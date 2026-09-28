"""Process-wide pre-decoded idle/pose clip cache for WebRTC playback.

Plan item 1.7. Flag: WEBRTC_IDLE_FRAME_CACHE=1 (default 0 = every
IdleVideoStreamTrack decodes its mp4 with PyAV inside recv(), on the event
loop, as before).

Each clip is decoded ONCE per process, in a background thread, into one packed
uint8 array of shape (N, H*3/2, W) holding exactly ``frame.reformat(
format="yuv420p").to_ndarray()`` for every decoded frame, i.e. the planes
IdleVideoStreamTrack.read_frame() has always returned. Readers get a fresh
``av.VideoFrame`` per call (``from_ndarray`` copy, ~0.1 ms) so pts/time_base
stamping never aliases between sessions.

Bounded: WEBRTC_IDLE_FRAME_CACHE_MAX_MB (default 1024) across all avatars, LRU
over clips no session is reading. A clip that cannot fit is not cached; that
session keeps decoding as today (counted as ``admission_rejected``).
"""
from __future__ import annotations

import os
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from scripts.webrtc_media_flags import env_int, idle_frame_cache_enabled

try:
    import av  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    av = None
    np = None


def clip_key(path: str):
    real = os.path.realpath(str(path))
    st = os.stat(real)
    return (real, int(st.st_size), int(st.st_mtime_ns))


def _avatar_label(real_path: str) -> str:
    parts = Path(real_path).parts
    if "avatars" in parts:
        index = parts.index("avatars")
        if index + 1 < len(parts):
            return parts[index + 1]
    return Path(real_path).parent.name or real_path


class CachedClip:
    """One decoded clip. ``frames[i]`` is the packed I420 of decoded frame i."""

    __slots__ = ("key", "path", "frames", "count", "width", "height", "nbytes",
                 "stream_frames", "duration_seconds", "average_rate", "color_range",
                 "colorspace", "decode_seconds", "readers", "last_used", "created_at",
                 "__weakref__")

    def __init__(self, key, frames, width, height, stream_frames, duration_seconds,
                 average_rate, color_range, colorspace, decode_seconds):
        self.key = key
        self.path = key[0]
        self.frames = frames
        self.count = int(frames.shape[0])
        self.width = int(width)
        self.height = int(height)
        self.nbytes = int(frames.nbytes)
        self.stream_frames = int(stream_frames or 0)
        self.duration_seconds = duration_seconds
        self.average_rate = average_rate
        self.color_range = color_range
        self.colorspace = colorspace
        self.decode_seconds = decode_seconds
        self.readers = 0
        self.last_used = time.monotonic()
        self.created_at = time.monotonic()

    def video_frame(self, index: int):
        frame = av.VideoFrame.from_ndarray(self.frames[index], format="yuv420p")
        # Decoded frames of these clips carry unspecified range/colorspace, which
        # is also what from_ndarray produces. Copy anything else so downstream
        # colour conversions (motion blends, crossfades) see identical metadata.
        if self.color_range and frame.color_range != self.color_range:
            frame.color_range = self.color_range
        if self.colorspace is not None and frame.colorspace != self.colorspace:
            frame.colorspace = self.colorspace
        return frame


def decode_clip(path: str, decode_threads: int = 4) -> dict:
    """Decode a whole clip exactly as IdleVideoStreamTrack.read_frame() would."""
    key = clip_key(path)
    started = time.perf_counter()
    container = av.open(key[0])
    try:
        stream = container.streams.video[0]
        if decode_threads and decode_threads > 0:
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = int(decode_threads)
        rate = stream.average_rate
        average_rate = float(rate) if rate else None
        stream_frames = int(getattr(stream, "frames", 0) or 0)
        duration_seconds = None
        if getattr(stream, "duration", None) and getattr(stream, "time_base", None):
            duration_seconds = float(stream.duration * stream.time_base)
        elif getattr(container, "duration", None):
            duration_seconds = float(container.duration) / 1_000_000.0
        # Fill one preallocated array (no transient 2x copy); extra frames beyond
        # the container's frame count, if any, are appended and concatenated.
        frames = None
        extra = []
        count = 0
        color_range = None
        colorspace = None
        width = height = None
        for frame in container.decode(stream):
            yuv = frame.reformat(format="yuv420p")
            packed = yuv.to_ndarray()
            if frames is None:
                width, height = yuv.width, yuv.height
                color_range = int(yuv.color_range)
                colorspace = int(yuv.colorspace)
                frames = np.empty((max(1, stream_frames),) + packed.shape, dtype=np.uint8)
            if count < frames.shape[0]:
                frames[count] = packed
            else:
                extra.append(packed)
            count += 1
    finally:
        container.close()
    if frames is None or count == 0:
        raise ValueError(f"idle clip has no decodable frames: {path}")
    if extra:
        frames = np.concatenate([frames, np.stack(extra)])
    elif count < frames.shape[0]:
        frames = frames[:count].copy()
    frames.setflags(write=False)
    return {"key": key, "frames": frames, "width": width, "height": height,
            "stream_frames": stream_frames, "duration_seconds": duration_seconds,
            "average_rate": average_rate, "color_range": color_range,
            "colorspace": colorspace, "decode_seconds": time.perf_counter() - started}


def estimate_clip_bytes(path: str) -> int:
    container = av.open(os.path.realpath(str(path)))
    try:
        stream = container.streams.video[0]
        frames = int(getattr(stream, "frames", 0) or 0)
        width = int(stream.codec_context.width or stream.width or 0)
        height = int(stream.codec_context.height or stream.height or 0)
        if frames <= 0:
            rate = float(stream.average_rate or 25.0)
            duration = float(container.duration or 0) / 1_000_000.0
            frames = int(round(duration * rate)) or 1
    finally:
        container.close()
    return frames * width * height * 3 // 2


class IdleFrameCache:
    def __init__(self, max_bytes: int, decode_threads: int = 4, workers: int = 2):
        self.max_bytes = int(max_bytes)
        self.decode_threads = int(decode_threads)
        self._lock = threading.Lock()
        self._ready: "OrderedDict[tuple, CachedClip]" = OrderedDict()
        self._pending: dict = {}
        self._reserved: dict = {}
        self._rejected: set = set()
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                            thread_name_prefix="idle-frame-cache")
        self.hits = 0
        self.misses = 0
        self.builds = 0
        self.build_failures = 0
        self.build_seconds_total = 0.0
        self.evictions = 0
        self.admission_rejected = 0
        self.frames_served = 0

    # ------------------------------------------------------------------ budget
    def _used_bytes_locked(self) -> int:
        return sum(c.nbytes for c in self._ready.values()) + sum(self._reserved.values())

    def _make_room_locked(self, needed: int) -> bool:
        if needed > self.max_bytes:
            return False
        while self._used_bytes_locked() + needed > self.max_bytes:
            victim = next((k for k, c in self._ready.items() if c.readers <= 0), None)
            if victim is None:
                return False
            self._ready.pop(victim)
            self.evictions += 1
        return True

    # ------------------------------------------------------------------ readers
    def acquire(self, path: str) -> Optional[CachedClip]:
        """Return a ready clip (reader count +1) or None."""
        try:
            key = clip_key(path)
        except OSError:
            return None
        with self._lock:
            clip = self._ready.get(key)
            if clip is None:
                self.misses += 1
                return None
            clip.readers += 1
            clip.last_used = time.monotonic()
            self._ready.move_to_end(key)
            self.hits += 1
            return clip

    def release(self, clip: Optional[CachedClip]) -> None:
        if clip is None:
            return
        with self._lock:
            clip.readers = max(0, clip.readers - 1)
            clip.last_used = time.monotonic()
            if clip.readers == 0:
                # A clip freed its readers: earlier rejections may now fit.
                self._rejected.clear()

    def note_served(self, count: int = 1) -> None:
        self.frames_served += count

    def request(self, path: str) -> Optional[Future]:
        """Start a background build for ``path`` (idempotent). Never blocks."""
        try:
            key = clip_key(path)
        except OSError:
            return None
        with self._lock:
            if key in self._ready:
                done: Future = Future()
                done.set_result(True)
                return done
            if key in self._pending:
                return self._pending[key]
            if key in self._rejected:
                return None
            future = self._executor.submit(self._build, key)
            self._pending[key] = future
            return future

    def prewarm(self, paths) -> int:
        count = 0
        for path in paths or ():
            if path and self.request(str(path)) is not None:
                count += 1
        return count

    def _build(self, key) -> bool:
        try:
            estimate = estimate_clip_bytes(key[0])
            with self._lock:
                if not self._make_room_locked(estimate):
                    self.admission_rejected += 1
                    self._rejected.add(key)
                    return False
                self._reserved[key] = estimate
            decoded = decode_clip(key[0], self.decode_threads)
            if decoded["key"] != key:
                raise RuntimeError(f"idle clip changed while decoding: {key[0]}")
            clip = CachedClip(**decoded)
            with self._lock:
                self._reserved.pop(key, None)
                if not self._make_room_locked(clip.nbytes):
                    self.admission_rejected += 1
                    self._rejected.add(key)
                    return False
                self._ready[key] = clip
                self.builds += 1
                self.build_seconds_total += clip.decode_seconds
            print(f"🎞️ Idle frame cache: decoded {clip.count} frames "
                  f"{clip.width}x{clip.height} ({clip.nbytes / 2**20:.1f} MiB) in "
                  f"{clip.decode_seconds * 1000:.0f} ms: {key[0]}", flush=True)
            return True
        except Exception as exc:
            with self._lock:
                self.build_failures += 1
                self._reserved.pop(key, None)
            print(f"⚠️ Idle frame cache build failed for {key[0]}: {exc}", flush=True)
            return False
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = list(self._pending.values())
            if not pending:
                return True
            for future in pending:
                try:
                    future.result(timeout=max(0.0, deadline - time.monotonic()))
                except Exception:
                    pass
        return False

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> dict:
        with self._lock:
            clips = list(self._ready.values())
            used = self._used_bytes_locked()
            pending = len(self._pending)
            rejected = len(self._rejected)
        per_avatar: dict = {}
        for clip in clips:
            label = _avatar_label(clip.path)
            entry = per_avatar.setdefault(label, {"clips": 0, "frames": 0, "mib": 0.0})
            entry["clips"] += 1
            entry["frames"] += clip.count
            entry["mib"] = round(entry["mib"] + clip.nbytes / 2**20, 2)
        return {
            "enabled": True,
            "max_mib": round(self.max_bytes / 2**20, 1),
            "used_mib": round(used / 2**20, 2),
            "clips": len(clips),
            "pending_builds": pending,
            "rejected_keys": rejected,
            "hits": self.hits,
            "misses": self.misses,
            "builds": self.builds,
            "build_failures": self.build_failures,
            "avg_build_ms": round(self.build_seconds_total / max(1, self.builds) * 1000, 1),
            "evictions": self.evictions,
            "admission_rejected": self.admission_rejected,
            "frames_served": self.frames_served,
            "per_avatar": per_avatar,
            "clip_detail": [{"path": c.path, "frames": c.count, "mib": round(c.nbytes / 2**20, 2),
                             "readers": c.readers} for c in clips],
        }


_cache: Optional[IdleFrameCache] = None
_cache_lock = threading.Lock()


def get_idle_frame_cache(force: bool = False) -> Optional[IdleFrameCache]:
    """The process cache when WEBRTC_IDLE_FRAME_CACHE=1 (or ``force``), else None."""
    global _cache
    if not force and not idle_frame_cache_enabled():
        return None
    with _cache_lock:
        if _cache is None:
            _cache = IdleFrameCache(
                max_bytes=env_int("WEBRTC_IDLE_FRAME_CACHE_MAX_MB", 1024, minimum=1) * 2**20,
                decode_threads=env_int("WEBRTC_IDLE_FRAME_CACHE_DECODE_THREADS", 4, minimum=0),
                workers=env_int("WEBRTC_IDLE_FRAME_CACHE_WORKERS", 2, minimum=1),
            )
        return _cache


def reset_idle_frame_cache_for_tests() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def register_reader_finalizer(owner, cache: IdleFrameCache, clip: CachedClip):
    """Release ``clip`` when ``owner`` is collected without an explicit stop()."""
    return weakref.finalize(owner, cache.release, clip)
