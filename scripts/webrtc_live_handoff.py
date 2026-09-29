"""Non-blocking GPU-scheduler -> event-loop handoff for WebRTC live frames.

Plan item 1.5 (docs/musetalk_4070s_300fps_plan_2026-09-27.md). Flag:
WEBRTC_NONBLOCKING_HANDOFF=1 (default 0 keeps the blocking
``run_coroutine_threadsafe(...).result()`` handoff in api_server.py).

Design
------
* One ``LiveFrameHandoff`` per SwitchableVideoStreamTrack, created lazily.
* The scheduler thread calls ``begin_turn`` / ``submit_frames`` /
  ``submit_marker``; each appends to a thread-safe FIFO and, if no drain is
  running, wakes one with ``loop.call_soon_threadsafe``. Nothing waits on the
  event loop, and nothing waits on the track's strict-FIFO asyncio.Queue.
* A single drain task per track runs the items in submission order:
  ``start`` -> the unchanged ``_start_live_track`` coroutine; ``frames`` -> the
  unchanged ``track.push_bgr_frames_batch(frames, generation_id, metadata)``
  (same conversion, the same strict FIFO ``await queue.put`` backpressure,
  the same generation tokens); ``marker`` -> a callback that must observe every
  earlier frame already in the track queue (generation complete, A/V release
  on completion, playback drain).
* Frame content, order, RTP timestamps and the A/V release rule are therefore
  unchanged; only the scheduler thread stops waiting.
* ``depth_frames()`` = frames still in this FIFO + frames in the track queue.
  The scheduler can read it (``track.live_buffer_depth_frames()``) to skip a
  job whose playout buffer is full instead of generating further ahead.
* Bound: if more than ``max_pending_frames`` (WEBRTC_HANDOFF_MAX_PENDING_FRAMES,
  default 64 = 88 MB of 512x896 BGR per stream) frames are waiting to enter the
  track queue, ``submit_frames`` waits on a condition variable, bounded by the
  caller's push timeout. That only happens when the track queue itself is full
  (400 frames = 20 s of run-ahead), i.e. exactly when today's blocking handoff
  would also have blocked the scheduler; otherwise the scheduler never waits.
* ``close()`` (track stop) drops every queued item and wakes a waiting
  scheduler thread; ``submit_marker`` then returns False so the caller runs its
  completion path directly.
* WEBRTC_HANDOFF_VERIFY=1 (smoke tests only) hashes every frame at submit and
  again before the push, and compares converter-thread I420 with the on-loop
  conversion, so a live run can prove order and content were preserved.

Producer contract for WEBRTC_YUV_IN_COMPOSE=1: a frame may be any object with
``.bgr`` (uint8 HxWx3) and ``.yuv420p`` (an ``av.VideoFrame`` in yuv420p, or a
packed (H*3/2, W) uint8 I420 array) made by ``bgr_to_yuv420p_frame`` — the
exact PyAV call the track uses today — so the pushed I420 is SHA-identical.
``ComposedFrame`` is a ready-made carrier. Frames handed to the callback are
owned by the handoff afterwards and must not be mutated by the producer.
"""
from __future__ import annotations

import asyncio
import collections
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import hashlib

from scripts.webrtc_media_flags import (  # noqa: F401
    env_int,
    handoff_verify_enabled,
    nonblocking_handoff_enabled,
    queue_packed_i420_enabled,
)

DEFAULT_MAX_PENDING_FRAMES = 64

try:  # PyAV / NumPy are present in the server venv; keep import-light for tooling.
    import av  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    av = None
    np = None


# ---------------------------------------------------------------------------
# Frame conversion helpers (the single definition of "today's conversion")
# ---------------------------------------------------------------------------

def bgr_to_yuv420p_frame(frame_bgr):
    """Exactly the conversion SwitchableVideoStreamTrack.push_* has always used."""
    return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")


class ComposedFrame:
    """Composed output that carries its exact I420 conversion.

    ``np.asarray(frame)`` still returns the BGR array so shape checks and
    capture taps that expect ndarrays keep working.
    """

    __slots__ = ("bgr", "yuv420p")

    def __init__(self, bgr, yuv420p=None):
        self.bgr = bgr
        self.yuv420p = yuv420p

    @classmethod
    def from_bgr(cls, bgr):
        return cls(bgr, bgr_to_yuv420p_frame(bgr))

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.bgr)
        return array if dtype is None else array.astype(dtype)

    @property
    def shape(self):
        return np.asarray(self.bgr).shape


def video_frame_from_live_item(item):
    """Return (av.VideoFrame yuv420p, converted_here: bool) for one live item.

    Plain BGR ndarrays take today's exact conversion. Objects that carry a
    pre-converted ``yuv420p`` (see module docstring) are passed through.
    """
    yuv = getattr(item, "yuv420p", None)
    if yuv is not None:
        if isinstance(yuv, av.VideoFrame):
            if yuv.format.name != "yuv420p":
                raise ValueError(f"pre-converted live frame has format {yuv.format.name}")
            return yuv, False
        array = np.ascontiguousarray(yuv)
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError("pre-converted live frame must be a packed uint8 I420 array")
        return av.VideoFrame.from_ndarray(array, format="yuv420p"), False
    bgr = getattr(item, "bgr", None)
    if bgr is None:
        bgr = item
    return bgr_to_yuv420p_frame(bgr), True


def packed_i420_from_live_item(item):
    """WEBRTC_QUEUE_PACKED_I420=1: the packed (H*3/2, W) uint8 I420 of one live item.

    The bytes are exactly ``video_frame_from_live_item(item)[0].to_ndarray()``;
    ``live_frame_from_queue`` turns them back into the same av.VideoFrame.
    """
    yuv = getattr(item, "yuv420p", None)
    if isinstance(yuv, np.ndarray):
        array = np.ascontiguousarray(yuv)
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError("pre-converted live frame must be a packed uint8 I420 array")
        return array
    frame, _ = video_frame_from_live_item(item)
    return frame.to_ndarray()


def live_frame_from_queue(frame):
    """Inverse of ``packed_i420_from_live_item`` (av.VideoFrame passes through).

    Reformatted and from_ndarray frames both carry unspecified range/colorspace,
    so the rebuilt frame matches the one today's FIFO would have held.
    """
    if isinstance(frame, np.ndarray):
        return av.VideoFrame.from_ndarray(frame, format="yuv420p")
    return frame


_convert_pool: Optional[ThreadPoolExecutor] = None
_convert_pool_lock = threading.Lock()


def _converter_pool() -> Optional[ThreadPoolExecutor]:
    global _convert_pool
    workers = env_int("WEBRTC_HANDOFF_CONVERT_THREADS", 0, minimum=0, maximum=32)
    if workers <= 0:
        return None
    with _convert_pool_lock:
        if _convert_pool is None:
            _convert_pool = ThreadPoolExecutor(max_workers=workers,
                                               thread_name_prefix="webrtc-yuv")
        return _convert_pool


def _frame_digest(item) -> bytes:
    bgr = getattr(item, "bgr", None)
    if bgr is None:
        bgr = item
    return hashlib.sha1(np.ascontiguousarray(bgr).data).digest()


def _preconvert_batch(frames):
    """Converter-thread version of today's per-frame conversion (same call)."""
    packed = queue_packed_i420_enabled()
    converted = []
    for item in frames:
        if getattr(item, "yuv420p", None) is not None:
            converted.append(item)
            continue
        bgr = getattr(item, "bgr", None)
        if bgr is None:
            bgr = item
        yuv = bgr_to_yuv420p_frame(bgr)
        # The packed array keeps no av.VideoFrame (and its VideoFormat cycle) alive in the FIFO.
        converted.append(ComposedFrame(bgr, yuv.to_ndarray() if packed else yuv))
    return converted


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------

class HandoffTurn:
    """One speaking turn's ownership record inside a track handoff."""

    __slots__ = ("label", "start_factory", "on_started", "can_publish", "generation_id",
                 "state", "cancelled", "frames_submitted", "frames_pushed",
                 "frames_skipped", "created_at", "started_at", "verify_next_seq")

    def __init__(self, start_factory, on_started=None, can_publish=None, label=None):
        self.label = label
        self.start_factory = start_factory
        self.on_started = on_started
        self.can_publish = can_publish
        self.generation_id = None
        self.state = "pending"  # pending -> started | failed
        self.cancelled = False
        self.frames_submitted = 0
        self.frames_pushed = 0
        self.frames_skipped = 0
        self.created_at = time.monotonic()
        self.started_at = None
        self.verify_next_seq = 0

    def publishable(self) -> bool:
        if self.cancelled:
            return False
        if self.can_publish is None:
            return True
        try:
            return bool(self.can_publish())
        except Exception:
            return False


class LiveFrameHandoff:
    def __init__(self, track, loop: asyncio.AbstractEventLoop,
                 max_pending_frames: Optional[int] = None):
        self._track = track
        self._loop = loop
        if max_pending_frames is None:
            max_pending_frames = env_int("WEBRTC_HANDOFF_MAX_PENDING_FRAMES",
                                         DEFAULT_MAX_PENDING_FRAMES, minimum=0)
        if not max_pending_frames:
            max_pending_frames = int(getattr(track, "_max_queue", 0) or 400)
        self.max_pending_frames = int(max_pending_frames)
        self.verify = handoff_verify_enabled()
        self._closed = False
        self._lock = threading.Lock()
        self._space = threading.Condition(self._lock)
        self._items: collections.deque = collections.deque()
        self._pending_frames = 0
        self._drain_task: Optional[asyncio.Task] = None
        self._wake_scheduled = False
        self._loop_thread_id: Optional[int] = None
        # Statistics (read without the lock; single-writer or monotonic).
        self.turns_started = 0
        self.turns_failed = 0
        self.batches_submitted = 0
        self.frames_submitted = 0
        self.frames_pushed = 0
        self.frames_skipped = 0
        self.markers_run = 0
        self.max_pending_frames_seen = 0
        self.max_depth_frames_seen = 0
        self.overflow_waits = 0
        self.overflow_wait_total_s = 0.0
        self.overflow_wait_max_s = 0.0
        self.submit_total_s = 0.0
        self.submit_max_s = 0.0
        self.push_batches = 0
        self.push_total_s = 0.0
        self.push_max_s = 0.0
        self.latency_total_s = 0.0
        self.latency_max_s = 0.0
        self.preconvert_batches = 0
        self.errors = 0
        self.frames_dropped_on_close = 0
        self.verify_frames = 0
        self.verify_content_mismatch = 0
        self.verify_order_errors = 0
        self.verify_i420_checked = 0
        self.verify_i420_mismatch = 0

    # ------------------------------------------------------------------ any thread
    def _track_queue_size(self) -> int:
        queue = getattr(self._track, "_queue", None)
        try:
            return int(queue.qsize()) if queue is not None else 0
        except Exception:
            return 0

    def pending_frames(self) -> int:
        return self._pending_frames

    def depth_frames(self) -> int:
        """Frames produced but not yet played: FIFO backlog + track queue."""
        return self._pending_frames + self._track_queue_size()

    def _on_loop_thread(self) -> bool:
        return self._loop_thread_id is not None and threading.get_ident() == self._loop_thread_id

    def _enqueue_locked(self, item) -> None:
        self._items.append(item)
        if not self._wake_scheduled:
            self._wake_scheduled = True
            try:
                self._loop.call_soon_threadsafe(self._wake)
            except RuntimeError:
                self._wake_scheduled = False
                raise

    def begin_turn(self, start_factory: Callable[[], Any], *, on_started=None,
                   can_publish=None, label=None) -> HandoffTurn:
        """Queue the turn's live start. ``start_factory()`` returns an awaitable
        (today's ``_start_live_track`` coroutine) that yields the generation id.
        ``start_factory`` is called on the loop, so the coroutine is created there."""
        turn = HandoffTurn(start_factory, on_started=on_started,
                           can_publish=can_publish, label=label)
        with self._lock:
            if self._closed:
                turn.cancelled = True
                turn.state = "failed"
                return turn
            self._enqueue_locked(("start", turn, None, 0))
        return turn

    def submit_frames(self, turn: HandoffTurn, frames, metadata=None,
                      on_pushed: Optional[Callable[[bool], None]] = None,
                      wait_timeout_s: float = 30.0) -> int:
        """Hand one composed batch to the track without waiting for the loop."""
        count = len(frames)
        if count <= 0 or self._closed:
            return self.depth_frames()
        started_at = time.monotonic()
        digests = None
        if self.verify:
            digests = [_frame_digest(frame) for frame in frames]
        with self._space:
            if (self.max_pending_frames > 0 and self._pending_frames > 0
                    and self._pending_frames + count > self.max_pending_frames
                    and not self._on_loop_thread()):
                self.overflow_waits += 1
                wait_started = time.monotonic()
                deadline = wait_started + max(0.0, float(wait_timeout_s))
                while (not self._closed and self._pending_frames > 0
                       and self._pending_frames + count > self.max_pending_frames):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._space.wait(timeout=min(remaining, 0.25))
                waited = time.monotonic() - wait_started
                self.overflow_wait_total_s += waited
                self.overflow_wait_max_s = max(self.overflow_wait_max_s, waited)
                if waited >= max(0.0, float(wait_timeout_s)):
                    print(f"⚠️ [{turn.label}] WebRTC live handoff: FIFO still above "
                          f"{self.max_pending_frames} frames after {waited:.1f}s; queueing "
                          f"{count} more", flush=True)
            if self._closed:
                return self.depth_frames()
            first_seq = turn.frames_submitted
            self._pending_frames += count
            turn.frames_submitted += count
            self.batches_submitted += 1
            self.frames_submitted += count
            self.max_pending_frames_seen = max(self.max_pending_frames_seen, self._pending_frames)
            self._enqueue_locked(("frames", turn, (list(frames), metadata, on_pushed,
                                                    time.monotonic(), first_seq, digests),
                                  count))
            pending = self._pending_frames
        depth = pending + self._track_queue_size()
        self.max_depth_frames_seen = max(self.max_depth_frames_seen, depth)
        elapsed = time.monotonic() - started_at
        self.submit_total_s += elapsed
        self.submit_max_s = max(self.submit_max_s, elapsed)
        return depth

    def submit_marker(self, turn: HandoffTurn, callback: Callable[[Optional[int]], Any]) -> bool:
        """Run ``callback(generation_id)`` on the loop after every earlier item.

        Returns False (callback not queued) once the handoff is closed; the
        caller then runs its completion path directly."""
        with self._lock:
            if self._closed:
                return False
            self._enqueue_locked(("marker", turn, callback, 0))
            return True

    def cancel_turn(self, turn: Optional[HandoffTurn]) -> None:
        """Drop this turn's not-yet-pushed frames (and its start if still queued)."""
        if turn is not None:
            turn.cancelled = True

    def close(self) -> int:
        """Track teardown: drop every queued item (frames, starts, markers) and
        release any scheduler thread waiting on the FIFO bound. Returns the
        number of frames dropped. Safe from any thread; idempotent."""
        with self._space:
            self._closed = True
            dropped = 0
            while self._items:
                kind, turn, _payload, count = self._items.popleft()
                if turn is not None:
                    turn.cancelled = True
                dropped += count
            self._pending_frames = max(0, self._pending_frames - dropped)
            self.frames_dropped_on_close += dropped
            self._space.notify_all()
        return dropped

    # ------------------------------------------------------------------ loop thread
    def _wake(self) -> None:
        self._loop_thread_id = threading.get_ident()
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = self._loop.create_task(self._drain())

    async def _drain(self) -> None:
        self._loop_thread_id = threading.get_ident()
        while True:
            with self._lock:
                if not self._items:
                    self._wake_scheduled = False
                    self._drain_task = None
                    return
                item = self._items.popleft()
            await self._process(item)
            # A push that never suspends (queue not full) runs synchronously;
            # yield between items so one track's backlog cannot hold the loop
            # (and every other track's recv() pacing) for several batches.
            await asyncio.sleep(0)

    async def _process(self, item) -> None:
        kind, turn, payload, count = item
        try:
            if kind == "start":
                await self._process_start(turn)
            elif kind == "frames":
                await self._process_frames(turn, payload, count)
            elif kind == "marker":
                self.markers_run += 1
                payload(turn.generation_id)
        except Exception as exc:
            self.errors += 1
            if kind == "start":
                turn.state = "failed"
                self.turns_failed += 1
            print(f"⚠️ [{turn.label}] WebRTC live handoff {kind} failed: {exc}", flush=True)
            if kind != "start":
                traceback.print_exc()
        finally:
            if count:
                with self._space:
                    self._pending_frames -= count
                    self._space.notify_all()

    async def _process_start(self, turn: HandoffTurn) -> None:
        if not turn.publishable():
            turn.state = "failed"
            self.turns_failed += 1
            return
        generation_id = await turn.start_factory()
        turn.generation_id = generation_id
        turn.state = "started"
        turn.started_at = time.monotonic()
        self.turns_started += 1
        if turn.on_started is not None:
            try:
                turn.on_started(generation_id)
            except Exception as exc:
                self.errors += 1
                print(f"⚠️ [{turn.label}] WebRTC live handoff on_started failed: {exc}",
                      flush=True)
                traceback.print_exc()

    async def _process_frames(self, turn: HandoffTurn, payload, count: int) -> None:
        frames, metadata, on_pushed, submitted_at, first_seq, digests = payload
        if turn.state != "started" or not turn.publishable():
            turn.frames_skipped += count
            self.frames_skipped += count
            return
        if digests is not None:
            self._verify_batch(turn, frames, first_seq, digests)
        pool = _converter_pool()
        if pool is not None and any(getattr(f, "yuv420p", None) is None for f in frames):
            frames = await self._loop.run_in_executor(pool, _preconvert_batch, frames)
            self.preconvert_batches += 1
            if digests is not None:
                self._verify_i420(frames)
            if turn.state != "started" or not turn.publishable():
                turn.frames_skipped += count
                self.frames_skipped += count
                return
        push_started = time.monotonic()
        ready = await self._track.push_bgr_frames_batch(
            frames,
            generation_id=turn.generation_id,
            metadata=metadata,
        )
        now = time.monotonic()
        push_s = now - push_started
        latency = now - submitted_at
        self.push_batches += 1
        self.push_total_s += push_s
        self.push_max_s = max(self.push_max_s, push_s)
        self.latency_total_s += latency
        self.latency_max_s = max(self.latency_max_s, latency)
        turn.frames_pushed += count
        self.frames_pushed += count
        if on_pushed is not None:
            on_pushed(bool(ready))

    # ------------------------------------------------------------------ verify
    def _verify_batch(self, turn: HandoffTurn, frames, first_seq: int, digests) -> None:
        """Every frame reaches the push in submission order with unchanged bytes."""
        if first_seq != turn.verify_next_seq:
            self.verify_order_errors += 1
        turn.verify_next_seq = first_seq + len(frames)
        for frame, digest in zip(frames, digests):
            self.verify_frames += 1
            if _frame_digest(frame) != digest:
                self.verify_content_mismatch += 1

    def _verify_i420(self, frames) -> None:
        """Converter-thread I420 equals the on-loop conversion, byte for byte."""
        for item in frames:
            yuv = getattr(item, "yuv420p", None)
            if yuv is None:
                continue
            self.verify_i420_checked += 1
            reference = bgr_to_yuv420p_frame(item.bgr).to_ndarray()
            candidate = yuv.to_ndarray() if isinstance(yuv, av.VideoFrame) else np.asarray(yuv)
            if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
                self.verify_i420_mismatch += 1

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> dict:
        batches = max(1, self.push_batches)
        submits = max(1, self.batches_submitted)
        return {
            "mode": "nonblocking",
            "pending_frames": self._pending_frames,
            "queued_items": len(self._items),
            "depth_frames": self.depth_frames(),
            "max_pending_frames": self.max_pending_frames,
            "max_pending_frames_seen": self.max_pending_frames_seen,
            "max_depth_frames_seen": self.max_depth_frames_seen,
            "turns_started": self.turns_started,
            "turns_failed": self.turns_failed,
            "batches_submitted": self.batches_submitted,
            "frames_submitted": self.frames_submitted,
            "frames_pushed": self.frames_pushed,
            "frames_skipped": self.frames_skipped,
            "markers_run": self.markers_run,
            "overflow_waits": self.overflow_waits,
            "overflow_wait_total_s": round(self.overflow_wait_total_s, 4),
            "overflow_wait_max_s": round(self.overflow_wait_max_s, 4),
            "avg_submit_ms": round(self.submit_total_s / submits * 1000.0, 4),
            "max_submit_ms": round(self.submit_max_s * 1000.0, 4),
            "avg_push_ms": round(self.push_total_s / batches * 1000.0, 3),
            "max_push_ms": round(self.push_max_s * 1000.0, 3),
            "avg_handoff_latency_ms": round(self.latency_total_s / batches * 1000.0, 3),
            "max_handoff_latency_ms": round(self.latency_max_s * 1000.0, 3),
            "preconvert_batches": self.preconvert_batches,
            "convert_threads": env_int("WEBRTC_HANDOFF_CONVERT_THREADS", 0, minimum=0),
            "errors": self.errors,
            "closed": self._closed,
            "frames_dropped_on_close": self.frames_dropped_on_close,
            **({"verify_frames": self.verify_frames,
                "verify_content_mismatch": self.verify_content_mismatch,
                "verify_order_errors": self.verify_order_errors,
                "verify_i420_checked": self.verify_i420_checked,
                "verify_i420_mismatch": self.verify_i420_mismatch} if self.verify else {}),
        }


def get_live_handoff(track, loop: asyncio.AbstractEventLoop) -> LiveFrameHandoff:
    """The track's handoff, created on first use and bound to its event loop."""
    handoff = getattr(track, "_musetalk_live_handoff", None)
    if handoff is None or handoff._loop is not loop or handoff._closed:
        handoff = LiveFrameHandoff(track, loop)
        try:
            track._musetalk_live_handoff = handoff
        except Exception:
            pass
    return handoff
