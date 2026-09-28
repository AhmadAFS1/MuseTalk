"""One-line env flags for the WebRTC media path (300 fps plan, 2026-09-27).

Plan items 0.4, 0.5, 1.5, 1.6 (flag only), 1.7, 1.8, 1.11. Every flag's default
reproduces the behaviour before these changes exactly; rollback is deleting the
line from the overlay env or setting it back to its default.

This module is stdlib-only so it can be imported by tests, the load harness and
the server without pulling in aiortc, PyAV or torch.
"""
from __future__ import annotations

import os
from typing import Optional

_TRUE = ("1", "true", "yes", "on")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in _TRUE


def env_int(name: str, default: int, minimum: Optional[int] = None,
            maximum: Optional[int] = None) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        result = default
    else:
        try:
            result = int(value.strip())
        except ValueError:
            result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


# name -> (default, effect). The default string is what an unset variable means.
FLAGS = {
    "WEBRTC_NONBLOCKING_HANDOFF": (
        "0",
        "1: the GPU scheduler thread appends composed batches to a per-track FIFO and "
        "returns at once; one drain task on the event loop pushes them through the "
        "unchanged push_bgr_frames_batch (same conversion, order, tokens, A/V release). "
        "0: scheduler blocks on run_coroutine_threadsafe(...).result() per batch (today)."),
    "WEBRTC_HANDOFF_MAX_PENDING_FRAMES": (
        "0",
        "Safety valve for the non-blocking FIFO: when more than N frames are still waiting "
        "to enter the track queue the scheduler waits (bounded by the push timeout). "
        "0 = the track's max_queue (400 in strict FIFO)."),
    "WEBRTC_HANDOFF_CONVERT_THREADS": (
        "0",
        ">0: with the non-blocking handoff, BGR->yuv420p runs on N shared converter threads "
        "(the exact PyAV call used today) instead of on the event loop. 0 = on the loop."),
    "WEBRTC_YUV_IN_COMPOSE": (
        "0",
        "Producer contract: when 1 the compose side may hand objects with .bgr and "
        ".yuv420p (av.VideoFrame or packed I420 ndarray made by the exact PyAV conversion). "
        "push_bgr_frames_batch always accepts them; plain ndarrays keep today's conversion."),
    "WEBRTC_IDLE_FRAME_CACHE": (
        "0",
        "1: each idle/pose clip is decoded once per process into a shared yuv420p array; "
        "idle playback, pose switches, completion staging and motion entry/return builders "
        "index it instead of decoding on the event loop. 0: per-session PyAV decode (today)."),
    "WEBRTC_IDLE_FRAME_CACHE_MAX_MB": (
        "1024",
        "LRU budget of the idle frame cache across avatars; a clip that does not fit (all "
        "resident clips in use) is not cached and that session decodes as today."),
    "WEBRTC_IDLE_FRAME_CACHE_DECODE_THREADS": (
        "4", "FFmpeg frame threads used for the one-time background decode of a clip."),
    "WEBRTC_IDLE_FRAME_CACHE_WORKERS": (
        "2", "Background threads that build cache entries."),
    "WEBRTC_LIFETIME_COUNTERS": (
        "0",
        "1: per-track monotonic counters (frames_played, frames_duplicated, "
        "strict_video_stall_seconds, underruns, output frames, turns) that never reset per "
        "turn, plus a ring of server send (recv-return) timestamps; exposed under "
        "track_stats.video.lifetime and GET /webrtc/sessions/stats?view=lifetime."),
    "WEBRTC_LIFETIME_SEND_RING": (
        "256", "Entries kept in the per-track send-timestamp ring."),
    "WEBRTC_GROUP_MAX_COUNT": (
        "12", "Upper bound for count on /webrtc/groups/create and /hls/groups/create."),
    "MUSETALK_DISABLE_LOCAL_TTS": (
        "0", "1: POST /webrtc/tts/kokoro returns 503 without loading Kokoro."),
    "MUSETALK_THREAD_CAPS": (
        "0",
        "1: persistent idle decoders use MUSETALK_IDLE_DECODE_THREADS FFmpeg threads "
        "instead of FFmpeg's auto 16 slice threads (same decoded pixels)."),
    "MUSETALK_IDLE_DECODE_THREADS": (
        "1", "Idle decoder threads when MUSETALK_THREAD_CAPS=1."),
    "MUSETALK_MOTION_DECODE_THREADS": (
        "", "Motion entry/return builder decoder threads when MUSETALK_THREAD_CAPS=1; "
            "unset keeps today's 16 (their decode latency is a visible hold)."),
    "WEBRTC_NATIVE_VP8_THREADS": (
        "",
        "Native VP8 cfg.g_threads. Unset = aiortc's number_of_threads() (2 for 512x896 on "
        "this 32-CPU box), i.e. today."),
    "WEBRTC_H264_IMPL": (
        "aiortc",
        "aiortc: aiortc 1.14 built-in libx264 (preset medium, zerolatency, Baseline, auto "
        "threads) = today. x264tuned: libx264 with WEBRTC_H264_X264_PRESET/THREADS. "
        "nvenc: h264_nvenc under a process-wide session cap, x264tuned fallback."),
    "WEBRTC_H264_X264_PRESET": ("veryfast", "x264 preset for x264tuned (and NVENC fallback)."),
    "WEBRTC_H264_X264_THREADS": ("1", "x264 threads for x264tuned (and NVENC fallback)."),
    "WEBRTC_H264_NVENC_PRESET": ("p2", "NVENC preset for WEBRTC_H264_IMPL=nvenc."),
    "WEBRTC_H264_NVENC_TUNE": ("ll", "NVENC tune for WEBRTC_H264_IMPL=nvenc."),
    "WEBRTC_NVENC_MAX_SESSIONS": (
        "12", "Process-wide NVENC session semaphore; encoders beyond it use x264tuned."),
}


def nonblocking_handoff_enabled() -> bool:
    return env_bool("WEBRTC_NONBLOCKING_HANDOFF", False)


def yuv_in_compose_enabled() -> bool:
    return env_bool("WEBRTC_YUV_IN_COMPOSE", False)


def idle_frame_cache_enabled() -> bool:
    return env_bool("WEBRTC_IDLE_FRAME_CACHE", False)


def lifetime_counters_enabled() -> bool:
    return env_bool("WEBRTC_LIFETIME_COUNTERS", False)


def thread_caps_enabled() -> bool:
    return env_bool("MUSETALK_THREAD_CAPS", False)


def local_tts_disabled() -> bool:
    return env_bool("MUSETALK_DISABLE_LOCAL_TTS", False)


def group_max_count() -> int:
    return env_int("WEBRTC_GROUP_MAX_COUNT", 12, minimum=1)


def effective_decode_threads(requested: int) -> int:
    """FFmpeg decoder thread count for an idle-clip decoder.

    ``requested`` is what the caller asked for today (0 = FFmpeg auto, which is
    16 slice threads for these clips). Decoded pixels do not depend on it.
    """
    requested = max(0, int(requested or 0))
    if not thread_caps_enabled():
        return requested
    if requested == 0:
        return env_int("MUSETALK_IDLE_DECODE_THREADS", 1, minimum=1, maximum=64)
    motion = os.environ.get("MUSETALK_MOTION_DECODE_THREADS", "").strip()
    if motion:
        return env_int("MUSETALK_MOTION_DECODE_THREADS", requested, minimum=1, maximum=64)
    return requested


def snapshot() -> dict:
    """Current value of every flag (unset flags report their default)."""
    return {name: os.environ.get(name, default) for name, (default, _effect) in FLAGS.items()}


def non_default() -> dict:
    return {name: value for name, value in snapshot().items()
            if value != FLAGS[name][0]}


def startup_line(label: str = "api_server") -> str:
    changed = non_default()
    body = " ".join(f"{key}={value}" for key, value in sorted(changed.items())) or "all defaults"
    return f"[{label}] WebRTC media flags: {body}"
