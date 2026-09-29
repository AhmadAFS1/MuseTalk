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
        "64",
        "Bound of the per-track non-blocking FIFO (frames not yet in the track queue). When a "
        "batch would exceed it the scheduler waits, like today's blocking handoff, bounded by "
        "the push timeout. 64 x 1.38 MB (512x896 BGR) = 88 MB per stream worst case. "
        "0 = the track's max_queue (400 in strict FIFO)."),
    "WEBRTC_HANDOFF_CONVERT_THREADS": (
        "0",
        ">0: with the non-blocking handoff, BGR->yuv420p runs on N shared converter threads "
        "(the exact PyAV call used today; PyAV releases the GIL in sws_scale) instead of on "
        "the event loop. 0 = on the loop inside push_bgr_frames_batch (today's placement)."),
    "WEBRTC_HANDOFF_VERIFY": (
        "0",
        "Smoke-test only: SHA-1 every frame at submit and again right before the push and count "
        "order/content mismatches (live_handoff.verify_*). Costs ~0.4 ms/frame of CPU."),
    "WEBRTC_PREENCODE_SHA_DIR": (
        "",
        "Smoke-test only: directory; each track appends one JSON line per live frame entering "
        "its queue (generation, index, SHA-256 of the packed I420 planes) for A/B exactness."),
    "WEBRTC_QUEUE_PACKED_I420": (
        "0",
        "1: live frames wait in the track FIFO as packed I420 uint8 arrays (the exact bytes of today's "
        "PyAV conversion; the handoff converter threads produce them) and become an av.VideoFrame when "
        "recv() pops them. 0: the FIFO holds av.VideoFrame objects (today). Each av.VideoFrame owns a "
        "VideoFormat whose components reference it back, a reference cycle; queued for seconds, those "
        "cycles reach generation 2, so full collections (30 ms at 15 streams, every ~10 s) freeze the "
        "event loop. Same pixels, pts and colour metadata either way."),
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
    "WEBRTC_IDLE_FRAME_CACHE_WARM": (
        "0",
        "1 (with WEBRTC_IDLE_FRAME_CACHE=1): POST /avatars/{id}/cache/warm also builds that avatar's idle and pose "
        "clips into the idle frame cache (and waits for them when wait=true). 0: a clip is built at the avatar's "
        "first session create, where its decode competes with the live streams for the GIL (today)."),
    "WEBRTC_LIFETIME_COUNTERS": (
        "0",
        "1: per-track monotonic counters (frames_played, frames_duplicated, "
        "strict_video_stall_seconds, underruns, output frames, turns) that never reset per "
        "turn, plus a ring of server send (recv-return) timestamps; exposed under "
        "track_stats.video.lifetime and GET /webrtc/sessions/stats?view=lifetime."),
    "WEBRTC_LIFETIME_SEND_RING": (
        "256", "Entries kept in the per-track send-timestamp ring."),
    "WEBRTC_DEADLINE_PACING": (
        "0",
        "1: SwitchableVideoStreamTrack.recv() advances its pacing deadline by one frame time "
        "(as the motion-bank path already does) instead of re-anchoring on each wake-up, so "
        "asyncio oversleep no longer stretches the average send interval (~51 ms measured) and "
        "RTP media time stays on wall time. Frames are chosen by output index: same content."),
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
    "MUSETALK_TORCH_INTRAOP_THREADS": (
        "4", "torch intra-op thread cap applied at startup when MUSETALK_THREAD_CAPS=1 "
             "(0 = leave torch alone). Only lowers, never raises."),
    "MUSETALK_CV2_THREADS": (
        "", "cv2.setNumThreads() applied at startup when MUSETALK_THREAD_CAPS=1; unset = "
            "leave OpenCV alone (per-worker caps belong to the compose owner)."),
    "MUSETALK_FFMPEG_EXECUTOR_WORKERS": (
        "4", "With MUSETALK_THREAD_CAPS=1 the per-turn ffmpeg conversion and PCM load run on "
             "this dedicated executor instead of the loop's default executor."),
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
    "MUSETALK_OFFLOOP_DIAGNOSTICS": (
        "0",
        "1: the per-turn '🎬 WebRTC stream request' resource snapshot (it runs nvidia-smi, "
        "40-80 ms) is taken on a worker thread and logged when ready, and GET /stats, GET /health and "
        "GET /worker/state (the last two run nvidia-smi through the worker metrics provider) build their "
        "replies on a worker thread. 0: all run on the event loop (today), which freezes every stream's "
        "pacing for that long (live 15-stream test, 2026-09-29)."),
}


def nonblocking_handoff_enabled() -> bool:
    return env_bool("WEBRTC_NONBLOCKING_HANDOFF", False)


def queue_packed_i420_enabled() -> bool:
    return env_bool("WEBRTC_QUEUE_PACKED_I420", False)


def yuv_in_compose_enabled() -> bool:
    return env_bool("WEBRTC_YUV_IN_COMPOSE", False)


def idle_frame_cache_enabled() -> bool:
    return env_bool("WEBRTC_IDLE_FRAME_CACHE", False)


def idle_frame_cache_warm_enabled() -> bool:
    return env_bool("WEBRTC_IDLE_FRAME_CACHE_WARM", False)


def lifetime_counters_enabled() -> bool:
    return env_bool("WEBRTC_LIFETIME_COUNTERS", False)


def offloop_diagnostics_enabled() -> bool:
    return env_bool("MUSETALK_OFFLOOP_DIAGNOSTICS", False)


def thread_caps_enabled() -> bool:
    return env_bool("MUSETALK_THREAD_CAPS", False)


def local_tts_disabled() -> bool:
    return env_bool("MUSETALK_DISABLE_LOCAL_TTS", False)


def group_max_count() -> int:
    return env_int("WEBRTC_GROUP_MAX_COUNT", 12, minimum=1)


def handoff_verify_enabled() -> bool:
    return env_bool("WEBRTC_HANDOFF_VERIFY", False)


def preencode_sha_dir() -> str:
    return env_str("WEBRTC_PREENCODE_SHA_DIR", "")


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


_thread_caps_summary: Optional[dict] = None


def apply_thread_caps(label: str = "api_server") -> dict:
    """MUSETALK_THREAD_CAPS=1: targeted, process-wide thread caps (plan item 1.11).

    Idle-decoder caps are applied where decoders are opened
    (effective_decode_threads). Here: torch intra-op threads (lower only) and,
    only if MUSETALK_CV2_THREADS is set, OpenCV's pool. Global OMP/MKL caps
    (MUSETALK_CPU_TUNING) are deliberately not touched. Idempotent.
    """
    global _thread_caps_summary
    if _thread_caps_summary is not None:
        return dict(_thread_caps_summary)
    summary: dict = {"enabled": thread_caps_enabled()}
    if summary["enabled"]:
        summary["idle_decode_threads"] = env_int("MUSETALK_IDLE_DECODE_THREADS", 1, minimum=1)
        summary["motion_decode_threads"] = (
            os.environ.get("MUSETALK_MOTION_DECODE_THREADS", "").strip() or "unchanged")
        torch_cap = env_int("MUSETALK_TORCH_INTRAOP_THREADS", 4, minimum=0, maximum=256)
        if torch_cap > 0:
            try:
                import torch  # already imported by the server stack at this point
                before = int(torch.get_num_threads())
                if before > torch_cap:
                    torch.set_num_threads(torch_cap)
                summary["torch_intraop_threads"] = {"before": before,
                                                    "after": int(torch.get_num_threads())}
            except Exception as exc:  # pragma: no cover - torch always present in the server
                summary["torch_intraop_threads"] = f"error: {exc}"
        cv2_text = os.environ.get("MUSETALK_CV2_THREADS", "").strip()
        if cv2_text:
            try:
                import cv2
                before = int(cv2.getNumThreads())
                cv2.setNumThreads(int(cv2_text))
                summary["cv2_threads"] = {"before": before, "after": int(cv2.getNumThreads())}
            except Exception as exc:
                summary["cv2_threads"] = f"error: {exc}"
        summary["ffmpeg_executor_workers"] = env_int(
            "MUSETALK_FFMPEG_EXECUTOR_WORKERS", 4, minimum=1, maximum=64)
        print(f"[{label}] MUSETALK_THREAD_CAPS=1: {summary}", flush=True)
    _thread_caps_summary = summary
    return dict(summary)


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
