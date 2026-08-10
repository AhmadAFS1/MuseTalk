"""
WebRTC media tracks for MuseTalk.

Classes:
- Video tracks: IdleVideoStreamTrack, SwitchableVideoStreamTrack, LiveVideoStreamTrack
- Audio tracks: SilenceAudioStreamTrack, SyncedAudioStreamTrack
"""

import asyncio
import fractions
import json
import math
import os
import subprocess
import threading
import time
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import av
import numpy as np
from aiortc import VideoStreamTrack, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

# Environment configuration
WEBRTC_SYNC_MODE = os.getenv("WEBRTC_SYNC_MODE", "strict_fifo").strip().lower()
WEBRTC_STRICT_FIFO_SYNC = WEBRTC_SYNC_MODE in ("strict_fifo", "fifo", "hls_like", "hls-like", "hls")
WEBRTC_AUDIO_SYNC_STRATEGY = os.getenv(
    "WEBRTC_AUDIO_SYNC_STRATEGY",
    "timestamp_locked",
).strip().lower()
WEBRTC_VIDEO_PREBUFFER_SECONDS = float(os.getenv("WEBRTC_VIDEO_PREBUFFER_SECONDS", "2.0"))
WEBRTC_ADAPTIVE_FPS = os.getenv("WEBRTC_ADAPTIVE_FPS", "0").lower() in ("1", "true", "yes")
WEBRTC_MIN_FPS_RATIO = float(os.getenv("WEBRTC_MIN_FPS_RATIO", "0.75"))  # Allow slowdown to 75%
WEBRTC_QUEUE_LOG_INTERVAL = int(os.getenv("WEBRTC_QUEUE_LOG_INTERVAL", "30"))
WEBRTC_TARGET_QUEUE_FILL = float(os.getenv("WEBRTC_TARGET_QUEUE_FILL", "0.4"))  # Target 40% queue fill
WEBRTC_VIDEO_CLOCK_RATE = 90000
WEBRTC_VIDEO_TIME_BASE = fractions.Fraction(1, WEBRTC_VIDEO_CLOCK_RATE)
WEBRTC_SYNC_EPSILON_SECONDS = float(os.getenv("WEBRTC_SYNC_EPSILON_SECONDS", "0.005"))
WEBRTC_STRICT_AUDIO_WAIT_TIMEOUT_SECONDS = float(os.getenv("WEBRTC_STRICT_AUDIO_WAIT_TIMEOUT_SECONDS", "30.0"))
WEBRTC_STRICT_VIDEO_WAIT_TIMEOUT_SECONDS = float(os.getenv("WEBRTC_STRICT_VIDEO_WAIT_TIMEOUT_SECONDS", "30.0"))
WEBRTC_VIDEO_MAX_QUEUE_FRAMES = int(os.getenv("WEBRTC_VIDEO_MAX_QUEUE_FRAMES", "400" if WEBRTC_STRICT_FIFO_SYNC else "100"))
WEBRTC_IDLE_SYNC_HOLD = os.getenv("WEBRTC_IDLE_SYNC_HOLD", "1").lower() in ("1", "true", "yes", "on")
WEBRTC_POSE_CROSSFADE_FRAMES = max(
    0,
    int(os.getenv("WEBRTC_POSE_CROSSFADE_FRAMES", "0")),
)

# ============================================================================
# Video Tracks
# ============================================================================

class VideoSyncClock:
    def __init__(self, source_fps: float, strict_fifo: Optional[bool] = None):
        self.source_fps = float(source_fps) if source_fps > 0 else 1.0
        self.source_frames = 0
        self.active = False
        self.started = asyncio.Event()
        self.playout_released = asyncio.Event()
        self.audio_complete = asyncio.Event()
        self.strict_fifo = WEBRTC_STRICT_FIFO_SYNC if strict_fifo is None else bool(strict_fifo)
        self._coverage_changed = asyncio.Event()
        self._closed = False
        self.audio_waiting = False
        self.video_waiting = False
        self.audio_stalls = 0
        self.video_stalls = 0
        self.audio_stall_seconds = 0.0
        self.video_stall_seconds = 0.0
        self.last_audio_wait_target_seconds = 0.0
        self.audio_ready_at: Optional[float] = None
        self.video_ready_at: Optional[float] = None
        self.playout_released_at: Optional[float] = None
        self.playout_start_time: Optional[float] = None
        self.first_video_frame_at: Optional[float] = None
        self.first_audio_packet_at: Optional[float] = None
        self.playout_released_wall_time: Optional[float] = None
        self.playout_start_wall_time: Optional[float] = None
        self.first_video_frame_wall_time: Optional[float] = None
        self.first_audio_packet_wall_time: Optional[float] = None
        self.turn_request_id: Optional[str] = None
        self.turn_session_id: Optional[str] = None
        self._av_start_summary_logged = False
        self.audio_completed_at: Optional[float] = None
        self.audio_media_seconds: Optional[float] = None
        self.audio_playout_seconds = 0.0
        # Session-level RTP phase published continuously by the persistent
        # audio transport, including while the avatar is idle.
        self.audio_transport_next_pts_seconds: Optional[float] = None
        self.first_live_audio_target_seconds: Optional[float] = None
        self.first_live_video_rtp_seconds: Optional[float] = None
        self.first_tts_transport_pts_seconds: Optional[float] = None
        self.audio_transport_rebase_target_seconds: Optional[float] = None
        self.video_rtp_phase_correction_seconds = 0.0
        self.audio_rtp_phase_correction_seconds = 0.0
        self.first_live_rtp_max_mismatch_seconds: Optional[float] = None

    def reset(self) -> None:
        self.source_frames = 0
        self.active = True
        self._closed = False
        self.audio_waiting = False
        self.video_waiting = False
        self.audio_stalls = 0
        self.video_stalls = 0
        self.audio_stall_seconds = 0.0
        self.video_stall_seconds = 0.0
        self.last_audio_wait_target_seconds = 0.0
        self.audio_ready_at = None
        self.video_ready_at = None
        self.playout_released_at = None
        self.playout_start_time = None
        self.first_video_frame_at = None
        self.first_audio_packet_at = None
        self.playout_released_wall_time = None
        self.playout_start_wall_time = None
        self.first_video_frame_wall_time = None
        self.first_audio_packet_wall_time = None
        self._av_start_summary_logged = False
        self.audio_completed_at = None
        self.audio_media_seconds = None
        self.audio_playout_seconds = 0.0
        self.first_live_audio_target_seconds = None
        self.first_live_video_rtp_seconds = None
        self.first_tts_transport_pts_seconds = None
        self.audio_transport_rebase_target_seconds = None
        self.video_rtp_phase_correction_seconds = 0.0
        self.audio_rtp_phase_correction_seconds = 0.0
        self.first_live_rtp_max_mismatch_seconds = None
        self.started.clear()
        self.playout_released.clear()
        self.audio_complete.clear()
        self._coverage_changed.set()

    def deactivate(self) -> None:
        self.active = False
        self.started.set()
        self.playout_released.set()
        self._coverage_changed.set()

    def close(self) -> None:
        self._closed = True
        self.active = False
        self.started.set()
        self.playout_released.set()
        self.audio_complete.set()
        self._coverage_changed.set()

    def mark_started(self) -> None:
        if self.active and not self.started.is_set():
            self.started.set()

    def mark_audio_ready(self) -> None:
        if self.audio_ready_at is None:
            self.audio_ready_at = time.monotonic()

    def mark_video_ready(self) -> None:
        if self.video_ready_at is None:
            self.video_ready_at = time.monotonic()

    def set_turn_context(self, request_id: str, session_id: str) -> None:
        """Attach searchable request/session IDs to one A/V playout turn."""
        self.turn_request_id = str(request_id or "").strip() or None
        self.turn_session_id = str(session_id or "").strip() or None

    @staticmethod
    def _rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
        return None if value is None else round(float(value), digits)

    def _log_av_timing(
        self,
        event: str,
        *,
        event_at: Optional[float] = None,
        event_wall_time: Optional[float] = None,
        **details,
    ) -> None:
        monotonic_now = time.monotonic() if event_at is None else float(event_at)
        wall_now = time.time() if event_wall_time is None else float(event_wall_time)
        payload = {
            "event": event,
            "request_id": self.turn_request_id,
            "session_id": self.turn_session_id,
            "unix_ms": self._rounded(wall_now * 1000.0),
            "monotonic_seconds": self._rounded(monotonic_now, 6),
            "scheduled_t0_monotonic_seconds": self._rounded(
                self.playout_start_time,
                6,
            ),
            "after_gate_release_ms": self._rounded(
                (monotonic_now - self.playout_released_at) * 1000.0
                if self.playout_released_at is not None
                else None
            ),
            "after_scheduled_t0_ms": self._rounded(
                (monotonic_now - self.playout_start_time) * 1000.0
                if self.playout_start_time is not None
                else None
            ),
            **details,
        }
        print(
            "📐 WEBRTC_AV_TIMING " + json.dumps(payload, sort_keys=True),
            flush=True,
        )

    def _log_av_start_summary(self) -> None:
        if (
            self._av_start_summary_logged
            or self.first_audio_packet_at is None
            or self.first_video_frame_at is None
        ):
            return
        self._av_start_summary_logged = True
        audio_minus_video_ms = (
            self.first_audio_packet_at - self.first_video_frame_at
        ) * 1000.0
        if abs(audio_minus_video_ms) < 0.5:
            leading_media = "simultaneous"
        elif audio_minus_video_ms > 0:
            leading_media = "video"
        else:
            leading_media = "audio"
        rtp_delta_ms = None
        if (
            self.first_tts_transport_pts_seconds is not None
            and self.first_live_video_rtp_seconds is not None
        ):
            rtp_delta_ms = (
                self.first_tts_transport_pts_seconds
                - self.first_live_video_rtp_seconds
            ) * 1000.0
        self._log_av_timing(
            "av_start_summary",
            event_at=max(
                self.first_audio_packet_at,
                self.first_video_frame_at,
            ),
            event_wall_time=max(
                self.first_audio_packet_wall_time or 0.0,
                self.first_video_frame_wall_time or 0.0,
            ),
            audio_start_minus_video_start_ms=self._rounded(
                audio_minus_video_ms
            ),
            absolute_start_skew_ms=self._rounded(abs(audio_minus_video_ms)),
            leading_media=leading_media,
            audio_after_gate_release_ms=self._rounded(
                (self.first_audio_packet_at - self.playout_released_at) * 1000.0
                if self.playout_released_at is not None
                else None
            ),
            video_after_gate_release_ms=self._rounded(
                (self.first_video_frame_at - self.playout_released_at) * 1000.0
                if self.playout_released_at is not None
                else None
            ),
            first_tts_audio_rtp_seconds=self._rounded(
                self.first_tts_transport_pts_seconds,
                6,
            ),
            first_live_video_rtp_seconds=self._rounded(
                self.first_live_video_rtp_seconds,
                6,
            ),
            audio_rtp_minus_video_rtp_ms=self._rounded(rtp_delta_ms),
            rtp_aligned=self.first_live_rtp_alignment_valid(),
        )

    def release_playout(self, start_time: Optional[float] = None) -> float:
        if self.playout_start_time is None:
            now = time.monotonic()
            wall_now = time.time()
            self.playout_start_time = now if start_time is None else start_time
            self.playout_released_at = now
            self.playout_released_wall_time = wall_now
            self.playout_start_wall_time = wall_now + (
                self.playout_start_time - now
            )
            self.playout_released.set()
            self._coverage_changed.set()
            self._log_av_timing(
                "playout_gate_released",
                event_at=now,
                event_wall_time=wall_now,
                scheduled_t0_unix_ms=self._rounded(
                    self.playout_start_wall_time * 1000.0
                ),
                scheduled_delay_ms=self._rounded(
                    (self.playout_start_time - now) * 1000.0
                ),
            )
        return self.playout_start_time

    def playout_due(self) -> bool:
        return (
            self.playout_released.is_set()
            and (
                self.playout_start_time is None
                or time.monotonic() >= self.playout_start_time
            )
        )

    async def wait_for_playout_start(self, timeout: Optional[float] = None) -> Optional[float]:
        wait_timeout = 60.0 if timeout is None else timeout
        await asyncio.wait_for(self.playout_released.wait(), timeout=wait_timeout)
        if self.playout_start_time is not None:
            delay = self.playout_start_time - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        return self.playout_start_time

    def mark_first_video_frame(self) -> None:
        if self.first_video_frame_at is None:
            self.first_video_frame_at = time.monotonic()
            self.first_video_frame_wall_time = time.time()
            self._log_av_timing(
                "first_live_video_frame",
                event_at=self.first_video_frame_at,
                event_wall_time=self.first_video_frame_wall_time,
                video_rtp_seconds=self._rounded(
                    self.first_live_video_rtp_seconds,
                    6,
                ),
                source_frames=self.source_frames,
            )
            self._log_av_start_summary()

    def mark_first_audio_packet(self) -> None:
        if self.first_audio_packet_at is None:
            self.first_audio_packet_at = time.monotonic()
            self.first_audio_packet_wall_time = time.time()
            self._log_av_timing(
                "first_tts_audio_packet",
                event_at=self.first_audio_packet_at,
                event_wall_time=self.first_audio_packet_wall_time,
                audio_rtp_seconds=self._rounded(
                    self.first_tts_transport_pts_seconds,
                    6,
                ),
            )
            self._log_av_start_summary()

    def mark_audio_complete(self, media_seconds: Optional[float] = None) -> None:
        """Mark the exact media-time endpoint shared by audio and live video."""
        if self.audio_complete.is_set():
            return
        self.audio_media_seconds = (
            None if media_seconds is None else max(0.0, float(media_seconds))
        )
        if self.audio_media_seconds is not None:
            self.audio_playout_seconds = max(
                self.audio_playout_seconds,
                self.audio_media_seconds,
            )
        self.audio_completed_at = time.monotonic()
        self.audio_complete.set()
        self._coverage_changed.set()

    def mark_audio_progress(self, media_seconds: float) -> None:
        """Publish the PCM media position already emitted by the audio sender."""
        if not self.active or self.audio_complete.is_set():
            return
        self.audio_playout_seconds = max(
            self.audio_playout_seconds,
            max(0.0, float(media_seconds)),
        )
        self._coverage_changed.set()

    def publish_audio_transport_next_pts(self, media_seconds: float) -> None:
        """Publish the persistent audio RTP timestamp of its next packet."""
        self.audio_transport_next_pts_seconds = max(0.0, float(media_seconds))

    def note_first_live_rtp_alignment(
        self,
        *,
        audio_target_seconds: float,
        video_rtp_seconds: float,
        correction_seconds: float,
        max_mismatch_seconds: Optional[float] = None,
    ) -> None:
        if self.first_live_video_rtp_seconds is not None:
            return
        self.first_live_audio_target_seconds = float(audio_target_seconds)
        self.first_live_video_rtp_seconds = float(video_rtp_seconds)
        self.video_rtp_phase_correction_seconds = max(
            0.0,
            float(correction_seconds),
        )
        if max_mismatch_seconds is not None:
            self.first_live_rtp_max_mismatch_seconds = max(
                0.0,
                float(max_mismatch_seconds),
            )

    def request_audio_transport_rebase(self, target_seconds: float) -> None:
        """Request a forward-only rebase before the armed TTS source starts."""
        target = max(0.0, float(target_seconds))
        current = self.audio_transport_rebase_target_seconds
        if current is None or target > current:
            self.audio_transport_rebase_target_seconds = target

    def note_audio_transport_rebase(
        self,
        *,
        previous_seconds: float,
        rebased_seconds: float,
    ) -> None:
        """Record the actual silent-audio RTP correction applied for this turn."""
        self.audio_rtp_phase_correction_seconds = max(
            0.0,
            float(rebased_seconds) - float(previous_seconds),
        )

    def note_first_tts_transport_pts(self, media_seconds: float) -> None:
        """Record the actual RTP PTS of the first packet containing TTS PCM."""
        if self.first_tts_transport_pts_seconds is None:
            self.first_tts_transport_pts_seconds = max(0.0, float(media_seconds))

    def first_live_rtp_alignment_valid(
        self,
        max_abs_mismatch_seconds: Optional[float] = None,
    ) -> Optional[bool]:
        """Validate actual first TTS PTS against actual first-live video PTS."""
        if (
            self.first_tts_transport_pts_seconds is None
            or self.first_live_video_rtp_seconds is None
        ):
            return None
        tolerance = (
            self.first_live_rtp_max_mismatch_seconds
            if max_abs_mismatch_seconds is None
            else max(0.0, float(max_abs_mismatch_seconds))
        )
        if tolerance is None:
            return None
        mismatch = abs(
            self.first_tts_transport_pts_seconds
            - self.first_live_video_rtp_seconds
        )
        return mismatch <= tolerance + 1e-9

    async def wait_for_audio_complete(
        self,
        timeout: Optional[float] = None,
    ) -> None:
        if timeout is None:
            await self.audio_complete.wait()
            return
        await asyncio.wait_for(self.audio_complete.wait(), timeout=timeout)

    def add_frames(self, frames: int) -> None:
        if self.active and frames > 0:
            self.source_frames += frames
            self._coverage_changed.set()

    def video_time(self) -> float:
        return self.source_frames / self.source_fps

    async def wait_for_audio_coverage(
        self,
        target_seconds: float,
        timeout: Optional[float] = None,
    ) -> float:
        """Wait until emitted video covers the requested audio media time."""
        if not self.strict_fifo or not self.active or self._closed:
            return 0.0

        self.last_audio_wait_target_seconds = target_seconds
        if self.video_time() + WEBRTC_SYNC_EPSILON_SECONDS >= target_seconds:
            return 0.0

        stall_start = time.monotonic()
        self.audio_stalls += 1
        self.audio_waiting = True
        try:
            while (
                self.strict_fifo
                and self.active
                and not self._closed
                and self.video_time() + WEBRTC_SYNC_EPSILON_SECONDS < target_seconds
            ):
                self._coverage_changed.clear()
                if self.video_time() + WEBRTC_SYNC_EPSILON_SECONDS >= target_seconds:
                    break
                wait_timeout = WEBRTC_STRICT_AUDIO_WAIT_TIMEOUT_SECONDS if timeout is None else timeout
                try:
                    await asyncio.wait_for(self._coverage_changed.wait(), timeout=wait_timeout)
                except asyncio.TimeoutError:
                    print(
                        f"🔊 Strict FIFO audio wait timed out at target={target_seconds:.3f}s, "
                        f"video={self.video_time():.3f}s"
                    )
                    break
        finally:
            elapsed = time.monotonic() - stall_start
            self.audio_stall_seconds += elapsed
            self.audio_waiting = False
        return elapsed

    def note_video_stall(self, duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        self.video_stalls += 1
        self.video_stall_seconds += duration_seconds
        self._coverage_changed.set()

    def get_stats(self) -> dict:
        first_live_rtp_delta_seconds = None
        first_live_rtp_abs_mismatch_seconds = None
        if (
            self.first_tts_transport_pts_seconds is not None
            and self.first_live_video_rtp_seconds is not None
        ):
            first_live_rtp_delta_seconds = (
                self.first_tts_transport_pts_seconds
                - self.first_live_video_rtp_seconds
            )
            first_live_rtp_abs_mismatch_seconds = abs(
                first_live_rtp_delta_seconds
            )
        return {
            "sync_mode": "strict_fifo" if self.strict_fifo else "free_run",
            "audio_sync_strategy": WEBRTC_AUDIO_SYNC_STRATEGY,
            "active": self.active,
            "started": self.started.is_set(),
            "source_fps": self.source_fps,
            "source_frames": self.source_frames,
            "video_coverage_seconds": self.video_time(),
            "audio_waiting": self.audio_waiting,
            "video_waiting": self.video_waiting,
            "audio_stalls": self.audio_stalls,
            "video_stalls": self.video_stalls,
            "audio_stall_seconds": self.audio_stall_seconds,
            "video_stall_seconds": self.video_stall_seconds,
            "last_audio_wait_target_seconds": self.last_audio_wait_target_seconds,
            "audio_ready": self.audio_ready_at is not None,
            "video_ready": self.video_ready_at is not None,
            "playout_released": self.playout_released.is_set(),
            "turn_request_id": self.turn_request_id,
            "turn_session_id": self.turn_session_id,
            "playout_released_unix_ms": (
                self._rounded(self.playout_released_wall_time * 1000.0)
                if self.playout_released_wall_time is not None
                else None
            ),
            "scheduled_playout_start_unix_ms": (
                self._rounded(self.playout_start_wall_time * 1000.0)
                if self.playout_start_wall_time is not None
                else None
            ),
            "first_video_frame_unix_ms": (
                self._rounded(self.first_video_frame_wall_time * 1000.0)
                if self.first_video_frame_wall_time is not None
                else None
            ),
            "first_audio_packet_unix_ms": (
                self._rounded(self.first_audio_packet_wall_time * 1000.0)
                if self.first_audio_packet_wall_time is not None
                else None
            ),
            "audio_ready_to_release_seconds": (
                self.playout_released_at - self.audio_ready_at
                if self.playout_released_at is not None and self.audio_ready_at is not None
                else None
            ),
            "video_ready_to_release_seconds": (
                self.playout_released_at - self.video_ready_at
                if self.playout_released_at is not None and self.video_ready_at is not None
                else None
            ),
            "first_audio_packet_after_release_seconds": (
                self.first_audio_packet_at - self.playout_released_at
                if self.first_audio_packet_at is not None and self.playout_released_at is not None
                else None
            ),
            "first_video_frame_after_release_seconds": (
                self.first_video_frame_at - self.playout_released_at
                if self.first_video_frame_at is not None and self.playout_released_at is not None
                else None
            ),
            "initial_av_start_delta_seconds": (
                self.first_audio_packet_at - self.first_video_frame_at
                if self.first_audio_packet_at is not None and self.first_video_frame_at is not None
                else None
            ),
            "audio_complete": self.audio_complete.is_set(),
            "audio_media_seconds": self.audio_media_seconds,
            "audio_playout_seconds": self.audio_playout_seconds,
            "audio_transport_next_pts_seconds": (
                self.audio_transport_next_pts_seconds
            ),
            "first_live_audio_target_seconds": (
                self.first_live_audio_target_seconds
            ),
            "first_live_video_rtp_seconds": self.first_live_video_rtp_seconds,
            "first_tts_transport_pts_seconds": (
                self.first_tts_transport_pts_seconds
            ),
            "audio_transport_rebase_target_seconds": (
                self.audio_transport_rebase_target_seconds
            ),
            "video_rtp_phase_correction_seconds": (
                self.video_rtp_phase_correction_seconds
            ),
            "audio_rtp_phase_correction_seconds": (
                self.audio_rtp_phase_correction_seconds
            ),
            "first_live_rtp_delta_seconds": first_live_rtp_delta_seconds,
            "first_live_rtp_abs_mismatch_seconds": (
                first_live_rtp_abs_mismatch_seconds
            ),
            "first_live_rtp_max_mismatch_seconds": (
                self.first_live_rtp_max_mismatch_seconds
            ),
            "first_live_rtp_aligned": self.first_live_rtp_alignment_valid(),
            "audio_complete_after_release_seconds": (
                self.audio_completed_at - self.playout_released_at
                if self.audio_completed_at is not None
                and self.playout_released_at is not None
                else None
            ),
        }


class IdleVideoStreamTrack(VideoStreamTrack):
    """
    Loops a local MP4 file as a WebRTC video track.
    """

    def __init__(self, video_path: str, fps: Optional[float] = None):
        super().__init__()
        self.video_path = video_path
        self._fps = fps
        self._frame_time = None
        self._last_ts = None
        self._rtp_frame_index = 0
        self._container = None
        self._stream = None
        self._frame_iter = None
        self._position_lock = threading.Lock()
        self._source_frame_count: Optional[int] = None
        self._source_duration_seconds: Optional[float] = None
        self._next_source_frame_index = 0
        self._last_source_frame_index: Optional[int] = None
        self._last_frame_read_at: Optional[float] = None
        self._last_read_started_cycle = False
        self._completed_cycles = 0
        self._open_container()

    def _open_container(self) -> None:
        self._container = av.open(self.video_path)
        self._stream = self._container.streams.video[0]
        if self._fps is None:
            rate = self._stream.average_rate
            self._fps = float(rate) if rate else 25.0
        self._frame_time = 1.0 / float(self._fps)
        self._frame_iter = self._container.decode(self._stream)
        frame_count = int(getattr(self._stream, "frames", 0) or 0)
        duration_seconds = None
        if getattr(self._stream, "duration", None) and getattr(self._stream, "time_base", None):
            duration_seconds = float(self._stream.duration * self._stream.time_base)
        elif getattr(self._container, "duration", None):
            duration_seconds = float(self._container.duration) / 1_000_000.0
        if frame_count <= 0 and duration_seconds and self._fps:
            frame_count = int(round(duration_seconds * float(self._fps)))
        with self._position_lock:
            self._source_frame_count = frame_count if frame_count > 0 else None
            self._source_duration_seconds = duration_seconds
            self._next_source_frame_index = 0

    def _reset_container(self) -> None:
        if self._container is not None:
            self._container.close()
        self._open_container()

    def read_frame(self):
        started_cycle = False
        try:
            frame = next(self._frame_iter)
        except StopIteration:
            self._reset_container()
            frame = next(self._frame_iter)
            started_cycle = True
        with self._position_lock:
            frame_index = self._next_source_frame_index
            if self._source_frame_count:
                frame_index = frame_index % self._source_frame_count
            self._last_source_frame_index = frame_index
            self._next_source_frame_index = frame_index + 1
            self._last_frame_read_at = time.monotonic()
            self._last_read_started_cycle = started_cycle
            if started_cycle:
                self._completed_cycles += 1
        return frame.reformat(format="yuv420p")

    def next_frame_starts_cycle(self) -> bool:
        """Return whether the next decode is the first frame of a new loop.

        Container metadata normally gives us the exact frame count, allowing a
        queued pose to be installed before decoding frame zero. Some codecs do
        not expose a frame count; ``last_read_started_cycle`` is the fallback
        used by ``SwitchableVideoStreamTrack`` after the decoder reports EOF.
        """
        with self._position_lock:
            if self._last_source_frame_index is None:
                return self._next_source_frame_index == 0
            return bool(
                self._source_frame_count
                and self._next_source_frame_index >= self._source_frame_count
            )

    def last_read_started_cycle(self) -> bool:
        with self._position_lock:
            return self._last_read_started_cycle

    def get_timing(self) -> dict:
        with self._position_lock:
            frame_index = self._last_source_frame_index
            frame_count = self._source_frame_count
            duration_seconds = self._source_duration_seconds
            last_frame_read_at = self._last_frame_read_at
            fps = float(self._fps or 0.0)

        if frame_index is None:
            frame_index = 0
        if frame_count and frame_count > 0:
            frame_index = frame_index % frame_count
        elapsed_seconds = frame_index / fps if fps > 0 else 0.0
        if duration_seconds is None and frame_count and fps > 0:
            duration_seconds = frame_count / fps

        return {
            "source_frame_index": frame_index,
            "source_frame_count": frame_count,
            "source_fps": fps,
            "idle_elapsed_seconds": elapsed_seconds,
            "idle_duration_seconds": duration_seconds,
            "last_frame_read_at": last_frame_read_at,
            "completed_cycles": self._completed_cycles,
        }

    async def recv(self):
        if self._last_ts is None:
            self._last_ts = time.monotonic()
        else:
            now = time.monotonic()
            wait = self._frame_time - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.monotonic()

        frame = self.read_frame()
        pts = int(round(self._rtp_frame_index * WEBRTC_VIDEO_CLOCK_RATE / float(self._fps)))
        self._rtp_frame_index += 1
        frame.pts = pts
        frame.time_base = WEBRTC_VIDEO_TIME_BASE
        return frame

    def stop(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        super().stop()

    def reset(self) -> None:
        self._reset_container()


class LiveVideoStreamTrack(VideoStreamTrack):
    """
    Video track fed by pushed frames (e.g., inference output).
    """

    def __init__(self, fps: float = 10.0, max_queue: int = 30):
        super().__init__()
        self._fps = fps
        self._frame_time = 1.0 / float(self._fps)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._last_ts = None
        self._rtp_frame_index = 0
        self._closed = False

    async def push_bgr_frame(self, frame_bgr) -> None:
        if self._closed:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")
        await self._queue.put(frame)

    async def recv(self):
        if self._closed:
            raise asyncio.CancelledError()

        if self._last_ts is None:
            self._last_ts = time.monotonic()
        else:
            now = time.monotonic()
            wait = self._frame_time - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.monotonic()

        frame = await self._queue.get()
        pts = int(round(self._rtp_frame_index * WEBRTC_VIDEO_CLOCK_RATE / float(self._fps)))
        self._rtp_frame_index += 1
        frame.pts = pts
        frame.time_base = WEBRTC_VIDEO_TIME_BASE
        return frame

    def stop(self) -> None:
        self._closed = True
        super().stop()


class SwitchableVideoStreamTrack(VideoStreamTrack):
    """
    Single video track that switches between idle frames and live frames.
    
    HLS-like buffering behavior:
    - Prebuffering: waits for N seconds of frames before switching to live playback
    - Fixed media clock by default; optional adaptive FPS remains opt-in
    - Smooth transitions between idle and live modes
    - Repeats/holds video frames instead of speeding up/slowing down audio
    """

    def __init__(
        self,
        idle_video_path: str,
        source_fps: float = 10.0,
        output_fps: Optional[float] = None,
        max_queue: Optional[int] = None,
        sync_clock: Optional[VideoSyncClock] = None,
        prebuffer_seconds: Optional[float] = None,
        adaptive_fps: Optional[bool] = None,
        min_fps_ratio: Optional[float] = None,
        idle_pose_id: str = "default",
        on_idle_pose_changed: Optional[Callable[[str, str], None]] = None,
        idle_source_fps: Optional[float] = None,
        pose_crossfade_frames: Optional[int] = None,
    ):
        super().__init__()
        self._source_fps = float(source_fps)
        self._idle_source_fps = (
            float(idle_source_fps)
            if idle_source_fps is not None
            else self._source_fps
        )
        if self._idle_source_fps <= 0:
            self._idle_source_fps = self._source_fps
        self._output_fps = float(output_fps) if output_fps is not None else self._source_fps
        self._sync_clock = sync_clock
        self._strict_fifo = bool(getattr(sync_clock, "strict_fifo", WEBRTC_STRICT_FIFO_SYNC))
        if self._strict_fifo and self._output_fps < self._source_fps:
            print(
                f"⚠️ Strict FIFO requires output_fps >= source_fps to avoid skipping; "
                f"raising output_fps {self._output_fps:g} -> {self._source_fps:g}"
            )
            self._output_fps = self._source_fps
        if self._output_fps <= 0:
            self._output_fps = self._source_fps
        self._base_frame_time = 1.0 / float(self._output_fps)
        self._frame_time = self._base_frame_time
        self._source_step = self._source_fps / self._output_fps
        self._idle_source_step = self._idle_source_fps / self._output_fps
        self._idle_source_accum = 0.0
        self._live_output_index = 0
        self._last_live_output_frames = 0
        self._last_required_live_output_frames = 0
        self._live_source_consumed = 0
        self._live_generation_id = 0
        self._live_rtp_alignment_applied = False
        self._live_rtp_phase_correction_frames = 0
        self._last_live_rtp_phase_correction_frames = 0
        self._max_queue = WEBRTC_VIDEO_MAX_QUEUE_FRAMES if max_queue is None else max_queue
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._idle = IdleVideoStreamTrack(
            idle_video_path,
            fps=self._idle_source_fps,
        )
        self._current_idle_video_path = str(idle_video_path)
        self._current_idle_pose_id = str(idle_pose_id or "default")
        self._pending_idle_switches = deque()
        self._completion_idle_switch = None
        self._completion_idle_stage_id = 0
        self._on_idle_pose_changed = on_idle_pose_changed
        self._idle_switch_count = 0
        self._last_idle_switch_reason: Optional[str] = None
        self._idle_transition_frames = []
        self._pose_crossfade_frames = max(
            0,
            (
                WEBRTC_POSE_CROSSFADE_FRAMES
                if pose_crossfade_frames is None
                else int(pose_crossfade_frames)
            ),
        )
        self._live_active = False
        self._live_released = False
        self._last_idle_frame = None
        self._last_live_frame = None
        self._last_ts = None
        self._rtp_frame_index = 0
        self._closed = False

        # Prebuffer support - strict FIFO defaults to a deeper HLS-like live queue.
        if prebuffer_seconds is None:
            prebuffer_seconds = WEBRTC_VIDEO_PREBUFFER_SECONDS
        self._prebuffer_seconds = prebuffer_seconds
        self._prebuffer_frames = max(0, int(round(prebuffer_seconds * self._source_fps)))
        self._prebuffer_ready = asyncio.Event()
        self._frames_received = 0
        
        # Adaptive FPS support - more aggressive than before
        if adaptive_fps is None:
            adaptive_fps = WEBRTC_ADAPTIVE_FPS
        if min_fps_ratio is None:
            min_fps_ratio = WEBRTC_MIN_FPS_RATIO
        self._adaptive_fps = adaptive_fps
        self._min_fps_ratio = min_fps_ratio
        self._target_fill = WEBRTC_TARGET_QUEUE_FILL
        
        # Stats tracking
        self._frames_played = 0
        self._frames_dropped = 0
        self._frames_duplicated = 0
        self._queue_underruns = 0
        self._strict_video_stalls = 0
        self._strict_video_stall_seconds = 0.0
        self._output_frames_sent = 0
        self._slowdown_active = False
        self._current_slowdown = 1.0
        self._generation_complete = False
        self._generation_complete_event = asyncio.Event()
        self._playback_complete = asyncio.Event()
        self._playback_complete.set()
        self._reset_timing_stats()
        self._idle_sync_hold_active = False
        self._idle_sync_anchor_timing: Optional[dict] = None
        self._last_live_timing: Optional[dict] = None
        
        # Smoothing for adaptive FPS (prevents jitter)
        self._slowdown_history = []
        self._slowdown_window = 5  # Average over 5 samples
        
        print(f"🎬 SwitchableVideoStreamTrack: prebuffer={self._prebuffer_frames} frames "
              f"({prebuffer_seconds}s), adaptive_fps={adaptive_fps}, min_ratio={min_fps_ratio}, "
              f"max_queue={self._max_queue}, target_fill={self._target_fill}, "
              f"pose_crossfade_frames={self._pose_crossfade_frames}, "
              f"sync_mode={'strict_fifo' if self._strict_fifo else 'free_run'}")

    def _reset_source_timing(self) -> None:
        self._idle_source_accum = 0.0
        self._live_output_index = 0
        self._live_source_consumed = 0

    @staticmethod
    def _safe_avg(total: float, count: int) -> float:
        return total / count if count else 0.0

    def _reset_timing_stats(self) -> None:
        self._push_frames = 0
        self._push_total_s = 0.0
        self._push_max_s = 0.0
        self._push_convert_total_s = 0.0
        self._push_convert_max_s = 0.0
        self._push_queue_wait_total_s = 0.0
        self._push_queue_wait_max_s = 0.0
        self._recv_frames = 0
        self._recv_total_s = 0.0
        self._recv_max_s = 0.0
        self._recv_pace_wait_count = 0
        self._recv_pace_wait_total_s = 0.0
        self._recv_pace_wait_max_s = 0.0

    def _build_idle_transition_frames(self, old_frame, next_idle, frame_count: int) -> list:
        """Create a short crossfade from the last displayed idle frame into a new idle loop."""
        if old_frame is None or frame_count <= 0:
            return []

        frames = []
        try:
            for idx in range(frame_count):
                next_frame = next_idle.read_frame()
                new_bgr = next_frame.to_ndarray(format="bgr24")
                old_bgr = old_frame.reformat(
                    width=next_frame.width,
                    height=next_frame.height,
                    format="bgr24",
                ).to_ndarray()
                progress = float(idx + 1) / float(frame_count + 1)
                alpha = 0.5 - 0.5 * np.cos(np.pi * progress)
                blended = (
                    old_bgr.astype(np.float32) * (1.0 - alpha)
                    + new_bgr.astype(np.float32) * alpha
                )
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                frames.append(
                    av.VideoFrame.from_ndarray(blended, format="bgr24").reformat(format="yuv420p")
                )
        except Exception as exc:
            print(f"⚠️ Failed to build idle transition frames: {exc}", flush=True)
            try:
                next_idle.reset()
            except Exception:
                pass
            return []
        return frames

    def set_idle_pose_change_callback(
        self,
        callback: Optional[Callable[[str, str], None]],
    ) -> None:
        """Install a lightweight callback invoked after an idle pose activates."""
        self._on_idle_pose_changed = callback

    def _notify_idle_pose_changed(self) -> None:
        if self._on_idle_pose_changed is None:
            return
        try:
            self._on_idle_pose_changed(
                self._current_idle_pose_id,
                self._current_idle_video_path,
            )
        except Exception as exc:
            print(f"⚠️ Idle pose change callback failed: {exc}", flush=True)

    def _stop_pending_idle_switches(self) -> None:
        while self._pending_idle_switches:
            pending = self._pending_idle_switches.popleft()
            next_idle = pending.get("idle_track")
            if next_idle is not None:
                next_idle.stop()

    def clear_pending_idle_switches(self) -> int:
        """Discard queued idle decoders and return how many were removed."""
        pending_count = len(self._pending_idle_switches)
        self._stop_pending_idle_switches()
        return pending_count

    def _apply_idle_switch(
        self,
        next_idle,
        *,
        idle_video_path: str,
        pose_id: str,
        reason: str,
        transition_frames: Optional[list] = None,
    ) -> None:
        previous_idle = self._idle
        self._idle = next_idle
        self._current_idle_video_path = str(idle_video_path)
        self._current_idle_pose_id = str(pose_id or "default")
        self._idle_transition_frames = list(transition_frames or [])
        self._last_idle_frame = None
        self._idle_sync_hold_active = False
        self._idle_sync_anchor_timing = None
        self._idle_switch_count += 1
        self._last_idle_switch_reason = reason
        if previous_idle is not None:
            previous_idle.stop()
        self._notify_idle_pose_changed()

    def _activate_next_queued_idle_switch(self) -> bool:
        if not self._pending_idle_switches:
            return False
        pending = self._pending_idle_switches.popleft()
        transition_frames = []
        if self._pose_crossfade_frames > 0 and not self._live_active:
            transition_frames = self._build_idle_transition_frames(
                self._last_idle_frame,
                pending["idle_track"],
                self._pose_crossfade_frames,
            )
        self._apply_idle_switch(
            pending["idle_track"],
            idle_video_path=pending["idle_video_path"],
            pose_id=pending["pose_id"],
            reason=pending["reason"],
            transition_frames=transition_frames,
        )
        print(
            f"🎬 Activated queued idle pose={self._current_idle_pose_id} "
            f"path={self._current_idle_video_path} "
            f"crossfade_frames={len(transition_frames)} "
            f"pending={len(self._pending_idle_switches)}",
            flush=True,
        )
        return True

    async def queue_idle_video(
        self,
        idle_video_path: str,
        *,
        pose_id: str = "default",
        reason: str = "pose_protocol",
        replace_pending: bool = False,
    ) -> dict:
        """Queue an idle source for activation at the next clip boundary.

        The new decoder is opened when queued, so malformed/missing media fails
        at request time rather than in the WebRTC sender loop. When configured,
        the first frames of the new clip are replaced by a short cross-dissolve
        against the last outgoing frame without adding media-clock duration.
        """
        if self._closed:
            return {"changed": False, "queued": False, "reason": "track_closed"}

        idle_video_path = str(idle_video_path)
        pose_id = str(pose_id or "default")
        if replace_pending:
            self._stop_pending_idle_switches()

        if not self._pending_idle_switches:
            if (
                idle_video_path == self._current_idle_video_path
                and pose_id == self._current_idle_pose_id
            ):
                return {
                    "changed": False,
                    "queued": False,
                    "reason": "already_active",
                    **self.get_pose_status(),
                }
        else:
            last_pending = self._pending_idle_switches[-1]
            if (
                idle_video_path == last_pending["idle_video_path"]
                and pose_id == last_pending["pose_id"]
            ):
                return {
                    "changed": False,
                    "queued": False,
                    "reason": "already_queued",
                    **self.get_pose_status(),
                }

        next_idle = IdleVideoStreamTrack(
            idle_video_path,
            fps=self._idle_source_fps,
        )
        self._pending_idle_switches.append(
            {
                "idle_track": next_idle,
                "idle_video_path": idle_video_path,
                "pose_id": pose_id,
                "reason": str(reason or "pose_protocol"),
            }
        )
        print(
            f"🎬 Queued idle pose={pose_id} path={idle_video_path} "
            f"effective=next_boundary pending={len(self._pending_idle_switches)}",
            flush=True,
        )
        return {
            "changed": False,
            "queued": True,
            "effective": "next_boundary",
            **self.get_pose_status(),
        }

    async def switch_idle_video(
        self,
        idle_video_path: str,
        transition_seconds: float = 0.35,
        *,
        pose_id: Optional[str] = None,
        effective: str = "immediate",
        reason: str = "legacy_switch",
        replace_pending: bool = False,
    ) -> dict:
        """Switch the idle loop without renegotiating the WebRTC media track."""
        if self._closed:
            return {"changed": False, "reason": "track_closed"}

        idle_video_path = str(idle_video_path)
        target_pose_id = str(pose_id or self._current_idle_pose_id or "default")
        switch_mode = str(effective or "immediate").strip().lower()
        if switch_mode not in ("immediate", "next_boundary"):
            raise ValueError("effective must be immediate or next_boundary")
        if switch_mode == "next_boundary":
            return await self.queue_idle_video(
                idle_video_path,
                pose_id=target_pose_id,
                reason=reason,
                replace_pending=replace_pending,
            )
        if (
            idle_video_path == self._current_idle_video_path
            and target_pose_id == self._current_idle_pose_id
        ):
            return {
                "changed": False,
                "reason": "already_active",
                "idle_video_path": self._current_idle_video_path,
                "pose_id": self._current_idle_pose_id,
            }

        next_idle = None
        transition_frames = []
        try:
            next_idle = IdleVideoStreamTrack(
                idle_video_path,
                fps=self._idle_source_fps,
            )
            frame_count = max(0, int(round(float(transition_seconds or 0.0) * self._output_fps)))
            if frame_count > 0 and not self._live_active:
                transition_frames = self._build_idle_transition_frames(
                    self._last_idle_frame,
                    next_idle,
                    frame_count,
                )

            self._apply_idle_switch(
                next_idle,
                idle_video_path=idle_video_path,
                pose_id=target_pose_id,
                reason=reason,
                transition_frames=transition_frames,
            )
        except Exception:
            if next_idle is not None:
                next_idle.stop()
            raise

        print(
            f"🎬 Switched idle pose={target_pose_id} path={idle_video_path} "
            f"transition_frames={len(transition_frames)} live_active={self._live_active}",
            flush=True,
        )
        return {
            "changed": True,
            "idle_video_path": idle_video_path,
            "pose_id": target_pose_id,
            "transition_frames": len(transition_frames),
            "live_active": self._live_active,
        }

    def _advance_source(self) -> int:
        self._idle_source_accum += self._idle_source_step
        advance = int(self._idle_source_accum)
        if advance > 0:
            self._idle_source_accum -= advance
        return advance

    def _live_source_steps_for_output(self) -> int:
        desired_consumed = int(self._live_output_index * self._source_step) + 1
        steps = max(0, desired_consumed - self._live_source_consumed)
        self._live_output_index += 1
        return steps

    def _pop_live_frames(self, steps: int):
        """Pop frames from queue, returning the last frame and count popped"""
        frame = None
        popped = 0
        for _ in range(max(steps, 0)):
            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    return frame, popped
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and item[0] == "live_frame"
                ):
                    generation_id, candidate = item[1], item[2]
                    if generation_id != self._live_generation_id:
                        self._frames_dropped += 1
                        continue
                    frame = candidate
                else:
                    # Backward compatibility for callers/tests that directly
                    # seed the queue with an AV frame.
                    frame = item
                popped += 1
                break
        return frame, popped

    def _unwrap_live_queue_item(self, item):
        if (
            isinstance(item, tuple)
            and len(item) == 3
            and item[0] == "live_frame"
        ):
            generation_id, frame = item[1], item[2]
            if generation_id != self._live_generation_id:
                self._frames_dropped += 1
                return None
            return frame
        return item

    def _pop_live_frames_timestamp_locked(self):
        """Select a source frame for the receiver-visible RTP media timestamp.

        The audio RTP stream is contiguous, so its receiver-visible media clock
        advances at normal speed even if producer callbacks briefly run late or
        catch up. Drive source selection from the matching video output-frame
        horizon instead of producer-side audio callbacks. A missing inference
        frame never blocks the WebRTC sender; the last frame is held.
        """
        output_seconds = self._live_output_index / self._output_fps
        desired_consumed = max(
            1,
            int(math.floor(output_seconds * self._source_fps + 1e-9))
            + 1,
        )
        # Never discard recovered inference frames after an underrun.  When the
        # producer refills, consume at most one source frame on each output
        # opportunity; the normally duplicated output slots let a 15 fps
        # source catch back up on a 30 fps transport without jumping over lip
        # motion.
        steps = min(
            1,
            max(0, desired_consumed - self._live_source_consumed),
        )
        if self._last_live_frame is None:
            steps = max(1, steps)
        next_frame, popped = self._pop_live_frames(steps)
        if popped < steps and not self._generation_complete:
            self._queue_underruns += 1
        if popped > 1:
            self._frames_dropped += popped - 1
        if next_frame is not None or self._last_live_frame is not None:
            self._live_output_index += 1
        return next_frame, popped

    def _required_live_output_frames(self) -> int:
        """Number of live RTP video frames covering the complete audio media."""
        if self._sync_clock is None:
            return 0
        media_seconds = self._sync_clock.audio_media_seconds
        if media_seconds is None:
            return 0
        return max(
            0,
            int(math.ceil(max(0.0, media_seconds) * self._output_fps - 1e-9)),
        )

    def _audio_media_horizon_reached(self) -> bool:
        """Return true only when neutral can share the audio RTP endpoint."""
        if not self._live_active:
            return True
        if self._sync_clock is None:
            return True
        if not self._sync_clock.audio_complete.is_set():
            return False
        return self._live_output_index >= self._required_live_output_frames()

    async def _pop_live_frames_strict(self, steps: int):
        """Pop exactly the next FIFO frames, waiting instead of dropping/holding."""
        frame = None
        popped = 0
        stalled_seconds = 0.0
        for _ in range(max(steps, 0)):
            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    item = None
                if item is not None:
                    candidate = self._unwrap_live_queue_item(item)
                    if candidate is None:
                        continue
                    frame = candidate
                    popped += 1
                    break

                if (
                    self._generation_complete
                    or not self._live_active
                    or (
                        self._sync_clock is not None
                        and self._sync_clock.audio_complete.is_set()
                    )
                ):
                    return frame, popped, stalled_seconds

                self._queue_underruns += 1
                self._strict_video_stalls += 1
                stall_start = time.monotonic()
                if self._sync_clock:
                    self._sync_clock.video_waiting = True
                queue_task = asyncio.create_task(self._queue.get())
                endpoint_tasks = [
                    asyncio.create_task(self._generation_complete_event.wait())
                ]
                if self._sync_clock is not None:
                    endpoint_tasks.append(
                        asyncio.create_task(self._sync_clock.audio_complete.wait())
                    )
                try:
                    done, pending = await asyncio.wait(
                        [queue_task, *endpoint_tasks],
                        timeout=WEBRTC_STRICT_VIDEO_WAIT_TIMEOUT_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if queue_task in done:
                        candidate = self._unwrap_live_queue_item(queue_task.result())
                        if candidate is not None:
                            frame = candidate
                            popped += 1
                            break
                        continue
                    if not done:
                        print(
                            f"🎬 Strict FIFO video wait timed out "
                            f"(queue={self._queue.qsize()}, played={self._frames_played})"
                        )
                    return frame, popped, stalled_seconds
                finally:
                    for task in [queue_task, *endpoint_tasks]:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(
                        queue_task,
                        *endpoint_tasks,
                        return_exceptions=True,
                    )
                    elapsed = time.monotonic() - stall_start
                    stalled_seconds += elapsed
                    self._strict_video_stall_seconds += elapsed
                    if self._sync_clock:
                        self._sync_clock.video_waiting = False
                        self._sync_clock.note_video_stall(elapsed)

        return frame, popped, stalled_seconds

    def _advance_idle_frame(self, steps: int):
        if self._idle_sync_hold_active and self._last_idle_frame is not None:
            return self._last_idle_frame
        if self._idle_transition_frames:
            frame = self._idle_transition_frames.pop(0)
            self._last_idle_frame = frame
            return frame
        if steps <= 0 and self._last_idle_frame is not None:
            return self._last_idle_frame
        steps = max(1, steps)
        frame = self._last_idle_frame
        for _ in range(steps):
            if (
                self._pending_idle_switches
                and self._idle.next_frame_starts_cycle()
            ):
                activated = self._activate_next_queued_idle_switch()
                if activated and self._idle_transition_frames:
                    frame = self._idle_transition_frames.pop(0)
                else:
                    frame = self._idle.read_frame()
            else:
                frame = self._idle.read_frame()
            if (
                self._pending_idle_switches
                and self._idle.last_read_started_cycle()
            ):
                # Frame-count metadata is not guaranteed. If EOF was the first
                # detectable boundary, discard that old loop's frame zero and
                # emit frame zero from the queued source instead.
                activated = self._activate_next_queued_idle_switch()
                if activated and self._idle_transition_frames:
                    frame = self._idle_transition_frames.pop(0)
                else:
                    frame = self._idle.read_frame()
        self._last_idle_frame = frame
        return frame

    def capture_idle_sync_timing(
        self,
        generation_fps: float,
        cycle_frames: Optional[int] = None,
        reveal_delay_seconds: float = 0.0,
        hold: bool = True,
    ) -> dict:
        """Capture the displayed idle position and map it to a MuseTalk cycle offset."""
        idle_timing = self._idle.get_timing()
        source_fps = float(idle_timing.get("source_fps") or self._source_fps or 0.0)
        source_frame_count = idle_timing.get("source_frame_count")
        source_frame_index = int(idle_timing.get("source_frame_index") or 0)
        safe_delay = max(0.0, float(reveal_delay_seconds or 0.0))
        delay_frames = int(round(safe_delay * source_fps)) if source_fps > 0 else 0

        target_source_frame = source_frame_index + delay_frames
        if source_frame_count and source_frame_count > 0:
            target_source_frame %= int(source_frame_count)

        # Convert through media time because the idle MP4 can run at a different
        # frame rate from generated WebRTC output.
        idle_phase_seconds = (
            target_source_frame / source_fps
            if source_fps > 0
            else 0.0
        )
        offset_frames = max(
            0,
            int(round(idle_phase_seconds * float(generation_fps or 0.0))),
        )
        if cycle_frames and cycle_frames > 0:
            offset_frames %= int(cycle_frames)
        offset_seconds = (
            offset_frames / float(generation_fps)
            if generation_fps and generation_fps > 0
            else 0.0
        )

        timing = {
            "timing_source": "webrtc_idle_track",
            "mapping": "single_video_source_frame",
            "offset_seconds": offset_seconds,
            "offset_frames": offset_frames,
            "idle_source_frame_index": source_frame_index,
            "target_source_frame_index": target_source_frame,
            "idle_phase_seconds": idle_phase_seconds,
            "source_frame_count": source_frame_count,
            "source_fps": source_fps,
            "cycle_frames": cycle_frames,
            "generation_fps": generation_fps,
            "reveal_delay_seconds": safe_delay,
            "idle_elapsed_seconds": idle_timing.get("idle_elapsed_seconds"),
            "idle_duration_seconds": idle_timing.get("idle_duration_seconds"),
            "hold_enabled": bool(hold and WEBRTC_IDLE_SYNC_HOLD),
        }
        self._last_live_timing = timing
        if hold and WEBRTC_IDLE_SYNC_HOLD:
            self._idle_sync_hold_active = True
            self._idle_sync_anchor_timing = timing
        return timing

    def start_live(self) -> int:
        """Start live mode - will show idle frames until prebuffer is ready"""
        # A cancelled producer from the previous turn may have completed its
        # queue put after end_live() drained.  Rotate the ownership token and
        # clear anything left before accepting this turn's frames.
        self._live_generation_id += 1
        stale_frames = 0
        try:
            while True:
                self._queue.get_nowait()
                stale_frames += 1
        except asyncio.QueueEmpty:
            pass
        self._live_active = True
        self._live_released = False
        self._live_rtp_alignment_applied = False
        self._live_rtp_phase_correction_frames = 0
        self._last_live_frame = None
        self._reset_source_timing()
        self._frames_received = 0
        self._frames_played = 0
        self._frames_dropped = 0
        self._frames_duplicated = 0
        self._queue_underruns = 0
        self._strict_video_stalls = 0
        self._strict_video_stall_seconds = 0.0
        self._output_frames_sent = 0
        self._reset_timing_stats()
        self._prebuffer_ready.clear()
        if self._prebuffer_frames <= 0:
            self._prebuffer_ready.set()
        self._slowdown_active = False
        self._current_slowdown = 1.0
        self._generation_complete = False
        self._generation_complete_event.clear()
        self._slowdown_history = []
        self._playback_complete.clear()
        if self._sync_clock:
            self._sync_clock.reset()
        print(
            f"🎬 Live mode started generation={self._live_generation_id}, "
            f"prebuffering {self._prebuffer_frames} frames "
            f"({self._prebuffer_seconds}s), stale_drained={stale_frames}..."
        )
        return self._live_generation_id

    async def stage_completion_idle_video(
        self,
        idle_video_path: str,
        *,
        pose_id: str = "neutral_resting",
        reason: str = "audio_media_complete",
    ) -> dict:
        """Decode the post-speech idle source before playout starts.

        It is activated atomically by ``end_live`` so the first frame after the
        final audio packet cannot come from a stale listening/speaking decoder.
        """
        if self._closed:
            return {"staged": False, "reason": "track_closed"}

        # A newer staging request supersedes an older one even if its decoder
        # happens to open first. This token also lets end_live()/stop()
        # invalidate a predecode that is still running in a worker thread.
        self._completion_idle_stage_id += 1
        stage_id = self._completion_idle_stage_id

        def _open_and_predecode():
            idle_track = IdleVideoStreamTrack(
                str(idle_video_path),
                fps=self._idle_source_fps,
            )
            try:
                return idle_track, idle_track.read_frame()
            except Exception:
                idle_track.stop()
                raise

        # Decoder startup and frame-zero decode are blocking operations. Keep
        # them off the event loop so the persistent 20 ms audio RTP cadence is
        # uninterrupted even while a new turn is staged.
        predecode_task = asyncio.create_task(
            asyncio.to_thread(_open_and_predecode)
        )

        def _stop_cancelled_predecode(done_task) -> None:
            # asyncio.to_thread cannot stop its worker when the awaiting request
            # is cancelled. Retrieve and close the eventual decoder result so
            # it cannot leak a PyAV container after endpoint rollback/deletion.
            try:
                cancelled_idle, _first_frame = done_task.result()
            except (asyncio.CancelledError, Exception):
                return
            cancelled_idle.stop()

        try:
            next_idle, first_frame = await asyncio.shield(predecode_task)
        except asyncio.CancelledError:
            if self._completion_idle_stage_id == stage_id:
                self._completion_idle_stage_id += 1
            predecode_task.add_done_callback(_stop_cancelled_predecode)
            raise

        if self._closed:
            next_idle.stop()
            return {"staged": False, "reason": "track_closed"}
        if stage_id != self._completion_idle_stage_id:
            next_idle.stop()
            return {"staged": False, "reason": "staging_superseded"}

        previous = self._completion_idle_switch
        self._completion_idle_switch = {
            "idle_track": next_idle,
            "first_frame": first_frame,
            "idle_video_path": str(idle_video_path),
            "pose_id": str(pose_id or "neutral_resting"),
            "reason": str(reason or "audio_media_complete"),
        }
        if previous is not None:
            previous["idle_track"].stop()
        return {
            "staged": True,
            "idle_video_path": str(idle_video_path),
            "pose_id": str(pose_id or "neutral_resting"),
        }

    def _activate_completion_idle_video(self) -> None:
        pending = self._completion_idle_switch
        self._completion_idle_switch = None
        if pending is None:
            return
        self._stop_pending_idle_switches()
        self._apply_idle_switch(
            pending["idle_track"],
            idle_video_path=pending["idle_video_path"],
            pose_id=pending["pose_id"],
            reason=pending["reason"],
            transition_frames=[pending["first_frame"]],
        )
        print(
            f"🎬 Activated completion idle pose={pending['pose_id']} "
            f"at shared audio endpoint",
            flush=True,
        )

    def end_live(self) -> None:
        """End live mode - drain queue and return to idle"""
        # Prevent an in-flight predecode from installing itself after this
        # turn has already handed back to idle.
        self._completion_idle_stage_id += 1
        self._live_generation_id += 1
        self._last_live_output_frames = self._live_output_index
        self._last_required_live_output_frames = self._required_live_output_frames()
        self._last_live_rtp_phase_correction_frames = (
            self._live_rtp_phase_correction_frames
        )
        self._live_active = False
        self._live_released = False
        drained = 0
        try:
            while True:
                self._queue.get_nowait()
                drained += 1
        except asyncio.QueueEmpty:
            pass
        self._activate_completion_idle_video()
        self._last_live_frame = None
        self._reset_source_timing()
        self._frames_received = 0
        self._prebuffer_ready.clear()
        self._generation_complete = False
        self._generation_complete_event.set()
        if self._sync_clock:
            self._sync_clock.deactivate()
        self._idle_sync_hold_active = False
        self._idle_sync_anchor_timing = None
        self._playback_complete.set()
        print(f"🎬 Live mode ended. Played: {self._frames_played}, Dropped: {self._frames_dropped}, Drained: {drained}")

    def signal_generation_complete(
        self,
        generation_id: Optional[int] = None,
    ) -> bool:
        """Called when all frames have been pushed - allows queue to drain naturally"""
        if (
            generation_id is not None
            and int(generation_id) != self._live_generation_id
        ):
            print(
                f"🎬 Ignoring stale generation-complete callback "
                f"generation={generation_id} active={self._live_generation_id}",
                flush=True,
            )
            return False
        self._generation_complete = True
        self._generation_complete_event.set()
        if self._live_active and not self._prebuffer_ready.is_set() and self._queue.qsize() > 0:
            print(
                f"🎬 Generation complete before full prebuffer; "
                f"releasing {self._queue.qsize()} queued frames"
            )
            self._prebuffer_ready.set()
            if self._sync_clock:
                self._sync_clock.mark_video_ready()
        print(f"🎬 Generation complete signaled. Queue: {self._queue.qsize()}, Played: {self._frames_played}")
        return True

    async def wait_for_playback_complete(self, timeout: Optional[float] = None) -> None:
        """Wait until the live queue has drained and the track has returned to idle."""
        if timeout is None:
            await self._playback_complete.wait()
            return
        await asyncio.wait_for(self._playback_complete.wait(), timeout=timeout)

    async def push_bgr_frame(
        self,
        frame_bgr,
        generation_id: Optional[int] = None,
    ) -> bool:
        """Push a single BGR frame to the queue - never drops, waits if full"""
        owner_generation_id = (
            self._live_generation_id
            if generation_id is None
            else int(generation_id)
        )
        if (
            self._closed
            or not self._live_active
            or owner_generation_id != self._live_generation_id
        ):
            return False

        push_started_at = time.monotonic()
        convert_started_at = push_started_at
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")
        convert_s = time.monotonic() - convert_started_at
        return await self._push_video_frame(
            frame,
            push_started_at,
            convert_s,
            generation_id=owner_generation_id,
        )

    def _remove_queued_item_identity(self, target) -> bool:
        """Remove a late stale put without disturbing current queue ordering."""
        queue_items = getattr(self._queue, "_queue", None)
        if queue_items is None:
            return False
        for index, item in enumerate(queue_items):
            if item is target:
                del queue_items[index]
                # A successful put increments this internal counter. No code in
                # this track uses join(), but keep Queue accounting coherent.
                unfinished = getattr(self._queue, "_unfinished_tasks", 0)
                if unfinished > 0:
                    self._queue._unfinished_tasks = unfinished - 1
                self._queue._wakeup_next(self._queue._putters)
                return True
        return False

    async def _push_video_frame(
        self,
        frame,
        push_started_at: float,
        convert_s: float,
        generation_id: Optional[int] = None,
    ) -> bool:
        # Audio is the authoritative turn endpoint.  If it has already returned
        # the transport to idle, discard any late inference callbacks instead
        # of refilling an orphaned queue and blocking the GPU worker.
        owner_generation_id = (
            self._live_generation_id
            if generation_id is None
            else int(generation_id)
        )
        if (
            self._closed
            or not self._live_active
            or owner_generation_id != self._live_generation_id
        ):
            return False
        queued_item = ("live_frame", owner_generation_id, frame)
        queue_wait_started_at = time.monotonic()
        if self._strict_fifo:
            # Strict FIFO preserves every generated frame and applies backpressure
            # instead of dropping old frames when the playout buffer is full.
            await self._queue.put(queued_item)
        else:
            # Wait for space instead of dropping (MSE-like behavior)
            try:
                # Use a short timeout to avoid blocking forever
                await asyncio.wait_for(self._queue.put(queued_item), timeout=1.0)
            except asyncio.TimeoutError:
                # Only drop if absolutely necessary
                try:
                    self._queue.get_nowait()
                    self._frames_dropped += 1
                except asyncio.QueueEmpty:
                    pass
                await self._queue.put(queued_item)

        # end_live() may have run while a full queue was applying
        # backpressure.  The token makes that late item invisible to this or a
        # subsequent turn; do not count it toward the new prebuffer.
        if (
            self._closed
            or not self._live_active
            or owner_generation_id != self._live_generation_id
        ):
            self._remove_queued_item_identity(queued_item)
            return False

        queue_wait_s = time.monotonic() - queue_wait_started_at
        push_s = time.monotonic() - push_started_at
        self._push_frames += 1
        self._push_total_s += push_s
        self._push_max_s = max(self._push_max_s, push_s)
        self._push_convert_total_s += convert_s
        self._push_convert_max_s = max(self._push_convert_max_s, convert_s)
        self._push_queue_wait_total_s += queue_wait_s
        self._push_queue_wait_max_s = max(self._push_queue_wait_max_s, queue_wait_s)

        self._frames_received += 1
        
        # Check if prebuffer is ready
        if not self._prebuffer_ready.is_set() and self._frames_received >= self._prebuffer_frames:
            print(f"🎬 Prebuffer ready: {self._frames_received} frames buffered, queue: {self._queue.qsize()}/{self._max_queue}")
            self._prebuffer_ready.set()
            if self._sync_clock:
                self._sync_clock.mark_video_ready()
        elif self._prebuffer_ready.is_set() and self._frames_received > 0 and self._sync_clock:
            self._sync_clock.mark_video_ready()
        return self._prebuffer_ready.is_set()

    async def push_bgr_frames_batch(
        self,
        frames: list,
        generation_id: Optional[int] = None,
    ) -> None:
        """Push multiple BGR frames in one event-loop handoff."""
        owner_generation_id = (
            self._live_generation_id
            if generation_id is None
            else int(generation_id)
        )
        if (
            self._closed
            or not frames
            or not self._live_active
            or owner_generation_id != self._live_generation_id
        ):
            return False

        prebuffer_ready = False
        converted_frames = []
        convert_started_at = time.monotonic()
        for frame_bgr in frames:
            if self._closed:
                break
            converted_frames.append(
                av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")
            )
        total_convert_s = time.monotonic() - convert_started_at
        if not converted_frames:
            return self._prebuffer_ready.is_set()

        per_frame_convert_s = total_convert_s / len(converted_frames)
        for frame in converted_frames:
            push_started_at = time.monotonic()
            prebuffer_ready = await self._push_video_frame(
                frame,
                push_started_at=push_started_at,
                convert_s=per_frame_convert_s,
                generation_id=owner_generation_id,
            )
        return prebuffer_ready

    def _calculate_adaptive_slowdown(self) -> float:
        """
        Calculate slowdown factor based on queue depth.
        
        This implements MSE-like behavior:
        - When queue is filling up (generation faster than playback): speed up slightly
        - When queue is draining (generation slower than playback): slow down
        - Target is to maintain a stable queue level
        """
        if not self._adaptive_fps or not self._live_active or not self._prebuffer_ready.is_set():
            return 1.0
        
        queue_size = self._queue.qsize()
        fill_ratio = queue_size / self._max_queue if self._max_queue > 0 else 0
        
        # If generation is complete, don't slow down - let it play out
        if self._generation_complete:
            return 1.0
        
        # Calculate slowdown based on how far we are from target fill
        # target_fill = 0.4 means we want 40% of queue filled
        
        if fill_ratio >= self._target_fill:
            # Queue is healthy or overfull - normal speed or slightly faster
            # At 100% fill, speed up by 10%. At target fill, normal speed.
            speedup = 1.0 + (fill_ratio - self._target_fill) * 0.2
            slowdown = 1.0 / min(speedup, 1.1)  # Cap speedup at 10%
        else:
            # Queue is below target - slow down proportionally
            # At 0% fill, slow to min_fps_ratio. At target fill, normal speed.
            deficit = self._target_fill - fill_ratio
            max_deficit = self._target_fill
            slowdown_factor = deficit / max_deficit  # 0 to 1
            
            # Interpolate between 1.0 and (1.0 / min_fps_ratio)
            max_slowdown = 1.0 / self._min_fps_ratio
            slowdown = 1.0 + slowdown_factor * (max_slowdown - 1.0)
        
        # Smooth the slowdown to prevent jitter
        self._slowdown_history.append(slowdown)
        if len(self._slowdown_history) > self._slowdown_window:
            self._slowdown_history.pop(0)
        
        smoothed_slowdown = sum(self._slowdown_history) / len(self._slowdown_history)
        
        return smoothed_slowdown

    def _get_adaptive_frame_time(self) -> float:
        """Get frame time adjusted by adaptive slowdown"""
        slowdown = self._calculate_adaptive_slowdown()
        self._current_slowdown = slowdown
        
        queue_size = self._queue.qsize()
        fill_ratio = queue_size / self._max_queue if self._max_queue > 0 else 0
        
        # Log status periodically
        if self._output_frames_sent % WEBRTC_QUEUE_LOG_INTERVAL == 0 and self._output_frames_sent > 0:
            effective_fps = self._output_fps / slowdown
            status = "⚡" if slowdown <= 1.0 else ("🐢" if slowdown > 1.2 else "📊")
            print(f"{status} Queue: {queue_size}/{self._max_queue} ({fill_ratio*100:.0f}%), "
                  f"slowdown: {slowdown:.2f}x, effective_fps: {effective_fps:.1f}, "
                  f"played: {self._frames_played}, duplicated: {self._frames_duplicated}, "
                  f"gen_complete: {self._generation_complete}")
        
        # Track slowdown state changes
        was_slow = self._slowdown_active
        self._slowdown_active = slowdown > 1.05
        
        if self._slowdown_active and not was_slow:
            print(f"📉 Slowing playback: {slowdown:.2f}x (queue: {fill_ratio*100:.0f}%)")
        elif was_slow and not self._slowdown_active:
            print(f"📈 Resuming normal speed (queue: {fill_ratio*100:.0f}%)")
        
        return self._base_frame_time * slowdown

    def _stamp_video_frame(self, frame) -> None:
        pts = int(round(self._rtp_frame_index * WEBRTC_VIDEO_CLOCK_RATE / self._output_fps))
        self._rtp_frame_index += 1
        frame.pts = pts
        frame.time_base = WEBRTC_VIDEO_TIME_BASE

    def _align_first_live_rtp_to_audio(self) -> None:
        """Choose one forward-only RTP anchor for first live video and TTS."""
        if self._live_rtp_alignment_applied:
            return
        if self._sync_clock is None:
            self._live_rtp_alignment_applied = True
            return
        audio_target = self._sync_clock.audio_transport_next_pts_seconds
        if audio_target is None:
            return
        self._live_rtp_alignment_applied = True

        previous_index = self._rtp_frame_index
        target_index = max(
            previous_index,
            int(round(audio_target * self._output_fps)),
        )
        self._rtp_frame_index = target_index
        correction_frames = target_index - previous_index
        self._live_rtp_phase_correction_frames = correction_frames
        aligned_seconds = target_index / self._output_fps
        max_mismatch_seconds = 1.0 / self._output_fps
        self._sync_clock.note_first_live_rtp_alignment(
            audio_target_seconds=audio_target,
            video_rtp_seconds=aligned_seconds,
            correction_seconds=correction_frames / self._output_fps,
            max_mismatch_seconds=max_mismatch_seconds,
        )
        # If independently paced video has moved materially ahead, the audio
        # transport cannot move backward to meet it. Ask the still-silent,
        # armed audio sender to move its next PTS forward immediately before it
        # substitutes TTS PCM. Small sub-frame quantization differences are
        # intentionally left alone so normal starts keep contiguous audio RTP.
        if aligned_seconds - audio_target > max_mismatch_seconds + 1e-9:
            self._sync_clock.request_audio_transport_rebase(aligned_seconds)
        if correction_frames:
            print(
                "🎬 Aligned first live video RTP to persistent audio "
                f"target={audio_target:.6f}s video={aligned_seconds:.6f}s "
                f"forward_frames={correction_frames}",
                flush=True,
            )

    def _get_current_frame_time(self) -> float:
        """Get the current effective frame time (for stats)"""
        return self._base_frame_time * self._current_slowdown

    async def recv(self):
        if self._closed:
            raise asyncio.CancelledError()

        recv_started_at = time.monotonic()
        pace_wait_s = 0.0

        # Calculate adaptive frame time
        frame_time = self._get_adaptive_frame_time()

        # Timing control
        if self._last_ts is None:
            self._last_ts = time.monotonic()
        else:
            now = time.monotonic()
            wait = frame_time - (now - self._last_ts)
            if wait > 0.001:
                pace_wait_started_at = time.monotonic()
                await asyncio.sleep(wait)
                pace_wait_s += time.monotonic() - pace_wait_started_at
            self._last_ts = time.monotonic()

        idle_advance_frames = self._advance_source()
        frame = None
        
        if self._live_active:
            if self._audio_media_horizon_reached():
                print("🎬 Shared audio endpoint reached - returning to idle")
                self.end_live()

        if self._live_active:
            # Check if we're still prebuffering
            if self._prebuffer_frames > 0 and not self._prebuffer_ready.is_set():
                # Still prebuffering - show idle frames while we wait
                frame = self._advance_idle_frame(idle_advance_frames)
            elif (
                self._strict_fifo
                and self._sync_clock is not None
                and not self._sync_clock.playout_due()
            ):
                # The FIFO has enough video, but strict A/V mode waits until
                # audio has also been prepared. Keep showing idle frames so the
                # first live frame and first audio packet share one release point.
                self._sync_clock.mark_video_ready()
                frame = self._advance_idle_frame(idle_advance_frames)
            else:
                if not self._live_released:
                    self._live_released = True
                    self._last_ts = time.monotonic()
                # Prebuffer ready - consume live frames
                attempted_live_pop = False
                timestamp_locked = (
                    self._sync_clock is not None
                    and WEBRTC_AUDIO_SYNC_STRATEGY == "timestamp_locked"
                )
                live_steps = (
                    1
                    if timestamp_locked
                    else self._live_source_steps_for_output()
                )
                if live_steps > 0:
                    attempted_live_pop = True
                    if timestamp_locked:
                        next_frame, popped = self._pop_live_frames_timestamp_locked()
                    elif self._strict_fifo:
                        next_frame, popped, stalled_seconds = await self._pop_live_frames_strict(live_steps)
                        if stalled_seconds > 0:
                            # A strict FIFO stall is intentional buffering. Re-anchor
                            # the local video pacing so we do not burst frames after it.
                            self._last_ts = time.monotonic()
                    else:
                        next_frame, popped = self._pop_live_frames(live_steps)
                    if next_frame is not None:
                        self._last_live_frame = next_frame
                        self._frames_played += popped
                        self._live_source_consumed += popped
                        if self._sync_clock:
                            # Commit the shared RTP anchor before opening the
                            # audio start gate. The persistent audio sender is
                            # not allowed to emit TTS until this value exists.
                            if not self._live_rtp_alignment_applied:
                                self._align_first_live_rtp_to_audio()
                            self._sync_clock.mark_first_video_frame()
                            self._sync_clock.add_frames(popped)
                            self._sync_clock.mark_started()
                    elif self._generation_complete and self._queue.qsize() == 0:
                        audio_done = (
                            self._sync_clock is None
                            or self._audio_media_horizon_reached()
                        )
                        # In timestamp-locked mode the pop above may have just
                        # advanced the receiver-visible horizon by holding the
                        # last generated frame.  Emit that Nth live frame now;
                        # the top of the next recv() performs the neutral
                        # handoff.  Ending here would count N but transmit idle
                        # in its place (one-frame-early completion).
                        held_timestamp_frame = bool(
                            timestamp_locked
                            and self._last_live_frame is not None
                        )
                        if audio_done and not held_timestamp_frame:
                            print("🎬 Playback complete - returning to idle")
                            self.end_live()
                    elif self._last_live_frame is not None and (
                        not timestamp_locked and not self._strict_fifo
                    ):
                        self._queue_underruns += 1
                
                if self._live_active:
                    frame = self._last_live_frame
                    if frame is not None and (
                        not attempted_live_pop or (timestamp_locked and popped == 0)
                    ):
                        self._frames_duplicated += 1
                
                # If queue is empty and generation is complete, end live mode
                if (
                    self._live_active
                    and attempted_live_pop
                    and frame is None
                    and self._generation_complete
                    and self._queue.qsize() == 0
                    and (
                        self._sync_clock is None
                        or self._audio_media_horizon_reached()
                    )
                ):
                    print("🎬 Playback complete - returning to idle")
                    self.end_live()

        # Fallback to idle frame if no live frame available
        if frame is None:
            frame = self._advance_idle_frame(idle_advance_frames)

        if (
            self._live_active
            and self._live_released
            and self._last_live_frame is not None
            and frame is self._last_live_frame
            and not self._live_rtp_alignment_applied
        ):
            self._align_first_live_rtp_to_audio()

        self._output_frames_sent += 1
        self._stamp_video_frame(frame)
        recv_s = time.monotonic() - recv_started_at
        self._recv_frames += 1
        self._recv_total_s += recv_s
        self._recv_max_s = max(self._recv_max_s, recv_s)
        if pace_wait_s > 0.0:
            self._recv_pace_wait_count += 1
            self._recv_pace_wait_total_s += pace_wait_s
            self._recv_pace_wait_max_s = max(self._recv_pace_wait_max_s, pace_wait_s)
        return frame

    def get_pose_status(self) -> dict:
        """Return boundary-switch state without exposing decoder objects."""
        return {
            "current_pose_id": self._current_idle_pose_id,
            "current_idle_video_path": self._current_idle_video_path,
            "pending_pose_ids": [
                pending["pose_id"] for pending in self._pending_idle_switches
            ],
            "pending_pose_count": len(self._pending_idle_switches),
            "idle_switch_count": self._idle_switch_count,
            "last_idle_switch_reason": self._last_idle_switch_reason,
        }

    def get_stats(self) -> dict:
        """Get current track statistics"""
        queue_size = self._queue.qsize()
        return {
            'live_active': self._live_active,
            'live_released': self._live_released,
            'prebuffer_ready': self._prebuffer_ready.is_set(),
            'queue_size': queue_size,
            'queue_max': self._max_queue,
            'queue_fill_pct': (queue_size / self._max_queue * 100) if self._max_queue > 0 else 0,
            'frames_received': self._frames_received,
            'frames_played': self._frames_played,
            'frames_dropped': self._frames_dropped,
            'frames_duplicated': self._frames_duplicated,
            'queue_underruns': self._queue_underruns,
            'strict_video_stalls': self._strict_video_stalls,
            'strict_video_stall_seconds': self._strict_video_stall_seconds,
            'output_frames_sent': self._output_frames_sent,
            'prebuffer_seconds': self._prebuffer_seconds,
            'prebuffer_frames': self._prebuffer_frames,
            'source_fps': self._source_fps,
            'idle_source_fps': self._idle_source_fps,
            'output_fps': self._output_fps,
            'current_idle_video_path': self._current_idle_video_path,
            'current_pose_id': self._current_idle_pose_id,
            'pending_pose_ids': [
                pending["pose_id"] for pending in self._pending_idle_switches
            ],
            'pending_pose_count': len(self._pending_idle_switches),
            'idle_switch_count': self._idle_switch_count,
            'last_idle_switch_reason': self._last_idle_switch_reason,
            'idle_transition_frames': len(self._idle_transition_frames),
            'pose_crossfade_frames': self._pose_crossfade_frames,
            'sync_mode': 'strict_fifo' if self._strict_fifo else 'free_run',
            'audio_sync_strategy': WEBRTC_AUDIO_SYNC_STRATEGY,
            'live_generation_id': self._live_generation_id,
            'live_rtp_phase_correction_frames': self._live_rtp_phase_correction_frames,
            'last_live_rtp_phase_correction_frames': (
                self._last_live_rtp_phase_correction_frames
            ),
            'live_output_frames': self._live_output_index,
            'required_live_output_frames': self._required_live_output_frames(),
            'last_live_output_frames': self._last_live_output_frames,
            'last_required_live_output_frames': self._last_required_live_output_frames,
            'audio_media_horizon_reached': self._audio_media_horizon_reached(),
            'sync_clock': self._sync_clock.get_stats() if self._sync_clock else None,
            'idle_timing': self._idle.get_timing() if self._idle else None,
            'live_timing': self._last_live_timing,
            'idle_sync_hold_active': self._idle_sync_hold_active,
            'idle_sync_anchor_timing': self._idle_sync_anchor_timing,
            'adaptive_fps': self._adaptive_fps,
            'slowdown_active': self._slowdown_active,
            'current_slowdown': self._current_slowdown,
            'effective_fps': self._output_fps / self._current_slowdown,
            'generation_complete': self._generation_complete,
            'completion_idle_staged': self._completion_idle_switch is not None,
            'push_frame_count': self._push_frames,
            'avg_push_s': self._safe_avg(self._push_total_s, self._push_frames),
            'max_push_s': self._push_max_s,
            'avg_push_convert_s': self._safe_avg(self._push_convert_total_s, self._push_frames),
            'max_push_convert_s': self._push_convert_max_s,
            'avg_push_queue_wait_s': self._safe_avg(self._push_queue_wait_total_s, self._push_frames),
            'max_push_queue_wait_s': self._push_queue_wait_max_s,
            'recv_frame_count': self._recv_frames,
            'avg_recv_s': self._safe_avg(self._recv_total_s, self._recv_frames),
            'max_recv_s': self._recv_max_s,
            'recv_pace_wait_count': self._recv_pace_wait_count,
            'avg_recv_pace_wait_s': self._safe_avg(self._recv_pace_wait_total_s, self._recv_pace_wait_count),
            'max_recv_pace_wait_s': self._recv_pace_wait_max_s,
        }

    def stop(self) -> None:
        self._closed = True
        self._completion_idle_stage_id += 1
        # Invalidate every producer before making queue capacity available.
        # Draining an asyncio.Queue wakes blocked putters; once awakened, their
        # generation-token check removes the late item and returns False.
        self._live_generation_id += 1
        self._live_active = False
        self._live_released = False
        drained = 0
        try:
            while True:
                self._queue.get_nowait()
                drained += 1
        except asyncio.QueueEmpty:
            pass
        self._generation_complete = True
        self._generation_complete_event.set()
        self._prebuffer_ready.clear()
        self._playback_complete.set()
        self._stop_pending_idle_switches()
        if self._completion_idle_switch is not None:
            self._completion_idle_switch["idle_track"].stop()
            self._completion_idle_switch = None
        if self._sync_clock:
            self._sync_clock.close()
        if self._idle is not None:
            self._idle.stop()
        print(
            "🎬 SwitchableVideoStreamTrack stopped "
            f"(drained={drained}). Final stats: {self.get_stats()}"
        )
        super().stop()


# ============================================================================
# Audio Tracks
# ============================================================================

class SilenceAudioStreamTrack(MediaStreamTrack):
    """
    Persistent session audio transport.

    The track emits silence while the avatar is idle and reads staged TTS PCM
    only after the shared A/V gate opens.  Keeping this *same* track attached
    to the RTCRtpSender avoids timestamp holes when inference/prebuffering takes
    longer for a multipose turn.  Its RTP/sample counter never resets across
    idle -> speech -> idle or across consecutive turns.
    """

    kind = "audio"

    def __init__(
        self,
        sample_rate: int = 48000,
        samples: int = 960,
        sync_clock: Optional[VideoSyncClock] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples = samples
        self._timestamp = 0
        self._transport_sync_clock = sync_clock
        self._frame_time = self.samples / float(self.sample_rate)
        self._transport_start_time: Optional[float] = None
        self._frames_sent = 0
        self._armed_source: Optional["SyncedAudioStreamTrack"] = None
        self._active_source: Optional["SyncedAudioStreamTrack"] = None
        self._finishing_source: Optional["SyncedAudioStreamTrack"] = None
        self._source_sync_clock: Optional[VideoSyncClock] = None
        self._source_start_time: Optional[float] = None
        self._turns_started = 0
        self._turns_completed = 0
        self._pace_reanchors = 0
        self._pace_reanchor_seconds = 0.0
        self._source_rtp_phase_correction_samples = 0
        self._last_source_rtp_phase_correction_samples = 0
        self._last_source_stats: Optional[dict] = None
        self._closed = False
        if self._transport_sync_clock is not None:
            self._transport_sync_clock.publish_audio_transport_next_pts(0.0)

    def arm_source(
        self,
        source: "SyncedAudioStreamTrack",
        *,
        sync_clock: Optional[VideoSyncClock],
        start_time: Optional[float],
    ) -> None:
        """Stage decoded TTS without interrupting the continuous silence RTP."""
        if self._closed:
            raise MediaStreamError("Persistent audio transport is closed")
        if not source.is_prepared:
            raise RuntimeError("TTS audio must be prepared before it is armed")
        if (
            source._sample_rate != self.sample_rate
            or source._samples_per_frame != self.samples
            or source._channels != 1
        ):
            raise ValueError(
                "Persistent audio source must match the mono transport "
                f"({self.sample_rate}Hz/{self.samples} samples); got "
                f"{source._sample_rate}Hz/{source._samples_per_frame} samples/"
                f"{source._channels} channels"
            )
        if (
            self._armed_source is not None
            or self._active_source is not None
            or self._finishing_source is not None
        ):
            raise RuntimeError("Persistent audio transport already has an active turn")
        self._armed_source = source
        self._source_sync_clock = sync_clock
        self._source_start_time = start_time
        self._source_rtp_phase_correction_samples = 0
        if self._transport_sync_clock is None:
            self._transport_sync_clock = sync_clock
        if self._transport_sync_clock is not None:
            self._transport_sync_clock.publish_audio_transport_next_pts(
                self._timestamp / float(self.sample_rate)
            )
        source.signal_start(start_time=start_time)

    def cancel_source(
        self,
        source: Optional["SyncedAudioStreamTrack"] = None,
    ) -> bool:
        """Return to continuous silence immediately, normally on error/abort."""
        current = self._active_source or self._armed_source or self._finishing_source
        if current is None or (source is not None and current is not source):
            return False
        current.cancel_playout()
        self._last_source_stats = current.get_stats()
        self._last_source_rtp_phase_correction_samples = (
            self._source_rtp_phase_correction_samples
        )
        self._source_rtp_phase_correction_samples = 0
        self._armed_source = None
        self._active_source = None
        self._finishing_source = None
        self._source_sync_clock = None
        self._source_start_time = None
        return True

    def _source_is_due(self, now: float) -> bool:
        if self._armed_source is None:
            return False
        if self._source_start_time is not None and now < self._source_start_time:
            return False
        clock = self._source_sync_clock
        if clock is None:
            return True
        # Video emits frame zero first; audio begins on the next 20 ms packet.
        # This bounds start skew to one audio packet without ever pausing RTP.
        if not (clock.playout_due() and clock.started.is_set()):
            return False
        if (
            WEBRTC_AUDIO_SYNC_STRATEGY == "timestamp_locked"
            and clock.first_live_video_rtp_seconds is None
        ):
            return False
        return True

    def _apply_armed_source_rtp_rebase(self) -> None:
        """Move only still-silent audio RTP forward to the shared live anchor."""
        if self._armed_source is None:
            return
        if self._active_source is not None or self._finishing_source is not None:
            return
        clock = self._source_sync_clock
        if clock is None:
            return
        target_seconds = clock.audio_transport_rebase_target_seconds
        if target_seconds is None:
            return

        previous_pts = self._timestamp
        previous_seconds = previous_pts / float(self.sample_rate)
        tolerance = clock.first_live_rtp_max_mismatch_seconds or 0.0
        # A silence packet may have advanced after the original comparison but
        # before the first live frame opened the gate. If it is now within one
        # video frame, preserve contiguous audio instead of creating a tiny gap.
        if target_seconds - previous_seconds <= tolerance + 1e-9:
            return

        target_pts = int(round(target_seconds * self.sample_rate))
        rebased_pts = max(previous_pts, target_pts)
        if rebased_pts <= previous_pts:
            return
        self._timestamp = rebased_pts
        correction_samples = rebased_pts - previous_pts
        self._source_rtp_phase_correction_samples = correction_samples
        rebased_seconds = rebased_pts / float(self.sample_rate)
        clock.note_audio_transport_rebase(
            previous_seconds=previous_seconds,
            rebased_seconds=rebased_seconds,
        )
        clock.publish_audio_transport_next_pts(rebased_seconds)
        print(
            "🔊 Rebased silent audio RTP to first live video "
            f"from={previous_seconds:.6f}s to={rebased_seconds:.6f}s "
            f"forward_samples={correction_samples}",
            flush=True,
        )

    async def _pace(self) -> float:
        now = time.monotonic()
        if self._transport_start_time is None:
            self._transport_start_time = now
            return now
        target = self._transport_start_time + self._frames_sent * self._frame_time
        lateness = now - target
        if lateness > 0.002:
            # A delayed event-loop turn must become a real transport delay, not
            # a burst of back-to-back audio producer callbacks. The RTP sample
            # clock remains contiguous; only its wall-clock pacing is re-anchored.
            self._transport_start_time = now - self._frames_sent * self._frame_time
            target = now
            self._pace_reanchors += 1
            self._pace_reanchor_seconds += lateness
        wait = target - now
        if wait > 0.001:
            await asyncio.sleep(wait)
        return time.monotonic()

    async def recv(self):
        if self._closed:
            raise MediaStreamError("Persistent audio transport stopped")
        now = await self._pace()

        # The previous call returned the final TTS packet.  Complete the media
        # turn on this next 20 ms transport tick, exactly when the persistent
        # sender resumes silence.  This prevents video from returning to idle
        # before the final packet has left the sender.
        if self._finishing_source is not None:
            completed_source = self._finishing_source
            completed_source.complete_transport_playout()
            self._turns_completed += 1
            self._last_source_stats = completed_source.get_stats()
            self._last_source_rtp_phase_correction_samples = (
                self._source_rtp_phase_correction_samples
            )
            self._source_rtp_phase_correction_samples = 0
            self._finishing_source = None
            self._active_source = None
            self._source_sync_clock = None
            self._source_start_time = None

        if self._armed_source is not None and self._source_is_due(now):
            self._apply_armed_source_rtp_rebase()
            self._active_source = self._armed_source
            self._armed_source = None
            self._turns_started += 1

        audio_bytes: Optional[bytes] = None
        if self._active_source is not None:
            audio_bytes, is_final = self._active_source.read_samples_for_transport(
                self.samples,
            )
            self._active_source.note_transport_frame(
                transport_pts=self._timestamp,
                emitted_at=now,
                is_final=is_final,
            )
            if is_final:
                self._finishing_source = self._active_source

        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples)
        if audio_bytes is None:
            audio_bytes = b"\x00" * frame.planes[0].buffer_size
        frame.planes[0].update(audio_bytes)
        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += self.samples
        self._frames_sent += 1
        if self._transport_sync_clock is not None:
            self._transport_sync_clock.publish_audio_transport_next_pts(
                self._timestamp / float(self.sample_rate)
            )

        return frame

    def get_stats(self) -> dict:
        current = self._active_source or self._armed_source or self._finishing_source
        return {
            "transport": "persistent_timestamp_locked",
            "sample_rate": self.sample_rate,
            "samples_per_frame": self.samples,
            "transport_frames_sent": self._frames_sent,
            "transport_pts": self._timestamp,
            "transport_seconds": self._timestamp / float(self.sample_rate),
            "transport_sync_clock_attached": self._transport_sync_clock is not None,
            "source_armed": self._armed_source is not None,
            "source_active": self._active_source is not None,
            "source_finishing": self._finishing_source is not None,
            "turns_started": self._turns_started,
            "turns_completed": self._turns_completed,
            "pace_reanchors": self._pace_reanchors,
            "pace_reanchor_seconds": self._pace_reanchor_seconds,
            "source_rtp_phase_correction_seconds": (
                self._source_rtp_phase_correction_samples
                / float(self.sample_rate)
            ),
            "last_source_rtp_phase_correction_seconds": (
                self._last_source_rtp_phase_correction_samples
                / float(self.sample_rate)
            ),
            "current_source": current.get_stats() if current is not None else None,
            "last_source": self._last_source_stats,
        }

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel_source()
        super().stop()


def _convert_audio_with_ffmpeg(
    input_path: str,
    sample_rate: int = 48000,
    channels: int = 1,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert audio to optimal format for WebRTC using FFmpeg.
    """
    input_path = Path(input_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else input_path.parent / f"{input_path.stem}_webrtc.wav"
    )
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-af", "aresample=resampler=soxr:precision=33:dither_method=triangular",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            cmd_simple = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", str(sample_rate), "-ac", str(channels),
                "-c:a", "pcm_s16le", "-f", "wav", str(output_path)
            ]
            result = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"🔊 FFmpeg converted: {input_path.name} -> {output_path.name}")
        return str(output_path)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found")


class SyncedAudioStreamTrack(MediaStreamTrack):
    """
    High-quality audio track synced with video generation.
    """

    kind = "audio"

    def __init__(
        self,
        audio_path: str,
        sample_rate: int = 48000,
        samples_per_frame: int = 960,
        use_stereo: bool = False,
        use_ffmpeg_convert: bool = True,
        sync_clock: Optional[VideoSyncClock] = None,
    ):
        super().__init__()
        self._original_audio_path = audio_path
        self._audio_path = audio_path
        self._sample_rate = sample_rate
        self._samples_per_frame = samples_per_frame
        self._frame_duration = samples_per_frame / sample_rate
        self._channels = 2 if use_stereo else 1
        self._layout = "stereo" if use_stereo else "mono"
        self._use_ffmpeg = use_ffmpeg_convert
        self._sync_clock = sync_clock
        
        try:
            self._max_audio_lead = max(0.0, float(os.getenv("WEBRTC_AUDIO_MAX_LEAD_SECONDS", "0.15")))
        except ValueError:
            self._max_audio_lead = 0.15
        try:
            self._max_audio_lag = max(0.0, float(os.getenv("WEBRTC_AUDIO_MAX_LAG_SECONDS", "0.25")))
        except ValueError:
            self._max_audio_lag = 0.25
        
        self._timestamp = 0
        self._started = asyncio.Event()
        self._stopped = False
        self._bytes_per_sample = 2
        self._start_time: Optional[float] = None
        self._start_signal_time: Optional[float] = None
        self._playout_start_time: Optional[float] = None
        self._first_packet_at: Optional[float] = None
        self._first_transport_pts: Optional[int] = None
        self._frames_sent = 0
        self._eof = False
        self._eof_event = asyncio.Event()
        
        self._audio_samples: bytes = b""
        self._read_position = 0
        self._fully_loaded = False
        self._load_lock = asyncio.Lock()
        self._converted_path: Optional[str] = None
        if self._use_ffmpeg:
            input_path = Path(self._original_audio_path)
            self._converted_path = str(
                input_path.parent
                / f"{input_path.stem}_{uuid.uuid4().hex[:10]}_webrtc.wav"
            )
        self._source_info = {}
        self._last_drift_seconds: Optional[float] = None
        self._drift_log_interval = int(os.getenv("WEBRTC_AUDIO_DRIFT_LOG_INTERVAL", "100"))
        self._strict_audio_stalls = 0
        self._strict_audio_stall_seconds = 0.0
        self._prepare_started_at: Optional[float] = None
        self._prepare_finished_at: Optional[float] = None

    @property
    def is_prepared(self) -> bool:
        return self._fully_loaded

    @property
    def media_duration_seconds(self) -> float:
        bytes_per_sample_frame = self._bytes_per_sample * self._channels
        if bytes_per_sample_frame <= 0:
            return 0.0
        sample_frames = len(self._audio_samples) / float(bytes_per_sample_frame)
        return sample_frames / float(self._sample_rate)

    def signal_start(self, start_time: Optional[float] = None):
        self._start_signal_time = time.monotonic()
        if start_time is None and self._sync_clock is not None:
            start_time = self._sync_clock.playout_start_time
        self._playout_start_time = start_time
        self._started.set()
        if start_time is None:
            print(f"🔊 SyncedAudioStreamTrack: Start signaled at {self._start_signal_time:.3f}")
        else:
            print(
                f"🔊 SyncedAudioStreamTrack: Start signaled at {self._start_signal_time:.3f}, "
                f"t0={start_time:.3f}"
            )

    async def prepare(self) -> None:
        """Decode/convert audio before the shared A/V playout gate opens."""
        await self._load_audio_async()
        if self._sync_clock is not None:
            self._sync_clock.mark_audio_ready()

    async def _load_audio_async(self):
        if self._fully_loaded:
            if self._sync_clock is not None:
                self._sync_clock.mark_audio_ready()
            return
            
        async with self._load_lock:
            if self._fully_loaded:
                if self._sync_clock is not None:
                    self._sync_clock.mark_audio_ready()
                return
            
            loop = asyncio.get_event_loop()
            if self._prepare_started_at is None:
                self._prepare_started_at = time.monotonic()
            
            if self._use_ffmpeg:
                conversion_future = None
                try:
                    conversion_future = loop.run_in_executor(
                        None, _convert_audio_with_ffmpeg,
                        self._original_audio_path,
                        self._sample_rate,
                        self._channels,
                        self._converted_path,
                    )
                    converted_path = await asyncio.shield(conversion_future)
                    self._audio_path = converted_path
                except asyncio.CancelledError:
                    # The executor subprocess cannot be cancelled. Register a
                    # completion callback so a file written after endpoint
                    # rollback cannot be orphaned.
                    converted_path = self._converted_path

                    def _remove_cancelled_conversion(_future):
                        if converted_path:
                            try:
                                Path(converted_path).unlink(missing_ok=True)
                            except OSError:
                                pass

                    if conversion_future is not None:
                        conversion_future.add_done_callback(
                            _remove_cancelled_conversion
                        )
                    raise
                except Exception as e:
                    print(f"⚠️ FFmpeg conversion failed: {e}")
                    if self._converted_path:
                        Path(self._converted_path).unlink(missing_ok=True)
            
            self._audio_samples = await loop.run_in_executor(None, self._load_pcm_audio)
            self._fully_loaded = True
            self._prepare_finished_at = time.monotonic()
            if self._sync_clock is not None:
                self._sync_clock.mark_audio_ready()
            
            duration_ms = len(self._audio_samples) / (self._bytes_per_sample * self._channels) / self._sample_rate * 1000
            prepare_ms = (
                (self._prepare_finished_at - self._prepare_started_at) * 1000
                if self._prepare_started_at is not None
                else 0.0
            )
            print(
                f"🔊 Audio loaded: {len(self._audio_samples)} bytes, "
                f"{duration_ms:.0f}ms media, prepare={prepare_ms:.0f}ms"
            )

    def _load_pcm_audio(self) -> bytes:
        audio_path = Path(self._audio_path)
        if audio_path.suffix.lower() == '.wav' and '_webrtc' in audio_path.stem:
            return self._load_wav_pcm()
        return self._decode_with_pyav()

    def _load_wav_pcm(self) -> bytes:
        try:
            with wave.open(self._audio_path, 'rb') as wf:
                self._source_info = {
                    'sample_rate': wf.getframerate(),
                    'channels': wf.getnchannels(),
                    'source': 'ffmpeg_wav'
                }
                return wf.readframes(wf.getnframes())
        except Exception as e:
            print(f"⚠️ WAV load failed: {e}")
            return self._decode_with_pyav()

    def _decode_with_pyav(self) -> bytes:
        result = bytearray()
        try:
            container = av.open(self._audio_path)
            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout=self._layout, rate=self._sample_rate)
            
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    for rf in resampler.resample(frame):
                        if rf:
                            result.extend(bytes(rf.planes[0]))
            
            for rf in resampler.resample(None):
                if rf:
                    result.extend(bytes(rf.planes[0]))
            
            container.close()
        except Exception as e:
            print(f"⚠️ PyAV decode error: {e}")
        return bytes(result)

    def _mark_eof(self) -> None:
        if self._eof:
            return
        self._eof = True
        self._eof_event.set()
        media_seconds = self.media_duration_seconds
        if self._sync_clock is not None:
            self._sync_clock.mark_audio_complete(media_seconds)
        print(
            f"🔊 Audio EOF after {self._frames_sent} frames "
            f"({media_seconds:.3f}s media)"
        )

    def _get_samples(self, num_samples: int) -> bytes:
        bytes_per_frame = num_samples * self._bytes_per_sample * self._channels
        
        if self._read_position + bytes_per_frame <= len(self._audio_samples):
            result = self._audio_samples[self._read_position:self._read_position + bytes_per_frame]
            self._read_position += bytes_per_frame
        else:
            remaining = self._audio_samples[self._read_position:]
            padding = bytes_per_frame - len(remaining)
            result = remaining + (b"\x00" * padding)
            self._read_position = len(self._audio_samples)
        return result

    def read_samples_for_transport(self, num_samples: int) -> tuple[bytes, bool]:
        """Read one packet for the persistent session RTP transport."""
        if not self._fully_loaded:
            raise RuntimeError("TTS audio is not prepared")
        samples = self._get_samples(num_samples)
        return samples, self._read_position >= len(self._audio_samples)

    def note_transport_frame(
        self,
        *,
        transport_pts: int,
        emitted_at: float,
        is_final: bool,
    ) -> None:
        del is_final
        if self._first_transport_pts is None:
            self._first_transport_pts = int(transport_pts)
            if self._sync_clock is not None:
                self._sync_clock.note_first_tts_transport_pts(
                    self._first_transport_pts / float(self._sample_rate)
                )
        if self._first_packet_at is None:
            self._first_packet_at = emitted_at
            if self._sync_clock is not None:
                self._sync_clock.mark_first_audio_packet()
        if self._sync_clock is not None:
            # Publish the packet's media *start* timestamp. Publishing its end
            # before the packet is played can move video one 20 ms packet ahead.
            self._sync_clock.mark_audio_progress(
                min(
                    self.media_duration_seconds,
                    self._frames_sent * self._frame_duration,
                )
            )
        self._timestamp += self._samples_per_frame
        self._frames_sent += 1

    def complete_transport_playout(self) -> None:
        """Mark EOF after the final packet has occupied one full RTP tick."""
        self._mark_eof()

    def cancel_playout(self) -> None:
        """Unblock turn cleanup and force the shared video clock back to idle."""
        self._mark_eof()

    async def recv(self):
        if self._stopped:
            raise MediaStreamError("Track stopped")

        try:
            await asyncio.wait_for(self._started.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            raise MediaStreamError("Timeout waiting for start")
        if not self._fully_loaded:
            await self._load_audio_async()

        if self._sync_clock is not None:
            if self._sync_clock.strict_fifo:
                try:
                    playout_start = await self._sync_clock.wait_for_playout_start(timeout=60.0)
                    if self._playout_start_time is None:
                        self._playout_start_time = playout_start
                except asyncio.TimeoutError:
                    raise MediaStreamError("Timeout waiting for A/V playout release")
            try:
                await asyncio.wait_for(self._sync_clock.started.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                raise MediaStreamError("Timeout waiting for video")

        if (
            self._sync_clock is not None
            and self._sync_clock.strict_fifo
            and WEBRTC_AUDIO_SYNC_STRATEGY in (
                "coverage_wait",
                "strict_fifo_coverage",
                "legacy_coverage",
            )
        ):
            packet_end_time = (self._frames_sent + 1) * self._frame_duration
            stalled_seconds = await self._sync_clock.wait_for_audio_coverage(packet_end_time)
            if stalled_seconds > 0:
                self._strict_audio_stalls += 1
                self._strict_audio_stall_seconds += stalled_seconds
                if self._start_time is not None:
                    # Insert silence-free buffering time instead of catching up in
                    # a burst, which would sound like audio speedup.
                    self._start_time = time.monotonic() - (self._frames_sent * self._frame_duration)

        if self._start_time is None:
            now = time.monotonic()
            playout_t0 = self._playout_start_time or now
            self._start_time = max(playout_t0, now)
            print(
                f"🔊 SyncedAudioStreamTrack: Playout started at {now:.3f} "
                f"(t0={playout_t0:.3f})"
            )
        else:
            target = self._start_time + (self._frames_sent * self._frame_duration)
            wait = target - time.monotonic()
            if wait > 0.002:
                await asyncio.sleep(wait)

        if self._sync_clock is not None and self._sync_clock.active:
            video_time = self._sync_clock.video_time()
            audio_time = self._frames_sent * self._frame_duration
            drift = audio_time - video_time
            self._last_drift_seconds = drift
            if (
                self._drift_log_interval > 0
                and self._frames_sent > 0
                and self._frames_sent % self._drift_log_interval == 0
                and abs(drift) > max(self._max_audio_lead, self._max_audio_lag)
            ):
                print(f"🔊 Audio/video drift observed: {drift:.3f}s (audio not retimed)")

        audio_bytes = self._get_samples(self._samples_per_frame)
        is_final = self._read_position >= len(self._audio_samples)

        frame = av.AudioFrame(format="s16", layout=self._layout, samples=self._samples_per_frame)
        frame.planes[0].update(audio_bytes)
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = fractions.Fraction(1, self._sample_rate)

        if self._first_transport_pts is None:
            self._first_transport_pts = int(frame.pts)
            if self._sync_clock is not None:
                self._sync_clock.note_first_tts_transport_pts(
                    self._first_transport_pts / float(self._sample_rate)
                )

        self._timestamp += self._samples_per_frame
        self._frames_sent += 1
        if self._first_packet_at is None:
            self._first_packet_at = time.monotonic()
            if self._sync_clock is not None:
                self._sync_clock.mark_first_audio_packet()
        if self._sync_clock is not None:
            self._sync_clock.mark_audio_progress(
                min(
                    self.media_duration_seconds,
                    max(0, self._frames_sent - 1) * self._frame_duration,
                )
            )
        if is_final:
            self._mark_eof()
        return frame

    def get_stats(self) -> dict:
        prepare_seconds = (
            self._prepare_finished_at - self._prepare_started_at
            if self._prepare_finished_at is not None and self._prepare_started_at is not None
            else None
        )
        return {
            "sample_rate": self._sample_rate,
            "samples_per_frame": self._samples_per_frame,
            "frames_sent": self._frames_sent,
            "audio_seconds_sent": self._frames_sent * self._frame_duration,
            "first_tts_transport_pts_seconds": (
                self._first_transport_pts / float(self._sample_rate)
                if self._first_transport_pts is not None
                else None
            ),
            "eof": self._eof,
            "last_drift_seconds": self._last_drift_seconds,
            "fully_loaded": self._fully_loaded,
            "prepare_started": self._prepare_started_at is not None,
            "prepare_seconds": prepare_seconds,
            "start_signaled": self._started.is_set(),
            "playout_start_time_set": self._playout_start_time is not None,
            "first_packet_after_signal_seconds": (
                self._first_packet_at - self._start_signal_time
                if self._first_packet_at is not None and self._start_signal_time is not None
                else None
            ),
            "sync_mode": (
                "strict_fifo"
                if self._sync_clock is not None and self._sync_clock.strict_fifo
                else "free_run"
            ),
            "audio_sync_strategy": WEBRTC_AUDIO_SYNC_STRATEGY,
            "media_duration_seconds": self.media_duration_seconds,
            "strict_audio_stalls": self._strict_audio_stalls,
            "strict_audio_stall_seconds": self._strict_audio_stall_seconds,
            "sync_clock": self._sync_clock.get_stats() if self._sync_clock else None,
        }

    async def wait_for_eof(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            await self._eof_event.wait()
            return
        await asyncio.wait_for(self._eof_event.wait(), timeout=timeout)

    def stop(self):
        self._stopped = True
        self._started.set()
        self.cancel_playout()
        self._audio_samples = b""
        
        if self._converted_path and os.path.exists(self._converted_path):
            try:
                os.remove(self._converted_path)
                print(f"🧹 Cleaned up: {self._converted_path}")
            except Exception:
                pass
        
        print(f"🔊 SyncedAudioStreamTrack stopped after {self._frames_sent} frames")
