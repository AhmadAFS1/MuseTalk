"""
WebRTC media tracks for MuseTalk.

Classes:
- Video tracks: IdleVideoStreamTrack, SwitchableVideoStreamTrack, LiveVideoStreamTrack
- Audio tracks: SilenceAudioStreamTrack, SyncedAudioStreamTrack
"""

import asyncio
import fractions
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional

import av
from aiortc import VideoStreamTrack, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

# Environment configuration
WEBRTC_SYNC_MODE = os.getenv("WEBRTC_SYNC_MODE", "strict_fifo").strip().lower()
WEBRTC_STRICT_FIFO_SYNC = WEBRTC_SYNC_MODE in ("strict_fifo", "fifo", "hls_like", "hls-like", "hls")
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
        self.started.clear()
        self.playout_released.clear()
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

    def release_playout(self, start_time: Optional[float] = None) -> float:
        if self.playout_start_time is None:
            now = time.monotonic()
            self.playout_start_time = now if start_time is None else start_time
            self.playout_released_at = now
            self.playout_released.set()
            self._coverage_changed.set()
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

    def mark_first_audio_packet(self) -> None:
        if self.first_audio_packet_at is None:
            self.first_audio_packet_at = time.monotonic()

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
        return {
            "sync_mode": "strict_fifo" if self.strict_fifo else "free_run",
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
        self._open_container()

    def _open_container(self) -> None:
        self._container = av.open(self.video_path)
        self._stream = self._container.streams.video[0]
        if self._fps is None:
            rate = self._stream.average_rate
            self._fps = float(rate) if rate else 25.0
        self._frame_time = 1.0 / float(self._fps)
        self._frame_iter = self._container.decode(self._stream)

    def _reset_container(self) -> None:
        if self._container is not None:
            self._container.close()
        self._open_container()

    def read_frame(self):
        try:
            frame = next(self._frame_iter)
        except StopIteration:
            self._reset_container()
            frame = next(self._frame_iter)
        return frame.reformat(format="yuv420p")

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
    ):
        super().__init__()
        self._source_fps = float(source_fps)
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
        self._source_accum = 0.0
        self._live_output_index = 0
        self._live_source_consumed = 0
        self._max_queue = WEBRTC_VIDEO_MAX_QUEUE_FRAMES if max_queue is None else max_queue
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._idle = IdleVideoStreamTrack(idle_video_path, fps=source_fps)
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
        self._playback_complete = asyncio.Event()
        self._playback_complete.set()
        self._reset_timing_stats()
        
        # Smoothing for adaptive FPS (prevents jitter)
        self._slowdown_history = []
        self._slowdown_window = 5  # Average over 5 samples
        
        print(f"🎬 SwitchableVideoStreamTrack: prebuffer={self._prebuffer_frames} frames "
              f"({prebuffer_seconds}s), adaptive_fps={adaptive_fps}, min_ratio={min_fps_ratio}, "
              f"max_queue={self._max_queue}, target_fill={self._target_fill}, "
              f"sync_mode={'strict_fifo' if self._strict_fifo else 'free_run'}")

    def _reset_source_timing(self) -> None:
        self._source_accum = 0.0
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

    def _advance_source(self) -> int:
        self._source_accum += self._source_step
        advance = int(self._source_accum)
        if advance > 0:
            self._source_accum -= advance
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
            try:
                frame = self._queue.get_nowait()
                popped += 1
            except asyncio.QueueEmpty:
                break
        return frame, popped

    async def _pop_live_frames_strict(self, steps: int):
        """Pop exactly the next FIFO frames, waiting instead of dropping/holding."""
        frame = None
        popped = 0
        stalled_seconds = 0.0
        for _ in range(max(steps, 0)):
            try:
                frame = self._queue.get_nowait()
                popped += 1
                continue
            except asyncio.QueueEmpty:
                pass

            if self._generation_complete:
                break

            self._queue_underruns += 1
            self._strict_video_stalls += 1
            stall_start = time.monotonic()
            if self._sync_clock:
                self._sync_clock.video_waiting = True
            try:
                frame = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=WEBRTC_STRICT_VIDEO_WAIT_TIMEOUT_SECONDS,
                )
                popped += 1
            except asyncio.TimeoutError:
                print(
                    f"🎬 Strict FIFO video wait timed out "
                    f"(queue={self._queue.qsize()}, played={self._frames_played})"
                )
                break
            finally:
                elapsed = time.monotonic() - stall_start
                stalled_seconds += elapsed
                self._strict_video_stall_seconds += elapsed
                if self._sync_clock:
                    self._sync_clock.video_waiting = False
                    self._sync_clock.note_video_stall(elapsed)

        return frame, popped, stalled_seconds

    def _advance_idle_frame(self, steps: int):
        if steps <= 0 and self._last_idle_frame is not None:
            return self._last_idle_frame
        steps = max(1, steps)
        frame = self._last_idle_frame
        for _ in range(steps):
            frame = self._idle.read_frame()
        self._last_idle_frame = frame
        return frame

    def start_live(self) -> None:
        """Start live mode - will show idle frames until prebuffer is ready"""
        self._live_active = True
        self._live_released = False
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
        self._slowdown_history = []
        self._playback_complete.clear()
        if self._sync_clock:
            self._sync_clock.reset()
        print(f"🎬 Live mode started, prebuffering {self._prebuffer_frames} frames ({self._prebuffer_seconds}s)...")

    def end_live(self) -> None:
        """End live mode - drain queue and return to idle"""
        self._live_active = False
        self._live_released = False
        drained = 0
        try:
            while True:
                self._queue.get_nowait()
                drained += 1
        except asyncio.QueueEmpty:
            pass
        self._last_live_frame = None
        self._reset_source_timing()
        self._frames_received = 0
        self._prebuffer_ready.clear()
        self._generation_complete = False
        if self._sync_clock:
            self._sync_clock.deactivate()
        self._playback_complete.set()
        print(f"🎬 Live mode ended. Played: {self._frames_played}, Dropped: {self._frames_dropped}, Drained: {drained}")

    def signal_generation_complete(self) -> None:
        """Called when all frames have been pushed - allows queue to drain naturally"""
        self._generation_complete = True
        if self._live_active and not self._prebuffer_ready.is_set() and self._queue.qsize() > 0:
            print(
                f"🎬 Generation complete before full prebuffer; "
                f"releasing {self._queue.qsize()} queued frames"
            )
            self._prebuffer_ready.set()
            if self._sync_clock:
                self._sync_clock.mark_video_ready()
        print(f"🎬 Generation complete signaled. Queue: {self._queue.qsize()}, Played: {self._frames_played}")

    async def wait_for_playback_complete(self, timeout: Optional[float] = None) -> None:
        """Wait until the live queue has drained and the track has returned to idle."""
        if timeout is None:
            await self._playback_complete.wait()
            return
        await asyncio.wait_for(self._playback_complete.wait(), timeout=timeout)

    async def push_bgr_frame(self, frame_bgr) -> bool:
        """Push a single BGR frame to the queue - never drops, waits if full"""
        if self._closed:
            return False

        push_started_at = time.monotonic()
        convert_started_at = push_started_at
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")
        convert_s = time.monotonic() - convert_started_at

        queue_wait_started_at = time.monotonic()
        if self._strict_fifo:
            # Strict FIFO preserves every generated frame and applies backpressure
            # instead of dropping old frames when the playout buffer is full.
            await self._queue.put(frame)
        else:
            # Wait for space instead of dropping (MSE-like behavior)
            try:
                # Use a short timeout to avoid blocking forever
                await asyncio.wait_for(self._queue.put(frame), timeout=1.0)
            except asyncio.TimeoutError:
                # Only drop if absolutely necessary
                try:
                    self._queue.get_nowait()
                    self._frames_dropped += 1
                except asyncio.QueueEmpty:
                    pass
                await self._queue.put(frame)

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

    async def push_bgr_frames_batch(self, frames: list) -> None:
        """Push multiple BGR frames at once"""
        if self._closed or not frames:
            return
        
        for frame_bgr in frames:
            await self.push_bgr_frame(frame_bgr)

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
                live_steps = self._live_source_steps_for_output()
                if live_steps > 0:
                    attempted_live_pop = True
                    if self._strict_fifo:
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
                            self._sync_clock.mark_first_video_frame()
                            self._sync_clock.add_frames(popped)
                            self._sync_clock.mark_started()
                    elif self._generation_complete and self._queue.qsize() == 0:
                        print("🎬 Playback complete - returning to idle")
                        self.end_live()
                    elif self._last_live_frame is not None and not self._strict_fifo:
                        self._queue_underruns += 1
                
                if self._live_active:
                    frame = self._last_live_frame
                    if frame is not None and not attempted_live_pop:
                        self._frames_duplicated += 1
                
                # If queue is empty and generation is complete, end live mode
                if (
                    self._live_active
                    and attempted_live_pop
                    and frame is None
                    and self._generation_complete
                    and self._queue.qsize() == 0
                ):
                    print("🎬 Playback complete - returning to idle")
                    self.end_live()

        # Fallback to idle frame if no live frame available
        if frame is None:
            frame = self._advance_idle_frame(idle_advance_frames)

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
            'prebuffer_frames': self._prebuffer_frames,
            'source_fps': self._source_fps,
            'output_fps': self._output_fps,
            'sync_mode': 'strict_fifo' if self._strict_fifo else 'free_run',
            'sync_clock': self._sync_clock.get_stats() if self._sync_clock else None,
            'adaptive_fps': self._adaptive_fps,
            'slowdown_active': self._slowdown_active,
            'current_slowdown': self._current_slowdown,
            'effective_fps': self._output_fps / self._current_slowdown,
            'generation_complete': self._generation_complete,
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
        self._playback_complete.set()
        if self._sync_clock:
            self._sync_clock.close()
        if self._idle is not None:
            self._idle.stop()
        print(f"🎬 SwitchableVideoStreamTrack stopped. Final stats: {self.get_stats()}")
        super().stop()


# ============================================================================
# Audio Tracks
# ============================================================================

class SilenceAudioStreamTrack(MediaStreamTrack):
    """
    Silence audio source to keep the audio m-line alive until real audio is available.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, samples: int = 960):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples = samples
        self._timestamp = 0
        self._frame_time = self.samples / float(self.sample_rate)

    async def recv(self):
        await asyncio.sleep(self._frame_time)
        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples)
        for p in frame.planes:
            p.update(b"\x00" * p.buffer_size)
        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += self.samples
        return frame


def _convert_audio_with_ffmpeg(
    input_path: str,
    sample_rate: int = 48000,
    channels: int = 1,
) -> str:
    """
    Convert audio to optimal format for WebRTC using FFmpeg.
    """
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}_webrtc.wav"
    
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
        self._frames_sent = 0
        self._eof = False
        self._eof_event = asyncio.Event()
        
        self._audio_samples: bytes = b""
        self._read_position = 0
        self._fully_loaded = False
        self._load_lock = asyncio.Lock()
        self._converted_path: Optional[str] = None
        self._source_info = {}
        self._last_drift_seconds: Optional[float] = None
        self._drift_log_interval = int(os.getenv("WEBRTC_AUDIO_DRIFT_LOG_INTERVAL", "100"))
        self._strict_audio_stalls = 0
        self._strict_audio_stall_seconds = 0.0
        self._prepare_started_at: Optional[float] = None
        self._prepare_finished_at: Optional[float] = None

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
                try:
                    self._converted_path = await loop.run_in_executor(
                        None, _convert_audio_with_ffmpeg,
                        self._original_audio_path, self._sample_rate, self._channels
                    )
                    self._audio_path = self._converted_path
                except Exception as e:
                    print(f"⚠️ FFmpeg conversion failed: {e}")
            
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
            if not self._eof:
                self._eof = True
                self._eof_event.set()
                print(f"🔊 Audio EOF after {self._frames_sent} frames")
        return result

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

        if self._sync_clock is not None and self._sync_clock.strict_fifo:
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
            if target < time.monotonic() - 0.002:
                # Any upstream stall should become a real pause, not an audio
                # catch-up burst.
                self._start_time = time.monotonic() - (self._frames_sent * self._frame_duration)
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

        frame = av.AudioFrame(format="s16", layout=self._layout, samples=self._samples_per_frame)
        frame.planes[0].update(audio_bytes)
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = fractions.Fraction(1, self._sample_rate)

        self._timestamp += self._samples_per_frame
        self._frames_sent += 1
        if self._first_packet_at is None:
            self._first_packet_at = time.monotonic()
            if self._sync_clock is not None:
                self._sync_clock.mark_first_audio_packet()
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
        self._eof_event.set()
        self._audio_samples = b""
        
        if self._converted_path and os.path.exists(self._converted_path):
            try:
                os.remove(self._converted_path)
                print(f"🧹 Cleaned up: {self._converted_path}")
            except Exception:
                pass
        
        print(f"🔊 SyncedAudioStreamTrack stopped after {self._frames_sent} frames")
