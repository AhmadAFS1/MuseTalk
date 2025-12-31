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
WEBRTC_VIDEO_PREBUFFER_SECONDS = float(os.getenv("WEBRTC_VIDEO_PREBUFFER_SECONDS", "2.0"))  # Match chunk duration
WEBRTC_ADAPTIVE_FPS = os.getenv("WEBRTC_ADAPTIVE_FPS", "1").lower() in ("1", "true", "yes")
WEBRTC_MIN_FPS_RATIO = float(os.getenv("WEBRTC_MIN_FPS_RATIO", "0.75"))  # Allow slowdown to 75%
WEBRTC_QUEUE_LOG_INTERVAL = int(os.getenv("WEBRTC_QUEUE_LOG_INTERVAL", "30"))
WEBRTC_TARGET_QUEUE_FILL = float(os.getenv("WEBRTC_TARGET_QUEUE_FILL", "0.4"))  # Target 40% queue fill

# ============================================================================
# Video Tracks
# ============================================================================

class VideoSyncClock:
    def __init__(self, source_fps: float):
        self.source_fps = float(source_fps) if source_fps > 0 else 1.0
        self.source_frames = 0
        self.active = False
        self.started = asyncio.Event()

    def reset(self) -> None:
        self.source_frames = 0
        self.active = True
        self.started.clear()

    def deactivate(self) -> None:
        self.active = False
        self.started.set()

    def mark_started(self) -> None:
        if self.active and not self.started.is_set():
            self.started.set()

    def add_frames(self, frames: int) -> None:
        if self.active and frames > 0:
            self.source_frames += frames

    def video_time(self) -> float:
        return self.source_frames / self.source_fps


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
            self._last_ts = time.time()
        else:
            now = time.time()
            wait = self._frame_time - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.time()

        frame = self.read_frame()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
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
            self._last_ts = time.time()
        else:
            now = time.time()
            wait = self._frame_time - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.time()

        frame = await self._queue.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        self._closed = True
        super().stop()


class SwitchableVideoStreamTrack(VideoStreamTrack):
    """
    Single video track that switches between idle frames and live frames.
    
    MSE-like buffering behavior:
    - Prebuffering: waits for N seconds of frames before switching to live playback
    - Adaptive FPS: dynamically adjusts playback speed based on queue depth
    - Smooth transitions between idle and live modes
    - Never drops frames - slows down instead to maintain smoothness
    """

    def __init__(
        self,
        idle_video_path: str,
        source_fps: float = 10.0,
        output_fps: Optional[float] = None,
        max_queue: int = 100,  # Increased for more buffering headroom
        sync_clock: Optional[VideoSyncClock] = None,
        prebuffer_seconds: Optional[float] = None,
        adaptive_fps: Optional[bool] = None,
        min_fps_ratio: Optional[float] = None,
    ):
        super().__init__()
        self._source_fps = float(source_fps)
        self._output_fps = float(output_fps) if output_fps is not None else self._source_fps
        if self._output_fps <= 0:
            self._output_fps = self._source_fps
        self._base_frame_time = 1.0 / float(self._output_fps)
        self._frame_time = self._base_frame_time
        self._source_step = self._source_fps / self._output_fps
        self._source_accum = 0.0
        self._max_queue = max_queue
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._idle = IdleVideoStreamTrack(idle_video_path, fps=source_fps)
        self._live_active = False
        self._last_idle_frame = None
        self._last_live_frame = None
        self._last_ts = None
        self._closed = False
        self._sync_clock = sync_clock

        # Prebuffer support - default to 2 seconds like MSE chunks
        if prebuffer_seconds is None:
            prebuffer_seconds = WEBRTC_VIDEO_PREBUFFER_SECONDS
        self._prebuffer_seconds = prebuffer_seconds
        self._prebuffer_frames = int(prebuffer_seconds * self._source_fps)
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
        self._slowdown_active = False
        self._current_slowdown = 1.0
        self._generation_complete = False
        
        # Smoothing for adaptive FPS (prevents jitter)
        self._slowdown_history = []
        self._slowdown_window = 5  # Average over 5 samples
        
        print(f"🎬 SwitchableVideoStreamTrack: prebuffer={self._prebuffer_frames} frames "
              f"({prebuffer_seconds}s), adaptive_fps={adaptive_fps}, min_ratio={min_fps_ratio}, "
              f"max_queue={max_queue}, target_fill={self._target_fill}")

    def _reset_source_timing(self) -> None:
        self._source_accum = 0.0

    def _advance_source(self) -> int:
        self._source_accum += self._source_step
        advance = int(self._source_accum)
        if advance > 0:
            self._source_accum -= advance
        return advance

    def _pop_live_frames(self, steps: int):
        """Pop frames from queue, returning the last frame and count popped"""
        frame = None
        popped = 0
        for _ in range(max(steps, 1)):
            try:
                frame = self._queue.get_nowait()
                popped += 1
            except asyncio.QueueEmpty:
                break
        return frame, popped

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
        self._last_live_frame = None
        self._reset_source_timing()
        self._frames_received = 0
        self._frames_played = 0
        self._frames_dropped = 0
        self._prebuffer_ready.clear()
        self._slowdown_active = False
        self._current_slowdown = 1.0
        self._generation_complete = False
        self._slowdown_history = []
        if self._sync_clock:
            self._sync_clock.reset()
        print(f"🎬 Live mode started, prebuffering {self._prebuffer_frames} frames ({self._prebuffer_seconds}s)...")

    def end_live(self) -> None:
        """End live mode - drain queue and return to idle"""
        self._live_active = False
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
        print(f"🎬 Live mode ended. Played: {self._frames_played}, Dropped: {self._frames_dropped}, Drained: {drained}")

    def signal_generation_complete(self) -> None:
        """Called when all frames have been pushed - allows queue to drain naturally"""
        self._generation_complete = True
        print(f"🎬 Generation complete signaled. Queue: {self._queue.qsize()}, Played: {self._frames_played}")

    async def push_bgr_frame(self, frame_bgr) -> None:
        """Push a single BGR frame to the queue - never drops, waits if full"""
        if self._closed:
            return
        
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24").reformat(format="yuv420p")
        
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
        
        self._frames_received += 1
        
        # Check if prebuffer is ready
        if not self._prebuffer_ready.is_set() and self._frames_received >= self._prebuffer_frames:
            print(f"🎬 Prebuffer ready: {self._frames_received} frames buffered, queue: {self._queue.qsize()}/{self._max_queue}")
            self._prebuffer_ready.set()

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
        if self._frames_played % WEBRTC_QUEUE_LOG_INTERVAL == 0 and self._frames_played > 0:
            effective_fps = self._output_fps / slowdown
            status = "⚡" if slowdown <= 1.0 else ("🐢" if slowdown > 1.2 else "📊")
            print(f"{status} Queue: {queue_size}/{self._max_queue} ({fill_ratio*100:.0f}%), "
                  f"slowdown: {slowdown:.2f}x, effective_fps: {effective_fps:.1f}, "
                  f"played: {self._frames_played}, gen_complete: {self._generation_complete}")
        
        # Track slowdown state changes
        was_slow = self._slowdown_active
        self._slowdown_active = slowdown > 1.05
        
        if self._slowdown_active and not was_slow:
            print(f"📉 Slowing playback: {slowdown:.2f}x (queue: {fill_ratio*100:.0f}%)")
        elif was_slow and not self._slowdown_active:
            print(f"📈 Resuming normal speed (queue: {fill_ratio*100:.0f}%)")
        
        return self._base_frame_time * slowdown

    def _get_current_frame_time(self) -> float:
        """Get the current effective frame time (for stats)"""
        return self._base_frame_time * self._current_slowdown

    async def recv(self):
        if self._closed:
            raise asyncio.CancelledError()

        # Calculate adaptive frame time
        frame_time = self._get_adaptive_frame_time()

        # Timing control
        if self._last_ts is None:
            self._last_ts = time.time()
        else:
            now = time.time()
            wait = frame_time - (now - self._last_ts)
            if wait > 0.001:
                await asyncio.sleep(wait)
            self._last_ts = time.time()

        advance_frames = self._advance_source()
        frame = None
        
        if self._live_active:
            # Check if we're still prebuffering
            if self._prebuffer_frames > 0 and not self._prebuffer_ready.is_set():
                # Still prebuffering - show idle frames while we wait
                frame = self._advance_idle_frame(advance_frames)
            else:
                # Prebuffer ready - consume live frames
                if advance_frames > 0 or self._last_live_frame is None:
                    next_frame, popped = self._pop_live_frames(advance_frames)
                    if next_frame is not None:
                        self._last_live_frame = next_frame
                        self._frames_played += popped
                        if self._sync_clock:
                            self._sync_clock.add_frames(popped)
                            self._sync_clock.mark_started()
                
                frame = self._last_live_frame
                
                # If queue is empty and generation is complete, end live mode
                if frame is None and self._generation_complete and self._queue.qsize() == 0:
                    print("🎬 Playback complete - returning to idle")
                    self.end_live()

        # Fallback to idle frame if no live frame available
        if frame is None:
            frame = self._advance_idle_frame(advance_frames)

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def get_stats(self) -> dict:
        """Get current track statistics"""
        queue_size = self._queue.qsize()
        return {
            'live_active': self._live_active,
            'prebuffer_ready': self._prebuffer_ready.is_set(),
            'queue_size': queue_size,
            'queue_max': self._max_queue,
            'queue_fill_pct': (queue_size / self._max_queue * 100) if self._max_queue > 0 else 0,
            'frames_received': self._frames_received,
            'frames_played': self._frames_played,
            'frames_dropped': self._frames_dropped,
            'prebuffer_frames': self._prebuffer_frames,
            'adaptive_fps': self._adaptive_fps,
            'slowdown_active': self._slowdown_active,
            'current_slowdown': self._current_slowdown,
            'effective_fps': self._output_fps / self._current_slowdown,
            'generation_complete': self._generation_complete,
        }

    def stop(self) -> None:
        self._closed = True
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
        self._frames_sent = 0
        self._eof = False
        
        self._audio_samples: bytes = b""
        self._read_position = 0
        self._fully_loaded = False
        self._load_lock = asyncio.Lock()
        self._converted_path: Optional[str] = None
        self._source_info = {}

    def signal_start(self):
        self._start_time = time.time()
        self._started.set()
        print(f"🔊 SyncedAudioStreamTrack: Started at {self._start_time:.3f}")

    async def _load_audio_async(self):
        if self._fully_loaded:
            return
            
        async with self._load_lock:
            if self._fully_loaded:
                return
            
            loop = asyncio.get_event_loop()
            
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
            
            duration_ms = len(self._audio_samples) / (self._bytes_per_sample * self._channels) / self._sample_rate * 1000
            print(f"🔊 Audio loaded: {len(self._audio_samples)} bytes, {duration_ms:.0f}ms")

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
                print(f"🔊 Audio EOF after {self._frames_sent} frames")
        return result

    def _skip_audio_frames(self, count: int) -> None:
        if count <= 0:
            return
        skip_bytes = count * self._samples_per_frame * self._bytes_per_sample * self._channels
        self._read_position = min(len(self._audio_samples), self._read_position + skip_bytes)
        self._timestamp += count * self._samples_per_frame
        self._frames_sent += count

    async def recv(self):
        if self._stopped:
            raise MediaStreamError("Track stopped")

        try:
            await asyncio.wait_for(self._started.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            raise MediaStreamError("Timeout waiting for start")
        
        if self._sync_clock is not None:
            try:
                await asyncio.wait_for(self._sync_clock.started.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                raise MediaStreamError("Timeout waiting for video")

        if not self._fully_loaded:
            await self._load_audio_async()

        # Simple timing - sync to video clock if available
        if self._start_time is not None:
            target = self._start_time + (self._frames_sent * self._frame_duration)
            wait = target - time.time()
            if wait > 0.002:
                await asyncio.sleep(wait)

        # Sync with video clock
        if self._sync_clock is not None and self._sync_clock.active:
            video_time = self._sync_clock.video_time()
            audio_time = self._frames_sent * self._frame_duration
            drift = audio_time - video_time
            
            if drift > self._max_audio_lead:
                await asyncio.sleep(drift - self._max_audio_lead)
            elif drift < -self._max_audio_lag:
                skip = int((-drift - self._max_audio_lag) / self._frame_duration)
                if skip > 0:
                    self._skip_audio_frames(skip)

        audio_bytes = self._get_samples(self._samples_per_frame)

        frame = av.AudioFrame(format="s16", layout=self._layout, samples=self._samples_per_frame)
        frame.planes[0].update(audio_bytes)
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = fractions.Fraction(1, self._sample_rate)

        self._timestamp += self._samples_per_frame
        self._frames_sent += 1
        return frame

    def stop(self):
        self._stopped = True
        self._started.set()
        self._audio_samples = b""
        
        if self._converted_path and os.path.exists(self._converted_path):
            try:
                os.remove(self._converted_path)
                print(f"🧹 Cleaned up: {self._converted_path}")
            except Exception:
                pass
        
        print(f"🔊 SyncedAudioStreamTrack stopped after {self._frames_sent} frames")
