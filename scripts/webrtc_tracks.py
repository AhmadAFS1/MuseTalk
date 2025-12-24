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
import tempfile
import time
from pathlib import Path
from typing import Optional

import av
from aiortc import VideoStreamTrack, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError


# ============================================================================
# Video Tracks
# ============================================================================

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
    Avoids replaceTrack issues by always producing frames.
    """

    def __init__(
        self,
        idle_video_path: str,
        source_fps: float = 10.0,
        output_fps: Optional[float] = None,
        max_queue: int = 30,
    ):
        super().__init__()
        self._source_fps = float(source_fps)
        self._output_fps = float(output_fps) if output_fps is not None else self._source_fps
        if self._output_fps <= 0:
            self._output_fps = self._source_fps
        self._frame_time = 1.0 / float(self._output_fps)
        # Track how often to advance source frames vs output frames.
        self._source_step = self._source_fps / self._output_fps
        self._source_accum = 1.0
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._idle = IdleVideoStreamTrack(idle_video_path, fps=source_fps)
        self._live_active = False
        self._last_idle_frame = None
        self._last_live_frame = None
        self._last_ts = None
        self._closed = False

    def _reset_source_timing(self) -> None:
        self._source_accum = 1.0

    def _advance_source(self) -> int:
        self._source_accum += self._source_step
        advance = int(self._source_accum)
        if advance > 0:
            self._source_accum -= advance
        return advance

    def _pop_live_frames(self, steps: int):
        frame = None
        for _ in range(max(steps, 0)):
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        return frame

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
        self._live_active = True
        self._last_live_frame = None
        self._reset_source_timing()

    def end_live(self) -> None:
        self._live_active = False
        try:
            while True:
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self._last_live_frame = None
        self._reset_source_timing()

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

        advance_frames = self._advance_source()
        frame = None
        if self._live_active:
            if advance_frames > 0:
                next_frame = self._pop_live_frames(advance_frames)
                if next_frame is not None:
                    self._last_live_frame = next_frame
            frame = self._last_live_frame

        if frame is None:
            frame = self._advance_idle_frame(advance_frames)

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        self._closed = True
        if self._idle is not None:
            self._idle.stop()
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
    
    Uses high-quality resampling (soxr) and outputs raw PCM s16le.
    Returns path to the converted WAV file.
    """
    input_path = Path(input_path)
    
    # Create output path in same directory
    output_path = input_path.parent / f"{input_path.stem}_webrtc.wav"
    
    # Build FFmpeg command with high-quality settings
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_path),
        # High-quality resampling
        "-af", f"aresample=resampler=soxr:precision=33:dither_method=triangular",
        "-ar", str(sample_rate),  # Sample rate
        "-ac", str(channels),  # Channels
        "-sample_fmt", "s16",  # 16-bit signed
        "-c:a", "pcm_s16le",  # PCM codec
        "-f", "wav",  # WAV container
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"⚠️ FFmpeg warning: {result.stderr}")
            # Try simpler command without soxr
            cmd_simple = [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-ar", str(sample_rate),
                "-ac", str(channels),
                "-c:a", "pcm_s16le",
                "-f", "wav",
                str(output_path)
            ]
            result = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"🔊 FFmpeg converted: {input_path.name} -> {output_path.name}")
        return str(output_path)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install ffmpeg.")


class SyncedAudioStreamTrack(MediaStreamTrack):
    """
    High-quality audio track synced with video generation.
    
    Features:
    - Uses FFmpeg for high-quality resampling (soxr)
    - Pre-loads entire audio into memory
    - PTS-based timing for accurate sync
    - Optimal format for Opus encoding (48kHz, mono, s16)
    """

    kind = "audio"

    def __init__(
        self,
        audio_path: str,
        sample_rate: int = 48000,
        samples_per_frame: int = 960,  # 20ms at 48kHz - optimal for Opus
        use_stereo: bool = False,
        use_ffmpeg_convert: bool = True,  # Use FFmpeg for high-quality conversion
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
        
        self._timestamp = 0
        self._started = asyncio.Event()
        self._stopped = False
        self._bytes_per_sample = 2  # s16 = 2 bytes
        self._start_time: Optional[float] = None
        self._frames_sent = 0
        self._eof = False
        
        # Pre-decoded audio buffer
        self._audio_samples: bytes = b""
        self._read_position = 0
        self._fully_loaded = False
        self._load_lock = asyncio.Lock()
        
        # Converted file path (for cleanup)
        self._converted_path: Optional[str] = None
        
        # Debug info
        self._source_info = {}

    def signal_start(self):
        """Called when first video frame is generated - starts audio playback"""
        self._start_time = time.time()
        self._started.set()
        print(f"🔊 SyncedAudioStreamTrack: Started at {self._start_time:.3f}")

    async def _load_audio_async(self):
        """Load and decode entire audio file into memory"""
        if self._fully_loaded:
            return
            
        async with self._load_lock:
            if self._fully_loaded:
                return
            
            loop = asyncio.get_event_loop()
            
            # Convert with FFmpeg first (in thread pool)
            if self._use_ffmpeg:
                try:
                    self._converted_path = await loop.run_in_executor(
                        None,
                        _convert_audio_with_ffmpeg,
                        self._original_audio_path,
                        self._sample_rate,
                        self._channels
                    )
                    self._audio_path = self._converted_path
                except Exception as e:
                    print(f"⚠️ FFmpeg conversion failed, falling back to PyAV: {e}")
                    self._audio_path = self._original_audio_path
            
            # Load the (converted) audio
            self._audio_samples = await loop.run_in_executor(None, self._load_pcm_audio)
            self._fully_loaded = True
            
            duration_ms = len(self._audio_samples) / (self._bytes_per_sample * self._channels) / self._sample_rate * 1000
            print(f"🔊 Audio loaded: {len(self._audio_samples)} bytes, {duration_ms:.0f}ms, "
                  f"{self._channels}ch @ {self._sample_rate}Hz")

    def _load_pcm_audio(self) -> bytes:
        """
        Load audio file. If it's a WAV file from FFmpeg, read raw PCM.
        Otherwise, decode with PyAV.
        """
        audio_path = Path(self._audio_path)
        
        # Check if it's our converted WAV file
        if audio_path.suffix.lower() == '.wav' and '_webrtc' in audio_path.stem:
            return self._load_wav_pcm()
        else:
            return self._decode_with_pyav()

    def _load_wav_pcm(self) -> bytes:
        """Load raw PCM from WAV file (faster than decoding)"""
        import wave
        
        try:
            with wave.open(self._audio_path, 'rb') as wf:
                # Verify format
                if wf.getsampwidth() != 2:
                    print(f"⚠️ WAV sample width is {wf.getsampwidth()}, expected 2")
                if wf.getframerate() != self._sample_rate:
                    print(f"⚠️ WAV sample rate is {wf.getframerate()}, expected {self._sample_rate}")
                if wf.getnchannels() != self._channels:
                    print(f"⚠️ WAV channels is {wf.getnchannels()}, expected {self._channels}")
                
                self._source_info = {
                    'sample_rate': wf.getframerate(),
                    'channels': wf.getnchannels(),
                    'format': 's16le',
                    'codec': 'pcm',
                    'source': 'ffmpeg_converted'
                }
                print(f"🔊 Source (FFmpeg WAV): {self._source_info}")
                
                # Read all frames
                data = wf.readframes(wf.getnframes())
                print(f"🔊 Loaded {len(data)} bytes from WAV")
                return data
                
        except Exception as e:
            print(f"⚠️ WAV load failed, falling back to PyAV: {e}")
            return self._decode_with_pyav()

    def _decode_with_pyav(self) -> bytes:
        """Decode audio file with PyAV (fallback)"""
        result = bytearray()
        
        try:
            container = av.open(self._audio_path)
            audio_stream = container.streams.audio[0]
            
            self._source_info = {
                'sample_rate': audio_stream.sample_rate,
                'channels': audio_stream.layout.channels,
                'format': str(audio_stream.format.name),
                'codec': audio_stream.codec_context.name,
                'source': 'pyav'
            }
            print(f"🔊 Source (PyAV): {self._source_info}")
            
            # Create resampler
            resampler = av.AudioResampler(
                format="s16",
                layout=self._layout,
                rate=self._sample_rate,
            )
            
            # Decode all frames
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    resampled_frames = resampler.resample(frame)
                    for rf in resampled_frames:
                        if rf is not None:
                            result.extend(bytes(rf.planes[0]))
            
            # Flush resampler
            flushed_frames = resampler.resample(None)
            for rf in flushed_frames:
                if rf is not None:
                    result.extend(bytes(rf.planes[0]))
            
            container.close()
            print(f"🔊 Decoded {len(result)} bytes with PyAV")
            
        except Exception as e:
            print(f"⚠️ Audio decode error: {e}")
            import traceback
            traceback.print_exc()
            
        return bytes(result)

    def _get_samples(self, num_samples: int) -> bytes:
        """Get samples from pre-loaded buffer"""
        bytes_per_frame = num_samples * self._bytes_per_sample * self._channels
        
        if self._read_position + bytes_per_frame <= len(self._audio_samples):
            result = self._audio_samples[self._read_position:self._read_position + bytes_per_frame]
            self._read_position += bytes_per_frame
        else:
            remaining = self._audio_samples[self._read_position:]
            padding_needed = bytes_per_frame - len(remaining)
            result = remaining + (b"\x00" * padding_needed)
            self._read_position = len(self._audio_samples)
            if not self._eof:
                self._eof = True
                print(f"🔊 Audio EOF after {self._frames_sent} frames")
            
        return result

    async def recv(self):
        if self._stopped:
            raise MediaStreamError("Track stopped")

        # Wait for start signal
        try:
            await asyncio.wait_for(self._started.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            raise MediaStreamError("Timeout waiting for start signal")

        # Load audio on first recv
        if not self._fully_loaded:
            await self._load_audio_async()

        # PTS-based timing
        if self._start_time is not None:
            target_time = self._start_time + (self._frames_sent * self._frame_duration)
            now = time.time()
            sleep_time = target_time - now
            
            if sleep_time > 0.002:
                await asyncio.sleep(sleep_time)
            elif sleep_time < -0.05:
                if self._frames_sent % 50 == 0:
                    print(f"⚠️ Audio {-sleep_time*1000:.1f}ms behind")

        # Get audio samples
        audio_bytes = self._get_samples(self._samples_per_frame)

        # Create frame
        frame = av.AudioFrame(
            format="s16",
            layout=self._layout,
            samples=self._samples_per_frame
        )
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
        
        # Cleanup converted file
        if self._converted_path and os.path.exists(self._converted_path):
            try:
                os.remove(self._converted_path)
                print(f"🧹 Cleaned up converted audio: {self._converted_path}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup {self._converted_path}: {e}")
        
        print(f"🔊 SyncedAudioStreamTrack stopped after {self._frames_sent} frames")
