import asyncio
import fractions
import math
import time
from array import array
from typing import Optional

import av
from aiortc import VideoStreamTrack, MediaStreamTrack


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
        # Drop the oldest frame if the queue is full to keep latency low.
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

    def __init__(self, idle_video_path: str, fps: float = 10.0, max_queue: int = 30):
        super().__init__()
        self._fps = fps
        self._frame_time = 1.0 / float(self._fps)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._idle = IdleVideoStreamTrack(idle_video_path, fps=fps)
        self._live_active = False
        self._last_ts = None
        self._closed = False

    def start_live(self) -> None:
        self._live_active = True

    def end_live(self) -> None:
        self._live_active = False
        # Clear any buffered live frames so we return to idle immediately.
        try:
            while True:
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

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

        frame = None
        if self._live_active:
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                frame = None

        if frame is None:
            frame = self._idle.read_frame()

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        self._closed = True
        if self._idle is not None:
            self._idle.stop()
        super().stop()


class SilenceAudioStreamTrack(MediaStreamTrack):
    """
    Simple silence audio source to keep the audio m-line alive until real audio is available.
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


class ToneAudioStreamTrack(MediaStreamTrack):
    """
    Generate a simple sine tone for debugging audio playback.
    """

    kind = "audio"

    def __init__(
        self,
        frequency: float = 440.0,
        sample_rate: int = 48000,
        samples: int = 960,
        amplitude: float = 0.3,
    ):
        super().__init__()
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.samples = samples
        self.amplitude = amplitude
        self._timestamp = 0
        self._frame_time = samples / sample_rate

    async def recv(self):
        # Wait to simulate real-time audio pacing
        await asyncio.sleep(self._frame_time)

        # Create audio frame
        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples)
        frame.sample_rate = self.sample_rate

        # Generate sine wave
        tone = array("h")
        for i in range(self.samples):
            sample_index = self._timestamp + i
            t = sample_index / self.sample_rate
            value = int(self.amplitude * 32767 * math.sin(2 * math.pi * self.frequency * t))
            tone.append(value)

        # Copy samples to frame buffer
        frame.planes[0].update(tone.tobytes())

        # Set timing
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self._timestamp += self.samples
        return frame


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
        """Restart the container/iterator (used when reattaching)."""
        self._reset_container()


class FileAudioStreamTrack(MediaStreamTrack):
    """
    Audio track that reads from a file with proper WebRTC timing.
    Unlike MediaPlayer, this ensures the audio is paced correctly for WebRTC.
    """

    kind = "audio"

    def __init__(self, audio_path: str, sample_rate: int = 48000, samples_per_frame: int = 960):
        super().__init__()
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self._timestamp = 0
        self._frame_time = samples_per_frame / sample_rate
        self._container = None
        self._resampler = None
        self._audio_fifo = []
        self._finished = False
        self._open_container()

    def _open_container(self):
        try:
            self._container = av.open(self.audio_path)
            self._stream = self._container.streams.audio[0]
            # Resampler to convert to 48kHz mono s16
            self._resampler = av.audio.resampler.AudioResampler(
                format='s16',
                layout='mono',
                rate=self.sample_rate
            )
            self._decoder = self._container.decode(self._stream)
        except Exception as e:
            print(f"⚠️ FileAudioStreamTrack: Failed to open {self.audio_path}: {e}")
            self._finished = True

    def _read_samples(self, num_samples: int) -> bytes:
        """Read exactly num_samples from the audio file."""
        result = b""
        bytes_needed = num_samples * 2  # s16 = 2 bytes per sample

        # First, drain from FIFO buffer
        while self._audio_fifo and len(result) < bytes_needed:
            chunk = self._audio_fifo.pop(0)
            result += chunk

        # If we still need more, decode from file
        while len(result) < bytes_needed and not self._finished:
            try:
                frame = next(self._decoder)
                resampled = self._resampler.resample(frame)
                for rf in resampled:
                    if rf is not None:
                        # Get raw bytes from the frame
                        data = bytes(rf.planes[0])
                        result += data
            except StopIteration:
                self._finished = True
                break
            except Exception as e:
                print(f"⚠️ FileAudioStreamTrack decode error: {e}")
                self._finished = True
                break

        # If we have more than needed, put excess back in FIFO
        if len(result) > bytes_needed:
            self._audio_fifo.insert(0, result[bytes_needed:])
            result = result[:bytes_needed]

        # Pad with silence if we don't have enough
        if len(result) < bytes_needed:
            result += b'\x00' * (bytes_needed - len(result))

        return result

    async def recv(self):
        await asyncio.sleep(self._frame_time)

        # Read samples from file
        audio_data = self._read_samples(self.samples_per_frame)

        # Create frame
        frame = av.AudioFrame(format='s16', layout='mono', samples=self.samples_per_frame)
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(audio_data)
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self._timestamp += self.samples_per_frame
        return frame

    def stop(self):
        if self._container:
            self._container.close()
            self._container = None
        super().stop()


class SyncedAudioStreamTrack(MediaStreamTrack):
    """
    Audio track that stays in sync with video by:
    1. Waiting for video generation to start before playing audio
    2. Using a frame-based timing approach matching video FPS
    """

    kind = "audio"

    def __init__(self, audio_path: str, sample_rate: int = 48000, samples_per_frame: int = 960):
        super().__init__()
        self._audio_path = audio_path
        self._sample_rate = sample_rate
        self._samples_per_frame = samples_per_frame
        self._frame_duration = samples_per_frame / sample_rate  # ~20ms per frame
        self._timestamp = 0
        self._started = asyncio.Event()  # Wait for video to signal start
        self._stopped = False
        self._container = None
        self._resampler = None
        self._audio_buffer = b""
        self._bytes_per_sample = 2  # s16
        self._channels = 1
        self._start_time = None
        self._frames_sent = 0

    def signal_start(self):
        """Called when first video frame is generated - starts audio playback"""
        self._start_time = time.time()
        self._started.set()

    def _open_container(self):
        if self._container is not None:
            return
        self._container = av.open(self._audio_path)
        self._audio_stream = self._container.streams.audio[0]
        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self._sample_rate,
        )

    def _read_samples(self, num_samples: int) -> bytes:
        """Read and resample audio data"""
        needed_bytes = num_samples * self._bytes_per_sample * self._channels
        
        while len(self._audio_buffer) < needed_bytes:
            try:
                packet = next(self._container.demux(self._audio_stream))
                for frame in packet.decode():
                    resampled = self._resampler.resample(frame)
                    for rf in resampled:
                        self._audio_buffer += bytes(rf.planes[0])
            except (StopIteration, av.error.EOFError):
                # End of file - pad with silence
                silence_needed = needed_bytes - len(self._audio_buffer)
                self._audio_buffer += b"\x00" * silence_needed
                break

        result = self._audio_buffer[:needed_bytes]
        self._audio_buffer = self._audio_buffer[needed_bytes:]
        return result

    async def recv(self):
        if self._stopped:
            raise MediaStreamError("Track stopped")

        # Wait for video to start before playing audio
        try:
            await asyncio.wait_for(self._started.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise MediaStreamError("Timeout waiting for video start")

        self._open_container()

        # Calculate expected time for this frame
        expected_time = self._start_time + (self._frames_sent * self._frame_duration)
        now = time.time()
        
        # Sleep to maintain proper timing (don't send frames too fast)
        sleep_time = expected_time - now
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        # Read audio samples
        audio_bytes = self._read_samples(self._samples_per_frame)

        # Create frame
        frame = av.AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.planes[0].update(audio_bytes)
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = fractions.Fraction(1, self._sample_rate)

        self._timestamp += self._samples_per_frame
        self._frames_sent += 1
        
        return frame

    def stop(self):
        self._stopped = True
        self._started.set()  # Unblock any waiting recv()
        if self._container:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
