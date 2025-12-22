import asyncio
import time
import fractions
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

    async def recv(self):
        if self._last_ts is None:
            self._last_ts = time.time()
        else:
            now = time.time()
            wait = self._frame_time - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.time()

        try:
            frame = next(self._frame_iter)
        except StopIteration:
            self._reset_container()
            frame = next(self._frame_iter)

        frame = frame.reformat(format="yuv420p")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        super().stop()
