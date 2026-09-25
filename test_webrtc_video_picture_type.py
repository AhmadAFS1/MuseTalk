"""Actual PyAV/VP8 checks for source metadata leaking into transport encoding."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import av
import numpy as np
from aiortc import RTCRtpSender
from aiortc.codecs import CODECS
from aiortc.codecs.vpx import vp8_depayload
from aiortc.rtp import RTCP_PSFB_PLI, RtcpPsfbPacket

from scripts.webrtc_tracks import (
    IdleVideoStreamTrack, LiveVideoStreamTrack, SwitchableVideoStreamTrack,
)


class VideoPictureTypeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "all-intra.mp4"
        pixels = np.zeros((128, 128, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(128, dtype=np.uint8)[None, :]
        pixels[:, :, 1] = np.arange(128, dtype=np.uint8)[:, None]
        pixels[:, :, 2] = 160
        with av.open(str(self.path), "w") as container:
            stream = container.add_stream("libx264", rate=20)
            stream.width = 128
            stream.height = 128
            stream.pix_fmt = "yuv420p"
            stream.options = {"g": "1", "qp": "18", "threads": "1"}
            for index in range(6):
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 20)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(self.path)) as container:
            frames = list(container.decode(video=0))
            self.assertTrue(all(int(frame.pict_type) == 1 for frame in frames))
            self.source_pixels = frames[0].to_ndarray(format="yuv420p")

    async def _assert_interframes_and_pli(self, track, *, held=False):
        sender = RTCRtpSender(track, SimpleNamespace(state="new"))
        codec = next(codec for codec in CODECS["video"] if codec.mimeType == "video/VP8")
        tags, timestamps = [], []
        cached_frame = None
        for index in range(4):
            if index == 2:
                await sender._handle_rtcp_packet(RtcpPsfbPacket(
                    fmt=RTCP_PSFB_PLI, ssrc=123, media_ssrc=sender._ssrc))
            encoded = await sender._next_encoded_frame(codec)
            tags.append(vp8_depayload(encoded.payloads[0])[0] & 1)
            timestamps.append(encoded.timestamp)
            if held:
                if index == 0:
                    cached_frame = track._last_idle_frame
                    track._idle_sync_hold_active = True
                self.assertIs(track._last_idle_frame, cached_frame)
                np.testing.assert_array_equal(
                    cached_frame.to_ndarray(format="yuv420p"), self.source_pixels)
        # First frame and the explicit PLI are keyframes; source I metadata and
        # a sender-mutated cached I frame must not force the other two frames.
        self.assertEqual(tags, [0, 1, 0, 1])
        self.assertEqual(timestamps, [0, 4500, 9000, 13500])

    async def test_standalone_idle_uses_interframes_and_honors_sender_pli(self):
        track = IdleVideoStreamTrack(str(self.path), fps=20, decode_threads=1)
        self.addCleanup(track.stop)
        await self._assert_interframes_and_pli(track)

    async def test_switchable_held_frame_clears_source_and_previous_pli_metadata(self):
        track = SwitchableVideoStreamTrack(str(self.path), source_fps=20,
            output_fps=20, prebuffer_seconds=0, adaptive_fps=False)
        self.addCleanup(track.stop)
        await self._assert_interframes_and_pli(track, held=True)

    async def test_live_output_clears_metadata_without_changing_pixels_or_clock(self):
        track = LiveVideoStreamTrack(fps=20)
        self.addCleanup(track.stop)
        with av.open(str(self.path)) as container:
            frame = next(container.decode(video=0))
        await track._queue.put(frame)
        output = await track.recv()
        self.assertIs(output, frame)
        self.assertEqual(int(output.pict_type), 0)
        self.assertEqual(output.pts, 0)
        self.assertEqual(output.time_base, Fraction(1, 90000))
        np.testing.assert_array_equal(output.to_ndarray(format="yuv420p"), self.source_pixels)


if __name__ == "__main__":
    unittest.main()
