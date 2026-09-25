import asyncio
import math
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import Mock, patch
from fractions import Fraction

from scripts.test_pose_webrtc import (
    SmokeTestError,
    MediaRecorder,
    RTPMP4Recorder,
    WallClockAudioTrack,
    WallClockVideoTrack,
    validate_recording_timestamp_proof,
)


try:
    import av
    from aiortc.mediastreams import MediaStreamError
except ModuleNotFoundError:
    av = None


class _FixedClock:
    @staticmethod
    def elapsed():
        return 0.0


class _Frame:
    def __init__(
        self,
        pts,
        time_base,
        *,
        sample_rate=None,
        samples=None,
    ):
        self.pts = pts
        self.time_base = time_base
        self.sample_rate = sample_rate
        self.samples = samples


class _Source:
    def __init__(self, frames):
        self.frames = list(frames)

    async def recv(self):
        await asyncio.sleep(0)
        return self.frames.pop(0)


async def _video_stats(points):
    source = _Source(
        [_Frame(pts, Fraction(1, 90_000)) for pts in points]
    )
    track = WallClockVideoTrack(source, _FixedClock(), nominal_fps=30)
    for _ in points:
        await track.recv()
    return track.get_stats()


async def _audio_stats(points, *, samples=960):
    source = _Source(
        [
            _Frame(
                pts,
                Fraction(1, 48_000),
                sample_rate=48_000,
                samples=samples,
            )
            for pts in points
        ]
    )
    track = WallClockAudioTrack(source, _FixedClock())
    for _ in points:
        await track.recv()
    return track.get_stats()


def _case(
    *,
    first_live,
    correction_frames=0,
    first_tts=None,
    audio_correction=0.0,
):
    sync_clock = {
        "first_live_video_rtp_seconds": first_live,
        "video_rtp_phase_correction_seconds": correction_frames / 30.0,
        "audio_rtp_phase_correction_seconds": audio_correction,
    }
    if first_tts is not None:
        sync_clock["first_tts_transport_pts_seconds"] = first_tts
    return {
        "final_track_stats": {
            "sync_clock": sync_clock,
            "video": {
                "last_live_rtp_phase_correction_frames": correction_frames,
                "sync_clock": dict(sync_clock),
            },
            "audio_transport": {
                "last_source": {
                    **(
                        {"first_tts_transport_pts_seconds": first_tts}
                        if first_tts is not None
                        else {}
                    ),
                    "sync_clock": dict(sync_clock),
                }
            },
        }
    }


class PoseWebRTCRecordingTimestampTest(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_only_declared_first_live_video_phase_gap(self):
        video = await _video_stats([0, 3_000, 6_000, 21_000, 24_000])
        audio = await _audio_stats([0, 960, 1_920, 2_880, 3_840])
        first_live = 21_000 / 90_000.0

        result = validate_recording_timestamp_proof(
            {"video": video, "audio": audio},
            [
                _case(
                    first_live=first_live,
                    correction_frames=4,
                    first_tts=first_live + 0.006,
                )
            ],
            playback_fps=30,
        )

        self.assertTrue(result["validated"])
        self.assertEqual(result["video_source_timestamp_anomalies"], 1)
        self.assertEqual(
            result["declared_video_phase_corrections"][0]["correction_frames"],
            4,
        )
        self.assertTrue(result["first_live_av_rtp_checks"][0]["aligned"])

    async def test_rejects_undeclared_video_gap(self):
        video = await _video_stats([0, 3_000, 18_000, 21_000])
        audio = await _audio_stats([0, 960, 1_920, 2_880])

        with self.assertRaisesRegex(SmokeTestError, "undeclared gap"):
            validate_recording_timestamp_proof(
                {"video": video, "audio": audio},
                [_case(first_live=18_000 / 90_000.0)],
                playback_fps=30,
            )

    async def test_rejects_phase_gap_at_wrong_receiver_pts(self):
        video = await _video_stats([0, 3_000, 18_000, 21_000])
        audio = await _audio_stats([0, 960, 1_920, 2_880])

        with self.assertRaisesRegex(SmokeTestError, "not located"):
            validate_recording_timestamp_proof(
                {"video": video, "audio": audio},
                [
                    _case(
                        first_live=21_000 / 90_000.0,
                        correction_frames=4,
                    )
                ],
                playback_fps=30,
            )

    async def test_rejects_first_tts_video_delta_over_one_frame(self):
        video = await _video_stats([0, 3_000, 6_000, 9_000])
        audio = await _audio_stats([0, 960, 1_920, 2_880])

        with self.assertRaisesRegex(SmokeTestError, "exceeded one video frame"):
            validate_recording_timestamp_proof(
                {"video": video, "audio": audio},
                [
                    _case(
                        first_live=9_000 / 90_000.0,
                        first_tts=9_000 / 90_000.0 + 0.04,
                    )
                ],
                playback_fps=30,
            )

    async def test_old_server_without_actual_first_tts_stat_remains_auditable(self):
        video = await _video_stats([0, 3_000, 6_000, 9_000])
        audio = await _audio_stats([0, 960, 1_920, 2_880])

        result = validate_recording_timestamp_proof(
            {"video": video, "audio": audio},
            [_case(first_live=9_000 / 90_000.0)],
            playback_fps=30,
        )

        self.assertTrue(result["validated"])
        self.assertFalse(result["actual_first_tts_rtp_available"])

    async def test_accepts_declared_audio_rebase_only_at_first_tts_packet(self):
        video = await _video_stats([0, 3_000, 6_000, 9_000])
        # Normal next PTS after 1,920 is 2,880. The first TTS packet advances
        # to 4,800, declaring a 40 ms excess gap.
        audio = await _audio_stats([0, 960, 1_920, 4_800, 5_760])
        first_tts = 4_800 / 48_000.0

        result = validate_recording_timestamp_proof(
            {"video": video, "audio": audio},
            [
                _case(
                    first_live=first_tts,
                    first_tts=first_tts,
                    audio_correction=0.04,
                )
            ],
            playback_fps=30,
        )

        self.assertTrue(result["validated"])
        self.assertEqual(result["audio_source_timestamp_anomalies"], 1)
        self.assertEqual(len(result["declared_audio_phase_corrections"]), 1)

    async def test_accepts_jitter_buffer_start_one_packet_after_audio_rebase(self):
        video = await _video_stats([0, 3_000, 6_000, 9_000])
        # The server declares first TTS at 80 ms, while the receiver jitter
        # buffer exposes the next ordinary 20 ms packet at 100 ms. The measured
        # excess gap remains exactly the declared 40 ms rebase.
        audio = await _audio_stats([0, 960, 1_920, 4_800, 5_760])

        result = validate_recording_timestamp_proof(
            {"video": video, "audio": audio},
            [
                _case(
                    first_live=0.10,
                    first_tts=0.08,
                    audio_correction=0.04,
                )
            ],
            playback_fps=30,
        )

        declaration = result["declared_audio_phase_corrections"][0]
        self.assertAlmostEqual(
            declaration["receiver_first_packet_offset_seconds"],
            0.02,
            places=6,
        )

    async def test_rejects_audio_rebase_more_than_one_packet_after_first_tts(self):
        video = await _video_stats([0, 3_000, 6_000, 9_000])
        audio = await _audio_stats([0, 960, 1_920, 5_760, 6_720])

        with self.assertRaisesRegex(SmokeTestError, "following packet"):
            validate_recording_timestamp_proof(
                {"video": video, "audio": audio},
                [
                    _case(
                        first_live=0.10,
                        first_tts=0.08,
                        audio_correction=0.06,
                    )
                ],
                playback_fps=30,
            )


@unittest.skipIf(av is None, "PyAV and aiortc are required for real MP4 roundtrips")
class MP4RecorderCadenceTest(unittest.IsolatedAsyncioTestCase):
    async def _roundtrip(self, output, video_points, *, fps=20):
        class FiniteSource:
            def __init__(self, frames):
                self.frames = iter(frames)
                self.exhausted = asyncio.Event()

            async def recv(self):
                await asyncio.sleep(0)
                try:
                    return next(self.frames)
                except StopIteration:
                    self.exhausted.set()
                    raise MediaStreamError

        class OriginClock:
            def __init__(self, elapsed):
                self.origin = elapsed

            def elapsed(self):
                return self.origin

        frames = []
        for index, pts in enumerate(video_points):
            frame = av.VideoFrame(64, 64, "yuv420p")
            for plane_index, plane in enumerate(frame.planes):
                plane.update(bytes([40 + index if plane_index == 0 else 128]) * plane.buffer_size)
            frame.pts, frame.time_base = pts, Fraction(1, 90_000)
            frames.append(frame)
        audio_frames = []
        for index in range(50):
            frame = av.AudioFrame(format="s16", layout="mono", samples=960)
            frame.sample_rate = 48_000
            frame.pts, frame.time_base = 144_000 + index * 960, Fraction(1, 48_000)
            pcm = array("h", (int(4000 * math.sin(2 * math.pi * 440 * (index * 960 + n) / 48_000))
                              for n in range(960)))
            frame.planes[0].update(pcm.tobytes())
            audio_frames.append(frame)
        video_source, audio_source = FiniteSource(frames), FiniteSource(audio_frames)
        video = WallClockVideoTrack(video_source, OriginClock(7130 / 90_000), nominal_fps=fps)
        audio = WallClockAudioTrack(audio_source, OriginClock(2173 / 48_000))
        recorder = RTPMP4Recorder(str(output), video_fps=fps)
        recorder.addTrack(video)
        recorder.addTrack(audio)
        try:
            await recorder.start()
            await asyncio.wait_for(asyncio.gather(video_source.exhausted.wait(),
                                                  audio_source.exhausted.wait()), timeout=5)
        finally:
            await recorder.stop()
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            rate = stream.average_rate
            decoded_video = [(frame.pts, frame.time_base) for frame in container.decode(video=0)]
        with av.open(str(output)) as container:
            audio_frames = list(container.decode(audio=0))
        return video.get_stats(), audio.get_stats(), decoded_video, rate, audio_frames

    async def test_real_20hz_video_and_audio_preserve_pts_and_uniform_cadence(self):
        source_pts = [630_000 + index * 4500 for index in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            video, audio, frames, rate, audio_frames = await self._roundtrip(
                Path(directory) / "twenty-fps.mp4", source_pts)
        times = [pts * base for pts, base in frames]
        self.assertEqual(rate, 20)
        self.assertEqual(times, [Fraction(7130 + index * 4500, 90_000) for index in range(20)])
        self.assertEqual({right - left for left, right in zip(times, times[1:])}, {Fraction(1, 20)})
        self.assertEqual(video["frames"], 20)
        self.assertEqual(video["source_origin_seconds"], 7.0)
        self.assertEqual(video["last_source_seconds"], 7.95)
        self.assertEqual(video["recording_origin_pts"], 7130)
        self.assertEqual(video["last_recording_pts"], 7130 + 19 * 4500)
        self.assertEqual(video["recording_time_base"], {"numerator": 1, "denominator": 90_000})
        self.assertEqual(video["source_timestamp_anomalies"], 0)
        self.assertEqual(audio["source_timestamp_anomalies"], 0)
        self.assertEqual(audio["recording_origin_pts"], 2173)
        self.assertEqual(audio["source_origin_seconds"], 3.0)
        self.assertGreater(len(audio_frames), 0)
        # AAC retains its upstream one-frame encoder priming packet. The first
        # media timestamp after that packet remains the exact receiver origin.
        self.assertEqual(audio_frames[0].pts * audio_frames[0].time_base,
                         Fraction(2173 - 1024, 48_000))
        self.assertEqual(audio_frames[1].pts * audio_frames[1].time_base,
                         Fraction(2173, 48_000))
        for previous, current in zip(audio_frames, audio_frames[1:]):
            self.assertEqual(current.pts * current.time_base - previous.pts * previous.time_base,
                             Fraction(previous.samples, previous.sample_rate))
        self.assertTrue(any(abs(frame.to_ndarray()).max() > 0.01 for frame in audio_frames))

    async def test_real_mp4_retains_receiver_gap_and_audit_still_rejects_it(self):
        # A missing 50 ms frame and an extra 7 ms RTP discontinuity must survive
        # muxing; neither may be normalized onto a nominal 20 Hz grid.
        source_pts = [630_000, 634_500, 643_500, 648_630, 653_130]
        with tempfile.TemporaryDirectory() as directory:
            video, audio, frames, _, _ = await self._roundtrip(
                Path(directory) / "gap.mp4", source_pts)
        times = [pts * base for pts, base in frames]
        self.assertEqual(times, [Fraction(7130 + pts - source_pts[0], 90_000) for pts in source_pts])
        self.assertEqual(video["source_timestamp_anomalies"], 2)
        with self.assertRaisesRegex(SmokeTestError, "undeclared gap"):
            validate_recording_timestamp_proof(
                {"video": video, "audio": audio}, [], playback_fps=20)

    async def test_incompatible_upstream_registry_fails_before_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RTPMP4Recorder(str(Path(directory) / "bad-api.mp4"), video_fps=20)
            try:
                with patch.object(recorder, "_MediaRecorder__tracks", None):
                    with self.assertRaisesRegex(SmokeTestError, "stream registry unavailable"):
                        recorder.addTrack(WallClockVideoTrack(_Source([]), _FixedClock(), nominal_fps=20))
            finally:
                await recorder.stop()

    async def test_constructor_api_rejection_closes_already_opened_container(self):
        container = Mock()

        def incompatible_init(recorder, *args, **kwargs):
            recorder._MediaRecorder__container = container
            recorder._MediaRecorder__tracks = None

        with patch.object(MediaRecorder, "__init__", incompatible_init):
            with self.assertRaisesRegex(SmokeTestError, "stream registry unavailable"):
                RTPMP4Recorder("unused.mp4", video_fps=20)
        container.close.assert_called_once_with()

    async def test_recorder_can_close_before_any_track_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RTPMP4Recorder(str(Path(directory) / "unstarted.mp4"), video_fps=20)
            recorder.addTrack(WallClockVideoTrack(_Source([]), _FixedClock(), nominal_fps=20))
            await recorder.stop()
            await recorder.stop()
            with self.assertRaisesRegex(SmokeTestError, "closed aiortc recorder"):
                recorder._recorder_parts()

    async def test_rejects_invalid_fps_before_opening_output(self):
        for fps in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                RTPMP4Recorder("unused.mp4", video_fps=fps)


if __name__ == "__main__":
    unittest.main()
