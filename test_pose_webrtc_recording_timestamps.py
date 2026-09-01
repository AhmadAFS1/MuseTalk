import asyncio
import unittest
from fractions import Fraction

from scripts.test_pose_webrtc import (
    SmokeTestError,
    WallClockAudioTrack,
    WallClockVideoTrack,
    validate_recording_timestamp_proof,
)


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


if __name__ == "__main__":
    unittest.main()
