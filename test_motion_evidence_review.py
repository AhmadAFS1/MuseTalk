"""Ownership and following-turn proof in saved barge-in recording evidence."""
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from scripts.motion_transitions import MotionBank
from scripts.review_motion_evidence import verify_barge_in, verify_recording, verify_saved_video_cadence
from test_motion_transitions import fixture


def live(output, generation):
    return {"mode": "live", "output_frame": output, "generation_frame": generation,
            "pose_id": "neutral_resting", "source_frame": generation}


def status(motion, *, active_stream=None, protocol=None):
    return {"active_stream": active_stream,
            "pose_protocol": protocol or {"assistant_active": False},
            "track_stats": {"video": {"live_active": bool(active_stream), "motion": motion}}}


def barge_in_fixture():
    bank = MotionBank(fixture())
    first_entry = {"generation_id": 1, "status": "completed", "first_live_generation_frame": 0,
                   "first_live_output_frame": 20, "additional_start_seconds": .3}
    next_entry = {**first_entry, "generation_id": 3, "first_live_output_frame": 70}
    returned = {"status": "completed", "source_output_frame": 45, "total_seconds": .4}
    first_motion = {"bank": {"routing_sha256": bank.routing_sha256}, "settled": True,
                    "entries": [first_entry], "returns": [returned],
                    "trace": [live(20, 0), live(40, 20), live(45, 25)]}
    next_motion = copy.deepcopy(first_motion)
    next_motion["entries"].append(next_entry)
    next_motion["returns"].append({**returned, "source_output_frame": 85})
    next_motion["trace"].extend([live(70, 0), live(71, 1), live(85, 15)])
    owner_motion = copy.deepcopy(first_motion)
    owner_motion["last_emitted"] = live(40, 20)
    owner_motion["settled"] = False
    events = lambda event, seq, turn_id: {"accepted": True, "event": event, "seq": seq, "turn_id": turn_id}
    first_turn = {
        "turn_id": "assistant-A", "accepted": {"request_id": "request-A", "pose_plan": {
            "accepted": True, "seq": 1, "turn_id": "assistant-A"}},
        "first_possible_output_frame": 10, "submitted_at_seconds": .5,
        "interrupted_at_seconds": 2.5, "complete_at_seconds": 3,
        "final_status": status(first_motion), "observed_body_poses": ["neutral_resting"],
        "interrupt_response": {**events("assistant_turn_aborted", 3, "assistant-A"), "motion_return_started": True,
                               "pose_status": {"user_speaking": True}},
        "barge_in": {
            "user_turn_id": "user-B", "started": events("user_speech_started", 2, "user-B"),
            "ended": {**events("user_speech_ended", 4, "user-B"), "pose_status": {"user_speaking": False}},
            "status_before_abort": status(owner_motion, active_stream="request-A", protocol={
                "active_turn_id": "assistant-A", "assistant_active": True, "user_speaking": True,
                "last_event": "user_speech_started", "last_seq": 2}),
        },
    }
    next_turn = {
        "turn_id": "assistant-C", "accepted": {"request_id": "request-C", "pose_plan": {
            "accepted": True, "seq": 11, "turn_id": "assistant-C"}},
        "first_possible_output_frame": 60, "submitted_at_seconds": 3.01,
        "complete_at_seconds": 4.5, "final_status": status(next_motion),
        "observed_body_poses": ["neutral_resting"],
    }
    tracks = {kind: {"frames": 100, "source_timestamp_anomalies": 0,
                     "source_timestamp_missing": 0} for kind in ("audio", "video")}
    return bank, {"case": "barge-in", "success": True, "timestamp_audit_passed": True,
                  "fps": 20, "bank": {"routing_sha256": bank.routing_sha256,
                  "source_hashes": {p: s["sha256"] for p, s in bank.sources.items()}},
                  "receiver_tracks": tracks, "turns": [first_turn, next_turn],
                  "final_status": status(next_motion), "longest_idle_source_hold_frames": 1}


class BargeInEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.bank, self.report = barge_in_fixture()

    def test_distinct_user_id_preserves_assistant_abort_and_clean_next_reply(self):
        proof = verify_barge_in(self.report)
        self.assertEqual(proof["events_seq"], [1, 2, 3, 4, 11])
        self.assertEqual(proof["assistant_turn_id"], "assistant-A")
        self.assertEqual(proof["user_turn_id"], "user-B")
        self.assertEqual(proof["following_first_live_generation_frame"], 0)

    def test_missing_barge_in_evidence_cannot_pass_on_success_flag_alone(self):
        self.report["turns"][0].pop("barge_in")
        with self.assertRaisesRegex(ValueError, "turn IDs"):
            verify_barge_in(self.report)

    def test_user_event_must_not_replace_assistant_owner_or_active_request(self):
        for field, value in (("active_turn_id", "user-B"), ("user_speaking", False),
                             ("assistant_active", False), ("last_seq", 1)):
            report = copy.deepcopy(self.report)
            report["turns"][0]["barge_in"]["status_before_abort"]["pose_protocol"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ownership changed"):
                verify_barge_in(report)
        self.report["turns"][0]["barge_in"]["status_before_abort"]["active_stream"] = "request-C"
        with self.assertRaisesRegex(ValueError, "ownership changed"):
            verify_barge_in(self.report)

    def test_wrong_abort_owner_rejection_and_reordered_events_fail(self):
        for field, value, message in (("turn_id", "user-B", "accepted assistant_turn_aborted"),
                                      ("accepted", False, "accepted assistant_turn_aborted"),
                                      ("seq", 2, "strictly ordered"),
                                      ("motion_return_started", False, "motion recovery")):
            report = copy.deepcopy(self.report)
            report["turns"][0]["interrupt_response"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                verify_barge_in(report)

    def test_abort_must_preserve_ongoing_user_speech(self):
        for value in (False, None):
            self.report["turns"][0]["interrupt_response"]["pose_status"]["user_speaking"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "incorrectly ended"):
                verify_barge_in(self.report)

    def test_abort_requires_actual_live_overlap_and_settled_return(self):
        report = copy.deepcopy(self.report)
        report["turns"][0]["barge_in"]["status_before_abort"]["track_stats"]["video"]["motion"]["last_emitted"]["mode"] = "idle"
        with self.assertRaisesRegex(ValueError, "overlap emitted"):
            verify_barge_in(report)
        self.report["turns"][0]["final_status"]["track_stats"]["video"]["motion"]["settled"] = False
        with self.assertRaisesRegex(ValueError, "settle and release"):
            verify_barge_in(self.report)

    def test_late_cancelled_live_frame_and_stale_following_generation_fail(self):
        report = copy.deepcopy(self.report)
        report["turns"][0]["final_status"]["track_stats"]["video"]["motion"]["trace"].append(live(46, 26))
        with self.assertRaisesRegex(ValueError, "continued after its return"):
            verify_barge_in(report)
        report = copy.deepcopy(self.report)
        report["turns"][1]["final_status"]["track_stats"]["video"]["motion"]["trace"][-3]["generation_frame"] = 26
        with self.assertRaisesRegex(ValueError, "initial generation frame"):
            verify_barge_in(report)
        self.report["turns"][1]["final_status"]["track_stats"]["video"]["motion"]["trace"].insert(-1, live(72, 26))
        with self.assertRaisesRegex(ValueError, "stale or reordered"):
            verify_barge_in(self.report)

    def test_following_reply_waits_for_recovery_and_user_end(self):
        report = copy.deepcopy(self.report)
        report["turns"][1]["submitted_at_seconds"] = 2.9
        with self.assertRaisesRegex(ValueError, "before.*settled"):
            verify_barge_in(report)
        self.report["turns"][0]["barge_in"]["ended"]["pose_status"]["user_speaking"] = True
        with self.assertRaisesRegex(ValueError, "did not end"):
            verify_barge_in(self.report)

    def test_recording_verifier_runs_barge_in_validation_and_exports_proof(self):
        media = {"streams": [{"codec_type": "audio"}, {"codec_type": "video", "time_base": "1/90000"}],
                 "format": {"duration": "5.0"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "barge-in.json"
            path.write_text(json.dumps(self.report))
            decoded = {"frames": [{"pts": 7130 + 4500 * index} for index in range(100)]}
            with patch("scripts.review_motion_evidence.subprocess.check_output",
                       side_effect=[json.dumps(media), json.dumps(decoded)]), \
                    patch("scripts.review_motion_evidence.file_hash", return_value="media-hash"):
                result = verify_recording(path, self.bank)
                self.assertEqual(result["barge_in"]["following_turn_id"], "assistant-C")
                self.assertEqual(result["saved_video_cadence"]["decoded_frames"], 100)
                self.assertEqual(result["saved_video_cadence"]["max_frame_seconds"], .05)
                self.report["turns"][0]["interrupt_response"]["turn_id"] = "user-B"
                path.write_text(json.dumps(self.report))
                with self.assertRaisesRegex(ValueError, "accepted assistant_turn_aborted"):
                    verify_recording(path, self.bank)


class SavedVideoCadenceTest(unittest.TestCase):
    def _verify(self, points, *, receiver_count=None, time_base="1/90000"):
        media = {"streams": [{"codec_type": "video", "time_base": time_base}]}
        decoded = {"frames": [{"pts": point} if point is not None else {} for point in points]}
        with patch("scripts.review_motion_evidence.subprocess.check_output", return_value=json.dumps(decoded)):
            return verify_saved_video_cadence(Path("received.mp4"), media,
                {"frames": len(points) if receiver_count is None else receiver_count}, 20)

    def test_preserves_nonzero_origin_and_allows_only_one_video_tick_rounding(self):
        metrics = self._verify([7130, 11631, 16130, 20630])
        self.assertEqual(metrics["first_seconds"], 7130 / 90000)
        self.assertEqual(metrics["max_frame_error_seconds"], 1 / 90000)
        self.assertEqual(metrics["decoded_frames"], 4)
        with self.assertRaisesRegex(ValueError, "cadence error"):
            self._verify([7130, 11632, 16130])

    def test_legacy_quantization_and_real_timestamp_gap_both_fail(self):
        for points, time_base in (([1536, 2048, 3072, 3584], "1/15360"),
                                  ([7130, 11630, 20630], "1/90000"),
                                  ([7130, 11630, 11630], "1/90000")):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "cadence error"):
                self._verify(points, time_base=time_base)

    def test_missing_pts_and_decoded_frame_loss_fail(self):
        with self.assertRaisesRegex(ValueError, "decoded count"):
            self._verify([7130, 11630], receiver_count=3)
        with self.assertRaisesRegex(ValueError, "without PTS"):
            self._verify([7130, None, 16130])

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is required for real MP4 verification")
    def test_real_encoded_files_accept_20hz_and_reject_old_30hz_quantization(self):
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is required to encode the CPU fixtures")

        def encode(path, *, precise):
            with av.open(str(path), "w", options={"movie_timescale": "720000"}) as container:
                video = container.add_stream("libx264", rate=20 if precise else 30)
                video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
                if precise:
                    video.time_base = video.codec_context.time_base = Fraction(1, 90000)
                audio = container.add_stream("aac", rate=48000)
                for index in range(12):
                    frame = av.VideoFrame(64, 64, "yuv420p")
                    for plane in frame.planes:
                        plane.update(bytes([100]) * plane.buffer_size)
                    frame.pts, frame.time_base = 7130 + 4500 * index, Fraction(1, 90000)
                    for packet in video.encode(frame):
                        container.mux(packet)
                    frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
                    frame.sample_rate = 48000
                    frame.pts, frame.time_base = 2173 + 1024 * index, Fraction(1, 48000)
                    frame.planes[0].update(bytes(frame.planes[0].buffer_size))
                    for packet in audio.encode(frame):
                        container.mux(packet)
                for stream in (video, audio):
                    for packet in stream.encode(None):
                        container.mux(packet)

        with tempfile.TemporaryDirectory() as directory:
            for precise in (True, False):
                video_path = Path(directory) / f"cadence-{precise}.mp4"
                encode(video_path, precise=precise)
                media = json.loads(subprocess.check_output([
                    "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path)]))
                if precise:
                    metrics = verify_saved_video_cadence(video_path, media, {"frames": 12}, 20)
                    self.assertEqual(metrics["decoded_frames"], 12)
                    self.assertEqual(metrics["min_frame_seconds"], .05)
                    self.assertEqual(metrics["max_frame_seconds"], .05)
                    self.assertEqual(metrics["first_seconds"], 7130 / 90000)
                    origins = {stream["codec_type"]: stream["start_time"] for stream in media["streams"]}
                    self.assertNotEqual(origins["audio"], origins["video"])
                else:
                    with self.assertRaisesRegex(ValueError, "Saved MP4 cadence error"):
                        verify_saved_video_cadence(video_path, media, {"frames": 12}, 20)
                # Integration must fail even when the saved JSON declares a
                # successful RTP audit, if muxing quantized the actual frames.
                bank, report = barge_in_fixture()
                report["receiver_tracks"]["video"]["frames"] = 12
                evidence_path = video_path.with_suffix(".json")
                evidence_path.write_text(json.dumps(report))
                if precise:
                    self.assertTrue(verify_recording(evidence_path, bank)["saved_video_cadence"]["validated"])
                else:
                    with self.assertRaisesRegex(ValueError, "Saved MP4 cadence error"):
                        verify_recording(evidence_path, bank)


if __name__ == "__main__":
    unittest.main()
