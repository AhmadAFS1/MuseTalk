"""CPU-only operator review tests using synthetic media and local receipts."""
import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from character_factory.scripts import review_realtime_character as review
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, atomic_json, configured_bank, file_hash, publish_bank
from scripts.review_motion_evidence import build_review


class RecordedReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_temp = tempfile.TemporaryDirectory()
        cls.media = Path(cls.media_temp.name) / "received.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x96:r=24",
                        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.5",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(cls.media)],
                       check=True, stdout=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls):
        cls.media_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.character = self.root / "character"
        self.character.mkdir()
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.atlas_path = self.character / "motion-atlas.json"
        self.package_path = self.character / "character.json"
        self.manifest = {"version": 1, "fps": 24, "status": "candidate_requires_recorded_review",
            "sources": {}, "edges": {p: {q: [{"target_frame": (i+1)%12, "score": 0, "admissible": True}
                                             for i in range(12)] for q in (IDLE,TALK,SMILE)}
                                      for p in (IDLE,TALK,SMILE)}}
        self.paths = {}
        for pose, name in review.SOURCE_NAMES.items():
            path = self.sources / (name+".mp4")
            shutil.copy2(self.media, path)
            self.paths[pose] = str(path)
            self.manifest["sources"][pose] = {"frame_count": 12, "path": str(path), "sha256": file_hash(path)}
        publish_bank(self.atlas_path, self.manifest)
        self.bank = MotionBank(self.manifest)
        self.package = {"version": 1, "character_id": "synthetic_review_character", "source_dir": str(self.sources),
                        "status": self.manifest["status"], "motion_atlas": str(self.atlas_path),
                        "source_hashes": {review.SOURCE_NAMES[p]: v["sha256"] for p,v in self.bank.sources.items()},
                        "motion_atlas_content_sha256": self.bank.routing_sha256,
                        "prepared": {"synthetic_avatar": {"status": "ready"}}}
        atomic_json(self.package_path, self.package)
        for case in sorted(review.REQUIRED_CASES):
            shutil.copy2(self.media, self.evidence / (case+".mp4"))
            poses = [IDLE,TALK,SMILE] if case == "long-talking-smiling" else [IDLE]
            turns = [{"observed_body_poses": poses}]
            if case == "interrupted-and-next-turn":
                turns = [{"observed_body_poses": [IDLE,TALK], "interrupted_at_seconds": 3.0,
                          "interrupt_response": {"status": "accepted"},
                          "motion_before_interrupt": {"last_emitted": {"mode": "live", "pose_id": TALK}}},
                         {"observed_body_poses": [IDLE]}]
            motion = {"settled": True, "bank": {"routing_sha256": self.bank.routing_sha256},
                      "trace": [{"pose_id": IDLE, "source_frame": 0}],
                      "returns": [{"status": "completed", "total_seconds": .35} for _ in turns],
                      "entries": [{"status": "completed", "first_live_generation_frame": 0,
                                   "additional_start_seconds": .2} for _ in turns]}
            report = {"case": case, "success": True, "timestamp_audit_passed": True, "fps": 24,
                      "bank": {"routing_sha256": self.bank.routing_sha256,
                               "source_hashes": {p:v["sha256"] for p,v in self.bank.sources.items()}},
                      "receiver_tracks": {p: {"frames": 12, "source_timestamp_anomalies": 0,
                                               "source_timestamp_missing": 0} for p in ("audio", "video")},
                      "turns": turns, "final_status": {"motion": motion}, "longest_idle_source_hold_frames": 1}
            atomic_json(self.evidence / (case+".json"), report)
        self.verification = build_review(self.evidence, self.atlas_path, "Synthetic CPU review test")

    def decide(self, decision="accepted", watched=True):
        return review.review_character(self.character, self.evidence, decision=decision,
                                       reviewer="CPU test operator", notes="Synthetic fixture; test only",
                                       watched_normal_speed=watched)

    def current(self):
        return (json.loads(self.atlas_path.read_text()), json.loads(self.package_path.read_text()))

    def test_metrics_cannot_approve_without_explicit_normal_speed_attestation(self):
        with self.assertRaisesRegex(ValueError, "watched-normal-speed"):
            self.decide(watched=False)
        self.assertEqual(self.current(), (self.manifest, self.package))
        self.assertFalse((self.character / "reviews").exists())

    def test_acceptance_receipt_preserves_preparation_and_route_identity(self):
        result = self.decide()
        atlas, package = self.current()
        self.assertEqual(atlas["status"], "reviewed")
        self.assertEqual(package["status"], "reviewed")
        self.assertEqual(package["prepared"], self.package["prepared"])
        self.assertEqual(MotionBank(atlas).routing_sha256, self.bank.routing_sha256)
        self.assertEqual(package["motion_atlas_content_sha256"], self.bank.routing_sha256)
        receipt = json.loads(Path(result["receipt"]).read_text())
        self.assertEqual(receipt["signature_kind"], "operator_supplied_name_not_cryptographic")
        self.assertTrue(receipt["watched_normal_speed"])
        self.assertEqual(receipt["verification_sha256"], file_hash(self.evidence / "verification.json"))
        self.assertEqual(atlas["review"]["receipt_sha256"], file_hash(result["receipt"]))
        with patch.dict(os.environ, {"WEBRTC_MOTION_ATLAS": str(self.atlas_path), "WEBRTC_MOTION_ATLAS_DIR": "",
                                     "WEBRTC_MOTION_ALLOW_UNREVIEWED": "0"}):
            self.assertEqual(configured_bank(self.paths).routing_sha256, self.bank.routing_sha256)

    def test_explicit_rejection_after_acceptance_disables_bank_without_pilot_override(self):
        first = self.decide()
        result = self.decide(decision="rejected", watched=False)
        atlas, package = self.current()
        self.assertEqual(atlas["status"], "rejected")
        self.assertEqual(package["status"], "rejected")
        self.assertEqual(len(package["review_history"]), 2)
        self.assertTrue(Path(first["receipt"]).exists())
        self.assertNotEqual(first["receipt"], result["receipt"])
        with patch.dict(os.environ, {"WEBRTC_MOTION_ATLAS": str(self.atlas_path), "WEBRTC_MOTION_ATLAS_DIR": "",
                                     "WEBRTC_MOTION_ALLOW_UNREVIEWED": "0"}):
            with self.assertRaisesRegex(ValueError, "recorded review"):
                configured_bank(self.paths)

    def test_missing_required_case_or_failed_automated_validation_cannot_publish(self):
        for change in (lambda v: v.update(automated_validation_passed=False),
                       lambda v: v.update(recordings=v["recordings"][:-1])):
            verification = copy.deepcopy(self.verification)
            change(verification)
            atomic_json(self.evidence / "verification.json", verification)
            with self.assertRaises(ValueError): self.decide()
            self.assertEqual(self.current(), (self.manifest, self.package))

    def test_unexplained_full_atlas_hash_or_modified_prior_receipt_is_rejected(self):
        verification = copy.deepcopy(self.verification)
        verification["atlas_sha256"] = "0" * 64
        atomic_json(self.evidence / "verification.json", verification)
        with self.assertRaisesRegex(ValueError, "Atlas file changed"):
            self.decide()
        atomic_json(self.evidence / "verification.json", self.verification)
        accepted = self.decide()
        receipt = Path(accepted["receipt"])
        receipt.write_text(receipt.read_text()+" ")
        with self.assertRaisesRegex(ValueError, "Atlas file changed"):
            self.decide(decision="rejected", watched=False)

    def test_source_report_video_and_routing_tampering_cannot_publish(self):
        paths = [self.sources / "talking.mp4", self.evidence / "short-idle.json", self.evidence / "short-idle.mp4"]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original+b"changed")
            with self.subTest(path=path), self.assertRaises(ValueError): self.decide()
            path.write_bytes(original)
            self.assertEqual(self.current(), (self.manifest, self.package))
        altered = copy.deepcopy(self.manifest)
        altered["bridge_seconds"] = .25
        publish_bank(self.atlas_path, altered)
        with self.assertRaisesRegex(ValueError, "routing digest"):
            self.decide()
        self.assertEqual(json.loads(self.package_path.read_text()), self.package)

    def test_case_name_without_actual_interruption_is_not_enough(self):
        path = self.evidence / "interrupted-and-next-turn.json"
        report = json.loads(path.read_text())
        del report["turns"][0]["interrupted_at_seconds"]
        atomic_json(path, report)
        build_review(self.evidence, self.atlas_path, "Missing actual interruption")
        with self.assertRaisesRegex(ValueError, "lacks a live talking interruption"):
            self.decide()
        self.assertEqual(self.current(), (self.manifest, self.package))

    def test_package_publication_failure_rolls_back_bank_and_registration(self):
        def write(path, value):
            if Path(path) == self.package_path and value.get("status") == "reviewed":
                raise OSError("Injected package write failure")
            atomic_json(path, value)
        with patch.object(review, "atomic_json", side_effect=write):
            with self.assertRaisesRegex(OSError, "Injected"):
                self.decide()
        self.assertEqual(self.current(), (self.manifest, self.package))
        registration = json.loads((self.character / "motion-registration.json").read_text())
        self.assertEqual(registration["atlas_sha256"], file_hash(self.atlas_path))


if __name__ == "__main__": unittest.main()
