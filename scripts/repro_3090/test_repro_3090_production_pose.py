"""CPU-only production audit contract tests; not render/GPU/visual evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import production_pose_audit as audit


def publication():
    return {"characters": [{"id": f"character_{i:02d}", "poses": {
        pose: {"avatar_id": f"character_{i:02d}_{pose}",
               "s3_uri": f"s3://private-bucket/avatars/v15/character_{i:02d}_{pose}.tar.gz",
               "bytes": 100, "source_video_sha256": hashlib.sha256(b"video").hexdigest(),
               "metadata": {"avatar-id": f"character_{i:02d}_{pose}", "musetalk-version": "v15"}}
        for pose in ("idle", "talking", "smiling")}} for i in range(16)]}


class AuditContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pub = self.root / "publication.json"
        self.pub.write_text(json.dumps(publication()))
        self.audio = self.root / "known-speech.wav"
        self.audio.write_bytes(b"not decoded in plan mode")
        self.audio_sha = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        self.avatars = self.root / "avatars"
        self.avatars.mkdir()

    def plan(self, selected=()):
        return audit.make_plan(self.pub, self.avatars, self.audio, self.audio_sha, "Known recorded speech fixture", selected)

    def arguments(self, out):
        return ["--publication", str(self.pub), "--avatars-root", str(self.avatars), "--audio", str(self.audio),
                "--audio-sha256", self.audio_sha, "--speech-source", "Known speech fixture", "--out", str(out)]

    def test_all_48_are_separate_from_six_fixture_claims(self):
        plan = self.plan()
        self.assertEqual(len(plan["items"]), 48)
        self.assertEqual(plan["status"], "PLAN_ONLY")
        self.assertFalse(plan["gpu_used"])
        self.assertFalse(plan["coverage_complete"])
        self.assertFalse(plan["release_ready"])
        self.assertEqual(plan["visual_review_status"], "NOT_PERFORMED")
        self.assertEqual(plan["recipe"]["chin_strength"], 1.0)
        self.assertFalse(plan["recipe"]["resize_or_crop_output"])
        self.assertEqual({p["pose"] for p in plan["items"]}, {"idle", "talking", "smiling"})

    def test_subset_cannot_be_full_coverage(self):
        plan = self.plan(["character_00_talking"])
        self.assertTrue(plan["diagnostic_subset"])
        self.assertEqual(plan["selected_poses"], 1)
        result = audit.summarize([{"avatar_id": "x", "status": "PASS"}], 1)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["full_48_render_compatibility"], "INCOMPLETE")
        self.assertFalse(result["release_ready"])

    def test_missing_pose_duplicate_unknown_and_path_ids_rejected(self):
        bad = publication()
        del bad["characters"][0]["poses"]["idle"]
        with self.assertRaises(ValueError):
            audit.select_items(bad)
        for selection in (["character_00_idle"] * 2, ["missing"], ["../elsewhere"]):
            with self.assertRaises(audit.checks.Invalid):
                audit.select_items(publication(), selection)

    def test_audio_reference_is_bound_not_guessed(self):
        self.audio.write_bytes(b"changed")
        with self.assertRaisesRegex(audit.checks.Invalid, "audio differs"):
            self.plan()

    def test_output_cannot_write_inside_avatars_or_overwrite(self):
        for out in (self.avatars / "audit", self.root, self.avatars):
            with self.assertRaises(audit.checks.Invalid):
                audit.independent_output(out, self.avatars)
        audit.independent_output(self.root / "new-run", self.avatars)

    def test_plan_cli_never_imports_gpu_or_starts_subprocess(self):
        with mock.patch.object(audit.runner, "child", side_effect=AssertionError("GPU child forbidden")):
            result = audit.main(self.arguments(self.root / "plan"))
        self.assertEqual(result, 0)
        self.assertNotIn("torch", sys.modules)
        report = json.loads((self.root / "plan/report.json").read_text())
        self.assertEqual(report["status"], "PLAN_ONLY")

    def test_venv_interpreter_symlink_is_not_resolved_to_system_python(self):
        system = self.root / "system-python"
        system.write_bytes(b"synthetic interpreter placeholder")
        interpreter = self.root / "venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(system)
        args = argparse.Namespace(tracker_python=interpreter, audio=self.audio)
        audit.normalize_paths(args)
        self.assertEqual(args.tracker_python, interpreter.absolute())
        self.assertNotEqual(args.tracker_python, system.resolve())
        self.assertEqual(args.audio, self.audio.resolve())

    def test_short_smiling_source_uses_existing_forward_reverse_cache(self):
        source = {"width": 512, "height": 896, "avg_frame_rate": "24/1", "nb_frames": "158"}
        result = audit.validate_source_timeline(source, 316)
        self.assertEqual(result["selected_cache_frames"], 240)
        self.assertEqual(result["cache_cycle_frames"], 316)
        self.assertFalse(result["new_frames_generated_or_resampled"])
        for change, count in (({}, 315), ({"nb_frames": "100"}, 200), ({"avg_frame_rate": "25/1"}, 316),
                              ({"width": 256}, 316), ({"nb_frames": "0"}, 316)):
            with self.subTest(change=change, count=count), self.assertRaises(audit.checks.Invalid):
                audit.validate_source_timeline({**source, **change}, count)

    def test_execution_missing_settings_is_invalid_without_gpu(self):
        with mock.patch.object(audit.runner, "child", side_effect=AssertionError("GPU child forbidden")):
            result = audit.main(self.arguments(self.root / "invalid") + ["--execute-render"])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads((self.root / "invalid/report.json").read_text())["status"], "INVALID")

    def test_no_visual_approval_even_all_render_pass(self):
        rows = [{"avatar_id": str(i), "status": "PASS"} for i in range(48)]
        result = audit.summarize(rows, 48)
        self.assertEqual(result["full_48_render_compatibility"], "PASS")
        self.assertEqual(result["visual_review_status"], "NOT_PERFORMED")
        self.assertFalse(result["release_ready"])
        self.assertEqual(audit.summarize(rows[:-1], 48)["status"], "INVALID")
        rows[0]["status"] = "FAIL"
        self.assertEqual(audit.summarize(rows, 48)["status"], "FAIL")
        rows[0]["status"] = "INVALID"
        self.assertEqual(audit.summarize(rows, 48)["status"], "INVALID")
        self.assertEqual(audit.summarize([rows[0]] * 48, 48)["status"], "INVALID")
        self.assertEqual(audit.summarize([], 0)["status"], "INVALID")
        self.assertEqual(audit.summarize([{"avatar_id": "x", "status": "UNKNOWN"}], 1)["status"], "INVALID")

    def test_geometry_preserves_native_resolution_and_complete_face_box(self):
        good = ([140, 247, 378, 500], [19, 167, 499, 647], [896, 512, 3], [480, 480])
        audit.validate_geometry(*good)
        audit.validate_geometry(good[0], [-10, 167, 550, 647], good[2], [480, 560])
        for index, bad in ((0, [140, 247, 600, 500]), (1, [19, 167, 350, 647]),
                           (2, [448, 256, 3]), (3, [240, 240])):
            args = copy.deepcopy(good)
            args = list(args)
            args[index] = bad
            with self.assertRaises(audit.checks.Invalid):
                audit.validate_geometry(*args)

    def test_plain_coordinate_pickles_and_reject_trailing_object(self):
        path = self.root / "coords.pkl"
        path.write_bytes(pickle.dumps([(1, 2, 3, 4)]))
        self.assertEqual(audit.coordinates(path, 1), [(1, 2, 3, 4)])
        path.write_bytes(pickle.dumps([(1, 2, 3, 4)]) + pickle.dumps(42))
        with self.assertRaisesRegex(audit.checks.Invalid, "trailing"):
            audit.coordinates(path, 1)
        for obj in ([[(1,), 2, 3, 4]], [[True, 2, 3, 4]], [[1., 2, 3, 4]], [[1, 2, 3]]):
            path.write_bytes(pickle.dumps(obj))
            with self.assertRaises(audit.checks.Invalid):
                audit.coordinates(path, 1)

    def test_numpy_integer_pickle_without_loading_numpy(self):
        # Protocol4 fragment matching the real restored cache's dtype/scalar
        # globals, but consumed by stdlib integer stand-ins, never NumPy reducers.
        dtype = (b"cnumpy\ndtype\n\x8c\x02i4\x89\x88\x87R"
                 b"(K\x03\x8c\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb")
        scalar = b"cnumpy.core.multiarray\nscalar\n" + dtype + b"C\x04\x8c\x00\x00\x00\x86R"
        data = b"\x80\x04]](" + scalar + b"K\x02K\x03K\x04ea."
        path = self.root / "numpy-coords.pkl"
        path.write_bytes(data)
        self.assertEqual(audit.coordinates(path, 1), [(140, 2, 3, 4)])
        with self.assertRaises(audit.checks.Invalid):
            audit._IntegerDType("O8")

    def test_coordinate_pickle_cannot_execute_globals(self):
        class Malicious:
            def __reduce__(self):
                return eval, ("1+1",)
        path = self.root / "malicious.pkl"
        path.write_bytes(pickle.dumps(Malicious()))
        with self.assertRaisesRegex(audit.checks.Invalid, "forbidden global"):
            audit.coordinates(path, 1)

    def test_frozen_input_missing_changed_or_duplicate_rejected(self):
        manifest = self.root / "inputs.json"
        row = {"path": self.audio.name, "sha256": self.audio_sha}
        manifest.write_text(json.dumps({"files": [row]}))
        audit.verify_frozen_inputs(manifest, [self.audio])
        with self.assertRaises(audit.checks.Invalid):
            audit.verify_frozen_inputs(manifest, [self.pub])
        manifest.write_text(json.dumps({"files": [row, row]}))
        with self.assertRaises(audit.checks.Invalid):
            audit.verify_frozen_inputs(manifest, [self.audio])
        manifest.write_text(json.dumps({"files": [row]}))
        self.audio.write_bytes(b"changed")
        with self.assertRaises(audit.checks.Invalid):
            audit.verify_frozen_inputs(manifest, [self.audio])

    def test_runtime_policy_requires_full_face_no_build_verified_probes(self):
        profile = self.root / "profile.env"
        profile.write_text("MUSETALK_TRT_FALLBACK=1\nMUSETALK_UNET_STAGEWISE_PROBE_TOL=1\nMUSETALK_SOURCE_MOUTH_BLEND=1\nMUSETALK_TAESD_TRT_BUILD=1\n")
        args = argparse.Namespace(profile=profile, engine_root=self.root / "engines", taesd_dir=self.root / "taesd")
        policy = audit.runtime_policy(args)
        for key in ("MUSETALK_TRT_FALLBACK", "MUSETALK_UNET_STAGEWISE_PROBE_TOL", "MUSETALK_SOURCE_MOUTH_BLEND", "MUSETALK_TAESD_TRT_BUILD"):
            self.assertEqual(policy[key], "0")
        self.assertEqual(policy["MUSETALK_UNET_STAGEWISE_VERIFY_SHA"], "1")
        self.assertEqual(policy["MUSETALK_UNET_STAGEWISE_PROBE_CHECK"], "1")
        self.assertEqual(policy["MUSETALK_TAESD_TRT_STRICT"], "1")


if __name__ == "__main__":
    unittest.main()
