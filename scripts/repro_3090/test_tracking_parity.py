"""Synthetic CPU contracts only: no real FaceMesh, GPU or release acceptance."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import report
import runner
import tracking_parity as pair


def npy(dtype, shape, payload):
    header = (repr({"descr": dtype, "fortran_order": False, "shape": shape}) + "\n").encode("ascii")
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + payload


def write_arrays(path, *, changed=None, nonfinite=False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (dtype, shape, fmt) in pair.ARRAYS.items():
            number = float("nan") if nonfinite and name == "generated_landmarks.npy" else (1. if changed == name else 0.)
            payload = struct.pack(fmt, number) + bytes(math.prod(shape) * struct.calcsize(fmt) - struct.calcsize(fmt))
            archive.writestr(name, npy(dtype, shape, payload))


class TrackingParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        engine_root = str(self.root / "synthetic-engine")
        self.environment = {"gpu": {"uuid": "SYNTHETIC_CPU_ONLY"}, "actual_gpu": "synthetic",
                            "compute_capability": "synthetic", "runtime": {"scope": "synthetic"},
                            "input_manifest_sha256": "1" * 64, "input_count": 878, "profile_sha256": "2" * 64,
                            "effective_profile": {}, "engines": [{"root": engine_root, "manifest_sha256": "3" * 64,
                            "plan_sha256": {"synthetic": "4" * 64}, "manifest": {"probe": {"output_sha256": "5" * 64}}}],
                            "taesd": {"key": "synthetic-key", "decoder_plan_sha256": "6" * 64}}
        self.mock_face = mock.patch.object(pair.quality_envelope, "face_array_hash", side_effect=lambda p: report.sha256(p))
        self.mock_face.start()
        self.addCleanup(self.mock_face.stop)
        self.a = self.make_capture("synthetic_tracking_serial", False)
        self.b = self.make_capture("synthetic_tracking_overlap", True)

    def make_capture(self, label, overlap):
        root = self.root / label
        root.mkdir()
        hashes, workers = {}, {}
        for i, ident in enumerate(report.IDENTITIES):
            prefix = root / f"stream{i:02d}_{ident}"
            face = Path(str(prefix) + "_faces.npz")
            face.write_bytes(b"SYNTHETIC mocked face payload " + ident.encode())
            write_arrays(Path(str(prefix) + "_arrays.npz"))
            face_sha = report.sha256(face)
            hashes[ident] = [["7" * 64, face_sha]]
            workers[str(i)] = {"identity": ident, "frames": 240, "tracking_overlap": overlap,
                               "clips": [{"loop": 0, "raw_refined_sha256": "7" * 64, "generated_faces_sha256": face_sha}]}
        canonical = pair.ROOT / "character_factory/h3_avatar_workflow"
        data = {"status": "complete", "label": label,
                "args": {"streams": 6, "loops": 1, "repeats": 1, "pack": 16, "decode_split": 8,
                         "mode": "multi", "backend": "stagewise16_taesdtrt", "identity_list": list(report.IDENTITIES),
                         "save_arrays": True, "encode": True, "tracking_overlap": overlap, "label": label,
                         "out_root": str(self.root), "cv2_threads": 2},
                "code_integrity": {"matches_accepted_render_json": True,
                                   "files": {n: report.sha256(canonical / n) for n in ("backend.py", "tracker_worker.py", "chin.py", "render_stage.py")},
                                   "blending_py": report.sha256(pair.ROOT / "musetalk/utils/blending.py")},
                "harness_code_sha256": pair.current_harness_hashes(),
                "summary": {"deterministic_per_identity": True},
                "exactness": {"deterministic_per_identity": {i: True for i in report.IDENTITIES},
                              "clips_checked": 6, "clips_expected": 6, "distinct_hash_pairs_per_identity": hashes},
                "repeats": [{"frames": 1440, "per_worker": workers}], "versions": {"scope": "synthetic"}, "env": {},
                "backends": {"unet_name": "tensorrt_unet_stagewise", "decoder_name": "taesd_trt",
                             "unet_describe": {"engine_dir": self.environment["engines"][0]["root"] + "/bs16", "batch": 16,
                                               "probe_status": "exact", "probe_validation": {"kind": "exact", "expected_sha256": "5" * 64, "actual_sha256": "5" * 64}},
                             "decoder_trt_key": "synthetic-key", "decoder_trt_plan_sha256": "6" * 64}}
        path = self.root / (label + ".json")
        path.write_text(json.dumps(data))
        return path

    def edit(self, path, fn):
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data))

    def test_all_six_exact_arrays_and_hashes_pass_only_scheduler_scope(self):
        result = pair.compare(self.a, self.b, self.environment)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["matches"]), 6)
        self.assertTrue(all(all(x.values()) for x in result["matches"].values()))
        self.assertFalse(result["release_quality_accepted"])
        self.assertIn("not_release_quality_or_throughput", result["scope"])

    def test_changed_finite_landmark_or_delta_is_valid_fail_not_parity(self):
        path = self.root / "synthetic_tracking_overlap" / f"stream00_{report.IDENTITIES[0]}_arrays.npz"
        for changed in pair.ARRAYS:
            write_arrays(path, changed=changed)
            result = pair.compare(self.a, self.b, self.environment)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["matches"][report.IDENTITIES[0]][changed])

    def test_nonfinite_truncated_unknown_member_and_malformed_npy_are_invalid(self):
        path = self.root / "bad.npz"
        write_arrays(path, nonfinite=True)
        with self.assertRaisesRegex(report.Invalid, "nonfinite"):
            pair.array_hashes(path)
        for name, payload in (("unknown.npy", b"bad"), ("generated_landmarks.npy", b"bad")):
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(name, payload)
            with self.assertRaises(report.Invalid):
                pair.array_hashes(path)
        path.write_bytes(b"not zip")
        with self.assertRaisesRegex(report.Invalid, "malformed"):
            pair.array_hashes(path)

    def test_exact_member_wrong_shape_dtype_magic_version_and_payload_are_invalid(self):
        path = self.root / "bad-npy.npz"
        payload = bytes(240 * 478 * 2 * 4)
        valid = npy("<f4", (240, 478, 2), payload)
        cases = [npy("|O", (240, 478, 2), payload), npy("<f4", (240, 477, 2), payload),
                 b"BADMAG" + valid[6:], valid[:6] + b"\x03\x00" + valid[8:],
                 npy("<f4", (240, 478, 2), payload[:-1]), valid + b"EXTR"]
        for bad in cases:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("generated_landmarks.npy", bad)
                archive.writestr("chin_delta.npy", npy("<f8", (240, 161), bytes(240 * 161 * 8)))
            with self.assertRaises(report.Invalid):
                pair.array_hashes(path)

    def test_uuid_recheck_must_match_before_another_child(self):
        with mock.patch.object(runner, "capture", return_value="DIFFERENT_SYNTHETIC_GPU"):
            with self.assertRaisesRegex(report.Invalid, "GPU identity changed"):
                runner.verify_current_gpu(self.environment)

    def test_uuid_stage_runs_through_real_bounded_cpu_subprocess(self):
        def synthetic_nvml(command, **kwargs):
            self.assertEqual(command[0], "nvidia-smi")
            return runner.safe_capture.capture(
                [sys.executable, "-c", "print('SYNTHETIC_CPU_ONLY')"], cwd=pair.ROOT, **kwargs)
        with mock.patch.object(runner, "capture", side_effect=synthetic_nvml):
            runner.verify_current_gpu(self.environment)

    def test_missing_avatar_ignored_mode_source_and_workload_changes_are_invalid(self):
        original = self.b.read_text()
        changes = [lambda d: d["repeats"][0]["per_worker"].pop("5"),
                   lambda d: d["args"].update(tracking_overlap=False),
                   lambda d: d["repeats"][0]["per_worker"]["0"].update(tracking_overlap=False),
                   lambda d: d["harness_code_sha256"].update({"worker.py": "0" * 64}),
                   lambda d: d["code_integrity"]["files"].update({"chin.py": "0" * 64}),
                   lambda d: d["args"].update(cv2_threads=4)]
        for fn in changes:
            self.b.write_text(original)
            self.edit(self.b, fn)
            with self.assertRaises(report.Invalid):
                pair.compare(self.a, self.b, self.environment)

    def test_same_capture_path_wrong_engine_or_reported_pixels_cannot_pass(self):
        with self.assertRaisesRegex(report.Invalid, "distinct"):
            pair.compare(self.a, self.a, self.environment)
        self.edit(self.b, lambda d: d["backends"].update(decoder_trt_plan_sha256="0" * 64))
        with self.assertRaisesRegex(report.Invalid, "TAESD plan"):
            pair.compare(self.a, self.b, self.environment)

    def test_saved_face_mismatch_and_missing_array_cannot_use_reported_hashes(self):
        face = self.root / "synthetic_tracking_overlap" / f"stream00_{report.IDENTITIES[0]}_faces.npz"
        face.write_bytes(b"changed synthetic face pixels")
        with self.assertRaisesRegex(report.Invalid, "saved faces differ"):
            pair.compare(self.a, self.b, self.environment)
        face.write_bytes(b"SYNTHETIC mocked face payload " + report.IDENTITIES[0].encode())
        arrays = self.root / "synthetic_tracking_overlap" / f"stream00_{report.IDENTITIES[0]}_arrays.npz"
        arrays.unlink()
        with self.assertRaisesRegex(report.Invalid, "missing or unsafe raw"):
            pair.compare(self.a, self.b, self.environment)

    def test_completed_refined_hash_difference_is_valid_fail(self):
        ident = report.IDENTITIES[0]
        def change(d):
            d["exactness"]["distinct_hash_pairs_per_identity"][ident][0][0] = "8" * 64
            d["repeats"][0]["per_worker"]["0"]["clips"][0]["raw_refined_sha256"] = "8" * 64
        self.edit(self.b, change)
        result = pair.compare(self.a, self.b, self.environment)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["matches"][ident]["raw_refined_sha256"])

    def test_receipt_is_recomputed_and_cannot_transfer_to_another_gpu(self):
        row = pair.compare(self.a, self.b, self.environment)
        parent = {"schema": "repro_3090_v1", "suite": "tracking-parity", "status": "PASS", "results": [row]}
        path = self.root / "receipt.json"
        path.write_text(json.dumps(parent))
        self.assertTrue(pair.require_passed(path, self.environment)["verified"])
        changed = copy.deepcopy(self.environment)
        changed["gpu"]["uuid"] = "OTHER_SYNTHETIC_GPU"
        with self.assertRaisesRegex(report.Invalid, "identity differs"):
            pair.require_passed(path, changed)
        arrays = self.root / "synthetic_tracking_overlap" / f"stream00_{report.IDENTITIES[0]}_arrays.npz"
        write_arrays(arrays, changed="chin_delta.npy")
        with self.assertRaisesRegex(report.Invalid, "evidence changed"):
            pair.require_passed(path, self.environment)

    def test_command_pair_differs_only_in_label_and_explicit_overlap(self):
        args = SimpleNamespace(python="synthetic")
        serial = runner.tracking_capture_command(args, self.root, "serial", False)
        overlap = runner.tracking_capture_command(args, self.root, "overlap", True)
        self.assertNotIn("--tracking-overlap", serial)
        self.assertEqual(overlap[-1], "--tracking-overlap")
        serial[serial.index("--label") + 1] = "same"
        overlap[overlap.index("--label") + 1] = "same"
        self.assertEqual(serial, overlap[:-1])
        self.assertIn("--save-arrays", serial)
        self.assertIn("--encode", serial)

    def test_overlap_cli_requires_successful_parity_before_any_gpu_work(self):
        command = [sys.executable, str(pair.ROOT / "scripts/repro_3090/runner.py"), "aggregate", "--tracking-overlap"]
        for name in ("profile", "engine-root", "taesd-key", "taesd-dir", "input-manifest"):
            command += ["--" + name, "synthetic"]
        command += ["--out", str(self.root / "never-created"), "--label", "synthetic"]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("requires --tracking-parity-report", run.stderr)
        self.assertFalse((self.root / "never-created").exists())

    def test_pair_suite_executes_two_guarded_children_with_four_uuid_checks(self):
        out = self.root / "suite"
        profile = self.root / "profile.env"
        profile.write_text("MUSETALK_TAESD_TRT_OPT_LEVEL=3\n")
        inputs = self.root / "inputs.json"
        inputs.write_text(json.dumps({"schema": "repro_3090_inputs_v1", "files": []}))
        argv = ["runner.py", "tracking-parity", "--profile", str(profile), "--engine-root", self.environment["engines"][0]["root"],
                "--taesd-key", "synthetic-key", "--taesd-dir", str(self.root), "--input-manifest", str(inputs), "--out", str(out), "--label", "synthetic"]
        calls = []
        def child(args, root, env, name, command, gb):
            calls.append((name, list(command), gb))
            (root / ("synthetic_" + name + ".json")).write_text("{}")
            return 0
        row = {"schema": pair.SCHEMA, "status": "PASS", "scope": "synthetic_cpu_test"}
        with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "preflight", return_value=self.environment), \
                mock.patch.object(runner, "capture", return_value="SYNTHETIC_CPU_ONLY"), \
                mock.patch.object(runner, "child", side_effect=child), \
                mock.patch.object(runner, "engine", return_value=self.environment["engines"][0]), \
                mock.patch.object(runner.checks, "verify_files") as verify_inputs, \
                mock.patch.object(runner.tracking_parity, "compare", return_value=row), \
                mock.patch.object(runner, "verify_current_gpu", wraps=runner.verify_current_gpu) as verify_gpu:
            self.assertEqual(runner.main(), 0)
        self.assertEqual(verify_gpu.call_count, 4)
        verify_inputs.assert_called_once()
        self.assertEqual([x[0] for x in calls], ["tracking_serial", "tracking_overlap"])
        self.assertNotIn("--tracking-overlap", calls[0][1])
        self.assertIn("--tracking-overlap", calls[1][1])
        parent = json.loads((out / "synthetic_tracking-parity/report.json").read_text())
        self.assertEqual(parent["quality_parity_with_reference"], "NOT_EVALUATED")
        self.assertIn("scheduler equality only", parent["release_quality_decision"])


if __name__ == "__main__":
    unittest.main()
