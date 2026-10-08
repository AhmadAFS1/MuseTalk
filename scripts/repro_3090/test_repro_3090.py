"""CPU-only contract tests. No test constitutes GPU/performance evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report
import runner


def aggregate_fixture(fps=400.0, repeats=2, streams=6):
    loops = 24
    total = streams * loops * 240
    wall = total / fps
    return {
        "status": "complete", "code_integrity": {"matches_accepted_render_json": True},
        "args": {"identity_list": list(report.IDENTITIES), "mode": "multi", "backend": "stagewise16_taesdtrt",
                 "pack": 16, "decode_split": 8, "encode": False, "save_arrays": False, "streams": streams, "loops": loops},
        "summary": {"deterministic_per_identity": True},
        "thermal_warmup": {"seconds": 120, "last_30s": {"n": 60, "temperature.gpu": {"min": 69, "max": 71}}},
        "repeats": [{"repeat": i, "frames": total, "wall_s": wall, "aggregate_fps": fps, "gpu": {"smi": {"n": 100}},
                     "per_worker": {str(s): {"frames": loops * 240, "fps": loops * 240 / wall, "done_after_t0_s": wall - .1}
                                    for s in range(streams)}} for i in range(repeats)]}


class Reports(unittest.TestCase):
    def test_shared_wall_denominator(self):
        result = report.aggregate(aggregate_fixture(), "T", 400)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["windows"][0]["fps"], 400)

    def test_do_not_round_399_96_up(self):
        self.assertEqual(report.aggregate(aggregate_fixture(399.96), "T", 400)["status"], "FAIL")

    def test_portable_threshold(self):
        self.assertEqual(report.aggregate(aggregate_fixture(307), "T", 300)["status"], "PASS")

    def test_all_sustained_windows_required(self):
        data = aggregate_fixture(repeats=5)
        data["repeats"][-1]["aggregate_fps"] = 399.96
        with self.assertRaisesRegex(report.Invalid, "denominator"):
            report.aggregate(data, "SUST", 400)
        with self.assertRaisesRegex(report.Invalid, "missing streams/windows"):
            report.aggregate(aggregate_fixture(), "SUST", 400)

    def test_short_measurement(self):
        with self.assertRaisesRegex(report.Invalid, "too short"):
            report.aggregate(aggregate_fixture(1000), "T", 400)

    def test_stream_denominator(self):
        data = aggregate_fixture()
        data["repeats"][0]["per_worker"]["0"]["fps"] *= 2
        with self.assertRaisesRegex(report.Invalid, "per-stream denominator"):
            report.aggregate(data, "T", 400)

    def test_timestamp_ordering(self):
        data = aggregate_fixture()
        data["repeats"][0]["per_worker"]["0"]["done_after_t0_s"] = -1
        with self.assertRaisesRegex(report.Invalid, "timestamp"):
            report.aggregate(data, "T", 400)

    def test_nonconsecutive_windows(self):
        data = aggregate_fixture()
        data["repeats"][1]["repeat"] = 3
        with self.assertRaisesRegex(report.Invalid, "ordering"):
            report.aggregate(data, "T", 400)

    def test_no_padded_or_missing_frames(self):
        data = aggregate_fixture()
        data["repeats"][0]["per_worker"]["0"]["frames"] -= 8
        with self.assertRaisesRegex(report.Invalid, "frame count"):
            report.aggregate(data, "T", 400)

    def test_thermal_settling(self):
        data = aggregate_fixture()
        data["thermal_warmup"]["last_30s"]["temperature.gpu"]["max"] = 79
        with self.assertRaisesRegex(report.Invalid, "thermally settled"):
            report.aggregate(data, "T", 400)

    def test_wrong_gpu_and_ti(self):
        for name in ("NVIDIA GeForce RTX 4070 SUPER", "NVIDIA GeForce RTX 3090 Ti"):
            with self.assertRaises(report.Invalid):
                report.gpu_identity(name, "8.6")
        self.assertEqual(report.gpu_identity("NVIDIA GeForce RTX 4070 SUPER", "8.9", True)["comparison_scope"], "general_gpu")

    def test_missing_report(self):
        with self.assertRaisesRegex(report.Invalid, "missing report"):
            report.child_result(0, ["/definitely-absent-3090/report.json"])

    def test_failed_child_even_with_existing_report(self):
        with self.assertRaisesRegex(report.Invalid, "child failed"):
            report.child_result(1, [__file__])

    def test_nonfinite_report(self):
        data = aggregate_fixture()
        data["repeats"][0]["aggregate_fps"] = float("nan")
        with self.assertRaisesRegex(report.Invalid, "nonfinite"):
            report.aggregate(data, "T", 400)

    def test_backend_and_fingerprint(self):
        data = {"backends": {"unet_name": "tensorrt_unet_stagewise", "decoder_name": "taesd_trt",
                            "unet_describe": {"engine_dir": "/engines/bs16", "batch": 16, "probe_status": "exact",
                                              "probe_validation": {"kind": "exact", "expected_sha256": "a" * 64, "actual_sha256": "a" * 64}},
                            "decoder_trt_key": "abc", "decoder_trt_plan_sha256": "def"}}
        manifest = {"probe": {"output_sha256": "a" * 64}}
        report.loaded_backend(data, "/engines", "abc", "def", manifest)
        for key, value in (("unet_name", "pytorch"), ("decoder_name", "taesd"), ("decoder_trt_key", "wrong"), ("decoder_trt_plan_sha256", "wrong")):
            bad = copy.deepcopy(data)
            bad["backends"][key] = value
            with self.assertRaises(report.Invalid):
                report.loaded_backend(bad, "/engines", "abc", "def")
        for status in (None, "FAIL", "skipped", "within_tol:0.0", "unknown"):
            bad = copy.deepcopy(data)
            bad["backends"]["unet_describe"]["probe_status"] = status
            with self.assertRaisesRegex(report.Invalid, "UNet probe"):
                report.loaded_backend(bad, "/engines", "abc", "def", manifest)
        for validation in ({}, {"kind": "exact", "expected_sha256": "a" * 64, "actual_sha256": "b" * 64}):
            bad = copy.deepcopy(data)
            bad["backends"]["unet_describe"]["probe_validation"] = validation
            with self.assertRaises(report.Invalid):
                report.loaded_backend(bad, "/engines", "abc", "def", manifest)
        manifest["probe"]["output_sha256"] = "b" * 64
        with self.assertRaisesRegex(report.Invalid, "reference differs"):
            report.loaded_backend(data, "/engines", "abc", "def", manifest)

    def test_portable_probe_requires_frozen_bound(self):
        manifest = {"hardware_compatibility_level": "ampere_plus",
                    "probe": {"output_sha256": "a" * 64, "cross_gpu_rel_l2_max": .01}}
        data = {"backends": {"unet_name": "tensorrt_unet_stagewise", "decoder_name": "taesd_trt",
                            "unet_describe": {"engine_dir": "/engines/bs16", "batch": 16, "probe_status": "cross_gpu:rel_l2=0.004",
                                              "probe_validation": {"kind": "cross_gpu", "rel_l2": .004, "limit": .01,
                                                                   "expected_sha256": "a" * 64, "actual_sha256": "b" * 64}},
                            "decoder_trt_key": "abc", "decoder_trt_plan_sha256": "def"}}
        report.loaded_backend(data, "/engines", "abc", "def", manifest)
        for reference in (None, {**manifest, "hardware_compatibility_level": "none"}):
            with self.assertRaisesRegex(report.Invalid, "nonportable"):
                report.loaded_backend(data, "/engines", "abc", "def", reference)
        for key, value in (("rel_l2", .01001), ("limit", .02), ("rel_l2", float("nan"))):
            bad = copy.deepcopy(data)
            bad["backends"]["unet_describe"]["probe_validation"][key] = value
            with self.assertRaises(report.Invalid):
                report.loaded_backend(bad, "/engines", "abc", "def", manifest)

    def test_baseline_is_explicit_and_cannot_fake_comparison(self):
        self.assertEqual(runner.gpu_roots("/portable", [], True), ["/portable"])
        self.assertEqual(runner.gpu_roots("/native", ["/portable"], False), ["/native", "/portable"])
        for candidate, comparisons, baseline in (("/p", [], False), ("/p", ["/p"], False),
                                                  ("/p", ["/n", "/n"], False), ("/p", ["/n"], True)):
            with self.assertRaises(report.Invalid):
                runner.gpu_roots(candidate, comparisons, baseline)

    def test_nested_harness_environment_is_explicit(self):
        ambient = {"REPRO_GATE_OUT": "/stale", "LIVE15_SKIP_SERVER": "1", "MUSETALK_TRT_FALLBACK": "1",
                   "OMP_NUM_THREADS": "99", "PYTORCH_CUDA_ALLOC_CONF": "bad", "CUDA_VISIBLE_DEVICES": "0",
                   "AWS_PROFILE": "test", "PATH": "/usr/bin"}
        actual, removed = runner.clean_environment(ambient, {"MUSETALK_TRT_FALLBACK": "0", "OMP_NUM_THREADS": "1"})
        self.assertEqual(actual, {"MUSETALK_TRT_FALLBACK": "0", "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "0",
                                  "AWS_PROFILE": "test", "PATH": "/usr/bin"})
        self.assertIn("LIVE15_SKIP_SERVER", removed)
        self.assertIn("REPRO_GATE_OUT", removed)

    def test_live_missing_freshness_is_invalid_not_pass(self):
        with self.assertRaisesRegex(report.Invalid, "missing live evidence"):
            report.live({"streams": [{"window_s": 3600, "server_fresh_fraction": None}]}, 1, 3600)

    def test_failure_not_hidden_by_later_success(self):
        self.assertEqual(report.combined_status([{"status": "FAIL"}, {"status": "PASS"}]), "FAIL")
        self.assertEqual(report.combined_status([{"status": "INVALID"}, {"status": "FAIL"}]), "INVALID")

    def test_missing_input_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.bin"
            manifest = {"schema": "repro_3090_inputs_v1", "files": [{"path": path.name, "sha256": "incorrect"}]}
            with self.assertRaisesRegex(report.Invalid, "missing input"):
                report.verify_files(manifest, tmp)
            path.write_bytes(b"fixture")
            with self.assertRaisesRegex(report.Invalid, "hash mismatch"):
                report.verify_files(manifest, tmp)

    def test_profile_rejects_secrets_and_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.env"
            for body in ("MUSETALK_API_TOKEN=secret", "MUSETALK_TAESD_TRT_DIR=$(command)"):
                path.write_text(body)
                with self.assertRaises(report.Invalid):
                    runner.profile(path)

    def test_profiles_are_literal(self):
        for path in (Path(__file__).parent / "profiles").glob("*.env"):
            self.assertEqual(runner.profile(path)["MUSETALK_TRT_FALLBACK"], "0")

    def test_missing_s3_object_fails_closed(self):
        with mock.patch.object(runner, "capture", side_effect=subprocess.CalledProcessError(254, ["aws"])):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.verify_s3_objects([{"bucket": "fixture", "key": "missing", "bytes": 1}])

    def test_missing_chain_block(self):
        data = {"input_kind": "synthetic_random_buffers", "sets": {"root": {"block_ms": {}, "round_ms": {}, "sum_ms": 0}}, "rounds": 9}
        with self.assertRaisesRegex(report.Invalid, "missing chain blocks"):
            report.blocks(data, {"root": {"spec": [{"name": "down1"}]}})

    def test_missing_raw_block_rounds_cannot_pass(self):
        data = {"input_kind": "synthetic_random_buffers", "sets": {"root": {"block_ms": {"down1": 1}, "round_ms": {}, "sum_ms": 1}}, "rounds": 9}
        with self.assertRaisesRegex(report.Invalid, "raw chain block"):
            report.blocks(data, {"root": {"spec": [{"name": "down1"}]}})

    def test_baseline_cli_executes_one_block_set_and_two_full_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "portable")
            manifest = {"spec": [{"name": "down1"}], "probe": {"output_sha256": "a" * 64}}
            eng = {"root": root, "manifest": manifest, "plan_sha256": {"down1": "sha"}, "manifest_sha256": "manifest-sha"}
            environment = {"engines": [eng], "taesd": {"decoder_plan_sha256": "def"}}
            calls = []
            def fake_child(args, out, env, label, command, gb):
                calls.append(command)
                path = Path(command[command.index("--out") + 1])
                if label == "blocks":
                    value = {"input_kind": "synthetic_random_buffers", "rounds": 9,
                             "sets": {root: {"block_ms": {"down1": 1}, "round_ms": {"down1": [1] * 9}, "sum_ms": 1}}}
                else:
                    value = {"backends": {"unet": "tensorrt_unet_stagewise", "vae_decode": "taesd_trt", "stage_sync": False, "decode_sync": False,
                                          "unet_describe": {"engine_dir": root + "/bs16", "batch": 16, "probe_status": "exact",
                                                            "probe_validation": {"kind": "exact", "expected_sha256": "a" * 64, "actual_sha256": "a" * 64}},
                                          "decoder_trt_key": "abc", "decoder_trt_plan_sha256": "def"},
                             "args": {"batch": 16, "window_s": 10, "live_env": False}, "warmup": {"seconds": 20},
                             "inputs": {"distinct_batches": 1}, "golden": {"finite_checked": True, "per_input_sha256": ["hash"]},
                             "measured": {"wall_s": 180, "batches": 1800, "frames": 28800, "aggregate_fps": 160,
                                          "fps_windows": [{"t_start_s": 10 * i, "fps": 160} for i in range(18)]}, "gpu_smi": {"samples": 180}}
                path.write_text(json.dumps(value))
                return 0
            argv = ["runner.py", "gpu", "--baseline-only", "--target", "portable", "--profile", str(Path(__file__).parent / "profiles/portable.env"),
                    "--engine-root", root, "--taesd-key", "abc", "--taesd-dir", tmp, "--input-manifest", tmp + "/inputs.json",
                    "--out", tmp, "--label", "baseline"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "preflight", return_value=environment), \
                    mock.patch.object(runner, "engine", return_value=eng), mock.patch.object(runner, "child", side_effect=fake_child):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0].count("--root"), 1)
            for command in calls[1:]:
                self.assertEqual(command[command.index("--seconds") + 1], "180")
                self.assertIn("--no-live-env", command)
            saved = json.loads((Path(tmp) / "baseline_gpu/report.json").read_text())
            self.assertEqual(saved["comparison_claim"], "none: single-engine diagnostic baseline")
            self.assertEqual(len(saved["results"]), 3)

    def test_cli_missing_gpu_writes_invalid_and_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["check", "--profile", str(Path(__file__).parent / "profiles/native.env"), "--engine-root", tmp,
                    "--taesd-key", "unbuilt", "--taesd-dir", tmp, "--input-manifest", str(Path(tmp) / "absent.json"),
                    "--out", tmp, "--label", "negative"]
            with mock.patch.object(sys, "argv", ["runner.py", *args]), mock.patch.object(runner, "capture", side_effect=FileNotFoundError("nvidia-smi absent")):
                self.assertEqual(runner.main(), 2)
            result = json.loads((Path(tmp) / "negative_check/report.json").read_text())
            self.assertEqual(result["status"], "INVALID")
            self.assertEqual(result["results"], [])


if __name__ == "__main__":
    unittest.main()
