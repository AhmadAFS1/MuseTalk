"""CPU-only staging/negative provenance tests; never launch GPU preparation."""
import hashlib
import datetime as dt
import io
import json
import os
import signal
import subprocess
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import avatar_latent_candidate as candidate


class CandidateTests(unittest.TestCase):
    def test_optional_allocation_binding_is_paired_utc_exact_and_not_expired(self):
        now = dt.datetime(2026, 10, 9, 4, tzinfo=dt.timezone.utc)
        self.assertIsNone(candidate.allocation_deadline(None, None, now))
        with patch.object(candidate.socket, "gethostname", return_value="owned"):
            self.assertEqual(candidate.allocation_deadline("2026-10-09T05:30:00Z", "owned", now).hour, 5)
            for value, host in ((None, "owned"), ("2026-10-09T05:30:00Z", None),
                                ("2026-10-09T05:30:00", "owned"),
                                ("2026-10-09T05:30:00+01:00", "owned"),
                                ("2026-10-09T04:10:00Z", "owned"),
                                ("2026-10-09T05:30:00Z", "foreign")):
                with self.assertRaises(ValueError):
                    candidate.allocation_deadline(value, host, now)

    def test_child_timeout_stops_only_new_owned_guard_group_and_propagates(self):
        process = Mock(pid=424242)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["guard"], 600), 143]
        deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        with patch.object(candidate.subprocess, "Popen", return_value=process) as popen, \
             patch.object(candidate.os, "killpg") as kill, patch("sys.stdout", io.StringIO()):
            with self.assertRaises(subprocess.TimeoutExpired):
                candidate.run_guarded_child(["guard"], self.workspace, {}, io.StringIO(), deadline)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(process.wait.call_args_list[0].kwargs["timeout"], 600)
        kill.assert_called_once_with(424242, signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list[1].kwargs["timeout"], 20)

    def test_child_default_preserves_unbounded_wait_and_success_exit(self):
        process = Mock(pid=424242)
        process.wait.return_value = 0
        with patch.object(candidate.subprocess, "Popen", return_value=process), \
             patch.object(candidate.os, "killpg") as kill, patch("sys.stdout", io.StringIO()):
            self.assertEqual(candidate.run_guarded_child(["guard"], self.workspace, {}, io.StringIO()), 0)
        process.wait.assert_called_once_with(timeout=None)
        kill.assert_not_called()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.repo = self.workspace / "MuseTalk"
        self.repo.mkdir()
        self.source = self.workspace / "accepted"
        self.identity = candidate.IDS[1]
        self.rows = []
        for relative in (*candidate.PINNED_CODE, *("models/" + n for n in candidate.MODEL_FILES)):
            self.row(self.repo / relative, "../../../../" + relative, relative.encode())
        base = self.source / self.identity
        self.bindings = {}
        for name in candidate.FIXTURE_FILES:
            raw = name.encode()
            if name == "preparation.json":
                raw = json.dumps({"seed": 123, "encoder": "Native MuseTalk FP16 SD-VAE", "torch": "2.5.1+cu121",
                                  "numpy": "1.23.5", "opencv": "4.9.0",
                                  "source_sha256": hashlib.sha256(b"source.mp4").hexdigest(),
                                  "audio_sha256": hashlib.sha256(b"speech.wav").hexdigest()}).encode()
            row = self.row(base / name, "../../../../../experiments/avatar_diversity_20260927/" + self.identity + "/" + name, raw)
            self.bindings[self.identity + "/" + name] = {**row, "path": str(base / name)}
        # Padding is deliberately unavailable: only declared prep scope verified.
        while len(self.rows) < 878:
            self.rows.append({"path": "unrelated-missing-" + str(len(self.rows)), "bytes": 1, "sha256": "0" * 64})
        self.manifest = self.workspace / "inputs.json"
        self.manifest.write_text(json.dumps({"schema": "repro_3090_inputs_v1", "files": self.rows}))
        self.pin = patch.object(candidate, "PARENT_SHA", candidate.sha(self.manifest))
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def row(self, path, key, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        row = {"path": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.rows.append(row)
        return row

    def verify(self):
        return candidate.verify_inputs(self.manifest, self.source, self.repo, [self.identity])

    def test_exact_scope_verifies_only_needed_frozen_inputs(self):
        bindings = self.verify()
        self.assertEqual(len(bindings), len(candidate.PINNED_CODE) + len(candidate.MODEL_FILES) + len(candidate.FIXTURE_FILES))
        self.assertFalse(any("unrelated" in p for p in bindings))

    def test_changed_manifest_source_model_or_prep_code_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "original frozen"):
            self.verify()
        self.manifest.write_text(self.manifest.read_text()[:-1])
        paths = [self.source / self.identity / "source.mp4", self.repo / "models" / candidate.MODEL_FILES[0],
                 self.repo / candidate.PINNED_CODE[0]]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "pinned input changed"):
                self.verify()
            path.write_bytes(original)

    def test_duplicate_manifest_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            candidate.rows_for({"files": [self.rows[0], self.rows[0]]})

    def test_fresh_disjoint_paths_only(self):
        out = self.workspace / "new"
        self.assertEqual(candidate.prepare_paths(self.source, out, self.repo, [self.identity])[1], out.resolve())
        for unsafe in (self.source, self.source / "candidate", self.workspace, self.repo / "results/x", self.repo / "models/x"):
            with self.assertRaises(ValueError):
                candidate.prepare_paths(self.source, unsafe, self.repo, [self.identity])
        out.symlink_to(self.workspace / "missing-target")
        with self.assertRaises(ValueError):
            candidate.prepare_paths(self.source, out, self.repo, [self.identity])

    def test_no_arbitrary_or_duplicate_identity(self):
        for identities in (["new_production_avatar"], [self.identity, self.identity], []):
            with self.assertRaises(ValueError):
                candidate.prepare_paths(self.source, self.workspace / "new", self.repo, identities)

    def test_repeat_staging_independent_and_never_overwrites(self):
        out = self.workspace / "experiment"
        left = candidate.stage_repeat(out, "repeat1", self.identity, self.source, self.workspace, self.bindings)
        right = candidate.stage_repeat(out, "repeat2", self.identity, self.source, self.workspace, self.bindings)
        self.assertNotEqual(left.parent / "_audio", right.parent / "_audio")
        for path in (left, right):
            spec = json.loads((path / "spec.json").read_text())
            self.assertEqual(spec["preparation_seed"], 123)
            self.assertEqual(spec["preparation_policy"], candidate.preparation_policy(False))
            self.assertEqual(spec["historical_preparation_versions"]["numpy"], "1.23.5")
            self.assertEqual((path / "source.mp4").read_bytes(), b"source.mp4")
            self.assertFalse((path / "cache.pt").exists())
        with self.assertRaises(FileExistsError):
            candidate.stage_repeat(out, "repeat1", self.identity, self.source, self.workspace, self.bindings)

    def test_environment_has_no_ambient_credentials_or_recipe(self):
        with patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "secret", "PYTHONPATH": "foreign",
                                     "MUSETALK_S3FD_PATH": "foreign", "MUSETALK_VAE_BACKEND": "wrong"}):
            env = candidate.child_environment(self.repo, "GPU-test")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("MUSETALK_VAE_BACKEND", env)
        self.assertEqual(env["MUSETALK_S3FD_PATH"], str(self.repo / "models/face_detection/s3fd.pth"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-test")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env[candidate.POLICY_ENV], "canonical_preparation_unmodified_v1")
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", env)

    def test_gpu_identity_rejects_multiple_wrong_uuid_non3090_and_busy(self):
        uuid = "GPU-" + "a" * 36
        good = "NVIDIA GeForce RTX 3090, " + uuid + ", 8.6, 595.84"
        with patch("safe_capture.capture", side_effect=[good, ""]), patch.object(candidate, "active_pipeline_pids", return_value=[]):
            self.assertEqual(candidate.gpu_identity(self.repo, uuid, True)["uuid"], uuid)
        for raw in (good + "\n" + good, good.replace("3090", "4090"), good.replace(uuid, "GPU-wrong")):
            with patch("safe_capture.capture", return_value=raw), self.assertRaises(ValueError):
                candidate.gpu_identity(self.repo, uuid, True)
        with patch("safe_capture.capture", side_effect=[good, "123"]), self.assertRaisesRegex(ValueError, "GPU busy"):
            candidate.gpu_identity(self.repo, uuid, True)
        with patch("safe_capture.capture", side_effect=[good, ""]), patch.object(candidate, "active_pipeline_pids", return_value=[42]), self.assertRaisesRegex(ValueError, "API/pipeline"):
            candidate.gpu_identity(self.repo, uuid, True)

    def test_cpu_idle_api_detected_without_logging_args(self):
        proc = self.workspace / "proc"
        for pid, argv in ((111, b"python\0/workspace/MuseTalk/api_server.py\0secret\0"),
                          (112, b"python\0/workspace/MuseTalk/scripts/chin_multistream_render.py\0"),
                          (113, b"python\0normal_script.py\0api_server.py-is-not-a-path\0")):
            path = proc / str(pid)
            path.mkdir(parents=True)
            (path / "cmdline").write_bytes(argv)
        self.assertEqual(candidate.active_pipeline_pids(proc), [111, 112])

    def test_unowned_worker_refuses_before_torch_import(self):
        with patch.object(Path, "is_file", return_value=False), self.assertRaisesRegex(ValueError, "box_guard/watch"):
            candidate.worker(self.workspace / "missing_spec.json", "GPU-test")

    def test_canonical_child_has_one_guard_one_watch_and_no_seed_override(self):
        command = candidate.guarded_command(self.workspace / "new", self.identity, "repeat1", "GPU-test")
        self.assertEqual(command.count("scripts/box_guard.sh"), 1)
        self.assertEqual(command.count(str(candidate.HERE / "watch.py")), 1)
        self.assertEqual(command[command.index("--wait-min") + 1], "0")
        self.assertEqual(command[command.index("--min-avail-gb") + 1], "14")
        self.assertIn("--worker-spec", command)
        self.assertNotIn("--seed", command)
        self.assertNotIn("--fixed-cudnn-preparation", command)

    def test_fixed_policy_propagates_to_spec_child_command_and_clean_environment(self):
        target = candidate.stage_repeat(self.workspace / "fixed", "repeat1", self.identity, self.source,
                                        self.workspace, self.bindings, True)
        spec = json.loads((target / "spec.json").read_text())
        self.assertEqual(spec["preparation_policy"], candidate.preparation_policy(True))
        self.assertTrue(spec["preparation_policy"]["non_historical_candidate"])
        self.assertIn("prep_fixed_cudnn_selection_v1", spec["experiment"])
        self.assertEqual(spec["preparation_seed"], 123)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", candidate.POLICY_ENV: "foreign"}):
            env = candidate.child_environment(self.repo, "GPU-test", True)
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", env)
        self.assertEqual(env[candidate.POLICY_ENV], "prep_fixed_cudnn_selection_v1")
        command = candidate.guarded_command(target, self.identity, "repeat1", "GPU-test", True)
        self.assertEqual(command.count("--fixed-cudnn-preparation"), 1)
        self.assertNotIn("--seed", command)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(candidate.validate_policy_binding(spec, True), candidate.preparation_policy(True))
            with self.assertRaisesRegex(ValueError, "policy spec/CLI/environment"):
                candidate.validate_policy_binding(spec, False)
        with patch.dict(os.environ, {candidate.POLICY_ENV: "foreign"}):
            with self.assertRaises(ValueError):
                candidate.validate_policy_binding(spec, True)

    def test_cli_default_and_option_are_forwarded_without_gpu_execution(self):
        uuid = "GPU-a39e62bc-2405-e19d-ba7f-2c51647a46b2"
        for selected in (False, True):
            args = ["--expected-gpu-uuid", uuid, "--worker-spec", str(self.workspace / "spec.json")]
            if selected:
                args.append("--fixed-cudnn-preparation")
            with patch.object(candidate, "worker") as worker:
                self.assertEqual(candidate.main(args), 0)
                self.assertEqual(worker.call_args.args[-2:], (selected, False))
            args = ["--expected-gpu-uuid", uuid, "--inputs", str(self.manifest), "--out", str(self.workspace / "plan")]
            if selected:
                args.append("--fixed-cudnn-preparation")
            output = io.StringIO()
            with patch.object(candidate, "ROOT", self.repo), patch.object(candidate, "verify_inputs", return_value={}), \
                    patch.object(candidate, "code_snapshot", return_value={}), patch("sys.stdout", output):
                self.assertEqual(candidate.main(args), 0)
            receipt = json.loads(output.getvalue())
            self.assertEqual(receipt["preparation_policy"], candidate.preparation_policy(selected))
            self.assertFalse(receipt["gpu_execution"])
            self.assertFalse(receipt["warm_fps_speedup_claim"])

    def test_geometry_policy_binding_staging_command_environment_and_cli(self):
        target = candidate.stage_repeat(self.workspace / "geometry", "repeat1", self.identity, self.source,
                                        self.workspace, self.bindings, False, True)
        spec = json.loads((target / "spec.json").read_text())
        policy = candidate.preparation_policy(False, True)
        self.assertEqual(policy["name"], "prep_fixed_cudnn_selection_v2")
        self.assertEqual(policy["required_geometry_calls"], 240)
        self.assertEqual(spec["preparation_policy"], policy)
        env = candidate.child_environment(self.repo, "GPU-test", False, True)
        self.assertEqual(env[candidate.POLICY_ENV], policy["name"])
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", env)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(candidate.validate_policy_binding(spec, False, True), policy)
            with self.assertRaises(ValueError):
                candidate.validate_policy_binding(spec, True)
        command = candidate.guarded_command(target, self.identity, "repeat1", "GPU-test", False, True)
        self.assertIn("--fixed-cudnn-geometry", command)
        self.assertNotIn("--fixed-cudnn-preparation", command)
        uuid = "GPU-a39e62bc-2405-e19d-ba7f-2c51647a46b2"
        args = ["--expected-gpu-uuid", uuid, "--worker-spec", str(target / "spec.json"), "--fixed-cudnn-geometry"]
        with patch.object(candidate, "worker") as worker:
            candidate.main(args)
            self.assertEqual(worker.call_args.args[-2:], (False, True))
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            candidate.main(args + ["--fixed-cudnn-preparation"])
        with self.assertRaises(ValueError):
            candidate.preparation_policy(True, True)

    def test_wrong_historical_seed_is_not_a_canonical_candidate(self):
        path = self.source / self.identity / "preparation.json"
        payload = json.loads(path.read_text())
        payload["seed"] = 999
        path.write_text(json.dumps(payload))
        for row in self.rows:
            if row["path"].endswith(self.identity + "/preparation.json"):
                row.update(bytes=path.stat().st_size, sha256=candidate.sha(path))
        self.manifest.write_text(json.dumps({"schema": "repro_3090_inputs_v1", "files": self.rows}))
        with patch.object(candidate, "PARENT_SHA", candidate.sha(self.manifest)), self.assertRaisesRegex(ValueError, "seed/encoder"):
            self.verify()


class FixedCudnnAdapterTests(unittest.TestCase):
    def test_geometry_resets_before_every_original_call_preserving_result_and_arguments(self):
        cudnn = types.SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=cudnn))
        result = object()
        calls, rows = [], []

        def inference(*args, **kwargs):
            self.assertFalse(cudnn.benchmark)
            self.assertFalse(rows[-1]["original_returned_successfully"])
            calls.append((args, kwargs))
            return result

        module = types.SimpleNamespace(inference_topdown=inference)
        self.assertIs(candidate.install_fixed_geometry_wrapper(module, torch, rows), inference)
        self.assertTrue(cudnn.benchmark)  # Installation alone changes nothing.
        for index in range(240):
            cudnn.benchmark = True  # S3FD's flag leak between DWPose calls.
            self.assertIs(module.inference_topdown("model", index, option="same"), result)
        self.assertEqual(len(calls), 240)
        self.assertEqual(calls[-1], (("model", 239), {"option": "same"}))
        self.assertEqual([r["call_index"] for r in rows], list(range(240)))
        self.assertTrue(all(r["benchmark_before_reset"] and r["original_returned_successfully"] for r in rows))
        self.assertTrue(all(r["benchmark_before_original_call"] is False
                            and r["benchmark_after_original_return"] is False for r in rows))
        self.assertFalse(cudnn.deterministic)
        self.assertTrue(cudnn.allow_tf32)

    def test_geometry_exception_remains_failure_and_both_adapters_restore(self):
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True)))
        result = object()
        for fails in (False, True):
            post_rows, geometry_rows = [], []
            def inference():
                if fails:
                    raise RuntimeError("original pose failure")
                return result
            def bbox():
                return module.inference_topdown()
            module = types.SimpleNamespace(inference_topdown=inference, get_landmark_and_bbox=bbox)
            def run():
                with candidate.fixed_preparation_adapters(module, torch, post_rows, geometry_rows):
                    self.assertIs(module.get_landmark_and_bbox(), result)
            if fails:
                with self.assertRaisesRegex(RuntimeError, "original pose failure"):
                    run()
                self.assertEqual(post_rows, [])
                self.assertFalse(geometry_rows[0]["original_returned_successfully"])
                self.assertNotIn("benchmark_after_original_return", geometry_rows[0])
            else:
                run()
                self.assertEqual(len(post_rows), 1)
                self.assertTrue(geometry_rows[0]["original_returned_successfully"])
            self.assertIs(module.inference_topdown, inference)
            self.assertIs(module.get_landmark_and_bbox, bbox)

    def test_v1_context_never_touches_geometry_and_restores_on_failure(self):
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True)))
        def bbox():
            raise RuntimeError("original detector failure")
        module = types.SimpleNamespace(get_landmark_and_bbox=bbox)
        with self.assertRaisesRegex(RuntimeError, "original detector failure"):
            with candidate.fixed_preparation_adapters(module, torch, []):
                module.get_landmark_and_bbox()
        self.assertIs(module.get_landmark_and_bbox, bbox)
        self.assertFalse(hasattr(module, "inference_topdown"))
        with self.assertRaises(AttributeError):
            with candidate.fixed_preparation_adapters(module, torch, [], []):
                self.fail("missing geometry adapter must fail")
        self.assertIs(module.get_landmark_and_bbox, bbox)

    def test_v2_receipt_requires_exactly240_successful_ordered_geometry_calls_and_post_reset(self):
        rows = [{"call_index": index, "benchmark_before_reset": True,
                 "benchmark_before_original_call": False, "original_returned_successfully": True,
                 "benchmark_after_original_return": False} for index in range(240)]
        runtime = {"preparation_policy": candidate.preparation_policy(False, True),
                   "geometry_cudnn_receipts": rows,
                   "post_detector_cudnn_receipts": [{"after_original_return": True, "benchmark_before_reset": True,
                                                    "benchmark_after_reset": False}],
                   "cudnn_benchmark_after_preparation": False}
        candidate.validate_policy_receipt(runtime, False, True)
        for bad_rows in (rows[:-1], rows + [rows[-1]],
                         [{**rows[0], "original_returned_successfully": False}] + rows[1:],
                         [{**rows[0], "benchmark_before_original_call": True}] + rows[1:],
                         [{**rows[0], "benchmark_after_original_return": True}] + rows[1:],
                         [{**rows[0], "call_index": 1}] + rows[1:]):
            with self.assertRaisesRegex(ValueError, "240"):
                candidate.validate_policy_receipt({**runtime, "geometry_cudnn_receipts": bad_rows}, False, True)
        with self.assertRaisesRegex(ValueError, "post-detector"):
            candidate.validate_policy_receipt({**runtime, "post_detector_cudnn_receipts": []}, False, True)

    def test_reset_occurs_after_detector_return_and_preserves_exact_result_and_other_flags(self):
        cudnn = types.SimpleNamespace(benchmark=False, deterministic=False, allow_tf32=True)
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=cudnn))
        rows, calls = [], []
        result = (object(), object())

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            self.assertEqual(rows, [])
            cudnn.benchmark = True  # Actual S3FD behavior, not a startup-only flag.
            return result

        module = types.SimpleNamespace(get_landmark_and_bbox=original)
        self.assertIs(candidate.install_fixed_cudnn_wrapper(module, torch, rows), original)
        self.assertFalse(cudnn.benchmark)  # Installing alone makes no flag change.
        self.assertIs(module.get_landmark_and_bbox(["frame.png"], 0, sentinel=5), result)
        self.assertEqual(calls, [((["frame.png"], 0), {"sentinel": 5})])
        self.assertEqual(rows, [{"after_original_return": True, "benchmark_before_reset": True,
                                "benchmark_after_reset": False}])
        self.assertFalse(cudnn.benchmark)
        self.assertFalse(cudnn.deterministic)
        self.assertTrue(cudnn.allow_tf32)

    def test_failed_detector_is_not_silently_reset_or_accepted(self):
        cudnn = types.SimpleNamespace(benchmark=False)
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=cudnn))
        rows = []

        def original():
            cudnn.benchmark = True
            raise RuntimeError("detector failure")

        module = types.SimpleNamespace(get_landmark_and_bbox=original)
        candidate.install_fixed_cudnn_wrapper(module, torch, rows)
        with self.assertRaisesRegex(RuntimeError, "detector failure"):
            module.get_landmark_and_bbox()
        self.assertEqual(rows, [])
        self.assertTrue(cudnn.benchmark)

    def test_receipts_require_actual_post_detector_reset_not_just_requested_policy(self):
        runtime = {"preparation_policy": candidate.preparation_policy(True),
                   "post_detector_cudnn_receipts": [{"after_original_return": True, "benchmark_before_reset": True,
                                                    "benchmark_after_reset": False}],
                   "cudnn_benchmark_after_preparation": False}
        candidate.validate_policy_receipt(runtime, True)
        for changed in ({"post_detector_cudnn_receipts": []}, {"cudnn_benchmark_after_preparation": True},
                        {"post_detector_cudnn_receipts": [{"after_original_return": False, "benchmark_after_reset": False}]},
                        {"preparation_policy": candidate.preparation_policy(False)}):
            with self.assertRaises(ValueError):
                candidate.validate_policy_receipt({**runtime, **changed}, True)
        candidate.validate_policy_receipt({"preparation_policy": candidate.preparation_policy(False)}, False)
        policy = candidate.preparation_policy(True)
        for name in ("early_seed_added", "deterministic_algorithms_changed", "cublas_workspace_config_added",
                     "precision_tf32_attention_changed"):
            self.assertFalse(policy[name])


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy needed for CPU numerical comparison tests")
class ArrayTests(unittest.TestCase):
    def test_exact_values_dtype_shape_and_changes(self):
        a = np.arange(8, dtype=np.float16).reshape(2, 4)
        self.assertTrue(candidate.array_comparison(a, a.copy())["exact"])
        self.assertFalse(candidate.array_comparison(a, a.astype(np.float32))["exact"])
        self.assertFalse(candidate.array_comparison(a, a.reshape(4, 2))["exact"])
        b = a.copy()
        b[0, 0] = .5
        result = candidate.array_comparison(a, b)
        self.assertFalse(result["exact"])
        self.assertEqual(result["max_abs"], .5)
        self.assertEqual(result["changed_values"], 1)
        self.assertEqual(result["mean_abs"], .0625)

    def test_nan_inf_empty_and_object_refused(self):
        for bad in (np.array([float("nan")]), np.array([float("inf")]), np.array([]), np.array(["a"])):
            with self.assertRaises(ValueError):
                candidate.array_comparison(bad, bad)

    def test_cache_comparison_keeps_latent_audio_geometry_and_masks_separate(self):
        class Tensor:
            def __init__(self, array):
                self.value = array
                self.shape = array.shape
                self.dtype = "torch.float16"
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "left", Path(tmp) / "right"
            for path in (left, right):
                path.mkdir()
                np.savez(path / "masks.npz", **{str(i): np.zeros((2, 2), np.uint8) for i in range(240)})
            a = {"latents": Tensor(np.zeros((240, 8, 32, 32), np.float16)), "audio": Tensor(np.zeros((240, 2), np.float16)),
                 "boxes": np.zeros((240, 4), np.int32), "cropboxes": [[0, 0, 2, 2]] * 240}
            b = {**a, "latents": Tensor(a["latents"].value.copy()), "audio": Tensor(a["audio"].value.copy())}
            fake = types.SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor),
                                         load=lambda path, **kwargs: a if path.parent == left else b)
            with patch.dict("sys.modules", {"torch": fake}):
                result = candidate.compare_caches(left, right)
                self.assertTrue(result["all_exact"])
                b["latents"].value[0, 0, 0, 0] = 1
                result = candidate.compare_caches(left, right)
                self.assertFalse(result["all_exact"])
                self.assertTrue(result["nonlatent_components_exact"])
                b["audio"].value[0, 0] = 1
                self.assertFalse(candidate.compare_caches(left, right)["nonlatent_components_exact"])


if __name__ == "__main__":
    unittest.main()
