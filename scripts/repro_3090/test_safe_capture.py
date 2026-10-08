"""CPU-only subprocess failure contracts, never GPU acceptance evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner
import safe_capture


class SafeCaptureTests(unittest.TestCase):
    def run_python(self, source, **kwargs):
        return safe_capture.capture([sys.executable, "-c", source], cwd=Path(__file__).parent,
                                    timeout_s=kwargs.pop("timeout_s", 3), **kwargs)

    def test_stdout_compatibility_and_stderr_not_mixed(self):
        self.assertEqual(self.run_python("import sys; print('  value  '); print('ignored',file=sys.stderr)"), "value")

    def test_failure_has_known_diagnosis_without_raw_secrets(self):
        secret = "NEVER_SERIALIZE_THIS_CREDENTIAL"
        source = "import sys; print('token=" + secret + "'); print('RuntimeError: CUDA driver initialization failed token=" + secret + "',file=sys.stderr); sys.exit(9)"
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python(source, env={**os.environ, "PRIVATE_TOKEN": secret}, stage="runtime_import")
        error = caught.exception
        encoded = json.dumps(error.record) + str(error) + repr(error.__dict__)
        self.assertNotIn(secret, encoded)
        self.assertNotIn(source, encoded)
        self.assertEqual(error.record["returncode"], 9)
        self.assertEqual(error.record["stage"], "runtime_import")
        self.assertEqual(error.record["diagnostics"], [{"code": "CUDA_DRIVER_INITIALIZATION_FAILED", "message": "CUDA driver initialization failed"}])

    def test_unknown_stderr_is_suppressed(self):
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python("import sys; sys.stderr.write('secret arbitrary error'); sys.exit(2)")
        self.assertEqual(caught.exception.record["diagnostics"], [])
        self.assertNotIn("arbitrary", json.dumps(caught.exception.record))

    def test_multiple_fixed_diagnostics_and_no_dynamic_module_name(self):
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python("import sys; sys.stderr.write('Unable to read CUDA capable devices\\nModuleNotFoundError: No module named secret_module\\nFailed to initialize NVML: Driver/library version mismatch'); sys.exit(1)")
        codes = {row["code"] for row in caught.exception.record["diagnostics"]}
        self.assertEqual(codes, {"CUDA_DEVICES_UNREADABLE", "MISSING_MODULE", "NVML_INITIALIZATION_FAILED", "DRIVER_LIBRARY_VERSION_MISMATCH"})
        self.assertNotIn("secret_module", json.dumps(caught.exception.record))

    def test_timeout_is_bounded_and_requires_check_before_retry(self):
        start = time.monotonic()
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python("import time; time.sleep(30)", timeout_s=.2, terminate_grace_s=.2, reap_grace_s=.2)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(caught.exception.record["failure"], "TIMEOUT")
        self.assertTrue(caught.exception.record["process_state_check_required_before_retry"])
        self.assertEqual(caught.exception.record["descendant_cleanup"], "not_certified")
        self.assertIsNotNone(caught.exception.record["returncode"])

    def test_term_ignoring_process_is_killed(self):
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)",
                            timeout_s=.4, terminate_grace_s=.1, reap_grace_s=.2)
        self.assertTrue(caught.exception.record["kill_requested"])
        self.assertEqual(caught.exception.record["returncode"], -9)

    def test_unconfirmed_reap_stays_bounded_and_never_claims_cleanup(self):
        # Simulate an unresponsive kernel wait observation without creating a
        # real D-state task. Reap the real, ordinary CPU child after the mock.
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            start = time.monotonic()
            with mock.patch.object(subprocess, "Popen", return_value=process), mock.patch.object(process, "poll", return_value=None):
                with self.assertRaises(safe_capture.CaptureFailure) as caught:
                    self.run_python("unused", timeout_s=.1, terminate_grace_s=.1, reap_grace_s=.1)
            self.assertLess(time.monotonic() - start, 2)
            self.assertEqual(caught.exception.record["cleanup"], "unconfirmed_kernel_or_child_state")
            self.assertIsNone(caught.exception.record["returncode"])
            self.assertTrue(caught.exception.record["process_state_check_required_before_retry"])
        finally:
            process.kill()
            process.wait(timeout=2)

    def test_fixed_diagnosis_across_read_boundary_survives_bounded_tail(self):
        # os.read uses 16KiB chunks; split the recognized phrase across that
        # boundary, then put it beyond the retained 64KiB diagnostic tail.
        source = "import sys; sys.stderr.write('x' * 16380 + 'CUDA driver initialization failed' + 'y' * 100000); sys.exit(1)"
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python(source)
        self.assertEqual(caught.exception.record["diagnostics"][0]["code"], "CUDA_DRIVER_INITIALIZATION_FAILED")

    def test_pipe_inheriting_child_cannot_hold_capture_forever(self):
        # The parent exits, but its owned process group still has an open pipe.
        source = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
        start = time.monotonic()
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            self.run_python(source, timeout_s=.3, terminate_grace_s=.2, reap_grace_s=.2)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(caught.exception.record["failure"], "TIMEOUT")

    def test_output_limit_on_either_stream_does_not_deadlock(self):
        for stream in ("stdout", "stderr"):
            with self.subTest(stream=stream), self.assertRaises(safe_capture.CaptureFailure) as caught:
                self.run_python(f"import sys; sys.{stream}.write('sensitive' * 100000)",
                                output_limit_bytes=4096, terminate_grace_s=.1, reap_grace_s=.2)
            self.assertEqual(caught.exception.record["failure"], "OUTPUT_LIMIT")
            self.assertNotIn("sensitive", json.dumps(caught.exception.record))

    def test_missing_executable_never_echoes_path(self):
        with self.assertRaises(safe_capture.CaptureFailure) as caught:
            safe_capture.capture(["/does-not-exist/credential-in-name"], cwd=Path(__file__).parent)
        self.assertEqual(caught.exception.record["failure"], "EXECUTABLE_UNAVAILABLE")
        self.assertEqual(caught.exception.record["executable"], "other")
        self.assertNotIn("credential-in-name", str(caught.exception.record))

    def test_invalid_bounds_and_stage_are_rejected_before_spawn(self):
        for options in ({"timeout_s": 0}, {"timeout_s": float("inf")}, {"output_limit_bytes": 0}, {"stage": "secret_stage"},
                        {"terminate_grace_s": float("nan")}, {"terminate_grace_s": float("inf")},
                        {"reap_grace_s": float("nan")}, {"reap_grace_s": float("inf")}):
            with mock.patch.object(subprocess, "Popen") as spawn, self.assertRaises(ValueError):
                self.run_python("pass", **options)
            spawn.assert_not_called()

    def test_runner_persists_safe_failure_artifact_and_stays_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["runner.py", "check", "--profile", str(Path(__file__).parent / "profiles/native.env"),
                    "--engine-root", tmp, "--taesd-key", "unbuilt", "--taesd-dir", tmp,
                    "--input-manifest", tmp + "/absent.json", "--out", tmp, "--label", "failure"]
            failure = {"stage": "runtime_import", "failure": "NONZERO_EXIT", "diagnostics": [
                {"code": "CUDA_DRIVER_INITIALIZATION_FAILED", "message": "CUDA driver initialization failed"}]}
            with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "preflight", side_effect=safe_capture.CaptureFailure(failure)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 2)
            out = Path(tmp) / "failure_check"
            saved = json.loads((out / "report.json").read_text())
            self.assertEqual(saved["status"], "INVALID")
            self.assertEqual(saved["results"], [])
            self.assertEqual(saved["preflight_failure"], failure)
            self.assertEqual(json.loads((out / "preflight-failure.json").read_text()), failure)

    def test_preflight_nvml_limits_and_stage_are_explicit(self):
        args = mock.Mock(general_gpu=False)
        side_effect = ["NVIDIA GeForce RTX 3090, GPU-example, 8.6, 24576, 550.00, 350, 210, 405, 40", "", RuntimeError("stop before CUDA")]
        with mock.patch.object(runner, "capture", side_effect=side_effect) as capture, self.assertRaisesRegex(RuntimeError, "stop before CUDA"):
            runner.preflight(args, {}, {})
        self.assertEqual(capture.call_args_list[0].kwargs, {"stage": "nvml_identity", "timeout_s": 20})
        self.assertEqual(capture.call_args_list[1].kwargs, {"stage": "nvml_workloads", "timeout_s": 20})
        self.assertEqual(capture.call_args_list[2].kwargs, {"env": {}, "stage": "runtime_import", "timeout_s": 120})


if __name__ == "__main__":
    unittest.main()
