import os
import unittest
from unittest.mock import patch

from scripts.worker_control_plane import LinguaWorkerControlPlane


class ControlPlaneIsolationTest(unittest.TestCase):
    configured = {
        "LINGUA_CONTROL_PLANE_BASE_URL": "https://control.example",
        "LINGUA_WORKER_TOKEN": "test-only-token",
        "LINGUA_WORKER_BASE_URL": "http://worker.example:8000",
        "LINGUA_WORKER_GPU_TYPE": "test_gpu",
    }

    def make_worker(self, extra):
        with patch.dict(os.environ, {**self.configured, **extra}, clear=True):
            return LinguaWorkerControlPlane(internal_port=8000, profile="test", metrics_provider=lambda: {}, log_fn=lambda _: None)

    def test_explicit_isolation_overrides_injected_credentials(self):
        for value in ("0", "false", "off", "no"):
            worker = self.make_worker({"LINGUA_CONTROL_PLANE_ENABLED": value})
            with patch.object(worker, "_post_json", side_effect=AssertionError("network forbidden")), patch("threading.Thread", side_effect=AssertionError("no callbacks")):
                worker.start()
                worker.mark_local_ready()
                worker.begin_draining("test")
                worker.stop()
            self.assertFalse(worker.control_plane_requested)
            self.assertFalse(worker.control_plane_configured)
            self.assertFalse(worker.is_registered())

    def test_isolated_health_does_not_claim_registration(self):
        worker = self.make_worker({"LINGUA_CONTROL_PLANE_ENABLED": "0"})
        worker.mark_local_ready()
        self.assertTrue(worker.ready_for_health())
        self.assertFalse(worker.is_registered())
        self.assertEqual(worker.current_status(), "healthy")

    def test_default_behavior_preserved(self):
        worker = self.make_worker({})
        self.assertTrue(worker.control_plane_requested)
        self.assertTrue(worker.control_plane_configured)
        worker.mark_local_ready()
        self.assertFalse(worker.ready_for_health())
        self.assertEqual(worker.current_status(), "registering")

    def test_invalid_flag_fails_closed(self):
        with self.assertRaises(ValueError):
            self.make_worker({"LINGUA_CONTROL_PLANE_ENABLED": "maybe"})


if __name__ == "__main__":
    unittest.main()
