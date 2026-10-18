#!/usr/bin/env python3
"""Heartbeat metadata: deployed revision and per-boot startup stage stamps."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import worker_control_plane as wcp


def plane(env):
    with patch.dict(os.environ, env, clear=False):
        return wcp.LinguaWorkerControlPlane(internal_port=8000, profile="throughput_record",
                                            metrics_provider=lambda: {}, log_fn=lambda m: None)


class StartupMetadataTests(unittest.TestCase):
    def test_revision_and_stamped_stages_reach_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            stamps = Path(folder) / "startup_timings"
            stamps.write_text("bootstrap_start=1792350000.12\ncode_ready=1792350019.5\n"
                              "bad stage=1\nr5_ready=not-a-number\nserver_healthy=1792350400.0\n")
            p = plane({"MUSETALK_SOURCE_REVISION": "A" * 40, "MUSETALK_STARTUP_TIMINGS_FILE": str(stamps)})
            meta = p._payload()["metadata"]
        self.assertEqual(meta["source_revision"], "a" * 40)
        self.assertEqual(meta["startup_timings"], {"bootstrap_start": 1792350000.12, "code_ready": 1792350019.5,
                                                   "server_healthy": 1792350400.0})

    def test_missing_or_invalid_inputs_are_omitted(self):
        p = plane({"MUSETALK_SOURCE_REVISION": "not-a-sha", "MUSETALK_STARTUP_TIMINGS_FILE": "/nonexistent/x"})
        meta = p._payload()["metadata"]
        self.assertIsNone(meta["source_revision"])
        self.assertEqual(meta["startup_timings"], {})

    def test_stage_count_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            stamps = Path(folder) / "startup_timings"
            stamps.write_text("".join("s%d=%d\n" % (i, 1000 + i) for i in range(50)))
            p = plane({"MUSETALK_STARTUP_TIMINGS_FILE": str(stamps)})
            self.assertEqual(len(p._payload()["metadata"]["startup_timings"]), 32)


if __name__ == "__main__":
    unittest.main()
