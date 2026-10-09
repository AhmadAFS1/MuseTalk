"""Synthetic cgroup reads/deltas only; no GPU, remote host or performance proof."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chin_multistream.telemetry import cgroup_cpu_interval, cgroup_cpu_snapshot
import chin_multistream_cpu_telemetry as wrapper


class CPUWindowTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.membership = self.root / "membership"
        self.mountinfo = self.root / "mountinfo"
        self.mount = self.root / "mounted"
        self.mount.mkdir()
        self.membership.write_text("0::/owned/leaf\n")
        self.mountinfo.write_text(f"1 2 0:3 /owned {self.mount} rw - cgroup2 cgroup rw\n")
        self.leaf = self.mount / "leaf"
        self.leaf.mkdir()
        self.stats = "usage_usec 1000\nuser_usec 800\nsystem_usec 200\nnr_periods 10\nnr_throttled 2\nthrottled_usec 300\n"
        (self.leaf / "cpu.stat").write_text(self.stats)
        # Distinct host-root values: must not read this instead of owned/leaf.
        (self.mount / "cpu.stat").write_text(self.stats.replace("1000", "9999"))

    def snapshot(self):
        return cgroup_cpu_snapshot(self.membership, self.mountinfo)

    def test_current_descendant_not_host_root(self):
        data = self.snapshot()
        self.assertEqual(data["status"], "AVAILABLE")
        self.assertEqual(data["counters"]["usage_usec"], 1000)
        self.assertEqual(data["source"], str(self.leaf / "cpu.stat"))

    def test_namespace_root_uses_mountpoint(self):
        self.membership.write_text("0::/\n")
        self.assertEqual(self.snapshot()["counters"]["usage_usec"], 9999)

    def test_missing_stats_explicitly_unavailable(self):
        (self.leaf / "cpu.stat").unlink()
        self.assertEqual(self.snapshot()["status"], "UNAVAILABLE")

    def test_unsupported_ambiguous_or_unresolved_membership(self):
        for text in ("2:cpu:/owned/leaf\n", "0::/owned/leaf\n0::/owned/other\n",
                     "0::/outside/leaf\n", "0::/owned/../leaf\n"):
            with self.subTest(text=text):
                self.membership.write_text(text)
                self.assertEqual(self.snapshot()["status"], "UNAVAILABLE")

    def test_ambiguous_or_malformed_mounts(self):
        good = self.mountinfo.read_text()
        for text in (good + good, "bad\n", good.replace("cgroup2", "cgroup")):
            with self.subTest(text=text):
                self.mountinfo.write_text(text)
                self.assertEqual(self.snapshot()["status"], "UNAVAILABLE")

    def test_unknown_counters_ignored_but_bad_required_counters_rejected(self):
        for text in (self.stats + "future_counter 42\n", self.stats.replace("1000", "-1"),
                     self.stats.replace("1000", "x"), self.stats + "nr_periods 10\n",
                     self.stats.replace("nr_periods 10\n", "")):
            with self.subTest(text=text):
                (self.leaf / "cpu.stat").write_text(text)
                self.assertEqual(self.snapshot()["status"], "AVAILABLE" if "future_counter" in text else "UNAVAILABLE")

    def pair(self):
        before = self.snapshot()
        after = {**before, "sample_monotonic_s": before["sample_monotonic_s"] + 2,
                 "counters": {k: v + (4 if k == "nr_periods" else 1) for k, v in before["counters"].items()}}
        return before, after

    def test_delta_and_fraction_are_not_fps_denominator(self):
        before, after = self.pair()
        data = cgroup_cpu_interval(before, after)
        self.assertEqual(data["status"], "AVAILABLE")
        self.assertEqual(data["sample_elapsed_s"], 2)
        self.assertEqual(data["throttled_period_fraction"], .25)
        self.assertEqual(data["throttled_time_s"], .000001)
        self.assertNotIn("fps", data)

    def test_missing_snapshot_is_not_zero_throttling(self):
        before, after = self.pair()
        data = cgroup_cpu_interval({"status": "UNAVAILABLE"}, after)
        self.assertEqual(data["status"], "UNAVAILABLE")
        self.assertNotIn("counters_delta", data)

    def test_counter_reset_migration_and_time_reversal_invalid(self):
        before, after = self.pair()
        changes = [{"source": "different"}, {"counters": {}},
                   {"counters": {**after["counters"], "usage_usec": 0}},
                   {"sample_monotonic_s": before["sample_monotonic_s"]},
                   {"counters": {**after["counters"], "nr_throttled": 100}}]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(cgroup_cpu_interval(before, {**after, **change})["status"], "INVALID")

    def test_no_period_delta_does_not_invent_ratio(self):
        before, after = self.pair()
        after["counters"] = dict(before["counters"])
        data = cgroup_cpu_interval(before, after)
        self.assertEqual(data["status"], "AVAILABLE")
        self.assertIsNone(data["throttled_period_fraction"])


class WrapperTests(unittest.TestCase):
    def test_import_inert_and_no_gpu_packages(self):
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import chin_multistream_cpu_telemetry;'
                'assert not any(n in sys.modules for n in ("torch","numpy","tensorrt","cv2"))')
        result = subprocess.run([sys.executable, "-B", "-c", code, str(Path(wrapper.__file__).parent)],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bad_renderer_bytes_fail_before_hooks_or_sampling(self):
        with mock.patch.object(wrapper, "RENDERER_SHA256", "0" * 64), \
             mock.patch.object(wrapper.telemetry, "cgroup_cpu_snapshot") as sample, \
             mock.patch.object(wrapper, "_original_run_multi") as run:
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                wrapper.run_multi(None, None)
        sample.assert_not_called()
        run.assert_not_called()

    def invoke(self, run, snapshots=None):
        with mock.patch.object(wrapper, "_original_run_multi", run), \
             mock.patch.object(wrapper.gpu, "run_repeat", return_value=({"unchanged": True}, "drain")), \
             mock.patch.object(wrapper.renderer.Collector, "wait_for", return_value={}), \
             mock.patch.object(wrapper.telemetry, "cgroup_cpu_snapshot", side_effect=snapshots or [{"status": "UNAVAILABLE"}] * 4):
            issue = wrapper.gpu.run_repeat
            collect = wrapper.renderer.Collector.wait_for
            try:
                return wrapper.run_multi(None, None)
            finally:
                self.assertIs(wrapper.gpu.run_repeat, issue)
                self.assertIs(wrapper.renderer.Collector.wait_for, collect)
                self.assertFalse(wrapper._lock.locked())

    def test_two_windows_keep_frames_fps_and_gpu_result(self):
        rows = [{"frames": 240, "wall_s": 2, "aggregate_fps": 120},
                {"frames": 240, "wall_s": 3, "aggregate_fps": 80}]
        def run(args, directory):
            for row in rows:
                self.assertEqual(wrapper.gpu.run_repeat(), ({"unchanged": True}, "drain"))
                wrapper.renderer.Collector.wait_for(object(), "repdone", [0])
            return {"repeats": rows}
        result = self.invoke(run)
        self.assertEqual([r["aggregate_fps"] for r in result["repeats"]], [120, 80])
        self.assertEqual([r["cgroup_cpu"]["status"] for r in result["repeats"]], ["UNAVAILABLE"] * 2)
        self.assertFalse(result["cpu_window_telemetry"]["renderer_bytes_changed"])

    def test_original_failure_restores_hooks(self):
        def run(args, directory):
            wrapper.gpu.run_repeat()
            raise RuntimeError("synthetic original failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic original failure"):
            self.invoke(run)

    def test_missing_extra_or_unmatched_window_rejected(self):
        def pending(args, directory):
            wrapper.gpu.run_repeat()
            return {"repeats": [{}]}
        def missing(args, directory):
            return {"repeats": [{}]}
        def unmatched(args, directory):
            wrapper.renderer.Collector.wait_for(object(), "repdone", [0])
        for run in (pending, missing, unmatched):
            with self.subTest(run=run), self.assertRaises(RuntimeError):
                self.invoke(run)

    def test_uncollected_previous_window_rejected(self):
        def run(args, directory):
            wrapper.gpu.run_repeat()
            wrapper.gpu.run_repeat()
        with self.assertRaises(RuntimeError):
            self.invoke(run)

    def test_main_serial_rejected_and_original_main_restored(self):
        original = wrapper.renderer.run_multi
        with self.assertRaises(ValueError):
            wrapper.main(["--label", "synthetic", "--mode", "serial"])
        with mock.patch.object(wrapper.renderer, "main", side_effect=RuntimeError("synthetic")):
            with self.assertRaises(RuntimeError):
                wrapper.main(["--label", "synthetic"])
        self.assertIs(wrapper.renderer.run_multi, original)


if __name__ == "__main__":
    unittest.main()
