"""Synthetic cgroup reads/deltas only; no GPU, remote host or performance proof."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chin_multistream.telemetry import cgroup_cpu_interval, cgroup_cpu_snapshot


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


if __name__ == "__main__":
    unittest.main()
