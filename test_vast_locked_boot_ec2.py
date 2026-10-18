#!/usr/bin/env python3
"""Mock tests for scripts/vast_locked_boot_ec2.py (no cloud, SSH or network)."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import vast_locked_boot_ec2 as m


SESSION = {"AccessKeyId": "ASIAFIXTUREFIXTURE00", "SecretAccessKey": "fixture-secret", "SessionToken": "fixture-token"}
SECRET = {"AVATAR_S3_BUCKET": "bucket-fixture", "AVATAR_S3_REGION": "us-east-1", "AVATAR_S3_PREFIX": "avatars",
          "AVATAR_S3_ENABLED": "1"}
TEMPLATE_ENV = {"-p 8000:8000": "1", "-p 3478:3478/udp": "1", "NVIDIA_VISIBLE_DEVICES": "all"}


class RunTests(unittest.TestCase):
    def test_label_is_outside_every_autoscaler_prefix(self):
        run = m.Run("cold-a1")
        self.assertEqual(run.label, "mtboot-cold-a1")
        self.assertFalse(run.label.startswith(m.AUTOSCALER_PREFIXES))
        for bad in ("", "x", "Has-Caps", "../escape", "a" * 41):
            with self.assertRaises(ValueError):
                m.Run(bad)


class EnvTests(unittest.TestCase):
    def test_launch_env_matches_locked_bootstrap_contract(self):
        tickets = {"manifest": {"url": "https://example.invalid/m"}, "wheels": {}}
        env = m.compose_env(TEMPLATE_ENV, "a" * 40, tickets, SESSION, SECRET)
        self.assertEqual(env["MUSETALK_SOURCE_REVISION"], "a" * 40)
        self.assertEqual(json.loads(env["MUSETALK_DEPENDENCY_TICKETS_JSON"]), tickets)
        self.assertEqual(env["MUSETALK_RUNTIME_CONFIG_SOURCE"], "injected")
        self.assertEqual(env["TRT_ARTIFACT_S3_BUCKET"], "bucket-fixture")
        self.assertEqual(env["AWS_SESSION_TOKEN"], "fixture-token")
        self.assertIn("-p 8000:8000", env)
        for absent in ("MUSETALK_AWS_SECRET_ID", "VENV_PATH", "SETUP_CLEAN", "LINGUA_WORKER_TOKEN"):
            self.assertNotIn(absent, env)

    def test_template_env_must_not_carry_registration_or_legacy_keys(self):
        for key in ("LINGUA_WORKER_TOKEN", "VENV_PATH", "SETUP_CLEAN", "MUSETALK_AWS_SECRET_ID"):
            with self.assertRaises(ValueError):
                m.compose_env({**TEMPLATE_ENV, key: "x"}, "a" * 40, {}, SESSION, SECRET)


class StubTests(unittest.TestCase):
    def test_onstart_stub_is_small_pinned_and_valid_bash(self):
        import subprocess
        stub = m.onstart_stub("b" * 40, "c" * 64)
        self.assertLess(len(stub.encode()), 1500)
        self.assertIn("c" * 64 + "  $f.tmp", stub)
        self.assertIn(m.REPO_RAW + "b" * 40 + "/" + m.BOOTSTRAP_PATH, stub)
        self.assertIn('exec bash "$f"', stub)
        subprocess.run(["bash", "-n"], input=stub.encode(), check=True)
        for bad in (("short", "c" * 64), ("b" * 40, "short")):
            with self.assertRaises(ValueError):
                m.onstart_stub(*bad)


class OfferTests(unittest.TestCase):
    def setUp(self):
        self.b = types.SimpleNamespace(acceptable=lambda row, allow_unverified=False: row.get("ok", True))

    def test_cheapest_acceptable_offer_and_machine_pin(self):
        rows = [{"id": 3, "machine_id": 30, "dph_total_adj": .25}, {"id": 1, "machine_id": 10, "dph_total_adj": .20},
                {"id": 2, "machine_id": 20, "dph_total_adj": .10, "ok": False}]
        self.assertEqual(m.choose_offer(rows, self.b)["id"], 1)
        self.assertEqual(m.choose_offer(rows, self.b, machine=30)["id"], 3)
        with self.assertRaises(ValueError):
            m.choose_offer(rows, self.b, machine=20)

    def test_production_id_is_never_an_offer(self):
        rows = [{"id": 51074906, "machine_id": 150024, "dph_total_adj": .01}]
        with self.assertRaises(ValueError):
            m.choose_offer(rows, self.b)


class OwnershipTests(unittest.TestCase):
    """Production RTX 4070 SUPER 51074906 must be refused on every path."""

    def run_with(self, row, ledger_extra=None):
        run = m.Run("own-a1")
        ledger = {"label": run.label, "protected_instance_ids": ["51074906"], "instance_id": None, "machine_id": 1,
                  **(ledger_extra or {})}
        b = types.SimpleNamespace(owned=lambda client, ledger: row, PROTECTED=None)
        return m.owned(run, b, None, ledger)

    def test_production_instance_refused_even_if_matched(self):
        for row in ({"id": 51074906, "label": "mtboot-own-a1", "gpu_name": "RTX 3090"},
                    {"id": 51074906, "label": None, "gpu_name": "RTX 4070S"}):
            with self.assertRaisesRegex(ValueError, "non-owned or production"):
                self.run_with(row)

    def test_other_label_or_gpu_refused_and_owned_3090_accepted(self):
        with self.assertRaises(ValueError):
            self.run_with({"id": 777, "label": "lingua-musetalk-1", "gpu_name": "RTX 3090"})
        with self.assertRaises(ValueError):
            self.run_with({"id": 777, "label": "mtboot-own-a1", "gpu_name": "RTX 4070S"})
        with self.assertRaises(ValueError):
            self.run_with({"id": 778, "label": "mtboot-own-a1", "gpu_name": "RTX 3090"}, {"protected_instance_ids": ["778"]})
        row = {"id": 779, "label": "mtboot-own-a1", "gpu_name": "RTX 3090"}
        self.assertEqual(self.run_with(row), row)

    def test_cleanup_checks_ownership_before_destroy(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(m, "STATE", Path(folder)):
            run = m.Run("own-a2")
            run.root.mkdir()
            run.save("ledger.json", {"label": run.label, "state": "created", "instance_id": "51074906",
                                     "protected_instance_ids": ["51074906"], "machine_id": 1,
                                     "deadline_utc": "2000-01-01T00:00:00+00:00"})
            calls = []
            lib = types.SimpleNamespace(bootstrap=lambda: object(), PROTECTED=None,
                                        owned=lambda client, ledger: {"id": 51074906, "label": run.label, "gpu_name": "RTX 3090"},
                                        cleanup=lambda client, ledger: calls.append("destroy"))
            with patch.object(m, "library", return_value=lib):
                with self.assertRaises(ValueError):
                    m.cleanup(run)
            self.assertEqual(calls, [])

    def test_wipe_paths_are_exact_worker_workspace_paths(self):
        for p in m.WIPE_PATHS:
            self.assertTrue((p.startswith("/workspace/") or p == "/root/.cache/huggingface")
                            and ".." not in p and p.count("/") >= 2, p)
        self.assertNotIn("/workspace", m.WIPE_PATHS)
        self.assertNotIn("/root", m.WIPE_PATHS)


BOOT = """[bootstrap] [2026-10-18T10:00:00Z] start boot_id=old
[bootstrap] [2026-10-18T10:05:00Z] Code ready (300s); revision=aaa
[bootstrap] [2026-10-18T19:00:00Z] start boot_id=new
[bootstrap] [2026-10-18T19:00:07Z] Code ready (7s); revision=bbb
[bootstrap] [2026-10-18T19:09:30Z] Complete (563s after checkout phase began)
"""
ONSTART = """VAST_ONSTART BEGIN: 2026-10-18 10:05:01 UTC
[vast_onstart.sh] [10:20:00] Bootstrap/setup phase finished in 15m00s
VAST_ONSTART COMPLETE: 2026-10-18 10:30:00 UTC
VAST_ONSTART BEGIN: 2026-10-18 19:00:08 UTC
[install_musetalk.sh][PHASE][2026-10-18T19:03:10+0000] DONE: Phase 3: Packages (3m00s)
[install_musetalk.sh][PHASE][2026-10-18T19:05:00+0000] DONE: Phase 4: Weights (1m50s)
[vast_onstart.sh] [19:06:00] Bootstrap/setup phase finished in 5m52s
[vast_onstart.sh] [19:07:10] r5 engine bundle restore phase finished in 1m10s
VAST_ONSTART COMPLETE: 2026-10-18 19:09:29 UTC
TOTAL ELAPSED: 9m21s
"""


class TimelineTests(unittest.TestCase):
    def test_last_boot_only_with_absolute_stage_times(self):
        stages = m.timeline(BOOT, ONSTART)
        names = [s["stage"] for s in stages]
        self.assertEqual(names[0], "bootstrap_start")
        self.assertEqual(stages[0]["at"], "2026-10-18T19:00:00+00:00")
        self.assertNotIn("10:", "".join(s["at"][11:14] for s in stages))
        by = {s["stage"]: s["since_bootstrap_s"] for s in stages}
        self.assertEqual(by["code_ready"], 7)
        self.assertEqual(by["install:Phase 4: Weights (1m50s)"], 300)
        self.assertEqual(by["onstart:r5 engine bundle restore"], 430)
        self.assertEqual(by["onstart_complete"], 569)
        self.assertEqual(by["bootstrap_complete"], 570)

    def test_midnight_rollover_for_clock_only_lines(self):
        boot = "[bootstrap] [2026-10-18T23:59:00Z] start boot_id=x\n"
        onstart = ("VAST_ONSTART BEGIN: 2026-10-18 23:59:01 UTC\n"
                   "[install_musetalk.sh][PHASE][2026-10-18T23:59:50+0000] DONE: Phase 3: Packages (49s)\n"
                   "[vast_onstart.sh] [00:01:00] Bootstrap/setup phase finished in 2m00s\n")
        stages = {s["stage"]: s for s in m.timeline(boot, onstart)}
        self.assertEqual(stages["onstart:Bootstrap/setup"]["at"], "2026-10-19T00:01:00+00:00")

    def test_missing_start_marker_is_an_error(self):
        with self.assertRaises(ValueError):
            m.timeline("no markers\n", "")


if __name__ == "__main__":
    unittest.main()
