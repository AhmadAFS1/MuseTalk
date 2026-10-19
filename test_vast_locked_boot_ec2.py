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

    def test_only_allowlisted_experiment_settings_reach_the_worker(self):
        settings = m.parse_settings(["MUSETALK_DEPENDENCY_RANGE_STREAMS=4", "MUSETALK_DEPENDENCY_MAX_CONNECTIONS=8"])
        env = m.compose_env(TEMPLATE_ENV, "a" * 40, {}, SESSION, SECRET, settings)
        self.assertEqual(env["MUSETALK_DEPENDENCY_RANGE_STREAMS"], "4")
        for bad in (["MUSETALK_DEPENDENCY_RANGE_STREAMS=16"], ["VENV_PATH=/x"], ["AWS_ACCESS_KEY_ID=x"], ["noequals"]):
            with self.assertRaises(ValueError):
                m.parse_settings(bad)

    def test_template_env_must_not_carry_registration_or_legacy_keys(self):
        for key in ("LINGUA_WORKER_TOKEN", "VENV_PATH", "SETUP_CLEAN", "MUSETALK_AWS_SECRET_ID"):
            with self.assertRaises(ValueError):
                m.compose_env({**TEMPLATE_ENV, key: "x"}, "a" * 40, {}, SESSION, SECRET)


class StubTests(unittest.TestCase):
    def test_relaxed_filters_are_bounded_and_never_touch_transfer_caps(self):
        fast = {**GOOD, "verification": "unverified", "inet_down": 471}
        self.assertFalse(m.acceptable(fast))
        self.assertTrue(m.acceptable(fast, relax={"allow_unverified": True, "min_inet_down": 450}))
        self.assertFalse(m.acceptable({**fast, "verification": "deverified"}, relax={"allow_unverified": True, "min_inet_down": 450}))
        self.assertFalse(m.acceptable({**fast, "inet_down": 250}, relax={"allow_unverified": True, "min_inet_down": 100}))  # floor 300
        self.assertFalse(m.acceptable({**fast, "inet_down_cost": 0.009}, relax={"allow_unverified": True, "min_inet_down": 450}))
        lean = {**fast, "cpu_ram": 31992}
        self.assertFalse(m.acceptable(lean, relax={"allow_unverified": True, "min_inet_down": 450}))
        self.assertTrue(m.acceptable(lean, relax={"allow_unverified": True, "min_inet_down": 450, "min_cpu_ram_mb": 31000}))
        self.assertFalse(m.acceptable({**lean, "cpu_ram": 16000}, relax={"allow_unverified": True, "min_inet_down": 450,
                                                                         "min_cpu_ram_mb": 8000}))  # floor 24 GB

    def test_stock_launch_matches_the_locked_autoscaler_shape(self):
        b = type("B", (), {"IMAGE": "vastai/pytorch:cuda-12.1-auto", "TEMPLATE": "4dc0868ee56886c667c217d6c3634a58"})
        env = m.compose_env(TEMPLATE_ENV, "a" * 40, {"wheels": {}}, SESSION, SECRET)
        stock = m.create_request(b, "stock", 100, "mtboot-x", env, "stub")
        self.assertNotIn("template_hash_id", stock)  # the autoscaler refuses the legacy template
        self.assertEqual((stock["image"], stock["runtype"]), ("vastai/pytorch:cuda-12.1-auto", "ssh_direc ssh_proxy"))
        template = m.create_request(b, "template", 100, "mtboot-x", env, "stub")
        self.assertEqual(template["template_hash_id"], b.TEMPLATE)
        self.assertNotIn("runtype", template)
        for request in (stock, template):
            self.assertEqual((request["env"], request["onstart"], request["disk"], request["cancel_unavail"]), (env, "stub", 100, True))
        shape = m.request_shape(stock)
        self.assertEqual(shape["ports"], ["-p 3478:3478/udp", "-p 8000:8000"])
        # Every top-level field production must copy, including cancel_unavail (fail fast on a lost offer).
        self.assertEqual(shape["fields"], {"client_id": "me", "image": "vastai/pytorch:cuda-12.1-auto", "disk": 100,
                                           "cancel_unavail": True, "runtype": "ssh_direc ssh_proxy"})
        portal = m.request_shape(m.create_request(b, "stock", 100, "x", {**env, "PORTAL_CONFIG": "localhost:1111:11111:/:Portal",
                                                                          "OPEN_BUTTON_TOKEN": "1"}, "stub"))
        self.assertEqual((portal["plain_env"]["PORTAL_CONFIG"], portal["plain_env"]["OPEN_BUTTON_TOKEN"]),
                         ("localhost:1111:11111:/:Portal", "1"))
        odd = m.request_shape(m.create_request(b, "stock", 100, "x", {**env, "OPEN_BUTTON_TOKEN": "tok-abc"}, "stub"))
        self.assertEqual(odd["plain_env"]["OPEN_BUTTON_TOKEN"], "<redacted>")
        text = json.dumps(shape)
        for secret in (SESSION["SecretAccessKey"], SESSION["SessionToken"], SESSION["AccessKeyId"]):
            self.assertNotIn(secret, text)  # the shape is safe to record and to copy into config
        self.assertIn("AWS_SECRET_ACCESS_KEY", shape["env_keys"])
        self.assertIn("MUSETALK_SKIP_EAGER_UNET", m.TUNABLE_ENV)
        self.assertEqual(m.parse_settings(["MUSETALK_SKIP_EAGER_UNET=0"]), {"MUSETALK_SKIP_EAGER_UNET": "0"})

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


GOOD = {"gpu_name": "RTX 3090", "num_gpus": 1, "cpu_arch": "amd64", "verification": "verified",
        "geolocation": "Utah, US", "cpu_cores_effective": 16, "cpu_ram": 64000, "disk_space": 200,
        "inet_down": 900, "cuda_max_good": 12.4, "reliability": .97, "inet_down_cost": .002,
        "inet_up_cost": .002, "storage_cost": .1, "dph_total": .25, "dph_total_adj": .25}


class OfferTests(unittest.TestCase):
    def test_cheapest_acceptable_offer_and_machine_pin(self):
        rows = [{**GOOD, "id": 3, "machine_id": 30, "dph_total": .25, "dph_total_adj": .25},
                {**GOOD, "id": 1, "machine_id": 10, "dph_total": .20, "dph_total_adj": .20},
                {**GOOD, "id": 2, "machine_id": 20, "dph_total": .10, "dph_total_adj": .10, "verification": "unverified"}]
        self.assertEqual(m.choose_offer(rows)["id"], 1)
        self.assertEqual(m.choose_offer(rows, machine=30)["id"], 3)
        with self.assertRaises(ValueError):
            m.choose_offer(rows, machine=20)

    def test_explicit_caps(self):
        pricey = [{**GOOD, "id": 7, "machine_id": 70, "dph_total": .40, "dph_total_adj": .40}]
        with self.assertRaises(ValueError):
            m.choose_offer(pricey)
        self.assertEqual(m.choose_offer(pricey, max_price=.45)["id"], 7)
        abroad = [{**GOOD, "id": 8, "machine_id": 80, "geolocation": "The Netherlands, NL"}]
        with self.assertRaises(ValueError):
            m.choose_offer(abroad)
        self.assertEqual(m.choose_offer(abroad, any_geo=True)["id"], 8)
        costly = [{**GOOD, "id": 9, "machine_id": 90, "inet_down_cost": .006}]
        with self.assertRaises(ValueError):
            m.choose_offer(costly)
        for bad in ({"cpu_ram": 16000}, {"cpu_cores_effective": 6}, {"inet_down": 300}, {"gpu_name": "RTX 4070S"}):
            self.assertFalse(m.acceptable({**GOOD, **bad}))

    def test_production_id_is_never_an_offer(self):
        rows = [{**GOOD, "id": 51074906, "machine_id": 150024, "dph_total_adj": .01, "dph_total": .01}]
        with self.assertRaises(ValueError):
            m.choose_offer(rows)


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


class ExtendTests(unittest.TestCase):
    def test_extension_is_bounded_and_arms_new_timer_before_stopping_old(self):
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as folder, patch.object(m, "STATE", Path(folder)), \
                patch.object(m.subprocess, "run", side_effect=fake_run):
            run = m.Run("ext-a1")
            run.root.mkdir()
            run.save("ledger.json", {"label": run.label, "state": "created",
                                     "request_started_utc": (now - dt.timedelta(minutes=30)).isoformat(),
                                     "deadline_utc": (now + dt.timedelta(minutes=90)).isoformat()})
            m.extend(run, 60)
            ledger = run.load()
            self.assertEqual(ledger["extensions"][0]["unit"], "mtboot-ext-a1-expiry-e1")
            armed = next(i for i, c in enumerate(calls) if c[0] == "systemd-run")
            stopped = next(i for i, c in enumerate(calls) if c[:2] == ["systemctl", "stop"])
            self.assertLess(armed, stopped)
            self.assertIn("mtboot-ext-a1-expiry.timer", calls[stopped])
            m.extend(run, 60)  # 30 min elapsed + 90 + 60 + 60 = 4 h exactly
            with self.assertRaisesRegex(ValueError, "exceed 4 hours"):
                m.extend(run, 10)
            for bad in (5, 121):
                with self.assertRaises(ValueError):
                    m.extend(run, bad)


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
