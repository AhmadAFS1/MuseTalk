"""CPU-only startup observer tests: no cloud credentials, requests, or GPU."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("startup_observer", ROOT / "scripts/repro_3090/50_startup.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)
PREP_SPEC = importlib.util.spec_from_file_location("startup_prepare", ROOT / "scripts/repro_3090/prepare_ec2_development.py")
prepare = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(prepare)
RUNNER_SPEC = importlib.util.spec_from_file_location("ec2_runner", ROOT / "scripts/repro_3090/ec2_startup_runner.py")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.ledger = observer.initialize(self.run, "test-startup-unique", "http://127.0.0.1:8000")
        self.request = {"count": 1, "offer_id": 123, "create_request": {"label": "test-startup-unique", "disk": 150}}
        self.budget = {"approval_reference": "test fixture; not authority for actual spending",
                       "lifetime_enforcer_reference": "unit-test external enforcer",
                       "total_usd": 30, "max_hourly_usd": .30, "max_lifetime_hours": 24,
                       "planned_lifetime_hours": 4, "max_inet_down_usd_per_gb": 1.50 / 1024,
                       "max_inet_up_usd_per_gb": 1.50 / 1024, "reserved_inet_down_gb": 100,
                       "reserved_inet_up_gb": 100, "additional_reserve_usd": 1}
        self.offer = {"observed_at_utc": observer.utc_now(), "allocated_storage_gb": 150,
                      "offer": {"id": 123, "gpu_name": "RTX 3090", "num_gpus": 1,
                                "verification": "verified", "dph_total": .25, "disk_space": 200,
                                "inet_down_cost": .0013, "inet_up_cost": .0013, "storage_cost": .20}}
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def fake_create(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"success": True, "result": {"actions": [{"action": "create", "label": "test-startup-unique", "instance_id": "456"}]}}

    def create(self, api_fn=None):
        return observer.create_once(self.run, self.request, self.budget, self.offer,
                                    self.root / "budget-ledger.json", "test-token", api_fn or self.fake_create)

    def test_create_exact_contract_and_no_secret_persistence(self):
        self.request["create_request"]["env"] = {"AWS_SECRET_ACCESS_KEY": "SECRET-NEVER-PERSIST"}
        result = self.create()
        self.assertEqual(result["instance_id"], "456")
        self.assertEqual(self.calls[0][0][1], "/api/runtime/workers/musetalk/create")
        self.assertEqual(self.calls[0][1]["method"], "POST")
        self.assertNotIn("SECRET-NEVER-PERSIST", (self.run / "startup-ledger.json").read_text())
        with self.assertRaises(observer.Invalid):
            self.create()
        self.assertEqual(len(self.calls), 1)

    def test_timeout_never_reposts_and_keeps_budget_reserved(self):
        def timeout(*args, **kwargs):
            self.calls.append(1)
            raise TimeoutError("arbitrary secret error")
        with self.assertRaisesRegex(observer.Invalid, "ambiguous"):
            self.create(timeout)
        with self.assertRaises(observer.Invalid):
            self.create()
        self.assertEqual(len(self.calls), 1)
        ledger = observer.load_json(self.run / "startup-ledger.json")
        self.assertEqual(ledger["state"], "ambiguous")
        self.assertNotIn("arbitrary secret", json.dumps(ledger))
        self.assertTrue(observer.load_json(self.root / "budget-ledger.json")["reservations"])

    def test_shared_budget_prevents_unfunded_create(self):
        observer.atomic_json(self.root / "budget-ledger.json", {"total_usd": 30, "reservations": {"another": {"reserved_usd": 29.99}}})
        with self.assertRaisesRegex(observer.Invalid, "Shared experiment budget"):
            self.create()
        self.assertFalse(self.calls)

    def test_budget_wrong_gpu_cost_nonfinite_missing_authority_stale_offer(self):
        cases = [("offer", "gpu_name", "RTX 3090 Ti"), ("offer", "dph_total", .31),
                 ("offer", "inet_up_cost", .0015), ("offer", "dph_total", float("nan")),
                 ("budget", "approval_reference", ""), ("budget", "lifetime_enforcer_reference", ""),
                 ("budget", "planned_lifetime_hours", 25)]
        for source, key, value in cases:
            budget, offer = copy.deepcopy(self.budget), copy.deepcopy(self.offer)
            (offer["offer"] if source == "offer" else budget)[key] = value
            with self.subTest(source=source, key=key), self.assertRaises(observer.Invalid):
                observer.validate_budget(budget, offer, self.request)
        self.offer["observed_at_utc"] = "2020-01-01T00:00:00Z"
        with self.assertRaises(observer.Invalid):
            observer.validate_budget(self.budget, self.offer, self.request)

    def test_reconcile_requires_unique_label_and_retains_unknown_acceptance(self):
        inventory = {"observed_at_utc": observer.utc_now(), "instances": []}
        with self.assertRaises(observer.Invalid):
            observer.reconcile(self.ledger, inventory)
        inventory["instances"] = [{"id": 456, "label": self.ledger["label"]}] * 2
        with self.assertRaises(observer.Invalid):
            observer.reconcile(self.ledger, inventory)
        inventory["instances"] = inventory["instances"][:1]
        result = observer.reconcile(self.ledger, inventory)
        self.assertEqual(result["instance_id"], "456")
        self.assertEqual(result["events"][0]["source"], "provider_reconciliation_upper_bound")
        self.assertNotIn("monotonic_ns", result["events"][0])

    def test_health_or_registration_is_never_usable_call(self):
        ledger = self.create()
        observer.add_event(ledger, "health_ready", source="test")
        observer.observe_once(ledger, "test-token", 5, lambda *a, **k: {
            "success": True, "workers": [{"worker_id": "musetalk-456", "instance_id": 456, "status": "healthy"}],
            "pool_status": {"servers": [{"worker_id": "musetalk-456", "status": "healthy", "assignable": True}]}})
        self.assertEqual(observer.report(ledger)["verdict"], "INCOMPLETE")
        self.assertIn("first_usable_frame", observer.report(ledger)["missing_readiness_evidence"])

    def test_foreign_monotonic_never_subtracted(self):
        ledger = self.create()
        start = ledger["events"][0]
        later = observer.utc_parse(start["utc"]) + dt.timedelta(seconds=12)
        ledger["events"].append({"name": "models_loaded", "utc": later.isoformat(), "clock_domain": "another-host",
                                 "monotonic_ns": 1, "uncertainty_seconds": None})
        delta = observer.report(ledger)["durations"]["models_loaded"]
        self.assertEqual(delta["seconds_from_request"], 12)
        self.assertEqual(delta["method"], "UTC_correlation")

    def test_timestamp_order_invalid(self):
        ledger = self.create()
        ledger["events"].append({"name": "first_usable_frame", "utc": "2020-01-01T00:00:00Z"})
        self.assertEqual(observer.report(ledger)["verdict"], "INVALID")

    def test_timeline_marks_unavailable_stages_and_conflicting_time_fails(self):
        ledger = self.create()
        observer.write_report(self.run, ledger, observer.report(ledger))
        text = (self.run / "startup-timeline.md").read_text()
        self.assertIn("image_pull_started | unavailable", text)
        self.assertIn("INCOMPLETE", text)
        observer.add_event(ledger, "models_loaded", source="worker", utc="2026-10-08T08:00:00Z", local=False)
        with self.assertRaises(observer.Invalid):
            observer.add_event(ledger, "models_loaded", source="worker", utc="2026-10-08T08:01:00Z", local=False)

    def media(self):
        raw = self.root / "client-trace.json"
        raw.write_text('{"test_fixture":true}\n')
        start = dt.datetime.now(dt.timezone.utc)
        return {"schema": "musetalk_startup_media_v1", "instance_id": "456", "network_path": "ec2_routed_webrtc",
                "client_kind": "real_browser", "audio_response_verified": True, "avatar_verified": True,
                "backend_verified": True, "no_deferred_build_or_download": True, "test_audio_sha256": "a" * 64,
                "first_usable_frame_utc": start.isoformat(), "validated_at_utc": (start + dt.timedelta(seconds=20)).isoformat(),
                "measurement_seconds": 20, "streams": [{"minimum_anchored_1s_decoded_fps": 20,
                    "speaking_fresh_fraction": 1, "content_fps": 20, "decoded_valid_frames": 400,
                    "maximum_held_run": 0, "maximum_gap_ms": 50, "gaps_over_100ms": 0,
                    "send_cadence_caused_gaps": 0, "negotiated_codec": "VP8"}],
                "artifacts": [{"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}]}

    def test_real_media_requires_duration_trace_identity_and_each_gate(self):
        doc = self.media()
        self.assertTrue(observer.validate_media(doc, "456", self.root))
        for key, bad in (("network_path", "direct_loopback"), ("measurement_seconds", 1),
                         ("instance_id", "another"), ("artifacts", []), ("backend_verified", False)):
            altered = copy.deepcopy(doc); altered[key] = bad
            with self.subTest(key=key), self.assertRaises(observer.Invalid):
                observer.validate_media(altered, "456", self.root)
        doc["streams"][0]["content_fps"] = 17.9
        with self.assertRaises(observer.Invalid):
            observer.validate_media(doc, "456", self.root)

    def test_missing_or_modified_media_artifact_fails(self):
        doc = self.media()
        doc["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaises(observer.Invalid):
            observer.validate_media(doc, "456", self.root)
        doc["artifacts"][0]["path"] = "missing.json"
        with self.assertRaises(observer.Invalid):
            observer.validate_media(doc, "456", self.root)

    def test_credential_urls_and_external_http_rejected(self):
        for url in ("http://example.com", "https://user:pass@example.com", "https://example.com?token=private"):
            with self.subTest(url=url), self.assertRaises(observer.Invalid):
                observer.safe_base_url(url)

    def test_sanitizer_redacts_both_aws_key_parts(self):
        value = prepare.sanitize_script('export AWS_ACCESS_KEY_ID="PRIVATEID"\nexport AWS_SECRET_ACCESS_KEY="PRIVATEVALUE"\n')
        self.assertNotIn("PRIVATEID", value)
        self.assertNotIn("PRIVATEVALUE", value)
        self.assertEqual(value.count("[REDACTED]"), 2)

    def expiry_ledger(self):
        ledger = self.create()
        ledger["label"] = "musetalk-r5-3090-dev-test"
        return ledger

    def test_expiry_only_owned_id_label_gpu_after_deadline(self):
        ledger = self.expiry_ledger()
        instance = {"id": 456, "label": ledger["label"], "gpu_name": "RTX 3090", "num_gpus": 1}
        calls = []
        def destroy(*a, **kw):
            calls.append(kw)
            return {"success": True, "result": {"action": "destroy"}}
        outcome = runner.expire_owned(ledger, "2020-01-01T00:00:00Z", [instance], destroy, "token", observer)
        self.assertEqual(outcome["status"], "destroy")
        self.assertFalse(calls[0]["payload"]["force"])
        for key, value in (("id", 51074906), ("id", 457), ("gpu_name", "RTX 4070S")):
            altered = dict(instance, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(observer.Invalid):
                runner.expire_owned(ledger, "2020-01-01T00:00:00Z", [altered], destroy, "token", observer)
        future = "2099-01-01T00:00:00Z"
        self.assertEqual(runner.expire_owned(ledger, future, [instance], destroy, "token", observer)["status"], "not_due")
        self.assertEqual(len(calls), 1)

    def test_expiry_relabelled_or_duplicate_target_not_destroyed(self):
        ledger = self.expiry_ledger()
        def forbidden(*a, **kw):
            raise AssertionError("Must not call destroy")
        with self.assertRaises(observer.Invalid):
            runner.expire_owned(ledger, "2020-01-01T00:00:00Z", [{"id": 456, "label": "production"}], forbidden, "token", observer)
        duplicated = [{"id": 456, "label": ledger["label"]}] * 2
        with self.assertRaises(observer.Invalid):
            runner.expire_owned(ledger, "2020-01-01T00:00:00Z", duplicated, forbidden, "token", observer)


if __name__ == "__main__":
    unittest.main()
