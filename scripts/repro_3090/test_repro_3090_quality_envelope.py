"""CPU/synthetic comparisons only; no measured 3090 envelope is created here."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_envelope as q


def historical_reports():
    policy = q.checks.read(q.POLICY)
    return {entry["identity"]["name"]: q.checks.read(q.ROOT / path)
            for path, entry in policy["historical_reference_sources"].items() if "identity" in entry}


def reference(label="reference1"):
    avatars = {ident: {"metrics": q.extract_avatar(data), "signature": q.signature(data, "a" * 64),
                       "original_verdict": data["verdict"], "B_frames_sha256": data["B"]["info"]["frames_sha256"]}
               for ident, data in historical_reports().items()}
    return {"role": "reference", "label": label, "run": {"path": "/SYNTHETIC/" + label, "sha256": "b" * 64},
            "started_utc": "2026-10-08T10:00:00Z", "input_manifest_sha256": "c" * 64,
            "engine": {"synthetic": "reference-engine"}, "taesd": {"synthetic": "reference-decoder"},
            "metric_tool_sha256": "d" * 64, "capture_pixel_hashes": {i: [["e" * 64, "f" * 64]] for i in q.checks.IDENTITIES},
            "unet_capture_rows": {"main": [], "holdout": []}, "avatars": avatars,
            "global_metrics": {"unet.main.mae_max": .0044, "unet.main.max_abs_max": .7766,
                               "unet.holdout.mae_max": .0039, "unet.holdout.max_abs_max": 1.2601,
                               "taesd.G_TAESD_full_max": 5, "taesd.G_TAESD_full_mean": .0657},
            "strict_original_gates": {"unet_main": "FAIL", "unet_holdout": "FAIL", "taesd": "FAIL",
                                      **{i: a["original_verdict"] for i, a in avatars.items()}}}


def frozen_fixture():
    first, second = reference(), reference("reference2")
    historical = {"avatars": {i: a["metrics"] for i, a in first["avatars"].items()},
                  "global_metrics": first["global_metrics"], "comparability": []}
    policy = q.checks.read(q.POLICY)
    env = q.freeze(policy, [first, second], historical, "2026-10-08T11:00:00Z")
    candidate = copy.deepcopy(first)
    candidate.update(role="candidate", started_utc="2026-10-08T12:00:00Z")
    return policy, env, candidate


class QualityEnvelopeTests(unittest.TestCase):
    def test_real_historical_numerical_metrics_and_closed_taesd_avatar_set(self):
        gate = q.ROOT / "docs/fps_comparisons/4070s_400fps_20260928/gate"
        unets = {s: q.checks.read(gate / f"gunet_srcg50_{s}.json") for s in ("main", "holdout")}
        taesd = q.checks.read(q.ROOT / "docs/fps_comparisons/ampere_plus_r5_20260930/taesd_gate/gate_taesd_trt.json")
        metrics = q.global_metrics(unets, taesd)
        self.assertEqual(len(metrics), 104)
        self.assertEqual(metrics["taesd.G_TAESD_full_max"], 5)
        for mutation in ("missing", "unknown", "exactness", "batch", "crop", "forged_verdict", "new_gate"):
            bad = copy.deepcopy(taesd)
            if mutation == "missing":
                del bad["per_avatar_trt_vs_compiled"]["codex_smoke"]
            elif mutation == "unknown":
                bad["per_avatar_trt_vs_compiled"]["codex_smoke"]["full"]["new_metric"] = 0
            elif mutation == "exactness":
                bad["gate"]["rerun_mismatched_bytes"] = 1
            elif mutation == "batch":
                bad["batching"]["n3"]["fp16_equal"] = False
            elif mutation == "crop":
                bad["vae_decode_latents"]["fused_dtype_shape"][1][1] = 152
            elif mutation == "forged_verdict":
                bad["gate"]["verdict"] = "PASS"
            else:
                bad["gate"]["new_similarity_metric"] = 1.
            with self.assertRaises(q.checks.Invalid):
                q.global_metrics(unets, bad)

    def test_capture_uses_canonical_exactness_object_not_summary_boolean_as_map(self):
        cap = {"status": "complete", "code_integrity": {"matches_accepted_render_json": True},
               "summary": {"deterministic_per_identity": True},
               "args": {"streams": 6, "loops": 1, "repeats": 1, "pack": 16, "decode_split": 8,
                        "mode": "multi", "backend": "stagewise16_taesdtrt", "identity_list": list(q.checks.IDENTITIES),
                        "save_arrays": True, "encode": True},
               "exactness": {"clips_checked": 6, "clips_expected": 6,
                             "deterministic_per_identity": {i: True for i in q.checks.IDENTITIES},
                             "distinct_hash_pairs_per_identity": {i: [["a" * 64, "b" * 64]] for i in q.checks.IDENTITIES}}}
        self.assertEqual(q.validate_capture(cap), cap["exactness"]["distinct_hash_pairs_per_identity"])
        for mutation in ("clips", "resolution_recipe", "identity_order", "hash", "determinism"):
            bad = copy.deepcopy(cap)
            if mutation == "clips":
                bad["exactness"]["clips_checked"] = 5
            elif mutation == "resolution_recipe":
                bad["args"]["decode_split"] = 0
            elif mutation == "identity_order":
                bad["args"]["identity_list"].reverse()
            elif mutation == "hash":
                bad["exactness"]["distinct_hash_pairs_per_identity"][q.checks.IDENTITIES[0]] = [["invalid", "b" * 64]]
            else:
                bad["exactness"]["deterministic_per_identity"][q.checks.IDENTITIES[0]] = False
            with self.assertRaises(q.checks.Invalid):
                q.validate_capture(bad)

    def test_actual_six_historical_reports_use_explicit_registry(self):
        reports = historical_reports()
        self.assertEqual(set(reports), set(q.checks.IDENTITIES))
        for data in reports.values():
            self.assertEqual(set(q.extract_avatar(data)), set(q.METRICS))
        self.assertEqual(len(q.METRICS), 99)
        self.assertEqual(q.METRICS["lip_sync.pearson_A_vs_B"]["direction"], "similarity")
        self.assertEqual(q.METRICS["seam.band_max_abs.max"]["direction"], "error")

    def test_actual_metric_source_lineage_only_changed_two_path_assignments(self):
        result = q.code_lineage()
        self.assertEqual(result["historical_metric_sha256"], "d987c623f3b6759cdd294c8f7ccb90ce270e2495b450895d1c4490de651d6359")
        self.assertEqual(result["excluded_path_assignments"], ["FACEMESH_PY", "DIV"])

    def test_noise_comes_only_from_reference_repeats(self):
        b = q.bound([2., 3.], [10.], "error")
        self.assertEqual(b["observed_reference_repeat_spread"], 1.)
        self.assertEqual(b["limit"], 11.)
        b = q.bound([.99, .98], [.97], "similarity")
        self.assertEqual(b["observed_reference_repeat_spread"], .99 - .98)
        self.assertEqual(b["limit"], .97 - (.99 - .98))
        self.assertEqual(q.bound([4., 4.], [5.], "error")["limit"], 5.)
        for values, direction in (([1.], "error"), ([1., float("nan")], "error"), ([1., 2.], "invented")):
            with self.assertRaises(q.checks.Invalid):
                q.bound(values, [], direction)

    def test_strict_fail_is_not_relabelled_pass_on_numerical_parity(self):
        policy, env, candidate = frozen_fixture()
        result = q.compare(env, candidate, policy)
        self.assertEqual(result["quality_parity_with_reference"], "PASS")
        self.assertEqual(result["strict_original_gates"]["unet_holdout"], "FAIL")
        self.assertEqual(result["strict_original_gates"]["taesd"], "FAIL")
        self.assertEqual(result["decision"], "incomplete")
        self.assertEqual(result["visual_inspection_status"], "NOT_EVALUATED_BY_HELPER")
        self.assertFalse(result["release_ready"])

    def test_unrounded_tiny_regression_fails_with_zero_reference_noise(self):
        policy, env, candidate = frozen_fixture()
        candidate["global_metrics"]["taesd.G_TAESD_full_max"] = math.nextafter(5., math.inf)
        result = q.compare(env, candidate, policy)
        self.assertEqual(result["quality_parity_with_reference"], "FAIL")
        self.assertEqual(result["decision"], "rejected")

    def test_each_avatar_compared_independently_both_metric_directions(self):
        policy, env, candidate = frozen_fixture()
        ident = q.checks.IDENTITIES[0]
        key = "lip_sync.pearson_A_vs_B"
        candidate["avatars"][ident]["metrics"][key] = math.nextafter(env["avatars"][ident]["bounds"][key]["limit"], -math.inf)
        result = q.compare(env, candidate, policy)
        self.assertEqual(result["quality_parity_with_reference"], "FAIL")
        failed = [r for r in result["metrics"] if r["status"] == "FAIL"]
        self.assertEqual([(r["scope"], r["metric"]) for r in failed], [(ident, key)])

    def test_sharpness_has_both_blur_and_oversharpening_bounds(self):
        policy, env, candidate = frozen_fixture()
        ident = q.checks.IDENTITIES[0]
        for suffix, value in (("lower", 0.), ("upper", 100.)):
            bad = copy.deepcopy(candidate)
            bad["avatars"][ident]["metrics"]["overall.sharpness.mouth.ratio." + suffix] = value
            self.assertEqual(q.compare(env, bad, policy)["quality_parity_with_reference"], "FAIL")

    def test_candidate_before_freeze_or_changed_inputs_are_invalid(self):
        policy, env, candidate = frozen_fixture()
        for key, value in (("started_utc", "2026-10-08T10:59:59Z"), ("input_manifest_sha256", "wrong")):
            bad = {**candidate, key: value}
            with self.assertRaises(q.checks.Invalid):
                q.compare(env, bad, policy)

    def test_unknown_or_missing_metric_never_defaults_to_pass(self):
        policy, env, candidate = frozen_fixture()
        for mutation in ("unknown", "missing", "null", "nan"):
            bad = copy.deepcopy(candidate)
            metrics = bad["avatars"][q.checks.IDENTITIES[0]]["metrics"]
            name = next(iter(metrics))
            if mutation == "unknown":
                metrics["new.metric"] = 0.
            elif mutation == "missing":
                del metrics[name]
            else:
                metrics[name] = None if mutation == "null" else float("nan")
            with self.assertRaises(q.checks.Invalid):
                q.compare(env, bad, policy)

    def test_unknown_fields_in_canonical_output_are_rejected(self):
        report = historical_reports()[q.checks.IDENTITIES[0]]
        for group in (report, report["overall"], report["overall"]["psnr"]["full"], report["flicker"]["mouth"]):
            group["unknown_metric"] = 123
            with self.assertRaises(q.checks.Invalid):
                q.extract_avatar(report)
            del group["unknown_metric"]

    def test_hard_invariants_cannot_be_covered_by_reference_noise(self):
        source = historical_reports()[q.checks.IDENTITIES[0]]
        for mutate in (lambda r: r["lip_sync"].update(xcorr_lag_frames=1),
                       lambda r: r["seam"]["protected_lip"]["B"].update(max_rgb_difference=1),
                       lambda r: r["chin"]["landmark_deviation_A_vs_B"].update(missing_face_frames=1),
                       lambda r: r.update(frames=239),
                       lambda r: r["environment"]["blend_env"].update(MUSETALK_BLEND_FIXED_POINT="0")):
            bad = copy.deepcopy(source)
            mutate(bad)
            with self.assertRaises(q.checks.Invalid):
                q.extract_avatar(bad)

    def test_forged_original_verdict_rejected(self):
        report = historical_reports()[q.checks.IDENTITIES[0]]
        self.assertEqual(report["verdict"], "FAIL")
        report["verdict"] = "PASS"
        for row in report["gates"]:
            if row["result"] == "fail":
                row["result"] = "pass"
        with self.assertRaisesRegex(q.checks.Invalid, "canonical calculations"):
            q.extract_avatar(report)

    def test_reference_repeat_must_have_distinct_run_and_same_engine_pixels(self):
        policy, env, _ = frozen_fixture()
        refs = copy.deepcopy(env["references"])
        for key in ("engine", "capture_pixel_hashes", "input_manifest_sha256"):
            bad = copy.deepcopy(refs)
            bad[1][key] = "changed"
            with self.assertRaises(q.checks.Invalid):
                q.freeze(policy, bad, env["historical"], env["frozen_utc"])
        with self.assertRaises(q.checks.Invalid):
            q.freeze(policy, [refs[0], refs[0]], env["historical"], env["frozen_utc"])

    def test_face_capture_shape_and_raw_pixels_are_bound(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "faces.npz"
            header = repr({"descr": "|u1", "fortran_order": False, "shape": (240, 256, 256, 3)}).encode() + b"\n"
            raw = b"\0" * (240 * 256 * 256 * 3)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("faces.npy", b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + raw)
            self.assertEqual(q.face_array_hash(path), hashlib.sha256(raw).hexdigest())
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("faces.npy", b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + b"bad")
            with self.assertRaisesRegex(q.checks.Invalid, "truncated"):
                q.face_array_hash(path)

    def test_envelope_file_digest_mismatch_is_invalid_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            envelope = base / "envelope.json"
            envelope.write_text("{}")
            inputs = base / "inputs.json"
            inputs.write_text(json.dumps({"schema": "repro_3090_inputs_v1", "files": [{"path": envelope.name, "sha256": q.checks.sha256(envelope)}]}))
            out = base / "comparison.json"
            argv = ["compare", "--inputs", str(inputs), "--candidate", str(base / "absent"), "--envelope", str(envelope),
                    "--envelope-sha256", "f" * 64, "--out", str(out)]
            self.assertEqual(q.main(argv), 2)
            self.assertEqual(q.checks.read(out)["status"], "INVALID")
            with self.assertRaises(q.checks.Invalid):
                q.main(argv)


if __name__ == "__main__":
    unittest.main()
