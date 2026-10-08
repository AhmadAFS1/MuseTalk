#!/usr/bin/env python3
"""Freeze repeated-reference r5 quality bounds, then compare without widening.

CPU/stdlib only. Does not run inference, invent absent historical metrics, change
original gates, inspect images, or approve a release. Reference freeze and native
comparison are separate commands; a comparison requires the earlier envelope SHA.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import report as checks

POLICY = ROOT / "docs/fps_comparisons/rtx3090_r5_20261008/quality/quality_acceptance.json"
HISTORICAL_REVISION = "5cc706e90e50e93da1310628c025a84199cd8042"
BUNDLES = {
    "fixtures": ("repro-avatar-diversity-20260927", "4fbb421484b119814c52ed40840ae41c0481086a53960847b9e215e01b015149"),
    "calibration": ("repro-calibration-unet-multi-avatar-20260928", "5b38ed6d0d776b43d405d85839cabeeaf143972ea67c56dd4eaac7e46bab3ded"),
}
SCHEMA = "r5_measured_quality_envelope_v1"
TAESD_AVATARS = {
    "chin_h3_japanese_new", "chin_h3_latina_new", "chinese_bob_pink_bedroom_talking_3373c10448", "codex_smoke",
    *("div_" + ident for ident in checks.IDENTITIES), "indian_realtime_talking_20f9845543",
    "japanese_realtime_talking_7d94520b7f", "japanese_realtime_talking_7d94520b7f_fh1", "latina_guided_20260925_talking_84c5bc80b8",
}


def sha_data(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": checks.sha256(path), "bytes": path.stat().st_size}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    checks.require(parsed.tzinfo is not None, "quality timestamps must include timezone")
    return parsed


def number(value, name):
    checks.require(type(value) in (int, float) and math.isfinite(value), "missing/nonfinite metric: " + name)
    return value


def get(data, path):
    for key in path.split("."):
        checks.require(isinstance(data, dict) and key in data, "missing metric field: " + path)
        data = data[key]
    return data


def keys(data, expected, where):
    checks.require(isinstance(data, dict) and set(data) == set(expected.split()), "unknown/missing metric fields: " + where)


def metric_registry():
    """Every bound has a named semantic direction; no numeric-tree flattening."""
    out = {}
    def add(path, direction, name=None):
        out[name or path] = {"path": path, "direction": direction}
    for key in ("pearson_A_vs_B", "pearson_A_vs_B_px", "xcorr_lag_corr"):
        add("lip_sync." + key, "similarity")
    for key in ("mean_abs_delta_px", "p95_abs_delta_px", "mean_abs_delta_eye_spans"):
        add("lip_sync." + key, "error")
    for region in ("mouth", "jaw", "ring", "face"):
        for diff in ("first", "second"):
            add(f"flicker.{region}.ratio_{diff}_diff", "error")
    for region in ("mouth_box", "jaw_box"):
        add(f"flicker.temporal_hf_power_gt_6hz.{region}.ratio", "error")
    for region in ("band", "ring", "outside", "jaw_band"):
        for kind in ("max", "mean"):
            for stat in ("mean", "p99", "max"):
                add(f"seam.{region}_{kind}_abs.{stat}", "error")
    for kind in ("target_chin_abs_error_px", "positive_excess_chin_length_px", "source_chin_error_percent_eye_span"):
        for stat in ("mean", "median", "p95", "p99", "max"):
            add(f"chin.B.{kind}.{stat}", "error")
    add("chin.B.source_relative_jaw_step_percent_eye_span", "error")
    for stat in ("mean", "p99", "max", "jaw_mean", "lips_mean", "window_mean", "window_p99"):
        add("chin.landmark_deviation_A_vs_B.tracked_final." + stat, "error")
    for region in ("full", "face_box", "mouth_roi"):
        # global_db can be null for identical pixels (+inf before JSON cleanup).
        # The tool's finite 100dB-capped mean and worst frame need no invented cap.
        for stat in ("mean_frame_db_capped", "worst_frame_db"):
            add(f"overall.psnr.{region}.{stat}", "similarity")
        for stat in ("mean", "p99", "max"):
            add(f"overall.ssim.{region}.{stat}", "similarity")
        add("overall.ssim.worst." + region, "similarity")
    add("overall.mouth_mae_normalized", "error")
    for kind in ("face_max_abs", "full_max_abs"):
        for stat in ("mean", "p99", "max"):
            add(f"overall.{kind}.{stat}", "error")
    for region in ("mouth", "face"):
        # Both blur and oversharpening can be regressions. Preserve the reference
        # band; do not treat arbitrarily increasing sharpness as always better.
        path = f"overall.sharpness.{region}.ratio"
        add(path, "similarity", path + ".lower")
        add(path, "error", path + ".upper")
    for key in ("deltaE76_of_mean", "per_pixel_deltaE76_mean"):
        add("overall.color_lab_face_box." + key, "error")
    for stat in ("mean", "median", "p95", "p99", "max"):
        add("overall.color_lab_face_box.deltaE76_of_per_frame_means." + stat, "error")
    return out


METRICS = metric_registry()
EXCLUDED_FIELDS = {
    "overall.psnr.*.global_db": "Can be undefined/infinite for identical frames; finite canonical capped mean/worst are bounded instead",
    "lip_sync.A/B.aperture_*": "Absolute aperture is not monotonic quality; correlation/delta/zero lag are bounded",
    "chin.A.*": "Fixed comparison arm, not the candidate; its frame hash and configuration must match",
    "chin.B.lower_lip_to_chin_px": "Absolute anatomy has no monotonic quality direction",
    "chin.delta_target_error_mean_px": "Signed diagnostic; bound absolute B error directly",
    "chin.landmark_deviation_A_vs_B.generated_in_render": "Render tracker diagnostic; final composed-frame landmark metrics are bounded",
    "flicker.*.A/B_first/second_diff": "Bound relative candidate/reference temporal ratios rather than absolute scene motion",
    "flicker.temporal_hf_power_gt_6hz.*.A/B_power/fraction": "Bound canonical power ratio; no independent invented fraction threshold",
    "overall.sharpness.*.A/B": "Bound the reference-relative ratio on both sides",
    "overall.color_lab_face_box.*_mean_Lab": "Signed color channels lack a scalar quality direction; bound canonical deltaE measures",
    "series/xcorr_curve/counts/coordinates/timing": "Raw supporting diagnostics or coverage, never generic error metrics",
    "chin.landmark_dev_calibrated_proposal": "Historical report-only proposal; not promoted to an original strict gate",
    "syncnet": "Not collected by canonical e1 wrapper; a new metric requires a separate preregistered policy",
    "TAESD.frames_with_max_ge3": "Threshold-count diagnostic; existing per-avatar max/mean/nonzero fraction are bounded instead",
    "TAESD.lsb/fp16_max_abs/psnr_min_db": "Auxiliary compiled-vs-eager noise context and duplicate corpus diagnostics; registered full-height/rows104 uint8 metrics and per-avatar errors are the primary TAESD contract",
}


def validate_metric_schema(r):
    keys(r, "tool tool_sha256 helper_sha256 name profile frames_mode frames identical_frames_sha identity A B environment lip_sync flicker seam chin overall notes gates verdict seconds peak_rss_mb", "root")
    keys(r["lip_sync"], "definition A B pearson_A_vs_B pearson_A_vs_B_px xcorr_lag_frames xcorr_lag_corr xcorr_curve mean_abs_delta_px p95_abs_delta_px mean_abs_delta_eye_spans series", "lip_sync")
    keys(r["flicker"], "mouth jaw ring face temporal_hf_power_gt_6hz", "flicker")
    for region in ("mouth", "jaw", "ring", "face"):
        keys(r["flicker"][region], "A_first_diff B_first_diff ratio_first_diff A_second_diff B_second_diff ratio_second_diff", "flicker." + region)
    keys(r["flicker"]["temporal_hf_power_gt_6hz"], "mouth_box jaw_box", "flicker.hf")
    for value in r["flicker"]["temporal_hf_power_gt_6hz"].values():
        keys(value, "box_xyxy A_power B_power ratio A_fraction B_fraction", "flicker.hf.region")
    keys(r["seam"], "regions band_max_abs band_mean_abs ring_max_abs ring_mean_abs outside_max_abs outside_mean_abs jaw_band_max_abs jaw_band_mean_abs ring_flicker_ratio series protected_lip", "seam")
    keys(r["chin"], "window formula A B delta_target_error_mean_px landmark_deviation_A_vs_B", "chin")
    keys(r["overall"], "psnr ssim mouth_mae_normalized face_max_abs full_max_abs sharpness color_lab_face_box", "overall")
    keys(r["overall"]["psnr"], "full face_box mouth_roi", "psnr")
    keys(r["overall"]["ssim"], "full face_box mouth_roi worst", "ssim")
    keys(r["overall"]["sharpness"], "definition mouth face", "sharpness")
    keys(r["overall"]["color_lab_face_box"], "A_mean_Lab B_mean_Lab delta_mean_Lab deltaE76_of_mean deltaE76_of_per_frame_means per_pixel_deltaE76_mean", "color")
    for arm in ("A", "B"):
        keys(r["chin"][arm], "target_chin_abs_error_px positive_excess_chin_length_px lower_lip_to_chin_px source_chin_error_percent_eye_span source_relative_jaw_step_percent_eye_span series_signed_error_px", "chin." + arm)
    lm = r["chin"]["landmark_deviation_A_vs_B"]
    keys(lm, "landmarks tracked_final missing_face_frames generated_in_render", "landmarks")
    keys(lm["tracked_final"], "mean p99 max jaw_mean lips_mean window_mean window_p99", "tracked landmarks")
    # Metric-leaf dictionaries are also closed: a new scalar is not silently
    # dropped just because its surrounding metric group is already recognized.
    parents = {spec["path"].rsplit(".", 1)[0] for spec in METRICS.values()}
    for parent in parents:
        obj = get(r, parent)
        if parent.startswith("overall.psnr."):
            expected = "global_db mean_frame_db_capped worst_frame_db identical_frames"
        elif parent.startswith("overall.sharpness."):
            expected = "A B ratio"
        elif parent.startswith("chin.B.") or parent.endswith("deltaE76_of_per_frame_means"):
            expected = "mean median p95 p99 max n"
        elif parent.startswith("seam.") or parent in ("overall.face_max_abs", "overall.full_max_abs") or parent in ("overall.ssim.full", "overall.ssim.face_box", "overall.ssim.mouth_roi"):
            expected = "mean p99 max"
        elif parent == "overall.ssim.worst":
            expected = "full face_box mouth_roi"
        else:
            continue
        keys(obj, expected, parent)


def extract_avatar(r):
    validate_metric_schema(r)
    checks.finite(r)
    checks.require(r["profile"] == "e1" and r["frames_mode"] == "raw" and r["frames"] == 240, "not canonical raw e1/240 metric run")
    checks.require(r["identity"]["name"] in checks.IDENTITIES and r["chin"]["window"] == [24, 216], "unknown identity/chin window")
    checks.require(r["environment"]["chin_is_accepted"] is True and r["environment"]["blending_is_accepted"] is True, "composition implementation changed")
    for value in r["environment"]["blend_env"].values():
        checks.require(value is None or str(value).lower() in ("", "1", "true", "yes", "on"), "canonical blend optimization policy changed")
    checks.require(r["lip_sync"]["xcorr_lag_frames"] == 0, "hard invariant: nonzero lip lag")
    checks.require(r["chin"]["landmark_deviation_A_vs_B"]["missing_face_frames"] == 0, "hard invariant: missing faces")
    for arm in ("A", "B"):
        a = r[arm]
        checks.require(a["compose"] == "refined" and a["perturb_lsb"] == 0 and a["info"]["frames_source"] == "raw", "non-refined/perturbed arm")
        checks.require(a["info"]["facemesh"]["frames"] == 240 and a["info"]["facemesh"]["missing_frames"] == [], "incomplete FaceMesh")
        pl = r["seam"]["protected_lip"][arm]
        checks.require(pl and pl["frames"] == 240 and pl["max_rgb_difference"] == 0 and pl["frames_nonzero"] == 0, "hard invariant: protected lips changed")
        checks.require(re.fullmatch(r"[0-9a-f]{64}", a["info"]["frames_sha256"]) is not None, "missing frame hash")
        checks.require(r["chin"][arm]["target_chin_abs_error_px"]["n"] == 192, "chin window coverage missing")
    values = {name: number(get(r, spec["path"]), name) for name, spec in METRICS.items()}
    gate_names = {row["gate"] for row in r["gates"]}
    expected = {"lip.aperture_corr", "lip.xcorr_lag_frames", "lip.mean_abs_delta_px", "flicker.mouth_ratio", "flicker.jaw_ratio", "flicker.ring_ratio", "flicker.hf_mouth_box_ratio", "flicker.hf_jaw_box_ratio", "chin.target_error_mean_px", "frames.raw_pre_encode", "chin.landmark_dev_mean_px", "chin.landmark_dev_p99_px", "chin.landmark_dev_calibrated_proposal", "track.missing_face_frames", "seam.protected_lip_A", "seam.protected_lip_B", "overall.mouth_sharpness_ratio", "overall.mouth_sharpness_upper", "overall.face_psnr_db"}
    checks.require(gate_names == expected and len(r["gates"]) == len(expected), "unknown/missing strict gate")
    checks.require(all(row["result"] in ("pass", "fail", "report") for row in r["gates"]), "incomplete strict gate")
    # Recompute original active e1 gates from raw metrics; the historical proposed
    # calibration and report-only diagnostics do not silently become strict PASS.
    active = [r["lip_sync"]["pearson_A_vs_B"] >= .97, r["lip_sync"]["mean_abs_delta_px"] <= .5,
              *(r["flicker"][reg]["ratio_first_diff"] <= 1.05 for reg in ("mouth", "jaw", "ring")),
              r["chin"]["B"]["target_chin_abs_error_px"]["mean"] <= r["chin"]["A"]["target_chin_abs_error_px"]["mean"] + .05 + 1e-12,
              r["chin"]["landmark_deviation_A_vs_B"]["tracked_final"]["mean"] <= .05,
              r["chin"]["landmark_deviation_A_vs_B"]["tracked_final"]["p99"] <= .15,
              r["overall"]["sharpness"]["mouth"]["ratio"] >= .95]
    strict = "PASS" if all(active) else "FAIL"
    checks.require(("FAIL" if any(row["result"] == "fail" for row in r["gates"]) else "PASS") == strict, "strict raw gate rows disagree with canonical calculations")
    checks.require(r["verdict"] == strict, "strict verdict disagrees with raw gates")
    return values


def signature(r, audio_sha256):
    env = r["environment"]
    return {"identity": {k: r["identity"][k] for k in ("name", "source_sha256", "source_landmarks_sha256", "cache_sha256", "masks_sha256")},
            "audio_sha256": audio_sha256, "A_frames_sha256": r["A"]["info"]["frames_sha256"],
            "helper_sha256": r["helper_sha256"], "chin_sha256": env["chin_sha256"], "blending_sha256": env["blending_sha256"],
            "numpy": env["numpy"], "opencv": env["opencv"],
            "mediapipe": {a: r[a]["info"]["facemesh"]["mediapipe"] for a in ("A", "B")},
            "frames": r["frames"], "frames_mode": r["frames_mode"], "profile": r["profile"]}


def global_metrics(unets, taesd):
    values = {}
    for split, expected_count in (("main", 176), ("holdout", 48)):
        data = unets[split]
        keys(data, "backend precision capture_dir padded_batch_size actual_only summary files", "UNet report")
        checks.require(data["summary"]["files"] == len(data["files"]) == expected_count, "UNet corpus coverage missing")
        checks.require(data["precision"] == "fp16" and data["padded_batch_size"] == 8 and data["actual_only"] is False, "UNet comparison contract changed")
        allowed = {"files", "latency_ms_mean", "latency_ms_max", "frames_per_sec_mean", "frames_per_sec_min"}
        for metric in ("mae", "rmse", "p95_abs", "max_abs"):
            for stat in ("mean", "max"):
                name = metric + "_" + stat
                allowed.add(name)
                values[f"unet.{split}.{name}"] = number(data["summary"][name], name)
        checks.require(set(data["summary"]) == allowed, "unknown/missing UNet summary metric")
    keys(taesd, "schema created_utc env gpu torch engine probe_at_load files frames gate lsb per_avatar_trt_vs_compiled fp16_max_abs psnr_min_db batching vae_decode_latents seconds tags worst_frame_png", "TAESD report")
    gate = taesd["gate"]
    keys(gate, "G_TAESD_full_max G_TAESD_full_mean G_TAESD_rows104_max G_TAESD_rows104_mean thresholds fused_post_mismatched_bytes fused_frames rerun_mismatched_bytes G_TAESD bit_exact_checks verdict", "TAESD gate")
    checks.require(taesd["files"] == 448 and taesd["frames"] == 3584 and gate["fused_frames"] == 3584, "TAESD corpus coverage missing")
    checks.require(gate["bit_exact_checks"] == "PASS" and gate["fused_post_mismatched_bytes"] == gate["rerun_mismatched_bytes"] == 0, "TAESD hard exactness failed")
    checks.require(gate["thresholds"] == {"max_lsb": 3, "mean_lsb": .2}, "original TAESD thresholds changed")
    batching, vae = taesd["batching"], taesd["vae_decode_latents"]
    keys(batching, "n1 n3 n5 n7 n9 n13 n16 n17 n24 n64 shuffled_composition_equal every_slot_position_equal fp32_input_equal compiled_shuffled_composition_equal", "TAESD batching")
    for name, value in batching.items():
        if name.startswith("n"):
            keys(value, "u8_equal fp16_equal", "TAESD partial batch")
            checks.require(value["u8_equal"] is True and value["fp16_equal"] is True, "TAESD hard partial-batch equality failed")
        else:
            checks.require(value is True, "TAESD hard composition equality failed")
    keys(vae, "backend_name_trt fused_equals_nonfused fused_equals_reference fused_dtype_shape nonfused_dtype_shape compiled_backend_equals_compiled_reference", "TAESD VAE dispatch")
    checks.require(vae["backend_name_trt"] == "taesd_trt"
                   and all(vae[k] is True for k in ("fused_equals_nonfused", "fused_equals_reference", "compiled_backend_equals_compiled_reference"))
                   and vae["fused_dtype_shape"] == vae["nonfused_dtype_shape"] == ["uint8", [8, 256, 256, 3], True],
                   "TAESD hard full-height VAE dispatch equality failed")
    for name in ("G_TAESD_full_max", "G_TAESD_full_mean", "G_TAESD_rows104_max", "G_TAESD_rows104_mean"):
        values["taesd." + name] = number(gate[name], name)
    checks.require(set(taesd["per_avatar_trt_vs_compiled"]) == TAESD_AVATARS, "unknown/missing TAESD corpus avatar")
    checks.require(sum(row["frames"] for row in taesd["per_avatar_trt_vs_compiled"].values()) == 3584, "TAESD per-avatar frame coverage mismatch")
    for ident, row in taesd["per_avatar_trt_vs_compiled"].items():
        keys(row, "split frames full rows104 frames_with_max_ge3", "TAESD avatar")
        for area in ("full", "rows104"):
            keys(row[area], "max mean frac_nonzero", "TAESD avatar region")
            for stat in ("max", "mean", "frac_nonzero"):
                name = f"taesd.avatar.{ident}.{area}.{stat}"
                values[name] = number(row[area][stat], name)
    strict = "PASS" if all(gate[f"G_TAESD_{area}_max"] <= 3 and gate[f"G_TAESD_{area}_mean"] <= .2
                           for area in ("full", "rows104")) else "FAIL"
    checks.require(gate["G_TAESD"] == gate["verdict"] == strict, "original TAESD verdict disagrees with raw values")
    return values


def validate_policy(policy):
    p = policy["quality_parity_policy"]
    checks.require(policy["schema"] == "r5_quality_acceptance_v1" and p["original_failures_remain_failures"] is True
                   and p["no_widening_after_candidate_results"] is True and p["per_avatar_non_regression"] is True
                   and p["inherited_exception_requires_frozen_measured_reference"] is True, "unrecognized/weakened quality policy")
    checks.require(policy["strict_original_gates"] == {"unet_main_and_holdout_mae_max": .01, "unet_main_and_holdout_max_abs": .5,
                                                      "taesd_full_max_lsb": 3, "taesd_full_mean_lsb": .2}, "strict original thresholds changed")


def code_lineage():
    path = "scripts/quality_ab_metrics.py"
    old = subprocess.check_output(["git", "show", HISTORICAL_REVISION + ":" + path], cwd=ROOT)
    current = (ROOT / path).read_bytes()
    def semantic(raw):
        tree = ast.parse(raw)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in ("FACEMESH_PY", "DIV") for t in node.targets))]
        return ast.dump(tree, include_attributes=False)
    checks.require(semantic(old) == semantic(current), "historical metric implementation changed beyond approved path routing")
    unchanged = {}
    for source in ("scripts/validate_unet_backend.py", "scripts/repro_400fps/gate_taesd_trt.py"):
        raw = subprocess.check_output(["git", "show", HISTORICAL_REVISION + ":" + source], cwd=ROOT)
        checks.require(raw == (ROOT / source).read_bytes(), "historical numerical gate implementation changed")
        unchanged[source] = hashlib.sha256(raw).hexdigest()
    return {"historical_revision": HISTORICAL_REVISION, "historical_metric_sha256": hashlib.sha256(old).hexdigest(),
            "current_metric_sha256": hashlib.sha256(current).hexdigest(), "semantic_ast_sha256": hashlib.sha256(semantic(old).encode()).hexdigest(),
            "excluded_path_assignments": ["FACEMESH_PY", "DIV"], "unchanged_numerical_sources": unchanged}


def verify_bundle(kind, root, sidecar, frozen):
    name, archive_sha = BUNDLES[kind]
    manifest = Path(sidecar) / ".musetalk_trt_artifact_manifest.json"
    stamp = Path(sidecar) / ".musetalk_trt_artifact_restored.json"
    restored, bundle = checks.read(stamp), checks.read(manifest)
    checks.require(restored["archive_sha256"] == archive_sha and restored["mode"] in ("restored", "adopted"), "unrecognized historical input bundle receipt")
    files = bundle["files"]
    checks.require(bool(files), "empty input bundle manifest")
    bound = {}
    for row in files:
        file = (Path(root) / row["path"]).resolve()
        checks.require(file.is_relative_to(Path(root).resolve()) and str(file) not in bound, "unsafe/duplicate bundle entry")
        checks.require(checks.sha256(file) == row["sha256"], "bundle content mismatch")
        if str(file) in frozen:
            checks.require(frozen[str(file)] == row["sha256"], "bundle/input freeze content mismatch")
        bound[str(file)] = row["sha256"]
    return {"name": name, "archive_sha256": archive_sha, "receipt": evidence(stamp), "manifest": evidence(manifest),
            "mode": restored["mode"], "files_verified": len(bound), "paths": bound,
            "provenance_limit": "Existing canonical restored/adopted receipt plus freshly verified sidecar files; archive bytes are not re-downloaded"}


def face_array_hash(path):
    """Hash canonical uint8 faces.npz pixels without importing NumPy or pickle."""
    with zipfile.ZipFile(path) as archive:
        checks.require(archive.namelist() == ["faces.npy"], "unexpected face capture NPZ members")
        with archive.open("faces.npy") as stream:
            checks.require(stream.read(6) == b"\x93NUMPY", "invalid face NPY magic")
            version = stream.read(2)
            checks.require(version in (b"\x01\x00", b"\x02\x00"), "unsupported face NPY version")
            length = int.from_bytes(stream.read(2 if version[0] == 1 else 4), "little")
            checks.require(0 < length <= 4096, "invalid face NPY header length")
            header = ast.literal_eval(stream.read(length).decode("ascii"))
            checks.require(header == {"descr": "|u1", "fortran_order": False, "shape": (240, 256, 256, 3)}, "face capture shape/dtype changed")
            digest, count = hashlib.sha256(), 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(chunk)
                checks.require(count <= 240 * 256 * 256 * 3, "face capture has extra pixels")
                digest.update(chunk)
            checks.require(count == 240 * 256 * 256 * 3, "face capture truncated")
            return digest.hexdigest()


def validate_capture(cap):
    checks.require(cap["status"] == "complete" and cap["code_integrity"]["matches_accepted_render_json"] is True, "invalid quality capture")
    exact = cap["exactness"]
    checks.require(cap["summary"]["deterministic_per_identity"] is True
                   and set(exact["deterministic_per_identity"]) == set(checks.IDENTITIES)
                   and all(v is True for v in exact["deterministic_per_identity"].values()), "capture determinism invariant failed")
    checks.require(exact["clips_checked"] == exact["clips_expected"] == 6, "capture clip coverage missing")
    checks.require(cap["args"]["streams"] == 6 and cap["args"]["loops"] == 1 and cap["args"]["repeats"] == 1
                   and cap["args"]["pack"] == 16 and cap["args"]["decode_split"] == 8
                   and cap["args"]["mode"] == "multi" and cap["args"]["backend"] == "stagewise16_taesdtrt"
                   and tuple(cap["args"]["identity_list"]) == checks.IDENTITIES
                   and cap["args"]["save_arrays"] is True and cap["args"]["encode"] is True,
                   "quality capture is not the canonical six-stream full-resolution recipe")
    pairs = exact["distinct_hash_pairs_per_identity"]
    checks.require(set(pairs) == set(checks.IDENTITIES), "missing capture pixel identities")
    for pair in pairs.values():
        checks.require(len(pair) == 1 and len(pair[0]) == 2
                       and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in pair[0]), "invalid capture pixel hashes")
    return pairs


def load_run(path, inputs, role):
    path, inputs = Path(path).resolve(), Path(inputs).resolve()
    r = checks.read(path)
    checks.require(r["schema"] == "repro_3090_v1" and r["suite"] == "quality" and r["status"] in ("PASS", "FAIL"), "incomplete/nonquality wrapper report")
    env = r["environment"]
    checks.gpu_identity(env["gpu"]["name"], env["gpu"]["compute_cap"], False)
    checks.require(checks.sha256(inputs) == env["input_manifest_sha256"], "quality input manifest binding mismatch")
    frozen = {str((inputs.parent / row["path"]).resolve()): row["sha256"] for row in checks.read(inputs)["files"]}
    eng = env["engines"][0]
    compatibility = eng["manifest"].get("hardware_compatibility_level") or "none"
    checks.require(compatibility == ("ampere_plus" if role == "reference" else "none"), "wrong reference/candidate engine compatibility")
    checks.require(env["taesd"]["fingerprint"].get("hardware_compatibility_level", "none") == compatibility, "TAESD role mismatch")
    capture_name = r["label"] + "_quality_capture"
    cap_path = path.parent / (capture_name + ".json")
    cap = checks.read(cap_path)
    capture_pairs = validate_capture(cap)
    checks.loaded_backend(cap, eng["root"], env["taesd"]["key"], env["taesd"]["decoder_plan_sha256"], eng["manifest"])
    for key in ("graph_equals_direct_enqueue", "deterministic_run_to_run"):
        checks.require(eng["manifest"]["probe"][key] is True, "engine hard invariant failed")
    source_prefix = checks.read(path.parent / "srccache.json")
    checks.require(source_prefix["cached_equals_forward"] is True and source_prefix["permuted_rows_exact"] is True, "source prefix hard invariant failed")
    unets = {s: checks.read(path.parent / f"unet_{s}.json") for s in ("main", "holdout")}
    taesd = checks.read(path.parent / "taesd/gate_taesd_trt.json")
    checks.require(taesd["engine"]["key"] == env["taesd"]["key"] and taesd["engine"]["decoder_plan_sha256"] == env["taesd"]["decoder_plan_sha256"], "TAESD gate identity mismatch")
    scalars = global_metrics(unets, taesd)
    avatars, artifacts = {}, [evidence(path), evidence(cap_path), evidence(path.parent / "srccache.json"),
                              evidence(path.parent / "taesd/gate_taesd_trt.json")]
    artifacts += [evidence(path.parent / f"unet_{s}.json") for s in unets]
    for ident in checks.IDENTITIES:
        metric_path = path.parent / "quality_metrics" / f"{ident}__{r['label']}.json"
        metric = checks.read(metric_path)
        checks.require(metric["identity"]["name"] == ident, "metric/avatar identity mismatch")
        source = Path(metric["identity"]["source"])
        for key, filename in (("source_sha256", source.name), ("source_landmarks_sha256", "source_landmarks.npy"),
                              ("cache_sha256", "cache.pt"), ("masks_sha256", "masks.npz")):
            checks.require(frozen.get(str(source.parent / filename)) == metric["identity"][key], "metric fixture hash differs from frozen input")
        audio = str(Path(metric["identity"]["audio"]).resolve())
        checks.require(audio in frozen, "audio missing from frozen manifest")
        checks.require(metric["tool_sha256"] == checks.sha256(ROOT / "scripts/quality_ab_metrics.py")
                       and metric["helper_sha256"] == checks.sha256(ROOT / "scripts/quality_ab_facemesh.py"), "metric tool/helper code mismatch")
        avatars[ident] = {"metrics": extract_avatar(metric), "signature": signature(metric, frozen[audio]),
                          "original_verdict": metric["verdict"], "raw_report": evidence(metric_path),
                          "B_frames_sha256": metric["B"]["info"]["frames_sha256"]}
        artifacts.append(evidence(metric_path))
        for suffix in ("faces.npz", "arrays.npz", "refined.mp4", "faces.mp4"):
            found = list((path.parent / capture_name).glob(f"stream*_{ident}_{suffix}"))
            checks.require(len(found) == 1, "missing/duplicate visual or raw capture artifact")
            artifacts.append(evidence(found[0]))
            if suffix == "faces.npz":
                pairs = capture_pairs[ident]
                checks.require(len(pairs) == 1 and face_array_hash(found[0]) == pairs[0][1], "saved face pixels differ from completed GPU capture")
        checks.require(Path(metric["B"]["faces"]).name == f"stream{checks.IDENTITIES.index(ident):02d}_{ident}_faces.npz", "metric is not bound to matching capture")
    strict_global = {f"unet_{s}": "PASS" if unets[s]["summary"]["mae_max"] <= .01 and unets[s]["summary"]["max_abs_max"] <= .5 else "FAIL" for s in unets}
    strict_global["taesd"] = taesd["gate"]["verdict"]  # recomputed above, including the original rows104 checks
    return {"role": role, "run": evidence(path), "label": r["label"], "started_utc": env["utc"],
            "input_manifest_sha256": env["input_manifest_sha256"], "avatars": avatars, "global_metrics": scalars,
            "engine": eng, "taesd": env["taesd"], "metric_tool_sha256": checks.sha256(ROOT / "scripts/quality_ab_metrics.py"),
            "strict_original_gates": {**strict_global, **{i: a["original_verdict"] for i, a in avatars.items()}},
            "capture_pixel_hashes": capture_pairs, "evidence": artifacts,
            "unet_capture_rows": {s: [{k: row[k] for k in ("path", "actual_batch", "padded_batch", "compared_batch", "items")} for row in unets[s]["files"]] for s in unets}}


def historical_references(policy, reference, inputs, fixture_root, fixture_sidecar, calibration_root, calibration_sidecar):
    lineage = code_lineage()
    frozen = {str((Path(inputs).resolve().parent / row["path"]).resolve()): row["sha256"] for row in checks.read(inputs)["files"]}
    fixtures = verify_bundle("fixtures", fixture_root, fixture_sidecar, frozen)
    calibration = verify_bundle("calibration", calibration_root, calibration_sidecar, frozen)
    captures = {p: h for p, h in calibration["paths"].items() if Path(p).name.startswith("unet_io_") and p.endswith(".pt")}
    checks.require(len(captures) == 448 and all(frozen.get(p) == h for p, h in captures.items()), "canonical capture corpus not fully frozen")
    avatars, unets, taesd, decisions = {}, {}, None, []
    for name, record in policy["historical_reference_sources"].items():
        path = ROOT / name
        checks.require(checks.sha256(path) == record["sha256"], "historical report hash mismatch")
        report = checks.read(path)
        if "identity" in report:
            ident = report["identity"]["name"]
            checks.require(report["tool_sha256"] == lineage["historical_metric_sha256"], "historical metric source lineage mismatch")
            now = reference["avatars"][ident]["signature"]
            # Conditioning is inside cache.pt. The pinned fixture bundle verifies
            # both that identical cache and its speech.wav; this supplements the
            # old report's missing standalone audio checksum, without inventing it.
            audio = [p for p in fixtures["paths"] if p.endswith(f"/{ident}/speech.wav")]
            checks.require(len(audio) == 1 and fixtures["paths"][audio[0]] == now["audio_sha256"], "historical fixture audio provenance missing")
            for key, filename in (("source_sha256", "source.mp4"), ("source_landmarks_sha256", "source_landmarks.npy"),
                                  ("cache_sha256", "cache.pt"), ("masks_sha256", "masks.npz")):
                matches = [p for p in fixtures["paths"] if p.endswith(f"/{ident}/{filename}")]
                checks.require(len(matches) == 1 and fixtures["paths"][matches[0]] == now["identity"][key], "historical fixture bundle hash mismatch")
            comparable = signature(report, now["audio_sha256"]) == now
            checks.require(comparable, "historical metric inputs/A pixels/runtime are not comparable: " + ident)
            avatars[ident] = extract_avatar(report)
            decisions.append({"file": evidence(path), "applicable": True, "scope": "all explicitly registered per-avatar metrics",
                              "basis": "matching source/cache/mask/landmark hashes, A frame hash, pinned fixture/audio bundle, helper/chin/blending, library versions and metric AST lineage"})
        elif name.endswith("gunet_srcg50_main.json") or name.endswith("gunet_srcg50_holdout.json"):
            split = "main" if name.endswith("_main.json") else "holdout"
            rows = [{k: row[k] for k in ("path", "actual_batch", "padded_batch", "compared_batch", "items")} for row in report["files"]]
            checks.require(rows == reference["unet_capture_rows"][split], "historical UNet capture rows differ")
            unets[split] = report
            decisions.append({"file": evidence(path), "applicable": True, "scope": "UNet accuracy summaries only; excludes historical timing/FPS",
                              "basis": "pinned calibration bundle, identical capture row membership/batching and unchanged canonical gate implementation"})
        elif report.get("schema") == "gate_taesd_trt_v1":
            for key in ("recipe", "post_recipe", "onnx_sha256", "batch", "latent_shape", "precision"):
                checks.require(report["engine"]["fingerprint"][key] == reference["taesd"]["fingerprint"][key], "historical TAESD model/graph fingerprint mismatch")
            taesd = report
            decisions.append({"file": evidence(path), "applicable": True, "scope": "full-height/rows104 TAESD LSB summaries only",
                              "basis": "same pinned 448-capture calibration bundle, original 3584 frames, unchanged gate implementation; exactness remains hard"})
        else:
            raise checks.Invalid("unknown historical metric report")
    checks.require(set(avatars) == set(checks.IDENTITIES) and set(unets) == {"main", "holdout"} and taesd is not None,
                   "missing required historical references")
    return {"avatars": avatars, "global_metrics": global_metrics(unets, taesd), "comparability": decisions,
            "source_lineage": lineage, "input_bundles": {"fixtures": fixtures, "calibration": calibration}}


def bound(reference_values, historical_values, direction):
    checks.require(direction in ("error", "similarity"), "unknown metric direction")
    checks.require(len(reference_values) >= 2, "independent reference repeat missing")
    for value in [*reference_values, *historical_values]:
        number(value, "reference")
    noise = max(reference_values) - min(reference_values)
    values = [*reference_values, *historical_values]
    edge = max(values) if direction == "error" else min(values)
    return {"direction": direction, "reference_values": reference_values, "historical_values": historical_values,
            "observed_reference_repeat_spread": noise, "applicable_reference_edge": edge,
            "limit": edge + noise if direction == "error" else edge - noise}


def freeze(policy, references, historical, frozen_at=None):
    validate_policy(policy)
    checks.require(len(references) >= 2 and len({r["run"]["path"] for r in references}) == len(references)
                   and len({r["label"] for r in references}) == len(references), "need separately recorded reference runs")
    first = references[0]
    at = frozen_at or utc()
    for r in references:
        checks.require(r["role"] == "reference" and timestamp(r["started_utc"]) < timestamp(at), "reference must precede freeze")
        for key in ("input_manifest_sha256", "engine", "taesd", "metric_tool_sha256", "capture_pixel_hashes", "unet_capture_rows"):
            checks.require(r[key] == first[key], "reference runs differ in inputs/engines/pixels/implementation: " + key)
        checks.require(set(r["avatars"]) == set(checks.IDENTITIES) and set(r["global_metrics"]) == set(first["global_metrics"]), "reference metric coverage differs")
        checks.require(all(set(a["metrics"]) == set(METRICS) for a in r["avatars"].values()), "unknown/missing reference metric")
    avatars = {}
    for ident in checks.IDENTITIES:
        checks.require(all(r["avatars"][ident]["signature"] == first["avatars"][ident]["signature"] for r in references), "reference avatar comparability mismatch")
        avatars[ident] = {"signature": first["avatars"][ident]["signature"], "bounds": {
            name: bound([r["avatars"][ident]["metrics"][name] for r in references], [historical["avatars"][ident][name]], spec["direction"])
            for name, spec in METRICS.items()}}
    global_bounds = {name: bound([r["global_metrics"][name] for r in references], [historical["global_metrics"][name]], "error") for name in first["global_metrics"]}
    return {"schema": SCHEMA, "status": "FROZEN_REFERENCE_ENVELOPE", "frozen_utc": at,
            "candidate_evaluated": False, "policy_sha256": sha_data(policy), "helper_sha256": checks.sha256(__file__),
            "registry": METRICS, "excluded_fields": EXCLUDED_FIELDS, "references": references, "historical": historical,
            "input_manifest_sha256": first["input_manifest_sha256"], "avatars": avatars, "global_bounds": global_bounds,
            "noise_method": "per metric observed max-minus-min across independently repeated same-input portable3090 reference runs; no epsilon",
            "visual_inspection_status": "NOT_EVALUATED_BY_HELPER", "release_ready": False}


def compare(envelope, candidate, policy, visual_evidence=()):
    validate_policy(policy)
    checks.require(envelope["schema"] == SCHEMA and envelope["registry"] == METRICS
                   and envelope["helper_sha256"] == checks.sha256(__file__) and envelope["policy_sha256"] == sha_data(policy), "frozen policy/helper/registry changed")
    checks.require(candidate["role"] == "candidate" and timestamp(candidate["started_utc"]) > timestamp(envelope["frozen_utc"]),
                   "candidate predates envelope freeze; post-result widening forbidden")
    checks.require(candidate["input_manifest_sha256"] == envelope["input_manifest_sha256"], "candidate inputs differ from frozen reference")
    rows = []
    def check(scope, values, bounds):
        checks.require(set(values) == set(bounds), "unknown/missing candidate metric: " + scope)
        for name, b in bounds.items():
            v = number(values[name], name)
            checks.require(b["direction"] in ("error", "similarity"), "unknown bound direction")
            ok = v <= b["limit"] if b["direction"] == "error" else v >= b["limit"]
            rows.append({"scope": scope, "metric": name, "value": v, "limit": b["limit"],
                         "direction": b["direction"], "status": "PASS" if ok else "FAIL"})
    checks.require(set(candidate["avatars"]) == set(envelope["avatars"]) == set(checks.IDENTITIES), "candidate avatar coverage differs")
    for ident, expected in envelope["avatars"].items():
        checks.require(candidate["avatars"][ident]["signature"] == expected["signature"], "candidate avatar comparability mismatch")
        check(ident, candidate["avatars"][ident]["metrics"], expected["bounds"])
    check("numerical_gates", candidate["global_metrics"], envelope["global_bounds"])
    parity = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    return {"schema": "r5_frozen_quality_comparison_v1", "quality_parity_with_reference": parity,
            "strict_original_gates": candidate["strict_original_gates"], "metrics": rows, "candidate": candidate,
            "visual_evidence": [evidence(p) for p in visual_evidence], "visual_inspection_status": "NOT_EVALUATED_BY_HELPER",
            "decision": "incomplete" if parity == "PASS" else "rejected", "release_ready": False,
            "note": "Numerical parity is not strict-original PASS, visual approval, production-pose approval, or a release decision"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--reference", action="append", required=True, type=Path)
    f.add_argument("--inputs", required=True, type=Path)
    for name in ("fixture-root", "fixture-sidecar", "calibration-root", "calibration-sidecar", "out"):
        f.add_argument("--" + name, required=True, type=Path)
    c = sub.add_parser("compare")
    for name in ("candidate", "inputs", "envelope", "out"):
        c.add_argument("--" + name, required=True, type=Path)
    c.add_argument("--envelope-sha256", required=True)
    c.add_argument("--visual-evidence", action="append", default=[], type=Path)
    a = parser.parse_args(argv)
    checks.require(not a.out.exists(), "refusing to overwrite frozen envelope/comparison")
    try:
        policy = checks.read(a.policy)
        checks.verify_files(checks.read(a.inputs), a.inputs.resolve().parent)
        if a.command == "freeze":
            refs = [load_run(p, a.inputs, "reference") for p in a.reference]
            historical = historical_references(policy, refs[0], a.inputs, a.fixture_root, a.fixture_sidecar, a.calibration_root, a.calibration_sidecar)
            result = freeze(policy, refs, historical)
        else:
            checks.require(checks.sha256(a.envelope) == a.envelope_sha256, "frozen envelope digest mismatch")
            envelope = checks.read(a.envelope)
            result = compare(envelope, load_run(a.candidate, a.inputs, "candidate"), policy, a.visual_evidence)
            result["frozen_envelope"] = evidence(a.envelope)
        result["policy_file"] = evidence(a.policy)
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "INVALID", "error_type": type(exc).__name__, "release_ready": False}
        if isinstance(exc, checks.Invalid):
            result["reason"] = str(exc)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"out": str(a.out), "sha256": checks.sha256(a.out), "status": result.get("status", result.get("quality_parity_with_reference"))}))
    return 2 if result.get("status") == "INVALID" else 1 if result.get("quality_parity_with_reference") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
