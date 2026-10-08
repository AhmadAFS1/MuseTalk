#!/usr/bin/env python3
"""Fail-closed, dependency-free validators for the canonical r5 measurements.

PASS is scoped to the named measurement, never an overall release/visual approval.
FAIL means a valid measurement missed a gate; INVALID means evidence is unusable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

IDENTITIES = (
    "black_man_short_beard", "black_woman", "east_asian_man_goatee",
    "middle_eastern_man_full_beard", "south_asian_woman", "white_man_clean_shaven",
)
EXIT = {"PASS": 0, "FAIL": 1, "INVALID": 2, "HISTORICAL_QUALITY_EXCEPTION": 3}


class Invalid(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise Invalid(reason)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite report value")
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v)


def read(path):
    require(Path(path).is_file(), f"missing report: {path}")
    data = json.loads(Path(path).read_text())
    finite(data)
    return data


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_identity(name, capability, general=False):
    require(bool(name) and bool(capability), "GPU identity unavailable")
    if not general:
        require(name in ("NVIDIA GeForce RTX 3090", "GeForce RTX 3090"), f"expected RTX 3090, got {name}")
        require(str(capability).replace(".", "") == "86", f"expected sm86, got {capability}")
    return {"actual_gpu": name, "compute_capability": capability,
            "comparison_scope": "general_gpu" if general else "rtx3090"}


def verify_files(manifest, base):
    require(manifest.get("schema") == "repro_3090_inputs_v1", "unknown input manifest schema")
    rows = manifest.get("files", [])
    require(bool(rows), "empty input manifest")
    seen = set()
    for row in rows:
        path = (Path(base) / row["path"]).resolve()
        require(str(path) not in seen, f"duplicate input: {path}")
        seen.add(str(path))
        require(path.is_file(), f"missing input: {path}")
        require(sha256(path) == row["sha256"], f"input hash mismatch: {path}")
    return len(rows)


def loaded_backend(data, engine_root, taesd_key, taesd_sha, manifest=None):
    be = data["backends"]
    require(be.get("unet", be.get("unet_name")) == "tensorrt_unet_stagewise", "UNet fallback/mismatch")
    require(be.get("vae_decode", be.get("decoder_name")) == "taesd_trt", "TAESD fallback/mismatch")
    desc = be.get("unet_describe") or {}
    require(isinstance(desc, dict), "missing structured UNet description")
    require(Path(desc.get("engine_dir", "")).resolve() == (Path(engine_root) / "bs16").resolve(), "loaded wrong engine directory")
    require(desc.get("batch") == 16, "wrong UNet batch")
    probe = desc.get("probe_validation") or {}
    require(isinstance(probe, dict), "missing structured UNet probe evidence")
    finite(probe)
    if desc.get("probe_status") == "exact":
        require(probe.get("kind") == "exact" and probe.get("actual_sha256") == probe.get("expected_sha256")
                and bool(probe.get("actual_sha256")), "UNet exact probe evidence absent/mismatched")
    elif str(desc.get("probe_status", "")).startswith("cross_gpu:rel_l2="):
        require(manifest and manifest.get("hardware_compatibility_level") == "ampere_plus", "cross-GPU probe on nonportable engine")
        bound = manifest.get("probe", {}).get("cross_gpu_rel_l2_max")
        require(type(bound) in (int, float) and bound >= 0, "portable probe bound absent")
        require(probe.get("kind") == "cross_gpu" and probe.get("limit") == bound
                and type(probe.get("rel_l2")) in (int, float) and 0 <= probe["rel_l2"] <= bound,
                "portable probe failed frozen bound")
    else:
        raise Invalid("UNet probe absent, disabled, failed, or using unapproved tolerance")
    if manifest:
        require(probe.get("expected_sha256") == manifest["probe"]["output_sha256"], "probe reference differs from frozen engine manifest")
    require(be.get("decoder_trt_key") == taesd_key, "loaded wrong TAESD identity")
    require(be.get("decoder_trt_plan_sha256") == taesd_sha, "loaded wrong TAESD plan hash")


def aggregate(data, stage, target):
    finite(data)
    require(data.get("status") == "complete" and not data.get("error"), "aggregate child incomplete")
    require(data["code_integrity"]["matches_accepted_render_json"] is True, "canonical composition changed")
    args = data["args"]
    require(tuple(args["identity_list"]) == IDENTITIES, "noncanonical six-avatar workload")
    require(args["mode"] == "multi" and args["backend"] == "stagewise16_taesdtrt", "wrong workload/backend")
    require(args["pack"] == 16 and args["decode_split"] == 8, "wrong batch/decode split")
    require(not args["encode"] and not args["save_arrays"], "capture overhead mixed into throughput")
    require(data["summary"]["deterministic_per_identity"] is True, "nondeterministic output")
    thermal = data.get("thermal_warmup", {})
    require(thermal.get("seconds", 0) >= 120, "missing thermal preconditioning")
    tail = thermal.get("last_30s", {})
    require(tail.get("n", 0) >= 40, "insufficient thermal stability samples")
    temperature = tail["temperature.gpu"]
    require(temperature["max"] - temperature["min"] <= 3, "GPU not thermally settled; repeat with longer warmup")
    n, repeats = {"T": (6, 2), "SUST": (6, 5), "N15": (15, 10)}[stage]
    require(args["streams"] == n and len(data["repeats"]) >= repeats, "missing streams/windows")
    windows = []
    for index, row in enumerate(data["repeats"]):
        require(row["repeat"] == index, "repeat ordering is not consecutive")
        wall = row["wall_s"]
        require(wall >= 60, f"window {index} too short: {wall}")
        require(row["gpu"]["smi"]["n"] > 0, "missing aggregate thermal telemetry")
        workers = row["per_worker"]
        require(len(workers) == n, "missing worker reports")
        frames = sum(w["frames"] for w in workers.values())
        require(frames == row["frames"] == n * args["loops"] * 240, "completed frame count mismatch")
        # Never sum independently timed stream FPS. All streams share this wall.
        fps = frames / wall
        require(math.isclose(fps, row["aggregate_fps"], rel_tol=1e-9), "aggregate denominator mismatch")
        for worker in workers.values():
            require(math.isclose(worker["fps"], worker["frames"] / wall, rel_tol=1e-9), "per-stream denominator mismatch")
            require(0 <= worker["done_after_t0_s"] <= wall + 1e-6, "worker timestamp outside shared interval")
        windows.append({"window": index, "completed_frames": frames, "shared_wall_s": wall, "fps": fps})
    passed = target is None or all(w["fps"] >= target for w in windows)
    return {"status": "PASS" if passed else "FAIL", "scope": "aggregate_full_recipe_offline",
            "stage": stage, "target_fps": target, "windows": windows,
            "includes_live_encoding_or_RTP": False}


def gpu_path(data, minimum=180):
    finite(data)
    m = data["measured"]
    require(m["wall_s"] >= minimum, "GPU measurement too short")
    require(data["warmup"]["seconds"] >= 20, "insufficient GPU warmup")
    require(data["golden"].get("finite_checked") is True, "missing golden finite checks")
    require(data["args"]["batch"] == 16 and data["args"]["window_s"] == 10, "wrong GPU batch/window")
    require(data["backends"]["stage_sync"] is False and data["backends"]["decode_sync"] is False,
            "timed GPU path includes stage synchronizations")
    require(data["inputs"]["distinct_batches"] > 0 and len(data["golden"]["per_input_sha256"]) == data["inputs"]["distinct_batches"],
            "incomplete golden input coverage")
    require(m["frames"] == m["batches"] * data["args"]["batch"], "GPU frame count mismatch")
    require(math.isclose(m["aggregate_fps"], m["frames"] / m["wall_s"], rel_tol=1e-9), "GPU denominator mismatch")
    require(len(m["fps_windows"]) >= int(minimum / 10), "missing ten-second GPU windows")
    require(all(w["t_start_s"] == i * 10 and w["fps"] > 0 for i, w in enumerate(m["fps_windows"])),
            "GPU windows missing, reordered, or empty")
    require(data["gpu_smi"]["samples"] > 0, "missing thermal/power telemetry")
    require(data["args"]["live_env"] is False, "legacy live env was loaded")
    return {"status": "PASS", "scope": "gpu_path_not_aggregate", "frames": m["frames"],
            "wall_s": m["wall_s"], "fps": m["aggregate_fps"], "windows": m["fps_windows"]}


def blocks(data, manifests):
    finite(data)
    require(data.get("input_kind") == "synthetic_random_buffers", "unlabelled synthetic block inputs")
    require(data.get("rounds", 0) >= 9, "insufficient block timing rounds")
    require(set(data["sets"]) == set(manifests), "missing comparison engine set")
    for root, manifest in manifests.items():
        required = {x["name"] for x in manifest["spec"] if x["name"] != "prefix"}
        row = data["sets"][root]
        require(set(row["block_ms"]) == required, f"missing chain blocks for {root}")
        require(set(row["round_ms"]) == required, f"missing raw chain block rounds for {root}")
        require(all(len(x) == data["rounds"] for x in row["round_ms"].values()), "missing block rounds")
        for name, values in row["round_ms"].items():
            require(all(value > 0 for value in values), "invalid block duration")
            require(math.isclose(statistics.median(values), row["block_ms"][name], rel_tol=1e-9), "block median mismatch")
        require(math.isclose(sum(row["block_ms"].values()), row["sum_ms"], rel_tol=1e-9), "block sum mismatch")
    return {"status": "PASS", "scope": "synthetic_block_diagnostic", "sets": data["sets"]}


def live(data, expected_streams, min_seconds):
    finite(data)
    rows = data["streams"]
    require(len(rows) == expected_streams, "missing live streams")
    passes = []
    for row in rows:
        require(not row.get("error"), "live stream error")
        require(row["window_s"] >= min_seconds, "live window too short")
        for key in ("server_fresh_fraction", "server_held_run_max", "client_content_1s_min", "pts_join_matched"):
            require(row.get(key) is not None, f"missing live evidence: {key}")
        require(row["pts_join_matched"] >= .99, "invalid client/server trace join")
        passes.append(row["P1_rate"] is True and row["P2_fresh"] is True and row["P3_no_buffering"] is True
                      and row["anchored_1s_min_frames"] >= 20 and row["server_fresh_fraction"] >= .995
                      and row["server_held_run_max"] <= 2 and row["client_content_1s_min"] >= 18
                      and row["gap_max_ms"] <= 120 and row["gaps_over_100ms_per_10min"] <= 1
                      and not row["gap_causes"].get("server_send", 0) and not row["gap_causes"].get("server_late", 0))
    return {"status": "PASS" if all(passes) else "FAIL", "scope": "local_loopback_live_not_EC2_deployment",
            "streams": expected_streams, "per_stream_pass": passes, "summary": data["summary"]}


def child_result(returncode, expected_reports):
    require(returncode == 0, f"child failed with exit {returncode}")
    return [read(path) for path in expected_reports]


def combined_status(results):
    require(bool(results), "no measurements requested")
    states = {x["status"] for x in results}
    for state in ("INVALID", "FAIL", "HISTORICAL_QUALITY_EXCEPTION"):
        if state in states:
            return state
    require(states == {"PASS"}, "unknown measurement status")
    return "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        rows = [read(p) for p in args.reports]
        result = {"schema": "repro_3090_summary_v1", "status": combined_status(rows),
                  "reports": [{"path": p, "sha256": sha256(p), "status": d["status"]} for p, d in zip(args.reports, rows)]}
    except (Invalid, KeyError, ValueError, OSError) as exc:
        result = {"schema": "repro_3090_summary_v1", "status": "INVALID", "reason": str(exc)}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return EXIT[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
