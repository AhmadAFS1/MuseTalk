#!/usr/bin/env python
"""Time the exported Core ML models on this Mac's Neural Engine and check their numerics.

For every model in models/ios_bench/coreml/manifest.json: load with the requested
compute units, warm up, time predictions, and report ms per call and per frame.
With --compare, also time CPU_ONLY and ALL so an ANE fallback shows up as a
missing gap. With --check, compare each UNet variant's output on the saved
16-frame reference set against the fp32 PyTorch reference (latent mean/max abs
error, and decoded-face mean abs error through the fp32 TAESD). --sustain N runs
one model for N seconds and logs fps every 30 s.

Usage (repo root, Python 3.12 venv):
  python scripts/ios/bench_coreml.py --compare --check --out docs/ios_unet_bench/coreml_m5pro.json
  python scripts/ios/bench_coreml.py --only unet_int8w_b4 --sustain 300
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"
UNITS = {"ane": "CPU_AND_NE", "cpu": "CPU_ONLY", "all": "ALL", "gpu": "CPU_AND_GPU"}


def load(path: str, units: str):
    import coremltools as ct

    return ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, units))


def inputs_for(entry: dict, ref: dict | None):
    b = entry["batch"]
    if entry["kind"] == "unet":
        if ref is not None:
            idx = np.arange(b) % ref["latents"].shape[0]
            return {"sample": ref["latents"][idx].astype(np.float16), "audio": ref["audio"][idx].astype(np.float16)}
        rng = np.random.default_rng(0)
        return {"sample": rng.standard_normal((b, 8, 32, 32)).astype(np.float16),
                "audio": (0.5 * rng.standard_normal((b, 50, 384))).astype(np.float16)}
    if ref is not None:
        idx = np.arange(b) % ref["ref_pred_latents"].shape[0]
        return {"latents": ref["ref_pred_latents"][idx].astype(np.float16)}
    rng = np.random.default_rng(0)
    return {"latents": rng.standard_normal((b, 4, 32, 32)).astype(np.float16)}


def time_model(model, feed: dict, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        model.predict(feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def compute_plan_summary(path: str, units: str) -> dict | None:
    """Per-op device placement from Core ML's compute plan (macOS 14.4+)."""
    try:
        import coremltools as ct
        from coremltools.models.compute_plan import MLComputePlan

        compiled = ct.models.utils.compile_model(path) if not path.endswith(".mlmodelc") else path
        plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=getattr(ct.ComputeUnit, units))
        prog = plan.model_structure.program
        fn = prog.functions["main"]
        counts: dict[str, int] = {}
        for op in fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            dev = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            counts[dev] = counts.get(dev, 0) + 1
        return counts
    except Exception as exc:  # the API moved between coremltools releases
        return {"error": f"{type(exc).__name__}: {exc}"}


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def turn_test(manifest: dict, unet_name: str, turns: int, seconds: int, fps: int, ref, units: str) -> dict:
    """Render N back-to-back TTS turns of `seconds` s at `fps`, in chunks of the model's batch.

    This is the real-time question in the product's own terms: for a 10 s turn,
    how long until the first chunk of frames exists, how long until all frames
    exist (must be < 10 s), and does the Neural Engine slow down over a few
    minutes of continuous turns.
    """
    entry = manifest[unet_name]
    b = entry["batch"]
    taesd_entry = manifest.get(f"taesd_fp16_b{b}")
    unet = load(entry["path"], units)
    taesd = load(taesd_entry["path"], units) if taesd_entry else None
    feed = inputs_for(entry, ref)

    def chunk():
        out = unet.predict(feed)
        if taesd is not None:
            taesd.predict({"latents": np.asarray(out["pred_latents"], dtype=np.float16)})

    for _ in range(5):
        chunk()
    frames = fps * seconds
    chunks = -(-frames // b)
    rows = []
    for t in range(turns):
        t0 = time.perf_counter()
        chunk()
        first_ms = (time.perf_counter() - t0) * 1000
        for _ in range(chunks - 1):
            chunk()
        total = time.perf_counter() - t0
        rows.append({"turn": t + 1, "first_chunk_ms": round(first_ms, 1), "render_s": round(total, 2),
                     "realtime_factor": round(seconds / total, 2), "ms_per_frame": round(total / frames * 1000, 2)})
        print(f"  turn {t + 1}/{turns}: first chunk {first_ms:.0f} ms, {frames} frames in {total:.2f} s "
              f"({seconds / total:.2f}x real time)", flush=True)
    first, last = rows[0]["ms_per_frame"], rows[-1]["ms_per_frame"]
    return {"model": unet_name, "with_taesd": taesd is not None, "batch": b, "fps": fps, "turn_seconds": seconds,
            "frames_per_turn": frames, "chunks_per_turn": chunks, "turns": rows,
            "drift_pct_first_to_last": round((last - first) / first * 100, 1),
            "worst_realtime_factor": min(r["realtime_factor"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--units", default="ane", choices=list(UNITS))
    ap.add_argument("--compare", action="store_true", help="also time cpu and all")
    ap.add_argument("--check", action="store_true", help="numeric check against the PyTorch reference")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--sustain", type=int, default=0, help="seconds of sustained loop for the --only model")
    ap.add_argument("--turn-model", default=None, help="run the back-to-back TTS-turn test on this UNet model and exit")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--turn-seconds", type=int, default=10)
    ap.add_argument("--turn-fps", type=int, default=20)
    ap.add_argument("--ref", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "bench_inputs.pt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "coreml_bench.json")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    names = [n for n in manifest if "error" not in manifest[n]]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    ref = None
    if args.ref.exists():
        import torch

        d = torch.load(args.ref, map_location="cpu", weights_only=True)
        ref = {k: v.float().numpy() for k, v in d.items() if hasattr(v, "numpy")}

    host = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "macos": platform.mac_ver()[0]}
    if args.turn_model:
        res = turn_test(manifest, args.turn_model, args.turns, args.turn_seconds, args.turn_fps, ref, UNITS[args.units])
        out = {"host": host, "units": UNITS[args.units], "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "turn_test": res}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out}")
        return
    results = []
    for name in names:
        entry = manifest[name]
        feed = inputs_for(entry, ref)
        rec = {"name": name, "kind": entry["kind"], "variant": entry["variant"], "batch": entry["batch"],
               "bytes": entry["bytes"], "mb": round(entry["bytes"] / 2**20)}
        unit_list = [args.units] + (["cpu", "all"] if args.compare else [])
        for u in unit_list:
            units = UNITS[u]
            t0 = time.time()
            model = load(entry["path"], units)
            load_s = time.time() - t0
            iters = args.iters if u == args.units else max(5, args.iters // 5)
            times = time_model(model, feed, args.warmup if u == args.units else 3, iters)
            med = statistics.median(times)
            rec[f"{u}_ms_per_call"] = round(med, 2)
            rec[f"{u}_ms_per_frame"] = round(med / entry["batch"], 3)
            rec[f"{u}_p90_ms_per_frame"] = round(sorted(times)[int(0.9 * (len(times) - 1))] / entry["batch"], 3)
            rec[f"{u}_load_s"] = round(load_s, 1)
            if u == args.units:
                rec["rss_mb_after"] = round(rss_mb())
                rec["weight_bytes_per_frame_mb"] = round(entry["bytes"] / entry["batch"] / 2**20, 1)
                if entry["kind"] == "unet":
                    rec["fps_single_stream_unet_only"] = round(1000 / (med / entry["batch"]), 1)
                rec["compute_plan_ops_by_device"] = compute_plan_summary(entry["path"], units)
            if args.check and ref is not None and u == args.units:
                if entry["kind"] == "unet":
                    outs = []
                    for start in range(0, 16, entry["batch"]):
                        idx = np.arange(start, start + entry["batch"]) % 16
                        out = model.predict({"sample": ref["latents"][idx].astype(np.float16),
                                             "audio": ref["audio"][idx].astype(np.float16)})["pred_latents"]
                        outs.append(np.asarray(out, dtype=np.float32))
                    pred = np.concatenate(outs)[:16]
                    diff = np.abs(pred - ref["ref_pred_latents"][:16])
                    rec["latent_mae"] = float(diff.mean())
                    rec["latent_max_abs"] = float(diff.max())
                    # decoded-face error through the fp32 TAESD, in 0..255 units
                    import torch
                    from diffusers import AutoencoderTiny

                    taesd = AutoencoderTiny.from_pretrained(str(REPO_ROOT / "models" / "taesd"), torch_dtype=torch.float32).eval()
                    with torch.inference_mode():
                        faces = (taesd.decode(torch.from_numpy(pred)).sample / 2 + 0.5).clamp(0, 1).numpy()
                    rec["face_mae_255"] = float(np.abs(faces - ref["ref_faces"][:16]).mean() * 255)
                    rec["face_max_abs_255"] = float(np.abs(faces - ref["ref_faces"][:16]).max() * 255)
                else:
                    outs = []
                    for start in range(0, 16, entry["batch"]):
                        idx = np.arange(start, start + entry["batch"]) % 16
                        out = model.predict({"latents": ref["ref_pred_latents"][idx].astype(np.float16)})["faces"]
                        outs.append(np.asarray(out, dtype=np.float32))
                    faces = np.concatenate(outs)[:16]
                    rec["face_mae_255"] = float(np.abs(faces - ref["ref_faces"][:16]).mean() * 255)
                    rec["face_max_abs_255"] = float(np.abs(faces - ref["ref_faces"][:16]).max() * 255)
            if args.sustain and u == args.units:
                log, t_start, n, last = [], time.time(), 0, time.time()
                while time.time() - t_start < args.sustain:
                    model.predict(feed)
                    n += 1
                    if time.time() - last >= 30:
                        fps = n * entry["batch"] / (time.time() - last)
                        log.append({"t_s": round(time.time() - t_start), "fps": round(fps, 1)})
                        print(f"  sustain {log[-1]}", flush=True)
                        n, last = 0, time.time()
                rec["sustain"] = log
            del model
        print(json.dumps(rec), flush=True)
        results.append(rec)

    out = {"host": host, "units": UNITS[args.units], "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.only:
        prev = json.loads(args.out.read_text())
        keep = [r for r in prev.get("results", []) if r["name"] not in {r2["name"] for r2 in results}]
        out["results"] = keep + results
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
