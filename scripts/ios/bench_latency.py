#!/usr/bin/env python
"""Latency view of the Core ML build: what a streaming client would feel.

Three numbers the throughput benches do not give:

  load      - compile the .mlpackage once (what Xcode does at build time), then
              time loading the compiled model the way an app does at launch,
              and the first prediction after load (Neural Engine warm-up).
  jitter    - per-frame latency distribution of UNet + TAESD at batch 1 over a
              long run of real inputs; p99 and the count of frames over the
              25 fps (40 ms) and 20 fps (50 ms) budgets decide smoothness,
              not the mean.
  blend     - CPU cost of pasting the 256x256 face back into the 9:16 frame
              (cv2.resize + musetalk.utils.blending.get_image_blending), which
              runs on the CPU next to the Neural Engine in a real pipeline.

Usage (repo root, .venv-ct):
  python scripts/ios/bench_latency.py --unet unet_int8w_b1 --avatar <avatar_dir> --frames 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from render_evidence import load_avatar, to_bgr_u8  # noqa: E402
from musetalk.utils.blending import get_image_blending  # noqa: E402

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"


def pct(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p90_ms": round(float(np.percentile(a, 90)), 2), "p99_ms": round(float(np.percentile(a, 99)), 2),
            "max_ms": round(float(a.max()), 2), "n": len(xs)}


def bench_load(path: str, units, loads: int) -> tuple[dict, object]:
    import coremltools as ct
    from coremltools.models.utils import compile_model

    t0 = time.perf_counter()
    compiled = compile_model(path)
    compile_s = time.perf_counter() - t0
    load_s, first_ms = [], []
    model = None
    for _ in range(loads):
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(str(compiled), compute_units=units)
        load_s.append(time.perf_counter() - t0)
    return {"compile_s": round(compile_s, 2), "load_s": [round(x, 2) for x in load_s]}, model


def main() -> None:
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="unet_int8w_b1")
    ap.add_argument("--avatar", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--blend-frames", type=int, default=150)
    ap.add_argument("--loads", type=int, default=3)
    ap.add_argument("--bench-inputs", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "bench_inputs.pt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "latency_m5pro.json")
    args = ap.parse_args()

    import torch

    manifest = json.loads(MANIFEST.read_text())
    units = ct.ComputeUnit.CPU_AND_NE
    ref = {k: v.numpy() for k, v in torch.load(args.bench_inputs, map_location="cpu", weights_only=True).items()
           if hasattr(v, "numpy")}
    n_ref = ref["latents"].shape[0]
    result = {"unet": args.unet, "compute_units": "CPU_AND_NE"}

    # --- load -----------------------------------------------------------------
    print(f"loading {args.unet} (compile once, then {args.loads} loads)", flush=True)
    load_info, unet = bench_load(manifest[args.unet]["path"], units, args.loads)
    feed = {"sample": ref["latents"][:1].astype(np.float16), "audio": ref["audio"][:1].astype(np.float16)}
    t0 = time.perf_counter()
    out = unet.predict(feed)
    load_info["first_predict_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    unet.predict(feed)
    load_info["second_predict_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["unet_load"] = load_info
    print(f"  compile {load_info['compile_s']} s, loads {load_info['load_s']} s, "
          f"first predict {load_info['first_predict_ms']} ms, second {load_info['second_predict_ms']} ms", flush=True)

    taesd_info, taesd = bench_load(manifest["taesd_fp16_b1"]["path"], units, args.loads)
    lat_feed = {"latents": np.asarray(out["pred_latents"], dtype=np.float16)}
    t0 = time.perf_counter()
    faces = taesd.predict(lat_feed)["faces"]
    taesd_info["first_predict_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["taesd_load"] = taesd_info
    print(f"  taesd compile {taesd_info['compile_s']} s, loads {taesd_info['load_s']} s, "
          f"first predict {taesd_info['first_predict_ms']} ms", flush=True)

    # --- jitter ---------------------------------------------------------------
    print(f"per-frame latency over {args.frames} frames (UNet + TAESD, batch 1)", flush=True)
    unet_ms, taesd_ms, frame_ms = [], [], []
    for i in range(args.frames):
        j = i % n_ref
        feed = {"sample": ref["latents"][j:j + 1].astype(np.float16), "audio": ref["audio"][j:j + 1].astype(np.float16)}
        t0 = time.perf_counter()
        pred = unet.predict(feed)["pred_latents"]
        t1 = time.perf_counter()
        taesd.predict({"latents": np.asarray(pred, dtype=np.float16)})
        t2 = time.perf_counter()
        unet_ms.append((t1 - t0) * 1000)
        taesd_ms.append((t2 - t1) * 1000)
        frame_ms.append((t2 - t0) * 1000)
    fm = np.asarray(frame_ms)
    result["jitter"] = {"unet": pct(unet_ms), "taesd": pct(taesd_ms), "frame": pct(frame_ms),
                        "frames_over_40ms": int((fm > 40).sum()), "frames_over_50ms": int((fm > 50).sum()),
                        "first_frame_ms": round(float(fm[0]), 2)}
    print(f"  frame: {result['jitter']['frame']}  >40ms: {result['jitter']['frames_over_40ms']}  "
          f">50ms: {result['jitter']['frames_over_50ms']}", flush=True)

    # --- blend ----------------------------------------------------------------
    frames, masks, coords, mask_coords, _ = load_avatar(args.avatar)
    face_u8 = to_bgr_u8(np.asarray(faces, dtype=np.float32))[0]  # [256,256,3] BGR
    h, w = frames[0].shape[:2]
    print(f"blend cost on {len(frames)} avatar frames of {w}x{h}", flush=True)
    resize_ms, blend_ms = [], []
    for i in range(args.blend_frames):
        j = i % len(frames)
        x1, y1, x2, y2 = coords[j]
        t0 = time.perf_counter()
        face = cv2.resize(face_u8, (x2 - x1, y2 - y1))
        t1 = time.perf_counter()
        get_image_blending(frames[j].copy(), face, coords[j], masks[j], mask_coords[j])
        t2 = time.perf_counter()
        resize_ms.append((t1 - t0) * 1000)
        blend_ms.append((t2 - t1) * 1000)
    result["blend"] = {"frame_size": [w, h], "face_box_px": [int(x2 - x1), int(y2 - y1)],
                       "resize": pct(resize_ms), "blend": pct(blend_ms)}
    print(f"  resize {result['blend']['resize']['mean_ms']} ms, blend {result['blend']['blend']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
