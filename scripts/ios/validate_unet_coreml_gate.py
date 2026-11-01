#!/usr/bin/env python
"""Run the repo's UNet quality gate on a Core ML UNet.

`scripts/validate_unet_backend.py` is what the TensorRT engines on the servers
are gated with: production captures of real UNet inputs with the fp16 PyTorch
outputs (`calibration/unet_portable_bs8/*.pt`), and per-file thresholds
mae <= 0.01 and max_abs <= 0.5 on the predicted latents. This runs the same
captures and statistics through a Core ML model on the Neural Engine, so the
device build and the server engines are judged on identical inputs.

Usage (repo root, .venv-ct):
  python scripts/ios/validate_unet_coreml_gate.py --model unet_int8w_b1 [--pytorch-fp32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_unet_backend import _load_capture_group, discover_captures, summarize, tensor_stats  # noqa: E402

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"
GATE = {"mae": 0.01, "max_abs": 0.5}


def coreml_runner(model_name: str, units: str, pe_table: torch.Tensor):
    import coremltools as ct

    entry = json.loads(MANIFEST.read_text())[model_name]
    model = ct.models.MLModel(entry["path"], compute_units=getattr(ct.ComputeUnit, units))
    b = entry["batch"]

    def run(lat: torch.Tensor, aud: torch.Tensor) -> torch.Tensor:
        # The captures hold the audio features as the server's UNet saw them, i.e. with the
        # positional encoding already added; the Core ML step adds it inside the graph.
        aud = aud - pe_table
        outs = []
        for s in range(0, lat.shape[0], b):
            l, a = lat[s:s + b], aud[s:s + b]
            pad = b - l.shape[0]
            if pad:  # pad the last chunk up to the model's static batch
                l = torch.cat([l, l[-1:].expand(pad, -1, -1, -1)])
                a = torch.cat([a, a[-1:].expand(pad, -1, -1)])
            out = model.predict({"sample": l.numpy().astype(np.float16), "audio": a.numpy().astype(np.float16)})
            outs.append(torch.from_numpy(np.asarray(out["pred_latents"], dtype=np.float32))[:l.shape[0] - pad])
        return torch.cat(outs)

    return run, b


def pytorch_runner(unet_dir: Path):
    from musetalk.models.unet import UNet

    wrap = UNet(unet_config=str(unet_dir / "musetalk.json"), model_path=str(unet_dir / "unet.pth"),
                device=torch.device("cpu"))
    unet = wrap.model.eval()
    ts = torch.zeros(1, dtype=torch.long)

    def run(lat, aud):  # captured audio already carries the positional encoding
        with torch.inference_mode():
            return unet(lat, ts, encoder_hidden_states=aud).sample.float()

    return run, 8


def pe_table(unet_dir: Path) -> torch.Tensor:
    from musetalk.models.unet import PositionalEncoding

    return PositionalEncoding(d_model=384).pe[:, :50, :].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_int8w_b1", help="manifest name, or 'pytorch_fp32' for the CPU anchor")
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--capture-dir", type=Path, default=REPO_ROOT / "calibration" / "unet_portable_bs8")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.model == "pytorch_fp32":
        run, b = pytorch_runner(args.unet_dir)
    else:
        run, b = coreml_runner(args.model, args.units, pe_table(args.unet_dir))
    paths = discover_captures(args.capture_dir, args.limit)
    print(f"{args.model}: {len(paths)} capture files from {args.capture_dir} (reference = fp16 PyTorch production outputs)",
          flush=True)

    rows = []
    t_total, n_frames = 0.0, 0
    for path in paths:
        payload = _load_capture_group([path])
        lat = payload["latent_batch"].float()
        aud = payload["audio_feature_batch"].float()
        ref = payload["pred_latents"].float()
        n = int(payload.get("actual_batch", lat.shape[0]))
        lat, aud, ref = lat[:n], aud[:n], ref[:n]
        t0 = time.perf_counter()
        cand = run(lat, aud)
        t_total += time.perf_counter() - t0
        n_frames += n
        stats = tensor_stats(cand, ref)
        rows.append({"path": path.name, "frames": n, "avatar": payload["items"][0].get("avatar_id") if payload.get("items") else None,
                     **stats, "benchmark": None})
        flag = "" if stats["mae"] <= GATE["mae"] and stats["max_abs"] <= GATE["max_abs"] else "  <-- over gate"
        print(f"  {path.name}: mae {stats['mae']:.5f}  p95 {stats['p95_abs']:.4f}  max_abs {stats['max_abs']:.4f}{flag}", flush=True)

    summary = summarize(rows)
    passed = summary["mae_max"] <= GATE["mae"] and summary["max_abs_max"] <= GATE["max_abs"]
    summary["gate"] = GATE
    summary["pass"] = passed
    summary["ms_per_frame_incl_io"] = round(t_total / n_frames * 1000, 2)
    print(f"\nsummary: mae mean {summary['mae_mean']:.5f} / max {summary['mae_max']:.5f}; "
          f"max_abs mean {summary['max_abs_mean']:.4f} / max {summary['max_abs_max']:.4f}; "
          f"p95 max {summary['p95_abs_max']:.4f}; {n_frames} frames; "
          f"{'PASS' if passed else 'FAIL'} (gate mae<=0.01, max_abs<=0.5 per file)", flush=True)

    out = args.out or (REPO_ROOT / "docs" / "ios_unet_bench" / f"gate_{args.model}_m5pro.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "units": args.units, "capture_dir": str(args.capture_dir),
                               "summary": summary, "files": rows}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
