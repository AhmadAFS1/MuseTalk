#!/usr/bin/env python
"""Precision attribution: which op classes make the Neural Engine's fp16 UNet inaccurate?

On the repo's UNet gate (production captures, fp16 PyTorch reference) the same
fp16 Core ML model scores mae 0.0004 on the GPU and 0.013 on the Neural Engine,
so the error is the ANE's fp16 arithmetic, not the conversion. Core ML can keep
chosen ops in fp32 (they then run on the CPU/GPU, splitting the graph), which
makes a clean experiment: export the fp16 UNet with one op class at a time held
in fp32, gate each on CPU_AND_NE, and see which class recovers the accuracy.
Speed of these hybrids is secondary; the point is to learn what to reformulate
so it can stay on the ANE.

Variants (op types held in fp32):
  gn32    reduce_mean, sub, square, sqrt, real_div  (GroupNorm statistics and normalisation)
  sm32    softmax                                   (attention probabilities)
  mm32    matmul, softmax                           (attention QK^T, softmax, attn.V)
  gnsm32  gn32 + sm32
  all32   everything (fp32 model; cannot use the ANE; the accuracy ceiling)

Usage (repo root, .venv-ct):
  python scripts/ios/export_coreml_precision.py --variants gn32,sm32,mm32,gnsm32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from export_coreml import OUT_DIR, UNetStep  # noqa: E402
from musetalk.models.unet import UNet  # noqa: E402

FP32_OPS = {
    "gn32": {"reduce_mean", "sub", "square", "sqrt", "real_div"},
    "sm32": {"softmax"},
    "mm32": {"matmul", "softmax"},
    "gnsm32": {"reduce_mean", "sub", "square", "sqrt", "real_div", "softmax"},
}


def main() -> None:
    import coremltools as ct
    from diffusers.models.attention_processor import AttnProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="gn32,sm32,mm32,gnsm32")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    wrap = UNet(unet_config=str(args.unet_dir / "musetalk.json"), model_path=str(args.unet_dir / "unet.pth"),
                device=torch.device("cpu"))
    unet = wrap.model.eval().requires_grad_(False)
    unet.set_attn_processor(AttnProcessor())
    step = UNetStep(unet, wrap.pe).eval()
    b = args.batch
    with torch.no_grad():
        traced = torch.export.export(step, (torch.randn(b, 8, 32, 32), torch.randn(b, 50, 384)), strict=False)
        traced = traced.run_decompositions({})

    WEIGHTED = {"conv", "linear", "matmul"}
    regions: dict[str, set[str]] = {}
    if any(v.startswith("region_") for v in args.variants.split(",")):
        # Depth attribution: hold the weighted ops (conv / linear / matmul) of one third of the
        # graph in fp32. The op names are fixed by the trace, so convert once to read them in order.
        probe = ct.convert(traced,
                           inputs=[ct.TensorType(name="sample", shape=(b, 8, 32, 32), dtype=np.float16),
                                   ct.TensorType(name="audio", shape=(b, 50, 384), dtype=np.float16)],
                           outputs=[ct.TensorType(name="pred_latents")], convert_to="mlprogram",
                           minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16,
                           skip_model_load=True)
        names = [op.name.replace("_cast_fp16", "") for op in probe._mil_program.functions["main"].operations
                 if op.op_type in WEIGHTED]
        n = len(names)
        regions = {"region_early": set(names[:n // 3]), "region_mid": set(names[n // 3:2 * n // 3]),
                   "region_late": set(names[2 * n // 3:])}
        print(f"{n} weighted ops; thirds of {n // 3}: early ends at {names[n // 3 - 1]}, mid ends at {names[2 * n // 3 - 1]}",
              flush=True)
        del probe

    def acc_len(op) -> int:
        """Multiply-accumulate length of a weighted op: Cin*kH*kW for conv, Cin for linear."""
        try:
            if op.op_type == "conv":
                s = op.weight.shape  # [Cout, Cin/groups, kH, kW]
                return int(s[1] * s[2] * s[3])
            if op.op_type == "linear":
                return int(op.weight.shape[1])
        except Exception:
            pass
        return 0

    for variant in args.variants.split(","):
        if variant == "all32":
            precision = ct.precision.FLOAT32
        elif variant.startswith("acc"):
            # Hold every conv/linear whose accumulation is at least N terms in fp32: a direct
            # test of "the ANE's error grows with the length of the fp16 accumulation".
            thresh = int(variant[3:].replace("k", "000"))
            precision = ct.transform.FP16ComputePrecision(
                op_selector=lambda op, thresh=thresh: acc_len(op) < thresh)
        elif variant in regions:
            keep_names = regions[variant]
            precision = ct.transform.FP16ComputePrecision(
                op_selector=lambda op, keep_names=keep_names: not (op.op_type in WEIGHTED and op.name in keep_names))
        else:
            keep = FP32_OPS[variant]
            precision = ct.transform.FP16ComputePrecision(op_selector=lambda op, keep=keep: op.op_type not in keep)
        print(f"{variant}: converting (fp32 ops: "
              f"{sorted(FP32_OPS[variant]) if variant in FP32_OPS else variant})", flush=True)
        model = ct.convert(traced,
                           inputs=[ct.TensorType(name="sample", shape=(b, 8, 32, 32), dtype=np.float16),
                                   ct.TensorType(name="audio", shape=(b, 50, 384), dtype=np.float16)],
                           outputs=[ct.TensorType(name="pred_latents")], convert_to="mlprogram",
                           minimum_deployment_target=ct.target.iOS17, compute_precision=precision)
        name = f"unet_{variant}_b{b}"
        path = args.out_dir / f"{name}.mlpackage"
        model.save(str(path))
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        manifest = json.loads(manifest_path.read_text())  # other exports may have written meanwhile
        if variant in FP32_OPS:
            fp32_ops = sorted(FP32_OPS[variant])
        elif variant in regions:
            fp32_ops = f"{len(regions[variant])} weighted ops ({variant})"
        else:
            fp32_ops = variant
        manifest[name] = {"path": str(path), "bytes": size, "kind": "unet", "variant": variant, "batch": b,
                          "fp32_ops": fp32_ops}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  saved {path} ({size / 2**20:.0f} MB)", flush=True)
        del model


if __name__ == "__main__":
    main()
