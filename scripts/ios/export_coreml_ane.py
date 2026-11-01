#!/usr/bin/env python
"""Export the MuseTalk UNet through Apple's Neural-Engine-friendly UNet implementation.

`export_coreml.py` converts the diffusers UNet as it is: linear layers on
[B, S, C] tensors, attention as matmul + softmax over [B*heads, S, S], which the
compute plan shows spending ~16% of its cost on reshapes, transposes and splits,
and which the ANE compiler rejects at batch > 1. Apple's ml-stable-diffusion
re-implements the same UNet in the layout the ANE wants (Conv2d on [B, C, 1, S],
LayerNormANE, "split-einsum" attention) and loads diffusers weights through
state-dict hooks. This script builds that model from MuseTalk's config and
weights, checks it against the diffusers UNet in fp32, and exports it.

Needs a clone of https://github.com/apple/ml-stable-diffusion (only its
python_coreml_stable_diffusion package is imported; nothing is installed).

Usage (repo root, .venv-ct):
  python scripts/ios/export_coreml_ane.py --apple-repo /path/to/ml-stable-diffusion --batches 1,4 --variants fp16,int8w
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from export_coreml import OUT_DIR, convert, quantize_weights  # noqa: E402
from musetalk.models.unet import UNet  # noqa: E402


class UNetStepANE(torch.nn.Module):
    """sample [B,8,32,32] + audio [B,50,384] -> pred latent [B,4,32,32], Apple layout inside."""

    def __init__(self, ane_unet, pe):
        super().__init__()
        self.unet = ane_unet
        self.register_buffer("pe_table", pe.pe[:, :50, :].clone())
        self.register_buffer("timestep", torch.zeros(1, dtype=torch.long))

    def forward(self, sample, audio):
        hidden = (audio + self.pe_table).transpose(1, 2).unsqueeze(2)  # [B, 384, 1, 50]
        out = self.unet(sample, self.timestep, hidden)
        return out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, (tuple, list)) else out)


class GroupNormHier(torch.nn.Module):
    """GroupNorm with the statistics taken as three short reductions (W, then H, then
    the group's channels) instead of one flat mean over ~10k elements.

    The flat reduction is what coremltools emits for nn.GroupNorm, and on the Neural
    Engine it runs in fp16; a 10k-element fp16 sum loses the low bits of the mean and
    variance. Chained short reductions keep each partial sum small. Same math in fp32.
    """

    def __init__(self, gn: torch.nn.GroupNorm):
        super().__init__()
        self.groups, self.eps = gn.num_groups, gn.eps
        self.weight, self.bias = gn.weight, gn.bias

    @staticmethod
    def _mean(x):
        return x.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)

    def forward(self, x):
        b, c, h, w = x.shape
        xg = x.reshape(b, self.groups, c // self.groups, h, w)
        d = xg - self._mean(xg)
        y = d * torch.rsqrt(self._mean(d * d) + self.eps)
        return y.reshape(b, c, h, w) * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def swap_groupnorm(module: torch.nn.Module) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GroupNorm):
            setattr(module, name, GroupNormHier(child))
            n += 1
        else:
            n += swap_groupnorm(child)
    return n


def build_ane_unet(apple_repo: Path, unet_dir: Path, attention: str, groupnorm: str = "torch"):
    sys.path.insert(0, str(apple_repo))
    from python_coreml_stable_diffusion import unet as ane

    ane.ATTENTION_IMPLEMENTATION_IN_EFFECT = ane.AttentionImplementations[attention]
    config = {k: v for k, v in json.loads((unet_dir / "musetalk.json").read_text()).items() if not k.startswith("_")}
    model = ane.UNet2DConditionModel(**config).eval()
    wrap = UNet(unet_config=str(unet_dir / "musetalk.json"), model_path=str(unet_dir / "unet.pth"),
                device=torch.device("cpu"))
    ref = wrap.model.eval().requires_grad_(False)
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    print(f"  state dict: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    if missing:
        print(f"    missing (first 10): {missing[:10]}", flush=True)
    if unexpected:
        print(f"    unexpected (first 10): {unexpected[:10]}", flush=True)
    if groupnorm == "hier":
        print(f"  swapped {swap_groupnorm(model)} GroupNorm modules for GroupNormHier", flush=True)
    return model.requires_grad_(False), ref, wrap.pe


def check_vs_diffusers(step_ane, ref_unet, pe, bench_inputs: Path, n: int = 8) -> float:
    d = torch.load(bench_inputs, map_location="cpu", weights_only=True)
    lat, aud = d["latents"][:n].float(), d["audio"][:n].float()
    with torch.no_grad():
        want = ref_unet(lat, torch.zeros(1, dtype=torch.long), encoder_hidden_states=pe(aud)).sample
        got = step_ane(lat, aud)
    err = (want - got).abs()
    print(f"  fp32 torch, Apple layout vs diffusers on {n} real frames: max |diff| {err.max():.2e}, "
          f"mean {err.mean():.2e} (ref |x| mean {want.abs().mean():.3f})", flush=True)
    return float(err.max())


def main() -> None:
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--apple-repo", type=Path, required=True, help="clone of apple/ml-stable-diffusion")
    ap.add_argument("--attention", default="SPLIT_EINSUM", choices=["ORIGINAL", "SPLIT_EINSUM", "SPLIT_EINSUM_V2"])
    ap.add_argument("--groupnorm", default="torch", choices=["torch", "hier"],
                    help="hier: GroupNorm statistics as chained short reductions (fp16 accuracy on the ANE)")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--variants", default="fp16,int8w")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--bench-inputs", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "bench_inputs.pt")
    ap.add_argument("--target", default="iOS17", choices=["iOS17", "iOS18"])
    ap.add_argument("--exporter", default="trace", choices=["trace", "export"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    target = getattr(ct.target, args.target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tag = {"ORIGINAL": "aneorig", "SPLIT_EINSUM": "ane", "SPLIT_EINSUM_V2": "anev2"}[args.attention]
    if args.groupnorm == "hier":
        tag += "gn"

    def record(name, path, extra):
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        manifest[name] = {"path": str(path), "bytes": size, **extra}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  saved {path} ({size / 2**20:.0f} MB)", flush=True)

    print(f"building Apple-layout UNet ({args.attention}) from MuseTalk weights", flush=True)
    ane_unet, ref_unet, pe = build_ane_unet(args.apple_repo, args.unet_dir, args.attention, args.groupnorm)
    step = UNetStepANE(ane_unet, pe).eval()
    max_err = check_vs_diffusers(step, ref_unet, pe, args.bench_inputs)
    if max_err > 1e-2:
        raise SystemExit("Apple-layout UNet does not reproduce the diffusers UNet; stop here")
    del ref_unet

    for b in [int(x) for x in args.batches.split(",")]:
        print(f"UNet ({tag}) batch {b}: exporting ({args.exporter})", flush=True)
        ex_sample, ex_audio = torch.randn(b, 8, 32, 32), torch.randn(b, 50, 384)
        t0 = time.time()
        with torch.no_grad():
            if args.exporter == "trace":
                traced = torch.jit.trace(step, (ex_sample, ex_audio), check_trace=False)
            else:
                traced = torch.export.export(step, (ex_sample, ex_audio), strict=False).run_decompositions({})
        print(f"  traced in {time.time() - t0:.0f}s", flush=True)
        base = convert(traced, [ct.TensorType(name="sample", shape=(b, 8, 32, 32), dtype=np.float16),
                                ct.TensorType(name="audio", shape=(b, 50, 384), dtype=np.float16)],
                       "pred_latents", target)
        extra = {"kind": "unet", "batch": b, "attention": args.attention, "layout": "apple-ane",
                 "groupnorm": args.groupnorm}
        if "fp16" in args.variants:
            path = args.out_dir / f"unet_{tag}_fp16_b{b}.mlpackage"
            base.save(str(path))
            record(f"unet_{tag}_fp16_b{b}", path, {**extra, "variant": f"{tag}_fp16"})
        if "int8w" in args.variants:
            w8 = quantize_weights(base)
            path = args.out_dir / f"unet_{tag}_int8w_b{b}.mlpackage"
            w8.save(str(path))
            record(f"unet_{tag}_int8w_b{b}", path, {**extra, "variant": f"{tag}_int8w"})
            del w8
        del base


if __name__ == "__main__":
    main()
