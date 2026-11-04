#!/usr/bin/env python
"""How much smaller does the MuseTalk UNet have to be for 25 fps on an iPhone 12 Pro? A speed sweep before any training.

The A14's Neural Engine is compute-bound at ~53 ms per frame for the current UNet (ios_coreml_implementation.md 3.5,
5.6), and 25 fps needs ~40 ms for the whole frame. This exports smaller UNets *without training*, so only their speed
means anything, and times them next to the full model.

The shapes are BK-SDM's (Kim et al., 2023, "BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion"),
which removes whole blocks of the same SD 1.x UNet layout MuseTalk uses and recovers quality by distilling from the
full model, starting from the full model's weights for every block it keeps:
  bk_base   one resnet (+ attention) per down stage instead of two, two per up stage instead of three
  bk_small  bk_base without the mid block
  bk_tiny   bk_small without the innermost (4x4, 1280-channel) stage
Kept blocks get the full model's weights wherever the shapes match (BK-SDM's initialisation); the rest stay random.

Plain layout (diffusers, explicit attention), int8 weights, fp16 compute: the same export as
`export_coreml.py`'s unet_int8w_b1, which is the full model's reference here, so the ratio between them carries to
the Apple-layout chunks the phone runs. `build` is also what `distill_small_unet.py` trains: there it uses SDPA
attention (processor="train"), which trains faster and in less memory but is not what Core ML converts.

A trained student (`--weights`, a state_dict from distill_small_unet.py) is exported as unet_<variant>_<tag>_int8w_b1,
always re-exported and never over the untrained unet_<variant>_int8w_b1 the speed sweep timed.

Usage (repo root, .venv-ct):
  python scripts/ios/export_small_unets.py --variants bk_base,bk_small,bk_tiny --time
  python scripts/ios/export_small_unets.py --variants bk_tiny --weights <run>/bk_tiny_best.pt --tag r1 --time
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from export_coreml import OUT_DIR, UNetStep, convert, quantize_weights  # noqa: E402
from musetalk.models.unet import PositionalEncoding  # noqa: E402

VARIANTS = {
    "full": {},
    "bk_base": {"layers_per_block": 1},
    "bk_small": {"layers_per_block": 1, "mid_block_type": None},
    "bk_tiny": {"layers_per_block": 1, "mid_block_type": None, "block_out_channels": [320, 640, 1280],
                "down_block_types": ["CrossAttnDownBlock2D"] * 3, "up_block_types": ["CrossAttnUpBlock2D"] * 3},
}


def variant_config(variant: str, unet_dir: Path) -> dict:
    """musetalk.json with the variant's changes: the config the stock musetalk.models.unet.UNet wrapper builds it from."""
    config = json.loads((unet_dir / "musetalk.json").read_text())
    config.update(VARIANTS[variant])
    return config


def load_full(unet_dir: Path):
    """The full UNet, loaded strictly from unet.pth as the stock wrapper does. The weights are memory-mapped into a model
    built on the meta device: no random init and no second 3.4 GB copy, which matters on a Mac that is already swapping."""
    from diffusers import UNet2DConditionModel

    with torch.device("meta"):
        model = UNet2DConditionModel(**variant_config("full", unet_dir))
    state = torch.load(unet_dir / "unet.pth", map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, assign=True)
    return model.eval().requires_grad_(False)


def build(variant: str, unet_dir: Path, processor: str = "export", full=None, weights: Path | None = None):
    """UNetStep around `variant`, from a trained state_dict (`weights`, loaded strictly) or else BK-SDM's
    initialisation from the full model (`full`, loaded if not given). processor "export": explicit attention, the
    graph Core ML converts; "train": SDPA (AttnProcessor2_0)."""
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

    if processor not in ("export", "train"):
        raise ValueError(f"processor must be export or train, not {processor!r}")
    if weights is None and full is None:
        full = load_full(unet_dir)
    if weights is None and variant == "full":
        model, source = full, "the full model"
    else:
        model = UNet2DConditionModel(**variant_config(variant, unet_dir)).eval().requires_grad_(False)
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            source = f"trained, {weights}"
        else:
            want, have = model.state_dict(), full.state_dict()
            # a variant without the innermost stage: its up block i is the full model's up block i + shift
            shift = len(full.up_blocks) - len(model.up_blocks)
            if shift:
                moved = {}
                for k, v in have.items():
                    m = re.match(r"up_blocks\.(\d+)\.(.*)", k)
                    if m:
                        if int(m.group(1)) < shift:
                            continue
                        k = f"up_blocks.{int(m.group(1)) - shift}.{m.group(2)}"
                    moved[k] = v
                have = moved
            keep = {k: v for k, v in have.items() if k in want and want[k].shape == v.shape}
            model.load_state_dict(keep, strict=False)
            source = f"{len(keep)} of {len(want)} tensors from the full model"
    params = sum(p.numel() for p in model.parameters())
    print(f"{variant}: {params / 1e6:.0f}M parameters ({source})", flush=True)
    model.set_attn_processor(AttnProcessor() if processor == "export" else AttnProcessor2_0())
    return UNetStep(model, PositionalEncoding(d_model=384)).eval(), params


def gflops(step) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        step(torch.randn(1, 8, 32, 32), torch.randn(1, 50, 384))
    return fc.get_total_flops() / 1e9


def export(step, name: str, out_dir: Path) -> Path:
    import coremltools as ct

    with torch.no_grad():
        ep = torch.export.export(step, (torch.randn(1, 8, 32, 32), torch.randn(1, 50, 384)), strict=False)
        ep = ep.run_decompositions({})
    base = convert(ep, [ct.TensorType(name="sample", shape=(1, 8, 32, 32), dtype=np.float16),
                        ct.TensorType(name="audio", shape=(1, 50, 384), dtype=np.float16)], "pred_latents", ct.target.iOS17)
    path = out_dir / f"{name}.mlpackage"
    quantize_weights(base).save(str(path))
    print(f"  saved {path}", flush=True)
    return path


def time_ane(path: Path, n: int = 50) -> tuple[float, float]:
    """Mean and p90 ms per call on CPU_AND_NE after a warm-up, from a compiled copy."""
    import coremltools as ct

    compiled = ct.models.utils.compile_model(str(path))
    t0 = time.time()
    m = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    load = time.time() - t0
    x = {"sample": np.random.randn(1, 8, 32, 32).astype(np.float16), "audio": np.random.randn(1, 50, 384).astype(np.float16)}
    for _ in range(5):
        m.predict(x)
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        m.predict(x)
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    print(f"  {path.name}: {np.mean(ms):.2f} ms mean, p90 {ms[int(0.9 * n)]:.2f} (load {load:.1f} s)", flush=True)
    return float(np.mean(ms)), ms[int(0.9 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bk_base,bk_small,bk_tiny")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "small")
    ap.add_argument("--time", action="store_true", help="time each on this Mac's Neural Engine, and the full model's int8w export")
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "small_unets_2026-11-03" / "small_unets_m5pro.json")
    ap.add_argument("--weights", type=Path, help="a trained student's state_dict (distill_small_unet.py's <variant>_best.pt) "
                                                 "for the one variant in --variants")
    ap.add_argument("--tag", help="with --weights: the export is unet_<variant>_<tag>_int8w_b1, results key <variant>_<tag>")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v != "full"]
    if args.weights is not None and (len(variants) != 1 or not args.tag or not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag)):
        ap.error("--weights needs exactly one variant in --variants and a --tag of letters, digits, _ or -")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    full = load_full(args.unet_dir)
    for v in ["full"] + variants:
        trained = args.weights is not None and v != "full"
        step, params = build(v, args.unet_dir, full=full, weights=args.weights if trained else None)
        key, name = (f"{v}_{args.tag}", f"unet_{v}_{args.tag}_int8w_b1") if trained else (v, f"unet_{v}_int8w_b1")
        r = {"params_m": round(params / 1e6, 1), "gflops": round(gflops(step), 1), **{k: v2 for k, v2 in VARIANTS[v].items()}}
        if trained:
            r["weights"] = str(args.weights)
        if v == "full":
            path = OUT_DIR / "unet_int8w_b1.mlpackage"  # export_coreml.py's, the same export path
        else:
            path = args.out_dir / f"{name}.mlpackage"
            if trained or not path.exists():
                export(step, name, args.out_dir)
        r["path"] = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
        if args.time:
            r["mac_ane_ms"], r["mac_ane_p90_ms"] = time_ane(path)
        results[key] = r
        print(f"  {key}: {json.dumps(r)}", flush=True)
        args.results.write_text(json.dumps(results, indent=1) + "\n")
    if args.time and "full" in results:
        f = results["full"]
        print("\nvariant     params  GFLOPs  Mac ANE ms  vs full  -> A14 UNet at 53.8 ms x ratio")
        for v, r in results.items():
            if "mac_ane_ms" not in r:
                continue
            ratio = r["mac_ane_ms"] / f["mac_ane_ms"]
            print(f"{v:10s} {r['params_m']:6.0f}M {r['gflops']:7.1f} {r['mac_ane_ms']:10.2f} {ratio:8.2f}x  {53.8 * ratio:6.1f} ms")


if __name__ == "__main__":
    main()
