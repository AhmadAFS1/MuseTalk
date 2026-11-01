#!/usr/bin/env python
"""Export the MuseTalk v1.5 UNet and the TAESD decoder to Core ML ML Programs.

The UNet model takes the raw per-frame Whisper window (50x384) and the 8-channel
face latent; the positional encoding and the fixed timestep 0 are inside the
graph, so the Core ML model is the whole per-frame UNet step the server runs.
One model per batch size (the Neural Engine wants static shapes). Variants:

  fp16      - float16 weights and activations (the reference device build)
  int8w     - int8 per-channel linear weights, fp16 activations (halves the bytes
              read per frame; what memory bandwidth on a phone needs)
  int8wa    - int8 weights and activations (uses the int8 Neural Engine compute
              on A17 Pro / M4 and newer; needs calibration data)

Usage (repo root, Python 3.12 venv):
  python scripts/ios/export_coreml.py --batches 1,4,8 --variants fp16,int8w,int8wa
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

from musetalk.models.unet import UNet  # noqa: E402

OUT_DIR = REPO_ROOT / "models" / "ios_bench" / "coreml"


class UNetStep(torch.nn.Module):
    """sample [B,8,32,32] + audio [B,50,384] -> pred latent [B,4,32,32]."""

    def __init__(self, unet, pe):
        super().__init__()
        self.unet = unet
        self.register_buffer("pe_table", pe.pe[:, :50, :].clone())
        self.register_buffer("timestep", torch.zeros(1, dtype=torch.long))

    def forward(self, sample, audio):
        hidden = audio + self.pe_table
        return self.unet(sample, self.timestep, encoder_hidden_states=hidden).sample


class TaesdDecode(torch.nn.Module):
    """latent [B,4,32,32] -> face [B,3,256,256] in [0,1] (vae_fast_decoder convention)."""

    def __init__(self, taesd):
        super().__init__()
        self.decoder = taesd.decoder

    def forward(self, latents):
        image = self.decoder(latents)
        return (image / 2 + 0.5).clamp(0, 1)


def convert(traced, inputs, name: str, target) -> "ct.models.MLModel":
    import coremltools as ct

    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=name)],
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    print(f"  converted in {time.time() - t0:.0f}s", flush=True)
    return model


def quantize_weights(model):
    import coremltools.optimize.coreml as cto

    cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    t0 = time.time()
    out = cto.linear_quantize_weights(model, config=cfg)
    print(f"  int8 weights in {time.time() - t0:.0f}s", flush=True)
    return out


def quantize_activations(model, samples: list[dict]):
    import coremltools.optimize.coreml as cto

    fn = getattr(cto, "linear_quantize_activations", None)
    if fn is None:  # coremltools 8.x location
        from coremltools.optimize.coreml.experimental import linear_quantize_activations as fn
    cfg = cto.OptimizationConfig(global_config=cto.experimental.OpActivationLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_tensor"))
    t0 = time.time()
    out = fn(model, config=cfg, sample_data=samples)
    print(f"  int8 activations in {time.time() - t0:.0f}s", flush=True)
    return out


def calibration_samples(bench_inputs: Path, batch: int, n: int = 16) -> list[dict]:
    d = torch.load(bench_inputs, map_location="cpu", weights_only=True)
    lat, aud = d["latents"], d["audio"]
    g = torch.Generator().manual_seed(1)
    samples = []
    for i in range(n):
        idx = (torch.arange(batch) + i) % lat.shape[0]
        jitter = 1.0 + 0.05 * torch.randn(1, generator=g).item()
        samples.append({"sample": (lat[idx] * jitter).numpy().astype(np.float16),
                        "audio": (aud[idx] * jitter).numpy().astype(np.float16)})
    return samples


def main() -> None:
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--variants", default="fp16,int8w,int8wa")
    ap.add_argument("--unet-dir", default=str(REPO_ROOT / "models" / "musetalkV15"))
    ap.add_argument("--bench-inputs", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "bench_inputs.pt")
    ap.add_argument("--target", default="iOS17", choices=["iOS17", "iOS18"])
    ap.add_argument("--exporter", default="export", choices=["export", "trace"])
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--skip-taesd", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    target = getattr(ct.target, args.target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    batches = [int(b) for b in args.batches.split(",")]
    variants = args.variants.split(",")
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def record(name, path, extra):
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        manifest[name] = {"path": str(path), "bytes": size, **extra}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  saved {path} ({size / 2**20:.0f} MB)", flush=True)

    if not args.skip_taesd:
        from diffusers import AutoencoderTiny

        taesd = AutoencoderTiny.from_pretrained(str(REPO_ROOT / "models" / "taesd"), torch_dtype=torch.float32).eval()
        wrapper = TaesdDecode(taesd).eval()
        for b in batches:
            print(f"TAESD decoder batch {b}", flush=True)
            ex = torch.randn(b, 4, 32, 32)
            traced = torch.jit.trace(wrapper, ex)
            model = convert(traced, [ct.TensorType(name="latents", shape=(b, 4, 32, 32), dtype=np.float16)], "faces", target)
            path = args.out_dir / f"taesd_fp16_b{b}.mlpackage"
            model.save(str(path))
            record(f"taesd_fp16_b{b}", path, {"kind": "taesd", "variant": "fp16", "batch": b})

    if args.skip_unet:
        return

    print("loading UNet...", flush=True)
    wrap = UNet(unet_config=os.path.join(args.unet_dir, "musetalk.json"),
                model_path=os.path.join(args.unet_dir, "unet.pth"), device=torch.device("cpu"))
    unet = wrap.model.eval().requires_grad_(False)
    # Explicit attention (matmul + softmax) traces and converts predictably; the
    # fused SDPA op needs an iOS 18 target and is not needed for the ANE.
    from diffusers.models.attention_processor import AttnProcessor
    unet.set_attn_processor(AttnProcessor())
    step = UNetStep(unet, wrap.pe).eval()

    for b in batches:
        print(f"UNet batch {b}: exporting ({args.exporter})", flush=True)
        ex_sample, ex_audio = torch.randn(b, 8, 32, 32), torch.randn(b, 50, 384)
        with torch.no_grad():
            if args.exporter == "export":
                # torch.export specializes the shape arithmetic (height*width,
                # batch//heads) to constants; torch.jit.trace leaves aten::Int
                # casts on 1-element tensors that coremltools 9 cannot fold.
                traced = torch.export.export(step, (ex_sample, ex_audio), strict=False)
                traced = traced.run_decompositions({})  # coremltools wants the ATEN dialect
            else:
                traced = torch.jit.trace(step, (ex_sample, ex_audio), check_trace=False)
        base = convert(traced, [ct.TensorType(name="sample", shape=(b, 8, 32, 32), dtype=np.float16),
                                ct.TensorType(name="audio", shape=(b, 50, 384), dtype=np.float16)],
                       "pred_latents", target)
        if "fp16" in variants:
            path = args.out_dir / f"unet_fp16_b{b}.mlpackage"
            base.save(str(path))
            record(f"unet_fp16_b{b}", path, {"kind": "unet", "variant": "fp16", "batch": b})
        if "int8w" in variants or "int8wa" in variants:
            w8 = quantize_weights(base)
            if "int8w" in variants:
                path = args.out_dir / f"unet_int8w_b{b}.mlpackage"
                w8.save(str(path))
                record(f"unet_int8w_b{b}", path, {"kind": "unet", "variant": "int8w", "batch": b})
            if "int8wa" in variants:
                try:
                    wa8 = quantize_activations(w8, calibration_samples(args.bench_inputs, b))
                    path = args.out_dir / f"unet_int8wa_b{b}.mlpackage"
                    wa8.save(str(path))
                    record(f"unet_int8wa_b{b}", path, {"kind": "unet", "variant": "int8wa", "batch": b})
                except Exception as exc:  # report, keep going with the other batches
                    print(f"  int8wa FAILED for batch {b}: {type(exc).__name__}: {exc}", flush=True)
                    manifest[f"unet_int8wa_b{b}"] = {"error": f"{type(exc).__name__}: {exc}"}
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        del base


if __name__ == "__main__":
    main()
