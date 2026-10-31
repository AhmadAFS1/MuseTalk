#!/usr/bin/env python
"""Split a Core ML UNet into two chunks for the iPhone (Apple's ml-stable-diffusion recipe).

Why: on iOS the Neural Engine program for the whole 850M-parameter UNet is compiled
inside the app's process, where the int8 weights expand to fp16 (2.46 GB measured on
an iPhone 12 Pro against a 3.07 GB allowance) and the compiled program (1.6 GB) must be
written to the app's cache; the ml-stable-diffusion README calls chunking "required for
Neural Engine deployment on iOS and iPadOS if weights are not quantized to 6-bits or
less". `coremltools.models.utils.bisect_model` (what Apple's chunk_mlprogram.py calls)
cuts the ML Program at the op where half the weight bytes have been consumed; the
second chunk takes the first chunk's outputs (and any original inputs it still needs).

Usage (repo root, .venv-ct):
  python scripts/ios/chunk_coreml.py --model unet_ane_int8w_b1 --check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"
DTYPE_NAMES = {65552: "float16", 65568: "float32", 131104: "int32", 65600: "float64"}


def io_names(mlpackage: Path):
    import coremltools as ct

    spec = ct.utils.load_spec(str(mlpackage))
    ins = [(f.name, [int(s) for s in f.type.multiArrayType.shape], DTYPE_NAMES.get(f.type.multiArrayType.dataType))
           for f in spec.description.input]
    outs = [(f.name, [int(s) for s in f.type.multiArrayType.shape], DTYPE_NAMES.get(f.type.multiArrayType.dataType))
            for f in spec.description.output]
    return ins, outs


def run_chain(models, feed: dict) -> dict:
    """Run chunk1 -> chunk2: each stage takes what it needs from the running feature dict by name."""
    feats = dict(feed)
    for model in models:
        names = [i.name for i in model.get_spec().description.input]
        out = model.predict({n: feats[n] for n in names})
        feats.update(out)
    return feats


def main() -> None:
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_ane_int8w_b1")
    ap.add_argument("--check", action="store_true", help="compare chunk1->chunk2 with the unchunked model on real frames")
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--capture-dir", type=Path, default=REPO_ROOT / "calibration" / "unet_portable_bs8")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    entry = manifest[args.model]
    src = Path(entry["path"])
    out_dir = src.parent
    t0 = time.time()
    print(f"bisecting {src.name} ({entry['bytes'] / 2**20:.0f} MB)", flush=True)
    ct.models.utils.bisect_model(model=str(src), output_dir=str(out_dir), merge_chunks_to_pipeline=False,
                                 check_output_correctness=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    chunks = []
    for i in (1, 2):
        path = out_dir / f"{src.stem}_chunk{i}.mlpackage"
        if not path.exists():
            raise SystemExit(f"expected {path}")
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        ins, outs = io_names(path)
        name = f"{args.model}_chunk{i}"
        manifest = json.loads(MANIFEST.read_text())
        manifest[name] = {"path": str(path), "bytes": size, "kind": "unet_chunk", "chunk_of": args.model, "chunk_index": i,
                          "batch": entry.get("batch", 1), "variant": entry.get("variant"), "layout": entry.get("layout"),
                          "inputs": [n for n, _, _ in ins], "outputs": [n for n, _, _ in outs]}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        chunks.append(path)
        print(f"  chunk{i}: {size / 2**20:.0f} MB; inputs {[(n, s, d) for n, s, d in ins]}; outputs {[(n, s, d) for n, s, d in outs]}",
              flush=True)

    if not args.check:
        return

    from musetalk.models.unet import PositionalEncoding
    from scripts.validate_unet_backend import _load_capture_group, discover_captures, tensor_stats

    units = getattr(ct.ComputeUnit, args.units)
    full = ct.models.MLModel(str(src), compute_units=units)
    parts = [ct.models.MLModel(str(p), compute_units=units) for p in chunks]
    pe = PositionalEncoding(d_model=384).pe[:, :50, :].float()
    paths = discover_captures(args.capture_dir, max(1, args.frames // 8))
    b = entry.get("batch", 1)
    full_out, chain_out, refs = [], [], []
    t_full = t_chain = 0.0
    for path in paths:
        payload = _load_capture_group([path])
        n = int(payload.get("actual_batch", payload["latent_batch"].shape[0]))
        lat = payload["latent_batch"].float()[:n]
        aud = payload["audio_feature_batch"].float()[:n] - pe
        ref = payload["pred_latents"].float()[:n]
        for s in range(0, n, b):
            feed = {"sample": lat[s:s + b].numpy().astype(np.float16), "audio": aud[s:s + b].numpy().astype(np.float16)}
            if feed["sample"].shape[0] < b:
                continue
            t = time.perf_counter()
            fo = np.asarray(full.predict(feed)["pred_latents"], dtype=np.float32)
            t_full += time.perf_counter() - t
            t = time.perf_counter()
            co = np.asarray(run_chain(parts, feed)["pred_latents"], dtype=np.float32)
            t_chain += time.perf_counter() - t
            full_out.append(fo)
            chain_out.append(co)
            refs.append(ref[s:s + b].numpy())
    full_t, chain_t, ref_t = (torch.from_numpy(np.concatenate(x)) for x in (full_out, chain_out, refs))
    n = full_t.shape[0]
    d = (full_t - chain_t).abs()
    print(f"\n[{args.units}] chunked vs unchunked on {n} real frames: max |diff| {d.max():.2e}, mean {d.mean():.2e}", flush=True)
    s_full, s_chain = tensor_stats(full_t, ref_t), tensor_stats(chain_t, ref_t)
    print(f"  vs capture reference: unchunked mae {s_full['mae']:.5f} max_abs {s_full['max_abs']:.4f}; "
          f"chunked mae {s_chain['mae']:.5f} max_abs {s_chain['max_abs']:.4f}", flush=True)
    print(f"  ms/frame incl. Python I/O: unchunked {t_full / n * 1000:.1f}, chunked {t_chain / n * 1000:.1f}", flush=True)
    result = {"model": args.model, "units": args.units, "frames": n, "chunk_vs_full_max_abs": float(d.max()),
              "chunk_vs_full_mean_abs": float(d.mean()), "unchunked_vs_ref": s_full, "chunked_vs_ref": s_chain,
              "ms_per_frame_unchunked": t_full / n * 1000, "ms_per_frame_chunked": t_chain / n * 1000,
              "chunks": [{"name": f"{args.model}_chunk{i}", "bytes": json.loads(MANIFEST.read_text())[f"{args.model}_chunk{i}"]["bytes"]}
                         for i in (1, 2)]}
    out = REPO_ROOT / "docs" / "ios_unet_bench" / f"chunk_{args.model}_m5pro.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
