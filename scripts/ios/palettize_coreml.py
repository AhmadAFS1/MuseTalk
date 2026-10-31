#!/usr/bin/env python
"""Palettize a converted Core ML UNet to n-bit weights (k-means lookup tables).

Weight bytes per frame set the floor on a phone's memory bus (iPhone 12 Pro:
~34 GB/s, so 812 MB of int8 weights is 24 ms before any compute). Palettized
weights are decompressed to fp16 on the fly, so this trades nothing in compute
and only quality, which the repo gate measures afterwards.

Usage (repo root, .venv-ct):
  python scripts/ios/palettize_coreml.py --model unet_ane_fp16_b1 --nbits 6,4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"


def main() -> None:
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_ane_fp16_b1")
    ap.add_argument("--nbits", default="6,4")
    ap.add_argument("--mode", default="kmeans", choices=["kmeans", "uniform"])
    ap.add_argument("--granularity", default="per_tensor", choices=["per_tensor", "per_grouped_channel"])
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest[args.model]
    out_dir = Path(entry["path"]).parent

    for nbits in [int(x) for x in args.nbits.split(",")]:
        print(f"{args.model}: palettizing to {nbits} bits ({args.mode}, {args.granularity})", flush=True)
        model = ct.models.MLModel(entry["path"], compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        kw = {"mode": args.mode, "nbits": nbits, "granularity": args.granularity}
        if args.granularity == "per_grouped_channel":
            kw["group_size"] = args.group_size
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(**kw))
        t0 = time.time()
        out = cto.palettize_weights(model, config=cfg)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
        name = f"{args.model}_pal{nbits}"
        path = out_dir / f"{name}.mlpackage"
        out.save(str(path))
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        manifest = json.loads(MANIFEST.read_text())  # another export may have written meanwhile
        manifest[name] = {**{k: v for k, v in entry.items() if k not in ("path", "bytes")}, "path": str(path), "bytes": size,
                          "variant": f"{entry['variant']}_pal{nbits}", "palettize": kw}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  saved {path} ({size / 2**20:.0f} MB)", flush=True)
        del out, model


if __name__ == "__main__":
    main()
