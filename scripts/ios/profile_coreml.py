#!/usr/bin/env python
"""Where a Core ML model's Neural Engine time goes, from Core ML's own compute plan.

`MLComputePlan` gives each op a relative estimated cost (weights sum to 1 over
the program) and its device placement. Grouped by op type and, where the op
names carry it, by UNet block, that is the map for deciding what to change
first: attention vs convolution, which resolution level, and anything that
fell off the Neural Engine.

Usage (repo root, .venv-ct):
  python scripts/ios/profile_coreml.py --model unet_int8w_b1 [--top 25]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"

# torch.export names ops after the module path they came from, e.g.
# "down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q"; pick the block out.
BLOCK_RE = re.compile(r"(down_blocks_\d|mid_block|up_blocks_\d|conv_in|conv_out|conv_norm_out|time_)")
KIND_RE = re.compile(r"(attn1|attn2|ff_|proj_in|proj_out|resnets|downsamplers|upsamplers|norm)")


def main() -> None:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_int8w_b1")
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    path = manifest[args.model]["path"]
    compiled = ct.models.utils.compile_model(path)
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=getattr(ct.ComputeUnit, args.units))
    fn = plan.model_structure.program.functions["main"]

    by_type, by_block, by_kind, by_device, rows = (defaultdict(float) for _ in range(5))
    n_ops, n_cost = 0, 0
    for op in fn.block.operations:
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        n_ops += 1
        if cost is None:
            continue
        w = cost.weight
        n_cost += 1
        name = (op.outputs[0].name if op.outputs else op.operator_name) or ""
        dev = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "") if usage else "?"
        by_type[op.operator_name] += w
        by_device[dev] += w
        m = BLOCK_RE.search(name)
        by_block[m.group(1) if m else "other"] += w
        k = KIND_RE.search(name)
        by_kind[k.group(1) if k else "other"] += w
        rows[name] = w

    def show(title, d, top=None):
        items = sorted(d.items(), key=lambda kv: -kv[1])
        if top:
            items = items[:top]
        print(f"\n{title}")
        for k, v in items:
            print(f"  {v * 100:6.2f}%  {k}")

    print(f"{args.model}: {n_ops} ops, {n_cost} with a cost estimate, compute units {args.units}")
    show("by device", by_device)
    show("by op type", by_type, 15)
    show("by UNet block", by_block)
    show("by layer kind", by_kind)
    show(f"top {args.top} ops", rows, args.top)

    if args.out:
        args.out.write_text(json.dumps({"model": args.model, "units": args.units, "ops": n_ops,
                                        "by_device": by_device, "by_type": by_type, "by_block": by_block,
                                        "by_kind": by_kind,
                                        "top_ops": dict(sorted(rows.items(), key=lambda kv: -kv[1])[:args.top])},
                                       indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
