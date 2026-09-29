"""Per-layer TensorRT profile of one stagewise UNet block engine (where the block's time goes).

Direct enqueue (no CUDA graph) with an IProfiler attached; prints the top layers by median time and a
coarse grouping by layer-name keywords. Engines built without DETAILED profiling verbosity still report
fused-kernel names, which is enough to see attention / norm / conv / GEMM shares.
  scripts/box_guard.sh run --min-avail-gb 6 --label profile_block -- /workspace/.venvs/musetalk_trt_stagewise/bin/python \
      scripts/profile_stagewise_block.py --root models/tensorrt_unet_stagewise_sm89_srcmix --block up3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import unet_stagewise_trt as sw  # noqa: E402


def main() -> int:
    import tensorrt as trt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", action="append", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    d = Path(args.root) / f"bs{args.batch}"
    man = json.loads((d / "manifest.json").read_text())
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    report = {}
    for name in args.block:
        entry = man["blocks"][name]
        eng = rt.deserialize_cuda_engine((d / entry["engine_file"]).read_bytes())
        blk = next(b for b in man["spec"] if b["name"] == name)
        ch = sw.StageChain({name: eng}, [blk], device, require_io=False)
        for t in ch.buffers.values():
            if t.dtype.is_floating_point:
                t.normal_()
        times: dict[str, list] = {}

        class Prof(trt.IProfiler):
            def report_layer_time(self, layer_name, ms):
                times.setdefault(layer_name, []).append(ms)

        _, ctx = ch.contexts[0]
        for _ in range(3):
            ch.enqueue_current()
        torch.cuda.synchronize()
        ctx.profiler = Prof()
        stream = torch.cuda.Stream()
        for _ in range(args.iters):
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        total = sum(med.values())
        groups = {}
        for k, v in med.items():
            lk = k.lower()
            g = ("attention/mha" if any(s in lk for s in ("mha", "attention", "softmax", "fmha", "bmm"))
                 else "norm" if any(s in lk for s in ("norm", "reduce", "instancenorm"))
                 else "conv" if "conv" in lk else "gemm/matmul" if any(s in lk for s in ("matmul", "gemm", "linear"))
                 else "reformat/copy" if any(s in lk for s in ("reformat", "copy", "shuffle", "transpose"))
                 else "pointwise/other")
            groups[g] = groups.get(g, 0.0) + v
        top = sorted(med.items(), key=lambda kv: -kv[1])[: args.top]
        print(f"== {name} ({entry['engine_file']}): {len(med)} layers, profiled sum {total:.3f} ms")
        for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
            print(f"   {g:18s} {v:7.3f} ms  {100 * v / total:5.1f}%")
        for k, v in top:
            print(f"   {v:7.3f} ms  {k[:150]}")
        report[name] = {"total_ms": total, "groups": groups, "layers": med}
        del ctx, ch, eng
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
