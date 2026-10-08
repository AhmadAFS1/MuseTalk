"""Per-block CUDA-graph replay time of stagewise UNet engine sets (which blocks INT8 actually speeds up).

For every engine set (--root, repeatable) and every block of its chain, one single-block StageChain is
captured into a CUDA graph and timed; sets are interleaved block by block so power-cap clock drift hits
them alike. Inputs are independently randomized buffers, not captured speech. This is a synthetic
diagnostic and is not full-pipeline throughput or quality evidence.
Run under the GPU lease:
  scripts/box_guard.sh run --min-avail-gb 6 --label bench_blocks -- /workspace/.venvs/musetalk_trt_stagewise/bin/python \
      scripts/bench_stagewise_blocks.py --root models/tensorrt_unet_stagewise_sm89_srcmix --root ... --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import unet_stagewise_trt as sw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    sets = {}
    for r in args.root:
        d = (ROOT / r / f"bs{args.batch}") if not Path(r).is_absolute() else Path(r) / f"bs{args.batch}"
        man = json.loads((d / "manifest.json").read_text())
        import tensorrt as trt

        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engines = {}
        # partial sets (a few rebuilt blocks) are fine: time whatever blocks exist
        for name, entry in man.get("blocks", {}).items():
            p = d / entry["engine_file"]
            if name != "prefix" and p.exists():
                engines[name] = rt.deserialize_cuda_engine(p.read_bytes())
        spec = [blk for blk in man["spec"] if blk["name"] in engines]
        chains = {}
        for blk in spec:
            ch = sw.StageChain({blk["name"]: engines[blk["name"]]}, [blk], device, require_io=False)
            for t in ch.buffers.values():
                if t.dtype.is_floating_point:
                    t.normal_()
            ch.capture_graph()
            chains[blk["name"]] = ch
        sets[r] = {"runtime": rt, "engines": engines, "chains": chains,
                   "files": {b: man["blocks"][b]["engine_file"] for b in chains}}
        print(f"loaded {r}: {list(chains)}", flush=True)
    names = []
    for s in sets.values():
        for b in s["chains"]:
            if b not in names:
                names.append(b)
    times = {r: {b: [] for b in s["chains"]} for r, s in sets.items()}
    t0 = time.time()
    for _ in range(args.rounds):
        for b in names:
            for r, s in sets.items():
                ch = s["chains"].get(b)
                if ch is None:
                    continue
                for _ in range(5):
                    ch.run()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(args.reps):
                    ch.run()
                e1.record()
                e1.synchronize()
                times[r][b].append(e0.elapsed_time(e1) / args.reps)
    out = {"batch": args.batch, "rounds": args.rounds, "reps": args.reps, "seconds": time.time() - t0,
           "input_kind": "synthetic_random_buffers", "measurement_scope": "isolated_blocks_not_pipeline", "sets": {}}
    for r, s in sets.items():
        med = {b: sorted(v)[len(v) // 2] for b, v in times[r].items()}
        out["sets"][r] = {"block_ms": med, "round_ms": times[r], "sum_ms": sum(med.values()), "engine_files": s["files"]}
    hdr = "%-10s" % "block" + "".join("%14s" % Path(r).name[-14:] for r in sets)
    print(hdr)
    for b in names:
        print("%-10s" % b + "".join("%14s" % (f"{out['sets'][r]['block_ms'][b]:.3f}" if b in out["sets"][r]["block_ms"]
                                               else "-") for r in sets))
    print("%-10s" % "sum" + "".join("%14.3f" % out["sets"][r]["sum_ms"] for r in sets))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
