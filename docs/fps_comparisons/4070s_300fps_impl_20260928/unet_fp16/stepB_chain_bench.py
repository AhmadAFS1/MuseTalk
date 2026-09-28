"""Step B (plan item 2.2): sustained go/no-go timing of the chained stagewise FP16 TensorRT UNet.

Loads the persisted engine set through StagewiseTrtUnetBackend (11 raw contexts, one shared arena,
skip tensors in static buffers, execute_async_v3, one CUDA graph over all 11 enqueues) and calls it
back to back exactly as the scheduler would (input copy into the static buffers + replay + output
clone), on real multi-avatar corpus batches, for --seconds at the power cap with nvidia-smi at 1 Hz.
Also times the bare graph replay (no copies) for a short window, and records VRAM.

  scripts/box_guard.sh run --min-avail-gb 6 -- /workspace/.venvs/musetalk_trt_stagewise/bin/python \
     docs/fps_comparisons/4070s_300fps_impl_20260928/unet_fp16/stepB_chain_bench.py --batch 16 --seconds 90
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402

from scripts import unet_stagewise_trt as sw  # noqa: E402

FIELDS = "clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active"


class Smi:
    def __init__(self):
        self.rows = []
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "-lms=1000"],
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            self.rows.append((time.time(), [x.strip() for x in line.split(",")]))

    def stop(self):
        self.p.terminate()

    def stats(self, t0, t1):
        sel = [r for t, r in self.rows if t0 <= t <= t1]
        out = {"samples": len(sel)}
        for i, k in enumerate(["sm_mhz", "mem_mhz", "power_w", "power_limit_w", "temp_c", "util_pct", "mem_used_mib"]):
            vals = []
            for r in sel:
                try:
                    vals.append(float(r[i]))
                except (ValueError, IndexError):
                    pass
            vals.sort()
            if vals:
                out[k] = {"median": statistics.median(vals), "min": vals[0], "max": vals[-1],
                          "p05": vals[int(0.05 * (len(vals) - 1))], "p95": vals[int(0.95 * (len(vals) - 1))]}
        th = {}
        for r in sel:
            if len(r) > 7:
                th[r[7]] = th.get(r[7], 0) + 1
        out["throttle_hist"] = th
        return out


def smi_mem_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out.splitlines()[0])
    except Exception:
        return None


def dist(v):
    v = sorted(v)
    return {"median": statistics.median(v), "mean": statistics.fmean(v), "p90": v[int(0.9 * (len(v) - 1))],
            "p99": v[int(0.99 * (len(v) - 1))], "min": v[0], "max": v[-1], "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--raw-seconds", type=float, default=15.0)
    ap.add_argument("--root", default=str(sw.default_cache_root()))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(args), "tags": "[M]"}
    free0, total = torch.cuda.mem_get_info()
    smi_before = smi_mem_used()
    t = time.time()
    backend = sw.StagewiseTrtUnetBackend.load(Path(args.root) / f"bs{args.batch}", device=dev)
    res["load_s"] = time.time() - t
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    res["vram"] = {"device_used_by_backend_mib": (free0 - free1) / 2**20, "smi_used_before_mib": smi_before,
                   "smi_used_after_load_mib": smi_mem_used(), "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                   **backend.describe()}
    # real inputs: consecutive corpus captures concatenated to the engine batch
    files = sorted((ROOT / "calibration/unet_multi_avatar_20260928").glob("unet_io_*.pt"))
    step = max(1, len(files) // 32)
    rows_l, rows_a = [], []
    for f in files[::step][:32]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        rows_l.append(d["latent_batch"].half())
        rows_a.append(d["audio_feature_batch"].half())
    L, A = torch.cat(rows_l), torch.cat(rows_a)
    bs = args.batch
    inputs = [(L[i * bs:(i + 1) * bs].to(dev), A[i * bs:(i + 1) * bs].to(dev)) for i in range(L.shape[0] // bs)]
    ts = torch.tensor([0], device=dev)
    smi = Smi()

    def call(i):
        lat, aud = inputs[i % len(inputs)]
        return backend(lat, ts, encoder_hidden_states=aud).sample

    with torch.inference_mode():
        t_w = time.time()
        i = 0
        while time.time() - t_w < args.warmup:
            call(i)
            i += 1
            if i % 4 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        # sustained, scheduler-like (copy in + replay + clone out), bounded run-ahead of 2
        evs, host = [], []
        t0 = time.time()
        p0 = time.perf_counter()
        n = 0
        while time.perf_counter() - p0 < args.seconds:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            h0 = time.perf_counter()
            s.record()
            call(n)
            e.record()
            host.append((time.perf_counter() - h0) * 1e3)
            evs.append((s, e))
            if len(evs) > 2:
                evs[-3][1].synchronize()
            n += 1
        torch.cuda.synchronize()
        wall = time.perf_counter() - p0
        t1 = time.time()
        ms = [s.elapsed_time(e) for s, e in evs]
        # windows of 10 s
        # bare graph replay, back to back
        chain = backend._chain
        t2 = time.time()
        p2 = time.perf_counter()
        raw = []
        while time.perf_counter() - p2 < args.raw_seconds:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                chain.run()
            e.record()
            e.synchronize()
            raw.append(s.elapsed_time(e) / 10)
        t3 = time.time()
    smi.stop()
    d = dist(ms)
    res["sustained"] = {
        "calls": n, "wall_s": wall, "wall_ms_per_call": wall * 1e3 / n, "wall_ms_per_frame": wall * 1e3 / (n * bs),
        "wall_fps": n * bs / wall, "event_ms_per_call": d, "event_ms_per_frame_median": d["median"] / bs,
        "host_ms_per_call_median": statistics.median(host), "gpu": smi.stats(t0, t1),
    }
    rd = dist(raw)
    res["raw_graph_replay"] = {"ms_per_call": rd, "ms_per_frame_median": rd["median"] / bs, "gpu": smi.stats(t2, t3)}
    res["vram"]["torch_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    per_frame = res["sustained"]["event_ms_per_frame_median"]
    if bs == 16:
        verdict = "GO" if per_frame <= 2.51 else ("MARGINAL" if per_frame <= 2.64 else "NO-GO (K2)")
    else:
        verdict = "GO (2.3a fallback)" if per_frame <= 2.60 else "NO-GO"
    res["decision"] = {"ms_per_frame": per_frame, "verdict": verdict,
                       "rule": "bs16: <=2.51 GO, 2.51-2.64 marginal, >2.64 K2; bs8: <=2.60 GO"}
    out = Path(args.out) if args.out else OUT / f"stepB_chain_bench_bs{bs}{('_' + args.label) if args.label else ''}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    with open(out.with_suffix(".smi.csv"), "w") as f:
        f.write("host_time," + FIELDS + "\n")
        for tt, r in smi.rows:
            f.write(f"{tt:.3f}," + ",".join(r) + "\n")
    print(json.dumps({k: res[k] for k in ("load_s", "vram", "sustained", "raw_graph_replay", "decision")}, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
