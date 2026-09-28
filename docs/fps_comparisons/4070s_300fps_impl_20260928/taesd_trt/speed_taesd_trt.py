"""Standalone TAESD decode speed: compiled (today) vs the persisted TRT backend.

Per path, CUDA-event median over `--iters` back-to-back-synchronised calls after
`--warmup` calls, on real corpus latents, plus a `--pipelined` throughput figure
(one sync per `--iters` calls) and nvidia-smi clock/power samples at 1 Hz.
  compiled_bs8       TaesdVaeDecodeBackend.decode (max-autotune) + repo fast post (GPU part)
  trt_fp16_bs8       TaesdTrtBackend.decode                     (fp16 NCHW)
  trt_fp16_bs8_post  TaesdTrtBackend.decode + repo fast post    (what --post pinned measures)
  trt_u8_bs8         TaesdTrtBackend.decode_bgr_u8              (fused post)
  trt_*_bs16         the same with 16 latents = 2 x bs8 sub-batches
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import vae_fast_decoder as vfd  # noqa: E402

DEV = torch.device("cuda:0")


class Smi:
    def __init__(self):
        self.rows = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms=1000"], stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.rows.append((time.time(), [float(x) for x in line.split(",")]))
            except ValueError:
                pass

    def window(self, t0, t1):
        rows = [r for t, r in self.rows if t0 <= t <= t1]
        if not rows:
            return None
        a = np.array(rows)
        return {"samples": len(rows), "sm_mhz_median": float(np.median(a[:, 0])),
                "power_w_median": float(np.median(a[:, 1])), "temp_c_max": float(a[:, 2].max())}

    def stop(self):
        self.proc.terminate()


def measure(fn, bs, warmup, iters, smi):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    t1 = time.time()
    t = np.array(times)
    return {"batch": bs, "median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
            "p99_ms": float(np.percentile(t, 99)), "ms_per_frame": float(np.median(t)) / bs,
            "pipelined_ms_per_frame": s.elapsed_time(e) / iters / bs, "iters": iters, "warmup": warmup,
            "smi": smi.window(t0, t1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=2, help="interleaved repeats of the whole path list")
    ap.add_argument("--out", default=str(OUT / "speed_taesd_trt.json"))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    files = sorted((ROOT / "calibration/unet_multi_avatar_20260928").glob("unet_io_*.pt"))[::32][:2]
    Z = torch.cat([torch.load(f, map_location="cpu", weights_only=False)["pred_latents"] for f in files])
    Z = Z.to(DEV, torch.float16).contiguous()
    z8, z16 = Z[:8].contiguous(), Z[:16].contiguous()

    compiled = vfd.TaesdVaeDecodeBackend.load(device=DEV, runtime_dtype=torch.float16)
    compiled.warmup([8])
    trt = vfd.load_taesd_trt_backend(DEV, torch.float16, model=compiled.model)
    trt.warmup([8])
    post = vfd.repo_fast_postprocess_gpu
    paths = {
        "compiled_bs8_decode": (lambda: compiled.decode(z8, 1.0, torch.float16), 8),
        "compiled_bs8_decode_post": (lambda: post(compiled.decode(z8, 1.0, torch.float16)), 8),
        "trt_fp16_bs8": (lambda: trt.decode(z8, 1.0, torch.float16), 8),
        "trt_fp16_bs8_post": (lambda: post(trt.decode(z8, 1.0, torch.float16)), 8),
        "trt_u8_bs8": (lambda: trt.decode_bgr_u8(z8), 8),
        "trt_fp16_bs16": (lambda: trt.decode(z16, 1.0, torch.float16), 16),
        "trt_u8_bs16": (lambda: trt.decode_bgr_u8(z16), 16),
    }
    smi = Smi()
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "engine_key": trt.meta.get("key"),
           "rounds": []}
    with torch.inference_mode():
        for r in range(args.rounds):
            rnd = {}
            for name, (fn, bs) in paths.items():
                rnd[name] = measure(fn, bs, args.warmup, args.iters, smi)
                print(f"round {r} {name:26s} {rnd[name]['ms_per_frame']:.4f} ms/frame "
                      f"(pipelined {rnd[name]['pipelined_ms_per_frame']:.4f}) smi={rnd[name]['smi']}", flush=True)
            res["rounds"].append(rnd)
    smi.stop()
    res["median_over_rounds_ms_per_frame"] = {
        name: float(np.median([rnd[name]["ms_per_frame"] for rnd in res["rounds"]])) for name in paths}
    res["tags"] = "[M] measured by this run (burst, not power-capped steady state)"
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["median_over_rounds_ms_per_frame"], indent=1))


if __name__ == "__main__":
    main()
