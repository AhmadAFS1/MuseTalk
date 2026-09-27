"""Shared helpers for the TAESD 300fps probe (read-only on /workspace)."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parent
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(WORK / "cache" / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(WORK / "cache" / "triton"))
os.environ.setdefault("TMPDIR", str(WORK / "cache" / "tmp"))
for _d in ("cache/inductor", "cache/triton", "cache/tmp"):
    (WORK / _d).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

ROOT = Path("/workspace/MuseTalk")
TAESD_DIR = ROOT / "models" / "taesd"
CAPTURES = "/workspace/benchmarks/same-avatar/unet-captures/unet_io_*_bs8_*.pt"
DEV = torch.device("cuda:0")
USED_Y0 = 104  # latent row 13


def gpu_state(tag: str = "") -> dict:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    own_reserved_mb = torch.cuda.memory_reserved() / 2**20 if torch.cuda.is_initialized() else 0.0
    rec = {"tag": tag, "t": time.time(), "smi": q, "compute_apps": apps,
           "own_torch_reserved_MiB": round(own_reserved_mb, 1), "own_pid": os.getpid()}
    print(f"[gpu_state {tag}] {q} | apps: {apps.replace(chr(10), '; ') or '-'} | own_reserved={own_reserved_mb:.0f}MiB",
          flush=True)
    return rec


def load_taesd(dtype=torch.float16):
    from diffusers import AutoencoderTiny
    m = AutoencoderTiny.from_pretrained(str(TAESD_DIR), torch_dtype=dtype).to(DEV).eval()
    m.requires_grad_(False)
    return m


def real_latents(n: int | None = None) -> torch.Tensor:
    """Real post-UNet latents (pred_latents) captured from the live server path."""
    files = sorted(glob.glob(CAPTURES))
    zs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        zs.append(d["pred_latents"].to(torch.float16))
    z = torch.cat(zs, 0)
    if n is not None:
        reps = (n + z.shape[0] - 1) // z.shape[0]
        z = z.repeat(reps, 1, 1, 1)[:n]
    return z.contiguous()


def real_unet_inputs():
    files = sorted(glob.glob(CAPTURES))
    d = torch.load(files[0], map_location="cpu", weights_only=False)
    return d["latent_batch"].to(torch.float16), d["audio_feature_batch"].to(torch.float16)


def repo_raw_decode(model):
    """Exact repo convention (scripts/vae_fast_decoder.py::_raw_decode)."""
    def _raw_decode(latents):
        image = model.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)
    return _raw_decode


def bench_events(fn, warmup=10, iters=40, stream=None, mark_step=False):
    """Median/mean/p90 of per-call CUDA-event time and per-call wall time (sync each call)."""
    stream = stream or torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            fn()
        torch.cuda.synchronize()
        ev = []
        walls = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            s.record(stream)
            if mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            fn()
            e.record(stream)
            e.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)
            ev.append(s.elapsed_time(e))
    ev = np.array(ev)
    walls = np.array(walls)
    return {"median_ms": float(np.median(ev)), "mean_ms": float(ev.mean()),
            "p90_ms": float(np.percentile(ev, 90)), "min_ms": float(ev.min()),
            "wall_median_ms": float(np.median(walls)), "n": int(iters), "warmup": int(warmup)}


def bench_throughput(fn, iters=40, warmup=10, mark_step=False):
    """Back-to-back launches, single sync: aggregate ms/call (captures pipelined steady state)."""
    for _ in range(warmup):
        if mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    s.record()
    for _ in range(iters):
        if mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        fn()
    e.record()
    e.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    return {"gpu_ms_per_call": s.elapsed_time(e) / iters, "wall_ms_per_call": wall / iters, "n": iters}


def dump(name: str, obj):
    p = WORK / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {p}", flush=True)
