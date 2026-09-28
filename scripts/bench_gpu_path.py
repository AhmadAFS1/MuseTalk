"""Sustained benchmark of the live-equivalent MuseTalk GPU path, per scheduler batch.

Plan item 0.8 (L1-equivalent without a server). Adapted from
docs/fps_comparisons/4070s_300fps_20260927/taesd_probe/p6_combined.py, but:
- backends are loaded through the repo's own loaders (scripts/trt_runtime.py
  load_unet_trt_backend / load_vae_trt_decoder, the same calls
  ParallelAvatarManager._init_models makes), so later backends are selected purely
  by env flags;
- the per-batch body mirrors HLSGPUStreamScheduler._run_generation_batch:
  pinned staging -> H2D (non_blocking) -> [stage sync] -> UNet(latent, timesteps,
  encoder_hidden_states).sample -> [stage sync] -> cast to the VAE dtype ->
  VAE.decode_latents_tensor (the attached decode backend) -> [decode sync] ->
  the repo's uint8 BGR postprocess (musetalk/models/vae.py MUSETALK_VAE_FAST_POSTPROCESS)
  -> D2H into a pinned buffer (default) or the repo's pageable .cpu() (--post repo);
- inputs are real scheduler captures (default: the multi-avatar corpus);
- it runs for a fixed wall duration with 1 Hz nvidia-smi logging of clocks, power,
  temperature, utilisation and throttle reasons, and reports CUDA-event stage times.

Environment: by default the live launcher's configuration is reproduced without
touching it: .runtime/musetalk_trt_local_sm89.env, then the overrides that
experiments/chinese_bob_webrtc_20260927/run_local_api.sh exports (TAESD decoder,
TRT UNet, ...), then an optional --overlay file (e.g. .runtime/musetalk_300fps.env).
Each layer only sets keys that are not already set, and variables set by the
caller always win, so `FLAG=1 scripts/bench_gpu_path.py ...` measures a lever.

Run under the GPU lease (loading the 2.2 GB TRT .ts has an ~8.5 GB host-RSS peak):
  scripts/box_guard.sh run --min-avail-gb 11 -- \
    /workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/bench_gpu_path.py --seconds 180
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LIVE_ENV_FILE = ROOT / ".runtime/musetalk_trt_local_sm89.env"
# Exported by experiments/chinese_bob_webrtc_20260927/run_local_api.sh after it sources
# LIVE_ENV_FILE (copied here; the launcher itself is not read or modified at runtime).
LIVE_LAUNCHER_OVERRIDES = {
    "MUSETALK_VAE_BACKEND": "taesd",
    "MUSETALK_UNET_BACKEND": "trt",
    "MUSETALK_TRT_FALLBACK": "0",
    "MUSETALK_BLEND_FIXED_POINT": "1",
    "MUSETALK_BLEND_SHRINK_MASK_BBOX": "1",
    "MUSETALK_TAESD_WARMUP_BATCHES": "8",
    "AVATAR_S3_ENABLED": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
ENV_PREFIXES = ("MUSETALK_", "HLS_", "WEBRTC_", "TORCH", "TRITON", "CUDA_", "PYTORCH_", "HF_", "TRANSFORMERS_")


def parse_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


def apply_env(args) -> dict:
    """Layer the live env under the caller's env. Returns provenance per key."""
    caller = set(os.environ)
    layers = []
    if args.live_env:
        layers.append((str(LIVE_ENV_FILE.relative_to(ROOT)), parse_env_file(LIVE_ENV_FILE)))
        layers.append(("run_local_api.sh overrides", dict(LIVE_LAUNCHER_OVERRIDES)))
    if args.overlay:
        layers.append((args.overlay, parse_env_file(Path(args.overlay))))
    source = {}
    merged: dict[str, str] = {}
    for name, values in layers:
        for key, value in values.items():
            merged[key] = value
            source[key] = name
    for key, value in merged.items():
        if key in caller:
            source[key] = "caller"
            continue
        os.environ[key] = value
    for key in os.environ:
        if key.startswith(ENV_PREFIXES) and key not in source:
            source[key] = "caller"
    return {k: {"value": os.environ.get(k), "source": source[k]} for k in sorted(source) if k in os.environ}


class SmiLogger:
    """nvidia-smi -lms sampler; every line is stamped with host time on arrival."""

    FIELDS = "clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active"

    def __init__(self, interval_ms: int, csv_path: Path | None):
        self.rows: list[tuple[float, list[str]]] = []
        self.csv = open(csv_path, "w") if csv_path else None
        if self.csv:
            self.csv.write("host_time," + self.FIELDS + "\n")
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", f"-lms={interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            t = time.time()
            parts = [p.strip() for p in line.strip().split(",")]
            self.rows.append((t, parts))
            if self.csv:
                self.csv.write(f"{t:.3f}," + ",".join(parts) + "\n")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.thread.join(timeout=5)
        if self.csv:
            self.csv.close()

    def stats(self, t0: float, t1: float) -> dict:
        rows = [p for t, p in self.rows if t0 <= t <= t1]

        def col(i):
            out = []
            for p in rows:
                try:
                    out.append(float(p[i]))
                except (ValueError, IndexError):
                    pass
            return out

        def summary(values):
            if not values:
                return None
            values = sorted(values)
            return {"median": statistics.median(values), "mean": statistics.fmean(values),
                    "min": values[0], "max": values[-1],
                    "p05": values[int(0.05 * (len(values) - 1))], "p95": values[int(0.95 * (len(values) - 1))]}

        throttle = {}
        for p in rows:
            if len(p) > 7:
                throttle[p[7]] = throttle.get(p[7], 0) + 1
        return {"samples": len(rows), "sm_clock_mhz": summary(col(0)), "mem_clock_mhz": summary(col(1)),
                "power_w": summary(col(2)), "power_limit_w": summary(col(3)), "temp_c": summary(col(4)),
                "util_pct": summary(col(5)), "mem_used_mib": summary(col(6)),
                "throttle_reasons_active_hist": throttle}


def proc_mem() -> dict:
    """Host memory of this process: current and peak (high-water) RSS, in MiB."""
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0]] = round(int(line.split()[1]) / 1024, 1)
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                out["MemAvailable_MiB"] = round(int(line.split()[1]) / 1024, 1)
    return out


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def dist(values):
    if not values:
        return None
    return {"median": statistics.median(values), "mean": statistics.fmean(values), "p90": pct(values, 0.90),
            "p99": pct(values, 0.99), "min": min(values), "max": max(values), "n": len(values)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, default=8, help="scheduler padded batch (live: 8)")
    ap.add_argument("--seconds", type=float, default=180.0, help="measured wall duration")
    ap.add_argument("--warmup", type=float, default=20.0, help="warmup wall seconds before measuring")
    ap.add_argument("--capture-dir", default=str(ROOT / "calibration/unet_multi_avatar_20260928"),
                    help="real scheduler UNet captures used as inputs")
    ap.add_argument("--inputs", type=int, default=32, help="distinct input batches cycled through")
    ap.add_argument("--post", choices=["pinned", "repo"], default="pinned",
                    help="pinned: repo uint8 BGR ops + non_blocking copy into a pinned buffer; "
                         "repo: VAE.decode_latents() itself (pageable .cpu(), as the scheduler calls it)")
    ap.add_argument("--stage-sync", choices=["env", "on", "off"], default="env",
                    help="torch.cuda.synchronize after H2D/UNet (HLS_GPU_STAGE_SYNC_TIMING) and after decode "
                         "(MUSETALK_VAE_DECODE_TIMING_SYNC); env = the live defaults (both on)")
    ap.add_argument("--no-live-env", dest="live_env", action="store_false",
                    help="do not layer the live launcher env under the caller env")
    ap.add_argument("--overlay", default="", help="extra env file layered above the live env (e.g. .runtime/musetalk_300fps.env)")
    ap.add_argument("--smi-interval-ms", type=int, default=1000)
    ap.add_argument("--window-s", type=float, default=10.0, help="fps time-series window")
    ap.add_argument("--out", default="", help="JSON report path (a sibling .smi.csv is written too)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    env_report = apply_env(args)

    # Imports after the env is final: musetalk/models/vae.py reads its flags at import.
    import numpy as np
    import torch

    from musetalk.models.vae import VAE
    from scripts.trt_runtime import load_unet_trt_backend, load_vae_trt_decoder

    def env_bool(name, default):
        value = os.getenv(name)
        if value is None or value == "":
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    if args.stage_sync == "env":
        stage_sync = env_bool("HLS_GPU_STAGE_SYNC_TIMING", True)
        decode_sync = env_bool("MUSETALK_VAE_DECODE_TIMING_SYNC", True)
    else:
        stage_sync = decode_sync = args.stage_sync == "on"

    device = torch.device("cuda:0")
    # ParallelAvatarManager._init_models backend flags.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    smi = SmiLogger(args.smi_interval_ms, out_path.with_suffix(".smi.csv") if out_path else None)
    t_load = time.time()
    mem = {"start": proc_mem()}

    # --- models, in ParallelAvatarManager order (VAE wrapper, UNet backend, decode backend)
    vae = VAE(model_path="./models/sd-vae")
    vae.vae = vae.vae.half().to(device).eval()
    vae.vae.requires_grad_(False)
    vae.runtime_dtype = vae.vae.dtype
    vae_dtype = vae.vae.dtype
    mem["after_sd_vae"] = proc_mem()

    unet_backend = load_unet_trt_backend(device=device)
    if unet_backend is not None:
        unet_model = unet_backend
        unet_name = getattr(unet_backend, "name", type(unet_backend).__name__)
    else:
        from scripts.build_unet_multi_avatar_corpus import load_eager_unet
        unet_model = load_eager_unet(device)
        unet_name = "pytorch_eager_fp16"
    mem["after_unet_backend"] = proc_mem()

    vae_backend = load_vae_trt_decoder(device=device, scaling_factor=vae.scaling_factor, vae_module=vae.vae)
    vae.set_decode_backend(vae_backend)
    vae_name = vae.get_decode_backend_name()
    timesteps = torch.tensor([0], device=device)
    load_s = time.time() - t_load
    mem["after_decode_backend"] = proc_mem()
    print(f"backends: unet={unet_name} vae={vae_name} load={load_s:.1f}s stage_sync={stage_sync} "
          f"decode_sync={decode_sync} post={args.post}", flush=True)

    # --- real inputs, pinned like the scheduler staging buffers
    files = sorted(Path(args.capture_dir).glob("unet_io_*.pt"))
    if not files:
        raise SystemExit(f"no captures in {args.capture_dir}")
    step = max(1, len(files) // max(1, args.inputs))
    picked = files[::step][: args.inputs]
    rows_lat, rows_aud, used = [], [], []
    for f in picked:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        rows_lat.append(payload["latent_batch"].to(torch.float16))
        rows_aud.append(payload["audio_feature_batch"].to(torch.float16))
        used.append({"file": f.name, "avatar": (payload.get("items") or [{}])[0].get("avatar_id")})
    all_lat, all_aud = torch.cat(rows_lat), torch.cat(rows_aud)
    n_rows = all_lat.shape[0]
    bs = args.batch
    inputs = []
    for i in range(max(1, n_rows // bs)):
        idx = [(i * bs + j) % n_rows for j in range(bs)]
        inputs.append((all_aud[idx].contiguous().pin_memory(), all_lat[idx].contiguous().pin_memory()))
    pinned_out = torch.empty((bs, 256, 256, 3), dtype=torch.uint8, pin_memory=True)

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def run_batch(i: int):
        cond_cpu, lat_cpu = inputs[i % len(inputs)]
        ev[0].record()
        audio_feature_batch = cond_cpu.to(device, non_blocking=True)
        latent_batch = lat_cpu.to(device=device, dtype=audio_feature_batch.dtype, non_blocking=True)
        if stage_sync:
            torch.cuda.synchronize()
        ev[1].record()
        pred_latents = unet_model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        if stage_sync:
            torch.cuda.synchronize()
        ev[2].record()
        pred_latents = pred_latents.to(device=device, dtype=vae_dtype)
        if args.post == "repo":
            frames = vae.decode_latents(pred_latents)  # decode + (sync) + uint8 BGR + pageable D2H
            ev[3].record()
            ev[4].record()
            ev[4].synchronize()
            return frames
        image = vae.decode_latents_tensor(pred_latents)
        if decode_sync:
            torch.cuda.synchronize(device)
        ev[3].record()
        u8 = (image.detach().float().mul(255).round().clamp_(0, 255).to(torch.uint8)
              .flip(1).permute(0, 2, 3, 1).contiguous())
        pinned_out.copy_(u8, non_blocking=True)
        ev[4].record()
        ev[4].synchronize()  # the host needs the frames before compose, as in the scheduler
        return pinned_out

    with torch.inference_mode():
        # Golden pass: one batch per distinct input, SHA-256 of the uint8 BGR frames.
        golden = []
        for i in range(len(inputs)):
            frames = run_batch(i)
            arr = frames if isinstance(frames, np.ndarray) else frames.numpy()
            golden.append(hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest())
        combined = hashlib.sha256("".join(golden).encode()).hexdigest()

        t_w = time.time()
        i = 0
        while time.time() - t_w < args.warmup:
            run_batch(i)
            i += 1
        warmup_batches = i
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        per = {"total": [], "h2d": [], "unet": [], "vae": [], "post_d2h": [], "host_wall": []}
        stamps = []
        t0 = time.time()
        p0 = time.perf_counter()
        n = 0
        while True:
            h0 = time.perf_counter()
            run_batch(n)
            h1 = time.perf_counter()
            per["host_wall"].append((h1 - h0) * 1e3)
            per["total"].append(ev[0].elapsed_time(ev[4]))
            per["h2d"].append(ev[0].elapsed_time(ev[1]))
            per["unet"].append(ev[1].elapsed_time(ev[2]))
            per["vae"].append(ev[2].elapsed_time(ev[3]))
            per["post_d2h"].append(ev[3].elapsed_time(ev[4]))
            stamps.append(h1 - p0)
            n += 1
            if h1 - p0 >= args.seconds:
                break
        wall = time.perf_counter() - p0
        t1 = time.time()
        mem["end"] = proc_mem()
        max_alloc = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
    smi.stop()

    frames_total = n * bs
    # fps per fixed wall window (batches are attributed to the window they finished in).
    w = args.window_s
    buckets: dict[int, int] = {}
    for s in stamps:
        buckets[int(s // w)] = buckets.get(int(s // w), 0) + 1
    windows = []
    for b in sorted(buckets):
        span = min(w, wall - b * w)
        if span >= 0.5 * w:  # drop a short trailing window
            windows.append({"t_start_s": b * w, "fps": buckets[b] * bs / span})
    median_total = statistics.median(per["total"])
    median_host = statistics.median(per["host_wall"])
    fps_windows = [x["fps"] for x in windows if x["fps"]]

    import torch_tensorrt  # noqa: F401  (version only)
    result = {
        "schema": "bench_gpu_path_v1",
        "label": args.label,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "backends": {"unet": unet_name, "unet_class": type(unet_model).__name__, "vae_decode": vae_name,
                     "taesd_compile_mode": getattr(vae_backend, "compile_mode", None),
                     "taesd_compile_enabled": getattr(vae_backend, "compile_enabled", None),
                     "stage_sync": stage_sync, "decode_sync": decode_sync, "post": args.post,
                     "load_seconds": load_s},
        "env": env_report,
        "versions": {"torch": torch.__version__, "torch_tensorrt": torch_tensorrt.__version__,
                     "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0)},
        "inputs": {"capture_dir": args.capture_dir, "files": used, "distinct_batches": len(inputs)},
        "golden": {"per_input_sha256": golden, "combined_sha256": combined},
        "warmup": {"seconds": args.warmup, "batches": warmup_batches},
        "measured": {
            "batches": n, "frames": frames_total, "wall_s": wall,
            "aggregate_fps": frames_total / wall, "wall_ms_per_frame": wall * 1e3 / frames_total,
            "event_ms_per_batch": dist(per["total"]),
            "event_ms_per_frame_median": median_total / bs,
            "event_fps_from_median": bs * 1e3 / median_total,
            "host_ms_per_batch": dist(per["host_wall"]),
            "host_fps_from_median": bs * 1e3 / median_host,
            "stage_event_ms_median": {k: statistics.median(v) for k, v in per.items() if k not in ("total", "host_wall")},
            "stage_event_ms_p99": {k: pct(v, 0.99) for k, v in per.items() if k not in ("total", "host_wall")},
            "fps_windows": windows,
            "fps_window_min": min(fps_windows) if fps_windows else None,
            "fps_window_max": max(fps_windows) if fps_windows else None,
            "torch_max_allocated_mib": max_alloc, "torch_reserved_mib": reserved,
        },
        "host_memory_mib": mem,
        "gpu_smi": smi.stats(t0, t1),
        "gpu_smi_during_load_and_warmup": smi.stats(t_load, t0),
        "tags": "[M] measured by this run",
    }
    m = result["measured"]
    print(json.dumps({"label": args.label, "backends": result["backends"], "aggregate_fps": m["aggregate_fps"],
                      "wall_ms_per_frame": m["wall_ms_per_frame"], "event_ms_per_batch": m["event_ms_per_batch"],
                      "stage_event_ms_median": m["stage_event_ms_median"], "fps_window_min": m["fps_window_min"],
                      "fps_window_max": m["fps_window_max"], "gpu_smi": result["gpu_smi"],
                      "golden_combined_sha256": combined}, indent=1), flush=True)
    if out_path:
        out_path.write_text(json.dumps(result, indent=1))
        print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
