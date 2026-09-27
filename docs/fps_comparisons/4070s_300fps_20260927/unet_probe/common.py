"""Shared helpers for the 4070S UNet probe (read-only w.r.t. /workspace)."""
import glob
import json
import os
import subprocess
import sys
import time

import torch

REPO = "/workspace/MuseTalk"
OUT = os.path.dirname(os.path.abspath(__file__))
CAPTURES = sorted(glob.glob("/workspace/benchmarks/same-avatar/unet-captures/unet_io_*_bs8_*.pt"))
SERVER_PID = 3240171

sys.path.insert(0, REPO)


def gpu_state(tag, log):
    """Sync, idle 1.2 s, then sample nvidia-smi so our own work is not in the sample."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(1.2)
    samples = []
    for _ in range(3):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        samples.append(out)
        time.sleep(0.3)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    utils = []
    for s in samples:
        try:
            utils.append(int(s.split(",")[0].strip().split()[0]))
        except Exception:
            pass
    foreign = [a for a in apps if a.split(",")[0].strip() not in (str(SERVER_PID), str(os.getpid()))]
    contaminated = (max(utils) if utils else 0) > 5
    rec = {"tag": tag, "samples": samples, "apps": apps, "foreign_apps": foreign, "contaminated": contaminated}
    log.append(rec)
    print(f"[gpu_state {tag}] {samples} foreign={foreign} contaminated={contaminated}", flush=True)
    return rec


def wait_clean(tag, log, retries=6, delay=20):
    """Check GPU idle; if contaminated, wait and retry (records every attempt)."""
    rec = gpu_state(tag, log)
    n = 0
    while rec["contaminated"] and n < retries:
        time.sleep(delay)
        n += 1
        rec = gpu_state(f"{tag}#retry{n}", log)
    return rec


def cuda_time(fn, warmup=10, iters=30):
    """Median/mean/min GPU ms using CUDA events around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "p90_ms": times[int(len(times) * 0.9) - 1],
        "n": len(times),
    }


def load_unet_fp16(device="cuda"):
    from musetalk.models.unet import UNet
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        unet = UNet(unet_config="models/musetalkV15/musetalk.json",
                    model_path="models/musetalkV15/unet.pth", use_float16=True, device=torch.device(device))
    finally:
        os.chdir(cwd)
    unet.model.eval()
    unet.model.requires_grad_(False)
    return unet


def load_captures(n=None):
    lat, aud, pred = [], [], []
    for p in CAPTURES[: n or len(CAPTURES)]:
        d = torch.load(p, map_location="cpu", weights_only=False)
        lat.append(d["latent_batch"])
        aud.append(d["audio_feature_batch"])
        pred.append(d["pred_latents"])
    return torch.cat(lat), torch.cat(aud), torch.cat(pred)


def batch_inputs(bs, device="cuda"):
    """Real captured latents/audio tiled to batch size bs (fp16)."""
    lat, aud, _ = load_captures()
    idx = torch.arange(bs) % lat.shape[0]
    return lat[idx].to(device).half().contiguous(), aud[idx].to(device).half().contiguous()


def save_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print("wrote", path)


def load_unet_fp16_lowmem(device="cuda"):
    """Low host-RAM loader: build the UNet directly as fp16 on the GPU and copy weights from an mmap'd checkpoint
    (page-cache backed), so no 3.4 GB fp32 CPU copy is materialised."""
    from diffusers import UNet2DConditionModel
    cfg = json.load(open(os.path.join(REPO, "models/musetalkV15/musetalk.json")))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device(device):
            model = UNet2DConditionModel(**cfg)
    finally:
        torch.set_default_dtype(prev)
    sd = torch.load(os.path.join(REPO, "models/musetalkV15/unet.pth"), map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    assert not unexpected, unexpected[:5]
    # buffers not in the checkpoint (none expected) stay default
    model = model.half().eval().requires_grad_(False)
    return model, missing
