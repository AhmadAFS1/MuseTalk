#!/usr/bin/env python3
"""GPU probe for startup-rework component A (scripts/musetalk_host_profile.py).

Loads exactly the UNet and VAE-decode backends a resolved env selects, through the SAME loader functions
scripts/avatar_manager_parallel.py calls (trt_runtime.load_vae_trt_decoder / load_unet_trt_backend), and
prints the SAME two log lines the server prints, so `musetalk_host_profile.py verify-log` can be checked
against real backend names on real hardware without starting a server:

    ✅ VAE decode backend active: <name>    |  ℹ️  VAE decode backend: PyTorch
    ✅ UNet backend active: <name>          |  ℹ️  UNet backend: PyTorch

Then one forward per scheduler bucket (finite-output check) and a JSON record (load seconds, host VmHWM,
peak CUDA memory). Imports torch: run it ONLY under scripts/box_guard.sh (see gpu_sequence.sh).
Written by a CPU-only agent; not executed yet.

usage: backend_probe.py --repo-root R --env RESOLVED_ENV --log OUT.log --json OUT.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path


def _load_host_profile(repo_root: Path):
    spec = importlib.util.spec_from_file_location("musetalk_host_profile",
                                                  str(repo_root / "scripts" / "musetalk_host_profile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--env", required=True, help="resolved env file (loaded only-if-unset)")
    parser.add_argument("--log", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    hp = _load_host_profile(repo_root)
    loaded = {}
    for key, value, _lineno in hp.parse_env_file(args.env):
        if key not in os.environ:  # same only-if-unset rule as the launcher
            os.environ[key] = value
            loaded[key] = value
    sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    record = {"schema": "musetalk_backend_probe_v1", "env_file": args.env, "loaded_keys": sorted(loaded),
              "vae": None, "unet": None, "forward": [], "errors": []}
    log = open(args.log, "w", encoding="utf-8")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    started = time.time()
    try:
        import torch  # noqa: F401  (GPU step: allowed here, never in the resolver)
        from scripts.trt_runtime import load_unet_trt_backend, load_vae_trt_decoder

        device = torch.device("cuda:0")
        record["device"] = torch.cuda.get_device_name(device)
        t0 = time.time()
        vae_backend = load_vae_trt_decoder(device=device, scaling_factor=0.18215, vae_module=None)
        record["vae"] = {"name": getattr(vae_backend, "name", "pytorch") if vae_backend is not None else "pytorch",
                         "load_s": round(time.time() - t0, 2)}
        if vae_backend is not None:
            say(f"✅ VAE decode backend active: {record['vae']['name']}")
        else:
            say("ℹ️  VAE decode backend: PyTorch")
        t0 = time.time()
        unet_backend = load_unet_trt_backend(device=device)
        record["unet"] = {"name": getattr(unet_backend, "name", "tensorrt_unet") if unet_backend is not None
                          else "pytorch", "load_s": round(time.time() - t0, 2)}
        if unet_backend is not None:
            say(f"✅ UNet backend active: {record['unet']['name']}")
        else:
            say("ℹ️  UNet backend: PyTorch")
        if not args.no_forward:
            buckets = hp.parse_buckets(os.environ.get("HLS_SCHEDULER_FIXED_BATCH_SIZES", "8"))
            for batch in buckets:
                item = {"batch": batch}
                with torch.no_grad():
                    if unet_backend is not None:
                        latent = torch.randn(batch, 8, 32, 32, device=device, dtype=torch.float16)
                        audio = torch.randn(batch, 50, 384, device=device, dtype=torch.float16)
                        t0 = time.time()
                        out = unet_backend(latent, None, encoder_hidden_states=audio).sample
                        torch.cuda.synchronize(device)
                        item["unet_ms"] = round((time.time() - t0) * 1000, 2)
                        item["unet_finite"] = bool(torch.isfinite(out).all())
                    if vae_backend is not None:
                        latents = torch.randn(batch, 4, 32, 32, device=device, dtype=torch.float16)
                        t0 = time.time()
                        image = vae_backend.decode(latents, 0.18215)
                        torch.cuda.synchronize(device)
                        item["vae_ms"] = round((time.time() - t0) * 1000, 2)
                        item["vae_finite"] = bool(torch.isfinite(image).all())
                        item["vae_shape"] = list(image.shape)
                record["forward"].append(item)
                say(f"probe forward bs={batch}: {json.dumps(item)}")
        record["cuda_max_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)
    except Exception as exc:  # recorded, then exit 1
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        say(f"probe ERROR {type(exc).__name__}: {exc}")
        traceback.print_exc()
    record["seconds"] = round(time.time() - started, 2)
    record["vmhwm_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    finite = all(v for item in record["forward"] for k, v in item.items() if k.endswith("_finite"))
    record["ok"] = not record["errors"] and finite
    Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
    log.close()
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
