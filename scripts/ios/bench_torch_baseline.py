#!/usr/bin/env python
"""PyTorch baseline for the MuseTalk per-frame path on an Apple Silicon Mac.

Measures, per batch size, what the live server does per frame once an avatar is
prepared: positional encoding + UNet (one step) + TAESD decode. Also times the
Whisper-tiny encoder once per 30 s audio segment, since that is the only other
model on the live path. Face analysis is offline (avatar prep) and not timed.

Inputs are real when an avatar cache and an audio file are given (the same
latents.pt and Whisper windows the server uses); otherwise random tensors of the
same shapes. The fixed 16-frame input set and the fp32 CPU reference outputs are
saved so the Core ML numeric check compares against the same frames.

Usage (from the repo root, inside the Python 3.12 venv):
  python scripts/ios/bench_torch_baseline.py \
      --avatar-latents /path/to/avatar/latents.pt --audio data/audio/ai-assistant.mpga \
      --out docs/ios_unet_bench/baseline_torch.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from musetalk.models.unet import PositionalEncoding, UNet  # noqa: E402

LATENT_SHAPE = (8, 32, 32)
AUDIO_SHAPE = (50, 384)
REF_FRAMES = 16


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def host_info() -> dict:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)
    return {
        "chip": chip,
        "ram_gb": round(mem / 2**30, 1),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def load_taesd(device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderTiny

    model = AutoencoderTiny.from_pretrained(str(REPO_ROOT / "models" / "taesd"), torch_dtype=dtype)
    return model.to(device).eval().requires_grad_(False)


def taesd_decode(model, latents: torch.Tensor) -> torch.Tensor:
    # Same convention as scripts/vae_fast_decoder.py::_raw_decode: TAESD takes the
    # scaled latent the UNet emits and returns [-1, 1].
    image = model.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


def real_inputs(avatar_latents: Path, audio: Path, whisper_dir: Path, device: torch.device, dtype: torch.dtype):
    from transformers import WhisperModel

    from musetalk.utils.audio_processor import AudioProcessor

    latent_list = torch.load(avatar_latents, map_location="cpu", weights_only=True)
    latents = torch.cat([t.float() for t in latent_list[:REF_FRAMES]], dim=0)
    if latents.shape[0] < REF_FRAMES:
        reps = -(-REF_FRAMES // latents.shape[0])
        latents = latents.repeat(reps, 1, 1, 1)[:REF_FRAMES]

    processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
    whisper = WhisperModel.from_pretrained(str(whisper_dir)).to(device=device, dtype=dtype).eval()
    features, librosa_length = processor.get_audio_feature(str(audio), weight_dtype=dtype)
    # Time the encoder on the first 30 s segment: this is the per-turn audio cost.
    with torch.inference_mode():
        seg = features[0].to(device=device, dtype=dtype)
        for _ in range(2):
            whisper.encoder(seg, output_hidden_states=True)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(5):
            whisper.encoder(seg, output_hidden_states=True)
        sync(device)
        whisper_ms = (time.perf_counter() - t0) / 5 * 1000
        chunks = processor.get_whisper_chunk(features, device, dtype, whisper, librosa_length, fps=25)
    audio_windows = chunks[:REF_FRAMES].float().cpu()
    return latents, audio_windows, {"whisper_encoder_30s_segment_ms": round(whisper_ms, 2), "audio_frames": int(chunks.shape[0])}


def random_inputs():
    g = torch.Generator().manual_seed(0)
    latents = torch.randn((REF_FRAMES, *LATENT_SHAPE), generator=g)
    audio = torch.randn((REF_FRAMES, *AUDIO_SHAPE), generator=g) * 0.5
    return latents, audio, {"whisper_encoder_30s_segment_ms": None, "audio_frames": None}


def bench(unet, pe, taesd, latents, audio, device, dtype, batch, iters, warmup) -> dict:
    timesteps = torch.tensor([0], device=device)
    idx = torch.arange(batch) % latents.shape[0]
    lat = latents[idx].to(device=device, dtype=dtype)
    aud = audio[idx].to(device=device, dtype=dtype)
    unet_ms, dec_ms, step_ms = [], [], []
    with torch.inference_mode():
        for i in range(warmup + iters):
            sync(device)
            t0 = time.perf_counter()
            hidden = pe(aud).to(dtype)  # the PE table is fp32; the server casts pe to half
            pred = unet(lat, timesteps, encoder_hidden_states=hidden).sample
            sync(device)
            t1 = time.perf_counter()
            faces = taesd_decode(taesd, pred)
            sync(device)
            t2 = time.perf_counter()
            if i >= warmup:
                unet_ms.append((t1 - t0) * 1000)
                dec_ms.append((t2 - t1) * 1000)
                step_ms.append((t2 - t0) * 1000)
    med = statistics.median
    return {
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "iters": iters,
        "unet_ms_per_call": round(med(unet_ms), 2),
        "unet_ms_per_frame": round(med(unet_ms) / batch, 3),
        "taesd_ms_per_call": round(med(dec_ms), 2),
        "taesd_ms_per_frame": round(med(dec_ms) / batch, 3),
        "step_ms_per_frame": round(med(step_ms) / batch, 3),
        "step_fps_single_stream": round(1000 / (med(step_ms) / batch), 1),
        "step_ms_p90_per_frame": round(sorted(step_ms)[int(0.9 * (len(step_ms) - 1))] / batch, 3),
        "faces_shape": list(faces.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet-dir", default=str(REPO_ROOT / "models" / "musetalkV15"))
    ap.add_argument("--whisper-dir", default=str(REPO_ROOT / "models" / "whisper"))
    ap.add_argument("--avatar-latents", type=Path, default=None, help="an avatar's latents.pt (real inputs)")
    ap.add_argument("--audio", type=Path, default=None, help="an audio file for real Whisper windows")
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cpu-iters", type=int, default=5)
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "baseline_torch.json")
    ap.add_argument("--inputs-out", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "bench_inputs.pt")
    args = ap.parse_args()

    mps = torch.device("mps") if torch.backends.mps.is_available() else None
    cpu = torch.device("cpu")
    if mps is None:
        print("MPS not available; only the CPU baseline will run", file=sys.stderr)

    print("loading UNet (fp32 weights, 3.4 GB)...", flush=True)
    t0 = time.time()
    unet_wrap = UNet(unet_config=os.path.join(args.unet_dir, "musetalk.json"),
                     model_path=os.path.join(args.unet_dir, "unet.pth"), device=cpu)
    unet32 = unet_wrap.model.eval().requires_grad_(False)
    pe = unet_wrap.pe.eval()
    n_params = sum(p.numel() for p in unet32.parameters())
    print(f"UNet loaded in {time.time() - t0:.1f}s, {n_params / 1e6:.1f}M params", flush=True)

    feature_device = mps or cpu
    feature_dtype = torch.float16 if mps is not None else torch.float32
    if args.avatar_latents and args.audio:
        latents, audio, extra = real_inputs(args.avatar_latents, args.audio, Path(args.whisper_dir), feature_device, feature_dtype)
        source = {"latents": str(args.avatar_latents), "audio": str(args.audio)}
    else:
        latents, audio, extra = random_inputs()
        source = {"latents": "random", "audio": "random"}
    print(f"inputs: latents {tuple(latents.shape)} audio {tuple(audio.shape)} from {source}", flush=True)

    # fp32 CPU reference on the fixed 16 frames, kept for the Core ML numeric check.
    taesd32 = load_taesd(cpu, torch.float32)
    with torch.inference_mode():
        ref_pred = unet32(latents, torch.tensor([0]), encoder_hidden_states=pe(audio)).sample
        ref_faces = taesd_decode(taesd32, ref_pred)
    args.inputs_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "audio": audio, "ref_pred_latents": ref_pred, "ref_faces": ref_faces,
                "source": source}, args.inputs_out)
    print(f"saved reference inputs/outputs to {args.inputs_out}", flush=True)

    results = []
    batches = [int(b) for b in args.batches.split(",")]
    if mps is not None:
        unet16 = unet32.to(device=mps, dtype=torch.float16)
        pe16 = pe.to(mps)
        taesd16 = load_taesd(mps, torch.float16)
        for b in batches:
            r = bench(unet16, pe16, taesd16, latents, audio, mps, torch.float16, b, args.iters, 5)
            print(json.dumps(r), flush=True)
            results.append(r)
        # fp32 on the GPU at one batch, to show the fp16 gain.
        unet32m = unet16.to(torch.float32)
        taesd32m = taesd16.to(torch.float32)
        r = bench(unet32m, pe16, taesd32m, latents, audio, mps, torch.float32, 4, max(5, args.iters // 3), 3)
        print(json.dumps(r), flush=True)
        results.append(r)
        unet32 = unet32m.to(cpu)
        taesd32 = taesd32m.to(cpu)
        pe = pe16.to(cpu)
    if not args.skip_cpu:
        for b in [1, 4]:
            r = bench(unet32, pe, taesd32, latents, audio, cpu, torch.float32, b, args.cpu_iters, 1)
            print(json.dumps(r), flush=True)
            results.append(r)

    out = {"host": host_info(), "unet_params_m": round(n_params / 1e6, 1), "inputs": source, **extra,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
