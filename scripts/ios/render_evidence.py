#!/usr/bin/env python
"""Render labelled video evidence for the Core ML MuseTalk builds against the PyTorch reference.

For one prepared avatar and one audio file, renders the same N-second turn through:

  ref_fp32_sdvae   PyTorch fp32 UNet + SD-VAE decoder (stock MuseTalk inference look)
  ref_fp32_taesd   same UNet latents, TAESD decoder (the production server's decoder)
  torch_mps_fp16   PyTorch fp16 on the Mac GPU + TAESD (the baseline measured earlier)
  coreml_fp16      Core ML fp16 UNet + Core ML TAESD on the Neural Engine
  coreml_int8w     Core ML int8-weight UNet + Core ML TAESD on the Neural Engine
  coreml_int8wa    Core ML int8 weight+activation UNet + Core ML TAESD on the Neural Engine

Every variant uses the repo's own audio windows (musetalk.utils.audio_processor), the
avatar's cached latents, frames, masks and coords, and the repo's blending
(musetalk.utils.blending.get_image_blending), i.e. the realtime_inference.py loop minus
preprocessing. Outputs: one full-frame mp4 per variant with the audio muxed in, a
side-by-side face-crop grid video, a full-frame comparison video, still frames for the
docs, and a metrics JSON (face-crop and full-frame mean abs error and PSNR against
ref_fp32_taesd, plus render ms per frame).

Usage (repo root, the torch-2.7 venv):
  python scripts/ios/render_evidence.py --avatar /path/to/avatars/codex_smoke \
      --audio data/audio/ai-assistant.mpga --seconds 10 --fps 25
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from musetalk.models.unet import UNet  # noqa: E402
from musetalk.utils.blending import get_image_blending  # noqa: E402

COREML_DIR = REPO_ROOT / "models" / "ios_bench" / "coreml"
SCALING_FACTOR = 0.18215  # sd-vae-ft-mse, as musetalk/models/vae.py applies it


# ----------------------------------------------------------------------------- inputs
def load_avatar(avatar_dir: Path):
    def numbered(folder):
        files = glob.glob(str(folder / "*.[jpJP][pnPN]*[gG]"))
        return sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    frames = [cv2.imread(p) for p in numbered(avatar_dir / "full_imgs")]
    masks = [cv2.imread(p) for p in numbered(avatar_dir / "mask")]
    with open(avatar_dir / "coords.pkl", "rb") as f:
        coords = pickle.load(f)
    with open(avatar_dir / "mask_coords.pkl", "rb") as f:
        mask_coords = pickle.load(f)
    latents = torch.load(avatar_dir / "latents.pt", map_location="cpu", weights_only=True)
    latents = torch.cat([t.float() for t in latents], dim=0)  # [N,8,32,32]
    n = min(len(frames), len(masks), len(coords), len(mask_coords), latents.shape[0])
    return frames[:n], masks[:n], coords[:n], mask_coords[:n], latents[:n]


def audio_windows(audio: Path, whisper_dir: Path, fps: int) -> torch.Tensor:
    from transformers import WhisperModel

    from musetalk.utils.audio_processor import AudioProcessor

    device, dtype = torch.device("cpu"), torch.float32
    processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
    whisper = WhisperModel.from_pretrained(str(whisper_dir)).to(device=device, dtype=dtype).eval()
    features, librosa_length = processor.get_audio_feature(str(audio), weight_dtype=dtype)
    with torch.inference_mode():
        chunks = processor.get_whisper_chunk(features, device, dtype, whisper, librosa_length, fps=fps)
    return chunks.float()  # [frames, 50, 384]


# ---------------------------------------------------------------------------- decoders
def to_bgr_u8(image01: torch.Tensor | np.ndarray) -> np.ndarray:
    """[B,3,256,256] in [0,1] RGB -> list-able [B,256,256,3] uint8 BGR (vae.py convention)."""
    x = torch.as_tensor(np.asarray(image01) if not torch.is_tensor(image01) else image01).float()
    x = x.mul(255).round().clamp_(0, 255).to(torch.uint8).flip(1).permute(0, 2, 3, 1).contiguous()
    return x.cpu().numpy()


def taesd_torch(device, dtype):
    from diffusers import AutoencoderTiny

    m = AutoencoderTiny.from_pretrained(str(REPO_ROOT / "models" / "taesd"), torch_dtype=dtype)
    return m.to(device).eval().requires_grad_(False)


def sdvae_torch(device, dtype):
    from diffusers import AutoencoderKL

    m = AutoencoderKL.from_pretrained(str(REPO_ROOT / "models" / "sd-vae"), torch_dtype=dtype)
    return m.to(device).eval().requires_grad_(False)


# ----------------------------------------------------------------------------- variants
def run_torch(unet, pe, decoder, decoder_kind, lat, aud, device, dtype, batch=1):
    """Returns (faces uint8 BGR [N,256,256,3], pred latents fp32 [N,4,32,32], ms/frame)."""
    unet = unet.to(device=device, dtype=dtype)
    pe = pe.to(device)
    timesteps = torch.tensor([0], device=device)
    faces, preds, t_total = [], [], 0.0
    with torch.inference_mode():
        for s in range(0, lat.shape[0], batch):
            l = lat[s:s + batch].to(device=device, dtype=dtype)
            a = aud[s:s + batch].to(device=device, dtype=dtype)
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            pred = unet(l, timesteps, encoder_hidden_states=pe(a).to(dtype)).sample
            if decoder_kind == "sdvae":
                img = decoder.decode((pred / SCALING_FACTOR).to(decoder.dtype)).sample
            else:
                img = decoder.decode(pred.to(decoder.dtype)).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            if device.type == "mps":
                torch.mps.synchronize()
            t_total += time.perf_counter() - t0
            faces.append(to_bgr_u8(img))
            preds.append(pred.float().cpu())
    return np.concatenate(faces), torch.cat(preds), t_total / lat.shape[0] * 1000


def run_coreml(variant: str, lat, aud):
    import coremltools as ct

    units = ct.ComputeUnit.CPU_AND_NE
    unet = ct.models.MLModel(str(COREML_DIR / f"unet_{variant}_b1.mlpackage"), compute_units=units)
    taesd = ct.models.MLModel(str(COREML_DIR / "taesd_fp16_b1.mlpackage"), compute_units=units)
    feed0 = {"sample": lat[:1].numpy().astype(np.float16), "audio": aud[:1].numpy().astype(np.float16)}
    for _ in range(5):
        taesd.predict({"latents": np.asarray(unet.predict(feed0)["pred_latents"], dtype=np.float16)})
    faces, preds, t_total = [], [], 0.0
    for i in range(lat.shape[0]):
        feed = {"sample": lat[i:i + 1].numpy().astype(np.float16), "audio": aud[i:i + 1].numpy().astype(np.float16)}
        t0 = time.perf_counter()
        pred = np.asarray(unet.predict(feed)["pred_latents"], dtype=np.float32)
        img = np.asarray(taesd.predict({"latents": pred.astype(np.float16)})["faces"], dtype=np.float32)
        t_total += time.perf_counter() - t0
        faces.append(to_bgr_u8(img))
        preds.append(torch.from_numpy(pred))
    return np.concatenate(faces), torch.cat(preds), t_total / lat.shape[0] * 1000


# ------------------------------------------------------------------------------ compose
def compose(faces: np.ndarray, frames, masks, coords, mask_coords) -> list[np.ndarray]:
    out = []
    n_cycle = len(frames)
    for i in range(faces.shape[0]):
        j = i % n_cycle
        x1, y1, x2, y2 = coords[j]
        face = cv2.resize(faces[i].astype(np.uint8), (x2 - x1, y2 - y1))
        out.append(get_image_blending(frames[j].copy(), face, coords[j], masks[j], mask_coords[j]))
    return out


def face_crops(composed: list[np.ndarray], coords, size=256) -> np.ndarray:
    crops = []
    for i, img in enumerate(composed):
        x1, y1, x2, y2 = coords[i % len(coords)]
        crops.append(cv2.resize(img[y1:y2, x1:x2], (size, size)))
    return np.stack(crops)


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def write_video(frames: list[np.ndarray], path: Path, fps: int, audio: Path | None, seconds: float):
    tmp = path.with_suffix(".raw.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp)]
    if audio is not None:
        cmd += ["-i", str(audio), "-t", f"{seconds:.3f}", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, check=True)
    tmp.unlink(missing_ok=True)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    mse = float((d ** 2).mean())
    return {"mae_255": round(float(d.mean()), 3), "max_abs_255": round(float(d.max()), 1),
            "psnr_db": round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else None}


# --------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--variants", default="ref_fp32_sdvae,ref_fp32_taesd,torch_mps_fp16,coreml_fp16,coreml_int8w,coreml_int8wa")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "evidence")
    ap.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "evidence")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")

    frames, masks, coords, mask_coords, av_latents = load_avatar(args.avatar)
    aud_all = audio_windows(args.audio, args.whisper_dir, args.fps)
    n = min(int(round(args.seconds * args.fps)), aud_all.shape[0])
    seconds = n / args.fps
    aud = aud_all[:n]
    lat = torch.stack([av_latents[i % av_latents.shape[0]] for i in range(n)])  # datagen's cycle
    print(f"avatar {args.avatar.name}: {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}; "
          f"turn: {n} frames = {seconds:.2f} s at {args.fps} fps", flush=True)

    print("loading UNet fp32...", flush=True)
    w = UNet(unet_config=str(args.unet_dir / "musetalk.json"), model_path=str(args.unet_dir / "unet.pth"),
             device=torch.device("cpu"))
    unet, pe = w.model.eval().requires_grad_(False), w.pe.eval()
    cpu = torch.device("cpu")
    mps = torch.device("mps") if torch.backends.mps.is_available() else None

    results: dict[str, dict] = {}
    composed: dict[str, list[np.ndarray]] = {}
    ref_pred = None
    for v in variants:
        t0 = time.time()
        if v == "ref_fp32_sdvae":
            faces, ref_pred, ms = run_torch(unet, pe, sdvae_torch(cpu, torch.float32), "sdvae", lat, aud, cpu, torch.float32, batch=4)
        elif v == "ref_fp32_taesd":
            if ref_pred is None:
                _, ref_pred, _ = run_torch(unet, pe, taesd_torch(cpu, torch.float32), "taesd", lat, aud, cpu, torch.float32, batch=4)
            dec = taesd_torch(cpu, torch.float32)
            with torch.inference_mode():
                img = (dec.decode(ref_pred).sample / 2 + 0.5).clamp(0, 1)
            faces, ms = to_bgr_u8(img), None
        elif v == "torch_mps_fp16":
            if mps is None:
                print("  skipping torch_mps_fp16: no MPS", flush=True)
                continue
            faces, _, ms = run_torch(unet, pe, taesd_torch(mps, torch.float16), "taesd", lat, aud, mps, torch.float16, batch=1)
            unet = unet.to(device=cpu, dtype=torch.float32)
            pe = pe.to(cpu)
        elif v.startswith("coreml_"):
            faces, _, ms = run_coreml(v.split("_", 1)[1], lat, aud)
        else:
            raise SystemExit(f"unknown variant {v}")
        composed[v] = compose(faces, frames, masks, coords, mask_coords)
        write_video(composed[v], args.out_dir / f"{v}.mp4", args.fps, args.audio, seconds)
        results[v] = {"render_ms_per_frame": round(ms, 2) if ms is not None else None, "wall_s": round(time.time() - t0, 1)}
        print(f"  {v}: {ms if ms is None else round(ms, 1)} ms/frame, {time.time() - t0:.0f} s wall -> {v}.mp4", flush=True)

    ref_name = "ref_fp32_taesd" if "ref_fp32_taesd" in composed else variants[0]
    ref_full = np.stack(composed[ref_name])
    ref_crop = face_crops(composed[ref_name], coords)
    for v, frs in composed.items():
        full = np.stack(frs)
        crop = face_crops(frs, coords)
        results[v]["vs_" + ref_name] = {"face_crop": metrics(crop, ref_crop), "full_frame": metrics(full, ref_full)}
        if "ref_fp32_sdvae" in composed and v != "ref_fp32_sdvae":
            results[v]["vs_ref_fp32_sdvae"] = {"face_crop": metrics(crop, face_crops(composed["ref_fp32_sdvae"], coords))}

    # Side-by-side face-crop grid and full-frame comparison videos, plus stills.
    names = list(composed)
    crops = {v: face_crops(composed[v], coords) for v in names}
    grid = [np.hstack([label(crops[v][i], v) for v in names]) for i in range(n)]
    write_video(grid, args.out_dir / "grid_face_crops.mp4", args.fps, args.audio, seconds)
    cmp_names = [x for x in [ref_name, "coreml_fp16", "coreml_int8w"] if x in composed]
    h = 540
    def fit(img):
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    full_cmp = [np.hstack([label(fit(composed[v][i]), v) for v in cmp_names]) for i in range(n)]
    write_video(full_cmp, args.out_dir / "full_frame_ref_vs_coreml.mp4", args.fps, args.audio, seconds)
    for sec in (2, 5, 8):
        i = min(int(sec * args.fps), n - 1)
        cv2.imwrite(str(args.docs_dir / f"face_grid_{sec}s.png"), grid[i])
        cv2.imwrite(str(args.docs_dir / f"full_frame_{sec}s.png"), full_cmp[i])
        if "coreml_fp16" in composed:
            d = np.abs(crops["coreml_fp16"][i].astype(np.int16) - ref_crop[i].astype(np.int16)).max(axis=2)
            heat = cv2.applyColorMap(np.clip(d * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(args.docs_dir / f"diff_x8_coreml_fp16_{sec}s.png"), label(heat, "|coreml_fp16 - ref| x8"))

    out = {"avatar": str(args.avatar), "audio": str(args.audio), "fps": args.fps, "frames": n, "seconds": seconds,
           "reference": ref_name, "videos_dir": str(args.out_dir), "results": results}
    (args.docs_dir / "evidence_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["results"], indent=1))
    print("EVIDENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
