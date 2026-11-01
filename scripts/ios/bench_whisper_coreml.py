#!/usr/bin/env python
"""Bench and validate the Core ML Whisper front end, alone and through the UNet.

1. Time `whisper_logmel_fp32` and `whisper_encoder_fp16` on each compute unit and
   show where the encoder's ops land (Neural Engine or not).
2. Numerics: Core ML features vs the PyTorch fp32 features on the real clip.
3. The check that matters: the first `--seconds` of the clip on a real avatar,
   rendered through the Core ML int8w UNet + Core ML TAESD, once with PyTorch
   features and once with the all-Core-ML features, both against the fp32
   PyTorch reference (fp32 Whisper, fp32 UNet, fp32 TAESD). Writes the
   all-Core-ML full-chain video next to the other evidence.

Usage (repo root, .venv-ct):
  python scripts/ios/bench_whisper_coreml.py --avatar <avatar_dir> --audio data/audio/ai-assistant.mpga
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from bench_coreml import compute_plan_summary  # noqa: E402
from render_evidence import (COREML_DIR, compose, face_crops, load_avatar, metrics, run_coreml,  # noqa: E402
                             run_torch, taesd_torch, write_video)
from musetalk.models.unet import UNet  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor, pad_whisper_feature_for_musetalk  # noqa: E402

SAMPLES_30S = 30 * 16000
UNITS = {"ane": "CPU_AND_NE", "cpu": "CPU_ONLY", "all": "ALL"}


def timed(model, feed, warmup=3, iters=20):
    for _ in range(warmup):
        model.predict(feed)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed)
        t.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(t)), 2), "p99_ms": round(float(np.percentile(t, 99)), 2),
            "max_ms": round(float(np.max(t)), 2)}


def windows_from_features(features: torch.Tensor, processor: AudioProcessor, librosa_length: int, fps: int):
    """The tail of AudioProcessor.encode_whisper_feature + build_audio_prompts, on a [1,1500,5,384] stack."""
    num_frames = int(np.floor(librosa_length / 16000 * fps))
    actual = int(np.floor(librosa_length / 16000 * 50))
    feat = pad_whisper_feature_for_musetalk(features[:, :actual].float())
    return processor.build_audio_prompts(whisper_feature=feat, num_frames=num_frames, fps=fps)


def main() -> None:
    import coremltools as ct
    import librosa
    from transformers import WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", type=Path, required=True)
    ap.add_argument("--audio", type=Path, default=REPO_ROOT / "data" / "audio" / "ai-assistant.mpga")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--unet-variant", default="int8w")
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "whisper_coreml_m5pro.json")
    ap.add_argument("--video-dir", type=Path, default=None)
    args = ap.parse_args()
    manifest = json.loads((COREML_DIR / "manifest.json").read_text())
    result = {"audio": str(args.audio), "avatar": args.avatar.name, "seconds": args.seconds, "fps": args.fps}

    wav, _ = librosa.load(str(args.audio), sr=16000)
    seg = np.zeros(SAMPLES_30S, dtype=np.float32)
    seg[:min(len(wav), SAMPLES_30S)] = wav[:SAMPLES_30S]
    librosa_length = min(len(wav), args.seconds * 16000)  # the first N seconds, like render_evidence

    # --- 1. timing + placement -------------------------------------------------
    result["timing"] = {}
    for name in ("whisper_logmel_fp32", "whisper_encoder_fp16"):
        path = manifest[name]["path"]
        result["timing"][name] = {}
        for key, units in UNITS.items():
            model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, units))
            if name == "whisper_logmel_fp32":
                feed = {"waveform": seg[None]}
            else:
                mel = np.asarray(ct.models.MLModel(manifest["whisper_logmel_fp32"]["path"],
                                                   compute_units=ct.ComputeUnit.CPU_ONLY).predict(
                    {"waveform": seg[None]})["logmel"], dtype=np.float16)
                feed = {"logmel": mel}
            t = timed(model, feed)
            result["timing"][name][key] = t
            print(f"{name} [{units}]: {t}", flush=True)
        plan = compute_plan_summary(path, "CPU_AND_NE")
        result["timing"][name]["compute_plan_ane"] = plan
        print(f"  compute plan (CPU_AND_NE): {plan}", flush=True)

    # --- 2. features: Core ML vs torch fp32 -----------------------------------
    processor = AudioProcessor(feature_extractor_path=str(args.whisper_dir))
    whisper = WhisperModel.from_pretrained(str(args.whisper_dir), attn_implementation="eager").eval()
    ref_mel = processor.feature_extractor(seg, return_tensors="pt", sampling_rate=16000).input_features
    with torch.inference_mode():
        ref_feat = torch.stack(whisper.encoder(ref_mel, output_hidden_states=True).hidden_states, dim=2)
    logmel_cm = ct.models.MLModel(manifest["whisper_logmel_fp32"]["path"], compute_units=ct.ComputeUnit.CPU_ONLY)
    enc_cm = ct.models.MLModel(manifest["whisper_encoder_fp16"]["path"], compute_units=ct.ComputeUnit.CPU_AND_NE)
    t0 = time.perf_counter()
    mel_cm = np.asarray(logmel_cm.predict({"waveform": seg[None]})["logmel"], dtype=np.float32)
    feat_cm = np.asarray(enc_cm.predict({"logmel": mel_cm.astype(np.float16)})["features"], dtype=np.float32)
    chain_ms = (time.perf_counter() - t0) * 1000
    feat_cm = torch.from_numpy(feat_cm)

    # Where the feature error comes from: the encoder's compute unit (fp16 on the Neural
    # Engine vs the GPU) crossed with the log-mel source (the HF extractor vs our Core ML model).
    enc_gpu = ct.models.MLModel(manifest["whisper_encoder_fp16"]["path"], compute_units=ct.ComputeUnit.ALL)
    grid = {}
    for enc_name, enc in (("ane", enc_cm), ("gpu", enc_gpu)):
        for mel_name, mel in (("hf_mel", ref_mel.numpy()), ("coreml_mel", mel_cm)):
            f = torch.from_numpy(np.asarray(enc.predict({"logmel": mel.astype(np.float16)})["features"], dtype=np.float32))
            e = (f - ref_feat).abs()
            grid[f"{enc_name}_{mel_name}"] = {"max_abs_err": round(float(e.max()), 3), "mean_abs_err": round(float(e.mean()), 5)}
    d = (feat_cm - ref_feat).abs()
    result["features"] = {"max_abs_err": round(float(d.max()), 4), "mean_abs_err": round(float(d.mean()), 5),
                          "ref_mean_abs": round(float(ref_feat.abs().mean()), 4),
                          "ref_max_abs": round(float(ref_feat.abs().max()), 2),
                          "logmel_max_abs_err": round(float(np.abs(mel_cm - ref_mel.numpy()).max()), 6),
                          "front_end_ms_cold_call": round(chain_ms, 1), "error_grid_vs_fp32": grid}
    print(f"features: {result['features']}", flush=True)

    aud_ref = windows_from_features(ref_feat, processor, librosa_length, args.fps)
    aud_cm = windows_from_features(feat_cm, processor, librosa_length, args.fps)
    n_frames = aud_ref.shape[0]
    print(f"{n_frames} frames of windows; window max |diff| {float((aud_ref - aud_cm).abs().max()):.4f}", flush=True)

    # --- 3. through the UNet on a real avatar ---------------------------------
    frames, masks, coords, mask_coords, latents = load_avatar(args.avatar)
    lat = latents[torch.arange(n_frames) % latents.shape[0]]

    print("reference: fp32 Whisper -> fp32 UNet (CPU) -> fp32 TAESD", flush=True)
    wrap = UNet(unet_config=os.path.join(args.unet_dir, "musetalk.json"),
                model_path=os.path.join(args.unet_dir, "unet.pth"), device=torch.device("cpu"))
    unet, pe = wrap.model.eval(), wrap.pe
    faces_ref, pred_ref, ms_ref = run_torch(unet, pe, taesd_torch(torch.device("cpu"), torch.float32), "taesd",
                                            lat, aud_ref, torch.device("cpu"), torch.float32, batch=4)
    print(f"  {ms_ref:.0f} ms/frame", flush=True)

    print(f"Core ML {args.unet_variant} UNet with torch fp32 features", flush=True)
    faces_a, pred_a, ms_a = run_coreml(args.unet_variant, lat, aud_ref)
    print(f"Core ML {args.unet_variant} UNet with Core ML features (full chain)", flush=True)
    faces_b, pred_b, ms_b = run_coreml(args.unet_variant, lat, aud_cm)

    # Two views of the same frames: the raw 256x256 decoder output (everything the UNet
    # changed) and, as in render_evidence, the face crop of the composed frame (what the
    # viewer sees: only the blended mouth region differs from the avatar frame).
    composed = {k: compose(f, frames, masks, coords, mask_coords) for k, f in
                (("ref", faces_ref), ("a", faces_a), ("b", faces_b))}
    crops = {k: face_crops(c, coords) for k, c in composed.items()}

    def report(tag, faces, crop, pred, ms):
        d = (pred - pred_ref).abs()
        m = {"raw_face": metrics(faces, faces_ref), "composed_face_crop": metrics(crop, crops["ref"]),
             "latent_vs_fp32": {"max_abs": round(float(d.max()), 3), "p99_9_abs": round(float(d.flatten().quantile(0.999)), 3),
                                "frac_over_0_5": round(float((d > 0.5).float().mean()), 6)},
             "ms_per_frame": round(ms, 2)}
        print(f"  {tag}: {m}", flush=True)
        return m

    result["through_unet"] = {
        "coreml_unet_torch_features_vs_ref": report("torch features", faces_a, crops["a"], pred_a, ms_a),
        "coreml_unet_coreml_features_vs_ref": report("Core ML features", faces_b, crops["b"], pred_b, ms_b),
    }
    m_ab = {"raw_face": metrics(faces_b, faces_a), "composed_face_crop": metrics(crops["b"], crops["a"]),
            "latent_max_abs": round(float((pred_b - pred_a).abs().max()), 3)}
    result["through_unet"]["coreml_features_vs_torch_features_same_unet"] = m_ab
    print(f"  Core ML vs torch features through the same UNet: {m_ab}", flush=True)

    video_dir = args.video_dir or (REPO_ROOT / "models" / "ios_bench" / "evidence" / args.avatar.name)
    video_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = video_dir / f"coreml_full_chain_{args.unet_variant}.mp4"
    write_video(composed["b"], out_mp4, args.fps, args.audio, args.seconds)
    result["full_chain_video"] = str(out_mp4)
    print(f"wrote {out_mp4}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
