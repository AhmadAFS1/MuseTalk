#!/usr/bin/env python
"""Assemble the resources the iOS timing harness (ios/MuseTalkBench) bundles.

The harness is data-driven: it reads `meta.json` from its HarnessBundle folder and
finds there the compiled Core ML models (.mlmodelc), the UNet gate captures as
flat binaries, and a 30 s Whisper segment with the fp32 torch features as the
reference. Everything in HarnessBundle/ is generated here from the artifacts the
other scripts/ios tools already produced, so the folder is gitignored.

UNet captures: `calibration/unet_portable_bs8/*.pt` are the production inputs the
server engines are gated on (`scripts/validate_unet_backend.py`). The stored audio
features already carry the positional encoding; the Core ML step adds it inside
the graph, so the PE is subtracted here (same as validate_unet_coreml_gate.py).
The stored `pred_latents` (server fp16 PyTorch) are the reference, so a device
run of the harness reports the same per-file mae / max_abs statistics as the
Mac gate JSONs and the server engine results.

Usage (repo root, .venv-ct):
  python scripts/ios/prepare_harness_bundle.py \
      --models unet_ane_int8w_b1,unet_ane_int8w_b4,taesd_fp16_b1,whisper_encoder_fp16,whisper_logmel_fp32
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_unet_backend import _load_capture_group, discover_captures  # noqa: E402

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"
RESULTS_DIR = REPO_ROOT / "docs" / "ios_unet_bench"
DEFAULT_MODELS = "unet_anec0_int8w_b1_chunk1,unet_anec0_int8w_b1_chunk2,taesd_fp16_b1,whisper_encoder_fp16,whisper_logmel_fp32"
SAMPLES_30S = 16000 * 30
DTYPE_NAMES = {65552: "float16", 65568: "float32", 131104: "int32", 65600: "float64"}


def compile_model(src: Path, out_dir: Path) -> Path:
    dst = out_dir / (src.stem + ".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    t0 = time.time()
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(src), str(out_dir)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if not dst.exists():
        raise RuntimeError(f"coremlcompiler did not produce {dst}")
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"  compiled {src.name} -> {dst.name} ({size / 2**20:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    return dst


def io_description(mlpackage: Path) -> dict:
    import coremltools as ct

    spec = ct.utils.load_spec(str(mlpackage))

    def features(fs):
        out = []
        for f in fs:
            mat = f.type.multiArrayType
            out.append({"name": f.name, "shape": [int(s) for s in mat.shape],
                        "dtype": DTYPE_NAMES.get(mat.dataType, str(mat.dataType))})
        return out

    return {"inputs": features(spec.description.input), "outputs": features(spec.description.output)}


def write_bin(path: Path, arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    path.write_bytes(arr.tobytes())
    return {"file": path.name, "shape": list(arr.shape), "dtype": str(arr.dtype), "bytes": int(arr.nbytes)}


def export_unet_captures(out_dir: Path, capture_dir: Path, limit: int, unet_dir: Path) -> dict:
    from musetalk.models.unet import PositionalEncoding

    pe = PositionalEncoding(d_model=384).pe[:, :50, :].float()
    paths = discover_captures(capture_dir, limit)
    lats, auds, refs, files = [], [], [], []
    start = 0
    for path in paths:
        payload = _load_capture_group([path])
        n = int(payload.get("actual_batch", payload["latent_batch"].shape[0]))
        lat = payload["latent_batch"].float()[:n]
        aud = payload["audio_feature_batch"].float()[:n] - pe
        ref = payload["pred_latents"].float()[:n]
        lats.append(lat.numpy().astype(np.float16))
        auds.append(aud.numpy().astype(np.float16))
        refs.append(ref.numpy().astype(np.float16))
        avatar = payload["items"][0].get("avatar_id") if payload.get("items") else None
        files.append({"name": path.name, "start": start, "count": n, "avatar": avatar})
        start += n
    lat_all, aud_all, ref_all = np.concatenate(lats), np.concatenate(auds), np.concatenate(refs)
    print(f"  UNet captures: {len(paths)} files, {start} frames; latents {lat_all.shape}, audio {aud_all.shape}, "
          f"ref {ref_all.shape}", flush=True)
    return {
        "frames": start,
        "capture_dir": str(capture_dir.relative_to(REPO_ROOT)),
        "reference": "server fp16 PyTorch pred_latents stored in the captures",
        "audio_note": "positional encoding subtracted; the Core ML UNet adds it inside the graph",
        "latents": write_bin(out_dir / "unet_latents_f16.bin", lat_all),
        "audio": write_bin(out_dir / "unet_audio_f16.bin", aud_all),
        "ref": write_bin(out_dir / "unet_ref_f16.bin", ref_all),
        "files": files,
    }


def export_whisper_segment(out_dir: Path, audio: Path, whisper_dir: Path) -> dict:
    import librosa
    from transformers import WhisperFeatureExtractor, WhisperModel

    wav, _ = librosa.load(str(audio), sr=16000)
    seg = np.zeros(SAMPLES_30S, dtype=np.float32)
    seg[:min(len(wav), SAMPLES_30S)] = wav[:SAMPLES_30S]
    fe = WhisperFeatureExtractor.from_pretrained(str(whisper_dir))
    mel = fe(seg, return_tensors="pt", sampling_rate=16000).input_features  # [1, 80, 3000]
    whisper = WhisperModel.from_pretrained(str(whisper_dir), attn_implementation="eager").eval()
    with torch.inference_mode():
        feats = torch.stack(whisper.encoder(mel, output_hidden_states=True).hidden_states, dim=2)  # [1,1500,5,384]
    print(f"  Whisper segment: {audio.name}, {min(len(wav), SAMPLES_30S) / 16000:.1f}s of speech; "
          f"mel {tuple(mel.shape)}, features {tuple(feats.shape)}", flush=True)
    return {
        "audio": str(audio.relative_to(REPO_ROOT)) if audio.is_relative_to(REPO_ROOT) else str(audio),
        "speech_seconds": round(min(len(wav), SAMPLES_30S) / 16000, 2),
        "reference": "transformers WhisperModel encoder, fp32, hidden states stacked [1,1500,5,384]",
        "waveform": write_bin(out_dir / "whisper_waveform_f32.bin", seg[None]),
        "mel": write_bin(out_dir / "whisper_mel_f16.bin", mel.numpy().astype(np.float16)),
        "features_ref": write_bin(out_dir / "whisper_features_ref_f16.bin", feats.numpy().astype(np.float16)),
    }


def mac_reference(model_names: list[str]) -> dict:
    """Numbers already measured on this Mac, for side-by-side display on the device."""
    ref: dict = {"host": platform.machine(), "note": "M5 Pro Neural Engine, coremltools 9 Python bench"}
    gates = {}
    for name in model_names:
        p = RESULTS_DIR / f"gate_{name}_m5pro.json"
        if p.exists():
            s = json.loads(p.read_text())["summary"]
            gates[name] = {k: s[k] for k in ("mae_mean", "mae_max", "max_abs_mean", "max_abs_max", "pass") if k in s}
    for name in ("unet_fp16_b1", "pytorch_fp32"):
        p = RESULTS_DIR / f"gate_{name}_m5pro.json"
        if p.exists() and name not in gates:
            s = json.loads(p.read_text())["summary"]
            gates[name] = {k: s[k] for k in ("mae_mean", "mae_max", "max_abs_mean", "max_abs_max", "pass") if k in s}
    ref["gate"] = gates
    speed = RESULTS_DIR / "coreml_ane_layout_m5pro.json"
    if speed.exists():
        ref["ane_layout_bench"] = json.loads(speed.read_text())
    ref["server_engines"] = {
        "trt_fp16": {"mae_mean": 0.0018, "mae_max": 0.0025, "max_abs_max": 0.414},
        "trt_int8_srcg50_shipping": {"mae_mean": 0.0032, "mae_max": 0.0044, "max_abs_max": 0.777},
    }
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated manifest names to bundle")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "ios" / "MuseTalkBench" / "HarnessBundle")
    ap.add_argument("--capture-dir", type=Path, default=REPO_ROOT / "calibration" / "unet_portable_bs8")
    ap.add_argument("--limit", type=int, default=16, help="capture files (8 frames each)")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--audio", type=Path, default=REPO_ROOT / "data" / "audio" / "ai-assistant.mpga")
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    ap.add_argument("--skip-whisper", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    names = [n for n in args.models.split(",") if n]
    args.out.mkdir(parents=True, exist_ok=True)

    models = []
    print("compiling models", flush=True)
    for name in names:
        entry = manifest[name]
        src = Path(entry["path"])
        compiled = compile_model(src, args.out)
        models.append({"name": name, "file": compiled.name, "kind": entry["kind"], "batch": entry.get("batch", 1),
                       "variant": entry.get("variant"), "layout": entry.get("layout", "diffusers"),
                       "mlpackage_bytes": entry.get("bytes"), "chunk_of": entry.get("chunk_of"),
                       "chunk_index": entry.get("chunk_index"), **io_description(src)})

    print("exporting UNet gate captures", flush=True)
    unet = export_unet_captures(args.out, args.capture_dir, args.limit, args.unet_dir)
    whisper = None
    if not args.skip_whisper and any(m["kind"].startswith("whisper") for m in models):
        print("exporting Whisper segment", flush=True)
        whisper = export_whisper_segment(args.out, args.audio, args.whisper_dir)

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                      text=True).stdout.strip(),
        "gate": {"mae": 0.01, "max_abs": 0.5, "source": "scripts/validate_unet_backend.py per-file thresholds"},
        "models": models,
        "unet": unet,
        "whisper": whisper,
        "mac_reference": mac_reference(names),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"wrote {args.out / 'meta.json'}; HarnessBundle total {total / 2**20:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
