"""Build a multi-avatar UNet validation corpus in the live scheduler's capture format.

Plan item 0.7 (docs/musetalk_4070s_300fps_plan_2026-09-27.md). The output files are
byte-for-byte the format `HLSGPUStreamScheduler._capture_unet_calibration_batch` writes
(`unet_io_<seq>_bs8_pid<pid>.pt`, kind=unet_io_batch), so
`scripts/validate_unet_backend.py --capture-dir <dir>` and
`scripts/tensorrt_export.py --unet-capture-dir <dir>` consume them unchanged.

Scheduler equivalence (every step uses the repo's own code where it exists):
- audio: AudioProcessor.get_audio_feature(path, 0, fp16) ->
  encode_whisper_feature(..., fps=<live fps, 20>, padding 2/2) -> build_audio_prompts ->
  HLSGPUStreamScheduler._apply_positional_encoding_cpu (the live CPU-side PE add, fp16);
- latents: prepared avatar latents.pt (already the forward+reverse cycle) or an H3
  experiment cache (240 native-encoder latents; cycled forward+reverse exactly like
  APIAvatar prepare), normalised by APIAvatar._finalize_latent_cycle;
- batch assembly: pinned staging buffers, frame i uses cycle[(start_offset + i) % N]
  (the no-pose-router path of _run_generation_batch), H2D non_blocking, fp16;
- reference output: the PyTorch FP16 eager UNet with the live server's backend flags
  (tf32 on, cudnn.benchmark on, inference_mode, timesteps=[0]). This matches the
  existing single-avatar captures (/workspace/benchmarks/same-avatar/unet-captures),
  which were recorded from the eager UNet; the script re-checks that on those files.
- files are written by HLSGPUStreamScheduler._capture_unet_calibration_batch itself.

Run under the GPU lease:
  scripts/box_guard.sh run --min-avail-gb 8 -- \
    /workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/build_unet_multi_avatar_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.hls_gpu_scheduler import HLSGPUStreamScheduler  # noqa: E402
from scripts.api_avatar import APIAvatar as _APIAvatarFactory  # noqa: E402


def _unwrap_class(obj, name: str):
    """api_avatar decorates the class with @torch.no_grad(), which hides it behind lambdas."""
    seen, todo = set(), [obj]
    while todo:
        cur = todo.pop()
        if isinstance(cur, type) and cur.__name__ == name:
            return cur
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        todo.extend(c.cell_contents for c in (getattr(cur, "__closure__", None) or ()))
        if hasattr(cur, "__wrapped__"):
            todo.append(cur.__wrapped__)
    raise RuntimeError(f"could not unwrap {name}")


APIAvatar = _unwrap_class(_APIAvatarFactory, "APIAvatar")

EXP = Path("/workspace/experiments")
AVATARS = ROOT / "results/v15/avatars"

AUDIO = {
    # af_heart (female Kokoro), dense 10 s; the audio the chin experiments render with.
    "A_af_heart_dense": EXP / "cheek_drift_ab_20260926/h3_reduced_mouth_screen_20260926/dense_tts/speech_dense_10s.wav",
    # am_michael (male Kokoro), 10 s; the diversity batch's second voice.
    "B_am_michael": EXP / "avatar_diversity_20260927/_audio/am_michael/speech.wav",
    # repo sample speech, 60 s; used only by the holdout split (unseen voice).
    "C_repo_eng": ROOT / "data/audio/eng.wav",
}

# (name, kind, path, split, audio keys)
AVATAR_SPECS = [
    # Chin-experiment H3 sources (portrait_jaw_video_20260926 *_new caches, used by
    # chin_boundary / chin_fps_validation / quiet_taesd_chin100).
    ("chin_h3_japanese_new", "h3_cache", EXP / "portrait_jaw_video_20260926/japanese_new/cache.pt", "main", ("A_af_heart_dense", "B_am_michael")),
    ("chin_h3_latina_new", "h3_cache", EXP / "portrait_jaw_video_20260926/latina_new/cache.pt", "main", ("A_af_heart_dense", "B_am_michael")),
    # Live prepared talking avatars.
    ("japanese_realtime_talking_7d94520b7f", "results_v15", AVATARS / "japanese_realtime_talking_7d94520b7f", "main", ("A_af_heart_dense", "B_am_michael")),
    ("japanese_realtime_talking_7d94520b7f_fh1", "results_v15", AVATARS / "japanese_realtime_talking_7d94520b7f_fh1", "main", ("A_af_heart_dense", "B_am_michael")),
    ("latina_guided_20260925_talking_84c5bc80b8", "results_v15", AVATARS / "latina_guided_20260925_talking_84c5bc80b8", "main", ("A_af_heart_dense", "B_am_michael")),
    ("chinese_bob_pink_bedroom_talking_3373c10448", "results_v15", AVATARS / "chinese_bob_pink_bedroom_talking_3373c10448", "main", ("A_af_heart_dense", "B_am_michael")),
    ("indian_realtime_talking_20f9845543", "results_v15", AVATARS / "indian_realtime_talking_20f9845543", "main", ("A_af_heart_dense", "B_am_michael")),
    # Diversity-batch H3 identities (avatar_diversity_20260927).
    ("div_black_woman", "h3_cache", EXP / "avatar_diversity_20260927/black_woman/cache.pt", "main", ("A_af_heart_dense", "B_am_michael")),
    ("div_east_asian_man_goatee", "h3_cache", EXP / "avatar_diversity_20260927/east_asian_man_goatee/cache.pt", "main", ("A_af_heart_dense", "B_am_michael")),
    ("div_middle_eastern_man_full_beard", "h3_cache", EXP / "avatar_diversity_20260927/middle_eastern_man_full_beard/cache.pt", "main", ("A_af_heart_dense", "B_am_michael")),
    ("div_white_man_clean_shaven", "h3_cache", EXP / "avatar_diversity_20260927/white_man_clean_shaven/cache.pt", "main", ("A_af_heart_dense", "B_am_michael")),
    # Holdout: identities that appear nowhere in the main split, plus one unseen voice.
    ("div_south_asian_woman", "h3_cache", EXP / "avatar_diversity_20260927/south_asian_woman/cache.pt", "holdout", ("B_am_michael", "C_repo_eng")),
    ("div_black_man_short_beard", "h3_cache", EXP / "avatar_diversity_20260927/black_man_short_beard/cache.pt", "holdout", ("B_am_michael", "C_repo_eng")),
    ("codex_smoke", "results_v15", AVATARS / "codex_smoke", "holdout", ("B_am_michael", "C_repo_eng")),
]

# Per-voice start offset into the latent cycle (fraction of the cycle), so the two
# voices of one avatar see different source frames, as live turns do.
AUDIO_START_FRACTION = {"A_af_heart_dense": 0.0, "B_am_michael": 1.0 / 3.0, "C_repo_eng": 2.0 / 3.0}


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit_bytes
        while True:
            chunk = f.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def load_eager_unet(device: torch.device):
    """The live eager UNet (models/musetalkV15, fp16) with a lower host-RAM peak.

    musetalk.models.unet.UNet loads the FP32 state dict and the FP32 model (6.8 GB
    host RAM), then ParallelAvatarManager casts to fp16. Here the state dict is
    mmap'd and copied into fp16 parameters; copy_ from FP32 into FP16 rounds to
    nearest-even exactly like .half(), so the weights are bit-identical.
    """
    from diffusers import UNet2DConditionModel

    cfg = json.loads((ROOT / "models/musetalkV15/musetalk.json").read_text())
    model = UNet2DConditionModel(**cfg).half()
    try:
        state = torch.load(ROOT / "models/musetalkV15/unet.pth", map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        state = torch.load(ROOT / "models/musetalkV15/unet.pth", map_location="cpu")
    model.load_state_dict(state)
    del state
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def latent_cycle_for(kind: str, path: Path) -> torch.Tensor:
    if kind == "results_v15":
        latents = torch.load(path / "latents.pt", map_location="cpu", weights_only=False)
    else:
        frames = torch.load(path, map_location="cpu", weights_only=False)["latents"]
        per_frame = [frames[i:i + 1] for i in range(frames.shape[0])]
        latents = per_frame + per_frame[::-1]  # APIAvatar prepare: list + list[::-1]
    stub = types.SimpleNamespace(input_latent_list_cycle=latents)
    APIAvatar._finalize_latent_cycle(stub)
    return stub.input_latent_cycle_batch_tensor  # [N, 8, 32, 32]


def batch_starts(total_frames: int, batch: int, count: int) -> list[int]:
    last = (total_frames - batch) // batch
    if last + 1 <= count:
        return [i * batch for i in range(last + 1)]
    return [int(round(x)) * batch for x in np.linspace(0, last, count)]


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs().reshape(-1)
    return {"mae": float(d.mean()), "max_abs": float(d.max()), "exact": bool(torch.equal(a, b))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(ROOT / "calibration/unet_multi_avatar_20260928"))
    ap.add_argument("--fps", type=int, default=20, help="live generation fps (WebRTC path uses 20)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches-per-pair", type=int, default=16)
    ap.add_argument("--manifest", default="", help="optional extra copy of the manifest JSON")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    # Same backend flags ParallelAvatarManager._init_models sets.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    main_dir, holdout_dir = out_dir, out_dir / "holdout"
    for d in (main_dir, holdout_dir):
        d.mkdir(parents=True, exist_ok=True)
        stale = list(d.glob("unet_io_*.pt"))
        if stale:
            raise SystemExit(f"{d} already has {len(stale)} capture files; remove them first")

    t0 = time.time()
    from musetalk.models.unet import PositionalEncoding
    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel

    pe = PositionalEncoding(d_model=384).half().to(device).eval()
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    whisper = WhisperModel.from_pretrained("./models/whisper").to(device=device, dtype=torch.float16).eval()
    whisper.requires_grad_(False)

    sched_stub = types.SimpleNamespace(
        manager=types.SimpleNamespace(pe=pe),
        _cpu_pe_cache={},
        unet_calibration_capture=True,
        unet_calibration_dir=main_dir,
        unet_calibration_max_batches=0,
        _unet_calibration_capture_count=0,
        _unet_calibration_limit_logged=False,
    )

    conditioning = {}
    audio_meta = {}
    with torch.inference_mode():
        for key, path in AUDIO.items():
            feats, librosa_length = audio_processor.get_audio_feature(str(path), 0, torch.float16)
            whisper_feature, total_frames = audio_processor.encode_whisper_feature(
                feats, device, torch.float16, whisper, librosa_length, fps=args.fps,
                audio_padding_length_left=2, audio_padding_length_right=2,
            )
            prompts = audio_processor.build_audio_prompts(
                whisper_feature=whisper_feature, num_frames=total_frames, fps=args.fps,
                audio_padding_length_left=2, audio_padding_length_right=2, start_frame=0, end_frame=total_frames,
            )
            cond = HLSGPUStreamScheduler._apply_positional_encoding_cpu(sched_stub, prompts)
            # Cross-check: the GPU PE module path (offline harnesses) on the same prompts.
            gpu_pe = pe(prompts[:64].to(device)).cpu()
            conditioning[key] = cond
            audio_meta[key] = {
                "path": str(path), "sha256": sha256_file(path), "seconds": librosa_length / 16000.0,
                "frames_at_fps": int(total_frames), "conditioning_shape": list(cond.shape),
                "conditioning_dtype": str(cond.dtype),
                "cpu_pe_vs_gpu_pe_module": tensor_stats(cond[:64], gpu_pe),
            }
            print(f"audio {key}: {total_frames} frames @ {args.fps} fps, cond {tuple(cond.shape)} {cond.dtype}", flush=True)
    del whisper
    torch.cuda.empty_cache()

    unet = load_eager_unet(device)
    timesteps = torch.tensor([0], device=device)
    print(f"models ready in {time.time() - t0:.1f}s", flush=True)

    bs = args.batch
    staging_cond = torch.empty((bs, 50, 384), dtype=torch.float16, pin_memory=True)
    staging_lat = torch.empty((bs, 8, 32, 32), dtype=torch.float16, pin_memory=True)
    files = []
    avatar_meta = {}
    determinism = []
    with torch.inference_mode():
        for name, kind, path, split, audio_keys in AVATAR_SPECS:
            cycle = latent_cycle_for(kind, path)
            n_cycle = int(cycle.shape[0])
            src = path / "latents.pt" if kind == "results_v15" else path
            avatar_meta[name] = {"kind": kind, "path": str(path), "split": split, "cycle_frames": n_cycle,
                                 "latent_dtype": str(cycle.dtype), "latents_sha256": sha256_file(src)}
            sched_stub.unet_calibration_dir = main_dir if split == "main" else holdout_dir
            for akey in audio_keys:
                cond = conditioning[akey]
                start_offset = int(round(AUDIO_START_FRACTION[akey] * n_cycle))
                starts = batch_starts(int(cond.shape[0]), bs, args.batches_per_pair)
                for s in starts:
                    staging_cond.copy_(cond[s:s + bs])
                    for j in range(bs):
                        staging_lat[j].copy_(cycle[(start_offset + s + j) % n_cycle])
                    audio_batch = staging_cond.to(device, non_blocking=True)
                    latent_batch = staging_lat.to(device=device, dtype=audio_batch.dtype, non_blocking=True)
                    pred = unet(latent_batch, timesteps, encoder_hidden_states=audio_batch).sample
                    job = types.SimpleNamespace(
                        request_id=f"corpus:{name}:{akey}:off{start_offset}",
                        session_id=f"corpus:{name}",
                        session=types.SimpleNamespace(avatar_id=name),
                        current_frame_idx=int(s),
                    )
                    HLSGPUStreamScheduler._capture_unet_calibration_batch(
                        sched_stub, latent_batch=latent_batch, audio_feature_batch=audio_batch,
                        timesteps=timesteps, pred_latents=pred, selected=[(job, bs)],
                        actual_batch=bs, padded_batch=bs,
                    )
                    seq = sched_stub._unet_calibration_capture_count
                    fname = f"unet_io_{seq:06d}_bs{bs}_pid{os.getpid()}.pt"
                    fpath = sched_stub.unet_calibration_dir / fname
                    if not fpath.exists():
                        raise RuntimeError(f"capture was not written: {fpath}")
                    files.append({"file": str(fpath.relative_to(out_dir)), "split": split, "avatar": name,
                                  "audio": akey, "start_frame": int(s), "start_offset_frames": start_offset,
                                  "cycle_indices": [int((start_offset + s + j) % n_cycle) for j in range(bs)]})
                    if len(determinism) < 4:
                        again = unet(latent_batch, timesteps, encoder_hidden_states=audio_batch).sample
                        determinism.append(tensor_stats(again, pred))
            print(f"avatar {name} ({split}): {sum(1 for f in files if f['avatar'] == name)} batches", flush=True)

        # Provenance: the existing single-avatar captures were produced by this eager UNet.
        legacy = []
        for p in sorted(Path("/workspace/benchmarks/same-avatar/unet-captures").glob("unet_io_*_bs8_*.pt"))[:4]:
            payload = torch.load(p, map_location="cpu", weights_only=False)
            lat = payload["latent_batch"].to(device, torch.float16)
            aud = payload["audio_feature_batch"].to(device, torch.float16)
            out = unet(lat, timesteps, encoder_hidden_states=aud).sample
            legacy.append({"file": p.name, **tensor_stats(out.cpu(), payload["pred_latents"])})

    sizes = sum((out_dir / f["file"]).stat().st_size for f in files)
    manifest = {
        "schema": "unet_multi_avatar_corpus_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256_file(Path(__file__)),
        "capture_format": "HLSGPUStreamScheduler._capture_unet_calibration_batch schema_version=1 (kind=unet_io_batch)",
        "reference_backend": "PyTorch FP16 eager UNet (models/musetalkV15/unet.pth), tf32+cudnn.benchmark on, inference_mode",
        "fps": args.fps, "batch": bs, "batches_per_pair": args.batches_per_pair,
        "audio_padding_left_right": [2, 2],
        "splits": {"main": str(main_dir), "holdout": str(holdout_dir)},
        "counts": {
            "main_files": sum(1 for f in files if f["split"] == "main"),
            "holdout_files": sum(1 for f in files if f["split"] == "holdout"),
            "main_avatars": sorted({f["avatar"] for f in files if f["split"] == "main"}),
            "holdout_avatars": sorted({f["avatar"] for f in files if f["split"] == "holdout"}),
            "total_bytes": int(sizes),
        },
        "audio": audio_meta,
        "avatars": avatar_meta,
        "checks": {
            "same_process_rerun_vs_saved_reference": determinism,
            "legacy_same_avatar_captures_eager_vs_stored_reference": legacy,
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seconds": time.time() - t0,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    print(json.dumps({k: manifest[k] for k in ("counts", "checks")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
