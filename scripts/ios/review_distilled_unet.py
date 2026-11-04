#!/usr/bin/env python
"""Video review of a distilled MuseTalk UNet: can the student replace the full UNet on the phone?

25 fps on the iPhone 12 Pro needs a smaller UNet (ios_coreml_implementation.md 5.8): a BK-SDM shape from
export_small_unets.py (bk_tiny is the candidate), distilled from the full UNet by distill_small_unet.py. The gate
frames say how close the student's latents are. This review shows what that closeness does to the lipsync. It uses
the set and metrics the Neural Engine chain was judged on (review_ane_vs_fp16.py, review_mixed_units.py; 4.2, 5.7):
six production characters x two audios, 10 s at 25 fps.

Arms per clip. All arms share the avatar cache, the Whisper windows and the blend, and all decode with torch TAESD
in fp32 on --device:
  A    reference  full UNet fp32 with TF32 off (cuDNN and matmul), so A is true fp32 on Ampere too
  N    noise      full UNet fp16, the server's precision: the size of difference shipping already accepts
  S    student    the distilled UNet in fp32, TF32 off
  S16  student    the same weights in fp16, the precision the phone runs it at (scored, not part of the verdict)

Verdict on S: PASS when
  (1) every clip passes the r5 thresholds against A: aperture corr >= 0.97, lag 0, mean |delta| <= 0.5 px,
      mouth and face flicker ratio <= 1.05, mouth sharpness ratio >= 0.95; and
  (2) every clip is non-inferior to the chain the app ships today. The bar is the Neural Engine chain's D-vs-A
      numbers on the same clip (--mixed-metrics, review_mixed_units' arm D), within the margins below.
FAIL when any scored check fails. INCOMPLETE when nothing failed but something went unscored: no FaceMesh,
reduced landmark coverage, mismatched frame count/fps/ROIs, a clip outside D's set or a D clip not run. All six
visual no-worse comparisons use zero deterioration margins. Latent error is reported but is not a visual gate.

One UNet sits on the device at a time. Every clip runs through the teacher (A, then the same weights cast once to
fp16 for N), then through the student (S, then S16), and only then is anything decoded. The weights are never cast
back, so A always runs the checkpoint's own fp32 weights.

Writes <out>/<clip>/: A.mp4 N.mp4 S.mp4 S16.mp4, <clip>_AS.mp4 (A | S labelled, 3x mouth zoom, |A-S| x8; the
zoom caption is the review helper's "A | B", and B is S), latents_*.npy, landmarks_*.npy, metrics.json. Also
<out>/stills/, <out>/summary.json and <out>/summary.md.

Usage (repo root; on the CUDA box the server venv, which has diffusers 0.30: mid_block_type=None is supported there):
  python scripts/ios/review_distilled_unet.py --avatars <avatars>/en_01_maya_... <avatars>/... \\
      --audio data/audio/ai-assistant.mpga data/audio/eng.wav \\
      --student-json <run>/bk_tiny.json --student-weights <run>/bk_tiny_best.pt \\
      --out experiments/video_validation/distilled_unet --facemesh-python <venv with mediapipe 0.10.14>/bin/python
A tiny Mac test (the verdict is then INCOMPLETE at best, or FAIL): add --device cpu --fp16-device mps --frames 12.

New identities use an explicitly separate protocol: --protocol new-static-r5 --new-avatar-manifest <frozen.json>.
The manifest binds 30 static portraits/caches, two clips and a checkpoint SHA256 (validate_new_manifest documents
paths). All 60 clips require 250 frames at 25 fps and full finite FaceMesh coverage. N/S/S16 get independent r5-only
verdicts; no old D metrics are read. Static head/body inputs do not establish moving-pose or universal quality,
and CUDA fp16 is not evidence of iPhone ANE/Core ML quantization quality. The original default protocol is unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import diffusers
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from render_evidence import audio_windows, compose, load_avatar, taesd_torch, to_bgr_u8, write_video  # noqa: E402
from review_ane_vs_fp16 import (FACEMESH_HELPER, R5_GATES, aperture_metrics, aperture_series, compare, even,  # noqa: E402
                                facemesh, gates, labelled_frame, mouth_box_from_landmarks, mouth_box_heuristic, rel,
                                union_box, wrap)
from review_mixed_units import error_flicker, r5_status, summary_line  # noqa: E402
from musetalk.models.unet import UNet  # noqa: E402

MIXED_METRICS = REPO_ROOT / "docs" / "ios_unet_bench" / "review_mixed_units" / "metrics.json"
ARMS = ("A", "N", "S", "S16")
SCORED = ("N", "S", "S16")

# The user's acceptance criterion is literal: no worse than today's phone chain on the six visual checks.
# The unreviewed WIP allowed deterioration margins; those are deliberately zero for the overnight verdict.
LIP_CORR_MARGIN = 0.0
LIP_DELTA_MARGIN_PX = 0.0
MOUTH_PSNR_MARGIN_DB = 0.0
MOUTH_SHARPNESS_MARGIN = 0.0
ERROR_FLICKER_MARGIN = 0.0
FLICKER_RATIO_MARGIN = 0.0
# (key, higher is better, margin)
CHECKS = (("lip_corr", True, LIP_CORR_MARGIN), ("lip_mean_abs_delta_px", False, LIP_DELTA_MARGIN_PX),
          ("mouth_psnr_db", True, MOUTH_PSNR_MARGIN_DB), ("mouth_sharpness_ratio", True, MOUTH_SHARPNESS_MARGIN),
          ("error_flicker_mouth", False, ERROR_FLICKER_MARGIN), ("mouth_flicker_ratio", False, FLICKER_RATIO_MARGIN))


NEW_PROTOCOL = "new_static_avatars_r5_v1"
NEW_R5_GATE_KEYS = ("flicker_mouth_ratio_le_1.05", "flicker_face_ratio_le_1.05",
                    "mouth_sharpness_ratio_ge_0.95", "aperture_corr_ge_0.97", "aperture_lag_0",
                    "aperture_mean_abs_delta_le_0.5px")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_new_manifest(path: Path, avatars: list[Path], audios: list[Path], weights: Path,
                          fps: int, seconds: float, frame_limit: int | None) -> dict:
    """Bind the opt-in static-portrait experiment to its frozen inputs before any model is loaded.

    Paths in portrait/audio entries are relative to the manifest, or absolute. cache_files keys are relative
    to the corresponding --avatars directory and cover every file load_avatar consumes. Additional provenance
    fields are retained. The manifest is evidence of evaluation inputs, never a training/checkpoint selector.
    """
    m = json.loads(path.read_text())
    if (m.get("schema_version"), m.get("protocol"), m.get("avatar_kind"), m.get("fps"), m.get("frames")) != (
            1, NEW_PROTOCOL, "static_portrait", 25, 250):
        raise ValueError("new-avatar manifest must declare schema 1, static_portrait, 250 frames at 25 fps")
    if fps != 25 or seconds != 10 or frame_limit not in (None, 250):
        raise ValueError("new-static-r5 requires the complete 250-frame/10-second, 25-fps protocol")
    av_rows, au_rows = m.get("avatars"), m.get("audio")
    if not isinstance(av_rows, list) or len(av_rows) != 30 or not isinstance(au_rows, list) or len(au_rows) != 2:
        raise ValueError("new-avatar manifest requires exactly 30 avatars and two audio files")

    def resolve(value):
        p = Path(value)
        return (p if p.is_absolute() else path.parent / p).resolve()

    def verify(file, expected):
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"invalid SHA256 for {file}")
        if sha256_file(file) != expected:
            raise ValueError(f"SHA256 mismatch: {file}")

    if any(not isinstance(row, dict) for row in av_rows + au_rows):
        raise ValueError("avatar and audio manifest entries must be objects")
    av_by_id = {a.name: a.resolve() for a in avatars}
    ids = [r["id"] for r in av_rows]
    if any(not isinstance(ident, str) or not ident for ident in ids):
        raise ValueError("avatar manifest IDs must be nonempty strings")
    if len(av_by_id) != 30 or len(set(ids)) != 30 or set(ids) != set(av_by_id):
        raise ValueError("--avatars must match the 30 unique manifest IDs exactly")
    portrait_hashes, image_hashes, latent_hashes = [], [], []
    for row in av_rows:
        av = av_by_id[row["id"]]
        if row.get("cache_dir") is not None and resolve(row["cache_dir"]) != av:
            raise ValueError(f"cache_dir mismatch: {row['id']}")
        portrait = resolve(row["portrait"])
        verify(portrait, row["portrait_sha256"])
        portrait_hashes.append(row["portrait_sha256"])
        files = row["cache_files"]
        if not isinstance(files, dict):
            raise ValueError(f"cache_files must be a relative-path/hash mapping: {row['id']}")
        images = sorted((av / "full_imgs").glob("*.[jpJP][pnPN]*[gG]"))
        masks = sorted((av / "mask").glob("*.[jpJP][pnPN]*[gG]"))
        expected_files = {"coords.pkl", "mask_coords.pkl", "latents.pt"} | {
            f.relative_to(av).as_posix() for f in images + masks}
        if not images or len(images) != len(masks) or set(files) != expected_files:
            raise ValueError(f"cache_files must cover all reviewed frames/masks and three cache tensors: {row['id']}")
        for relative, digest in files.items():
            f = av / relative
            if not f.resolve().is_relative_to(av):
                raise ValueError(f"cache file escapes avatar directory: {relative}")
            verify(f, digest)
        if len({files[f.relative_to(av).as_posix()] for f in images}) != 1 or len({
                files[f.relative_to(av).as_posix()] for f in masks}) != 1:
            raise ValueError(f"static-portrait frames and masks must be repeated unchanged: {row['id']}")
        image_hashes.append(files[images[0].relative_to(av).as_posix()])
        latent_hashes.append(files["latents.pt"])
    if len(set(portrait_hashes)) != 30 or len(set(image_hashes)) != 30 or len(set(latent_hashes)) != 30:
        raise ValueError("new-avatar identities must have 30 distinct portrait, frame and latent-cache hashes")
    manifest_audio = [resolve(row["file"]) for row in au_rows]
    if (len(set(manifest_audio)) != 2 or len({row["sha256"] for row in au_rows}) != 2
            or set(manifest_audio) != {a.resolve() for a in audios}):
        raise ValueError("--audio must match the two manifest audio files exactly")
    if len({a.stem for a in manifest_audio}) != 2:
        raise ValueError("new-avatar audio stems must be unique")
    for row, audio in zip(au_rows, manifest_audio):
        verify(audio, row["sha256"])
    verify(weights, m["checkpoint_sha256"])
    return {"manifest": str(path.resolve()), "manifest_sha256": sha256_file(path), "inputs": m,
            "expected_clips": sorted(f"{ident}__{a.stem}" for ident in ids for a in manifest_audio)}


def protocol_inputs(args, avatars: list[Path], audios: list[Path]) -> tuple[dict, dict | None]:
    """Keep baseline selection explicit: the opt-in identity test never even opens legacy D metrics."""
    if args.protocol == "new-static-r5":
        if args.new_avatar_manifest is None or args.facemesh_python is None:
            raise ValueError("new-static-r5 requires --new-avatar-manifest and --facemesh-python")
        info = validate_new_manifest(args.new_avatar_manifest, avatars, audios, args.student_weights,
                                     args.fps, args.seconds, args.frames)
        return {}, info
    if args.new_avatar_manifest is not None:
        raise ValueError("--new-avatar-manifest requires --protocol new-static-r5")
    rows = json.loads(args.mixed_metrics.read_text())["clips"]
    clips = {row["clip"]: row for row in rows}
    if not clips or len(clips) != len(rows):
        raise ValueError("D metrics must have unique, nonempty clips")
    return clips, None


def finite_metric_tree(value) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(finite_metric_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(finite_metric_tree(v) for v in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def new_r5_verdict(clips: list[dict], expected: list[str], arm: str) -> tuple[str, list[str]]:
    """Independent r5 verdict for the frozen new-identity set; no shipped-chain D comparison is implied."""
    seen = [m["clip"] for m in clips]
    incomplete, failed = [], []
    if len(expected) != 60 or len(set(expected)) != 60 or len(seen) != len(set(seen)) or set(seen) != set(expected):
        incomplete.append("the complete, unique manifest set of 60 clips was not scored")
    for m in clips:
        tag = m["clip"]
        if (m.get("frames"), m.get("fps")) != (250, 25):
            incomplete.append(f"{tag}: requires 250 frames at 25 fps")
        coverage = m.get("landmark_coverage") or {}
        if coverage.get("A") != 250 or coverage.get(arm) != 250:
            incomplete.append(f"{tag}: incomplete finite FaceMesh coverage for A or {arm}")
        g = m.get(f"r5_gates_{arm}") or {}
        if any(type(g.get(k)) is not bool for k in NEW_R5_GATE_KEYS):
            incomplete.append(f"{tag}: not all six r5 gates were scored")
        else:
            failed.extend(f"{tag}: r5 {key}" for key in NEW_R5_GATE_KEYS if not g[key])
        try:
            ap = m["aperture"][f"{arm}_vs_A"]
            trees = (m["latent_vs_A"][arm], m[f"{arm}_vs_A"], m["error_flicker_mouth"][arm],
                     m["error_flicker_face"][arm], ap)
            if ap.get("frames") != 250 or not all(finite_metric_tree(v) for v in trees):
                incomplete.append(f"{tag}: missing/nonfinite metric or incomplete aperture coverage")
        except (KeyError, TypeError):
            incomplete.append(f"{tag}: required metric missing")
    if incomplete:
        return "INCOMPLETE", incomplete + failed
    return ("FAIL", failed) if failed else ("PASS", [])


# ------------------------------------------------------------------------------- arms
def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def run_unet(unet, pe, lat, aud, dev, dtype, batch):
    """pred latents fp32 [N,4,32,32] on the CPU, ms/frame for the UNet alone (after one warm-up batch)."""
    ts = torch.tensor([0], device=dev)
    preds, t_total = [], 0.0
    with torch.inference_mode():
        unet(lat[:batch].to(dev, dtype), ts, encoder_hidden_states=pe(aud[:batch].to(dev, dtype)).to(dtype))
        for s in range(0, lat.shape[0], batch):
            l, a = lat[s:s + batch].to(dev, dtype), aud[s:s + batch].to(dev, dtype)
            sync(dev)
            t0 = time.perf_counter()
            pred = unet(l, ts, encoder_hidden_states=pe(a).to(dtype)).sample
            sync(dev)
            t_total += time.perf_counter() - t0
            preds.append(pred.float().cpu())
    return torch.cat(preds), t_total / lat.shape[0] * 1000


def unet_arms(clips: list[dict], nets: list[tuple], batch) -> tuple[dict, dict]:
    """Every clip through each network's arms in order: ({clip: {arm: (pred, ms)}}, {net: params}).

    nets: (name, config, weights, [(arm, dtype, device), ...]), the fp32 arm first: the cast to fp16 happens once."""
    out, params = {c["clip"]: {} for c in clips}, {}
    for name, config, weights, arms in nets:
        print(f"loading {name} UNet {Path(weights).name}...", flush=True)
        w = UNet(unet_config=str(config), model_path=str(weights), device=torch.device("cpu"))
        unet, pe = w.model.eval().requires_grad_(False), w.pe
        params[name] = sum(p.numel() for p in unet.parameters())
        for arm, dtype, dev in arms:
            unet, pe = unet.to(dev, dtype), pe.to(dev, dtype)
            for c in clips:
                out[c["clip"]][arm] = run_unet(unet, pe, c["lat"], c["aud"], dev, dtype, batch)
                print(f"  {arm} {c['clip']}: {out[c['clip']][arm][1]:.1f} ms/frame on {dev.type}", flush=True)
        del unet, pe, w
        gc.collect()
        for dev in {d for _, _, d in arms}:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            elif dev.type == "mps":
                torch.mps.empty_cache()
    return out, params


def decode(dec, preds: torch.Tensor, dev, batch=8) -> np.ndarray:
    """TAESD fp32 on the device (review_ane_vs_fp16.decode_torch only works with the decoder on the CPU)."""
    faces = []
    with torch.inference_mode():
        for s in range(0, preds.shape[0], batch):
            img = (dec.decode(preds[s:s + batch].to(dev, torch.float32)).sample / 2 + 0.5).clamp(0, 1)
            faces.append(to_bgr_u8(img))
    return np.concatenate(faces)


# ----------------------------------------------------------------- non-inferiority vs D
def numbers(m: dict, arm: str) -> dict:
    """The per-clip numbers the verdict uses, from this review's metrics or a review_mixed_units clip (arm D)."""
    c = m[f"{arm}_vs_A"]
    ap = (m.get("aperture") or {}).get(f"{arm}_vs_A") or {}
    g = m[f"r5_gates_{arm}"]
    return {"frames": m["frames"], "fps": m["fps"], "face_box": m["face_box"], "mouth_box": m["mouth_box"],
            "aperture_frames": ap.get("frames", 0), "latent_mae": m["latent_vs_A"][arm]["mae"], "lip_corr": ap.get("corr"),
            "lip_lag_frames": ap.get("lag_frames"), "lip_mean_abs_delta_px": ap.get("mean_abs_delta_px"),
            "mouth_psnr_db": c["mouth_roi"]["psnr_db"], "mouth_flicker_ratio": c["mouth_roi"]["flicker_ratio"],
            "error_flicker_mouth": m["error_flicker_mouth"][arm], "mouth_sharpness_ratio": c["mouth_sharpness"]["ratio"],
            "r5": r5_status(g)}


def non_inferiority(x: dict, d: dict | None) -> dict:
    if d is None:
        return {"status": "incomplete", "checks": {}, "incomplete_because": ["clip not in the D set"]}
    checks, why = {}, []
    for key, higher, margin in CHECKS:
        if x.get(key) is None or d.get(key) is None:
            why.append(f"{key} not scored for student or D")
            continue
        bound = d[key] - margin if higher else d[key] + margin
        finite = math.isfinite(x[key]) and math.isfinite(d[key])
        checks[key] = {"value": x[key], "D": d[key], "delta_vs_D": x[key] - d[key], "bound": bound,
                       "pass": bool(finite and (x[key] >= bound if higher else x[key] <= bound))}
    for key in ("frames", "fps", "face_box", "mouth_box"):
        if x.get(key) != d.get(key):
            why.append(f"{key} differs from D: {x.get(key)} vs {d.get(key)}")
    if x.get("aperture_frames", 0) < d.get("aperture_frames", 0):
        why.append(f"aperture coverage {x.get('aperture_frames', 0)} frames, D has {d.get('aperture_frames', 0)}")
    status = "FAIL" if any(not c["pass"] for c in checks.values()) else ("incomplete" if why else "pass")
    return {"status": status, "checks": checks, "incomplete_because": why}


def failures(m: dict, arm: str) -> list[str]:
    out = [f"r5 {k}" for k, v in m[f"r5_gates_{arm}"].items() if k != "all_pass" and not v]
    for k, c in ((m.get("vs_D") or {}).get(arm) or {}).get("checks", {}).items():
        if not c["pass"]:
            out.append(f"vs D {k} {c['value']} (bound {c['bound']}, D {c['D']})")
    return out


def verdict(clips: list[dict], d_clips: dict, arm: str) -> tuple[str, list[str]]:
    reasons, failed = [], False
    if not clips:
        return "INCOMPLETE", ["no clips scored"]
    for m in clips:
        f = failures(m, arm)
        if f:
            failed = True
            reasons.append(f"{m['clip']}: " + "; ".join(f))
    if not failed:
        for m in clips:
            r5, vd = r5_status(m[f"r5_gates_{arm}"]), m["vs_D"][arm]
            if r5 != "pass" or vd["status"] != "pass":
                why = "; ".join(vd["incomplete_because"])
                reasons.append(f"{m['clip']}: r5 {r5}, vs D {vd['status']}" + (f" ({why})" if why else ""))
    missing = sorted(set(d_clips) - {m["clip"] for m in clips})
    if missing:
        reasons.append(f"{len(missing)} of the {len(d_clips)} D clips not run: {', '.join(missing)}")
    return ("FAIL" if failed else "INCOMPLETE" if reasons else "PASS"), reasons


# -------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatars", nargs="+", required=True, help="avatar directories (full_imgs, mask, coords.pkl, latents.pt); commas also split")
    ap.add_argument("--audio", nargs="+", default=["data/audio/ai-assistant.mpga", "data/audio/eng.wav"])
    ap.add_argument("--student-json", type=Path, required=True, help="UNet2DConditionModel config of the student")
    ap.add_argument("--student-weights", type=Path, required=True, help="the student's state dict (torch.save(unet.state_dict()))")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16-device", default=None, help="device for the fp16 arms N and S16 (default --device). The CPU's fp16 "
                    "is ~60x slower than its fp32 (2026-11-03, M5 Pro), so a Mac run passes mps, as review_ane_vs_fp16's N arm did")
    ap.add_argument("--frames", type=int, default=None, help="limit frames per clip (tests); the verdict is then INCOMPLETE at best")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--facemesh-python", type=Path, default=None, help="interpreter with mediapipe<=0.10.14 for scripts/quality_ab_facemesh.py")
    ap.add_argument("--protocol", choices=("legacy-r5-d", "new-static-r5"), default="legacy-r5-d",
                    help="default retains the original r5 + shipped-chain D acceptance; new-static-r5 is separate held-out identity evidence")
    ap.add_argument("--new-avatar-manifest", type=Path, help="frozen 30-portrait/two-audio SHA256 manifest; required for new-static-r5")
    ap.add_argument("--mixed-metrics", type=Path, default=MIXED_METRICS, help="review_mixed_units metrics.json: arm D's per-clip numbers")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    args = ap.parse_args()
    if args.seconds <= 0 or args.fps <= 0 or args.batch <= 0 or (args.frames is not None and args.frames <= 0):
        ap.error("seconds, fps, batch and frames must be positive")
    if args.facemesh_python is not None and not FACEMESH_HELPER.exists():
        sys.exit(f"--facemesh-python given but {FACEMESH_HELPER} is missing")
    # A and S are the fp32 references; TF32 would round their convolutions and matmuls to a 10-bit mantissa.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    dev16 = torch.device(args.fp16_device or args.device)
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
    dev16_name = torch.cuda.get_device_name(dev16) if dev16.type == "cuda" else dev16.type
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stills").mkdir(exist_ok=True)

    avatars = [Path(p) for a in args.avatars for p in a.split(",") if p]
    audios = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for a in args.audio for p in a.split(",") if p]
    try:
        d_clips, args.new_manifest_info = protocol_inputs(args, avatars, audios)
    except (OSError, ValueError, TypeError, KeyError) as e:
        ap.error(str(e))
    if len(avatars) != len(set(avatars)) or len(audios) != len(set(audios)):
        ap.error("duplicate avatars or audio inputs")

    windows = {a: audio_windows(a, args.whisper_dir, args.fps) for a in audios}
    clips = []
    for avatar in avatars:
        if args.new_manifest_info is not None:
            cache = load_avatar(avatar)
            av_latents = cache[4]
            expected_frames = len(list((avatar / "full_imgs").glob("*.[jpJP][pnPN]*[gG]")))
            if (len(cache[0]) != expected_frames or not torch.isfinite(av_latents).all()
                    or any(not np.array_equal(f, cache[0][0]) for f in cache[0])
                    or any(not np.array_equal(f, cache[1][0]) for f in cache[1])
                    or any(not np.array_equal(c, cache[2][0]) for c in cache[2])
                    or any(not np.array_equal(c, cache[3][0]) for c in cache[3])
                    or not torch.equal(av_latents, av_latents[:1].expand_as(av_latents))):
                ap.error(f"{avatar.name}: cache is not a complete, finite static-portrait cache")
            del cache
        else:
            av_latents = load_avatar(avatar)[4]  # preserve the original legacy cache lifetime
        for audio in audios:
            n = min(int(round(args.seconds * args.fps)), windows[audio].shape[0], args.frames or 1 << 30)
            clips.append({"clip": f"{avatar.name}__{audio.stem}", "avatar": avatar, "audio": audio, "n": n,
                          "lat": torch.stack([av_latents[i % av_latents.shape[0]] for i in range(n)]),
                          "aud": windows[audio][:n]})
    if len({c["clip"] for c in clips}) != len(clips):
        ap.error("avatar names and audio stems must form unique clip identifiers")
    if any(c["n"] < 10 for c in clips):
        ap.error("every clip needs at least 10 frames for aperture/flicker scoring")
    if args.new_manifest_info is not None:
        if sorted(c["clip"] for c in clips) != args.new_manifest_info["expected_clips"] or any(c["n"] != 250 for c in clips):
            ap.error("new-static-r5 requires every manifest clip at exactly 250 frames")
        if any(not torch.isfinite(c["aud"]).all() for c in clips):
            ap.error("new-static-r5 audio inputs contain nonfinite values")
    print(f"{len(clips)} clips, {sum(c['n'] for c in clips)} frames per arm, on {dev_name}", flush=True)

    nets = [("teacher", args.unet_dir / "musetalk.json", args.unet_dir / "unet.pth",
             [("A", torch.float32, dev), ("N", torch.float16, dev16)]),
            ("student", args.student_json, args.student_weights, [("S", torch.float32, dev), ("S16", torch.float16, dev16)])]
    preds, params = unet_arms(clips, nets, args.batch)
    sha = hashlib.sha256()
    with args.student_weights.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            sha.update(block)
    if args.new_manifest_info is not None and sha.hexdigest() != args.new_manifest_info["inputs"]["checkpoint_sha256"]:
        raise ValueError("student weights changed after frozen-manifest validation")
    student = {"json": rel(args.student_json), "weights": rel(args.student_weights), "sha256": sha.hexdigest(), "sha256_12": sha.hexdigest()[:12],
               "params_m": round(params["student"] / 1e6, 1), "teacher_params_m": round(params["teacher"] / 1e6, 1)}
    dec = taesd_torch(dev, torch.float32)

    results = []
    for avatar in avatars:
        frames, masks, coords, mask_coords, _ = load_avatar(avatar)
        H, W = frames[0].shape[:2]
        for c in (c for c in clips if c["avatar"] == avatar):
            t_clip, clip, n, audio = time.time(), c["clip"], c["n"], c["audio"]
            print(f"\n[{clip}] {n} frames", flush=True)
            clip_dir = args.out / clip
            clip_dir.mkdir(parents=True, exist_ok=True)
            pred = {arm: preds[clip][arm][0] for arm in ARMS}
            if args.new_manifest_info is not None and any(not torch.isfinite(p).all() for p in pred.values()):
                raise ValueError(f"{clip}: nonfinite UNet predictions; new-identity review cannot be scored")
            faces = {arm: decode(dec, pred[arm], dev) for arm in ARMS}
            full = {arm: np.stack(compose(faces[arm], frames, masks, coords, mask_coords)) for arm in ARMS}
            for arm in ARMS:
                np.save(clip_dir / f"latents_{arm}.npy", pred[arm].numpy().astype(np.float16))

            lms = {}
            if args.facemesh_python is not None:
                for arm in ARMS:
                    lms[arm] = facemesh(full[arm], args.facemesh_python, clip_dir / f"landmarks_{arm}.npy")
            # D's boxes when the clip is in the D set, so S and D are scored on the same pixels
            dm = d_clips.get(clip)
            fits = lambda b: len(b) == 4 and b[0] >= 0 and b[1] >= 0 and b[2] > 0 and b[3] > 0 and b[0] + b[2] <= W and b[1] + b[3] <= H  # noqa: E731
            if dm is not None and fits(dm["face_box"]) and fits(dm["mouth_box"]):
                face_box, mouth_box, box_source = tuple(dm["face_box"]), tuple(dm["mouth_box"]), "review_mixed_units (D's boxes)"
            else:
                face_box = union_box(coords, min(n, len(coords)), W, H, pad=8)
                mouth_box = mouth_box_from_landmarks(lms["A"], W, H) if lms.get("A") is not None else None
                box_source = "facemesh_lips_A" if mouth_box else "heuristic_face_box"
                mouth_box = mouth_box or mouth_box_heuristic(face_box, W, H)

            m = {"clip": clip, "identity": avatar.name, "audio": rel(audio), "frames": n, "fps": args.fps,
                 "frame_size": [W, H], "face_box": list(face_box), "mouth_box": list(mouth_box), "box_source": box_source,
                 "arms": {"A": f"full UNet fp32, TF32 off, {dev_name} + torch TAESD fp32",
                          "N": f"full UNet fp16, {dev16_name} + torch TAESD fp32",
                          "S": f"student {args.student_weights.name} fp32, TF32 off, {dev_name} + torch TAESD fp32",
                          "S16": f"student {args.student_weights.name} fp16, {dev16_name} + torch TAESD fp32"},
                 "unet_ms_per_frame": {arm: round(preds[clip][arm][1], 2) for arm in ARMS},
                 "latent_vs_A": {}, "error_flicker_mouth": {}, "error_flicker_face": {}, "aperture": None}
            for arm in SCORED:
                d = (pred[arm] - pred["A"]).abs()
                m["latent_vs_A"][arm] = {"mae": round(float(d.mean()), 5), "max_abs": round(float(d.max()), 4),
                                         "p95_abs": round(float(torch.quantile(d.flatten(), 0.95)), 4)}
                m[f"{arm}_vs_A"] = compare(full["A"], full[arm], faces["A"], faces[arm], face_box, mouth_box)
                m["error_flicker_mouth"][arm] = round(error_flicker(full["A"], full[arm], mouth_box), 4)
                m["error_flicker_face"][arm] = round(error_flicker(full["A"], full[arm], face_box), 4)
            m["latent_S16_vs_S_mae"] = round(float((pred["S16"] - pred["S"]).abs().mean()), 5)
            if lms.get("A") is not None:
                apa = aperture_series(lms["A"])
                m["aperture"] = {f"{arm}_vs_A": aperture_metrics(apa, aperture_series(lms[arm]))
                                 for arm in SCORED if lms.get(arm) is not None}
            d_num = numbers(dm, "D") if dm is not None else None
            m["D"] = d_num
            if args.new_manifest_info is not None:
                m["protocol"] = NEW_PROTOCOL
                m["avatar_kind"] = "static_portrait; head/body frozen"
                m["landmark_coverage"] = {arm: int(np.isfinite(lms[arm]).all(axis=(1, 2)).sum())
                                          if lms.get(arm) is not None and lms[arm].ndim == 3 and lms[arm].shape[0] == n
                                          and lms[arm].shape[1] >= 416 and lms[arm].shape[2] == 2 else 0 for arm in ARMS}
            m["vs_D"] = {}
            for arm in SCORED:
                m[f"r5_gates_{arm}"] = gates(m, f"{arm}_vs_A")
                m["vs_D"][arm] = ({"status": "not_evaluated", "checks": {},
                                    "incomplete_because": ["new identities: no shipped-chain D baseline was measured"]}
                                   if args.new_manifest_info is not None else non_inferiority(numbers(m, arm), d_num))

            seconds = n / args.fps
            for arm in ARMS:
                write_video(list(full[arm]), clip_dir / f"{arm}.mp4", args.fps, audio, seconds)
            la = wrap(f"A (left) REFERENCE: full UNet fp32, TF32 off, {dev_name} + TAESD | {m['unet_ms_per_frame']['A']:.1f} ms/frame UNet")
            lb = wrap(f"S (right; B in the zoom caption) STUDENT: {args.student_weights.name}, {student['params_m']:.0f}M parameters, "
                      f"fp32 | {summary_line(m, 'S')} | r5 {r5_status(m['r5_gates_S'])}, vs shipped chain D {m['vs_D']['S']['status']}")
            bar_d = (f"shipped Neural Engine chain D on this clip: mouth PSNR {d_num['mouth_psnr_db']:.1f} dB, error flicker "
                     f"{d_num['error_flicker_mouth']:.3f}, lip corr {fmt(d_num['lip_corr'], '.4f')} |d| "
                     f"{fmt(d_num['lip_mean_abs_delta_px'], '.2f')} px" if d_num else "clip not in the D set")
            if args.new_manifest_info is not None:
                bar_d = "STATIC portrait; head/body frozen | r5 vs full fp32 only; D not evaluated; no ANE quality claim"
            ml = wrap(f"{bar_d} | fp16 noise arm {summary_line(m, 'N')}")
            lab = [even(labelled_frame(full["A"][i], full["S"][i], faces["A"][i], faces["S"][i], mouth_box, la, lb, ml)) for i in range(n)]
            write_video(lab, clip_dir / f"{clip}_AS.mp4", args.fps, audio, seconds)
            for sec in (3, 7):
                i = min(int(sec * args.fps), n - 1)
                cv2.imwrite(str(args.out / "stills" / f"{clip}_{sec}s.jpg"), lab[i], [cv2.IMWRITE_JPEG_QUALITY, 82])
            m["files"] = {"dir": rel(clip_dir), "labelled": f"{clip}_AS.mp4", **{arm: f"{arm}.mp4" for arm in ARMS}}
            m["wall_s"] = round(time.time() - t_clip, 1)
            (clip_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
            results.append(m)
            for arm in SCORED:
                print(f"  {summary_line(m, arm)} | r5 {r5_status(m[f'r5_gates_{arm}'])} | vs D {m['vs_D'][arm]['status']}", flush=True)
            print(f"  wrote {clip_dir} in {m['wall_s']} s", flush=True)
        frames = masks = None

    env = {"device": dev_name, "fp16_device": dev16_name, "torch": torch.__version__, "diffusers": diffusers.__version__}
    summarize(results, d_clips, student, env, args)
    print("REVIEW_DONE", flush=True)


# ----------------------------------------------------------------------------- summary
def fmt(x, spec: str) -> str:
    return "—" if x is None else format(x, spec)


def table(results: list[dict]) -> list[list[str]]:
    rows = []
    for m in results:
        for arm in ("D",) + SCORED:
            x = m["D"] if arm == "D" else numbers(m, arm)
            if x is None:
                continue
            vs_d = "bar" if arm == "D" else m["vs_D"][arm]["status"]
            rows.append([m["clip"], arm, fmt(x["latent_mae"], ".4f"), fmt(x["lip_corr"], ".4f"), fmt(x["lip_mean_abs_delta_px"], ".3f"),
                         fmt(x["lip_lag_frames"], "d"), fmt(x["mouth_psnr_db"], ".2f"), fmt(x["mouth_flicker_ratio"], ".3f"),
                         fmt(x["error_flicker_mouth"], ".3f"), fmt(x["mouth_sharpness_ratio"], ".3f"), x["r5"], vs_d])
    return rows


HEADER = ["clip", "arm", "latent mae", "lip corr", "lip mean abs delta px", "lag", "mouth PSNR dB", "mouth flicker ratio",
          "error flicker mouth", "mouth sharpness", "r5", "vs D"]


def summarize(results: list[dict], d_clips: dict, student: dict, env: dict, args) -> None:
    if args.new_manifest_info is not None:
        summarize_new(results, student, env, args)
        return
    v, reasons = verdict(results, d_clips, "S")
    v16, reasons16 = verdict(results, d_clips, "S16")
    n_r5 = {arm: sum(r5_status(m[f"r5_gates_{arm}"]) == "pass" for m in results) for arm in SCORED}
    summary = {"acceptance": {"arm": "S", "all_clips_required": True,
                              "all_r5_gates_required": True, "no_worse_visual_metrics_vs_D": [k for k, _, _ in CHECKS],
                              "deterioration_margins": "zero", "latent_mae_is_informational": True,
                              "matching_frames_fps_rois_and_full_D_aperture_coverage_required": True}, "verdict": v, "reasons": reasons, "verdict_S16_informational": v16, "reasons_S16": reasons16,
               "student": student, "env": env, "frames_per_clip": sorted({m["frames"] for m in results}), "fps": args.fps,
               "r5_gates": R5_GATES, "r5_pass_clips": n_r5, "clips_run": len(results), "d_set": rel(args.mixed_metrics),
               "d_clips": len(d_clips),
               "margins_vs_D": {k: {"higher_is_better": h, "margin": mg} for k, h, mg in CHECKS},
               "clips": [{"clip": m["clip"], "files": m["files"], "D": m["D"],
                          **{arm: {**numbers(m, arm), "vs_D": m["vs_D"][arm]} for arm in SCORED}} for m in results]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = table(results)
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(HEADER)]
    print(f"\n{'  '.join(h.ljust(w) for h, w in zip(HEADER, widths))}")
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(f"\nVERDICT S: {v}")
    for r in reasons:
        print(f"  {r}")
    print(f"S16 (informational): {v16}; r5 pass clips N {n_r5['N']}, S {n_r5['S']}, S16 {n_r5['S16']} of {len(results)}")

    margin_rows = "\n".join(f"| {k} | {'≥ D −' if h else '≤ D +'} {mg} |" for k, h, mg in CHECKS)
    md = f"""# Distilled UNet video review: {v}

Student `{student['weights']}` (config `{student['json']}`, sha256 {student['sha256_12']}…), {student['params_m']:.0f}M parameters
against the teacher's {student['teacher_params_m']:.0f}M. {len(results)} clips, {', '.join(map(str, summary['frames_per_clip']))} frames each
at {args.fps} fps, on {env['device']} (fp16 arms on {env['fp16_device']}; torch {env['torch']}, diffusers {env['diffusers']}). Generated by
`scripts/ios/review_distilled_unet.py`.

Arms against A (full UNet fp32, TF32 off): N full UNet fp16 (the server's precision, the noise floor); S the student
fp32; S16 the student fp16 (the phone's precision, informational). D is the bar: the Neural Engine chain the app ships,
from `{summary['d_set']}`, the same clips.

**Verdict on S: {v}.** S passes when every clip passes r5 against A (aperture corr ≥ {R5_GATES['aperture_corr_min']},
lag {R5_GATES['aperture_lag_frames']}, mean |Δ| ≤ {R5_GATES['aperture_mean_abs_delta_px_max']} px, mouth and face flicker ratio
≤ {R5_GATES['flicker_ratio_max']}, mouth sharpness ratio ≥ {R5_GATES['sharpness_ratio_min']}) and is non-inferior to D on
every one of D's {len(d_clips)} clips. The six visual comparisons below allow zero deterioration. Latent MAE
is informational. Comparisons require matching frame count, fps and ROIs, and at least D's valid aperture coverage.

{chr(10).join(f'- {r}' for r in reasons) or '- every clip passes both'}

S16 (informational): {v16}. r5 passes on {n_r5['N']} (N), {n_r5['S']} (S), {n_r5['S16']} (S16) of {len(results)} clips.

## No-worse checks against D (per clip; all deterioration margins are zero)

| metric | S must be |
|---|---|
{margin_rows}

## Per clip

| {' | '.join(HEADER)} |
|{'---|' * len(HEADER)}
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in rows)}
"""
    (args.out / "summary.md").write_text(md)
    print(f"wrote {args.out / 'summary.json'} and {args.out / 'summary.md'}", flush=True)



def summarize_new(results: list[dict], student: dict, env: dict, args) -> None:
    info = args.new_manifest_info
    assessments = {arm: new_r5_verdict(results, info["expected_clips"], arm) for arm in SCORED}
    limitations = ["Static generated portraits: source head/body remain frozen; this does not test moving-source poses.",
                   "Results characterize these 30 identities and two clips; they cannot prove quality for all future characters.",
                   "No shipped-chain D baseline was measured for these identities; D non-inferiority is not evaluated.",
                   "S16 is PyTorch fp16 on the reported device, not Core ML quantization or iPhone ANE quality evidence."]
    n_r5 = {arm: sum(r5_status(m[f"r5_gates_{arm}"]) == "pass" for m in results) for arm in SCORED}
    summary = {"protocol": NEW_PROTOCOL, "avatar_kind": "static_portrait", "verdict": assessments["S"][0],
               "reasons": assessments["S"][1], "verdicts": {arm: {"verdict": v, "reasons": reasons}
                                                            for arm, (v, reasons) in assessments.items()},
               "acceptance": {"r5_only_against": "A: full UNet fp32, TF32 off", "expected_clips": 60,
                              "frames_per_clip": 250, "fps": 25, "full_finite_facemesh_coverage_required": True,
                              "all_numeric_metrics_finite_required": True, "D_comparison": "not_evaluated",
                              "each_arm_reported_independently": list(SCORED)},
               "student": student, "env": env, "provenance": info, "limitations": limitations,
               "teacher": {"weights_sha256": sha256_file(args.unet_dir / "unet.pth"),
                           "config_sha256": sha256_file(args.unet_dir / "musetalk.json")},
               "student_config_sha256": sha256_file(args.student_json), "r5_gates": R5_GATES,
               "r5_pass_clips": n_r5, "clips_run": len(results),
               "clips": [{"clip": m["clip"], "files": m["files"], "landmark_coverage": m.get("landmark_coverage"),
                          **{arm: numbers(m, arm) for arm in SCORED}} for m in results]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = table(results)
    md = f"""# New static-avatar r5 review: {summary['verdict']}

Thirty held-out generated portraits × two audio clips, 250 frames at 25 fps per clip. Source head and body are
frozen; only generated lips/face regions change. Manifest SHA256: `{info['manifest_sha256']}`.
Student checkpoint SHA256: `{student['sha256']}`. All arms share the source cache, audio windows, blend and fp32 TAESD.
A is full fp32 with TF32 off; N is full fp16; S is student fp32; S16 is student fp16 on {env['fp16_device']}.

Acceptance uses the unchanged r5 thresholds against A: aperture correlation ≥ {R5_GATES['aperture_corr_min']},
lag {R5_GATES['aperture_lag_frames']}, mean aperture error ≤ {R5_GATES['aperture_mean_abs_delta_px_max']} px, mouth/face
flicker ratios ≤ {R5_GATES['flicker_ratio_max']}, mouth sharpness ratio ≥ {R5_GATES['sharpness_ratio_min']}.
The complete manifest set, finite metrics and all 250 finite FaceMesh frames are required independently for each arm.

| Arm | Independent verdict | r5 clips passing |
|---|---|---|
{chr(10).join(f'| {arm} | {assessments[arm][0]} | {n_r5[arm]}/{len(results)} |' for arm in SCORED)}

{chr(10).join('- ' + item for item in limitations)}

## Reasons by arm

{chr(10).join(f'### {arm}: {v}' + chr(10) + chr(10) + (chr(10).join('- ' + r for r in reasons) or '- Every expected clip passes.') + chr(10) for arm, (v, reasons) in assessments.items())}

## Per clip (D is not evaluated)

| {' | '.join(HEADER)} |
|{'---|' * len(HEADER)}
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in rows)}
"""
    (args.out / "summary.md").write_text(md)
    print(f"\nNEW_STATIC_R5 verdicts: " + ", ".join(f"{a} {assessments[a][0]} ({n_r5[a]}/{len(results)} r5)" for a in SCORED), flush=True)
    print("D not evaluated; static portraits; no universal or on-device ANE quality claim.", flush=True)


if __name__ == "__main__":
    main()
