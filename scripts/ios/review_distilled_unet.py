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
FAIL when any scored check fails. INCOMPLETE when nothing failed but something went unscored: no FaceMesh, a clip
outside the D set or shorter than D's, or a D clip not run.

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
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

# Non-inferiority margins against D, the Neural Engine chain the app ships (review_mixed_units/metrics.json,
# 2026-11-03, the same 12 clips). D ran once per clip, so there is no run-to-run spread to size them from. Five of
# them are the largest change in D's own score between the two audios of one character, over the five characters
# that pass r5 (the Japanese one fails it), rounded up to one significant figure. A student worse than D by less
# than that is within what changing the clip alone does to the shipped chain. The flicker ratio margin is the cost
# 5.7 accepted for the mixed Neural Engine + GPU arm.
LIP_CORR_MARGIN = 0.002         # es_03_mateo 0.9953 / 0.9968, 0.0015 apart
LIP_DELTA_MARGIN_PX = 0.05      # en_01_maya 0.246 / 0.202 px, 0.044 apart; the GPU arm G, numerically A, scores 0.03-0.05
MOUTH_PSNR_MARGIN_DB = 0.8      # en_01_maya 46.95 / 47.69 dB, 0.74 apart
MOUTH_SHARPNESS_MARGIN = 0.005  # latina_guided 0.9933 / 0.9977, 0.0044 apart (blur is a distilled student's likely failure)
ERROR_FLICKER_MARGIN = 0.04     # es_03_mateo 0.859 / 0.895, 0.036 apart; G scores 0.047-0.067
FLICKER_RATIO_MARGIN = 0.02     # arm M's mouth flicker ratio was 0.007-0.019 above D's, accepted within r5's 1.05
# (key, higher is better, margin)
CHECKS = (("lip_corr", True, LIP_CORR_MARGIN), ("lip_mean_abs_delta_px", False, LIP_DELTA_MARGIN_PX),
          ("mouth_psnr_db", True, MOUTH_PSNR_MARGIN_DB), ("mouth_sharpness_ratio", True, MOUTH_SHARPNESS_MARGIN),
          ("error_flicker_mouth", False, ERROR_FLICKER_MARGIN), ("mouth_flicker_ratio", False, FLICKER_RATIO_MARGIN))


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
            unet, pe = unet.to(dev, dtype), pe.to(dev)
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
    return {"frames": m["frames"], "latent_mae": m["latent_vs_A"][arm]["mae"], "lip_corr": ap.get("corr"),
            "lip_lag_frames": ap.get("lag_frames"), "lip_mean_abs_delta_px": ap.get("mean_abs_delta_px"),
            "mouth_psnr_db": c["mouth_roi"]["psnr_db"], "mouth_flicker_ratio": c["mouth_roi"]["flicker_ratio"],
            "error_flicker_mouth": m["error_flicker_mouth"][arm], "mouth_sharpness_ratio": c["mouth_sharpness"]["ratio"],
            "r5": r5_status(g)}


def non_inferiority(x: dict, d: dict | None) -> dict:
    if d is None:
        return {"status": "incomplete", "checks": {}, "incomplete_because": ["clip not in the D set"]}
    checks = {}
    for key, higher, margin in CHECKS:
        if x.get(key) is None or d.get(key) is None:
            continue
        bound = d[key] - margin if higher else d[key] + margin
        checks[key] = {"value": x[key], "D": d[key], "bound": round(bound, 5),
                       "pass": bool(x[key] >= bound if higher else x[key] <= bound)}
    why = []
    if "lip_corr" not in checks:
        why.append("lip aperture not scored (no FaceMesh)")
    if x["frames"] != d["frames"]:
        why.append(f"{x['frames']} frames, D has {d['frames']}")
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
    ap.add_argument("--mixed-metrics", type=Path, default=MIXED_METRICS, help="review_mixed_units metrics.json: arm D's per-clip numbers")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    args = ap.parse_args()
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
    d_clips = {c["clip"]: c for c in json.loads(args.mixed_metrics.read_text())["clips"]}

    windows = {a: audio_windows(a, args.whisper_dir, args.fps) for a in audios}
    clips = []
    for avatar in avatars:
        av_latents = load_avatar(avatar)[4]  # cycled as load_avatar truncates it, like the earlier reviews
        for audio in audios:
            n = min(int(round(args.seconds * args.fps)), windows[audio].shape[0], args.frames or 1 << 30)
            clips.append({"clip": f"{avatar.name}__{audio.stem}", "avatar": avatar, "audio": audio, "n": n,
                          "lat": torch.stack([av_latents[i % av_latents.shape[0]] for i in range(n)]),
                          "aud": windows[audio][:n]})
    print(f"{len(clips)} clips, {sum(c['n'] for c in clips)} frames per arm, on {dev_name}", flush=True)

    nets = [("teacher", args.unet_dir / "musetalk.json", args.unet_dir / "unet.pth",
             [("A", torch.float32, dev), ("N", torch.float16, dev16)]),
            ("student", args.student_json, args.student_weights, [("S", torch.float32, dev), ("S16", torch.float16, dev16)])]
    preds, params = unet_arms(clips, nets, args.batch)
    sha = hashlib.sha256()
    with args.student_weights.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            sha.update(block)
    student = {"json": rel(args.student_json), "weights": rel(args.student_weights), "sha256_12": sha.hexdigest()[:12],
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
            fits = lambda b: b[0] + b[2] <= W and b[1] + b[3] <= H  # noqa: E731
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
            m["vs_D"] = {}
            for arm in SCORED:
                m[f"r5_gates_{arm}"] = gates(m, f"{arm}_vs_A")
                m["vs_D"][arm] = non_inferiority(numbers(m, arm), d_num)

            seconds = n / args.fps
            for arm in ARMS:
                write_video(list(full[arm]), clip_dir / f"{arm}.mp4", args.fps, audio, seconds)
            la = wrap(f"A (left) REFERENCE: full UNet fp32, TF32 off, {dev_name} + TAESD | {m['unet_ms_per_frame']['A']:.1f} ms/frame UNet")
            lb = wrap(f"S (right; B in the zoom caption) STUDENT: {args.student_weights.name}, {student['params_m']:.0f}M parameters, "
                      f"fp32 | {summary_line(m, 'S')} | r5 {r5_status(m['r5_gates_S'])}, vs shipped chain D {m['vs_D']['S']['status']}")
            bar_d = (f"shipped Neural Engine chain D on this clip: mouth PSNR {d_num['mouth_psnr_db']:.1f} dB, error flicker "
                     f"{d_num['error_flicker_mouth']:.3f}, lip corr {fmt(d_num['lip_corr'], '.4f')} |d| "
                     f"{fmt(d_num['lip_mean_abs_delta_px'], '.2f')} px" if d_num else "clip not in the D set")
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
    v, reasons = verdict(results, d_clips, "S")
    v16, reasons16 = verdict(results, d_clips, "S16")
    n_r5 = {arm: sum(r5_status(m[f"r5_gates_{arm}"]) == "pass" for m in results) for arm in SCORED}
    summary = {"verdict": v, "reasons": reasons, "verdict_S16_informational": v16, "reasons_S16": reasons16,
               "student": student, "env": env, "frames_per_clip": sorted({m["frames"] for m in results}), "fps": args.fps,
               "r5_gates": R5_GATES, "r5_pass_clips": n_r5, "clips_run": len(results), "d_set": rel(args.mixed_metrics),
               "d_clips": len(d_clips),
               "margins_vs_D": {k: {"higher_is_better": h, "margin": mg} for k, h, mg in CHECKS},
               "clips": [{"clip": m["clip"], "files": m["files"], "D": m["D"],
                          **{arm: {**numbers(m, arm), "vs_D": m["vs_D"][arm]} for arm in SCORED}} for m in results]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = table(results)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(HEADER)]
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
every one of D's {len(d_clips)} clips.

{chr(10).join(f'- {r}' for r in reasons) or '- every clip passes both'}

S16 (informational): {v16}. r5 passes on {n_r5['N']} (N), {n_r5['S']} (S), {n_r5['S16']} (S16) of {len(results)} clips.

## Non-inferiority margins against D (per clip; sources in the script's constants)

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


if __name__ == "__main__":
    main()
