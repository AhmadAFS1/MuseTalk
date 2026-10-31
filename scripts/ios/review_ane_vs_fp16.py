#!/usr/bin/env python
"""Labelled A/B quality review package: the Neural Engine UNet against the PyTorch reference,
the way the r5 INT8 recipe was reviewed (docs/fps_comparisons/4070s_400fps_20260928, scripts/video_ab_round.py).

Arms, per production character and audio clip (same avatar cache, same audio windows, same blend):
  A  reference   PyTorch fp32 UNet on the CPU, TAESD (torch fp32) decode. The repo gate's reference is the
                 server's fp16 PyTorch; fp32 differs from it by ~0.0004 latent mae, a thirtieth of arm B's.
  N  noise arm   PyTorch fp16 UNet on the Mac GPU (the server's precision), same torch TAESD decode. It is
                 what r5's "±1 LSB noise arm" was: the size of difference shipping already accepts, in every
                 metric below. Optional (--no-noise-arm).
  B  candidate   Core ML int8-weight UNet in two fp16-boundary chunks on the Neural Engine (CPU_AND_NE),
                 Core ML TAESD on the Neural Engine: exactly the iPhone chain (bit-identical gate statistics
                 on the iPhone 12 Pro and this Mac). Metrics also report B's latents through the torch decoder
                 (Btd), which isolates the UNet from the decoder.

Per clip it writes (experiments/video_validation/ane_vs_fp16/<identity>__<audio>/):
  A.mp4 B.mp4 [N.mp4]            full-frame renders with audio
  <clip>_ab.mp4                  labelled: A | B full frame; 3x nearest mouth zoom A | B | raw 256 px face |A-B| x8;
                                 burned-in labels and a metrics line
  <clip_id>_blind.mp4            blind: LEFT | RIGHT (A/B randomised per clip), mouth zoom LEFT | RIGHT, no diff,
                                 no metrics; the assignment is in the key file
  faces_{A,B}.npz latents_*.npy  the raw generated faces and latents (reproducibility; metrics re-runnable)
and in docs/ios_unet_bench/review_ane_vs_fp16/: metrics.json, metrics.md, review_sheet.csv (blank), key/blind_key.json,
stills/*.jpg, README.md (what a reviewer does, the acceptance rule).

Metrics (quality_ab_metrics.py's spirit, computed here on RAW composed frames, no codec noise):
  latent mae / max_abs vs A; raw-face mae and PSNR; face-box and mouth-ROI PSNR + SSIM; flicker (mean |f(t)-f(t-1)|)
  ratio B/A in the mouth ROI and face box; mouth Laplacian-variance sharpness ratio B/A; Lab colour shift in the
  face box; and, when a FaceMesh interpreter is given (--facemesh-python, see scripts/quality_ab_facemesh.py),
  the inner-lip aperture series per arm: Pearson A vs B, lag of max cross-correlation, mean |delta| px.
  r5's thresholds are applied where they exist: aperture corr >= 0.97, lag 0, mean |delta| <= 0.5 px, flicker
  ratio <= 1.05, sharpness ratio >= 0.95.

Usage (repo root, .venv-ct):
  python scripts/ios/review_ane_vs_fp16.py --avatars /path/a,/path/b --audios data/audio/ai-assistant.mpga,data/audio/eng.wav \
      --seconds 10 --fps 25 --facemesh-python .venv-mp/bin/python
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from render_evidence import audio_windows, compose, load_avatar, taesd_torch, to_bgr_u8, write_video  # noqa: E402
from musetalk.models.unet import UNet  # noqa: E402

COREML_DIR = REPO_ROOT / "models" / "ios_bench" / "coreml"
MANIFEST = COREML_DIR / "manifest.json"
FACEMESH_HELPER = REPO_ROOT / "scripts" / "quality_ab_facemesh.py"

# MediaPipe FaceMesh indices: inner-lip aperture pairs (quality_ab_metrics.py) and the lips outline.
APERTURE_PAIRS = [(13, 14), (82, 87), (312, 317), (81, 178), (311, 402)]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
R5_GATES = {"aperture_corr_min": 0.97, "aperture_lag_frames": 0, "aperture_mean_abs_delta_px_max": 0.5,
            "flicker_ratio_max": 1.05, "sharpness_ratio_min": 0.95}


# ------------------------------------------------------------------------------- arms
def unet_torch(unet, pe, lat, aud, device, dtype, batch):
    """pred latents fp32 [N,4,32,32], ms/frame (UNet only)."""
    unet = unet.to(device=device, dtype=dtype)
    pe = pe.to(device)
    ts = torch.tensor([0], device=device)
    preds, t_total = [], 0.0
    with torch.inference_mode():
        for s in range(0, lat.shape[0], batch):
            l = lat[s:s + batch].to(device=device, dtype=dtype)
            a = aud[s:s + batch].to(device=device, dtype=dtype)
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            pred = unet(l, ts, encoder_hidden_states=pe(a).to(dtype)).sample
            if device.type == "mps":
                torch.mps.synchronize()
            t_total += time.perf_counter() - t0
            preds.append(pred.float().cpu())
    return torch.cat(preds), t_total / lat.shape[0] * 1000


def decode_torch(dec, preds, batch=8):
    faces = []
    with torch.inference_mode():
        for s in range(0, preds.shape[0], batch):
            img = (dec.decode(preds[s:s + batch].to(dec.dtype)).sample / 2 + 0.5).clamp(0, 1)
            faces.append(to_bgr_u8(img))
    return np.concatenate(faces)


class CoreMLChunks:
    """chunk1 -> chunk2 on the Neural Engine: each stage takes what it needs by name from the inputs and earlier outputs."""

    def __init__(self, parent: str, units):
        import coremltools as ct

        man = json.loads(MANIFEST.read_text())
        names = sorted([k for k, v in man.items() if v.get("chunk_of") == parent], key=lambda k: man[k]["chunk_index"])
        if not names:
            names = [parent]  # a single-program UNet
        self.models = [ct.models.MLModel(man[n]["path"], compute_units=units) for n in names]
        self.inputs = [[i.name for i in m.get_spec().description.input] for m in self.models]
        self.names = names

    def __call__(self, sample16: np.ndarray, audio16: np.ndarray) -> np.ndarray:
        feats = {"sample": sample16, "audio": audio16}
        out = None
        for m, names in zip(self.models, self.inputs):
            out = m.predict({n: feats[n] for n in names})
            feats.update(out)
        return np.asarray(out["pred_latents"], dtype=np.float32)


def unet_coreml(chunks: CoreMLChunks, lat, aud):
    preds, t_total = [], 0.0
    feed0 = (lat[:1].numpy().astype(np.float16), aud[:1].numpy().astype(np.float16))
    for _ in range(3):
        chunks(*feed0)
    for i in range(lat.shape[0]):
        s, a = lat[i:i + 1].numpy().astype(np.float16), aud[i:i + 1].numpy().astype(np.float16)
        t0 = time.perf_counter()
        preds.append(torch.from_numpy(chunks(s, a)))
        t_total += time.perf_counter() - t0
    return torch.cat(preds), t_total / lat.shape[0] * 1000


def decode_coreml(preds, units):
    import coremltools as ct

    dec = ct.models.MLModel(str(COREML_DIR / "taesd_fp16_b1.mlpackage"), compute_units=units)
    faces = []
    for i in range(preds.shape[0]):
        img = np.asarray(dec.predict({"latents": preds[i:i + 1].numpy().astype(np.float16)})["faces"], dtype=np.float32)
        faces.append(to_bgr_u8(img))
    return np.concatenate(faces)


# ---------------------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on luma, Gaussian window 11 / sigma 1.5, the usual constants."""
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def roi(frames: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return frames[:, y:y + h, x:x + w]


def flicker(frames: np.ndarray) -> float:
    f = frames.astype(np.float32)
    return float(np.abs(f[1:] - f[:-1]).mean())


def sharpness(frames: np.ndarray) -> float:
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in frames]))


def lab_shift(a: np.ndarray, b: np.ndarray) -> float:
    la = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3).mean(0) for f in a], axis=0)
    lb = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3).mean(0) for f in b], axis=0)
    return float(np.linalg.norm(la - lb))


def union_box(coords, n_frames, W, H, pad=0):
    xs1 = min(int(c[0]) for c in coords[:n_frames]); ys1 = min(int(c[1]) for c in coords[:n_frames])
    xs2 = max(int(c[2]) for c in coords[:n_frames]); ys2 = max(int(c[3]) for c in coords[:n_frames])
    x1, y1 = max(0, xs1 - pad), max(0, ys1 - pad)
    x2, y2 = min(W, xs2 + pad), min(H, ys2 + pad)
    return (x1, y1, (x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2)


def mouth_box_heuristic(face_box, W, H):
    """Lower-middle of the face box when no landmarks are available (MuseTalk's crop is face-centred)."""
    x, y, w, h = face_box
    mw, mh = int(w * 0.56) // 2 * 2, int(h * 0.30) // 2 * 2
    mx = int(x + w / 2 - mw / 2)
    my = int(y + h * 0.58)
    return (max(0, min(W - mw, mx)), max(0, min(H - mh, my)), mw, mh)


def mouth_box_from_landmarks(lms: np.ndarray, W, H):
    """video_ab_round.py's rule: lips extent over all frames, width x1.6, height 0.62 w, shifted 8 px down."""
    g = lms[:, LIPS]
    ok = ~np.isnan(g[..., 0]).any(axis=1)
    g = g[ok]
    if len(g) == 0:
        return None
    x0, y0 = np.floor(g[..., 0].min()), np.floor(g[..., 1].min())
    x1, y1 = np.ceil(g[..., 0].max()), np.ceil(g[..., 1].max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = int(max(x1 - x0, 1) * 1.6) // 2 * 2
    h = int(w * 0.62) // 2 * 2
    X = int(max(0, min(W - w, cx - w / 2)))
    Y = int(max(0, min(H - h, cy - h / 2 + 8)))
    return (X, Y, w, h)


def facemesh(frames: np.ndarray, python: Path, out_npy: Path) -> np.ndarray | None:
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    cmd = [str(python), str(FACEMESH_HELPER), "--stdin-frames", str(n), str(h), str(w), "--out", str(out_npy)]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        print(f"  facemesh failed: {p.stderr.decode(errors='replace')[-400:]}", flush=True)
        return None
    return np.load(out_npy)


def aperture_series(lms: np.ndarray) -> np.ndarray:
    d = [np.linalg.norm(lms[:, i] - lms[:, j], axis=1) for i, j in APERTURE_PAIRS]
    return np.nanmean(np.stack(d, 1), axis=1)


def aperture_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return {"frames": int(len(a)), "corr": None}
    def pearson(x, y):
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / den) if den > 0 else 0.0
    corr = pearson(a, b)
    lags = {}
    for k in range(-6, 7):
        if k >= 0:
            lags[k] = pearson(a[:len(a) - k] if k else a, b[k:])
        else:
            lags[k] = pearson(a[-k:], b[:len(b) + k])
    best = max(lags.values())
    lag = min((k for k, v in lags.items() if v == best), key=abs)
    return {"frames": int(len(a)), "corr": round(corr, 5), "lag_frames": int(lag),
            "mean_abs_delta_px": round(float(np.abs(a - b).mean()), 4), "p95_abs_delta_px": round(float(np.percentile(np.abs(a - b), 95)), 4),
            "mean_aperture_px_A": round(float(a.mean()), 3), "mean_aperture_px_B": round(float(b.mean()), 3)}


def compare(ref_full, cand_full, ref_faces, cand_faces, face_box, mouth_box) -> dict:
    out = {"raw_face": {"mae_255": round(float(np.abs(ref_faces.astype(np.float32) - cand_faces.astype(np.float32)).mean()), 4),
                        "max_abs_255": int(np.abs(ref_faces.astype(np.int16) - cand_faces.astype(np.int16)).max()),
                        "psnr_db": round(psnr(ref_faces, cand_faces), 2)}}
    for name, box in (("face_box", face_box), ("mouth_roi", mouth_box)):
        ra, cb = roi(ref_full, box), roi(cand_full, box)
        out[name] = {"psnr_db": round(psnr(ra, cb), 2),
                     "ssim": round(float(np.mean([ssim_gray(ra[i], cb[i]) for i in range(len(ra))])), 5),
                     "flicker_ref": round(flicker(ra), 4), "flicker_cand": round(flicker(cb), 4),
                     "flicker_ratio": round(flicker(cb) / max(flicker(ra), 1e-6), 4)}
    rm, cm = roi(ref_full, mouth_box), roi(cand_full, mouth_box)
    sa, sb = sharpness(rm), sharpness(cm)
    out["mouth_sharpness"] = {"ref": round(sa, 2), "cand": round(sb, 2), "ratio": round(sb / max(sa, 1e-6), 4)}
    out["lab_shift_face_box"] = round(lab_shift(roi(ref_full, face_box), roi(cand_full, face_box)), 4)
    return out


# ------------------------------------------------------------------------------ clips
def bar(width: int, lines: list[str], color, size=0.5, step=20) -> np.ndarray:
    img = np.zeros((8 + step * len(lines), width, 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 6 + step * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    return img


def wrap(text: str, width: int = 120) -> list[str]:
    lines, cur = [], ""
    for part in text.split(" | "):
        cand = part if not cur else cur + " | " + part
        if len(cand) > width and cur:
            lines.append(cur); cur = part
        else:
            cur = cand
    return lines + ([cur] if cur else [])


def zoom(frame: np.ndarray, box, factor=3) -> np.ndarray:
    x, y, w, h = box
    return cv2.resize(frame[y:y + h, x:x + w], (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def labelled_frame(fa, fb, face_a, face_b, mouth_box, label_a: list[str], label_b: list[str], metrics_lines: list[str]) -> np.ndarray:
    top = np.hstack([fa, fb])
    width = top.shape[1]
    za, zb = zoom(fa, mouth_box), zoom(fb, mouth_box)
    diff = np.clip(np.abs(face_a.astype(np.int16) - face_b.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    diff = cv2.resize(diff, (za.shape[0], za.shape[0]), interpolation=cv2.INTER_NEAREST)
    bottom = np.hstack([za, zb, diff])
    bottom = cv2.resize(bottom, (width, int(bottom.shape[0] * width / bottom.shape[1])), interpolation=cv2.INTER_NEAREST)
    caption = bar(width, ["3x mouth zoom (nearest)  A | B  |  raw 256 px generated face |A-B| x8 (black = identical)"], (255, 255, 255), 0.45, 18)
    out = np.vstack([bar(width, label_a, (255, 255, 255)), bar(width, label_b, (0, 255, 255)), top, caption, bottom,
                     bar(width, metrics_lines, (255, 255, 0), 0.42, 16)])
    return out


def blind_frame(fl, fr, mouth_box) -> np.ndarray:
    top = np.hstack([fl, fr])
    width = top.shape[1]
    zl, zr = zoom(fl, mouth_box), zoom(fr, mouth_box)
    bottom = np.hstack([zl, np.zeros((zl.shape[0], 24, 3), np.uint8), zr])
    bottom = cv2.resize(bottom, (width, int(bottom.shape[0] * width / bottom.shape[1])), interpolation=cv2.INTER_NEAREST)
    half = width // 2
    head = np.zeros((34, width, 3), np.uint8)
    cv2.putText(head, "LEFT", (half // 2 - 30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(head, "RIGHT", (half + half // 2 - 40, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([head, top, bar(width, ["3x mouth zoom (nearest)   LEFT | RIGHT"], (255, 255, 255), 0.45, 18), bottom])


def even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h // 2 * 2, : w // 2 * 2]


def rel(p: Path) -> str:
    """Repo-relative when inside the repo (the published layout), absolute otherwise."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(p)


# -------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatars", required=True, help="comma-separated avatar directories (full_imgs, mask, coords.pkl, latents.pt)")
    ap.add_argument("--audios", default="data/audio/ai-assistant.mpga,data/audio/eng.wav")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--chunks", default="unet_anec0_int8w_b1", help="manifest parent name of the chunked UNet (or a single UNet)")
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--no-noise-arm", action="store_true")
    ap.add_argument("--facemesh-python", type=Path, default=None, help="interpreter with mediapipe<=0.10.14 for scripts/quality_ab_facemesh.py")
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--videos-dir", type=Path, default=REPO_ROOT / "experiments" / "video_validation" / "ane_vs_fp16")
    ap.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "review_ane_vs_fp16")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--only-package", action="store_true", help="rebuild sheet/key/metrics.md from existing per-clip metrics.json files")
    args = ap.parse_args()
    import coremltools as ct

    units = getattr(ct.ComputeUnit, args.units)
    args.videos_dir.mkdir(parents=True, exist_ok=True)
    (args.docs_dir / "stills").mkdir(parents=True, exist_ok=True)
    (args.docs_dir / "key").mkdir(parents=True, exist_ok=True)
    avatars = [Path(p) for p in args.avatars.split(",") if p]
    audios = [REPO_ROOT / a if not Path(a).is_absolute() else Path(a) for a in args.audios.split(",") if a]
    rng = random.Random(args.seed)

    cpu = torch.device("cpu")
    mps = torch.device("mps") if (torch.backends.mps.is_available() and not args.no_noise_arm) else None
    unet = pe = dec32 = chunks = None
    if not args.only_package:
        print("loading PyTorch UNet fp32 and TAESD...", flush=True)
        w = UNet(unet_config=str(args.unet_dir / "musetalk.json"), model_path=str(args.unet_dir / "unet.pth"), device=cpu)
        unet, pe = w.model.eval().requires_grad_(False), w.pe.eval()
        dec32 = taesd_torch(cpu, torch.float32)
        print(f"loading Core ML {args.chunks} on {args.units}...", flush=True)
        chunks = CoreMLChunks(args.chunks, units)
        print(f"  stages: {chunks.names}", flush=True)

    clips = []
    for avatar in avatars:
        ident = avatar.name
        frames, masks, coords, mask_coords, av_latents = (None,) * 5
        for audio in audios:
            aname = audio.stem
            clip_dir = args.videos_dir / f"{ident}__{aname}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            mpath = clip_dir / "metrics.json"
            if args.only_package and mpath.exists():
                clips.append(json.loads(mpath.read_text()))
                continue
            if frames is None:
                frames, masks, coords, mask_coords, av_latents = load_avatar(avatar)
            H, W = frames[0].shape[:2]
            t_clip = time.time()
            aud_all = audio_windows(audio, args.whisper_dir, args.fps)
            n = min(int(round(args.seconds * args.fps)), aud_all.shape[0])
            seconds = n / args.fps
            aud = aud_all[:n]
            lat = torch.stack([av_latents[i % av_latents.shape[0]] for i in range(n)])
            print(f"\n[{ident} / {aname}] {n} frames = {seconds:.2f} s at {args.fps} fps, {W}x{H}", flush=True)

            # A: fp32 CPU + torch TAESD
            pred_a, ms_a = unet_torch(unet, pe, lat, aud, cpu, torch.float32, batch=4)
            faces_a = decode_torch(dec32, pred_a)
            full_a = np.stack(compose(faces_a, frames, masks, coords, mask_coords))
            print(f"  A fp32 CPU: {ms_a:.0f} ms/frame UNet", flush=True)
            # N: fp16 MPS + torch TAESD (noise arm)
            pred_n = faces_n = full_n = None
            ms_n = None
            if mps is not None:
                pred_n, ms_n = unet_torch(unet, pe, lat, aud, mps, torch.float16, batch=1)
                unet.to(device=cpu, dtype=torch.float32); pe.to(cpu)
                faces_n = decode_torch(dec32, pred_n)
                full_n = np.stack(compose(faces_n, frames, masks, coords, mask_coords))
                print(f"  N fp16 MPS: {ms_n:.1f} ms/frame UNet", flush=True)
            # B: Core ML chunks on the ANE + Core ML TAESD
            pred_b, ms_b = unet_coreml(chunks, lat, aud)
            faces_b = decode_coreml(pred_b, units)
            full_b = np.stack(compose(faces_b, frames, masks, coords, mask_coords))
            faces_btd = decode_torch(dec32, pred_b)
            full_btd = np.stack(compose(faces_btd, frames, masks, coords, mask_coords))
            print(f"  B ANE chunks: {ms_b:.1f} ms/frame UNet", flush=True)

            np.savez_compressed(clip_dir / "faces_A.npz", faces=faces_a)
            np.savez_compressed(clip_dir / "faces_B.npz", faces=faces_b)
            np.save(clip_dir / "latents_A.npy", pred_a.numpy().astype(np.float16))
            np.save(clip_dir / "latents_B.npy", pred_b.numpy().astype(np.float16))
            if pred_n is not None:
                np.save(clip_dir / "latents_N.npy", pred_n.numpy().astype(np.float16))

            # landmarks (optional) and boxes
            face_box = union_box(coords, min(n, len(coords)), W, H, pad=8)
            lms_a = lms_b = lms_n = None
            if args.facemesh_python is not None and FACEMESH_HELPER.exists():
                lms_a = facemesh(full_a, args.facemesh_python, clip_dir / "landmarks_A.npy")
                lms_b = facemesh(full_b, args.facemesh_python, clip_dir / "landmarks_B.npy")
                if full_n is not None:
                    lms_n = facemesh(full_n, args.facemesh_python, clip_dir / "landmarks_N.npy")
            mouth_box = mouth_box_from_landmarks(lms_a, W, H) if lms_a is not None else None
            mouth_source = "facemesh_lips_A" if mouth_box else "heuristic_face_box"
            if mouth_box is None:
                mouth_box = mouth_box_heuristic(face_box, W, H)

            # metrics
            m = {"clip": f"{ident}__{aname}", "identity": ident, "audio": rel(audio),
                 "frames": n, "seconds": seconds, "fps": args.fps, "frame_size": [W, H], "face_box": list(face_box),
                 "mouth_box": list(mouth_box), "mouth_box_source": mouth_source,
                 "arms": {"A": "PyTorch fp32 UNet (CPU) + torch TAESD fp32", "N": "PyTorch fp16 UNet (MPS) + torch TAESD fp32" if pred_n is not None else None,
                          "B": f"Core ML {args.chunks} ({' -> '.join(chunks.names)}) on {args.units} + Core ML TAESD fp16 ({args.units})",
                          "Btd": "B's latents decoded with torch TAESD fp32 (UNet-only attribution)"},
                 "unet_ms_per_frame": {"A_cpu_fp32": round(ms_a, 1), "N_mps_fp16": round(ms_n, 2) if ms_n else None, "B_ane_chunks": round(ms_b, 2)},
                 "latent_vs_A": {}, "B_vs_A": compare(full_a, full_b, faces_a, faces_b, face_box, mouth_box),
                 "Btd_vs_A": compare(full_a, full_btd, faces_a, faces_btd, face_box, mouth_box)}
            d = (pred_b - pred_a).abs()
            m["latent_vs_A"]["B"] = {"mae": round(float(d.mean()), 5), "max_abs": round(float(d.max()), 4), "p95_abs": round(float(torch.quantile(d.flatten(), 0.95)), 4)}
            if pred_n is not None:
                dn = (pred_n - pred_a).abs()
                m["latent_vs_A"]["N"] = {"mae": round(float(dn.mean()), 5), "max_abs": round(float(dn.max()), 4), "p95_abs": round(float(torch.quantile(dn.flatten(), 0.95)), 4)}
                m["N_vs_A"] = compare(full_a, full_n, faces_a, faces_n, face_box, mouth_box)
            if lms_a is not None and lms_b is not None:
                apa, apb = aperture_series(lms_a), aperture_series(lms_b)
                m["aperture"] = {"B_vs_A": aperture_metrics(apa, apb)}
                if lms_n is not None:
                    m["aperture"]["N_vs_A"] = aperture_metrics(apa, aperture_series(lms_n))
                m["aperture"]["missing_frames"] = {"A": int(np.isnan(lms_a[:, 0, 0]).sum()), "B": int(np.isnan(lms_b[:, 0, 0]).sum())}
            else:
                m["aperture"] = None
            m["r5_gates_B"] = gates(m, "B_vs_A")
            if pred_n is not None:
                m["r5_gates_N"] = gates(m, "N_vs_A")

            # videos
            write_video(list(full_a), clip_dir / "A.mp4", args.fps, audio, seconds)
            write_video(list(full_b), clip_dir / "B.mp4", args.fps, audio, seconds)
            if full_n is not None:
                write_video(list(full_n), clip_dir / "N.mp4", args.fps, audio, seconds)
            la = wrap(f"A (left) REFERENCE: PyTorch fp32 UNet + TAESD, this Mac CPU | {ms_a:.0f} ms/frame here; the server runs the same network in fp16 at 2.5 ms")
            lb = wrap(f"B (right) CANDIDATE: Core ML int8 UNet, two fp16-boundary chunks, Neural Engine + Core ML TAESD | {ms_b:.1f} ms/frame on this Mac's ANE, "
                      f"59.6 ms on iPhone 12 Pro (bit-identical) | gate mae {m['latent_vs_A']['B']['mae']:.4f} vs reference")
            ml = wrap(metrics_line(m))
            lab = [even(labelled_frame(full_a[i], full_b[i], faces_a[i], faces_b[i], mouth_box, la, lb, ml)) for i in range(n)]
            write_video(lab, clip_dir / f"{ident}__{aname}_ab.mp4", args.fps, audio, seconds)
            for sec in (3, 7):
                i = min(int(sec * args.fps), n - 1)
                cv2.imwrite(str(args.docs_dir / "stills" / f"{ident}__{aname}_{sec}s.jpg"), lab[i], [cv2.IMWRITE_JPEG_QUALITY, 82])
            left_is_a = rng.random() < 0.5
            L, R = (full_a, full_b) if left_is_a else (full_b, full_a)
            blind = [even(blind_frame(L[i], R[i], mouth_box)) for i in range(n)]
            blind_name = f"{ident}__{aname}_blind.mp4"
            write_video(blind, clip_dir / blind_name, args.fps, audio, seconds)
            m["files"] = {"dir": rel(clip_dir), "labelled": f"{ident}__{aname}_ab.mp4", "blind": blind_name,
                          "A": "A.mp4", "B": "B.mp4", "N": "N.mp4" if full_n is not None else None}
            m["blind"] = {"left": "A" if left_is_a else "B", "right": "B" if left_is_a else "A"}
            m["wall_s"] = round(time.time() - t_clip, 1)
            mpath.write_text(json.dumps(m, indent=2) + "\n")
            clips.append(m)
            print(f"  {metrics_line(m)}", flush=True)
            print(f"  wrote {clip_dir} in {m['wall_s']} s", flush=True)
        frames = None  # free the avatar

    package(clips, args)
    print("REVIEW_DONE", flush=True)


def gates(m: dict, arm: str) -> dict:
    c = m[arm]
    g = {"flicker_mouth_ratio_le_1.05": c["mouth_roi"]["flicker_ratio"] <= R5_GATES["flicker_ratio_max"],
         "flicker_face_ratio_le_1.05": c["face_box"]["flicker_ratio"] <= R5_GATES["flicker_ratio_max"],
         "mouth_sharpness_ratio_ge_0.95": c["mouth_sharpness"]["ratio"] >= R5_GATES["sharpness_ratio_min"]}
    ap = (m.get("aperture") or {}).get(arm)
    if ap and ap.get("corr") is not None:
        g["aperture_corr_ge_0.97"] = ap["corr"] >= R5_GATES["aperture_corr_min"]
        g["aperture_lag_0"] = ap["lag_frames"] == R5_GATES["aperture_lag_frames"]
        g["aperture_mean_abs_delta_le_0.5px"] = ap["mean_abs_delta_px"] <= R5_GATES["aperture_mean_abs_delta_px_max"]
    g["all_pass"] = all(v for k, v in g.items() if k != "all_pass")
    return g


def metrics_line(m: dict) -> str:
    b, lat = m["B_vs_A"], m["latent_vs_A"]["B"]
    parts = [f"latent mae {lat['mae']:.4f} max {lat['max_abs']:.2f}",
             f"raw face mae {b['raw_face']['mae_255']:.3f}/255 max {b['raw_face']['max_abs_255']}",
             f"face PSNR {b['face_box']['psnr_db']:.1f} dB mouth {b['mouth_roi']['psnr_db']:.1f} dB SSIM {b['mouth_roi']['ssim']:.4f}",
             f"flicker ratio mouth {b['mouth_roi']['flicker_ratio']:.3f} face {b['face_box']['flicker_ratio']:.3f}",
             f"mouth sharpness ratio {b['mouth_sharpness']['ratio']:.3f}"]
    ap = (m.get("aperture") or {}).get("B_vs_A")
    if ap and ap.get("corr") is not None:
        parts.append(f"lip aperture corr {ap['corr']:.4f} lag {ap['lag_frames']} mean|d| {ap['mean_abs_delta_px']:.2f} px")
    if "N_vs_A" in m:
        n, latn = m["N_vs_A"], m["latent_vs_A"]["N"]
        parts.append(f"[fp16 noise arm: latent mae {latn['mae']:.4f}, raw face mae {n['raw_face']['mae_255']:.3f}, mouth PSNR {n['mouth_roi']['psnr_db']:.1f} dB]")
    return " | ".join(parts)


def package(clips: list[dict], args) -> None:
    docs = args.docs_dir
    clips = sorted(clips, key=lambda c: c["clip"])
    (docs / "metrics.json").write_text(json.dumps({"chunks": args.chunks, "units": args.units, "seconds": args.seconds, "fps": args.fps,
                                                   "r5_gates": R5_GATES, "clips": clips}, indent=2) + "\n")
    # per-clip metrics table (B and the fp16 noise arm N side by side)
    def f(x, fmt):
        return "—" if x is None else format(x, fmt)
    rows = ["| identity | audio | latent mae B / N | raw face mae /255 B / N | face PSNR dB B / N | mouth PSNR dB B / N | mouth SSIM B / N | flicker mouth B / N | sharpness mouth B / N | lip corr B / N | lip mean\\|Δ\\| px B / N | lag B | r5 gates B |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in clips:
        b, n = c["B_vs_A"], c.get("N_vs_A")
        lb, ln = c["latent_vs_A"]["B"], c["latent_vs_A"].get("N")
        apb = (c.get("aperture") or {}).get("B_vs_A") or {}
        apn = (c.get("aperture") or {}).get("N_vs_A") or {}
        g = c.get("r5_gates_B", {})
        passed = sum(1 for k, v in g.items() if k != "all_pass" and v); total = sum(1 for k in g if k != "all_pass")
        rows.append(f"| {c['identity']} | {Path(c['audio']).stem} | {lb['mae']:.4f} / {f(ln and ln['mae'], '.4f')} | "
                    f"{b['raw_face']['mae_255']:.3f} / {f(n and n['raw_face']['mae_255'], '.3f')} | "
                    f"{b['face_box']['psnr_db']:.1f} / {f(n and n['face_box']['psnr_db'], '.1f')} | "
                    f"{b['mouth_roi']['psnr_db']:.1f} / {f(n and n['mouth_roi']['psnr_db'], '.1f')} | "
                    f"{b['mouth_roi']['ssim']:.4f} / {f(n and n['mouth_roi']['ssim'], '.4f')} | "
                    f"{b['mouth_roi']['flicker_ratio']:.3f} / {f(n and n['mouth_roi']['flicker_ratio'], '.3f')} | "
                    f"{b['mouth_sharpness']['ratio']:.3f} / {f(n and n['mouth_sharpness']['ratio'], '.3f')} | "
                    f"{f(apb.get('corr'), '.4f')} / {f(apn.get('corr'), '.4f')} | {f(apb.get('mean_abs_delta_px'), '.2f')} / {f(apn.get('mean_abs_delta_px'), '.2f')} | "
                    f"{f(apb.get('lag_frames'), 'd')} | {passed}/{total} |")
    (docs / "metrics.md").write_text("\n".join(rows) + "\n")

    # blind order: shuffled, numbered; key kept apart from the sheet
    rng = random.Random(args.seed + 1)
    order = list(range(len(clips)))
    rng.shuffle(order)
    key = {}
    with (docs / "review_sheet.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "identity", "audio", "seconds", "blind_video", "reviewer",
                    "lipsync_worse_side (L/R/none)", "lipsync_severity (0 none,1 slight,2 clear,3 unacceptable)",
                    "mouth_artifacts_side (L/R/none)", "artifacts_severity (0-3)",
                    "flicker_side (L/R/none)", "flicker_severity (0-3)",
                    "seam_or_blend_side (L/R/none)", "seam_severity (0-3)",
                    "overall_worse_side (L/R/none)", "overall_severity (0-3)", "notes"])
        for k, idx in enumerate(order, 1):
            c = clips[idx]
            cid = f"c{k:02d}"
            key[cid] = {"clip": c["clip"], "left": c["blind"]["left"], "right": c["blind"]["right"],
                        "labelled_video": f"{c['files']['dir']}/{c['files']['labelled']}"}
            w.writerow([cid, c["identity"], Path(c["audio"]).stem, f"{c['seconds']:.1f}", f"{c['files']['dir']}/{c['files']['blind']}",
                        "", "", "", "", "", "", "", "", "", "", "", ""])
    (docs / "key" / "blind_key.json").write_text(json.dumps({"seed": args.seed, "A": "reference (PyTorch fp32 + TAESD)",
                                                              "B": "candidate (Core ML Neural Engine chunks + Core ML TAESD)",
                                                              "clips": key}, indent=2) + "\n")
    # reviewer instructions
    n_clips = len(clips)
    idents = sorted({c["identity"] for c in clips})
    readme = f"""# Labelled quality review: Neural Engine UNet vs the PyTorch reference

{n_clips} clips, {len(idents)} production characters ({', '.join(idents)}), {args.seconds:.0f} s each at {args.fps} fps.
Arm A is the PyTorch fp32 UNet with TAESD (the reference the repo's gate uses, in fp32); arm B is the two-chunk
Core ML int8 UNet on the Neural Engine with Core ML TAESD: the chain the iPhone runs, bit-identical to the
iPhone 12 Pro's output on the gate frames. Both arms share the avatar cache, the Whisper windows and the blend.
A third arm, N (PyTorch fp16 on the Mac GPU, the server's precision), is not in the clips but is in
`metrics.md`: it shows how large a difference the shipping server already accepts, metric by metric.

This follows how r5 (the INT8 TensorRT recipe) was accepted: labelled A/B clips per identity with a mouth zoom
and a raw-face difference panel, a per-identity metrics table, and a human verdict on the whole set.

## What the reviewer does

1. **Blind pass.** Open the clips in `review_sheet.csv` order (`blind_video` column; paths are relative to the repo
   root, clips live under `experiments/video_validation/ane_vs_fp16/`). Each shows LEFT | RIGHT full frame with
   audio and a 3x mouth zoom. Which side is A or B is randomised per clip and recorded only in `key/blind_key.json`;
   do not open the key before finishing the sheet. Watch each clip twice, once at the mouth zoom.
2. For each clip fill the sheet: for lip sync, mouth artefacts (teeth, blur, ghosting, texture), flicker, and
   seam/blend, write the worse side (`L`, `R`, or `none`) and a severity: 0 = cannot tell, 1 = slight (visible
   when looking for it), 2 = clear, 3 = unacceptable. Then an overall worse side and severity, and notes.
3. **Unblind** with `key/blind_key.json` and count, per identity, the verdicts that fell on B.
4. **Labelled pass.** Open the `_ab.mp4` clips (A left, B right, labels, mouth zoom, |A-B| x8 face panel and the
   metrics line) for every clip where B was called worse, to see what the difference is.

## Acceptance rule (r5's: every identity acceptable, no systematic artefact)

B is **accepted** when, after unblinding, (a) no clip has a severity ≥ 2 attributed to B in any column, (b) every
identity has at least one clip where B was not called worse overall, and (c) across all clips B is called worse
overall at most 3 times more often than A (with {n_clips} clips, side picks under indistinguishability split
about evenly; a run of B-worse calls is the signal). If (a) fails on one identity only, B is **accepted with a
reservation** for that character. Otherwise B is **rejected** and the labelled clips plus `metrics.md` say which
component (UNet precision: compare the B and N columns; decoder: compare `B_vs_A` with `Btd_vs_A` in
`metrics.json`) carries the difference.

## Files

- `review_sheet.csv` — blank sheet, one row per clip in blind order.
- `key/blind_key.json` — LEFT/RIGHT assignment per clip (open after the blind pass).
- `metrics.md`, `metrics.json` — per-clip metrics for B and the fp16 noise arm N against A; r5's thresholds
  (aperture corr ≥ 0.97, lag 0, mean |Δ| ≤ 0.5 px, flicker ratio ≤ 1.05, sharpness ratio ≥ 0.95).
- `stills/` — one labelled frame per clip at 3 s and 7 s.
- Clips and raw material: `experiments/video_validation/ane_vs_fp16/<identity>__<audio>/` (A.mp4, B.mp4, N.mp4,
  `*_ab.mp4`, `*_blind.mp4`, faces_A/B.npz, latents_*.npy, landmarks_*.npy). Rebuild with
  `scripts/ios/review_ane_vs_fp16.py`.
"""
    (docs / "README.md").write_text(readme)
    print(f"package: {docs} ({n_clips} clips, {len(idents)} identities)", flush=True)


if __name__ == "__main__":
    main()
