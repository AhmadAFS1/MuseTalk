#!/usr/bin/env python
"""Quality review of the Neural Engine + GPU lipsync pipeline (`LipsyncPipeline.gpuWorkers`).

With GPU workers the frames of one turn come from two chains with different numerics: the Neural Engine's (int8
chunks, fp16 accumulation, gate mae ~0.013) and the GPU's (plain fp16 UNet, fp32 accumulation, ~0.0004). Each chain
passes alone; the question is whether the mix does, in particular whether switching numerics from frame to frame
flickers where neither chain does.

Arms per production character and audio clip (same avatar caches, audio windows and blend as review_ane_vs_fp16.py,
whose saved arm A is reused from experiments/video_validation/ane_vs_fp16/):
  A  reference   PyTorch fp32 UNet + torch TAESD (faces_A.npz, landmarks_A.npy of the earlier package)
  D  before      Core ML chunks on the Neural Engine (--ane-chunks; the 2026-11-03 run used the folded V2 ones) +
                 Core ML TAESD on the Neural Engine: every frame of the pipeline without GPU workers
  G  GPU         Core ML plain fp16 UNet on the GPU + Core ML TAESD on the GPU: the GPU workers' frames
  M  mixed       D on even frames, G on odd ones. The pipeline gives the GPU 38-50 % of the frames, not in a fixed
                 pattern; alternating every frame is the most switches a clip can have
Metrics: review_ane_vs_fp16.compare() (raw face, face box and mouth PSNR / SSIM / flicker ratio, mouth sharpness,
Lab shift), FaceMesh lip aperture against A, the r5 gates, and the error flicker: mean |e(t) - e(t-1)| of the error
e = arm - A in the mouth ROI, which is where alternating numerics would show.

Writes experiments/video_validation/mixed_units/<clip>/ (D.mp4, G.mp4, M.mp4, <clip>_DM.mp4: D | M labelled with
mouth zoom; faces and latents) and docs/ios_unet_bench/review_mixed_units/ (metrics.json, metrics.md, stills).

Usage (repo root, .venv-ct):
  python scripts/ios/review_mixed_units.py --avatars /path/a,/path/b \\
      --ane-chunks <dir>/unet_anec0_int8w_b1_chunk1.mlpackage,<dir>/unet_anec0_int8w_b1_chunk2.mlpackage \\
      --gpu-unet models/ios_bench/coreml/unet_fp16_b1.mlpackage --facemesh-python .venv-mp/bin/python
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from render_evidence import audio_windows, compose, load_avatar, to_bgr_u8, write_video  # noqa: E402
from review_ane_vs_fp16 import (COREML_DIR, FACEMESH_HELPER, R5_GATES, aperture_metrics, aperture_series, compare, even,  # noqa: E402
                                facemesh, gates, labelled_frame, rel, roi, wrap)

ARMS = ("D", "G", "M")


class Chain:
    """UNet stages (each takes what it needs by name from the inputs and earlier outputs) + TAESD, on one compute unit."""

    def __init__(self, unet_paths: list[Path], units):
        import coremltools as ct

        self.models = [ct.models.MLModel(str(p), compute_units=units) for p in unet_paths]
        self.inputs = [[i.name for i in m.get_spec().description.input] for m in self.models]
        self.taesd = ct.models.MLModel(str(COREML_DIR / "taesd_fp16_b1.mlpackage"), compute_units=units)

    def latent(self, sample16: np.ndarray, audio16: np.ndarray) -> np.ndarray:
        feats = {"sample": sample16, "audio": audio16}
        out = None
        for m, names in zip(self.models, self.inputs):
            out = m.predict({n: feats[n] for n in names})
            feats.update(out)
        return np.asarray(out["pred_latents"], dtype=np.float32)

    def face(self, latent: np.ndarray) -> np.ndarray:
        img = np.asarray(self.taesd.predict({"latents": latent.astype(np.float16)})["faces"], dtype=np.float32)
        return to_bgr_u8(img)

    def run(self, lat: torch.Tensor, aud: torch.Tensor):
        s0, a0 = lat[:1].numpy().astype(np.float16), aud[:1].numpy().astype(np.float16)
        for _ in range(2):
            self.face(self.latent(s0, a0))
        preds, faces, t = [], [], 0.0
        for i in range(lat.shape[0]):
            s, a = lat[i:i + 1].numpy().astype(np.float16), aud[i:i + 1].numpy().astype(np.float16)
            t0 = time.perf_counter()
            p = self.latent(s, a)
            faces.append(self.face(p))
            t += time.perf_counter() - t0
            preds.append(p)
        return np.concatenate(preds), np.concatenate(faces), t / lat.shape[0] * 1000


def error_flicker(ref: np.ndarray, cand: np.ndarray, box) -> float:
    e = roi(cand, box).astype(np.float32) - roi(ref, box).astype(np.float32)
    return float(np.abs(e[1:] - e[:-1]).mean())


def r5_status(g: dict) -> str:
    """'FAIL' if any gate failed, 'incomplete' if the aperture gates could not be scored (no FaceMesh), else 'pass'."""
    if not all(v for k, v in g.items() if k != "all_pass"):
        return "FAIL"
    return "pass" if "aperture_corr_ge_0.97" in g else "incomplete"


def chain_name(paths: list[Path]) -> str:
    return " -> ".join(f"{p.parent.name}/{p.name}" for p in paths)


def summary_line(m: dict, arm: str) -> str:
    c = m[f"{arm}_vs_A"]
    parts = [f"{arm}: latent mae {m['latent_vs_A'][arm]['mae']:.4f}", f"mouth PSNR {c['mouth_roi']['psnr_db']:.1f} dB",
             f"flicker ratio mouth {c['mouth_roi']['flicker_ratio']:.3f}", f"error flicker {m['error_flicker_mouth'][arm]:.3f}",
             f"sharpness {c['mouth_sharpness']['ratio']:.3f}"]
    ap = (m.get("aperture") or {}).get(f"{arm}_vs_A")
    if ap and ap.get("corr") is not None:
        parts.append(f"lip corr {ap['corr']:.4f} lag {ap['lag_frames']} |d| {ap['mean_abs_delta_px']:.2f} px")
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatars", required=True, help="comma-separated avatar directories (full_imgs, mask, coords.pkl, latents.pt)")
    ap.add_argument("--audios", default="data/audio/ai-assistant.mpga,data/audio/eng.wav")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--ane-chunks", required=True, help="comma-separated .mlpackage stages of the Neural Engine UNet, in order")
    ap.add_argument("--gpu-unet", type=Path, default=COREML_DIR / "unet_fp16_b1.mlpackage")
    ap.add_argument("--facemesh-python", type=Path, default=None)
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    ap.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "experiments" / "video_validation" / "ane_vs_fp16")
    ap.add_argument("--videos-dir", type=Path, default=REPO_ROOT / "experiments" / "video_validation" / "mixed_units")
    ap.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs" / "ios_unet_bench" / "review_mixed_units")
    args = ap.parse_args()
    if args.facemesh_python is not None and not FACEMESH_HELPER.exists():
        sys.exit(f"--facemesh-python given but {FACEMESH_HELPER} is missing")

    import coremltools as ct

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    (args.docs_dir / "stills").mkdir(parents=True, exist_ok=True)
    avatars = [Path(p) for p in args.avatars.split(",") if p]
    audios = [REPO_ROOT / a if not Path(a).is_absolute() else Path(a) for a in args.audios.split(",") if a]
    ane_paths = [Path(p) for p in args.ane_chunks.split(",") if p]
    print(f"loading Neural Engine chain {chain_name(ane_paths)} and GPU chain {args.gpu_unet.name}...", flush=True)
    ane = Chain(ane_paths, ct.ComputeUnit.CPU_AND_NE)
    gpu = Chain([args.gpu_unet], ct.ComputeUnit.CPU_AND_GPU)

    clips = []
    for avatar in avatars:
        ident = avatar.name
        frames = None
        for audio in audios:
            clip = f"{ident}__{audio.stem}"
            ref_dir = args.reference_dir / clip
            ref_m = json.loads((ref_dir / "metrics.json").read_text())
            if frames is None:
                frames, masks, coords, mask_coords, av_latents = load_avatar(avatar)
            n = min(ref_m["frames"], int(round(args.seconds * args.fps)))
            face_box, mouth_box = tuple(ref_m["face_box"]), tuple(ref_m["mouth_box"])
            t_clip = time.time()
            aud = audio_windows(audio, args.whisper_dir, args.fps)[:n]
            lat = torch.stack([av_latents[i % av_latents.shape[0]] for i in range(n)])
            print(f"\n[{clip}] {n} frames", flush=True)

            faces_a = np.load(ref_dir / "faces_A.npz")["faces"][:n]
            pred_a = np.load(ref_dir / "latents_A.npy").astype(np.float32)[:n]
            full_a = np.stack(compose(faces_a, frames, masks, coords, mask_coords))
            pred_d, faces_d, ms_d = ane.run(lat, aud)
            pred_g, faces_g, ms_g = gpu.run(lat, aud)
            odd = (np.arange(n) % 2 == 1)
            pred_m = np.where(odd[:, None, None, None], pred_g, pred_d)
            faces_m = np.where(odd[:, None, None, None], faces_g, faces_d)
            full = {"D": np.stack(compose(faces_d, frames, masks, coords, mask_coords)),
                    "G": np.stack(compose(faces_g, frames, masks, coords, mask_coords))}
            full["M"] = np.where(odd[:, None, None, None], full["G"], full["D"])
            faces = {"D": faces_d, "G": faces_g, "M": faces_m}
            preds = {"D": pred_d, "G": pred_g, "M": pred_m}
            print(f"  D Neural Engine {ms_d:.1f} ms/frame, G GPU {ms_g:.1f} ms/frame (Python, UNet + TAESD)", flush=True)

            clip_dir = args.videos_dir / clip
            clip_dir.mkdir(parents=True, exist_ok=True)
            m = {"clip": clip, "identity": ident, "audio": rel(audio), "frames": n, "fps": args.fps,
                 "face_box": list(face_box), "mouth_box": list(mouth_box),
                 "arms": {"A": "PyTorch fp32 UNet + torch TAESD (reused from review_ane_vs_fp16)",
                          "D": f"Core ML {chain_name(ane_paths)} on CPU_AND_NE + Core ML TAESD on CPU_AND_NE",
                          "G": f"Core ML {args.gpu_unet.name} on CPU_AND_GPU + Core ML TAESD on CPU_AND_GPU",
                          "M": "D on even frames, G on odd frames"},
                 "ms_per_frame_python": {"D": round(ms_d, 2), "G": round(ms_g, 2)},
                 "latent_vs_A": {}, "error_flicker_mouth": {}, "error_flicker_face": {}}
            for arm in ARMS:
                d = np.abs(preds[arm] - pred_a)
                m["latent_vs_A"][arm] = {"mae": round(float(d.mean()), 5), "max_abs": round(float(d.max()), 4)}
                m[f"{arm}_vs_A"] = compare(full_a, full[arm], faces_a, faces[arm], face_box, mouth_box)
                m["error_flicker_mouth"][arm] = round(error_flicker(full_a, full[arm], mouth_box), 4)
                m["error_flicker_face"][arm] = round(error_flicker(full_a, full[arm], face_box), 4)
                np.savez_compressed(clip_dir / f"faces_{arm}.npz", faces=faces[arm])
            m["M_vs_D"] = compare(full["D"], full["M"], faces_d, faces_m, face_box, mouth_box)

            m["aperture"] = None
            lms_a_path = ref_dir / "landmarks_A.npy"
            if args.facemesh_python is not None and lms_a_path.exists():
                apa = aperture_series(np.load(lms_a_path)[:n])
                m["aperture"] = {}
                for arm in ARMS:
                    lms = facemesh(full[arm], args.facemesh_python, clip_dir / f"landmarks_{arm}.npy")
                    if lms is not None:
                        m["aperture"][f"{arm}_vs_A"] = aperture_metrics(apa, aperture_series(lms))
            for arm in ARMS:
                m[f"r5_gates_{arm}"] = gates(m, f"{arm}_vs_A")

            seconds = n / args.fps
            for arm in ARMS:
                write_video(list(full[arm]), clip_dir / f"{arm}.mp4", args.fps, audio, seconds)
            ld = wrap(f"D (left) BEFORE: every frame on the Neural Engine ({chain_name(ane_paths)} + TAESD) | {summary_line(m, 'D')}")
            lm = wrap(f"M (right) MIXED: even frames as D, odd frames on the GPU ({args.gpu_unet.name} + TAESD) | {summary_line(m, 'M')}")
            dm = [even(labelled_frame(full["D"][i], full["M"][i], faces_d[i], faces_m[i], mouth_box, ld, lm, [f"frame {i} ({'GPU' if odd[i] else 'Neural Engine'})"]))
                  for i in range(n)]
            write_video(dm, clip_dir / f"{clip}_DM.mp4", args.fps, audio, seconds)
            for sec in (3, 7):
                i = min(int(sec * args.fps), n - 1)
                cv2.imwrite(str(args.docs_dir / "stills" / f"{clip}_{sec}s.jpg"), dm[i], [cv2.IMWRITE_JPEG_QUALITY, 82])
            m["files"] = {"dir": rel(clip_dir), "labelled": f"{clip}_DM.mp4", "D": "D.mp4", "G": "G.mp4", "M": "M.mp4"}
            m["wall_s"] = round(time.time() - t_clip, 1)
            (clip_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
            clips.append(m)
            for arm in ARMS:
                print(f"  {summary_line(m, arm)} | r5 {r5_status(m[f'r5_gates_{arm}'])}", flush=True)
            print(f"  wrote {clip_dir} in {m['wall_s']} s", flush=True)
        frames = None

    package(clips, args)
    print("REVIEW_DONE", flush=True)


def package(clips: list[dict], args) -> None:
    (args.docs_dir / "metrics.json").write_text(json.dumps({"r5_gates": R5_GATES, "clips": clips}, indent=2) + "\n")
    rows = ["| Clip | Arm | latent mae | mouth PSNR | flicker ratio mouth | error flicker mouth | sharpness | lip corr | lip mean abs delta px | r5 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m in clips:
        for arm in ARMS:
            c = m[f"{arm}_vs_A"]
            ap = (m.get("aperture") or {}).get(f"{arm}_vs_A") or {}
            rows.append(f"| {m['clip']} | {arm} | {m['latent_vs_A'][arm]['mae']:.4f} | {c['mouth_roi']['psnr_db']:.1f} | "
                        f"{c['mouth_roi']['flicker_ratio']:.3f} | {m['error_flicker_mouth'][arm]:.3f} | {c['mouth_sharpness']['ratio']:.3f} | "
                        f"{ap.get('corr', '')} | {ap.get('mean_abs_delta_px', '')} | {r5_status(m[f'r5_gates_{arm}'])} |")
    (args.docs_dir / "metrics.md").write_text(
        "# Neural Engine + GPU pipeline: mixed-numerics review\n\n"
        "Arms against A (PyTorch fp32): D every frame on the Neural Engine, G every frame on the GPU, M the two "
        "alternating frame by frame. Generated by `scripts/ios/review_mixed_units.py`.\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
