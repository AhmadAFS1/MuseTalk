"""G-TRACK + chin-target A/B for the TRT TAESD backend (plan item 2.1).

For each complete chin render (experiments/avatar_diversity_20260927/<id>: TensorRT
FP16 bs8 UNet + compiled TAESD + 100% refined chin, 24 fps, 240 frames) this:
  1. regenerates the 240 UNet predictions with the shipping .ts UNet, exactly as
     character_factory/h3_avatar_workflow/render_stage.py does;
  2. decodes them with (a) compiled TAESD + the render's own postprocess and (b) the
     TRT TAESD fused uint8 path;
  3. runs the render's per-frame chin loop for each arm with a fresh FaceMesh tracker
     (same worker, same shared-memory paste, same .25/.5/.25 filter, same
     chin.corrected_refined), then re-tracks the refined output frames;
  4. reports: compiled faces vs the stored reference faces.npz (harness reproduction),
     generated jaw+lip landmark deviation TRT vs compiled (G-TRACK), chin-delta and
     target-chin-error differences, output pixel differences, and writes a labelled
     side-by-side video (compiled | TRT | |diff| x16) per identity.
The target chin error uses validate_stage.py's formula on raw (not re-encoded)
output frames for both arms, window 24..216.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk")
OUT = Path(__file__).resolve().parent
H3 = ROOT / "character_factory/h3_avatar_workflow"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(H3)]
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV = torch.device("cuda:0")
DEFAULT_IDS = ["black_woman", "east_asian_man_goatee", "south_asian_woman"]
DIV = Path("/workspace/experiments/avatar_diversity_20260927")


def load_env():
    for line in (ROOT / ".runtime/musetalk_trt_local_sm89.env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)
    os.environ["MUSETALK_TRT_FALLBACK"] = "0"


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def render_post(pixels):  # render_stage.generate()'s postprocess, verbatim
    return pixels.float().mul(255).round().clamp(0, 255).to(torch.uint8).flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def chin_loop(chin, tracker, frames, d, faces):
    """render_stage.main's tracking + emit loop, serial (same order and filter)."""
    n = len(frames)
    d["g"] = np.zeros((n, 478, 2), np.float32)
    d["chin_delta"] = np.zeros_like(d["chin_delta"])
    tracker.reset()
    outputs, queue = [], []
    state = {"previous": None}

    def emit(record, next_delta):
        i, face, g, delta = record
        d["g"][i] = g
        old = delta if state["previous"] is None else state["previous"]
        d["chin_delta"][i] = np.clip(.25 * old + .5 * delta + .25 * next_delta, 0, .18)
        outputs.append(chin.corrected_refined(frames[i], d, i, face))
        state["previous"] = delta

    for i in range(n):
        g, _ = tracker.track(frames[i], faces[i], d["cache"]["boxes"][i])
        source, target = chin.curves(d["p"][i], g)
        delta = source - target
        queue.append((i, faces[i], g, delta))
        if len(queue) == 2:
            emit(queue.pop(0), delta)
    if queue:
        emit(queue[0], queue[0][3])
    return outputs, d["g"].copy(), d["chin_delta"].copy()


def track_frames(tracker, frames):
    tracker.reset()
    rows = []
    for f in frames:
        tracker.frame[:] = f
        answer = tracker.command("frame").split()
        assert answer[0] == "ok"
        rows.append(tracker.points.copy())
    return np.asarray(rows, np.float32)


def target_error(chin, source, generated, final):
    err = []
    for p, g, q in zip(source, generated, final):
        _, _, down, _ = chin.axes(p)
        err.append(float((q[152] - g[152]) @ down))
    a = np.abs(np.asarray(err))[24:216]
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}, np.asarray(err)


def dist_stats(a):
    a = np.asarray(a, float).ravel()
    return {"mean": float(a.mean()), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def write_video(path, left, right, audio, label_l, label_r, crf):
    h, w = left[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w * 3}x{h}", "-r", "24",
           "-i", "pipe:0", "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "slow",
           "-crf", str(crf), "-pix_fmt", "yuv444p", "-c:a", "aac", "-b:a", "96k", "-t", "10",
           "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for a, b in zip(left, right):
        diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 16, 0, 255).astype(np.uint8)
        tiles = [a.copy(), b.copy(), diff]
        for t, lab in zip(tiles, (label_l, label_r, "|diff| x16")):
            cv2.rectangle(t, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(t, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        p.stdin.write(np.ascontiguousarray(np.concatenate(tiles, axis=1)).tobytes())
    p.stdin.close()
    assert p.wait() == 0


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--video-crf", type=int, default=12)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    load_env()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    import chin
    from backend import Tracker
    from scripts import vae_fast_decoder as vfd
    from scripts.trt_runtime import load_unet_trt_backend

    t0 = time.time()
    compiled = vfd.TaesdVaeDecodeBackend.load(device=DEV, runtime_dtype=torch.float16)
    compiled.warmup([8])
    trt = vfd.load_taesd_trt_backend(DEV, torch.float16, model=compiled.model)
    trt.warmup([8])
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    unet = load_unet_trt_backend(device=DEV, force=True)
    assert unet is not None
    stamp = torch.tensor([0], device="cuda")
    res = {"schema": "gtrack_taesd_trt_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "engine_key": trt.meta.get("key"), "unet": getattr(unet, "name", type(unet).__name__),
           "load_s": time.time() - t0, "identities": {}}
    tracker = Tracker("/workspace", OUT / "gtrack_tracker.log")
    idx = np.array(chin.JAW + chin.LIPS)
    try:
        for ident in [s for s in args.ids.split(",") if s]:
            out = DIV / ident
            frames = read_frames(out / "source.mp4")
            assert len(frames) == 240, len(frames)
            d = dict(cache=torch.load(out / "cache.pt", map_location="cpu", weights_only=False),
                     masks=np.load(out / "masks.npz"), p=np.load(out / "source_landmarks.npy"),
                     g=np.zeros((240, 478, 2), np.float32))
            chin.prepare_refined(d, frames)
            stored_faces = np.load(out / "faces.npz")["faces"]
            stored_g = np.load(out / "generated_landmarks.npy")
            stored_delta = np.load(out / "chin_delta.npy")
            preds = []
            for i in range(0, 240, 8):
                z = unet(d["cache"]["latents"][i:i + 8].cuda(), stamp,
                         encoder_hidden_states=d["cache"]["audio"][i:i + 8].cuda()).sample
                preds.append(z.clone())
            faces = {"compiled": np.concatenate([render_post(compiled.decode(z, .18215, torch.float16)) for z in preds]),
                     "trt": np.concatenate([trt.decode_bgr_u8(z).cpu().numpy() for z in preds])}
            torch.cuda.synchronize()
            rec = {"frames": 240}
            rec["compiled_faces_equal_stored_reference"] = bool(np.array_equal(faces["compiled"], stored_faces))
            fd = np.abs(faces["trt"].astype(np.int16) - faces["compiled"].astype(np.int16))
            rec["face_lsb_trt_vs_compiled"] = {"max": int(fd.max()), "mean": float(fd.mean()),
                                               "rows104_max": int(fd[:, 104:].max()),
                                               "rows104_mean": float(fd[:, 104:].mean())}
            arms = {}
            for arm in ("compiled", "trt", "compiled_rerun"):
                src = faces["trt" if arm == "trt" else "compiled"]
                t_arm = time.time()
                outputs, g, delta = chin_loop(chin, tracker, frames, d, list(src))
                final = track_frames(tracker, outputs)
                terr, err = target_error(chin, d["p"], g, final)
                arms[arm] = {"outputs": outputs, "g": g, "delta": delta, "final": final, "terr": terr, "err": err,
                             "seconds": time.time() - t_arm}
            c, t, r = arms["compiled"], arms["trt"], arms["compiled_rerun"]
            spans = np.array([chin.axes(p)[3] for p in d["p"]])
            rec["compiled_landmarks_equal_stored_reference"] = bool(np.array_equal(c["g"], stored_g))
            rec["compiled_landmarks_vs_stored_max_px"] = float(np.abs(c["g"] - stored_g).max())
            rec["compiled_chin_delta_equal_stored"] = bool(np.array_equal(c["delta"], stored_delta))
            dev = np.linalg.norm(t["g"][:, idx] - c["g"][:, idx], axis=-1)
            rec["G_TRACK_jaw_lip_deviation_px"] = dist_stats(dev)
            rec["G_TRACK_jaw_only_px"] = dist_stats(np.linalg.norm(t["g"][:, chin.JAW] - c["g"][:, chin.JAW], axis=-1))
            rec["G_TRACK_lips_only_px"] = dist_stats(np.linalg.norm(t["g"][:, chin.LIPS] - c["g"][:, chin.LIPS], axis=-1))
            rec["tracker_rerun_noise_px"] = dist_stats(np.linalg.norm(r["g"][:, idx] - c["g"][:, idx], axis=-1))
            rec["chin_delta_diff_px"] = dist_stats(np.abs(t["delta"] - c["delta"]) * spans[:, None])
            rec["target_chin_abs_error_px"] = {"compiled": c["terr"], "trt": t["terr"], "compiled_rerun": r["terr"],
                                               "mean_diff_trt_minus_compiled": t["terr"]["mean"] - c["terr"]["mean"]}
            rec["per_frame_target_error_diff_px"] = dist_stats(np.abs(t["err"] - c["err"])[24:216])
            od = np.stack([np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
                           for a, b in zip(c["outputs"], t["outputs"])])
            rec["output_frame_lsb_trt_vs_compiled"] = {"max": int(od.max()), "mean": float(od.mean()),
                                                       "frac_nonzero": float((od > 0).mean())}
            rec["output_rerun_identical"] = bool(all(np.array_equal(a, b) for a, b in zip(c["outputs"], r["outputs"])))
            rec["final_landmarks_jaw_lip_deviation_px"] = dist_stats(
                np.linalg.norm(t["final"][:, idx] - c["final"][:, idx], axis=-1))
            rec["arm_seconds"] = {k: v["seconds"] for k, v in arms.items()}
            gate = (rec["G_TRACK_jaw_lip_deviation_px"]["mean"] <= 0.05
                    and rec["G_TRACK_jaw_lip_deviation_px"]["p99"] <= 0.15
                    and abs(rec["target_chin_abs_error_px"]["mean_diff_trt_minus_compiled"]) <= 0.05)
            rec["G_TRACK"] = "PASS" if gate else "FAIL"
            if not args.no_video:
                vid = OUT / f"gtrack_{ident}_compiled_vs_trt_refined_chin100.mp4"
                write_video(vid, c["outputs"], t["outputs"], out / "speech.wav",
                            f"{ident}: compiled TAESD (today)", "TRT TAESD fp16 (MUSETALK_TAESD_BACKEND=trt)",
                            args.video_crf)
                rec["video"] = {"file": vid.name, "bytes": vid.stat().st_size, "crf": args.video_crf,
                                "layout": "refined 100% chin output: compiled | TRT | |diff| x16; 24 fps, 10 s"}
            res["identities"][ident] = rec
            print(ident, json.dumps({k: rec[k] for k in ("compiled_faces_equal_stored_reference",
                                                         "compiled_landmarks_equal_stored_reference",
                                                         "face_lsb_trt_vs_compiled", "G_TRACK_jaw_lip_deviation_px",
                                                         "tracker_rerun_noise_px", "target_chin_abs_error_px",
                                                         "chin_delta_diff_px", "output_frame_lsb_trt_vs_compiled",
                                                         "G_TRACK")}, default=str), flush=True)
            (OUT / "gtrack_taesd_trt.json").write_text(json.dumps(res, indent=1, default=str))
    finally:
        tracker.close()
    res["verdict"] = "PASS" if all(v["G_TRACK"] == "PASS" for v in res["identities"].values()) else "FAIL"
    res["seconds"] = time.time() - t0
    res["tags"] = "[M] measured by this run"
    (OUT / "gtrack_taesd_trt.json").write_text(json.dumps(res, indent=1, default=str))
    print("verdict", res["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
