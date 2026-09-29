"""Single-stream serial baseline: render_stage.py's preparation, warmup and timed loop, verbatim.

Only the offline tail of render_stage.main() (diagnostics, npz/mp4 writing, render.json) is left
out, and the loop runs `repeats` times per identity with a tracker reset before each run (the first
run is exactly render_stage's). Everything is in one process with backends loaded once.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

from . import paths


def hash_frames(frames):
    h = hashlib.sha256()
    for f in frames:
        h.update(f.tobytes())
    return h.hexdigest()


def run_serial(unet, decoder, sf, identities, repeats, log_dir):
    import cv2
    import numpy as np
    import torch

    paths.add_import_paths()
    import chin
    from backend import Tracker

    results = {}
    with torch.inference_mode():
        for ident in identities:
            out = paths.ACCEPTED_ROOT / ident
            ref = paths.accepted_render(ident)
            cap = cv2.VideoCapture(str(out / "source.mp4"))
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            assert len(frames) == 240
            d = dict(cache=torch.load(out / "cache.pt", map_location="cpu", weights_only=False), masks=np.load(out / "masks.npz"),
                     p=np.load(out / "source_landmarks.npy"), g=np.zeros((240, 478, 2), np.float32))
            chin.prepare_refined(d, frames)
            stamp = torch.tensor([0], device="cuda")
            for _ in range(4):
                z = unet(d["cache"]["latents"][:8].cuda(), stamp, encoder_hidden_states=d["cache"]["audio"][:8].cuda()).sample
                decoder.decode(z, sf, torch.float16)
            torch.cuda.synchronize()
            tracker = Tracker(paths.WORKSPACE, log_dir / f"serial_{ident}_tracking.log")
            runs = []
            try:
                for r in range(repeats):
                    tracker.reset()
                    outputs = []
                    faces = []
                    queue = []
                    previous = None
                    timing = dict(tracking_ipc_ms=0., compose_ms=0.)

                    def emit(record, next_delta):
                        nonlocal previous
                        i, face, g, delta = record
                        d["g"][i] = g
                        old = delta if previous is None else previous
                        d["chin_delta"][i] = np.clip(.25 * old + .5 * delta + .25 * next_delta, 0, .18)
                        start = time.perf_counter()
                        outputs.append(chin.corrected_refined(frames[i], d, i, face))
                        timing["compose_ms"] += (time.perf_counter() - start) * 1000
                        previous = delta

                    @torch.inference_mode()
                    def generate(i):
                        z = unet(d["cache"]["latents"][i:i + 8].cuda(), stamp, encoder_hidden_states=d["cache"]["audio"][i:i + 8].cuda()).sample
                        pixels = decoder.decode(z, sf, torch.float16)
                        if not torch.isfinite(pixels).all():
                            raise RuntimeError("Non-finite generated pixels")
                        return pixels.float().mul(255).round().clamp(0, 255).to(torch.uint8).flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy()

                    with ThreadPoolExecutor(max_workers=1) as pool:
                        start = time.perf_counter()
                        future = pool.submit(generate, 0)
                        for base in range(0, 240, 8):
                            pixels = future.result()
                            if base + 8 < 240:
                                future = pool.submit(generate, base + 8)
                            faces.extend(pixels)
                            for j, face in enumerate(pixels):
                                i = base + j
                                t = time.perf_counter()
                                g, _ = tracker.track(frames[i], face, d["cache"]["boxes"][i])
                                timing["tracking_ipc_ms"] += (time.perf_counter() - t) * 1000
                                source, target = chin.curves(d["p"][i], g)
                                delta = source - target
                                queue.append((i, face, g, delta))
                                if len(queue) == 2:
                                    emit(queue.pop(0), delta)
                        if queue:
                            emit(queue[0], queue[0][3])
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                    assert len(outputs) == 240
                    raw, fh = hash_frames(outputs), hash_frames(faces)
                    runs.append(dict(repeat=r, render_seconds=elapsed, warm_render_fps=240 / elapsed, timing=timing,
                                     raw_refined_sha256=raw, generated_faces_sha256=fh,
                                     raw_match=raw == ref["raw_refined_sha256"], faces_match=fh == ref["generated_faces_sha256"]))
                    print("SERIAL", ident, r, f"{240 / elapsed:.1f} fps", "raw_match", raw == ref["raw_refined_sha256"],
                          "faces_match", fh == ref["generated_faces_sha256"], flush=True)
                    del outputs, faces
            finally:
                tracker.close()
            fps = sorted(x["warm_render_fps"] for x in runs)
            results[ident] = dict(runs=runs, median_fps=fps[len(fps) // 2] if len(fps) % 2 else (fps[len(fps) // 2 - 1] + fps[len(fps) // 2]) / 2,
                                  accepted_warm_render_fps=ref["warm_render_fps"])
            del d, frames
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    return results
