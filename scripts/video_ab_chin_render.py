#!/usr/bin/env python
"""Chin-recipe clip renderer for the standing video A/B tool (scripts/video_ab.py).

A copy/adaptation of
  /workspace/experiments/chin_fps_validation_20260927/run.py      (timed warm repeats, identity data)
  character_factory/h3_avatar_workflow/backend.py + render_stage.py (setup, tracker, refined seam loop)
The originals are not modified. The recipe is unchanged: native-encoder avatar-cache latents ->
TensorRT FP16 UNet (bs8) -> TAESD decode -> per-frame MediaPipe FaceMesh tracking of the generated
face -> 100% chin alignment with the refined seam (chin.prepare_refined / chin.corrected_refined)
and the symmetric .25/.5/.25 one-frame-lookahead jaw filter, one GPU batch prefetched.

What is different from render_stage.py:
  * --repo selects the code tree. Every product module (musetalk.*, scripts.*, chin.py,
    tracker_worker.py) and the .runtime env file come from that tree, so the SAME render loop runs
    against the pre-change tree (/workspace/MuseTalk) or a candidate tree (the worktree). This
    script's own directory is removed from sys.path, and after loading, every imported module
    whose file lies in a known tree must lie in --repo (recorded as module_origin_check).
  * --flags K=V,... are applied after the tree's .runtime env file and the recipe overrides, so
    candidate backends are picked up (MUSETALK_TAESD_BACKEND=trt, MUSETALK_UNET_BACKEND=trt_stagewise,
    MUSETALK_TRT_UNET_CUDAGRAPHS=manual, ...). A requested backend that silently falls back is an
    error unless --allow-fallback. The env layer is applied BEFORE chin/blending are imported (so
    MUSETALK_BLEND_* flags would take effect); render_stage.py imports chin first, which is
    equivalent because the env file sets the blend flags to their defaults.
  * --repeats timed warm runs (default 3, the run.py protocol); the first run's frames are kept,
    the others are hashed and must be identical (determinism = the noise floor of an A/B).
  * Frames are stored losslessly (libx264rgb -qp 0, bgr24, native 24 fps) with a per-frame
    SHA-256 list (same scheme as replay_scheduler_exactness.sha_array) in clip.json, the input
    format of video_ab.py. The mp4 re-encode of render_stage.py is not produced.

Modes:
  GPU (default): needs CUDA; run under scripts/box_guard.sh (video_ab.py does this in
    docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/gpu_sequence.sh).
  --dry-run: CPU only, no CUDA, no tracker. Uses the identity's SAVED faces (faces.npz) and saved
    generated landmarks instead of fresh inference/tracking, and composes --frames frames with the
    repo's chin.py/blending. It checks the import isolation and the composition half of the recipe.

Usage:
  python scripts/video_ab_chin_render.py --repo /workspace/MuseTalk --out-root DIR \
      [--identities japanese,latina] [--flags K=V,...] [--arm-name NAME] [--repeats 3]
Writes DIR/<identity>/{frames.mkv, clip.json, generated_landmarks.npy, chin_delta.npy}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parent.parent
# Drop this script's directory (and anything else inside the worktree) from the import path before
# anything product-related is imported; --repo is inserted explicitly below.
sys.path[:] = [p for p in sys.path if p not in ("", ".") and not Path(p or ".").resolve().is_relative_to(SCRIPT.parent)]

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import importlib.util  # noqa: E402

SCHEMA = "video_ab_clip_v1"
MAIN_TREE = Path(os.getenv("VIDEO_AB_MAIN_TREE", "/workspace/MuseTalk"))
TREE_CACHE = Path(os.getenv("VIDEO_AB_TREE_CACHE", "/workspace/.cache/video_ab/trees"))
DATA_ROOT = Path("/workspace/experiments/portrait_jaw_video_20260926")
TRACKER_PYTHON = Path("/workspace/SoulX-FlashHead/.venv/bin/python")
ENV_FILE_REL = ".runtime/musetalk_trt_local_sm89.env"
# backend.setup(): forced after the env file (TAESD recipe, no silent fallback).
RECIPE_ENV = {"MUSETALK_TRT_FALLBACK": "0", "MUSETALK_VAE_BACKEND": "taesd", "MUSETALK_TAESD_WARMUP_BATCHES": "8"}
BATCH = 8
FPS = 24


# ----------------------------------------------------------------------------- helpers
def sha_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(array) -> str:
    """Identical to replay_scheduler_exactness.sha_array (shape|dtype| prefix + raw bytes)."""
    import numpy as np
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.shape}|{a.dtype}|".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def digest(hashes) -> str:
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def dump(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def parse_flags(text: str | None) -> dict:
    flags = {}
    for item in filter(None, (x.strip() for x in (text or "").split(","))):
        if "=" not in item:
            raise SystemExit(f"--flags expects K=V items, got {item!r}")
        k, v = item.split("=", 1)
        flags[k.strip()] = v.strip()
    return flags


def parse_env_file(path: Path) -> dict:
    """backend.setup() semantics: KEY=VALUE lines, '#' comments skipped."""
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k] = v
    return values


def apply_env_layers(repo: Path, flags: dict) -> dict:
    """env file -> recipe overrides -> arm flags; returns {key: {value, source}}."""
    layers = [(ENV_FILE_REL, parse_env_file(repo / ENV_FILE_REL)), ("recipe (backend.setup)", RECIPE_ENV),
              ("arm flags", flags)]
    report = {}
    for name, values in layers:
        for k, v in values.items():
            os.environ[k] = v
            report[k] = {"value": v, "source": name}
    return report


def module_origin_check(repo: Path) -> dict:
    """Every loaded module file inside a known tree must be inside --repo (this script excepted)."""
    repo = repo.resolve()
    roots = {p.resolve() for p in (WORKTREE, MAIN_TREE, TREE_CACHE) if p.exists()}
    foreign, in_repo = [], 0
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            path = Path(f).resolve()
        except OSError:
            continue
        if path == SCRIPT:
            continue
        if path.is_relative_to(repo):
            in_repo += 1
        elif any(path.is_relative_to(r) for r in roots):
            foreign.append({"module": name, "file": str(path)})
    return {"repo": str(repo), "modules_in_repo": in_repo, "foreign": foreign, "ok": not foreign}


def load_chin(repo: Path):
    path = repo / "character_factory/h3_avatar_workflow/chin.py"
    spec = importlib.util.spec_from_file_location("video_ab_h3_chin", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def identity_paths(who: str, data_root: Path) -> dict:
    folder = data_root / f"{who}_new"
    lm = data_root / "analysis" / f"{who}_new_portrait"
    manifest = json.loads((folder / "manifest.json").read_text())
    return {"folder": folder, "lm": lm, "manifest": manifest, "source": Path(manifest["source"]),
            "audio": Path(manifest["audio"]), "cache": folder / "cache.pt", "masks": folder / "masks.npz",
            "faces": folder / "faces.npz", "source_landmarks": lm / "source_landmarks.npy",
            "saved_generated_landmarks": lm / "generated_landmarks.npy"}


def read_frames(path: Path, n: int):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) != n:
        raise RuntimeError(f"{path}: read {len(frames)} frames, expected {n}")
    return frames


def mouth_box(chin, p, width: int, height: int) -> list:
    """Fixed mouth+chin ROI for the 3x zoom: W//3 wide, centred on the median source lip centre,
    from just above the lips to below the source chin point (FaceMesh 152)."""
    import numpy as np
    lips = p[:, chin.LIPS]
    cx = float(np.median(lips[:, :, 0].mean(1)))
    top = float(np.median(lips[:, :, 1].min(1)))
    chin_y = float(np.median(p[:, 152, 1]))
    w = width // 3
    h = int(round(min(max(chin_y - top + 34, 0.55 * w), 0.95 * w)))
    x0 = int(round(cx - w / 2))
    y0 = int(round(top - 12))
    x0 = max(0, min(width - w, x0))
    y0 = max(0, min(height - h, y0))
    return [x0, y0, x0 + w, y0 + h]


class LosslessWriter:
    """bgr24 -> libx264rgb -qp 0 (the replay harness's LosslessWriter settings)."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264rgb", "-qp", "0",
             "-preset", "ultrafast", "-threads", "2", str(path)], stdin=subprocess.PIPE)

    def write(self, frame) -> None:
        import numpy as np
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def decoded_hashes(path: Path, width: int, height: int) -> list:
    import numpy as np
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path), "-vsync", "0", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    out = []
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        out.append(sha_array(np.frombuffer(buf, np.uint8).reshape(height, width, 3)))
    proc.stdout.close()
    proc.wait()
    return out


def store_clip(dest: Path, frames, fps: int) -> dict:
    h, w = frames[0].shape[:2]
    hashes = [sha_array(f) for f in frames]
    writer = LosslessWriter(dest / "frames.mkv", w, h, fps)
    for f in frames:
        writer.write(f)
    writer.close()
    decoded = decoded_hashes(dest / "frames.mkv", w, h)
    if decoded != hashes:
        bad = next((i for i, (a, b) in enumerate(zip(decoded, hashes)) if a != b), min(len(decoded), len(hashes)))
        raise RuntimeError(f"lossless round trip failed at frame {bad} ({len(decoded)} decoded / {len(hashes)})")
    return {"frames_video": "frames.mkv", "frames_video_sha256": sha_file(dest / "frames.mkv"),
            "frames_video_codec": "libx264rgb -qp 0 (lossless, bgr24); decode round trip verified",
            "frame_sha256": hashes, "frames_digest": digest(hashes), "width": w, "height": h}


# ----------------------------------------------------------------------------- tracker
class Tracker:
    """backend.Tracker with the worker taken from --repo (FaceMesh in the SoulX venv, shared-memory IPC)."""

    def __init__(self, repo: Path, python: Path, log_path: Path):
        import numpy as np
        from multiprocessing import shared_memory
        self.shm = shared_memory.SharedMemory(create=True, size=896 * 512 * 3 + 478 * 2 * 4)
        self.frame = np.ndarray((896, 512, 3), np.uint8, buffer=self.shm.buf)
        self.points = np.ndarray((478, 2), np.float32, buffer=self.shm.buf, offset=self.frame.nbytes)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a")
        self.worker = repo / "character_factory/h3_avatar_workflow/tracker_worker.py"
        self.proc = subprocess.Popen([str(python), str(self.worker), self.shm.name], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)

    def command(self, command: str) -> str:
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        answer = self.proc.stdout.readline().strip()
        if not answer:
            raise RuntimeError(("tracker failed", self.proc.poll()))
        return answer

    def reset(self) -> None:
        assert self.command("reset") == "ready"

    def track(self, frame, face, box):
        import cv2
        self.frame[:] = frame
        x, y, x1, y1 = map(int, box)
        self.frame[y:y1, x:x1] = cv2.resize(face, (x1 - x, y1 - y))
        answer = self.command("frame").split()
        assert answer[0] == "ok"
        return self.points.copy(), float(answer[1])

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        self.log.close()
        self.shm.close()
        self.shm.unlink()


# ----------------------------------------------------------------------------- backends
def describe_unet(unet) -> dict:
    info = {"class": type(unet).__name__, "name": getattr(unet, "name", None)}
    if hasattr(unet, "describe"):
        try:
            info["describe"] = unet.describe()
        except Exception as exc:  # noqa: BLE001 - informational only
            info["describe_error"] = repr(exc)
    modes = []
    subs = getattr(unet, "backends_by_batch", None)
    for sub in (subs.values() if isinstance(subs, dict) else [unet]):
        if hasattr(sub, "cudagraphs_mode"):
            modes.append(sub.cudagraphs_mode)
    info["cudagraphs_modes"] = sorted(set(modes))
    return info


def describe_decoder(decoder) -> dict:
    return {"class": type(decoder).__name__, "name": getattr(decoder, "name", None),
            "compile_enabled": getattr(decoder, "compile_enabled", None),
            "compile_mode": getattr(decoder, "compile_mode", None),
            "fused_post_enabled": getattr(decoder, "fused_post_enabled", None)}


def backend_expectations(flags: dict, decoder_info: dict, unet_info: dict) -> list:
    """Requested-but-not-active backends (a silent fallback would mislabel the arm)."""
    problems = []
    env = os.environ
    if env.get("MUSETALK_TAESD_BACKEND", "").strip().lower() == "trt":
        if decoder_info["name"] != "taesd_trt":
            problems.append(f"MUSETALK_TAESD_BACKEND=trt but decoder is {decoder_info['name']}")
    else:
        if decoder_info["name"] != "taesd":
            problems.append(f"expected compiled TAESD, got {decoder_info['name']}")
        elif "MUSETALK_TAESD_COMPILE" not in flags and not decoder_info["compile_enabled"]:
            problems.append("compiled TAESD expected (backend.setup asserts compile_enabled)")
    if env.get("MUSETALK_UNET_BACKEND", "").strip().lower() in {"trt_stagewise", "tensorrt_stagewise"}:
        if unet_info["name"] != "tensorrt_unet_stagewise":
            problems.append(f"MUSETALK_UNET_BACKEND=trt_stagewise but UNet is {unet_info['name']}")
    graphs = env.get("MUSETALK_TRT_UNET_CUDAGRAPHS", "0").strip().lower()
    graphs = {"1": "manual", "on": "manual", "true": "manual", "yes": "manual"}.get(graphs, graphs)
    if graphs in {"manual", "runtime"} and unet_info["name"] != "tensorrt_unet_stagewise":
        if unet_info["cudagraphs_modes"] != [graphs]:
            problems.append(f"MUSETALK_TRT_UNET_CUDAGRAPHS={graphs} but UNet modes are {unet_info['cudagraphs_modes']}")
    return problems


def setup_backends(repo: Path):
    """backend.setup() with the tree taken from --repo (the env layer is already applied)."""
    import torch
    from scripts.trt_runtime import load_unet_trt_backend
    from scripts.vae_fast_decoder import load_taesd_decoder
    started = time.perf_counter()
    decoder = load_taesd_decoder(device=torch.device("cuda:0"), runtime_dtype=torch.float16, force=True)
    if decoder is None:
        raise RuntimeError("load_taesd_decoder returned None")
    gc.collect()
    torch.cuda.empty_cache()
    unet = load_unet_trt_backend(device=torch.device("cuda:0"), force=True)
    if unet is None:
        raise RuntimeError("load_unet_trt_backend returned None")
    return unet, decoder, 0.18215, time.perf_counter() - started


# ----------------------------------------------------------------------------- render
def load_identity(who: str, data_root: Path, n: int):
    import numpy as np
    import torch
    paths = identity_paths(who, data_root)
    frames = read_frames(paths["source"], n)
    d = dict(cache=torch.load(paths["cache"], map_location="cpu", weights_only=False),
             masks=np.load(paths["masks"]), p=np.load(paths["source_landmarks"])[:n],
             g=np.zeros((n, 478, 2), np.float32))
    return paths, frames, d


def input_hashes(paths: dict, repo: Path, chin_path: Path) -> dict:
    out = {k: sha_file(paths[k]) for k in ("source", "audio", "cache", "masks", "source_landmarks")}
    out["chin.py"] = sha_file(chin_path)
    out["tracker_worker.py"] = sha_file(repo / "character_factory/h3_avatar_workflow/tracker_worker.py")
    out["blending.py"] = sha_file(repo / "musetalk/utils/blending.py")
    out["render_script"] = sha_file(SCRIPT)
    return out


def render_gpu(chin, repo, who, data_root, tracker, unet, decoder, sf, repeats, n):
    """render_stage.main's timed loop, repeated; returns frames of repeat 0 + per-repeat records."""
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor

    paths, frames, d = load_identity(who, data_root, n)
    if frames[0].shape != (896, 512, 3):  # tracker_worker.py's shared-memory frame slot is fixed
        raise RuntimeError(f"{who}: source frames are {frames[0].shape}, the tracker needs (896, 512, 3)")
    t = time.perf_counter()
    chin.prepare_refined(d, frames)
    prep_s = time.perf_counter() - t
    stamp = torch.tensor([0], device="cuda")
    with torch.inference_mode():
        for _ in range(4):
            z = unet(d["cache"]["latents"][:BATCH].cuda(), stamp, encoder_hidden_states=d["cache"]["audio"][:BATCH].cuda()).sample
            decoder.decode(z, sf, torch.float16)
    torch.cuda.synchronize()

    @torch.inference_mode()
    def generate(i):
        z = unet(d["cache"]["latents"][i:i + BATCH].cuda(), stamp,
                 encoder_hidden_states=d["cache"]["audio"][i:i + BATCH].cuda()).sample
        pixels = decoder.decode(z, sf, torch.float16)
        if not torch.isfinite(pixels).all():
            raise RuntimeError("Non-finite generated pixels")
        return pixels.float().mul(255).round().clamp(0, 255).to(torch.uint8).flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    records, kept = [], None
    for repeat in range(repeats):
        d["g"][:] = 0
        d["chin_delta"][:] = 0
        tracker.reset()
        outputs, faces, queue = [], [], []
        previous = None
        timing = dict(tracking_ipc_ms=0.0, tracking_worker_ms=0.0, compose_ms=0.0)

        def emit(record, next_delta):
            nonlocal previous
            i, face, g, delta = record
            d["g"][i] = g
            old = delta if previous is None else previous
            d["chin_delta"][i] = np.clip(.25 * old + .5 * delta + .25 * next_delta, 0, .18)
            s = time.perf_counter()
            outputs.append(chin.corrected_refined(frames[i], d, i, face))
            timing["compose_ms"] += (time.perf_counter() - s) * 1000
            previous = delta

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as pool:
            torch.cuda.synchronize()
            start = time.perf_counter()
            future = pool.submit(generate, 0)
            for base in range(0, n, BATCH):
                pixels = future.result()
                if base + BATCH < n:
                    future = pool.submit(generate, base + BATCH)
                faces.extend(pixels)
                for j, face in enumerate(pixels):
                    i = base + j
                    s = time.perf_counter()
                    g, worker_s = tracker.track(frames[i], face, d["cache"]["boxes"][i])
                    timing["tracking_ipc_ms"] += (time.perf_counter() - s) * 1000
                    timing["tracking_worker_ms"] += worker_s * 1000
                    source, target = chin.curves(d["p"][i], g)
                    queue.append((i, face, g, source - target))
                    if len(queue) == 2:
                        emit(queue.pop(0), source - target)
            if queue:
                emit(queue[0], queue[0][3])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        if len(outputs) != n:
            raise RuntimeError(f"{who}: {len(outputs)} outputs, expected {n}")
        hashes = [sha_array(f) for f in outputs]
        rec = dict(repeat=repeat, frames=n, seconds=elapsed, fps=n / elapsed, stages_ms=timing,
                   frames_digest=digest(hashes), faces_digest=digest([sha_array(f) for f in faces]),
                   loadavg=os.getloadavg())
        records.append(rec)
        print("MEASURED", who, json.dumps({k: v for k, v in rec.items() if "digest" not in k}), flush=True)
        if repeat == 0:
            kept = dict(outputs=outputs, faces=np.asarray(faces), g=d["g"].copy(), delta=d["chin_delta"].copy())
        else:
            rec["landmarks_max_abs_vs_repeat0"] = float(np.max(np.abs(d["g"] - kept["g"])))
            del outputs, faces
            gc.collect()
    d["g"], d["chin_delta"] = kept["g"], kept["delta"]
    return paths, frames, d, kept, records, prep_s


def recipe_checks(chin, frames, d, faces, outputs) -> dict:
    """render_stage.py's offline diagnostics (excluded from timing): protected lips unchanged vs the
    standard composite, optimized ROI path == full-frame reference, Jacobian bound."""
    import cv2
    import numpy as np
    max_lip, min_jac, max_ref = 0, 1.0, 0
    assertion_failures = []
    for i, (frame, face) in enumerate(zip(frames, faces)):
        plain = chin.standard(frame, d, i, face)
        mask = d["refined_masks"][i].current(d["g"][i], d["chin_delta"][i])
        b = d["cache"]["boxes"][i]
        cb = d["cache"]["cropboxes"][i]
        x, y, x1, y1 = map(int, b)
        retained = chin.get_image_blending(frame.copy(), cv2.resize(face, (x1 - x, y1 - y)), b, mask, cb)
        try:
            reference, meta = chin.aligned(retained, d, i, 1.)
        except AssertionError as exc:  # chin.aligned asserts lip stationarity and the Jacobian bound
            assertion_failures.append({"frame": i, "error": str(exc)[:300]})
            continue
        max_ref = max(max_ref, int(np.abs(reference.astype(np.int16) - outputs[i].astype(np.int16)).max()))
        _, _, _, span = chin.axes(d["p"][i])
        lip = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(lip, cv2.convexHull(np.rint(d["g"][i][chin.LIPS]).astype(np.int32)), 255)
        radius = max(4, int(round(.06 * span)))
        lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
        max_lip = max(max_lip, int(np.abs(plain.astype(np.int16) - outputs[i].astype(np.int16))[lip].max()))
        min_jac = min(min_jac, meta["min_map_jacobian"])
    return dict(frames=len(frames), protected_lip_max_rgb_difference=max_lip, reference_max_rgb_difference=max_ref,
                minimum_jacobian=min_jac, aligned_assertion_failures=assertion_failures,
                passes=max_lip == 0 and max_ref == 0 and min_jac > .25 and not assertion_failures)


def render_dry(chin, who, data_root, n):
    """CPU: saved faces + saved generated landmarks through the same emit()/corrected_refined path."""
    import numpy as np
    paths, frames, d = load_identity(who, data_root, n)
    chin.prepare_refined(d, frames)
    faces = np.load(paths["faces"])["faces"][:n]
    saved_g = np.load(paths["saved_generated_landmarks"])[:n].astype(np.float32)
    outputs, queue, previous = [], [], None
    start = time.perf_counter()
    for i in range(n):
        g = saved_g[i]
        source, target = chin.curves(d["p"][i], g)
        queue.append((i, faces[i], g, source - target))
        if len(queue) == 2:
            j, face, gj, delta = queue.pop(0)
            d["g"][j] = gj
            old = delta if previous is None else previous
            d["chin_delta"][j] = np.clip(.25 * old + .5 * delta + .25 * (source - target), 0, .18)
            outputs.append(chin.corrected_refined(frames[j], d, j, face))
            previous = delta
    j, face, gj, delta = queue[0]
    d["g"][j] = gj
    old = delta if previous is None else previous
    d["chin_delta"][j] = np.clip(.25 * old + .5 * delta + .25 * delta, 0, .18)
    outputs.append(chin.corrected_refined(frames[j], d, j, face))
    elapsed = time.perf_counter() - start
    rec = dict(repeat=0, frames=n, seconds=elapsed, fps=n / elapsed, mode="dry_run_compose_only",
               frames_digest=digest([sha_array(f) for f in outputs]))
    return paths, frames, d, dict(outputs=outputs, faces=faces, g=d["g"].copy(), delta=d["chin_delta"].copy()), [rec], 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path, help="code tree to import (pre-change or candidate)")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--identities", default="japanese,latina")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--flags", default="", help="K=V,... applied after the env file and recipe overrides")
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--tree-meta", default="{}", help="JSON tree metadata from video_ab.py (recorded verbatim)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--tracker-python", type=Path, default=TRACKER_PYTHON)
    ap.add_argument("--allow-fallback", action="store_true", help="do not fail when a requested backend is not active")
    ap.add_argument("--allow-foreign-modules", action="store_true")
    ap.add_argument("--skip-checks", action="store_true", help="skip the offline recipe diagnostics")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: saved faces/landmarks, no CUDA, no tracker")
    args = ap.parse_args()

    repo = args.repo.resolve()
    flags = parse_flags(args.flags)
    if args.frames % BATCH and not args.dry_run:
        raise SystemExit(f"--frames must be a multiple of {BATCH}")
    os.chdir(repo)
    for p in (str(repo / "scripts"), str(repo)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    env_report = apply_env_layers(repo, flags)
    import cv2
    import numpy as np
    chin, chin_path = load_chin(repo)
    cv2.setNumThreads(2)
    tree_meta = json.loads(args.tree_meta)
    base = dict(schema=SCHEMA, kind="chin_render", fps=FPS, native_fps=FPS,
                recipe="native-encoder cache latents -> TRT FP16 UNet bs8 -> TAESD -> per-frame FaceMesh -> "
                       "100% chin alignment, refined seam (chin.corrected_refined), .25/.5/.25 lookahead filter",
                arm={**tree_meta, "name": args.arm_name, "tree": str(repo), "flags": flags},
                env=env_report, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                host=dict(node=platform.node(), python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__),
                argv=sys.argv)
    identities = [w for w in args.identities.split(",") if w]
    unet = decoder = tracker = None
    load_s = 0.0
    if args.dry_run:
        backends = {"decoder": {"name": "saved faces.npz (dry run, no inference)"}, "unet": {"name": "none (dry run)"}}
    else:
        import torch
        torch.set_num_threads(4)
        unet, decoder, sf, load_s = setup_backends(repo)
        backends = {"decoder": describe_decoder(decoder), "unet": describe_unet(unet), "torch": torch.__version__,
                    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                          capture_output=True, text=True).stdout.strip()}
        problems = backend_expectations(flags, backends["decoder"], backends["unet"])
        backends["expectation_problems"] = problems
        if problems and not args.allow_fallback:
            print("FAIL backend expectation:", problems, flush=True)
            return 3
        tracker = Tracker(repo, args.tracker_python, args.out_root / "tracker.log")
    origins = module_origin_check(repo)
    if not origins["ok"] and not args.allow_foreign_modules:
        print("FAIL foreign modules imported:", origins["foreign"][:5], flush=True)
        return 4
    status = 0
    try:
        for who in identities:
            dest = args.out_root / who
            dest.mkdir(parents=True, exist_ok=True)
            if args.dry_run:
                paths, frames, d, kept, records, prep_s = render_dry(chin, who, args.data_root, args.frames)
            else:
                paths, frames, d, kept, records, prep_s = render_gpu(chin, repo, who, args.data_root, tracker, unet,
                                                                     decoder, sf, args.repeats, args.frames)
            checks = {"skipped": True} if args.skip_checks else recipe_checks(chin, frames, d, kept["faces"], kept["outputs"])
            stored = store_clip(dest, kept["outputs"], FPS)
            np.save(dest / "generated_landmarks.npy", kept["g"])
            np.save(dest / "chin_delta.npy", kept["delta"])
            fps_values = [r["fps"] for r in records]
            identical = len({r["frames_digest"] for r in records}) == 1
            clip = dict(base, clip=f"chin_{who}", identity=who, frame_count=len(kept["outputs"]),
                        audio=str(paths["audio"]), mouth_box=mouth_box(chin, d["p"], stored["width"], stored["height"]),
                        backends=backends, load_seconds=load_s, preparation_seconds=prep_s,
                        measured_fps=float(np.median(fps_values)),
                        measured_fps_desc=(f"median of {len(records)} warm {args.frames}-frame runs: fresh UNet+TAESD, "
                                           "D2H, per-frame FaceMesh tracking, refined chin composite (excl. load/prep/encode)"
                                           if not args.dry_run else "dry run: compose-only CPU time (not a render fps)"),
                        fps_runs=records, repeat_outputs_identical=identical, recipe_checks=checks,
                        generated_landmarks="generated_landmarks.npy", chin_delta="chin_delta.npy",
                        inputs={k: str(paths[k]) for k in ("source", "audio", "cache", "masks", "source_landmarks")},
                        input_sha256=input_hashes(paths, repo, chin_path), module_origin_check=origins, **stored)
            dump(dest / "clip.json", clip)
            ok = identical and (args.skip_checks or checks.get("passes", False))
            status = status or (0 if ok else 5)
            print(f"{'PASS' if ok else 'FAIL'} chin_render arm={args.arm_name} identity={who} frames={clip['frame_count']} "
                  f"fps={clip['measured_fps']:.1f} repeats_identical={identical} checks={checks.get('passes', 'skipped')} "
                  f"digest={stored['frames_digest'][:12]} -> {dest / 'clip.json'}", flush=True)
            del paths, frames, d, kept
            gc.collect()
    finally:
        if tracker is not None:
            tracker.close()
    return status


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:  # argparse / explicit exits keep their code
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
        if not isinstance(exc.code, int) and exc.code is not None:
            print(exc.code, file=sys.stderr)
    except BaseException:  # noqa: BLE001 - print, then exit without TRT/compile teardown
        import traceback
        traceback.print_exc()
        print("FAIL chin_render: exception (traceback above)", flush=True)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # skip interpreter teardown of TRT/compiled graphs (as the replay harness does)
