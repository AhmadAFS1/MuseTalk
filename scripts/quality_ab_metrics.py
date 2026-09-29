"""Quality A/B metrics for MuseTalk chin100 renders (the quality half of the 300 fps goal).

Compares two renders of the SAME identity + audio: A = accepted reference, B = candidate.
Reports, for both arms and their delta, with documented PASS thresholds (see GATES and
docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/README.md):

  1. lip sync      FaceMesh inner-lip aperture (px and / source eye spacing) per arm; Pearson A vs B;
                   lag of max cross-correlation; mean |delta| px;
                   optional SyncNet (models/syncnet, GPU, run under box_guard, relative only).
  2. flicker       mean |f(t)-f(t-1)| and |f(t+1)-2f(t)+f(t-1)| in mouth ROI, chin/jaw band, seam ring,
                   face box, per arm, ratio B/A; temporal >6 Hz power in fixed mouth/jaw boxes.
  3. seam          per-frame max/mean |A-B| inside the blend-mask feather band, along the 50% mask
                   boundary ring, outside the mask; protected-lip pixel change per arm vs its own
                   standard compose (render_stage.py pixel_checks semantics, raw frames only).
  4. chin          validate_stage.py geometry (target chin error = (q152-g152).down, frames 24..215)
                   per arm; FaceMesh jaw+lip landmark deviation A vs B (mean, p99, max px).
  5. overall       PSNR/SSIM A vs B (full, face box, mouth ROI), mouth/face Laplacian-variance
                   sharpness ratio B/A, Lab colour shift in the face box.

Frames: an arm is analysed either from RAW pre-encode frames (reconstructed bit-exactly with the
unchanged character_factory/h3_avatar_workflow/chin.py from faces.npz + generated_landmarks.npy,
verified against render.json raw_refined_sha256, or loaded from a .npy/.npz frame dump) or from its
encoded .mp4. --frames auto uses raw when both arms support it.

CPU only (except `syncnet`). Run with /workspace/.venvs/musetalk_trt_stagewise/bin/python from the
worktree root. FaceMesh runs in /workspace/SoulX-FlashHead/.venv/bin/python via
scripts/quality_ab_facemesh.py (identical tracker configuration to the workflow).

Arm spec (--a / --b): comma-separated key=value
    dir=RENDER_DIR        render_stage layout: refined_raw.mp4, standard_raw.mp4, faces.npz,
                          generated_landmarks.npy, chin_delta.npy, render.json, pixel_checks.json
    compose=refined|standard|source   which compose the arm is (default refined). 'source' is the
                          earlier SourceMask chin correction (chin.corrected).
    video=PATH  faces=PATH  g=PATH  chin_delta=PATH  raw=PATH  render_json=PATH  label=TEXT
Identity (--identity-dir, or explicit --source/--source-landmarks/--cache/--masks/--audio).

Examples
    PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
    ID=/workspace/experiments/avatar_diversity_20260927/black_woman
    $PY scripts/quality_ab_metrics.py pair --identity-dir $ID --a dir=$ID --b dir=/path/to/candidate \
        --profile e1 --name black_woman_trt_taesd
    $PY scripts/quality_ab_metrics.py calibrate            # noise floors + sensitivity (writes summary)
    scripts/box_guard.sh run --min-avail-gb 5 --wait-min 90 --label qm-syncnet -- \
        $PY scripts/quality_ab_metrics.py syncnet-calibrate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H3 = ROOT / "character_factory/h3_avatar_workflow"
sys.path[:0] = [str(ROOT), str(H3)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(2)

import chin  # noqa: E402  (unchanged accepted chin algorithm; imports musetalk.utils.blending from ROOT)

FACEMESH_PY = Path("/workspace/SoulX-FlashHead/.venv/bin/python")
FACEMESH_HELPER = ROOT / "scripts/quality_ab_facemesh.py"
OUT_ROOT = ROOT / "docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics"
DIV = Path("/workspace/experiments/avatar_diversity_20260927")
DIV_IDS = ["black_man_short_beard", "black_woman", "east_asian_man_goatee",
           "middle_eastern_man_full_beard", "south_asian_woman", "white_man_clean_shaven"]
CFV = Path("/workspace/experiments/chin_fps_validation_20260927")
PJV = Path("/workspace/experiments/portrait_jaw_video_20260926")
ACCEPTED_BLENDING_SHA = "6e5de1bf3f30378493c098b76dfc6fe6644e354d60df2492ffb1fc27726ba121"
ACCEPTED_CHIN_SHA = "fd753e7d86b14f1800719ec7f26663c1b362cb512ee24b4b8eb9b1e07f2b99e1"

TOOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HELPER_SHA256 = hashlib.sha256(FACEMESH_HELPER.read_bytes()).hexdigest()
FPS = 24
CHIN_WINDOW = (24, 216)          # validate_stage.py window
LAG_RANGE = 6                    # frames searched for the cross-correlation lag
HF_CUTOFF_HZ = 6.0               # temporal "high frequency" = above 6 Hz (Nyquist 12 Hz at 24 fps)
PSNR_CAP_DB = 100.0              # identical frames are capped here in per-frame means
INNER_PAIRS = [(13, 14), (82, 87), (312, 317), (81, 178), (311, 402)]
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
JAW_LOWER = chin.JAW[3:18]       # 58 ... 152 ... 288 (lower jaw / chin contour)
LANDMARK_SET = sorted(set(chin.JAW) | set(chin.LIPS) | set(INNER_LIPS))

# ------------------------------------------------------------------------------------ gates
# Thresholds (documented in README.md). 'e1' = FP16 runtime change (plan class E1, G-TRACK),
# 'lossy' = plan class L (L3-L6), 'exact' = plan class E0 (SHA identity of pre-encode frames).
GATES = {
    "aperture_corr_min": 0.97,          # L3 / task
    "aperture_lag_frames": 0,           # task: lag of max cross-correlation must be 0
    "aperture_mean_abs_delta_px_max": 0.5,   # L3
    "flicker_ratio_max": 1.05,          # L6 (mouth, jaw band and seam ring, first difference)
    "chin_error_margin_px": {"e1": 0.05, "lossy": 0.20, "exact": 0.0},   # G-TRACK / L4
    "landmark_dev_mean_px_max": 0.05,   # G-TRACK (e1 only; reported for lossy)
    "landmark_dev_p99_px_max": 0.15,    # G-TRACK (e1 only; reported for lossy)
    "protected_lip_max_rgb": 0,         # pixel_checks semantics, per arm vs its own standard compose
    # Proposed (report-only, pending a user decision): calibrated G-TRACK. 0.05/0.15 sits at FaceMesh's own
    # sensitivity floor (0.2-LSB sparse face noise: 0.034-0.083 mean, 0.10-0.43 p99 px); 0.10/0.35 passes every
    # 1-LSB synthetic floor and still fails codec noise (>= 0.24 mean) and every known visible change (>= 0.47).
    "landmark_dev_mean_px_proposed": 0.10,
    "landmark_dev_p99_px_proposed": 0.35,
    "sharpness_ratio_min": 0.95,        # L5 lower bound (upper 1.05 reported as a warning)
    "sharpness_ratio_warn_max": 1.05,
}


def ram_available_gb():
    """min(/proc/meminfo MemAvailable, cgroup estimate), same definition as scripts/box_guard.sh."""
    vals = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                vals.append(int(line.split()[1]) / 1048576)
    except OSError:
        pass
    try:
        mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if mx != "max":
            cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
            st = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
            rec = sum(int(st.get(k, 0)) for k in ("active_file", "inactive_file", "slab_reclaimable"))
            vals.append((int(mx) - cur + rec) / 1073741824)
    except (OSError, ValueError):
        pass
    return min(vals) if vals else float("nan")


def ram_guard(where, need_gb=float(os.environ.get("QAB_MIN_AVAIL_GB", "4.0")), wait_s=float(os.environ.get("QAB_RAM_WAIT_S", "900"))):
    """Shared box: never push MemAvailable toward 3 GB. Wait (up to wait_s) for headroom, else abort."""
    deadline = time.time() + wait_s
    while True:
        avail = ram_available_gb()
        if not (avail == avail and avail < need_gb):
            return
        if time.time() > deadline:
            raise SystemExit(f"[quality_ab] MemAvailable {avail:.2f} GB < {need_gb} GB at {where}; aborting to protect the box")
        print(f"[quality_ab] waiting for RAM: {avail:.2f} GB < {need_gb} GB at {where}", file=sys.stderr, flush=True)
        time.sleep(10)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_frames(frames):
    h = hashlib.sha256()
    for f in frames:
        h.update(np.ascontiguousarray(f).tobytes())
    return h.hexdigest()


def clean(x):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, Path):
        return str(x)
    return x


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=None, median=None, p95=None, p99=None, max=None, n=0)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()), n=int(a.size))


def ratio(b, a):
    if a is None or b is None:
        return None
    if a == 0:
        return 1.0 if b == 0 else float("inf")
    return float(b / a)


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def encode_video(path, frames, audio=None, crf=18, preset="fast", threads=2, fps=FPS):
    """Encode BGR frames; defaults are exactly character_factory backend.encode()."""
    h, w = frames[0].shape[:2]
    args = ["ffmpeg", "-v", "error", "-xerror", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "pipe:0"]
    if audio:
        args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
    args += ["-c:v", "libx264", "-threads", str(threads), "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-t", str(len(frames) / fps)]
    args += ["-movflags", "+faststart", str(path)]
    p = subprocess.Popen(args, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, args


# --------------------------------------------------------------------------------- facemesh
def facemesh(frames=None, video=None, tag="arm", count=None, shape=None):
    """Track (T,478,2) float32 landmarks with the workflow's FaceMesh config (NaN where no face).

    frames: a sequence (or an iterator, with count and (H, W) shape) of BGR uint8 frames streamed
    to the helper over stdin; or video: an mp4 decoded by the helper itself.
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="qab_fm_") as tmp:
        out = Path(tmp) / f"{tag}.npy"
        cmd = [str(FACEMESH_PY), str(FACEMESH_HELPER), "--out", str(out)]
        if video is not None and frames is None:
            cmd += ["--video", str(video)]
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stdout = res.stdout
        else:
            if count is None:
                count, shape = len(frames), frames[0].shape[:2]
            h, w = shape
            cmd += ["--stdin-frames", str(count), str(h), str(w)]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            err_buf = []
            t = threading.Thread(target=lambda: err_buf.append(p.stderr.read()), daemon=True)
            t.start()
            n = 0
            for f in frames:
                p.stdin.write(np.ascontiguousarray(f).tobytes())
                n += 1
            p.stdin.close()
            assert n == count, (n, count)
            stdout = p.stdout.read().decode()
            if p.wait() != 0:
                t.join(5)
                raise RuntimeError(f"facemesh helper failed: {b''.join(err_buf).decode()[-2000:]}")
        info = json.loads(stdout.strip().splitlines()[-1])
        return np.load(out), info


def track_generated(ident, faces, tag="generated"):
    """Re-derive the render's generated landmarks g from generated faces.

    Same as render_stage.py's Tracker: source frame with cv2.resize(face) pasted at the face box,
    one fresh tracking-mode FaceMesh per sequence, frames in order.
    """
    boxes = ident.d["cache"]["boxes"]

    def pasted():
        for i, face in enumerate(faces):
            f = ident.frames[i].copy()
            x, y, x1, y1 = map(int, boxes[i])
            f[y:y1, x:x1] = cv2.resize(face, (x1 - x, y1 - y))
            yield f
    return facemesh(pasted(), tag=tag, count=len(faces), shape=ident.frames[0].shape[:2])


class _Lazy:
    """Per-frame chin objects built on demand with chin.py's own constructors (small LRU).

    chin.prepare_refined() keeps SourceMask + RefinedMask + blend plan for all frames (about 5 MB
    per frame, 1.2 GB per identity); building them per frame from identical inputs is bit-identical
    and keeps this tool's RSS low on the shared box.
    """

    def __init__(self, n, build, keep=4):
        self.n, self.build, self.keep, self.cache = n, build, keep, {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = self.build(i)
        return self.cache[i]


# --------------------------------------------------------------------------------- identity
@dataclass
class Identity:
    name: str
    source: Path
    landmarks: Path
    cache: Path
    masks: Path
    audio: Path | None = None
    _frames: list | None = None
    _d: dict | None = None

    @classmethod
    def from_dir(cls, path, name=None):
        path = Path(path)
        return cls(name=name or path.name, source=path / "source.mp4", landmarks=path / "source_landmarks.npy",
                   cache=path / "cache.pt", masks=path / "masks.npz",
                   audio=(path / "speech.wav") if (path / "speech.wav").exists() else None)

    @property
    def frames(self):
        if self._frames is None:
            self._frames = read_video(self.source)
        return self._frames

    @property
    def d(self):
        """Source-only chin state (masks, plans, SourceMask/RefinedMask objects), shared by arms."""
        if self._d is None:
            import torch
            torch.set_num_threads(2)
            cache = torch.load(self.cache, map_location="cpu", weights_only=False)
            cache = dict(boxes=np.asarray(cache["boxes"]), cropboxes=list(cache["cropboxes"]))
            d = dict(cache=cache, masks=np.load(self.masks), p=np.load(self.landmarks))
            T = len(self.frames)
            assert len(d["p"]) == T == len(cache["boxes"]), (len(d["p"]), T, len(cache["boxes"]))
            d["g"] = np.zeros_like(d["p"])
            # Equivalent to chin.prepare_refined(d, frames) (same constructors and arguments), built lazily.
            from musetalk.utils.blending import prepare_image_blending_plan
            shape = self.frames[0].shape
            m, cb, p = d["masks"], cache["cropboxes"], d["p"]
            d["source_masks"] = _Lazy(T, lambda i: chin.SourceMask(m[str(i)], cb[i], p[i]))
            d["plans"] = _Lazy(T, lambda i: prepare_image_blending_plan(shape, cache["boxes"][i], m[str(i)], cb[i]))
            d["refined_masks"] = _Lazy(T, lambda i: chin.RefinedMask(m[str(i)], cb[i], p[i]))
            d["chin_delta"] = np.zeros((T, len(chin.GRID)))
            self._d = d
        return self._d

    @property
    def p(self):
        return self.d["p"]

    def signature(self):
        return dict(name=self.name, source=str(self.source), source_sha256=sha_file(self.source),
                    source_landmarks_sha256=sha_file(self.landmarks), cache_sha256=sha_file(self.cache),
                    masks_sha256=sha_file(self.masks), audio=str(self.audio) if self.audio else None)


# ------------------------------------------------------------------------------------- arms
@dataclass
class Arm:
    label: str
    compose: str = "refined"
    video: Path | None = None
    faces: Path | None = None
    g: Path | None = None
    chin_delta: Path | None = None
    raw: Path | None = None
    render_json: Path | None = None
    pixel_checks: Path | None = None
    mask_samples: Path | None = None
    retrack: bool = False          # re-derive g from faces with the render's tracker procedure
    perturb_lsb: int = 0           # calibration only: add iid uniform integer noise in [-k, k] to faces
    perturb_seed: int = 0
    perturb_frac: float = 1.0      # calibration only: fraction of face samples perturbed (sparse +-k)
    info: dict = field(default_factory=dict)
    # filled by load()
    frames: list | None = None
    g_arr: np.ndarray | None = None
    delta_arr: np.ndarray | None = None
    q: np.ndarray | None = None

    @classmethod
    def parse(cls, spec, default_label):
        kv = {}
        for part in spec.split(","):
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"bad arm spec part {part!r} (want key=value)")
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
        compose = kv.pop("compose", "refined")
        arm = cls(label=kv.pop("label", default_label), compose=compose)
        if "dir" in kv:
            d = Path(kv.pop("dir"))
            name = "standard_raw.mp4" if compose == "standard" else "refined_raw.mp4"
            arm.video = d / name if (d / name).exists() else None
            for attr, fname in [("faces", "faces.npz"), ("g", "generated_landmarks.npy"),
                                ("render_json", "render.json"), ("pixel_checks", "pixel_checks.json"),
                                ("mask_samples", "mask_samples.npz")]:
                if (d / fname).exists():
                    setattr(arm, attr, d / fname)
            if compose == "refined" and (d / "chin_delta.npy").exists():
                arm.chin_delta = d / "chin_delta.npy"
            if compose != "refined":
                arm.render_json = arm.pixel_checks = arm.mask_samples = None
        for k, v in kv.items():
            if k == "retrack":
                arm.retrack = v not in ("0", "false", "no")
            elif k == "perturb":
                arm.perturb_lsb = int(v)
            elif k == "seed":
                arm.perturb_seed = int(v)
            elif k == "frac":
                arm.perturb_frac = float(v)
            elif k in ("video", "faces", "g", "chin_delta", "raw", "render_json", "pixel_checks", "mask_samples"):
                setattr(arm, k, Path(v))
            else:
                raise SystemExit(f"unknown arm key {k!r}")
        if compose not in ("refined", "standard", "source"):
            raise SystemExit(f"compose must be refined|standard|source, got {compose}")
        return arm

    def can_raw(self):
        return self.raw is not None or self.faces is not None

    def describe(self):
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in
                dict(label=self.label, compose=self.compose, video=self.video, faces=self.faces, g=self.g,
                     chin_delta=self.chin_delta, raw=self.raw, render_json=self.render_json, retrack=self.retrack,
                     perturb_lsb=self.perturb_lsb, perturb_seed=self.perturb_seed, perturb_frac=self.perturb_frac).items()}


def arm_chin_state(ident, arm):
    """Per-arm view of the identity's chin state with this arm's generated landmarks and delta."""
    d = dict(ident.d)
    d["g"] = arm.g_arr
    chin.prepare(d)          # the render's streaming .25/.5/.25 filter == chin.prepare's padded filter
    if arm.delta_arr is not None:
        arm.info["chin_delta_recomputed_equals_saved"] = bool(np.array_equal(d["chin_delta"], arm.delta_arr))
        d["chin_delta"] = arm.delta_arr
    return d


def load_arm(ident, arm, mode):
    """Load frames (raw reconstruction or decoded video) and per-arm generated landmarks."""
    ram_guard(f"load {arm.label}")
    T = len(ident.frames)
    if arm.g is not None:
        arm.g_arr = np.load(arm.g).astype(np.float32)
        assert arm.g_arr.shape == (T, 478, 2), arm.g_arr.shape
    if arm.chin_delta is not None:
        arm.delta_arr = np.load(arm.chin_delta)
    arm.info.update(frames_source=mode)
    lip_rows = None
    if mode == "raw" and arm.raw is not None:
        z = np.load(arm.raw)
        frames = z[z.files[0]] if hasattr(z, "files") else z
        arm.frames = [np.ascontiguousarray(f) for f in frames]
    elif mode == "raw":
        z = np.load(arm.faces)
        faces = z["faces"] if hasattr(z, "files") else z
        assert len(faces) == T, (len(faces), T)
        if arm.perturb_lsb:
            rng = np.random.default_rng(arm.perturb_seed)
            k = arm.perturb_lsb
            if arm.perturb_frac >= 1:
                noise = rng.integers(-k, k + 1, size=faces.shape, dtype=np.int16)
                what = f"iid uniform integer noise in [-{k}, {k}]"
            else:
                noise = (rng.random(faces.shape) < arm.perturb_frac) * rng.choice(np.array([-k, k], np.int16), size=faces.shape)
                what = f"sparse +-{k} noise on {arm.perturb_frac:.0%} of samples"
            faces = np.clip(faces.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
            arm.info["perturbation"] = (f"faces + {what} (seed {arm.perturb_seed}); mean |noise| "
                                        f"{float(np.abs(noise).mean()):.3f} LSB; g re-tracked; chin_delta recomputed")
            del noise
        if arm.g_arr is None or arm.retrack or arm.perturb_lsb:
            g2, fm = track_generated(ident, faces, tag=f"{arm.label}_generated")
            arm.info["generated_tracking"] = fm
            if arm.g_arr is not None and not arm.perturb_lsb:
                arm.info["g_retracked_equals_saved"] = bool(np.array_equal(g2, arm.g_arr))
                arm.info["g_retracked_max_abs_vs_saved"] = float(np.nanmax(np.abs(g2 - arm.g_arr)))
            arm.g_arr = g2
            arm.info["g_source"] = "re-tracked from faces (render_stage Tracker procedure)"
            if arm.delta_arr is not None:
                arm.info["saved_chin_delta_ignored"] = True
                arm.delta_arr = None
        else:
            arm.info["g_source"] = str(arm.g)
        d = arm_chin_state(ident, arm)
        out, lip_rows = [], []
        for i, (frame, face) in enumerate(zip(ident.frames, faces)):
            if arm.compose == "standard":
                o = chin.standard(frame, d, i, face)
            elif arm.compose == "refined":
                o = chin.corrected_refined(frame, d, i, face)
            else:
                o = chin.corrected(frame, d, i, face)
            out.append(o)
            if arm.compose != "standard":
                # render_stage.py pixel_checks semantics: generated lip hull + max(4, .06 span) dilation.
                plain = chin.standard(frame, d, i, face)
                _, _, _, span = chin.axes(d["p"][i])
                lip = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(lip, cv2.convexHull(np.rint(d["g"][i][chin.LIPS]).astype(np.int32)), 255)
                radius = max(4, int(round(.06 * span)))
                lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
                lip_rows.append(int(np.abs(plain.astype(np.int16) - o.astype(np.int16))[lip].max()))
            else:
                lip_rows.append(0)
        arm.frames = out
        arm.info["raw_reconstructed_with"] = f"chin.py {sha_file(H3 / 'chin.py')[:12]} compose={arm.compose}"
    else:
        if arm.video is None:
            raise SystemExit(f"arm {arm.label}: no video and raw mode not possible")
        arm.frames = read_video(arm.video)
        arm.info["video_sha256"] = sha_file(arm.video)
    assert len(arm.frames) == T, (arm.label, len(arm.frames), T)
    arm.info["frames_sha256"] = sha_frames(arm.frames)
    if mode == "raw" and arm.render_json is not None and arm.compose == "refined":
        rj = json.loads(Path(arm.render_json).read_text())
        if "raw_refined_sha256" in rj:
            arm.info["render_json_raw_refined_sha256"] = rj["raw_refined_sha256"]
            arm.info["raw_matches_render_json"] = rj["raw_refined_sha256"] == arm.info["frames_sha256"]
    if lip_rows is not None:
        arm.info["protected_lip"] = dict(max_rgb_difference=int(max(lip_rows)), frames=len(lip_rows),
                                         frames_nonzero=int(sum(r > 0 for r in lip_rows)))
        if arm.pixel_checks is not None and arm.compose == "refined":
            pc = json.loads(Path(arm.pixel_checks).read_text())
            ref_rows = [r["protected_lip_difference"] for r in pc["rows"]]
            arm.info["protected_lip"]["matches_pixel_checks_rows"] = ref_rows == lip_rows
    return arm


def alpha_full(ident, d, arm, i):
    """Full-frame effective blend alpha (uint8) for this arm, as render_stage's mask_samples."""
    frame_shape = ident.frames[i].shape[:2]
    b = ident.d["cache"]["boxes"][i]
    cb = ident.d["cache"]["cropboxes"][i]
    if arm.compose == "refined" and d is not None:
        mask = ident.d["refined_masks"][i].current(d["g"][i], d["chin_delta"][i])
    elif arm.compose == "source" and d is not None:
        mask = ident.d["source_masks"][i].current(d["g"][i])
    else:
        mask = ident.d["masks"][str(i)]
    x, y, x1, y1 = map(int, b)
    cx, cy = int(cb[0]), int(cb[1])
    H, W = frame_shape
    xa, ya, xb, yb = max(x, 0, cx), max(y, 0, cy), min(x1, W, cx + mask.shape[1]), min(y1, H, cy + mask.shape[0])
    alpha = np.zeros(frame_shape, np.uint8)
    if xb > xa and yb > ya:
        alpha[ya:yb, xa:xb] = mask[ya - cy:yb - cy, xa - cx:xb - cx]
    if arm.compose in ("refined", "source") and d is not None:
        alpha = chin.warp_roi(np.repeat(alpha[:, :, None], 3, 2), d, i, 1.)[:, :, 0]
    return alpha


# ------------------------------------------------------------------------------ primitives
def aperture_px(q):
    return np.mean([np.linalg.norm(q[:, a] - q[:, b], axis=-1) for a, b in INNER_PAIRS], axis=0)


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    if np.array_equal(a, b):
        return 1.0
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def lagged_corr(a, b, max_lag=LAG_RANGE):
    """corr(a[t], b[t+k]) for k in [-max_lag, max_lag]; positive k = b lags a."""
    out = {}
    n = len(a)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            out[k] = pearson(a[:n - k], b[k:])
        else:
            out[k] = pearson(a[-k:], b[:n + k])
    return out


def best_lag(curve):
    finite = {k: v for k, v in curve.items() if v == v}
    if not finite:
        return None, float("nan")
    best = max(finite.values())
    # ties (e.g. identical series) resolve to the smallest |k|
    k = min((k for k, v in finite.items() if v == best), key=abs)
    return int(k), float(best)


def ssim_map(x, y):
    """Gaussian-window SSIM map (11x11, sigma 1.5) on float64 luma, Wang et al. 2004 constants."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def psnr(mse):
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def rect_from_points(pts, pad_x, pad_y, shape):
    pts = pts[np.isfinite(pts).all(1)]
    H, W = shape
    x0 = int(max(0, math.floor(pts[:, 0].min() - pad_x)))
    x1 = int(min(W, math.ceil(pts[:, 0].max() + pad_x) + 1))
    y0 = int(max(0, math.floor(pts[:, 1].min() - pad_y)))
    y1 = int(min(H, math.ceil(pts[:, 1].max() + pad_y) + 1))
    return x0, y0, x1, y1


def valid(q):
    return q is not None and np.isfinite(q).all()


# ---------------------------------------------------------------------------------- compare
def compare(ident, A, B, mode="auto", profile="e1", name=None, out_dir=None, syncnet=False, notes=None):
    t0 = time.perf_counter()
    if mode == "auto":
        mode = "raw" if (A.can_raw() and B.can_raw()) else "video"
    if mode == "raw" and not (A.can_raw() and B.can_raw()):
        raise SystemExit("raw mode needs faces+g (or raw=) for both arms")
    for arm in (A, B):
        if arm.frames is None:
            load_arm(ident, arm, mode)
    T = len(ident.frames)
    shape = ident.frames[0].shape[:2]
    p = ident.p
    spans = np.array([chin.axes(pi)[3] for pi in p])
    downs = np.array([chin.axes(pi)[2] for pi in p])
    identical = A.info["frames_sha256"] == B.info["frames_sha256"]

    # ---- FaceMesh on the analysed frames of each arm
    for arm in (A, B):
        if arm.q is None:
            if mode == "video" and arm.video is not None:
                arm.q, fm = facemesh(video=arm.video, tag=arm.label)
            else:
                arm.q, fm = facemesh(frames=arm.frames, tag=arm.label)
            arm.info["facemesh"] = fm
    qA, qB = A.q, B.q
    missing = int((~np.isfinite(qA).all((1, 2))).sum() + (~np.isfinite(qB).all((1, 2))).sum())

    # ---- chin states for alphas
    dA = arm_chin_state(ident, A) if A.g_arr is not None and A.compose != "standard" else None
    dB = arm_chin_state(ident, B) if B.g_arr is not None and B.compose != "standard" else None

    # ---- mask-sample cross-check (render_stage mask_samples.npz semantics)
    for arm, d in ((A, dA), (B, dB)):
        if arm.mask_samples is not None and d is not None and arm.compose == "refined":
            ms = np.load(arm.mask_samples)
            arm.info["mask_samples_match"] = all(np.array_equal(alpha_full(ident, d, arm, int(k)), ms[k]) for k in ms.files)

    # ---- per-frame loop
    regions = ("mouth", "jaw", "ring", "face")
    fl = {arm: {r: [] for r in regions} for arm in "AB"}
    fl2 = {arm: {r: [] for r in regions} for arm in "AB"}
    sharp = {arm: {"mouth": [], "face": []} for arm in "AB"}
    lab_mean = {arm: [] for arm in "AB"}
    rows = []
    mouth_rects, jaw_union_mask = [], np.zeros(shape, bool)
    prev = None
    ellipse = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring_kernel = ellipse(3)
    for i in range(T):
        fa, fb = A.frames[i], B.frames[i]
        span = spans[i]
        lip_pts = [p[i][chin.LIPS]] + [q[i][chin.LIPS] for q in (qA, qB) if np.isfinite(q[i]).all()]
        mr = rect_from_points(np.concatenate(lip_pts), .12 * span, .10 * span, shape)
        mouth_rects.append(mr)
        jaw = np.zeros(shape, np.uint8)
        thick = 2 * max(3, int(round(.08 * span))) + 1
        for q in [p[i]] + [q[i] for q in (qA, qB) if np.isfinite(q[i]).all()]:
            cv2.polylines(jaw, [np.rint(q[JAW_LOWER]).astype(np.int32)], False, 255, thick)
        jaw_b = jaw > 0
        jaw_union_mask |= jaw_b
        aA = alpha_full(ident, dA, A, i)
        aB = aA if (B.compose == A.compose and dB is None and dA is None) else alpha_full(ident, dB, B, i)
        band = ((aA > 0) & (aA < 255)) | ((aB > 0) & (aB < 255))
        ring = np.zeros(shape, np.uint8)
        for a in (aA, aB):
            m = (a >= 128).astype(np.uint8)
            ring |= cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ring_b = cv2.dilate(ring, ring_kernel) > 0
        outside = (aA == 0) & (aB == 0)
        x, y, x1, y1 = [int(v) for v in ident.d["cache"]["boxes"][i]]
        x, y, x1, y1 = max(0, x), max(0, y), min(shape[1], x1), min(shape[0], y1)
        mx0, my0, mx1, my1 = mr
        diff = cv2.absdiff(fa, fb)
        d32 = diff.astype(np.float32)
        sq = d32 * d32
        row = dict(frame=i)
        row["full_mse"] = float(sq.mean())
        row["face_mse"] = float(sq[y:y1, x:x1].mean())
        row["mouth_mse"] = float(sq[my0:my1, mx0:mx1].mean())
        row["mouth_mae"] = float(d32[my0:my1, mx0:mx1].mean())
        row["face_max"] = int(diff[y:y1, x:x1].max())
        row["full_max"] = int(diff.max())
        for rname, rm in (("band", band), ("ring", ring_b), ("outside", outside), ("jaw", jaw_b)):
            v = diff[rm]
            row[f"{rname}_max"] = int(v.max()) if v.size else 0
            row[f"{rname}_mean"] = float(v.mean()) if v.size else 0.0
        if identical:
            smap = None
            row.update(ssim_full=1.0, ssim_face=1.0, ssim_mouth=1.0)
        else:
            ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY).astype(np.float64)
            smap = ssim_map(ga, gb)
            row.update(ssim_full=float(smap.mean()), ssim_face=float(smap[y:y1, x:x1].mean()),
                       ssim_mouth=float(smap[my0:my1, mx0:mx1].mean()))
        # colour (Lab of face box)
        la = cv2.cvtColor(fa[y:y1, x:x1].astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        lb = cv2.cvtColor(fb[y:y1, x:x1].astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        lab_mean["A"].append(la.reshape(-1, 3).mean(0))
        lab_mean["B"].append(lb.reshape(-1, 3).mean(0))
        row["face_deltaE76_mean"] = float(np.sqrt(((la - lb) ** 2).sum(-1)).mean())
        # sharpness + flicker per arm
        for key, f in (("A", fa), ("B", fb)):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sharp[key]["mouth"].append(float(cv2.Laplacian(g[my0:my1, mx0:mx1], cv2.CV_64F).var()))
            sharp[key]["face"].append(float(cv2.Laplacian(g[y:y1, x:x1], cv2.CV_64F).var()))
        if prev is not None:
            for key, f, pf in (("A", fa, prev[0]), ("B", fb, prev[1])):
                td = cv2.absdiff(f, pf)
                fl[key]["mouth"].append(float(td[my0:my1, mx0:mx1].mean()))
                fl[key]["jaw"].append(float(td[jaw_b].mean()))
                fl[key]["ring"].append(float(td[ring_b].mean()) if ring_b.any() else 0.0)
                fl[key]["face"].append(float(td[y:y1, x:x1].mean()))
        if 1 <= i < T - 1:
            for key, arm in (("A", A), ("B", B)):
                s2 = np.abs(arm.frames[i + 1].astype(np.int16) - 2 * arm.frames[i].astype(np.int16)
                            + arm.frames[i - 1].astype(np.int16))
                fl2[key]["mouth"].append(float(s2[my0:my1, mx0:mx1].mean()))
                fl2[key]["jaw"].append(float(s2[jaw_b].mean()))
                fl2[key]["ring"].append(float(s2[ring_b].mean()) if ring_b.any() else 0.0)
                fl2[key]["face"].append(float(s2[y:y1, x:x1].mean()))
        prev = (fa, fb)
        rows.append(row)

    # ---- temporal >6 Hz power in fixed union boxes
    def hf_power(frames, box):
        x0, y0, x1, y1 = box
        stack = np.stack([cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
        stack = stack.reshape(len(frames), -1)
        spec = np.abs(np.fft.rfft(stack - stack.mean(0), axis=0)) ** 2 / len(frames)
        freqs = np.fft.rfftfreq(len(frames), 1 / FPS)
        hf = spec[freqs > HF_CUTOFF_HZ].sum(0)
        total = spec[1:].sum(0)
        return float(hf.mean()), float(hf.sum() / max(total.sum(), 1e-12))
    mouth_union = (min(r[0] for r in mouth_rects), min(r[1] for r in mouth_rects),
                   max(r[2] for r in mouth_rects), max(r[3] for r in mouth_rects))
    jys, jxs = np.where(jaw_union_mask)
    jaw_union = (int(jxs.min()), int(jys.min()), int(jxs.max()) + 1, int(jys.max()) + 1)
    hf = {}
    for rname, box in (("mouth_box", mouth_union), ("jaw_box", jaw_union)):
        pa, fa_ = hf_power(A.frames, box)
        pb, fb_ = hf_power(B.frames, box)
        hf[rname] = dict(box_xyxy=list(box), A_power=pa, B_power=pb, ratio=ratio(pb, pa), A_fraction=fa_, B_fraction=fb_)

    # ---- lip sync
    apA, apB = aperture_px(qA), aperture_px(qB)
    nA, nB = apA / spans, apB / spans
    curve = lagged_corr(nA, nB)
    lag, lag_corr = best_lag(curve)
    lip = dict(
        definition="mean of inner-lip vertical distances (13-14, 82-87, 312-317, 81-178, 311-402); normalized by source eye spacing (chin.axes span)",
        A=dict(aperture_px=stats(apA), aperture_eye_spans=stats(nA)),
        B=dict(aperture_px=stats(apB), aperture_eye_spans=stats(nB)),
        pearson_A_vs_B=pearson(nA, nB), pearson_A_vs_B_px=pearson(apA, apB),
        xcorr_lag_frames=lag, xcorr_lag_corr=lag_corr, xcorr_curve={str(k): v for k, v in curve.items()},
        mean_abs_delta_px=float(np.nanmean(np.abs(apA - apB))),
        p95_abs_delta_px=float(np.nanpercentile(np.abs(apA - apB), 95)),
        mean_abs_delta_eye_spans=float(np.nanmean(np.abs(nA - nB))),
        series=dict(A_px=apA, B_px=apB))
    # ---- flicker
    flick = {}
    for r in regions:
        a1, b1 = float(np.mean(fl["A"][r])), float(np.mean(fl["B"][r]))
        a2, b2 = float(np.mean(fl2["A"][r])), float(np.mean(fl2["B"][r]))
        flick[r] = dict(A_first_diff=a1, B_first_diff=b1, ratio_first_diff=ratio(b1, a1),
                        A_second_diff=a2, B_second_diff=b2, ratio_second_diff=ratio(b2, a2))
    flick["temporal_hf_power_gt_6hz"] = hf

    # ---- seam / masking
    def agg(key):
        v = np.array([r[key] for r in rows], float)
        return dict(mean=float(v.mean()), p99=float(np.percentile(v, 99)), max=float(v.max()))
    seam = dict(
        regions="band: 0<alpha<255 of either arm's effective full-frame blend alpha (render_stage mask_samples semantics); "
                "ring: 50% alpha contour of either arm dilated by 3 px; outside: alpha==0 in both arms (includes chin-warp neck pixels).",
        band_max_abs=agg("band_max"), band_mean_abs=agg("band_mean"),
        ring_max_abs=agg("ring_max"), ring_mean_abs=agg("ring_mean"),
        outside_max_abs=agg("outside_max"), outside_mean_abs=agg("outside_mean"),
        jaw_band_max_abs=agg("jaw_max"), jaw_band_mean_abs=agg("jaw_mean"),
        ring_flicker_ratio=flick["ring"]["ratio_first_diff"],
        series=dict(band_max=[r["band_max"] for r in rows], band_mean=[r["band_mean"] for r in rows],
                    ring_max=[r["ring_max"] for r in rows], ring_mean=[r["ring_mean"] for r in rows]),
        protected_lip=dict(A=A.info.get("protected_lip"), B=B.info.get("protected_lip")))
    if mode == "video":
        seam["protected_lip"]["note"] = "not computable on encoded frames (needs faces + generated landmarks, raw mode)"

    # ---- chin (validate_stage.py formula) + landmark deviation
    lo, hi = CHIN_WINDOW
    chin_rep = dict(window=list(CHIN_WINDOW), formula="validate_stage.py: error=(q[152]-g[152]).down(source), stats over frames 24..215")
    for key, arm in (("A", A), ("B", B)):
        if arm.g_arr is None:
            chin_rep[key] = None
            continue
        q, g = arm.q, arm.g_arr
        err, length, src_err, resid = [], [], [], []
        for i in range(T):
            center, right, down, span = chin.axes(p[i])
            err.append(float((q[i][152] - g[i][152]) @ down))
            length.append(float((q[i][152] - q[i][17]) @ down))
            src_err.append(float(np.linalg.norm(q[i][152] - p[i][152]) / span * 100))
            resid.append((q[i][chin.JAW] - p[i][chin.JAW]) @ np.stack([right, down], axis=1) / span)
        err = np.asarray(err)
        chin_rep[key] = dict(
            target_chin_abs_error_px=stats(np.abs(err)[lo:hi]),
            positive_excess_chin_length_px=stats(np.maximum(err, 0)[lo:hi]),
            lower_lip_to_chin_px=stats(np.asarray(length)[lo:hi]),
            source_chin_error_percent_eye_span=stats(np.asarray(src_err)[lo:hi]),
            source_relative_jaw_step_percent_eye_span=float(np.linalg.norm(np.diff(np.asarray(resid)[lo:hi], axis=0), axis=-1).mean() * 100),
            series_signed_error_px=err)
    if chin_rep.get("A") and chin_rep.get("B"):
        chin_rep["delta_target_error_mean_px"] = (chin_rep["B"]["target_chin_abs_error_px"]["mean"]
                                                  - chin_rep["A"]["target_chin_abs_error_px"]["mean"])
    dev = np.linalg.norm(qA[:, LANDMARK_SET] - qB[:, LANDMARK_SET], axis=-1)
    lm = dict(landmarks="JAW(21)+LIPS(20)+inner lips(20) = %d FaceMesh points, all frames" % len(LANDMARK_SET),
              tracked_final=dict(mean=float(np.nanmean(dev)), p99=float(np.nanpercentile(dev, 99)), max=float(np.nanmax(dev)),
                                 jaw_mean=float(np.nanmean(np.linalg.norm(qA[:, chin.JAW] - qB[:, chin.JAW], axis=-1))),
                                 lips_mean=float(np.nanmean(np.linalg.norm(qA[:, chin.LIPS + INNER_LIPS] - qB[:, chin.LIPS + INNER_LIPS], axis=-1))),
                                 window_mean=float(np.nanmean(dev[lo:hi])), window_p99=float(np.nanpercentile(dev[lo:hi], 99))),
              missing_face_frames=missing)
    if A.g_arr is not None and B.g_arr is not None:
        gd = np.linalg.norm(A.g_arr[:, LANDMARK_SET] - B.g_arr[:, LANDMARK_SET], axis=-1)
        lm["generated_in_render"] = dict(mean=float(gd.mean()), p99=float(np.percentile(gd, 99)), max=float(gd.max()),
                                         note="tracker landmarks recorded inside each render (generated face pasted on source); report-only")
    chin_rep["landmark_deviation_A_vs_B"] = lm

    # ---- overall
    def psnr_block(key):
        mses = np.array([r[f"{key}_mse"] for r in rows])
        per = np.array([min(psnr(m), PSNR_CAP_DB) for m in mses])
        return dict(global_db=psnr(float(mses.mean())), mean_frame_db_capped=float(per.mean()),
                    worst_frame_db=float(per.min()), identical_frames=int((mses == 0).sum()))
    labA, labB = np.mean(lab_mean["A"], 0), np.mean(lab_mean["B"], 0)
    per_frame_mean_delta = np.linalg.norm(np.asarray(lab_mean["B"]) - np.asarray(lab_mean["A"]), axis=1)
    overall = dict(
        psnr=dict(full=psnr_block("full"), face_box=psnr_block("face"), mouth_roi=psnr_block("mouth")),
        ssim=dict(full=agg("ssim_full"), face_box=agg("ssim_face"), mouth_roi=agg("ssim_mouth"),
                  worst=dict(full=float(min(r["ssim_full"] for r in rows)), face_box=float(min(r["ssim_face"] for r in rows)),
                             mouth_roi=float(min(r["ssim_mouth"] for r in rows)))),
        mouth_mae_normalized=float(np.mean([r["mouth_mae"] for r in rows]) / 255),
        face_max_abs=agg("face_max"), full_max_abs=agg("full_max"),
        sharpness=dict(definition="variance of cv2.Laplacian(gray) per frame, mean over frames",
                       mouth=dict(A=float(np.mean(sharp["A"]["mouth"])), B=float(np.mean(sharp["B"]["mouth"])),
                                  ratio=ratio(float(np.mean(sharp["B"]["mouth"])), float(np.mean(sharp["A"]["mouth"])))),
                       face=dict(A=float(np.mean(sharp["A"]["face"])), B=float(np.mean(sharp["B"]["face"])),
                                 ratio=ratio(float(np.mean(sharp["B"]["face"])), float(np.mean(sharp["A"]["face"]))))),
        color_lab_face_box=dict(A_mean_Lab=labA, B_mean_Lab=labB, delta_mean_Lab=labB - labA,
                                deltaE76_of_mean=float(np.linalg.norm(labB - labA)),
                                deltaE76_of_per_frame_means=stats(per_frame_mean_delta),
                                per_pixel_deltaE76_mean=float(np.mean([r["face_deltaE76_mean"] for r in rows]))))

    report = dict(
        tool="scripts/quality_ab_metrics.py", tool_sha256=TOOL_SHA256, helper_sha256=HELPER_SHA256,
        name=name, profile=profile, frames_mode=mode, frames=T, identical_frames_sha=identical,
        identity=ident.signature(), A=A.describe() | {"info": A.info}, B=B.describe() | {"info": B.info},
        environment=dict(chin_sha256=sha_file(H3 / "chin.py"), chin_is_accepted=sha_file(H3 / "chin.py") == ACCEPTED_CHIN_SHA,
                         blending_sha256=sha_file(ROOT / "musetalk/utils/blending.py"),
                         blending_is_accepted=sha_file(ROOT / "musetalk/utils/blending.py") == ACCEPTED_BLENDING_SHA,
                         blend_env={k: os.environ.get(k) for k in ("MUSETALK_BLEND_FIXED_POINT", "MUSETALK_BLEND_SHRINK_MASK_BBOX")},
                         numpy=np.__version__, opencv=cv2.__version__),
        lip_sync=lip, flicker=flick, seam=seam, chin=chin_rep, overall=overall, notes=notes)
    if syncnet:
        report["syncnet"] = syncnet_eval(ident, [A, B])
    report["gates"] = evaluate_gates(report, profile)
    report["verdict"] = verdict(report["gates"])
    report["seconds"] = time.perf_counter() - t0
    import resource
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report = clean(report)
    if out_dir is not None and name:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
        (out_dir / f"{name}.md").write_text(markdown(report) + "\n")
    return report


# ------------------------------------------------------------------------------------ gates
def evaluate_gates(r, profile):
    g = []

    def add(gate, value, threshold, ok, applies=True, note=None):
        res = "not-run" if value is None else ("pass" if ok else "fail")
        extra = {"note": note} if note else {}
        if not applies and res != "not-run":
            res = "report"
            extra["would_pass"] = bool(ok)
        g.append(dict(gate=gate, value=value, threshold=threshold, result=res, **extra))

    raw_ok = r["frames_mode"] == "raw"
    ls = r["lip_sync"]
    corr = ls["pearson_A_vs_B"]
    add("lip.aperture_corr", corr, f">= {GATES['aperture_corr_min']}", corr is not None and corr >= GATES["aperture_corr_min"])
    lag = ls["xcorr_lag_frames"]
    add("lip.xcorr_lag_frames", lag, "== 0", lag == 0)
    mad = ls["mean_abs_delta_px"]
    add("lip.mean_abs_delta_px", mad, f"<= {GATES['aperture_mean_abs_delta_px_max']}", mad is not None and mad <= GATES["aperture_mean_abs_delta_px_max"])
    for reg in ("mouth", "jaw", "ring"):
        v = r["flicker"][reg]["ratio_first_diff"]
        add(f"flicker.{reg}_ratio", v, f"<= {GATES['flicker_ratio_max']}", v is not None and v <= GATES["flicker_ratio_max"])
    for reg in ("mouth_box", "jaw_box"):
        v = r["flicker"]["temporal_hf_power_gt_6hz"][reg]["ratio"]
        add(f"flicker.hf_{reg}_ratio", v, f"<= {GATES['flicker_ratio_max']} (report)", v is not None and v <= GATES["flicker_ratio_max"], applies=False)
    ch = r["chin"]
    margin = GATES["chin_error_margin_px"][profile]
    if ch.get("A") and ch.get("B"):
        a = ch["A"]["target_chin_abs_error_px"]["mean"]
        b = ch["B"]["target_chin_abs_error_px"]["mean"]
        add("chin.target_error_mean_px", b, f"<= A ({a:.4f}) + {margin}", b <= a + margin + 1e-12,
            applies=raw_ok or profile == "lossy")
    else:
        add("chin.target_error_mean_px", None, f"<= A + {margin}", False, note="needs generated_landmarks for both arms")
    lm = ch["landmark_deviation_A_vs_B"]["tracked_final"]
    e1 = profile in ("e1", "exact") and raw_ok
    if profile in ("e1", "exact"):
        add("frames.raw_pre_encode", r["frames_mode"], "== raw (codec noise alone exceeds the G-TRACK thresholds)", raw_ok,
            note=None if raw_ok else "encoded input: landmark/chin gates downgraded to report")
        if not raw_ok:
            g[-1]["result"] = "not-run"
    add("chin.landmark_dev_mean_px", lm["mean"], f"<= {GATES['landmark_dev_mean_px_max']}", lm["mean"] <= GATES["landmark_dev_mean_px_max"], applies=e1)
    add("chin.landmark_dev_p99_px", lm["p99"], f"<= {GATES['landmark_dev_p99_px_max']}", lm["p99"] <= GATES["landmark_dev_p99_px_max"], applies=e1)
    add("chin.landmark_dev_calibrated_proposal", lm["mean"],
        f"mean <= {GATES['landmark_dev_mean_px_proposed']} and p99 <= {GATES['landmark_dev_p99_px_proposed']} (proposed, report)",
        lm["mean"] <= GATES["landmark_dev_mean_px_proposed"] and lm["p99"] <= GATES["landmark_dev_p99_px_proposed"], applies=False)
    add("track.missing_face_frames", ch["landmark_deviation_A_vs_B"]["missing_face_frames"], "== 0",
        ch["landmark_deviation_A_vs_B"]["missing_face_frames"] == 0)
    for key in ("A", "B"):
        pl = r["seam"]["protected_lip"].get(key)
        v = None if not pl else pl["max_rgb_difference"]
        add(f"seam.protected_lip_{key}", v, "== 0 vs own standard compose", v == 0,
            note=None if pl else "raw mode only (faces + generated landmarks)")
    sr = r["overall"]["sharpness"]["mouth"]["ratio"]
    add("overall.mouth_sharpness_ratio", sr, f">= {GATES['sharpness_ratio_min']}", sr is not None and sr >= GATES["sharpness_ratio_min"])
    add("overall.mouth_sharpness_upper", sr, f"<= {GATES['sharpness_ratio_warn_max']} (L5 band, report)",
        sr is not None and sr <= GATES["sharpness_ratio_warn_max"], applies=False)
    add("overall.face_psnr_db", r["overall"]["psnr"]["face_box"]["mean_frame_db_capped"], "report-only", True, applies=False)
    if profile == "exact":
        add("exact.frames_sha_identical", r["identical_frames_sha"], "== true (pre-encode frames)", bool(r["identical_frames_sha"]))
    return g


def verdict(gates):
    res = [x["result"] for x in gates]
    if "fail" in res:
        return "FAIL"
    if "not-run" in res:
        return "INCOMPLETE"
    return "PASS"


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if abs(v) >= 1000:
            return f"{v:.1f}"
        return f"{v:.{nd}f}"
    return str(v)


def markdown(r):
    A, B = r["A"], r["B"]
    ls, fl, se, ch, ov = r["lip_sync"], r["flicker"], r["seam"], r["chin"], r["overall"]
    L = [f"### {r['name']} — {r['verdict']} (profile {r['profile']}, frames {r['frames_mode']}, {r['frames']} frames)",
         "", f"A = `{A['label']}` ({A['compose']}), B = `{B['label']}` ({B['compose']}); "
             f"identity `{r['identity']['name']}`; pre-encode/decoded frames SHA-identical: **{r['identical_frames_sha']}**", "",
         "| Metric | A | B | Delta / ratio | Threshold | Result |", "|---|---:|---:|---:|---|---|"]
    gates = {g["gate"]: g for g in r["gates"]}

    def res(name):
        g = gates.get(name)
        if not g:
            return ("", "")
        tag = g["result"].upper()
        if "would_pass" in g:
            tag += " (ok)" if g["would_pass"] else " (over)"
        return (g["threshold"], tag)

    def row(metric, a, b, d, gate=None):
        t, s = res(gate) if gate else ("report", "")
        L.append(f"| {metric} | {fmt(a)} | {fmt(b)} | {fmt(d)} | {t} | {s} |")
    row("Lip aperture mean (eye spans); delta = Pearson A vs B", ls["A"]["aperture_eye_spans"]["mean"], ls["B"]["aperture_eye_spans"]["mean"],
        ls["pearson_A_vs_B"], "lip.aperture_corr")
    row("Aperture xcorr lag (frames)", "-", "-", ls["xcorr_lag_frames"], "lip.xcorr_lag_frames")
    row("Aperture mean abs delta (px)", ls["A"]["aperture_px"]["mean"], ls["B"]["aperture_px"]["mean"], ls["mean_abs_delta_px"], "lip.mean_abs_delta_px")
    if r.get("syncnet"):
        sn = r["syncnet"]
        row("SyncNet conf (uncalibrated) / offset", sn["arms"][0]["confidence"], sn["arms"][1]["confidence"],
            f"offsets {sn['arms'][0]['offset_frames']}/{sn['arms'][1]['offset_frames']}")
    for reg, gate in (("mouth", "flicker.mouth_ratio"), ("jaw", "flicker.jaw_ratio"), ("ring", "flicker.ring_ratio")):
        row(f"Flicker {reg} mean abs(dt) RGB", fl[reg]["A_first_diff"], fl[reg]["B_first_diff"], fl[reg]["ratio_first_diff"], gate)
    row("Flicker mouth 2nd-diff", fl["mouth"]["A_second_diff"], fl["mouth"]["B_second_diff"], fl["mouth"]["ratio_second_diff"])
    hfm = fl["temporal_hf_power_gt_6hz"]["mouth_box"]
    row("Mouth-box >6 Hz temporal power", hfm["A_power"], hfm["B_power"], hfm["ratio"], "flicker.hf_mouth_box_ratio")
    row("Seam band abs(A-B): per-frame max p99 / max; delta = mean", "-", f"{fmt(se['band_max_abs']['p99'], 1)} / {fmt(se['band_max_abs']['max'], 1)}", se["band_mean_abs"]["mean"])
    row("Seam ring abs(A-B): per-frame max p99 / max; delta = mean", "-", f"{fmt(se['ring_max_abs']['p99'], 1)} / {fmt(se['ring_max_abs']['max'], 1)}", se["ring_mean_abs"]["mean"])
    row("Outside-mask abs(A-B): max; delta = mean", "-", fmt(se["outside_max_abs"]["max"], 1), se["outside_mean_abs"]["mean"])
    pa, pb = se["protected_lip"].get("A"), se["protected_lip"].get("B")
    row("Protected-lip max RGB change vs own standard", pa["max_rgb_difference"] if pa else None, pb["max_rgb_difference"] if pb else None, None, "seam.protected_lip_B")
    if ch.get("A") and ch.get("B"):
        row("Chin-target abs error mean (px)", ch["A"]["target_chin_abs_error_px"]["mean"], ch["B"]["target_chin_abs_error_px"]["mean"],
            ch["delta_target_error_mean_px"], "chin.target_error_mean_px")
        row("Chin positive excess p95 (px)", ch["A"]["positive_excess_chin_length_px"]["p95"], ch["B"]["positive_excess_chin_length_px"]["p95"],
            ch["B"]["positive_excess_chin_length_px"]["p95"] - ch["A"]["positive_excess_chin_length_px"]["p95"])
    lm = ch["landmark_deviation_A_vs_B"]["tracked_final"]
    row("Jaw+lip landmark dev A vs B mean (px)", "-", "-", lm["mean"], "chin.landmark_dev_mean_px")
    row("Jaw+lip landmark dev A vs B p99 (px)", "-", "-", lm["p99"], "chin.landmark_dev_p99_px")
    if "chin.landmark_dev_calibrated_proposal" in gates:
        row("Landmark dev vs calibrated proposal (mean/p99)", "-", "-", f"{fmt(lm['mean'])} / {fmt(lm['p99'])}",
            "chin.landmark_dev_calibrated_proposal")
    ps, ss = ov["psnr"], ov["ssim"]
    row("PSNR A vs B, dB mean frame (cap 100)", "-", f"full {fmt(ps['full']['mean_frame_db_capped'], 2)} / face {fmt(ps['face_box']['mean_frame_db_capped'], 2)}",
        f"mouth {fmt(ps['mouth_roi']['mean_frame_db_capped'], 2)}; worst face {fmt(ps['face_box']['worst_frame_db'], 2)}", "overall.face_psnr_db")
    row("SSIM A vs B mean", "-", f"full {fmt(ss['full']['mean'])} / face {fmt(ss['face_box']['mean'])}",
        f"mouth {fmt(ss['mouth_roi']['mean'])}; worst mouth {fmt(ss['worst']['mouth_roi'])}")
    row("Mouth sharpness (Laplacian var)", ov["sharpness"]["mouth"]["A"], ov["sharpness"]["mouth"]["B"], ov["sharpness"]["mouth"]["ratio"], "overall.mouth_sharpness_ratio")
    c = ov["color_lab_face_box"]
    row("Face Lab L* mean; delta = dE76 of means", c["A_mean_Lab"][0], c["B_mean_Lab"][0], c["deltaE76_of_mean"])
    return "\n".join(L)


# ---------------------------------------------------------------------------------- syncnet
_SYNCNET = {}


def syncnet_eval(ident, arms, max_offset=8, stride=2):
    """LatentSync SyncNet (models/syncnet/latentsync_syncnet.pt) confidence/offset per arm.

    Uncalibrated for these avatars and for 24 fps (the model was trained at 25 fps; the 52-frame mel
    window spans 16/24 s here with 4% time-scale mismatch). Use only relative A-vs-B numbers.
    Visual input: identity face box (same for both arms) -> 256x256 RGB [-1,1], lower half, 16 frames.
    """
    import torch
    import torch.nn.functional as F
    if "model" not in _SYNCNET:
        from omegaconf import OmegaConf
        from musetalk.models.syncnet import SyncNet
        cfg = OmegaConf.load(ROOT / "configs/training/syncnet.yaml")
        with torch.device("cuda"):
            model = SyncNet(OmegaConf.to_container(cfg.model))
        ck = torch.load(ROOT / "models/syncnet/latentsync_syncnet.pt", map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(ck["state_dict"])
        del ck
        _SYNCNET["model"] = model.half().eval()
    model = _SYNCNET["model"]
    from musetalk.data import audio as mt_audio
    import librosa
    wav = librosa.load(str(ident.audio), sr=16000)[0]
    mel = mt_audio.melspectrogram(wav).T          # (N, 80)
    T = len(ident.frames)
    boxes = ident.d["cache"]["boxes"]
    steps = 52

    def mel_at(t):
        s = int(80. * (t / float(FPS)))
        s = min(max(s, 0), len(mel) - steps)
        return mel[s:s + steps].T

    starts = list(range(max_offset, T - 16 - max_offset + 1, stride))
    with torch.inference_mode():
        needed = sorted({t + k for t in starts for k in range(-max_offset, max_offset + 1)})
        a_in = torch.from_numpy(np.stack([mel_at(t) for t in needed])[:, None]).float().cuda().half()
        a_emb = torch.cat([model.get_audio_embed(a_in[j:j + 64]) for j in range(0, len(a_in), 64)]).float()
        a_idx = {t: j for j, t in enumerate(needed)}
        out = []
        for arm in arms:
            crops = []
            for i, f in enumerate(arm.frames):
                x, y, x1, y1 = [int(v) for v in boxes[i]]
                c = cv2.resize(f[max(0, y):y1, max(0, x):x1], (256, 256), interpolation=cv2.INTER_AREA)
                crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)[128:])
            crops = torch.from_numpy(np.stack(crops)).cuda().half().div(127.5).sub(1).permute(0, 3, 1, 2)  # T,3,128,256
            sims = []
            for j in range(0, len(starts), 16):
                chunk = starts[j:j + 16]
                v_in = torch.stack([crops[t:t + 16].reshape(48, 128, 256) for t in chunk])
                v_emb = model.get_image_embed(v_in).float()
                for n, t in enumerate(chunk):
                    ae = a_emb[[a_idx[t + k] for k in range(-max_offset, max_offset + 1)]]
                    sims.append(F.cosine_similarity(v_emb[n:n + 1], ae, dim=1).cpu().numpy())
            sims = np.stack(sims)                      # windows x offsets
            curve = sims.mean(0)
            k = int(np.argmax(curve)) - max_offset
            out.append(dict(label=arm.label, offset_frames=k, sim_at_0=float(curve[max_offset]), sim_best=float(curve.max()),
                            confidence=float(curve.max() - np.median(curve)), windows=len(starts),
                            curve={str(o - max_offset): float(v) for o, v in enumerate(curve)}))
    return dict(model="LatentSync SyncNet 16-frame pixel (models/syncnet/latentsync_syncnet.pt)",
                note="uncalibrated; relative A-vs-B only; confidence = max - median of the mean cosine-similarity offset curve",
                arms=out, delta_confidence=out[1]["confidence"] - out[0]["confidence"],
                delta_sim_at_0=out[1]["sim_at_0"] - out[0]["sim_at_0"])


# ------------------------------------------------------------------------------ calibration
def pjv_identity(who):
    man = json.loads((PJV / f"{who}_new/manifest.json").read_text())
    return Identity(name=who, source=PJV / who / "source.mp4", landmarks=PJV / "analysis" / f"{who}_new_portrait/source_landmarks.npy",
                    cache=PJV / f"{who}_new/cache.pt", masks=PJV / f"{who}_new/masks.npz", audio=Path(man["audio"]))


def summary_line(r):
    ls, fl, ch, ov, se = r["lip_sync"], r["flicker"], r["chin"], r["overall"], r["seam"]
    lm = ch["landmark_deviation_A_vs_B"]["tracked_final"]
    ca = ch["A"]["target_chin_abs_error_px"]["mean"] if ch.get("A") else None
    cb = ch["B"]["target_chin_abs_error_px"]["mean"] if ch.get("B") else None
    pl = se["protected_lip"]
    return dict(name=r["name"], verdict=r["verdict"], frames_mode=r["frames_mode"], sha_identical=r["identical_frames_sha"],
                aperture_corr=ls["pearson_A_vs_B"], lag=ls["xcorr_lag_frames"], aperture_mad_px=ls["mean_abs_delta_px"],
                flicker_mouth=fl["mouth"]["ratio_first_diff"], flicker_jaw=fl["jaw"]["ratio_first_diff"], flicker_ring=fl["ring"]["ratio_first_diff"],
                hf_mouth=fl["temporal_hf_power_gt_6hz"]["mouth_box"]["ratio"],
                band_max_p99=se["band_max_abs"]["p99"], ring_mean=se["ring_mean_abs"]["mean"],
                lip_A=pl["A"]["max_rgb_difference"] if pl.get("A") else None, lip_B=pl["B"]["max_rgb_difference"] if pl.get("B") else None,
                chin_A=ca, chin_B=cb, lm_mean=lm["mean"], lm_p99=lm["p99"],
                psnr_face=ov["psnr"]["face_box"]["mean_frame_db_capped"], psnr_mouth=ov["psnr"]["mouth_roi"]["mean_frame_db_capped"],
                ssim_mouth=ov["ssim"]["mouth_roi"]["mean"], sharp_ratio=ov["sharpness"]["mouth"]["ratio"],
                dE_mean=ov["color_lab_face_box"]["deltaE76_of_mean"],
                failed=[g["gate"] for g in r["gates"] if g["result"] == "fail"])


def calibrate(out_dir, ids=None, include_pair=True, include_reencode=True):
    out_dir = Path(out_dir)
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("QAB_SCRATCH", "/tmp")) / "qab_calibration"
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    extra = {}

    def run(ident, A, B, name, mode, category, notes=None, profile="e1"):
        print(f"[calibrate] {name} ({mode})", flush=True)
        r = compare(ident, A, B, mode=mode, profile=profile, name=name, out_dir=runs,
                    notes=dict(category=category, identity=ident.name, text=notes))
        s = summary_line(r)
        s["category"] = category
        s["identity"] = ident.name
        results.append(s)
        print(f"[calibrate]   -> {r['verdict']} {r['seconds']:.1f}s failed={s['failed']}", flush=True)
        return r

    for who in (ids or DIV_IDS):
        ident = Identity.from_dir(DIV / who)
        base = DIV / who
        # 1. self (video): the accepted mp4 against itself, FaceMesh run twice independently.
        run(ident, Arm.parse(f"dir={base}", "A_refined_mp4"), Arm.parse(f"dir={base}", "A_refined_mp4_again"),
            f"{who}__self_video", "video", "self")
        # 2. self (raw): two independent bit-exact reconstructions from faces.npz with chin.py.
        rawA = Arm.parse(f"dir={base}", "A_refined_raw")
        rawA2 = Arm.parse(f"dir={base}", "A_refined_raw_again")
        run(ident, rawA, rawA2, f"{who}__self_raw", "raw", "self")
        del rawA, rawA2
        # 3. codec noise: raw pre-encode frames vs the accepted crf18 encode of the same frames.
        vidA = Arm.parse(f"dir={base}", "A_refined_mp4")
        vidA.faces = None  # force video
        rawC = Arm.parse(f"dir={base}", "A_refined_raw")
        load_arm(ident, rawC, "raw")
        load_arm(ident, vidA, "video")
        run(ident, rawC, vidA, f"{who}__codec_raw_vs_crf18", "raw_vs_video", "codec")
        # 4. codec noise: two different re-encodes of the same raw frames.
        if include_reencode:
            e1, e2 = scratch / f"{who}_enc_crf18_fast.mp4", scratch / f"{who}_enc_crf18_medium.mp4"
            encode_video(e1, rawC.frames, ident.audio, crf=18, preset="fast", threads=2)
            encode_video(e2, rawC.frames, ident.audio, crf=18, preset="medium", threads=4)
            dec1 = read_video(e1)
            extra[f"{who}_reencode_crf18_fast_equals_accepted_mp4"] = sha_frames(dec1) == vidA.info["frames_sha256"]
            del dec1
            E1 = Arm(label="reencode_crf18_fast", compose="refined", video=e1, g=base / "generated_landmarks.npy", chin_delta=base / "chin_delta.npy")
            E2 = Arm(label="reencode_crf18_medium", compose="refined", video=e2, g=base / "generated_landmarks.npy", chin_delta=base / "chin_delta.npy")
            run(ident, E1, E2, f"{who}__codec_crf18fast_vs_crf18medium", "video", "codec")
            e1.unlink(missing_ok=True)
            e2.unlink(missing_ok=True)
        del rawC, vidA
        # 5. sensitivity: accepted refined vs accepted standard (chin100 on vs off), video and raw.
        # 5. faces-only candidate path: g re-tracked from faces.npz with the render's Tracker procedure,
        #    chin_delta recomputed; must be SHA-identical to the accepted raw render.
        run(ident, Arm.parse(f"dir={base}", "A_refined_raw"), Arm.parse(f"faces={base / 'faces.npz'},retrack=1", "B_faces_only"),
            f"{who}__faces_only_retracked_raw", "raw", "self")
        # 6. synthetic E1-like perturbation: generated faces + iid integer noise, full re-track + recompose.
        for tag, spec in (("sparse1lsb20pct", "perturb=1,frac=0.2"), ("noise1lsb", "perturb=1"), ("noise3lsb", "perturb=3")):
            run(ident, Arm.parse(f"dir={base}", "A_refined_raw"), Arm.parse(f"faces={base / 'faces.npz'},{spec},seed=7", f"B_faces_{tag}"),
                f"{who}__synthetic_faces_{tag}_raw", "raw", f"synthetic_{tag}")
        # 7. sensitivity: accepted refined vs accepted standard (chin100 on vs off), video and raw.
        run(ident, Arm.parse(f"dir={base}", "A_refined"), Arm.parse(f"dir={base},compose=standard", "B_standard"),
            f"{who}__refined_vs_standard_video", "video", "sensitivity_refined_vs_standard")
        run(ident, Arm.parse(f"dir={base}", "A_refined"), Arm.parse(f"dir={base},compose=standard", "B_standard"),
            f"{who}__refined_vs_standard_raw", "raw", "sensitivity_refined_vs_standard")
        ident._frames = ident._d = None
    if include_pair:
        for who in ("japanese", "latina"):
            ident = pjv_identity(who)
            A = Arm(label="approved_INT8_chin100", compose="source", video=CFV / who / "int8_pipelined/aligned_raw.mp4",
                    g=CFV / who / "int8_pipelined/generated_landmarks.npy")
            B = Arm(label="TAESD_chin100", compose="source", video=CFV / who / "taesd_pipelined/aligned_raw.mp4",
                    g=CFV / who / "taesd_pipelined/generated_landmarks.npy")
            r = run(ident, A, B, f"{who}__int8_vs_taesd_video", "video", "sensitivity_int8_vs_taesd", profile="lossy")
            prev = CFV / who / "taesd_final_landmarks.npy"
            if prev.exists():
                extra[f"{who}_taesd_tracked_equals_chin_fps_validation"] = bool(np.array_equal(B.q, np.load(prev)))
    (out_dir / "calibration_checks.json").write_text(json.dumps(clean(extra), indent=1) + "\n")
    return write_reports(out_dir)


def write_reports(out_dir):
    """Rebuild calibration_summary.{json,md} and per_identity/<id>.md from runs/*.json."""
    out_dir = Path(out_dir)
    runs = sorted((out_dir / "runs").glob("*.json"), key=lambda q: q.stat().st_mtime)
    results, per_id = [], {}
    for path in runs:
        r = json.loads(path.read_text())
        s = summary_line(r)
        notes = r.get("notes") or {}
        s["category"] = notes.get("category")
        s["identity"] = notes.get("identity") or r["identity"]["name"]
        s["tool_sha256"] = r["tool_sha256"]
        results.append(s)
        per_id.setdefault(s["identity"], []).append(r)
    checks_path = out_dir / "calibration_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    sync_path = out_dir / "syncnet_calibration.json"
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else None
    summary = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gates=GATES, results=results,
                   checks=checks, syncnet=sync, floors=floors(results))
    (out_dir / "calibration_summary.json").write_text(json.dumps(clean(summary), indent=1) + "\n")
    (out_dir / "calibration_summary.md").write_text(calibration_markdown(clean(summary)) + "\n")
    pid = out_dir / "per_identity"
    pid.mkdir(exist_ok=True)
    for who, rs in per_id.items():
        body = [f"# {who}: quality A/B metrics", "",
                "Generated by scripts/quality_ab_metrics.py. Thresholds: ../README.md. JSON per comparison: ../runs/.", ""]
        for r in rs:
            body += [markdown(r), ""]
        (pid / f"{who}.md").write_text("\n".join(body) + "\n")
    return summary


FLOOR_KEYS = [("aperture_corr", min), ("lag", lambda v: max(v, key=abs)), ("aperture_mad_px", max), ("flicker_mouth", None),
              ("flicker_jaw", None), ("flicker_ring", None), ("band_max_p99", max), ("chin_delta_px", None), ("lm_mean", max),
              ("lm_p99", max), ("psnr_face", min), ("psnr_mouth", min), ("ssim_mouth", min), ("sharp_ratio", None), ("dE_mean", max)]


def floors(results):
    """Per calibration category: [min, max] across identities for each summary metric."""
    out = {}
    for r in results:
        r = dict(r)
        if r.get("chin_A") is not None and r.get("chin_B") is not None:
            r["chin_delta_px"] = r["chin_B"] - r["chin_A"]
        cat = out.setdefault(r.get("category") or "uncategorised", dict(n=0))
        cat["n"] += 1
        for k, _ in FLOOR_KEYS:
            v = r.get(k)
            if v is None:
                continue
            lo, hi = cat.get(k, [v, v])
            cat[k] = [min(lo, v), max(hi, v)]
        cat.setdefault("verdicts", []).append(r["verdict"])
    return out


def floors_markdown(f):
    cols = [("n", "n"), ("aperture_corr", "Ap. corr"), ("lag", "Lag"), ("aperture_mad_px", "Ap. abs-delta px"), ("flicker_mouth", "Flk mouth"),
            ("flicker_jaw", "Flk jaw"), ("flicker_ring", "Flk ring"), ("band_max_p99", "Band max p99"), ("chin_delta_px", "Chin B-A px"),
            ("lm_mean", "LM mean px"), ("lm_p99", "LM p99 px"), ("psnr_face", "PSNR face"), ("psnr_mouth", "PSNR mouth"),
            ("ssim_mouth", "SSIM mouth"), ("sharp_ratio", "Sharp B/A"), ("dE_mean", "dE76")]
    L = ["| Category | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for cat, v in f.items():
        cells = []
        for k, _ in cols:
            x = v.get(k)
            if isinstance(x, list):
                cells.append(fmt(x[0], 3) if x[0] == x[1] else f"{fmt(x[0], 3)} .. {fmt(x[1], 3)}")
            else:
                cells.append(fmt(x, 3))
        L.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def calibration_markdown(s):
    cols = [("name", "Comparison"), ("verdict", "Verdict"), ("aperture_corr", "Ap. corr"), ("lag", "Lag"), ("aperture_mad_px", "Ap. abs-delta px"),
            ("flicker_mouth", "Flk mouth B/A"), ("flicker_jaw", "Flk jaw B/A"), ("flicker_ring", "Flk ring B/A"),
            ("band_max_p99", "Band max p99"), ("lip_A", "Lip A"), ("lip_B", "Lip B"), ("chin_A", "Chin A px"), ("chin_B", "Chin B px"),
            ("lm_mean", "LM mean px"), ("lm_p99", "LM p99 px"), ("psnr_face", "PSNR face"), ("psnr_mouth", "PSNR mouth"),
            ("ssim_mouth", "SSIM mouth"), ("sharp_ratio", "Sharp B/A"), ("dE_mean", "dE76")]
    L = []
    if s.get("floors"):
        L += ["Range across identities per calibration category (min .. max):", "", floors_markdown(s["floors"]), "",
              "Every comparison:", ""]
    L += ["| " + " | ".join(c[1] for c in cols) + " | Failed gates |", "|" + "---|" * (len(cols) + 1)]
    for r in s["results"]:
        r = dict(r, name=r["name"].replace("__", " "))
        L.append("| " + " | ".join(fmt(r[c[0]], 3) for c in cols) + " | " + (", ".join(r["failed"]) or "-") + " |")
    if s.get("syncnet"):
        L += ["", "SyncNet (LatentSync 16-frame pixel model; uncalibrated, relative only). Offset k = the audio window starting k frames "
              "after the video window matches best (k > 0: lips lead the audio). Same model, crop and audio for every arm:", "",
              "| Identity | Arm | Confidence (max - median) | Offset | Cos-sim at 0 | Delta conf vs first arm |", "|---|---|---:|---:|---:|---:|"]
        for r in s["syncnet"]["results"]:
            c0 = r["arms"][0]["confidence"]
            for a in r["arms"]:
                L.append(f"| {r['identity']} | {a['label']} | {fmt(a['confidence'])} | {a['offset_frames']} | {fmt(a['sim_at_0'])} | {fmt(a['confidence'] - c0)} |")
    return "\n".join(L)


# --------------------------------------------------------------------------------------- cli
def identity_from_args(a):
    if a.identity_dir:
        ident = Identity.from_dir(a.identity_dir)
    else:
        ident = Identity(name=a.identity_name or Path(a.source).parent.name, source=Path(a.source), landmarks=Path(a.source_landmarks),
                         cache=Path(a.cache), masks=Path(a.masks), audio=Path(a.audio) if a.audio else None)
    if a.audio:
        ident.audio = Path(a.audio)
    return ident


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pair", help="compare arm A (reference) with arm B (candidate)")
    pp.add_argument("--identity-dir")
    pp.add_argument("--identity-name")
    pp.add_argument("--source")
    pp.add_argument("--source-landmarks")
    pp.add_argument("--cache")
    pp.add_argument("--masks")
    pp.add_argument("--audio")
    pp.add_argument("--a", required=True)
    pp.add_argument("--b", required=True)
    pp.add_argument("--frames", default="auto", choices=["auto", "raw", "video"])
    pp.add_argument("--profile", default="e1", choices=["e1", "lossy", "exact"])
    pp.add_argument("--name", required=True)
    pp.add_argument("--out-dir", default=str(OUT_ROOT / "runs"))
    pp.add_argument("--syncnet", action="store_true", help="GPU; run under scripts/box_guard.sh")
    cp = sub.add_parser("calibrate", help="noise floors + sensitivity on the accepted data")
    cp.add_argument("--out-dir", default=str(OUT_ROOT))
    cp.add_argument("--ids", nargs="*")
    cp.add_argument("--no-pair", action="store_true")
    cp.add_argument("--no-reencode", action="store_true")
    rp = sub.add_parser("report", help="rebuild summary and per-identity tables from runs/*.json")
    rp.add_argument("--out-dir", default=str(OUT_ROOT))
    sp = sub.add_parser("syncnet-calibrate", help="GPU SyncNet on refined vs standard and INT8 vs TAESD (box_guard)")
    sp.add_argument("--out-dir", default=str(OUT_ROOT))
    sp.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    ram_guard("start")
    if a.cmd == "pair":
        ident = identity_from_args(a)
        A = Arm.parse(a.a, "A")
        B = Arm.parse(a.b, "B")
        r = compare(ident, A, B, mode=a.frames, profile=a.profile, name=a.name, out_dir=a.out_dir, syncnet=a.syncnet)
        print(markdown(r))
        print(f"\nVERDICT {r['verdict']}  json={Path(a.out_dir) / (a.name + '.json')}")
        sys.exit(0 if r["verdict"] == "PASS" else 1)
    elif a.cmd == "calibrate":
        s = calibrate(a.out_dir, ids=a.ids, include_pair=not a.no_pair, include_reencode=not a.no_reencode)
        print(calibration_markdown(clean(s)))
        print(json.dumps(clean(s["checks"]), indent=1))
    elif a.cmd == "report":
        s = write_reports(a.out_dir)
        print(calibration_markdown(clean(s)))
    elif a.cmd == "syncnet-calibrate":
        syncnet_calibrate(a.out_dir, a.ids)


def syncnet_calibrate(out_dir, ids=None):
    out = []
    for who in (ids or DIV_IDS):
        ident = Identity.from_dir(DIV / who)
        A = load_arm(ident, Arm.parse(f"dir={DIV / who}", "refined_mp4"), "video")
        B = load_arm(ident, Arm.parse(f"dir={DIV / who},compose=standard", "standard_mp4"), "video")
        r = syncnet_eval(ident, [A, B])
        r["identity"] = who
        out.append(r)
        print(who, [(x["label"], round(x["confidence"], 4), x["offset_frames"], round(x["sim_at_0"], 4)) for x in r["arms"]], flush=True)
    for who in ("japanese", "latina"):
        ident = pjv_identity(who)
        A = load_arm(ident, Arm(label="approved_INT8_chin100", compose="source", video=CFV / who / "int8_pipelined/aligned_raw.mp4"), "video")
        B = load_arm(ident, Arm(label="TAESD_chin100", compose="source", video=CFV / who / "taesd_pipelined/aligned_raw.mp4"), "video")
        S = load_arm(ident, Arm(label="H3_source", compose="standard", video=ident.source), "video")
        r = syncnet_eval(ident, [A, B, S])
        r["identity"] = who
        out.append(r)
        print(who, [(x["label"], round(x["confidence"], 4), x["offset_frames"], round(x["sim_at_0"], 4)) for x in r["arms"]], flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "syncnet_calibration.json").write_text(json.dumps(clean(dict(results=out)), indent=1) + "\n")


if __name__ == "__main__":
    main()
