#!/usr/bin/env python
"""Standing video A/B validation tool: every optimization round ships video evidence against the
PRE-CHANGE output, repeatable any time.

Concepts
--------
arm   = (code tree, env flags). The PRE-CHANGE arm is the clean main checkout /workspace/MuseTalk at
        its git HEAD with no new flags (what ships today). CANDIDATE arms are the worktree
        /workspace/MuseTalk-perf300 plus flags. A subprocess environment is scrubbed of inherited
        MUSETALK_/HLS_/WEBRTC_/AVATAR_ variables and PYTHONPATH, so an arm is defined only by its
        tree and flags. Bytecode goes to PYTHONPYCACHEPREFIX (nothing is written into a tree).
clip  = one reviewable sequence of pre-encoder frames, stored losslessly with per-frame SHA-256
        (clip.json, schema video_ab_clip_v1):
          chin_japanese, chin_latina  chin recipe renders (TAESD + native encoder + 100% chin +
                                      refined seam) by scripts/video_ab_chin_render.py, 24 fps
          replay_<job>                live scheduler path: scripts/replay_scheduler_exactness.py golden
                                      replay, composed BGR frames of one job, 20 fps. Defaults:
                                      replay_bob_mid (3-pose chinese_bob motion avatar, mid-turn pose
                                      switch) and replay_jp_d10 (standard avatar)
        The same instrument (render script / harness) runs against either tree: the harness is
        executed with its module __file__ placed in the arm's tree (ROOT = that tree), so all of
        scripts.*, musetalk.*, .runtime/ and results/ resolve inside the arm's tree.
round = a named set of candidate arms, e.g. r1_engines, r1_serving.

Outputs (experiments/video_validation/)
---------------------------------------
  baselines/<clip>/<main-HEAD-12>/clip.json (+ frames)   pre-change renders, cached by main's HEAD;
                                                         re-rendered only when main's HEAD (or the
                                                         instrument/clip parameters) change
  <round>/<clip>__<arm>_ab.mp4        column A pre-change | column B candidate | |A-B|x8, full frame
                                      on top, nearest-neighbour 3x mouth zoom below, burned-in labels
                                      (arm, tree, flags, backends, measured fps, gate), clip native
                                      fps (1x speed), libx264 crf<=12, <=20 s, clip audio
  <round>/<clip>__<arm>_ab.json       per-frame SHA equality, PSNR, max/mean abs LSB (full frame and
                                      mouth ROI), summary, gate, verdict, layout and label strings
  <round>/<clip>__<arm>_contact.jpg   contact sheet (evenly spaced + worst frames)
  README.md                           index of all rounds, one-line verdict each (generated between
                                      markers; the rest of the README is hand-written)

Commands
--------
  selftest                   CPU only: synthetic A/B composition + JSON/label/panel checks, lossless
                             round trip, cache keys, replay adapter, harness tree shim, and a real
                             dry-run chin composition on both trees (no CUDA)
  tree-info                  print the pre-change / candidate tree resolution (CPU)
  baseline [--check]         render missing/stale pre-change baselines (GPU unless cached)
  render-arm                 render one candidate arm's clips into <round>/_dumps/<arm>/ (GPU)
  compose-arm                compose A/B videos + JSON for one arm and update README (CPU)
  compose                    compose two arbitrary clip.json dumps (CPU)
  index                      regenerate the README index (CPU)
GPU commands must run under scripts/box_guard.sh; see
docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/gpu_sequence.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parent.parent
MAIN_TREE = Path(os.getenv("VIDEO_AB_MAIN_TREE", "/workspace/MuseTalk"))
OUT_ROOT = Path(os.getenv("VIDEO_AB_OUT_ROOT", str(WORKTREE / "experiments/video_validation")))
DOCS_DIR = WORKTREE / "docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab"
PY = os.getenv("VIDEO_AB_PYTHON", "/workspace/.venvs/musetalk_trt_stagewise/bin/python")
TREE_CACHE = Path(os.getenv("VIDEO_AB_TREE_CACHE", "/workspace/.cache/video_ab/trees"))
PYCACHE_PREFIX = Path(os.getenv("VIDEO_AB_PYCACHE", "/workspace/.cache/video_ab/pycache"))
HARNESS = WORKTREE / "scripts/replay_scheduler_exactness.py"
CHIN_RENDER = WORKTREE / "scripts/video_ab_chin_render.py"
SCHEMA_AB = "video_ab_v1"
SCHEMA_CLIP = "video_ab_clip_v1"
ENV_SCRUB_PREFIXES = ("MUSETALK_", "HLS_", "WEBRTC_", "AVATAR_")
CODE_SUFFIXES = (".py", ".pyx", ".pyi", ".so", ".c", ".cc", ".cpp", ".cu", ".h")
CODE_DIRS = ("scripts/", "musetalk/", "configs/")
SNAPSHOT_PATHS = ("scripts", "musetalk", "configs", "templates", "character_factory/h3_avatar_workflow")
CHIN_IDENTITIES = ("japanese", "latina")
DEFAULT_REPLAY_JOBS = ("bob_mid", "jp_d10")
DEFAULT_CLIPS = ("chin_japanese", "chin_latina", "replay_bob_mid", "replay_jp_d10")
MAX_SECONDS = 20.0
ZOOM = 3
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
README_BEGIN = "<!-- VIDEO_AB_INDEX:BEGIN (generated by scripts/video_ab.py index; do not edit by hand) -->"
README_END = "<!-- VIDEO_AB_INDEX:END -->"
# fp16 (E1) numeric gate, proposed: the user decides on the video (plan D14). exact (E0) is SHA-only.
FP16_GATE = {"full_psnr_min_db": 40.0, "mouth_psnr_min_db": 36.0, "full_mean_abs_lsb_max": 0.5}
GTRACK_PROPOSED = {"mean_px_max": 0.05, "p99_px_max": 0.15}
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


# ============================================================================ small helpers
def sha_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(array) -> str:
    """Identical to replay_scheduler_exactness.sha_array (shape|dtype| prefix + raw bytes)."""
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
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_flags(text: str | None) -> dict:
    flags = {}
    for item in filter(None, (x.strip() for x in (text or "").split(","))):
        if "=" not in item:
            raise SystemExit(f"flags expect K=V items, got {item!r}")
        k, v = item.split("=", 1)
        flags[k.strip()] = v.strip()
    return flags


def flags_str(flags: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in flags.items())


def rel(path: Path, start: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return -1.0


# ============================================================================ trees
def git(tree: Path, *args: str) -> str:
    """Read-only git (--no-optional-locks: never refresh/lock the other tree's index)."""
    return subprocess.run(["git", "--no-optional-locks", "-C", str(tree), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _is_code_path(path: str) -> bool:
    path = path.rstrip("/")
    return path.endswith(CODE_SUFFIXES) or any((path + "/").startswith(d) for d in CODE_DIRS)


def tree_info(tree: Path) -> dict:
    """HEAD, branch and code-affecting dirty paths (tracked changes + untracked) of a git tree."""
    tree = Path(tree)
    info = {"tree": str(tree), "git": False}
    try:
        head = git(tree, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return info
    status = git(tree, "status", "--porcelain", "--untracked-files=normal")
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    dirty_code = sorted(p for p in dirty if _is_code_path(p))
    h = hashlib.sha256()
    if dirty_code:
        h.update(git(tree, "diff", "HEAD", "--", *[p for p in dirty_code if not p.endswith("/")]).encode()
                 if any(not p.endswith("/") for p in dirty_code) else b"")
        for p in dirty_code:
            full = tree / p
            if full.is_file():
                h.update(p.encode())
                h.update(full.read_bytes())
    info.update(git=True, head=head, head12=head[:12], branch=git(tree, "rev-parse", "--abbrev-ref", "HEAD"),
                dirty_paths=len(dirty), dirty_code=dirty_code, dirty_code_digest=h.hexdigest()[:12] if dirty_code else None)
    return info


def tree_label(meta: dict) -> str:
    if not meta.get("git"):
        return f"{meta.get('tree')} (no git)"
    s = f"{meta['tree']} @{meta['head'][:10]}"
    if meta.get("tree_mode") == "snapshot":
        s += " (HEAD snapshot)"
    elif meta.get("dirty_code"):
        s += f" +dirty:{meta['dirty_code_digest']}"
    else:
        s += " (clean)"
    return s


def snapshot_tree(main: Path, head: str, cache: Path = TREE_CACHE) -> Path:
    """Exact HEAD copy of main's code paths (git archive, read-only) + symlinks to everything else.

    Used when main's working copy has code-affecting uncommitted changes (another session edits
    it), so the pre-change arm is still exactly HEAD."""
    dest = cache / head[:12]
    marker = dest / ".video_ab_snapshot.json"
    if marker.exists():
        return dest
    tmp = cache / f".{head[:12]}.tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    top_py = [n for n in git(main, "ls-tree", "--name-only", head).splitlines() if n.endswith(".py")]
    paths = [p for p in SNAPSHOT_PATHS
             if subprocess.run(["git", "-C", str(main), "cat-file", "-e", f"{head}:{p}"], capture_output=True).returncode == 0]
    archive = subprocess.run(["git", "--no-optional-locks", "-C", str(main), "archive", "--format=tar", head, "--",
                              *paths, *top_py], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
    _link_missing(main, tmp)
    dump(tmp / ".video_ab_snapshot.json", {"source": str(main), "head": head, "paths": paths + top_py,
                                           "created_utc": utc()})
    if dest.exists():
        shutil.rmtree(tmp)
        return dest
    tmp.rename(dest)
    return dest


def _link_missing(src: Path, dst: Path) -> None:
    """In every real directory of dst, symlink src's entries that dst lacks (not .git/__pycache__)."""
    for dirpath, dirnames, _ in os.walk(dst):
        d = Path(dirpath)
        s = src / d.relative_to(dst)
        if not s.is_dir():
            continue
        for entry in s.iterdir():
            if entry.name in (".git", "__pycache__"):
                continue
            target = d / entry.name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(entry)
        dirnames[:] = [n for n in dirnames if not (d / n).is_symlink()]


def resolve_pre_tree(mode: str = "auto") -> tuple[Path, dict]:
    """PRE-CHANGE arm tree. auto: live main when its code is clean, else an exact HEAD snapshot."""
    info = tree_info(MAIN_TREE)
    if not info.get("git"):
        raise SystemExit(f"{MAIN_TREE} is not a git tree")
    dirty = bool(info["dirty_code"])
    if mode == "snapshot" or (mode == "auto" and dirty):
        tree = snapshot_tree(MAIN_TREE, info["head"])
        meta = dict(info, tree=str(tree), source_tree=str(MAIN_TREE), tree_mode="snapshot", key=info["head12"])
        return tree, meta
    key = info["head12"] + (f"-dirty{info['dirty_code_digest']}" if dirty else "")
    return MAIN_TREE, dict(info, tree_mode="live", key=key)


def candidate_tree_meta(tree: Path = WORKTREE) -> dict:
    info = tree_info(tree)
    info["tree_mode"] = "live"
    return info


def arm_env(flags: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_SCRUB_PREFIXES) and k != "PYTHONPATH"}
    env.update(flags)
    env["PYTHONPYCACHEPREFIX"] = str(PYCACHE_PREFIX)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ============================================================================ harness (replay) glue
HARNESS_SHIM = r"""
import sys
tree, harness = sys.argv[1], sys.argv[2]
fake = tree.rstrip('/') + '/scripts/replay_scheduler_exactness.py'
sys.argv = [fake] + sys.argv[3:]
sys.path[:] = [p for p in sys.path if p not in ('', '.')]
probe = sys.argv[1:2] == ['--video-ab-probe']
g = {'__name__': 'video_ab_probe' if probe else '__main__', '__file__': fake, '__builtins__': __builtins__}
exec(compile(open(harness).read(), harness, 'exec'), g)
if probe:
    import importlib.util, json
    spec = importlib.util.find_spec('scripts.hls_gpu_scheduler')
    print(json.dumps({'ROOT': str(g['ROOT']), 'LIVE_ENV_FILE': str(g['LIVE_ENV_FILE']),
                      'hls_gpu_scheduler': spec.origin if spec else None, 'sys_path0': sys.path[0]}))
"""


def load_harness_module():
    """Import the harness's tables/compare() (stdlib-only at import time; no model code runs)."""
    spec = importlib.util.spec_from_file_location("video_ab_replay_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harness_command(tree: Path, harness_args: list[str]) -> list[str]:
    return [PY, "-c", HARNESS_SHIM, str(tree), str(HARNESS), *harness_args]


def replay_mouth_box(avatar_id: str, width: int, height: int, results_root: Path) -> list:
    """W//3 x W//4 ROI centred at 72% down the median MuseTalk face bbox of the generation avatar."""
    coords = pickle.load(open(results_root / "results/v15/avatars" / avatar_id / "coords.pkl", "rb"))
    arr = np.array([[float(v) for v in c] for c in coords if c is not None and len(c) == 4])
    x1, y1, x2, y2 = np.median(arr, axis=0)
    w, h = width // 3, width // 4
    cx, cy = (x1 + x2) / 2, y1 + 0.72 * (y2 - y1)
    x0 = int(round(min(max(cx - w / 2, 0), width - w)))
    y0 = int(round(min(max(cy - 0.45 * h, 0), height - h)))
    return [x0, y0, x0 + w, y0 + h]


def generation_avatar(harness, identity: str) -> str:
    spec = harness.IDENTITIES[identity]
    if spec["kind"] == "motion":
        return json.loads(Path(spec["pose_set"]).read_text())["poses"]["speaking_direct"]["avatar_id"]
    return spec["avatar_id"]


def replay_clips_from_golden(golden_path: Path, video_dir: Path, label: str, jobs: list[str], dest_root: Path,
                             arm_meta: dict, results_root: Path = WORKTREE) -> dict:
    """Turn one harness golden run into per-job clip.json dumps; the lossless mkv frames must hash to
    the harness's own pre-encoder frame hashes (sha_array) for every frame the video covers."""
    harness = load_harness_module()
    golden = load_json(golden_path)
    run = next(r for r in golden["runs"] if r["label"] == label)
    if run.get("error"):
        raise RuntimeError(f"harness run {label} failed: {run['error']}")
    jobs_by_id = {j["id"]: j for j in harness.GOLDEN_JOBS}
    out = {}
    for job_id in jobs:
        jr = run["jobs"][job_id]
        mkv = video_dir / f"{label}_{job_id}.mkv"
        w, h = probe_size(mkv)
        decoded = [sha_array(f) for f in iter_frames(mkv, w, h)]
        expected = jr["frames"][:len(decoded)]
        mismatch = [i for i, (a, b) in enumerate(zip(decoded, expected)) if a != b]
        if mismatch or not decoded:
            raise RuntimeError(f"{mkv}: lossless frames do not match harness hashes (first mismatch {mismatch[:1]})")
        job = jobs_by_id[job_id]
        wav = harness.WAVS[job["wav"]]
        dest = dest_root / f"replay_{job_id}"
        dest.mkdir(parents=True, exist_ok=True)
        clip = dict(
            schema=SCHEMA_CLIP, kind="replay", clip=f"replay_{job_id}", job=job, fps=20, native_fps=20,
            frame_count=len(decoded), job_total_frames=len(jr["frames"]), width=w, height=h,
            frames_video=rel(mkv, dest), frames_video_sha256=sha_file(mkv),
            frames_video_codec="libx264rgb -qp 0 (harness LosslessWriter); every frame's SHA equals the harness pre-encoder hash",
            frame_sha256=decoded, frames_digest=digest(decoded),
            audio=None if wav.startswith("@") else wav,
            mouth_box=replay_mouth_box(generation_avatar(harness, job["identity"]), w, h, results_root),
            arm=arm_meta, backends={"vae": golden.get("vae_backend"), "unet": golden.get("unet_backend")},
            measured_fps=run.get("fps"),
            measured_fps_desc=(f"golden replay aggregate: {run.get('frames')} frames of {len(run['jobs'])} concurrent "
                               f"jobs in {run.get('wall_s')} s (unpaced, scheduler+compose, no transport)"),
            golden={"path": rel(golden_path, dest), "label": label, "status": jr.get("status"),
                    "frames_digest": jr.get("frames_digest"), "faces_digest": jr.get("faces_digest"),
                    "yuv_digest": jr.get("yuv_digest"), "order_errors": len(jr.get("order_errors") or []),
                    "yuv_contract_mismatches": len(jr.get("yuv_contract_mismatches") or []),
                    "jobs": sorted(run["jobs"]), "scheduler_sha256": run.get("scheduler_sha256"),
                    "scheduler_path": run.get("scheduler_path"), "harness_sha256": sha_file(HARNESS),
                    "env": golden.get("env")},
            created_utc=utc())
        dump(dest / "clip.json", clip)
        out[f"replay_{job_id}"] = dest / "clip.json"
    return out


# ============================================================================ frame io
def probe_size(path: Path) -> tuple[int, int]:
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height", "-of", "json", str(path)], text=True))["streams"][0]
    return int(s["width"]), int(s["height"])


def iter_frames(path: Path, width: int, height: int, limit: int | None = None):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path), "-vsync", "0",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    count = 0
    try:
        while limit is None or count < limit:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            count += 1
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        if proc.poll() is None:
            proc.kill()  # closed early: kill first so ffmpeg does not report a broken pipe
        proc.stdout.close()
        proc.wait()


def write_lossless(path: Path, frames, fps: int) -> None:
    frames = list(frames)
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264rgb", "-qp", "0",
                          "-preset", "ultrafast", "-threads", "2", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


class Clip:
    """A clip.json dump (frames are read lazily from its lossless video and verified on read)."""

    def __init__(self, clip_json: Path):
        self.path = Path(clip_json).resolve()
        self.meta = load_json(self.path)
        if self.meta.get("schema") != SCHEMA_CLIP:
            raise ValueError(f"{self.path}: not a {SCHEMA_CLIP} dump")
        self.video = (self.path.parent / self.meta["frames_video"]).resolve()
        self.fps = int(self.meta["fps"])
        self.width, self.height = int(self.meta["width"]), int(self.meta["height"])
        self.hashes = self.meta.get("frame_sha256") or []
        self.n = int(self.meta.get("frame_count") or len(self.hashes))

    def frames(self, limit: int | None = None, verify: bool = True):
        if not self.video.exists():
            raise FileNotFoundError(f"{self.video} (frames of {self.path}) is missing")
        for i, f in enumerate(iter_frames(self.video, self.width, self.height, limit)):
            if verify and i < len(self.hashes) and sha_array(f) != self.hashes[i]:
                raise RuntimeError(f"{self.video}: frame {i} does not match its recorded SHA-256")
            yield f

    def landmarks(self):
        name = self.meta.get("generated_landmarks")
        path = self.path.parent / name if name else None
        return np.load(path) if path and path.exists() else None


# ============================================================================ metrics
def region_metrics(d: np.ndarray) -> dict:
    """d = |A-B| uint8 (H,W,3)."""
    mx = int(d.max()) if d.size else 0
    if mx == 0:
        return {"max_abs_lsb": 0, "mean_abs_lsb": 0.0, "mse": 0.0, "psnr_db": None, "diff_pixels": 0}
    f = d.astype(np.float32)
    mse = float(np.mean(f * f))
    return {"max_abs_lsb": mx, "mean_abs_lsb": round(float(f.mean()), 6), "mse": round(mse, 6),
            "psnr_db": round(10.0 * math.log10(255.0 ** 2 / mse), 4), "diff_pixels": int(np.count_nonzero(d.max(axis=2)))}


def frame_metrics(i: int, a: np.ndarray, b: np.ndarray, roi: list, sha_a: str, sha_b: str) -> dict:
    import cv2
    d = cv2.absdiff(a, b)
    x0, y0, x1, y1 = roi
    full, mouth = region_metrics(d), region_metrics(d[y0:y1, x0:x1])
    equal = sha_a == sha_b
    if equal != (full["max_abs_lsb"] == 0):
        raise RuntimeError(f"frame {i}: SHA equality ({equal}) disagrees with pixel diff (max {full['max_abs_lsb']})")
    return {"i": i, "sha_equal": equal, "sha_a": sha_a, "sha_b": sha_b, "full": full, "mouth": mouth}


def summarize(per_frame: list) -> dict:
    def agg(key):
        rows = [p[key] for p in per_frame]
        psnrs = [r["psnr_db"] for r in rows if r["psnr_db"] is not None]
        mse = float(np.mean([r["mse"] for r in rows])) if rows else 0.0
        worst = max(range(len(rows)), key=lambda i: (rows[i]["max_abs_lsb"], rows[i]["mse"])) if rows else None
        return {"psnr_min_db": min(psnrs) if psnrs else None,
                "psnr_median_db": float(np.median(psnrs)) if psnrs else None,
                "psnr_global_db": round(10 * math.log10(255.0 ** 2 / mse), 4) if mse > 0 else None,
                "max_abs_lsb": max((r["max_abs_lsb"] for r in rows), default=0),
                "mean_abs_lsb": round(float(np.mean([r["mean_abs_lsb"] for r in rows])), 6) if rows else 0.0,
                "frames_with_diff": sum(1 for r in rows if r["max_abs_lsb"]), "worst_frame": worst}
    eq = sum(1 for p in per_frame if p["sha_equal"])
    first = next((p["i"] for p in per_frame if not p["sha_equal"]), None)
    return {"frames_compared": len(per_frame), "sha_equal_frames": eq, "all_sha_equal": eq == len(per_frame),
            "first_differing_frame": first, "full": agg("full"), "mouth": agg("mouth")}


def tracking_deviation(la, lb) -> dict | None:
    """G-TRACK-style FaceMesh jaw+lip deviation between the arms' generated landmarks (chin clips)."""
    if la is None or lb is None:
        return None
    n = min(len(la), len(lb))
    idx = JAW + LIPS
    dev = np.linalg.norm(la[:n, idx] - lb[:n, idx], axis=-1)
    return {"frames": n, "points": "jaw(21)+lips(20)", "mean_px": round(float(dev.mean()), 5),
            "p99_px": round(float(np.percentile(dev, 99)), 5), "max_px": round(float(dev.max()), 5),
            "gate_proposed": GTRACK_PROPOSED,
            "within_proposed": bool(dev.mean() <= GTRACK_PROPOSED["mean_px_max"] and np.percentile(dev, 99) <= GTRACK_PROPOSED["p99_px_max"])}


def evaluate_gate(expect: str, summary: dict, counts_equal: bool, thresholds: dict, extra_fail: list) -> dict:
    reasons = list(extra_fail)
    if not counts_equal:
        reasons.append("frame counts differ")
    if expect == "exact":
        if not summary["all_sha_equal"]:
            reasons.append(f"{summary['frames_compared'] - summary['sha_equal_frames']}/{summary['frames_compared']} frames not SHA-identical")
        result = "PASS" if not reasons else "FAIL"
    elif expect == "fp16":
        f, m = summary["full"], summary["mouth"]
        if f["psnr_min_db"] is not None and f["psnr_min_db"] < thresholds["full_psnr_min_db"]:
            reasons.append(f"full min PSNR {f['psnr_min_db']:.2f} < {thresholds['full_psnr_min_db']} dB")
        if m["psnr_min_db"] is not None and m["psnr_min_db"] < thresholds["mouth_psnr_min_db"]:
            reasons.append(f"mouth min PSNR {m['psnr_min_db']:.2f} < {thresholds['mouth_psnr_min_db']} dB")
        if f["mean_abs_lsb"] > thresholds["full_mean_abs_lsb_max"]:
            reasons.append(f"full mean {f['mean_abs_lsb']:.3f} > {thresholds['full_mean_abs_lsb_max']} LSB")
        result = "PASS" if not reasons else "FAIL"
    else:
        result = "REPORT"
    return {"expect": expect, "thresholds": thresholds if expect == "fp16" else {"sha256": "every frame identical"} if expect == "exact" else {},
            "result": result, "reasons": reasons,
            "note": {"exact": "E0: pre-encoder frames SHA-identical (serving/scheduling/memory levers).",
                     "fp16": "E1 numeric gate [proposed]; the video decides (plan D14). Visual review required.",
                     "report": "No gate; numbers reported."}[expect]}


def fmt_db(v) -> str:
    return "inf" if v is None else f"{v:.2f}"


def verdict_line(summary: dict, gate: dict, fps_a, fps_b, golden: dict | None) -> str:
    n = summary["frames_compared"]
    f, m = summary["full"], summary["mouth"]
    speed = ""
    if fps_a and fps_b:
        speed = f"; fps A {fps_a:.1f} -> B {fps_b:.1f} ({(fps_b / fps_a - 1) * 100:+.1f}%)"
    gold = ""
    if golden is not None:
        gold = "; golden replay all jobs identical" if golden.get("identical") else \
            f"; golden replay NOT identical ({golden.get('jobs_differing')} jobs differ)"
    if summary["all_sha_equal"]:
        body = f"{n}/{n} frames SHA-identical (black diff panel)"
    else:
        body = (f"{summary['sha_equal_frames']}/{n} SHA-identical, min PSNR {fmt_db(f['psnr_min_db'])} dB "
                f"(mouth {fmt_db(m['psnr_min_db'])}), max {f['max_abs_lsb']} LSB, mean {f['mean_abs_lsb']:.3f} LSB")
    tag = {"exact": "EXACT", "fp16": "FP16-GATE", "report": "REPORT"}[gate["expect"]]
    tail = f" [{'; '.join(gate['reasons'])}]" if gate["reasons"] else ""
    review = " (visual review required)" if gate["expect"] == "fp16" and gate["result"] == "PASS" else ""
    return f"{tag} {gate['result']}{review}: {body}{gold}{speed}{tail}"


# ============================================================================ rendering (labels, layout)
class Fonts:
    def __init__(self):
        from PIL import ImageFont

        def load(name, size):
            try:
                return ImageFont.truetype(str(FONT_DIR / name), size)
            except OSError:
                return ImageFont.load_default()
        self.title = load("DejaVuSans-Bold.ttf", 17)
        self.body = load("DejaVuSans.ttf", 13)
        self.mono = load("DejaVuSansMono.ttf", 12)
        self.footer = load("DejaVuSansMono.ttf", 13)
        self.caption = load("DejaVuSansMono.ttf", 15)


def text_width(font, text: str) -> float:
    return font.getlength(text) if hasattr(font, "getlength") else len(text) * 7


def wrap_text(text: str, font, width: int, max_lines: int = 4) -> list[str]:
    words, lines, cur = re.split(r"(?<=[ ,])", text), [], ""
    for w in words:
        if text_width(font, cur + w) <= width:
            cur += w
            continue
        if cur:
            lines.append(cur.rstrip())
        cur = w
        while text_width(font, cur) > width:  # hard-split an over-long token
            cut = max(1, int(len(cur) * width / max(1.0, text_width(font, cur))) - 1)
            lines.append(cur[:cut])
            cur = cur[cut:]
    if cur.strip():
        lines.append(cur.rstrip())
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, len(lines[-1]) - 2)] + " …"
    return lines or [""]


def render_block(width: int, lines: list[tuple], bg=(30, 30, 30), pad: int = 6, spacing: int = 3,
                 height: int | None = None) -> tuple[np.ndarray, list[str]]:
    """lines = [(text, font, rgb, wrap_max_lines)], bg is BGR; returns (BGR image, rendered strings)."""
    from PIL import Image, ImageDraw
    laid = []
    for text, font, color, max_lines in lines:
        for s in wrap_text(text, font, width - 2 * pad, max_lines):
            laid.append((s, font, color))
    heights = [font.getbbox("Ag")[3] + spacing for _, font, _ in laid]
    total = pad * 2 + sum(heights)
    img = Image.new("RGB", (width, max(total, height or 0)), bg[::-1])
    draw = ImageDraw.Draw(img)
    y = pad
    for (s, font, color), hgt in zip(laid, heights):
        draw.text((pad, y), s, font=font, fill=tuple(color))
        y += hgt
    return np.asarray(img)[:, :, ::-1].copy(), [s for s, _, _ in laid]


def backend_summary(meta: dict) -> str:
    b = meta.get("backends") or {}
    as_dict = lambda v: v if isinstance(v, dict) else {"name": v}  # noqa: E731
    if "decoder" in b:  # chin render: {decoder: {...}, unet: {...}}
        unet = as_dict(b.get("unet"))
        cg = ",".join(unet.get("cudagraphs_modes") or []) or "-"
        return f"decoder {as_dict(b.get('decoder')).get('name')} | unet {unet.get('name')} (cudagraphs {cg})"
    return f"vae {b.get('vae')} | unet {b.get('unet')}"


def arm_lines(side: str, meta: dict, fonts: Fonts) -> list[tuple]:
    arm = meta.get("arm") or {}
    flags = arm.get("flags") or {}
    color = (140, 190, 255) if side == "A" else (255, 185, 110)
    role = "PRE-CHANGE" if side == "A" else "CANDIDATE"
    fps = meta.get("measured_fps")
    fps_s = f"{fps:.1f} fps" if isinstance(fps, (int, float)) else "n/a"
    desc = meta.get("measured_fps_desc") or ""
    return [(f"{side}  {role}  arm: {arm.get('name', '?')}", fonts.title, color, 1),
            (f"tree: {tree_label(arm)}", fonts.body, (230, 230, 230), 2),
            (f"flags: {flags_str(flags) if flags else '(none)'}", fonts.mono, (230, 230, 150), 4),
            (f"backends: {backend_summary(meta)}", fonts.body, (210, 210, 210), 2),
            (f"measured: {fps_s} - {desc}", fonts.body, (170, 255, 170), 3)]


def diff_lines(clip: str, round_name: str, fps: int, n_video: int, summary: dict, gate: dict, fonts: Fonts) -> list[tuple]:
    f, m = summary["full"], summary["mouth"]
    n = summary["frames_compared"]
    ok = gate["result"] == "PASS"
    gcolor = (120, 255, 120) if ok else (255, 110, 110) if gate["result"] == "FAIL" else (230, 230, 230)
    eq = (f"SHA-identical: {n}/{n} frames (diff panel black)" if summary["all_sha_equal"]
          else f"SHA-identical: {summary['sha_equal_frames']}/{n} frames; first diff at {summary['first_differing_frame']}")
    return [("|A-B| x8  (per channel, clipped)", fonts.title, (230, 230, 230), 1),
            (f"clip: {clip}   round: {round_name}", fonts.body, (230, 230, 230), 1),
            (f"playback: {fps} fps native, 1x speed; {n_video} frames = {n_video / fps:.2f} s", fonts.body, (230, 230, 230), 2),
            (eq, fonts.body, (230, 230, 230), 2),
            (f"full : PSNR min {fmt_db(f['psnr_min_db'])} dB, max {f['max_abs_lsb']} LSB, mean {f['mean_abs_lsb']:.3f}", fonts.mono, (230, 230, 150), 2),
            (f"mouth: PSNR min {fmt_db(m['psnr_min_db'])} dB, max {m['max_abs_lsb']} LSB, mean {m['mean_abs_lsb']:.3f}", fonts.mono, (230, 230, 150), 2),
            (f"gate [{gate['expect']}]: {gate['result']}" + (f" - {'; '.join(gate['reasons'])}" if gate["reasons"] else ""),
             fonts.body, gcolor, 3)]


def build_layout(width: int, height: int, roi: list, header_h: int, fonts: Fonts) -> dict:
    rw, rh = roi[2] - roi[0], roi[3] - roi[1]
    zw, zh = rw * ZOOM, rh * ZOOM
    gap, zlabel = 8, 20
    cw = max(width, zw)
    footer_h = 2 * (fonts.footer.getbbox("Ag")[3] + 3) + 12 + 22
    cols = [gap + c * (cw + gap) for c in range(3)]
    y_full = header_h
    y_zlabel = y_full + height + gap
    y_zoom = y_zlabel + zlabel
    y_footer = y_zoom + zh + gap
    canvas_w = 3 * cw + 4 * gap
    canvas_h = y_footer + footer_h
    canvas_w += canvas_w % 2
    canvas_h += canvas_h % 2
    rect = lambda x, y, w, h: [int(x), int(y), int(w), int(h)]  # noqa: E731
    lay = {"canvas": [canvas_w, canvas_h], "column_width": cw, "gap": gap, "zoom": ZOOM, "roi": list(roi),
           "footer": rect(0, y_footer, canvas_w, canvas_h - y_footer)}
    for c, name in enumerate("abd"):
        lay[f"header_{name}"] = rect(cols[c], 0, cw, header_h)
        lay[f"full_{name}"] = rect(cols[c] + (cw - width) // 2, y_full, width, height)
        lay[f"zlabel_{name}"] = rect(cols[c], y_zlabel, cw, zlabel)
        lay[f"zoom_{name}"] = rect(cols[c] + (cw - zw) // 2, y_zoom, zw, zh)
    lay["timeline"] = rect(gap, canvas_h - 18, canvas_w - 2 * gap, 10)
    return lay


def nn_zoom(img: np.ndarray, roi: list) -> np.ndarray:
    x0, y0, x1, y1 = roi
    crop = img[y0:y1, x0:x1]
    return np.repeat(np.repeat(crop, ZOOM, axis=0), ZOOM, axis=1)


def paste(canvas: np.ndarray, r: list, img: np.ndarray) -> None:
    x, y, w, h = r
    canvas[y:y + h, x:x + w] = img[:h, :w]


class Composer:
    """Builds the A/B canvas for one frame; static parts (headers, zoom labels, timeline) are cached."""

    BG = (18, 18, 18)

    def __init__(self, meta_a: dict, meta_b: dict, clip: str, round_name: str, fps: int, width: int, height: int,
                 roi: list, per_frame: list, summary: dict, gate: dict, n_video: int):
        import cv2
        self.cv2 = cv2
        self.fonts = Fonts()
        self.roi, self.fps, self.per_frame, self.n_video = roi, fps, per_frame, n_video
        self.width, self.height = width, height
        cw = max(width, (roi[2] - roi[0]) * ZOOM)
        tints = {"a": (60, 38, 20), "b": (20, 38, 60), "d": (34, 34, 34)}  # BGR header backgrounds
        blocks, self.labels = [], {}
        for key, lines in (("a", arm_lines("A", meta_a, self.fonts)), ("b", arm_lines("B", meta_b, self.fonts)),
                           ("d", diff_lines(clip, round_name, fps, n_video, summary, gate, self.fonts))):
            img, strings = render_block(cw, lines, bg=tints[key])
            blocks.append((key, img))
            self.labels[key] = strings
        header_h = max(img.shape[0] for _, img in blocks) + 8
        self.layout = build_layout(width, height, roi, header_h, self.fonts)
        W, H = self.layout["canvas"]
        base = np.empty((H, W, 3), np.uint8)
        base[:] = self.BG
        for key, img in blocks:
            x, y, w, _ = self.layout[f"header_{key}"]
            base[y:y + header_h - 4, x:x + w] = tints[key]
            paste(base, [x, y, w, img.shape[0]], img)
        zl = f"mouth ROI x{ZOOM} nearest-neighbour  [{roi[0]},{roi[1]} {roi[2] - roi[0]}x{roi[3] - roi[1]}]"
        for key in "abd":
            img, _ = render_block(cw, [(zl if key != "d" else "|A-B| x8 of the mouth ROI", self.fonts.mono, (200, 200, 200), 1)],
                                  bg=self.BG, pad=3, height=self.layout[f"zlabel_{key}"][3])
            paste(base, self.layout[f"zlabel_{key}"], img)
        tl = self.layout["timeline"]
        strip = np.zeros((tl[3], tl[2], 3), np.uint8)
        n = max(1, len(per_frame))
        for p in per_frame:
            x0 = int(p["i"] * tl[2] / n)
            x1 = max(x0 + 1, int((p["i"] + 1) * tl[2] / n))
            color = (0, 150, 0) if p["sha_equal"] else (0, 170, 230) if p["mouth"]["max_abs_lsb"] == 0 else (40, 40, 220)
            strip[:, x0:x1] = color
        self.timeline = strip
        self.base = base
        self.labels["zoom"] = zl
        self.labels["timeline_legend"] = "green: SHA-identical, amber: differs outside mouth ROI, red: mouth ROI differs"

    def footer_text(self, i: int) -> list[str]:
        p = self.per_frame[i] if i < len(self.per_frame) else None
        head = f"frame {i:4d}/{len(self.per_frame)}  t={i / self.fps:6.3f}s  {self.fps} fps native (1x)"
        if p is None:
            return [head, ""]
        f, m = p["full"], p["mouth"]
        state = "SHA-identical" if p["sha_equal"] else "DIFFERS"
        line2 = (f"{state:13s} full: PSNR {fmt_db(f['psnr_db']):>6s} dB max {f['max_abs_lsb']:3d} mean {f['mean_abs_lsb']:.3f} LSB "
                 f"px {f['diff_pixels']:6d} | mouth: PSNR {fmt_db(m['psnr_db']):>6s} dB max {m['max_abs_lsb']:3d} "
                 f"mean {m['mean_abs_lsb']:.3f} LSB")
        return [head, line2]

    def caption_text(self, i: int) -> list[str]:
        """Compact two-line caption for the half-width contact-sheet tiles."""
        p = self.per_frame[i]
        f, m = p["full"], p["mouth"]
        return [f"frame {i}/{len(self.per_frame)}  t={i / self.fps:.3f}s  {'SHA-identical' if p['sha_equal'] else 'DIFFERS'}",
                f"full {fmt_db(f['psnr_db'])} dB max {f['max_abs_lsb']} | mouth {fmt_db(m['psnr_db'])} dB max {m['max_abs_lsb']}"]

    def frame(self, i: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        canvas = self.base.copy()
        d = np.minimum(cv2.absdiff(a, b).astype(np.uint16) * 8, 255).astype(np.uint8)
        paste(canvas, self.layout["full_a"], a)
        paste(canvas, self.layout["full_b"], b)
        paste(canvas, self.layout["zoom_a"], nn_zoom(a, self.roi))
        paste(canvas, self.layout["zoom_b"], nn_zoom(b, self.roi))
        paste(canvas, self.layout["zoom_d"], nn_zoom(d, self.roi))
        dd = d.copy()
        x0, y0, x1, y1 = self.roi
        cv2.rectangle(dd, (x0, y0), (x1 - 1, y1 - 1), (0, 110, 200), 1)  # ROI outline, diff panel only
        paste(canvas, self.layout["full_d"], dd)
        fx, fy, fw, fh = self.layout["footer"]
        lines = self.footer_text(i)
        img, _ = render_block(fw, [(s, self.fonts.footer, (235, 235, 235), 1) for s in lines], bg=self.BG, pad=4, height=fh - 22)
        paste(canvas, [fx, fy, fw, img.shape[0]], img)
        tl = self.layout["timeline"]
        strip = self.timeline.copy()
        cx = int((i + 0.5) * tl[2] / max(1, len(self.per_frame)))
        strip[:, max(0, cx - 1):cx + 2] = 255
        paste(canvas, tl, strip)
        return canvas


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int, audio: str | None, seconds: float):
        if crf > 12:
            raise ValueError("crf must be <= 12")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio:
            cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
                "-threads", "4", "-t", f"{seconds:.6f}", "-movflags", "+faststart", str(path)]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {' '.join(self.cmd)}")


def probe_video(path: Path) -> dict:
    streams = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,nb_read_frames,avg_frame_rate,width,height,pix_fmt,duration", "-of", "json", str(path)],
        text=True))["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    return {"codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "frames": int(v.get("nb_read_frames", 0)),
            "avg_frame_rate": v.get("avg_frame_rate"), "width": int(v["width"]), "height": int(v["height"]),
            "duration_s": float(v.get("duration") or 0), "audio": any(s["codec_type"] == "audio" for s in streams)}


def contact_sheet(path: Path, tiles: list, composer: Composer, title_lines: list[tuple]) -> dict:
    import cv2
    lay = composer.layout
    y0 = lay["full_a"][1]
    y1 = lay["zoom_a"][1] + lay["zoom_a"][3]
    W = lay["canvas"][0]
    scaled = []
    for i, canvas in tiles:
        body = cv2.resize(canvas[y0:y1], (W // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA)
        cap, _ = render_block(W // 2, [(s, composer.fonts.caption, (240, 240, 240), 1) for s in composer.caption_text(i)],
                              bg=(40, 40, 40), pad=4)
        scaled.append(np.vstack([cap, body]))
    th, tw = scaled[0].shape[:2]
    rows = (len(scaled) + 1) // 2
    title, _ = render_block(tw * 2, title_lines, bg=(30, 30, 30), pad=8)
    sheet = np.empty((title.shape[0] + rows * (th + 6), tw * 2 + 6, 3), np.uint8)
    sheet[:] = 18
    sheet[:title.shape[0], :title.shape[1]] = title
    for k, img in enumerate(scaled):
        r, c = divmod(k, 2)
        y = title.shape[0] + r * (th + 6)
        x = c * (tw + 6)
        sheet[y:y + th, x:x + tw] = img
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"failed to write {path}")
    return {"path": str(path), "frames": [i for i, _ in tiles], "size": [int(sheet.shape[1]), int(sheet.shape[0])]}


def pick_contact_frames(per_frame: list, n_video: int, count: int = 6) -> list[int]:
    picks = [int(round(k * (n_video - 1) / max(1, count - 1))) for k in range(count)] if n_video else []
    if per_frame:
        worst_full = max(per_frame, key=lambda p: (p["full"]["max_abs_lsb"], p["full"]["mse"]))["i"]
        worst_mouth = max(per_frame, key=lambda p: (p["mouth"]["max_abs_lsb"], p["mouth"]["mse"]))["i"]
        for w in (worst_full, worst_mouth):
            if per_frame[w]["full"]["max_abs_lsb"] and w not in picks:
                picks.append(w)
    return sorted(set(picks))


# ============================================================================ compose
def compose(a_path: Path, b_path: Path, out_prefix: Path, round_name: str, clip_name: str, expect: str = "report",
            crf: int = 10, max_seconds: float = MAX_SECONDS, mouth_box: list | None = None, thresholds: dict | None = None,
            audio: str | None = "auto", golden: dict | None = None, extra_fail: list | None = None) -> dict:
    """Compose the A/B review video, JSON and contact sheet for two clip dumps (CPU only)."""
    A, B = Clip(a_path), Clip(b_path)
    if (A.width, A.height) != (B.width, B.height):
        raise RuntimeError(f"frame size mismatch A {A.width}x{A.height} vs B {B.width}x{B.height}")
    if A.fps != B.fps:
        raise RuntimeError(f"fps mismatch A {A.fps} vs B {B.fps}")
    fps = A.fps
    roi = list(mouth_box or A.meta.get("mouth_box") or B.meta.get("mouth_box") or
               [A.width // 3, A.height // 2, A.width // 3 * 2, A.height // 2 + A.width // 4])
    n = min(A.n, B.n)
    n_video = min(n, int(math.floor(max_seconds * fps + 1e-9)))
    thresholds = dict(FP16_GATE, **(thresholds or {}))
    # pass 1: metrics over every frame both clips have
    per_frame = []
    for i, (fa, fb) in enumerate(zip(A.frames(n), B.frames(n))):
        per_frame.append(frame_metrics(i, fa, fb, roi, A.hashes[i] if i < len(A.hashes) else sha_array(fa),
                                       B.hashes[i] if i < len(B.hashes) else sha_array(fb)))
    if len(per_frame) != n:
        raise RuntimeError(f"decoded {len(per_frame)} frames, expected {n}")
    summary = summarize(per_frame)
    counts_equal = A.n == B.n
    gate = evaluate_gate(expect, summary, counts_equal, thresholds, (extra_fail or []) +
                         (["golden replay not identical"] if golden is not None and not golden.get("identical") else []))
    tracking = tracking_deviation(A.landmarks(), B.landmarks())
    verdict = verdict_line(summary, gate, A.meta.get("measured_fps"), B.meta.get("measured_fps"), golden)
    # pass 2: the video (<= max_seconds) + contact frames
    if audio == "auto":
        audio = A.meta.get("audio") if A.meta.get("audio") and Path(A.meta["audio"]).exists() else None
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    video_path = Path(str(out_prefix) + "_ab.mp4")
    json_path = Path(str(out_prefix) + "_ab.json")
    contact_path = Path(str(out_prefix) + "_contact.jpg")
    composer = Composer(A.meta, B.meta, clip_name, round_name, fps, A.width, A.height, roi, per_frame, summary, gate, n_video)
    W, H = composer.layout["canvas"]
    picks = pick_contact_frames(per_frame, n_video)
    tiles = []
    writer = VideoWriter(video_path, W, H, fps, crf, audio, n_video / fps)
    last = max([n_video - 1] + picks)
    try:
        for i, (fa, fb) in enumerate(zip(A.frames(last + 1, verify=False), B.frames(last + 1, verify=False))):
            canvas = composer.frame(i, fa, fb)
            if i < n_video:
                writer.write(canvas)
            if i in picks:
                tiles.append((i, canvas))
    finally:
        writer.close()
    encoded = probe_video(video_path)
    problems = []
    if encoded["frames"] != n_video:
        problems.append(f"encoded {encoded['frames']} frames, expected {n_video}")
    if encoded["avg_frame_rate"] != f"{fps}/1":
        problems.append(f"encoded rate {encoded['avg_frame_rate']} != {fps}/1")
    if encoded["duration_s"] > max_seconds + 0.05:
        problems.append(f"duration {encoded['duration_s']} > {max_seconds}")
    if problems:
        raise RuntimeError(f"{video_path}: {problems}")
    title = [(f"{clip_name}  |  round {round_name}  |  A {(A.meta.get('arm') or {}).get('name')} vs B {(B.meta.get('arm') or {}).get('name')}",
              composer.fonts.title, (240, 240, 240), 1), (verdict, composer.fonts.body, (200, 255, 200) if gate["result"] == "PASS" else (255, 200, 200), 3)]
    sheet = contact_sheet(contact_path, tiles, composer, title)
    arm_view = lambda C: {k: C.meta.get(k) for k in ("arm", "backends", "measured_fps", "measured_fps_desc", "kind",  # noqa: E731
                                                     "frame_count", "frames_digest", "fps_runs", "repeat_outputs_identical",
                                                     "recipe_checks", "golden", "job", "identity", "audio")}
    report = {
        "schema": SCHEMA_AB, "created_utc": utc(), "round": round_name, "clip": clip_name,
        "verdict": verdict, "gate": gate, "summary": summary, "tracking_deviation": tracking,
        "golden_replay": golden,
        "fps": fps, "playback": f"{fps} fps (clip native), 1x speed", "frame_count_a": A.n, "frame_count_b": B.n,
        "frame_counts_equal": counts_equal, "video_frames": n_video, "video_seconds": round(n_video / fps, 4),
        "video_covers_all_frames": n_video == n, "mouth_box": roi,
        "arm_a": arm_view(A), "arm_b": arm_view(B),
        "inputs": {"a": {"clip_json": str(A.path), "sha256": sha_file(A.path), "frames_video_sha256": A.meta.get("frames_video_sha256")},
                   "b": {"clip_json": str(B.path), "sha256": sha_file(B.path), "frames_video_sha256": B.meta.get("frames_video_sha256")}},
        "outputs": {"video": str(video_path), "video_sha256": sha_file(video_path), "json": str(json_path),
                    "contact_sheet": sheet, "encode": dict(encoded, encoder="libx264", crf=crf, preset="medium",
                                                           audio_source=audio)},
        "layout": composer.layout, "labels": composer.labels, "per_frame": per_frame,
        "tool": {"script": str(Path(__file__).resolve()), "sha256": sha_file(Path(__file__).resolve())},
    }
    dump(json_path, report)
    return report


# ============================================================================ baselines / arms
def clip_kind(clip: str) -> str:
    if clip.startswith("chin_") and clip[5:] in CHIN_IDENTITIES:
        return "chin"
    if clip.startswith("replay_"):
        return "replay"
    raise SystemExit(f"unknown clip {clip!r} (chin_<{'|'.join(CHIN_IDENTITIES)}> or replay_<golden job id>)")


def replay_params(args) -> dict:
    return {"jobs": args.replay_jobs, "identities": args.replay_identities, "video_seconds": args.video_seconds}


def chin_params(args) -> dict:
    return {"repeats": args.repeats, "frames": 240}


def baseline_signature(kind: str, key: str, params: dict) -> dict:
    instrument = CHIN_RENDER if kind == "chin" else HARNESS
    return {"schema": SCHEMA_CLIP, "key": key, "kind": kind, "instrument": instrument.name,
            "instrument_sha256": sha_file(instrument), "params": params}


def baseline_dir(clip: str, key: str) -> Path:
    return OUT_ROOT / "baselines" / clip / key


def baseline_state(clip: str, key: str, signature: dict) -> str:
    d = baseline_dir(clip, key)
    if not (d / "baseline.json").exists() or not (d / "clip.json").exists():
        return "missing"
    b = load_json(d / "baseline.json")
    if b.get("signature") != signature:
        return "stale"
    try:
        c = Clip(d / "clip.json")
    except (ValueError, OSError):
        return "missing"
    return "ok" if c.video.exists() else "missing"


def run_logged(cmd: list[str], cwd: Path, env: dict, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[video_ab] $ (cwd={cwd}) {' '.join(cmd[:3])} ... > {log}", flush=True)
    with log.open("a") as f:
        f.write(f"\n=== {utc()} cwd={cwd}\n{cmd}\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def render_chin(tree: Path, tree_meta: dict, arm_name: str, flags: dict, identities: list[str], out_root: Path,
                repeats: int, log: Path) -> dict:
    cmd = [PY, str(CHIN_RENDER), "--repo", str(tree), "--out-root", str(out_root), "--identities", ",".join(identities),
           "--flags", flags_str(flags), "--arm-name", arm_name, "--tree-meta", json.dumps(tree_meta), "--repeats", str(repeats)]
    rc = run_logged(cmd, tree, arm_env({}), log)
    out = {}
    for who in identities:
        cj = out_root / who / "clip.json"
        out[f"chin_{who}"] = cj if cj.exists() else None
    return {"rc": rc, "clips": out}


def render_replay(tree: Path, tree_meta: dict, arm_name: str, flags: dict, params: dict, jobs: list[str],
                  out_dir: Path, log: Path) -> dict:
    label = re.sub(r"[^A-Za-z0-9_]", "_", arm_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    harness_args = ["--mode", "golden", "--run", f"{label}:worktree", "--out", str(out_dir / "golden.json"),
                    "--identities", params["identities"], "--jobs", params["jobs"], "--video-dir", str(out_dir / "videos"),
                    "--video-jobs", ",".join(jobs), "--video-seconds", str(params["video_seconds"])]
    rc = run_logged(harness_command(tree, harness_args), tree, arm_env(flags), log)
    clips = {}
    if (out_dir / "golden.json").exists():
        arm_meta = dict(name=arm_name, tree=str(tree), flags=flags, **{k: v for k, v in tree_meta.items() if k != "tree"})
        try:
            clips = replay_clips_from_golden(out_dir / "golden.json", out_dir / "videos", label, jobs, out_dir.parent, arm_meta,
                                             results_root=tree)
        except Exception as exc:  # noqa: BLE001 - reported as a FAIL line by the caller
            print(f"[video_ab] replay adapter failed: {type(exc).__name__}: {exc}", flush=True)
            rc = rc or 6
    return {"rc": rc, "clips": clips, "label": label}


def cmd_baseline(args) -> int:
    tree, meta = resolve_pre_tree(args.pre_mode)
    key = meta["key"]
    clips = [c for c in args.clips.split(",") if c]
    want = {"chin": [], "replay": []}
    states = {}
    for clip in clips:
        kind = clip_kind(clip)
        sig = baseline_signature(kind, key, chin_params(args) if kind == "chin" else replay_params(args))
        states[clip] = baseline_state(clip, key, sig)
        if states[clip] != "ok" or args.force:
            want[kind].append(clip)
    if want["replay"]:  # the replay clips of one request share one harness run: render them together
        want["replay"] = [c for c in clips if clip_kind(c) == "replay"]
    print(f"[video_ab] pre-change tree {tree_label(meta)} key={key} states={states}", flush=True)
    if args.check:
        missing = [c for c, s in states.items() if s != "ok"]
        print(("PASS" if not missing else "MISSING") + f" baselines key={key} missing={missing}", flush=True)
        return 0 if not missing else 3
    rc = 0
    stage = OUT_ROOT / "baselines" / f"_staging_{key}"
    if want["chin"]:
        ids = [c[5:] for c in want["chin"]]
        res = render_chin(tree, meta, "pre_change", {}, ids, stage / "chin", args.repeats, stage / "chin_render.log")
        for clip in want["chin"]:
            src = res["clips"].get(clip)
            if src is None:
                print(f"FAIL baseline {clip}: render rc={res['rc']} (log {stage / 'chin_render.log'})", flush=True)
                rc = rc or 1
                continue
            dest = baseline_dir(clip, key)
            c = install_clip(src, dest, move_dir=True)
            dump(dest / "baseline.json", {"signature": baseline_signature("chin", key, chin_params(args)), "tree": meta,
                                          "created_utc": utc(), "render_rc": res["rc"]})
            problems = chin_render_problems(c)
            rc = rc or (1 if problems else 0)
            print(f"{'FAIL' if problems else 'PASS'} baseline {clip} key={key} frames={c['frame_count']} "
                  f"fps={c['measured_fps']:.1f} repeats_identical={c.get('repeat_outputs_identical')} "
                  f"checks={(c.get('recipe_checks') or {}).get('passes')} {problems or ''}-> {dest}", flush=True)
    if want["replay"]:
        params = replay_params(args)
        jobs = [c[len("replay_"):] for c in want["replay"]]
        # one harness output dir per (key, recorded job set): clips of other requests keep their frames
        gdir = OUT_ROOT / "baselines" / "replay_golden" / key / "+".join(sorted(jobs))
        shutil.rmtree(gdir, ignore_errors=True)
        log = gdir.parent / f"{'+'.join(sorted(jobs))}.log"
        res = render_replay(tree, meta, "pre_change", {}, params, jobs, gdir, log)
        for clip in want["replay"]:
            src = res["clips"].get(clip)
            if src is None:
                print(f"FAIL baseline {clip}: harness rc={res['rc']} (log {log})", flush=True)
                rc = rc or 1
                continue
            dest = baseline_dir(clip, key)
            meta_c = install_clip(src, dest, move_dir=False)
            dump(dest / "baseline.json", {"signature": baseline_signature("replay", key, params), "tree": meta,
                                          "created_utc": utc(), "render_rc": res["rc"]})
            print(f"PASS baseline {clip} key={key} frames={meta_c['frame_count']}/{meta_c['job_total_frames']} "
                  f"golden_fps={meta_c['measured_fps']} -> {dest}", flush=True)
    shutil.rmtree(stage, ignore_errors=True) if rc == 0 else None
    if args.report:
        rendered = {}
        for clip in clips:
            cj = baseline_dir(clip, key) / "clip.json"
            if cj.exists():
                c = load_json(cj)
                rendered[clip] = {"frames": c.get("frame_count"), "fps": c.get("fps"), "measured_fps": c.get("measured_fps"),
                                  "measured_fps_desc": c.get("measured_fps_desc"), "frames_digest": c.get("frames_digest"),
                                  "repeat_outputs_identical": c.get("repeat_outputs_identical"),
                                  "recipe_checks": {k: v for k, v in (c.get("recipe_checks") or {}).items() if k != "rows"},
                                  "backends": c.get("backends"), "clip_json": str(cj)}
        dump(Path(args.report), {"step": "baseline", "key": key, "tree": meta, "states_before": states,
                                 "rendered_now": want, "baselines": rendered, "result": "PASS" if rc == 0 else "FAIL",
                                 "created_utc": utc()})
    return rc


def install_clip(src: Path, dest: Path, move_dir: bool) -> dict:
    """Place a rendered clip dump at dest. move_dir: move its whole directory (chin: frames live inside it);
    else rewrite its relative references (replay: frames stay in the shared harness output dir)."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if move_dir:
        shutil.move(str(src.parent), str(dest))
        return load_json(dest / "clip.json")
    meta = load_json(src)
    meta["frames_video"] = rel(src.parent / meta["frames_video"], dest)
    if meta.get("golden"):
        meta["golden"]["path"] = rel(src.parent / meta["golden"]["path"], dest)
    dump(dest / "clip.json", meta)
    shutil.rmtree(src.parent, ignore_errors=True)
    return meta


def dumps_dir(round_name: str, arm: str) -> Path:
    return OUT_ROOT / round_name / "_dumps" / arm


def cmd_render_arm(args) -> int:
    flags = parse_flags(args.flags)
    tree = Path(args.tree).resolve()
    meta = candidate_tree_meta(tree)
    _, pre = resolve_pre_tree(args.pre_mode)
    clips = [c for c in args.clips.split(",") if c]
    ddir = dumps_dir(args.round, args.arm)
    ddir.mkdir(parents=True, exist_ok=True)
    manifest = {"round": args.round, "arm": args.arm, "tree": meta, "flags": flags, "clips": {}, "baseline_key": pre["key"],
                "chin_params": chin_params(args), "replay_params": replay_params(args), "started_utc": utc(),
                "mem_available_gb_start": round(mem_available_gb(), 2)}
    rc = 0
    chin = [c for c in clips if clip_kind(c) == "chin"]
    replay = [c for c in clips if clip_kind(c) == "replay"]
    if chin:
        res = render_chin(tree, meta, args.arm, flags, [c[5:] for c in chin], ddir / "_chin", args.repeats, ddir / "chin_render.log")
        for clip in chin:
            src = res["clips"].get(clip)
            if src is None:
                manifest["clips"][clip] = {"status": "failed", "rc": res["rc"]}
                print(f"FAIL render {args.round}/{args.arm} {clip}: rc={res['rc']} (log {ddir / 'chin_render.log'})", flush=True)
                rc = rc or 1
                continue
            dest = ddir / clip
            c = install_clip(src, dest, move_dir=True)
            problems = chin_render_problems(c)
            manifest["clips"][clip] = {"status": "ok", "clip_json": str(dest / "clip.json"), "measured_fps": c["measured_fps"],
                                       "render_problems": problems}
            rc = rc or (1 if problems else 0)
            print(f"{'FAIL' if problems else 'PASS'} render {args.round}/{args.arm} {clip}: {c['frame_count']} frames, "
                  f"{c['measured_fps']:.1f} fps, backends {backend_summary(c)} {problems or ''}", flush=True)
        shutil.rmtree(ddir / "_chin", ignore_errors=True)
    if replay:
        gdir = ddir / "replay_golden"
        shutil.rmtree(gdir, ignore_errors=True)
        res = render_replay(tree, meta, args.arm, flags, replay_params(args), [c[len("replay_"):] for c in replay], gdir,
                            ddir / "replay.log")
        for clip in replay:
            src = res["clips"].get(clip)
            if src is None:
                manifest["clips"][clip] = {"status": "failed", "rc": res["rc"]}
                print(f"FAIL render {args.round}/{args.arm} {clip}: harness rc={res['rc']} (log {ddir / 'replay.log'})", flush=True)
                rc = rc or 1
                continue
            c = load_json(src)
            manifest["clips"][clip] = {"status": "ok", "clip_json": str(src), "measured_fps": c["measured_fps"]}
            print(f"PASS render {args.round}/{args.arm} {clip}: {c['frame_count']}/{c['job_total_frames']} frames, "
                  f"golden {c['measured_fps']} fps, backends {backend_summary(c)}", flush=True)
    manifest["finished_utc"] = utc()
    manifest["result"] = "PASS" if rc == 0 else "FAIL"
    dump(ddir / "render_arm.json", manifest)
    if args.report:
        dump(Path(args.report), dict(manifest, step="render-arm"))
    return rc


def golden_compare(a_meta: dict, a_path: Path, b_meta: dict, b_path: Path) -> dict | None:
    ga, gb = a_meta.get("golden"), b_meta.get("golden")
    if not ga or not gb:
        return None
    harness = load_harness_module()
    ja, jb = load_json(a_path.parent / ga["path"]), load_json(b_path.parent / gb["path"])
    ra = next(r for r in ja["runs"] if r["label"] == ga["label"])
    rb = next(r for r in jb["runs"] if r["label"] == gb["label"])
    rep = harness.compare(ra, rb, ra["label"], rb["label"])
    return {"identical": bool(rep["identical"]), "frames_total": rep.get("frames_total"),
            "jobs_differing": sum(1 for j in rep["jobs"].values() if not j.get("identical")),
            "jobs": rep["jobs"], "same_job_set": sorted(ra["jobs"]) == sorted(rb["jobs"]),
            "fps_a": ra.get("fps"), "fps_b": rb.get("fps"),
            "scope": "all golden jobs: faces (TAESD uint8), composed BGR frames, PyAV yuv420p, order, status"}


def cmd_compose_arm(args) -> int:
    ddir = dumps_dir(args.round, args.arm)
    manifest = load_json(ddir / "render_arm.json")
    key = manifest["baseline_key"]
    clips = [c for c in (args.clips or ",".join(manifest["clips"])).split(",") if c]
    report = {"round": args.round, "arm": args.arm, "flags": manifest["flags"], "tree": manifest["tree"], "baseline_key": key,
              "expect": args.expect, "clips": {}, "created_utc": utc()}
    rc = 0
    for clip in clips:
        entry = manifest["clips"].get(clip, {})
        a = baseline_dir(clip, key) / "clip.json"
        if entry.get("status") != "ok" or not a.exists():
            why = "candidate render failed" if entry.get("status") != "ok" else f"baseline {a} missing"
            report["clips"][clip] = {"result": "FAIL", "verdict": why}
            print(f"FAIL video_ab {args.round}/{args.arm} {clip}: {why}", flush=True)
            rc = 1
            continue
        b = Path(entry["clip_json"])
        am, bm = load_json(a), load_json(b)
        golden = golden_compare(am, a, bm, b)
        extra = []
        if golden is not None and not golden["same_job_set"]:
            extra.append("golden job sets differ between arms")
        if bm.get("kind") == "replay":
            extra += replay_backend_problems(manifest["flags"], bm.get("backends") or {})
        extra += [f"baseline: {x}" for x in chin_render_problems(am)] + [f"candidate: {x}" for x in chin_render_problems(bm)]
        prefix = OUT_ROOT / args.round / f"{clip}__{args.arm}"
        rep = compose(a, b, prefix, args.round, f"{clip}__{args.arm}", expect=args.expect, crf=args.crf,
                      max_seconds=args.max_seconds, golden=golden, extra_fail=extra)
        report["clips"][clip] = {"result": rep["gate"]["result"], "verdict": rep["verdict"], "video": rep["outputs"]["video"],
                                 "json": rep["outputs"]["json"], "contact": rep["outputs"]["contact_sheet"]["path"],
                                 "summary": {k: rep["summary"][k] for k in ("frames_compared", "sha_equal_frames", "all_sha_equal")},
                                 "full": rep["summary"]["full"], "mouth": rep["summary"]["mouth"],
                                 "fps_a": am.get("measured_fps"), "fps_b": bm.get("measured_fps"),
                                 "tracking_deviation": rep["tracking_deviation"]}
        ok = rep["gate"]["result"] in ("PASS", "REPORT")
        rc = rc or (0 if ok else 1)
        print(f"{'PASS' if ok else 'FAIL'} video_ab {args.round}/{args.arm} {clip}: {rep['verdict']} -> {rep['outputs']['video']}", flush=True)
        if not args.keep_dumps and ok:
            drop_candidate_frames(b)
    write_index()
    out = Path(args.report) if args.report else DOCS_DIR / f"{args.round}__{args.arm}.json"
    report["all_pass"] = rc == 0
    dump(out, report)
    print(f"[video_ab] report -> {out}", flush=True)
    return rc


def chin_render_problems(meta: dict) -> list[str]:
    """Render-level failures of a chin clip dump (empty for other kinds)."""
    if meta.get("kind") != "chin_render":
        return []
    out = []
    if not meta.get("repeat_outputs_identical", True):
        out.append("chin repeats not SHA-identical (non-deterministic render)")
    checks = meta.get("recipe_checks") or {}
    if not checks.get("skipped") and not checks.get("passes", False):
        out.append(f"recipe checks failed (lip {checks.get('protected_lip_max_rgb_difference')}, "
                   f"reference {checks.get('reference_max_rgb_difference')}, jacobian {checks.get('minimum_jacobian')})")
    if (meta.get("backends") or {}).get("expectation_problems"):
        out.append(f"backend expectation: {meta['backends']['expectation_problems']}")
    return out


def replay_backend_problems(flags: dict, backends: dict) -> list[str]:
    """A requested decoder/UNet backend that the harness's manager did not activate (silent fallback)."""
    problems = []
    vae, unet = str(backends.get("vae")), str(backends.get("unet"))
    if flags.get("MUSETALK_TAESD_BACKEND", "").lower() == "trt" and vae != "taesd_trt":
        problems.append(f"MUSETALK_TAESD_BACKEND=trt but the replay VAE backend is {vae}")
    if flags.get("MUSETALK_UNET_BACKEND", "").lower() in ("trt_stagewise", "tensorrt_stagewise") and "stagewise" not in unet:
        problems.append(f"MUSETALK_UNET_BACKEND=trt_stagewise but the replay UNet backend is {unet}")
    return problems


def drop_candidate_frames(clip_json: Path) -> None:
    """Delete a candidate's lossless frames after a successful compose (frame SHAs stay in clip.json)."""
    meta = load_json(clip_json)
    video = (clip_json.parent / meta["frames_video"]).resolve()
    if OUT_ROOT.resolve() / "baselines" in video.parents:
        return
    if video.exists():
        video.unlink()
    meta["frames_video_deleted_utc"] = utc()
    dump(clip_json, meta)


def cmd_compose(args) -> int:
    rep = compose(Path(args.a), Path(args.b), Path(args.out_prefix), args.round, args.clip, expect=args.expect, crf=args.crf,
                  max_seconds=args.max_seconds, mouth_box=[int(v) for v in args.mouth_box.split(",")] if args.mouth_box else None)
    ok = rep["gate"]["result"] in ("PASS", "REPORT")
    print(f"{'PASS' if ok else 'FAIL'} video_ab {args.round} {args.clip}: {rep['verdict']} -> {rep['outputs']['video']}")
    if args.index:
        write_index()
    return 0 if ok else 1


# ============================================================================ README index
def write_index(root: Path | None = None) -> Path:
    root = Path(root or OUT_ROOT)  # resolved at call time (tests swap OUT_ROOT)
    readme = root / "README.md"
    lines = [README_BEGIN, "", f"_Index regenerated {utc()}._", ""]
    rounds = sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("baselines",) and not p.name.startswith(("_", ".")))\
        if root.exists() else []
    if not rounds:
        lines += ["No rounds recorded yet.", ""]
    for rd in rounds:
        reports = sorted(rd.glob("*_ab.json"))
        external = sorted(v for v in rd.glob("*_ab.mp4") if not v.with_suffix(".json").exists())
        if not reports and not external:
            continue
        lines += [f"### {rd.name}", "", "| Clip | Arm | Result | One-line verdict | Files |", "|---|---|---|---|---|"]
        for v in external:  # videos another tool wrote here without a video_ab JSON
            lines.append(f"| {v.name[:-len('_ab.mp4')]} | - | n/a | not produced by video_ab.py (no per-frame JSON); "
                         f"see the producing area's notes | [video]({rd.name}/{v.name}) |")
        for rp in reports:
            try:
                r = load_json(rp)
            except (OSError, ValueError):
                continue
            base = rp.name[: -len("_ab.json")]
            if r.get("schema") != SCHEMA_AB:
                lines.append(f"| {base} | - | n/a | JSON not written by video_ab.py (schema {r.get('schema')!r}) | "
                             f"[json]({rd.name}/{rp.name}) |")
                continue
            clip, _, arm = base.partition("__")
            files = f"[video]({rd.name}/{base}_ab.mp4) · [json]({rd.name}/{base}_ab.json) · [contact]({rd.name}/{base}_contact.jpg)"
            lines.append(f"| {clip} | {arm or '-'} | {r['gate']['result']} | {r['verdict'].replace('|', '/')} | {files} |")
        lines.append("")
    bdir = root / "baselines"
    rows = []
    if bdir.exists():
        for bj in sorted(bdir.glob("*/*/baseline.json")):
            clip_dir = bj.parent
            try:
                c = load_json(clip_dir / "clip.json")
                b = load_json(bj)
            except (OSError, ValueError):
                continue
            fps = c.get("measured_fps")
            rows.append(f"| {clip_dir.parent.name} | {clip_dir.name} | {tree_label(b.get('tree') or {})} | {c.get('frame_count')} "
                        f"@ {c.get('fps')} fps | {fps if fps is None else round(fps, 1)} | {b.get('created_utc')} |")
    lines += ["### Pre-change baselines (cache)", ""]
    lines += (["| Clip | Key (main HEAD) | Tree | Frames | Measured fps | Rendered |", "|---|---|---|---|---|---|"] + rows
              if rows else ["None rendered yet."])
    lines += ["", README_END]
    block = "\n".join(lines)
    text = readme.read_text() if readme.exists() else f"# Video A/B validation\n\n{README_BEGIN}\n{README_END}\n"
    if README_BEGIN in text and README_END in text:
        pre, rest = text.split(README_BEGIN, 1)
        _, post = rest.split(README_END, 1)
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n## Index\n\n" + block + "\n"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(text)
    return readme


# ============================================================================ selftest (CPU)
def _synthetic(n: int, w: int, h: int, seed: int) -> list[np.ndarray]:
    import cv2
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    frames = []
    for i in range(n):
        f = base.copy()
        cv2.circle(f, (int(w * (0.3 + 0.4 * i / max(1, n - 1))), h // 3), max(4, w // 12), (40, 200, 240), -1)
        cv2.ellipse(f, (w // 2, int(h * 0.62)), (w // 8, 4 + (i % 7) * 2), 0, 0, 360, (30, 30, 160), -1)
        frames.append(f)
    return frames


def _write_synthetic_clip(dest: Path, frames, fps: int, name: str, roi, extra=None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    write_lossless(dest / "frames.mkv", frames, fps)
    hashes = [sha_array(f) for f in frames]
    h, w = frames[0].shape[:2]
    meta = dict(schema=SCHEMA_CLIP, kind="synthetic", clip="synthetic", fps=fps, native_fps=fps, frame_count=len(frames),
                width=w, height=h, frames_video="frames.mkv", frames_video_sha256=sha_file(dest / "frames.mkv"),
                frame_sha256=hashes, frames_digest=digest(hashes), mouth_box=list(roi), audio=None,
                arm={"name": name, "tree": "/synthetic/" + name, "flags": {"MUSETALK_EXAMPLE_FLAG": "1"} if name == "cand" else {},
                     "git": False}, backends={"vae": "synthetic", "unet": "synthetic"},
                measured_fps=123.4 if name == "pre" else 150.0, measured_fps_desc="synthetic")
    meta.update(extra or {})
    dump(dest / "clip.json", meta)
    return dest / "clip.json"


MOCK_GPU_LOOP = r"""
# CPU mock of video_ab_chin_render.render_gpu: fake UNet/TAESD/tracker, CUDA hidden (CUDA_VISIBLE_DEVICES='').
import importlib.util, json, sys
from pathlib import Path
render_path, repo, out, n = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location('video_ab_chin_render_mock', render_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sys.path[:0] = [str(repo), str(repo / 'scripts')]
mod.apply_env_layers(repo, {})
import numpy as np, torch
_tensor = torch.tensor
torch.tensor = lambda *a, **k: _tensor(*a, **{x: y for x, y in k.items() if x != 'device'})
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.synchronize = lambda *a, **k: None
chin, _ = mod.load_chin(repo)
class Out:
    def __init__(self, s): self.sample = s
class FakeUnet:
    name = 'fake_unet'
    def __call__(self, x, t, encoder_hidden_states=None): return Out(x[:, :4] * 0.5 + encoder_hidden_states.mean() * 0)
class FakeDecoder:
    name = 'fake_decoder'
    def decode(self, z, sf, dtype):
        img = torch.nn.functional.interpolate(z[:, :3].float(), size=(256, 256), mode='bilinear', align_corners=False)
        return torch.sigmoid(img).to(dtype)
saved = np.load(mod.identity_paths('japanese', mod.DATA_ROOT)['saved_generated_landmarks']).astype(np.float32)
class FakeTracker:
    def reset(self): self.i = 0
    def track(self, frame, face, box):
        g = saved[self.i]; self.i += 1; return g.copy(), 0.001
paths, frames, d, kept, records, prep = mod.render_gpu(chin, repo, 'japanese', mod.DATA_ROOT, FakeTracker(), FakeUnet(),
                                                       FakeDecoder(), 0.18215, 2, n)
checks = mod.recipe_checks(chin, frames, d, kept['faces'], kept['outputs'])
stored = mod.store_clip(out, kept['outputs'], 24)
print(json.dumps({'frames': len(kept['outputs']), 'repeats': len(records),
                  'repeats_identical': len({r['frames_digest'] for r in records}) == 1,
                  'landmarks_restored': bool(np.array_equal(d['g'], saved[:n])), 'checks': checks['passes'],
                  'lip_max': checks['protected_lip_max_rgb_difference'], 'ref_max': checks['reference_max_rgb_difference'],
                  'unet': mod.describe_unet(FakeUnet())['name'], 'cuda_initialized': torch.cuda.is_initialized(),
                  'digest': stored['frames_digest'][:12]}))
"""


def _fake_golden(label: str, job: str, frames: list, fps: float) -> dict:
    h = [sha_array(f) for f in frames]
    return {"runs": [{"label": label, "fps": fps, "frames": len(frames), "wall_s": round(len(frames) / fps, 3), "jobs": {job: {
        "frames": h, "yuv": [x[::-1] for x in h], "faces": h[:4], "order_errors": [], "status": "completed",
        "raw_no_gpu_faces": 0, "yuv_contract_mismatches": [], "frames_digest": digest(h)}}}],
        "vae_backend": "taesd", "unet_backend": "tensorrt_unet_multi"}


def _selftest_flow(root: Path) -> tuple[bool, dict]:
    global OUT_ROOT
    saved = OUT_ROOT
    shutil.rmtree(root, ignore_errors=True)
    OUT_ROOT = root / "vv"
    try:
        key, roi = "testkey", [171, 500, 341, 628]
        frames = _synthetic(24, 512, 896, 7)
        lm = np.random.default_rng(0).uniform(100, 400, (24, 478, 2)).astype(np.float32)
        # pre-change baselines
        src = _write_synthetic_clip(root / "stage/japanese", frames, 24, "pre", roi, {"kind": "chin_render",
                                    "generated_landmarks": "generated_landmarks.npy", "audio": None,
                                    "recipe_checks": {"passes": True}, "repeat_outputs_identical": True})
        np.save(root / "stage/japanese/generated_landmarks.npy", lm)
        install_clip(src, baseline_dir("chin_japanese", key), move_dir=True)
        sig = baseline_signature("chin", key, {"repeats": 3, "frames": 240})
        dump(baseline_dir("chin_japanese", key) / "baseline.json", {"signature": sig, "tree": {"tree": "/fake", "git": False}})
        states = (baseline_state("chin_japanese", key, sig),
                  baseline_state("chin_japanese", key, baseline_signature("chin", key, {"repeats": 1, "frames": 240})),
                  baseline_state("chin_latina", key, sig))
        rframes = _synthetic(20, 512, 832, 9)
        gdir = OUT_ROOT / "baselines/replay_golden" / key / "jp_d10"
        write_lossless(gdir / "videos/pre_change_jp_d10.mkv", rframes, 20)
        dump(gdir / "golden.json", _fake_golden("pre_change", "jp_d10", rframes, 200.0))
        made = replay_clips_from_golden(gdir / "golden.json", gdir / "videos", "pre_change", ["jp_d10"], gdir.parent,
                                        {"name": "pre_change", "tree": "/fake", "flags": {}})
        install_clip(made["replay_jp_d10"], baseline_dir("replay_jp_d10", key), move_dir=False)
        dump(baseline_dir("replay_jp_d10", key) / "baseline.json", {"signature": {}, "tree": {"tree": "/fake", "git": False}})
        # candidate arm dumps (identical frames, landmarks shifted by 0.01 px)
        ddir = dumps_dir("rtest", "armx")
        c1 = _write_synthetic_clip(ddir / "chin_japanese", frames, 24, "cand", roi, {"kind": "chin_render",
                                   "generated_landmarks": "generated_landmarks.npy", "audio": None,
                                    "recipe_checks": {"passes": True}, "repeat_outputs_identical": True})
        np.save(ddir / "chin_japanese/generated_landmarks.npy", lm + 0.01)
        gc = ddir / "replay_golden"
        write_lossless(gc / "videos/armx_jp_d10.mkv", rframes, 20)
        dump(gc / "golden.json", _fake_golden("armx", "jp_d10", rframes, 230.0))
        c2 = replay_clips_from_golden(gc / "golden.json", gc / "videos", "armx", ["jp_d10"], ddir,
                                      {"name": "armx", "tree": str(WORKTREE), "flags": {"HLS_X": "1"}})["replay_jp_d10"]
        dump(ddir / "render_arm.json", {"baseline_key": key, "flags": {"HLS_X": "1"}, "tree": {"tree": str(WORKTREE)},
                                        "clips": {"chin_japanese": {"status": "ok", "clip_json": str(c1)},
                                                  "replay_jp_d10": {"status": "ok", "clip_json": str(c2)}}})
        rc = cmd_compose_arm(argparse.Namespace(round="rtest", arm="armx", clips=None, expect="exact", crf=10,
                                                max_seconds=MAX_SECONDS, keep_dumps=False, report=str(root / "report.json")))
        rep = load_json(root / "report.json")
        rj = load_json(OUT_ROOT / "rtest/replay_jp_d10__armx_ab.json")
        cj = load_json(OUT_ROOT / "rtest/chin_japanese__armx_ab.json")
        readme = (OUT_ROOT / "README.md").read_text()
        detail = {"rc": rc, "states": states, "results": {k: v["result"] for k, v in rep["clips"].items()},
                  "golden_identical": (rj.get("golden_replay") or {}).get("identical"),
                  "tracking_mean_px": (cj.get("tracking_deviation") or {}).get("mean_px"),
                  "candidate_frames_deleted": not (ddir / "chin_japanese/frames.mkv").exists(),
                  "baseline_frames_kept": (baseline_dir("chin_japanese", key) / "frames.mkv").exists()
                  and (gdir / "videos/pre_change_jp_d10.mkv").exists(),
                  "readme_rows": readme.count("| armx | PASS |")}
        ok = (rc == 0 and states == ("ok", "stale", "missing") and set(detail["results"].values()) == {"PASS"}
              and detail["golden_identical"] is True and abs((detail["tracking_mean_px"] or 0) - 0.01414) < 1e-3
              and detail["candidate_frames_deleted"] and detail["baseline_frames_kept"] and detail["readme_rows"] == 2)
        return ok, detail
    except Exception as exc:  # noqa: BLE001 - reported as a failed test
        import traceback
        return False, {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-1200:]}
    finally:
        OUT_ROOT = saved


def cmd_selftest(args) -> int:
    import cv2
    work = Path(args.work or tempfile.mkdtemp(prefix="video_ab_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    results = []

    def check(name, cond, detail=""):
        results.append({"test": name, "result": "pass" if cond else "fail", "detail": str(detail)})
        print(f"{'PASS' if cond else 'FAIL'} selftest {name}: {detail}", flush=True)
        return cond

    t0 = time.time()
    W, H, FPS_ = 512, 896, 24
    roi = [171, 500, 341, 628]
    A = _synthetic(36, W, H, 1)
    B = [f.copy() for f in A]
    changed = {}
    for i in (20, 21, 22):  # +-k LSB inside the mouth ROI
        noise = np.zeros_like(B[i], dtype=np.int16)
        noise[roi[1] + 10:roi[1] + 60, roi[0] + 20:roi[0] + 120] = (i - 18)
        B[i] = np.clip(B[i].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        changed[i] = i - 18
    B[30][50:90, 40:100] = 255 - B[30][50:90, 40:100]  # a large change outside the mouth
    changed[30] = None
    ca = _write_synthetic_clip(work / "a", A, FPS_, "pre", roi)
    cb = _write_synthetic_clip(work / "b", B, FPS_, "cand", roi)
    decoded = [sha_array(f) for f in iter_frames(work / "a/frames.mkv", W, H)]
    check("lossless_roundtrip", decoded == [sha_array(f) for f in A], f"{len(decoded)} frames libx264rgb qp0 decoded SHA-exact")

    rep = compose(ca, cb, work / "out/synth", "selftest", "synthetic", expect="exact", crf=10)
    pf = rep["per_frame"]
    check("per_frame_sha_equality", [p["sha_equal"] for p in pf] == [i not in changed for i in range(36)],
          f"{sum(p['sha_equal'] for p in pf)}/36 equal; differing {sorted(changed)}")
    ok_lsb = all(pf[i]["mouth"]["max_abs_lsb"] == k and pf[i]["full"]["max_abs_lsb"] == k for i, k in changed.items() if k)
    ref = cv2.absdiff(A[21], B[21]).astype(np.float64)
    psnr_ref = 10 * math.log10(255 ** 2 / np.mean(ref ** 2))
    check("metrics_lsb_psnr", ok_lsb and abs(pf[21]["full"]["psnr_db"] - psnr_ref) < 1e-3 and pf[30]["mouth"]["max_abs_lsb"] == 0,
          f"frame21 max {pf[21]['full']['max_abs_lsb']} psnr {pf[21]['full']['psnr_db']:.3f} (ref {psnr_ref:.3f}); frame30 mouth max 0")
    check("gate_exact_fails_on_diff", rep["gate"]["result"] == "FAIL" and rep["verdict"].startswith("EXACT FAIL"), rep["verdict"])
    enc = rep["outputs"]["encode"]
    check("video_encode", enc["frames"] == 36 and enc["avg_frame_rate"] == "24/1" and enc["codec"] == "h264" and enc["crf"] <= 12
          and enc["duration_s"] <= MAX_SECONDS, enc)
    lay = rep["layout"]
    # decode composed video; compare panels with the source frames (lossy crf10/yuv420p -> PSNR bound)
    Wc, Hc = lay["canvas"]
    out_frames = list(iter_frames(Path(rep["outputs"]["video"]), Wc, Hc))

    def crop(img, r):
        x, y, w, h = r
        return img[y:y + h, x:x + w]

    def psnr(x, y):
        m = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
        return 99.0 if m == 0 else 10 * math.log10(255 ** 2 / m)
    p_a = psnr(crop(out_frames[21], lay["full_a"]), A[21])
    p_b = psnr(crop(out_frames[21], lay["full_b"]), B[21])
    p_z = psnr(crop(out_frames[21], lay["zoom_b"]), nn_zoom(B[21], roi))
    check("panels_full_and_zoom", p_a > 34 and p_b > 34 and p_z > 34, f"PSNR vs source: A {p_a:.1f}, B {p_b:.1f}, zoom(3x NN) {p_z:.1f} dB")
    d_same = crop(out_frames[5], lay["full_d"]).astype(np.float64)
    d_diff = crop(crop(out_frames[30], lay["full_d"]), [40, 50, 60, 40]).astype(np.float64)
    expect_blk = np.minimum(cv2.absdiff(A[30], B[30]).astype(np.float64) * 8, 255)[50:90, 40:100]
    check("diff_panel", d_same.mean() < 3.0 and expect_blk.mean() > 20 and abs(d_diff.mean() - expect_blk.mean()) < 6,
          f"identical frame diff-panel mean {d_same.mean():.2f}; changed block mean {d_diff.mean():.1f} "
          f"(expected |A-B|x8 {expect_blk.mean():.1f})")
    labels = rep["labels"]
    lab_ok = (any("PRE-CHANGE" in s for s in labels["a"]) and any("CANDIDATE" in s for s in labels["b"])
              and any("MUSETALK_EXAMPLE_FLAG=1" in s for s in labels["b"]) and any("123.4 fps" in s for s in labels["a"])
              and any("150.0 fps" in s for s in labels["b"]) and any("24 fps native" in s for s in labels["d"]))
    hdr = crop(out_frames[0], lay["header_b"])
    check("labels", lab_ok and (hdr.max(axis=2) > 200).sum() > 500, f"{len(labels['a'])}/{len(labels['b'])}/{len(labels['d'])} lines, "
          f"bright header px {(hdr.max(axis=2) > 200).sum()}")
    js = load_json(rep["outputs"]["json"])
    need = {"schema", "verdict", "gate", "summary", "per_frame", "mouth_box", "layout", "labels", "outputs", "arm_a", "arm_b"}
    check("json_fields", need <= set(js) and js["summary"]["mouth"]["max_abs_lsb"] == 4 and Path(js["outputs"]["contact_sheet"]["path"]).exists(),
          sorted(need - set(js)) or "all present; contact sheet written")

    # exact PASS on identical inputs (the lossless-round verdict + black diff panel)
    cb2 = _write_synthetic_clip(work / "b2", A, FPS_, "cand", roi)
    rep2 = compose(ca, cb2, work / "out/same", "selftest", "identical", expect="exact")
    of2 = list(iter_frames(Path(rep2["outputs"]["video"]), *rep2["layout"]["canvas"], limit=3))
    dmax = max(float(crop(f, rep2["layout"]["full_d"]).mean()) for f in of2)
    check("exact_pass_identical", rep2["gate"]["result"] == "PASS" and "SHA-identical" in rep2["verdict"] and dmax < 2.0,
          f"{rep2['verdict']} (diff panel mean {dmax:.2f})")
    # fp16 gate evaluation
    g = evaluate_gate("fp16", rep["summary"], True, FP16_GATE, [])
    check("gate_fp16_numeric", g["result"] == ("PASS" if rep["summary"]["full"]["psnr_min_db"] >= 40 else "FAIL"), g)
    # > 20 s input is truncated in the video, fully covered in the JSON; frame-count mismatch fails
    small_roi = [8, 20, 29, 36]
    long_a = _synthetic(10 * 25, 64, 96, 3)
    la = _write_synthetic_clip(work / "la", long_a, 10, "pre", small_roi)
    lb = _write_synthetic_clip(work / "lb", long_a[:-1], 10, "cand", small_roi)
    rep3 = compose(la, lb, work / "out/long", "selftest", "long", expect="exact")
    check("max_20s_and_count_mismatch", rep3["outputs"]["encode"]["frames"] == 200 and rep3["video_seconds"] == 20.0 and
          rep3["summary"]["frames_compared"] == 249 and rep3["gate"]["result"] == "FAIL" and "frame counts differ" in rep3["gate"]["reasons"],
          f"video {rep3['outputs']['encode']['frames']} frames/{rep3['video_seconds']} s; compared {rep3['summary']['frames_compared']}; {rep3['gate']['reasons']}")
    # corrupted dump is detected
    bad = load_json(cb)
    bad["frame_sha256"][3] = "0" * 64
    dump(work / "b/clip_bad.json", bad)
    try:
        compose(ca, work / "b/clip_bad.json", work / "out/bad", "selftest", "bad")
        check("corrupt_dump_detected", False, "no error")
    except RuntimeError as exc:
        check("corrupt_dump_detected", "frame 3" in str(exc), exc)
    # README index between markers (temp root)
    idx_root = work / "idx"
    (idx_root / "r0").mkdir(parents=True, exist_ok=True)
    shutil.copy(rep["outputs"]["json"], idx_root / "r0/synthetic__cand_ab.json")
    (idx_root / "README.md").write_text(f"# hand\n\nkeep me\n\n{README_BEGIN}\nold\n{README_END}\n\ntail stays\n")
    text = write_index(idx_root).read_text()
    check("readme_index", "keep me" in text and "tail stays" in text and "old" not in text.split(README_BEGIN)[1].split(README_END)[0]
          and "| synthetic | cand | FAIL | EXACT FAIL" in text, "markers respected, one row per report")
    # cache keys / dirty detection on a scratch git repo
    g_repo = work / "gitrepo"
    shutil.rmtree(g_repo, ignore_errors=True)
    (g_repo / "scripts").mkdir(parents=True)
    (g_repo / "scripts/x.py").write_text("print(1)\n")
    (g_repo / "notes.md").write_text("n\n")
    genv = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    for c in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run(["git", "-C", str(g_repo), *c], check=True, env=genv, capture_output=True)
    clean = tree_info(g_repo)
    (g_repo / "notes.md").write_text("changed\n")
    doc_only = tree_info(g_repo)
    (g_repo / "scripts/x.py").write_text("print(2)\n")
    dirty = tree_info(g_repo)
    check("tree_dirty_detection", not clean["dirty_code"] and not doc_only["dirty_code"] and dirty["dirty_code"] == ["scripts/x.py"]
          and dirty["dirty_code_digest"], f"clean {clean['head12']}, doc-only dirty ignored, code dirty {dirty['dirty_code']}")
    snap = snapshot_tree(g_repo, clean["head"], cache=work / "trees")
    check("snapshot_is_head", (snap / "scripts/x.py").read_text() == "print(1)\n" and (snap / "notes.md").is_symlink(),
          f"{snap}: scripts from HEAD, other entries symlinked")
    # harness shim: ROOT and imports resolve inside the arm's tree
    probes = {}
    for tree in (MAIN_TREE, WORKTREE):
        out = subprocess.run(harness_command(tree, ["--video-ab-probe"]), cwd=tree, env=arm_env({}), capture_output=True, text=True)
        probes[str(tree)] = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-400:]}
    shim_ok = all(p.get("ROOT") == t and str(p.get("hls_gpu_scheduler", "")).startswith(t + "/") for t, p in probes.items())
    check("harness_tree_shim", shim_ok, probes)
    # replay adapter: a fake golden run whose frames hash like the harness
    rep_dir = work / "replay"
    frames_r = _synthetic(12, 64, 96, 5)
    write_lossless(rep_dir / "videos/lab_jp_d10.mkv", frames_r, 20)
    golden = {"runs": [{"label": "lab", "fps": 99.0, "frames": 12, "wall_s": 0.12, "jobs": {"jp_d10": {
        "frames": [sha_array(f) for f in frames_r] + ["x" * 64] * 3, "status": "completed", "order_errors": [],
        "faces_digest": "f", "frames_digest": "g", "yuv_digest": "y"}}}], "vae_backend": "taesd", "unet_backend": "tensorrt_unet"}
    dump(rep_dir / "golden.json", golden)
    made = replay_clips_from_golden(rep_dir / "golden.json", rep_dir / "videos", "lab", ["jp_d10"], rep_dir, {"name": "lab"})
    rc_meta = load_json(made["replay_jp_d10"])
    golden["runs"][0]["jobs"]["jp_d10"]["frames"][4] = "0" * 64
    dump(rep_dir / "golden_bad.json", golden)
    try:
        replay_clips_from_golden(rep_dir / "golden_bad.json", rep_dir / "videos", "lab", ["jp_d10"], rep_dir / "bad", {"name": "lab"})
        detect = False
    except RuntimeError:
        detect = True
    check("replay_adapter", rc_meta["frame_count"] == 12 and rc_meta["job_total_frames"] == 15 and rc_meta["fps"] == 20 and detect
          and len(rc_meta["mouth_box"]) == 4, f"12 video frames of 15, mouth_box {rc_meta['mouth_box']}, mismatch detected={detect}")
    # orchestration: baseline install/cache state, candidate dumps, compose-arm, golden compare, cleanup, README
    flow_ok, flow_detail = _selftest_flow(work / "flow")
    check("compose_arm_flow", flow_ok, flow_detail)
    # real data, CPU: dry-run chin composition through both trees must be SHA-identical
    if not args.no_real:
        env = dict(arm_env({}), CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([PY, "-c", MOCK_GPU_LOOP, str(CHIN_RENDER), str(WORKTREE), str(work / "mock_gpu"), "16"],
                           cwd=WORKTREE, env=env, capture_output=True, text=True)
        try:
            mres = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            mres = {"error": (p.stdout + p.stderr)[-800:]}
        check("chin_gpu_loop_mock", p.returncode == 0 and mres.get("frames") == 16 and mres.get("repeats_identical")
              and mres.get("landmarks_restored") and mres.get("checks") and not mres.get("cuda_initialized"), mres)
        dry = {}
        for name, tree in (("pre_change", MAIN_TREE), ("wt_default", WORKTREE)):
            outd = work / f"dry_{name}"
            shutil.rmtree(outd, ignore_errors=True)
            cmd = [PY, str(CHIN_RENDER), "--repo", str(tree), "--out-root", str(outd), "--identities", "japanese",
                   "--frames", str(args.dry_frames), "--dry-run", "--arm-name", name, "--tree-meta", json.dumps(tree_info(tree))]
            p = subprocess.run(cmd, cwd=tree, env=arm_env({}), capture_output=True, text=True)
            dry[name] = (p.returncode, outd / "japanese/clip.json", (p.stdout + p.stderr)[-600:])
        ok = all(rc == 0 and cj.exists() for rc, cj, _ in dry.values())
        if check("dry_run_chin_both_trees", ok, {k: (v[0], v[2][-200:]) for k, v in dry.items()}):
            ma, mb = load_json(dry["pre_change"][1]), load_json(dry["wt_default"][1])
            origin_ok = ma["module_origin_check"]["ok"] and mb["module_origin_check"]["ok"]
            rep4 = compose(dry["pre_change"][1], dry["wt_default"][1], work / "out/dry_chin", "selftest", "chin_japanese__dry", expect="exact")
            check("dry_run_chin_ab_exact", rep4["gate"]["result"] == "PASS" and origin_ok and ma["recipe_checks"]["passes"],
                  f"{rep4['verdict']}; module origins ok={origin_ok}; mouth_box {rep4['mouth_box']}; video {rep4['outputs']['video']}")
    summary = {"schema": "video_ab_selftest_v1", "created_utc": utc(), "work_dir": str(work), "seconds": round(time.time() - t0, 1),
               "tests": results, "all_pass": all(r["result"] == "pass" for r in results)}
    if args.out:
        dump(Path(args.out), summary)
    print(f"{'PASS' if summary['all_pass'] else 'FAIL'} selftest {sum(r['result'] == 'pass' for r in results)}/{len(results)} "
          f"in {summary['seconds']} s (work {work})", flush=True)
    return 0 if summary["all_pass"] else 1


def cmd_rollup(args) -> int:
    """One line per clip verdict across all compose-arm reports in --dir (+ step statuses)."""
    d = Path(args.dir)
    rows, steps = [], {}
    for rp in sorted(d.glob("*.json")):
        try:
            r = load_json(rp)
        except (OSError, ValueError):
            continue
        if "clips" in r and "expect" in r:  # compose-arm report
            for clip, c in r["clips"].items():
                rows.append({"round": r["round"], "arm": r["arm"], "clip": clip, "result": c.get("result"),
                             "verdict": c.get("verdict"), "video": c.get("video"), "report": rp.name})
        elif "tests" in r and "all_pass" in r:  # selftest
            steps[rp.stem] = "PASS" if r["all_pass"] else "FAIL"
        elif "step" in r or "result" in r:
            steps[rp.stem] = r.get("result") or r.get("status")
    for row in rows:
        print(f"{row['result']:6s} {row['round']}/{row['arm']} {row['clip']}: {row['verdict']}")
    out = {"schema": "video_ab_rollup_v1", "created_utc": utc(), "clips": rows, "steps": steps,
           "all_pass": bool(rows) and all(r["result"] in ("PASS", "REPORT") for r in rows),
           "readme": str(write_index())}
    dump(Path(args.out) if args.out else d / "rollup.json", out)
    print(f"{'PASS' if out['all_pass'] else 'FAIL'} rollup: {sum(r['result'] == 'PASS' for r in rows)}/{len(rows)} clip verdicts pass")
    return 0 if out["all_pass"] else 1


def cmd_tree_info(args) -> int:
    tree, meta = resolve_pre_tree(args.pre_mode)
    print(json.dumps({"pre_change": dict(meta, resolved_tree=str(tree)), "candidate": candidate_tree_meta(WORKTREE)}, indent=1))
    return 0


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, clips=True):
        p.add_argument("--pre-mode", choices=["auto", "live", "snapshot"], default="auto",
                       help="pre-change tree: live main if code-clean (auto), else exact HEAD snapshot")
        if clips:
            p.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
        p.add_argument("--repeats", type=int, default=3, help="chin: timed warm repeats")
        p.add_argument("--replay-jobs", default="all", help="harness golden jobs run per arm (same set both arms)")
        p.add_argument("--replay-identities", default="bob,jp,latfh1")
        p.add_argument("--video-seconds", type=float, default=MAX_SECONDS)

    p = sub.add_parser("baseline", help="render missing/stale pre-change baselines (GPU unless cached)")
    common(p)
    p.add_argument("--check", action="store_true", help="only report cache state (exit 3 if any missing/stale)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", default=None, help="write a JSON step report here")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("render-arm", help="render a candidate arm's clips (GPU)")
    common(p)
    p.add_argument("--round", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--flags", default="")
    p.add_argument("--tree", default=str(WORKTREE))
    p.add_argument("--report", default=None, help="write a JSON step report here")
    p.set_defaults(fn=cmd_render_arm)

    p = sub.add_parser("rollup", help="summarize the step reports of a GPU sequence (CPU)")
    p.add_argument("--dir", default=str(DOCS_DIR))
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_rollup)

    p = sub.add_parser("compose-arm", help="compose A/B videos for a rendered arm (CPU)")
    p.add_argument("--round", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--clips", default=None)
    p.add_argument("--expect", choices=["exact", "fp16", "report"], default="exact")
    p.add_argument("--crf", type=int, default=10)
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    p.add_argument("--keep-dumps", action="store_true", help="keep the candidate's lossless frames")
    p.add_argument("--report", default=None)
    p.set_defaults(fn=cmd_compose_arm)

    p = sub.add_parser("compose", help="compose two clip.json dumps (CPU)")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--round", default="adhoc")
    p.add_argument("--clip", default="clip")
    p.add_argument("--expect", choices=["exact", "fp16", "report"], default="report")
    p.add_argument("--crf", type=int, default=10)
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    p.add_argument("--mouth-box", default=None, help="x0,y0,x1,y1 (default: from the clip dump)")
    p.add_argument("--index", action="store_true")
    p.set_defaults(fn=cmd_compose)

    p = sub.add_parser("index", help="regenerate the README index (CPU)")
    p.set_defaults(fn=lambda a: (print(write_index()), 0)[1])

    p = sub.add_parser("tree-info", help="show tree resolution (CPU)")
    p.add_argument("--pre-mode", choices=["auto", "live", "snapshot"], default="auto")
    p.set_defaults(fn=cmd_tree_info)

    p = sub.add_parser("selftest", help="CPU-only tests")
    p.add_argument("--work", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-real", action="store_true", help="skip the real-data dry-run chin composition")
    p.add_argument("--dry-frames", type=int, default=12)
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
