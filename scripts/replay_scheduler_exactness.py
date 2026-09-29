#!/usr/bin/env python
"""Golden replay exactness + unpaced speed harness for HLSGPUStreamScheduler.

Plan items 0.3 (golden capture/replay), 0.2 (event-vs-sync timing gate), and the
speed/exactness gates of 1.1, 1.4, 1.9, 1.10 and part of 1.5
(docs/musetalk_4070s_300fps_plan_2026-09-27.md).

What it does
------------
It loads the real model stack exactly like api_server.py does (ParallelAvatarManager
with the live launcher's env), builds real WebRTC-shaped sessions (the real
WebRTCSession dataclass and the real LivePoseVideoRouter / MotionBank, as
scripts/webrtc_manager.py create_session + stage_pose_plan would), and drives the
real HLSGPUStreamScheduler through submit_webrtc_stream() with fixed prepared
avatars and fixed WAVs, N jobs concurrently.

Only the transport is replaced:
  * the idle track is a stub with a FROZEN idle phase per job (the live track reads
    a wall-clock phase; freezing it is what makes the replay deterministic);
  * frame_batch_callback is a sink that records, in order, the SHA-256 of every
    composed pre-encoder BGR frame (and of its PyAV yuv420p conversion, the exact
    call webrtc_tracks.push_bgr_frames_batch makes), plus the api_server
    motion_metadata() router calls for fidelity;
  * a tap on scheduler._dispatch_compose_batch records the SHA-256 of every decoded
    face (the TAESD uint8 BGR output row) per (job, generation frame).
Hashes are computed in memory on a thread pool; nothing raw is written. Optional
per-job lossless (libx264rgb -qp 0) mp4s go to --video-dir for review videos.

Modes
-----
  golden  all jobs submitted at once, run to completion, hash everything.
  speed   N sessions chaining turns back-to-back for --warmup-s + --seconds; the
          sink only counts (unpaced null sink), or simulates the 20 fps strict-FIFO
          WebRTC consumer with --paced. Reports generated fps in the window, GPU
          utilisation (nvidia-smi 5 Hz), process / scheduler-thread CPU, and the
          scheduler's CUDA-event capacity telemetry when HLS_GPU_EVENT_TIMING=1.
  compare CPU only: compare two golden JSONs frame by frame.

Which tree
----------
--repo PATH selects the code under test: the harness puts PATH first on sys.path,
drops its own tree from sys.path, and chdirs to PATH before importing anything
from `scripts.` / `musetalk.`, so e.g. `--repo /workspace/MuseTalk` replays the
clean main checkout (the pre-change arm) and `--repo /workspace/MuseTalk-perf300`
(the default: this file's own tree) replays the candidates. With a foreign repo
the harness never writes into it (no .pyc: sys.dont_write_bytecode; git is only
read with --no-optional-locks). Launch it with cwd = this file's tree.

Runs
----
Several runs can execute sequentially in ONE process (the model loads once):
  --run LABEL:SOURCE[:KEY=VAL,KEY=VAL...]
SOURCE is `repo` (alias `worktree`: --repo's working copy of
scripts/hls_gpu_scheduler.py), `head` (git HEAD's copy in --repo), `base`
(merge-base(HEAD, main)'s copy: today's scheduler running inside --repo's other
modules, which isolates scheduler changes from the rest of the tree) or a path.
KEY=VAL are env overrides applied while that run's scheduler is constructed and
runs (every scheduler flag is read in __init__ or at use time). Flags read at
module import time elsewhere (e.g. MUSETALK_VAE_DECODE_TIMING_SYNC in
musetalk/models/vae.py, MUSETALK_TRT_UNET_CUDAGRAPHS at model load) must be set
for the whole process instead. Harness-only keys (not exported to the env):
  REPLAY_JOBS=id+id+...   golden: this run replays only these jobs.
  REPLAY_CONSUMER_DEPTH=zero|none   golden: overrides --consumer-depth for this run.

Outputs
-------
golden: per job, in generation order, SHA-256 of every decoded face (the TAESD
uint8 BGR row; "RAW_NO_GPU" where HLS_SKIP_GPU_FOR_RAW skipped it) and of every
composed pre-encoder BGR frame (+ its PyAV yuv420p unless --no-yuv), batch order
checks, first-frame latency, the scheduler capacity telemetry when present, and
an event-vs-host stage-time summary. --video-dir writes short H.264 mp4s
(libx264 -crf --video-crf, <= 12; --video-lossless for libx264rgb -qp 0 mkv)
for the labelled comparison video tool.
speed: unpaced null sink (the consumer reports an empty queue, so run-ahead caps
never throttle) or --paced; generated fps over a --seconds window per N.

Run it under the GPU lease (model load has an ~8.5-10.6 GB host-RSS peak):
  scripts/box_guard.sh run --min-avail-gb 14 --wait-min 60 --label sched_golden -- \
    /workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/replay_scheduler_exactness.py \
    --repo /workspace/MuseTalk --mode golden --run main_r1:repo --out docs/.../scheduler/golden_main_r1.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent


def _repo_from_argv(argv) -> Path:
    """--repo is resolved before any repo module is imported (see docstring)."""
    for index, arg in enumerate(argv):
        if arg == "--repo" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
        if arg.startswith("--repo="):
            return Path(arg.split("=", 1)[1]).resolve()
    return HARNESS_ROOT


ROOT = _repo_from_argv(sys.argv[1:]) if __name__ == "__main__" else HARNESS_ROOT
FOREIGN_REPO = ROOT != HARNESS_ROOT
if FOREIGN_REPO:
    # Import nothing from the harness's own tree, and write nothing (.pyc)
    # into the repo under test.
    sys.dont_write_bytecode = True
    _own = {str(HARNESS_ROOT), str(HARNESS_ROOT / "scripts")}
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in _own]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_ENV_FILE = ROOT / ".runtime/musetalk_trt_local_sm89.env"
BOB_DIR = Path("/workspace/experiments/chinese_bob_webrtc_20260927")
# Exported by /workspace/experiments/chinese_bob_webrtc_20260927/run_local_api.sh after it
# sources LIVE_ENV_FILE (copied; the launcher itself is never read or modified).
LIVE_LAUNCHER_OVERRIDES = {
    "MUSETALK_VAE_BACKEND": "taesd",
    "MUSETALK_UNET_BACKEND": "trt",
    "MUSETALK_TRT_FALLBACK": "0",
    "MUSETALK_BLEND_FIXED_POINT": "1",
    "MUSETALK_BLEND_SHRINK_MASK_BBOX": "1",
    "MUSETALK_TAESD_WARMUP_BATCHES": "8",
    "WEBRTC_MOTION_ATLAS": str(BOB_DIR / "motion-atlas.json"),
    "WEBRTC_MOTION_ALLOW_UNREVIEWED": "1",
    "AVATAR_S3_ENABLED": "0",
    "KOKORO_TTS_DEVICE": "cpu",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
ENV_PREFIXES = ("MUSETALK_", "HLS_", "WEBRTC_", "TORCH", "CUDA_", "PYTORCH_", "GPU_", "AVATAR_")

IDENTITIES = {
    # 3-pose WebRTC motion avatar (pose protocol v1 + motion atlas), as the wall uses it.
    "bob": {"kind": "motion", "avatar_id": "chinese_bob_pink_bedroom_idle_d4b06da317",
            "pose_set": str(BOB_DIR / "session-pose-set.json")},
    # Standard single-pose avatars.
    "jp": {"kind": "standard", "avatar_id": "japanese_realtime_talking_7d94520b7f"},
    "latfh1": {"kind": "standard", "avatar_id": "latina_guided_20260925_talking_84c5bc80b8_fh1"},
}
WAVS = {
    "bob1": str(BOB_DIR / "audio/turn_one.wav"),
    "bob2": str(BOB_DIR / "audio/turn_two.wav"),
    "dense10": "/workspace/experiments/cheek_drift_ab_20260926/h3_reduced_mouth_screen_20260926/dense_tts/speech_dense_10s.wav",
    "michael10": "/workspace/experiments/avatar_diversity_20260927/_audio/am_michael/speech.wav",
    "eng60": str(ROOT / "data/audio/eng.wav"),
    "out26": str(ROOT / "data/audio/outputnew2.wav"),
    "sun22": str(ROOT / "data/audio/sun.wav"),
    "zero3": "@zeros:3.0",  # synthesized exact-zero PCM (the exact_silence path)
}
# wall pose plan (templates/webrtc_wall.py POSE_PLAN / verify_wall_concurrency.py)
WALL_POSE_PLAN = {"version": 2, "clock": "audio_progress",
                  "segments": [{"at_permille": 0, "pose_id": "speaking_direct"}],
                  "switch_mode": "next_boundary", "on_complete": "neutral_resting"}
# A plan that switches speech pose inside the turn (exercises the pose crossfade path).
# Neutral (WEBRTC_RAW_IDLE_POSE) frames already occur in every motion turn: the entry
# bridge and the return to neutral_resting are rendered inside the turn.
MIDTURN_SMILE_PLAN = {"version": 2, "clock": "audio_progress",
                      "segments": [{"at_permille": 0, "pose_id": "speaking_direct"},
                                   {"at_permille": 300, "pose_id": "light_smile"},
                                   {"at_permille": 700, "pose_id": "speaking_direct"}],
                      "switch_mode": "next_boundary", "on_complete": "neutral_resting"}
GOLDEN_JOBS = [
    {"id": "bob_t1", "identity": "bob", "wav": "bob1", "idle_frame": 0},
    {"id": "bob_t2", "identity": "bob", "wav": "bob2", "idle_frame": 97},
    {"id": "bob_d10", "identity": "bob", "wav": "dense10", "idle_frame": 183},
    {"id": "bob_mid", "identity": "bob", "wav": "michael10", "idle_frame": 41, "plan": "midturn_smile"},
    {"id": "jp_d10", "identity": "jp", "wav": "dense10", "idle_frame": 0},
    {"id": "jp_m10", "identity": "jp", "wav": "michael10", "idle_frame": 211},
    {"id": "lat_m10", "identity": "latfh1", "wav": "michael10", "idle_frame": 57},
    {"id": "lat_b1", "identity": "latfh1", "wav": "bob1", "idle_frame": 300},
    {"id": "bob_zero", "identity": "bob", "wav": "zero3", "idle_frame": 11},
    {"id": "jp_zero", "identity": "jp", "wav": "zero3", "idle_frame": 5},
]
SPEED_WAVS = ["eng60", "out26", "sun22", "dense10", "michael10", "bob1", "bob2"]


# ----------------------------------------------------------------------------- env
def parse_env_file(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip('"').strip("'")
    return values


def apply_env(overlay: str | None) -> dict:
    """Live env file, then the launcher's overrides, then --overlay; the caller wins."""
    caller = set(os.environ)
    merged, source = {}, {}
    layers = [(str(LIVE_ENV_FILE.relative_to(ROOT)), parse_env_file(LIVE_ENV_FILE)),
              ("run_local_api.sh overrides", dict(LIVE_LAUNCHER_OVERRIDES))]
    if overlay:
        layers.append((overlay, parse_env_file(Path(overlay))))
    for name, values in layers:
        for key, value in values.items():
            merged[key] = value
            source[key] = name
    for key, value in merged.items():
        if key in caller:
            source[key] = "caller"
        else:
            os.environ[key] = value
    for key in os.environ:
        if key.startswith(ENV_PREFIXES) and key not in source:
            source[key] = "caller"
    return {k: {"value": os.environ.get(k), "source": source[k]} for k in sorted(source) if k in os.environ}


def apply_gpu_aware_runtime_defaults(gpu_id: int = 0) -> dict:
    """Same set-if-unset logic as api_server._apply_gpu_aware_runtime_defaults."""
    from scripts.concurrent_gpu_manager import (default_reserved_memory_gb, detect_total_gpu_memory_gb,
                                                recommended_scheduler_batch_config)
    total_gb, src = detect_total_gpu_memory_gb(gpu_id=gpu_id)
    reserved_gb = default_reserved_memory_gb(total_gb)
    rec = recommended_scheduler_batch_config(total_gb, profile=os.getenv("PROFILE", "baseline"))
    changed = {}

    def setdefault(name, value):
        if os.getenv(name) in (None, ""):
            os.environ[name] = str(value)
            changed[name] = str(value)

    setdefault("GPU_TOTAL_MEMORY_GB", f"{total_gb:.1f}")
    setdefault("GPU_RESERVED_MEMORY_GB", f"{reserved_gb:.1f}")
    setdefault("GPU_MEMORY_DETECTION_SOURCE", src)
    setdefault("HLS_SCHEDULER_MAX_BATCH", rec["max_combined_batch_size"])
    setdefault("HLS_SCHEDULER_FIXED_BATCH_SIZES", ",".join(str(b) for b in rec["fixed_batch_sizes"]))
    setdefault("MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES", ",".join(str(b) for b in rec["warmup_batches"]))
    setdefault("HLS_SCHEDULER_STARTUP_SLICE_SIZE", rec["startup_slice_size"])
    available_gb = max(1.0, total_gb - reserved_gb)
    setdefault("AVATAR_CACHE_MAX_MEMORY_MB", int(max(6000, min(24000, available_gb * 1024 * 0.75))))
    return changed


def env_int(name, default):
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def env_float(name, default):
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return -1.0


def rss_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1048576
    return -1.0


def proc_cpu_s() -> float:
    fields = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def thread_cpu_s(native_id) -> float:
    """CPU seconds of one OS thread (Python thread names are not OS names here)."""
    if not native_id:
        return 0.0
    try:
        fields = open(f"/proc/self/task/{native_id}/stat").read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except OSError:
        return 0.0


# ------------------------------------------------------------------ hashing / sinks
def sha_array(array) -> str:
    import numpy as np
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.shape}|{a.dtype}|".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def bgr_to_yuv420p(bgr):
    """The exact conversion webrtc_tracks.push_bgr_frames_batch performs today."""
    import av
    return av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p").to_ndarray()


MAX_VIDEO_CRF = 12


class LosslessWriter:
    """Encodes BGR frames in order for review videos.

    Default: H.264 mp4 (libx264 -crf <= 12, yuv420p) for the labelled comparison
    video tool; lossless=True: RGB H.264 mkv (libx264rgb -qp 0). ffmpeg is
    started on the first frame so the size always matches the composed frames
    (prepared avatar frames can differ from the idle video's size)."""

    def __init__(self, path: Path, fps: int, max_frames: int, crf: int = MAX_VIDEO_CRF,
                 lossless: bool = False):
        self.lossless = bool(lossless)
        self.crf = min(MAX_VIDEO_CRF, max(0, int(crf)))
        self.path = Path(path).with_suffix(".mkv" if self.lossless else ".mp4")
        self.fps = fps
        self.max_frames = max_frames
        self.count = 0
        self.proc = None
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _codec_args(self, width: int, height: int) -> list:
        if self.lossless:
            return ["-c:v", "libx264rgb", "-qp", "0", "-preset", "ultrafast"]
        # yuv420p needs even dimensions; pad by one pixel if a clip is odd.
        pad = [] if width % 2 == 0 and height % 2 == 0 else ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        return pad + ["-c:v", "libx264", "-crf", str(self.crf), "-preset", "medium",
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    def put(self, bgr):
        if self.count >= self.max_frames:
            return
        self.count += 1
        self.q.put(bgr)

    def _run(self):
        import numpy as np
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.proc is None:
                h, w = item.shape[:2]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.proc = subprocess.Popen(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", *self._codec_args(w, h),
                     "-threads", "2", str(self.path)],
                    stdin=subprocess.PIPE)
            self.proc.stdin.write(np.ascontiguousarray(item).tobytes())

    def close(self):
        self.q.put(None)
        self.thread.join()
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


class JobRecorder:
    """Per-job ordered record of face / composed / yuv hashes and callback batches."""

    def __init__(self, job_id: str, hash_pool, hash_faces: bool, hash_frames: bool, hash_yuv: bool,
                 writer: LosslessWriter | None = None):
        self.job_id = job_id
        self.pool = hash_pool
        self.hash_faces, self.hash_frames, self.hash_yuv = hash_faces, hash_frames, hash_yuv
        self.writer = writer
        self.faces: dict = {}
        self.frames: dict = {}
        self.yuv: dict = {}
        self.yuv_mismatch: list = []
        self.callbacks: list = []
        self.pose_meta: dict = {}
        self.order_errors: list = []
        self.next_expected = 1
        self.frame_count = 0
        self.frame_times: list = []
        self.lock = threading.Lock()

    def record_faces(self, start_frame_idx: int, batch_frames):
        if not self.hash_faces:
            return
        import numpy as np
        for i, face in enumerate(batch_frames):
            idx = int(start_frame_idx) + i
            # Copy before hashing off-thread: at HLS_GPU_PIPELINE_DEPTH>=2 faces are
            # views of a pinned ring slot the scheduler reuses once compose is done.
            self.faces[idx] = "RAW_NO_GPU" if face is None else self.pool.submit(sha_array, np.array(face))

    def record_frames(self, frames, start_frame_idx: int, now: float):
        with self.lock:
            if start_frame_idx != self.next_expected:
                self.order_errors.append({"expected": self.next_expected, "got": start_frame_idx,
                                          "count": len(frames)})
            self.next_expected = start_frame_idx + len(frames)
            self.callbacks.append((start_frame_idx, len(frames)))
            self.frame_count += len(frames)
            self.frame_times.append((now, len(frames)))
        for i, frame in enumerate(frames):
            idx = start_frame_idx - 1 + i  # 0-based generation frame index
            bgr = getattr(frame, "bgr", frame)
            yuv_given = getattr(frame, "yuv420p", None)
            if self.hash_frames:
                self.frames[idx] = self.pool.submit(sha_array, bgr)
            if self.hash_yuv:
                if yuv_given is not None:
                    self.yuv[idx] = self.pool.submit(self._check_yuv, idx, bgr, yuv_given)
                else:
                    self.yuv[idx] = self.pool.submit(lambda b: sha_array(bgr_to_yuv420p(b)), bgr)
            if self.writer is not None:
                self.writer.put(bgr)

    def _check_yuv(self, idx, bgr, yuv_given):
        """WEBRTC_YUV_IN_COMPOSE producer contract: the carried I420 must equal
        today's conversion of the same BGR; the recorded hash is of the carried one."""
        import numpy as np
        if hasattr(yuv_given, "to_ndarray"):
            yuv_given = yuv_given.to_ndarray()
        expected = bgr_to_yuv420p(bgr)
        if not np.array_equal(expected, yuv_given):
            self.yuv_mismatch.append(idx)
        return sha_array(yuv_given)

    def result(self) -> dict:
        def ordered(d):
            out = []
            for k in sorted(d):
                v = d[k]
                out.append(v.result() if isinstance(v, Future) else v)
            return out, sorted(d)

        faces, face_idx = ordered(self.faces)
        frames, frame_idx = ordered(self.frames)
        yuv, _ = ordered(self.yuv)
        digest = lambda xs: hashlib.sha256("".join(xs).encode()).hexdigest()
        return {
            "frame_count": self.frame_count,
            "face_indices_contiguous": face_idx == list(range(len(face_idx))),
            "frame_indices_contiguous": frame_idx == list(range(len(frame_idx))),
            "order_errors": self.order_errors,
            "callback_batches": len(self.callbacks),
            "faces_digest": digest(faces), "frames_digest": digest(frames), "yuv_digest": digest(yuv),
            "raw_no_gpu_faces": sum(1 for f in faces if f == "RAW_NO_GPU"),
            "yuv_contract_mismatches": self.yuv_mismatch,
            "faces": faces, "frames": frames, "yuv": yuv,
            "pose_ids": [self.pose_meta.get(i) for i in range(len(frame_idx))],
        }


# --------------------------------------------------------------- sessions / idle stub
class InlineLoop:
    """Stands in for the asyncio loop the scheduler resolves completion futures on."""

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


class FrozenIdleTrack:
    """Idle-track stub: a frozen idle phase instead of the wall-clock one.

    capture_idle_sync_timing mirrors SwitchableVideoStreamTrack.capture_idle_sync_timing
    (scripts/webrtc_tracks.py) with idle_timing = {source_frame_index: frozen}.
    """

    def __init__(self, source_frame_index: int, source_fps: float, source_frame_count: int,
                 idle_pose_id: str, motion_bank=None, video_path: str = ""):
        self.source_frame_index = int(source_frame_index) % max(1, int(source_frame_count))
        self.source_fps = float(source_fps)
        self.source_frame_count = int(source_frame_count)
        self.idle_pose_id = idle_pose_id
        self.motion_bank = motion_bank
        self.video_path = video_path

    def get_pose_status(self):
        return {"current_pose_id": self.idle_pose_id, "pending_pose_ids": [],
                "current_idle_video_path": self.video_path}

    def clear_pending_idle_switches(self):
        return None

    def capture_idle_sync_timing(self, generation_fps, cycle_frames=None, reveal_delay_seconds=0.0, hold=True):
        if self.motion_bank is not None:
            hold = False
        source_fps = self.source_fps
        delay_frames = int(round(max(0.0, float(reveal_delay_seconds or 0.0)) * source_fps)) if source_fps > 0 else 0
        target = (self.source_frame_index + delay_frames) % self.source_frame_count
        idle_phase_seconds = target / source_fps if source_fps > 0 else 0.0
        offset_frames = max(0, int(round(idle_phase_seconds * float(generation_fps or 0.0))))
        if cycle_frames and cycle_frames > 0:
            offset_frames %= int(cycle_frames)
        return {"timing_source": "replay_frozen_idle", "mapping": "single_video_source_frame",
                "offset_seconds": offset_frames / float(generation_fps) if generation_fps else 0.0,
                "offset_frames": offset_frames, "idle_source_frame_index": self.source_frame_index,
                "target_source_frame_index": target, "source_frame_count": self.source_frame_count,
                "source_fps": source_fps, "cycle_frames": cycle_frames, "generation_fps": generation_fps,
                "reveal_delay_seconds": reveal_delay_seconds, "hold_enabled": bool(hold)}


def probe_video(path: str) -> tuple[float, int]:
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 25)
        frames = int(stream.frames or 0)
        if frames <= 0:
            frames = sum(1 for _ in container.decode(video=0))
    return fps, frames


def resolve_idle_video(avatar_id: str) -> str:
    """api_server._resolve_avatar_video_path(avatar_id, role='idle') for pose_id=None."""
    avatar_dir = ROOT / "results/v15/avatars" / avatar_id
    info = json.loads((avatar_dir / "avator_info.json").read_text())
    candidates = [info.get("idle_video_path"), avatar_dir / "idle_video.mp4", info.get("input_video_path"),
                  info.get("talking_video_path"), avatar_dir / "input_video.mp4"]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file() and p.stat().st_size >= 1024:
            return str(p.resolve())
    raise FileNotFoundError(f"no idle video for {avatar_id}")


_VIDEO_PROBE_CACHE: dict = {}


def build_session(identity: str, session_id: str, idle_frame: int, batch_size: int = 8, fps: int = 20,
                  prebuffer_seconds: float = 2.0, chunk_duration: int = 2):
    """Create a WebRTCSession as webrtc_manager.create_session would (no RTCPeerConnection)."""
    from scripts.pose_protocol import POSE_IDS, normalize_pose_set
    from scripts.webrtc_manager import WebRTCSession
    from scripts.webrtc_pose_router import LivePoseVideoRouter

    spec = IDENTITIES[identity]
    avatar_id = spec["avatar_id"]
    pose_set = {}
    if spec["kind"] == "motion":
        pose_set = normalize_pose_set(json.loads(Path(spec["pose_set"]).read_text()))
        pose_video_paths, prepared = {}, {}
        for pose_id in POSE_IDS:
            entry = pose_set["poses"][pose_id]
            pose_video_paths[pose_id] = resolve_idle_video(entry["avatar_id"])
            prepared[pose_id] = entry["avatar_id"]
        idle_pose_id = pose_set["default_pose_id"]
        idle_video_path = pose_video_paths[idle_pose_id]
        pose_video_paths["default"] = idle_video_path
        prepared["default"] = avatar_id
        live_pose_id = idle_pose_id
        generation_avatar_id = pose_set["poses"]["speaking_direct"]["avatar_id"]
        for pose_id, entry in pose_set["poses"].items():
            prepared.setdefault(pose_id, entry["avatar_id"])
        idle_source_fps = float(pose_set["poses"][idle_pose_id].get("fps") or fps)
    else:
        idle_video_path = resolve_idle_video(avatar_id)
        pose_video_paths = {"default": idle_video_path}
        prepared = {"default": avatar_id}
        idle_pose_id = live_pose_id = "default"
        generation_avatar_id = avatar_id
        idle_source_fps = float(fps)
    prepared = {str(k).strip().lower(): str(v).strip() for k, v in prepared.items()}
    prepared.setdefault("default", avatar_id)
    router = LivePoseVideoRouter(dict(pose_video_paths), prepared_pose_id="default",
                                 prepared_pose_ids=set(prepared), pose_variant_render_keys={},
                                 initial_pose_id=live_pose_id)
    if spec["kind"] == "motion" and router.motion_bank is None:
        raise RuntimeError("motion bank did not load for the 3-pose identity (check WEBRTC_MOTION_ATLAS)")
    if idle_video_path not in _VIDEO_PROBE_CACHE:
        _VIDEO_PROBE_CACHE[idle_video_path] = probe_video(idle_video_path)
    vfps, vframes = _VIDEO_PROBE_CACHE[idle_video_path]
    track = FrozenIdleTrack(idle_frame, idle_source_fps if spec["kind"] == "motion" else vfps, vframes,
                            idle_pose_id, motion_bank=router.motion_bank, video_path=idle_video_path)
    session = WebRTCSession(
        session_id=session_id, avatar_id=avatar_id, fps=fps, playback_fps=fps, batch_size=batch_size,
        chunk_duration=chunk_duration, prebuffer_seconds=prebuffer_seconds, idle_track=track,
        idle_pose_id=idle_pose_id, idle_video_path=idle_video_path,
        pose_protocol_enabled=bool(pose_set), pose_set=pose_set,
        pose_switch_mode="next_boundary" if pose_set else "immediate", pose_video_paths=pose_video_paths,
        current_pose_id=idle_pose_id, generation_avatar_id=generation_avatar_id, live_pose_id=live_pose_id,
        live_pose_router=router, prepared_pose_avatar_ids=prepared,
    )
    session.webrtc_live_reveal_delay_seconds = 0.0
    return session


def stage_turn(session, turn_id: str, plan_name: str | None):
    """webrtc_manager.stage_pose_plan (no asyncio): only pose-protocol sessions carry a plan."""
    from scripts.pose_protocol import normalize_pose_plan
    if not session.pose_protocol_enabled:
        return
    plan = MIDTURN_SMILE_PLAN if plan_name == "midturn_smile" else WALL_POSE_PLAN
    normalized = normalize_pose_plan(plan)
    session.assistant_active = True
    session.active_turn_id = turn_id
    context_key = ":".join((str(session.pose_set.get("pose_set_id") or "pose-set"), session.session_id, turn_id))
    session.live_pose_router.set_variant_context(context_key)
    session.active_pose_plan = normalized
    session.compiled_pose_plan = None
    session.reset_rendered_pose_trace()
    session.live_pose_id = normalized["segments"][0]["pose_id"]


def finish_turn(session):
    """webrtc_manager.finish_assistant_turn (no asyncio)."""
    if not session.pose_protocol_enabled:
        return
    session.assistant_active = False
    session.active_pose_plan = {}
    session.active_turn_id = None
    try:
        session.live_pose_router.switch_pose("neutral_resting")
        session.live_pose_id = "neutral_resting"
    except (KeyError, ValueError):
        pass


def motion_metadata(session, start_index, count, total_frames):
    """api_server frame_batch_callback's motion_metadata(): same router calls."""
    router = session.live_pose_router
    if router is None or getattr(router, "motion_bank", None) is None:
        return None
    raw_idle = os.getenv("WEBRTC_RAW_IDLE_POSE", "").strip().lower() in {"1", "true", "yes", "on"}
    final_index = total_frames - 1
    final_snapshot = (router.snapshot(final_index, session.fps)
                      if raw_idle and start_index + count >= max(0, total_frames - 16) else None)
    result = []
    for index in range(start_index, start_index + count):
        snapshot = router.snapshot(index, session.fps)
        result.append({"pose_id": snapshot.pose_id, "source_frame": router.source_frame_index(snapshot, index)})
    del final_snapshot
    return result


# ---------------------------------------------------------------- audio preparation
def prepare_wavs(names, workdir: Path) -> dict:
    """prepare_webrtc_audio_timeline on a copy of each WAV (api_server's upload path)."""
    import numpy as np
    import shutil
    import wave
    from scripts.webrtc_audio_timeline import prepare_webrtc_audio_timeline
    out = {}
    workdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = WAVS[name]
        dst = workdir / f"{name}.wav"
        if src.startswith("@zeros:"):
            seconds = float(src.split(":", 1)[1])
            with wave.open(str(dst), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(np.zeros(int(24000 * seconds), np.int16).tobytes())
        else:
            shutil.copyfile(src, dst)
        timeline = prepare_webrtc_audio_timeline(dst)
        out[name] = {"media_path": str(timeline.media_path), "exact_silence": bool(timeline.exact_silence),
                     "media_duration_s": timeline.media_duration_seconds,
                     "sha256": hashlib.sha256(Path(timeline.media_path).read_bytes()).hexdigest()}
    return out


# ------------------------------------------------------------------ scheduler loading
def git_read(*args) -> str:
    """Read-only git on --repo (never takes the index lock)."""
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *args], check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"<git {' '.join(args)} failed: {exc}>"


def repo_provenance() -> dict:
    status = git_read("status", "--porcelain", "--untracked-files=no")
    return {"repo": str(ROOT), "harness": str(Path(__file__).resolve()), "foreign_repo": FOREIGN_REPO,
            "head": git_read("rev-parse", "HEAD").strip(),
            "branch": git_read("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "modified_tracked_files": [line[3:] for line in status.splitlines() if line.strip()]}


def load_scheduler_class(source: str, scratch: Path):
    if source in ("repo", "worktree"):
        path = ROOT / "scripts/hls_gpu_scheduler.py"
        mod_name = "scripts.hls_gpu_scheduler"
        import scripts.hls_gpu_scheduler as module
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"imported {module.__file__}, expected {path} (sys.path leak)")
    else:
        if source in ("head", "base"):
            rev = "HEAD" if source == "head" else git_read("merge-base", "HEAD", "main").strip()
            if rev.startswith("<git "):
                raise RuntimeError(rev)
            text = git_read("show", f"{rev}:scripts/hls_gpu_scheduler.py")
            if text.startswith("<git "):
                raise RuntimeError(text)
            tag = hashlib.sha1(f"{ROOT}:{rev}".encode()).hexdigest()[:8]
            path = scratch / f"hls_gpu_scheduler_{source}_{tag}.py"
            path.write_text(text)
            mod_name = f"hls_gpu_scheduler_{source}"
        else:
            path = Path(source).resolve()
            mod_name = "hls_gpu_scheduler_" + hashlib.sha1(str(path).encode()).hexdigest()[:8]
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return module.HLSGPUStreamScheduler, str(path), sha


def make_scheduler(cls, manager):
    """Same constructor arguments api_server.startup_event passes."""
    return cls(
        manager=manager, hls_session_manager=None,
        max_combined_batch_size=env_int("HLS_SCHEDULER_MAX_BATCH", 8),
        startup_slice_size=env_int("HLS_SCHEDULER_STARTUP_SLICE_SIZE", 2),
        aggressive_fill_max_active_jobs=env_int("HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS", 4),
        prep_workers=env_int("HLS_PREP_WORKERS", 2), compose_workers=env_int("HLS_COMPOSE_WORKERS", 2),
        encode_workers=env_int("HLS_ENCODE_WORKERS", 2), max_pending_jobs=env_int("HLS_MAX_PENDING_JOBS", 16),
        startup_chunk_duration_seconds=env_float("HLS_STARTUP_CHUNK_DURATION_SECONDS", 0.5),
        startup_chunk_count=env_int("HLS_STARTUP_CHUNK_COUNT", 1),
    )


class EnvOverride:
    def __init__(self, overrides: dict):
        self.overrides = overrides
        self.saved = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def parse_run(spec: str) -> dict:
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise SystemExit(f"--run needs LABEL:SOURCE[:K=V,...], got {spec!r}")
    overrides, harness = {}, {}
    if len(parts) == 3 and parts[2]:
        for item in parts[2].split(","):
            k, v = item.split("=", 1)
            k, v = k.strip(), v.strip()
            (harness if k.startswith("REPLAY_") else overrides)[k] = v
    return {"label": parts[0], "source": parts[1], "env": overrides, "harness": harness}


# ------------------------------------------------------------------------- the runs
class JobHandle:
    def __init__(self, job_id, session, recorder):
        self.job_id = job_id
        self.session = session
        self.recorder = recorder
        self.done = threading.Event()
        self.status = None
        self.error = None
        self.turns = 0
        self.request_id = None
        self.cancel_event = None
        self.submitted_at = None
        self.first_frame_at = None
        self.total_frames = 0
        # --paced consumer simulation (webrtc_tracks strict FIFO)
        self.queue_frames = 0
        self.played = 0
        self.held = 0
        self.playing = False
        self.blocked_s = 0.0
        self.qlock = threading.Condition()


def submit_turn(scheduler, handle: JobHandle, job: dict, wavs: dict, sink, run_label: str, plan: str | None,
                registry: dict | None = None):
    handle.turns += 1
    turn_id = f"{run_label}_{handle.job_id}_t{handle.turns}"
    handle.request_id = f"{handle.session.avatar_id}_webrtc_{turn_id}"
    if registry is not None:
        registry[handle.request_id] = handle
    stage_turn(handle.session, turn_id, plan)
    handle.cancel_event = threading.Event()
    handle.done.clear()
    wav = wavs[job["wav"]]

    def on_complete(status, error=None):
        handle.status = status
        handle.error = error
        finish_turn(handle.session)
        handle.done.set()

    def frame_batch_callback(frames, start_frame_idx, total_frames):
        sink(handle, frames, start_frame_idx, total_frames)

    accepted = scheduler.submit_webrtc_stream(
        session=handle.session, request_id=handle.request_id, audio_path=wav["media_path"],
        exact_silence=wav["exact_silence"], generation_fps=handle.session.fps, cancel_event=handle.cancel_event,
        completion_future=Future(), main_loop=InlineLoop(), frame_callback=lambda *a: None,
        frame_batch_callback=frame_batch_callback, generation_complete_callback=on_complete)
    if not accepted:
        raise RuntimeError(f"scheduler refused {handle.request_id}")
    handle.submitted_at = time.time()


def install_face_tap(scheduler, handles_by_request: dict):
    original = scheduler._dispatch_compose_batch

    def tapped(job, batch_frames, start_frame_idx, *args, **kwargs):
        handle = handles_by_request.get(job.request_id)
        if handle is not None and handle.recorder is not None:
            handle.recorder.record_faces(start_frame_idx, batch_frames)
        return original(job, batch_frames, start_frame_idx, *args, **kwargs)

    scheduler._dispatch_compose_batch = tapped


def capacity_stats(scheduler):
    fn = getattr(scheduler, "get_capacity_stats", None)
    if fn is None:
        return None
    try:
        return fn(include_batches=True)
    except TypeError:
        return fn()


def run_golden(cls, manager, run: dict, jobs: list, wavs: dict, args, hash_pool) -> dict:
    handles, by_request = [], {}
    writers = {}
    video_jobs = set(filter(None, (args.video_jobs or "").split(",")))
    for job in jobs:
        session = build_session(job["identity"], f"{run['label']}_{job['id']}", job["idle_frame"])
        if (run.get("harness") or {}).get("REPLAY_CONSUMER_DEPTH", args.consumer_depth) == "zero":
            # A consumer that is never behind: EDF run-ahead caps never throttle,
            # so batch composition does not depend on wall-clock timing.
            session.webrtc_playback_queue_frames = lambda: 0
        writer = None
        if args.video_dir and job["id"] in video_jobs:
            writer = LosslessWriter(Path(args.video_dir) / f"{run['label']}_{job['id']}", 20,
                                    int(args.video_seconds * 20), crf=args.video_crf,
                                    lossless=args.video_lossless)
            writers[job["id"]] = writer
        rec = JobRecorder(job["id"], hash_pool, True, True, not args.no_yuv, writer)
        handles.append(JobHandle(job["id"], session, rec))

    def sink(handle, frames, start_frame_idx, total_frames):
        now = time.time()
        if handle.first_frame_at is None:
            handle.first_frame_at = now
        handle.total_frames = total_frames
        meta = motion_metadata(handle.session, start_frame_idx - 1, len(frames), total_frames)
        if meta:
            for i, m in enumerate(meta):
                handle.recorder.pose_meta[start_frame_idx - 1 + i] = m["pose_id"]
        handle.recorder.record_frames(frames, start_frame_idx, now)

    scheduler = make_scheduler(cls, manager)
    install_face_tap(scheduler, by_request)
    scheduler.start()
    t0 = time.time()
    try:
        for handle, job in zip(handles, jobs):
            submit_turn(scheduler, handle, job, wavs, sink, run["label"], job.get("plan"), by_request)
            if args.stagger_ms:
                time.sleep(args.stagger_ms / 1000.0)
        deadline = time.time() + args.timeout_s
        for handle in handles:
            if not handle.done.wait(max(0.1, deadline - time.time())):
                raise TimeoutError(f"{handle.job_id} did not finish in {args.timeout_s}s")
        wall = time.time() - t0
        cap = capacity_stats(scheduler)
        if cap is not None:
            cap = {k: v for k, v in cap.items() if k != "batches"} | {
                "batches_tail": (cap.get("batches") or [])[-64:]}
        sched_stats = scheduler.get_stats()
    finally:
        scheduler.shutdown()
        for w in writers.values():
            w.close()
    results = {}
    total_frames = 0
    for handle in handles:
        r = handle.recorder.result()
        r.update({"status": handle.status, "error": handle.error, "request_id": handle.request_id,
                  "identity": handle.session.avatar_id, "total_frames": handle.total_frames,
                  "first_frame_latency_s": (round(handle.first_frame_at - handle.submitted_at, 3)
                                            if handle.first_frame_at else None)})
        total_frames += r["frame_count"]
        results[handle.job_id] = r
        try:
            handle.session.live_pose_router.close()
        except Exception:
            pass
    latencies = sorted(r["first_frame_latency_s"] for r in results.values() if r["first_frame_latency_s"] is not None)
    return {"wall_s": round(wall, 3), "frames": total_frames, "fps": round(total_frames / wall, 2),
            "jobs": results, "capacity": cap, "event_vs_host": event_vs_host(cap),
            "scheduler_pipeline": sched_stats.get("pipeline"),
            "first_frame_latency_s": {"max": latencies[-1] if latencies else None,
                                      "median": latencies[len(latencies) // 2] if latencies else None},
            "videos": {k: str(w.path) for k, w in writers.items()}}


def event_vs_host(cap: dict | None) -> dict | None:
    """Plan item 0.2 gate input: CUDA-event stage totals vs host (sync) totals.

    Only meaningful at depth 1 with HLS_GPU_STAGE_SYNC_TIMING=1 (host clocks
    bracket host syncs). vae compares event vae+d2h with host vae (the host
    decode includes the D2H)."""
    if not cap or not cap.get("enabled"):
        return None
    t = cap.get("totals") or {}
    out = {"stage_sync_on": bool((cap.get("config") or {}).get("gpu_stage_sync_timing")),
           "depth": (cap.get("config") or {}).get("pipeline_depth"), "gpu_batches": t.get("gpu_batches")}
    for name, ev, host in (("h2d", t.get("h2d_ms", 0.0), t.get("host_copy_ms", 0.0)),
                           ("unet", t.get("unet_ms", 0.0), t.get("host_unet_ms", 0.0)),
                           ("vae", t.get("vae_ms", 0.0) + t.get("d2h_ms", 0.0), t.get("host_vae_ms", 0.0))):
        out[name] = {"event_ms": round(ev, 3), "host_ms": round(host, 3),
                     "rel_diff": round((ev - host) / host, 5) if host else None}
    return out


class SmiSampler:
    FIELDS = "utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used"

    def __init__(self, interval_ms=200):
        self.rows = []
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits",
                                      f"-lms={interval_ms}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.rows.append((time.time(), [float(x) for x in line.strip().split(",")]))
            except ValueError:
                continue

    def summary(self, t0, t1):
        rows = [r for t, r in self.rows if t0 <= t <= t1]
        if not rows:
            return {}
        cols = list(zip(*rows))
        names = ["util_pct", "sm_mhz", "power_w", "temp_c", "mem_mib"]
        return {n: round(sum(c) / len(c), 2) for n, c in zip(names, cols)} | {"samples": len(rows)}

    def close(self):
        self.proc.terminate()


def run_speed(cls, manager, run: dict, n_jobs: int, wavs: dict, args) -> dict:
    identities = [x for x in args.speed_identities.split(",") if x]
    handles, jobs = [], []
    for i in range(n_jobs):
        job = {"id": f"s{i:02d}", "identity": identities[i % len(identities)],
               "wav": SPEED_WAVS[i % len(SPEED_WAVS)], "idle_frame": (37 * i) % 240}
        jobs.append(job)
        session = build_session(job["identity"], f"{run['label']}_n{n_jobs}_{job['id']}", job["idle_frame"])
        handles.append(JobHandle(job["id"], session, None))
    fps = 20
    prebuffer_frames = int(round(2.0 * fps))
    max_queue = 400
    per_second: dict = {}
    counters = {"frames": 0}
    count_lock = threading.Lock()
    stop = threading.Event()

    def sink(handle, frames, start_frame_idx, total_frames):
        now = time.time()
        n = len(frames)
        motion_metadata(handle.session, start_frame_idx - 1, n, total_frames)
        with count_lock:
            counters["frames"] += n
            sec = int(now)
            per_second[sec] = per_second.get(sec, 0) + n
        if handle.first_frame_at is None:
            handle.first_frame_at = now
        if args.paced:
            t_block = time.time()
            with handle.qlock:
                while handle.queue_frames + n > max_queue and not stop.is_set():
                    handle.qlock.wait(0.05)
                handle.queue_frames += n
                if handle.queue_frames >= prebuffer_frames:
                    handle.playing = True
            handle.blocked_s += time.time() - t_block

    if not args.paced:
        # Null sink: an infinitely fast consumer, never behind (EDF run-ahead
        # caps do not throttle an unpaced throughput run).
        for h in handles:
            h.session.webrtc_playback_queue_frames = lambda: 0
    if args.paced:
        for h in handles:
            h.session.webrtc_playback_queue_frames = (lambda hh=h: hh.queue_frames)

        def consumer():
            period = 1.0 / fps
            next_tick = time.time()
            while not stop.is_set():
                next_tick += period
                for h in handles:
                    with h.qlock:
                        if not h.playing:
                            continue
                        if h.queue_frames > 0:
                            h.queue_frames -= 1
                            h.played += 1
                            h.qlock.notify_all()
                        elif not h.done.is_set():
                            h.held += 1
                        else:
                            h.playing = False
                delay = next_tick - time.time()
                if delay > 0:
                    time.sleep(delay)
        consumer_thread = threading.Thread(target=consumer, daemon=True, name="paced-consumer")
        consumer_thread.start()

    scheduler = make_scheduler(cls, manager)
    scheduler.start()
    sched_tid = getattr(scheduler.scheduler_thread, "native_id", None)
    smi = SmiSampler(200)
    t_start = time.time()
    t_measure0 = t_start + args.warmup_s
    t_measure1 = t_measure0 + args.seconds
    cpu0 = cpu1 = sched0 = sched1 = None
    frames0 = frames1 = None
    cap0 = cap1 = None
    try:
        for h, job in zip(handles, jobs):
            submit_turn(scheduler, h, job, wavs, sink, run["label"], None)
        while time.time() < t_measure1:
            now = time.time()
            if cpu0 is None and now >= t_measure0:
                cpu0, sched0 = proc_cpu_s(), thread_cpu_s(sched_tid)
                with count_lock:
                    frames0 = counters["frames"]
                cap0 = capacity_stats(scheduler)
            for h, job in zip(handles, jobs):
                if h.done.is_set():
                    if h.status not in ("completed",):
                        raise RuntimeError(f"{h.job_id} ended {h.status}: {h.error}")
                    submit_turn(scheduler, h, job, wavs, sink, run["label"], None)
            time.sleep(0.01)
        cpu1, sched1 = proc_cpu_s(), thread_cpu_s(sched_tid)
        with count_lock:
            frames1 = counters["frames"]
        cap1 = capacity_stats(scheduler)
    finally:
        stop.set()
        for h in handles:
            if h.cancel_event is not None:
                h.cancel_event.set()
        for h in handles:
            h.done.wait(30)
        scheduler.shutdown()
        smi.close()
        for h in handles:
            try:
                h.session.live_pose_router.close()
            except Exception:
                pass
    window = t_measure1 - t_measure0
    series = [per_second.get(s, 0) for s in range(int(t_measure0) + 1, int(t_measure1))]
    out = {
        "n_jobs": n_jobs, "window_s": round(window, 2), "warmup_s": args.warmup_s,
        "generated_frames": frames1 - frames0, "generated_fps": round((frames1 - frames0) / window, 2),
        "per_second_fps_min": min(series) if series else None, "per_second_fps_max": max(series) if series else None,
        "process_cpu_cores": round((cpu1 - cpu0) / window, 3),
        "scheduler_thread_cpu_cores": round((sched1 - sched0) / window, 3),
        "smi": smi.summary(t_measure0, t_measure1),
        "turns": {h.job_id: h.turns for h in handles},
        "mem_available_gb_end": round(mem_available_gb(), 2), "rss_gb_end": round(rss_gb(), 2),
    }
    if args.paced:
        out["paced"] = {"held_frames": {h.job_id: h.held for h in handles},
                        "held_total": sum(h.held for h in handles),
                        "played_total": sum(h.played for h in handles),
                        "callback_blocked_s": {h.job_id: round(h.blocked_s, 2) for h in handles}}
    if cap0 is not None and cap1 is not None:
        out["capacity_window"] = diff_capacity(cap0, cap1)
        out["capacity_end"] = {k: v for k, v in cap1.items() if k != "batches"}
    return out


def diff_capacity(c0: dict, c1: dict) -> dict:
    t0, t1 = c0.get("totals", {}), c1.get("totals", {})
    d = {k: t1.get(k, 0) - t0.get(k, 0) for k in t1 if isinstance(t1.get(k), (int, float))}
    wall = d.get("wall_s", 0) or 1e-9
    batches = d.get("batches", 0) or 1
    return {
        "batches": d.get("batches"),
        "gpu_busy_fraction": round(d.get("gpu_span_ms", 0) / 1000.0 / wall, 4),
        "gpu_ms_per_batch": round(d.get("gpu_span_ms", 0) / batches, 3),
        "idle_gap_ms_per_batch": round(d.get("idle_gap_ms", 0) / batches, 3),
        "callback_ms_per_batch": round(d.get("callback_ms", 0) / batches, 3),
        "feeder_cpu_ms_per_batch": round(d.get("feeder_cpu_ms", 0) / batches, 3),
        "fill": round(d.get("actual_frames", 0) / max(1, d.get("padded_frames", 0)), 4),
        "jobs_per_batch": round(d.get("jobs", 0) / batches, 3),
        "gpu_frames": d.get("actual_frames"),
        "gpu_fps": round(d.get("actual_frames", 0) / wall, 2),
        "raw_frames_skipped": d.get("raw_frames"),
        "unet_ms_per_batch": round(d.get("unet_ms", 0) / batches, 3),
        "vae_ms_per_batch": round(d.get("vae_ms", 0) / batches, 3),
        "h2d_ms_per_batch": round(d.get("h2d_ms", 0) / batches, 3),
        "d2h_ms_per_batch": round(d.get("d2h_ms", 0) / batches, 3),
    }


# -------------------------------------------------------------------------- compare
def compare(a: dict, b: dict, label_a="A", label_b="B", subset_ok: bool | None = None) -> dict:
    """Frame-by-frame comparison of two golden runs (faces are compared only where
    both computed a face; frames marked RAW_NO_GPU had their GPU output discarded).

    subset_ok (default: b ran a REPLAY_JOBS subset) compares only b's jobs."""
    if subset_ok is None:
        subset_ok = bool((b.get("harness_options") or {}).get("REPLAY_JOBS"))
    report = {"a": label_a, "b": label_b, "jobs": {}, "identical": True, "subset": bool(subset_ok)}
    job_ids = set(b["jobs"]) if subset_ok else set(a["jobs"]) | set(b["jobs"])
    for job_id in sorted(job_ids):
        ja, jb = a["jobs"].get(job_id), b["jobs"].get(job_id)
        if ja is None or jb is None:
            report["jobs"][job_id] = {"missing_in": label_a if ja is None else label_b}
            report["identical"] = False
            continue
        fa, fb = ja["frames"], jb["frames"]
        frame_diff = [i for i in range(max(len(fa), len(fb))) if i >= len(fa) or i >= len(fb) or fa[i] != fb[i]]
        ya, yb = ja.get("yuv") or [], jb.get("yuv") or []
        yuv_diff = [i for i in range(max(len(ya), len(yb))) if i >= len(ya) or i >= len(yb) or ya[i] != yb[i]]
        face_pairs = [(x, y) for x, y in zip(ja["faces"], jb["faces"]) if "RAW_NO_GPU" not in (x, y)]
        face_diff = sum(1 for x, y in face_pairs if x != y)
        ok = (not frame_diff and not yuv_diff and face_diff == 0 and len(ja["faces"]) == len(jb["faces"])
              and not ja["order_errors"] and not jb["order_errors"]
              and not jb.get("yuv_contract_mismatches")
              and ja["status"] == jb["status"] == "completed")
        report["jobs"][job_id] = {
            "frames": len(fa), "frames_b": len(fb), "frame_mismatches": len(frame_diff),
            "first_frame_mismatch": frame_diff[0] if frame_diff else None,
            "yuv_mismatches": len(yuv_diff), "faces_compared": len(face_pairs), "face_mismatches": face_diff,
            "raw_no_gpu_faces_b": jb.get("raw_no_gpu_faces", 0),
            "order_errors": len(ja["order_errors"]) + len(jb["order_errors"]),
            "yuv_contract_mismatches_b": len(jb.get("yuv_contract_mismatches") or []),
            "identical": ok,
        }
        report["identical"] &= ok
    report["frames_total"] = sum(j.get("frames", 0) for j in report["jobs"].values())
    return report


def strip_hashes(run_result: dict) -> dict:
    out = json.loads(json.dumps(run_result))
    for j in out.get("jobs", {}).values():
        for k in ("faces", "frames", "yuv", "pose_ids"):
            j.pop(k, None)
    return out


# ------------------------------------------------------------------------ selfcheck
def selfcheck(args) -> int:
    """CPU-only preflight (no CUDA, no model): the modules the replay uses import
    from --repo, the fixtures exist, WAV prep works, the scheduler sources load."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    report = {"provenance": repo_provenance(), "checks": {}}
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            report["checks"][name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            ok = False
            report["checks"][name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    def modules():
        import importlib
        out = {}
        for name in ("scripts.hls_gpu_scheduler", "scripts.webrtc_manager", "scripts.webrtc_pose_router",
                     "scripts.pose_protocol", "scripts.webrtc_audio_timeline"):
            path = Path(importlib.import_module(name).__file__).resolve()
            if not str(path).startswith(str(ROOT) + os.sep):
                raise RuntimeError(f"{name} imported from {path}, not {ROOT}")
            out[name] = str(path)
        return out

    def sources():
        scratch = Path(os.getenv("REPLAY_SCRATCH", "/tmp/claude-0/replay_scheduler"))
        scratch.mkdir(parents=True, exist_ok=True)
        out = {}
        for source in ("repo", "head", "base"):
            _cls, path, sha = load_scheduler_class(source, scratch)
            out[source] = {"path": path, "sha256": sha}
        out["repo_equals_head"] = out["repo"]["sha256"] == out["head"]["sha256"]
        return out

    def fixtures():
        missing = [p for p in [str(LIVE_ENV_FILE), IDENTITIES["bob"]["pose_set"], str(BOB_DIR / "motion-atlas.json")]
                   + [w for w in WAVS.values() if not w.startswith("@")] if not Path(p).exists()]
        for spec in IDENTITIES.values():
            if not (ROOT / "results/v15/avatars" / spec["avatar_id"]).is_dir():
                missing.append(spec["avatar_id"])
        if missing:
            raise FileNotFoundError(missing)
        return "all present"

    def wav_prep():
        scratch = Path(os.getenv("REPLAY_SCRATCH", "/tmp/claude-0/replay_scheduler")) / "selfcheck_wav"
        return prepare_wavs(["zero3", "bob1"], scratch)

    check("modules_from_repo", modules)
    check("scheduler_sources", sources)
    check("fixtures", fixtures)
    check("wav_prep", wav_prep)
    import torch
    report["cuda_initialized"] = bool(torch.cuda.is_initialized())
    ok &= not report["cuda_initialized"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(("PASS" if ok else "FAIL") + f" selfcheck repo={ROOT} "
          + " ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in report["checks"].items()), flush=True)
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["golden", "speed", "compare", "selfcheck"], default="golden")
    ap.add_argument("--run", action="append", default=[], help="LABEL:SOURCE[:K=V,...] (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", default=None, help="extra env file layered over the live env")
    ap.add_argument("--jobs", default="all", help="comma list of golden job ids, or 'all'")
    ap.add_argument("--identities", default="bob,jp,latfh1", help="golden identities to include")
    ap.add_argument("--stagger-ms", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--no-yuv", action="store_true", help="skip the PyAV yuv420p hash")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--video-jobs", default="bob_t1,jp_d10")
    ap.add_argument("--video-seconds", type=float, default=12.0)
    ap.add_argument("--video-crf", type=int, default=MAX_VIDEO_CRF,
                    help=f"libx264 crf of the mp4 dumps (capped at {MAX_VIDEO_CRF})")
    ap.add_argument("--video-lossless", action="store_true", help="libx264rgb -qp 0 mkv instead of crf mp4")
    ap.add_argument("--repo", default=str(HARNESS_ROOT),
                    help="tree under test (resolved before import; see docstring)")
    ap.add_argument("--consumer-depth", choices=["zero", "none"], default="zero",
                    help="golden: 'zero' = consumer never behind (deterministic EDF); 'none' = no depth hook "
                         "(EDF falls back to its wall-clock estimate)")
    ap.add_argument("--hash-workers", type=int, default=6)
    ap.add_argument("--n-jobs", default="8,12,16", help="speed: comma list of N")
    ap.add_argument("--seconds", type=float, default=60.0, help="speed: measured window")
    ap.add_argument("--warmup-s", type=float, default=15.0)
    ap.add_argument("--paced", action="store_true", help="speed: simulate the 20 fps strict-FIFO consumer")
    ap.add_argument("--speed-identities", default="bob,jp")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()

    if Path(args.repo).resolve() != ROOT:
        raise SystemExit(f"--repo resolved to {ROOT} at import but {args.repo} at parse time")
    if args.video_crf > MAX_VIDEO_CRF:
        print(f"[replay] --video-crf {args.video_crf} capped at {MAX_VIDEO_CRF}", flush=True)
    if args.mode == "compare":
        a, b = (json.loads(Path(p).read_text()) for p in args.compare)
        ra = a["runs"][0] if "runs" in a else a
        reports = []
        runs_b = b["runs"] if "runs" in b else [b]
        for rb in runs_b:
            reports.append(compare(ra, rb, ra.get("label", "A"), rb.get("label", "B")))
        Path(args.out).write_text(json.dumps({"compare": reports}, indent=1))
        for r in reports:
            print(f"{r['a']} vs {r['b']}: identical={r['identical']} frames={r['frames_total']}")
        return 0 if all(r["identical"] for r in reports) else 1

    os.chdir(ROOT)
    if args.mode == "selfcheck":
        return selfcheck(args)
    env_report = apply_env(args.overlay)
    from scripts.runtime_cpu_tuning import apply_cpu_tuning_early
    apply_cpu_tuning_early("api_server")
    import torch  # noqa: F401  (after the env layer, like api_server)
    from argparse import Namespace
    from scripts.runtime_cpu_tuning import apply_cpu_tuning_runtime
    apply_cpu_tuning_runtime("api_server")
    gpu_defaults = apply_gpu_aware_runtime_defaults(0)
    from scripts.avatar_manager_parallel import ParallelAvatarManager

    scratch = Path(os.getenv("REPLAY_SCRATCH", "/tmp/claude-0/replay_scheduler"))
    scratch.mkdir(parents=True, exist_ok=True)
    runs = [parse_run(r) for r in (args.run or ["base:head"])]

    if args.mode == "golden":
        wanted = set(args.identities.split(","))
        jobs = [j for j in GOLDEN_JOBS if j["identity"] in wanted and (args.jobs == "all" or j["id"] in args.jobs.split(","))]
        wav_names = sorted({j["wav"] for j in jobs})
    else:
        jobs = []
        wav_names = sorted(set(SPEED_WAVS))
    wavs = prepare_wavs(wav_names, scratch / f"wav_{os.getpid()}")

    t_load = time.time()
    margs = Namespace(version="v15", gpu_id=0, vae_type="sd-vae", unet_config="./models/musetalkV15/musetalk.json",
                      unet_model_path="./models/musetalkV15/unet.pth", whisper_dir="./models/whisper",
                      left_cheek_width=90, right_cheek_width=90, extra_margin=10, parsing_mode="jaw",
                      audio_padding_length_left=2, audio_padding_length_right=2, result_dir="./results",
                      ffmpeg_path="./ffmpeg-4.4-amd64-static/")
    manager = ParallelAvatarManager(margs, max_concurrent_inferences=5)
    load_s = time.time() - t_load
    leaked = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                     if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(
                         str(HARNESS_ROOT) + os.sep) and Path(m.__file__).resolve() != Path(__file__).resolve()}
                    ) if FOREIGN_REPO else []
    if leaked:
        raise SystemExit(f"modules imported from the harness tree instead of --repo {ROOT}: {leaked[:5]}")
    print(f"[replay] models loaded in {load_s:.1f}s rss={rss_gb():.2f}GB avail={mem_available_gb():.2f}GB", flush=True)

    out = {"schema": "replay_scheduler_exactness_v2", "mode": args.mode, "argv": sys.argv,
           "provenance": repo_provenance(),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "env": env_report,
           "gpu_defaults_applied": gpu_defaults, "wavs": wavs, "model_load_s": round(load_s, 1),
           "unet_backend": getattr(manager, "unet_backend_name", None),
           "vae_backend": getattr(manager, "vae_decode_backend_name", None), "runs": []}
    hash_pool = ThreadPoolExecutor(max_workers=args.hash_workers, thread_name_prefix="replay-hash")
    exit_code = 0
    for run in runs:
        with EnvOverride(run["env"]):
            cls, path, sha = load_scheduler_class(run["source"], scratch)
            print(f"[replay] run {run['label']} source={run['source']} sha={sha[:12]} env={run['env']}", flush=True)
            entry = {"label": run["label"], "source": run["source"], "scheduler_path": path, "scheduler_sha256": sha,
                     "env_overrides": run["env"], "harness_options": run["harness"],
                     "mem_available_gb_start": round(mem_available_gb(), 2)}
            try:
                if args.mode == "golden":
                    run_jobs = jobs
                    if run["harness"].get("REPLAY_JOBS"):
                        wanted_ids = set(run["harness"]["REPLAY_JOBS"].split("+"))
                        run_jobs = [j for j in jobs if j["id"] in wanted_ids]
                        if not run_jobs:
                            raise RuntimeError(f"REPLAY_JOBS matched no golden job: {wanted_ids}")
                    entry.update(run_golden(cls, manager, run, run_jobs, wavs, args, hash_pool))
                    bad = [k for k, v in entry["jobs"].items() if v["status"] != "completed" or v["order_errors"]]
                    entry["all_completed_in_order"] = not bad
                    print(f"[replay] {run['label']}: {entry['frames']} frames in {entry['wall_s']}s "
                          f"({entry['fps']} fps) bad={bad}", flush=True)
                else:
                    entry["speed"] = []
                    for n in [int(x) for x in args.n_jobs.split(",") if x]:
                        res = run_speed(cls, manager, run, n, wavs, args)
                        entry["speed"].append(res)
                        print(f"[replay] {run['label']} N={n}: {res['generated_fps']} fps "
                              f"util={res['smi'].get('util_pct')} cap={res.get('capacity_window')}", flush=True)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
                print(entry["traceback"], flush=True)
                exit_code = 2
            entry["mem_available_gb_end"] = round(mem_available_gb(), 2)
            entry["rss_gb_end"] = round(rss_gb(), 2)
            out["runs"].append(entry)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    if args.mode == "golden" and len(out["runs"]) > 1 and exit_code == 0:
        ref = out["runs"][0]
        out["compare_vs_first"] = [compare(ref, r, ref["label"], r["label"]) for r in out["runs"][1:]]
        for r in out["compare_vs_first"]:
            print(f"[replay] {r['a']} vs {r['b']}: identical={r['identical']} frames={r['frames_total']} "
                  + " ".join(f"{k}:{v['frame_mismatches']}/{v['face_mismatches']}" for k, v in r["jobs"].items()),
                  flush=True)
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(out, indent=1))
    hash_pool.shutdown(wait=True)
    # A short summary without the per-frame hash lists (for docs/greps).
    summary = dict(out)
    summary["runs"] = [strip_hashes(r) for r in out["runs"]]
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    os._exit(exit_code)  # skip interpreter teardown of TRT/compiled graphs


if __name__ == "__main__":
    sys.exit(main())
