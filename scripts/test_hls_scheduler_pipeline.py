"""CPU-only tests for the 300 fps scheduler levers in scripts/hls_gpu_scheduler.py.

Plan items 0.2, 1.1, 1.4, 1.9, 1.10, the crossfade-copy skip and the
WEBRTC_YUV_IN_COMPOSE producer side (docs/musetalk_4070s_300fps_plan_2026-09-27.md).

GPU calls are mocked (a per-sample UNet stand-in and a fake TAESD backend behind
the REAL musetalk.models.vae.VAE.decode_latents), composition uses the REAL
APIAvatar.compose_frame, and every scenario is compared frame by frame against
the baseline scheduler (today's code: merge-base(HEAD, main), or
HLS_TEST_BASELINE_REV) run on the same fixtures.

Run from the repo root with CUDA hidden:
  CUDA_VISIBLE_DEVICES= python -m unittest scripts.test_hls_scheduler_pipeline -v
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import musetalk.models.vae as vae_mod  # noqa: E402
import scripts.hls_gpu_scheduler as wt  # noqa: E402
from musetalk.utils.blending import prepare_image_blending_plan  # noqa: E402
from scripts.api_avatar import APIAvatar as _APIAvatarDecorated  # noqa: E402


def _unwrap_class(obj):
    """api_avatar decorates the class with @torch.no_grad(), which returns a
    constructor lambda; find the real class (compose_frame is its method)."""
    seen = []
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, type):
            return item
        if id(item) in seen:
            continue
        seen.append(id(item))
        stack.extend(c.cell_contents for c in (getattr(item, "__closure__", None) or ()))
        if getattr(item, "__wrapped__", None) is not None:
            stack.append(item.__wrapped__)
    raise RuntimeError("APIAvatar class not found")


APIAvatar = _unwrap_class(_APIAvatarDecorated)

vae_mod.MUSETALK_VAE_DECODE_TIMING_LOG_INTERVAL = 0
_HEAD_MODULE = None
_TMP = tempfile.TemporaryDirectory()
IDLE, TALK, SMILE = "neutral_resting", "speaking_direct", "light_smile"


def baseline_rev() -> str:
    """The scheduler 'today': HLS_TEST_BASELINE_REV, else merge-base(HEAD, main)
    (the commit this branch forked from, stable across commits on the branch)."""
    rev = os.environ.get("HLS_TEST_BASELINE_REV", "").strip()
    if rev:
        return rev
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), "merge-base", "HEAD", "main"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "HEAD"


def head_module():
    """The baseline revision's scheduler (the code as it was before these levers)."""
    global _HEAD_MODULE
    if _HEAD_MODULE is None:
        text = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), "show",
                               f"{baseline_rev()}:scripts/hls_gpu_scheduler.py"],
                              check=True, capture_output=True, text=True).stdout
        path = Path(_TMP.name) / "hls_gpu_scheduler_head.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location("hls_gpu_scheduler_head_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _HEAD_MODULE = module
    return _HEAD_MODULE


def sha(array) -> str:
    a = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(f"{a.shape}|{a.dtype}|".encode() + a.tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ mocked GPU
class FakeUNet:
    """Per-sample deterministic stand-in: a row's output depends only on its inputs."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def model(self, latent, timesteps, encoder_hidden_states=None):
        with self.lock:
            self.calls.append(encoder_hidden_states[:, 0, 0].detach().cpu().clone())
        # Bounded, so faces vary per frame (conditioning carries +1000*seed row tags).
        sample = latent[:, :4] * 0.7 + torch.sin(encoder_hidden_states.mean(dim=(1, 2)) * 3.1).view(-1, 1, 1, 1)
        return SimpleNamespace(sample=sample)


class FakeTaesdBackend:
    name = "fake_taesd"
    fused_post_enabled = False

    def decode(self, latents, scaling_factor, output_dtype):
        return torch.sigmoid(F.interpolate(latents[:, :3].float(), size=(16, 16), mode="nearest") * 2.0)


class FakeFusedBackend(FakeTaesdBackend):
    fused_post_enabled = True

    def decode_bgr_u8(self, latents):
        image = self.decode(latents, 1.0, torch.float32)
        return (image.float().mul(255).round().clamp_(0, 255).to(torch.uint8)
                .flip(1).permute(0, 2, 3, 1).contiguous())


def make_vae(backend=None):
    vae = vae_mod.VAE.__new__(vae_mod.VAE)
    vae._decode_backend = backend or FakeTaesdBackend()
    vae._decode_backend_name = vae._decode_backend.name
    vae.scaling_factor = 0.18215
    vae.runtime_dtype = torch.float32
    vae.vae = SimpleNamespace(dtype=torch.float32)
    vae._decode_timing_count = 0
    vae._decode_timing_tensor_total_s = 0.0
    vae._decode_timing_post_total_s = 0.0
    vae._decode_timing_total_s = 0.0
    vae._decode_timing_max_total_s = 0.0
    return vae


def make_manager(vae=None):
    return SimpleNamespace(
        device=torch.device("cpu"), unet=FakeUNet(), vae=vae or make_vae(),
        timesteps=torch.tensor([0]), gpu_memory=SimpleNamespace(allocate=lambda n: contextlib.nullcontext()),
        request_lock=threading.Lock(), active_requests={}, models_compiled=False,
        vae_dtype=torch.float32, unet_dtype=torch.float32,
    )


# ------------------------------------------------------------------ fixtures
BBOX = (8, 10, 32, 34)
CROP = (4, 6, 36, 38)


def make_avatar(seed: int, cycle: int = 12, plans: bool = True, compose_delay=None):
    rng = np.random.default_rng(seed)
    avatar = APIAvatar.__new__(APIAvatar)
    avatar.frame_list_cycle = [rng.integers(0, 256, (40, 48, 3), dtype=np.uint8) for _ in range(cycle)]
    for frame in avatar.frame_list_cycle:
        frame.setflags(write=False)
    avatar.coord_list_cycle = [BBOX] * cycle
    avatar.mask_coords_list_cycle = [CROP] * cycle
    avatar.mask_list_cycle = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(cycle)]
    avatar._compose_plan_cycle = (
        [prepare_image_blending_plan(f.shape, BBOX, m, CROP)
         for f, m in zip(avatar.frame_list_cycle, avatar.mask_list_cycle)] if plans else []
    )
    avatar._source_mouth_blend = False
    avatar._side_jaw_blend = False
    gen = torch.Generator().manual_seed(seed)
    avatar.input_latent_cycle_batch_tensor = torch.randn((cycle, 8, 4, 4), generator=gen)
    avatar.input_latent_cycle_tensor = avatar.input_latent_cycle_batch_tensor.unsqueeze(1)
    if compose_delay is not None:
        real = APIAvatar.compose_frame

        def delayed(res_frame, cycle_index, background_frame=None, return_layers=False):
            compose_delay(cycle_index)
            return real(avatar, res_frame, cycle_index, background_frame=background_frame,
                        return_layers=return_layers)
        avatar.compose_frame = delayed
    return avatar


class FakeBank:
    current_phoneme = None
    eye_blend = None

    def blend(self, anchor, source, alpha, anchor_pose, anchor_source, pose, source_index):
        out = anchor.astype(np.float32) * (1.0 - alpha) + source.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)


class FakeRouter:
    """Stateless pose router: a fixed per-generation-frame pose sequence."""

    def __init__(self, sequence, background_poses=()):
        self.motion_bank = FakeBank()
        self.sequence = sequence
        self.background_poses = set(background_poses)
        self.calls = 0

    def snapshots_for_range(self, start, count, fps):
        self.calls += 1
        out = []
        for index in range(start, start + count):
            pose, source = self.sequence[min(index, len(self.sequence) - 1)]
            out.append(SimpleNamespace(pose_id=pose, effective_render_key=pose, origin_generation_frame=0,
                                       is_queued=True, crossfade_frames=3, source_index=source,
                                       uses_prepared_background=pose not in self.background_poses))
        return out

    def source_frame_index(self, snapshot, generation_index):
        return snapshot.source_index

    def read_background_frames(self, snapshot, start, count):
        if snapshot.uses_prepared_background:
            return [None] * count
        frames = []
        for index in range(start, start + count):
            rng = np.random.default_rng(10_000 + index)
            frames.append(rng.integers(0, 256, (44, 52, 3), dtype=np.uint8))  # resized by compose
        return frames


def pose_sequence(total, pattern):
    seq, counters = [], {}
    for pose, n in pattern:
        for _ in range(n):
            seq.append((pose, counters.get(pose, 0)))
            counters[pose] = counters.get(pose, 0) + 1
    while len(seq) < total:
        seq.append((IDLE, counters.get(IDLE, 0)))
        counters[IDLE] = counters.get(IDLE, 0) + 1
    return seq[:total]


class InlineLoop:
    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


class InlineExecutor:
    """Runs compose synchronously (deterministic selection for HEAD comparisons)."""

    _max_workers = 1

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - surfaced by the job
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        return None


SPECS = [
    {"id": "std_a", "kind": "standard", "total": 37, "offset": 3, "seed": 1},
    {"id": "mot_b", "kind": "motion", "total": 58, "seed": 2,
     "pattern": [(IDLE, 6), (TALK, 20), (SMILE, 10), (TALK, 8)]},
    {"id": "std_c", "kind": "standard", "total": 21, "offset": 0, "seed": 3, "plans": False},
    {"id": "sil_d", "kind": "silence", "total": 30, "seed": 4},
    {"id": "mot_e", "kind": "motion", "total": 44, "seed": 5,
     "pattern": [(IDLE, 3), (TALK, 13), (IDLE, 2), (TALK, 12)]},
]


class Recorder:
    def __init__(self, sink_delay=0.0):
        self.frames = {}
        self.starts = {}
        self.yuv_bad = []
        self.yuv_carried = 0
        self.dispatch = []
        self.done = {}
        self.status = {}
        self.sink_delay = sink_delay
        self.lock = threading.Lock()

    def sink(self, job_id):
        def callback(frames, start_frame_idx, total_frames):
            if self.sink_delay:
                time.sleep(self.sink_delay)
            with self.lock:
                self.starts.setdefault(job_id, []).append((start_frame_idx, len(frames)))
                out = self.frames.setdefault(job_id, [])
                for frame in frames:
                    bgr = getattr(frame, "bgr", frame)
                    yuv = getattr(frame, "yuv420p", None)
                    if yuv is not None:
                        self.yuv_carried += 1
                        import av
                        expected = av.VideoFrame.from_ndarray(np.asarray(bgr), format="bgr24").reformat(
                            format="yuv420p").to_ndarray()
                        if not np.array_equal(expected, yuv.to_ndarray()):
                            self.yuv_bad.append((job_id, len(out)))
                    out.append(sha(bgr))
        return callback


def build_jobs(module, recorder, specs=SPECS, compose_delay=None, hooks=None):
    jobs = []
    for spec in specs:
        total = spec["total"]
        avatar = make_avatar(spec["seed"], plans=spec.get("plans", True), compose_delay=compose_delay)
        pose_avatars = {"default": avatar}
        router = None
        exact_silence = spec["kind"] == "silence"
        if spec["kind"] in ("motion", "silence"):
            pattern = spec.get("pattern", [])
            router = FakeRouter(pose_sequence(total, pattern), background_poses={IDLE})
            pose_avatars.update({
                IDLE: make_avatar(spec["seed"] + 100, compose_delay=compose_delay),
                TALK: avatar,
                SMILE: make_avatar(spec["seed"] + 200, plans=False, compose_delay=compose_delay),
            })
        session = SimpleNamespace(session_id=spec["id"], live_pose_router=router, prebuffer_seconds=0.5)
        if hooks is not None and spec["id"] in hooks:
            session.webrtc_playback_queue_frames = hooks[spec["id"]]
        cond = torch.arange(total * 5 * 8, dtype=torch.float32).reshape(total, 5, 8) / 97.0
        cond += spec["seed"] * 1000.0
        done = threading.Event()
        recorder.done[spec["id"]] = done

        def on_complete(status, error=None, job_id=spec["id"], done=done):
            recorder.status[job_id] = (status, error)
            done.set()

        job = module.HLSStreamJob(
            request_id=spec["id"], session_id=spec["id"], session=session, avatar=avatar,
            pose_avatars=pose_avatars, audio_path="unused.wav", chunk_output_dir=Path("/tmp/unused"),
            generation_fps=20, batch_size=8, conditioning_chunks=cond, conditioning_ready_frames=total,
            conditioning_complete=True, total_frames=total, frames_per_chunk=40, startup_chunk_frames=10,
            startup_chunk_count=1, total_chunks=2, start_offset_frames=spec.get("offset", 0),
            cancel_event=threading.Event(), completion_future=Future(), main_loop=InlineLoop(),
            output_mode="webrtc", exact_silence=exact_silence,
            frame_batch_callback=recorder.sink(spec["id"]), generation_complete_callback=on_complete,
        )
        jobs.append(job)
    return jobs


def light_job(module, job_id, total):
    """A job without avatar/router fixtures, for selection-only tests."""
    return module.HLSStreamJob(
        request_id=job_id, session_id=job_id, session=SimpleNamespace(live_pose_router=None),
        avatar=None, pose_avatars={}, audio_path="unused.wav", chunk_output_dir=Path("/tmp/unused"),
        generation_fps=20, batch_size=8, conditioning_chunks=None, conditioning_ready_frames=total,
        conditioning_complete=True, total_frames=total, frames_per_chunk=40, startup_chunk_frames=10,
        startup_chunk_count=1, total_chunks=2, start_offset_frames=0, cancel_event=threading.Event(),
        completion_future=None, main_loop=None, output_mode="webrtc",
    )


def run_scenario(module, env=None, inline=False, specs=SPECS, compose_delay=None, sink_delay=0.0,
                 hooks=None, manager=None, timeout=60.0, before_start=None):
    env = dict(env or {})
    recorder = Recorder(sink_delay)
    manager = manager or make_manager()
    with mock.patch.dict(os.environ, env, clear=False):
        scheduler = module.HLSGPUStreamScheduler(manager, None, max_combined_batch_size=8, startup_slice_size=8)
        if inline:
            scheduler.compose_executor.shutdown(wait=False)
            scheduler.compose_executor = InlineExecutor()
        original = scheduler._dispatch_compose_batch

        def tapped(job, batch_frames, start_frame_idx, *args, **kwargs):
            recorder.dispatch.append((job.request_id, start_frame_idx, len(batch_frames)))
            return original(job, batch_frames, start_frame_idx, *args, **kwargs)
        scheduler._dispatch_compose_batch = tapped
        jobs = build_jobs(module, recorder, specs, compose_delay=compose_delay, hooks=hooks)
        with scheduler.condition:
            for job in jobs:
                scheduler.jobs[job.request_id] = job
        if before_start is not None:
            before_start(scheduler, jobs)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            scheduler.start()
            try:
                for job_id, done in recorder.done.items():
                    if not done.wait(timeout):
                        raise AssertionError(f"{job_id} did not finish")
                stats = scheduler.get_stats()
                capacity = scheduler.get_capacity_stats(include_batches=True) if hasattr(
                    scheduler, "get_capacity_stats") else None
            finally:
                scheduler.shutdown()
    return SimpleNamespace(recorder=recorder, scheduler=scheduler, jobs=jobs, stats=stats,
                           capacity=capacity, manager=manager)


_BASELINES = {}


def baseline(raw_idle: bool):
    key = bool(raw_idle)
    if key not in _BASELINES:
        env = {"WEBRTC_RAW_IDLE_POSE": "1"} if raw_idle else {"WEBRTC_RAW_IDLE_POSE": "0"}
        _BASELINES[key] = run_scenario(head_module(), env=env, inline=True)
    return _BASELINES[key]


def assert_faces_vary(test, run):
    faces = set()
    for spec in SPECS:
        faces.update(run.recorder.frames.get(spec["id"], []))
    test.assertGreater(len(faces), 0.9 * sum(spec["total"] for spec in SPECS), "fixture frames are not distinct")


def assert_frames_equal(test, ref, got, label):
    for spec in SPECS:
        job_id = spec["id"]
        test.assertEqual(got.recorder.status.get(job_id, (None,))[0], "completed", f"{label}: {job_id}")
        ref_frames, got_frames = ref.recorder.frames[job_id], got.recorder.frames.get(job_id, [])
        test.assertEqual(len(got_frames), spec["total"], f"{label}: {job_id} frame count")
        mismatch = [i for i, (a, b) in enumerate(zip(ref_frames, got_frames)) if a != b]
        test.assertEqual(mismatch, [], f"{label}: {job_id} first mismatching frame {mismatch[:1]}")
        starts = got.recorder.starts[job_id]
        expected = 1
        for start, count in starts:
            test.assertEqual(start, expected, f"{label}: {job_id} out-of-order batch")
            expected += count


# ------------------------------------------------------------------ tests
class DefaultPathEquivalenceTest(unittest.TestCase):
    def test_default_flags_match_head_frames_and_batch_composition(self):
        ref = baseline(False)
        assert_faces_vary(self, ref)
        assert_faces_vary(self, baseline(True))
        got = run_scenario(wt, env={"WEBRTC_RAW_IDLE_POSE": "0"}, inline=True)
        assert_frames_equal(self, ref, got, "default")
        # Deterministic (inline compose): the exact same batches in the same order.
        self.assertEqual(got.recorder.dispatch, ref.recorder.dispatch)
        # Same number of UNet calls with the same rows.
        self.assertEqual(len(got.manager.unet.calls), len(ref.manager.unet.calls))
        for a, b in zip(ref.manager.unet.calls, got.manager.unet.calls):
            self.assertTrue(torch.equal(a, b))

    def test_default_stats_payload_unchanged(self):
        ref = baseline(False)
        got = run_scenario(wt, env={"WEBRTC_RAW_IDLE_POSE": "0"}, inline=True)
        self.assertEqual(set(got.stats), set(ref.stats))
        self.assertNotIn("pipeline", got.stats)
        self.assertNotIn("capacity", got.stats)

    def test_default_selection_matches_head_on_random_states(self):
        head = head_module()
        rng = random.Random(7)
        for trial in range(300):
            n = rng.randint(1, 9)
            states = []
            for i in range(n):
                total = rng.randint(1, 120)
                states.append({
                    "total": total, "current": rng.randint(0, total), "ready": rng.randint(0, total),
                    "complete": rng.random() < 0.6, "first": None if rng.random() < 0.4 else 1000.0 + i,
                    "encoded": 0, "startup_count": rng.choice([0, 1, 2]), "batch": rng.choice([2, 4, 8]),
                    "last": 500.0 + rng.random(),
                })
            cursor = rng.randint(0, 20)
            max_batch = rng.choice([4, 8, 16])
            slice_size = rng.choice([2, 8, 10])
            picks = []
            for module in (head, wt):
                with mock.patch.dict(os.environ, {}, clear=False):
                    sched = module.HLSGPUStreamScheduler(make_manager(), None, max_combined_batch_size=max_batch,
                                                         startup_slice_size=slice_size)
                sched.selection_cursor = cursor
                for i, st in enumerate(states):
                    job = light_job(module, f"j{i}", st["total"])
                    job.current_frame_idx = st["current"]
                    job.conditioning_ready_frames = max(st["ready"], st["current"])
                    job.conditioning_complete = st["complete"]
                    job.first_chunk_appended_at = st["first"]
                    job.encoded_frame_cursor = min(st["encoded"], st["current"])
                    job.startup_chunk_count = st["startup_count"]
                    job.batch_size = st["batch"]
                    job.last_progress_at = st["last"]
                    sched.jobs[job.request_id] = job
                with sched.condition:
                    if module is wt:
                        result = sched._select_jobs_for_batch_locked()
                    else:
                        result = sched._select_jobs_locked()
                picks.append(([(j.request_id, t) for j, t in result], sched.selection_cursor,
                              sorted(j.request_id for j in sched.jobs.values() if j.generation_done)))
                sched.shutdown()
            self.assertEqual(picks[0], picks[1], f"trial {trial}")


class FlagVariantsExactTest(unittest.TestCase):
    """Each lever alone and all together: per-job pre-encoder frames SHA-identical."""

    VARIANTS = {
        "event_timing": {"HLS_GPU_EVENT_TIMING": "1"},
        "stage_sync_off": {"HLS_GPU_STAGE_SYNC_TIMING": "0"},
        "depth2": {"HLS_GPU_PIPELINE_DEPTH": "2"},
        "depth2_min_ring_blocking": {"HLS_GPU_PIPELINE_DEPTH": "2", "HLS_GPU_OUTPUT_RING": "3",
                                     "HLS_GPU_BLOCKING_WAIT": "1"},
        "depth3": {"HLS_GPU_PIPELINE_DEPTH": "3"},
        "edf": {"HLS_SCHEDULER_POLICY": "edf"},
        "edf_prebuffer40_runahead": {"HLS_SCHEDULER_POLICY": "edf", "HLS_SCHEDULER_EDF_STARTUP_FRAMES": "40",
                                     "HLS_SCHEDULER_MAX_RUNAHEAD_S": "0.5"},
        "skip_raw": {"HLS_SKIP_GPU_FOR_RAW": "1"},
        "crossfade_copy_skip": {"HLS_SKIP_CROSSFADE_COPY": "1"},
        "yuv_in_compose": {"WEBRTC_YUV_IN_COMPOSE": "1"},
        "whisper_stream": {"MUSETALK_WHISPER_STREAM": "1"},
    }
    ALL = {"HLS_GPU_EVENT_TIMING": "1", "HLS_GPU_STAGE_SYNC_TIMING": "0", "HLS_GPU_PIPELINE_DEPTH": "2",
           "HLS_SCHEDULER_POLICY": "edf", "HLS_SKIP_GPU_FOR_RAW": "1", "HLS_SKIP_CROSSFADE_COPY": "1",
           "WEBRTC_YUV_IN_COMPOSE": "1", "MUSETALK_WHISPER_STREAM": "1"}

    def _check(self, name, env, raw_idle, inline, **kw):
        env = dict(env, WEBRTC_RAW_IDLE_POSE="1" if raw_idle else "0")
        got = run_scenario(wt, env=env, inline=inline, **kw)
        assert_frames_equal(self, baseline(raw_idle), got, f"{name} raw_idle={raw_idle} inline={inline}")
        for job in got.jobs:
            self.assertEqual(job.gpu_inflight_batches, 0)
        return got

    def test_each_flag_alone(self):
        for name, env in self.VARIANTS.items():
            for raw_idle in (False, True):
                for inline in (True, False):
                    with self.subTest(variant=name, raw_idle=raw_idle, inline=inline):
                        self._check(name, env, raw_idle, inline)

    def test_all_flags_together_with_slow_threaded_compose(self):
        rng = random.Random(3)
        lock = threading.Lock()

        def jitter(_index):
            with lock:
                delay = rng.random() * 0.004
            time.sleep(delay)
        for raw_idle in (False, True):
            with self.subTest(raw_idle=raw_idle):
                got = self._check("all", self.ALL, raw_idle, False, compose_delay=jitter, sink_delay=0.002)
                self.assertTrue(got.recorder.yuv_carried > 0)
                self.assertEqual(got.recorder.yuv_bad, [])
                self.assertIn("capacity", got.stats)
                self.assertIn("pipeline", got.stats)

    def test_yuv_carriers_are_exact_and_crossfaded_frames_stay_plain(self):
        got = self._check("yuv", {"WEBRTC_YUV_IN_COMPOSE": "1"}, False, True)
        self.assertEqual(got.recorder.yuv_bad, [])
        total = sum(spec["total"] for spec in SPECS)
        # Crossfaded frames (pose changes with 3-frame fades) are not carriers.
        self.assertTrue(0 < got.recorder.yuv_carried < total)


class SkipRawTest(unittest.TestCase):
    def test_exact_silence_frames_never_reach_the_gpu_and_rows_are_topped_up(self):
        base = baseline(False)
        got = run_scenario(wt, env={"HLS_SKIP_GPU_FOR_RAW": "1", "HLS_GPU_EVENT_TIMING": "1",
                                    "WEBRTC_RAW_IDLE_POSE": "0"}, inline=True)
        assert_frames_equal(self, base, got, "skip_raw")
        silence_seed = next(s["seed"] for s in SPECS if s["kind"] == "silence")
        for rows in got.manager.unet.calls:
            self.assertFalse(bool(((rows >= silence_seed * 1000.0) & (rows < silence_seed * 1000.0 + 999)).any()))
        base_rows = sum(len(r) for r in base.manager.unet.calls)
        got_rows = sum(len(r) for r in got.manager.unet.calls)
        silence_total = next(s["total"] for s in SPECS if s["kind"] == "silence")
        self.assertLess(got_rows, base_rows)
        self.assertGreaterEqual(got.capacity["totals"]["raw_frames"], silence_total)

    def test_raw_idle_frames_skipped_only_when_raw_idle_pose_enabled(self):
        got = run_scenario(wt, env={"HLS_SKIP_GPU_FOR_RAW": "1", "HLS_GPU_EVENT_TIMING": "1",
                                    "WEBRTC_RAW_IDLE_POSE": "1"}, inline=True)
        assert_frames_equal(self, baseline(True), got, "skip_raw_idle")
        idle_frames = sum(1 for spec in SPECS if spec["kind"] == "motion"
                          for pose, _ in pose_sequence(spec["total"], spec["pattern"]) if pose == IDLE)
        silence_total = next(s["total"] for s in SPECS if s["kind"] == "silence")
        self.assertEqual(int(got.capacity["totals"]["raw_frames"]), idle_frames + silence_total)

    def test_raw_layer_helper_matches_compose_frame_raw(self):
        avatar = make_avatar(11)
        plain = make_avatar(12, plans=False)
        rng = np.random.default_rng(0)
        face = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        backgrounds = {
            "none": None,
            "same_shape": rng.integers(0, 256, (40, 48, 3), dtype=np.uint8),
            "resized": rng.integers(0, 256, (44, 52, 3), dtype=np.uint8),
            "four_channel": rng.integers(0, 256, (40, 48, 4), dtype=np.uint8),
            "four_channel_resized": rng.integers(0, 256, (30, 20, 4), dtype=np.uint8),
            "float": rng.random((40, 48, 3)) * 300.0 - 20.0,
            "float_resized": (rng.random((50, 60, 3)) * 300.0).astype(np.float32),
            "two_dim_invalid": rng.integers(0, 256, (40, 48), dtype=np.uint8),
            "two_channel_invalid": rng.integers(0, 256, (40, 48, 2), dtype=np.uint8),
        }
        for label, background in backgrounds.items():
            for av_label, av in (("plans", avatar), ("masks", plain)):
                for cycle_index in (0, 5, 29):
                    with self.subTest(background=label, avatar=av_label, cycle=cycle_index):
                        expected = av.compose_frame(face, cycle_index, background_frame=background,
                                                    return_layers=True)["raw"]
                        got = wt.HLSGPUStreamScheduler._compose_raw_layer(av, cycle_index, background)
                        self.assertEqual(got.dtype, expected.dtype)
                        self.assertEqual(got.shape, expected.shape)
                        self.assertTrue(np.array_equal(got, expected))
                        self.assertFalse(got.flags.writeable)


class PipelineDepthTest(unittest.TestCase):
    def test_output_slots_are_not_overwritten_before_compose_reads_them(self):
        def slow_first_rows(cycle_index):
            if cycle_index % 4 == 0:
                time.sleep(0.02)
        env = {"HLS_GPU_PIPELINE_DEPTH": "2", "HLS_GPU_OUTPUT_RING": "3", "HLS_GPU_EVENT_TIMING": "1",
               "WEBRTC_RAW_IDLE_POSE": "0"}
        got = run_scenario(wt, env=env, compose_delay=slow_first_rows)
        assert_frames_equal(self, baseline(False), got, "depth2 slow compose")
        self.assertGreater(got.capacity["totals"].get("ring_waits", 0), 0)

    def test_negative_control_detects_overwrite_without_the_consumer_wait(self):
        def slow_first_rows(cycle_index):
            if cycle_index % 4 == 0:
                time.sleep(0.02)
        env = {"HLS_GPU_PIPELINE_DEPTH": "2", "HLS_GPU_OUTPUT_RING": "3", "WEBRTC_RAW_IDLE_POSE": "0"}
        with mock.patch.object(wt, "_wait_futures", lambda futures: None):
            got = run_scenario(wt, env=env, compose_delay=slow_first_rows)
        ref = baseline(False)
        mismatched = sum(1 for spec in SPECS for a, b in zip(ref.recorder.frames[spec["id"]],
                                                              got.recorder.frames.get(spec["id"], []))
                         if a != b)
        self.assertGreater(mismatched, 0, "the slot test would not notice an overwrite")

    def test_submit_runs_ahead_of_collect_and_counts_inflight_frames(self):
        order = []
        original_submit = wt.HLSGPUStreamScheduler._submit_generation_batch
        original_collect = wt.HLSGPUStreamScheduler._collect_generation_batch

        def submit(self, selected, pipelined=False):
            batch = original_submit(self, selected, pipelined)
            if batch is not None:
                order.append(("submit", batch.seq))
                # In-flight frames are already scheduled.
                for piece in batch.pieces:
                    assert piece.job.current_frame_idx >= piece.start_frame_idx + piece.take
                    assert piece.job.gpu_inflight_batches >= 1
            return batch

        def collect(self, batch):
            order.append(("collect", batch.seq))
            return original_collect(self, batch)

        with mock.patch.object(wt.HLSGPUStreamScheduler, "_submit_generation_batch", submit), \
                mock.patch.object(wt.HLSGPUStreamScheduler, "_collect_generation_batch", collect):
            got = run_scenario(wt, env={"HLS_GPU_PIPELINE_DEPTH": "2", "WEBRTC_RAW_IDLE_POSE": "0"})
        assert_frames_equal(self, baseline(False), got, "depth2 order")
        submits = [seq for kind, seq in order if kind == "submit"]
        collects = [seq for kind, seq in order if kind == "collect"]
        self.assertEqual(collects, sorted(collects))
        self.assertEqual(sorted(submits), sorted(collects))
        # Batch N+1 was submitted before batch N was collected.
        self.assertTrue(any(order[i] == ("submit", order[i][1]) and order[i + 1] == ("submit", order[i][1] + 1)
                            and order.index(("collect", order[i][1])) > i + 1
                            for i in range(len(order) - 1) if order[i][0] == "submit"))

    def test_generation_not_done_while_frames_in_flight(self):
        sched = wt.HLSGPUStreamScheduler(make_manager(), None, max_combined_batch_size=8)
        rec = Recorder()
        job = build_jobs(wt, rec, [{"id": "j", "kind": "standard", "total": 16, "seed": 1}])[0]
        job.current_frame_idx = 16
        job.gpu_inflight_batches = 1
        sched.jobs[job.request_id] = job
        with sched.condition:
            self.assertEqual(sched._ordered_schedulable_jobs_locked(), [])
        self.assertFalse(job.generation_done)
        job.cancel_event.set()
        sched._finalize_cancelled_jobs()
        self.assertFalse(job.finalized)
        job.gpu_inflight_batches = 0
        job.cancel_event.clear()
        with sched.condition:
            sched._ordered_schedulable_jobs_locked()
        self.assertTrue(job.generation_done)
        sched.shutdown()

    def test_cancel_with_batches_in_flight_finalizes_after_drain(self):
        cancelled = {}

        def before_start(scheduler, jobs):
            target = next(j for j in jobs if j.request_id == "mot_b")
            original = target.frame_batch_callback

            def cb(frames, start, total):
                original(frames, start, total)
                if start >= 9 and not target.cancel_event.is_set():
                    cancelled["at"] = start
                    target.cancel_event.set()
            target.frame_batch_callback = cb
        got = run_scenario(wt, env={"HLS_GPU_PIPELINE_DEPTH": "2", "WEBRTC_RAW_IDLE_POSE": "0"},
                           before_start=before_start)
        self.assertEqual(got.recorder.status["mot_b"][0], "cancelled")
        for job in got.jobs:
            self.assertEqual(job.gpu_inflight_batches, 0)
        ref = baseline(False)
        for spec in SPECS:
            if spec["id"] == "mot_b":
                frames = got.recorder.frames["mot_b"]
                self.assertEqual(frames, ref.recorder.frames["mot_b"][:len(frames)])
            else:
                self.assertEqual(got.recorder.frames[spec["id"]], ref.recorder.frames[spec["id"]])

    def test_device_u8_decode_matches_decode_latents_bytes(self):
        latents = torch.randn((8, 4, 4, 4), generator=torch.Generator().manual_seed(5))
        for backend in (FakeTaesdBackend(), FakeFusedBackend()):
            with self.subTest(backend=type(backend).__name__):
                manager = make_manager(make_vae(backend))
                sched = wt.HLSGPUStreamScheduler(manager, None)
                expected = manager.vae.decode_latents(latents)
                got = sched._vae_decode_device_u8(latents)
                self.assertIsNotNone(got)
                self.assertTrue(np.array_equal(got.numpy(), expected))
                sched.shutdown()
        with mock.patch.object(vae_mod, "MUSETALK_VAE_FAST_POSTPROCESS", False):
            sched = wt.HLSGPUStreamScheduler(make_manager(), None)
            self.assertIsNone(sched._vae_decode_device_u8(latents))
            sched.shutdown()


class EdfSelectionTest(unittest.TestCase):
    def make(self, env, specs, setup):
        with mock.patch.dict(os.environ, dict({"HLS_SCHEDULER_POLICY": "edf"}, **env), clear=False):
            sched = wt.HLSGPUStreamScheduler(make_manager(), None, max_combined_batch_size=8)
        rec = Recorder()
        jobs = build_jobs(wt, rec, specs)
        for index, job in enumerate(jobs):
            job.queued_at = 100.0 + index
            setup(job, index)
            sched.jobs[job.request_id] = job
        return sched, jobs

    @staticmethod
    def pick(sched):
        with sched.condition:
            return {job.request_id: take for job, take in sched._select_jobs_for_batch_locked()}

    @staticmethod
    def advance(sched, picks):
        for job in sched.jobs.values():
            job.current_frame_idx += picks.get(job.request_id, 0)
            job.composed_frame_idx = job.current_frame_idx

    def test_startup_slices_are_prebuffer_sized_and_packed(self):
        specs = [{"id": f"s{i}", "kind": "standard", "total": 200, "seed": i} for i in range(3)]

        def setup(job, index):
            job.session.webrtc_playback_queue_frames = lambda job=job: job.composed_frame_idx
        sched, jobs = self.make({"HLS_SCHEDULER_EDF_STARTUP_FRAMES": "10"}, specs, setup)
        seq = []
        for _ in range(4):
            picks = self.pick(sched)
            seq.append(picks)
            self.advance(sched, picks)
        self.assertEqual(seq[0], {"s0": 8})
        self.assertEqual(seq[1], {"s0": 2, "s1": 6})
        self.assertEqual(seq[2], {"s1": 4, "s2": 4})
        self.assertEqual(seq[3].get("s2"), 6)
        self.assertEqual(sum(seq[3].values()), 8)
        sched.shutdown()

    def test_startup_target_defaults_to_session_prebuffer(self):
        specs = [{"id": "p", "kind": "standard", "total": 200, "seed": 1}]

        def setup(job, index):
            job.session.prebuffer_seconds = 2.0
        sched, jobs = self.make({}, specs, setup)
        self.assertEqual(sched._edf_startup_target(jobs[0]), 40)
        jobs[0].session.prebuffer_seconds = 0.0
        self.assertEqual(sched._edf_startup_target(jobs[0]), 10)  # startup chunk
        sched.shutdown()

    def test_urgent_warmed_job_preempts_startup(self):
        specs = [{"id": "warm", "kind": "standard", "total": 400, "seed": 1},
                 {"id": "new", "kind": "standard", "total": 400, "seed": 2}]

        def setup(job, index):
            if job.request_id == "warm":
                job.current_frame_idx = job.composed_frame_idx = 100
                job.session.webrtc_playback_queue_frames = lambda: 4  # 0.2 s of slack
            else:
                job.session.webrtc_playback_queue_frames = lambda: 0
        sched, _ = self.make({"HLS_SCHEDULER_EDF_STARTUP_FRAMES": "10"}, specs, setup)
        self.assertEqual(self.pick(sched), {"warm": 8})
        sched.shutdown()

    def test_earliest_deadline_first_and_runahead_cap_and_queue_full(self):
        specs = [{"id": n, "kind": "standard", "total": 900, "seed": i}
                 for i, n in enumerate(["a", "b", "c", "far", "full"])]
        depth = {"a": 20, "b": 10, "c": 30, "far": 120, "full": 45}

        def setup(job, index):
            job.current_frame_idx = job.composed_frame_idx = 300
            job.session.webrtc_playback_queue_frames = lambda n=job.request_id: depth[n]
            if job.request_id == "full":
                job.session.idle_track = SimpleNamespace(_max_queue=50)
        sched, jobs = self.make({"HLS_SCHEDULER_EDF_STARTUP_FRAMES": "10", "HLS_SCHEDULER_MAX_RUNAHEAD_S": "5"},
                                specs, setup)
        picks = self.pick(sched)
        self.assertEqual(picks, {"b": 8})  # smallest slack (0.5 s) takes the whole batch
        depth["b"] = 60
        picks = self.pick(sched)
        self.assertEqual(picks, {"a": 8})
        depth.update({"a": 98, "b": 99, "c": 99, "full": 45})
        picks = self.pick(sched)
        self.assertNotIn("far", picks)   # 6 s ahead > 5 s cap
        self.assertNotIn("full", picks)  # 45 + 8 > its track's max_queue 50
        self.assertEqual(picks, {"a": 8})  # 4.9 s: still under the cap, least slack
        depth.update({"a": 150, "b": 150, "c": 150})
        self.assertEqual(self.pick(sched), {})  # everyone beyond the cap: GPU waits
        sched.shutdown()

    def test_runahead_counts_frames_already_in_flight(self):
        specs = [{"id": "x", "kind": "standard", "total": 900, "seed": 1}]

        def setup(job, index):
            job.current_frame_idx = 300
            job.composed_frame_idx = 210  # 90 frames in flight / composing
            job.session.webrtc_playback_queue_frames = lambda: 15
        sched, _ = self.make({"HLS_SCHEDULER_EDF_STARTUP_FRAMES": "10", "HLS_SCHEDULER_MAX_RUNAHEAD_S": "5"},
                             specs, setup)
        self.assertEqual(self.pick(sched), {})  # (15 + 90 + alloc) / 20 > 5 s
        sched.shutdown()


class CrossfadeCopySkipTest(unittest.TestCase):
    def test_history_equal_and_copies_only_last_frame(self):
        rng = np.random.default_rng(1)
        batches = []
        poses = [TALK] * 5 + [SMILE] * 4 + [TALK] * 7 + [IDLE] * 4
        frames = [rng.integers(0, 256, (12, 10, 3), dtype=np.uint8) for _ in poses]
        for start in (0, 3, 8, 9, 15):
            batches.append(start)
        bounds = batches + [len(poses)]
        outputs = {}
        copies = {}
        for skip in (False, True):
            sched = wt.HLSGPUStreamScheduler.__new__(wt.HLSGPUStreamScheduler)
            sched.webrtc_pose_crossfade_frames = 2
            sched.skip_crossfade_copy = skip
            job = build_jobs(wt, Recorder(), [{"id": "x", "kind": "motion", "total": len(poses), "seed": 1,
                                               "pattern": [(TALK, 5), (SMILE, 4), (TALK, 7)]}])[0]
            out = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                chunk = [f.copy() for f in frames[a:b]]
                result = sched._apply_webrtc_pose_crossfade(job, chunk, poses[a:b], [3] * (b - a),
                                                           list(range(a, b)))
                out.extend(sha(f) for f in result)
                copies.setdefault(skip, []).append(job.webrtc_last_pose_frame is not chunk[-1])
            outputs[skip] = out
        self.assertEqual(outputs[False], outputs[True])
        self.assertTrue(all(copies[True]))  # the last frame of each batch is still an owned copy


class CapacityTelemetryTest(unittest.TestCase):
    def test_capacity_fields_and_event_vs_host_totals(self):
        got = run_scenario(wt, env={"HLS_GPU_EVENT_TIMING": "1", "WEBRTC_RAW_IDLE_POSE": "0"}, inline=True)
        cap = got.stats["capacity"]
        derived = cap["derived"]
        self.assertGreater(derived["batches"], 0)
        self.assertGreater(derived["jobs_per_batch"], 0)
        self.assertTrue(0 < derived["fill"] <= 1.0)
        self.assertGreaterEqual(derived["feeder_cpu_ms_per_batch"], 0.0)
        self.assertIn("gpu_busy_fraction", derived)
        self.assertGreater(cap["totals"]["callback_ms"], 0.0)
        self.assertEqual(int(cap["totals"]["actual_frames"]), sum(s["total"] for s in SPECS))
        self.assertTrue(all("unet_ms" in b for b in got.capacity["batches"] if b["actual"] > 0))
        self.assertEqual(got.stats["pipeline"]["gpu_event_timing"], True)


class ReplayHarnessCpuTest(unittest.TestCase):
    """The GPU-free parts of scripts/replay_scheduler_exactness.py."""

    @classmethod
    def setUpClass(cls):
        import scripts.replay_scheduler_exactness as harness
        cls.h = harness
        cls.pool = ThreadPoolExecutor(max_workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.pool.shutdown()

    def recorder_result(self, frames_by_batch, faces_by_batch=None):
        rec = self.h.JobRecorder("j", self.pool, True, True, True)
        start = 1
        for i, frames in enumerate(frames_by_batch):
            faces = (faces_by_batch or [None] * len(frames_by_batch))[i]
            if faces is not None:
                rec.record_faces(start - 1, faces)
            rec.record_frames(frames, start, time.time())
            start += len(frames)
        result = rec.result()
        result.update({"status": "completed"})
        return result

    def test_yuv_contract_check_and_compare(self):
        from scripts.webrtc_live_handoff import ComposedFrame
        rng = np.random.default_rng(0)
        bgr = [rng.integers(0, 256, (40, 48, 3), dtype=np.uint8) for _ in range(6)]
        faces = [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(6)]
        plain = self.recorder_result([bgr[:3], bgr[3:]], [faces[:3], faces[3:]])
        carried = self.recorder_result([[ComposedFrame.from_bgr(b) for b in bgr[:3]], bgr[3:]],
                                       [[None, faces[1], faces[2]], faces[3:]])
        self.assertEqual(carried["yuv_contract_mismatches"], [])
        self.assertEqual(carried["raw_no_gpu_faces"], 1)
        wrong = ComposedFrame(bgr[0], ComposedFrame.from_bgr(bgr[1]).yuv420p)
        bad = self.recorder_result([[wrong] + bgr[1:3], bgr[3:]])
        self.assertEqual(bad["yuv_contract_mismatches"], [0])
        run = lambda job: {"jobs": {"j": job}}
        same = self.h.compare(run(plain), run(carried))
        self.assertTrue(same["identical"], same)
        changed = self.recorder_result([[bgr[0], bgr[2], bgr[1]], bgr[3:]])
        diff = self.h.compare(run(plain), run(changed))
        self.assertFalse(diff["identical"])
        self.assertEqual(diff["jobs"]["j"]["first_frame_mismatch"], 1)
        self.assertFalse(self.h.compare(run(plain), run(bad))["identical"])
        two = {"jobs": {"j": plain, "k": plain}}
        sub = {"jobs": {"j": plain}, "harness_options": {"REPLAY_JOBS": "j"}}
        self.assertTrue(self.h.compare(two, sub)["identical"])
        self.assertFalse(self.h.compare(two, {"jobs": {"j": plain}})["identical"])

    def test_order_errors_are_reported(self):
        rec = self.h.JobRecorder("j", self.pool, False, True, False)
        frame = np.zeros((4, 4, 3), np.uint8)
        rec.record_frames([frame, frame], 1, 0.0)
        rec.record_frames([frame], 4, 0.0)  # skipped index 3
        self.assertEqual(rec.result()["order_errors"], [{"expected": 3, "got": 4, "count": 1}])

    def test_video_writer_mp4_crf_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.h.LosslessWriter(Path(tmp) / "clip", 20, 10, crf=30)
            self.assertEqual(writer.crf, 12)
            rng = np.random.default_rng(1)
            for _ in range(12):
                writer.put(rng.integers(0, 256, (41, 47, 3), dtype=np.uint8))  # odd size -> padded
            writer.close()
            self.assertEqual(writer.path.suffix, ".mp4")
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                    "-show_entries", "stream=codec_name,pix_fmt,nb_read_frames,width,height",
                                    "-of", "csv=p=0", str(writer.path)], capture_output=True, text=True, check=True)
            codec, width, height, pix_fmt, frames = probe.stdout.strip().split(",")
            self.assertEqual((codec, pix_fmt, frames), ("h264", "yuv420p", "10"))
            self.assertEqual((int(width), int(height)), (48, 42))

    def test_event_vs_host_summary(self):
        cap = {"enabled": True, "config": {"gpu_stage_sync_timing": True, "pipeline_depth": 1},
               "totals": {"gpu_batches": 4, "h2d_ms": 2.0, "host_copy_ms": 2.1, "unet_ms": 98.0,
                          "host_unet_ms": 100.0, "vae_ms": 20.0, "d2h_ms": 1.0, "host_vae_ms": 21.0}}
        out = self.h.event_vs_host(cap)
        self.assertAlmostEqual(out["unet"]["rel_diff"], -0.02)
        self.assertAlmostEqual(out["vae"]["rel_diff"], 0.0)
        self.assertIsNone(self.h.event_vs_host(None))

    def test_run_spec_harness_keys_are_not_exported(self):
        run = self.h.parse_run("wt:repo:HLS_GPU_PIPELINE_DEPTH=2,REPLAY_JOBS=bob_t1+jp_d10")
        self.assertEqual(run["env"], {"HLS_GPU_PIPELINE_DEPTH": "2"})
        self.assertEqual(run["harness"], {"REPLAY_JOBS": "bob_t1+jp_d10"})


if __name__ == "__main__":
    unittest.main()
