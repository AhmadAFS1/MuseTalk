import collections
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as _wait_futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

# Added code (300 fps plan items 0.2, 1.1, 1.2 hookup, 1.4, 1.9, 1.10, 1.5
# crossfade-copy skip and the WEBRTC_YUV_IN_COMPOSE producer side;
# docs/musetalk_4070s_300fps_plan_2026-09-27.md). Every lever is one env line
# and every default reproduces the scheduler exactly as it was before them.
# name -> (default, effect)
SCHEDULER_PIPELINE_FLAGS = {
    "HLS_GPU_EVENT_TIMING": (
        "0",
        "1: CUDA events around H2D / UNet / TAESD / D2H of every batch; capacity telemetry "
        "(GPU busy fraction, GPU idle gap between batches, callback-blocked time, batch fill, "
        "jobs per batch, feeder-thread CPU per batch) in get_stats()['capacity'] and "
        "get_capacity_stats(). With HLS_GPU_STAGE_SYNC_TIMING=0 (or depth >= 2) the per-job "
        "stage times come from the events instead of host clocks."),
    "HLS_GPU_PIPELINE_DEPTH": (
        "1",
        "2: double-buffered loop: batch N+1 is assembled and launched before batch N is "
        "collected; per-slot pinned staging, pinned output ring, non_blocking D2H + events, "
        "no host syncs inside a batch. All UNet/TAESD launches stay on the scheduler thread."),
    "HLS_GPU_OUTPUT_RING": (
        "0", "Pinned output slots per shape for depth >= 2 (0 = depth + 2). A slot is reused only "
             "after every compose task that reads its faces has finished."),
    "HLS_GPU_TIMING_WINDOW": (
        "512", "Per-batch capacity records kept for get_capacity_stats(include_batches=True)."),
    "HLS_GPU_BLOCKING_WAIT": (
        "0", "1 (depth >= 2): the batch-done event is created with blocking=True so the collect "
             "wait sleeps instead of spinning (plan item 1.3)."),
    "HLS_SCHEDULER_POLICY": (
        "roundrobin",
        "edf: startup jobs packed to a prebuffer-sized first slice, then earliest-deadline-first "
        "on slack (frames queued ahead of playout), skipping jobs beyond the run-ahead cap or "
        "whose consumer queue is full. roundrobin: today's five selection rounds."),
    "HLS_SCHEDULER_MAX_RUNAHEAD_S": (
        "5", "edf: jobs with more than this many seconds generated ahead of playout wait (<=0 off)."),
    "HLS_SCHEDULER_EDF_STARTUP_FRAMES": (
        "0", "edf: first-slice size per job; 0 = session prebuffer_seconds x fps, else the "
             "startup chunk."),
    "HLS_SCHEDULER_EDF_URGENT_S": (
        "0.5", "edf: warmed jobs with less slack than this are served before startup jobs."),
    "HLS_SCHEDULER_EDF_MAX_QUEUE_FRAMES": (
        "0", "edf: consumer queue capacity used for the queue-full skip (0 = the track's max_queue)."),
    "HLS_SKIP_GPU_FOR_RAW": (
        "0",
        "1: frames whose generated face is discarded (exact_silence jobs, and neutral_resting "
        "frames when WEBRTC_RAW_IDLE_POSE=1) are left out of the UNet/TAESD batch and composed "
        "from the raw layer only; freed GPU rows are topped up with other jobs' frames."),
    "HLS_SKIP_RAW_MAX_PENDING_BATCHES": (
        "2", "With HLS_SKIP_GPU_FOR_RAW=1: an exact_silence job with this many compose batches "
             "outstanding is not selected (paces all-raw jobs that no longer wait on the GPU)."),
    "HLS_SKIP_CROSSFADE_COPY": (
        "0", "1: the WebRTC crossfade history keeps a reference to each frame and copies only the "
             "last frame of a batch (the only one that can outlive the batch) instead of every frame."),
    "WEBRTC_YUV_IN_COMPOSE": (
        "0", "1: compose workers attach the exact PyAV yuv420p conversion to each WebRTC frame "
             "(webrtc_live_handoff.ComposedFrame); frames changed by a crossfade stay plain BGR."),
    "MUSETALK_WHISPER_STREAM": (
        "0", "1: the Whisper encode in job prep runs on a per-prep-thread CUDA stream so it does "
             "not queue behind scheduler batches."),
}


def scheduler_pipeline_flag_snapshot() -> dict:
    return {
        name: os.environ.get(name, default)
        for name, (default, _effect) in SCHEDULER_PIPELINE_FLAGS.items()
    }


class _HostEvent:
    """CPU stand-in for torch.cuda.Event (CPU device or no CUDA): work is
    synchronous, so an event completes when it is recorded."""

    __slots__ = ("t",)

    def __init__(self):
        self.t = None

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        return None

    def query(self):
        return True

    def elapsed_time(self, end) -> float:
        if self.t is None or end.t is None:
            return 0.0
        return (end.t - self.t) * 1000.0


class _BatchPiece:
    """A contiguous run of one job's frames inside one GPU batch."""

    __slots__ = ("job", "start_frame_idx", "take", "snapshots", "raw_mask",
                 "gpu_offset", "gpu_rows")

    def __init__(self, job, start_frame_idx: int, take: int):
        self.job = job
        self.start_frame_idx = int(start_frame_idx)
        self.take = int(take)
        # Pose snapshots computed at assembly. Only pieces that skip raw
        # frames carry them to compose; otherwise compose recomputes them
        # exactly as before.
        self.snapshots = None
        # None: every frame goes to the GPU (the default). Else one bool per
        # frame, True = composed raw without a generated face.
        self.raw_mask = None
        self.gpu_offset = 0
        self.gpu_rows = int(take)


class _OutputSlot:
    """One pinned host buffer of the depth >= 2 output ring."""

    __slots__ = ("tensor", "array", "consumers", "uses")

    def __init__(self, tensor):
        self.tensor = tensor
        self.array = tensor.numpy()
        self.consumers = []
        self.uses = 0


class _InflightBatch:
    """Everything collect() needs about a submitted batch."""

    __slots__ = (
        "seq", "pieces", "jobs", "actual_batch", "padded_batch", "lease_batch_size",
        "recon", "done_event", "slot", "events", "pipelined", "raw_frames",
        "batch_started_at", "assembly_finished_at", "copy_started_at", "copy_finished_at",
        "pe_started_at", "pe_finished_at", "unet_started_at", "unet_finished_at",
        "vae_started_at", "vae_finished_at", "submit_cpu_s", "submitted_at_perf",
    )

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, None)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class HLSStreamJob:
    request_id: str
    session_id: str
    session: object
    avatar: object
    pose_avatars: Dict[str, object]
    audio_path: str
    chunk_output_dir: Path
    generation_fps: int
    batch_size: int
    conditioning_chunks: object
    conditioning_ready_frames: int
    conditioning_complete: bool
    total_frames: int
    frames_per_chunk: int
    startup_chunk_frames: int
    startup_chunk_count: int
    total_chunks: int
    start_offset_frames: int
    cancel_event: threading.Event
    completion_future: object
    main_loop: object
    output_mode: str = "hls"
    exact_silence: bool = False
    frame_callback: Optional[Callable[[object, int, int], None]] = None
    frame_batch_callback: Optional[Callable[[list, int, int], None]] = None
    generation_complete_callback: Optional[Callable[[str, Optional[str]], None]] = None
    audio_copy_path: Optional[str] = None
    idle_frames: list = field(default_factory=list)
    crossfade_tail_frames: int = 0
    current_frame_idx: int = 0
    composed_frame_idx: int = 0
    chunk_index: int = 0
    frame_buffer: list = field(default_factory=list)
    max_frame_buffer_len: int = 0
    generation_done: bool = False
    generation_done_at: Optional[float] = None
    finalized: bool = False
    finalized_at: Optional[float] = None
    last_progress_at: float = field(default_factory=time.time)
    compose_tasks: Dict[int, object] = field(default_factory=dict)
    composed_batches: Dict[int, dict] = field(default_factory=dict)
    compose_sequence: int = 0
    next_compose_sequence: int = 0
    max_pending_composes: int = 0
    encode_tasks: Dict[int, object] = field(default_factory=dict)
    encoded_chunks: Dict[int, dict] = field(default_factory=dict)
    next_append_chunk_index: int = 0
    max_pending_encodes: int = 0
    error_message: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    prep_started_at: float = 0.0
    queued_at: float = 0.0
    prep_total_s: float = 0.0
    prep_queue_wait_s: float = 0.0
    prep_work_s: float = 0.0
    avatar_load_s: float = 0.0
    audio_feature_s: float = 0.0
    audio_copy_prep_s: float = 0.0
    whisper_chunk_s: float = 0.0
    first_scheduled_at: Optional[float] = None
    first_chunk_appended_at: Optional[float] = None
    scheduler_turns: int = 0
    gpu_batch_count: int = 0
    batch_assembly_total_s: float = 0.0
    gpu_copy_total_s: float = 0.0
    pe_total_s: float = 0.0
    unet_total_s: float = 0.0
    vae_total_s: float = 0.0
    gpu_batch_total_s: float = 0.0
    compose_total_s: float = 0.0
    compose_queue_wait_total_s: float = 0.0
    compose_batch_count: int = 0
    max_compose_queue_wait_s: float = 0.0
    max_compose_s: float = 0.0
    frame_callback_count: int = 0
    frame_callback_total_s: float = 0.0
    frame_callback_max_s: float = 0.0
    chunks_encoded: int = 0
    encode_queue_wait_total_s: float = 0.0
    encode_total_s: float = 0.0
    max_encode_queue_wait_s: float = 0.0
    max_encode_s: float = 0.0
    chunks_appended: int = 0
    encoded_frame_cursor: int = 0
    webrtc_last_pose_frame: object = field(default=None, repr=False)
    webrtc_last_raw_pose_frame: object = field(default=None, repr=False)
    webrtc_last_pose_alpha: object = field(default=None, repr=False)
    webrtc_pose_crossfade_raw_anchor: object = field(default=None, repr=False)
    webrtc_pose_crossfade_alpha_anchor: object = field(default=None, repr=False)
    webrtc_last_pose_id: Optional[str] = None
    webrtc_last_source_frame: Optional[int] = None
    webrtc_pose_crossfade_anchor_pose: Optional[str] = None
    webrtc_pose_crossfade_anchor_source: Optional[int] = None
    webrtc_pose_crossfade_anchor: object = field(default=None, repr=False)
    webrtc_pose_crossfade_index: int = 0
    webrtc_pose_crossfade_target_frames: int = 0
    webrtc_pose_crossfade_count: int = 0
    webrtc_pose_crossfade_frames_applied: int = 0
    conditioning_lock: object = field(default_factory=threading.Lock, repr=False)
    # Added code (plan item 1.4): GPU batches submitted for this job but not
    # yet collected. Always 0 between batches at depth 1.
    gpu_inflight_batches: int = 0


class HLSGPUStreamScheduler:
    """
    Shared HLS GPU scheduler.

    One GPU thread batches work across active HLS sessions and a separate
    encode pool turns ready frame buffers into TS segments.
    """

    # Added code: class-level defaults of the 300 fps plan flags. __init__
    # reads the env; instances built with __new__ (unit tests) get today's
    # behaviour from these.
    gpu_event_timing = False
    pipeline_depth = 1
    output_ring_size = 0
    gpu_blocking_wait = False
    scheduler_policy = "roundrobin"
    max_runahead_s = 5.0
    edf_startup_frames = 0
    edf_urgent_s = 0.5
    edf_max_queue_frames = 0
    skip_gpu_for_raw = False
    skip_raw_max_pending_batches = 2
    skip_crossfade_copy = False
    webrtc_yuv_in_compose = False
    whisper_stream_enabled = False

    def __init__(
        self,
        manager,
        hls_session_manager,
        max_combined_batch_size: int = 8,
        startup_slice_size: int = 2,
        aggressive_fill_max_active_jobs: int = 4,
        prep_workers: int = 2,
        compose_workers: int = 2,
        encode_workers: int = 2,
        max_pending_jobs: int = 16,
        startup_chunk_duration_seconds: float = 0.5,
        startup_chunk_count: int = 1,
    ):
        self.manager = manager
        self.hls_session_manager = hls_session_manager
        self.max_combined_batch_size = max(1, int(max_combined_batch_size))
        self.startup_slice_size = max(1, int(startup_slice_size))
        self.aggressive_fill_max_active_jobs = max(0, int(aggressive_fill_max_active_jobs))
        self.max_pending_jobs = max(1, int(max_pending_jobs))
        self.startup_chunk_duration_seconds = max(0.0, float(startup_chunk_duration_seconds))
        self.startup_chunk_count = max(0, int(startup_chunk_count))
        self.prep_executor = ThreadPoolExecutor(max_workers=max(1, int(prep_workers)))
        # Local modification: this differs from the original MuseTalk code.
        # Prep now has a second executor so adjacent subtasks can overlap.
        prep_subtask_workers = max(
            2,
            min(
                max(1, os.cpu_count() or 8),
                max(2, int(prep_workers) * 2),
            ),
        )
        self.prep_subtask_executor = ThreadPoolExecutor(
            max_workers=prep_subtask_workers,
            thread_name_prefix="hls-prep-subtask",
        )
        self.backfill_executor = ThreadPoolExecutor(
            max_workers=max(1, min(2, int(prep_workers))),
            thread_name_prefix="hls-backfill",
        )

        # Scale compose workers: cv2/numpy release the GIL for heavy ops.
        # At 8 streams producing 32 frames/tick, 2 workers creates a queue.
        cpu_count = os.cpu_count() or 8
        effective_compose = max(
            6,
            int(compose_workers),
            min(10, cpu_count // 2),
        )
        self.compose_executor = ThreadPoolExecutor(
            max_workers=effective_compose,
            thread_name_prefix="hls-compose",
        )

        # Encode runs in worker threads because each ready chunk still performs
        # ffmpeg work off the main GPU scheduler loop.
        effective_encode = max(
            6,
            int(encode_workers),
            min(10, cpu_count // 2),
        )
        self.encode_executor = ThreadPoolExecutor(
            max_workers=effective_encode,
            thread_name_prefix="hls-encode",
        )
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.scheduler_thread: Optional[threading.Thread] = None
        self.jobs: Dict[str, HLSStreamJob] = {}
        self.preparing_requests: set[str] = set()
        self.selection_cursor = 0
        self._cpu_pe_cache: Dict[tuple[int, str], torch.Tensor] = {}
        self._cpu_staging_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.fixed_batch_sizes = self._resolve_fixed_batch_sizes(self.max_combined_batch_size)
        self.gpu_batch_timing_log_interval = max(0, _env_int("HLS_GPU_BATCH_TIMING_LOG_INTERVAL", 25))
        self.gpu_batch_timing_slow_s = max(0.0, _env_float("HLS_GPU_BATCH_TIMING_SLOW_SECONDS", 0.5))
        self.gpu_stage_sync_timing = _env_bool("HLS_GPU_STAGE_SYNC_TIMING", True)
        self.webrtc_pose_crossfade_frames = max(
            0,
            _env_int("WEBRTC_POSE_CROSSFADE_FRAMES", 0),
        )
        self._gpu_batch_timing_counter = 0
        self.vae_calibration_capture = _env_bool("MUSETALK_VAE_CALIBRATION_CAPTURE", False)
        self.vae_calibration_dir = Path(
            os.getenv("MUSETALK_VAE_CALIBRATION_DIR", "./calibration/vae_decoder")
        )
        self.vae_calibration_max_batches = max(
            0,
            _env_int("MUSETALK_VAE_CALIBRATION_MAX_BATCHES", 128),
        )
        self._vae_calibration_capture_count = 0
        self._vae_calibration_limit_logged = False
        self.unet_calibration_capture = _env_bool("MUSETALK_UNET_CALIBRATION_CAPTURE", False)
        self.unet_calibration_dir = Path(
            os.getenv("MUSETALK_UNET_CALIBRATION_DIR", "./calibration/unet")
        )
        self.unet_calibration_max_batches = max(
            0,
            _env_int("MUSETALK_UNET_CALIBRATION_MAX_BATCHES", 128),
        )
        self._unet_calibration_capture_count = 0
        self._unet_calibration_limit_logged = False
        self._init_pipeline_flags()

    def _init_pipeline_flags(self) -> None:
        """Added code: read the 300 fps plan flags (SCHEDULER_PIPELINE_FLAGS)."""
        self.gpu_event_timing = _env_bool("HLS_GPU_EVENT_TIMING", False)
        self.pipeline_depth = max(1, min(4, _env_int("HLS_GPU_PIPELINE_DEPTH", 1)))
        ring = _env_int("HLS_GPU_OUTPUT_RING", 0)
        self.output_ring_size = max(self.pipeline_depth + 1, ring if ring > 0 else self.pipeline_depth + 2)
        self.gpu_blocking_wait = _env_bool("HLS_GPU_BLOCKING_WAIT", False)
        policy = (os.getenv("HLS_SCHEDULER_POLICY") or "roundrobin").strip().lower()
        if policy not in ("roundrobin", "edf"):
            raise RuntimeError(
                f"Invalid HLS_SCHEDULER_POLICY={policy!r}; expected roundrobin or edf"
            )
        self.scheduler_policy = policy
        self.max_runahead_s = _env_float("HLS_SCHEDULER_MAX_RUNAHEAD_S", 5.0)
        self.edf_startup_frames = max(0, _env_int("HLS_SCHEDULER_EDF_STARTUP_FRAMES", 0))
        self.edf_urgent_s = max(0.0, _env_float("HLS_SCHEDULER_EDF_URGENT_S", 0.5))
        self.edf_max_queue_frames = max(0, _env_int("HLS_SCHEDULER_EDF_MAX_QUEUE_FRAMES", 0))
        self.skip_gpu_for_raw = _env_bool("HLS_SKIP_GPU_FOR_RAW", False)
        self.skip_raw_max_pending_batches = max(1, _env_int("HLS_SKIP_RAW_MAX_PENDING_BATCHES", 2))
        self.skip_crossfade_copy = _env_bool("HLS_SKIP_CROSSFADE_COPY", False)
        self.webrtc_yuv_in_compose = _env_bool("WEBRTC_YUV_IN_COMPOSE", False)
        self.whisper_stream_enabled = _env_bool("MUSETALK_WHISPER_STREAM", False)
        self._whisper_streams = threading.local()
        # depth >= 2 state (only touched by the scheduler thread).
        self._pipeline_seq = 0
        self._staging_slot_events: Dict[int, object] = {}
        self._output_rings: Dict[tuple, list] = {}
        self._output_ring_pos: Dict[tuple, int] = {}
        self._fallback_decode_logged = False
        # Capacity telemetry (HLS_GPU_EVENT_TIMING=1).
        self._cap_lock = threading.Lock()
        self._cap_started_at = time.monotonic()
        self._cap_totals = collections.defaultdict(float)
        self._cap_batches = collections.deque(
            maxlen=max(16, _env_int("HLS_GPU_TIMING_WINDOW", 512))
        )
        self._cap_prev_end_event = None
        self._cap_prev_seq = None

    def _pipeline_flags_non_default(self) -> bool:
        return bool(
            self.gpu_event_timing
            or self.pipeline_depth > 1
            or self.gpu_blocking_wait
            or self.scheduler_policy != "roundrobin"
            or self.skip_gpu_for_raw
            or self.skip_crossfade_copy
            or self.webrtc_yuv_in_compose
            or self.whisper_stream_enabled
        )

    def _pipeline_config(self) -> dict:
        unet_model = getattr(getattr(self.manager, "unet", None), "model", None)
        return {
            "gpu_event_timing": self.gpu_event_timing,
            "gpu_stage_sync_timing": bool(getattr(self, "gpu_stage_sync_timing", True)),
            "pipeline_depth": self.pipeline_depth,
            "output_ring_size": self.output_ring_size if self.pipeline_depth > 1 else 0,
            "gpu_blocking_wait": self.gpu_blocking_wait,
            "scheduler_policy": self.scheduler_policy,
            "max_runahead_s": self.max_runahead_s,
            "edf_startup_frames": self.edf_startup_frames,
            "edf_urgent_s": self.edf_urgent_s,
            "edf_max_queue_frames": self.edf_max_queue_frames,
            "skip_gpu_for_raw": self.skip_gpu_for_raw,
            "skip_raw_max_pending_batches": self.skip_raw_max_pending_batches,
            "skip_crossfade_copy": self.skip_crossfade_copy,
            "webrtc_yuv_in_compose": self.webrtc_yuv_in_compose,
            "whisper_stream": self.whisper_stream_enabled,
            "unet_cudagraphs_mode": getattr(unet_model, "cudagraphs_mode", None),
        }

    def start(self) -> None:
        if self.scheduler_thread is not None:
            return
        self.scheduler_thread = threading.Thread(target=self._run_loop, daemon=True, name="hls-gpu-scheduler")
        self.scheduler_thread.start()
        print(
            "🎛️  HLS GPU scheduler started "
            f"(max_combined_batch_size={self.max_combined_batch_size}, "
            f"fixed_batch_sizes={self.fixed_batch_sizes}, "
            f"startup_slice_size={self.startup_slice_size}, "
            f"startup_chunk_duration_seconds={self.startup_chunk_duration_seconds:.2f}, "
            f"startup_chunk_count={self.startup_chunk_count}, "
            f"aggressive_fill_max_active_jobs={self.aggressive_fill_max_active_jobs}, "
            f"compose_workers={self.compose_executor._max_workers}, "
            f"encode_workers={self.encode_executor._max_workers}, "
            f"webrtc_pose_crossfade_frames={self.webrtc_pose_crossfade_frames}, "
            f"gpu_batch_timing_log_interval={self.gpu_batch_timing_log_interval}, "
            f"gpu_batch_timing_slow_s={self.gpu_batch_timing_slow_s:.3f}, "
            f"gpu_stage_sync_timing={self.gpu_stage_sync_timing})"
        )
        if self._pipeline_flags_non_default():
            config = self._pipeline_config()
            print(
                "🎛️  HLS GPU scheduler 300fps flags: "
                + " ".join(f"{key}={value}" for key, value in config.items())
            )
            if self.pipeline_depth > 1 and self.gpu_stage_sync_timing:
                print(
                    "🎛️  HLS_GPU_PIPELINE_DEPTH>1 never host-syncs inside a batch; "
                    "HLS_GPU_STAGE_SYNC_TIMING is ignored (use HLS_GPU_EVENT_TIMING=1)"
                )
        if self.vae_calibration_capture:
            limit_label = (
                str(self.vae_calibration_max_batches)
                if self.vae_calibration_max_batches > 0
                else "unlimited"
            )
            print(
                "🧪 VAE calibration capture enabled "
                f"(dir={self.vae_calibration_dir}, max_batches={limit_label})"
            )
        if self.unet_calibration_capture:
            limit_label = (
                str(self.unet_calibration_max_batches)
                if self.unet_calibration_max_batches > 0
                else "unlimited"
            )
            print(
                "🧪 UNet calibration capture enabled "
                f"(dir={self.unet_calibration_dir}, max_batches={limit_label})"
            )

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.scheduler_thread is not None:
            self.scheduler_thread.join(timeout=10)
        self.prep_executor.shutdown(wait=False, cancel_futures=True)
        self.prep_subtask_executor.shutdown(wait=False, cancel_futures=True)
        self.backfill_executor.shutdown(wait=False, cancel_futures=True)
        self.compose_executor.shutdown(wait=False, cancel_futures=True)
        self.encode_executor.shutdown(wait=False, cancel_futures=True)
        print("🎛️  HLS GPU scheduler stopped")

    def submit_stream(
        self,
        session,
        request_id: str,
        audio_path: str,
        generation_fps: int,
        start_offset_seconds: Optional[float],
        cancel_event: threading.Event,
        completion_future,
        main_loop,
        *,
        output_mode: str = "hls",
        frame_callback: Optional[Callable[[object, int, int], None]] = None,
        frame_batch_callback: Optional[Callable[[list, int, int], None]] = None,
        generation_complete_callback: Optional[Callable[[str, Optional[str]], None]] = None,
        exact_silence: bool = False,
    ) -> bool:
        submitted_at = time.time()
        with self.condition:
            pending_count = len(self.jobs) + len(self.preparing_requests)
            if pending_count >= self.max_pending_jobs:
                return False
            self.preparing_requests.add(request_id)

        self.prep_executor.submit(
            self._prepare_job,
            session,
            request_id,
            audio_path,
            generation_fps,
            start_offset_seconds,
            cancel_event,
            completion_future,
            main_loop,
            output_mode,
            frame_callback,
            frame_batch_callback,
            generation_complete_callback,
            submitted_at,
            exact_silence,
        )
        return True

    def submit_webrtc_stream(
        self,
        session,
        request_id: str,
        audio_path: str,
        generation_fps: int,
        cancel_event: threading.Event,
        completion_future,
        main_loop,
        frame_callback: Callable[[object, int, int], None],
        generation_complete_callback: Callable[[str, Optional[str]], None],
        start_offset_seconds: Optional[float] = None,
        frame_batch_callback: Optional[Callable[[list, int, int], None]] = None,
        exact_silence: bool = False,
    ) -> bool:
        return self.submit_stream(
            session=session,
            request_id=request_id,
            audio_path=audio_path,
            generation_fps=generation_fps,
            start_offset_seconds=start_offset_seconds,
            cancel_event=cancel_event,
            completion_future=completion_future,
            main_loop=main_loop,
            output_mode="webrtc",
            frame_callback=frame_callback,
            frame_batch_callback=frame_batch_callback,
            generation_complete_callback=generation_complete_callback,
            exact_silence=exact_silence,
        )

    def get_stats(self) -> dict:
        with self.condition:
            stats = self._get_stats_locked()
            # Added code: only present when a 300 fps plan flag is non-default,
            # so the default stats payload is unchanged.
            if self._pipeline_flags_non_default():
                stats["pipeline"] = self._pipeline_config()
                if self.gpu_event_timing:
                    stats["capacity"] = self.get_capacity_stats()
            return stats

    def get_capacity_stats(self, include_batches: bool = False) -> dict:
        """Added code (plan item 0.2): capacity telemetry totals since start.

        Totals are monotonic, so a caller computes a window by differencing two
        snapshots. Event-based GPU times need HLS_GPU_EVENT_TIMING=1; host
        (sync) stage times are recorded as host_*_ms for the event-vs-sync gate.
        """
        with self._cap_lock:
            totals = dict(self._cap_totals)
            batches = list(self._cap_batches) if include_batches else None
        totals["wall_s"] = time.monotonic() - self._cap_started_at
        totals["scheduler_thread_cpu_ms"] = self._scheduler_thread_cpu_s() * 1000.0
        out = {
            "enabled": bool(self.gpu_event_timing),
            "config": self._pipeline_config(),
            "totals": totals,
            "derived": self._derive_capacity(totals),
        }
        if include_batches:
            out["batches"] = batches
        return out

    @staticmethod
    def _derive_capacity(totals: dict) -> dict:
        wall = totals.get("wall_s", 0.0) or 1e-9
        batches = totals.get("batches", 0.0) or 0.0
        gpu_batches = totals.get("gpu_batches", 0.0) or 0.0

        def per(key, count):
            return round(totals.get(key, 0.0) / count, 4) if count else None

        return {
            "batches": int(batches),
            "gpu_batches": int(gpu_batches),
            "gpu_busy_fraction": round(totals.get("gpu_span_ms", 0.0) / 1000.0 / wall, 4),
            "gpu_ms_per_batch": per("gpu_span_ms", gpu_batches),
            "idle_gap_ms_per_batch": per("idle_gap_ms", totals.get("idle_gap_count", 0.0)),
            "callback_ms_per_batch": per("callback_ms", batches),
            "feeder_cpu_ms_per_batch": per("feeder_cpu_ms", batches),
            "host_wait_ms_per_batch": per("host_wait_ms", gpu_batches),
            "ring_wait_ms_total": round(totals.get("ring_wait_ms", 0.0), 3),
            "fill": (
                round(totals.get("actual_frames", 0.0) / totals["padded_frames"], 4)
                if totals.get("padded_frames") else None
            ),
            "jobs_per_batch": per("jobs", batches),
            "gpu_fps": round(totals.get("actual_frames", 0.0) / wall, 2),
            "raw_frames": int(totals.get("raw_frames", 0.0)),
            "h2d_ms_per_batch": per("h2d_ms", gpu_batches),
            "unet_ms_per_batch": per("unet_ms", gpu_batches),
            "vae_ms_per_batch": per("vae_ms", gpu_batches),
            "d2h_ms_per_batch": per("d2h_ms", gpu_batches),
        }

    def _scheduler_thread_cpu_s(self) -> float:
        thread = getattr(self, "scheduler_thread", None)
        ident = getattr(thread, "ident", None)
        if (
            ident is None
            or not thread.is_alive()
            or not hasattr(time, "pthread_getcpuclockid")
        ):
            return 0.0
        try:
            return time.clock_gettime(time.pthread_getcpuclockid(ident))
        except (OSError, ValueError, OverflowError):
            return 0.0

    def _get_stats_locked(self) -> dict:
        # Body unchanged from the original get_stats (caller holds the lock).
        return {
                "queued_or_active_jobs": len(self.jobs),
                "preparing_jobs": len(self.preparing_requests),
                "prep_queue_depth": len(self.preparing_requests),
                "max_combined_batch_size": self.max_combined_batch_size,
                "startup_slice_size": self.startup_slice_size,
                "startup_chunk_duration_seconds": self.startup_chunk_duration_seconds,
                "startup_chunk_count": self.startup_chunk_count,
                "aggressive_fill_max_active_jobs": self.aggressive_fill_max_active_jobs,
                "startup_pending_jobs": len(
                    [job for job in self.jobs.values() if self._is_startup_job(job)]
                ),
                "compose_workers": self.compose_executor._max_workers,
                "encode_workers": self.encode_executor._max_workers,
                "jobs": [
                    {
                        "request_id": job.request_id,
                        "session_id": job.session_id,
                        "output_mode": job.output_mode,
                        "current_frame_idx": job.current_frame_idx,
                        "total_frames": job.total_frames,
                        "start_offset_frames": job.start_offset_frames,
                        "composed_frame_idx": job.composed_frame_idx,
                        "chunk_index": job.chunk_index,
                        "total_chunks": job.total_chunks,
                        "frame_buffer_len": len(job.frame_buffer),
                        "frames_per_chunk": job.frames_per_chunk,
                        "startup_chunk_frames": job.startup_chunk_frames,
                        "startup_chunk_count": job.startup_chunk_count,
                        "conditioning_ready_frames": job.conditioning_ready_frames,
                        "conditioning_complete": job.conditioning_complete,
                        "exact_silence": job.exact_silence,
                        "frames_until_next_chunk": max(0, self._next_chunk_target_frames(job) - len(job.frame_buffer)),
                        "frame_buffer_fill_pct": round(
                            (len(job.frame_buffer) / job.frames_per_chunk) if job.frames_per_chunk else 0.0,
                            3,
                        ),
                        "max_frame_buffer_len": job.max_frame_buffer_len,
                        "pending_composes": len(job.compose_tasks),
                        "pending_encodes": len(job.encode_tasks),
                        "max_pending_composes": job.max_pending_composes,
                        "max_pending_encodes": job.max_pending_encodes,
                        "startup_pending": self._is_startup_job(job),
                        "cancel_requested": job.cancel_event.is_set(),
                        "last_progress_age_s": round(time.time() - job.last_progress_at, 3),
                        "prep_total_s": round(job.prep_total_s, 3),
                        "prep_queue_wait_s": round(job.prep_queue_wait_s, 3),
                        "prep_work_s": round(job.prep_work_s, 3),
                        "avatar_load_s": round(job.avatar_load_s, 3),
                        "audio_feature_s": round(job.audio_feature_s, 3),
                        "whisper_chunk_s": round(job.whisper_chunk_s, 3),
                        "queue_wait_s": round(self._queue_wait_s(job), 3),
                        "time_to_first_chunk_s": round(self._time_to_first_chunk_s(job), 3),
                        "scheduler_turns": job.scheduler_turns,
                        "avg_gpu_batch_s": round(self._safe_avg(job.gpu_batch_total_s, job.gpu_batch_count), 4),
                        "avg_assemble_s": round(self._safe_avg(job.batch_assembly_total_s, job.gpu_batch_count), 4),
                        "avg_copy_s": round(self._safe_avg(job.gpu_copy_total_s, job.gpu_batch_count), 4),
                        "avg_pe_s": round(self._safe_avg(job.pe_total_s, job.gpu_batch_count), 4),
                        "avg_unet_s": round(self._safe_avg(job.unet_total_s, job.gpu_batch_count), 4),
                        "avg_vae_s": round(self._safe_avg(job.vae_total_s, job.gpu_batch_count), 4),
                        "avg_compose_queue_wait_s": round(self._safe_avg(job.compose_queue_wait_total_s, job.compose_batch_count), 4),
                        "avg_compose_s": round(self._safe_avg(job.compose_total_s, job.compose_batch_count), 4),
                        "max_compose_queue_wait_s": round(job.max_compose_queue_wait_s, 4),
                        "max_compose_s": round(job.max_compose_s, 4),
                        "avg_frame_callback_s": round(self._safe_avg(job.frame_callback_total_s, job.frame_callback_count), 4),
                        "max_frame_callback_s": round(job.frame_callback_max_s, 4),
                        "frame_callback_count": job.frame_callback_count,
                        "avg_encode_queue_wait_s": round(self._safe_avg(job.encode_queue_wait_total_s, job.chunks_encoded), 4),
                        "avg_encode_s": round(self._safe_avg(job.encode_total_s, job.chunks_encoded), 4),
                        "max_encode_queue_wait_s": round(job.max_encode_queue_wait_s, 4),
                        "max_encode_s": round(job.max_encode_s, 4),
                        "post_generation_drain_s": round(self._post_generation_drain_s(job), 4),
                    }
                    for job in self.jobs.values()
                ],
            }

    def _prepare_job(
        self,
        session,
        request_id: str,
        audio_path: str,
        generation_fps: int,
        start_offset_seconds: Optional[float],
        cancel_event: threading.Event,
        completion_future,
        main_loop,
        output_mode: str,
        frame_callback: Optional[Callable[[object, int, int], None]],
        frame_batch_callback: Optional[Callable[[list, int, int], None]],
        generation_complete_callback: Optional[Callable[[str, Optional[str]], None]],
        submitted_at: float,
        exact_silence: bool = False,
    ) -> None:
        prep_started_at = time.time()
        exact_silence = bool(exact_silence and output_mode == "webrtc")
        is_hls_output = output_mode == "hls"
        audio_copy_candidate_path = (
            str((session.segment_dir / request_id) / "chunk_audio.m4a")
            if is_hls_output
            else None
        )
        audio_copy_path = None
        try:
            if cancel_event.is_set():
                self._finish_before_enqueue(
                    request_id,
                    session,
                    audio_path,
                    audio_copy_candidate_path,
                    completion_future,
                    main_loop,
                    "cancelled",
                    output_mode=output_mode,
                    generation_complete_callback=generation_complete_callback,
                )
                return

            weight_dtype = getattr(self.manager, "unet_dtype", torch.float16)
            prepare_chunk_audio_copy_source = None
            if is_hls_output:
                from scripts.api_avatar import prepare_chunk_audio_copy_source

            # Local modification: this differs from the original MuseTalk code.
            # Avatar load and audio feature extraction are prepared in parallel.
            avatar_future = self.prep_subtask_executor.submit(
                self._timed_call,
                self.manager._get_or_load_avatar,
                session.avatar_id,
                session.batch_size,
            )
            audio_feature_future = self.prep_subtask_executor.submit(
                self._timed_call,
                self.manager.audio_processor.get_audio_feature,
                audio_path,
                0,
                weight_dtype,
            )
            audio_copy_future = None
            if prepare_chunk_audio_copy_source is not None and audio_copy_candidate_path is not None:
                audio_copy_future = self.prep_subtask_executor.submit(
                    self._timed_call,
                    prepare_chunk_audio_copy_source,
                    audio_path,
                    audio_copy_candidate_path,
                )

            avatar, avatar_load_s = avatar_future.result()
            pose_avatars = {"default": avatar}
            if output_mode == "webrtc":
                prepared_pose_avatar_ids = dict(
                    getattr(session, "prepared_pose_avatar_ids", {}) or {}
                )
                for pose_id, prepared_avatar_id in prepared_pose_avatar_ids.items():
                    if pose_id == "default" or prepared_avatar_id == session.avatar_id:
                        pose_avatars[pose_id] = avatar
                        continue
                    pose_avatars[pose_id] = self.manager._get_or_load_avatar(
                        prepared_avatar_id,
                        session.batch_size,
                    )
            (whisper_input_features, _librosa_length), audio_feature_s = audio_feature_future.result()
            audio_copy_prep_s = 0.0
            if audio_copy_future is not None:
                try:
                    audio_copy_path, audio_copy_prep_s = audio_copy_future.result()
                except Exception as audio_copy_exc:
                    print(f"⚠️  [{request_id}] reusable AAC sidecar unavailable: {audio_copy_exc}")

            if (
                hasattr(session, "idle_cycle_frames")
                and session.idle_cycle_frames is None
                and hasattr(avatar, "input_latent_cycle_tensor")
            ):
                session.idle_cycle_frames = len(avatar.input_latent_cycle_tensor)

            if whisper_input_features is None:
                raise RuntimeError("Audio feature extraction failed")

            whisper_chunk_start = time.time()
            whisper_stream = self._whisper_stream_for_thread()
            if whisper_stream is None:
                whisper_feature, total_frames = self.manager.audio_processor.encode_whisper_feature(
                    whisper_input_features,
                    self.manager.device,
                    weight_dtype,
                    self.manager.whisper,
                    _librosa_length,
                    fps=generation_fps,
                    audio_padding_length_left=self.manager.args.audio_padding_length_left,
                    audio_padding_length_right=self.manager.args.audio_padding_length_right,
                )
            else:
                # Added code (plan item 1.4, MUSETALK_WHISPER_STREAM=1): same
                # call on this prep thread's own stream. The result is a CPU
                # tensor; the stream is drained before it is used.
                with torch.cuda.stream(whisper_stream):
                    whisper_feature, total_frames = self.manager.audio_processor.encode_whisper_feature(
                        whisper_input_features,
                        self.manager.device,
                        weight_dtype,
                        self.manager.whisper,
                        _librosa_length,
                        fps=generation_fps,
                        audio_padding_length_left=self.manager.args.audio_padding_length_left,
                        audio_padding_length_right=self.manager.args.audio_padding_length_right,
                    )
                whisper_stream.synchronize()
            whisper_chunk_s = time.time() - whisper_chunk_start

            if cancel_event.is_set():
                self._finish_before_enqueue(
                    request_id,
                    session,
                    audio_path,
                    audio_copy_path,
                    completion_future,
                    main_loop,
                    "cancelled",
                    output_mode=output_mode,
                    generation_complete_callback=generation_complete_callback,
                )
                return

            segment_duration = getattr(session, "segment_duration", None)
            if segment_duration is None:
                segment_duration = getattr(session, "chunk_duration", 1)
            frames_per_chunk = max(1, int(round(float(segment_duration) * generation_fps)))
            startup_chunk_frames = self._startup_chunk_frames(frames_per_chunk, generation_fps)
            startup_chunk_count = self.startup_chunk_count if startup_chunk_frames < frames_per_chunk else 0
            total_chunks = self._estimate_total_chunks(
                total_frames=total_frames,
                frames_per_chunk=frames_per_chunk,
                startup_chunk_frames=startup_chunk_frames,
                startup_chunk_count=startup_chunk_count,
            )
            pose_plan_router = None
            active_pose_plan = (
                getattr(session, "active_pose_plan", None)
                if output_mode == "webrtc"
                else None
            )
            motion_router = getattr(session, "live_pose_router", None) if output_mode == "webrtc" else None
            if exact_silence and getattr(motion_router, "motion_bank", None) is not None:
                # Original-upload PCM proved entirely zero. Preserve the full
                # audio/frame schedule, but use the neutral physical source.
                active_pose_plan = {
                    "version": 2, "clock": "audio_progress", "switch_mode": "next_boundary",
                    "on_complete": "neutral_resting",
                    "segments": [{"at_permille": 0, "pose_id": "neutral_resting"}],
                }
            if getattr(motion_router, "motion_bank", None) is not None and not active_pose_plan:
                active_pose_plan = {
                    "version": 2, "clock": "audio_progress", "switch_mode": "next_boundary",
                    "on_complete": "neutral_resting",
                    "segments": [{"at_permille": 0, "pose_id": getattr(session, "live_pose_id", None) or "speaking_direct"}],
                }
            if active_pose_plan:
                pose_plan_router = getattr(
                    session,
                    "live_pose_router",
                    None,
                )
                if pose_plan_router is None:
                    raise RuntimeError(
                        "A v2 pose plan requires the live pose router"
                    )
                session.compiled_pose_plan = pose_plan_router.queue_pose_plan(
                    active_pose_plan,
                    total_frames,
                    float(generation_fps),
                    hold_last_pose=True,
                )
            if pose_plan_router is not None and getattr(pose_plan_router, "motion_bank", None) is not None:
                pose_plan_router.motion_initial_pose = session.idle_track.get_pose_status()["current_pose_id"]
            start_offset_frames = 0
            timing_debug = None
            if output_mode == "webrtc" and start_offset_seconds is None:
                cycle_frames = None
                latent_cycle = getattr(avatar, "input_latent_cycle_tensor", None)
                if isinstance(latent_cycle, torch.Tensor) and latent_cycle.shape[0] > 0:
                    cycle_frames = int(latent_cycle.shape[0])
                else:
                    coord_cycle = getattr(avatar, "coord_list_cycle", None)
                    if coord_cycle:
                        cycle_frames = len(coord_cycle)

                video_track = getattr(session, "idle_track", None)
                if hasattr(video_track, "capture_idle_sync_timing"):
                    timing_debug = video_track.capture_idle_sync_timing(
                        generation_fps=float(generation_fps),
                        cycle_frames=cycle_frames,
                        reveal_delay_seconds=float(
                            getattr(session, "webrtc_live_reveal_delay_seconds", 0.0) or 0.0
                        ),
                        hold=getattr(video_track, "motion_bank", None) is None,
                    )
                    try:
                        start_offset_frames = max(0, int(timing_debug.get("offset_frames") or 0))
                    except (TypeError, ValueError):
                        start_offset_frames = 0
                    timing_debug["offset_frames"] = start_offset_frames
                    timing_debug["offset_seconds"] = (
                        start_offset_frames / float(generation_fps)
                        if generation_fps > 0
                        else 0.0
                    )
                    try:
                        session.live_timing = timing_debug
                    except Exception:
                        pass
                    live_pose_router = getattr(
                        session,
                        "live_pose_router",
                        None,
                    )
                    if (
                        live_pose_router is not None
                        and hasattr(
                            live_pose_router,
                            "align_first_queued_pose",
                        )
                    ):
                        try:
                            timing_debug["live_pose_alignment"] = (
                                live_pose_router.align_first_queued_pose(
                                    int(
                                        timing_debug.get(
                                            "target_source_frame_index",
                                            0,
                                        )
                                        or 0
                                    ),
                                    float(generation_fps),
                                )
                            )
                            if pose_plan_router is live_pose_router:
                                session.compiled_pose_plan = (
                                    live_pose_router.get_compiled_pose_plan()
                                )
                        except Exception as exc:
                            timing_debug["live_pose_alignment_error"] = str(exc)
                            print(
                                f"⚠️ [{request_id}] Could not phase-align "
                                f"first live pose: {exc}",
                                flush=True,
                            )
                    print(
                        f"🎬 [{request_id}] WebRTC idle sync offset: "
                        f"source_frame={timing_debug.get('idle_source_frame_index')} "
                        f"offset_frames={start_offset_frames} "
                        f"offset_seconds={timing_debug.get('offset_seconds'):.3f} "
                        f"cycle_frames={cycle_frames} hold={timing_debug.get('hold_enabled')}",
                        flush=True,
                    )
                else:
                    start_offset_seconds = 0.0

            if output_mode != "webrtc" or start_offset_seconds is not None:
                try:
                    start_offset_value = max(0.0, float(start_offset_seconds or 0.0))
                except (TypeError, ValueError):
                    start_offset_value = 0.0
                start_offset_frames = int(round(start_offset_value * generation_fps))

            if pose_plan_router is not None:
                compiled_plan = pose_plan_router.get_compiled_pose_plan() or {}
                if compiled_plan.get("status") != "compiled":
                    pose_plan_router.align_first_queued_pose(
                        start_offset_frames,
                        float(generation_fps),
                    )
                    session.compiled_pose_plan = (
                        pose_plan_router.get_compiled_pose_plan()
                    )

            initial_ready_frames = self._initial_conditioning_frames(
                total_frames=total_frames,
                frames_per_chunk=frames_per_chunk,
                startup_chunk_frames=startup_chunk_frames,
                startup_chunk_count=startup_chunk_count,
            )

            idle_frames_future = None
            if is_hls_output:
                idle_frame_target = max(8, min(24, int(generation_fps * 0.8)))
                # Local modification: this differs from the original MuseTalk code.
                # Idle-frame preparation overlaps with conditioning setup during prep.
                idle_frames_future = self.prep_subtask_executor.submit(
                    self._timed_call,
                    avatar._get_idle_frames,
                    idle_frame_target,
                )

            if initial_ready_frames > 0:
                initial_prompts = self.manager.audio_processor.build_audio_prompts(
                    whisper_feature=whisper_feature,
                    num_frames=total_frames,
                    fps=generation_fps,
                    audio_padding_length_left=self.manager.args.audio_padding_length_left,
                    audio_padding_length_right=self.manager.args.audio_padding_length_right,
                    start_frame=0,
                    end_frame=initial_ready_frames,
                )
                initial_conditioning = self._apply_positional_encoding_cpu(initial_prompts)
                conditioning_chunks = torch.empty(
                    (total_frames,) + tuple(initial_conditioning.shape[1:]),
                    dtype=initial_conditioning.dtype,
                    pin_memory=torch.cuda.is_available(),
                )
                conditioning_chunks[:initial_ready_frames].copy_(initial_conditioning)
            else:
                conditioning_chunks = torch.empty((0, 0, 0), dtype=torch.float32)

            idle_frames = []
            if idle_frames_future is not None:
                idle_frames, _idle_frame_s = idle_frames_future.result()
            crossfade_tail_frames = 0
            if is_hls_output and idle_frames:
                crossfade_tail_frames = max(4, int(generation_fps * 0.15))
                crossfade_tail_frames = min(
                    crossfade_tail_frames,
                    len(idle_frames),
                    frames_per_chunk - 1 if frames_per_chunk > 1 else crossfade_tail_frames,
                )

            queued_at = time.time()
            job = HLSStreamJob(
                request_id=request_id,
                session_id=session.session_id,
                session=session,
                avatar=avatar,
                pose_avatars=pose_avatars,
                audio_path=audio_path,
                audio_copy_path=audio_copy_path,
                chunk_output_dir=(
                    session.segment_dir / request_id
                    if is_hls_output
                    else Path("chunks") / "_webrtc_scheduler" / request_id
                ),
                generation_fps=generation_fps,
                batch_size=max(1, int(session.batch_size)),
                conditioning_chunks=conditioning_chunks,
                conditioning_ready_frames=initial_ready_frames,
                conditioning_complete=initial_ready_frames >= total_frames,
                total_frames=total_frames,
                frames_per_chunk=frames_per_chunk,
                startup_chunk_frames=startup_chunk_frames,
                startup_chunk_count=startup_chunk_count,
                total_chunks=total_chunks,
                start_offset_frames=start_offset_frames,
                cancel_event=cancel_event,
                completion_future=completion_future,
                main_loop=main_loop,
                output_mode=output_mode,
                exact_silence=exact_silence,
                frame_callback=frame_callback,
                frame_batch_callback=frame_batch_callback,
                generation_complete_callback=generation_complete_callback,
                idle_frames=idle_frames,
                crossfade_tail_frames=crossfade_tail_frames,
                submitted_at=submitted_at,
                prep_started_at=prep_started_at,
                queued_at=queued_at,
                prep_total_s=queued_at - submitted_at,
                prep_queue_wait_s=prep_started_at - submitted_at,
                prep_work_s=queued_at - prep_started_at,
                avatar_load_s=avatar_load_s,
                audio_feature_s=audio_feature_s,
                audio_copy_prep_s=audio_copy_prep_s,
                whisper_chunk_s=whisper_chunk_s,
            )

            self._set_request_status(request_id, "queued")
            with self.condition:
                self.preparing_requests.discard(request_id)
                self.jobs[request_id] = job
                self.condition.notify_all()

            if total_frames == 0:
                self._finalize_job(job, "completed")
                return

            if not job.conditioning_complete:
                self.backfill_executor.submit(
                    self._backfill_conditioning_chunks,
                    job,
                    whisper_feature,
                )

            print(
                f"🎛️  [{request_id}] queued for shared {output_mode.upper()} GPU scheduler "
                f"(frames={total_frames}, chunks={total_chunks}, batch_size={job.batch_size}, "
                f"ready={job.conditioning_ready_frames}, "
                f"prep={job.prep_total_s:.2f}s, prep_wait={job.prep_queue_wait_s:.2f}s, "
                f"prep_work={job.prep_work_s:.2f}s, audio_copy={job.audio_copy_prep_s:.2f}s)"
            )
        except Exception as exc:
            print(f"❌ [{request_id}] {output_mode.upper()} prep failed: {exc}")
            traceback.print_exc()
            self._finish_before_enqueue(
                request_id,
                session,
                audio_path,
                audio_copy_path or audio_copy_candidate_path,
                completion_future,
                main_loop,
                "failed",
                error_message=str(exc),
                output_mode=output_mode,
                generation_complete_callback=generation_complete_callback,
            )

    def _whisper_stream_for_thread(self):
        """Added code: per-prep-thread CUDA stream for MUSETALK_WHISPER_STREAM=1."""
        if not self.whisper_stream_enabled or not torch.cuda.is_available():
            return None
        device = torch.device(getattr(self.manager, "device", "cuda"))
        if device.type != "cuda":
            return None
        local = getattr(self, "_whisper_streams", None)
        if local is None:
            return None
        stream = getattr(local, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            local.stream = stream
        return stream

    @staticmethod
    # Local modification: this differs from the original MuseTalk code.
    # Small helper used to parallelize prep work while still reporting timings.
    def _timed_call(fn, *args, **kwargs):
        started_at = time.time()
        result = fn(*args, **kwargs)
        return result, time.time() - started_at

    def _finish_before_enqueue(
        self,
        request_id: str,
        session,
        audio_path: str,
        audio_copy_path: Optional[str],
        completion_future,
        main_loop,
        status: str,
        error_message: Optional[str] = None,
        output_mode: str = "hls",
        generation_complete_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    ) -> None:
        with self.condition:
            self.preparing_requests.discard(request_id)
            self.condition.notify_all()

        if output_mode == "hls":
            self.hls_session_manager.finish_live_playlist(session)
            session.active_stream = None
        else:
            session.active_stream = None
            if generation_complete_callback is not None:
                try:
                    generation_complete_callback(status, error_message)
                except Exception as callback_exc:
                    print(f"⚠️  [{request_id}] WebRTC completion callback failed: {callback_exc}")
        if hasattr(session, "cancel_requested"):
            session.cancel_requested = False
        self._set_request_status(request_id, status)
        self._resolve_completion(completion_future, main_loop, status, error_message)
        if output_mode == "hls":
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass
            if audio_copy_path:
                try:
                    Path(audio_copy_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _initial_conditioning_frames(
        self,
        *,
        total_frames: int,
        frames_per_chunk: int,
        startup_chunk_frames: int,
        startup_chunk_count: int,
    ) -> int:
        if total_frames <= 0:
            return 0

        initial_frames = frames_per_chunk * 2
        if startup_chunk_count > 0 and startup_chunk_frames > 0:
            initial_frames = (startup_chunk_frames * startup_chunk_count) + frames_per_chunk

        return min(total_frames, max(frames_per_chunk, initial_frames))

    def _backfill_conditioning_chunks(self, job: HLSStreamJob, whisper_feature: torch.Tensor) -> None:
        try:
            block_frames = max(job.frames_per_chunk, self.max_combined_batch_size * 2)
            next_frame = job.conditioning_ready_frames

            while next_frame < job.total_frames:
                if job.cancel_event.is_set() or job.finalized:
                    return

                end_frame = min(job.total_frames, next_frame + block_frames)
                prompts = self.manager.audio_processor.build_audio_prompts(
                    whisper_feature=whisper_feature,
                    num_frames=job.total_frames,
                    fps=job.generation_fps,
                    audio_padding_length_left=self.manager.args.audio_padding_length_left,
                    audio_padding_length_right=self.manager.args.audio_padding_length_right,
                    start_frame=next_frame,
                    end_frame=end_frame,
                )
                conditioning = self._apply_positional_encoding_cpu(prompts)
                expected_frames = end_frame - next_frame
                if len(conditioning) != expected_frames:
                    raise RuntimeError(
                        f"conditioning backfill size mismatch: expected {expected_frames}, got {len(conditioning)}"
                    )

                with job.conditioning_lock:
                    job.conditioning_chunks[next_frame:end_frame].copy_(conditioning)
                    job.conditioning_ready_frames = end_frame
                    if end_frame >= job.total_frames:
                        job.conditioning_complete = True

                next_frame = end_frame
                with self.condition:
                    self.condition.notify_all()

            with job.conditioning_lock:
                job.conditioning_complete = True
            with self.condition:
                self.condition.notify_all()
        except Exception as exc:
            job.error_message = f"conditioning backfill failed: {exc}"
            with job.conditioning_lock:
                job.conditioning_complete = True
            with self.condition:
                self.condition.notify_all()
            print(f"❌ [{job.request_id}] conditioning backfill failed: {exc}")
            traceback.print_exc()

    def _run_loop(self) -> None:
        if self.pipeline_depth > 1:
            self._run_loop_pipelined()
            return
        while True:
            self._drain_completed_composes()
            self._drain_completed_encodes()
            self._finalize_ready_jobs()
            self._finalize_cancelled_jobs()

            with self.condition:
                if self.stop_event.is_set() and not self.jobs and not self.preparing_requests:
                    break

                selected = self._select_jobs_for_batch_locked()
                if not selected:
                    self.condition.wait(timeout=0.02)
                    continue

            try:
                self._run_generation_batch(selected)
            except Exception as exc:
                print(f"❌ HLS scheduler batch failed: {exc}")
                traceback.print_exc()
                for job, _ in selected:
                    job.error_message = str(exc)

            self._drain_completed_composes()
            self._drain_completed_encodes()
            self._finalize_ready_jobs()
            self._finalize_cancelled_jobs()

    def _run_loop_pipelined(self) -> None:
        """Added code (plan item 1.4, HLS_GPU_PIPELINE_DEPTH>=2).

        Batch N+1 is selected, assembled and launched before batch N is
        collected, so the GPU always has the next batch queued while this
        thread waits on N, composes, and runs callbacks. Selection counts
        in-flight frames as scheduled (current_frame_idx advances at submit),
        so no frame is generated twice, and per-job compose sequence numbers
        are assigned at collect in submit order, so frame order is unchanged.
        """
        inflight = collections.deque()
        while True:
            self._drain_completed_composes()
            self._drain_completed_encodes()
            self._finalize_ready_jobs()
            self._finalize_cancelled_jobs()

            selected = []
            with self.condition:
                if (
                    self.stop_event.is_set()
                    and not self.jobs
                    and not self.preparing_requests
                    and not inflight
                ):
                    break
                if len(inflight) < self.pipeline_depth:
                    selected = self._select_jobs_for_batch_locked()
                if not selected and not inflight:
                    self.condition.wait(timeout=0.02)
                    continue

            if selected:
                try:
                    batch = self._submit_generation_batch(selected, pipelined=True)
                    if batch is not None:
                        inflight.append(batch)
                except Exception as exc:
                    print(f"❌ HLS scheduler batch submit failed: {exc}")
                    traceback.print_exc()
                    for job, _ in selected:
                        job.error_message = str(exc)

            if inflight and (len(inflight) >= self.pipeline_depth or not selected):
                batch = inflight.popleft()
                try:
                    self._collect_generation_batch(batch)
                except Exception as exc:
                    print(f"❌ HLS scheduler batch collect failed: {exc}")
                    traceback.print_exc()
                    for job in batch.jobs or []:
                        job.error_message = str(exc)

            self._drain_completed_composes()
            self._drain_completed_encodes()
            self._finalize_ready_jobs()
            self._finalize_cancelled_jobs()

    def _select_jobs_for_batch_locked(self):
        """Added code: policy dispatch. The defaults call _select_jobs_locked()
        and return its result unchanged."""
        if self.scheduler_policy == "edf":
            selected = self._select_jobs_edf_locked()
        else:
            selected = self._select_jobs_locked()
        if self.skip_gpu_for_raw and selected:
            selected = [
                (job, take)
                for job, take in selected
                if not (
                    job.exact_silence
                    and self._pending_compose_batches(job) >= self.skip_raw_max_pending_batches
                )
            ]
        return selected

    @staticmethod
    def _pending_compose_batches(job: HLSStreamJob, extra: int = 0) -> int:
        return (
            len(job.compose_tasks)
            + len(job.composed_batches)
            + int(job.gpu_inflight_batches)
            + int(extra)
        )

    # ------------------------------------------------------------------ EDF
    # Added code (plan item 1.9, HLS_SCHEDULER_POLICY=edf).
    def _select_jobs_edf_locked(self):
        jobs = self._ordered_schedulable_jobs_locked()
        if not jobs:
            return []
        now = time.time()
        capacity = self.max_combined_batch_size
        allocations: Dict[str, int] = {}
        state = {job.request_id: self._edf_job_state(job, now) for job in jobs}
        ordinal = {job.request_id: index for index, job in enumerate(jobs)}
        total = [0]

        def grant(job: HLSStreamJob, want: int) -> int:
            capacity_left = capacity - total[0]
            if capacity_left <= 0 or want <= 0:
                return 0
            take = min(int(want), self._remaining_frames(job, allocations), capacity_left)
            if take <= 0:
                return 0
            allocations[job.request_id] = allocations.get(job.request_id, 0) + take
            total[0] += take
            return take

        def slack_s(job: HLSStreamJob) -> float:
            st = state[job.request_id]
            return (st["slack_frames"] + allocations.get(job.request_id, 0)) / st["fps"]

        def eligible(job: HLSStreamJob) -> bool:
            st = state[job.request_id]
            if st["queue_cap"] is not None and (
                st["slack_frames"] + allocations.get(job.request_id, 0) + job.batch_size
                > st["queue_cap"]
            ):
                return False
            return not (self.max_runahead_s > 0 and slack_s(job) > self.max_runahead_s)

        def edf_key(job: HLSStreamJob):
            return (slack_s(job), ordinal[job.request_id])

        startup = [job for job in jobs if state[job.request_id]["startup_need"] > 0]
        warmed = [job for job in jobs if state[job.request_id]["startup_need"] <= 0]

        # 1. Warmed jobs about to underrun are served before any new stream.
        for job in sorted(warmed, key=edf_key):
            if slack_s(job) >= self.edf_urgent_s:
                break
            if eligible(job):
                grant(job, job.batch_size)

        # 2. Startup jobs: a prebuffer-sized first slice each, packed across
        #    jobs into full batches, closest-to-ready first, then arrival.
        for job in sorted(
            startup,
            key=lambda j: (state[j.request_id]["startup_need"], j.queued_at, ordinal[j.request_id]),
        ):
            need = state[job.request_id]["startup_need"] - allocations.get(job.request_id, 0)
            grant(job, need)

        # 3. Earliest deadline first on slack, one per-job batch slice at a
        #    time, re-ranked after every grant.
        while total[0] < capacity:
            granted = False
            for job in sorted(jobs, key=edf_key):
                if not eligible(job):
                    continue
                if grant(job, job.batch_size) > 0:
                    granted = True
                    break
            if not granted:
                break

        return [
            (job, allocations[job.request_id])
            for job in jobs
            if allocations.get(job.request_id, 0) > 0
        ]

    def _edf_startup_target(self, job: HLSStreamJob) -> int:
        target = self.edf_startup_frames
        if target <= 0:
            prebuffer_s = 0.0
            if job.output_mode == "webrtc":
                try:
                    prebuffer_s = float(getattr(job.session, "prebuffer_seconds", 0.0) or 0.0)
                except (TypeError, ValueError):
                    prebuffer_s = 0.0
            if prebuffer_s > 0:
                target = int(round(prebuffer_s * float(job.generation_fps or 0)))
            if target <= 0:
                target = (
                    job.startup_chunk_frames
                    if job.startup_chunk_count > 0 and job.startup_chunk_frames > 0
                    else job.frames_per_chunk
                )
        return max(0, min(int(target), int(job.total_frames)))

    @staticmethod
    def _consumer_depth_frames(job: HLSStreamJob) -> Optional[int]:
        """Frames delivered to the consumer and not yet played, if knowable."""
        session = job.session
        hook = getattr(session, "webrtc_playback_queue_frames", None)
        if not callable(hook):
            track = getattr(session, "idle_track", None) if job.output_mode == "webrtc" else None
            hook = getattr(track, "live_buffer_depth_frames", None)
        if not callable(hook):
            return None
        try:
            return max(0, int(hook()))
        except Exception:
            return None

    def _consumer_queue_cap(self, job: HLSStreamJob) -> Optional[int]:
        if self.edf_max_queue_frames > 0:
            return self.edf_max_queue_frames
        if job.output_mode != "webrtc":
            return None
        cap = getattr(getattr(job.session, "idle_track", None), "_max_queue", None)
        if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            return cap
        return None

    def _edf_job_state(self, job: HLSStreamJob, now: float) -> dict:
        fps = max(1.0, float(job.generation_fps or 20))
        startup_need = max(0, self._edf_startup_target(job) - job.current_frame_idx)
        # Scheduled (incl. in flight) but not yet handed to the consumer.
        pipeline = max(0, job.current_frame_idx - job.composed_frame_idx)
        depth = self._consumer_depth_frames(job)
        if depth is None:
            # No consumer counter: assume playout started with the first
            # delivered block and runs in real time.
            if job.first_chunk_appended_at is None:
                queued = float(job.composed_frame_idx)
            else:
                queued = max(
                    0.0,
                    job.composed_frame_idx - (now - job.first_chunk_appended_at) * fps,
                )
            queue_cap = None
        else:
            queued = float(depth)
            queue_cap = self._consumer_queue_cap(job)
        return {
            "fps": fps,
            "startup_need": startup_need,
            "slack_frames": queued + pipeline,
            "queue_cap": queue_cap,
        }

    def _select_jobs_locked(self):
        jobs = self._ordered_schedulable_jobs_locked()
        if not jobs:
            return []

        allocations: Dict[str, int] = {}
        total_batch = 0

        startup_jobs = [job for job in jobs if self._is_startup_job(job)]
        warmed_jobs = [job for job in jobs if not self._is_startup_job(job)]

        # MUST-KEEP STARTUP FAIRNESS LOGIC:
        # These first two rounds are the change that compressed the old
        # "1/2/3/4/5s" live_ready wave into a much tighter startup band where
        # most streams become live at roughly the same time.
        #
        # Round 1 gives startup jobs an initial slice before warmed jobs are
        # considered at all.
        # --- Round 1: startup jobs get a small initial slice ---
        if startup_jobs:
            total_batch = self._allocate_round(
                jobs=startup_jobs,
                allocations=allocations,
                total_batch=total_batch,
                slice_cap=self.startup_slice_size,
            )

        # MUST-KEEP STARTUP FAIRNESS LOGIC:
        # Round 2 spends remaining budget on startup jobs again so they can
        # finish their first chunk before warmed jobs absorb the batch. This is
        # the main reason live_ready became much more even across sessions.
        # --- Round 2: finish startup jobs to their first chunk target where possible ---
        if total_batch < self.max_combined_batch_size and startup_jobs:
            total_batch = self._allocate_startup_priority_round(
                jobs=startup_jobs,
                allocations=allocations,
                total_batch=total_batch,
            )

        # --- Round 3: warmed jobs get their per-job batch_size ---
        if total_batch < self.max_combined_batch_size and warmed_jobs:
            total_batch = self._allocate_round(
                jobs=warmed_jobs,
                allocations=allocations,
                total_batch=total_batch,
                slice_cap=None,
            )

        # --- Round 4: finish the jobs that are closest to emitting a chunk ---
        # The hot path bottleneck is no longer compose; it is how many GPU
        # turns a stream needs before it can emit the next HLS chunk. Giving
        # every warmed job the same tiny slice keeps many jobs perpetually
        # "almost ready". This round spends any spare capacity first on the
        # jobs that can finish their next chunk with the fewest extra frames.
        if total_batch < self.max_combined_batch_size:
            all_schedulable = startup_jobs + warmed_jobs
            if all_schedulable:
                total_batch = self._allocate_chunk_completion_round(
                    jobs=all_schedulable,
                    allocations=allocations,
                    total_batch=total_batch,
                )

        # --- Round 5: ALWAYS fill any remaining GPU capacity fairly ---
        if total_batch < self.max_combined_batch_size:
            all_schedulable = startup_jobs + warmed_jobs
            if all_schedulable:
                total_batch = self._fill_remaining_capacity(
                    jobs=all_schedulable,
                    allocations=allocations,
                    total_batch=total_batch,
                )

        return [
            (job, allocations[job.request_id])
            for job in jobs
            if allocations.get(job.request_id, 0) > 0
        ]

    def _ordered_schedulable_jobs_locked(self) -> list[HLSStreamJob]:
        jobs = list(self.jobs.values())
        if not jobs:
            return []

        ordered_jobs: list[HLSStreamJob] = []
        count = len(jobs)
        start_idx = self.selection_cursor % count
        self.selection_cursor = (start_idx + 1) % count

        for offset in range(count):
            job = jobs[(start_idx + offset) % count]
            if job.finalized or job.cancel_event.is_set():
                continue
            if job.generation_done:
                continue
            if job.current_frame_idx >= job.total_frames:
                # Added guard: at depth >= 2 the last frames may still be in
                # flight; collect() marks the job done once they land.
                if job.gpu_inflight_batches <= 0:
                    job.generation_done = True
                    if job.generation_done_at is None:
                        job.generation_done_at = time.time()
                continue

            remaining_frames = self._remaining_frames(job)
            if remaining_frames <= 0:
                continue
            ordered_jobs.append(job)

        return ordered_jobs

    def _allocate_round(
        self,
        jobs: list[HLSStreamJob],
        allocations: Dict[str, int],
        total_batch: int,
        slice_cap: Optional[int],
    ) -> int:
        for job in jobs:
            capacity_left = self.max_combined_batch_size - total_batch
            if capacity_left <= 0:
                break
            remaining_frames = self._remaining_frames(job, allocations)
            if remaining_frames <= 0:
                continue
            per_turn_cap = job.batch_size if slice_cap is None else min(job.batch_size, slice_cap)
            take = min(per_turn_cap, remaining_frames, capacity_left)
            if take <= 0:
                continue
            allocations[job.request_id] = allocations.get(job.request_id, 0) + take
            total_batch += take
        return total_batch

    def _fill_remaining_capacity(
        self,
        jobs: list[HLSStreamJob],
        allocations: Dict[str, int],
        total_batch: int,
    ) -> int:
        if not jobs:
            return total_batch

        while total_batch < self.max_combined_batch_size:
            made_progress = False
            for job in jobs:
                capacity_left = self.max_combined_batch_size - total_batch
                if capacity_left <= 0:
                    break
                remaining_frames = self._remaining_frames(job, allocations)
                if remaining_frames <= 0:
                    continue
                take = min(job.batch_size, remaining_frames, capacity_left)
                if take <= 0:
                    continue
                allocations[job.request_id] = allocations.get(job.request_id, 0) + take
                total_batch += take
                made_progress = True
            if not made_progress:
                break

        return total_batch

    def _allocate_chunk_completion_round(
        self,
        jobs: list[HLSStreamJob],
        allocations: Dict[str, int],
        total_batch: int,
    ) -> int:
        if not jobs:
            return total_batch

        while total_batch < self.max_combined_batch_size:
            capacity_left = self.max_combined_batch_size - total_batch
            ranked_jobs = self._chunk_priority_jobs(jobs, allocations)
            if not ranked_jobs:
                break

            chosen_job = None
            chosen_take = 0
            fallback_job = None
            fallback_take = 0

            for job in ranked_jobs:
                remaining_frames = self._remaining_frames(job, allocations)
                if remaining_frames <= 0:
                    continue

                frames_to_next_chunk = self._frames_until_next_chunk(job, allocations)
                if frames_to_next_chunk <= 0:
                    continue

                take = min(frames_to_next_chunk, remaining_frames, capacity_left)
                if take <= 0:
                    continue

                if fallback_job is None:
                    fallback_job = job
                    fallback_take = take

                # Greedily finish the cheapest next chunk that fits in the
                # remaining batch budget. This reduces the number of turns a
                # stream needs before its next segment can be emitted.
                if frames_to_next_chunk <= capacity_left:
                    chosen_job = job
                    chosen_take = take
                    break

            if chosen_job is None:
                if fallback_job is None or fallback_take <= 0:
                    break
                chosen_job = fallback_job
                chosen_take = fallback_take

            allocations[chosen_job.request_id] = allocations.get(chosen_job.request_id, 0) + chosen_take
            total_batch += chosen_take

        return total_batch

    def _allocate_startup_priority_round(
        self,
        jobs: list[HLSStreamJob],
        allocations: Dict[str, int],
        total_batch: int,
    ) -> int:
        # MUST-KEEP STARTUP FAIRNESS HELPER:
        # This helper is the "finish the first chunk first" pass. If we roll
        # back other experiments, this is one of the pieces to keep so startup
        # does not spread back into a long convoy.
        for job in jobs:
            capacity_left = self.max_combined_batch_size - total_batch
            if capacity_left <= 0:
                break
            remaining_frames = self._remaining_frames(job, allocations)
            if remaining_frames <= 0:
                continue
            frames_to_first_chunk = self._frames_until_startup_chunk(job, allocations)
            if frames_to_first_chunk <= 0:
                continue
            take = min(frames_to_first_chunk, remaining_frames, capacity_left)
            if take <= 0:
                continue
            allocations[job.request_id] = allocations.get(job.request_id, 0) + take
            total_batch += take
        return total_batch

    def _chunk_priority_jobs(
        self,
        jobs: list[HLSStreamJob],
        allocations: Dict[str, int],
    ) -> list[HLSStreamJob]:
        ranked: list[tuple[int, int, int, float, HLSStreamJob]] = []
        for ordinal, job in enumerate(jobs):
            remaining_frames = self._remaining_frames(job, allocations)
            if remaining_frames <= 0:
                continue

            frames_to_next_chunk = self._frames_until_next_chunk(job, allocations)
            if frames_to_next_chunk <= 0:
                continue

            ranked.append(
                (
                    0 if self._is_startup_job(job) else 1,
                    frames_to_next_chunk,
                    -self._generated_unencoded_frames(job, allocations),
                    job.last_progress_at,
                    job,
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [job for *_unused, job in ranked]

    def _run_generation_batch(self, selected) -> None:
        # Added code (plan item 1.4): the batch is split into submit (assemble
        # + launch) and collect (wait + compose dispatch + stats). At depth 1
        # (the default) they run back to back, which is exactly the original
        # sequence of calls: the same stage syncs, the blocking
        # decode_latents(), the same compose dispatch order and per-job stats.
        batch = self._submit_generation_batch(selected, pipelined=False)
        if batch is not None:
            self._collect_generation_batch(batch)

    def _submit_generation_batch(self, selected, pipelined: bool = False):
        selected = [(job, take) for job, take in selected if take > 0 and not job.cancel_event.is_set()]
        if not selected:
            self._finalize_cancelled_jobs()
            return None

        telemetry = self.gpu_event_timing
        cpu_started = time.thread_time() if telemetry else 0.0
        batch_started_at = time.time()
        total_batch = sum(take for _, take in selected)
        lease_batch_size = self._memory_bucket(total_batch)

        for job, take in selected:
            self._mark_job_scheduled(job, batch_started_at)

        pieces = self._plan_batch_pieces(selected)
        jobs = []
        seen = set()
        for piece in pieces:
            if id(piece.job) not in seen:
                seen.add(id(piece.job))
                jobs.append(piece.job)
        if len(jobs) > len(selected):
            # Jobs added by the raw-frame top-up.
            selected_ids = {id(job) for job, _ in selected}
            for job in jobs:
                if id(job) not in selected_ids:
                    self._mark_job_scheduled(job, batch_started_at)

        gpu_offset = 0
        for piece in pieces:
            piece.gpu_offset = gpu_offset
            gpu_offset += piece.gpu_rows

        batch = _InflightBatch()
        batch.seq = self._pipeline_seq
        self._pipeline_seq += 1
        batch.pipelined = bool(pipelined)
        batch.pieces = pieces
        batch.jobs = jobs
        batch.actual_batch = gpu_offset
        batch.raw_frames = sum(piece.take - piece.gpu_rows for piece in pieces)
        batch.lease_batch_size = lease_batch_size
        batch.batch_started_at = batch_started_at
        batch.submitted_at_perf = time.perf_counter()

        if batch.actual_batch > 0:
            calib_selected = (
                selected
                if not self.skip_gpu_for_raw
                else [(piece.job, piece.gpu_rows) for piece in pieces if piece.gpu_rows > 0]
            )
            self._launch_gpu_batch(batch, calib_selected)
        else:
            # Every frame is raw (HLS_SKIP_GPU_FOR_RAW=1): nothing to launch.
            now = time.time()
            batch.padded_batch = 0
            batch.recon = None
            batch.assembly_finished_at = now
            batch.copy_started_at = batch.copy_finished_at = now
            batch.pe_started_at = batch.pe_finished_at = now
            batch.unet_started_at = batch.unet_finished_at = now
            batch.vae_started_at = batch.vae_finished_at = now

        # The frames are scheduled now; at depth >= 2 selection of the next
        # batch must not pick them again.
        for piece in pieces:
            piece.job.current_frame_idx = piece.start_frame_idx + piece.take
        for job in jobs:
            job.gpu_inflight_batches += 1
        batch.submit_cpu_s = (time.thread_time() - cpu_started) if telemetry else 0.0
        return batch

    def _mark_job_scheduled(self, job: HLSStreamJob, batch_started_at: float) -> None:
        if job.first_scheduled_at is None:
            job.first_scheduled_at = batch_started_at
        job.last_progress_at = time.time()
        if not getattr(job.session, "live_ready", False) and hasattr(job.session, "status"):
            job.session.status = "generating"
        self._set_request_status(job.request_id, "running")
        job.scheduler_turns += 1

    def _plan_batch_pieces(self, selected) -> list:
        pieces = [_BatchPiece(job, job.current_frame_idx, take) for job, take in selected]
        if not self.skip_gpu_for_raw:
            return pieces

        # Added code (plan item 1.10, HLS_SKIP_GPU_FOR_RAW=1).
        for piece in pieces:
            self._mark_raw_frames(piece)
        allocations: Dict[str, int] = {}
        for piece in pieces:
            allocations[piece.job.request_id] = allocations.get(piece.job.request_id, 0) + piece.take
        new_round = pieces
        for _round in range(4):
            gpu_rows = sum(piece.gpu_rows for piece in pieces)
            if gpu_rows >= self.max_combined_batch_size:
                break
            if not any(piece.raw_mask is not None for piece in new_round):
                break
            # Raw frames freed GPU rows: top them up with more frames, never
            # from exact_silence jobs (their frames are all raw).
            with self.condition:
                candidates = [job for job in self._topup_order_locked() if not job.exact_silence]
                before = dict(allocations)
                self._fill_remaining_capacity(candidates, allocations, gpu_rows)
            new_round = []
            for job in candidates:
                added = allocations.get(job.request_id, 0) - before.get(job.request_id, 0)
                if added <= 0:
                    continue
                piece = _BatchPiece(job, job.current_frame_idx + before.get(job.request_id, 0), added)
                self._mark_raw_frames(piece)
                new_round.append(piece)
            if not new_round:
                break
            pieces.extend(new_round)
        return pieces

    def _topup_order_locked(self) -> list:
        jobs = self._ordered_schedulable_jobs_locked()
        if self.scheduler_policy == "edf" and jobs:
            now = time.time()
            ordinal = {job.request_id: index for index, job in enumerate(jobs)}
            slack = {}
            for job in jobs:
                st = self._edf_job_state(job, now)
                slack[job.request_id] = st["slack_frames"] / st["fps"]
            jobs.sort(key=lambda job: (slack[job.request_id], ordinal[job.request_id]))
        return jobs

    def _mark_raw_frames(self, piece: _BatchPiece) -> None:
        """Mirror of the raw-compose decision in _dispatch_compose_batch."""
        job = piece.job
        router = (
            getattr(job.session, "live_pose_router", None)
            if job.output_mode == "webrtc"
            else None
        )
        if router is not None:
            piece.snapshots = router.snapshots_for_range(
                piece.start_frame_idx,
                piece.take,
                job.generation_fps,
            )
        mask = None
        if getattr(job, "exact_silence", False):
            mask = [True] * piece.take
        elif (
            router is not None
            and getattr(router, "motion_bank", None) is not None
            and _env_bool("WEBRTC_RAW_IDLE_POSE", False)
        ):
            mask = [
                (snapshot.pose_id if snapshot is not None else "default") == "neutral_resting"
                for snapshot in piece.snapshots
            ]
        if mask is not None and any(mask):
            piece.raw_mask = mask
            piece.gpu_rows = piece.take - sum(1 for is_raw in mask if is_raw)

    def _assemble_piece(self, piece: _BatchPiece, staging_conditioning, staging_latents) -> None:
        job = piece.job
        take = piece.take
        offset = piece.gpu_offset
        if piece.raw_mask is None:
            with job.conditioning_lock:
                conditioning_slice = job.conditioning_chunks[piece.start_frame_idx: piece.start_frame_idx + take]
            staging_conditioning[offset: offset + take].copy_(conditioning_slice)

        live_pose_router = (
            getattr(job.session, "live_pose_router", None)
            if job.output_mode == "webrtc"
            else None
        )
        live_pose_snapshots = piece.snapshots
        if live_pose_snapshots is None:
            live_pose_snapshots = (
                live_pose_router.snapshots_for_range(
                    piece.start_frame_idx,
                    take,
                    job.generation_fps,
                )
                if live_pose_router is not None
                else [None] * take
            )
        gathered_latents_by_frame = []
        for relative_frame, snapshot in enumerate(live_pose_snapshots):
            if piece.raw_mask is not None and piece.raw_mask[relative_frame]:
                continue
            pose_id = snapshot.pose_id if snapshot is not None else "default"
            render_key = (
                snapshot.effective_render_key
                if snapshot is not None
                else "default"
            )
            pose_avatar = job.pose_avatars.get(render_key, job.avatar)
            if render_key != pose_id and render_key not in job.pose_avatars:
                raise RuntimeError(
                    f"Prepared pose variant is unavailable: {render_key}"
                )
            latent_cycle = getattr(
                pose_avatar,
                "input_latent_cycle_batch_tensor",
                getattr(pose_avatar, "input_latent_cycle_tensor", None),
            )
            if not isinstance(latent_cycle, torch.Tensor):
                raise RuntimeError(
                    "Expected latent cycle tensor for scheduler batch assembly"
                )
            generation_index = piece.start_frame_idx + relative_frame
            cycle_index = (
                live_pose_router.source_frame_index(
                    snapshot,
                    generation_index,
                )
                if snapshot is not None and snapshot.is_queued
                else job.start_offset_frames + generation_index
            )
            gathered_latent = latent_cycle[
                cycle_index % latent_cycle.shape[0]
            ]
            if gathered_latent.dim() == 4 and gathered_latent.shape[0] == 1:
                gathered_latent = gathered_latent.squeeze(0)
            gathered_latents_by_frame.append((relative_frame, gathered_latent))
        # A pose transition can put CPU-pinned and GPU-resident avatar
        # latents in the same batch. Copy each frame into the common CPU
        # staging buffer before the model transfer.
        row = offset
        for relative_frame, gathered_latent in gathered_latents_by_frame:
            if piece.raw_mask is not None:
                with job.conditioning_lock:
                    conditioning_row = job.conditioning_chunks[piece.start_frame_idx + relative_frame]
                staging_conditioning[row].copy_(conditioning_row)
            staging_latents[row].copy_(gathered_latent)
            row += 1

    def _launch_gpu_batch(self, batch: _InflightBatch, calib_selected) -> None:
        pieces = batch.pieces
        pipelined = batch.pipelined
        actual_batch = batch.actual_batch

        # Pad to compile-friendly size to avoid torch.compile recompilation
        padded_batch = actual_batch
        for size in self.fixed_batch_sizes:
            if size >= actual_batch:
                padded_batch = size
                break
        else:
            padded_batch = actual_batch  # larger than 32, don't pad
        batch.padded_batch = padded_batch

        first_job = next(piece.job for piece in pieces if piece.gpu_rows > 0)
        conditioning_shape = tuple(first_job.conditioning_chunks.shape[1:])
        latent_cycle = getattr(
            first_job.avatar,
            "input_latent_cycle_batch_tensor",
            getattr(first_job.avatar, "input_latent_cycle_tensor", None),
        )
        if not isinstance(latent_cycle, torch.Tensor):
            raise RuntimeError("Expected latent cycle tensor for scheduler batch assembly")
        latent_shape = tuple(latent_cycle.shape[1:])
        staging_slot = (batch.seq % self.pipeline_depth) if pipelined else None
        staging_conditioning, staging_latents = self._get_staging_buffers(
            conditioning_shape=conditioning_shape,
            conditioning_dtype=first_job.conditioning_chunks.dtype,
            latent_shape=latent_shape,
            latent_dtype=latent_cycle.dtype,
            batch_size=padded_batch,
            slot=staging_slot,
        )
        if staging_slot is not None:
            # The batch that last used this slot was collected already; this
            # only guards its H2D copy explicitly.
            previous_h2d = self._staging_slot_events.get(staging_slot)
            if previous_h2d is not None:
                previous_h2d.synchronize()

        for piece in pieces:
            if piece.gpu_rows > 0:
                self._assemble_piece(piece, staging_conditioning, staging_latents)

        if padded_batch > actual_batch:
            pad_n = padded_batch - actual_batch
            # When actual_batch is less than half of the padded bucket, the
            # source and destination ranges can overlap inside the staging
            # buffer. Clone the repeated prefix before copying it into the pad.
            staging_conditioning[actual_batch:padded_batch].copy_(
                staging_conditioning[:pad_n].clone()
            )
            staging_latents[actual_batch:padded_batch].copy_(
                staging_latents[:pad_n].clone()
            )

        batch.assembly_finished_at = time.time()
        conditioning_batch = staging_conditioning[:padded_batch]
        latent_batch = staging_latents[:padded_batch]
        events = self._new_batch_events() if self.gpu_event_timing else None
        batch.events = events

        with self.manager.gpu_memory.allocate(batch.lease_batch_size):
            runtime_context = torch.no_grad if getattr(self.manager, "models_compiled", False) else torch.inference_mode
            with runtime_context():
                batch.copy_started_at = time.time()
                self._record_event(events, "start")
                audio_feature_batch = conditioning_batch.to(self.manager.device, non_blocking=True)
                # Use the prepared conditioning dtype instead of asking the
                # compiled model wrapper for dtype metadata.
                target_dtype = audio_feature_batch.dtype
                latent_batch = latent_batch.to(
                    device=self.manager.device,
                    dtype=target_dtype,
                    non_blocking=True,
                )
                if staging_slot is not None:
                    staging_event = self._make_event(timing=False)
                    staging_event.record()
                    self._staging_slot_events[staging_slot] = staging_event
                if not pipelined:
                    self._sync_gpu_for_stage_timing()
                self._record_event(events, "h2d")
                batch.copy_finished_at = time.time()

                batch.pe_started_at = batch.copy_finished_at
                batch.pe_finished_at = batch.copy_finished_at

                batch.unet_started_at = time.time()
                pred_latents = self.manager.unet.model(
                    latent_batch,
                    self.manager.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                if not pipelined:
                    self._sync_gpu_for_stage_timing()
                self._record_event(events, "unet")
                batch.unet_finished_at = time.time()
                self._capture_unet_calibration_batch(
                    latent_batch=latent_batch,
                    audio_feature_batch=audio_feature_batch,
                    timesteps=getattr(self.manager, "timesteps", None),
                    pred_latents=pred_latents,
                    selected=calib_selected,
                    actual_batch=actual_batch,
                    padded_batch=padded_batch,
                )

                batch.vae_started_at = time.time()
                pred_latents = pred_latents.to(
                    device=self.manager.device,
                    dtype=getattr(self.manager, "vae_dtype", pred_latents.dtype),
                )
                self._capture_vae_calibration_batch(
                    pred_latents=pred_latents,
                    selected=calib_selected,
                    actual_batch=actual_batch,
                    padded_batch=padded_batch,
                )
                # Depth >= 2: decode + postprocess stay on the GPU and the
                # uint8 faces are copied non_blocking into a pinned ring slot,
                # all enqueued on this stream before the next batch's UNet, so
                # static engine/graph outputs are consumed before any reuse.
                device_u8 = self._vae_decode_device_u8(pred_latents) if pipelined else None
                if device_u8 is None:
                    if pipelined and not self._fallback_decode_logged:
                        self._fallback_decode_logged = True
                        print(
                            "🎛️  depth>1: VAE decode has no device uint8 path "
                            "(MUSETALK_VAE_FAST_POSTPROCESS=0?); using the blocking decode_latents()"
                        )
                    recon = self.manager.vae.decode_latents(pred_latents)
                    if not pipelined:
                        self._sync_gpu_for_stage_timing()
                    self._record_event(events, "vae")
                    self._record_event(events, "d2h")
                else:
                    self._record_event(events, "vae")
                    slot = self._acquire_output_slot(tuple(device_u8.shape), device_u8.dtype)
                    slot.tensor.copy_(device_u8, non_blocking=True)
                    self._record_event(events, "d2h")
                    batch.slot = slot
                    recon = slot.array
                    del device_u8
                if pipelined:
                    done_event = self._make_event(timing=False, blocking=self.gpu_blocking_wait)
                    done_event.record()
                    batch.done_event = done_event
                batch.vae_finished_at = time.time()

        # After VAE decode, trim padding
        if padded_batch > actual_batch:
            recon = recon[:actual_batch]
        batch.recon = recon

    def _collect_generation_batch(self, batch: _InflightBatch) -> None:
        telemetry = self.gpu_event_timing
        cpu_started = time.thread_time() if telemetry else 0.0
        host_wait_s = 0.0
        stage_ms = None
        try:
            if batch.done_event is not None:
                wait_started = time.perf_counter()
                batch.done_event.synchronize()
                host_wait_s = time.perf_counter() - wait_started
            if telemetry:
                stage_ms = self._event_stage_ms(batch)

            batch_finished_at = time.time()
            assembly_s = batch.assembly_finished_at - batch.batch_started_at
            copy_s = batch.copy_finished_at - batch.copy_started_at
            pe_s = batch.pe_finished_at - batch.pe_started_at
            unet_s = batch.unet_finished_at - batch.unet_started_at
            vae_s = batch.vae_finished_at - batch.vae_started_at
            host_stage_s = (copy_s, unet_s, vae_s)
            if stage_ms is not None and (batch.pipelined or not self.gpu_stage_sync_timing):
                # Without host syncs the host clocks only time the enqueue.
                copy_s = stage_ms["h2d_ms"] / 1000.0
                unet_s = stage_ms["unet_ms"] / 1000.0
                vae_s = (stage_ms["vae_ms"] + stage_ms["d2h_ms"]) / 1000.0
            gpu_batch_s = batch_finished_at - batch.batch_started_at
            actual_batch = batch.actual_batch
            padded_batch = batch.padded_batch
            lease_batch_size = batch.lease_batch_size

            self._gpu_batch_timing_counter += 1
            log_by_interval = (
                self.gpu_batch_timing_log_interval > 0
                and self._gpu_batch_timing_counter % self.gpu_batch_timing_log_interval == 0
            )
            log_by_slow_batch = (
                self.gpu_batch_timing_slow_s > 0.0
                and gpu_batch_s >= self.gpu_batch_timing_slow_s
            )
            if log_by_interval or log_by_slow_batch:
                reason = "slow" if log_by_slow_batch else "interval"
                extra = ""
                if self._pipeline_flags_non_default():
                    extra = (
                        f" depth={self.pipeline_depth} raw={batch.raw_frames}"
                        f" pieces={len(batch.pieces)} host_wait={host_wait_s:.4f}s"
                    )
                    if stage_ms is not None:
                        extra += (
                            f" ev_h2d={stage_ms['h2d_ms']:.2f}ms ev_unet={stage_ms['unet_ms']:.2f}ms"
                            f" ev_vae={stage_ms['vae_ms']:.2f}ms ev_d2h={stage_ms['d2h_ms']:.2f}ms"
                        )
                print(
                    f"🎛️  GPU batch timing #{self._gpu_batch_timing_counter} reason={reason} "
                    f"jobs={len(batch.jobs)} actual={actual_batch} padded={padded_batch} "
                    f"lease={lease_batch_size} assemble={assembly_s:.4f}s copy={copy_s:.4f}s "
                    f"pe={pe_s:.4f}s unet={unet_s:.4f}s vae={vae_s:.4f}s total={gpu_batch_s:.4f}s"
                    f"{extra}"
                )

            recon = batch.recon
            for piece in batch.pieces:
                job = piece.job
                if piece.raw_mask is None:
                    batch_frames = recon[piece.gpu_offset: piece.gpu_offset + piece.take]
                else:
                    batch_frames = []
                    row = piece.gpu_offset
                    for is_raw in piece.raw_mask:
                        if is_raw:
                            batch_frames.append(None)
                        else:
                            batch_frames.append(recon[row])
                            row += 1
                if not job.cancel_event.is_set() and not job.finalized:
                    future = self._dispatch_compose_batch(
                        job,
                        batch_frames,
                        piece.start_frame_idx,
                        live_pose_snapshots=(
                            piece.snapshots if piece.raw_mask is not None else None
                        ),
                    )
                    if batch.slot is not None and future is not None:
                        batch.slot.consumers.append(future)
        finally:
            for job in batch.jobs:
                job.gpu_inflight_batches = max(0, job.gpu_inflight_batches - 1)

        for job in batch.jobs:
            if job.current_frame_idx >= job.total_frames and job.gpu_inflight_batches <= 0:
                job.generation_done = True
                if job.generation_done_at is None:
                    job.generation_done_at = time.time()
            job.batch_assembly_total_s += assembly_s
            job.gpu_copy_total_s += copy_s
            job.pe_total_s += pe_s
            job.unet_total_s += unet_s
            job.vae_total_s += vae_s
            job.gpu_batch_total_s += gpu_batch_s
            job.gpu_batch_count += 1

        if telemetry:
            collect_cpu_s = time.thread_time() - cpu_started
            self._record_capacity(batch, stage_ms, host_wait_s, collect_cpu_s, host_stage_s, gpu_batch_s)

        self._finalize_cancelled_jobs()
        self._finalize_ready_jobs()

    # ------------------------------------------------ depth >= 2 / telemetry helpers
    def _device_is_cuda(self) -> bool:
        if not torch.cuda.is_available():
            return False
        device = getattr(self.manager, "device", None)
        try:
            return torch.device(device).type == "cuda"
        except (TypeError, RuntimeError):
            return False

    def _make_event(self, timing: bool = False, blocking: bool = False):
        if self._device_is_cuda():
            return torch.cuda.Event(enable_timing=bool(timing), blocking=bool(blocking))
        return _HostEvent()

    def _new_batch_events(self) -> dict:
        return {name: self._make_event(timing=True) for name in ("start", "h2d", "unet", "vae", "d2h")}

    @staticmethod
    def _record_event(events, name: str) -> None:
        if events is not None:
            events[name].record()

    def _event_stage_ms(self, batch: _InflightBatch) -> Optional[dict]:
        events = batch.events
        if not events:
            return None
        events["d2h"].synchronize()
        return {
            "h2d_ms": events["start"].elapsed_time(events["h2d"]),
            "unet_ms": events["h2d"].elapsed_time(events["unet"]),
            "vae_ms": events["unet"].elapsed_time(events["vae"]),
            "d2h_ms": events["vae"].elapsed_time(events["d2h"]),
            "span_ms": events["start"].elapsed_time(events["d2h"]),
        }

    def _vae_decode_device_u8(self, pred_latents: torch.Tensor):
        """GPU half of VAE.decode_latents(): uint8 NHWC BGR on the device.

        Mirrors musetalk/models/vae.py decode_latents() op for op (the fused
        decode_bgr_u8 backend path, else decode_latents_tensor + the fast
        postprocess), so the bytes equal its .cpu().numpy() result. Returns
        None when that path does not apply (the caller then uses the blocking
        decode_latents(), unchanged).
        """
        vae = self.manager.vae
        hook = getattr(vae, "decode_latents_device_u8", None)
        if callable(hook):
            return hook(pred_latents)
        module = sys.modules.get(type(vae).__module__)
        if getattr(module, "MUSETALK_VAE_FAST_POSTPROCESS", None) is not True:
            return None
        if not callable(getattr(vae, "decode_latents_tensor", None)):
            return None
        backend = getattr(vae, "_decode_backend", None)
        fused_bgr_u8 = None
        if backend is not None and getattr(backend, "fused_post_enabled", False):
            fused_bgr_u8 = getattr(backend, "decode_bgr_u8", None)
        if fused_bgr_u8 is not None:
            return fused_bgr_u8(pred_latents)
        image = vae.decode_latents_tensor(pred_latents)
        return (
            image.detach()
            .float()
            .mul(255)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
            .flip(1)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def _acquire_output_slot(self, shape: tuple, dtype) -> _OutputSlot:
        key = (tuple(shape), str(dtype))
        ring = self._output_rings.setdefault(key, [])
        position = self._output_ring_pos.get(key, 0)
        index = position % self.output_ring_size
        self._output_ring_pos[key] = position + 1
        if index >= len(ring):
            with torch.inference_mode(False):
                tensor = torch.empty(
                    shape,
                    dtype=dtype,
                    pin_memory=self._device_is_cuda(),
                )
            ring.append(_OutputSlot(tensor))
        slot = ring[index]
        # The faces of this slot's previous batch are views handed to compose
        # workers; never overwrite them before those tasks finish.
        pending = [future for future in slot.consumers if not future.done()]
        if pending:
            wait_started = time.perf_counter()
            _wait_futures(pending)
            if self.gpu_event_timing:
                self._cap_add("ring_wait_ms", (time.perf_counter() - wait_started) * 1000.0)
                self._cap_add("ring_waits", 1.0)
        slot.consumers = []
        slot.uses += 1
        return slot

    def _cap_add(self, key: str, value: float) -> None:
        with self._cap_lock:
            self._cap_totals[key] += value

    def _record_capacity(self, batch, stage_ms, host_wait_s, collect_cpu_s, host_stage_s, gpu_batch_s) -> None:
        record = {
            "seq": batch.seq,
            "jobs": len(batch.jobs),
            "pieces": len(batch.pieces),
            "actual": batch.actual_batch,
            "padded": batch.padded_batch,
            "raw": batch.raw_frames,
            "host_wait_ms": round(host_wait_s * 1000.0, 3),
            "feeder_cpu_ms": round((batch.submit_cpu_s + collect_cpu_s) * 1000.0, 3),
            "wall_ms": round(gpu_batch_s * 1000.0, 3),
        }
        gap_ms = None
        if stage_ms is not None:
            if self._cap_prev_end_event is not None:
                gap_ms = max(0.0, self._cap_prev_end_event.elapsed_time(batch.events["start"]))
            self._cap_prev_end_event = batch.events["d2h"]
            record.update({key: round(value, 4) for key, value in stage_ms.items()})
            if gap_ms is not None:
                record["idle_gap_ms"] = round(gap_ms, 4)
        with self._cap_lock:
            totals = self._cap_totals
            totals["batches"] += 1
            totals["jobs"] += len(batch.jobs)
            totals["pieces"] += len(batch.pieces)
            totals["raw_frames"] += batch.raw_frames
            totals["feeder_cpu_ms"] += (batch.submit_cpu_s + collect_cpu_s) * 1000.0
            if batch.actual_batch > 0:
                totals["gpu_batches"] += 1
                totals["actual_frames"] += batch.actual_batch
                totals["padded_frames"] += batch.padded_batch
                totals["host_wait_ms"] += host_wait_s * 1000.0
                totals["host_copy_ms"] += host_stage_s[0] * 1000.0
                totals["host_unet_ms"] += host_stage_s[1] * 1000.0
                totals["host_vae_ms"] += host_stage_s[2] * 1000.0
                if stage_ms is not None:
                    totals["gpu_span_ms"] += stage_ms["span_ms"]
                    totals["h2d_ms"] += stage_ms["h2d_ms"]
                    totals["unet_ms"] += stage_ms["unet_ms"]
                    totals["vae_ms"] += stage_ms["vae_ms"]
                    totals["d2h_ms"] += stage_ms["d2h_ms"]
                if gap_ms is not None:
                    totals["idle_gap_ms"] += gap_ms
                    totals["idle_gap_count"] += 1
            self._cap_batches.append(record)

    def _capture_unet_calibration_batch(
        self,
        latent_batch: torch.Tensor,
        audio_feature_batch: torch.Tensor,
        timesteps,
        pred_latents: torch.Tensor,
        selected,
        actual_batch: int,
        padded_batch: int,
    ) -> None:
        if not self.unet_calibration_capture:
            return
        if (
            self.unet_calibration_max_batches > 0
            and self._unet_calibration_capture_count >= self.unet_calibration_max_batches
        ):
            if not self._unet_calibration_limit_logged:
                print(
                    "🧪 UNet calibration capture limit reached "
                    f"({self.unet_calibration_max_batches} batches)"
                )
                self._unet_calibration_limit_logged = True
            return

        self._unet_calibration_capture_count += 1
        sequence = self._unet_calibration_capture_count
        items = []
        offset = 0
        for job, take in selected:
            items.append(
                {
                    "request_id": job.request_id,
                    "session_id": job.session_id,
                    "avatar_id": getattr(job.session, "avatar_id", None),
                    "start_frame_idx": int(job.current_frame_idx),
                    "take": int(take),
                    "batch_offset": int(offset),
                }
            )
            offset += int(take)

        if isinstance(timesteps, torch.Tensor):
            saved_timesteps = timesteps.detach().to(device="cpu").contiguous()
        else:
            saved_timesteps = torch.tensor([0], dtype=torch.long)

        payload = {
            "schema_version": 1,
            "kind": "unet_io_batch",
            "created_at": time.time(),
            "sequence": int(sequence),
            "actual_batch": int(actual_batch),
            "padded_batch": int(padded_batch),
            "latent_dtype": str(latent_batch.dtype),
            "latent_shape": list(latent_batch.shape),
            "audio_feature_dtype": str(audio_feature_batch.dtype),
            "audio_feature_shape": list(audio_feature_batch.shape),
            "timesteps_dtype": str(saved_timesteps.dtype),
            "timesteps_shape": list(saved_timesteps.shape),
            "pred_latents_dtype": str(pred_latents.dtype),
            "pred_latents_shape": list(pred_latents.shape),
            "items": items,
            "latent_batch": latent_batch.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous(),
            "audio_feature_batch": audio_feature_batch.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous(),
            "timesteps": saved_timesteps,
            "pred_latents": pred_latents.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous(),
        }

        try:
            self.unet_calibration_dir.mkdir(parents=True, exist_ok=True)
            path = (
                self.unet_calibration_dir
                / f"unet_io_{sequence:06d}_bs{padded_batch}_pid{os.getpid()}.pt"
            )
            tmp_path = path.with_suffix(".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
            if sequence == 1 or sequence % 25 == 0:
                print(
                    "🧪 Saved UNet calibration batch "
                    f"#{sequence} actual={actual_batch} padded={padded_batch} path={path}"
                )
        except Exception as exc:
            print(f"⚠️  Failed to save UNet calibration batch #{sequence}: {type(exc).__name__}: {exc}")

    def _capture_vae_calibration_batch(
        self,
        pred_latents: torch.Tensor,
        selected,
        actual_batch: int,
        padded_batch: int,
    ) -> None:
        if not self.vae_calibration_capture:
            return
        if (
            self.vae_calibration_max_batches > 0
            and self._vae_calibration_capture_count >= self.vae_calibration_max_batches
        ):
            if not self._vae_calibration_limit_logged:
                print(
                    "🧪 VAE calibration capture limit reached "
                    f"({self.vae_calibration_max_batches} batches)"
                )
                self._vae_calibration_limit_logged = True
            return

        self._vae_calibration_capture_count += 1
        sequence = self._vae_calibration_capture_count
        items = []
        offset = 0
        for job, take in selected:
            items.append(
                {
                    "request_id": job.request_id,
                    "session_id": job.session_id,
                    "avatar_id": getattr(job.session, "avatar_id", None),
                    "start_frame_idx": int(job.current_frame_idx),
                    "take": int(take),
                    "batch_offset": int(offset),
                }
            )
            offset += int(take)

        payload = {
            "schema_version": 1,
            "kind": "vae_decoder_pred_latents",
            "created_at": time.time(),
            "sequence": int(sequence),
            "actual_batch": int(actual_batch),
            "padded_batch": int(padded_batch),
            "dtype": str(pred_latents.dtype),
            "shape": list(pred_latents.shape),
            "items": items,
            "pred_latents": pred_latents.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous(),
        }

        try:
            self.vae_calibration_dir.mkdir(parents=True, exist_ok=True)
            path = (
                self.vae_calibration_dir
                / f"vae_pred_latents_{sequence:06d}_bs{padded_batch}_pid{os.getpid()}.pt"
            )
            tmp_path = path.with_suffix(".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
            if sequence == 1 or sequence % 25 == 0:
                print(
                    "🧪 Saved VAE calibration batch "
                    f"#{sequence} actual={actual_batch} padded={padded_batch} path={path}"
                )
        except Exception as exc:
            print(f"⚠️  Failed to save VAE calibration batch #{sequence}: {type(exc).__name__}: {exc}")

    def _dispatch_compose_batch(
        self,
        job: HLSStreamJob,
        batch_frames,
        start_frame_idx: int,
        live_pose_snapshots=None,
    ):
        # Added: `live_pose_snapshots` is passed only for batches with skipped
        # raw frames (HLS_SKIP_GPU_FOR_RAW=1) so compose uses the snapshots
        # the skip decision was made with; otherwise they are computed here
        # as before. Returns the compose future (the depth >= 2 output ring
        # tracks it as a reader of the faces).
        given_snapshots = live_pose_snapshots
        compose_sequence = job.compose_sequence
        job.compose_sequence += 1
        compose_submitted_at = time.time()
        live_pose_router = None
        live_pose_snapshots = None
        if job.output_mode == "webrtc":
            live_pose_router = getattr(job.session, "live_pose_router", None)
            if live_pose_router is not None:
                live_pose_snapshots = (
                    given_snapshots
                    if given_snapshots is not None
                    else live_pose_router.snapshots_for_range(
                        start_frame_idx,
                        len(batch_frames),
                        job.generation_fps,
                    )
                )

        bank = getattr(live_pose_router, "motion_bank", None)
        carry_layers = getattr(bank, "current_phoneme", None) is not None
        yuv_in_compose = bool(self.webrtc_yuv_in_compose and job.output_mode == "webrtc")

        def compose_batch():
            compose_started_at = time.time()
            frames = []
            raw_layers = []
            source_frame_indices = []
            background_frames = [None] * len(batch_frames)
            if live_pose_router is not None and live_pose_snapshots is not None:
                group_start = 0
                while group_start < len(live_pose_snapshots):
                    snapshot = live_pose_snapshots[group_start]
                    group_end = group_start + 1
                    while (
                        group_end < len(live_pose_snapshots)
                        and live_pose_snapshots[group_end].pose_id == snapshot.pose_id
                        and live_pose_snapshots[group_end].effective_render_key
                        == snapshot.effective_render_key
                        and live_pose_snapshots[group_end].origin_generation_frame
                        == snapshot.origin_generation_frame
                    ):
                        group_end += 1
                    background_frames[group_start:group_end] = (
                        live_pose_router.read_background_frames(
                            snapshot,
                            start_frame_idx + group_start,
                            group_end - group_start,
                        )
                    )
                    group_start = group_end
            for rel_index, res_frame in enumerate(batch_frames):
                snapshot = (
                    live_pose_snapshots[rel_index]
                    if live_pose_snapshots is not None
                    else None
                )
                pose_id = snapshot.pose_id if snapshot is not None else "default"
                render_key = (
                    snapshot.effective_render_key
                    if snapshot is not None
                    else "default"
                )
                pose_avatar = job.pose_avatars.get(render_key, job.avatar)
                if render_key != pose_id and render_key not in job.pose_avatars:
                    raise RuntimeError(
                        f"Prepared pose variant is unavailable: {render_key}"
                    )
                generation_index = start_frame_idx + rel_index
                cycle_index = (
                    live_pose_router.source_frame_index(snapshot, generation_index)
                    if snapshot is not None and snapshot.is_queued
                    else job.start_offset_frames + generation_index
                )
                source_frame_indices.append(cycle_index)
                background_frame = (
                    background_frames[rel_index]
                    if rel_index < len(background_frames)
                    else None
                )
                raw_idle_pose = bool(
                    job.output_mode == "webrtc"
                    and bank is not None
                    and pose_id == "neutral_resting"
                    and _env_bool("WEBRTC_RAW_IDLE_POSE", False)
                )
                if getattr(job, "exact_silence", False) or raw_idle_pose:
                    # Do not gate individual quiet/voiceless speech frames.
                    # This flag comes only from the original entire upload's
                    # exact-zero decoded PCM. The separate opt-in raw idle mode
                    # applies only after the motion plan selects the idle pose.
                    if res_frame is None:
                        # Added code (HLS_SKIP_GPU_FOR_RAW=1): no face was
                        # generated; build the identical raw layer directly.
                        result = {"raw": self._compose_raw_layer(
                            pose_avatar, cycle_index, background_frame)}
                    else:
                        result = pose_avatar.compose_frame(res_frame, cycle_index,
                            background_frame=background_frame, return_layers=True)
                    frames.append(result["raw"])
                    if carry_layers:
                        empty_alpha = np.empty((0, 0), dtype=np.uint8)
                        empty_alpha.setflags(write=False)
                        raw_layers.append({"raw": result["raw"], "alpha": {
                            "bounds": (0, 0, 0, 0), "values": empty_alpha}})
                elif res_frame is None:
                    raise RuntimeError(
                        "GPU face was skipped for a frame that is not composed raw"
                    )
                elif carry_layers:
                    result = pose_avatar.compose_frame(res_frame, cycle_index,
                        background_frame=background_frame, return_layers=True)
                    frames.append(result["composed"])
                    raw_layers.append({"raw": result["raw"], "alpha": result["alpha"]})
                else:
                    frames.append(
                        pose_avatar.compose_frame(
                            res_frame,
                            cycle_index,
                            background_frame=background_frame,
                        )
                    )
            if yuv_in_compose:
                frames = self._attach_yuv420p(frames)
            return {
                "compose_sequence": compose_sequence,
                "frames": frames,
                "live_pose_ids": [
                    snapshot.pose_id
                    for snapshot in (live_pose_snapshots or [])
                ],
                "live_pose_render_keys": [
                    snapshot.effective_render_key
                    for snapshot in (live_pose_snapshots or [])
                ],
                "live_source_frame_indices": source_frame_indices,
                "live_raw_layers": raw_layers,
                "live_pose_crossfade_frames": [
                    snapshot.crossfade_frames
                    for snapshot in (live_pose_snapshots or [])
                ],
                "live_pose_id": (
                    live_pose_snapshots[0].pose_id
                    if live_pose_snapshots
                    else None
                ),
                "queue_wait_s": compose_started_at - compose_submitted_at,
                "compose_time": time.time() - compose_started_at,
            }

        future = self.compose_executor.submit(compose_batch)
        job.compose_tasks[compose_sequence] = future
        job.max_pending_composes = max(job.max_pending_composes, len(job.compose_tasks))
        return future

    @staticmethod
    def _attach_yuv420p(frames: list) -> list:
        """Added code (WEBRTC_YUV_IN_COMPOSE=1 producer side): carry the exact
        conversion push_bgr_frames_batch would make (webrtc_live_handoff
        bgr_to_yuv420p_frame) so the event loop does not convert."""
        from scripts.webrtc_live_handoff import ComposedFrame

        return [ComposedFrame.from_bgr(frame) for frame in frames]

    @staticmethod
    def _compose_raw_layer(pose_avatar, cycle_index: int, background_frame=None):
        """Added code (HLS_SKIP_GPU_FOR_RAW=1): exactly
        ``pose_avatar.compose_frame(face, cycle_index, background_frame=...,
        return_layers=True)["raw"]``, which never reads ``face``.

        Kept in lock-step with APIAvatar.compose_frame (scripts/api_avatar.py);
        scripts/test_hls_scheduler_pipeline.py compares the two bit for bit.
        An avatar may provide ``compose_raw_frame`` to own this itself.
        """
        own = getattr(pose_avatar, "compose_raw_frame", None)
        if callable(own):
            return own(cycle_index, background_frame=background_frame)
        cycle_pos = cycle_index % len(pose_avatar.coord_list_cycle)
        prepared_frame = pose_avatar.frame_list_cycle[cycle_pos]
        if background_frame is not None:
            ori_frame = np.asarray(background_frame)
            if ori_frame.ndim == 3 and ori_frame.shape[2] >= 3:
                if ori_frame.shape[:2] != prepared_frame.shape[:2]:
                    import cv2

                    ori_frame = cv2.resize(
                        ori_frame,
                        (prepared_frame.shape[1], prepared_frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if ori_frame.shape[2] > 3:
                    ori_frame = ori_frame[:, :, :3]
                if ori_frame.dtype != np.uint8:
                    ori_frame = np.clip(ori_frame, 0, 255).astype(np.uint8)
                raw_frame = ori_frame.copy()
                raw_frame.setflags(write=False)
                return raw_frame
        raw_frame = prepared_frame.view()
        raw_frame.setflags(write=False)
        return raw_frame

    def _drain_completed_composes(self) -> None:
        for job in list(self.jobs.values()):
            for compose_sequence, future in list(job.compose_tasks.items()):
                if not future.done():
                    continue
                del job.compose_tasks[compose_sequence]
                try:
                    compose_info = future.result()
                    job.compose_batch_count += 1
                    job.compose_queue_wait_total_s += compose_info.get("queue_wait_s", 0.0)
                    job.compose_total_s += compose_info.get("compose_time", 0.0)
                    job.max_compose_queue_wait_s = max(
                        job.max_compose_queue_wait_s,
                        compose_info.get("queue_wait_s", 0.0),
                    )
                    job.max_compose_s = max(
                        job.max_compose_s,
                        compose_info.get("compose_time", 0.0),
                    )
                    job.composed_batches[compose_sequence] = compose_info
                except Exception as exc:
                    job.error_message = str(exc)
                    print(f"❌ [{job.request_id}] compose batch failed: {exc}")
                    traceback.print_exc()

            try:
                self._append_ready_composed_frames(job)
            except Exception as exc:
                # Invalid configured layers must fail this job, not the shared
                # scheduler thread or silently select the legacy compositor.
                job.error_message = f"Composed frame append failed: {exc}"
                print(f"❌ [{job.request_id}] {job.error_message}")
                traceback.print_exc()

        self._finalize_ready_jobs()
        self._finalize_cancelled_jobs()

    def _append_ready_composed_frames(self, job: HLSStreamJob) -> None:
        if job.output_mode == "webrtc":
            self._append_ready_webrtc_frames(job)
            return

        while job.next_compose_sequence in job.composed_batches:
            compose_info = job.composed_batches.pop(job.next_compose_sequence)
            job.next_compose_sequence += 1
            job.last_progress_at = time.time()

            if job.cancel_event.is_set():
                continue

            for frame in compose_info["frames"]:
                job.frame_buffer.append(frame)
                job.composed_frame_idx += 1
                job.last_progress_at = time.time()
                job.max_frame_buffer_len = max(job.max_frame_buffer_len, len(job.frame_buffer))

                while len(job.frame_buffer) >= self._next_chunk_target_frames(job):
                    self._dispatch_encode(job)

        if (
            not job.cancel_event.is_set()
            and job.generation_done
            and not job.compose_tasks
            and job.next_compose_sequence >= job.compose_sequence
            and job.frame_buffer
        ):
            self._dispatch_encode(job, force_flush=True)

    def _append_ready_webrtc_frames(self, job: HLSStreamJob) -> None:
        while job.next_compose_sequence in job.composed_batches:
            compose_info = job.composed_batches.pop(job.next_compose_sequence)
            job.next_compose_sequence += 1
            job.last_progress_at = time.time()

            if job.cancel_event.is_set():
                continue

            frames = self._apply_webrtc_pose_crossfade(
                job,
                compose_info["frames"],
                compose_info.get("live_pose_ids") or [],
                compose_info.get("live_pose_crossfade_frames") or [],
                compose_info.get("live_source_frame_indices") or [],
                compose_info.get("live_raw_layers"),
            )
            if job.cancel_event.is_set():
                # Cancellation can arrive while CPU motion warping is running.
                # Do not hand that completed batch to any publisher.
                continue
            compose_info["frames"] = frames
            if frames and job.frame_batch_callback is not None:
                rendered_start_frame_idx = job.composed_frame_idx
                start_frame_idx = rendered_start_frame_idx + 1
                callback_started_at = time.time()
                try:
                    job.frame_batch_callback(frames, start_frame_idx, job.total_frames)
                except Exception as exc:
                    job.error_message = f"WebRTC frame batch callback failed: {exc}"
                    print(f"❌ [{job.request_id}] {job.error_message}")
                    traceback.print_exc()
                    return
                finally:
                    callback_s = time.time() - callback_started_at
                    job.frame_callback_count += len(frames)
                    job.frame_callback_total_s += callback_s
                    job.frame_callback_max_s = max(job.frame_callback_max_s, callback_s)
                    if self.gpu_event_timing:
                        self._cap_add("callback_ms", callback_s * 1000.0)

                if job.cancel_event.is_set():
                    continue
                if hasattr(job.session, "record_rendered_pose_batch"):
                    job.session.record_rendered_pose_batch(
                        compose_info.get("live_pose_ids") or [],
                        rendered_start_frame_idx,
                        compose_info.get("live_pose_render_keys") or [],
                    )
                job.composed_frame_idx += len(frames)
                job.last_progress_at = time.time()
                startup_target = self._next_chunk_target_frames(job)
                if job.first_chunk_appended_at is None and job.composed_frame_idx >= startup_target:
                    job.first_chunk_appended_at = job.last_progress_at
                    print(
                        f"🎛️  [{job.request_id}] first WebRTC startup block ready "
                        f"(frames={startup_target}, prep={job.prep_total_s:.2f}s, "
                        f"queue={self._queue_wait_s(job):.2f}s, "
                        f"first_block={self._time_to_first_chunk_s(job):.2f}s)"
                    )
                job.chunks_appended += 1
                continue

            rendered_start_frame_idx = job.composed_frame_idx
            rendered_frame_count = 0
            for frame in compose_info["frames"]:
                if job.cancel_event.is_set():
                    break
                job.composed_frame_idx += 1
                rendered_frame_count += 1
                job.last_progress_at = time.time()

                if job.frame_callback is None:
                    pass
                else:
                    callback_started_at = time.time()
                    try:
                        job.frame_callback(frame, job.composed_frame_idx, job.total_frames)
                    except Exception as exc:
                        job.error_message = f"WebRTC frame callback failed: {exc}"
                        print(f"❌ [{job.request_id}] {job.error_message}")
                        traceback.print_exc()
                        return
                    finally:
                        callback_s = time.time() - callback_started_at
                        job.frame_callback_count += 1
                        job.frame_callback_total_s += callback_s
                        job.frame_callback_max_s = max(job.frame_callback_max_s, callback_s)
                        if self.gpu_event_timing:
                            self._cap_add("callback_ms", callback_s * 1000.0)

            if (
                rendered_frame_count > 0
                and not job.cancel_event.is_set()
                and hasattr(job.session, "record_rendered_pose_batch")
            ):
                job.session.record_rendered_pose_batch(
                    (compose_info.get("live_pose_ids") or [])[
                        :rendered_frame_count
                    ],
                    rendered_start_frame_idx,
                    (compose_info.get("live_pose_render_keys") or [])[
                        :rendered_frame_count
                    ],
                )

                startup_target = self._next_chunk_target_frames(job)
                if job.first_chunk_appended_at is None and job.composed_frame_idx >= startup_target:
                    job.first_chunk_appended_at = job.last_progress_at
                    print(
                        f"🎛️  [{job.request_id}] first WebRTC startup block ready "
                        f"(frames={startup_target}, prep={job.prep_total_s:.2f}s, "
                        f"queue={self._queue_wait_s(job):.2f}s, "
                        f"first_block={self._time_to_first_chunk_s(job):.2f}s)"
                    )

            job.chunks_appended += 1

    def _apply_webrtc_pose_crossfade(
        self,
        job: HLSStreamJob,
        frames: list,
        pose_ids: list,
        pose_crossfade_frames: Optional[list] = None,
        pose_source_indices: Optional[list] = None,
        raw_layers: Optional[list] = None,
    ) -> list:
        """Blend the first N frames after a live pose change without retiming."""
        bank = getattr(getattr(getattr(job, "session", None), "live_pose_router", None), "motion_bank", None)
        current_only = getattr(bank, "current_phoneme", None) is not None
        if not frames:
            return frames
        if len(pose_ids) != len(frames):
            if current_only:
                raise ValueError("Current phoneme composition requires every pose ID")
            return frames
        requested_crossfades = list(pose_crossfade_frames or [])
        if len(requested_crossfades) != len(frames):
            if current_only:
                raise ValueError("Current phoneme composition requires every crossfade length")
            requested_crossfades = [0] * len(frames)

        source_indices = list(pose_source_indices or [])
        if len(source_indices) != len(frames):
            if current_only:
                raise ValueError("Current phoneme composition requires every source index")
            if bank is not None and bank.eye_blend is not None:
                raise ValueError("Eye-aware motion composition requires every source index")
            source_indices = [None] * len(frames)
        layers = [None] * len(frames)
        if current_only:
            from scripts.motion_current_phoneme import validate_alpha
            if not isinstance(raw_layers, (list, tuple)) or len(raw_layers) != len(frames):
                raise ValueError("Current phoneme composition requires every raw layer")
            layers = raw_layers
            # Validate the whole batch before changing history or publishing.
            for frame, layer, pose, index in zip(frames, layers, pose_ids, source_indices):
                if not isinstance(layer, dict) or "raw" not in layer or "alpha" not in layer:
                    raise ValueError("Current phoneme composition requires raw and alpha layers")
                raw = layer["raw"]
                if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint8
                        or raw.ndim != 3 or raw.shape[2] != 3
                        or raw.shape != np.asarray(frame).shape):
                    raise ValueError("Current phoneme composition requires matching uint8 BGR raw layers")
                if (pose not in bank.sources or isinstance(index, (bool, np.bool_))
                        or not isinstance(index, (int, np.integer))
                        or not 0 <= index < bank.count(pose)):
                    raise ValueError("Current phoneme composition requires exact original source indices")
                source = bank.sources[pose]
                if raw.shape[:2] != (source["height"], source["width"]):
                    raise ValueError("Current phoneme composition requires original source dimensions")
                validate_alpha(layer["alpha"], *raw.shape[:2])
        blended_frames = []
        last_frame_index = len(frames) - 1
        for frame_index, (frame, pose_id, requested_crossfade, source_index, layer) in enumerate(zip(
            frames,
            pose_ids,
            requested_crossfades,
            source_indices,
            layers,
        )):
            normalized_pose_id = str(pose_id or "default")
            source_frame = np.asarray(frame)
            if (
                job.webrtc_last_pose_id is not None
                and normalized_pose_id != job.webrtc_last_pose_id
                and job.webrtc_last_pose_frame is not None
            ):
                job.webrtc_pose_crossfade_anchor = np.asarray(
                    job.webrtc_last_pose_frame,
                ).copy()
                job.webrtc_pose_crossfade_anchor_pose = job.webrtc_last_pose_id
                job.webrtc_pose_crossfade_anchor_source = getattr(job, "webrtc_last_source_frame", None)
                if current_only:
                    # History advances only here, in generation order. Prepared
                    # raw layers are immutable cache views; freeze by reference.
                    job.webrtc_pose_crossfade_raw_anchor = job.webrtc_last_raw_pose_frame
                    job.webrtc_pose_crossfade_alpha_anchor = job.webrtc_last_pose_alpha
                job.webrtc_pose_crossfade_index = 0
                job.webrtc_pose_crossfade_target_frames = max(
                    self.webrtc_pose_crossfade_frames,
                    max(0, int(requested_crossfade or 0)),
                )
                if job.webrtc_pose_crossfade_target_frames > 0:
                    job.webrtc_pose_crossfade_count += 1
                    print(
                        f"🎞️ [{job.request_id}] WebRTC pose crossfade "
                        f"{job.webrtc_last_pose_id}->{normalized_pose_id} "
                        f"frames={job.webrtc_pose_crossfade_target_frames}",
                        flush=True,
                    )
                else:
                    job.webrtc_pose_crossfade_anchor = None
                    job.webrtc_pose_crossfade_raw_anchor = None
                    job.webrtc_pose_crossfade_alpha_anchor = None

            output_frame = source_frame
            anchor = job.webrtc_pose_crossfade_anchor
            fade_index = job.webrtc_pose_crossfade_index
            frame_count = job.webrtc_pose_crossfade_target_frames
            if anchor is not None and fade_index < frame_count:
                anchor_array = np.asarray(anchor)
                if current_only and anchor_array.shape != source_frame.shape:
                    raise ValueError("Current phoneme anchor has incompatible frame dimensions")
                if anchor_array.shape == source_frame.shape:
                    progress = float(fade_index + 1) / float(frame_count + 1)
                    alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
                    if current_only:
                        output_frame = bank.blend_current(
                            job.webrtc_pose_crossfade_raw_anchor, layer["raw"], source_frame,
                            job.webrtc_pose_crossfade_alpha_anchor, layer["alpha"], alpha,
                            job.webrtc_pose_crossfade_anchor_pose,
                            job.webrtc_pose_crossfade_anchor_source,
                            normalized_pose_id, source_index)
                    elif bank is not None:
                        output_frame = bank.blend(anchor_array, source_frame, alpha,
                            job.webrtc_pose_crossfade_anchor_pose,
                            job.webrtc_pose_crossfade_anchor_source,
                            normalized_pose_id, source_index)
                    else:
                        blended = (anchor_array.astype(np.float32) * (1.0 - alpha)
                                   + source_frame.astype(np.float32) * alpha)
                        output_frame = np.clip(blended, 0, 255).astype(source_frame.dtype)
                    job.webrtc_pose_crossfade_frames_applied += 1
                job.webrtc_pose_crossfade_index += 1
                if job.webrtc_pose_crossfade_index >= frame_count:
                    job.webrtc_pose_crossfade_anchor = None
                    job.webrtc_pose_crossfade_raw_anchor = None
                    job.webrtc_pose_crossfade_alpha_anchor = None
                    job.webrtc_pose_crossfade_target_frames = 0

            if (
                self.webrtc_yuv_in_compose
                and output_frame is source_frame
                and frame is not source_frame
                and getattr(frame, "yuv420p", None) is not None
            ):
                # Unchanged by a crossfade: keep the carrier with its exact
                # yuv420p. Blended frames stay plain BGR and are converted by
                # the consumer as before.
                output_frame = frame
            blended_frames.append(output_frame)
            if self.skip_crossfade_copy and frame_index < last_frame_index:
                # Added code (HLS_SKIP_CROSSFADE_COPY=1): the history is read
                # only by the next frame's pose-change check, which copies it
                # into the anchor; within a batch nothing mutates this frame
                # before that, so only the batch's last frame (which outlives
                # the batch) needs its own copy.
                job.webrtc_last_pose_frame = source_frame
            else:
                job.webrtc_last_pose_frame = source_frame.copy()
            if current_only:
                job.webrtc_last_raw_pose_frame = layer["raw"]
                job.webrtc_last_pose_alpha = layer["alpha"]
            job.webrtc_last_pose_id = normalized_pose_id
            job.webrtc_last_source_frame = source_index

        return blended_frames

    def _dispatch_encode(self, job: HLSStreamJob, force_flush: bool = False) -> None:
        if not job.frame_buffer:
            return
        target_frames = self._next_chunk_target_frames(job)
        if not force_flush and len(job.frame_buffer) < target_frames:
            return

        if force_flush:
            take = len(job.frame_buffer)
        else:
            take = min(len(job.frame_buffer), target_frames)

        frames = job.frame_buffer[:take]
        del job.frame_buffer[:take]
        chunk_index = job.chunk_index
        job.chunk_index += 1
        start_frame = job.encoded_frame_cursor
        job.encoded_frame_cursor += len(frames)
        total_frames = job.total_frames
        fps = job.generation_fps
        audio_path = job.audio_path
        audio_copy_path = job.audio_copy_path
        output_path = str(job.chunk_output_dir / f"chunk_{chunk_index:04d}.ts")
        is_final_chunk = (start_frame + len(frames)) >= total_frames
        encode_submitted_at = time.time()

        def encode_chunk():
            encode_started_at = time.time()
            if is_final_chunk and job.crossfade_tail_frames > 0:
                chunk_path = job.avatar._create_crossfade_chunk(
                    frames=frames,
                    idle_frames=job.idle_frames,
                    fade_frames=job.crossfade_tail_frames,
                    chunk_index=chunk_index,
                    audio_path=audio_path,
                    audio_copy_path=audio_copy_path,
                    fps=fps,
                    start_frame=start_frame,
                    total_frames=total_frames,
                    output_path=output_path,
                )
            else:
                chunk_path = job.avatar._create_chunk(
                    frames=frames,
                    chunk_index=chunk_index,
                    audio_path=audio_path,
                    audio_copy_path=audio_copy_path,
                    fps=fps,
                    start_frame=start_frame,
                    total_frames=total_frames,
                    output_path=output_path,
                )
            return {
                "chunk_path": chunk_path,
                "chunk_index": chunk_index,
                "total_chunks": job.total_chunks,
                "duration_seconds": len(frames) / fps,
                "queue_wait_s": encode_started_at - encode_submitted_at,
                "creation_time": time.time() - encode_started_at,
            }

        future = self.encode_executor.submit(encode_chunk)
        job.encode_tasks[chunk_index] = future
        job.max_pending_encodes = max(job.max_pending_encodes, len(job.encode_tasks))

    def _drain_completed_encodes(self) -> None:
        for job in list(self.jobs.values()):
            for chunk_index, future in list(job.encode_tasks.items()):
                if not future.done():
                    continue
                del job.encode_tasks[chunk_index]
                try:
                    chunk_info = future.result()
                    job.chunks_encoded += 1
                    job.encode_queue_wait_total_s += chunk_info.get("queue_wait_s", 0.0)
                    job.encode_total_s += chunk_info.get("creation_time", 0.0)
                    job.max_encode_queue_wait_s = max(
                        job.max_encode_queue_wait_s,
                        chunk_info.get("queue_wait_s", 0.0),
                    )
                    job.max_encode_s = max(
                        job.max_encode_s,
                        chunk_info.get("creation_time", 0.0),
                    )
                    job.encoded_chunks[chunk_index] = chunk_info
                except Exception as exc:
                    job.error_message = str(exc)
                    print(f"❌ [{job.request_id}] chunk encode failed: {exc}")
                    traceback.print_exc()

            self._append_ready_segments(job)

        self._finalize_ready_jobs()
        self._finalize_cancelled_jobs()

    def _append_ready_segments(self, job: HLSStreamJob) -> None:
        while job.next_append_chunk_index in job.encoded_chunks:
            chunk_info = job.encoded_chunks.pop(job.next_append_chunk_index)
            segment_path = Path(chunk_info["chunk_path"])
            try:
                segment_name = segment_path.relative_to(job.session.segment_dir).as_posix()
            except ValueError:
                segment_name = segment_path.name
            duration = chunk_info.get("duration_seconds") or job.session.segment_duration
            self.hls_session_manager.append_live_segment(job.session, segment_name, duration)
            job.next_append_chunk_index += 1
            job.last_progress_at = time.time()
            job.chunks_appended += 1
            if job.first_chunk_appended_at is None:
                job.first_chunk_appended_at = job.last_progress_at
                print(
                    f"🎛️  [{job.request_id}] first chunk ready "
                    f"(prep={job.prep_total_s:.2f}s, queue={self._queue_wait_s(job):.2f}s, "
                    f"first_chunk={self._time_to_first_chunk_s(job):.2f}s)"
                )

    def _finalize_cancelled_jobs(self) -> None:
        for job in list(self.jobs.values()):
            if job.finalized:
                continue
            if not job.cancel_event.is_set():
                continue
            if job.gpu_inflight_batches > 0:
                continue
            if job.compose_tasks:
                continue
            if job.encode_tasks:
                continue
            self._finalize_job(job, "cancelled")

    def _finalize_ready_jobs(self) -> None:
        for job in list(self.jobs.values()):
            if job.finalized:
                continue
            if job.error_message:
                self._finalize_job(job, "failed", error_message=job.error_message)
                continue
            if (
                job.generation_done
                and not job.compose_tasks
                and job.next_compose_sequence >= job.compose_sequence
                and not job.encode_tasks
                and not job.frame_buffer
                and job.next_append_chunk_index >= job.chunk_index
            ):
                self._finalize_job(job, "completed")

    def _finalize_job(self, job: HLSStreamJob, status: str, error_message: Optional[str] = None) -> None:
        if job.finalized:
            return

        job.finalized = True
        job.finalized_at = time.time()
        job.webrtc_last_raw_pose_frame = None
        job.webrtc_last_pose_alpha = None
        job.webrtc_pose_crossfade_raw_anchor = None
        job.webrtc_pose_crossfade_alpha_anchor = None
        job.composed_batches.clear()
        with self.condition:
            self.jobs.pop(job.request_id, None)
            self.condition.notify_all()

        if job.output_mode == "hls":
            self.hls_session_manager.finish_live_playlist(job.session)
            job.session.active_stream = None
        else:
            if job.generation_complete_callback is not None:
                try:
                    job.generation_complete_callback(status, error_message)
                except Exception as callback_exc:
                    print(f"⚠️  [{job.request_id}] WebRTC completion callback failed: {callback_exc}")
            if status != "completed":
                job.session.active_stream = None
        if hasattr(job.session, "cancel_requested"):
            job.session.cancel_requested = False
        self._set_request_status(job.request_id, status)
        self._resolve_completion(job.completion_future, job.main_loop, status, error_message)

        if job.output_mode == "hls":
            try:
                Path(job.audio_path).unlink(missing_ok=True)
            except OSError:
                pass
            if job.audio_copy_path:
                try:
                    Path(job.audio_copy_path).unlink(missing_ok=True)
                except OSError:
                    pass

        output_label = "HLS" if job.output_mode == "hls" else "WebRTC"
        output_count_label = "chunks" if job.output_mode == "hls" else "batches_pushed"
        print(
            f"🎛️  [{job.request_id}] {output_label} scheduler finished with status={status} "
            f"(prep={job.prep_total_s:.2f}s, prep_wait={job.prep_queue_wait_s:.2f}s, "
            f"prep_work={job.prep_work_s:.2f}s, audio_copy={job.audio_copy_prep_s:.2f}s, "
            f"queue={self._queue_wait_s(job):.2f}s, "
            f"first_chunk={self._time_to_first_chunk_s(job):.2f}s, "
            f"avg_gpu_batch={self._safe_avg(job.gpu_batch_total_s, job.gpu_batch_count):.3f}s, "
            f"gpu_batches={job.gpu_batch_count}, "
            f"avg_assemble={self._safe_avg(job.batch_assembly_total_s, job.gpu_batch_count):.3f}s, "
            f"avg_copy={self._safe_avg(job.gpu_copy_total_s, job.gpu_batch_count):.3f}s, "
            f"avg_pe={self._safe_avg(job.pe_total_s, job.gpu_batch_count):.3f}s, "
            f"avg_unet={self._safe_avg(job.unet_total_s, job.gpu_batch_count):.3f}s, "
            f"avg_vae={self._safe_avg(job.vae_total_s, job.gpu_batch_count):.3f}s, "
            f"avg_compose_wait={self._safe_avg(job.compose_queue_wait_total_s, job.compose_batch_count):.3f}s, "
            f"avg_compose={self._safe_avg(job.compose_total_s, job.compose_batch_count):.3f}s, "
            f"avg_callback={self._safe_avg(job.frame_callback_total_s, job.frame_callback_count):.3f}s, "
            f"max_callback={job.frame_callback_max_s:.3f}s, "
            f"avg_encode_wait={self._safe_avg(job.encode_queue_wait_total_s, job.chunks_encoded):.3f}s, "
            f"avg_encode={self._safe_avg(job.encode_total_s, job.chunks_encoded):.3f}s, "
            f"max_buffer={job.max_frame_buffer_len}/{job.frames_per_chunk}, "
            f"max_pending={job.max_pending_composes}/{job.max_pending_encodes}, "
            f"max_compose_wait={job.max_compose_queue_wait_s:.3f}s, "
            f"max_encode_wait={job.max_encode_queue_wait_s:.3f}s, "
            f"post_gen_drain={self._post_generation_drain_s(job):.3f}s, "
            f"{output_count_label}={job.chunks_appended})"
        )

    def _resolve_completion(self, completion_future, main_loop, status: str, error_message: Optional[str]) -> None:
        def _complete():
            if completion_future.done():
                return
            completion_future.set_result(
                {
                    "status": status,
                    "error": error_message,
                }
            )

        main_loop.call_soon_threadsafe(_complete)

    def _set_request_status(self, request_id: str, status: str) -> None:
        with self.manager.request_lock:
            req = self.manager.active_requests.get(request_id)
            if req is not None:
                req["status"] = status

    @staticmethod
    def _is_startup_job(job: HLSStreamJob) -> bool:
        return job.first_chunk_appended_at is None

    @staticmethod
    def _remaining_frames(job: HLSStreamJob, allocations: Optional[Dict[str, int]] = None) -> int:
        already_allocated = 0
        if allocations is not None:
            already_allocated = allocations.get(job.request_id, 0)
        with job.conditioning_lock:
            available_frames = job.total_frames if job.conditioning_complete else job.conditioning_ready_frames
        return max(0, available_frames - job.current_frame_idx - already_allocated)

    def _sync_gpu_for_stage_timing(self) -> None:
        if not self.gpu_stage_sync_timing or not torch.cuda.is_available():
            return
        device = getattr(self.manager, "device", None)
        device_type = getattr(device, "type", None)
        if device_type is None and isinstance(device, str):
            device_type = device.split(":", 1)[0]
        if device_type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _safe_avg(total: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return total / count

    @staticmethod
    def _queue_wait_s(job: HLSStreamJob) -> float:
        if job.first_scheduled_at is None:
            return 0.0
        return max(0.0, job.first_scheduled_at - job.queued_at)

    @staticmethod
    def _time_to_first_chunk_s(job: HLSStreamJob) -> float:
        if job.first_chunk_appended_at is None:
            return 0.0
        return max(0.0, job.first_chunk_appended_at - job.submitted_at)

    @staticmethod
    def _next_chunk_target_frames(job: HLSStreamJob) -> int:
        if job.chunk_index < job.startup_chunk_count and job.startup_chunk_frames > 0:
            return job.startup_chunk_frames
        return job.frames_per_chunk

    @staticmethod
    def _frames_until_startup_chunk(job: HLSStreamJob, allocations: Optional[Dict[str, int]] = None) -> int:
        if job.startup_chunk_count <= 0 or job.first_chunk_appended_at is not None:
            return 0
        already_allocated = allocations.get(job.request_id, 0) if allocations is not None else 0
        return max(0, job.startup_chunk_frames - job.current_frame_idx - already_allocated)

    @staticmethod
    def _generated_unencoded_frames(job: HLSStreamJob, allocations: Optional[Dict[str, int]] = None) -> int:
        already_allocated = allocations.get(job.request_id, 0) if allocations is not None else 0
        return max(0, (job.current_frame_idx - job.encoded_frame_cursor) + already_allocated)

    @classmethod
    def _frames_until_next_chunk(cls, job: HLSStreamJob, allocations: Optional[Dict[str, int]] = None) -> int:
        return max(0, cls._next_chunk_target_frames(job) - cls._generated_unencoded_frames(job, allocations))

    @staticmethod
    def _post_generation_drain_s(job: HLSStreamJob) -> float:
        if job.generation_done_at is None:
            return 0.0
        end_time = job.finalized_at or time.time()
        return max(0.0, end_time - job.generation_done_at)

    def _startup_chunk_frames(self, steady_chunk_frames: int, generation_fps: int) -> int:
        if self.startup_chunk_count <= 0 or self.startup_chunk_duration_seconds <= 0:
            return steady_chunk_frames
        startup_frames = max(1, int(round(self.startup_chunk_duration_seconds * generation_fps)))
        return min(steady_chunk_frames, startup_frames)

    @staticmethod
    def _estimate_total_chunks(
        *,
        total_frames: int,
        frames_per_chunk: int,
        startup_chunk_frames: int,
        startup_chunk_count: int,
    ) -> int:
        if total_frames <= 0:
            return 0
        if startup_chunk_count <= 0 or startup_chunk_frames >= frames_per_chunk:
            return int(math.ceil(total_frames / frames_per_chunk))

        remaining_frames = total_frames
        chunk_count = 0
        for _ in range(startup_chunk_count):
            if remaining_frames <= 0:
                break
            take = min(startup_chunk_frames, remaining_frames)
            remaining_frames -= take
            chunk_count += 1
        if remaining_frames > 0:
            chunk_count += int(math.ceil(remaining_frames / frames_per_chunk))
        return chunk_count

    def _get_staging_buffers(
        self,
        *,
        conditioning_shape: tuple[int, ...],
        conditioning_dtype: torch.dtype,
        latent_shape: tuple[int, ...],
        latent_dtype: torch.dtype,
        batch_size: int,
        slot: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (
            batch_size,
            conditioning_shape,
            str(conditioning_dtype),
            latent_shape,
            str(latent_dtype),
        )
        if slot is not None:
            # Added code (depth >= 2): one staging set per pipeline slot, so
            # assembling batch N+1 never overwrites batch N's pending H2D.
            cache_key = ("pipeline_slot", int(slot)) + cache_key
        buffers = self._cpu_staging_cache.get(cache_key)
        if buffers is None:
            pin_memory = torch.cuda.is_available()
            buffers = (
                torch.empty(
                    (batch_size,) + conditioning_shape,
                    dtype=conditioning_dtype,
                    pin_memory=pin_memory,
                ),
                torch.empty(
                    (batch_size,) + latent_shape,
                    dtype=latent_dtype,
                    pin_memory=pin_memory,
                ),
            )
            self._cpu_staging_cache[cache_key] = buffers
        return buffers

    def _apply_positional_encoding_cpu(self, audio_prompts: torch.Tensor) -> torch.Tensor:
        if not isinstance(audio_prompts, torch.Tensor) or audio_prompts.dim() != 3:
            return audio_prompts

        pe_buffer = getattr(self.manager.pe, "pe", None)
        if pe_buffer is None:
            return audio_prompts

        cache_key = (audio_prompts.shape[1], str(audio_prompts.dtype))
        pe_slice = self._cpu_pe_cache.get(cache_key)
        if pe_slice is None:
            pe_slice = pe_buffer[:, :audio_prompts.shape[1], :].detach().to(
                device="cpu",
                dtype=audio_prompts.dtype,
            ).contiguous()
            self._cpu_pe_cache[cache_key] = pe_slice

        return (audio_prompts + pe_slice).contiguous()

    @staticmethod
    def _memory_bucket(batch_size: int) -> int:
        """
        Returns a lease size for gpu_memory.allocate().
        
        IMPORTANT: This is NOT the GPU batch size. The actual forward pass
        processes `total_batch` frames regardless of this value.
        
        This is a slot count against the memory manager's semaphore pool.
        The HLS scheduler is the SOLE user of the GPU loop — it runs one
        batch at a time sequentially. So it only ever needs 1 slot.
        Requesting more than the pool has causes a permanent deadlock.
        """
        return 1

    @staticmethod
    def _parse_batch_size_list(raw: str) -> list[int]:
        values: list[int] = []
        seen = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = max(1, int(token))
            except ValueError:
                continue
            if value in seen:
                continue
            values.append(value)
            seen.add(value)
        values.sort()
        return values

    @classmethod
    def _resolve_fixed_batch_sizes(cls, max_combined_batch_size: int) -> list[int]:
        override = os.getenv("HLS_SCHEDULER_FIXED_BATCH_SIZES", "").strip()
        if override:
            parsed = cls._parse_batch_size_list(override)
            if parsed:
                return parsed

        # Keep the original compile-friendly buckets and, when the configured
        # combined batch exceeds 32, extend them in 16-frame increments so
        # 48/64-size turns stay on known shapes instead of falling back to
        # ad-hoc actual-batch recompilation.
        values = [4, 8, 16, 32]
        if max_combined_batch_size > 32:
            rounded_upper = ((int(max_combined_batch_size) + 15) // 16) * 16
            next_size = 48
            while next_size <= rounded_upper:
                values.append(next_size)
                next_size += 16
        return values
