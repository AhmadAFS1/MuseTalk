import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class HLSStreamJob:
    request_id: str
    session_id: str
    session: object
    avatar: object
    audio_path: str
    chunk_output_dir: Path
    generation_fps: int
    batch_size: int
    whisper_chunks: object
    total_frames: int
    frames_per_chunk: int
    total_chunks: int
    start_offset_frames: int
    cancel_event: threading.Event
    completion_future: object
    main_loop: object
    idle_frames: list = field(default_factory=list)
    crossfade_tail_frames: int = 0
    current_frame_idx: int = 0
    composed_frame_idx: int = 0
    chunk_index: int = 0
    frame_buffer: list = field(default_factory=list)
    generation_done: bool = False
    finalized: bool = False
    last_progress_at: float = field(default_factory=time.time)
    compose_tasks: Dict[int, object] = field(default_factory=dict)
    composed_batches: Dict[int, dict] = field(default_factory=dict)
    compose_sequence: int = 0
    next_compose_sequence: int = 0
    encode_tasks: Dict[int, object] = field(default_factory=dict)
    encoded_chunks: Dict[int, dict] = field(default_factory=dict)
    next_append_chunk_index: int = 0
    error_message: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    prep_started_at: float = 0.0
    queued_at: float = 0.0
    prep_total_s: float = 0.0
    avatar_load_s: float = 0.0
    audio_feature_s: float = 0.0
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
    chunks_encoded: int = 0
    encode_queue_wait_total_s: float = 0.0
    encode_total_s: float = 0.0
    chunks_appended: int = 0


class HLSGPUStreamScheduler:
    """
    Shared HLS GPU scheduler.

    One GPU thread batches work across active HLS sessions and a separate
    encode pool turns ready frame buffers into TS segments.
    """

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
    ):
        self.manager = manager
        self.hls_session_manager = hls_session_manager
        self.max_combined_batch_size = max(1, int(max_combined_batch_size))
        self.startup_slice_size = max(1, int(startup_slice_size))
        self.aggressive_fill_max_active_jobs = max(0, int(aggressive_fill_max_active_jobs))
        self.max_pending_jobs = max(1, int(max_pending_jobs))
        self.prep_executor = ThreadPoolExecutor(max_workers=max(1, int(prep_workers)))

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

        # Encode: each stream has its own persistent ffmpeg pipe,
        # but the submit_frames calls still need thread workers.
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

    def start(self) -> None:
        if self.scheduler_thread is not None:
            return
        self.scheduler_thread = threading.Thread(target=self._run_loop, daemon=True, name="hls-gpu-scheduler")
        self.scheduler_thread.start()
        print(
            "🎛️  HLS GPU scheduler started "
            f"(max_combined_batch_size={self.max_combined_batch_size}, "
            f"startup_slice_size={self.startup_slice_size}, "
            f"aggressive_fill_max_active_jobs={self.aggressive_fill_max_active_jobs}, "
            f"compose_workers={self.compose_executor._max_workers}, "
            f"encode_workers={self.encode_executor._max_workers})"
        )

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.scheduler_thread is not None:
            self.scheduler_thread.join(timeout=10)
        self.prep_executor.shutdown(wait=False, cancel_futures=True)
        self.compose_executor.shutdown(wait=False, cancel_futures=True)
        self.encode_executor.shutdown(wait=False, cancel_futures=True)
        print("🎛️  HLS GPU scheduler stopped")

    def submit_stream(
        self,
        session,
        request_id: str,
        audio_path: str,
        generation_fps: int,
        start_offset_seconds: float,
        cancel_event: threading.Event,
        completion_future,
        main_loop,
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
            submitted_at,
        )
        return True

    def get_stats(self) -> dict:
        with self.condition:
            return {
                "queued_or_active_jobs": len(self.jobs),
                "preparing_jobs": len(self.preparing_requests),
                "max_combined_batch_size": self.max_combined_batch_size,
                "startup_slice_size": self.startup_slice_size,
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
                        "current_frame_idx": job.current_frame_idx,
                        "total_frames": job.total_frames,
                        "composed_frame_idx": job.composed_frame_idx,
                        "chunk_index": job.chunk_index,
                        "total_chunks": job.total_chunks,
                        "pending_composes": len(job.compose_tasks),
                        "pending_encodes": len(job.encode_tasks),
                        "startup_pending": self._is_startup_job(job),
                        "cancel_requested": job.cancel_event.is_set(),
                        "last_progress_age_s": round(time.time() - job.last_progress_at, 3),
                        "prep_total_s": round(job.prep_total_s, 3),
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
                        "avg_encode_queue_wait_s": round(self._safe_avg(job.encode_queue_wait_total_s, job.chunks_encoded), 4),
                        "avg_encode_s": round(self._safe_avg(job.encode_total_s, job.chunks_encoded), 4),
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
        start_offset_seconds: float,
        cancel_event: threading.Event,
        completion_future,
        main_loop,
        submitted_at: float,
    ) -> None:
        prep_started_at = time.time()
        try:
            if cancel_event.is_set():
                self._finish_before_enqueue(
                    request_id, session, audio_path, completion_future, main_loop, "cancelled"
                )
                return

            avatar_load_start = time.time()
            avatar = self.manager._get_or_load_avatar(session.avatar_id, session.batch_size)
            avatar_load_s = time.time() - avatar_load_start
            if session.idle_cycle_frames is None and hasattr(avatar, "input_latent_list_cycle"):
                session.idle_cycle_frames = len(avatar.input_latent_list_cycle)

            weight_dtype = self.manager.unet.model.dtype
            audio_feature_start = time.time()
            whisper_input_features, librosa_length = self.manager.audio_processor.get_audio_feature(
                audio_path, weight_dtype=weight_dtype
            )
            audio_feature_s = time.time() - audio_feature_start
            if whisper_input_features is None:
                raise RuntimeError("Audio feature extraction failed")

            if cancel_event.is_set():
                self._finish_before_enqueue(
                    request_id, session, audio_path, completion_future, main_loop, "cancelled"
                )
                return

            whisper_chunk_start = time.time()
            whisper_chunks = self.manager.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.manager.device,
                weight_dtype,
                self.manager.whisper,
                librosa_length,
                fps=generation_fps,
                audio_padding_length_left=self.manager.args.audio_padding_length_left,
                audio_padding_length_right=self.manager.args.audio_padding_length_right,
            ).detach().cpu()
            whisper_chunk_s = time.time() - whisper_chunk_start

            total_frames = int(len(whisper_chunks))
            frames_per_chunk = max(1, int(round(session.segment_duration * generation_fps)))
            total_chunks = int(math.ceil(total_frames / frames_per_chunk)) if total_frames > 0 else 0
            start_offset_frames = int(round(max(0.0, float(start_offset_seconds)) * generation_fps))

            idle_frames = avatar._get_idle_frames(max_frames=max(8, min(24, int(generation_fps * 0.8))))
            crossfade_tail_frames = 0
            if idle_frames:
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
                audio_path=audio_path,
                chunk_output_dir=session.segment_dir / request_id,
                generation_fps=generation_fps,
                batch_size=max(1, int(session.batch_size)),
                whisper_chunks=whisper_chunks,
                total_frames=total_frames,
                frames_per_chunk=frames_per_chunk,
                total_chunks=total_chunks,
                start_offset_frames=start_offset_frames,
                cancel_event=cancel_event,
                completion_future=completion_future,
                main_loop=main_loop,
                idle_frames=idle_frames,
                crossfade_tail_frames=crossfade_tail_frames,
                submitted_at=submitted_at,
                prep_started_at=prep_started_at,
                queued_at=queued_at,
                prep_total_s=queued_at - submitted_at,
                avatar_load_s=avatar_load_s,
                audio_feature_s=audio_feature_s,
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

            print(
                f"🎛️  [{request_id}] queued for shared HLS GPU scheduler "
                f"(frames={total_frames}, chunks={total_chunks}, batch_size={job.batch_size}, "
                f"prep={job.prep_total_s:.2f}s)"
            )
        except Exception as exc:
            print(f"❌ [{request_id}] HLS prep failed: {exc}")
            traceback.print_exc()
            self._finish_before_enqueue(
                request_id,
                session,
                audio_path,
                completion_future,
                main_loop,
                "failed",
                error_message=str(exc),
            )

    def _finish_before_enqueue(
        self,
        request_id: str,
        session,
        audio_path: str,
        completion_future,
        main_loop,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        with self.condition:
            self.preparing_requests.discard(request_id)
            self.condition.notify_all()

        self.hls_session_manager.finish_live_playlist(session)
        session.active_stream = None
        session.cancel_requested = False
        self._set_request_status(request_id, status)
        self._resolve_completion(completion_future, main_loop, status, error_message)
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError:
            pass

    def _run_loop(self) -> None:
        while True:
            self._drain_completed_composes()
            self._drain_completed_encodes()
            self._finalize_ready_jobs()
            self._finalize_cancelled_jobs()

            with self.condition:
                if self.stop_event.is_set() and not self.jobs and not self.preparing_requests:
                    break

                selected = self._select_jobs_locked()
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

    def _select_jobs_locked(self):
        jobs = self._ordered_schedulable_jobs_locked()
        if not jobs:
            return []

        allocations: Dict[str, int] = {}
        total_batch = 0

        startup_jobs = [job for job in jobs if self._is_startup_job(job)]
        warmed_jobs = [job for job in jobs if not self._is_startup_job(job)]

        # --- Round 1: startup jobs get a small initial slice ---
        if startup_jobs:
            total_batch = self._allocate_round(
                jobs=startup_jobs,
                allocations=allocations,
                total_batch=total_batch,
                slice_cap=self.startup_slice_size,
            )

        # --- Round 2: warmed jobs get their per-job batch_size ---
        if total_batch < self.max_combined_batch_size and warmed_jobs:
            total_batch = self._allocate_round(
                jobs=warmed_jobs,
                allocations=allocations,
                total_batch=total_batch,
                slice_cap=None,
            )

        # --- Round 3: ALWAYS fill remaining GPU capacity ---
        # OLD: skipped when len(warmed_jobs) > aggressive_fill_max_active_jobs
        # At 8 streams with batch_size=2 and max_batch=8, Round 2 could only
        # fit 4 jobs (8/2=4). The other 4 jobs got NOTHING. And even those 4
        # jobs couldn't get extra frames because this guard blocked Round 3.
        # Now we always fill, regardless of how many jobs are active.
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
            if job.generation_done or job.current_frame_idx >= job.total_frames:
                continue

            remaining_frames = self._remaining_frames(job)
            if remaining_frames <= 0:
                job.generation_done = True
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

    def _run_generation_batch(self, selected) -> None:
        selected = [(job, take) for job, take in selected if take > 0 and not job.cancel_event.is_set()]
        if not selected:
            self._finalize_cancelled_jobs()
            return

        batch_started_at = time.time()
        total_batch = sum(take for _, take in selected)
        lease_batch_size = self._memory_bucket(total_batch)

        whisper_slices = []
        latent_slices = []

        for job, take in selected:
            if job.first_scheduled_at is None:
                job.first_scheduled_at = batch_started_at
            whisper_slices.append(job.whisper_chunks[job.current_frame_idx: job.current_frame_idx + take])
            for rel_index in range(take):
                cycle_index = job.start_offset_frames + job.current_frame_idx + rel_index
                latent_index = cycle_index % len(job.avatar.input_latent_list_cycle)
                latent_slices.append(job.avatar.input_latent_list_cycle[latent_index])
            job.last_progress_at = time.time()
            if not job.session.live_ready:
                job.session.status = "generating"
            self._set_request_status(job.request_id, "running")
            job.scheduler_turns += 1

        assembly_finished_at = time.time()
        whisper_batch = torch.cat(whisper_slices, dim=0)
        latent_batch = torch.cat(latent_slices, dim=0)

        with self.manager.gpu_memory.allocate(lease_batch_size):
            with torch.inference_mode():
                copy_started_at = time.time()
                audio_inputs = whisper_batch.to(self.manager.device, non_blocking=True)
                latent_batch = latent_batch.to(
                    device=self.manager.device,
                    dtype=self.manager.unet.model.dtype,
                    non_blocking=True,
                )
                copy_finished_at = time.time()

                pe_started_at = time.time()
                audio_feature_batch = self.manager.pe(audio_inputs)
                pe_finished_at = time.time()

                unet_started_at = time.time()
                pred_latents = self.manager.unet.model(
                    latent_batch,
                    self.manager.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                unet_finished_at = time.time()

                vae_started_at = time.time()
                pred_latents = pred_latents.to(device=self.manager.device, dtype=self.manager.vae.vae.dtype)
                recon = self.manager.vae.decode_latents(pred_latents)
                vae_finished_at = time.time()

        batch_finished_at = time.time()
        assembly_s = assembly_finished_at - batch_started_at
        copy_s = copy_finished_at - copy_started_at
        pe_s = pe_finished_at - pe_started_at
        unet_s = unet_finished_at - unet_started_at
        vae_s = vae_finished_at - vae_started_at
        gpu_batch_s = batch_finished_at - batch_started_at

        offset = 0
        for job, take in selected:
            batch_frames = recon[offset: offset + take]
            offset += take
            start_frame_idx = job.current_frame_idx
            job.current_frame_idx += take
            if not job.cancel_event.is_set():
                self._dispatch_compose_batch(job, batch_frames, start_frame_idx)
            if job.current_frame_idx >= job.total_frames:
                job.generation_done = True
            job.batch_assembly_total_s += assembly_s
            job.gpu_copy_total_s += copy_s
            job.pe_total_s += pe_s
            job.unet_total_s += unet_s
            job.vae_total_s += vae_s
            job.gpu_batch_total_s += gpu_batch_s
            job.gpu_batch_count += 1

        self._finalize_cancelled_jobs()
        self._finalize_ready_jobs()

    def _dispatch_compose_batch(self, job: HLSStreamJob, batch_frames, start_frame_idx: int) -> None:
        compose_sequence = job.compose_sequence
        job.compose_sequence += 1
        compose_submitted_at = time.time()

        def compose_batch():
            compose_started_at = time.time()
            frames = []
            for rel_index, res_frame in enumerate(batch_frames):
                cycle_index = job.start_offset_frames + start_frame_idx + rel_index
                frames.append(job.avatar.compose_frame(res_frame, cycle_index))
            return {
                "compose_sequence": compose_sequence,
                "frames": frames,
                "queue_wait_s": compose_started_at - compose_submitted_at,
                "compose_time": time.time() - compose_started_at,
            }

        future = self.compose_executor.submit(compose_batch)
        job.compose_tasks[compose_sequence] = future

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
                    job.composed_batches[compose_sequence] = compose_info
                except Exception as exc:
                    job.error_message = str(exc)
                    print(f"❌ [{job.request_id}] compose batch failed: {exc}")
                    traceback.print_exc()

            self._append_ready_composed_frames(job)

        self._finalize_ready_jobs()
        self._finalize_cancelled_jobs()

    def _append_ready_composed_frames(self, job: HLSStreamJob) -> None:
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

                while len(job.frame_buffer) >= job.frames_per_chunk:
                    self._dispatch_encode(job)

        if (
            not job.cancel_event.is_set()
            and job.generation_done
            and not job.compose_tasks
            and job.next_compose_sequence >= job.compose_sequence
            and job.frame_buffer
        ):
            self._dispatch_encode(job, force_flush=True)

    def _dispatch_encode(self, job: HLSStreamJob, force_flush: bool = False) -> None:
        if not job.frame_buffer:
            return
        if not force_flush and len(job.frame_buffer) < job.frames_per_chunk:
            return

        if force_flush:
            take = len(job.frame_buffer)
        else:
            take = min(len(job.frame_buffer), job.frames_per_chunk)

        frames = job.frame_buffer[:take]
        del job.frame_buffer[:take]
        chunk_index = job.chunk_index
        job.chunk_index += 1
        start_frame = chunk_index * job.frames_per_chunk
        total_frames = job.total_frames
        fps = job.generation_fps
        audio_path = job.audio_path
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
        with self.condition:
            self.jobs.pop(job.request_id, None)
            self.condition.notify_all()

        self.hls_session_manager.finish_live_playlist(job.session)
        job.session.active_stream = None
        job.session.cancel_requested = False
        self._set_request_status(job.request_id, status)
        self._resolve_completion(job.completion_future, job.main_loop, status, error_message)

        try:
            Path(job.audio_path).unlink(missing_ok=True)
        except OSError:
            pass

        print(
            f"🎛️  [{job.request_id}] HLS scheduler finished with status={status} "
            f"(prep={job.prep_total_s:.2f}s, queue={self._queue_wait_s(job):.2f}s, "
            f"first_chunk={self._time_to_first_chunk_s(job):.2f}s, "
            f"avg_gpu_batch={self._safe_avg(job.gpu_batch_total_s, job.gpu_batch_count):.3f}s, "
            f"avg_compose_wait={self._safe_avg(job.compose_queue_wait_total_s, job.compose_batch_count):.3f}s, "
            f"avg_compose={self._safe_avg(job.compose_total_s, job.compose_batch_count):.3f}s, "
            f"avg_encode_wait={self._safe_avg(job.encode_queue_wait_total_s, job.chunks_encoded):.3f}s, "
            f"avg_encode={self._safe_avg(job.encode_total_s, job.chunks_encoded):.3f}s, "
            f"chunks={job.chunks_appended})"
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
        return max(0, job.total_frames - job.current_frame_idx - already_allocated)

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
