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
    chunk_index: int = 0
    frame_buffer: list = field(default_factory=list)
    generation_done: bool = False
    finalized: bool = False
    last_progress_at: float = field(default_factory=time.time)
    encode_tasks: Dict[int, object] = field(default_factory=dict)
    encoded_chunks: Dict[int, dict] = field(default_factory=dict)
    next_append_chunk_index: int = 0
    error_message: Optional[str] = None


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
        prep_workers: int = 2,
        encode_workers: int = 2,
        max_pending_jobs: int = 16,
    ):
        self.manager = manager
        self.hls_session_manager = hls_session_manager
        self.max_combined_batch_size = max(1, int(max_combined_batch_size))
        self.max_pending_jobs = max(1, int(max_pending_jobs))
        self.prep_executor = ThreadPoolExecutor(max_workers=max(1, int(prep_workers)))
        self.encode_executor = ThreadPoolExecutor(max_workers=max(1, int(encode_workers)))
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.scheduler_thread: Optional[threading.Thread] = None
        self.jobs: Dict[str, HLSStreamJob] = {}
        self.preparing_requests: set[str] = set()

    def start(self) -> None:
        if self.scheduler_thread is not None:
            return
        self.scheduler_thread = threading.Thread(target=self._run_loop, daemon=True, name="hls-gpu-scheduler")
        self.scheduler_thread.start()
        print(
            "🎛️  HLS GPU scheduler started "
            f"(max_combined_batch_size={self.max_combined_batch_size}, "
            f"encode_workers={self.encode_executor._max_workers})"
        )

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.scheduler_thread is not None:
            self.scheduler_thread.join(timeout=10)
        self.prep_executor.shutdown(wait=False, cancel_futures=True)
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
        )
        return True

    def get_stats(self) -> dict:
        with self.condition:
            return {
                "queued_or_active_jobs": len(self.jobs),
                "preparing_jobs": len(self.preparing_requests),
                "max_combined_batch_size": self.max_combined_batch_size,
                "encode_workers": self.encode_executor._max_workers,
                "jobs": [
                    {
                        "request_id": job.request_id,
                        "session_id": job.session_id,
                        "current_frame_idx": job.current_frame_idx,
                        "total_frames": job.total_frames,
                        "chunk_index": job.chunk_index,
                        "total_chunks": job.total_chunks,
                        "pending_encodes": len(job.encode_tasks),
                        "cancel_requested": job.cancel_event.is_set(),
                        "last_progress_age_s": round(time.time() - job.last_progress_at, 3),
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
    ) -> None:
        try:
            if cancel_event.is_set():
                self._finish_before_enqueue(
                    request_id, session, audio_path, completion_future, main_loop, "cancelled"
                )
                return

            avatar = self.manager._get_or_load_avatar(session.avatar_id, session.batch_size)
            if session.idle_cycle_frames is None and hasattr(avatar, "input_latent_list_cycle"):
                session.idle_cycle_frames = len(avatar.input_latent_list_cycle)

            weight_dtype = self.manager.unet.model.dtype
            whisper_input_features, librosa_length = self.manager.audio_processor.get_audio_feature(
                audio_path, weight_dtype=weight_dtype
            )
            if whisper_input_features is None:
                raise RuntimeError("Audio feature extraction failed")

            if cancel_event.is_set():
                self._finish_before_enqueue(
                    request_id, session, audio_path, completion_future, main_loop, "cancelled"
                )
                return

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
                f"(frames={total_frames}, chunks={total_chunks}, batch_size={job.batch_size})"
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
            self._drain_completed_encodes()

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

            self._drain_completed_encodes()

    def _select_jobs_locked(self):
        ready_jobs = []
        total_batch = 0

        for job in list(self.jobs.values()):
            if job.finalized:
                continue
            if job.cancel_event.is_set() and not job.encode_tasks:
                ready_jobs.append((job, 0))
                continue
            if job.generation_done or job.current_frame_idx >= job.total_frames:
                continue

            remaining_frames = job.total_frames - job.current_frame_idx
            if remaining_frames <= 0:
                job.generation_done = True
                continue

            capacity_left = self.max_combined_batch_size - total_batch
            if capacity_left <= 0:
                break

            take = min(job.batch_size, remaining_frames, capacity_left)
            if take <= 0:
                continue
            ready_jobs.append((job, take))
            total_batch += take

        return ready_jobs

    def _run_generation_batch(self, selected) -> None:
        selected = [(job, take) for job, take in selected if take > 0 and not job.cancel_event.is_set()]
        if not selected:
            self._finalize_cancelled_jobs()
            return

        total_batch = sum(take for _, take in selected)
        lease_batch_size = self._memory_bucket(total_batch)

        whisper_slices = []
        latent_slices = []
        split_sizes = []

        for job, take in selected:
            whisper_slices.append(job.whisper_chunks[job.current_frame_idx: job.current_frame_idx + take])
            split_sizes.append(take)
            for rel_index in range(take):
                cycle_index = job.start_offset_frames + job.current_frame_idx + rel_index
                latent_index = cycle_index % len(job.avatar.input_latent_list_cycle)
                latent_slices.append(job.avatar.input_latent_list_cycle[latent_index])
            job.last_progress_at = time.time()
            if not job.session.live_ready:
                job.session.status = "generating"
            self._set_request_status(job.request_id, "running")

        whisper_batch = torch.cat(whisper_slices, dim=0)
        latent_batch = torch.cat(latent_slices, dim=0)

        with self.manager.gpu_memory.allocate(lease_batch_size):
            audio_feature_batch = self.manager.pe(whisper_batch.to(self.manager.device))
            latent_batch = latent_batch.to(device=self.manager.device, dtype=self.manager.unet.model.dtype)
            pred_latents = self.manager.unet.model(
                latent_batch,
                self.manager.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            pred_latents = pred_latents.to(device=self.manager.device, dtype=self.manager.vae.vae.dtype)
            recon = self.manager.vae.decode_latents(pred_latents)

        offset = 0
        for job, take in selected:
            batch_frames = recon[offset: offset + take]
            offset += take

            for res_frame in batch_frames:
                if job.cancel_event.is_set():
                    break
                cycle_index = job.start_offset_frames + job.current_frame_idx
                combined_frame = job.avatar.compose_frame(res_frame, cycle_index)
                job.frame_buffer.append(combined_frame)
                job.current_frame_idx += 1
                job.last_progress_at = time.time()

                if len(job.frame_buffer) >= job.frames_per_chunk or job.current_frame_idx >= job.total_frames:
                    self._dispatch_encode(job)

            if job.current_frame_idx >= job.total_frames:
                job.generation_done = True

        self._finalize_cancelled_jobs()
        self._finalize_ready_jobs()

    def _dispatch_encode(self, job: HLSStreamJob) -> None:
        if not job.frame_buffer:
            return

        frames = job.frame_buffer
        job.frame_buffer = []
        chunk_index = job.chunk_index
        job.chunk_index += 1
        start_frame = chunk_index * job.frames_per_chunk
        total_frames = job.total_frames
        fps = job.generation_fps
        audio_path = job.audio_path
        output_path = str(job.chunk_output_dir / f"chunk_{chunk_index:04d}.ts")
        is_final_chunk = job.current_frame_idx >= job.total_frames

        def encode_chunk():
            chunk_start = time.time()
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
                "creation_time": time.time() - chunk_start,
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
                    job.encoded_chunks[chunk_index] = future.result()
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

    def _finalize_cancelled_jobs(self) -> None:
        for job in list(self.jobs.values()):
            if job.finalized:
                continue
            if not job.cancel_event.is_set():
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
            if job.generation_done and not job.encode_tasks and job.next_append_chunk_index >= job.chunk_index:
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

        print(f"🎛️  [{job.request_id}] HLS scheduler finished with status={status}")

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
    def _memory_bucket(batch_size: int) -> int:
        if batch_size <= 1:
            return 1
        if batch_size <= 2:
            return 2
        if batch_size <= 4:
            return 4
        return 8
