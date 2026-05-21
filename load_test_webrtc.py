"""
MuseTalk WebRTC Concurrent Session Load Tester

Usage:
    python load_test_webrtc.py --base-url http://localhost:8000 \
                               --avatar-id test_avatar \
                               --audio-file ./data/audio/ai-assistant.mpga \
                               --ramp 1,2,3,4,5,6 \
                               --hold-seconds 120

This intentionally mirrors load_test.py's CLI, logging cadence, and report
shape, but drives the WebRTC player path instead of the HLS playlist path.
For WebRTC there are no segments to fetch, so the segment interval fields in
the output represent received live video-frame intervals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import shutil
import signal
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import aiohttp

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
except Exception as exc:  # pragma: no cover - allows --help/py_compile without aiortc.
    AIORTC_IMPORT_ERROR: Optional[Exception] = exc
    RTCConfiguration = None  # type: ignore[assignment]
    RTCIceServer = None  # type: ignore[assignment]
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
else:
    AIORTC_IMPORT_ERROR = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_test_webrtc")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class SessionMetrics:
    session_id: str = ""
    create_latency_s: float = 0.0
    stream_start_latency_s: float = 0.0
    time_to_live_ready_s: float = 0.0
    segment_intervals: list = field(default_factory=list)
    total_segments_fetched: int = 0
    stream_completed: bool = False
    errors: list = field(default_factory=list)

    # WebRTC-specific detail fields. The summary keeps the HLS report shape.
    offer_latency_s: float = 0.0
    connection_latency_s: float = 0.0
    first_video_frame_s: float = 0.0
    first_audio_frame_s: float = 0.0
    total_video_frames_received: int = 0
    total_audio_frames_received: int = 0
    live_video_frames_received: int = 0
    live_audio_frames_received: int = 0
    connection_state: str = ""
    ice_connection_state: str = ""
    ice_transport_policy: str = ""
    final_track_stats: dict = field(default_factory=dict)

    @property
    def avg_segment_interval(self) -> float:
        if not self.segment_intervals:
            return 0.0
        return sum(self.segment_intervals) / len(self.segment_intervals)

    @property
    def max_segment_interval(self) -> float:
        return max(self.segment_intervals) if self.segment_intervals else 0.0


@dataclass
class StageReport:
    concurrency: int = 0
    sessions: list = field(default_factory=list)
    gpu_peak_util: float = 0.0
    gpu_samples: list = field(default_factory=list)
    wall_time_s: float = 0.0

    def summary(self) -> dict:
        completed = [s for s in self.sessions if s.stream_completed]
        failed = [s for s in self.sessions if not s.stream_completed]
        avg_ttlr = (
            sum(s.time_to_live_ready_s for s in completed) / len(completed)
            if completed
            else 0.0
        )
        avg_seg = (
            sum(s.avg_segment_interval for s in completed) / len(completed)
            if completed
            else 0.0
        )
        max_seg = (
            max(s.max_segment_interval for s in completed)
            if completed
            else 0.0
        )
        gpu_util_values = [sample["gpu_util_pct"] for sample in self.gpu_samples]
        gpu_mem_values = [sample["memory_used_mb"] for sample in self.gpu_samples]
        gpu_mem_pct_values = [sample["memory_util_pct"] for sample in self.gpu_samples]
        return {
            "concurrency": self.concurrency,
            "completed": len(completed),
            "failed": len(failed),
            "avg_time_to_live_ready_s": round(avg_ttlr, 3),
            "avg_segment_interval_s": round(avg_seg, 3),
            "max_segment_interval_s": round(max_seg, 3),
            "wall_time_s": round(self.wall_time_s, 1),
            "gpu": {
                "samples": len(self.gpu_samples),
                "avg_util_pct": round(sum(gpu_util_values) / len(gpu_util_values), 2)
                if gpu_util_values else 0.0,
                "peak_util_pct": round(max(gpu_util_values), 2)
                if gpu_util_values else 0.0,
                "avg_memory_used_mb": round(sum(gpu_mem_values) / len(gpu_mem_values), 1)
                if gpu_mem_values else 0.0,
                "peak_memory_used_mb": round(max(gpu_mem_values), 1)
                if gpu_mem_values else 0.0,
                "avg_memory_util_pct": round(sum(gpu_mem_pct_values) / len(gpu_mem_pct_values), 2)
                if gpu_mem_pct_values else 0.0,
                "peak_memory_util_pct": round(max(gpu_mem_pct_values), 2)
                if gpu_mem_pct_values else 0.0,
            },
            "errors": [e for s in self.sessions for e in s.errors],
        }

    def detail(self) -> dict:
        return {
            "summary": self.summary(),
            "metric_note": (
                "For WebRTC, total_segments_fetched is live video frames received "
                "by the WebRTC client, and segment interval fields are live "
                "video-frame receive intervals."
            ),
            "gpu_samples": self.gpu_samples,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "create_latency_s": round(s.create_latency_s, 3),
                    "offer_latency_s": round(s.offer_latency_s, 3),
                    "connection_latency_s": round(s.connection_latency_s, 3),
                    "stream_start_latency_s": round(s.stream_start_latency_s, 3),
                    "time_to_live_ready_s": round(s.time_to_live_ready_s, 3),
                    "total_segments_fetched": s.total_segments_fetched,
                    "total_video_frames_received": s.total_video_frames_received,
                    "total_audio_frames_received": s.total_audio_frames_received,
                    "live_video_frames_received": s.live_video_frames_received,
                    "live_audio_frames_received": s.live_audio_frames_received,
                    "first_video_frame_s": round(s.first_video_frame_s, 3),
                    "first_audio_frame_s": round(s.first_audio_frame_s, 3),
                    "avg_segment_interval_s": round(s.avg_segment_interval, 3),
                    "max_segment_interval_s": round(s.max_segment_interval, 3),
                    "stream_completed": s.stream_completed,
                    "connection_state": s.connection_state,
                    "ice_connection_state": s.ice_connection_state,
                    "ice_transport_policy": s.ice_transport_policy,
                    "final_track_stats": s.final_track_stats,
                    "errors": s.errors,
                }
                for s in self.sessions
            ],
        }


async def collect_gpu_sample(gpu_index: int) -> Optional[dict]:
    if shutil.which("nvidia-smi") is None:
        return None

    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning("GPU sampling failed: %s", stderr.decode("utf-8", errors="ignore").strip())
        return None

    line = stdout.decode("utf-8", errors="ignore").strip().splitlines()
    if not line:
        return None

    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) != 4:
        return None

    gpu_util = float(parts[0])
    mem_util = float(parts[1])
    mem_used = float(parts[2])
    mem_total = float(parts[3])
    return {
        "ts": round(time.time(), 3),
        "gpu_index": gpu_index,
        "gpu_util_pct": gpu_util,
        "memory_util_pct": mem_util,
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
    }


async def gpu_monitor_loop(
    report: StageReport,
    gpu_index: int,
    sample_interval_s: float,
    log_interval_s: float,
    stop_event: asyncio.Event,
) -> None:
    last_log_monotonic = 0.0
    while not stop_event.is_set():
        sample = await collect_gpu_sample(gpu_index)
        if sample is not None:
            report.gpu_samples.append(sample)
            report.gpu_peak_util = max(report.gpu_peak_util, sample["gpu_util_pct"])
            if log_interval_s > 0:
                now = time.monotonic()
                if now - last_log_monotonic >= log_interval_s:
                    logger.info(
                        "GPU[%d] util=%.1f%% mem=%.1f%% used=%.0f/%.0fMB",
                        gpu_index,
                        sample["gpu_util_pct"],
                        sample["memory_util_pct"],
                        sample["memory_used_mb"],
                        sample["memory_total_mb"],
                    )
                    last_log_monotonic = now
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sample_interval_s)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Single-session lifecycle
# ---------------------------------------------------------------------------

async def wait_for_start_or_shutdown(
    start_event: asyncio.Event,
    shutdown_event: asyncio.Event,
) -> bool:
    """Return True if the stage start fired, False if shutdown was requested first."""
    if start_event.is_set():
        return True
    if shutdown_event.is_set():
        return False

    start_task = asyncio.create_task(start_event.wait())
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    done, pending = await asyncio.wait(
        {start_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    for task in pending:
        with suppress(asyncio.CancelledError):
            await task

    return start_task in done and start_event.is_set()


async def sleep_or_shutdown(delay_s: float, shutdown_event: asyncio.Event) -> bool:
    """Sleep for delay_s unless shutdown is requested first."""
    if delay_s <= 0:
        return True
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=delay_s)
        return False
    except asyncio.TimeoutError:
        return True


async def delete_webrtc_session(
    http: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
) -> None:
    async with http.delete(f"{base_url}/webrtc/sessions/{session_id}") as resp:
        await resp.text()


def build_rtc_configuration_from_payload(ice_servers: list[dict]):
    if RTCConfiguration is None or RTCIceServer is None:
        return None

    rtc_ice_servers = []
    for entry in ice_servers or []:
        urls = entry.get("urls")
        if not urls:
            continue
        rtc_ice_servers.append(
            RTCIceServer(
                urls=urls,
                username=entry.get("username"),
                credential=entry.get("credential"),
            )
        )
    return RTCConfiguration(iceServers=rtc_ice_servers)


async def wait_for_ice_gathering(pc, timeout_s: float) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change():
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "ICE gathering did not reach complete within %.1fs; sending current SDP",
            timeout_s,
        )


async def exchange_offer(
    http: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
    pc,
    metrics: SessionMetrics,
    ice_gather_timeout_s: float,
) -> None:
    t_offer = time.monotonic()
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await wait_for_ice_gathering(pc, ice_gather_timeout_s)

    async with http.post(
        f"{base_url}/webrtc/sessions/{session_id}/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        },
    ) as resp:
        metrics.offer_latency_s = time.monotonic() - t_offer
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"offer failed: {resp.status} {body[:200]}")
        answer = await resp.json()

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )


async def wait_for_peer_connection(
    pc,
    connected_event: asyncio.Event,
    failed_event: asyncio.Event,
    timeout_s: float,
) -> bool:
    if pc.connectionState == "connected":
        return True
    if pc.connectionState in {"failed", "closed"}:
        return False

    connected_task = asyncio.create_task(connected_event.wait())
    failed_task = asyncio.create_task(failed_event.wait())
    done, pending = await asyncio.wait(
        {connected_task, failed_task},
        timeout=timeout_s,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    for task in pending:
        with suppress(asyncio.CancelledError):
            await task

    return connected_task in done and connected_event.is_set()


def get_track_stats(status: dict) -> tuple[dict, dict, dict]:
    track_stats = status.get("track_stats") or {}
    video_stats = track_stats.get("video") or {}
    audio_stats = track_stats.get("audio") or {}
    sync_stats = track_stats.get("sync_clock") or video_stats.get("sync_clock") or {}
    return video_stats, audio_stats, sync_stats


def is_live_ready(status: dict) -> bool:
    video_stats, _audio_stats, sync_stats = get_track_stats(status)
    return bool(
        status.get("active_stream")
        and (
            sync_stats.get("playout_released")
            or video_stats.get("live_released")
            or video_stats.get("prebuffer_ready")
            or video_stats.get("frames_played", 0) > 0
        )
    )


def is_stream_complete(status: dict) -> bool:
    video_stats, _audio_stats, sync_stats = get_track_stats(status)
    active_stream = status.get("active_stream")
    return bool(
        not active_stream
        and not video_stats.get("live_active", False)
        and (
            sync_stats.get("active") is False
            or video_stats.get("generation_complete")
            or status.get("status") in {"connected", "completed", "idle", "ready"}
        )
    )


async def consume_track(
    track,
    metrics: SessionMetrics,
    started_at: float,
    live_event: asyncio.Event,
    stop_event: asyncio.Event,
) -> None:
    kind = getattr(track, "kind", "unknown")
    last_live_video_frame_time: Optional[float] = None
    try:
        while not stop_event.is_set():
            await track.recv()
            now = time.monotonic()

            if kind == "video":
                metrics.total_video_frames_received += 1
                if metrics.first_video_frame_s == 0.0:
                    metrics.first_video_frame_s = now - started_at
                if live_event.is_set():
                    metrics.live_video_frames_received += 1
                    metrics.total_segments_fetched = metrics.live_video_frames_received
                    if last_live_video_frame_time is not None:
                        metrics.segment_intervals.append(now - last_live_video_frame_time)
                    last_live_video_frame_time = now
            elif kind == "audio":
                metrics.total_audio_frames_received += 1
                if metrics.first_audio_frame_s == 0.0:
                    metrics.first_audio_frame_s = now - started_at
                if live_event.is_set():
                    metrics.live_audio_frames_received += 1
    except Exception as exc:
        if not stop_event.is_set():
            metrics.errors.append(f"{kind} track receive failed: {exc}")


async def run_session(
    http: aiohttp.ClientSession,
    base_url: str,
    avatar_id: str,
    audio_path: Path,
    metrics: SessionMetrics,
    segment_duration: float,
    playback_fps: int,
    musetalk_fps: int,
    batch_size: int,
    start_event: asyncio.Event,
    shutdown_event: asyncio.Event,
    ready_queue: Optional[asyncio.Queue] = None,
    user_id: Optional[str] = None,
    create_delay_s: float = 0.0,
    ice_gather_timeout_s: float = 10.0,
    connection_timeout_s: float = 30.0,
    completion_timeout_s: float = 300.0,
):
    """Full lifecycle: create -> connect WebRTC -> wait for go signal -> stream -> poll -> cleanup."""
    if shutdown_event.is_set():
        metrics.errors.append("aborted before session creation")
        return

    if not await sleep_or_shutdown(create_delay_s, shutdown_event):
        metrics.errors.append("aborted before staggered session create")
        return

    pc = None
    consumer_tasks: list[asyncio.Task] = []
    track_stop_event = asyncio.Event()
    live_event = asyncio.Event()
    started_at = time.monotonic()

    # --- 1) Create session ---------------------------------------------------
    t0 = time.monotonic()
    chunk_duration = max(1, int(round(segment_duration)))
    create_params = {
        "avatar_id": avatar_id,
        "fps": musetalk_fps,
        "playback_fps": playback_fps,
        "batch_size": batch_size,
        "chunk_duration": chunk_duration,
    }
    if user_id:
        create_params["user_id"] = user_id
    create_url = f"{base_url}/webrtc/sessions/create?{urlencode(create_params)}"

    try:
        async with http.post(create_url) as resp:
            if resp.status != 200:
                body = await resp.text()
                metrics.errors.append(f"create failed: {resp.status} {body[:200]}")
                return
            data = await resp.json()
            metrics.session_id = data["session_id"]
            metrics.create_latency_s = time.monotonic() - t0
            metrics.ice_transport_policy = data.get("ice_transport_policy", "")
            logger.info(
                "Session %s created in %.2fs",
                metrics.session_id,
                metrics.create_latency_s,
            )
    except Exception as exc:
        metrics.errors.append(f"create exception: {exc}")
        return

    sid = metrics.session_id

    try:
        # --- 2) Connect the WebRTC client, matching the browser player --------
        assert RTCPeerConnection is not None
        rtc_config = build_rtc_configuration_from_payload(data.get("ice_servers") or [])
        pc = RTCPeerConnection(configuration=rtc_config)

        connected_event = asyncio.Event()
        failed_event = asyncio.Event()

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            metrics.connection_state = pc.connectionState
            if pc.connectionState == "connected":
                connected_event.set()
            elif pc.connectionState in {"failed", "closed"}:
                failed_event.set()

        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            metrics.ice_connection_state = pc.iceConnectionState
            if pc.iceConnectionState in {"connected", "completed"}:
                connected_event.set()
            elif pc.iceConnectionState in {"failed", "closed"}:
                failed_event.set()

        @pc.on("track")
        def on_track(track):
            logger.info("Session %s received %s track", sid, track.kind)
            consumer_tasks.append(
                asyncio.create_task(
                    consume_track(
                        track=track,
                        metrics=metrics,
                        started_at=started_at,
                        live_event=live_event,
                        stop_event=track_stop_event,
                    )
                )
            )

        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")

        await exchange_offer(
            http=http,
            base_url=base_url,
            session_id=sid,
            pc=pc,
            metrics=metrics,
            ice_gather_timeout_s=ice_gather_timeout_s,
        )

        t_connect = time.monotonic()
        connected = await wait_for_peer_connection(
            pc=pc,
            connected_event=connected_event,
            failed_event=failed_event,
            timeout_s=connection_timeout_s,
        )
        metrics.connection_latency_s = time.monotonic() - t_connect
        if not connected:
            metrics.errors.append(
                f"webrtc connect failed/timeout: pc={pc.connectionState} ice={pc.iceConnectionState}"
            )
            return

        logger.info(
            "Session %s WebRTC connected in %.2fs",
            sid,
            metrics.connection_latency_s,
        )
        if ready_queue is not None:
            await ready_queue.put(sid)

        # --- 3) Wait for coordinated start -----------------------------------
        if not await wait_for_start_or_shutdown(start_event, shutdown_event):
            metrics.errors.append("aborted before stream start")
            return

        # --- 4) Upload audio to start stream ---------------------------------
        t_stream = time.monotonic()
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        with audio_path.open("rb") as audio_fp:
            form = aiohttp.FormData()
            form.add_field(
                "audio_file",
                audio_fp,
                filename=audio_path.name,
                content_type=content_type,
            )
            async with http.post(
                f"{base_url}/webrtc/sessions/{sid}/stream", data=form
            ) as resp:
                metrics.stream_start_latency_s = time.monotonic() - t_stream
                if resp.status != 200:
                    body = await resp.text()
                    metrics.errors.append(
                        f"stream failed: {resp.status} {body[:200]}"
                    )
                    return
                logger.info(
                    "Session %s stream accepted in %.2fs",
                    sid,
                    metrics.stream_start_latency_s,
                )

        # --- 5) Poll status until live, then until complete -------------------
        live_ready = False
        t_stream_start = time.monotonic()

        while True:
            if shutdown_event.is_set():
                metrics.errors.append("aborted during stream polling")
                logger.warning("Session %s interrupted by shutdown request", sid)
                break

            await asyncio.sleep(0.5)
            try:
                async with http.get(
                    f"{base_url}/webrtc/sessions/{sid}/status"
                ) as resp:
                    if resp.status != 200:
                        continue
                    status = await resp.json()
            except Exception:
                continue

            session_status = status.get("status", "")
            video_stats, _audio_stats, _sync_stats = get_track_stats(status)
            metrics.final_track_stats = status.get("track_stats") or {}

            if session_status in {"error", "failed", "cancelled", "rejected"}:
                metrics.errors.append(f"terminal status: {session_status}")
                logger.warning("Session %s ended with terminal status %s", sid, session_status)
                break

            if not live_ready and is_live_ready(status):
                live_ready = True
                live_event.set()
                metrics.time_to_live_ready_s = time.monotonic() - t_stream_start
                logger.info(
                    "Session %s live_ready in %.2fs",
                    sid,
                    metrics.time_to_live_ready_s,
                )

            if live_ready and is_stream_complete(status):
                metrics.stream_completed = True
                metrics.total_segments_fetched = max(
                    metrics.total_segments_fetched,
                    int(video_stats.get("frames_played") or 0),
                )
                logger.info(
                    "Session %s completed. Live frames received: %d, avg frame interval: %.3fs",
                    sid,
                    metrics.total_segments_fetched,
                    metrics.avg_segment_interval,
                )
                break

            if time.monotonic() - t_stream_start > completion_timeout_s:
                metrics.errors.append(
                    f"timeout: stream did not complete in {completion_timeout_s:.0f}s"
                )
                logger.warning("Session %s timed out", sid)
                break

    finally:
        # --- 6) Cleanup -------------------------------------------------------
        track_stop_event.set()
        if sid:
            try:
                cleanup_task = asyncio.create_task(delete_webrtc_session(http, base_url, sid))
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    await cleanup_task
                    raise
            except Exception:
                pass

        if pc is not None:
            with suppress(Exception):
                await pc.close()

        for task in consumer_tasks:
            task.cancel()
        for task in consumer_tasks:
            with suppress(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

async def run_stage(
    base_url: str,
    avatar_id: str,
    audio_path: Path,
    concurrency: int,
    segment_duration: float,
    playback_fps: int,
    musetalk_fps: int,
    batch_size: int,
    shutdown_event: asyncio.Event,
    stagger_seconds: float = 0.0,
    gpu_index: int = 0,
    gpu_sample_interval_s: float = 1.0,
    gpu_log_interval_s: float = 5.0,
    ice_gather_timeout_s: float = 10.0,
    connection_timeout_s: float = 30.0,
    completion_timeout_s: float = 300.0,
    stage_ready_timeout_s: float = 60.0,
) -> StageReport:
    """Run `concurrency` WebRTC sessions in parallel, return aggregated report."""

    report = StageReport(concurrency=concurrency)
    start_event = asyncio.Event()
    ready_queue: asyncio.Queue = asyncio.Queue()
    gpu_stop_event = asyncio.Event()
    gpu_monitor_task = None

    timeout = aiohttp.ClientTimeout(total=max(600, int(completion_timeout_s + 120)))
    async with aiohttp.ClientSession(timeout=timeout) as http:
        if gpu_sample_interval_s > 0 and shutil.which("nvidia-smi") is not None:
            gpu_monitor_task = asyncio.create_task(
                gpu_monitor_loop(
                    report=report,
                    gpu_index=gpu_index,
                    sample_interval_s=gpu_sample_interval_s,
                    log_interval_s=gpu_log_interval_s,
                    stop_event=gpu_stop_event,
                )
            )
        elif gpu_sample_interval_s > 0:
            logger.warning("nvidia-smi not found; GPU metrics disabled for this stage")

        metrics_list = [SessionMetrics() for _ in range(concurrency)]
        stage_tag = int(time.time())
        tasks = [
            asyncio.create_task(
                run_session(
                    http=http,
                    base_url=base_url,
                    avatar_id=avatar_id,
                    audio_path=audio_path,
                    metrics=m,
                    segment_duration=segment_duration,
                    playback_fps=playback_fps,
                    musetalk_fps=musetalk_fps,
                    batch_size=batch_size,
                    start_event=start_event,
                    shutdown_event=shutdown_event,
                    ready_queue=ready_queue,
                    user_id=f"load-test-webrtc-{stage_tag}-{index + 1}",
                    create_delay_s=index * stagger_seconds,
                    ice_gather_timeout_s=ice_gather_timeout_s,
                    connection_timeout_s=connection_timeout_s,
                    completion_timeout_s=completion_timeout_s,
                )
            )
            for index, m in enumerate(metrics_list)
        ]

        if stagger_seconds > 0:
            logger.info(
                "=== STAGE %d: Staggering %d WebRTC streams every %.2fs ===",
                concurrency,
                concurrency,
                stagger_seconds,
            )
            start_event.set()
        else:
            logger.info(
                "=== STAGE %d: Waiting for %d WebRTC peer connection(s) ===",
                concurrency,
                concurrency,
            )
            ready_sessions = set()
            deadline = time.monotonic() + stage_ready_timeout_s
            while len(ready_sessions) < concurrency:
                if shutdown_event.is_set():
                    break

                done_count = sum(1 for task in tasks if task.done())
                if len(ready_sessions) + done_count >= concurrency:
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                try:
                    sid = await asyncio.wait_for(
                        ready_queue.get(),
                        timeout=min(0.5, remaining),
                    )
                    ready_sessions.add(sid)
                except asyncio.TimeoutError:
                    pass

            if shutdown_event.is_set():
                logger.warning("Shutdown requested before stage %d started", concurrency)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                report.sessions = metrics_list
                return report

            if len(ready_sessions) < concurrency:
                logger.warning(
                    "Only %d/%d WebRTC peer connection(s) ready before start",
                    len(ready_sessions),
                    concurrency,
                )

            logger.info(
                "=== STAGE %d: Firing %d WebRTC streams simultaneously ===",
                concurrency,
                concurrency,
            )
            start_event.set()

        t_stage = time.monotonic()

        shutdown_task = asyncio.create_task(shutdown_event.wait())
        try:
            while True:
                pending_tasks = [task for task in tasks if not task.done()]
                if not pending_tasks:
                    break

                done, _ = await asyncio.wait(
                    set(pending_tasks) | {shutdown_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if shutdown_task in done:
                    logger.warning(
                        "Shutdown requested; cancelling %d active session task(s)",
                        len(pending_tasks),
                    )
                    for task in pending_tasks:
                        task.cancel()
                    break

            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            shutdown_task.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_task
            gpu_stop_event.set()
            if gpu_monitor_task is not None:
                with suppress(asyncio.CancelledError):
                    await gpu_monitor_task

        report.wall_time_s = time.monotonic() - t_stage
        report.sessions = metrics_list

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace):
    if AIORTC_IMPORT_ERROR is not None:
        logger.error("aiortc is required for WebRTC load testing: %s", AIORTC_IMPORT_ERROR)
        return

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        return

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown(signame: str):
        if shutdown_event.is_set():
            logger.warning("Received %s again; waiting for cancellation to finish", signame)
            return
        logger.warning("Received %s; stopping load test after cleanup", signame)
        shutdown_event.set()

    registered_signals = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown, sig.name)
            registered_signals.append(sig)
        except NotImplementedError:
            pass

    if args.concurrency is not None:
        ramp_levels = [args.concurrency]
    else:
        ramp_levels = [int(x) for x in args.ramp.split(",")]
    results = []
    detailed_results = []

    for level in ramp_levels:
        if shutdown_event.is_set():
            logger.warning("Shutdown requested; skipping remaining stages")
            break

        logger.info("\n%s", "=" * 60)
        logger.info("Starting WebRTC stage: concurrency=%d", level)
        logger.info("%s\n", "=" * 60)

        report = await run_stage(
            base_url=args.base_url,
            avatar_id=args.avatar_id,
            audio_path=audio_path,
            concurrency=level,
            segment_duration=args.segment_duration,
            playback_fps=args.playback_fps,
            musetalk_fps=args.musetalk_fps,
            batch_size=args.batch_size,
            shutdown_event=shutdown_event,
            stagger_seconds=args.stagger_seconds,
            gpu_index=args.gpu_index,
            gpu_sample_interval_s=args.gpu_sample_interval,
            gpu_log_interval_s=args.gpu_log_interval,
            ice_gather_timeout_s=args.ice_gather_timeout,
            connection_timeout_s=args.connection_timeout,
            completion_timeout_s=args.completion_timeout,
            stage_ready_timeout_s=args.stage_ready_timeout,
        )
        summary = report.summary()
        results.append(summary)
        detailed_results.append(report.detail())

        logger.info("\n--- WebRTC Stage %d Results ---", level)
        logger.info(json.dumps(summary, indent=2))

        target_frame_interval = 1.0 / max(1, args.playback_fps)
        if summary["max_segment_interval_s"] > 2 * target_frame_interval:
            logger.warning(
                "FRAME THROTTLING DETECTED at concurrency=%d "
                "(max frame interval %.3fs > %.3fs threshold)",
                level,
                summary["max_segment_interval_s"],
                2 * target_frame_interval,
            )

        if summary["failed"] > 0:
            logger.warning(
                "%d/%d sessions FAILED at concurrency=%d",
                summary["failed"],
                level,
                level,
            )

        if level != ramp_levels[-1] and not shutdown_event.is_set():
            logger.info("Cooling down for %ds before next stage...", args.hold_seconds)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=args.hold_seconds)
            except asyncio.TimeoutError:
                pass

    for sig in registered_signals:
        loop.remove_signal_handler(sig)

    logger.info("\n%s", "=" * 60)
    logger.info("FINAL WEBRTC LOAD TEST REPORT")
    logger.info("%s", "=" * 60)
    for r in results:
        logger.info(json.dumps(r, indent=2))

    report_path = Path(args.report_path)
    report_path.write_text(json.dumps(results, indent=2))
    logger.info("Report written to %s", report_path)

    detail_path = Path(args.detail_report_path)
    detail_path.write_text(json.dumps(detailed_results, indent=2))
    logger.info("Detailed report written to %s", detail_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuseTalk WebRTC Load Tester")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--avatar-id", default="test_avatar")
    p.add_argument("--audio-file", default="./data/audio/ai-assistant.mpga")
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Single-stage shortcut. If set, overrides --ramp.",
    )
    p.add_argument(
        "--ramp",
        default="1,2,3,4,5",
        help="Comma-separated concurrency levels to test",
    )
    p.add_argument("--hold-seconds", type=int, default=30, help="Cool-down between stages")
    p.add_argument(
        "--segment-duration",
        type=float,
        default=1.0,
        help="Mapped to WebRTC chunk_duration; kept for load_test.py parity.",
    )
    p.add_argument("--playback-fps", type=int, default=30)
    p.add_argument(
        "--musetalk-fps",
        type=int,
        default=15,
        help="Mapped to WebRTC generation fps.",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help="Delay between session arrivals. 0 preserves simultaneous burst behavior.",
    )
    p.add_argument("--gpu-index", type=int, default=0, help="GPU index to sample with nvidia-smi")
    p.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=1.0,
        help="Seconds between GPU samples. Set 0 to disable GPU sampling.",
    )
    p.add_argument(
        "--gpu-log-interval",
        type=float,
        default=5.0,
        help="Seconds between GPU log lines. Set 0 to disable periodic GPU logs.",
    )
    p.add_argument(
        "--ice-gather-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for local ICE gathering before posting offer.",
    )
    p.add_argument(
        "--connection-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for each WebRTC peer connection.",
    )
    p.add_argument(
        "--stage-ready-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for all peer connections before simultaneous start.",
    )
    p.add_argument(
        "--completion-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for each stream to finish after audio upload.",
    )
    p.add_argument(
        "--report-path",
        default="load_test_webrtc_report.json",
        help="Path for the summary report JSON.",
    )
    p.add_argument(
        "--detail-report-path",
        default="load_test_webrtc_report_detailed.json",
        help="Path for the detailed report JSON.",
    )
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
