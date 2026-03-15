"""
MuseTalk HLS Concurrent Session Load Tester

Usage:
    python load_test.py --base-url http://localhost:8000 \
                        --avatar-id test_avatar \
                        --audio-file ./data/audio/ai-assistant.mpga \
                        --ramp 1,2,3,4,5,6 \
                        --hold-seconds 120
"""

import argparse
import asyncio
import json
import logging
import shutil
import signal
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_test")


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
            "gpu_samples": self.gpu_samples,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "create_latency_s": round(s.create_latency_s, 3),
                    "stream_start_latency_s": round(s.stream_start_latency_s, 3),
                    "time_to_live_ready_s": round(s.time_to_live_ready_s, 3),
                    "total_segments_fetched": s.total_segments_fetched,
                    "avg_segment_interval_s": round(s.avg_segment_interval, 3),
                    "max_segment_interval_s": round(s.max_segment_interval, 3),
                    "stream_completed": s.stream_completed,
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


async def delete_hls_session(
    http: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
) -> None:
    async with http.delete(f"{base_url}/hls/sessions/{session_id}"):
        return

def build_segment_url(base_url: str, session_id: str, segment_ref: str) -> str:
    """
    Build a fetch URL from a manifest entry.

    The live playlist currently emits entries like:
      segments/{request_id}/chunk_0001.ts
    """
    segment_ref = segment_ref.lstrip("/")
    if segment_ref.startswith("segments/"):
        return f"{base_url}/hls/sessions/{session_id}/{segment_ref}"
    return f"{base_url}/hls/sessions/{session_id}/segments/{segment_ref}"


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
    create_delay_s: float = 0.0,
):
    """Full lifecycle: create → wait for go signal → stream → poll → fetch segments → cleanup."""
    if shutdown_event.is_set():
        metrics.errors.append("aborted before session creation")
        return

    if not await sleep_or_shutdown(create_delay_s, shutdown_event):
        metrics.errors.append("aborted before staggered session create")
        return

    # --- 1) Create session ---------------------------------------------------
    t0 = time.monotonic()
    create_url = (
        f"{base_url}/hls/sessions/create"
        f"?avatar_id={avatar_id}"
        f"&playback_fps={playback_fps}"
        f"&musetalk_fps={musetalk_fps}"
        f"&batch_size={batch_size}"
        f"&segment_duration={segment_duration}"
        f"&hls_server_timing=true"
    )
    try:
        async with http.post(create_url) as resp:
            if resp.status != 200:
                metrics.errors.append(f"create failed: {resp.status}")
                return
            data = await resp.json()
            metrics.session_id = data["session_id"]
            metrics.create_latency_s = time.monotonic() - t0
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
        # --- 2) Wait for coordinated start -----------------------------------
        if not await wait_for_start_or_shutdown(start_event, shutdown_event):
            metrics.errors.append("aborted before stream start")
            return

        # --- 3) Upload audio to start stream ---------------------------------
        t_stream = time.monotonic()
        with audio_path.open("rb") as audio_fp:
            form = aiohttp.FormData()
            form.add_field(
                "audio_file",
                audio_fp,
                filename=audio_path.name,
                content_type="audio/mpeg",
            )
            async with http.post(
                f"{base_url}/hls/sessions/{sid}/stream", data=form
            ) as resp:
                metrics.stream_start_latency_s = time.monotonic() - t_stream
                if resp.status != 200:
                    body = await resp.text()
                    metrics.errors.append(
                        f"stream failed: {resp.status} {body[:200]}"
                    )
                    return
                logger.info(
                    "Session %s stream started in %.2fs",
                    sid,
                    metrics.stream_start_latency_s,
                )

        # --- 4) Poll status until live_ready, then until complete -------------
        live_ready = False
        t_stream_start = time.monotonic()
        last_segment_time: Optional[float] = None
        seen_segments: set = set()

        while True:
            if shutdown_event.is_set():
                metrics.errors.append("aborted during stream polling")
                logger.warning("Session %s interrupted by shutdown request", sid)
                break

            await asyncio.sleep(0.5)
            try:
                async with http.get(
                    f"{base_url}/hls/sessions/{sid}/status"
                ) as resp:
                    if resp.status != 200:
                        continue
                    status = await resp.json()
            except Exception:
                continue

            session_status = status.get("status", "")
            if session_status in {"error", "failed", "cancelled", "rejected"}:
                metrics.errors.append(f"terminal status: {session_status}")
                logger.warning("Session %s ended with terminal status %s", sid, session_status)
                break

            # Detect live_ready transition
            if not live_ready and status.get("live_ready"):
                live_ready = True
                metrics.time_to_live_ready_s = (
                    time.monotonic() - t_stream_start
                )
                logger.info(
                    "Session %s live_ready in %.2fs",
                    sid,
                    metrics.time_to_live_ready_s,
                )

            # Simulate segment fetching if live
            if live_ready and session_status == "streaming":
                try:
                    async with http.get(
                        f"{base_url}/hls/sessions/{sid}/live.m3u8"
                    ) as resp:
                        if resp.status == 200:
                            playlist = await resp.text()
                            for line in playlist.splitlines():
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    seg_name = line.split("?")[0]
                                    if seg_name not in seen_segments:
                                        seen_segments.add(seg_name)
                                        # Fetch the segment
                                        seg_url = build_segment_url(base_url, sid, seg_name)
                                        async with http.get(seg_url) as seg_resp:
                                            if seg_resp.status == 200:
                                                await seg_resp.read()
                                                now = time.monotonic()
                                                if last_segment_time is not None:
                                                    interval = now - last_segment_time
                                                    metrics.segment_intervals.append(interval)
                                                last_segment_time = now
                                                metrics.total_segments_fetched += 1
                except Exception:
                    pass

            # Check for completion
            if session_status in ("idle", "ready") and live_ready:
                metrics.stream_completed = True
                logger.info(
                    "Session %s completed. Segments fetched: %d, avg interval: %.2fs",
                    sid,
                    metrics.total_segments_fetched,
                    metrics.avg_segment_interval,
                )
                break

            # Timeout guard (5 minutes)
            if time.monotonic() - t_stream_start > 300:
                metrics.errors.append("timeout: stream did not complete in 300s")
                logger.warning("Session %s timed out", sid)
                break

    finally:
        # --- 5) Cleanup -------------------------------------------------------
        try:
            cleanup_task = asyncio.create_task(delete_hls_session(http, base_url, sid))
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
                raise
        except Exception:
            pass


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
) -> StageReport:
    """Run `concurrency` sessions in parallel, return aggregated report."""

    report = StageReport(concurrency=concurrency)
    start_event = asyncio.Event()
    gpu_stop_event = asyncio.Event()
    gpu_monitor_task = None

    timeout = aiohttp.ClientTimeout(total=600)
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
                    create_delay_s=index * stagger_seconds,
                )
            )
            for index, m in enumerate(metrics_list)
        ]

        if stagger_seconds > 0:
            logger.info(
                "=== STAGE %d: Staggering %d streams every %.2fs ===",
                concurrency,
                concurrency,
                stagger_seconds,
            )
            start_event.set()
        else:
            # Give all sessions time to create, then fire simultaneously.
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=2.0)
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

            logger.info(
                "=== STAGE %d: Firing %d streams simultaneously ===",
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
            # add_signal_handler is not available on some platforms.
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
        logger.info("Starting stage: concurrency=%d", level)
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
        )
        summary = report.summary()
        results.append(summary)
        detailed_results.append(report.detail())

        logger.info("\n--- Stage %d Results ---", level)
        logger.info(json.dumps(summary, indent=2))

        # Throttle detection: if max segment interval > 2× segment_duration
        if summary["max_segment_interval_s"] > 2 * args.segment_duration:
            logger.warning(
                "⚠️  THROTTLING DETECTED at concurrency=%d "
                "(max segment interval %.2fs > %.2fs threshold)",
                level,
                summary["max_segment_interval_s"],
                2 * args.segment_duration,
            )

        if summary["failed"] > 0:
            logger.warning(
                "⚠️  %d/%d sessions FAILED at concurrency=%d",
                summary["failed"],
                level,
                level,
            )

        # Cool-down between stages
        if level != ramp_levels[-1] and not shutdown_event.is_set():
            logger.info("Cooling down for %ds before next stage...", args.hold_seconds)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=args.hold_seconds)
            except asyncio.TimeoutError:
                pass

    for sig in registered_signals:
        loop.remove_signal_handler(sig)

    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("FINAL LOAD TEST REPORT")
    logger.info("=" * 60)
    for r in results:
        logger.info(json.dumps(r, indent=2))

    # Write to file
    report_path = Path("load_test_report.json")
    report_path.write_text(json.dumps(results, indent=2))
    logger.info("Report written to %s", report_path)
    detail_path = Path("load_test_report_detailed.json")
    detail_path.write_text(json.dumps(detailed_results, indent=2))
    logger.info("Detailed report written to %s", detail_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuseTalk HLS Load Tester")
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
    p.add_argument("--segment-duration", type=float, default=1.0)
    p.add_argument("--playback-fps", type=int, default=30)
    p.add_argument("--musetalk-fps", type=int, default=15)
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
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
