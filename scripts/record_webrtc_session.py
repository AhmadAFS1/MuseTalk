#!/usr/bin/env python3
import argparse
import asyncio
import mimetypes
import time
import sys
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_test_webrtc import (
    SessionMetrics,
    build_rtc_configuration_from_payload,
    delete_webrtc_session,
    exchange_offer,
    get_track_stats,
    is_live_ready,
    is_stream_complete,
    wait_for_peer_connection,
)

try:
    from aiortc import RTCPeerConnection
except Exception as exc:  # pragma: no cover
    raise RuntimeError("aiortc is required to record WebRTC sessions") from exc


async def consume_video_track(
    track,
    record_event: asyncio.Event,
    stop_event: asyncio.Event,
    output_path: Path,
    fps: float,
    counters: dict,
) -> None:
    writer = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        while not stop_event.is_set():
            frame = await track.recv()
            if not record_event.is_set():
                continue
            image = frame.to_ndarray(format="bgr24")
            if writer is None:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
            writer.write(image)
            counters["video_frames_written"] = counters.get("video_frames_written", 0) + 1
    except Exception:
        if not stop_event.is_set():
            raise
    finally:
        if writer is not None:
            writer.release()


async def consume_audio_track(track, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await track.recv()
    except Exception:
        pass


async def record_once(args: argparse.Namespace) -> dict:
    timeout = aiohttp.ClientTimeout(total=max(300, args.completion_timeout + 60))
    metrics = SessionMetrics()
    pc = None
    sid = ""
    live_event = asyncio.Event()
    record_event = asyncio.Event()
    stop_event = asyncio.Event()
    counters = {"video_frames_written": 0}
    consumer_tasks: list[asyncio.Task] = []
    started_at = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout) as http:
        create_params = {
            "avatar_id": args.avatar_id,
            "fps": args.musetalk_fps,
            "playback_fps": args.playback_fps,
            "batch_size": args.batch_size,
            "chunk_duration": max(1, int(round(args.segment_duration))),
        }
        async with http.post(
            f"{args.base_url}/webrtc/sessions/create?{urlencode(create_params)}"
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"create failed: {resp.status} {body[:300]}")
            data = await resp.json()
            sid = data["session_id"]
            metrics.session_id = sid

        try:
            pc = RTCPeerConnection(
                configuration=build_rtc_configuration_from_payload(data.get("ice_servers") or [])
            )
            connected_event = asyncio.Event()
            failed_event = asyncio.Event()

            @pc.on("connectionstatechange")
            async def on_connection_state_change():
                if pc.connectionState == "connected":
                    connected_event.set()
                elif pc.connectionState in {"failed", "closed"}:
                    failed_event.set()

            @pc.on("iceconnectionstatechange")
            async def on_ice_connection_state_change():
                if pc.iceConnectionState in {"connected", "completed"}:
                    connected_event.set()
                elif pc.iceConnectionState in {"failed", "closed"}:
                    failed_event.set()

            @pc.on("track")
            def on_track(track):
                if track.kind == "video":
                    consumer_tasks.append(
                        asyncio.create_task(
                            consume_video_track(
                                track=track,
                                record_event=record_event,
                                stop_event=stop_event,
                                output_path=Path(args.output),
                                fps=float(args.playback_fps),
                                counters=counters,
                            )
                        )
                    )
                elif track.kind == "audio":
                    consumer_tasks.append(
                        asyncio.create_task(consume_audio_track(track, stop_event))
                    )

            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")
            await exchange_offer(
                http=http,
                base_url=args.base_url,
                session_id=sid,
                pc=pc,
                metrics=metrics,
                ice_gather_timeout_s=args.ice_gather_timeout,
            )
            connected = await wait_for_peer_connection(
                pc=pc,
                connected_event=connected_event,
                failed_event=failed_event,
                timeout_s=args.connection_timeout,
            )
            if not connected:
                raise RuntimeError(
                    f"WebRTC connection failed: pc={pc.connectionState} ice={pc.iceConnectionState}"
                )

            content_type = mimetypes.guess_type(args.audio_file.name)[0] or "application/octet-stream"
            with args.audio_file.open("rb") as audio_fp:
                form = aiohttp.FormData()
                form.add_field(
                    "audio_file",
                    audio_fp,
                    filename=args.audio_file.name,
                    content_type=content_type,
                )
                async with http.post(
                    f"{args.base_url}/webrtc/sessions/{sid}/stream",
                    data=form,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"stream failed: {resp.status} {body[:300]}")
            record_event.set()

            live_ready = False
            stream_started_at = time.monotonic()
            final_status = {}
            while time.monotonic() - stream_started_at < args.completion_timeout:
                await asyncio.sleep(0.5)
                async with http.get(f"{args.base_url}/webrtc/sessions/{sid}/status") as resp:
                    if resp.status != 200:
                        continue
                    status = await resp.json()
                final_status = status
                if not live_ready and is_live_ready(status):
                    live_ready = True
                    live_event.set()
                if live_ready and is_stream_complete(status):
                    break

            stop_event.set()
            for task in consumer_tasks:
                task.cancel()
            for task in consumer_tasks:
                with suppress(asyncio.CancelledError):
                    await task

            video_stats, audio_stats, sync_stats = get_track_stats(final_status)
            return {
                "session_id": sid,
                "output": str(Path(args.output).resolve()),
                "frames_written": int(counters.get("video_frames_written", 0)),
                "elapsed_s": round(time.monotonic() - started_at, 3),
                "video_stats": video_stats,
                "audio_stats": audio_stats,
                "sync_stats": sync_stats,
            }
        finally:
            stop_event.set()
            for task in consumer_tasks:
                task.cancel()
            if pc is not None:
                with suppress(Exception):
                    await pc.close()
            if sid:
                with suppress(Exception):
                    await delete_webrtc_session(http, args.base_url, sid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one MuseTalk WebRTC session to MP4.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--avatar-id", required=True)
    parser.add_argument("--audio-file", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--playback-fps", type=int, default=20)
    parser.add_argument("--musetalk-fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ice-gather-timeout", type=float, default=10.0)
    parser.add_argument("--connection-timeout", type=float, default=60.0)
    parser.add_argument("--completion-timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(record_once(args))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
