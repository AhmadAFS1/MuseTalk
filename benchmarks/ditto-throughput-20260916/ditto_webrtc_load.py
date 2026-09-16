#!/usr/bin/env python3
"""Loopback WebRTC load test for Ditto's TensorRT online pipeline."""

import argparse
import asyncio
import json
import queue
import statistics
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np
import torch

# Reuse aiortc/av already installed for MuseTalk without shadowing Ditto's TRT 8.6.
sys.path.append("/workspace/.venvs/musetalk_trt_stagewise/lib/python3.10/site-packages")
from aiortc import MediaStreamTrack, RTCPeerConnection
from av import VideoFrame

sys.path.insert(0, "/workspace/ditto-talkinghead")
import stream_pipeline_online as pipeline


SINKS = {}


class QueueWriter:
    """Replace MP4 writing with a bounded live-frame queue."""

    def __init__(self, output_path, **_kwargs):
        self.queue = SINKS[output_path]

    def __call__(self, image, fmt="bgr"):
        rgb = image[..., ::-1] if fmt == "bgr" else image
        self.queue.put((time.time(), np.ascontiguousarray(rgb)))

    def close(self):
        self.queue.put(None)


pipeline.VideoWriterByImageIO = QueueWriter


class QueueVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, frame_queue, fps=25, frame_limit=250):
        super().__init__()
        self.frame_queue = frame_queue
        self.fps = fps
        self.frame_limit = frame_limit
        self.index = 0
        self.started = None

    async def recv(self):
        if self.index >= self.frame_limit:
            raise asyncio.CancelledError
        _generated_at, pixels = await asyncio.to_thread(self.frame_queue.get)
        if self.started is None:
            self.started = time.monotonic()
        target = self.started + self.index / self.fps
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        frame = VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = self.index * 3600
        frame.time_base = Fraction(1, 90000)
        self.index += 1
        return frame


async def connect_peer(frame_queue, index, metrics, ready):
    server = RTCPeerConnection()
    client = RTCPeerConnection()
    # Send one transport-preroll frame; H.264 startup can consume one frame
    # before the receiver exposes decoded media.
    server.addTrack(QueueVideoTrack(frame_queue, frame_limit=251))
    client.addTransceiver("video", direction="recvonly")
    done = asyncio.Event()
    row = metrics[index]

    @client.on("track")
    def on_track(track):
        async def consume():
            try:
                while row["frames"] < 250:
                    await track.recv()
                    now = time.time()
                    if row["first_frame_at"] is None:
                        row["first_frame_at"] = now
                    row["arrivals"].append(now)
                    row["frames"] += 1
                    row["last_frame_at"] = now
            except Exception as exc:
                row["consumer_error"] = repr(exc)
            finally:
                done.set()
        asyncio.create_task(consume())

    offer = await client.createOffer()
    await client.setLocalDescription(offer)
    await server.setRemoteDescription(client.localDescription)
    answer = await server.createAnswer()
    await server.setLocalDescription(answer)
    await client.setRemoteDescription(server.localDescription)
    ready[index] = True
    return server, client, done


def feed_session(sdk, chunks, barrier, row):
    try:
        barrier.wait()
        row["audio_start_at"] = time.time()
        base = time.monotonic()
        for chunk_index, chunk in enumerate(chunks):
            target = base + chunk_index * .2
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            sdk.run_chunk(chunk, (3, 5, 2))
        row["audio_feed_end_at"] = time.time()
        sdk.close()
        torch.cuda.synchronize()
        row["generation_end_at"] = time.time()
    except Exception as exc:
        row["producer_error"] = repr(exc)


async def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    events = {"process_start": time.time(), "sessions_requested": args.sessions}
    frame_queues = [queue.Queue(maxsize=300) for _ in range(args.sessions)]
    metrics = [{"index": i, "frames": 0, "first_frame_at": None,
                "last_frame_at": None, "arrivals": []} for i in range(args.sessions)]
    sdks = []

    events["model_load_start"] = time.time()
    for index in range(args.sessions):
        key = f"session-{index}"
        SINKS[key + ".tmp.mp4"] = frame_queues[index]
        before = time.time()
        sdk = pipeline.StreamSDK(args.config, args.engines)
        torch.cuda.synchronize()
        metrics[index]["model_loaded_at"] = time.time()
        metrics[index]["model_load_s"] = metrics[index]["model_loaded_at"] - before
        sdks.append(sdk)
    events["model_load_end"] = time.time()

    events["avatar_setup_start"] = time.time()
    for index, sdk in enumerate(sdks):
        sdk.setup(args.image, f"session-{index}")
        sdk.setup_Nd(args.frames)
        torch.cuda.synchronize()
        metrics[index]["avatar_ready_at"] = time.time()
    events["avatar_setup_end"] = time.time()

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = np.concatenate([np.zeros(3 * 640, dtype=np.float32), audio.astype(np.float32)])
    split_len = int(10 * .04 * 16000) + 80
    chunks = []
    for offset in range(0, len(audio), 5 * 640):
        chunk = audio[offset:offset + split_len]
        if len(chunk) < split_len:
            chunk = np.pad(chunk, (0, split_len-len(chunk)))
        chunks.append(chunk)

    ready = [False] * args.sessions
    connections = await asyncio.gather(*(
        connect_peer(frame_queues[i], i, metrics, ready) for i in range(args.sessions)
    ))
    events["webrtc_ready"] = time.time()
    barrier = threading.Barrier(args.sessions)
    workers = [threading.Thread(target=feed_session,
                 args=(sdks[i], chunks, barrier, metrics[i]), daemon=True)
               for i in range(args.sessions)]
    for worker in workers:
        worker.start()
    events["workers_started"] = time.time()

    try:
        await asyncio.wait_for(asyncio.gather(*(item[2].wait() for item in connections)), args.timeout)
    except asyncio.TimeoutError:
        events["timeout"] = True
    for worker in workers:
        worker.join(timeout=10)
    events["test_end"] = time.time()
    for server, client, _done in connections:
        await server.close()
        await client.close()

    for row in metrics:
        arrivals = row.pop("arrivals")
        gaps = [b-a for a,b in zip(arrivals, arrivals[1:])]
        row["first_frame_s"] = (row["first_frame_at"] - row["audio_start_at"]
                                if row.get("first_frame_at") and row.get("audio_start_at") else None)
        row["playout_s"] = (row["last_frame_at"] - row["first_frame_at"]
                            if row.get("last_frame_at") and row.get("first_frame_at") else None)
        row["received_fps"] = ((row["frames"] - 1) / row["playout_s"]
                               if row.get("playout_s", 0) > 0 else 0)
        row["max_gap_s"] = max(gaps, default=0)
        row["p95_gap_s"] = sorted(gaps)[min(len(gaps)-1, int(len(gaps)*.95))] if gaps else 0
        row["gaps_over_80ms"] = sum(gap > .08 for gap in gaps)

    complete = sum(row["frames"] >= args.frames and not row.get("producer_error") for row in metrics)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "sessions_requested": args.sessions,
        "sessions_completed": complete,
        "fps": 25,
        "frames_per_session": args.frames,
        "audio_chunk_interval_s": .2,
        "online_config": args.config,
        "events": events,
        "peers": metrics,
        "aggregate_received_fps": sum(row["frames"] for row in metrics) /
            max(.001, events["test_end"] - events["workers_started"]),
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if complete == args.sessions else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--image", default="/workspace/benchmarks/same-avatar/shared.png")
    parser.add_argument("--audio", default="/workspace/benchmarks/same-avatar/audio.wav")
    parser.add_argument("--config", default="/workspace/ditto-talkinghead/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    parser.add_argument("--engines", default="/workspace/ditto-talkinghead/checkpoints/ditto_trt_Ampere_Plus")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
