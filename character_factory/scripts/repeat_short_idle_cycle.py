#!/usr/bin/env python3
"""Repeat an endpoint-matched short LTX idle clip into a ten-second pilot idle.

The 81-frame source has one shallow breath and the same decoded first and last
frame. Three cycles share their boundary frame, yielding 81 + 80 + 80 = 241
frames. This is an explicit assembly operation, not another LTX generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def command(argv: list[str]) -> str:
    result = subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode()


def probe(path: Path) -> dict[str, object]:
    data = json.loads(command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path),
    ]))
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}")
    return streams[0]


def decoded_frame_hashes(path: Path) -> list[str]:
    output = command([
        "ffmpeg", "-v", "error", "-i", str(path), "-an", "-pix_fmt", "rgb24",
        "-f", "framemd5", "-",
    ])
    return [line.rsplit(",", 1)[-1].strip() for line in output.splitlines()
            if line and not line.startswith("#")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Endpoint-matched 81-frame LTX idle MP4")
    parser.add_argument("output", type=Path, help="Output 241-frame MP4")
    parser.add_argument("--report", type=Path, help="Optional JSON provenance and validation report")
    args = parser.parse_args()
    source = probe(args.input)
    if (source.get("r_frame_rate"), source.get("width"), source.get("height"),
            int(source.get("nb_frames", 0))) != ("24/1", 512, 832, 81):
        raise ValueError(f"Expected 512x832, 24 fps, 81 frames; got {source}")
    source_hashes = decoded_frame_hashes(args.input)
    if len(source_hashes) != 81 or source_hashes[0] != source_hashes[-1]:
        raise ValueError("Source must have 81 frames with identical decoded endpoints")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-stream_loop", "2", "-i", str(args.input),
        "-vf", "select='not(eq(n,81)+eq(n,162))',setpts=N/(24*TB)",
        "-frames:v", "241", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-qp", "18", "-x264-params", "keyint=1:min-keyint=1:scenecut=0:bframes=0",
        str(args.output),
    ], check=True)
    result = probe(args.output)
    frame_hashes = decoded_frame_hashes(args.output)
    if (result.get("r_frame_rate"), result.get("width"), result.get("height"),
            int(result.get("nb_frames", 0))) != ("24/1", 512, 832, 241):
        raise ValueError(f"Unexpected output video metadata: {result}")
    if len(frame_hashes) != 241 or len({frame_hashes[i] for i in (0, 80, 160, 240)}) != 1:
        raise ValueError("The three loop joins and final endpoint are not identical")
    report = {
        "operation": "repeat_81_frame_endpoint_matched_LTX_idle_three_times_without_duplicate_join_frames",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "source_frames": 81,
        "cycles": 3,
        "output_frames": 241,
        "fps": 24,
        "width": 512,
        "height": 832,
        "decoded_loop_boundaries_equal": True,
        "decoded_boundary_framemd5": frame_hashes[0],
        "audio_streams": 0,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
