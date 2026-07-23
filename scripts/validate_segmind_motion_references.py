#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.avatar_generation import load_motion_reference_preset


ANCHOR_SSIM_MINIMUM = 0.99


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _probe(path: Path) -> Dict[str, Any]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def _extract_frame(video_path: Path, frame_number: int, output_path: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_number})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(output_path),
        ]
    )


def _ssim(first_path: Path, second_path: Path) -> float:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(first_path),
            "-i",
            str(second_path),
            "-lavfi",
            "[0:v][1:v]ssim",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.findall(r"All:([0-9.]+)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not read SSIM result: {result.stderr[-1000:]}")
    return float(match[-1])


def validate_preset(preset_id: str) -> Dict[str, Any]:
    preset = load_motion_reference_preset(preset_id)
    expected = preset.get("normalization") or {}
    expected_width, expected_height = (
        int(part) for part in str(expected.get("output_resolution", "720x1280")).split("x", 1)
    )
    expected_fps = float(expected.get("fps", 30))
    expected_duration = float(expected.get("duration_seconds", 10))
    pose_results: Dict[str, Dict[str, Any]] = {}
    first_frame_paths: Dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="segmind-motion-validation-") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        for pose in preset["poses"]:
            pose_id = pose["id"]
            video_path = pose["resolved_path"]
            probe = _probe(video_path)
            video_streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
            audio_streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "audio"]
            if len(video_streams) != 1:
                raise RuntimeError(f"{pose_id} must contain exactly one video stream")
            stream = video_streams[0]
            frame_count = int(stream.get("nb_frames") or 0)
            duration = float(probe["format"].get("duration") or stream.get("duration") or 0)
            fps = float(Fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"))

            first_frame_path = temp_dir / f"{pose_id}.first.png"
            last_frame_path = temp_dir / f"{pose_id}.last.png"
            _extract_frame(video_path, 0, first_frame_path)
            _extract_frame(video_path, frame_count - 1, last_frame_path)
            first_frame_paths[pose_id] = first_frame_path
            endpoint_ssim = _ssim(first_frame_path, last_frame_path)

            checks = {
                "codec_h264": stream.get("codec_name") == "h264",
                "resolution": (
                    int(stream.get("width") or 0) == expected_width
                    and int(stream.get("height") or 0) == expected_height
                ),
                "fps": abs(fps - expected_fps) < 0.001,
                "duration": abs(duration - expected_duration) < 0.05,
                "segmind_duration_range": 3.0 <= duration <= 30.0,
                "silent": not audio_streams,
                "neutral_loop_anchor": endpoint_ssim >= ANCHOR_SSIM_MINIMUM,
            }
            pose_results[pose_id] = {
                "path": _display_path(video_path),
                "codec": stream.get("codec_name"),
                "pixel_format": stream.get("pix_fmt"),
                "width": int(stream.get("width") or 0),
                "height": int(stream.get("height") or 0),
                "fps": fps,
                "duration_seconds": duration,
                "frame_count": frame_count,
                "audio_stream_count": len(audio_streams),
                "first_last_ssim": endpoint_ssim,
                "checks": checks,
                "passed": all(checks.values()),
            }

        default_pose_id = preset["default_pose_id"]
        canonical_first_path = first_frame_paths[default_pose_id]
        for pose_id, first_frame_path in first_frame_paths.items():
            anchor_ssim = _ssim(canonical_first_path, first_frame_path)
            pose_results[pose_id]["canonical_first_frame_ssim"] = anchor_ssim
            pose_results[pose_id]["checks"]["canonical_start_anchor"] = (
                anchor_ssim >= ANCHOR_SSIM_MINIMUM
            )
            pose_results[pose_id]["passed"] = all(
                pose_results[pose_id]["checks"].values()
            )

    return {
        "schema_version": 1,
        "preset_id": preset["id"],
        "default_pose_id": preset["default_pose_id"],
        "manifest_path": _display_path(preset["manifest_path"]),
        "anchor_ssim_minimum": ANCHOR_SSIM_MINIMUM,
        "poses": pose_results,
        "passed": all(result["passed"] for result in pose_results.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Segmind motion-reference preset")
    parser.add_argument("preset_id", nargs="?", default="facetime_v1")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = validate_preset(args.preset_id)
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
