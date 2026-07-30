#!/usr/bin/env python3
"""Cadence-neutral comparison of recent four-pose WebRTC showcases.

The earlier report's 400 ms measurements are already time-normalized, but its
consecutive-frame measurements compare a 30 fps capture with 20 fps captures.
This audit adds fixed 100 ms measurements and covers the internal pose
boundaries in both TTS cases.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path("/workspace/MuseTalk")
OUTPUT_DIR = ROOT / "generated/asset_analysis/2026-07-30_smooth_speaking_v3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.mmlab_compat import apply_mmlab_compat_patches

apply_mmlab_compat_patches()

from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.structures import merge_data_samples  # noqa: E402


CONFIG = ROOT / "musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
CHECKPOINT = ROOT / "models/dwpose/dw-ll_ucoco_384.pth"

RUNS = OrderedDict(
    (
        (
            "new_v3_2frames_30playback_15lipsync",
            {
                "video": ROOT
                / "generated/webrtc_pose_showcase/2026-07-30/"
                "speaking_smooth_v3_30playback_15lipsync/"
                "indian_tutor_speaking_smooth_v3_30playback_15lipsync_webrtc_capture.mp4",
                "result": ROOT
                / "generated/webrtc_pose_showcase/2026-07-30/"
                "speaking_smooth_v3_30playback_15lipsync/webrtc_result.json",
                "result_line": None,
                "crossfade_frames": 2,
                "asset_generation_fps": 15.0,
            },
        ),
        (
            "previous_v2_2frames_20playback_20lipsync",
            {
                "video": ROOT
                / "generated/webrtc_pose_showcase/2026-07-29/crossfade_comparison/2_frames/"
                "indian_tutor_mvp_four_v2_crossfade_2_webrtc_capture.mp4",
                "result": ROOT
                / "generated/webrtc_pose_showcase/2026-07-29/crossfade_comparison/2_frames/"
                "indian_tutor_mvp_four_v2_crossfade_2_run.log",
                "result_line": 60,
                "crossfade_frames": 2,
                "asset_generation_fps": 20.0,
            },
        ),
        (
            "previous_v2_0frames_20playback_20lipsync",
            {
                "video": ROOT
                / "generated/webrtc_pose_showcase/2026-07-29/crossfade_comparison/0_frames/"
                "indian_tutor_mvp_four_v2_crossfade_0_webrtc_capture.mp4",
                "result": ROOT
                / "generated/webrtc_pose_showcase/2026-07-29/crossfade_comparison/0_frames/"
                "indian_tutor_mvp_four_v2_crossfade_0_run.log",
                "result_line": 60,
                "crossfade_frames": 0,
                "asset_generation_fps": 20.0,
            },
        ),
    )
)

REGIONS = {
    "whole_face": np.arange(0, 68),
    "upper_face": np.arange(17, 48),
    "brows": np.arange(17, 27),
    "eyes": np.arange(36, 48),
    "nose": np.arange(27, 36),
    "mouth": np.arange(48, 68),
}


def load_result(path: Path, first_json_line: int | None) -> dict:
    text = path.read_text(encoding="utf-8")
    if first_json_line is not None:
        text = "\n".join(text.splitlines()[first_json_line - 1 :])
    return json.loads(text)


def similarity_rmse(source: np.ndarray, target: np.ndarray) -> float:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    left, singular_values, right_t = np.linalg.svd(source_centered.T @ target_centered)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    denominator = float(np.sum(source_centered * source_centered))
    scale = float(np.sum(singular_values) / denominator) if denominator else 1.0
    aligned = source_centered @ rotation * scale + target_mean
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))


def regional_rmse(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        name: round(similarity_rmse(source[indexes], target[indexes]), 6)
        for name, indexes in REGIONS.items()
    }


def media_facts(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or count <= 0:
        raise RuntimeError(f"Invalid media facts for {path}")
    return fps, count


def decode_selected(path: Path, selected: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    maximum = max(selected)
    index = 0
    while index <= maximum:
        ok, frame = capture.read()
        if not ok:
            break
        if index in selected:
            frames[index] = frame
        index += 1
    capture.release()
    missing = sorted(selected - frames.keys())
    if missing:
        raise RuntimeError(f"Missing frames in {path}: {missing[:10]}")
    return frames


def transition_events(result: dict, generation_fps: float) -> list[dict]:
    events: list[dict] = []
    trace = result["showcase_pose_trace"]
    for source, target in zip(trace, trace[1:]):
        events.append(
            {
                "id": f"idle:{source['pose_id']}->{target['pose_id']}",
                "group": (
                    "idle_speaking"
                    if "speaking_direct" in (source["pose_id"], target["pose_id"])
                    else "idle_non_speaking"
                ),
                "center_seconds": float(target["at_seconds"]),
                "source": source["pose_id"],
                "target": target["pose_id"],
            }
        )

    for case in result["speaking_cases"]:
        released = float(case["playout_released_at_seconds"])
        segments = case["rendered_pose_trace"]
        reaction = case["reaction_intent"]
        cumulative_frames = 0
        for source, target in zip(segments, segments[1:]):
            cumulative_frames += int(source["frame_count"])
            events.append(
                {
                    "id": (
                        f"tts:{reaction}:{source['pose_id']}->{target['pose_id']}"
                    ),
                    "group": "tts",
                    "center_seconds": released + cumulative_frames / generation_fps,
                    "source": source["pose_id"],
                    "target": target["pose_id"],
                }
            )
    return events


def maximum_fixed_step(
    detections: dict[int, np.ndarray],
    start: int,
    end: int,
    step: int,
) -> dict:
    candidates = []
    for first in range(start, end - step + 1):
        second = first + step
        rmse = regional_rmse(detections[first], detections[second])
        candidates.append((rmse["whole_face"], first, second, rmse))
    maximum = max(candidates, key=lambda item: item[0])
    return {
        "frame_pair": [maximum[1], maximum[2]],
        "regional_face_rmse_px": maximum[3],
    }


def summarize(events: dict[str, dict]) -> dict:
    summary: dict[str, dict] = {}
    for group in ("idle_speaking", "idle_non_speaking", "tts", "all"):
        values = [
            metrics
            for metrics in events.values()
            if group == "all" or metrics["group"] == group
        ]
        group_summary: dict[str, float | int] = {"count": len(values)}
        for duration in ("100ms", "400ms"):
            for region in ("whole_face", "upper_face"):
                numbers = [
                    value[duration]["regional_face_rmse_px"][region]
                    for value in values
                ]
                group_summary[f"mean_max_{duration}_{region}_rmse_px"] = round(
                    float(np.mean(numbers)), 6
                )
                group_summary[f"worst_max_{duration}_{region}_rmse_px"] = round(
                    float(np.max(numbers)), 6
                )
        summary[group] = group_summary
    return summary


def main() -> int:
    model = init_model(str(CONFIG), str(CHECKPOINT), device="cuda:0")
    output: dict[str, object] = {
        "schema_version": 1,
        "method": {
            "face_model": "MuseTalk DWPose 68-point face landmarks",
            "event_search_radius_seconds": 0.45,
            "fixed_steps_seconds": [0.1, 0.4],
            "alignment": "independent least-squares similarity transform per face region",
            "purpose": (
                "Fixed-time steps make 30 fps and 20 fps captures comparable. "
                "TTS boundaries are derived from playout release time plus the "
                "server-rendered pose frame trace."
            ),
            "caveat": (
                "The local maximum can include intentional pose or lip motion; "
                "upper-face TTS metrics are less confounded by lip sync."
            ),
        },
        "runs": {},
    }

    for run_name, paths in RUNS.items():
        video = paths["video"]
        result = load_result(paths["result"], paths["result_line"])
        fps, frame_count = media_facts(video)
        events = transition_events(result, paths["asset_generation_fps"])
        selected: set[int] = set()
        windows: list[tuple[dict, int, int]] = []
        for event in events:
            center = event["center_seconds"]
            start = max(0, int(np.floor((center - 0.45) * fps)))
            end = min(frame_count - 1, int(np.ceil((center + 0.45) * fps)))
            selected.update(range(start, end + 1))
            windows.append((event, start, end))

        frames = decode_selected(video, selected)
        detections: dict[int, np.ndarray] = {}
        for index in sorted(selected):
            merged = merge_data_samples(inference_topdown(model, frames[index]))
            keypoints = np.asarray(merged.pred_instances.keypoints)[0, :, :2].astype(
                np.float64
            )
            detections[index] = keypoints[23:91]

        event_metrics: dict[str, dict] = {}
        for event, start, end in windows:
            metrics = {
                "group": event["group"],
                "source": event["source"],
                "target": event["target"],
                "expected_center_seconds": round(event["center_seconds"], 6),
            }
            for label, seconds in (("100ms", 0.1), ("400ms", 0.4)):
                metrics[label] = maximum_fixed_step(
                    detections,
                    start,
                    end,
                    max(1, int(round(seconds * fps))),
                )
            event_metrics[event["id"]] = metrics

        output["runs"][run_name] = {
            "video": str(video),
            "capture_fps": fps,
            "capture_frame_count": frame_count,
            "crossfade_frames": paths["crossfade_frames"],
            "crossfade_duration_ms": round(
                paths["crossfade_frames"] / fps * 1000.0, 3
            ),
            "asset_generation_fps": paths["asset_generation_fps"],
            "summary": summarize(event_metrics),
            "events": event_metrics,
        }

    baseline = output["runs"]["previous_v2_2frames_20playback_20lipsync"]
    current = output["runs"]["new_v3_2frames_30playback_15lipsync"]
    comparisons = {}
    for group in ("idle_speaking", "idle_non_speaking", "tts", "all"):
        comparisons[group] = {}
        for metric in (
            "mean_max_100ms_whole_face_rmse_px",
            "worst_max_100ms_whole_face_rmse_px",
            "mean_max_100ms_upper_face_rmse_px",
            "worst_max_100ms_upper_face_rmse_px",
            "mean_max_400ms_whole_face_rmse_px",
            "worst_max_400ms_whole_face_rmse_px",
        ):
            old = baseline["summary"][group][metric]
            new = current["summary"][group][metric]
            comparisons[group][metric] = {
                "previous": old,
                "current": new,
                "delta_px": round(new - old, 6),
                "percent_change": (
                    round((new / old - 1.0) * 100.0, 3) if old else None
                ),
            }
    output["current_vs_previous_2frames"] = comparisons

    output_path = OUTPUT_DIR / "cadence_neutral_webrtc_validation.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
