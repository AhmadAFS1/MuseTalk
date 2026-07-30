#!/usr/bin/env python3
"""Compare old and new WebRTC pose-transition geometry with one method."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path("/workspace/MuseTalk")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.mmlab_compat import apply_mmlab_compat_patches

apply_mmlab_compat_patches()

from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.structures import merge_data_samples  # noqa: E402


OUTPUT_DIR = ROOT / "generated/asset_analysis/2026-07-30_smooth_speaking_v3"
CONFIG = ROOT / "musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
CHECKPOINT = ROOT / "models/dwpose/dw-ll_ucoco_384.pth"
RUNS = OrderedDict(
    (
        (
            "previous_20fps",
            {
                "video": ROOT
                / "generated/webrtc_pose_showcase/2026-07-29/crossfade_comparison/2_frames/"
                "indian_tutor_mvp_four_v2_crossfade_2_webrtc_capture.mp4",
                "result": ROOT
                / "generated/webrtc_pose_showcase/2026-07-29/crossfade_comparison/2_frames/"
                "indian_tutor_mvp_four_v2_crossfade_2_run.log",
            },
        ),
        (
            "new_30playback_15lipsync",
            {
                "video": ROOT
                / "generated/webrtc_pose_showcase/2026-07-30/"
                "speaking_smooth_v3_30playback_15lipsync/"
                "indian_tutor_speaking_smooth_v3_30playback_15lipsync_webrtc_capture.mp4",
                "result": ROOT
                / "generated/webrtc_pose_showcase/2026-07-30/"
                "speaking_smooth_v3_30playback_15lipsync/webrtc_result.json",
            },
        ),
    )
)

REGIONS = {
    "whole_face": np.arange(0, 68),
    "jaw": np.arange(0, 17),
    "brows": np.arange(17, 27),
    "eyes": np.arange(36, 48),
    "nose": np.arange(27, 36),
    "mouth": np.arange(48, 68),
}


def load_last_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() != "{":
            continue
        try:
            value = json.loads("\n".join(lines[index:]))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"No final JSON object in {path}")


def similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    return source_centered @ rotation * scale + target_mean


def regional_rmse(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    aligned = similarity_align(source, target)
    return {
        name: round(
            float(np.sqrt(np.mean(np.sum((aligned[indexes] - target[indexes]) ** 2, axis=1)))),
            6,
        )
        for name, indexes in REGIONS.items()
    }


def decode_selected(path: Path, selected: set[int]) -> tuple[dict[int, np.ndarray], float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
        raise RuntimeError(f"Missing selected frames in {path}: {missing[:10]}")
    return frames, fps, count


def make_contact_sheet(
    run_name: str,
    transitions: dict[str, dict],
    frames: dict[int, np.ndarray],
) -> None:
    rows = []
    for transition_name, metrics in transitions.items():
        if "speaking_direct" not in transition_name:
            continue
        before_index, after_index = metrics["max_400ms_frame_pair"]
        before = cv2.resize(frames[before_index], (288, 512), interpolation=cv2.INTER_AREA)
        after = cv2.resize(frames[after_index], (288, 512), interpolation=cv2.INTER_AREA)
        pair = np.hstack((before, after))
        cv2.rectangle(pair, (0, 0), (576, 48), (0, 0, 0), -1)
        cv2.putText(
            pair,
            f"{transition_name} 400ms",
            (8, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(pair)
    cv2.imwrite(str(OUTPUT_DIR / f"{run_name}_speaking_transition_pairs.jpg"), np.vstack(rows))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = init_model(str(CONFIG), str(CHECKPOINT), device="cuda:0")
    output: dict[str, object] = {
        "schema_version": 1,
        "method": {
            "face_model": "MuseTalk DWPose 68-point face landmarks",
            "search_window_seconds": 0.45,
            "long_window_seconds": 0.4,
            "alignment": "least-squares similarity transform",
            "note": (
                "For each reported activation time, metrics select the largest "
                "face change within the surrounding search window."
            ),
        },
        "runs": {},
    }

    for run_name, paths in RUNS.items():
        video_path = paths["video"]
        result = load_last_json(paths["result"])
        trace = result["showcase_pose_trace"]
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid media facts for {video_path}")

        selected: set[int] = set()
        windows: list[tuple[str, int, int]] = []
        for source, target in zip(trace, trace[1:]):
            transition_name = f"{source['pose_id']}->{target['pose_id']}"
            center = float(target["at_seconds"])
            start = max(0, int(np.floor((center - 0.45) * fps)))
            end = min(frame_count - 1, int(np.ceil((center + 0.45) * fps)))
            selected.update(range(start, end + 1))
            windows.append((transition_name, start, end))

        frames, decoded_fps, decoded_count = decode_selected(video_path, selected)
        detections: dict[int, np.ndarray] = {}
        for index in sorted(selected):
            merged = merge_data_samples(inference_topdown(model, frames[index]))
            keypoints = np.asarray(merged.pred_instances.keypoints)[0, :, :2].astype(np.float64)
            detections[index] = keypoints[23:91]

        transition_metrics: dict[str, dict] = {}
        long_step = max(1, int(round(0.4 * decoded_fps)))
        for transition_name, start, end in windows:
            consecutive_candidates = []
            for first in range(start, end):
                second = first + 1
                face_rmse = regional_rmse(detections[first], detections[second])
                nose_travel = float(np.linalg.norm(detections[first][30] - detections[second][30]))
                consecutive_candidates.append((face_rmse["whole_face"], nose_travel, first, second, face_rmse))
            maximum_consecutive = max(consecutive_candidates, key=lambda item: item[0])

            long_candidates = []
            for first in range(start, end - long_step + 1):
                second = first + long_step
                face_rmse = regional_rmse(detections[first], detections[second])
                nose_travel = float(np.linalg.norm(detections[first][30] - detections[second][30]))
                long_candidates.append((face_rmse["whole_face"], nose_travel, first, second, face_rmse))
            maximum_long = max(long_candidates, key=lambda item: item[0])

            transition_metrics[transition_name] = {
                "max_consecutive_face_rmse_px": maximum_consecutive[4],
                "max_consecutive_nose_travel_px": round(maximum_consecutive[1], 6),
                "max_consecutive_frame_pair": [maximum_consecutive[2], maximum_consecutive[3]],
                "max_400ms_face_rmse_px": maximum_long[4],
                "max_400ms_nose_travel_px": round(maximum_long[1], 6),
                "max_400ms_frame_pair": [maximum_long[2], maximum_long[3]],
            }

        speaking = [
            value
            for name, value in transition_metrics.items()
            if "speaking_direct" in name
        ]
        non_speaking = [
            value
            for name, value in transition_metrics.items()
            if "speaking_direct" not in name
        ]
        summary = {
            "speaking_transition_count": len(speaking),
            "non_speaking_transition_count": len(non_speaking),
            "speaking_mean_max_400ms_face_rmse_px": round(
                float(np.mean([item["max_400ms_face_rmse_px"]["whole_face"] for item in speaking])),
                6,
            ),
            "speaking_max_400ms_face_rmse_px": round(
                float(np.max([item["max_400ms_face_rmse_px"]["whole_face"] for item in speaking])),
                6,
            ),
            "speaking_mean_max_consecutive_face_rmse_px": round(
                float(np.mean([item["max_consecutive_face_rmse_px"]["whole_face"] for item in speaking])),
                6,
            ),
            "speaking_max_consecutive_face_rmse_px": round(
                float(np.max([item["max_consecutive_face_rmse_px"]["whole_face"] for item in speaking])),
                6,
            ),
        }
        output["runs"][run_name] = {
            "video": str(video_path),
            "fps": decoded_fps,
            "frame_count": decoded_count,
            "summary": summary,
            "transitions": transition_metrics,
        }
        make_contact_sheet(run_name, transition_metrics, frames)

    previous = output["runs"]["previous_20fps"]["transitions"]
    current = output["runs"]["new_30playback_15lipsync"]["transitions"]
    comparison = {}
    for name in previous:
        old_value = previous[name]["max_400ms_face_rmse_px"]["whole_face"]
        new_value = current[name]["max_400ms_face_rmse_px"]["whole_face"]
        comparison[name] = {
            "previous_max_400ms_whole_face_rmse_px": old_value,
            "new_max_400ms_whole_face_rmse_px": new_value,
            "delta_px": round(new_value - old_value, 6),
            "percent_change": round((new_value / old_value - 1.0) * 100.0, 3) if old_value else None,
        }
    output["comparison"] = comparison

    output_path = OUTPUT_DIR / "webrtc_transition_comparison.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
