#!/usr/bin/env python3
"""Measure review diagnostics for LTX portrait clips; never auto-approve motion.

Run with the LTX environment, which has OpenCV and MediaPipe installed:
  python measure_pose_motion.py idle.mp4 talking.mp4 smiling.mp4 --output motion.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def measure(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    rows: list[tuple[float, float, float, float, float, float]] = []
    eye_openings: list[float] = []
    shoulder_y: list[float | None] = []
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    frame_count = 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh, mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if first_frame is None:
                    first_frame = frame.copy()
                last_frame = frame
                frame_count += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    raise RuntimeError(f"Face not detected in {path} at frame {frame_count - 1}")
                points = result.multi_face_landmarks[0].landmark

                def point(index: int) -> tuple[float, float]:
                    return points[index].x * width, points[index].y * height

                left_eye, right_eye = point(33), point(263)
                eye_openings.append((
                    abs(point(159)[1] - point(145)[1])
                    + abs(point(386)[1] - point(374)[1])
                ) / 2)
                upper_lip, lower_lip = point(13), point(14)
                left_mouth, right_mouth = point(61), point(291)
                eye_x = (left_eye[0] + right_eye[0]) / 2
                eye_y = (left_eye[1] + right_eye[1]) / 2
                eye_distance = math.dist(left_eye, right_eye)
                tilt = math.degrees(math.atan2(
                    right_eye[1] - left_eye[1], right_eye[0] - left_eye[0],
                ))
                lip_gap = abs(upper_lip[1] - lower_lip[1])
                mouth_width = abs(left_mouth[0] - right_mouth[0])
                rows.append((eye_x, eye_y, eye_distance, tilt, lip_gap, mouth_width))
                body = pose.process(rgb)
                if body.pose_landmarks:
                    body_points = body.pose_landmarks.landmark
                    shoulder_y.append((body_points[11].y + body_points[12].y) * height / 2)
                else:
                    shoulder_y.append(None)
    finally:
        capture.release()

    if first_frame is None or last_frame is None:
        raise RuntimeError(f"Video is empty: {path}")
    x0, y0, eye_distance0, tilt0, _, _ = rows[0]
    second = max(1, round(fps))
    eye_opening_median = float(np.median(eye_openings))
    blink_flags = [value < eye_opening_median * 0.55 for value in eye_openings]
    blink_candidate_count = sum(
        active and (index == 0 or not blink_flags[index - 1])
        for index, active in enumerate(blink_flags)
    )
    last_open_frame = next(
        (index for index in range(len(rows) - 2, -1, -1) if rows[index][4] > 3),
        None,
    )
    mouth_by_second = []
    for start in range(0, len(rows) - 1, second):
        segment = rows[start:min(start + second, len(rows) - 1)]
        gaps = np.array([row[4] for row in segment])
        widths = np.array([row[5] for row in segment])
        mouth_by_second.append({
            "second": start // second,
            "mean_lip_gap_px": round(float(gaps.mean()), 1),
            "max_lip_gap_px": round(float(gaps.max()), 1),
            "fraction_frames_lip_gap_over_3px": round(float((gaps > 3).mean()), 2),
            "mean_mouth_shape_change_px_per_frame": round(float(
                np.abs(np.diff(gaps)).mean() + np.abs(np.diff(widths)).mean()
            ), 2) if len(segment) > 1 else 0.0,
        })

    detected_shoulders = sum(value is not None for value in shoulder_y)
    shoulder_summary: dict[str, object] = {"shoulder_detected_frames": detected_shoulders}
    if detected_shoulders == len(rows):
        # Independent per-frame pose detection avoids tracking drift. A small
        # median window filters occasional landmark jitter before comparison.
        smoothed = [float(np.median(shoulder_y[max(0, index - 6):index + 7]))
                    for index in range(len(shoulder_y))]
        baseline = smoothed[0]
        shoulder_summary.update({
            "max_shoulder_mid_y_excursion_px": round(max(abs(value - baseline) for value in smoothed), 1),
            "shoulder_mid_y_delta_each_second_px": [
                round(smoothed[index] - baseline, 1)
                for index in range(0, len(smoothed), second)
            ],
        })
    return {
        "file": str(path.resolve()),
        "frames_read": frame_count,
        "face_detections": len(rows),
        "width": width,
        "height": height,
        "fps": fps,
        "first_last_decoded_rgb_equal": bool(np.array_equal(first_frame, last_frame)),
        "endpoint_rgb_sha256": hashlib.sha256(
            cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB).tobytes()
        ).hexdigest(),
        "initial_eye_distance_px": round(eye_distance0, 1),
        "max_eye_x_excursion_px": round(max(abs(row[0] - x0) for row in rows), 1),
        "max_eye_y_excursion_px": round(max(abs(row[1] - y0) for row in rows), 1),
        "max_eye_y_excursion_pct_of_eye_distance": round(
            100 * max(abs(row[1] - y0) for row in rows) / eye_distance0, 1
        ),
        "max_eye_line_tilt_change_deg": round(max(abs(row[3] - tilt0) for row in rows), 1),
        "median_eye_opening_px": round(eye_opening_median, 1),
        "min_eye_opening_px": round(min(eye_openings), 1),
        "blink_candidate_count": blink_candidate_count,
        "max_central_lip_gap_px": round(max(row[4] for row in rows), 1),
        "last_lip_gap_over_3px_time_s": (
            round(last_open_frame / fps, 2) if last_open_frame is not None else None
        ),
        "trailing_lip_gap_under_3px_duration_s": (
            round((len(rows) - 2 - last_open_frame) / fps, 2)
            if last_open_frame is not None else round((len(rows) - 1) / fps, 2)
        ),
        "mouth_activity_each_second": mouth_by_second,
        "eye_y_delta_each_second_px": [
            round(rows[index][1] - y0, 1)
            for index in range(0, len(rows), second)
        ],
        **shoulder_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stems = [path.stem for path in args.videos]
    if len(stems) != len(set(stems)):
        parser.error("Video basenames must be unique; run same-named revisions separately")
    report = {
        "method": "MediaPipe FaceMesh on every decoded frame: eye midpoint 33/263, eye opening 159/145 and 386/374, central lip gap 13/14, mouth corners 61/291. Blink candidates are eye-opening runs below 55% of the median. Static per-frame Pose shoulders 11/12, median-smoothed over 13 frames. Excursions are relative to the first frame. These are diagnostics, not visual approval.",
        "videos": {path.stem: measure(path) for path in args.videos},
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
