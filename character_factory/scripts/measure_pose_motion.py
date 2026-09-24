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

    rows: list[tuple[float, float, float, float, float]] = []
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
        ) as face_mesh:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if first_frame is None:
                    first_frame = frame.copy()
                last_frame = frame
                frame_count += 1
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not result.multi_face_landmarks:
                    raise RuntimeError(f"Face not detected in {path} at frame {frame_count - 1}")
                points = result.multi_face_landmarks[0].landmark

                def point(index: int) -> tuple[float, float]:
                    return points[index].x * width, points[index].y * height

                left_eye, right_eye = point(33), point(263)
                upper_lip, lower_lip = point(13), point(14)
                eye_x = (left_eye[0] + right_eye[0]) / 2
                eye_y = (left_eye[1] + right_eye[1]) / 2
                eye_distance = math.dist(left_eye, right_eye)
                tilt = math.degrees(math.atan2(
                    right_eye[1] - left_eye[1], right_eye[0] - left_eye[0],
                ))
                lip_gap = abs(upper_lip[1] - lower_lip[1])
                rows.append((eye_x, eye_y, eye_distance, tilt, lip_gap))
    finally:
        capture.release()

    if first_frame is None or last_frame is None:
        raise RuntimeError(f"Video is empty: {path}")
    x0, y0, eye_distance0, tilt0, _ = rows[0]
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
        "max_central_lip_gap_px": round(max(row[4] for row in rows), 1),
        "eye_y_delta_each_second_px": [
            round(rows[index][1] - y0, 1)
            for index in range(0, len(rows), max(1, round(fps)))
        ],
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
        "method": "MediaPipe FaceMesh on every decoded frame. Eye midpoint from landmarks 33/263; central lip gap from landmarks 13/14. Excursions are relative to the first frame. These are diagnostics, not visual approval.",
        "videos": {path.stem: measure(path) for path in args.videos},
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
