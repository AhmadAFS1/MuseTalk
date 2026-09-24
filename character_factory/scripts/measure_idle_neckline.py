#!/usr/bin/env python3
"""Track center shirt-neckline motion in one fixed-framing idle clip.

This portrait-specific diagnostic separates visible upper-torso motion from
landmark shoulder motion. It requires a high-contrast shirt/skin boundary
inside the search region; inspect the returned trace and source frames before
using it to judge another portrait.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def measure(path: Path, search_top: int, search_bottom: int, threshold: float) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    positions: list[float] = []
    shirt_color: np.ndarray | None = None
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            height, width = frame.shape[:2]
            if not (0 <= search_top < search_bottom <= height):
                raise ValueError(f"Search bounds {search_top}:{search_bottom} outside height {height}")
            left, right = int(0.48 * width), int(0.52 * width)
            if shirt_color is None:
                # This lower shirt patch contains fabric, away from skin and hair
                # for the current FaceTime compositions.
                sample = frame[640:675, left:right].reshape(-1, 3)
                shirt_color = np.median(sample, axis=0)
            found: list[int] = []
            for x in range(left, right, 2):
                pixels = frame[search_top:search_bottom, x].astype(float)
                distances = np.linalg.norm(pixels - shirt_color, axis=1)
                matches = np.flatnonzero(distances < threshold)
                if len(matches):
                    found.append(search_top + int(matches[0]))
            if not found:
                raise RuntimeError(f"Shirt boundary not detected at frame {frame_count - 1}")
            positions.append(float(np.median(found)))
    finally:
        capture.release()
    second = max(1, round(fps))
    baseline = positions[0]
    return {
        "file": str(path.resolve()),
        "frames_read": frame_count,
        "fps": fps,
        "search_y_pixels": [search_top, search_bottom],
        "center_x_fraction": [0.48, 0.52],
        "shirt_sample_y_pixels": [640, 675],
        "shirt_color_bgr": [round(float(value), 1) for value in shirt_color],
        "color_distance_threshold": threshold,
        "initial_neckline_y_px": baseline,
        "neckline_y_range_px": round(max(positions) - min(positions), 1),
        "neckline_y_delta_each_second_px": [
            round(positions[index] - baseline, 1)
            for index in range(0, frame_count, second)
        ],
        "unique_neckline_y_values": len(set(positions)),
        "caveat": "Color-boundary proxy for this portrait, not direct lung or chest-volume measurement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--search-top", type=int, default=535)
    parser.add_argument("--search-bottom", type=int, default=630)
    parser.add_argument("--threshold", type=float, default=40.0)
    args = parser.parse_args()
    report = {
        "method": "First shirt-colored pixel per center column, median across 48–52% frame width; fixed first-frame shirt color. Calibrated to the Japanese FaceTime portrait and checked against the Indian reference.",
        "videos": [
            measure(path, args.search_top, args.search_bottom, args.threshold)
            for path in args.videos
        ],
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
