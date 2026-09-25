#!/usr/bin/env python3
"""Offline, source-hash-bound six-point eye contours for a motion-atlas bank.

Requires MediaPipe in the measurement environment; runtime blending does not.
This does not modify the atlas or grant perceptual acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_eye_blend import validate_eye_points
from scripts.motion_transitions import IDLE, TALK, SMILE, atomic_json, file_hash

LEFT = (33, 160, 158, 133, 153, 144)
RIGHT = (362, 385, 387, 263, 373, 380)


def measure_motion_eyes(atlas_path, *, mesh_factory=None):
    import cv2

    atlas_path = Path(atlas_path).resolve()
    atlas_hash = file_hash(atlas_path)
    bank = json.loads(atlas_path.read_text())
    if bank.get("version") != 1 or set(bank.get("sources", {})) != {IDLE, TALK, SMILE}:
        raise ValueError("Eye measurements require a version 1 three-pose motion atlas")
    fps = float(bank["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid atlas fps")
    if mesh_factory is None:
        import mediapipe as mp
        mesh_factory = mp.solutions.face_mesh.FaceMesh
    result = {"version": 1, "method": "incoming_roi_v1",
              "measurement_method": "static FaceMesh refined; six eye contour points per eye",
              "status": "candidate_requires_recorded_review", "atlas_sha256": atlas_hash,
              "fps": fps, "sources": {}, "source_hashes": {}, "frames": {}}
    with mesh_factory(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                      min_detection_confidence=.5) as mesh:
        for pose in (IDLE, TALK, SMILE):
            source = bank["sources"][pose]
            path = Path(source["path"])
            if not path.is_absolute():
                path = atlas_path.parent / path
            path = path.resolve()
            expected = source["sha256"]
            if file_hash(path) != expected:
                raise ValueError(f"Source hash mismatch before measuring {pose}")
            width, height = int(source["width"]), int(source["height"])
            if min(width, height) < 32 or int(source["frame_count"]) < 2:
                raise ValueError(f"Invalid source geometry/count for {pose}")
            cap = cv2.VideoCapture(str(path))
            rows = []
            try:
                observed_fps = float(cap.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(observed_fps) or abs(observed_fps - fps) > .01:
                    raise ValueError(f"Source fps mismatch for {pose}")
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape != (height, width, 3):
                        raise ValueError(f"Source geometry mismatch for {pose}")
                    prediction = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    faces = prediction.multi_face_landmarks
                    if not faces or len(faces) != 1:
                        raise ValueError(f"Missing face in {pose} frame {len(rows)}")
                    landmarks = faces[0].landmark
                    eye_points = {eye: [[landmarks[i].x * width, landmarks[i].y * height]
                                        for i in indices]
                                  for eye, indices in (("left", LEFT), ("right", RIGHT))}
                    validated = validate_eye_points(eye_points, width, height)
                    rows.append({eye: points.tolist() for eye, points in validated.items()})
            finally:
                cap.release()
            if len(rows) != int(source["frame_count"]):
                raise ValueError(f"Source decoded frame-count mismatch for {pose}")
            if file_hash(path) != expected:
                raise ValueError(f"Source changed while measuring {pose}")
            result["sources"][pose] = {"sha256": expected, "width": width, "height": height,
                                        "fps": observed_fps, "frame_count": len(rows)}
            result["source_hashes"][pose] = expected
            result["frames"][pose] = rows
    if file_hash(atlas_path) != atlas_hash:
        raise ValueError("Atlas changed while measuring eye contours")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    atlas = args.atlas.resolve()
    output = args.output.resolve()
    bank = json.loads(atlas.read_text())
    inputs = {atlas}
    for source in bank.get("sources", {}).values():
        path = Path(source["path"])
        inputs.add((path if path.is_absolute() else atlas.parent / path).resolve())
    if output in inputs:
        parser.error("Output must not overwrite the atlas or any source video")
    atomic_json(output, measure_motion_eyes(atlas))
    print(f"Wrote source-bound eye measurements to {output}")


if __name__ == "__main__":
    main()
