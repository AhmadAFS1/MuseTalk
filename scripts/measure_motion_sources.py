#!/usr/bin/env python3
"""Measure every source frame against the same portrait coordinate system.

Requires OpenCV and MediaPipe 0.10.x (installed in the local LTX environment).
Measurements are diagnostics, not perceptual approval.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_transitions import file_hash


def measure_sources(source_dir):
    import mediapipe as mp
    reference_cap = cv2.VideoCapture(str(source_dir / "idle.mp4"))
    ok, reference = reference_cap.read()
    reference_cap.release()
    if not ok:
        raise ValueError("Cannot read idle reference")
    videos = {}
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=.5) as mesh:
        def points(frame):
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                raise ValueError("Face missing in source frame")
            h,w = frame.shape[:2]
            return np.array([(p.x*w,p.y*h) for p in result.multi_face_landmarks[0].landmark])
        ref = points(reference)
        ref_span = np.linalg.norm(ref[33]-ref[263])
        ref_mid = (ref[33]+ref[263])/2
        for name in ("idle", "talking", "smiling"):
            path = source_dir/(name+".mp4")
            cap = cv2.VideoCapture(str(path))
            fps, rows, hashes = cap.get(cv2.CAP_PROP_FPS), [], []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    p = points(frame)
                    span = np.linalg.norm(p[33]-p[263])
                    mid = (p[33]+p[263])/2
                    eye_open = (abs(p[159,1]-p[145,1])+abs(p[386,1]-p[374,1]))/2
                    rows.append({"frame": len(rows), "eye_mid_y_change_px": float(mid[1]-ref_mid[1]),
                                 "eye_mid_x_change_px": float(mid[0]-ref_mid[0]),
                                 "eye_scale_change_pct": float((span/ref_span-1)*100),
                                 "eye_line_roll_deg": math.degrees(math.atan2(*(p[263]-p[33])[::-1])),
                                 "eye_opening_px": float(eye_open),
                                 "lip_gap_px": float(abs(p[13,1]-p[14,1])),
                                 "mouth_width_px": float(abs(p[61,0]-p[291,0]))})
                    hashes.append(hashlib.sha256(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
            finally:
                cap.release()
            if not rows:
                raise ValueError(f"Empty source: {path}")
            videos[name] = {"file": str(path.resolve()), "sha256": file_hash(path),
                            "fps": fps, "frame_count": len(rows), "rows": rows,
                            "first_rgb_sha256": hashes[0], "last_rgb_sha256": hashes[-1]}
    return {"version": 1, "method": "static FaceMesh; common idle-first-frame reference; file-hash-bound",
            "videos": videos}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(measure_sources(args.source_dir), indent=2)+"\n")
