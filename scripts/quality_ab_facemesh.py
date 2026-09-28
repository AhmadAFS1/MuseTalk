"""FaceMesh landmark helper for scripts/quality_ab_metrics.py.

Run ONLY with the established FaceMesh interpreter:
    /workspace/SoulX-FlashHead/.venv/bin/python scripts/quality_ab_facemesh.py ...

It uses exactly the tracker configuration of the accepted chin workflow
(character_factory/h3_avatar_workflow/track_stage.py and tracker_worker.py):
FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
min_detection_confidence=.5, min_tracking_confidence=.5), one fresh instance per
sequence, BGR->RGB, landmark (x*W, y*H) as float32. Running it on an encoded
refined_raw.mp4 therefore reproduces validate_stage.py's <arm>_landmarks.npy.

Inputs (one sequence per invocation):
    --video PATH               decode with OpenCV (same decode as the workflow)
    --stdin-frames N H W       read N raw BGR uint8 frames of HxW from stdin
Output:
    --out PATH.npy             (N, 478, 2) float32; frames without a face are NaN
A one-line JSON summary is printed on stdout.
"""
import argparse
import json
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

cv2.setNumThreads(2)


def frames_from_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def frames_from_stdin(n, h, w):
    size = h * w * 3
    stream = sys.stdin.buffer
    for _ in range(n):
        buf = bytearray(size)
        view = memoryview(buf)
        got = 0
        while got < size:
            chunk = stream.readinto(view[got:])
            if not chunk:
                raise RuntimeError(f"stdin ended after {got} of {size} bytes")
            got += chunk
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--stdin-frames", nargs=3, type=int, metavar=("N", "H", "W"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frames = frames_from_video(a.video) if a.video else frames_from_stdin(*a.stdin_frames)
    rows, missing = [], []
    start = time.perf_counter()
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        for i, f in enumerate(frames):
            result = mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                missing.append(i)
                rows.append(np.full((478, 2), np.nan, np.float32))
                continue
            rows.append(np.asarray([(p.x * f.shape[1], p.y * f.shape[0])
                                    for p in result.multi_face_landmarks[0].landmark], np.float32))
    points = np.asarray(rows, np.float32)
    np.save(a.out, points)
    print(json.dumps(dict(frames=len(rows), missing_frames=missing, mediapipe=mp.__version__,
                          seconds=round(time.perf_counter() - start, 3))), flush=True)


if __name__ == "__main__":
    main()
