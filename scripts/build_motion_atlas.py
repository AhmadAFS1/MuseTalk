#!/usr/bin/env python3
"""Build source-hash-bound, all-phase candidate transitions for three clips."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_transitions import IDLE, TALK, SMILE, file_hash, read_video, MotionBank


def build(source_dir, output, measurements):
    provenance_path = source_dir / "manifest.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    poses = {IDLE: "idle", TALK: "talking", SMILE: "smiling"}
    sources, features, dynamics, landmarks, lips = {}, {}, {}, {}, {}
    dims, fps = None, None
    for pose, name in poses.items():
        path = source_dir / (name + ".mp4")
        frames, rate = read_video(path)
        shape = frames[0].shape
        if dims is not None and (shape != dims or rate != fps):
            raise ValueError("Source geometry and fps must match")
        dims, fps = shape, rate
        sources[pose] = {"path": str(path.resolve()), "sha256": file_hash(path),
                         "frame_count": len(frames), "width": shape[1], "height": shape[0]}
        # Match upper face, hair, shoulders, room; lip articulation is checked
        # separately on the actual MuseTalk recording and mouth-release bridge.
        low = np.array([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64,104))
                        for f in frames], dtype=np.float32)
        mask = np.ones((104,64), np.float32)
        mask[38:72, 15:50] = 0
        features[pose] = (low * mask).reshape(len(frames), -1) / 255
        dynamics[pose] = np.roll(features[pose], -2, axis=0) - np.roll(features[pose], 2, axis=0)
        rows = measurements["videos"][name]["rows"]
        if len(rows) != len(frames):
            raise ValueError("Measurement frame count mismatch")
        # Bind this legacy measurement report to its adjacent generation manifest
        # and verify decoded endpoints as well as file hash and frame count.
        measured_hash = measurements["videos"][name].get("sha256")
        if measured_hash is None and provenance is not None:
            measured_hash = provenance["poses"][name]["delivery"]["sha256"]
        if measured_hash != sources[pose]["sha256"]:
            raise ValueError("Source differs from measurement provenance")
        import hashlib
        for endpoint, index in (("first", 0), ("last", -1)):
            rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != measurements["videos"][name][endpoint+"_rgb_sha256"]:
                raise ValueError("Measurement endpoint mismatch")
        landmarks[pose] = np.array([[r["eye_mid_y_change_px"], r["eye_scale_change_pct"],
                                     r["eye_line_roll_deg"], r.get("eye_mid_x_change_px", 0),
                                     r.get("eye_opening_px", 0)] for r in rows])
        lips[pose] = np.array([r["lip_gap_px"] for r in rows])
        del frames
    endpoints = {measurements["videos"][name][endpoint+"_rgb_sha256"]
                 for name in poses.values() for endpoint in ("first", "last")}
    if len(endpoints) != 1:
        raise ValueError("All source endpoints must share the same decoded anchor")
    edges, coverage = {}, {}
    for pose in poses:
        edges[pose] = {}
        for target in poses:
            a, b = features[pose], features[target]
            distances = np.maximum(0, (a*a).sum(1)[:,None] + (b*b).sum(1)[None,:] - 2*a@b.T) / a.shape[1]
            v, w = dynamics[pose], dynamics[target]
            velocity = np.maximum(0, (v*v).sum(1)[:,None] + (w*w).sum(1)[None,:] - 2*v@w.T) / v.shape[1]
            delta = landmarks[pose][:,None,:] - landmarks[target][None,:,:]
            score = distances * 1000 + velocity * 250 + np.abs(delta[...,0])*.04 + np.abs(delta[...,1])*.08 + np.abs(delta[...,2])*.04
            score += np.abs(delta[...,3]) * .08 + np.abs(delta[...,4]) * .12
            allowed = ((np.abs(delta[...,0]) <= 28) & (np.abs(delta[...,1]) <= 7)
                       & (np.abs(delta[...,2]) <= 8) & (np.abs(delta[...,3]) <= 16)
                       & (np.abs(delta[...,4]) <= 5) & (distances <= .025))
            if target == IDLE:
                allowed &= lips[target][None,:] <= 3.5
            # Entry into talking/smile should actually reach the action, rather
            # than always choosing the repeated neutral ending.
            if target != IDLE and pose != target:
                score[:, :12] += 1
                score[:, -36:] += 2
            score = np.where(allowed, score, 1e6 + score)
            result = []
            for i in range(len(a)):
                j = (i + 1) % len(b) if pose == target else int(np.argmin(score[i]))
                # Same-source recovery is consecutive motion: blinks can change
                # quickly in one frame, and should retain their native timing.
                admissible = bool(lips[target][j] <= 3.5) if pose == target == IDLE else bool(allowed[i,j])
                result.append({"target_frame": j, "score": round(float(score[i,j]),5),
                               "admissible": admissible,
                               "eye_height_delta_px": round(float(delta[i,j,0]),3),
                               "eye_opening_delta_px": round(float(delta[i,j,4]),3),
                               "eye_horizontal_delta_px": round(float(delta[i,j,3]),3)})
            edges[pose][target] = result
        coverage[pose] = {"covered": sum(e["admissible"] for e in edges[pose][IDLE]),
                          "total": len(edges[pose][IDLE])}
    manifest = {"version": 1, "status": "candidate_requires_recorded_review",
                "fps": fps, "bridge_seconds": .3, "short_reply_seconds": 3,
                "sources": sources, "edges": edges, "exit_coverage": coverage,
                "method": "upper-frame appearance + four-frame dynamics + saved FaceMesh geometry/blink state and closed-mouth idle; bounded bidirectional optical-flow bridge"}
    MotionBank(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"output": str(output), "exit_coverage": coverage}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--measurements", type=Path, required=True)
    args = p.parse_args()
    build(args.source_dir, args.output, json.loads(args.measurements.read_text()))
