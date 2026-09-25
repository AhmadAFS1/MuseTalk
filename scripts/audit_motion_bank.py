#!/usr/bin/env python3
"""Exercise every cross-pose candidate with the actual bridge implementation.

Pixel-step diagnostics can find discontinuities; they do not certify identity,
natural motion, or invisible transitions. Always review received WebRTC video.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_transitions import MotionBank, file_hash, flow_blend, read_video


def audit(path, output_fps):
    bank = MotionBank(json.loads(path.read_text()))
    cv2.setNumThreads(2)
    sources = {}
    for pose, source in bank.sources.items():
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Source hash mismatch")
        sources[pose], _ = read_video(source["path"])
    count = max(2, math.ceil(bank.bridge_seconds*output_fps))
    summaries, rows = {}, []
    started = time.monotonic()
    for source, frames in sources.items():
        for target, incoming in sources.items():
            if source == target: continue
            pair_rows = []
            for index, old in enumerate(frames):
                edge = bank.edge(source,index,target)
                if not edge["admissible"]: continue
                dest = edge["target_frame"]
                cut = float(np.abs(old.astype(np.int16)-incoming[dest].astype(np.int16)).mean())
                previous, peak = old, 0
                for n in range(count):
                    frame = incoming[(dest+int(n*bank.fps/output_fps)) % len(incoming)]
                    t = (n+1)/count
                    blend = flow_blend(old,frame,.5-.5*math.cos(math.pi*t))
                    peak = max(peak,float(np.abs(blend.astype(np.int16)-previous.astype(np.int16)).mean()))
                    previous = blend
                row = {"source": source, "target": target, "frame": index,
                       "target_frame": dest, "hard_cut_mean_abs_bgr": round(cut,4),
                       "bridge_peak_step_mean_abs_bgr": round(peak,4)}
                pair_rows.append(row)
            rows.extend(pair_rows)
            summaries[source+"->"+target] = {
                "candidates_tested": len(pair_rows),
                "total_source_frames": len(frames),
                "hard_cut_median_mean_abs_bgr": float(np.median([r["hard_cut_mean_abs_bgr"] for r in pair_rows])) if pair_rows else None,
                "bridge_median_peak_step_mean_abs_bgr": float(np.median([r["bridge_peak_step_mean_abs_bgr"] for r in pair_rows])) if pair_rows else None,
                "bridge_worst_peak_step_mean_abs_bgr": max((r["bridge_peak_step_mean_abs_bgr"] for r in pair_rows), default=None)}
    return {"method": __doc__, "atlas_sha256": file_hash(path), "output_fps": output_fps,
            "elapsed_seconds": time.monotonic()-started, "summaries": summaries, "rows": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=float, default=20)
    args = p.parse_args()
    result = audit(args.atlas,args.fps)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summaries"],indent=2))
