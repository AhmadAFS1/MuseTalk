#!/usr/bin/env python3
"""Render v3/v4 WebRTC transition frame pairs selected by the comparison audit."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/workspace/MuseTalk")
OUTPUT_DIR = (
    ROOT
    / "generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries"
)
AUDIT_PATH = OUTPUT_DIR / "v4_vs_v3_cadence_neutral_webrtc_validation.json"
BASELINE = "baseline_v3_15frame_2crossfade_30playback_15lipsync"
CANDIDATE = "candidate_v4_30frame_2crossfade_30playback_15lipsync"
FRAME_SIZE = (180, 320)
LABEL_HEIGHT = 54


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index} from {path}")
    return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)


def labeled_row(
    event_id: str,
    old_video: Path,
    new_video: Path,
    old_event: dict,
    new_event: dict,
) -> np.ndarray:
    old_pair = old_event["400ms"]["frame_pair"]
    new_pair = new_event["400ms"]["frame_pair"]
    frames = [
        read_frame(old_video, old_pair[0]),
        read_frame(old_video, old_pair[1]),
        read_frame(new_video, new_pair[0]),
        read_frame(new_video, new_pair[1]),
    ]
    row = np.hstack(frames)
    header = np.zeros((LABEL_HEIGHT, row.shape[1], 3), dtype=np.uint8)
    old_rmse = old_event["400ms"]["regional_face_rmse_px"]["whole_face"]
    new_rmse = new_event["400ms"]["regional_face_rmse_px"]["whole_face"]
    change = (new_rmse / old_rmse - 1.0) * 100.0 if old_rmse else 0.0
    cv2.putText(
        header,
        f"{event_id.removeprefix('idle:')}  v3 {old_rmse:.3f}px  v4 {new_rmse:.3f}px  {change:+.1f}%",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "v3 before / after                         v4 before / after",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, row))


def main() -> int:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    old_run = audit["runs"][BASELINE]
    new_run = audit["runs"][CANDIDATE]
    old_events = old_run["events"]
    new_events = new_run["events"]
    idle_ids = [event_id for event_id in old_events if event_id.startswith("idle:")]
    rows = [
        labeled_row(
            event_id,
            Path(old_run["video"]),
            Path(new_run["video"]),
            old_events[event_id],
            new_events[event_id],
        )
        for event_id in idle_ids
    ]
    destination = OUTPUT_DIR / "v3_vs_v4_idle_transition_pairs_400ms.jpg"
    if not cv2.imwrite(str(destination), np.vstack(rows)):
        raise RuntimeError(f"Could not write {destination}")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
