#!/usr/bin/env python3
"""Audit authored eyebrow cues and decoded extrema in the certified ARDY MVP four."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ARDY_ROOT = Path("/workspace/ardy")
if str(ARDY_ROOT) not in sys.path:
    sys.path.insert(0, str(ARDY_ROOT))

from ardy.pose_video.renderer import BROW_RAISE_METERS, facial_cue_for_frame


DELIVERY = ARDY_ROOT / "outputs/pose_delivery_mvp_four_v3_smooth_speaking"
OUTPUT = Path("/workspace/MuseTalk/generated/asset_analysis/2026-07-30_ardy_eyebrow_audit")
ASSETS = OrderedDict(
    (
        ("neutral_resting", DELIVERY / "neutral_resting.mp4"),
        ("speaking_direct_v2", DELIVERY / "speaking_direct_v2.mp4"),
        ("light_smile", DELIVERY / "light_smile.mp4"),
        (
            "active_listening_empathetic_v1",
            DELIVERY / "active_listening_empathetic_v1.mp4",
        ),
    )
)
HANDLE_FRAMES = 15


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def read_frames(path: Path, indexes: set[int]) -> tuple[dict[int, np.ndarray], float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected: dict[int, np.ndarray] = {}
    index = 0
    while index <= max(indexes):
        ok, frame = capture.read()
        if not ok:
            break
        if index in indexes:
            selected[index] = frame
        index += 1
    capture.release()
    missing = indexes - selected.keys()
    if missing:
        raise RuntimeError(f"Missing frames in {path}: {sorted(missing)}")
    return selected, fps, frame_count


def labeled_crop(
    frame: np.ndarray,
    crop: tuple[int, int, int, int],
    label: str,
    size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    image = cv2.resize(frame[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (size[0], 38), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": 1,
        "delivery": str(DELIVERY),
        "method": {
            "source": "ARDY renderer facial_cue_for_frame evaluated for every frame",
            "brow_raise_scale_meters": BROW_RAISE_METERS,
            "transition_handle_frames": HANDLE_FRAMES,
            "decoded_visual_samples": "opening, maximum authored brow cue, closing",
        },
        "assets": {},
    }
    face_rows = []
    brow_rows = []

    for behavior_id, path in ASSETS.items():
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid media facts for {path}")

        cues = [
            facial_cue_for_frame(behavior_id, index, frame_count, fps)
            for index in range(frame_count)
        ]
        peak_index = max(
            range(frame_count),
            key=lambda index: max(
                cues[index]["left_brow"],
                cues[index]["right_brow"],
            ),
        )
        if (
            cues[peak_index]["left_brow"] == 0.0
            and cues[peak_index]["right_brow"] == 0.0
        ):
            peak_index = frame_count // 2

        selected, decoded_fps, decoded_count = read_frames(
            path, {0, peak_index, frame_count - 1}
        )
        opening = selected[0]
        peak = selected[peak_index]
        closing = selected[frame_count - 1]
        peak_cue = cues[peak_index]
        opening_zero = all(
            cue["left_brow"] == 0.0 and cue["right_brow"] == 0.0
            for cue in cues[:HANDLE_FRAMES]
        )
        closing_zero = all(
            cue["left_brow"] == 0.0 and cue["right_brow"] == 0.0
            for cue in cues[-HANDLE_FRAMES:]
        )
        nonzero = [
            index
            for index, cue in enumerate(cues)
            if cue["left_brow"] != 0.0 or cue["right_brow"] != 0.0
        ]

        report["assets"][behavior_id] = {
            "path": str(path),
            "fps": decoded_fps,
            "frame_count": decoded_count,
            "authored_brow_motion": bool(nonzero),
            "first_nonzero_frame": nonzero[0] if nonzero else None,
            "last_nonzero_frame": nonzero[-1] if nonzero else None,
            "opening_handle_brow_cues_zero": opening_zero,
            "closing_handle_brow_cues_zero": closing_zero,
            "peak_frame": peak_index,
            "peak_left_brow_cue": round(float(peak_cue["left_brow"]), 9),
            "peak_right_brow_cue": round(float(peak_cue["right_brow"]), 9),
            "peak_left_brow_raise_mm": round(
                float(peak_cue["left_brow"] * BROW_RAISE_METERS * 1000.0), 6
            ),
            "peak_right_brow_raise_mm": round(
                float(peak_cue["right_brow"] * BROW_RAISE_METERS * 1000.0), 6
            ),
            "decoded_first_frame_sha256": frame_hash(opening),
            "decoded_last_frame_sha256": frame_hash(closing),
            "decoded_first_equals_last": bool(np.array_equal(opening, closing)),
        }

        cue_label = (
            f"peak f{peak_index} "
            f"L{peak_cue['left_brow'] * BROW_RAISE_METERS * 1000.0:.2f}/"
            f"R{peak_cue['right_brow'] * BROW_RAISE_METERS * 1000.0:.2f}mm"
        )
        face_rows.append(
            np.hstack(
                (
                    labeled_crop(
                        opening,
                        (260, 150, 820, 770),
                        f"{behavior_id} open f0",
                        (360, 400),
                    ),
                    labeled_crop(
                        peak,
                        (260, 150, 820, 770),
                        cue_label,
                        (360, 400),
                    ),
                    labeled_crop(
                        closing,
                        (260, 150, 820, 770),
                        f"close f{frame_count - 1}",
                        (360, 400),
                    ),
                )
            )
        )
        brow_rows.append(
            np.hstack(
                (
                    labeled_crop(
                        opening,
                        (320, 300, 760, 505),
                        f"{behavior_id} open",
                        (440, 205),
                    ),
                    labeled_crop(
                        peak,
                        (320, 300, 760, 505),
                        cue_label,
                        (440, 205),
                    ),
                    labeled_crop(
                        closing,
                        (320, 300, 760, 505),
                        "close",
                        (440, 205),
                    ),
                )
            )
        )

    assets = report["assets"]
    report["summary"] = {
        "all_opening_handle_brow_cues_zero": all(
            item["opening_handle_brow_cues_zero"] for item in assets.values()
        ),
        "all_closing_handle_brow_cues_zero": all(
            item["closing_handle_brow_cues_zero"] for item in assets.values()
        ),
        "all_decoded_first_last_frames_exact": all(
            item["decoded_first_equals_last"] for item in assets.values()
        ),
        "assets_with_authored_brow_motion": [
            name
            for name, item in assets.items()
            if item["authored_brow_motion"]
        ],
    }

    json_path = OUTPUT / "ardy_eyebrow_audit.json"
    face_path = OUTPUT / "ardy_eyebrow_face_extrema.jpg"
    brow_path = OUTPUT / "ardy_eyebrow_closeup_extrema.jpg"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    cv2.imwrite(str(face_path), np.vstack(face_rows))
    cv2.imwrite(str(brow_path), np.vstack(brow_rows))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(json_path)
    print(face_path)
    print(brow_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
