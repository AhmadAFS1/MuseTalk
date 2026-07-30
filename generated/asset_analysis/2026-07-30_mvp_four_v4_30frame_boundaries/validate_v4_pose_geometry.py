#!/usr/bin/env python3
"""Measure v4 geometry and compare it with the v3 Indian-tutor pose set."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path("/workspace/MuseTalk")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.mmlab_compat import apply_mmlab_compat_patches

apply_mmlab_compat_patches()

from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.structures import merge_data_samples  # noqa: E402


ASSET_DIR = ROOT / "generated/downloads/indian_tutor_essential_six_v1"
OUTPUT_DIR = (
    ROOT
    / "generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries"
)
PREVIOUS_DIR = OUTPUT_DIR / "previous_runtime_assets"
CONFIG = ROOT / "musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
CHECKPOINT = ROOT / "models/dwpose/dw-ll_ucoco_384.pth"
BOUNDARY_FRAMES = 30

CURRENT = OrderedDict(
    (
        ("neutral", ASSET_DIR / "neutral_resting.mp4"),
        ("listener", ASSET_DIR / "active_listening.mp4"),
        ("speaking_new", ASSET_DIR / "speaking_direct.mp4"),
        ("smile", ASSET_DIR / "light_smile.mp4"),
    )
)
PREVIOUS = OrderedDict(
    (
        ("v3_neutral", PREVIOUS_DIR / "neutral_resting.mp4"),
        ("v3_listener", PREVIOUS_DIR / "active_listening.mp4"),
        ("v3_speaking", PREVIOUS_DIR / "speaking_direct.mp4"),
        ("v3_smile", PREVIOUS_DIR / "light_smile.mp4"),
    )
)
ALL = OrderedDict(
    (
        *CURRENT.items(),
        *PREVIOUS.items(),
    )
)

FACE_REGIONS = {
    "whole_face": np.arange(0, 68),
    "jaw": np.arange(0, 17),
    "brows": np.arange(17, 27),
    "eyes": np.arange(36, 48),
    "nose": np.arange(27, 36),
    "mouth": np.arange(48, 68),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No decoded frames in {path}")
    return frames


def similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    left, singular_values, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    denominator = float(np.sum(source_centered * source_centered))
    scale = float(np.sum(singular_values) / denominator) if denominator else 1.0
    return source_centered @ rotation * scale + target_mean


def regional_rmse(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    aligned = similarity_align(source, target)
    return {
        name: round(
            float(np.sqrt(np.mean(np.sum((aligned[indexes] - target[indexes]) ** 2, axis=1)))),
            6,
        )
        for name, indexes in FACE_REGIONS.items()
    }


def eye_aspect_ratio(face: np.ndarray, indexes: tuple[int, int, int, int, int, int]) -> float:
    p1, p2, p3, p4, p5, p6 = (face[index] for index in indexes)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def face_metrics(face: np.ndarray) -> dict[str, float]:
    outer_eye = float(np.linalg.norm(face[36] - face[45]))
    if outer_eye == 0:
        outer_eye = 1.0
    return {
        "outer_eye_distance_px": round(outer_eye, 6),
        "nose_width_over_outer_eye": round(float(np.linalg.norm(face[31] - face[35]) / outer_eye), 6),
        "nose_height_over_outer_eye": round(float(np.linalg.norm(face[27] - face[33]) / outer_eye), 6),
        "mouth_width_over_outer_eye": round(float(np.linalg.norm(face[48] - face[54]) / outer_eye), 6),
        "face_width_over_outer_eye": round(float(np.linalg.norm(face[0] - face[16]) / outer_eye), 6),
        "face_height_over_outer_eye": round(float(np.linalg.norm(face[8] - face[27]) / outer_eye), 6),
        "left_eye_aspect_ratio": round(eye_aspect_ratio(face, (36, 37, 38, 39, 40, 41)), 6),
        "right_eye_aspect_ratio": round(eye_aspect_ratio(face, (42, 43, 44, 45, 46, 47)), 6),
    }


def pixel_comparison(source: np.ndarray, target: np.ndarray) -> dict[str, float | bool]:
    if source.shape != target.shape:
        return {"exact": False, "shape_match": False}
    difference = source.astype(np.float32) - target.astype(np.float32)
    mse = float(np.mean(difference * difference))
    return {
        "exact": bool(np.array_equal(source, target)),
        "shape_match": True,
        "mae": round(float(np.mean(np.abs(difference))), 6),
        "rmse": round(float(np.sqrt(mse)), 6),
        "psnr_db": round(float(20.0 * np.log10(255.0 / np.sqrt(mse))), 6) if mse else float("inf"),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in ALL.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    videos = {name: read_frames(path) for name, path in ALL.items()}
    model = init_model(str(CONFIG), str(CHECKPOINT), device="cuda:0")
    detections: dict[tuple[str, int], dict[str, np.ndarray | float]] = {}

    def detect(name: str, index: int) -> dict[str, np.ndarray | float]:
        frame_count = len(videos[name])
        normalized_index = index if index >= 0 else frame_count + index
        key = (name, normalized_index)
        if key not in detections:
            merged = merge_data_samples(inference_topdown(model, videos[name][normalized_index]))
            keypoints = np.asarray(merged.pred_instances.keypoints)[0, :, :2].astype(np.float64)
            scores = np.asarray(merged.pred_instances.keypoint_scores)[0].astype(np.float64)
            detections[key] = {
                "face": keypoints[23:91],
                "body": keypoints[:23],
                "face_confidence": float(scores[23:91].mean()),
                "body_confidence": float(scores[:23].mean()),
            }
        return detections[key]

    boundary_indexes: dict[str, list[int]] = {}
    for name in CURRENT:
        count = len(videos[name])
        boundary_indexes[name] = [
            *range(BOUNDARY_FRAMES),
            *range(count - BOUNDARY_FRAMES, count),
        ]
        for index in boundary_indexes[name]:
            detect(name, index)
    for name in PREVIOUS:
        for index in (0, len(videos[name]) - 1):
            detect(name, index)

    first_faces = {name: np.asarray(detect(name, 0)["face"]) for name in ALL}
    last_faces = {name: np.asarray(detect(name, -1)["face"]) for name in ALL}
    neutral_face = first_faces["neutral"]
    first_frames = {name: videos[name][0] for name in ALL}
    last_frames = {name: videos[name][-1] for name in ALL}

    current_first_vs_neutral = {
        name: regional_rmse(face, neutral_face)
        for name, face in first_faces.items()
        if name in CURRENT
        if name != "neutral"
    }
    current_first_vs_neutral_pixel = {
        name: pixel_comparison(frame, first_frames["neutral"])
        for name, frame in first_frames.items()
        if name in CURRENT
        if name != "neutral"
    }
    previous_neutral_face = first_faces["v3_neutral"]
    previous_first_vs_neutral = {
        name: regional_rmse(face, previous_neutral_face)
        for name, face in first_faces.items()
        if name in PREVIOUS and name != "v3_neutral"
    }
    v4_to_v3 = {
        "neutral": "v3_neutral",
        "listener": "v3_listener",
        "speaking_new": "v3_speaking",
        "smile": "v3_smile",
    }
    v4_vs_v3_first_frame = {
        name: regional_rmse(first_faces[name], first_faces[previous_name])
        for name, previous_name in v4_to_v3.items()
    }
    v4_vs_v3_endpoint = {
        name: {
            "v4": regional_rmse(last_faces[name], first_faces[name]),
            "v3": regional_rmse(
                last_faces[previous_name],
                first_faces[previous_name],
            ),
        }
        for name, previous_name in v4_to_v3.items()
    }

    directed_transitions: dict[str, dict[str, object]] = {}
    for source_name in CURRENT:
        for target_name in CURRENT:
            if source_name == target_name:
                continue
            directed_transitions[f"{source_name}->{target_name}"] = {
                "face_rmse": regional_rmse(last_faces[source_name], first_faces[target_name]),
                "pixels": pixel_comparison(last_frames[source_name], first_frames[target_name]),
            }

    handle_variation: dict[str, dict[str, float]] = {}
    for name in CURRENT:
        first_reference = first_faces[name]
        opening = [
            regional_rmse(np.asarray(detect(name, index)["face"]), first_reference)["whole_face"]
            for index in range(BOUNDARY_FRAMES)
        ]
        closing = [
            regional_rmse(np.asarray(detect(name, index)["face"]), last_faces[name])["whole_face"]
            for index in range(
                len(videos[name]) - BOUNDARY_FRAMES,
                len(videos[name]),
            )
        ]
        handle_variation[name] = {
            "opening_median_whole_face_rmse_px": round(float(np.median(opening)), 6),
            "opening_max_whole_face_rmse_px": round(float(np.max(opening)), 6),
            "closing_median_whole_face_rmse_px": round(float(np.median(closing)), 6),
            "closing_max_whole_face_rmse_px": round(float(np.max(closing)), 6),
        }

    neutral_body = np.asarray(detect("neutral", 0)["body"])
    neutral_nose = neutral_body[0]
    neutral_shoulders = (neutral_body[5] + neutral_body[6]) / 2.0
    neutral_shoulder_width = float(np.linalg.norm(neutral_body[5] - neutral_body[6]))
    body_offsets = {}
    for name in CURRENT:
        body = np.asarray(detect(name, 0)["body"])
        shoulder_center = (body[5] + body[6]) / 2.0
        shoulder_width = float(np.linalg.norm(body[5] - body[6]))
        body_offsets[name] = {
            "nose_center_delta_px": np.round(body[0] - neutral_nose, 6).tolist(),
            "shoulder_center_delta_px": np.round(shoulder_center - neutral_shoulders, 6).tolist(),
            "shoulder_width_delta_px": round(shoulder_width - neutral_shoulder_width, 6),
        }

    decoded_hashes = {}
    for name, path in ALL.items():
        decoded_hashes[name] = {
            "file_sha256": sha256_file(path),
            "frame_count": len(videos[name]),
            "first_frame_sha256": hashlib.sha256(first_frames[name].tobytes()).hexdigest(),
            "last_frame_sha256": hashlib.sha256(last_frames[name].tobytes()).hexdigest(),
            "first_equals_last": bool(np.array_equal(first_frames[name], last_frames[name])),
        }

    result = {
        "schema_version": 1,
        "method": {
            "model": "MuseTalk DWPose 68-point face landmarks",
            "alignment": "least-squares similarity transform removing translation, rotation, and scale",
            "boundary_frames_sampled": BOUNDARY_FRAMES,
            "regions": {name: indexes.tolist() for name, indexes in FACE_REGIONS.items()},
        },
        "decoded_assets": decoded_hashes,
        "first_frame_face_metrics": {
            name: face_metrics(face) for name, face in first_faces.items()
        },
        "first_frame_confidence": {
            name: round(float(detect(name, 0)["face_confidence"]), 6) for name in ALL
        },
        "v4_first_frame_vs_v4_neutral_face_rmse_px": current_first_vs_neutral,
        "v4_first_frame_vs_v4_neutral_pixels": current_first_vs_neutral_pixel,
        "v3_first_frame_vs_v3_neutral_face_rmse_px": previous_first_vs_neutral,
        "v4_vs_v3_first_frame_face_rmse_px": v4_vs_v3_first_frame,
        "v4_vs_v3_endpoint_face_rmse_px": v4_vs_v3_endpoint,
        "current_pose_body_offsets_vs_neutral": body_offsets,
        "current_pose_handle_variation": handle_variation,
        "current_directed_transitions": directed_transitions,
        "exact_current_first_frame_pair_count": sum(
            int(np.array_equal(first_frames[source], first_frames[target]))
            for source_index, source in enumerate(CURRENT)
            for target_index, target in enumerate(CURRENT)
            if source_index < target_index
        ),
        "exact_current_directed_transition_count": sum(
            int(np.array_equal(last_frames[source], first_frames[target]))
            for source in CURRENT
            for target in CURRENT
            if source != target
        ),
    }
    output_json = OUTPUT_DIR / "v4_segmind_geometry_validation.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    labels = list(ALL)
    resized = []
    for name in labels:
        image = cv2.resize(first_frames[name], (288, 512), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (288, 42), (0, 0, 0), -1)
        cv2.putText(
            image,
            name,
            (8, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        resized.append(image)
    cv2.imwrite(
        str(OUTPUT_DIR / "v4_and_v3_first_frames.jpg"),
        np.hstack(resized),
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
