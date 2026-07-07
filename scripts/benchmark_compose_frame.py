#!/usr/bin/env python3
"""Benchmark APIAvatar.compose_frame for a prepared avatar."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import musetalk.utils.blending as blending  # noqa: E402
from scripts.api_avatar import APIAvatar  # noqa: E402


def load_avatar(avatar_id: str, batch_size: int, version: str) -> APIAvatar:
    info_path = ROOT / "results" / version / "avatars" / avatar_id / "avator_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Prepared avatar info not found: {info_path}")
    info = json.loads(info_path.read_text())
    args = SimpleNamespace(version=version)
    return APIAvatar(
        avatar_id=avatar_id,
        video_path=info.get("talking_video_path") or info.get("input_video_path") or info.get("video_path", ""),
        idle_video_path=info.get("idle_video_path"),
        bbox_shift=int(info.get("bbox_shift", 0) or 0),
        batch_size=batch_size,
        vae=None,
        unet=None,
        pe=None,
        fp=None,
        args=args,
        preparation=False,
        force_recreate=False,
    )


def rebuild_plans(avatar: APIAvatar) -> None:
    if hasattr(avatar, "_build_compose_plan_cycle"):
        avatar._build_compose_plan_cycle()


def run_case(
    avatar: APIAvatar,
    *,
    fixed_point: bool,
    shrink_mask_bbox: bool,
    iters: int,
    warmup: int,
) -> dict:
    blending.MUSETALK_BLEND_FIXED_POINT = fixed_point
    blending.MUSETALK_BLEND_SHRINK_MASK_BBOX = shrink_mask_bbox
    rebuild_plans(avatar)
    rng = np.random.default_rng(1234)
    face = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    frame_count = max(1, len(getattr(avatar, "coord_list_cycle", []) or []))

    for idx in range(warmup):
        avatar.compose_frame(face, idx % frame_count)

    samples = []
    for idx in range(iters):
        started_at = time.perf_counter()
        avatar.compose_frame(face, idx % frame_count)
        samples.append(time.perf_counter() - started_at)

    samples_ms = [value * 1000 for value in samples]
    return {
        "fixed_point": fixed_point,
        "shrink_mask_bbox": shrink_mask_bbox,
        "iters": iters,
        "mean_ms": round(statistics.fmean(samples_ms), 4),
        "median_ms": round(statistics.median(samples_ms), 4),
        "p95_ms": round(sorted(samples_ms)[max(0, int(len(samples_ms) * 0.95) - 1)], 4),
        "max_ms": round(max(samples_ms), 4),
        "fps_single_worker": round(1000.0 / statistics.fmean(samples_ms), 2)
        if samples_ms
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatar-id", required=True)
    parser.add_argument("--version", default="v15")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    avatar = load_avatar(args.avatar_id, args.batch_size, args.version)
    baseline_result = run_case(
        avatar,
        fixed_point=False,
        shrink_mask_bbox=False,
        iters=args.iters,
        warmup=args.warmup,
    )
    fixed_result = run_case(
        avatar,
        fixed_point=True,
        shrink_mask_bbox=False,
        iters=args.iters,
        warmup=args.warmup,
    )
    shrink_result = run_case(
        avatar,
        fixed_point=True,
        shrink_mask_bbox=True,
        iters=args.iters,
        warmup=args.warmup,
    )
    speedup = baseline_result["mean_ms"] / shrink_result["mean_ms"] if shrink_result["mean_ms"] else 0.0
    report = {
        "avatar_id": args.avatar_id,
        "version": args.version,
        "frame_count": len(getattr(avatar, "frame_list_cycle", []) or []),
        "baseline_float_full_mask": baseline_result,
        "fixed_point_full_mask": fixed_result,
        "fixed_point_shrunk_mask": shrink_result,
        "mean_speedup": round(speedup, 4),
        "mean_reduction_pct": round((1.0 - 1.0 / speedup) * 100, 2) if speedup else 0.0,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
