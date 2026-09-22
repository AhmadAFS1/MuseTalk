#!/usr/bin/env python3
"""Certify one character's renders into a cross-cuttable pose bank.

MuseTalk switches poses at a clip boundary. If two clips do not end and begin on the exact
same decoded pixels, every switch shows a visible jump. The production banks therefore share
one `decoded_boundary_rgb_sha256` across every physical file, which is what proves all 30
ordered transitions between six clips are clean.

LTX will not hand that over on its own, so this script enforces it:

1. pick one anchor frame for the whole character;
2. overwrite the first and last `canonical_handle_frames_each_end` frames of every render
   with that anchor;
3. cross-fade the next `blend_frames_each_end` frames in and out of the organic motion, so
   the freeze does not read as a stutter;
4. re-encode to the delivery contract with no audio stream;
5. decode the result again and require the handles to be byte-identical across every render.

Step 5 is a hard gate. A bank that fails it must not be packaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import factory_common as fc


def decode_frames(path: Path, width: int, height: int) -> np.ndarray:
    """Decode the whole clip to RGB. A 241-frame 480x832 clip is about 289 MB in memory."""
    process = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    frame_bytes = width * height * 3
    if len(process.stdout) % frame_bytes:
        raise fc.FactoryError(f"{path}: decoded byte count is not a whole number of {width}x{height} frames")
    count = len(process.stdout) // frame_bytes
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(count, height, width, 3)


def encode_frames(frames: np.ndarray, destination: Path, *, fps: int, qp: int, keyframes: list[int]) -> None:
    """Encode with a fixed QP and forced keyframes on both handle boundaries.

    Fixed QP rather than CRF removes rate-control state from the equation: the head and tail
    handles are then quantised the same way, which is what lets identical input frames decode
    to identical output frames across separate files.
    """
    height, width = frames.shape[1], frames.shape[2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", str(qp),
        "-x264-params", "scenecut=0",
        # One expression, not one per keyframe: ffmpeg accepts a single `expr:` value.
        "-force_key_frames", "expr:" + "+".join(f"eq(n,{index})" for index in keyframes),
        str(destination),
    ]
    process = subprocess.run(command, input=frames.tobytes(), capture_output=True)
    if process.returncode != 0:
        raise fc.FactoryError(f"ffmpeg failed encoding {destination}: {process.stderr.decode()[-800:]}")


def apply_handles(frames: np.ndarray, anchor: np.ndarray, handle: int, blend: int) -> np.ndarray:
    """Freeze both ends on the anchor and cross-fade the organic motion in and out."""
    total = frames.shape[0]
    if total < 2 * (handle + blend) + 1:
        raise fc.FactoryError(
            f"Clip has {total} frames, too short for {handle} handle + {blend} blend frames at each end."
        )
    out = frames.astype(np.float32).copy()
    anchor_f = anchor.astype(np.float32)

    out[:handle] = anchor_f
    out[total - handle:] = anchor_f

    # Weights run strictly between the endpoints so the frame next to the handle is not a
    # second copy of the anchor, which would lengthen the freeze by one frame.
    for step in range(blend):
        weight = (step + 1) / (blend + 1)
        head = handle + step
        out[head] = anchor_f * (1.0 - weight) + out[head] * weight
        tail = total - handle - 1 - step
        out[tail] = anchor_f * (1.0 - weight) + out[tail] * weight

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def boundary_hash(frames: np.ndarray, handle: int) -> str:
    """Hash the decoded first and last `handle` frames — the bytes a cross-cut actually joins."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(frames[:handle]).tobytes())
    digest.update(np.ascontiguousarray(frames[-handle:]).tobytes())
    return digest.hexdigest()


def probe_contract(path: Path, contract: dict[str, Any], frame_count: int) -> list[str]:
    problems: list[str] = []
    streams = fc.ffprobe_streams(path)
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]

    if len(audio) != contract["audio_streams"]:
        problems.append(f"{len(audio)} audio streams; contract requires {contract['audio_streams']}")
    if len(video) != 1:
        problems.append(f"{len(video)} video streams; expected exactly 1")
        return problems

    stream = video[0]
    if (stream.get("width"), stream.get("height")) != (contract["width"], contract["height"]):
        problems.append(f"{stream.get('width')}x{stream.get('height')}; contract requires "
                        f"{contract['width']}x{contract['height']}")
    if stream.get("r_frame_rate") != f"{contract['fps']}/1":
        problems.append(f"frame rate {stream.get('r_frame_rate')}; contract requires {contract['fps']}/1")
    if stream.get("pix_fmt") != contract["pixel_format"]:
        problems.append(f"pixel format {stream.get('pix_fmt')}; contract requires {contract['pixel_format']}")
    if stream.get("codec_name") != contract["video_codec"]:
        problems.append(f"codec {stream.get('codec_name')}; contract requires {contract['video_codec']}")
    if int(stream.get("nb_read_frames") or 0) != frame_count:
        problems.append(f"{stream.get('nb_read_frames')} frames; contract requires {frame_count}")
    return problems


def certify_character(char_id: str, *, anchor_render: str | None, qp: int, force: bool) -> dict[str, Any]:
    fc.require_binaries("ffmpeg", "ffprobe")
    spec = fc.load_pose_spec()
    contract = spec["delivery_contract"]
    handle = contract["canonical_handle_frames_each_end"]
    blend = contract["blend_frames_each_end"]
    width, height, fps = contract["width"], contract["height"], contract["fps"]

    roster = fc.roster_index()
    if char_id not in roster:
        raise fc.FactoryError(f"Unknown character: {char_id}")
    ledger = fc.load_ledger()
    entry = fc.ledger_entry(ledger, char_id)
    rendered = entry.get("renders") or {}
    if not rendered:
        raise fc.FactoryError(f"{char_id} has no renders in the ledger. Run generate_pose_videos.py first.")

    render_specs = {r["render_key"]: r for r in spec["renders"]}
    order = [key for key in render_specs if key in rendered]
    anchor_key = anchor_render or ("idle_active_listening" if "idle_active_listening" in order else order[0])
    if anchor_key not in order:
        raise fc.FactoryError(f"Anchor render {anchor_key!r} was not rendered for {char_id}.")

    out_dir = fc.CERTIFIED_DIR / char_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # One anchor for the whole character: frame 0 of the anchor render. Every clip is frozen
    # onto these exact pixels, which is what makes them interchangeable at a cut.
    anchor_source = fc.FACTORY_ROOT / rendered[anchor_key]["file"]
    anchor = decode_frames(anchor_source, width, height)[0].copy()

    results: dict[str, Any] = {}
    problems: list[str] = []
    for render_key in order:
        source = fc.FACTORY_ROOT / rendered[render_key]["file"]
        destination = out_dir / f"{render_key}.mp4"
        expected_frames = render_specs[render_key]["frame_count"]

        if destination.exists() and not force:
            print(f"  reuse  {render_key}")
        else:
            frames = decode_frames(source, width, height)
            if frames.shape[0] != expected_frames:
                problems.append(f"{render_key}: LTX produced {frames.shape[0]} frames, expected {expected_frames}")
                continue
            treated = apply_handles(frames, anchor, handle, blend)
            encode_frames(
                treated, destination, fps=fps, qp=qp,
                # Keyframes on both handle boundaries so each handle decodes from a fresh IDR
                # rather than inheriting prediction state that differs between clips.
                keyframes=[0, expected_frames - handle],
            )
            print(f"  encode {render_key}  {expected_frames} frames")

        contract_problems = probe_contract(destination, contract, expected_frames)
        problems.extend(f"{render_key}: {issue}" for issue in contract_problems)

        decoded = decode_frames(destination, width, height)
        results[render_key] = {
            "file": f"certified/{render_key}.mp4",
            "frame_count": int(decoded.shape[0]),
            "duration_seconds": round(decoded.shape[0] / fps, 6),
            "video_sha256": fc.sha256_file(destination),
            "decoded_boundary_rgb_sha256": boundary_hash(decoded, handle),
            "source_render_sha256": rendered[render_key]["sha256"],
            "handles_internally_identical": bool(
                np.array_equal(decoded[:handle], np.repeat(decoded[:1], handle, axis=0))
                and np.array_equal(decoded[-handle:], np.repeat(decoded[-1:], handle, axis=0))
            ),
        }

    # The gate: every clip must present the same decoded handles, or a cut between two of
    # them will jump.
    hashes = {key: value["decoded_boundary_rgb_sha256"] for key, value in results.items()}
    shared = len(set(hashes.values())) == 1 if hashes else False
    if not shared:
        problems.append(
            "Certified clips do not share one decoded boundary hash, so cross-pose cuts will "
            "jump. Per-clip hashes: "
            + ", ".join(f"{k}={v[:12]}" for k, v in sorted(hashes.items()))
        )
    for key, value in results.items():
        if not value["handles_internally_identical"]:
            problems.append(f"{key}: its own handle frames are not identical after re-decode")

    report = {
        "schema_version": 1,
        "character_id": char_id,
        "display_name": roster[char_id]["display_name"],
        "anchor_render": anchor_key,
        "encoder": {"codec": "libx264", "qp": qp, "forced_keyframes": "0 and first tail handle frame"},
        "delivery_contract": contract,
        "renders": results,
        "shared_decoded_boundary_rgb_sha256": next(iter(hashes.values())) if shared else None,
        "ordered_transitions_checked": len(results) * (len(results) - 1),
        "passed": not problems,
        "problems": problems,
    }
    fc.write_json(out_dir / "validation_report.json", report)

    if problems:
        entry["stage"] = "certification_failed"
    else:
        entry["certified"] = {k: v for k, v in results.items()}
        entry["stage"] = "certified"
    fc.save_ledger(ledger)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", required=True)
    parser.add_argument("--anchor-render", help="Render key whose frame 0 becomes the anchor.")
    parser.add_argument("--qp", type=int, default=18, help="Fixed x264 QP. Lower is higher quality.")
    parser.add_argument("--force", action="store_true", help="Re-encode clips already certified.")
    args = parser.parse_args()

    failures = 0
    for char_id in args.characters:
        print(f"{char_id}:")
        report = certify_character(char_id, anchor_render=args.anchor_render, qp=args.qp, force=args.force)
        if report["passed"]:
            print(f"  PASS  {len(report['renders'])} clips, "
                  f"{report['ordered_transitions_checked']} ordered transitions share boundary "
                  f"{report['shared_decoded_boundary_rgb_sha256'][:16]}")
        else:
            failures += 1
            print(f"  FAIL  {len(report['problems'])} problems:")
            for problem in report["problems"]:
                print(f"        - {problem}")
    if failures:
        print(f"\n{failures} character(s) failed certification and must not be packaged.")
        print("If only the boundary-hash gate failed, try a lower --qp; fixed-QP encoding is what")
        print("makes identical input handles decode identically across separate files.")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
