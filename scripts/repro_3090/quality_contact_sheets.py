#!/usr/bin/env python3
"""CPU-only canonical quality review extraction; never a visual acceptance gate.

Default is a reproducible plan. --extract decodes existing review videos at their
native resolution, with no inference, audio classification, resize or new metric.
The full-frame MP4s are lossy review evidence, NOT the raw frames measured by e1.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_envelope as quality
import report as checks

FRAMES, FPS, WIDTH, HEIGHT = 240, 24, 512, 896
SCHEMA = "canonical_quality_review_extraction_v1"


def series(values, name):
    checks.require(isinstance(values, list) and len(values) == FRAMES, "missing per-frame metric series: " + name)
    return [quality.number(v, name) for v in values]


def largest(values, start=0, end=FRAMES, absolute=False):
    """Deterministic first-frame tie break; no quality threshold is inferred."""
    return max(range(start, end), key=lambda i: abs(values[i]) if absolute else values[i])


def select_frames(metrics, silence=None):
    reasons = {}
    def add(frame, reason):
        checks.require(type(frame) is int and 0 <= frame < FRAMES, "selection frame outside canonical clip")
        reasons.setdefault(frame, []).append(reason)
    for label, frame in (("start", 0), ("middle", FRAMES // 2), ("end", FRAMES - 1)):
        add(frame, {"selection": label, "basis": "fixed canonical 240-frame index"})
    for label, metric in metrics.items():
        for arm in ("A", "B"):
            values = series(metric["lip_sync"]["series"][arm + "_px"], "lip aperture")
            frame = largest(values)
            add(frame, {"selection": "peak_aperture", "run": label, "arm": arm, "metric": "lip_sync.series." + arm + "_px", "value": values[frame]})
        # The canonical tool saves these per-frame series, but not full-frame
        # PSNR/SSIM or landmark error series. Never invent a global worst frame.
        for name in ("band_mean", "ring_mean"):
            values = series(metric["seam"]["series"][name], "seam error")
            frame = largest(values)
            add(frame, {"selection": "largest_reported_pixel_error", "run": label, "metric": "seam.series." + name,
                        "value": values[frame], "scope": "this reported seam-region mean absolute pixel error, not global worst quality"})
        values = series(metric["chin"]["B"]["series_signed_error_px"], "chin error")
        frame = largest(values, 24, 216, absolute=True)
        add(frame, {"selection": "largest_reported_absolute_chin_error", "run": label, "metric": "chin.B.series_signed_error_px",
                    "signed_value": values[frame], "scope": "original chin gate window [24,216)"})
    if silence:
        checks.require(silence.get("review_method") == "reviewed_audio_intervals" and bool(silence.get("reviewer"))
                       and bool(silence.get("evidence_note")), "silence requires attributed audio review, not aperture inference")
        checks.require(0 < len(silence["intervals"]) <= 8, "silence interval count must be 1..8")
        for interval in silence["intervals"]:
            lo, hi = interval["start_frame"], interval["end_frame_exclusive"]
            checks.require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= FRAMES, "invalid reviewed silence interval")
            add((lo + hi - 1) // 2, {"selection": "reviewed_silence", "interval_frames": [lo, hi],
                                   "reviewer": silence["reviewer"], "basis": silence["evidence_note"]})
    checks.require(len(reasons) <= 40, "too many review selections")
    return [{"frame_index": i, "canonical_time_seconds": {"numerator": i, "denominator": FPS}, "reasons": reasons[i]} for i in sorted(reasons)]


def crop_union(metrics, region):
    boxes = [m["flicker"]["temporal_hf_power_gt_6hz"][region]["box_xyxy"] for m in metrics.values()]
    for box in boxes:
        checks.require(len(box) == 4 and all(type(x) is int for x in box)
                       and 0 <= box[0] < box[2] <= WIDTH and 0 <= box[1] < box[3] <= HEIGHT, "invalid canonical review crop")
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def fixture_binding(manifest, ident, filename):
    suffix = f"/avatar_diversity_20260927/{ident}/{filename}"
    matches = [row for row in manifest["files"] if ("/" + row["path"]).endswith(suffix)]
    checks.require(len(matches) == 1, "missing/duplicate frozen fixture: " + ident + "/" + filename)
    return matches[0]


def checked_file(path, expected, missing):
    path = Path(path).resolve()
    if not path.is_file():
        missing.append(str(path))
        return {"path": str(path), "status": "MISSING", "expected_sha256": expected["sha256"], "expected_bytes": expected["bytes"]}
    item = quality.evidence(path)
    checks.require(item["sha256"] == expected["sha256"] and item["bytes"] == expected["bytes"], "fixture SHA/byte mismatch: " + str(path))
    return {**item, "status": "VERIFIED", "expected_sha256": expected["sha256"], "expected_bytes": expected["bytes"]}


def probe_video(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
                             "-show_entries", "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
                             "-of", "json", str(path)], capture_output=True, check=True, text=True)
    data = json.loads(result.stdout)
    checks.require(len(data["streams"]) == 1, "review video needs one selected video stream")
    stream, frames = data["streams"][0], data["frames"]
    checks.require(stream["width"] == WIDTH and stream["height"] == HEIGHT
                   and Fraction(stream["avg_frame_rate"]) == FPS and len(frames) == FRAMES, "review video resolution/FPS/frame-count changed")
    timestamps = [Fraction(f["best_effort_timestamp_time"]) for f in frames]
    # ffprobe prints timestamps to six decimal places. This is timestamp-format
    # rounding only, never a quality/error tolerance or resampling operation.
    checks.require(all(abs(t - Fraction(i, FPS)) <= Fraction(1, 1000000) for i, t in enumerate(timestamps)),
                   "review video timestamps are not canonical frame-zero aligned 24fps")
    return {"width": WIDTH, "height": HEIGHT, "fps": "24/1", "frames": FRAMES,
            "frame_pts_seconds": [f["best_effort_timestamp_time"] for f in frames]}


def plan(runs, inputs, fixture_root, silence_path=None):
    checks.require(1 <= len(runs) <= 3, "supply one to three actual portable/native runs")
    inputs = Path(inputs).resolve()
    manifest = checks.read(inputs)
    checks.require(manifest["schema"] == "repro_3090_inputs_v1", "unknown frozen-input schema")
    silence = checks.read(silence_path) if silence_path else {"avatars": {}}
    if silence_path:
        checks.require(silence["schema"] == "canonical_audio_silence_v1" and silence["fps"] == FPS, "unknown silence evidence schema")
        checks.require(set(silence["avatars"]) <= set(checks.IDENTITIES), "unknown silence identity")
    loaded, missing = {}, []
    for run_path in runs:
        run_path = Path(run_path).resolve()
        run = checks.read(run_path)
        checks.require(run["schema"] == "repro_3090_v1" and run["suite"] == "quality" and run["status"] in ("PASS", "FAIL"), "incomplete quality run")
        checks.gpu_identity(run["environment"]["gpu"]["name"], run["environment"]["gpu"]["compute_cap"], False)
        checks.require(run["environment"]["input_manifest_sha256"] == checks.sha256(inputs), "run/input manifest hash mismatch")
        label = run["label"]
        checks.require(label not in loaded and label != "accepted_A" and label.isascii()
                       and label.replace("_", "").replace("-", "").isalnum(), "duplicate/unsafe run label")
        capture_dir = run_path.parent / (label + "_quality_capture")
        cap_path = capture_dir.with_suffix(".json")
        cap = checks.read(cap_path)
        checks.require(cap["label"] == label + "_quality_capture", "capture label differs from requested run")
        hashes = quality.validate_capture(cap)
        eng, taesd = run["environment"]["engines"][0], run["environment"]["taesd"]
        checks.loaded_backend(cap, eng["root"], taesd["key"], taesd["decoder_plan_sha256"], eng["manifest"])
        loaded[label] = {"run_report": quality.evidence(run_path), "capture_report": quality.evidence(cap_path), "capture_dir": capture_dir,
                         "pixel_hashes": hashes, "engine": eng, "taesd": taesd, "strict_wrapper_status": run["status"],
                         "declared_review_videos": cap["videos"]}
    avatars = {}
    for ident in checks.IDENTITIES:
        fixtures = {name: checked_file(Path(fixture_root) / ident / name, fixture_binding(manifest, ident, name), missing)
                    for name in ("source.mp4", "refined_raw.mp4", "speech.wav", "render.json")}
        metrics, columns, metric_evidence = {}, [], {}
        for label, info in loaded.items():
            metric_path = Path(info["run_report"]["path"]).parent / "quality_metrics" / f"{ident}__{label}.json"
            metric = checks.read(metric_path)
            quality.extract_avatar(metric)
            checks.require(metric["name"] == f"{ident}__{label}" and metric["identity"]["name"] == ident
                           and metric["identity"]["source_sha256"] == fixtures["source.mp4"]["expected_sha256"], "metric source identity mismatch")
            for field, name in (("cache_sha256", "cache.pt"), ("masks_sha256", "masks.npz"), ("source_landmarks_sha256", "source_landmarks.npy")):
                checks.require(metric["identity"][field] == fixture_binding(manifest, ident, name)["sha256"], "metric fixture identity mismatch")
            metrics[label] = metric
            metric_evidence[label] = quality.evidence(metric_path)
            tag = f"stream{checks.IDENTITIES.index(ident):02d}_{ident}"
            files = {}
            for suffix in ("refined.mp4", "faces.npz", "arrays.npz"):
                file = info["capture_dir"] / f"{tag}_{suffix}"
                checks.require(file.is_file(), "missing capture artifact: " + str(file))
                files[suffix] = quality.evidence(file)
            declared_video = [v for v in info["declared_review_videos"] if v["file"] == tag + "_refined.mp4"]
            checks.require(len(declared_video) == 1 and declared_video[0]["bytes"] == files["refined.mp4"]["bytes"]
                           and str(declared_video[0]["frames"]) == str(FRAMES), "saved review video differs from capture size/frame metadata")
            checks.require(quality.face_array_hash(files["faces.npz"]["path"]) == info["pixel_hashes"][ident][0][1], "saved generated faces differ from captured pixels")
            checks.require(Path(metric["B"]["faces"]).name == tag + "_faces.npz", "metric B does not reference this captured identity")
            columns.append({"label": label, "kind": "decoded_lossy_review_video", "files": files,
                            "recorded_raw_full_frame_sha256": info["pixel_hashes"][ident][0][0],
                            "raw_generated_face_sha256": info["pixel_hashes"][ident][0][1],
                            "metric_reconstructed_B_sha256": metric["B"]["info"]["frames_sha256"]})
        first = next(iter(metrics.values()))
        checks.require(all(m["A"]["info"]["frames_sha256"] == first["A"]["info"]["frames_sha256"] for m in metrics.values()), "accepted comparison A pixels differ across runs")
        if fixtures["render.json"]["status"] == "VERIFIED":
            accepted = checks.read(fixtures["render.json"]["path"])
            checks.require(accepted["raw_refined_sha256"] == first["A"]["info"]["frames_sha256"], "accepted fixture render metadata differs from metric A")
        columns.insert(0, {"label": "accepted_A", "kind": "decoded_lossy_review_video", "files": {"refined.mp4": fixtures["refined_raw.mp4"]},
                           "recorded_raw_full_frame_sha256": first["A"]["info"]["frames_sha256"]})
        note = silence["avatars"].get(ident)
        if note:
            checks.require(note["audio_sha256"] == fixtures["speech.wav"]["expected_sha256"], "silence annotation audio hash mismatch")
        avatars[ident] = {"fixtures": fixtures, "columns": columns, "metric_reports": metric_evidence,
                          "selections": select_frames(metrics, note), "crops_xyxy": {k: crop_union(metrics, k) for k in ("mouth_box", "jaw_box")},
                          "crop_basis": "union of canonical reported temporal-HF mouth/jaw rectangles across supplied runs, same rectangle for every arm; no tracking recomputation",
                          "silence_status": "REVIEWED_INTERVALS_SUPPLIED" if note else "MISSING_AUDIO_REVIEW",
                          "missing_per_frame_metrics": ["full-frame PSNR/SSIM", "tracked-final landmark deviation"],
                          "strict_original_verdicts": {label: metric["verdict"] for label, metric in metrics.items()}}
    for info in loaded.values():
        info["capture_dir"] = str(info["capture_dir"])
    return {"schema": SCHEMA, "status": "MISSING_SOURCES" if missing else "PLAN_READY", "inputs": quality.evidence(inputs),
            "helper": quality.evidence(__file__), "runs": loaded, "avatars": avatars, "missing_sources": missing,
            "silence_evidence": quality.evidence(silence_path) if silence_path else None,
            "source_hash_scope": "Copied canonical review fixtures and selected capture artifacts freshly hashed; models/calibration are not rehashed by this visualization utility",
            "visual_review_status": "NOT_PERFORMED", "release_ready": False,
            "limitations": ["Full-frame sheets decode existing lossy MP4s; they are not the raw frames used for e1 numerical gates",
                            "Raw generated-face NPZ pixel hashes are verified but no full-frame reconstruction or inference is performed",
                            "Silence is never inferred from small mouth aperture; absent audio review remains explicitly missing",
                            "Sparse contact samples cannot establish temporal quality, lip sync, full-clip acceptance or production-pose compatibility"]}


def png_bytes(width, height, rgb):
    checks.require(len(rgb) == width * height * 3, "RGB byte count mismatch")
    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff)
    rows = b"".join(b"\0" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def crop_rgb(rgb, width, height, xyxy):
    x0, y0, x1, y1 = xyxy
    checks.require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and len(rgb) == width * height * 3, "invalid RGB crop")
    return b"".join(rgb[(y * width + x0) * 3:(y * width + x1) * 3] for y in range(y0, y1))


def pair_rgb(images, width, height):
    checks.require(bool(images) and all(len(rgb) == width * height * 3 for rgb in images), "contact column dimensions differ")
    return b"".join(rgb[y * width * 3:(y + 1) * width * 3] for y in range(height) for rgb in images)


def decode_frames(path, indices):
    indices = sorted(set(indices))
    checks.require(indices and len(indices) <= 40 and all(type(i) is int and 0 <= i < FRAMES for i in indices), "invalid decode selection")
    select = "select=" + "+".join(f"eq(n\\,{i})" for i in indices)
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", select, "-vsync", "0",
                             "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
    stride = WIDTH * HEIGHT * 3
    checks.require(len(result.stdout) == len(indices) * stride, "decoded selected-frame byte count mismatch")
    return {index: result.stdout[i * stride:(i + 1) * stride] for i, index in enumerate(indices)}


def extract(result, out):
    checks.require(not result["missing_sources"], "cannot extract with missing canonical fixtures")
    out = Path(out)
    result["decoder"] = {name: subprocess.check_output([name, "-version"], text=True).splitlines()[0] for name in ("ffmpeg", "ffprobe")}
    pages = ["<!doctype html><meta charset='utf-8'><title>Canonical review samples</title>",
             "<style>body{font:16px system-ui;margin:24px}img{max-width:100%;height:auto}section{margin-bottom:3rem}code{white-space:pre-wrap}</style>",
             "<h1>Canonical quality review samples — NOT visually accepted</h1><p>Native pixel dimensions preserved in PNG files. Full frames decode lossy review MP4s; these are not raw e1 gate pixels. Silence and temporal/lip-sync review remain separate.</p>"]
    for ident, avatar in result["avatars"].items():
        directory = out / ident
        directory.mkdir()
        speech = avatar["fixtures"]["speech.wav"]
        checks.require(checks.sha256(speech["path"]) == speech["sha256"], "review audio changed after plan")
        shutil.copyfile(speech["path"], directory / "speech.wav")
        avatar["review_audio"] = quality.evidence(directory / "speech.wav")
        indices = [row["frame_index"] for row in avatar["selections"]]
        decoded = []
        for column in avatar["columns"]:
            file = column["files"]["refined.mp4"]
            checks.require(checks.sha256(file["path"]) == file["sha256"], "review video changed after plan")
            column["video_probe"] = probe_video(file["path"])
            decoded.append(decode_frames(file["path"], indices))
        pages.append("<h2>" + html.escape(ident) + "</h2><p>Columns: " + " | ".join(html.escape(c["label"]) for c in avatar["columns"]) + "</p>")
        pages.append("<p>Silence: " + avatar["silence_status"] + "</p>")
        pages.append(f"<p>Canonical speech, frame zero aligned at 24fps:</p><audio controls preload='none' src='{ident}/speech.wav'></audio>")
        for row in avatar["selections"]:
            index, outputs = row["frame_index"], {}
            images = [column[index] for column in decoded]
            row["decoded_frame_sha256"] = {c["label"]: hashlib.sha256(rgb).hexdigest() for c, rgb in zip(avatar["columns"], images)}
            row["observed_video_pts_seconds"] = {c["label"]: c["video_probe"]["frame_pts_seconds"][index] for c in avatar["columns"]}
            for name, box in {"full": [0, 0, WIDTH, HEIGHT], **avatar["crops_xyxy"]}.items():
                width, height = box[2] - box[0], box[3] - box[1]
                pixels = pair_rgb([crop_rgb(rgb, WIDTH, HEIGHT, box) for rgb in images], width, height)
                file = directory / f"frame_{index:03d}_{name}.png"
                file.write_bytes(png_bytes(width * len(images), height, pixels))
                outputs[name] = {**quality.evidence(file), "width": width * len(images), "height": height,
                                 "crop_xyxy_per_column": box, "columns": [c["label"] for c in avatar["columns"]]}
            row["outputs"] = outputs
            pages.append(f"<section><h3>Frame {index} ({index}/24 s)</h3><code>" + html.escape(json.dumps(row["reasons"], indent=2)) + "</code>")
            for name, file in outputs.items():
                relative = Path(file["path"]).relative_to(out.resolve()).as_posix()
                pages.append(f"<p>{html.escape(name)} — {file['width']}×{file['height']}</p><a href='{relative}'><img src='{relative}' alt='{html.escape(ident)} frame {index} {name}'></a>")
            pages.append("</section>")
    (out / "index.html").write_text("\n".join(pages) + "\n")
    result["index"] = quality.evidence(out / "index.html")
    result["status"] = "EXTRACTED_REVIEW_ONLY"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, action="append", type=Path, help="one to three actual quality report.json files")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--silence-evidence", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args(argv)
    checks.require(not args.out.exists(), "refusing to overwrite review evidence")
    args.out.mkdir(parents=True)
    try:
        result = plan(args.run, args.inputs, args.fixture_root, args.silence_evidence)
        if args.extract and not result["missing_sources"]:
            extract(result, args.out)
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "INVALID", "error_type": type(exc).__name__,
                  "visual_review_status": "NOT_PERFORMED", "release_ready": False}
        if isinstance(exc, checks.Invalid):
            result["reason"] = str(exc)
    (args.out / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out.resolve()), "status": result["status"], "visual_review_status": "NOT_PERFORMED"}))
    return 0 if result["status"] in ("PLAN_READY", "EXTRACTED_REVIEW_ONLY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
