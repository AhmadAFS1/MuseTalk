#!/usr/bin/env python3
"""Verify received recordings and build a portable normal-speed review page.

Diagnostics do not approve visual quality. This command never promotes a bank.
"""
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_transitions import MotionBank, atomic_json, file_hash


def find_motion(value):
    if isinstance(value, dict):
        if isinstance(value.get("motion"), dict):
            return value["motion"]
        for child in value.values():
            found = find_motion(child)
            if found is not None:
                return found
    return None


def verify_recording(path, bank):
    report = json.loads(path.read_text())
    video = path.with_suffix(".mp4")
    if not report.get("success") or not report.get("timestamp_audit_passed"):
        raise ValueError(f"Unsuccessful recording: {path}")
    selected = report.get("bank", {})
    if selected.get("routing_sha256") != bank.routing_sha256:
        raise ValueError(f"Recording uses a different motion bank: {path}")
    expected = {p: s["sha256"] for p, s in bank.sources.items()}
    if selected.get("source_hashes") != expected:
        raise ValueError(f"Recording source hashes differ: {path}")
    tracks = report["receiver_tracks"]
    if set(tracks) != {"audio", "video"} or any(
            t["frames"] <= 0 or t["source_timestamp_anomalies"] or t["source_timestamp_missing"]
            for t in tracks.values()):
        raise ValueError(f"Missing media or RTP discontinuity: {path}")
    motion = find_motion(report["final_status"])
    if motion["bank"]["routing_sha256"] != bank.routing_sha256:
        raise ValueError(f"Runtime bank changed during recording: {path}")
    for row in motion["trace"]:
        if not 0 <= row["source_frame"] < bank.count(row["pose_id"]):
            raise ValueError(f"Source frame outside its actual clip: {path}")
    returns = motion["returns"]
    if any(r["status"] != "completed" or r["total_seconds"] > .5 for r in returns):
        raise ValueError(f"Failed or slow recovery: {path}")
    entries = motion["entries"]
    for entry in entries:
        if entry["status"] == "cancelled" and report["case"] == "entry-interruption":
            turns = [t for t in report["turns"] if t.get("cancelled_entry", {}).get("generation_id") == entry["generation_id"]]
            if len(turns) != 1 or "first_live_generation_frame" in entry or not entry["frames_emitted"]:
                raise ValueError(f"Unproven entry cancellation: {path}")
            from scripts.test_webrtc_motion_transitions import validate_entry_abort_candidate, validate_cancelled_entry
            turn = turns[0]
            validate_cancelled_entry(find_motion(turn["final_status"]), turn["first_possible_output_frame"], entry["generation_id"])
            proof = turn["entry_abort_proof"]
            validate_entry_abort_candidate({"track_stats": {"sync_clock": proof["sync_clock"]}},
                                          turn["motion_before_interrupt"], turn["first_possible_output_frame"])
        elif entry["status"] != "completed" or entry.get("first_live_generation_frame") != 0:
            raise ValueError(f"Failed entry or missing initial speech frames: {path}")
    media = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]))
    if {s["codec_type"] for s in media["streams"]} != {"audio", "video"}:
        raise ValueError(f"Recording file lacks audio/video: {video}")
    return {"case": report["case"], "video": video.name,
            "video_sha256": file_hash(video), "evidence_sha256": file_hash(path),
            "duration_seconds": float(media["format"]["duration"]),
            "fps": report["fps"], "turn_count": len(report["turns"]),
            "return_seconds": [round(r["total_seconds"], 4) for r in returns],
            "entry_seconds": [round(e["additional_start_seconds"], 4) for e in entries if e["status"] == "completed"],
            "longest_idle_hold_frames": report["longest_idle_source_hold_frames"],
            "timestamp_anomalies": 0,
            "observed_poses": [t["observed_body_poses"] for t in report["turns"]]}


def build_review(directory, atlas, label):
    bank = MotionBank(json.loads(atlas.read_text()))
    for source in bank.sources.values():
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Source master changed after atlas construction")
    cases = []
    for path in sorted(directory.glob("*.json")):
        if not path.with_suffix(".mp4").exists():
            continue
        cases.append(verify_recording(path, bank))
    if not cases:
        raise ValueError("No received recordings found")
    result = {"version": 1, "label": label, "automated_validation_passed": True,
              "visual_acceptance": "pending", "atlas": str(atlas.resolve()),
              "atlas_sha256": file_hash(atlas), "routing_sha256": bank.routing_sha256,
              "source_hashes": {p: s["sha256"] for p, s in bank.sources.items()},
              "source_frame_counts": {p: bank.count(p) for p in bank.sources},
              "recordings": cases}
    atomic_json(directory/"verification.json", result)
    cards = []
    rows = []
    for case in cases:
        name = case["case"]
        escaped = html.escape(case["video"], quote=True)
        cards.append(f'<article><h2>{html.escape(name)}</h2>'
                     f'<video controls playsinline preload="metadata" src="{escaped}"></video>'
                     f'<p>{case["duration_seconds"]:.1f} seconds · {case["turn_count"]} turns</p></article>')
        rows.append(f'| [{name}]({case["video"]}) | {case["duration_seconds"]:.2f} s | '
                    + ", ".join(f'{n:.3f} s' for n in case["return_seconds"]) + ' |')
    (directory/"review.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(label)}</title><style>'
        'body{font:16px system-ui;background:#10151d;color:#f1f3f6;margin:32px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}'
        'article{background:#1b2430;padding:18px;border-radius:14px}h2{font-size:18px}'
        'video{display:block;width:100%;max-height:660px;background:#000}p{color:#bec9d7}</style>'
        f'<h1>{html.escape(label)}</h1><p>Actual received WebRTC audio and video. Watch at normal speed. '
        'Check speech entry, pose changes, mouth release, face outline, and interrupted returns. '
        'Automated checks passed; visual acceptance is pending.</p><main>' + ''.join(cards) + '</main>')
    (directory/"README.md").write_text(
        f'# {label}\n\nReceived WebRTC audio/video, without post-recording smoothing or dubbed audio. '
        'Automated validation passed; normal-speed visual acceptance is pending.\n\n'
        '[Open video review page](review.html). [Source-bound verification](verification.json).\n\n'
        '| Recording | Duration | Returns to idle |\n| --- | ---: | --- |\n' + '\n'.join(rows) + '\n\n'
        'Each JSON retains the actual emitted source phases, entry/return timing, and receiver timestamp audit. '
        'The bank and every video are hash-bound in verification.json. The pixel/transport checks do not prove '
        'that a viewer cannot notice a transition.\n')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = build_review(args.directory, args.atlas, args.label)
    print(json.dumps({"verified_recordings": len(result["recordings"]),
                      "visual_acceptance": result["visual_acceptance"]}))
