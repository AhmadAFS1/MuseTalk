#!/usr/bin/env python3
"""Verify received recordings and build a portable normal-speed review page.

Diagnostics do not approve visual quality. This command never promotes a bank.
"""
import argparse
import html
import json
import subprocess
import sys
from fractions import Fraction
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


def verify_barge_in(report):
    """Prove a new user ID did not steal the interrupted assistant's ownership."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"Unproven barge-in: {message}")

    turns = report.get("turns", [])
    require(len(turns) == 2, "expected interrupted speech and one following reply")
    interrupted, following = turns
    proof = interrupted.get("barge_in", {})
    assistant_id = interrupted.get("turn_id")
    user_id = proof.get("user_turn_id")
    next_id = following.get("turn_id")
    require(all(isinstance(value, str) and value for value in (assistant_id, user_id, next_id))
            and len({assistant_id, user_id, next_id}) == 3, "turn IDs must be present and distinct")
    started, aborted, ended = (proof.get("started", {}),
                               interrupted.get("interrupt_response", {}), proof.get("ended", {}))
    for response, event, turn_id in ((started, "user_speech_started", user_id),
                                     (aborted, "assistant_turn_aborted", assistant_id),
                                     (ended, "user_speech_ended", user_id)):
        require(response.get("accepted") is True and response.get("event") == event
                and response.get("turn_id") == turn_id, f"missing accepted {event} for its owner")
    accepted = interrupted.get("accepted", {})
    next_accepted = following.get("accepted", {})
    first_plan = accepted.get("pose_plan", {})
    next_plan = next_accepted.get("pose_plan", {})
    require(first_plan.get("accepted") is True and first_plan.get("turn_id") == assistant_id
            and next_plan.get("accepted") is True and next_plan.get("turn_id") == next_id,
            "stream acceptance does not bind both assistant turns")
    sequence = [first_plan.get("seq"), started.get("seq"), aborted.get("seq"),
                ended.get("seq"), next_plan.get("seq")]
    require(all(type(seq) is int for seq in sequence)
            and all(a < b for a, b in zip(sequence, sequence[1:])), "events are not strictly ordered")
    request_id, next_request_id = accepted.get("request_id"), next_accepted.get("request_id")
    require(request_id and next_request_id and request_id != next_request_id,
            "following reply must own a new stream request")
    owner = proof.get("status_before_abort", {})
    protocol = owner.get("pose_protocol", {})
    require(owner.get("active_stream") == request_id and protocol.get("active_turn_id") == assistant_id
            and protocol.get("user_speaking") is True and protocol.get("assistant_active") is True
            and protocol.get("last_event") == "user_speech_started"
            and protocol.get("last_seq") == started["seq"], "active assistant ownership changed during user speech")
    owner_motion = find_motion(owner) or {}
    displayed = owner_motion.get("last_emitted", {})
    require(displayed.get("mode") == "live", "user speech did not overlap emitted assistant speech")
    require(aborted.get("motion_return_started") is True, "assistant abort did not start motion recovery")
    require(aborted.get("pose_status", {}).get("user_speaking") is True,
            "assistant abort incorrectly ended the ongoing user speech")
    require(ended.get("pose_status", {}).get("user_speaking") is False,
            "user speech did not end before the following reply")

    motions = []
    for turn in turns:
        status = turn.get("final_status", {})
        motion = find_motion(status) or {}
        require("active_stream" in status and status["active_stream"] is None
                and status.get("track_stats", {}).get("video", {}).get("live_active") is False
                and status.get("pose_protocol", {}).get("assistant_active") is False
                and motion.get("settled") is True, "turn did not settle and release its stream")
        motions.append(motion)
    first_motion, next_motion = motions
    first_entries, entries = first_motion.get("entries", []), next_motion.get("entries", [])
    require(len(first_entries) == 1 and len(entries) == 2
            and entries[0]["generation_id"] == first_entries[0]["generation_id"]
            and entries[1]["generation_id"] != entries[0]["generation_id"], "following entry is not a new generation")
    first_returns = first_motion.get("returns", [])
    require(len(first_returns) == 1 and first_returns[0].get("status") == "completed",
            "interrupted turn has no completed return")
    anchor = first_returns[0].get("source_output_frame", -1)
    require(anchor >= displayed.get("output_frame", float("inf")), "return predates the user speech ownership proof")
    require(not any(row.get("mode") == "live" and row["output_frame"] > anchor
                    for row in first_motion.get("trace", [])), "cancelled speech continued after its return anchor")
    require(following.get("submitted_at_seconds", -1) >= interrupted.get("complete_at_seconds", float("inf")),
            "following reply began before the interrupted return settled")
    start_frame = following.get("first_possible_output_frame")
    require(type(start_frame) is int and start_frame > anchor, "following turn frame boundary is missing or stale")
    live = [row for row in next_motion.get("trace", [])
            if row.get("mode") == "live" and row["output_frame"] >= start_frame]
    require(live and live[0].get("generation_frame") == 0
            and live[0]["output_frame"] == entries[1].get("first_live_output_frame")
            and entries[1].get("first_live_generation_frame") == 0,
            "following reply did not start with its initial generation frame")
    require(all(a["generation_frame"] <= b["generation_frame"] for a, b in zip(live, live[1:])),
            "following reply contains stale or reordered generation frames")
    return {"assistant_turn_id": assistant_id, "user_turn_id": user_id,
            "following_turn_id": next_id, "interrupted_request_id": request_id,
            "following_request_id": next_request_id, "events_seq": sequence,
            "return_anchor_output_frame": anchor,
            "following_first_live_output_frame": live[0]["output_frame"],
            "following_first_live_generation_frame": 0}


def verify_saved_video_cadence(video, media, receiver, fps):
    """Decode the saved MP4's video PTS; RTP-only audits cannot prove mux cadence.

    Audio retains its own receiver origin. This check neither shifts it nor
    assumes that its first timestamp equals the video's first timestamp.
    """
    streams = [stream for stream in media.get("streams", []) if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"Recording must have exactly one video stream: {video}")
    try:
        time_base = Fraction(streams[0]["time_base"])
        nominal_fps = Fraction(str(fps))
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"Missing or invalid MP4 video time base/fps: {video}") from exc
    if time_base <= 0 or nominal_fps <= 0:
        raise ValueError(f"Non-positive MP4 video time base/fps: {video}")
    decoded = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=pts,pkt_pts", "-of", "json", str(video)], timeout=120))
    frames = decoded.get("frames", [])
    if not frames or len(frames) != receiver.get("frames"):
        raise ValueError(f"Saved MP4 decoded count {len(frames)} differs from receiver count "
                         f"{receiver.get('frames')}: {video}")
    # FFmpeg 4 names a decoded frame's original presentation stamp pkt_pts;
    # newer ffprobe names it pts. Never substitute generated best-effort PTS.
    points = [frame.get("pts", frame.get("pkt_pts")) for frame in frames]
    if any(type(point) is not int for point in points):
        raise ValueError(f"Saved MP4 contains a decoded frame without PTS: {video}")
    times = [point * time_base for point in points]
    deltas = [current - previous for previous, current in zip(times, times[1:])]
    expected, tolerance = 1 / nominal_fps, Fraction(1, 90_000)
    errors = [abs(delta - expected) for delta in deltas]
    for index, (delta, error) in enumerate(zip(deltas, errors), start=1):
        if delta <= 0 or error > tolerance:
            raise ValueError(f"Saved MP4 cadence error at decoded frame {index}: "
                             f"{float(delta):.9f}s, expected {float(expected):.9f}s "
                             f"within one 90 kHz tick: {video}")
    return {"decoded_frames": len(frames), "receiver_frames": receiver["frames"],
            "time_base": str(time_base), "first_pts": points[0], "last_pts": points[-1],
            "first_seconds": float(times[0]), "last_seconds": float(times[-1]),
            "expected_frame_seconds": float(expected), "tolerance_seconds": float(tolerance),
            "min_frame_seconds": float(min(deltas)) if deltas else None,
            "max_frame_seconds": float(max(deltas)) if deltas else None,
            "max_frame_error_seconds": float(max(errors, default=0)),
            "validated": True}


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
    barge_in = verify_barge_in(report) if report["case"] == "barge-in" else None
    media = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], timeout=120))
    if {s["codec_type"] for s in media["streams"]} != {"audio", "video"}:
        raise ValueError(f"Recording file lacks audio/video: {video}")
    saved_video_cadence = verify_saved_video_cadence(video, media, tracks["video"], report["fps"])
    return {"case": report["case"], "video": video.name,
            "video_sha256": file_hash(video), "evidence_sha256": file_hash(path),
            "duration_seconds": float(media["format"]["duration"]),
            "fps": report["fps"], "turn_count": len(report["turns"]),
            "return_seconds": [round(r["total_seconds"], 4) for r in returns],
            "entry_seconds": [round(e["additional_start_seconds"], 4) for e in entries if e["status"] == "completed"],
            "longest_idle_hold_frames": report["longest_idle_source_hold_frames"],
            "timestamp_anomalies": 0,
            "saved_video_cadence": saved_video_cadence,
            "observed_poses": [t["observed_body_poses"] for t in report["turns"]],
            **({"barge_in": barge_in} if barge_in is not None else {})}


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
