#!/usr/bin/env python3
"""Record received WebRTC audio/video for the opt-in three-pose runtime.

Uses local WAV files and the existing avatar preparation API; no TTS provider.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import aiohttp
import test_pose_webrtc as helper
from scripts.motion_transitions import MotionBank, file_hash, atomic_json


def save(path, value):
    atomic_json(path, json.loads(json.dumps(value, default=str)))


STATUS_TIMEOUT_SECONDS = 5.0
CONTROL_TIMEOUT_SECONDS = 10.0
CLEANUP_TIMEOUT_SECONDS = 5.0


async def request_json(http, method, url, *, action, timeout_seconds, **kwargs):
    # Avatar preparation intentionally has a long timeout. Session polling and
    # control must not inherit it: a hung server otherwise defeats every local
    # turn deadline and prevents evidence from being finalized for 20 minutes.
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with http.request(method, url, timeout=timeout, **kwargs) as response:
        return await helper.require_json(response, action=action)


async def cleanup_step(action, operation, evidence, timeout_seconds=CLEANUP_TIMEOUT_SECONDS):
    try:
        await asyncio.wait_for(operation, timeout=timeout_seconds)
        return True
    except Exception as exc:
        evidence.setdefault("cleanup_errors", []).append({
            "action": action, "type": type(exc).__name__, "message": str(exc)})
        evidence["success"] = False
        return False


def pose_manifest(prefix):
    poses = {}
    for pose in helper.POSE_IDS:
        mode = {"speaking_direct": "talking", "light_smile": "smiling"}.get(pose, "idle")
        poses[pose] = {"avatar_id": f"{prefix}_{mode}", "asset_file": f"{mode}.mp4",
                       "role": "talking" if mode == "talking" else "idle" if pose == "neutral_resting" else "listening" if pose == "active_listening" else "reaction",
                       "fps": 24, "frame_count": 241, "duration_seconds": 241/24,
                       "cycle_seconds": 241/24}
    return {"version": 1, "pose_set_id": prefix, "test_only": True,
            "switch_safe": False, "default_pose_id": "neutral_resting",
            "switch_mode": "next_boundary", "poses": poses}


def speech_segments(smile=False):
    segments = [{"at_permille": 0, "pose_id": "speaking_direct"}]
    if smile == "late":
        segments += [{"at_permille": 935, "pose_id": "light_smile"},
                     {"at_permille": 985, "pose_id": "speaking_direct"}]
    elif smile:
        segments += [{"at_permille": 350, "pose_id": "light_smile"},
                     {"at_permille": 650, "pose_id": "speaking_direct"}]
    return segments


def validate_entry_abort_candidate(status, motion, first_output_frame):
    """Require proof of an emitted body bridge before any new speech starts."""
    displayed = motion.get("last_emitted") or {}
    assert displayed.get("mode") == "speech_entry_body_bridge", displayed
    assert 0 < displayed.get("progress", 0) < 1, displayed
    rows = [r for r in motion["trace"] if r["output_frame"] >= first_output_frame]
    assert not any(r.get("mode") == "live" for r in rows), "Entry abort missed: speech already displayed"
    sync = status["track_stats"]["sync_clock"]
    assert not sync["started"], "Entry abort missed: audio start gate already opened"
    for key in ("first_live_video_rtp_seconds", "first_tts_transport_pts_seconds",
                "first_audio_packet_unix_ms"):
        assert sync.get(key) is None, (key, sync.get(key))
    entry = motion["entries"][-1]
    assert entry["status"] == "playing", entry
    return {"generation_id": entry["generation_id"], "displayed": dict(displayed),
            "sync_clock": dict(sync)}


def validate_cancelled_entry(motion, first_output_frame, generation_id):
    rows = [r for r in motion["trace"] if r["output_frame"] >= first_output_frame]
    assert not any(r.get("mode") == "live" for r in rows), "Cancelled entry leaked a live frame"
    entries = [e for e in motion["entries"] if e["generation_id"] == generation_id]
    assert len(entries) == 1, entries
    entry = entries[0]
    assert entry["status"] == "cancelled" and entry["frames_emitted"] > 0, entry
    assert "first_live_generation_frame" not in entry, entry
    assert "first_live_output_frame" not in entry, entry
    return entry


async def record_case(http, args, poses, case, turns):
    params = {"avatar_id": poses["poses"]["neutral_resting"]["avatar_id"],
              "user_id": f"motion_{case}", "fps": args.fps, "playback_fps": args.fps,
              "batch_size": 8, "chunk_duration": 2, "pose_switch_mode": "next_boundary",
              "pose_set": json.dumps(helper.worker_pose_manifest(poses))}
    created = await request_json(http, "POST", f"{args.base_url}/webrtc/sessions/create?{urlencode(params)}",
        action="create motion session", timeout_seconds=30)
    sid = created["session_id"]
    pc = helper.RTCPeerConnection(configuration=helper.build_rtc_configuration_from_payload(created.get("ice_servers") or []))
    recorder = None
    clock, wrappers = helper.SharedRecordingClock(), {}
    connected, failed = asyncio.Event(), asyncio.Event()
    evidence = {"session_id": sid, "case": case, "fps": args.fps, "turns": [], "statuses": []}

    @pc.on("connectionstatechange")
    async def state_changed():
        if pc.connectionState == "connected": connected.set()
        if pc.connectionState in {"failed", "closed"}: failed.set()

    @pc.on("track")
    def got_track(track):
        wrapper = (helper.WallClockAudioTrack(track, clock) if track.kind == "audio"
                   else helper.WallClockVideoTrack(track, clock, nominal_fps=args.fps))
        wrappers[track.kind] = wrapper
        recorder.addTrack(wrapper)

    async def status():
        value = await request_json(http, "GET", f"{args.base_url}/webrtc/sessions/{sid}/status",
                                   action="motion status", timeout_seconds=STATUS_TIMEOUT_SECONDS)
        # Full source trace is retained at each turn boundary, not every poll.
        compact = json.loads(json.dumps(value))
        def strip_trace(obj):
            if isinstance(obj, dict):
                obj.pop("trace", None)
                for item in obj.values(): strip_trace(item)
            elif isinstance(obj, list):
                for item in obj: strip_trace(item)
        strip_trace(compact)
        evidence["statuses"].append({"at_seconds": clock.elapsed(), "status": compact})
        return value

    def motion(value):
        if isinstance(value, dict):
            if isinstance(value.get("motion"), dict): return value["motion"]
            for child in value.values():
                found = motion(child)
                if found: return found
        return None

    try:
        recorder = helper.RTPMP4Recorder(
            str(args.output / f"{case}.mp4"), video_fps=args.fps)
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        await asyncio.wait_for(helper.exchange_offer(http=http, base_url=args.base_url, session_id=sid,
            pc=pc, metrics=helper.SessionMetrics(session_id=sid), ice_gather_timeout_s=10), timeout=45)
        clock.start()
        await recorder.start()
        if not await helper.wait_for_peer_connection(pc=pc, connected_event=connected,
                failed_event=failed, timeout_s=60):
            raise RuntimeError("WebRTC connection failed")
        await asyncio.sleep(1)
        selected = motion(await status())
        assert selected, "Motion atlas did not match prepared sources"
        if args.atlas:
            expected = MotionBank(json.loads(args.atlas.read_text()))
            assert selected["bank"]["routing_sha256"] == expected.routing_sha256, "Wrong motion routing selected"
            evidence["bank"] = {"atlas": str(args.atlas.resolve()), "atlas_sha256": file_hash(args.atlas),
                                "routing_sha256": expected.routing_sha256,
                                "source_hashes": {p: v["sha256"] for p,v in expected.sources.items()}}
        evidence["selected_motion_sources"] = selected.get("bank", {}).get("sources")
        if case == "reactive-start":
            evidence["initial_pose_request"] = await request_json(http, "POST",
                f"{args.base_url}/webrtc/sessions/{sid}/pose",
                json={"pose_id": "light_smile", "effective": "next_boundary"},
                action="stage smiling idle", timeout_seconds=CONTROL_TIMEOUT_SECONDS)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                await asyncio.sleep(.1)
                current = await status()
                current_motion = motion(current)
                displayed = current_motion.get("last_emitted") or {}
                if (current["track_stats"]["video"]["current_pose_id"] == "light_smile"
                        and displayed.get("pose_id") == "light_smile"
                        and displayed.get("mode") == "idle"):
                    evidence["initial_pose_ready"] = current
                    evidence["initial_pose_ready_at_seconds"] = clock.elapsed()
                    break
            else:
                raise RuntimeError("Smiling idle pose did not become visible before speech")
        for index, (audio_name, smile, interrupt) in enumerate(turns):
            turn_id = f"{case}_{index}_{int(time.time())}"
            before_turn = motion(await status())
            turn = {"audio": audio_name, "turn_id": turn_id, "submitted_at_seconds": clock.elapsed(),
                    "first_possible_output_frame": (before_turn.get("last_emitted") or {}).get("output_frame", -1) + 1}
            # Keep partial-turn evidence if an assertion or network call fails.
            evidence["turns"].append(turn)
            entry_interrupt = interrupt == "entry"
            segments = speech_segments(smile)
            with (args.audio_dir / audio_name).open("rb") as audio:
                form = aiohttp.FormData()
                form.add_field("audio_file", audio, filename=audio_name, content_type="audio/wav")
                fields = {"reaction_intent": "none", "pose_id": "speaking_direct",
                          "pose_sequence": json.dumps(["speaking_direct", "neutral_resting"]),
                          "pose_plan": json.dumps({"version": 2, "clock": "audio_progress",
                              "segments": segments, "switch_mode": "next_boundary", "on_complete": "neutral_resting"}),
                          "turn_id": turn_id, "seq": str(index*10+1), "effective": "next_boundary",
                          "mouth_mode": "lip_sync", "audio_start": "immediate"}
                if args.legacy: fields.pop("pose_plan")
                for key, value in fields.items(): form.add_field(key, value)
                turn["accepted"] = await request_json(http, "POST",
                    f"{args.base_url}/webrtc/sessions/{sid}/stream", data=form,
                    action="submit motion speech", timeout_seconds=60)
            print(f"[{case}] submitted {audio_name}", flush=True)
            live_at = None
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                await asyncio.sleep(.02 if entry_interrupt else .2)
                current = await status()
                current_motion = motion(current)
                displayed = current_motion.get("last_emitted") or {}
                if displayed.get("mode") == "live" and live_at is None:
                    live_at = time.monotonic()
                    turn["live_observed_at_seconds"] = clock.elapsed()
                    turn["first_live_output_frame"] = displayed["output_frame"]
                if entry_interrupt and live_at and "interrupted_at_seconds" not in turn:
                    raise AssertionError("Missed the body-entry bridge: live speech started before abort")
                should_interrupt = (
                    displayed.get("mode") == "speech_entry_body_bridge"
                    and 0 < displayed.get("progress", 0) < 1
                ) if entry_interrupt else (
                    bool(interrupt and live_at)
                    and time.monotonic() - live_at >= (2.3 if interrupt is True else float(interrupt or 0))
                )
                if should_interrupt and "interrupted_at_seconds" not in turn:
                    if entry_interrupt:
                        turn["entry_abort_proof"] = validate_entry_abort_candidate(
                            current, current_motion, turn["first_possible_output_frame"])
                    abort_seq = index*10+2
                    if case == "barge-in":
                        user_turn_id = turn_id+"_new_user"
                        turn["barge_in"] = {"user_turn_id": user_turn_id,
                            "started": await request_json(http, "POST",
                                f"{args.base_url}/webrtc/sessions/{sid}/events",
                                json={"event": "user_speech_started", "seq": abort_seq,
                                      "turn_id": user_turn_id},
                                action="start user barge-in", timeout_seconds=CONTROL_TIMEOUT_SECONDS)}
                        assert turn["barge_in"]["started"].get("accepted"), turn["barge_in"]
                        owner = await status()
                        turn["barge_in"]["status_before_abort"] = owner
                        protocol = owner["pose_protocol"]
                        assert owner["active_stream"] == turn["accepted"]["request_id"], owner
                        assert protocol["active_turn_id"] == turn_id, protocol
                        assert protocol["user_speaking"] and protocol["assistant_active"], protocol
                        abort_seq += 1
                    turn["interrupted_at_seconds"] = clock.elapsed()
                    turn["motion_before_interrupt"] = current_motion
                    turn["interrupt_response"] = await request_json(http, "POST",
                        f"{args.base_url}/webrtc/sessions/{sid}/events",
                        json={"event": "assistant_turn_aborted", "seq": abort_seq, "turn_id": turn_id},
                        action="interrupt speech", timeout_seconds=CONTROL_TIMEOUT_SECONDS)
                    assert turn["interrupt_response"].get("accepted"), turn["interrupt_response"]
                started_or_aborted = live_at is not None or (entry_interrupt and "interrupted_at_seconds" in turn)
                if started_or_aborted and helper.is_stream_complete(current) and current_motion["settled"]:
                    turn["complete_at_seconds"] = clock.elapsed()
                    turn["final_status"] = current
                    break
            else:
                raise RuntimeError(f"Turn timed out in {case}")
            trace = motion(turn["final_status"])["trace"]
            live_rows = [r for r in trace if r.get("mode") == "live"
                         and r["output_frame"] >= turn["first_possible_output_frame"]]
            if entry_interrupt:
                turn["cancelled_entry"] = validate_cancelled_entry(
                    motion(turn["final_status"]), turn["first_possible_output_frame"],
                    turn["entry_abort_proof"]["generation_id"])
                sync = turn["final_status"]["track_stats"]["sync_clock"]
                assert sync.get("first_audio_packet_unix_ms") is None, sync
                assert sync.get("first_tts_transport_pts_seconds") is None, sync
            observed_poses = {r["pose_id"] for r in live_rows}
            if case == "reactive-start":
                assert live_rows[0]["generation_frame"] == 0, live_rows[0]
                assert live_rows[0]["pose_id"] == "light_smile", live_rows[0]
                assert motion(turn["final_status"])["entries"][index]["pose_id"] == "light_smile"
                assert live_rows[-1]["pose_id"] == "neutral_resting", live_rows[-1]
            if audio_name in {"short.wav", "silence.wav"}:
                assert observed_poses == {"neutral_resting"}, observed_poses
            elif not interrupt:
                assert "speaking_direct" in observed_poses, observed_poses
            if smile == "late":
                assert "light_smile" not in observed_poses, observed_poses
                assert live_rows[-1]["pose_id"] == "neutral_resting"
            elif smile:
                assert "light_smile" in observed_poses, observed_poses
            if interrupt and interrupt is not True and not entry_interrupt:
                interrupted = turn["motion_before_interrupt"]["last_emitted"]
                assert interrupted["pose_id"] == "speaking_direct", interrupted
                assert .3*args.fps <= interrupted["generation_frame"] < .6*args.fps, interrupted
            assert all(0 <= r["source_frame"] < poses["poses"][r["pose_id"]]["frame_count"] for r in live_rows)
            turn["observed_body_poses"] = sorted(observed_poses)
            if "barge_in" in turn:
                turn["barge_in"]["ended"] = await request_json(http, "POST",
                    f"{args.base_url}/webrtc/sessions/{sid}/events",
                    json={"event": "user_speech_ended", "seq": index*10+4,
                          "turn_id": turn["barge_in"]["user_turn_id"]},
                    action="finish user barge-in", timeout_seconds=CONTROL_TIMEOUT_SECONDS)
                assert turn["barge_in"]["ended"].get("accepted"), turn["barge_in"]
            print(f"[{case}] complete at {turn['complete_at_seconds']:.2f}s", flush=True)
            # Next turn begins as soon as the return is settled: back-to-back test.
            if index == len(turns)-1: await asyncio.sleep(1.2)
        evidence["final_status"] = await status()
        returns = motion(evidence["final_status"])["returns"]
        assert all(r["status"] == "completed" and r["total_seconds"] <= .5 for r in returns), returns
        final_motion = motion(evidence["final_status"])
        entries = final_motion.get("entries", [])
        assert len(entries) == len(turns), entries
        for entry, (_, _, interrupt) in zip(entries, turns):
            if interrupt == "entry":
                assert entry["status"] == "cancelled" and "first_live_generation_frame" not in entry, entry
            else:
                assert entry["status"] == "completed" and entry["first_live_generation_frame"] == 0, entry
        # Idle is allowed to advance slowly at non-native output fps, but must
        # never be held for inference prebuffer. Ignore intentional entry/return
        # blends, whose metadata names a different source-clock convention.
        longest_hold, hold, previous = 0, 0, None
        for row in final_motion["trace"]:
            key = (row["pose_id"], row["source_frame"]) if row["mode"] == "idle" else None
            hold = hold + 1 if key is not None and key == previous else 1
            longest_hold = max(longest_hold, hold)
            previous = key
        evidence["longest_idle_source_hold_frames"] = longest_hold
        assert longest_hold <= max(2, int(args.fps/24)+1), longest_hold
        evidence["rtc_stats"] = {key: vars(value) for key, value in (await pc.getStats()).items()}
        evidence["success"] = True
    except BaseException as exc:
        evidence["success"] = False
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if recorder is not None:
            await cleanup_step("stop recorder", recorder.stop(), evidence)
        evidence["recording_seconds"] = clock.elapsed()
        evidence["receiver_tracks"] = {key: value.get_stats() for key, value in wrappers.items()}
        timing_ok = set(wrappers) == {"audio", "video"} and all(stats["frames"] > 0
                        and stats["source_timestamp_anomalies"] == 0
                        and stats["source_timestamp_missing"] == 0
                        for stats in evidence["receiver_tracks"].values())
        evidence["timestamp_audit_passed"] = timing_ok
        if not timing_ok:
            evidence["success"] = False
        save(args.output / f"{case}.json", evidence)
        await cleanup_step("close receiver", pc.close(), evidence)
        await cleanup_step("delete server session", helper.delete_webrtc_session(http, args.base_url, sid), evidence)
        # Save before and after network cleanup, so an unavailable server cannot
        # prevent the primary failure, receiver timing, and partial turns being retained.
        save(args.output / f"{case}.json", evidence)
    assert not evidence.get("cleanup_errors"), "Recording cleanup failed; inspect saved evidence"
    assert timing_ok, "Received RTP timestamp discontinuity; inspect the recording evidence"


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    poses = json.loads(args.pose_set.read_text()) if args.pose_set else pose_manifest(args.avatar_prefix)
    if not args.atlas and args.pose_set:
        candidate = args.pose_set.parent / "motion-atlas.json"
        if candidate.exists(): args.atlas = candidate
    save(args.output / "pose-set.json", poses)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as http:
        prepared = await helper.ensure_six_avatars(http, base_url=args.base_url, pose_set=poses,
            asset_dir=args.asset_dir, prepare_missing=True, force_recreate=False, batch_size=8, warm_timeout=600)
        save(args.output / "avatar-preparation.json", prepared)
        cases = {"short-idle": [("short.wav", False, False)],
                 "long-talking-smiling": [("long.wav", True, False)],
                 "interrupted-and-next-turn": [("long.wav", False, True), ("short.wav", False, False)],
                 "barge-in": [("long.wav", False, True), ("short.wav", False, False)],
                 "late-smile": [("long.wav", "late", False)],
                 "entry-interruption": [("long.wav", False, "entry"), ("short.wav", False, False)],
                 "reactive-start": [("long.wav", False, False)],
                 "edge-cases": [("threshold.wav", False, False), ("silence.wav", False, False),
                                ("long.wav", False, .38), ("short.wav", False, False),
                                ("loop.wav", False, False)]}
        for name, turns in cases.items():
            if (not args.case and name not in {"edge-cases", "late-smile", "entry-interruption", "reactive-start", "barge-in"}) or args.case == name:
                await record_case(http, args, poses, name, turns)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--avatar-prefix", default="japanese_motion_20260925")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--pose-set", type=Path, help="Character package pose-set.json")
    p.add_argument("--atlas", type=Path, help="Expected runtime bank; inferred beside --pose-set when present")
    p.add_argument("--legacy", action="store_true", help="Exercise speech metadata without a v2 pose plan")
    p.add_argument("--case", choices=["short-idle", "long-talking-smiling", "interrupted-and-next-turn", "edge-cases", "late-smile", "entry-interruption", "reactive-start", "barge-in"])
    asyncio.run(main(p.parse_args()))
