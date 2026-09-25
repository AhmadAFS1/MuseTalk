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


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


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


async def record_case(http, args, poses, case, turns):
    params = {"avatar_id": poses["poses"]["neutral_resting"]["avatar_id"],
              "user_id": f"motion_{case}", "fps": args.fps, "playback_fps": args.fps,
              "batch_size": 8, "chunk_duration": 2, "pose_switch_mode": "next_boundary",
              "pose_set": json.dumps(helper.worker_pose_manifest(poses))}
    async with http.post(f"{args.base_url}/webrtc/sessions/create?{urlencode(params)}") as response:
        created = await helper.require_json(response, action="create motion session")
    sid = created["session_id"]
    pc = helper.RTCPeerConnection(configuration=helper.build_rtc_configuration_from_payload(created.get("ice_servers") or []))
    recorder = helper.MediaRecorder(str(args.output / f"{case}.mp4"))
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
        async with http.get(f"{args.base_url}/webrtc/sessions/{sid}/status") as response:
            value = await helper.require_json(response, action="motion status")
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

    started = False
    try:
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        await helper.exchange_offer(http=http, base_url=args.base_url, session_id=sid,
            pc=pc, metrics=helper.SessionMetrics(session_id=sid), ice_gather_timeout_s=10)
        clock.start()
        await recorder.start()
        started = True
        if not await helper.wait_for_peer_connection(pc=pc, connected_event=connected,
                failed_event=failed, timeout_s=60):
            raise RuntimeError("WebRTC connection failed")
        await asyncio.sleep(1)
        assert motion(await status()), "Motion atlas did not match prepared sources"
        for index, (audio_name, smile, interrupt) in enumerate(turns):
            turn_id = f"{case}_{index}_{int(time.time())}"
            turn = {"audio": audio_name, "turn_id": turn_id, "submitted_at_seconds": clock.elapsed()}
            segments = [{"at_permille": 0, "pose_id": "speaking_direct"}]
            if smile:
                segments += [{"at_permille": 350, "pose_id": "light_smile"},
                             {"at_permille": 650, "pose_id": "speaking_direct"}]
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
                async with http.post(f"{args.base_url}/webrtc/sessions/{sid}/stream", data=form) as response:
                    turn["accepted"] = await helper.require_json(response, action="submit motion speech")
            print(f"[{case}] submitted {audio_name}", flush=True)
            live_at = None
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                await asyncio.sleep(.2)
                current = await status()
                current_motion = motion(current)
                displayed = current_motion.get("last_emitted") or {}
                if displayed.get("mode") == "live" and live_at is None:
                    live_at = time.monotonic()
                    turn["live_observed_at_seconds"] = clock.elapsed()
                    turn["first_live_output_frame"] = displayed["output_frame"]
                if interrupt and live_at and time.monotonic() - live_at >= (2.3 if interrupt is True else float(interrupt)) and "interrupted_at_seconds" not in turn:
                    turn["interrupted_at_seconds"] = clock.elapsed()
                    turn["motion_before_interrupt"] = current_motion
                    async with http.post(f"{args.base_url}/webrtc/sessions/{sid}/events",
                            json={"event": "assistant_turn_aborted", "seq": index*10+2, "turn_id": turn_id}) as response:
                        turn["interrupt_response"] = await helper.require_json(response, action="interrupt speech")
                if live_at and helper.is_stream_complete(current) and current_motion["settled"]:
                    turn["complete_at_seconds"] = clock.elapsed()
                    turn["final_status"] = current
                    break
            else:
                raise RuntimeError(f"Turn timed out in {case}")
            trace = motion(turn["final_status"])["trace"]
            live_rows = [r for r in trace if r.get("mode") == "live"
                         and r["output_frame"] >= turn["first_live_output_frame"]]
            observed_poses = {r["pose_id"] for r in live_rows}
            if audio_name in {"short.wav", "silence.wav"}:
                assert observed_poses == {"neutral_resting"}, observed_poses
            elif not interrupt:
                assert "speaking_direct" in observed_poses, observed_poses
            if smile:
                assert "light_smile" in observed_poses, observed_poses
            if interrupt and interrupt is not True:
                interrupted = turn["motion_before_interrupt"]["last_emitted"]
                assert interrupted["pose_id"] == "speaking_direct", interrupted
                assert .3*args.fps <= interrupted["generation_frame"] < .6*args.fps, interrupted
            assert all(0 <= r["source_frame"] < 241 for r in live_rows)
            turn["observed_body_poses"] = sorted(observed_poses)
            evidence["turns"].append(turn)
            print(f"[{case}] complete at {turn['complete_at_seconds']:.2f}s", flush=True)
            # Next turn begins as soon as the return is settled: back-to-back test.
            if index == len(turns)-1: await asyncio.sleep(1.2)
        evidence["final_status"] = await status()
        returns = motion(evidence["final_status"])["returns"]
        assert all(r["status"] == "completed" and r["total_seconds"] <= .5 for r in returns), returns
        evidence["rtc_stats"] = {key: vars(value) for key, value in (await pc.getStats()).items()}
        evidence["success"] = True
    finally:
        if started: await recorder.stop()
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
        await pc.close()
        await helper.delete_webrtc_session(http, args.base_url, sid)
    assert timing_ok, "Received RTP timestamp discontinuity; inspect the recording evidence"


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    poses = json.loads(args.pose_set.read_text()) if args.pose_set else pose_manifest(args.avatar_prefix)
    save(args.output / "pose-set.json", poses)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as http:
        prepared = await helper.ensure_six_avatars(http, base_url=args.base_url, pose_set=poses,
            asset_dir=args.asset_dir, prepare_missing=True, force_recreate=False, batch_size=8, warm_timeout=600)
        save(args.output / "avatar-preparation.json", prepared)
        cases = {"short-idle": [("short.wav", False, False)],
                 "long-talking-smiling": [("long.wav", True, False)],
                 "interrupted-and-next-turn": [("long.wav", False, True), ("short.wav", False, False)],
                 "edge-cases": [("threshold.wav", False, False), ("silence.wav", False, False),
                                ("long.wav", False, .38), ("short.wav", False, False),
                                ("loop.wav", False, False)]}
        for name, turns in cases.items():
            if (not args.case and name != "edge-cases") or args.case == name:
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
    p.add_argument("--legacy", action="store_true", help="Exercise speech metadata without a v2 pose plan")
    p.add_argument("--case", choices=["short-idle", "long-talking-smiling", "interrupted-and-next-turn", "edge-cases"])
    asyncio.run(main(p.parse_args()))
