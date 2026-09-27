#!/usr/bin/env python3
"""Record one received idle-to-talking-to-idle WebRTC sentence.

This reuses the strict multi-pose recorder and the real avatar preparation API,
but accepts an arbitrary WAV instead of one of the fixed validation fixtures.
"""
import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import soundfile as sf

import test_pose_webrtc as helper
from scripts.motion_transitions import atomic_json
from scripts.test_webrtc_motion_transitions import record_case


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    poses = json.loads(args.pose_set.read_text())
    atomic_json(args.output / "pose-set.json", poses)
    args.audio_dir = args.audio.parent
    if args.speech_end_permille is not None:
        if not 1 <= args.speech_end_permille < 995:
            raise ValueError("--speech-end-permille must be between 1 and 994")
        args.pose_segments_override = [
            {"at_permille": 0, "pose_id": "speaking_direct"},
            {"at_permille": args.speech_end_permille, "pose_id": "empathetic_head_tilt"},
            {"at_permille": 995, "pose_id": "speaking_direct"},
        ]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as http:
        prepared = await helper.ensure_six_avatars(
            http,
            base_url=args.base_url,
            pose_set=poses,
            asset_dir=args.asset_dir,
            prepare_missing=True,
            force_recreate=False,
            batch_size=8,
            warm_timeout=600,
        )
        atomic_json(args.output / "avatar-preparation.json", prepared)
        await record_case(
            http,
            args,
            poses,
            "talking-only",
            [(args.audio.name, False, False)],
        )
        evidence_path = args.output / "talking-only.json"
        evidence = json.loads(evidence_path.read_text())
        source_frames = round(sf.info(str(args.audio)).duration * args.fps)
        timeline = evidence["turns"][0]["accepted"]["audio_timeline"]
        nominal_frames = (round(float(timeline["media_duration_seconds"]) * args.fps)
                          if args.expect_trimmed else source_frames)
        if args.expect_trimmed and not (0 < nominal_frames < source_frames):
            raise RuntimeError(f"Audio was not shortened: {timeline}")
        video = evidence["final_status"]["track_stats"]["video"]
        played_frames = int(video["frames_played"])
        dropped_frames = int(video["frames_dropped"])
        duplicated_frames = int(video["frames_duplicated"])
        delivered_frames = played_frames + duplicated_frames
        required_frames = int(video["last_required_live_output_frames"])
        if (abs(required_frames - nominal_frames) > 1
                or delivered_frames != required_frames or dropped_frames
                or (not args.expect_trimmed and played_frames != source_frames)
                or int(video["last_live_output_frames"]) != required_frames):
            reason = (
                f"Incomplete speech generation: delivered {delivered_frames}/"
                f"{required_frames} required frames, nominal {nominal_frames} "
                f"({played_frames} played, {duplicated_frames} boundary duplicates), "
                f"dropped {dropped_frames}"
            )
            evidence["success"] = False
            evidence["error"] = {"type": "IncompleteSpeechGeneration", "message": reason}
            atomic_json(evidence_path, evidence)
            raise RuntimeError(reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--pose-set", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--initial-idle-wait-seconds", type=float, default=1.0,
                        help="Idle playback time before submitting speech; useful for source-phase-matched A/B")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--speech-end-permille", type=int,
                        help="Switch to the idle-backed empathetic pose when speech ends within a padded WAV")
    parser.add_argument("--expect-trimmed", action="store_true",
                        help="Validate generated frames against the server's shortened audio timeline")
    parser.add_argument("--post-complete-idle-seconds", type=float, default=1.2,
                        help="Record natural idle after the speech turn; default keeps the transition test brief")
    asyncio.run(main(parser.parse_args()))
