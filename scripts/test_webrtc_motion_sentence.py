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

import test_pose_webrtc as helper
from scripts.motion_transitions import atomic_json
from scripts.test_webrtc_motion_transitions import record_case


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    poses = json.loads(args.pose_set.read_text())
    atomic_json(args.output / "pose-set.json", poses)
    args.audio_dir = args.audio.parent
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--pose-set", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--legacy", action="store_true")
    asyncio.run(main(parser.parse_args()))
