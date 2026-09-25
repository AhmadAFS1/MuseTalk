#!/usr/bin/env python3
"""Create a reusable three-pose character package for the WebRTC motion runtime.

Accept an existing LTX trio or render one portrait with the approved recipe.
GPU rendering and the MuseTalk server must run in separate phases on this host.
Run --prepare-url only after the MuseTalk server has started.
"""
import argparse
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from scripts.motion_transitions import file_hash

LTX_PYTHON = "/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python"
MUSE_PYTHON = "/workspace/.venvs/musetalk_trt_stagewise/bin/python"


def run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run([str(v) for v in command], cwd=REPO, check=True)


def adapt_subject(pack, subject):
    """Change identity nouns/pronouns only; preserve all motion constraints."""
    pack = json.loads(json.dumps(pack))
    if subject == "keep":
        return pack
    terms = {"man": {"female": "male", "She": "He", "she": "he", "Her": "His", "her": "his"},
             "person": {"female ": "", "She": "The tutor", "she": "the tutor", "Her": "The tutor's", "her": "the tutor's"}}[subject]
    for pose in pack["poses"].values():
        prompt = pose["positive_prompt"]
        for old, new in terms.items():
            prompt = re.sub(r"\b"+re.escape(old)+r"\b", lambda match: new, prompt)
        pose["positive_prompt"] = prompt
    pack["pack_id"] += "_"+subject
    pack["approval_status"] = "identity_adaptation_requires_review"
    pack["identity_adaptation"] = subject
    return pack


def make_pose_set(character_id, atlas):
    from scripts.pose_protocol import POSE_IDS
    logical = {}
    for pose in POSE_IDS:
        physical = pose if pose in atlas["sources"] else "neutral_resting"
        item = atlas["sources"][physical]
        name = {"neutral_resting": "idle", "speaking_direct": "talking", "light_smile": "smiling"}[physical]
        logical[pose] = {"avatar_id": f"{character_id}_{name}_{item['sha256'][:10]}",
                         "asset_file": name+".mp4", "fps": atlas["fps"],
                         "frame_count": item["frame_count"],
                         "duration_seconds": item["frame_count"]/atlas["fps"],
                         "cycle_seconds": item["frame_count"]/atlas["fps"],
                         "role": "idle" if pose == "neutral_resting" else "talking" if pose == "speaking_direct" else "listening" if pose == "active_listening" else "reaction"}
    return {"version": 1, "pose_set_id": character_id, "test_only": True, "switch_safe": False,
            "default_pose_id": "neutral_resting", "switch_mode": "next_boundary", "poses": logical}


async def prepare(url, pose_set, source_dir):
    import aiohttp
    import test_pose_webrtc as helper
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as http:
        return await helper.ensure_six_avatars(http, base_url=url.rstrip("/"), pose_set=pose_set,
            asset_dir=source_dir, prepare_missing=True, force_recreate=False, batch_size=8, warm_timeout=600)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--source-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--character-id", required=True)
    p.add_argument("--prompt-pack", type=Path, default=ROOT / "config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json")
    p.add_argument("--measurement-python", default=LTX_PYTHON)
    p.add_argument("--render-python", default=LTX_PYTHON)
    p.add_argument("--runtime-python", default=MUSE_PYTHON)
    p.add_argument("--subject", choices=["keep", "man", "person"], default="keep",
                   help="Only adapt identity nouns/pronouns; motion instructions remain fixed")
    p.add_argument("--prepare-url", help="Optional running local MuseTalk API URL")
    args = p.parse_args()
    if args.image and args.prepare_url:
        p.error("Render first with MuseTalk stopped, then use --source-dir and --prepare-url after starting MuseTalk")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.character_id):
        p.error("character-id must contain 1–64 letters, numbers, underscores, or hyphens")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir.resolve() if args.source_dir else out / "sources"
    package_path = out / "character.json"
    if args.image:
        selected_pack = args.prompt_pack.resolve()
        if args.subject != "keep":
            selected_pack = out / "identity-prompt-pack.json"
            selected_pack.write_text(json.dumps(adapt_subject(json.loads(args.prompt_pack.read_text()), args.subject), indent=2)+"\n")
        run([args.render_python, ROOT / "scripts/generate_three_pose_videos.py",
             "--image", args.image.resolve(), "--output-dir", source_dir,
             "--prompt-pack", selected_pack, "--guide-fit", "center_crop", "--shared-anchor"])
    hashes = {name: file_hash(source_dir/(name+".mp4")) for name in ("idle", "talking", "smiling")}
    if package_path.exists():
        existing = json.loads(package_path.read_text())
        if existing["source_hashes"] != hashes or existing["character_id"] != args.character_id:
            raise ValueError("Existing package has different identity/sources; use a new output directory")
    measurements = out / "source-measurements.json"
    if not measurements.exists():
        run([args.measurement_python, REPO / "scripts/measure_motion_sources.py",
             "--source-dir", source_dir, "--output", measurements])
    atlas_path = out / "motion-atlas.json"
    if not atlas_path.exists():
        run([args.runtime_python, REPO / "scripts/build_motion_atlas.py",
             "--source-dir", source_dir, "--measurements", measurements, "--output", atlas_path])
    atlas = json.loads(atlas_path.read_text())
    for pose,name in (("neutral_resting","idle"),("speaking_direct","talking"),("light_smile","smiling")):
        if atlas["sources"][pose]["sha256"] != hashes[name]:
            raise ValueError("Package atlas is stale; use a new version directory")
    pose_set = make_pose_set(args.character_id, atlas)
    (out / "pose-set.json").write_text(json.dumps(pose_set, indent=2)+"\n")
    # This wire payload has the actual fields accepted by the session API.
    import test_pose_webrtc as helper
    (out / "session-pose-set.json").write_text(json.dumps(helper.worker_pose_manifest(pose_set), indent=2)+"\n")
    package = {"version": 1, "character_id": args.character_id,
               "status": "candidate_requires_recorded_review", "source_dir": str(source_dir),
               "source_hashes": hashes, "motion_atlas": str(atlas_path),
               "pose_set": str(out / "session-pose-set.json"),
               "physical_avatar_count": 3, "public_pose_count": len(pose_set["poses"]),
               "exit_coverage": atlas["exit_coverage"]}
    if args.prepare_url:
        package["prepared"] = asyncio.run(prepare(args.prepare_url, pose_set, source_dir))
    package_path.write_text(json.dumps(package, indent=2)+"\n")
    print(json.dumps(package, indent=2))


if __name__ == "__main__": main()
