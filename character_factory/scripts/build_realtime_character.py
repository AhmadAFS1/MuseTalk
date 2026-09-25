#!/usr/bin/env python3
"""Create a reusable three-pose character package for the WebRTC motion runtime.

Accept an existing LTX trio or render one portrait with the approved recipe.
GPU rendering and the MuseTalk server must run in separate phases on this host.
Run --prepare-url only after the MuseTalk server has started.
"""
import argparse
import asyncio
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, atomic_json, file_hash
from character_factory.scripts.generate_three_pose_videos import (
    ACCEPTED_GRAPH_PATH, generation_fingerprint, verified_resume, write_json as render_json)


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


def validate_artifacts(source_dir, measurements, atlas, hashes):
    """Recheck cached diagnostics against actual bytes and decoded media.

    This is an integrity/route-availability gate, not perceptual approval.
    Frames are streamed to avoid retaining three decoded videos in memory.
    """
    import cv2
    bank = MotionBank(atlas)
    endpoints, dimensions, coverage = set(), set(), {}
    required = ("eye_mid_y_change_px", "eye_scale_change_pct", "eye_line_roll_deg", "lip_gap_px")
    for pose, name in ((IDLE, "idle"), (TALK, "talking"), (SMILE, "smiling")):
        measured = measurements["videos"][name]
        source = bank.sources[pose]
        if measured.get("sha256") != hashes[name] or source["sha256"] != hashes[name]:
            raise ValueError(f"{name}: cached measurements/atlas differ from source bytes; use a new version directory")
        rows = measured["rows"]
        for index, row in enumerate(rows):
            if row.get("frame") != index or any(not math.isfinite(float(row[key])) for key in required):
                raise ValueError(f"{name}: invalid measurement row {index}")
            if row["lip_gap_px"] < 0:
                raise ValueError(f"{name}: invalid lip measurement at {index}")
            for key in ("eye_mid_x_change_px", "eye_opening_px", "mouth_width_px"):
                if key in row and not math.isfinite(float(row[key])):
                    raise ValueError(f"{name}: invalid {key} at {index}")
        cap = cv2.VideoCapture(str(source_dir/(name+".mp4")))
        rate, count, first, last, shape = cap.get(cv2.CAP_PROP_FPS), 0, None, None, None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if shape is not None and shape != frame.shape:
                    raise ValueError(f"{name}: changing frame dimensions")
                shape = frame.shape
                if first is None:
                    first = hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
                last = frame
                count += 1
        finally:
            cap.release()
        if not count or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"{name}: source cannot be decoded")
        last_hash = hashlib.sha256(cv2.cvtColor(last, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        if (count != len(rows) or count != source["frame_count"] or count != measured["frame_count"]
                or not math.isclose(rate, bank.fps, abs_tol=1e-6)
                or not math.isclose(rate, float(measured["fps"]), abs_tol=1e-6)):
            raise ValueError(f"{name}: source/measurement/atlas timeline mismatch")
        if shape[:2] != (source["height"], source["width"]):
            raise ValueError(f"{name}: atlas dimensions differ from source")
        if first != measured["first_rgb_sha256"] or last_hash != measured["last_rgb_sha256"]:
            raise ValueError(f"{name}: measurement endpoint mismatch")
        endpoints.update((first, last_hash))
        dimensions.add(shape[:2])
        exits = atlas["edges"][pose][IDLE]
        covered = sum(bool(edge["admissible"]) for edge in exits)
        coverage[pose] = {"covered": covered, "total": count}
        if covered != count:
            raise ValueError(f"{name}: incomplete idle-return coverage ({covered}/{count}); cannot activate this pose")
        if pose != IDLE and not any(edge["admissible"] for edge in atlas["edges"][IDLE][pose]):
            raise ValueError(f"{name}: no admissible idle-to-pose entry; cannot activate this pose")
    if len(endpoints) != 1 or len(dimensions) != 1:
        raise ValueError("Sources do not share decoded endpoints and frame geometry")
    if atlas.get("exit_coverage") != coverage:
        raise ValueError("Atlas exit_coverage does not match its edges")
    return coverage


def atlas_content_hash(atlas):
    """Route integrity remains immutable when recorded-review status changes."""
    data = {key: value for key, value in atlas.items() if key not in ("status", "review")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    source_dir = args.source_dir.resolve() if args.source_dir else out / "sources"
    package_path = out / "character.json"
    existing = json.loads(package_path.read_text()) if package_path.exists() else {}
    if existing and existing["character_id"] != args.character_id:
        raise ValueError("Existing package has a different character ID; use a new output directory")
    fingerprint = None
    if args.image:
        adapted = None
        selected_pack = args.prompt_pack.resolve()
        if args.subject != "keep":
            adapted = adapt_subject(json.loads(args.prompt_pack.read_text()), args.subject)
            selected_pack = out / "identity-prompt-pack.json"
        selected_hash = (hashlib.sha256((json.dumps(adapted, indent=2, ensure_ascii=False)+"\n").encode()).hexdigest()
                         if adapted is not None else file_hash(selected_pack))
        fingerprint = generation_fingerprint(args.image.resolve(), selected_pack, ACCEPTED_GRAPH_PATH,
            guide_fit="center_crop", shared_anchor=True, prompt_pack_sha256=selected_hash)
        if existing.get("generation_fingerprint") not in (None, fingerprint):
            raise ValueError("Image or render inputs changed; use a new output directory")
        verified_resume(source_dir, fingerprint)
        # No output is changed until immutable generation inputs are verified.
        out.mkdir(parents=True, exist_ok=True)
        if adapted is not None:
            render_json(selected_pack, adapted)
        run([args.render_python, ROOT / "scripts/generate_three_pose_videos.py",
             "--image", args.image.resolve(), "--output-dir", source_dir,
             "--prompt-pack", selected_pack, "--guide-fit", "center_crop", "--shared-anchor"])
    hashes = {name: file_hash(source_dir/(name+".mp4")) for name in ("idle", "talking", "smiling")}
    if existing and existing["source_hashes"] != hashes:
        raise ValueError("Existing package has different source bytes; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    measurements = out / "source-measurements.json"
    if not measurements.exists():
        run([args.measurement_python, REPO / "scripts/measure_motion_sources.py",
             "--source-dir", source_dir, "--output", measurements])
    measurement_hash = file_hash(measurements)
    if existing.get("measurements_sha256") not in (None, measurement_hash):
        raise ValueError("Cached measurement report changed; use a new version directory")
    atlas_path = out / "motion-atlas.json"
    if not atlas_path.exists():
        run([args.runtime_python, REPO / "scripts/build_motion_atlas.py",
             "--source-dir", source_dir, "--measurements", measurements, "--output", atlas_path])
    atlas = json.loads(atlas_path.read_text())
    atlas_hash = atlas_content_hash(atlas)
    if existing.get("motion_atlas_content_sha256") not in (None, atlas_hash):
        raise ValueError("Cached atlas routes changed; use a new version directory")
    coverage = validate_artifacts(source_dir, json.loads(measurements.read_text()), atlas, hashes)
    registration = {"version": 1,
                    "source_hashes": {pose: atlas["sources"][pose]["sha256"] for pose in (IDLE,TALK,SMILE)},
                    "atlas_sha256": file_hash(atlas_path)}
    registration_path = out / "motion-registration.json"
    try:
        registered = json.loads(registration_path.read_text())
    except (OSError, ValueError):
        registered = None
    if registered != registration:
        atomic_json(registration_path, registration)
    pose_set = make_pose_set(args.character_id, atlas)
    atomic_json(out / "pose-set.json", pose_set)
    # This wire payload has the actual fields accepted by the session API.
    import test_pose_webrtc as helper
    atomic_json(out / "session-pose-set.json", helper.worker_pose_manifest(pose_set))
    package = dict(existing)
    package.update({"version": 1, "character_id": args.character_id,
               "status": atlas.get("status", "candidate_requires_recorded_review"), "source_dir": str(source_dir),
               "source_hashes": hashes, "motion_atlas": str(atlas_path),
               "pose_set": str(out / "session-pose-set.json"),
               "physical_avatar_count": 3, "public_pose_count": len(pose_set["poses"]),
               "measurements_sha256": measurement_hash, "motion_atlas_content_sha256": atlas_hash,
               "exit_coverage": coverage})
    if fingerprint is None and (source_dir / "manifest.json").exists():
        fingerprint = json.loads((source_dir / "manifest.json").read_text()).get("generation_fingerprint")
    if fingerprint is not None:
        package["generation_fingerprint"] = fingerprint
    if args.prepare_url:
        package["prepared"] = asyncio.run(prepare(args.prepare_url, pose_set, source_dir))
    atomic_json(package_path, package)
    print(json.dumps(package, indent=2))


if __name__ == "__main__": main()
