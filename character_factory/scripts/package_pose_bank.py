#!/usr/bin/env python3
"""Package a certified character into a MuseTalk pose bank and runtime manifest.

Writes two things, both outside this factory directory because they are what MuseTalk loads:

  assets/ltx23_pose_banks/<bank_dir>/   immutable asset package + provenance
  configs/pose_test/<pose_set_id>.json  runtime manifest read by the pose lab, wall, and
                                        headless WebRTC harness

The runtime manifest is validated here against the same rules as
scripts/test_pose_webrtc.py::load_pose_set, so a manifest that would be rejected at session
creation is rejected at packaging time instead.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import factory_common as fc

ROLE_BY_POSE = {
    "neutral_resting": "idle",
    "active_listening": "listening",
    "speaking_direct": "talking",
    "nod_agree": "reaction",
    "empathetic_head_tilt": "reaction",
    "light_smile": "reaction",
}


def resolve_pose_renders(certified: dict[str, Any], spec: dict[str, Any]) -> dict[str, str]:
    """Map each of the six logical pose IDs onto a certified render key.

    A tier that skipped a reaction render aliases it onto the idle loop, so the wire contract
    still exposes exactly six IDs. That is a visible quality compromise, not a silent one:
    package_pose_bank reports every alias it had to invent.
    """
    aliases = spec["logical_aliases"]
    mapping: dict[str, str] = {}
    for pose_id in fc.POSE_IDS:
        preferred = aliases.get(pose_id, pose_id)
        if preferred in certified:
            mapping[pose_id] = preferred
            continue
        if pose_id == "speaking_direct":
            for candidate in ("speaking_direct_v14_subtle", "speaking_direct_v15_reference_paced"):
                if candidate in certified:
                    mapping[pose_id] = candidate
                    break
            else:
                raise fc.FactoryError("No certified speaking_direct render; the bank is unusable.")
            continue
        fallback = aliases["neutral_resting"]
        if fallback not in certified:
            raise fc.FactoryError(f"No certified render for {pose_id} and no idle loop to alias onto.")
        mapping[pose_id] = fallback
    return mapping


def validate_runtime_manifest(manifest: dict[str, Any]) -> None:
    """Mirror of scripts/test_pose_webrtc.py::load_pose_set."""
    if manifest.get("version") != 1:
        raise fc.FactoryError("Runtime manifest must be a version 1 object.")
    if manifest.get("default_pose_id") != "neutral_resting":
        raise fc.FactoryError("default_pose_id must be neutral_resting.")
    if manifest.get("switch_mode") != "next_boundary":
        raise fc.FactoryError("switch_mode must be next_boundary.")
    poses = manifest.get("poses")
    if not isinstance(poses, dict) or set(poses) != set(fc.POSE_IDS):
        raise fc.FactoryError("Runtime manifest must contain exactly: " + ", ".join(fc.POSE_IDS))
    for pose_id, entry in poses.items():
        if not str(entry.get("avatar_id") or "").strip():
            raise fc.FactoryError(f"{pose_id} requires avatar_id.")
        asset_file = str(entry.get("asset_file") or "")
        if not asset_file or Path(asset_file).name != asset_file:
            raise fc.FactoryError(f"{pose_id}.asset_file must be a plain filename.")
        variants = entry.get("variants")
        if variants is None:
            continue
        if pose_id != "speaking_direct":
            raise fc.FactoryError("Only speaking_direct may define a variants array.")
        for variant in variants:
            for field in ("variant_id", "avatar_id", "asset_file"):
                if not str(variant.get(field) or "").strip():
                    raise fc.FactoryError(f"speaking_direct variants require {field}.")
            if Path(variant["asset_file"]).name != variant["asset_file"]:
                raise fc.FactoryError("speaking_direct variant asset_file must be a plain filename.")


def package(char_id: str, *, force: bool, activation_status: str) -> dict[str, Any]:
    spec = fc.load_pose_spec()
    roster = fc.roster_index()
    if char_id not in roster:
        raise fc.FactoryError(f"Unknown character: {char_id}")
    character = roster[char_id]

    ledger = fc.load_ledger()
    entry = fc.ledger_entry(ledger, char_id)
    certified = entry.get("certified") or {}
    if not certified:
        raise fc.FactoryError(f"{char_id} is not certified. Run certify_pose_bank.py first.")

    report_path = fc.CERTIFIED_DIR / char_id / "validation_report.json"
    report = fc.load_json(report_path)
    if not report.get("passed"):
        raise fc.FactoryError(f"{char_id} has a failing validation report; refusing to package.")

    bank_dir = fc.POSE_BANK_ROOT / character["bank_dir"]
    if bank_dir.exists() and not force:
        raise fc.FactoryError(
            f"{bank_dir} already exists. Pose banks are immutable by convention; pass --force "
            "only if you are deliberately replacing an unused bank."
        )
    (bank_dir / "certified").mkdir(parents=True, exist_ok=True)
    (bank_dir / "source").mkdir(parents=True, exist_ok=True)

    portrait = fc.PORTRAIT_CANONICAL / f"{char_id}.png"
    shutil.copy2(portrait, bank_dir / "source" / f"{char_id}.png")

    render_specs = {r["render_key"]: r for r in spec["renders"]}
    avatar_ids: dict[str, str] = {}
    poses_block: list[dict[str, Any]] = []
    for render_key, info in certified.items():
        source = fc.CERTIFIED_DIR / char_id / f"{render_key}.mp4"
        shutil.copy2(source, bank_dir / "certified" / f"{render_key}.mp4")
        avatar_ids[render_key] = fc.avatar_id(char_id, render_key, info["video_sha256"])
        poses_block.append({
            "name": render_key,
            "file": f"certified/{render_key}.mp4",
            "frame_count": info["frame_count"],
            "duration_seconds": info["duration_seconds"],
            "video_sha256": info["video_sha256"],
            "decoded_boundary_rgb_sha256": info["decoded_boundary_rgb_sha256"],
            "generation_mode": "ltx23_q4_text_to_motion_then_handle_certified",
            "prompt_profile": f"character_factory/config/prompt_packs/pose_prompt_pack_v1.json#renders.{render_key}",
            "seed_pair": entry["renders"][render_key]["seed_pair"],
            "prompt_id": entry["renders"][render_key]["prompt_id"],
        })

    mapping = resolve_pose_renders(certified, spec)
    aliased = {
        pose_id: render_key for pose_id, render_key in mapping.items()
        if pose_id not in ("neutral_resting", "active_listening", "speaking_direct")
        and render_key != pose_id
    }

    pose_set = character["pose_set_id"]
    bank_manifest = {
        "schema_version": 1,
        "pose_set_id": pose_set,
        "character_id": char_id,
        "display_name": character["display_name"],
        "language_code": character["language_code"],
        "language_name": character["language_name"],
        "language_agnostic": character["language_agnostic"],
        "tier": character["tier"],
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": activation_status,
        "purpose": (
            f"LTX 2.3 close-up pose bank for {character['display_name']}, "
            + (f"the {character['language_name']} " if character["language_name"] else "a language-agnostic ")
            + "companion, generated by character_factory."
        ),
        "source_portrait": {
            "file": f"source/{char_id}.png",
            "sha256": entry["portrait"]["canonical_sha256"],
            "generation_source": "interactive ChatGPT image generation",
            "prompt_file": f"character_factory/state/portrait_prompts/{char_id}.md",
        },
        "delivery_contract": {
            **spec["delivery_contract"],
            "shared_decoded_boundary_rgb_sha256": report["shared_decoded_boundary_rgb_sha256"],
        },
        "logical_aliases": spec["logical_aliases"],
        "aliased_reactions": aliased,
        "speaking_variant_policy": spec["speaking_variant_policy"],
        "avatar_ids": avatar_ids,
        "poses": poses_block,
    }
    fc.write_json(bank_dir / "manifest.json", bank_manifest)
    fc.write_json(bank_dir / "validation_report.json", report)

    speaking_variants = [
        {
            "variant_id": render_specs[key]["variant_id"],
            "avatar_id": avatar_ids[key],
            "asset_file": f"{key}.mp4",
        }
        for key in ("speaking_direct_v14_subtle", "speaking_direct_v15_reference_paced")
        if key in certified
    ]

    runtime = {
        "version": 1,
        "pose_set_id": pose_set,
        "activation_status": activation_status,
        "switch_safe": True,
        "test_only": activation_status != "production_default_live_verified",
        "default_pose_id": "neutral_resting",
        "switch_mode": "next_boundary",
        "asset_bundle": str(bank_dir.relative_to(fc.MUSETALK_ROOT)),
        "poses": {},
    }
    for pose_id in fc.POSE_IDS:
        render_key = mapping[pose_id]
        info = certified[render_key]
        block: dict[str, Any] = {
            "avatar_id": avatar_ids[render_key],
            "role": ROLE_BY_POSE[pose_id],
            "asset_file": f"{render_key}.mp4",
            "duration_seconds": info["duration_seconds"],
            "cycle_seconds": info["duration_seconds"],
            "fps": spec["delivery_contract"]["fps"],
            "frame_count": info["frame_count"],
        }
        # Only speaking_direct may carry variants, and only when a second one was rendered.
        if pose_id == "speaking_direct" and len(speaking_variants) > 1:
            block["variant_policy"] = spec["speaking_variant_policy"]["policy"]
            block["variants"] = speaking_variants
        runtime["poses"][pose_id] = block

    validate_runtime_manifest(runtime)
    manifest_path = fc.RUNTIME_MANIFEST_DIR / f"{pose_set}.json"
    fc.write_json(manifest_path, runtime)

    (bank_dir / "README.md").write_text(_bank_readme(character, bank_manifest, manifest_path), encoding="utf-8")

    entry["bank"] = str(bank_dir.relative_to(fc.MUSETALK_ROOT))
    entry["runtime_manifest"] = str(manifest_path.relative_to(fc.MUSETALK_ROOT))
    entry["avatar_ids"] = avatar_ids
    entry["stage"] = "packaged"
    fc.save_ledger(ledger)

    return {
        "bank_dir": bank_dir,
        "manifest_path": manifest_path,
        "avatar_ids": avatar_ids,
        "aliased_reactions": aliased,
        "physical_clips": len(certified),
    }


def _bank_readme(character: dict[str, Any], manifest: dict[str, Any], manifest_path: Path) -> str:
    language = character["language_name"] or "language-agnostic"
    rows = "\n".join(
        f"| `{pose['name']}` | {pose['frame_count']} | {pose['duration_seconds']} | `{pose['video_sha256'][:16]}` |"
        for pose in manifest["poses"]
    )
    return f"""# {character['display_name']} — {language} pose bank

Pose set ID: `{manifest['pose_set_id']}`
Runtime manifest: `{manifest_path.relative_to(fc.MUSETALK_ROOT)}`

Generated by `character_factory` from one ChatGPT source portrait and the LTX 2.3 Q4 graph.

| Clip | Frames | Seconds | SHA-256 (first 16) |
|---|---:|---:|---|
{rows}

All clips are {manifest['delivery_contract']['width']}x{manifest['delivery_contract']['height']},
{manifest['delivery_contract']['fps']} fps, H.264/yuv420p, silent, and share decoded boundary
hash `{manifest['delivery_contract']['shared_decoded_boundary_rgb_sha256']}`, so every ordered
transition between them cuts cleanly.

Prepare the MuseTalk caches with
`character_factory/scripts/prepare_musetalk_avatars.py --characters {character['character_id']}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing bank directory.")
    parser.add_argument("--activation-status", default="generated_pending_review",
                        help="Recorded in both manifests. Only a reviewed bank should claim a production status.")
    args = parser.parse_args()

    for char_id in args.characters:
        result = package(char_id, force=args.force, activation_status=args.activation_status)
        print(f"{char_id}: {result['physical_clips']} clips -> {result['bank_dir']}")
        print(f"          runtime manifest -> {result['manifest_path']}")
        if result["aliased_reactions"]:
            print("          NOTE aliased reactions (this tier did not render them): "
                  + ", ".join(f"{k}->{v}" for k, v in result["aliased_reactions"].items()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
