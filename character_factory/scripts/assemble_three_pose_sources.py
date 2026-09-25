#!/usr/bin/env python3
"""Assemble selected, verified LTX deliveries without rerendering accepted poses.

Each pose keeps its original generation manifest and seed provenance. This emits
assembly-provenance.json, never a fabricated single-job generation manifest.
Per-frame quality and received-video review still run after --source-dir ingest.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.motion_transitions import atomic_json, file_hash
from character_factory.scripts.generate_three_pose_videos import build_generation_graph

POSES = ("idle", "talking", "smiling")
APPROVED_PACK = REPO / "character_factory/config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json"


def _bound_file(path, digest, bindings):
    path = Path(path).resolve()
    if not path.is_file() or file_hash(path) != digest:
        raise ValueError(f"Missing or changed provenance file: {path}")
    bindings[str(path)] = digest
    return path


def _decode_metadata(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, count, shape, first, last = cap.get(cv2.CAP_PROP_FPS), 0, None, None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if shape is not None and shape != frame.shape:
                raise ValueError(f"Video changes frame dimensions: {path}")
            shape = frame.shape
            if first is None:
                first = hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            last = frame
            count += 1
    finally:
        cap.release()
    if not count or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Cannot decode selected delivery: {path}")
    last_hash = hashlib.sha256(cv2.cvtColor(last, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
    if first != last_hash:
        raise ValueError(f"Selected delivery lacks exact decoded loop endpoints: {path}")
    return {"width": shape[1], "height": shape[0], "fps": fps,
            "frame_count": count, "decoded_endpoint_rgb_sha256": first}


def select_sources(manifests, approved_pack):
    approved_pack = Path(approved_pack).resolve()
    approved_hash = file_hash(approved_pack)
    approved = json.loads(approved_pack.read_text())
    bindings = {str(approved_pack): approved_hash}
    selections, shared, shared_workflow, endpoint = {}, None, None, None
    for pose in POSES:
        manifest_path = Path(manifests[pose]).resolve()
        manifest_hash = file_hash(manifest_path)
        bindings[str(manifest_path)] = manifest_hash
        manifest = json.loads(manifest_path.read_text())
        fingerprint = manifest.get("generation_fingerprint")
        if not isinstance(fingerprint, dict) or fingerprint.get("version") != 1:
            raise ValueError(f"{pose}: generation lacks immutable input provenance")
        if not fingerprint.get("shared_anchor"):
            raise ValueError(f"{pose}: shared-anchor generation is required")
        source_image = _bound_file(manifest["source_image"], fingerprint["source_image_sha256"], bindings)
        if manifest["source_image_sha256"] != fingerprint["source_image_sha256"]:
            raise ValueError(f"{pose}: portrait identity differs within manifest")
        guide = _bound_file(manifest_path.parent / "guide-512x832.png", manifest["prepared_guide"]["sha256"], bindings)
        graph_path = _bound_file(manifest["accepted_graph"], fingerprint["accepted_graph_sha256"], bindings)
        pack_path = _bound_file(manifest["prompt_pack_path"], fingerprint["prompt_pack_sha256"], bindings)
        if manifest["prompt_pack_sha256"] != fingerprint["prompt_pack_sha256"]:
            raise ValueError(f"{pose}: prompt-pack provenance differs within manifest")
        profile = json.loads(pack_path.read_text())["poses"][pose]
        entry = manifest["poses"][pose]
        expected = approved["poses"][pose]
        if entry.get("status") != "completed":
            raise ValueError(f"{pose}: selected generation is incomplete")
        for key in ("positive_prompt", "negative_prompt"):
            if profile[key] != expected[key] or entry[key] != expected[key]:
                raise ValueError(f"{pose}: {key} differs from the chosen approved pack")
        # Seed and its descriptive citation may differ. Motion text and every
        # other per-pose generation option must still match the approved recipe.
        normalize = lambda value: {k:v for k,v in value.items() if k not in ("seed", "prompt_source")}
        if normalize(profile) != normalize(expected) or int(entry["seed"]) != int(profile["seed"]):
            raise ValueError(f"{pose}: generation options differ beyond a seed reroll")
        common = {k:v for k,v in fingerprint.items() if k != "prompt_pack_sha256"}
        common["guide_sha256"] = manifest["prepared_guide"]["sha256"]
        workflow = {k:v for k,v in manifest["workflow"].items()
                    if k not in ("frames_by_pose", "delivered_frames_by_pose")}
        if (workflow.get("resolution") != [fingerprint["width"], fingerprint["height"]]
                or workflow.get("fps") != fingerprint["fps"]
                or manifest["prepared_guide"].get("guide_dimensions") != workflow["resolution"]):
            raise ValueError(f"{pose}: guide/workflow dimensions differ within manifest")
        if shared is not None and (common != shared or workflow != shared_workflow):
            raise ValueError(f"{pose}: selected clips use different portrait/guide/graph geometry or workflow")
        shared, shared_workflow = common, workflow
        delivery = entry["delivery"]
        video_path = Path(delivery["file"])
        if not video_path.is_absolute():
            video_path = manifest_path.parent / video_path
        video = _bound_file(video_path, delivery["sha256"], bindings)
        metadata = _decode_metadata(video)
        for key, value in metadata.items():
            if delivery.get(key) != value:
                raise ValueError(f"{pose}: decoded {key} differs from recorded delivery metadata")
        frames = int(profile.get("frame_count", 241))
        delivered_frames = (frames-1)*int(profile.get("repeat_cycles", 1))+1
        if (metadata["frame_count"] != delivered_frames or metadata["width"] != fingerprint["width"]
                or metadata["height"] != fingerprint["height"] or metadata["fps"] != fingerprint["fps"]
                or not delivery.get("first_frame_replaced_with_shared_anchor")):
            raise ValueError(f"{pose}: delivery differs from the shared-anchor recipe")
        if endpoint is not None and endpoint != metadata["decoded_endpoint_rgb_sha256"]:
            raise ValueError("Selected poses do not share a decoded anchor")
        endpoint = metadata["decoded_endpoint_rgb_sha256"]
        generation_path = manifest_path.parent / "graphs" / (pose+"-generation.json")
        graph = json.loads(generation_path.read_text())
        bindings[str(generation_path.resolve())] = file_hash(generation_path)
        expected_graph = build_generation_graph(json.loads(graph_path.read_text()), profile,
            graph["image"]["inputs"]["image"], graph["save"]["inputs"]["filename_prefix"], frames)
        graph_dimensions = expected_graph["empty"]["inputs"]
        if (graph != expected_graph or graph_dimensions["width"] != fingerprint["width"]
                or graph_dimensions["height"] != fingerprint["height"]
                or manifest["workflow"]["frames_by_pose"].get(pose) != frames
                or manifest["workflow"]["delivered_frames_by_pose"].get(pose) != delivered_frames):
            raise ValueError(f"{pose}: recorded generation graph differs from the approved recipe")
        selections[pose] = {"generation_manifest": str(manifest_path), "generation_manifest_sha256": manifest_hash,
                            "delivery_source": str(video), "delivery_sha256": delivery["sha256"],
                            "source_image": str(source_image), "guide": str(guide),
                            "prompt_pack": str(pack_path), "generation_graph": str(generation_path.resolve()),
                            "seed": entry["seed"], "approved_seed": expected["seed"],
                            "positive_prompt": entry["positive_prompt"], "negative_prompt": entry["negative_prompt"],
                            "decoded_delivery": metadata}
    return {"version": 1, "kind": "selected_generation_deliveries",
            "approved_prompt_pack": {"path": str(approved_pack), "sha256": approved_hash, "pack_id": approved["pack_id"]},
            "shared_inputs": shared, "shared_workflow": shared_workflow,
            "shared_decoded_endpoint_rgb_sha256": endpoint,
            "selections": selections, "bound_input_files": bindings}


def assemble(manifests, output_dir, approved_pack=APPROVED_PACK):
    inputs = select_sources(manifests, approved_pack)
    output_dir = Path(output_dir).resolve()
    if any(output_dir == Path(item["generation_manifest"]).parent for item in inputs["selections"].values()):
        raise ValueError("Assembly output must differ from every generation source directory")
    targets = {pose+".mp4": (Path(item["delivery_source"]), item["delivery_sha256"])
               for pose, item in inputs["selections"].items()}
    targets["guide-512x832.png"] = (Path(inputs["selections"]["idle"]["guide"]), inputs["shared_inputs"]["guide_sha256"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".assembly.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        provenance_path = output_dir / "assembly-provenance.json"
        existing = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        if existing is not None and existing.get("inputs") != inputs:
            raise ValueError("Assembly inputs changed; use a new output directory")
        if existing is None and any(p.name != ".assembly.lock" for p in output_dir.iterdir()):
            raise ValueError("Output directory already contains untracked files; use a new directory")
        for name, (_, digest) in targets.items():
            target = output_dir / name
            if target.exists() and file_hash(target) != digest:
                raise ValueError(f"Assembled output changed: {target}")
        for path, digest in inputs["bound_input_files"].items():
            if file_hash(path) != digest:
                raise ValueError(f"Input changed during verification: {path}")
        result = {"version": 1, "kind": "selected_generation_deliveries", "status": "assembling", "inputs": inputs,
                  "outputs": {name: {"file": name, "sha256": digest} for name, (_,digest) in targets.items()},
                  "quality_status": "requires_per_frame_validation_and_recorded_review"}
        if existing is None:
            atomic_json(provenance_path, result)
        for name, (source, digest) in targets.items():
            destination = output_dir / name
            if destination.exists():
                continue
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix="."+name+".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                shutil.copyfile(source, temporary)
                if file_hash(temporary) != digest:
                    raise ValueError(f"Source changed while copying: {source}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        result["status"] = "complete"
        if existing != result:
            atomic_json(provenance_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for pose in POSES:
        parser.add_argument("--"+pose+"-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved-prompt-pack", type=Path, default=APPROVED_PACK)
    args = parser.parse_args()
    try:
        result = assemble({p: getattr(args,p+"_manifest") for p in POSES}, args.output_dir, args.approved_prompt_pack)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Assembly refused: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "status": result["status"],
                      "quality_status": result["quality_status"],
                      "selected_seeds": {p:v["seed"] for p,v in result["inputs"]["selections"].items()}}, indent=2))


if __name__ == "__main__": main()
