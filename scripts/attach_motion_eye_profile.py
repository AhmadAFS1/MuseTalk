#!/usr/bin/env python3
"""Publish a separate candidate eye-blend atlas from verified source measurements.

This never changes approved source videos or grants visual approval. Configure
only the resulting candidate registry for a pilot: duplicate source banks in one
registry are intentionally rejected by the runtime.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_transitions import MotionBank, file_hash, publish_bank


def attach(atlas_path, measurement_path, output):
    atlas_path, measurement_path, output = map(Path, (atlas_path, measurement_path, output))
    if (output.name != "motion-atlas.json" or output.resolve() == atlas_path.resolve()
            or output.exists() or output.with_name("motion-registration.json").exists()):
        raise ValueError("Choose a new motion-atlas.json and registry sidecar; existing banks cannot be overwritten")
    bank = MotionBank(json.loads(atlas_path.read_text()))
    measured = json.loads(measurement_path.read_text())
    if measured.get("version") != 1 or measured.get("method") != "incoming_roi_v1":
        raise ValueError("Unsupported eye measurement profile")
    if measured.get("atlas_sha256") != file_hash(atlas_path):
        raise ValueError("Eye measurements belong to a different parent atlas")
    hashes = measured.get("source_hashes")
    if hashes != {pose: source["sha256"] for pose, source in bank.sources.items()}:
        raise ValueError("Eye measurements do not belong to this source bank")
    for source in bank.sources.values():
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Source video changed since atlas creation")
    candidate = copy.deepcopy(bank.manifest)
    candidate.pop("review", None)
    candidate["status"] = "candidate_requires_recorded_review"
    candidate["eye_blend"] = {"method": "incoming_roi_v1", "source_hashes": hashes,
                              "frames": measured.get("frames")}
    candidate["eye_blend_provenance"] = {"parent_atlas_sha256": file_hash(atlas_path),
                                        "measurements_sha256": file_hash(measurement_path)}
    MotionBank(candidate)  # Includes finite geometry, bounds and all-frame coverage.
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_bank(output, candidate)
    return {"atlas": str(output.resolve()), "atlas_sha256": file_hash(output),
            "routing_sha256": MotionBank(candidate).routing_sha256,
            "status": candidate["status"], "profile": candidate["eye_blend"]["method"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(attach(args.atlas, args.measurements, args.output), indent=2))


if __name__ == "__main__":
    main()
