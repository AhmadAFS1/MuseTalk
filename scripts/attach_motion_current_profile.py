#!/usr/bin/env python3
"""Create an unreviewed current-phoneme candidate from a measured motion bank."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_transitions import MotionBank, file_hash, publish_bank


def attach(atlas_path, output):
    atlas_path, output = map(Path, (atlas_path, output))
    if (output.name != "motion-atlas.json" or output.resolve() == atlas_path.resolve()
            or output.exists() or output.with_name("motion-registration.json").exists()):
        raise ValueError("Choose a new motion-atlas.json and registry sidecar; existing banks cannot be overwritten")
    bank = MotionBank(json.loads(atlas_path.read_text()))
    if bank.eye_blend is None:
        raise ValueError("Measure and attach source-bound eye geometry first")
    for source in bank.sources.values():
        path = Path(source["path"])
        path = path if path.is_absolute() else atlas_path.parent / path
        if file_hash(path) != source["sha256"]:
            raise ValueError("Source video changed since atlas creation")
    candidate = copy.deepcopy(bank.manifest)
    candidate.pop("review", None)
    candidate["status"] = "candidate_requires_recorded_review"
    candidate["current_phoneme"] = {
        "method": "current_similarity_v1",
        "source_hashes": {p: s["sha256"] for p, s in bank.sources.items()},
    }
    candidate["current_phoneme_provenance"] = {"parent_atlas_sha256": file_hash(atlas_path)}
    MotionBank(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_bank(output, candidate)
    return {"atlas": str(output.resolve()), "atlas_sha256": file_hash(output),
            "routing_sha256": MotionBank(candidate).routing_sha256,
            "status": candidate["status"], "profile": candidate["current_phoneme"]["method"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(attach(args.atlas, args.output), indent=2))


if __name__ == "__main__":
    main()
