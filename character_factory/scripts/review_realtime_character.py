#!/usr/bin/env python3
"""Record an explicit operator decision about normal-speed WebRTC recordings.

Passing diagnostics never approves appearance. Acceptance requires a named
reviewer, notes, and an explicit --watched-normal-speed attestation. The receipt
records the supplied name locally; it is not an authenticated digital signature.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, atomic_json, file_hash, publish_bank
from scripts.review_motion_evidence import find_motion, verify_recording

REQUIRED_CASES = {"short-idle", "long-talking-smiling", "interrupted-and-next-turn"}
SOURCE_NAMES = {IDLE: "idle", TALK: "talking", SMILE: "smiling"}


def _case_coverage(report):
    """Require actual case behavior, rather than trusting recording filenames."""
    case = report["case"]
    if case not in REQUIRED_CASES:
        return
    turns = report["turns"]
    motion = find_motion(report["final_status"])
    if (not turns or not motion.get("settled") or len(motion["entries"]) != len(turns)
            or len(motion["returns"]) < len(turns)):
        raise ValueError(f"Incomplete entry/return evidence for {case}")
    observed = [set(turn["observed_body_poses"]) for turn in turns]
    if case == "short-idle" and any(poses != {IDLE} for poses in observed):
        raise ValueError("Short-reply recording did not stay on the idle body")
    if case == "long-talking-smiling" and not {IDLE, TALK, SMILE}.issubset(set().union(*observed)):
        raise ValueError("Long recording does not demonstrate all three body sources")
    if case == "interrupted-and-next-turn":
        interrupted = turns[0].get("interrupted_at_seconds")
        before = turns[0].get("motion_before_interrupt", {}).get("last_emitted", {})
        if (len(turns) < 2 or not isinstance(interrupted, (int, float))
                or not math.isfinite(interrupted) or "interrupt_response" not in turns[0]
                or before.get("mode") != "live" or before.get("pose_id") != TALK
                or observed[-1] != {IDLE}):
            raise ValueError("Interruption recording lacks a live talking interruption and following idle reply")


def validate_evidence(character_dir, evidence_dir):
    """Revalidate all evidence and return immutable inputs for one decision."""
    atlas_path = character_dir / "motion-atlas.json"
    package_path = character_dir / "character.json"
    verification_path = evidence_dir / "verification.json"
    guards = {path: file_hash(path) for path in (atlas_path, package_path, verification_path)}
    atlas = json.loads(atlas_path.read_text())
    package = json.loads(package_path.read_text())
    verification = json.loads(verification_path.read_text())
    bank = MotionBank(atlas)
    expected = {pose: source["sha256"] for pose, source in bank.sources.items()}
    if Path(package["motion_atlas"]).resolve() != atlas_path:
        raise ValueError("Character package points to a different atlas")
    if package.get("motion_atlas_content_sha256") not in (None, bank.routing_sha256):
        raise ValueError("Character package routing digest is stale")
    if package["source_hashes"] != {SOURCE_NAMES[p]: value for p, value in expected.items()}:
        raise ValueError("Character package source hashes differ from atlas")
    for pose, source in bank.sources.items():
        paths = {Path(source["path"]), Path(package["source_dir"]) / (SOURCE_NAMES[pose]+".mp4")}
        for path in paths:
            guards[path] = file_hash(path)
            if guards[path] != expected[pose]:
                raise ValueError(f"Source video changed: {path}")
    if (verification.get("version") != 1 or verification.get("automated_validation_passed") is not True
            or verification.get("routing_sha256") != bank.routing_sha256
            or verification.get("source_hashes") != expected
            or verification.get("source_frame_counts") != {p: bank.count(p) for p in bank.sources}):
        raise ValueError("Verification failed or belongs to different routing/source content")
    # Full atlas bytes can change through a prior review; routing excludes only
    # status/review metadata and must still match the exact recorded route data.
    recorded_atlas_hash = verification.get("atlas_sha256")
    if (not isinstance(recorded_atlas_hash, str) or len(recorded_atlas_hash) != 64
            or any(c not in "0123456789abcdef" for c in recorded_atlas_hash)):
        raise ValueError("Verification lacks its original atlas hash")
    if recorded_atlas_hash != guards[atlas_path]:
        # A prior explicit decision changes only status/review metadata. Require
        # its immutable receipt to explain the full-file hash change, otherwise
        # regenerate verification for the current atlas before making a decision.
        prior = atlas.get("review", {})
        receipt_path = Path(prior.get("receipt", ""))
        if not receipt_path.is_file() or file_hash(receipt_path) != prior.get("receipt_sha256"):
            raise ValueError("Atlas file changed without a verified review receipt; regenerate verification")
        guards[receipt_path] = file_hash(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("routing_sha256") != bank.routing_sha256 or receipt.get("source_hashes") != expected
                or recorded_atlas_hash not in {receipt.get("recorded_atlas_sha256"), receipt.get("atlas_before_review_sha256")}):
            raise ValueError("Prior review receipt does not explain the recorded atlas hash")
    rows = verification.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Verification has no recording list; regenerate it with review_motion_evidence.py")
    cases, recordings = set(), []
    for row in rows:
        case, filename = row["case"], row["video"]
        if case in cases:
            raise ValueError(f"Duplicate recording case: {case}")
        cases.add(case)
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".mp4":
            raise ValueError("Recording paths must be local MP4 filenames")
        video = evidence_dir / filename
        report_path = video.with_suffix(".json")
        for path, digest in ((video, row["video_sha256"]), (report_path, row["evidence_sha256"])):
            guards[path] = file_hash(path)
            if guards[path] != digest:
                raise ValueError(f"Recorded evidence changed: {path}")
        computed = verify_recording(report_path, bank)
        if any(row.get(key) != value for key, value in computed.items()):
            raise ValueError(f"Verification metadata differs from current recording: {case}")
        _case_coverage(json.loads(report_path.read_text()))
        recordings.append(computed)
    missing = REQUIRED_CASES - cases
    if missing:
        raise ValueError(f"Missing required received recordings: {sorted(missing)}")
    return atlas, package, verification, recordings, guards


def review_character(character_dir, evidence_dir, *, decision, reviewer, notes,
                     watched_normal_speed=False):
    if decision not in {"accepted", "rejected"}:
        raise ValueError("Decision must be accepted or rejected")
    reviewer, notes = reviewer.strip(), notes.strip()
    if not reviewer or not notes:
        raise ValueError("A named reviewer and nonempty review notes are required")
    if decision == "accepted" and not watched_normal_speed:
        raise ValueError("Acceptance requires --watched-normal-speed after watching the received recordings")
    character_dir, evidence_dir = Path(character_dir).resolve(), Path(evidence_dir).resolve()
    if not character_dir.is_dir():
        raise ValueError("Character directory does not exist")
    with (character_dir / ".recorded-review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atlas, package, verification, recordings, guards = validate_evidence(character_dir, evidence_dir)
        original_atlas, original_package = copy.deepcopy(atlas), copy.deepcopy(package)
        if not isinstance(package.get("review_history", []), list):
            raise ValueError("Existing review history is invalid")
        for path, digest in guards.items():
            if file_hash(path) != digest:
                raise ValueError(f"Review input changed during verification: {path}")
        now = datetime.now(timezone.utc)
        review_id = now.strftime("%Y%m%dT%H%M%S%fZ")+"-"+uuid.uuid4().hex[:8]
        receipt_path = character_dir / "reviews" / (review_id+".json")
        receipt = {"version": 1, "review_id": review_id, "character_id": package["character_id"],
                   "decision": decision, "reviewer": reviewer, "notes": notes,
                   "signed_at": now.isoformat(), "signature_kind": "operator_supplied_name_not_cryptographic",
                   "watched_normal_speed": bool(watched_normal_speed),
                   "automated_validation_passed": True, "routing_sha256": verification["routing_sha256"],
                   "source_hashes": verification["source_hashes"], "recordings": recordings,
                   "verification_file": str(evidence_dir / "verification.json"),
                   "verification_sha256": guards[evidence_dir / "verification.json"],
                   "atlas_before_review_sha256": guards[character_dir / "motion-atlas.json"],
                   "recorded_atlas_sha256": verification["atlas_sha256"]}
        atomic_json(receipt_path, receipt)
        summary = {"receipt": str(receipt_path), "receipt_sha256": file_hash(receipt_path),
                   "review_id": review_id, "reviewer": reviewer, "decision": decision,
                   "signed_at": receipt["signed_at"], "watched_normal_speed": bool(watched_normal_speed)}
        atlas.update(status="reviewed" if decision == "accepted" else "rejected", review=summary)
        package.update(status=atlas["status"], review=summary,
                       motion_atlas_content_sha256=verification["routing_sha256"])
        package["review_history"] = package.get("review_history", []) + [summary]
        try:
            publish_bank(character_dir / "motion-atlas.json", atlas)
            atomic_json(character_dir / "character.json", package)
        except Exception:
            # A receipt alone does not activate a bank. Restore the previous
            # authoritative state if package/registration publication fails.
            publish_bank(character_dir / "motion-atlas.json", original_atlas)
            atomic_json(character_dir / "character.json", original_package)
            raise
        return {"decision": decision, "status": atlas["status"], "reviewer": reviewer,
                "receipt": str(receipt_path), "routing_sha256": verification["routing_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--watched-normal-speed", action="store_true")
    args = parser.parse_args()
    try:
        result = review_character(args.character_dir, args.evidence_dir, decision=args.decision,
                                  reviewer=args.reviewer, notes=args.notes,
                                  watched_normal_speed=args.watched_normal_speed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Review not published: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
