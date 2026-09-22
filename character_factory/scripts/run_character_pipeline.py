#!/usr/bin/env python3
"""Run one character end to end, or report where every character currently stands.

The pipeline has one step this script cannot perform: portrait generation happens
interactively in ChatGPT. `--stage all` therefore stops at that boundary and tells you what
to generate, then picks the character back up once the portrait lands in the inbox.

Stages: prompt -> (human: ChatGPT) -> ingest -> render -> certify -> package -> prepare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import factory_common as fc

SCRIPTS = Path(__file__).resolve().parent
STAGE_ORDER = ["prompt", "ingest", "render", "certify", "package", "prepare"]
STAGE_REACHED = {
    "new": 0, "prompt_emitted": 1, "portrait_ingested": 2, "rendered": 3,
    "certification_failed": 3, "certified": 4, "packaged": 5, "prepared": 6,
}


def run(script: str, *args: str) -> int:
    command = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n$ {' '.join(command[1:])}", flush=True)
    return subprocess.run(command).returncode


def status_rows(roster: dict[str, Any], ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for char_id, character in roster.items():
        entry = ledger.get("characters", {}).get(char_id, {})
        stage = entry.get("stage", "new")
        rows.append({
            "character_id": char_id,
            "display_name": character["display_name"],
            "language": character["language_name"] or "agnostic",
            "tier": character["tier"],
            "stage": stage,
            "renders": len(entry.get("renders") or {}),
            "certified": len(entry.get("certified") or {}),
            "prepared": len(entry.get("prepared_avatars") or {}),
        })
    return rows


def print_status(rows: list[dict[str, Any]], limit: int | None) -> None:
    from collections import Counter
    counts = Counter(row["stage"] for row in rows)
    print(f"{len(rows)} characters")
    for stage in ["new", "prompt_emitted", "portrait_ingested", "rendered",
                  "certification_failed", "certified", "packaged", "prepared"]:
        if counts.get(stage):
            print(f"  {stage:22s} {counts[stage]:>4}")
    blocked = [r for r in rows if r["stage"] == "certification_failed"]
    if blocked:
        print(f"\n{len(blocked)} blocked on certification: "
              + ", ".join(r["character_id"] for r in blocked[:12]))
    shown = [r for r in rows if r["stage"] != "new"][: limit or 20]
    if shown:
        print(f"\n{'id':10s} {'name':12s} {'language':16s} {'tier':6s} {'stage':22s} r/c/p")
        for row in shown:
            print(f"{row['character_id']:10s} {row['display_name']:12s} {row['language'][:16]:16s} "
                  f"{row['tier']:6s} {row['stage']:22s} "
                  f"{row['renders']}/{row['certified']}/{row['prepared']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", help="Character IDs to advance.")
    parser.add_argument("--stage", default="all",
                        choices=["all", *STAGE_ORDER], help="Run one stage, or every stage that is ready.")
    parser.add_argument("--status", action="store_true", help="Report progress and exit.")
    parser.add_argument("--limit", type=int, help="Rows to show in --status.")
    parser.add_argument("--comfy-url", help="Override the ComfyUI base URL.")
    parser.add_argument("--musetalk-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roster = fc.roster_index()
    ledger = fc.load_ledger()

    if args.status or not args.characters:
        print_status(status_rows(roster, ledger), args.limit)
        if not args.characters:
            print("\nPass --characters <id> ... to advance specific characters.")
        return 0

    unknown = set(args.characters) - set(roster)
    if unknown:
        raise fc.FactoryError(f"Unknown character IDs: {', '.join(sorted(unknown))}")

    for char_id in args.characters:
        entry = ledger.get("characters", {}).get(char_id, {})
        reached = STAGE_REACHED.get(entry.get("stage", "new"), 0)
        wanted = STAGE_ORDER if args.stage == "all" else [args.stage]
        print(f"\n=== {char_id} ({roster[char_id]['display_name']}) stage={entry.get('stage', 'new')} ===")

        for stage in wanted:
            index = STAGE_ORDER.index(stage)
            if args.stage == "all" and index < reached:
                continue

            if stage == "prompt":
                if run("render_portrait_prompts.py", "--characters", char_id):
                    return 1
            elif stage == "ingest":
                inbox = fc.PORTRAIT_INBOX / f"{char_id}.png"
                canonical = fc.PORTRAIT_CANONICAL / f"{char_id}.png"
                if not inbox.exists() and not canonical.exists():
                    print(f"\nPAUSED: {char_id} needs its portrait generated in ChatGPT.")
                    print(f"  prompt: state/portrait_prompts/{char_id}.md")
                    print(f"  save to: {inbox}")
                    print("  then re-run this command to continue.")
                    break
                if not canonical.exists() and run("ingest_portraits.py", "--characters", char_id):
                    return 1
            elif stage == "render":
                extra = ["--dry-run"] if args.dry_run else []
                if args.comfy_url:
                    print(f"  (note: --comfy-url is advisory; set comfyui.base_url in "
                          f"config/generation_defaults.json to change it permanently)")
                if run("generate_pose_videos.py", "--characters", char_id, *extra):
                    return 1
                if args.dry_run:
                    break
            elif stage == "certify":
                if run("certify_pose_bank.py", "--characters", char_id):
                    print(f"\n{char_id} failed certification. Inspect "
                          f"state/certified/{char_id}/validation_report.json, then reroll the "
                          f"offending pose with:\n"
                          f"  generate_pose_videos.py --characters {char_id} --renders <key> --attempt 1 --force")
                    return 1
            elif stage == "package":
                if run("package_pose_bank.py", "--characters", char_id):
                    return 1
            elif stage == "prepare":
                extra = ["--dry-run"] if args.dry_run else []
                if run("prepare_musetalk_avatars.py", "--characters", char_id,
                       "--base-url", args.musetalk_url, *extra):
                    return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
