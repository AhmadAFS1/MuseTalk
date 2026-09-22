#!/usr/bin/env python3
"""Emit one ChatGPT image-generation prompt per character, plus a work queue.

Image generation is interactive: Codex pastes each prompt into its own ChatGPT image tool
and saves the result into state/portrait_inbox/. No API key is used anywhere in this
factory. This script only prepares the work and tracks what is still outstanding.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import factory_common as fc

# The first fenced block after the "## Template" heading is the prompt body. Prose may sit
# between the heading and the fence.
TEMPLATE_BLOCK = re.compile(r"## Template\b.*?```\n(.*?)\n```", re.DOTALL)


def load_template() -> str:
    text = fc.PORTRAIT_TEMPLATE_PATH.read_text(encoding="utf-8")
    match = TEMPLATE_BLOCK.search(text)
    if not match:
        raise fc.FactoryError(
            f"{fc.PORTRAIT_TEMPLATE_PATH} has no fenced block under '## Template'. "
            "The prompt body must stay inside that block so this script and a human reader "
            "can never drift apart."
        )
    return " ".join(match.group(1).split())


def fill(template: str, character: dict[str, Any]) -> str:
    casting = character["casting"]
    filled = template
    for key in ("subject", "setting", "hair", "wardrobe", "lighting"):
        filled = filled.replace("{" + key + "}", casting[key])
    leftover = re.findall(r"\{[a-z_]+\}", filled)
    if leftover:
        raise fc.FactoryError(
            f"Unfilled placeholders for {character['character_id']}: {', '.join(sorted(set(leftover)))}"
        )
    return filled


def prompt_document(character: dict[str, Any], prompt: str) -> str:
    language = character["language_name"] or "language-agnostic"
    return "\n".join([
        f"# {character['character_id']} — {character['display_name']} ({language})",
        "",
        f"- tier: `{character['tier']}`",
        f"- gender: {character['gender']} ({character['pronouns']['subject']}/{character['pronouns']['object']})",
        f"- casting pool: {character['casting']['heritage_source']}",
        f"- save the image to: `state/portrait_inbox/{character['character_id']}.png`",
        "",
        "## Prompt",
        "",
        prompt,
        "",
        "## Before saving",
        "",
        "Reject and regenerate if the mouth is open, the gaze is off-camera, hair or a hand",
        "crosses the face, any text or watermark is present, the crop is tighter than mid-chest,",
        "the skin reads as retouched, or the face sits far off the horizontal centre. The full",
        "rejection list is in `config/prompt_packs/portrait_prompt_template.md`.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", help="Restrict to these character IDs.")
    parser.add_argument("--languages", nargs="*", help="Restrict to these BCP-47 codes.")
    parser.add_argument("--agnostic-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Emit at most this many prompts.")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip characters whose portrait is already in the inbox or ingested.")
    parser.add_argument("--out-dir", type=Path, default=fc.PROMPT_OUT_DIR)
    args = parser.parse_args()

    roster = fc.load_roster()
    template = load_template()
    fc.ensure_state_dirs()

    selected = roster["characters"]
    if args.characters:
        wanted = set(args.characters)
        selected = [c for c in selected if c["character_id"] in wanted]
        unknown = wanted - {c["character_id"] for c in roster["characters"]}
        if unknown:
            raise fc.FactoryError(f"Unknown character IDs: {', '.join(sorted(unknown))}")
    if args.languages:
        codes = set(args.languages)
        selected = [c for c in selected if c["language_code"] in codes]
    if args.agnostic_only:
        selected = [c for c in selected if c["language_agnostic"]]
    if args.only_missing:
        selected = [
            c for c in selected
            if not (fc.PORTRAIT_INBOX / f"{c['character_id']}.png").exists()
            and not (fc.PORTRAIT_CANONICAL / f"{c['character_id']}.png").exists()
        ]
    if args.limit:
        selected = selected[: args.limit]

    if not selected:
        print("Nothing to do: every selected character already has a portrait.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    queue = []
    for character in selected:
        prompt = fill(template, character)
        path = args.out_dir / f"{character['character_id']}.md"
        path.write_text(prompt_document(character, prompt), encoding="utf-8")
        queue.append({
            "character_id": character["character_id"],
            "display_name": character["display_name"],
            "language_code": character["language_code"],
            "language_name": character["language_name"],
            "tier": character["tier"],
            "prompt_file": str(path.relative_to(fc.FACTORY_ROOT)),
            "expected_portrait": f"state/portrait_inbox/{character['character_id']}.png",
            "prompt": prompt,
        })

    queue_path = args.out_dir / "queue.json"
    fc.write_json(queue_path, {
        "schema_version": 1,
        "note": ("Work queue for interactive ChatGPT image generation. Generate each `prompt`, "
                 "save the PNG to `expected_portrait`, then run ingest_portraits.py. "
                 "Re-run this script with --only-missing to see what is left."),
        "pending": len(queue),
        "items": queue,
    })
    print(f"Wrote {len(queue)} prompt files to {args.out_dir}")
    print(f"Work queue: {queue_path}")
    print(f"Save each generated PNG to {fc.PORTRAIT_INBOX}/<character_id>.png, then run ingest_portraits.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
