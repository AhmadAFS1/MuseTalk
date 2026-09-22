#!/usr/bin/env python3
"""Drive the LTX 2.3 Q4 ComfyUI graph to render one character's pose set.

Patches only the nodes listed in config/generation_defaults.json["node_ids"] and leaves the
rest of the winning Q4 graph untouched, because the sampler schedule, NAG values, and
attention patches are the reason that graph produces usable teeth and a stable identity.

Resumable: an already-rendered pose is skipped unless --force is given, so a run that dies
on character 40 of 332 restarts where it stopped.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import factory_common as fc


def request_json(url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fill_prompt(text: str, character: dict[str, Any]) -> str:
    casting = character["casting"]
    pronouns = character["pronouns"]
    replacements = {
        "{subject}": casting["subject"],
        "{wardrobe}": casting["wardrobe"],
        "{hair}": casting["hair"],
        "{setting}": casting["setting"],
        "{lighting}": casting["lighting"],
        "{pronoun_subject}": pronouns["subject"],
        "{pronoun_object}": pronouns["object"],
        "{pronoun_possessive}": pronouns["possessive"],
    }
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _capitalise_sentences(text: str) -> str:
    """Sentence-initial pronoun substitution leaves lowercase 'she'/'they' after a period."""
    out, capitalise = [], True
    for char in text:
        out.append(char.upper() if capitalise and char.isalpha() else char)
        if char.isalpha():
            capitalise = False
        elif char in ".!?":
            capitalise = True
    return "".join(out)


def build_workflow(
    *,
    base_workflow: dict[str, Any],
    defaults: dict[str, Any],
    character: dict[str, Any],
    render: dict[str, Any],
    prompt_pack: dict[str, Any],
    input_name: str,
    render_index: int,
    attempt: int,
) -> dict[str, Any]:
    workflow = copy.deepcopy(base_workflow)
    nodes = defaults["node_ids"]
    sampling = defaults["sampling"]
    render_key = render["render_key"]
    pose_prompts = prompt_pack["renders"][render_key]

    global_prompt = _capitalise_sentences(fill_prompt(prompt_pack["global_prompt_template"], character))
    local_prompts = " | ".join(
        _capitalise_sentences(fill_prompt(segment, character)) for segment in pose_prompts["local_prompts"]
    )

    seed_1, seed_2 = fc.render_seed_pair(character["character_id"], render_index, attempt, defaults)

    workflow[nodes["load_image"]]["inputs"]["image"] = input_name
    workflow[nodes["duration_seconds_int"]]["inputs"]["value"] = render["seconds"]
    workflow[nodes["prompt_relay_encode"]]["inputs"]["global_prompt"] = global_prompt
    workflow[nodes["prompt_relay_encode"]]["inputs"]["local_prompts"] = local_prompts
    workflow[nodes["prompt_relay_encode"]]["inputs"]["segment_lengths"] = pose_prompts["segment_lengths"]
    workflow[nodes["negative_text_encode"]]["inputs"]["text"] = prompt_pack["negative_prompt"]
    workflow[nodes["add_guide"]]["inputs"]["strength"] = sampling["guide_strength"]
    workflow[nodes["preprocess_image"]]["inputs"]["img_compression"] = sampling["img_compression"]
    workflow[nodes["resize_image"]]["inputs"]["resize_type.width"] = sampling["resize_width"]
    workflow[nodes["attention_tuner"]]["inputs"]["audio_to_video_scale"] = sampling["audio_to_video_scale"]
    workflow[nodes["stage1_noise_seed"]]["inputs"]["noise_seed"] = seed_1
    workflow[nodes["stage2_noise_seed"]]["inputs"]["noise_seed"] = seed_2
    workflow[nodes["video_combine"]]["inputs"]["filename_prefix"] = (
        f"{defaults['comfyui']['filename_prefix_root']}/{character['character_id']}_{render_key}"
    )

    # The graph derives frame count from seconds x fps; verify it lands on the certified
    # count rather than discovering a one-frame drift after a 12-minute render.
    fps = int(workflow[nodes["fps_float"]]["inputs"]["value"])
    computed = 1 + 8 * round((render["seconds"] * fps - 1) / 8)
    if computed != render["frame_count"]:
        raise fc.FactoryError(
            f"{render_key}: graph would produce {computed} frames at {render['seconds']}s/{fps}fps, "
            f"but config/pose_spec.json certifies {render['frame_count']}."
        )
    return workflow


def find_output(history: dict[str, Any], prompt_id: str, output_dir: Path) -> Path:
    record = history[prompt_id]
    for output in record.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in output.get(key, []):
                filename = item.get("filename", "")
                if filename.endswith(".mp4"):
                    return output_dir / item.get("subfolder", "") / filename
    raise fc.FactoryError(f"ComfyUI produced no MP4 for prompt {prompt_id}")


def record_timing(entry: dict[str, Any]) -> None:
    """Append-only so a bulk run's real throughput stops being a guess after a few renders."""
    fc.TIMINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with fc.TIMINGS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_render(
    *,
    character: dict[str, Any],
    render: dict[str, Any],
    render_index: int,
    defaults: dict[str, Any],
    base_workflow: dict[str, Any],
    prompt_pack: dict[str, Any],
    attempt: int,
    dry_run: bool,
) -> dict[str, Any]:
    comfy = defaults["comfyui"]
    char_id = character["character_id"]
    render_key = render["render_key"]

    portrait = fc.PORTRAIT_CANONICAL / f"{char_id}.png"
    if not portrait.exists():
        raise fc.FactoryError(f"No canonical portrait for {char_id}. Run ingest_portraits.py first.")

    input_relative = Path(comfy["input_subfolder"]) / f"{char_id}.png"
    workflow = build_workflow(
        base_workflow=base_workflow, defaults=defaults, character=character, render=render,
        prompt_pack=prompt_pack, input_name=input_relative.as_posix(),
        render_index=render_index, attempt=attempt,
    )
    destination = fc.RENDER_DIR / char_id / f"{render_key}.mp4"

    if dry_run:
        nodes = defaults["node_ids"]
        print(f"  [dry-run] {char_id}/{render_key}: {render['frame_count']} frames, "
              f"seeds={workflow[nodes['stage1_noise_seed']]['inputs']['noise_seed']}/"
              f"{workflow[nodes['stage2_noise_seed']]['inputs']['noise_seed']}")
        print(f"            global: {workflow[nodes['prompt_relay_encode']]['inputs']['global_prompt'][:110]}...")
        return {"dry_run": True, "destination": str(destination)}

    input_destination = Path(comfy["input_dir"]) / input_relative
    input_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(portrait, input_destination)

    started = time.monotonic()
    queued = request_json(f"{comfy['base_url']}/prompt", {"prompt": workflow})
    prompt_id = queued["prompt_id"]
    print(f"  QUEUED {char_id}/{render_key} attempt={attempt} prompt_id={prompt_id}", flush=True)

    deadline = time.monotonic() + comfy["job_timeout_seconds"]
    while time.monotonic() < deadline:
        history = request_json(f"{comfy['base_url']}/history/{prompt_id}")
        status = history.get(prompt_id, {}).get("status", {})
        if not status.get("completed"):
            time.sleep(comfy["poll_seconds"])
            continue
        if status.get("status_str") != "success":
            raise fc.FactoryError(f"ComfyUI failed {char_id}/{render_key}: {status}")

        produced = find_output(history, prompt_id, Path(comfy["output_dir"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, destination)
        elapsed = time.monotonic() - started
        result = {
            "render_key": render_key,
            "file": str(destination.relative_to(fc.FACTORY_ROOT)),
            "prompt_id": prompt_id,
            "attempt": attempt,
            "seed_pair": fc.render_seed_pair(char_id, render_index, attempt, defaults),
            "frame_count_expected": render["frame_count"],
            "sha256": fc.sha256_file(destination),
            "elapsed_seconds": round(elapsed, 2),
            "comfy_output": str(produced),
        }
        record_timing({"character_id": char_id, **{k: result[k] for k in ("render_key", "attempt", "elapsed_seconds")}})
        print(f"  SAVED  {destination.relative_to(fc.FACTORY_ROOT)}  {elapsed/60:.1f} min", flush=True)
        return result

    raise fc.FactoryError(f"Timed out after {comfy['job_timeout_seconds']}s waiting for {char_id}/{render_key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", required=True, help="Character IDs to render.")
    parser.add_argument("--renders", nargs="*", help="Restrict to these render keys.")
    parser.add_argument("--tier", help="Override the roster tier for these characters.")
    parser.add_argument("--attempt", type=int, default=0, help="Reroll counter; shifts the seed pair by +1000 each.")
    parser.add_argument("--force", action="store_true", help="Re-render even when the ledger has the render.")
    parser.add_argument("--dry-run", action="store_true", help="Patch and print without contacting ComfyUI.")
    args = parser.parse_args()

    defaults = fc.load_generation_defaults()
    prompt_pack = fc.load_pose_prompt_pack()
    pose_spec = fc.load_pose_spec()
    roster = fc.roster_index()
    ledger = fc.load_ledger()
    fc.ensure_state_dirs()

    workflow_path = Path(defaults["comfyui"]["workflow_api_json"])
    if not workflow_path.exists():
        raise fc.FactoryError(
            f"Workflow graph not found: {workflow_path}. This host needs the LTX-2.3 repository "
            "checked out and the Q4 bootstrap run."
        )
    base_workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    if not args.dry_run:
        try:
            request_json(f"{defaults['comfyui']['base_url']}/system_stats", timeout=15)
        except (urllib.error.URLError, OSError) as error:
            raise fc.FactoryError(
                f"ComfyUI is not reachable at {defaults['comfyui']['base_url']}: {error}. "
                "Start it with /workspace/LTX-2.3/scripts/start-q4-comfyui.sh"
            )

    unknown = set(args.characters) - set(roster)
    if unknown:
        raise fc.FactoryError(f"Unknown character IDs: {', '.join(sorted(unknown))}")

    for char_id in args.characters:
        character = roster[char_id]
        tier = args.tier or character["tier"]
        renders = fc.renders_for_tier(tier, pose_spec)
        if args.renders:
            wanted = set(args.renders)
            renders = [r for r in renders if r["render_key"] in wanted]
        entry = fc.ledger_entry(ledger, char_id)

        print(f"{char_id} ({character['display_name']}, {character['language_name'] or 'agnostic'}) "
              f"tier={tier} renders={len(renders)}")
        for render_index, render in enumerate(renders):
            render_key = render["render_key"]
            if not args.force and render_key in entry["renders"]:
                existing = fc.RENDER_DIR / char_id / f"{render_key}.mp4"
                if existing.exists():
                    print(f"  SKIP   {render_key} already rendered")
                    continue
            result = run_render(
                character=character, render=render, render_index=render_index, defaults=defaults,
                base_workflow=base_workflow, prompt_pack=prompt_pack, attempt=args.attempt,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                entry["renders"][render_key] = result
                entry["stage"] = "rendered"
                fc.save_ledger(ledger)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
