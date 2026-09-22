#!/usr/bin/env python3
"""Shared paths, config loading, and ID derivation for the character factory.

Every downstream artifact — prompt file, portrait, render, pose bank, runtime manifest,
prepared MuseTalk avatar — is named from the functions in here. Change an ID rule and you
orphan everything already produced, so treat these as frozen once a bulk run starts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zlib
from pathlib import Path
from typing import Any

FACTORY_ROOT = Path(__file__).resolve().parents[1]
MUSETALK_ROOT = FACTORY_ROOT.parent

CONFIG_DIR = FACTORY_ROOT / "config"
PROMPT_PACK_DIR = CONFIG_DIR / "prompt_packs"
STATE_DIR = FACTORY_ROOT / "state"

LANGUAGES_PATH = CONFIG_DIR / "languages.json"
ARCHETYPES_PATH = CONFIG_DIR / "archetypes.json"
POSE_SPEC_PATH = CONFIG_DIR / "pose_spec.json"
GENERATION_DEFAULTS_PATH = CONFIG_DIR / "generation_defaults.json"
POSE_PROMPT_PACK_PATH = PROMPT_PACK_DIR / "pose_prompt_pack_v1.json"
PORTRAIT_TEMPLATE_PATH = PROMPT_PACK_DIR / "portrait_prompt_template.md"

ROSTER_PATH = STATE_DIR / "roster.json"
LEDGER_PATH = STATE_DIR / "ledger.json"
TIMINGS_PATH = STATE_DIR / "timings.jsonl"
PROMPT_OUT_DIR = STATE_DIR / "portrait_prompts"
PORTRAIT_INBOX = STATE_DIR / "portrait_inbox"
PORTRAIT_CANONICAL = STATE_DIR / "portraits"
RENDER_DIR = STATE_DIR / "renders"
CERTIFIED_DIR = STATE_DIR / "certified"

POSE_BANK_ROOT = MUSETALK_ROOT / "assets" / "ltx23_pose_banks"
RUNTIME_MANIFEST_DIR = MUSETALK_ROOT / "configs" / "pose_test"

# Fixed by scripts/pose_protocol.py. Never reorder or extend.
POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "nod_agree",
    "empathetic_head_tilt",
    "light_smile",
)


class FactoryError(RuntimeError):
    """Raised for any recoverable factory failure worth reporting by itself."""


# --------------------------------------------------------------------------- config


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FactoryError(f"Missing config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FactoryError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_languages() -> dict[str, Any]:
    return load_json(LANGUAGES_PATH)


def load_archetypes() -> dict[str, Any]:
    return load_json(ARCHETYPES_PATH)


def load_pose_spec() -> dict[str, Any]:
    return load_json(POSE_SPEC_PATH)


def load_generation_defaults() -> dict[str, Any]:
    return load_json(GENERATION_DEFAULTS_PATH)


def load_pose_prompt_pack() -> dict[str, Any]:
    return load_json(POSE_PROMPT_PACK_PATH)


def load_roster() -> dict[str, Any]:
    roster = load_json(ROSTER_PATH)
    if not roster.get("characters"):
        raise FactoryError(f"{ROSTER_PATH} holds no characters. Run build_character_roster.py first.")
    return roster


def roster_index() -> dict[str, dict[str, Any]]:
    return {entry["character_id"]: entry for entry in load_roster()["characters"]}


def renders_for_tier(tier: str, pose_spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    spec = pose_spec or load_pose_spec()
    tiers = spec["tiers"]
    if tier not in tiers:
        raise FactoryError(f"Unknown tier {tier!r}. Known: {', '.join(sorted(tiers))}")
    wanted = tiers[tier]["renders"]
    by_key = {render["render_key"]: render for render in spec["renders"]}
    return [by_key[key] for key in wanted]


# ------------------------------------------------------------------------------ ids


def character_id(language_code: str, slot: int) -> str:
    """`ar_01`. Slots are 1-based and stable; slot 2 is always the same person."""
    return f"{normalize_code(language_code)}_{slot:02d}"


def agnostic_character_id(slot: int) -> str:
    return f"agn_{slot:02d}"


def normalize_code(language_code: str) -> str:
    """BCP-47 codes carry dashes; IDs and filenames use underscores."""
    return language_code.replace("-", "_").lower()


def pose_set_id(char_id: str) -> str:
    return f"lingua_{char_id}_ltx23_facetime_closeup_v1"


def bank_dir_name(char_id: str) -> str:
    return f"lingua_{char_id}_facetime_closeup_v1"


def avatar_id(char_id: str, render_key: str, content_sha256: str) -> str:
    """Content-hashed, matching the convention set by the 2026-09-01 migration.

    The first eight characters of the video hash are embedded so a regenerated render can
    never collide with a stale prepared cache.
    """
    return f"lingua_{char_id}_{render_key}_{content_sha256[:8]}"


def render_seed_pair(char_id: str, render_index: int, attempt: int, defaults: dict[str, Any]) -> tuple[int, int]:
    """Deterministic per-character, per-render, per-attempt seeds.

    crc32 rather than a hash digest so the value is stable across Python builds and
    trivially reproducible by hand when a render has to be explained.
    """
    seeds = defaults["seeds"]
    offset = (zlib.crc32(char_id.encode("utf-8")) % 100_000) * 8 + render_index + attempt * 1000
    return seeds["base_stage1"] + offset, seeds["base_stage2"] + offset


def pronouns(gender: str) -> dict[str, str]:
    """Casting gender drives the portrait wording. Unknown values fall back to they/them."""
    if gender == "female":
        return {"subject": "she", "object": "her", "possessive": "her"}
    if gender == "male":
        return {"subject": "he", "object": "him", "possessive": "his"}
    return {"subject": "they", "object": "them", "possessive": "their"}


# ---------------------------------------------------------------------------- media


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
            "-count_frames", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout).get("streams", [])


def require_binaries(*names: str) -> None:
    missing = [
        name for name in names
        if subprocess.run(["which", name], capture_output=True).returncode != 0
    ]
    if missing:
        raise FactoryError(
            "Missing required binaries: " + ", ".join(missing)
            + ". ffmpeg/ffprobe are needed for certification; install them before running."
        )


# ---------------------------------------------------------------------------- ledger


def load_ledger() -> dict[str, Any]:
    if not LEDGER_PATH.exists():
        return {"schema_version": 1, "characters": {}}
    return load_json(LEDGER_PATH)


def ledger_entry(ledger: dict[str, Any], char_id: str) -> dict[str, Any]:
    return ledger.setdefault("characters", {}).setdefault(
        char_id,
        {
            "portrait": None,
            "renders": {},
            "certified": {},
            "bank": None,
            "runtime_manifest": None,
            "prepared_avatars": {},
            "stage": "new",
        },
    )


def save_ledger(ledger: dict[str, Any]) -> None:
    write_json(LEDGER_PATH, ledger)


def ensure_state_dirs() -> None:
    for directory in (
        STATE_DIR, PROMPT_OUT_DIR, PORTRAIT_INBOX, PORTRAIT_CANONICAL, RENDER_DIR, CERTIFIED_DIR
    ):
        directory.mkdir(parents=True, exist_ok=True)
