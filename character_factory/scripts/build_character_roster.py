#!/usr/bin/env python3
"""Expand languages x archetypes into the concrete character roster.

Casting is deterministic: the same roster comes out of the same configs every time, so a
character that has already been generated keeps its identity when the roster is rebuilt for
an unrelated language. Diff state/roster.json before and after any config edit.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import factory_common as fc


def _pick(pool: list[str], *parts: str) -> str:
    """Deterministic pick keyed by the attribute name as well as the slot.

    Keying each attribute on its own derived seed decorrelates them; offsetting one shared
    seed made hair and wardrobe move in lockstep and collide inside large pools.
    """
    return pool[_seed(*parts) % len(pool)]


def _seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def _phrase_gender(phrase: str) -> str:
    """Casting phrases are written with an explicit gendered noun.

    `"woman" in phrase` must be tested first: "woman" contains "man", so a naive membership
    test for "man" matches every female phrase too.
    """
    lowered = phrase.lower()
    if "woman" in lowered:
        return "female"
    if "man" in lowered:
        return "male"
    return "unspecified"


def _gender_for_slot(slot: int, seed: int) -> str:
    """Strict alternation, with the opening gender flipped per language.

    Three slots therefore come out two-to-one, and which gender gets the extra slot varies
    by language, so the roster stays balanced overall. A larger pool such as the
    language-agnostic set lands on an even split instead of a two-thirds skew.
    """
    pattern = ("female", "male") if seed % 2 == 0 else ("male", "female")
    return pattern[(slot - 1) % len(pattern)]


def _heritage_for_gender(pool: list[str], gender: str, *parts: str) -> str:
    """Prefer a casting phrase whose wording already matches the slot's gender."""
    matching = [phrase for phrase in pool if _phrase_gender(phrase) == gender]
    return _pick(matching or pool, *parts, "heritage")


def build_character(
    *,
    char_id: str,
    language: dict[str, Any] | None,
    region: dict[str, Any],
    shared: dict[str, Any],
    slot: int,
    tier: str,
    agnostic: bool,
    taken: set[str] | None = None,
    taken_names: set[str] | None = None,
    language_heritage: dict[str, list[str]] | None = None,
    language_names: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    scope = language["code"] if language else "agnostic"
    gender = _gender_for_slot(slot, _seed(scope))
    heritage_pool = region["heritage"]
    if language and language_heritage:
        heritage_pool = language_heritage.get(language["code"], heritage_pool)
    heritage = _heritage_for_gender(heritage_pool, gender, scope, str(slot))
    age = _pick(shared["age_bands"], scope, str(slot), "age")
    hair_pool = shared["hair_f"] if gender == "female" else shared["hair_m"]
    wardrobe_pool = shared["wardrobe_f"] if gender == "female" else shared["wardrobe_m"]
    names_key = "names_f" if gender == "female" else "names_m"
    names_pool = region[names_key]
    if language and language_names:
        names_pool = language_names.get(language["code"], {}).get(names_key, names_pool)
    hair = _pick(hair_pool, scope, str(slot), "hair")
    wardrobe = _pick(wardrobe_pool, scope, str(slot), "wardrobe")

    # Nor share a display name: two "Haruka"s in one language read as a data bug to a user.
    display_name = _pick(names_pool, scope, str(slot), "name")
    if taken_names is not None:
        start = names_pool.index(display_name) if display_name in names_pool else 0
        for step in range(len(names_pool)):
            candidate = names_pool[(start + step) % len(names_pool)]
            if candidate not in taken_names:
                display_name = candidate
                break
        taken_names.add(display_name)

    # Two slots of one language must never resolve to the same look. Rotate the heritage
    # pick deterministically until the triple is unique rather than leaving it to luck in a
    # small pool; `taken` is the set of triples already cast for this language.
    if taken is not None:
        candidates = [p for p in heritage_pool if _phrase_gender(p) == gender] or heritage_pool
        start = candidates.index(heritage) if heritage in candidates else 0
        for step in range(len(candidates)):
            candidate = candidates[(start + step) % len(candidates)]
            if f"{candidate}|{hair}|{wardrobe}" not in taken:
                heritage = candidate
                break
        taken.add(f"{heritage}|{hair}|{wardrobe}")

    return {
        "character_id": char_id,
        "display_name": display_name,
        "language_code": language["code"] if language else None,
        "language_name": language["name"] if language else None,
        "language_native_name": language["native_name"] if language else None,
        "script": language["script"] if language else None,
        "rtl": language["rtl"] if language else False,
        "region_key": language["region_key"] if language else "language_agnostic",
        "language_agnostic": agnostic,
        "slot": slot,
        "tier": tier,
        "gender": gender,
        "pronouns": fc.pronouns(gender),
        "casting": {
            "subject": f"{heritage} {age}",
            "heritage": heritage,
            "heritage_source": (
                "language_override"
                if language and language_heritage and language["code"] in language_heritage
                else "region_pool"
            ),
            "age_band": age,
            "hair": hair,
            "wardrobe": wardrobe,
            "setting": _pick(region["settings"], scope, str(slot), "setting"),
            "lighting": _pick(shared["lighting"], scope, str(slot), "lighting"),
        },
        "pose_set_id": fc.pose_set_id(char_id),
        "bank_dir": fc.bank_dir_name(char_id),
        "status": "planned",
    }


def build_roster(
    *,
    tier: str,
    flagship_tier: str,
    characters_per_language: int,
    agnostic_count: int,
    only_languages: set[str] | None,
) -> dict[str, Any]:
    languages = fc.load_languages()
    archetypes = fc.load_archetypes()
    shared = archetypes["shared"]
    regions = archetypes["regions"]
    language_heritage = archetypes.get("language_heritage", {})
    language_names = archetypes.get("language_names", {})

    characters: list[dict[str, Any]] = []
    for language in languages["languages"]:
        if only_languages and language["code"] not in only_languages:
            continue
        taken: set[str] = set()
        taken_names: set[str] = set()
        region = regions.get(language["region_key"])
        if region is None:
            raise fc.FactoryError(
                f"Language {language['code']} names region {language['region_key']!r}, "
                "which config/archetypes.json does not define."
            )
        for slot in range(1, characters_per_language + 1):
            char_id = fc.character_id(language["code"], slot)
            characters.append(
                build_character(
                    char_id=char_id,
                    language=language,
                    region=region,
                    shared=shared,
                    slot=slot,
                    # Slot 1 is that language's flagship and carries the richer pose bank.
                    tier=flagship_tier if slot == 1 else tier,
                    agnostic=False,
                    taken=taken,
                    taken_names=taken_names,
                    language_heritage=language_heritage,
                    language_names=language_names,
                )
            )

    agnostic_pool = archetypes["language_agnostic"]
    agnostic_region = {
        "heritage": agnostic_pool["heritage"],
        "names_f": agnostic_pool["names_f"],
        "names_m": agnostic_pool["names_m"],
        "settings": agnostic_pool["settings"],
    }
    agnostic_taken: set[str] = set()
    agnostic_taken_names: set[str] = set()
    for slot in range(1, agnostic_count + 1):
        characters.append(
            build_character(
                char_id=fc.agnostic_character_id(slot),
                language=None,
                region=agnostic_region,
                shared=shared,
                slot=slot,
                # Agnostic characters are reused everywhere, so they always get the full bank.
                tier="full",
                agnostic=True,
                taken=agnostic_taken,
                taken_names=agnostic_taken_names,
            )
        )

    spec = fc.load_pose_spec()
    render_total = sum(len(spec["tiers"][entry["tier"]]["renders"]) for entry in characters)

    return {
        "schema_version": 1,
        "source_languages": str(fc.LANGUAGES_PATH.relative_to(fc.FACTORY_ROOT)),
        "source_archetypes": str(fc.ARCHETYPES_PATH.relative_to(fc.FACTORY_ROOT)),
        "languages_status": fc.load_languages()["status"],
        "characters_per_language": characters_per_language,
        "default_tier": tier,
        "flagship_tier": flagship_tier,
        "totals": {
            "languages": len({c["language_code"] for c in characters if c["language_code"]}),
            "language_characters": sum(1 for c in characters if not c["language_agnostic"]),
            "agnostic_characters": sum(1 for c in characters if c["language_agnostic"]),
            "characters": len(characters),
            "ltx_renders": render_total,
        },
        "characters": characters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="core", help="Pose tier for non-flagship characters.")
    parser.add_argument("--flagship-tier", default="full", help="Pose tier for slot 1 of each language.")
    parser.add_argument("--characters-per-language", type=int, default=3)
    parser.add_argument("--agnostic-count", type=int, default=20)
    parser.add_argument("--languages", nargs="*", help="Restrict to these BCP-47 codes.")
    parser.add_argument("--out", type=Path, default=fc.ROSTER_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print the summary without writing.")
    args = parser.parse_args()

    roster = build_roster(
        tier=args.tier,
        flagship_tier=args.flagship_tier,
        characters_per_language=args.characters_per_language,
        agnostic_count=args.agnostic_count,
        only_languages=set(args.languages) if args.languages else None,
    )

    totals = roster["totals"]
    print(
        f"{totals['characters']} characters "
        f"({totals['language_characters']} across {totals['languages']} languages, "
        f"{totals['agnostic_characters']} language-agnostic) "
        f"=> {totals['ltx_renders']} LTX renders"
    )
    name_clashes = _duplicate_names(roster["characters"])
    if name_clashes:
        print(f"WARNING: {len(name_clashes)} languages reuse a display name across slots: "
              + ", ".join(sorted(name_clashes)[:8]))
        print("         Add more names for those codes under `language_names` in config/archetypes.json.")
    duplicates = _duplicate_faces(roster["characters"])
    if duplicates:
        print(f"WARNING: {len(duplicates)} languages cast two slots identically: "
              + ", ".join(sorted(duplicates)[:8]) + ("..." if len(duplicates) > 8 else ""))
        print("         Widen that region's `heritage` pool in config/archetypes.json.")

    if args.dry_run:
        return 0
    fc.ensure_state_dirs()
    fc.write_json(args.out, roster)
    print(f"Wrote {args.out}")
    return 0


def _duplicate_names(characters: list[dict[str, Any]]) -> set[str]:
    seen: dict[str, set[str]] = {}
    clashes: set[str] = set()
    for entry in characters:
        scope = entry["language_code"] or "agnostic"
        bucket = seen.setdefault(scope, set())
        if entry["display_name"] in bucket:
            clashes.add(scope)
        bucket.add(entry["display_name"])
    return clashes


def _duplicate_faces(characters: list[dict[str, Any]]) -> set[str]:
    """Two slots of one language must not resolve to the same casting phrase."""
    seen: dict[str, set[str]] = {}
    clashes: set[str] = set()
    for entry in characters:
        scope = entry["language_code"] or "agnostic"
        key = f"{entry['casting']['heritage']}|{entry['casting']['hair']}|{entry['casting']['wardrobe']}"
        bucket = seen.setdefault(scope, set())
        if key in bucket:
            clashes.add(scope)
        bucket.add(key)
    return clashes


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
