#!/usr/bin/env python3
"""Validate ChatGPT-generated portraits and canonicalise them to 480x832.

A bad portrait multiplies: it is re-encoded into five or six 10-second renders that then
need certification and avatar preparation. Everything cheap enough to check here is checked
here, and a failure names the character so it can simply be regenerated.

Face detection is optional. If OpenCV is importable the portrait is checked for exactly one
front-facing face; if not, that gate is reported as skipped rather than silently passing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

import factory_common as fc

TARGET_WIDTH = 480
TARGET_HEIGHT = 832
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
ASPECT_TOLERANCE = 0.02
MIN_SOURCE_HEIGHT = 832


def _face_gate(path: Path) -> dict[str, Any]:
    """One front-facing face, if OpenCV is available.

    The Haar frontal-face cascade is deliberately coarse: it is here to catch a portrait
    with no usable face or with a second person in shot, not to judge quality. MuseTalk's
    own detector runs later during avatar preparation and is the real authority.
    """
    try:
        import cv2  # noqa: PLC0415 - optional dependency, absence must not be fatal
    except ImportError:
        return {"checked": False, "reason": "opencv not importable in this interpreter"}

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return {"checked": False, "reason": f"cascade not loadable at {cascade_path}"}

    image = cv2.imread(str(path))
    if image is None:
        return {"checked": True, "passed": False, "reason": "opencv could not decode the file"}
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
    if len(faces) == 0:
        return {"checked": True, "passed": False, "reason": "no front-facing face detected"}
    if len(faces) > 1:
        return {"checked": True, "passed": False, "reason": f"{len(faces)} faces detected; expected one"}

    x, _, w, _ = faces[0]
    centre_offset = abs((x + w / 2) / image.shape[1] - 0.5)
    if centre_offset > 0.18:
        return {
            "checked": True, "passed": False,
            "reason": f"face centre sits {centre_offset:.0%} off the horizontal midline; it will drift further under motion",
        }
    return {"checked": True, "passed": True, "face_centre_offset": round(centre_offset, 4)}


def validate_and_canonicalise(source: Path, destination: Path) -> dict[str, Any]:
    with Image.open(source) as image:
        image.load()
        width, height = image.size
        mode = image.mode

        problems: list[str] = []
        if height < MIN_SOURCE_HEIGHT:
            problems.append(
                f"source is {width}x{height}; needs at least {MIN_SOURCE_HEIGHT} px tall so the "
                f"canonical {TARGET_WIDTH}x{TARGET_HEIGHT} is a downscale, never an upscale"
            )
        aspect = width / height
        if abs(aspect - TARGET_ASPECT) > ASPECT_TOLERANCE:
            problems.append(
                f"aspect {aspect:.4f} is not 9:16 ({TARGET_ASPECT:.4f} +/- {ASPECT_TOLERANCE}); "
                "regenerate at a vertical size rather than cropping, which would change the framing"
            )
        if problems:
            return {"ok": False, "problems": problems, "source_size": [width, height], "mode": mode}

        # Flatten alpha onto white rather than dropping it: a transparent background would
        # become black in the video encoder and change the lighting the renders inherit.
        converted = image
        if mode in ("RGBA", "LA", "P"):
            converted = image.convert("RGBA")
            flattened = Image.new("RGB", converted.size, (255, 255, 255))
            flattened.paste(converted, mask=converted.split()[-1])
            converted = flattened
        elif mode != "RGB":
            converted = image.convert("RGB")

        canonical = converted.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BICUBIC)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canonical.save(destination, format="PNG", optimize=True)

    return {
        "ok": True,
        "source_size": [width, height],
        "source_mode": mode,
        "canonical_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "source_sha256": fc.sha256_file(source),
        "canonical_sha256": fc.sha256_file(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", help="Restrict to these character IDs.")
    parser.add_argument("--force", action="store_true", help="Re-ingest characters already canonicalised.")
    parser.add_argument("--skip-face-gate", action="store_true")
    args = parser.parse_args()

    fc.ensure_state_dirs()
    roster = fc.roster_index()
    ledger = fc.load_ledger()

    inbox = sorted(fc.PORTRAIT_INBOX.glob("*.png"))
    if args.characters:
        wanted = set(args.characters)
        inbox = [p for p in inbox if p.stem in wanted]
    if not inbox:
        print(f"No PNGs waiting in {fc.PORTRAIT_INBOX}")
        return 0

    accepted = rejected = skipped = 0
    for source in inbox:
        char_id = source.stem
        if char_id not in roster:
            print(f"REJECT {char_id}: not in the roster. Filenames must be exactly <character_id>.png")
            rejected += 1
            continue

        entry = fc.ledger_entry(ledger, char_id)
        destination = fc.PORTRAIT_CANONICAL / f"{char_id}.png"
        if entry.get("portrait") and destination.exists() and not args.force:
            skipped += 1
            continue

        result = validate_and_canonicalise(source, destination)
        if not result["ok"]:
            for problem in result["problems"]:
                print(f"REJECT {char_id}: {problem}")
            rejected += 1
            continue

        face = {"checked": False, "reason": "skipped by flag"} if args.skip_face_gate else _face_gate(destination)
        if face.get("checked") and not face.get("passed"):
            print(f"REJECT {char_id}: {face['reason']}")
            destination.unlink(missing_ok=True)
            rejected += 1
            continue
        if not face.get("checked"):
            print(f"  note {char_id}: face gate skipped ({face['reason']})")

        entry["portrait"] = {
            "source_file": str(source.relative_to(fc.FACTORY_ROOT)),
            "canonical_file": str(destination.relative_to(fc.FACTORY_ROOT)),
            **{k: v for k, v in result.items() if k != "ok"},
            "face_gate": face,
        }
        entry["stage"] = "portrait_ingested"
        accepted += 1
        print(f"ACCEPT {char_id}  {result['source_size'][0]}x{result['source_size'][1]}"
              f" -> {TARGET_WIDTH}x{TARGET_HEIGHT}  sha={result['canonical_sha256'][:12]}")

    fc.save_ledger(ledger)
    print(f"\naccepted={accepted} rejected={rejected} already_done={skipped}")
    if rejected:
        print("Regenerate the rejected portraits from their prompt files and re-run this script.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
