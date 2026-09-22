#!/usr/bin/env python3
"""Prepare one MuseTalk avatar cache per physical clip in a packaged bank.

Each physical video needs its own prepared cache, because MuseTalk precomputes face
crops and latents per source video. The manifest's six logical poses therefore map onto
however many unique clips the tier produced — five or six for a normal bank.

Avatar IDs embed the first eight characters of each clip's content hash, so a regenerated
clip can never be served from a stale cache. That convention comes from the 2026-09-01
production migration and must not be loosened.

Uses only the standard library so it runs in whatever interpreter is nearest the API host.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import factory_common as fc


def _multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----characterfactory{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    for name, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
        )
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def prepare_one(
    *, base_url: str, avatar_id: str, video: Path, idle_video: Path | None,
    batch_size: int, bbox_shift: int, force_recreate: bool, timeout: int,
) -> dict[str, Any]:
    query = (
        f"?avatar_id={urllib.parse.quote(avatar_id)}&batch_size={batch_size}"
        f"&bbox_shift={bbox_shift}&force_recreate={str(force_recreate).lower()}"
    )
    files = {"video_file": video}
    if idle_video is not None:
        files["idle_video_file"] = idle_video
    body, content_type = _multipart({}, files)
    request = urllib.request.Request(
        f"{base_url}/avatars/prepare{query}",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", nargs="*", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--bbox-shift", type=int, default=0)
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roster = fc.roster_index()
    ledger = fc.load_ledger()
    failures = 0

    for char_id in args.characters:
        if char_id not in roster:
            raise fc.FactoryError(f"Unknown character: {char_id}")
        entry = fc.ledger_entry(ledger, char_id)
        if not entry.get("bank"):
            raise fc.FactoryError(f"{char_id} is not packaged. Run package_pose_bank.py first.")

        bank_dir = fc.MUSETALK_ROOT / entry["bank"]
        manifest = fc.load_json(bank_dir / "manifest.json")
        idle_clip = bank_dir / "certified" / "idle_active_listening.mp4"

        print(f"{char_id} ({manifest['display_name']}): {len(manifest['avatar_ids'])} caches")
        for render_key, avatar in manifest["avatar_ids"].items():
            clip = bank_dir / "certified" / f"{render_key}.mp4"
            if not clip.exists():
                raise fc.FactoryError(f"Missing packaged clip: {clip}")

            if not args.force_recreate and entry.get("prepared_avatars", {}).get(render_key) == avatar:
                print(f"  SKIP    {render_key} -> {avatar}")
                continue
            if args.dry_run:
                print(f"  [dry-run] POST /avatars/prepare avatar_id={avatar} video={clip.name}")
                continue

            started = time.monotonic()
            try:
                # The idle loop is sent alongside every cache so WebRTC has something to play
                # while no audio is active, matching how the production banks were prepared.
                response = prepare_one(
                    base_url=args.base_url, avatar_id=avatar, video=clip,
                    idle_video=idle_clip if idle_clip.exists() and clip != idle_clip else None,
                    batch_size=args.batch_size, bbox_shift=args.bbox_shift,
                    force_recreate=args.force_recreate, timeout=args.timeout,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
                print(f"  FAIL    {render_key} -> {avatar}: {error}")
                failures += 1
                continue

            elapsed = time.monotonic() - started
            entry.setdefault("prepared_avatars", {})[render_key] = avatar
            fc.save_ledger(ledger)
            print(f"  PREPARE {render_key} -> {avatar}  {elapsed:.0f}s  {json.dumps(response)[:120]}")

        if not args.dry_run and not failures:
            entry["stage"] = "prepared"
            fc.save_ledger(ledger)

    if failures:
        print(f"\n{failures} cache(s) failed to prepare.")
        return 1
    if not args.dry_run:
        print("\nNext: run a WebRTC pose cycle against the new pose set before calling it reviewed:")
        print("  python3 scripts/test_pose_webrtc.py --manifest configs/pose_test/<pose_set_id>.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except fc.FactoryError as error:
        raise SystemExit(f"ERROR: {error}")
