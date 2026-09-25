"""Resolve browser-lab characters only from the configured motion registry.

Catalog entries are package directories, never client-supplied filesystem paths.
The selected package is checked against the same motion registry as live sessions.
"""
import json
import logging
import os
from pathlib import Path
import re

from scripts.motion_transitions import IDLE, TALK, SMILE, configured_bank
from scripts.pose_protocol import normalize_pose_set

LOG = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_NAMES = {IDLE: "idle", TALK: "talking", SMILE: "smiling"}


class PoseLabCatalogError(ValueError):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def _package_directories():
    explicit = os.environ.get("WEBRTC_MOTION_ATLAS", "").strip()
    registry = os.environ.get("WEBRTC_MOTION_ATLAS_DIR", "").strip()
    paths = [Path(explicit).expanduser().resolve().parent] if explicit else []
    if registry:
        paths.extend(path.parent.resolve() for path in sorted(
            Path(registry).expanduser().glob("*/motion-atlas.json")))
    result = {}
    for path in dict.fromkeys(paths):
        if not _IDENTIFIER.fullmatch(path.name):
            LOG.warning("Skipping pose-lab package with unsupported directory name: %s", path)
            continue
        if path.name in result and result[path.name] != path:
            raise PoseLabCatalogError("Configured character directories have duplicate names; use unique package directory names.")
        result[path.name] = path
    return bool(explicit or registry), result


def _read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _selected_context(directory, package):
    atlas_path = directory / "motion-atlas.json"
    pose_path = directory / "session-pose-set.json"
    if (Path(package["motion_atlas"]).resolve() != atlas_path.resolve()
            or Path(package["pose_set"]).resolve() != pose_path.resolve()):
        raise ValueError("Package artifact paths do not match its directory")
    atlas = _read_json(atlas_path)
    paths = {pose: atlas["sources"][pose]["path"] for pose in _NAMES}
    bank = configured_bank(paths)
    if bank is None:
        raise ValueError("Selected sources do not activate a registered motion bank")
    if bank.routing_sha256 != package["motion_atlas_content_sha256"]:
        raise ValueError("Selected package routing differs from its registered motion bank")
    expected_hashes = {pose: bank.sources[pose]["sha256"] for pose in _NAMES}
    if package["source_hashes"] != {_NAMES[p]: value for p, value in expected_hashes.items()}:
        raise ValueError("Selected package source hashes differ from its motion bank")
    poses = normalize_pose_set(_read_json(pose_path))
    if poses["pose_set_id"] != package["character_id"]:
        raise ValueError("Session manifest names a different character")
    if poses["default_pose_id"] != IDLE:
        raise ValueError("Browser character must start on the neutral idle source")
    preparation = package.get("prepared", [])
    if not isinstance(preparation, list) or any(not isinstance(entry, dict) for entry in preparation):
        raise ValueError("Character preparation receipt must be a list of objects")
    prepared = {entry["avatar_id"] for entry in preparation if entry.get("status") == "ready"}
    for logical_pose, entry in poses["poses"].items():
        physical = logical_pose if logical_pose in _NAMES else IDLE
        source = bank.sources[physical]
        expected_id = f"{package['character_id']}_{_NAMES[physical]}_{source['sha256'][:10]}"
        if entry["avatar_id"] != expected_id or entry.get("variants"):
            raise ValueError("Session manifest does not use the package's three physical caches")
        if expected_id not in prepared:
            raise ValueError("Selected character caches need API preparation before browser playback")
        if (entry.get("frame_count") != source["frame_count"]
                or entry.get("fps") != bank.fps):
            raise ValueError("Session manifest timing differs from its source bank")
    return {"pose_set": poses,
            "expected_motion": {"routing_sha256": bank.routing_sha256,
                                "source_hashes": expected_hashes}}


def pose_lab_context(character=None):
    """Return safe template arguments, or an explicit error instead of fallback."""
    if character is not None and not _IDENTIFIER.fullmatch(character):
        raise PoseLabCatalogError("Character must be a configured package name, not a path.", 400)
    configured, directories = _package_directories()
    if not configured:
        if character is not None:
            raise PoseLabCatalogError("No motion character registry is configured.", 404)
        return {}  # The explicitly labelled legacy pose-protocol lab.
    if character is not None and character not in directories:
        raise PoseLabCatalogError("Character is not in the configured motion registry.", 404)
    packages = {}
    for name, directory in directories.items():
        try:
            package = _read_json(directory / "character.json")
            if not _IDENTIFIER.fullmatch(package["character_id"]):
                raise ValueError("Invalid package character ID")
            packages[name] = package
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if character == name:
                raise PoseLabCatalogError("Selected character package is missing or invalid.") from exc
            LOG.warning("Skipping incomplete pose-lab package %s: %s", directory, exc)
    if character is None:
        # Resolve only usable packages; an unfinished unrelated package must not
        # prevent a prepared character from appearing in the browser lab.
        for name, package in packages.items():
            try:
                selected = _selected_context(directories[name], package)
                character = name
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                LOG.warning("Skipping unavailable pose-lab character %s: %s", name, exc)
        else:
            raise PoseLabCatalogError("No prepared, permitted character package is available in the configured registry.")
    else:
        try:
            selected = _selected_context(directories[character], packages[character])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PoseLabCatalogError(f"Selected character is unavailable: {exc}") from exc
    return {**selected,
            "characters": [{"id": name, "label": package["character_id"]}
                           for name, package in packages.items()],
            "selected_character": character}
