"""Validation helpers for Lingua's deterministic MuseTalk pose protocol v1.

This module deliberately has no FastAPI or GPU dependencies so the contract can
be unit-tested without importing the MuseTalk runtime.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping


POSE_PROTOCOL_VERSION = 1
POSE_SWITCH_MODE = "next_boundary"
POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "nod_agree",
    "empathetic_head_tilt",
    "light_smile",
)
POSE_ID_SET = frozenset(POSE_IDS)

REACTION_INTENTS = ("none", "acknowledge", "warmth", "empathy")
REACTION_INTENT_SET = frozenset(REACTION_INTENTS)
REACTION_POSE = {
    "acknowledge": "nod_agree",
    "warmth": "light_smile",
    "empathy": "empathetic_head_tilt",
}

SESSION_EVENTS = (
    "user_speech_started",
    "user_speech_ended",
    "assistant_thinking",
    "assistant_reaction_ready",
    "assistant_turn_aborted",
)
SESSION_EVENT_SET = frozenset(SESSION_EVENTS)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class PoseProtocolError(ValueError):
    """Raised when pose-protocol input is malformed or unsupported."""


def _identifier(value: Any, label: str, *, required: bool = True) -> str:
    result = str(value or "").strip()
    if not result and not required:
        return ""
    if not _IDENTIFIER_RE.fullmatch(result):
        raise PoseProtocolError(
            f"{label} must be 1-128 letters, numbers, dots, underscores, or dashes."
        )
    if result in {".", ".."}:
        raise PoseProtocolError(f"{label} is not allowed.")
    return result


def _positive_number(value: Any, label: str, *, integral: bool = False) -> int | float:
    if isinstance(value, bool):
        raise PoseProtocolError(f"{label} must be a positive number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PoseProtocolError(f"{label} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise PoseProtocolError(f"{label} must be a positive number.")
    if integral:
        if not number.is_integer():
            raise PoseProtocolError(f"{label} must be a positive integer.")
        return int(number)
    return int(number) if number.is_integer() else number


def normalize_pose_set(value: str | Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact six-cache manifest sent during session creation."""

    raw: Any = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PoseProtocolError("pose_set must be valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise PoseProtocolError("pose_set must be a JSON object.")

    try:
        version = int(raw.get("version", POSE_PROTOCOL_VERSION))
    except (TypeError, ValueError) as exc:
        raise PoseProtocolError("pose_set.version must be an integer.") from exc
    if version != POSE_PROTOCOL_VERSION:
        raise PoseProtocolError(f"Unsupported pose_set version {version}.")

    raw_poses = raw.get("poses")
    if not isinstance(raw_poses, Mapping) or set(raw_poses) != POSE_ID_SET:
        raise PoseProtocolError(
            "pose_set.poses must contain exactly: " + ", ".join(POSE_IDS) + "."
        )

    poses: dict[str, dict[str, Any]] = {}
    for pose_id in POSE_IDS:
        entry = raw_poses.get(pose_id)
        if not isinstance(entry, Mapping):
            raise PoseProtocolError(f"pose_set.poses.{pose_id} must be an object.")
        normalized_entry: dict[str, Any] = {
            "avatar_id": _identifier(
                entry.get("avatar_id"),
                f"pose_set.poses.{pose_id}.avatar_id",
            ),
            "role": str(entry.get("role") or "").strip().lower(),
        }
        for field_name in ("duration_seconds", "cycle_seconds", "fps", "frame_count"):
            field_value = entry.get(field_name)
            if field_value is not None:
                normalized_entry[field_name] = _positive_number(
                    field_value,
                    f"pose_set.poses.{pose_id}.{field_name}",
                    integral=field_name == "frame_count",
                )
        poses[pose_id] = normalized_entry

    default_pose_id = str(
        raw.get("default_pose_id") or "neutral_resting"
    ).strip().lower()
    if default_pose_id not in POSE_ID_SET:
        raise PoseProtocolError("pose_set.default_pose_id is unsupported.")

    switch_mode = str(
        raw.get("switch_mode") or POSE_SWITCH_MODE
    ).strip().lower()
    if switch_mode != POSE_SWITCH_MODE:
        raise PoseProtocolError(
            f"pose_set.switch_mode must be {POSE_SWITCH_MODE}."
        )

    return {
        "version": POSE_PROTOCOL_VERSION,
        "pose_set_id": _identifier(
            raw.get("pose_set_id"),
            "pose_set.pose_set_id",
            required=False,
        ),
        "default_pose_id": default_pose_id,
        "switch_mode": POSE_SWITCH_MODE,
        "poses": poses,
    }


def normalize_reaction_intent(value: Any) -> str:
    intent = str(value or "none").strip().lower()
    if intent not in REACTION_INTENT_SET:
        raise PoseProtocolError(
            "reaction_intent must be one of: " + ", ".join(REACTION_INTENTS) + "."
        )
    return intent


def build_turn_pose_sequence(reaction_intent: Any = "none") -> list[str]:
    intent = normalize_reaction_intent(reaction_intent)
    sequence: list[str] = []
    reaction_pose = REACTION_POSE.get(intent)
    if reaction_pose:
        sequence.append(reaction_pose)
    sequence.extend(("speaking_direct", "neutral_resting"))
    return sequence


def normalize_pose_sequence(
    value: str | list[Any] | None,
    *,
    reaction_intent: Any = "none",
) -> list[str]:
    expected = build_turn_pose_sequence(reaction_intent)
    if value in (None, ""):
        return expected
    raw: Any = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PoseProtocolError("pose_sequence must be a JSON array.") from exc
    if not isinstance(raw, list) or not raw:
        raise PoseProtocolError("pose_sequence must be a non-empty JSON array.")
    sequence = [str(item or "").strip().lower() for item in raw]
    if sequence != expected:
        raise PoseProtocolError(
            "pose_sequence must match reaction → speaking_direct → neutral_resting."
        )
    return sequence


def normalize_session_event(value: Mapping[str, Any]) -> dict[str, Any]:
    event = str(value.get("event") or "").strip().lower()
    if event not in SESSION_EVENT_SET:
        raise PoseProtocolError(
            "event must be one of: " + ", ".join(SESSION_EVENTS) + "."
        )
    try:
        seq = int(value.get("seq"))
    except (TypeError, ValueError) as exc:
        raise PoseProtocolError("seq must be a non-negative integer.") from exc
    if seq < 0:
        raise PoseProtocolError("seq must be a non-negative integer.")

    result: dict[str, Any] = {"event": event, "seq": seq}
    turn_id = _identifier(value.get("turn_id"), "turn_id", required=False)
    if turn_id:
        result["turn_id"] = turn_id
    if event == "assistant_reaction_ready":
        result["reaction_intent"] = normalize_reaction_intent(
            value.get("reaction_intent")
        )
    return result


def normalize_stream_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate optional multipart fields forwarded with assistant audio."""

    protocol_keys = {
        "reaction_intent",
        "pose_sequence",
        "turn_id",
        "seq",
        "pose_id",
        "effective",
        "mouth_mode",
        "audio_start",
    }
    if not any(key in value and value.get(key) not in (None, "") for key in protocol_keys):
        return {}

    reaction_intent = normalize_reaction_intent(value.get("reaction_intent"))
    sequence = normalize_pose_sequence(
        value.get("pose_sequence"),
        reaction_intent=reaction_intent,
    )
    pose_id = str(value.get("pose_id") or "speaking_direct").strip().lower()
    if pose_id != "speaking_direct":
        raise PoseProtocolError("Assistant audio must use pose_id=speaking_direct.")

    turn_id = _identifier(value.get("turn_id"), "turn_id")
    try:
        seq = int(value.get("seq"))
    except (TypeError, ValueError) as exc:
        raise PoseProtocolError("seq must be a non-negative integer.") from exc
    if seq < 0:
        raise PoseProtocolError("seq must be a non-negative integer.")

    effective = str(value.get("effective") or POSE_SWITCH_MODE).strip().lower()
    mouth_mode = str(value.get("mouth_mode") or "lip_sync").strip().lower()
    audio_start = str(value.get("audio_start") or "immediate").strip().lower()
    if effective != POSE_SWITCH_MODE:
        raise PoseProtocolError(f"effective must be {POSE_SWITCH_MODE}.")
    if mouth_mode != "lip_sync":
        raise PoseProtocolError("mouth_mode must be lip_sync.")
    if audio_start != "immediate":
        raise PoseProtocolError("audio_start must be immediate.")

    return {
        "reaction_intent": reaction_intent,
        "pose_id": pose_id,
        "pose_sequence": sequence,
        "turn_id": turn_id,
        "seq": seq,
        "effective": POSE_SWITCH_MODE,
        "mouth_mode": "lip_sync",
        "audio_start": "immediate",
    }
