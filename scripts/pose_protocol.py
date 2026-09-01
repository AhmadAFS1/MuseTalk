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
POSE_PLAN_VERSION = 2
POSE_PLAN_CLOCK = "audio_progress"
POSE_PLAN_MAX_SEGMENTS = 3
POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "nod_agree",
    "empathetic_head_tilt",
    "light_smile",
)
POSE_ID_SET = frozenset(POSE_IDS)
POSE_VARIANT_POLICY = "deterministic_boundary_rotation"
POSE_VARIANT_MAX_COUNT = 4
POSE_VARIANT_SUPPORTED_POSES = frozenset({"speaking_direct"})

REACTION_INTENTS = ("none", "acknowledge", "warmth", "empathy")
REACTION_INTENT_SET = frozenset(REACTION_INTENTS)
REACTION_POSE = {
    "acknowledge": "nod_agree",
    "warmth": "light_smile",
    "empathy": "empathetic_head_tilt",
}

SPEECH_POSE_IDS = (
    "speaking_direct",
    "empathetic_head_tilt",
    "light_smile",
)
SPEECH_POSE_ID_SET = frozenset(SPEECH_POSE_IDS)

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


def pose_variant_render_key(pose_id: str, variant_id: str) -> str:
    """Return the internal cache/router key for one physical pose variant."""

    return f"{str(pose_id).strip().lower()}__variant__{str(variant_id).strip().lower()}"


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
    """Validate the six-logical-pose manifest sent during session creation."""

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
        raw_variants = entry.get("variants")
        raw_variant_policy = entry.get("variant_policy")
        if raw_variants is not None or raw_variant_policy is not None:
            if pose_id not in POSE_VARIANT_SUPPORTED_POSES:
                raise PoseProtocolError(
                    f"pose_set.poses.{pose_id} does not support variants."
                )
            if (
                not isinstance(raw_variants, list)
                or not 2 <= len(raw_variants) <= POSE_VARIANT_MAX_COUNT
            ):
                raise PoseProtocolError(
                    f"pose_set.poses.{pose_id}.variants must contain 2-"
                    f"{POSE_VARIANT_MAX_COUNT} items."
                )
            variant_policy = str(raw_variant_policy or "").strip().lower()
            if variant_policy != POSE_VARIANT_POLICY:
                raise PoseProtocolError(
                    f"pose_set.poses.{pose_id}.variant_policy must be "
                    f"{POSE_VARIANT_POLICY}."
                )
            variants: list[dict[str, str]] = []
            variant_ids: set[str] = set()
            variant_avatar_ids: set[str] = set()
            for index, raw_variant in enumerate(raw_variants):
                if not isinstance(raw_variant, Mapping):
                    raise PoseProtocolError(
                        f"pose_set.poses.{pose_id}.variants[{index}] must be an object."
                    )
                variant_id = _identifier(
                    raw_variant.get("variant_id"),
                    f"pose_set.poses.{pose_id}.variants[{index}].variant_id",
                ).lower()
                variant_avatar_id = _identifier(
                    raw_variant.get("avatar_id"),
                    f"pose_set.poses.{pose_id}.variants[{index}].avatar_id",
                )
                if variant_id in variant_ids:
                    raise PoseProtocolError(
                        f"pose_set.poses.{pose_id}.variants has duplicate "
                        f"variant_id '{variant_id}'."
                    )
                if variant_avatar_id in variant_avatar_ids:
                    raise PoseProtocolError(
                        f"pose_set.poses.{pose_id}.variants has duplicate "
                        f"avatar_id '{variant_avatar_id}'."
                    )
                variant_ids.add(variant_id)
                variant_avatar_ids.add(variant_avatar_id)
                variants.append(
                    {
                        "variant_id": variant_id,
                        "avatar_id": variant_avatar_id,
                    }
                )
            if normalized_entry["avatar_id"] not in variant_avatar_ids:
                raise PoseProtocolError(
                    f"pose_set.poses.{pose_id}.avatar_id must appear in variants."
                )
            normalized_entry["variants"] = variants
            normalized_entry["variant_policy"] = POSE_VARIANT_POLICY
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


def normalize_pose_plan(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate Lingua's semantic speech plan without trusting model timing."""

    if value in (None, ""):
        return {}
    raw: Any = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PoseProtocolError("pose_plan must be a JSON object.") from exc
    if not isinstance(raw, Mapping):
        raise PoseProtocolError("pose_plan must be a JSON object.")
    try:
        version = int(raw.get("version"))
    except (TypeError, ValueError) as exc:
        raise PoseProtocolError(
            f"pose_plan.version must be {POSE_PLAN_VERSION}."
        ) from exc
    if version != POSE_PLAN_VERSION:
        raise PoseProtocolError(
            f"Unsupported pose_plan version {version}."
        )
    clock = str(raw.get("clock") or "").strip().lower()
    if clock != POSE_PLAN_CLOCK:
        raise PoseProtocolError(
            f"pose_plan.clock must be {POSE_PLAN_CLOCK}."
        )
    switch_mode = str(
        raw.get("switch_mode") or POSE_SWITCH_MODE
    ).strip().lower()
    if switch_mode != POSE_SWITCH_MODE:
        raise PoseProtocolError(
            f"pose_plan.switch_mode must be {POSE_SWITCH_MODE}."
        )
    on_complete = str(
        raw.get("on_complete") or "neutral_resting"
    ).strip().lower()
    if on_complete != "neutral_resting":
        raise PoseProtocolError(
            "pose_plan.on_complete must be neutral_resting."
        )

    raw_segments = raw.get("segments")
    if (
        not isinstance(raw_segments, list)
        or not raw_segments
        or len(raw_segments) > POSE_PLAN_MAX_SEGMENTS
    ):
        raise PoseProtocolError(
            f"pose_plan.segments must contain 1-{POSE_PLAN_MAX_SEGMENTS} items."
        )

    segments: list[dict[str, Any]] = []
    previous_anchor = -1
    previous_pose = ""
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise PoseProtocolError(
                f"pose_plan.segments[{index}] must be an object."
            )
        pose_id = str(raw_segment.get("pose_id") or "").strip().lower()
        if pose_id not in SPEECH_POSE_ID_SET:
            raise PoseProtocolError(
                "pose_plan speech poses must be one of: "
                + ", ".join(SPEECH_POSE_IDS)
                + "."
            )
        raw_anchor = raw_segment.get("at_permille")
        if isinstance(raw_anchor, bool):
            raise PoseProtocolError(
                "pose_plan at_permille values must be integers."
            )
        try:
            anchor = int(raw_anchor)
        except (TypeError, ValueError) as exc:
            raise PoseProtocolError(
                "pose_plan at_permille values must be integers."
            ) from exc
        if anchor < 0 or anchor >= 1000:
            raise PoseProtocolError(
                "pose_plan at_permille values must be between 0 and 999."
            )
        if index == 0 and anchor != 0:
            raise PoseProtocolError(
                "pose_plan must start at at_permille=0."
            )
        if anchor <= previous_anchor:
            raise PoseProtocolError(
                "pose_plan at_permille values must be strictly increasing."
            )
        if pose_id == previous_pose:
            raise PoseProtocolError(
                "pose_plan cannot contain adjacent duplicate poses."
            )
        segments.append({"at_permille": anchor, "pose_id": pose_id})
        previous_anchor = anchor
        previous_pose = pose_id

    if segments[-1]["pose_id"] != "speaking_direct":
        raise PoseProtocolError(
            "pose_plan must end with pose_id=speaking_direct."
        )

    return {
        "version": POSE_PLAN_VERSION,
        "clock": POSE_PLAN_CLOCK,
        "segments": segments,
        "on_complete": "neutral_resting",
        "switch_mode": POSE_SWITCH_MODE,
    }


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
        "pose_plan",
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
    pose_plan = normalize_pose_plan(value.get("pose_plan"))
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

    result = {
        "reaction_intent": reaction_intent,
        "pose_id": pose_id,
        "pose_sequence": sequence,
        "turn_id": turn_id,
        "seq": seq,
        "effective": POSE_SWITCH_MODE,
        "mouth_mode": "lip_sync",
        "audio_start": "immediate",
    }
    if pose_plan:
        result["pose_plan"] = pose_plan
    return result
