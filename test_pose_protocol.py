import json
import unittest

from scripts.pose_protocol import (
    POSE_IDS,
    PoseProtocolError,
    build_turn_pose_sequence,
    normalize_pose_plan,
    normalize_pose_set,
    normalize_session_event,
    normalize_stream_metadata,
)


def complete_pose_set():
    return {
        "version": 1,
        "pose_set_id": "indian_tutor_essential_six_v1",
        "default_pose_id": "neutral_resting",
        "switch_mode": "next_boundary",
        "poses": {
            pose_id: {
                "avatar_id": f"indian_tutor_essential_six_v1_{pose_id}",
                "role": "talking" if pose_id == "speaking_direct" else "idle",
                "fps": 30,
                "frame_count": 300,
            }
            for pose_id in POSE_IDS
        },
    }


class PoseProtocolTests(unittest.TestCase):
    def test_accepts_complete_compact_manifest(self):
        normalized = normalize_pose_set(json.dumps(complete_pose_set()))
        self.assertEqual(set(normalized["poses"]), set(POSE_IDS))
        self.assertEqual(
            normalized["poses"]["speaking_direct"]["avatar_id"],
            "indian_tutor_essential_six_v1_speaking_direct",
        )

    def test_rejects_partial_manifest(self):
        value = complete_pose_set()
        del value["poses"]["light_smile"]
        with self.assertRaises(PoseProtocolError):
            normalize_pose_set(value)

    def test_accepts_direct_speaking_variants(self):
        value = complete_pose_set()
        default_avatar_id = value["poses"]["speaking_direct"]["avatar_id"]
        value["poses"]["speaking_direct"].update(
            {
                "variants": [
                    {
                        "variant_id": "calm",
                        "avatar_id": default_avatar_id,
                    },
                    {
                        "variant_id": "reference_paced",
                        "avatar_id": "indian_tutor_direct_reference_paced",
                    },
                ],
                "variant_policy": "deterministic_boundary_rotation",
            }
        )
        normalized = normalize_pose_set(value)
        self.assertEqual(
            [
                variant["variant_id"]
                for variant in normalized["poses"]["speaking_direct"]["variants"]
            ],
            ["calm", "reference_paced"],
        )

    def test_rejects_variant_list_without_default_avatar(self):
        value = complete_pose_set()
        value["poses"]["speaking_direct"].update(
            {
                "variants": [
                    {"variant_id": "one", "avatar_id": "direct_one"},
                    {"variant_id": "two", "avatar_id": "direct_two"},
                ],
                "variant_policy": "deterministic_boundary_rotation",
            }
        )
        with self.assertRaisesRegex(PoseProtocolError, "must appear in variants"):
            normalize_pose_set(value)

    def test_rejects_variants_on_non_direct_pose(self):
        value = complete_pose_set()
        default_avatar_id = value["poses"]["light_smile"]["avatar_id"]
        value["poses"]["light_smile"].update(
            {
                "variants": [
                    {"variant_id": "one", "avatar_id": default_avatar_id},
                    {"variant_id": "two", "avatar_id": "smile_two"},
                ],
                "variant_policy": "deterministic_boundary_rotation",
            }
        )
        with self.assertRaisesRegex(PoseProtocolError, "does not support variants"):
            normalize_pose_set(value)

    def test_builds_only_supported_turn_orders(self):
        self.assertEqual(
            build_turn_pose_sequence("warmth"),
            ["light_smile", "speaking_direct", "neutral_resting"],
        )
        self.assertEqual(
            build_turn_pose_sequence("none"),
            ["speaking_direct", "neutral_resting"],
        )

    def test_normalizes_event(self):
        self.assertEqual(
            normalize_session_event(
                {
                    "event": "assistant_reaction_ready",
                    "reaction_intent": "empathy",
                    "turn_id": "turn_1",
                    "seq": 4,
                }
            ),
            {
                "event": "assistant_reaction_ready",
                "reaction_intent": "empathy",
                "turn_id": "turn_1",
                "seq": 4,
            },
        )

    def test_validates_stream_metadata(self):
        result = normalize_stream_metadata(
            {
                "reaction_intent": "acknowledge",
                "pose_id": "speaking_direct",
                "pose_sequence": json.dumps(
                    ["nod_agree", "speaking_direct", "neutral_resting"]
                ),
                "turn_id": "turn_2",
                "seq": "5",
                "effective": "next_boundary",
                "mouth_mode": "lip_sync",
                "audio_start": "immediate",
            }
        )
        self.assertEqual(result["pose_sequence"][0], "nod_agree")
        self.assertEqual(result["seq"], 5)

    def test_pose_plan_audio_progress_contract(self):
        result = normalize_pose_plan(
            {
                "version": 2,
                "clock": "audio_progress",
                "segments": [
                    {
                        "at_permille": 0,
                        "pose_id": "empathetic_head_tilt",
                    },
                    {
                        "at_permille": 480,
                        "pose_id": "speaking_direct",
                    },
                ],
                "on_complete": "neutral_resting",
                "switch_mode": "next_boundary",
            }
        )
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["segments"][1]["at_permille"], 480)

    def test_pose_plan_rejects_listening_pose_during_assistant_audio(self):
        with self.assertRaises(PoseProtocolError):
            normalize_pose_plan(
                {
                    "version": 2,
                    "clock": "audio_progress",
                    "segments": [
                        {
                            "at_permille": 0,
                            "pose_id": "active_listening",
                        }
                    ],
                }
            )

    def test_pose_plan_requires_safe_direct_speaking_tail(self):
        with self.assertRaisesRegex(
            PoseProtocolError,
            "must end with pose_id=speaking_direct",
        ):
            normalize_pose_plan(
                {
                    "version": 2,
                    "clock": "audio_progress",
                    "segments": [
                        {
                            "at_permille": 0,
                            "pose_id": "speaking_direct",
                        },
                        {
                            "at_permille": 500,
                            "pose_id": "light_smile",
                        },
                    ],
                }
            )

    def test_rejects_arbitrary_stream_sequence(self):
        with self.assertRaises(PoseProtocolError):
            normalize_stream_metadata(
                {
                    "reaction_intent": "warmth",
                    "pose_id": "speaking_direct",
                    "pose_sequence": json.dumps(
                        ["speaking_direct", "light_smile", "neutral_resting"]
                    ),
                    "turn_id": "turn_3",
                    "seq": "5",
                }
            )


if __name__ == "__main__":
    unittest.main()
