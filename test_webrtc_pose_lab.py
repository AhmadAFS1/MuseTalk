import json
import unittest
from copy import deepcopy

from scripts.test_pose_webrtc import (
    DEFAULT_MANIFEST,
    MVP_FOUR_ORDERED_TRANSITION_CIRCUIT,
    MVP_FOUR_POSE_IDS,
    POSE_IDS,
    load_pose_set,
    pose_asset_entries,
    worker_pose_manifest,
)
from templates.webrtc_pose_lab import (
    DEFAULT_POSE_SET,
    get_webrtc_pose_lab_html,
)


class PoseWebRTCLabTests(unittest.TestCase):
    def test_mvp_four_circuit_covers_every_ordered_transition_once(self):
        transitions = list(
            zip(
                MVP_FOUR_ORDERED_TRANSITION_CIRCUIT,
                MVP_FOUR_ORDERED_TRANSITION_CIRCUIT[1:],
            )
        )
        expected = {
            (source, target)
            for source in MVP_FOUR_POSE_IDS
            for target in MVP_FOUR_POSE_IDS
            if source != target
        }
        self.assertEqual(len(transitions), 12)
        self.assertEqual(set(transitions), expected)
        self.assertEqual(len(set(transitions)), len(transitions))
        self.assertEqual(
            MVP_FOUR_ORDERED_TRANSITION_CIRCUIT[0],
            "neutral_resting",
        )
        self.assertEqual(
            MVP_FOUR_ORDERED_TRANSITION_CIRCUIT[-1],
            "neutral_resting",
        )

    def test_manifest_has_exact_canonical_six(self):
        pose_set = load_pose_set(DEFAULT_MANIFEST)
        self.assertEqual(tuple(pose_set["poses"]), POSE_IDS)
        self.assertEqual(pose_set["default_pose_id"], "neutral_resting")
        self.assertTrue(pose_set["switch_safe"])
        self.assertFalse(pose_set["test_only"])
        self.assertEqual(
            pose_set["pose_set_id"],
            "sample_ai_human_ltx23_facetime_closeup_production_v1",
        )

    def test_worker_manifest_excludes_local_and_activation_fields(self):
        manifest = worker_pose_manifest(load_pose_set(DEFAULT_MANIFEST))
        self.assertEqual(tuple(manifest["poses"]), POSE_IDS)
        self.assertNotIn("activation_status", manifest)
        self.assertNotIn("switch_safe", manifest)
        for entry in manifest["poses"].values():
            self.assertNotIn("asset_file", entry)

    def test_worker_manifest_preserves_direct_variants_and_prepares_unique_assets(self):
        pose_set = deepcopy(load_pose_set(DEFAULT_MANIFEST))
        direct = pose_set["poses"]["speaking_direct"]
        default_avatar_id = direct["avatar_id"]
        default_asset_file = direct["asset_file"]
        direct["variants"] = [
            {
                "variant_id": "calm",
                "avatar_id": default_avatar_id,
                "asset_file": default_asset_file,
            },
            {
                "variant_id": "reference_paced",
                "avatar_id": "direct_reference_paced",
                "asset_file": "speaking_direct_reference_paced.mp4",
            },
        ]
        direct["variant_policy"] = "deterministic_boundary_rotation"

        worker_manifest = worker_pose_manifest(pose_set)
        worker_direct = worker_manifest["poses"]["speaking_direct"]
        self.assertEqual(worker_direct["variants"][1]["variant_id"], "reference_paced")
        self.assertNotIn("asset_file", worker_direct["variants"][1])
        physical_assets = pose_asset_entries(pose_set)
        self.assertEqual(
            sum(entry["pose_id"] == "speaking_direct" for entry in physical_assets),
            2,
        )

    def test_page_exercises_full_direct_worker_contract(self):
        page = get_webrtc_pose_lab_html()
        for endpoint in (
            "/webrtc/sessions/create?",
            "/webrtc/sessions/${sessionId}/offer",
            "/webrtc/sessions/${sessionId}/ice",
            "/webrtc/sessions/${sessionId}/events",
            "/webrtc/sessions/${sessionId}/pose",
            "/webrtc/sessions/${sessionId}/stream",
            "/webrtc/sessions/${sessionId}/status",
        ):
            self.assertIn(endpoint, page)
        for pose_id in POSE_IDS:
            self.assertIn(pose_id, page)
        self.assertIn("audio_start", page)
        self.assertIn("immediate", page)
        self.assertIn("mouth_mode", page)
        self.assertIn("lip_sync", page)
        self.assertIn("replace_pending: replacePending", page)
        self.assertIn(
            "await queuePose(cyclePoseIds[index], index === 0)",
            page,
        )
        self.assertIn("rendered_pose_id", page)
        self.assertIn("setIdlePoseControlsDisabled(Boolean(body.active_stream))", page)

    def test_page_embeds_valid_pose_set_without_external_dependencies(self):
        page = get_webrtc_pose_lab_html()
        compact = json.dumps(DEFAULT_POSE_SET, separators=(",", ":"))
        self.assertIn(compact, page)
        self.assertNotIn("https://", page)
        self.assertNotIn("<script src=", page)

    def test_page_rejects_partial_pose_set(self):
        incomplete = {
            **DEFAULT_POSE_SET,
            "poses": {
                key: value
                for key, value in DEFAULT_POSE_SET["poses"].items()
                if key != "light_smile"
            },
        }
        with self.assertRaisesRegex(ValueError, "six protocol poses"):
            get_webrtc_pose_lab_html(incomplete)


if __name__ == "__main__":
    unittest.main()
