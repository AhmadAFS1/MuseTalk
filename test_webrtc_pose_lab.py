import json
import unittest

from scripts.test_pose_webrtc import (
    DEFAULT_MANIFEST,
    MVP_FOUR_ORDERED_TRANSITION_CIRCUIT,
    MVP_FOUR_POSE_IDS,
    POSE_IDS,
    load_pose_set,
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
        self.assertFalse(pose_set["switch_safe"])

    def test_worker_manifest_excludes_local_and_activation_fields(self):
        manifest = worker_pose_manifest(load_pose_set(DEFAULT_MANIFEST))
        self.assertEqual(tuple(manifest["poses"]), POSE_IDS)
        self.assertNotIn("activation_status", manifest)
        self.assertNotIn("switch_safe", manifest)
        for entry in manifest["poses"].values():
            self.assertNotIn("asset_file", entry)

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
