import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.webrtc_pose_router import LivePoseVideoRouter

try:
    from musetalk.utils.blending import (
        get_image_blending_with_plan,
        prepare_image_blending_plan,
    )
except ModuleNotFoundError:
    get_image_blending_with_plan = None
    prepare_image_blending_plan = None


class FakeDecoder:
    instances = {}

    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.fps = 30.0
        self.frame_count = 147
        self.closed = False
        self.requests = []
        self.instances[self.video_path] = self

    def read_frames(self, frame_indices):
        self.requests.append(list(frame_indices))
        return [(self.video_path, frame_index) for frame_index in frame_indices]

    def close(self):
        self.closed = True

    def get_stats(self):
        return {"requests": list(self.requests), "closed": self.closed}


class FailingDecoder(FakeDecoder):
    def read_frames(self, frame_indices):
        raise RuntimeError("decode failed")


class ProductionDurationDecoder(FakeDecoder):
    def __init__(self, video_path):
        super().__init__(video_path)
        self.frame_count = (
            180 if "light_smile" in self.video_path else 360
        )


class LivePoseVideoRouterTest(unittest.TestCase):
    def setUp(self):
        FakeDecoder.instances = {}
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.paths = {}
        for pose_id in ("default", "light_smile", "speaking_direct"):
            path = root / f"{pose_id}.mp4"
            path.write_bytes(b"test")
            self.paths[pose_id] = str(path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepared_pose_uses_no_extra_decoder(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)

        snapshot = router.snapshot(0, 10)
        frames = router.read_background_frames(snapshot, 0, 3)

        self.assertTrue(snapshot.uses_prepared_background)
        self.assertEqual(frames, [None, None, None])
        self.assertEqual(FakeDecoder.instances, {})

    def test_prepared_alternate_pose_uses_its_own_materials_without_decode(self):
        router = LivePoseVideoRouter(
            self.paths,
            prepared_pose_ids={"default", "light_smile"},
            decoder_factory=FakeDecoder,
        )
        router.switch_pose("light_smile")

        snapshot = router.snapshot(0, 10)
        frames = router.read_background_frames(snapshot, 0, 2)

        self.assertTrue(snapshot.uses_prepared_background)
        self.assertEqual(frames, [None, None])
        self.assertEqual(FakeDecoder.instances, {})

    def test_pose_switch_maps_generation_frames_to_source_frames(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        router.switch_pose("light_smile")

        snapshot = router.snapshot(20, 10)
        frames = router.read_background_frames(snapshot, 20, 3)

        self.assertEqual(snapshot.pose_id, "light_smile")
        self.assertEqual([frame[1] for frame in frames], [0, 3, 6])

    def test_direct_variant_selection_is_deterministic_and_semantic_id_stays_stable(self):
        root = Path(self.temp_dir.name)
        variant_keys = [
            "speaking_direct__variant__calm",
            "speaking_direct__variant__reference_paced",
        ]
        for render_key in variant_keys:
            path = root / f"{render_key}.mp4"
            path.write_bytes(b"test")
            self.paths[render_key] = str(path)
        router = LivePoseVideoRouter(
            self.paths,
            pose_variant_render_keys={"speaking_direct": variant_keys},
            decoder_factory=FakeDecoder,
        )

        first = router.set_variant_context("pose-set:session:turn-1")
        repeat = router.set_variant_context("pose-set:session:turn-1")
        self.assertEqual(first, repeat)

        second_turn = router.set_variant_context("pose-set:session:turn-2")
        self.assertNotEqual(
            first["speaking_direct"],
            second_turn["speaking_direct"],
        )
        router.set_variant_context("pose-set:session:turn-1")

        router.queue_pose_sequence(["speaking_direct"], generation_fps=10)
        snapshot = router.snapshot(0, 10)
        self.assertEqual(snapshot.pose_id, "speaking_direct")
        self.assertIn(snapshot.effective_render_key, variant_keys)
        self.assertIn("speaking_direct__variant__", snapshot.effective_render_key)

    def test_in_flight_snapshot_keeps_its_pose_after_switch(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        router.switch_pose("light_smile")
        smile_snapshot = router.snapshot(10, 10)

        router.switch_pose("speaking_direct")
        speaking_snapshot = router.snapshot(30, 10)

        smile_frames = router.read_background_frames(smile_snapshot, 10, 1)
        speaking_frames = router.read_background_frames(speaking_snapshot, 30, 1)

        self.assertIn("light_smile.mp4", smile_frames[0][0])
        self.assertIn("speaking_direct.mp4", speaking_frames[0][0])
        self.assertEqual(smile_frames[0][1], 0)
        self.assertEqual(speaking_frames[0][1], 0)

    def test_pose_queue_routes_complete_clips_in_strict_frame_order(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        segments = router.queue_pose_sequence(
            ["default", "light_smile", "speaking_direct", "light_smile"],
            generation_fps=10,
        )

        self.assertEqual([segment["duration_frames"] for segment in segments], [49, 49, 49, 49])
        snapshots = {
            frame: router.snapshot(frame, 10)
            for frame in (0, 48, 49, 97, 98, 146, 147, 195, 196)
        }

        self.assertEqual(snapshots[0].pose_id, "default")
        self.assertEqual(snapshots[48].pose_id, "default")
        self.assertEqual(snapshots[49].pose_id, "light_smile")
        self.assertEqual(snapshots[97].pose_id, "light_smile")
        self.assertEqual(snapshots[98].pose_id, "speaking_direct")
        self.assertEqual(snapshots[146].pose_id, "speaking_direct")
        self.assertEqual(snapshots[147].pose_id, "light_smile")
        self.assertEqual(snapshots[195].pose_id, "light_smile")
        self.assertEqual(snapshots[196].pose_id, "light_smile")
        self.assertEqual(snapshots[49].origin_generation_frame, 49)
        self.assertEqual(snapshots[98].origin_generation_frame, 98)
        self.assertEqual(snapshots[147].origin_generation_frame, 147)

    def test_pose_plan_forces_requested_time_when_boundary_is_too_far(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        staged = router.queue_pose_plan(
            {
                "version": 2,
                "clock": "audio_progress",
                "segments": [
                    {"at_permille": 0, "pose_id": "speaking_direct"},
                    {"at_permille": 200, "pose_id": "light_smile"},
                    {"at_permille": 600, "pose_id": "speaking_direct"},
                ],
                "on_complete": "neutral_resting",
                "switch_mode": "next_boundary",
            },
            total_generation_frames=200,
            generation_fps=10,
        )
        self.assertEqual(staged["status"], "pending_phase_alignment")

        compiled = router.align_first_queued_pose(0, 10)

        self.assertEqual(
            [
                (
                    segment["pose_id"],
                    segment["effective_start_generation_frame"],
                    segment["effective_end_generation_frame"],
                )
                for segment in compiled
            ],
            [
                ("speaking_direct", 0, 40),
                ("light_smile", 40, 120),
                ("speaking_direct", 120, 200),
            ],
        )
        self.assertEqual(router.snapshot(39, 10).pose_id, "speaking_direct")
        self.assertEqual(router.snapshot(40, 10).pose_id, "light_smile")
        self.assertEqual(router.snapshot(119, 10).pose_id, "light_smile")
        self.assertEqual(router.snapshot(120, 10).pose_id, "speaking_direct")
        self.assertEqual(
            [segment["semantic_drift_frames"] for segment in compiled],
            [0, 0, 0],
        )
        self.assertEqual(
            [segment["switch_strategy"] for segment in compiled],
            [
                "initial_phase_aligned",
                "requested_time_crossfade",
                "requested_time_crossfade",
            ],
        )
        self.assertEqual(
            [segment["crossfade_frames"] for segment in compiled],
            [0, 4, 4],
        )
        self.assertTrue(router.get_compiled_pose_plan()["semantic_timing_valid"])

    def test_pose_plan_does_not_require_a_complete_expression_loop(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        router.queue_pose_plan(
            {
                "version": 2,
                "clock": "audio_progress",
                "segments": [
                    {"at_permille": 0, "pose_id": "speaking_direct"},
                    {"at_permille": 200, "pose_id": "light_smile"},
                    {"at_permille": 600, "pose_id": "speaking_direct"},
                ],
                "on_complete": "neutral_resting",
                "switch_mode": "next_boundary",
            },
            total_generation_frames=100,
            generation_fps=10,
        )

        router.align_first_queued_pose(0, 10)
        compiled = router.get_compiled_pose_plan()

        self.assertEqual(
            [segment["pose_id"] for segment in compiled["segments"]],
            ["speaking_direct", "light_smile", "speaking_direct"],
        )
        self.assertEqual(
            [
                segment["effective_start_generation_frame"]
                for segment in compiled["segments"]
            ],
            [0, 20, 60],
        )
        self.assertEqual(compiled["skipped_segments"], [])

    def test_pose_plan_uses_nearest_phase_boundary_within_drift_limit(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        router.queue_pose_plan(
            {
                "version": 2,
                "clock": "audio_progress",
                "segments": [
                    {"at_permille": 0, "pose_id": "light_smile"},
                    {"at_permille": 350, "pose_id": "speaking_direct"},
                ],
                "on_complete": "neutral_resting",
                "switch_mode": "next_boundary",
            },
            total_generation_frames=200,
            generation_fps=10,
        )

        compiled = router.align_first_queued_pose(87, 10)

        self.assertEqual(
            compiled[0]["source_frame_offset"],
            87,
        )
        self.assertEqual(
            compiled[1]["effective_start_generation_frame"],
            69,
        )
        self.assertEqual(
            compiled[1]["requested_start_generation_frame"],
            70,
        )
        self.assertEqual(compiled[1]["semantic_drift_frames"], -1)
        self.assertEqual(
            compiled[1]["switch_strategy"],
            "nearest_safe_boundary",
        )
        self.assertEqual(compiled[1]["crossfade_frames"], 0)

    def test_production_duration_example_has_no_multi_second_pose_lag(self):
        router = LivePoseVideoRouter(
            self.paths,
            prepared_pose_id="speaking_direct",
            initial_pose_id="speaking_direct",
            decoder_factory=ProductionDurationDecoder,
        )
        router.queue_pose_plan(
            {
                "version": 2,
                "clock": "audio_progress",
                "segments": [
                    {"at_permille": 0, "pose_id": "speaking_direct"},
                    {"at_permille": 200, "pose_id": "light_smile"},
                    {"at_permille": 600, "pose_id": "speaking_direct"},
                ],
                "on_complete": "neutral_resting",
                "switch_mode": "next_boundary",
            },
            total_generation_frames=395,
            generation_fps=15,
        )

        compiled = router.align_first_queued_pose(90, 15)

        self.assertEqual(
            [
                segment["effective_start_generation_frame"]
                for segment in compiled
            ],
            [0, 79, 237],
        )
        self.assertEqual(
            [
                segment["requested_start_generation_frame"]
                for segment in compiled
            ],
            [0, 79, 237],
        )
        self.assertEqual(
            [segment["semantic_drift_frames"] for segment in compiled],
            [0, 0, 0],
        )
        self.assertEqual(router.snapshot(78, 15).pose_id, "speaking_direct")
        self.assertEqual(router.snapshot(79, 15).pose_id, "light_smile")
        self.assertEqual(router.snapshot(236, 15).pose_id, "light_smile")
        self.assertEqual(router.snapshot(237, 15).pose_id, "speaking_direct")
        self.assertEqual(router.snapshot(79, 15).crossfade_frames, 4)
        self.assertEqual(router.snapshot(237, 15).crossfade_frames, 4)
        telemetry = router.get_compiled_pose_plan()
        self.assertEqual(telemetry["max_abs_semantic_drift_frames"], 0)
        self.assertTrue(telemetry["semantic_timing_valid"])

    def test_decode_failure_falls_back_to_prepared_background(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FailingDecoder)
        router.switch_pose("speaking_direct")
        snapshot = router.snapshot(0, 10)

        self.assertEqual(router.read_background_frames(snapshot, 0, 2), [None, None])
        self.assertEqual(router.get_stats()["read_failures"], 1)

    def test_close_releases_open_decoders(self):
        router = LivePoseVideoRouter(self.paths, decoder_factory=FakeDecoder)
        router.switch_pose("speaking_direct")
        snapshot = router.snapshot(0, 10)
        router.read_background_frames(snapshot, 0, 1)

        decoder = FakeDecoder.instances[self.paths["speaking_direct"]]
        router.close()

        self.assertTrue(decoder.closed)


@unittest.skipIf(
    get_image_blending_with_plan is None,
    "full MuseTalk image dependencies are not installed",
)
class AlternateBackgroundCompositionTest(unittest.TestCase):
    def test_compose_frame_uses_alternate_background(self):
        bbox = [2, 2, 6, 6]
        plan = prepare_image_blending_plan(
            (8, 8, 3),
            bbox,
            np.full((8, 8), 255, dtype=np.uint8),
            [0, 0, 8, 8],
        )
        alternate = np.zeros((8, 8, 3), dtype=np.uint8)
        alternate[:, :] = [10, 20, 30]
        generated_face = np.full((4, 4, 3), 200, dtype=np.uint8)

        composed = get_image_blending_with_plan(alternate.copy(), generated_face, plan)

        np.testing.assert_array_equal(composed[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(composed[3, 3], [200, 200, 200])
        np.testing.assert_array_equal(alternate[0, 0], [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
