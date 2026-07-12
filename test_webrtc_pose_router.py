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
