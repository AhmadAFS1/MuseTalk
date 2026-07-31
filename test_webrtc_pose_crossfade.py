import types
import unittest

import numpy as np

from scripts.hls_gpu_scheduler import HLSGPUStreamScheduler


class WebRTCPoseCrossfadeTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = HLSGPUStreamScheduler.__new__(HLSGPUStreamScheduler)
        self.scheduler.webrtc_pose_crossfade_frames = 2
        self.job = types.SimpleNamespace(
            request_id="crossfade_test",
            webrtc_last_pose_frame=None,
            webrtc_last_pose_id=None,
            webrtc_pose_crossfade_anchor=None,
            webrtc_pose_crossfade_index=0,
            webrtc_pose_crossfade_target_frames=0,
            webrtc_pose_crossfade_count=0,
            webrtc_pose_crossfade_frames_applied=0,
        )

    def test_blends_across_batches_without_changing_frame_count(self):
        black = np.zeros((2, 2, 3), dtype=np.uint8)
        white = np.full((2, 2, 3), 100, dtype=np.uint8)

        first_batch = self.scheduler._apply_webrtc_pose_crossfade(
            self.job,
            [black.copy(), black.copy(), white.copy()],
            ["active_listening", "active_listening", "speaking_direct"],
        )
        second_batch = self.scheduler._apply_webrtc_pose_crossfade(
            self.job,
            [white.copy(), white.copy()],
            ["speaking_direct", "speaking_direct"],
        )

        self.assertEqual(len(first_batch) + len(second_batch), 5)
        self.assertTrue(np.array_equal(first_batch[0], black))
        self.assertTrue(np.array_equal(first_batch[1], black))
        self.assertTrue(np.all(first_batch[2] == 25))
        self.assertTrue(np.all(second_batch[0] == 75))
        self.assertTrue(np.array_equal(second_batch[1], white))
        self.assertEqual(self.job.webrtc_pose_crossfade_count, 1)
        self.assertEqual(self.job.webrtc_pose_crossfade_frames_applied, 2)

    def test_requested_time_switch_uses_longer_per_pose_crossfade(self):
        black = np.zeros((2, 2, 3), dtype=np.uint8)
        white = np.full((2, 2, 3), 100, dtype=np.uint8)

        first_batch = self.scheduler._apply_webrtc_pose_crossfade(
            self.job,
            [black.copy(), white.copy()],
            ["active_listening", "speaking_direct"],
            [0, 4],
        )
        second_batch = self.scheduler._apply_webrtc_pose_crossfade(
            self.job,
            [white.copy(), white.copy(), white.copy()],
            ["speaking_direct", "speaking_direct", "speaking_direct"],
            [4, 4, 4],
        )

        transition_values = [
            int(first_batch[1][0, 0, 0]),
            *[
                int(frame[0, 0, 0])
                for frame in second_batch
            ],
        ]
        self.assertEqual(len(first_batch) + len(second_batch), 5)
        self.assertEqual(transition_values, sorted(transition_values))
        self.assertLess(transition_values[0], 25)
        self.assertLess(transition_values[-1], 100)
        self.assertEqual(self.job.webrtc_pose_crossfade_count, 1)
        self.assertEqual(self.job.webrtc_pose_crossfade_frames_applied, 4)
        self.assertEqual(self.job.webrtc_pose_crossfade_target_frames, 0)


if __name__ == "__main__":
    unittest.main()
