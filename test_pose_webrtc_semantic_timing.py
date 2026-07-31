import unittest

from scripts.test_pose_webrtc import (
    SmokeTestError,
    validate_compiled_pose_plan,
)


REQUESTED_PLAN = {
    "version": 2,
    "clock": "audio_progress",
    "segments": [
        {"at_permille": 0, "pose_id": "speaking_direct"},
        {"at_permille": 200, "pose_id": "light_smile"},
        {"at_permille": 600, "pose_id": "speaking_direct"},
    ],
    "on_complete": "neutral_resting",
    "switch_mode": "next_boundary",
}


def compiled_plan(starts):
    total_frames = 395
    requested_starts = [0, 79, 237]
    pose_ids = ["speaking_direct", "light_smile", "speaking_direct"]
    ends = [starts[1], starts[2], total_frames]
    segments = []
    for index, (pose_id, requested, start, end) in enumerate(
        zip(pose_ids, requested_starts, starts, ends)
    ):
        drift = start - requested
        segments.append(
            {
                "pose_id": pose_id,
                "requested_at_permille": [0, 200, 600][index],
                "requested_start_generation_frame": requested,
                "effective_start_generation_frame": start,
                "effective_end_generation_frame": end,
                "boundary_snap_delay_frames": drift,
                "semantic_drift_frames": drift,
                "switch_strategy": (
                    "initial_phase_aligned"
                    if index == 0
                    else "requested_time_crossfade"
                ),
                "crossfade_frames": 0 if index == 0 else 4,
            }
        )
    return {
        "status": "compiled",
        "switch_policy": "bounded_semantic",
        "requested_segments": [
            dict(segment)
            for segment in REQUESTED_PLAN["segments"]
        ],
        "segments": segments,
        "skipped_segments": [],
        "total_generation_frames": total_frames,
        "generation_fps": 15,
    }


def rendered_trace(starts):
    ends = [starts[1] - 1, starts[2] - 1, 394]
    return [
        {
            "pose_id": pose_id,
            "start_frame_index": start,
            "end_frame_index": end,
            "frame_count": end - start + 1,
        }
        for pose_id, start, end in zip(
            ["speaking_direct", "light_smile", "speaking_direct"],
            starts,
            ends,
        )
    ]


class PoseWebRTCSemanticTimingTest(unittest.TestCase):
    def test_accepts_rendered_trace_within_audio_timing_limit(self):
        starts = [0, 79, 237]

        result = validate_compiled_pose_plan(
            REQUESTED_PLAN,
            compiled_plan(starts),
            rendered_trace(starts),
            max_semantic_drift_seconds=0.75,
        )

        self.assertTrue(result["validated"])
        self.assertEqual(result["max_abs_semantic_drift_frames"], 0)
        self.assertEqual(result["max_abs_semantic_drift_seconds"], 0.0)

    def test_rejects_old_next_boundary_multi_second_delay(self):
        starts = [0, 135, 315]

        with self.assertRaisesRegex(
            SmokeTestError,
            "semantic timing limit",
        ):
            validate_compiled_pose_plan(
                REQUESTED_PLAN,
                compiled_plan(starts),
                rendered_trace(starts),
                max_semantic_drift_seconds=0.75,
            )


if __name__ == "__main__":
    unittest.main()
