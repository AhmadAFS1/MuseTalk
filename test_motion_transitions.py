import asyncio
import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import cv2
import numpy as np

from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, flow_blend
from scripts.webrtc_motion_playback import MotionPlaybackMixin


def fixture():
    count = 41
    return {"version": 1, "fps": 24, "bridge_seconds": .3, "short_reply_seconds": 3,
            "sources": {p: {"frame_count": count, "path": p, "sha256": "test"} for p in (IDLE, TALK, SMILE)},
            "edges": {p: {q: [{"target_frame": (i+1) % count, "score": 0,
                                "admissible": True} for i in range(count)]
                           for q in (IDLE, TALK, SMILE)} for p in (IDLE, TALK, SMILE)}}


class MotionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture()
        self.bank = MotionBank(self.manifest)

    def plan(self, seconds, fps=20, cues=None):
        return self.bank.plan(int(seconds*fps), fps, IDLE, 39,
                              cues or [{"at_permille": 0, "pose_id": TALK}])

    def test_short_replies_keep_idle_even_when_smile_requested(self):
        for duration in (.1, 1, 2.95):
            result = self.plan(duration, cues=[{"at_permille": 0, "pose_id": SMILE}])
            self.assertTrue(result["short_reply"])
            self.assertEqual({f["pose_id"] for f in result["frames"]}, {IDLE})
            self.assertEqual(result["switches"], [])

    def test_switch_waits_for_the_last_composed_anchor_to_be_admissible(self):
        # At 24Hz source / 20Hz output, generation frame 6 would enter a pose.
        # The compositor still holds idle source 4 from frame 5; source 5 has
        # not been displayed. A safe edge for that future source cannot justify
        # mixing the actual, incompatible anchor into the incoming face.
        self.manifest["edges"][IDLE][TALK][4]["admissible"] = False
        result = self.plan(10)
        first = result["switches"][0]
        self.assertEqual(first["frame"], 7)
        self.assertEqual(result["frames"][6]["pose_id"], IDLE)
        self.assertEqual(result["frames"][6]["source_frame"], 5)
        self.assertEqual(first["target_frame"], 6)

    def test_three_seconds_enters_talking_then_returns_before_end(self):
        result = self.plan(3)
        self.assertFalse(result["short_reply"])
        self.assertEqual([s["to"] for s in result["switches"]], [TALK, IDLE])
        self.assertEqual(result["frames"][-1]["pose_id"], IDLE)
        self.assertLessEqual(result["switches"][-1]["frame"], 50)

    def test_smile_uses_its_own_source_then_talking_then_idle(self):
        result = self.plan(10, cues=[{"at_permille": 0, "pose_id": TALK},
                                    {"at_permille": 350, "pose_id": SMILE},
                                    {"at_permille": 650, "pose_id": TALK}])
        self.assertEqual([s["to"] for s in result["switches"]], [TALK, SMILE, TALK, IDLE])

    def test_late_semantic_cues_leave_complete_terminal_return(self):
        for fps in (15, 20, 24, 30):
            for permille in range(850, 1000, 5):
                with self.subTest(fps=fps, permille=permille):
                    result = self.plan(10, fps=fps, cues=[
                        {"at_permille": 0, "pose_id": TALK},
                        {"at_permille": permille, "pose_id": SMILE}])
                    import math
                    terminal = int(10*fps) - math.ceil(.3*fps) - math.ceil(.2*fps)
                    self.assertLessEqual(result["switches"][-1]["frame"], terminal)
                    self.assertEqual(result["switches"][-1]["to"], IDLE)
                    self.assertTrue(all(f["pose_id"] == IDLE for f in result["frames"][terminal:]))

    def test_any_uncovered_interruption_phase_disables_that_source(self):
        self.manifest["edges"][TALK][IDLE][20]["admissible"] = False
        result = self.plan(10)
        self.assertEqual({f["pose_id"] for f in result["frames"]}, {IDLE})

    def test_forward_phase_resampling_never_uses_prepared_reverse_half(self):
        for fps in (15, 20, 24, 30):
            frames = self.plan(2, fps)["frames"]
            expected = [int(39+n*24/fps) % 41 for n in range(len(frames))]
            # Integer phase accumulation must not develop a duplicate due to
            # floating point drift at an exact rational boundary.
            actual = [f["source_frame"] for f in frames]
            self.assertEqual(actual, expected)

    def test_invalid_atlas_and_inputs_fail(self):
        for change in (lambda m: m.update(fps=0),
                       lambda m: m["edges"][TALK][IDLE].pop(),
                       lambda m: m["edges"][IDLE][TALK][0].update(target_frame=999),
                       lambda m: m["edges"][IDLE][TALK][0].update(score=float("nan"))):
            data = copy.deepcopy(self.manifest)
            change(data)
            with self.assertRaises(ValueError): MotionBank(data)
        with self.assertRaises(ValueError): self.bank.plan(0, 20, IDLE, 0, [])

    def test_file_mismatch_cannot_activate_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.mp4"
            path.write_bytes(b"different source")
            self.assertFalse(self.bank.compatible({p: str(path) for p in (IDLE, TALK, SMILE)}))


class FlowTest(unittest.TestCase):
    def test_endpoints_exact_and_shifted_feature_keeps_single_center(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 100, size=(128,128,3), dtype=np.uint8)
        cv2.circle(a, (54,64), 15, (230,230,230), -1)
        b = cv2.warpAffine(a, np.float32([[1,0,12],[0,1,0]]), (128,128), borderMode=cv2.BORDER_REFLECT)
        self.assertTrue(np.array_equal(flow_blend(a,b,0), a))
        self.assertTrue(np.array_equal(flow_blend(a,b,1), b))
        mid = flow_blend(a,b,.5)
        center = np.where(mid[:,:,0] > 190)[1].mean()
        self.assertAlmostEqual(center, 60, delta=2)
        with self.assertRaises(ValueError): flow_blend(a,b[:64],.5)


class RecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_return_uses_last_emitted_phase_and_settles_only_after_playback(self):
        class Decoder:
            def __init__(self, path, fps, decode_threads=0): self.index = -1; self.closed = False
            def read_frame(self):
                self.index += 1
                return av.VideoFrame.from_ndarray(np.full((64,64,3), self.index, np.uint8), format="bgr24")
            def stop(self): self.closed = True
        track = MotionPlaybackMixin()
        track._current_idle_video_path = "prepared-idle.mp4"
        track._current_idle_pose_id = IDLE
        track._closed = False
        track._output_fps = 20
        track._rtp_frame_index = 9
        track._completion_idle_switch = None
        track._stop_pending_idle_switches = lambda: None
        track.configure_motion_bank(MotionBank(fixture()))
        frame = av.VideoFrame.from_ndarray(np.zeros((64,64,3),np.uint8), format="bgr24")
        track._last_live_frame = frame
        track._note_motion_output({"pose_id": TALK, "source_frame": 7})
        track._popped_motion = {"pose_id": TALK, "source_frame": 35}  # inference ahead
        applied = {}
        track._apply_idle_switch = lambda decoder, **kw: applied.update(decoder=decoder, **kw)
        with patch("scripts.webrtc_tracks.IdleVideoStreamTrack", Decoder):
            self.assertTrue(track._begin_motion_return())
            await track._motion_task
        self.assertFalse(track._motion_settled.is_set())
        self.assertEqual(track._motion_returns[-1]["from_frame"], 7)
        self.assertEqual(track._motion_returns[-1]["target_frame"], 8)
        self.assertEqual(applied["idle_video_path"], "prepared-idle.mp4")
        self.assertEqual(len(applied["transition_frames"]), 6)
        for _ in range(6): track._note_motion_idle_frame()
        self.assertTrue(track._motion_settled.is_set())
        self.assertEqual(track._emitted_motion["source_frame"], 14)

    async def test_cancel_before_first_display_does_not_reset_idle(self):
        track = MotionPlaybackMixin()
        track._current_idle_video_path = "idle"
        track._completion_idle_switch = None
        track._stop_pending_idle_switches = lambda: None
        track._last_live_frame = None
        track.configure_motion_bank(MotionBank(fixture()))
        self.assertTrue(track._begin_motion_return())
        self.assertIsNone(track._motion_task)
        self.assertTrue(track._motion_settled.is_set())


if __name__ == "__main__": unittest.main()
