"""CPU proof for opt-in eye geometry through real motion runtime paths.

These checks prove metadata ownership and limited pixel support, not visual
acceptance of an avatar or performance of a live GPU/WebRTC session.
"""
import asyncio
import copy
import json
import tempfile
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import patch

import av
import numpy as np

from scripts.hls_gpu_scheduler import HLSGPUStreamScheduler
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, flow_blend, file_hash
from scripts.webrtc_motion_playback import MotionPlaybackMixin
from test_motion_transitions import fixture
import test_motion_entry as entry_tests

Decoder = entry_tests.Decoder
frame = entry_tests.frame


def eye_fixture():
    manifest = fixture()
    for source in manifest["sources"].values():
        source.update(width=64, height=64)
    frames = {}
    for pose_index, pose in enumerate((IDLE, TALK, SMILE)):
        frames[pose] = []
        for index in range(manifest["sources"][pose]["frame_count"]):
            # Vary every original frame's contour, so a spy can distinguish an
            # adjacent-frame error without inferring anything from pixel color.
            shift = index * .02 + pose_index * .1
            frames[pose].append({
                "left": [[x + shift, y] for x, y in
                         ((9, 15), (12, 13), (16, 13), (19, 15), (16, 17), (12, 17))],
                "right": [[x + shift, y] for x, y in
                          ((41, 15), (44, 13), (48, 13), (51, 15), (48, 17), (44, 17))],
            })
    manifest["eye_blend"] = {"method": "incoming_roi_v1",
                             "source_hashes": {p: s["sha256"] for p, s in manifest["sources"].items()},
                             "frames": frames}
    return manifest


class EyeProfileValidationTest(unittest.TestCase):
    def test_omitted_profile_preserves_baseline_pixels_and_routing(self):
        bank = MotionBank(fixture())
        self.assertIsNone(bank.eye_blend)
        rng = np.random.default_rng(902)
        old = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        new = np.roll(old, 2, axis=1)
        with patch("scripts.motion_eye_blend.incoming_eye_blend",
                   side_effect=AssertionError("Unconfigured eye path ran")):
            actual = bank.blend(old, new, .4, IDLE, None, TALK, None)
        np.testing.assert_array_equal(actual, flow_blend(old, new, .4))

    def test_profile_is_bound_to_all_original_frames_and_hashes(self):
        valid = eye_fixture()
        self.assertIsNotNone(MotionBank(valid).eye_blend)
        invalid = {
            "method": lambda m: m["eye_blend"].update(method="unknown"),
            "stale hash": lambda m: m["eye_blend"]["source_hashes"].update({TALK: "changed"}),
            "missing source": lambda m: m["eye_blend"]["frames"].pop(SMILE),
            "missing frame": lambda m: m["eye_blend"]["frames"][TALK].pop(),
            "extra frame": lambda m: m["eye_blend"]["frames"][IDLE].append(m["eye_blend"]["frames"][IDLE][0]),
            "missing dimensions": lambda m: m["sources"][IDLE].pop("width"),
            "nan geometry": lambda m: m["eye_blend"]["frames"][IDLE][0]["left"][0].__setitem__(0, float("nan")),
            "outside geometry": lambda m: m["eye_blend"]["frames"][TALK][3]["left"][0].__setitem__(1, 65),
            "missing eye": lambda m: m["eye_blend"]["frames"][SMILE][8].pop("right"),
        }
        for label, mutate in invalid.items():
            with self.subTest(label=label):
                manifest = copy.deepcopy(valid)
                mutate(manifest)
                with self.assertRaises(ValueError):
                    MotionBank(manifest)

    def test_profile_changes_routing_identity_and_rejects_wrong_frame_geometry(self):
        baseline = fixture()
        profiled = eye_fixture()
        self.assertNotEqual(MotionBank(baseline).routing_sha256, MotionBank(profiled).routing_sha256)
        bank = MotionBank(profiled)
        image = np.zeros((64, 64, 3), np.uint8)
        for old_index, new_index in ((None, 0), (0, None), (-1, 0), (0, 41), (1.5, 0)):
            with self.subTest(indices=(old_index, new_index)), self.assertRaises(ValueError):
                bank.blend(image, image, .5, IDLE, old_index, TALK, new_index)
        with self.assertRaises(ValueError):
            bank.blend(image[:32], image[:32], .5, IDLE, 0, TALK, 0)

    def test_opt_in_eyes_leave_existing_mouth_composition_exact(self):
        bank = MotionBank(eye_fixture())
        rng = np.random.default_rng(137)
        old = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        new = np.roll(old, 2, axis=0)
        # This lower region models current composed mouth pixels. It is outside
        # the localized eye support; output must be byte-identical to baseline.
        new[44:60, 20:44] = (17, 217, 79)
        for progress in (.1, .5, .9):
            with self.subTest(progress=progress):
                expected = flow_blend(old, new, progress)
                actual = bank.blend(old, new, progress, IDLE, 4, TALK, 9)
                np.testing.assert_array_equal(actual[36:], expected[36:])
        # The existing full-frame blend itself can alter mouth pixels; this
        # deliberately does not claim phoneme-core preservation over baseline.


class EyeSchedulerOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.bank = MotionBank(eye_fixture())
        self.scheduler = HLSGPUStreamScheduler.__new__(HLSGPUStreamScheduler)
        self.scheduler.webrtc_pose_crossfade_frames = 2
        self.job = types.SimpleNamespace(
            request_id="eye_runtime_test", session=types.SimpleNamespace(
                live_pose_router=types.SimpleNamespace(motion_bank=self.bank)),
            webrtc_last_pose_frame=None, webrtc_last_pose_id=None,
            webrtc_pose_crossfade_anchor=None, webrtc_pose_crossfade_index=0,
            webrtc_pose_crossfade_target_frames=0, webrtc_pose_crossfade_count=0,
            webrtc_pose_crossfade_frames_applied=0)

    def test_actual_scheduler_freezes_previous_source_geometry_across_batches(self):
        images = [np.full((64, 64, 3), value, np.uint8) for value in (10, 20, 30, 40, 50)]
        calls = []
        def spy(old, new, progress, old_points, new_points):
            calls.append((old.copy(), new.copy(), copy.deepcopy(old_points), copy.deepcopy(new_points)))
            return new.copy()
        with patch("scripts.motion_eye_blend.incoming_eye_blend", spy):
            first = self.scheduler._apply_webrtc_pose_crossfade(
                self.job, images[:3], [IDLE, IDLE, TALK], [0, 0, 3], [11, 12, 29])
            second = self.scheduler._apply_webrtc_pose_crossfade(
                self.job, images[3:], [TALK, TALK], [3, 3], [30, 32])
        self.assertEqual(len(first) + len(second), 5)
        self.assertEqual(len(calls), 3)
        for call, source in zip(calls, (29, 30, 32)):
            np.testing.assert_array_equal(call[0], images[1])
            self.assertEqual(call[2], self.bank.eye_blend["frames"][IDLE][12])
            self.assertEqual(call[3], self.bank.eye_blend["frames"][TALK][source])
        self.assertEqual(self.job.webrtc_last_source_frame, 32)
        self.assertEqual(self.job.webrtc_pose_crossfade_frames_applied, 3)

    def test_configured_scheduler_refuses_missing_source_indices(self):
        image = np.zeros((64, 64, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "every source index"):
            self.scheduler._apply_webrtc_pose_crossfade(self.job, [image], [IDLE])


class EyeEntryOwnershipTest(unittest.IsolatedAsyncioTestCase):
    # Reuse production-track setup but do not inherit/re-run every baseline test.
    tick = entry_tests.MovingIdleEntryTest.tick
    queue_turn = entry_tests.MovingIdleEntryTest.queue_turn

    def setUp(self):
        entry_tests.MovingIdleEntryTest.setUp(self)
        self.track.configure_motion_bank(MotionBank(eye_fixture()))

    async def asyncTearDown(self):
        await entry_tests.MovingIdleEntryTest.asyncTearDown(self)

    async def test_entry_contours_follow_current_decoded_idle_on_every_tick(self):
        await self.tick()
        _, generated = await self.queue_turn()
        await self.track._motion_entry_task
        calls = []
        def spy(old, new, progress, old_pose, old_index, new_pose, new_index):
            calls.append((old.copy(), old_pose, old_index, new_pose, new_index))
            return new.copy()
        with patch.object(self.track.motion_bank, "blend", spy):
            for _ in range(6):
                await self.tick()
                actual = calls[-1]
                timing = self.track._idle.get_timing()["source_frame_index"]
                self.assertEqual(actual[1:], (IDLE, timing, IDLE, 3))
                np.testing.assert_array_equal(actual[0], frame(timing * 4).to_ndarray(format="bgr24"))
                self.assertEqual(self.track._emitted_motion["outgoing_idle_frame"], timing)
                self.assertEqual(self.track._queue.qsize(), 2)
        self.assertEqual(len({call[2] for call in calls}), 6)
        self.assertIs(await self.tick(), generated[0])
        self.assertEqual(self.track._emitted_motion["generation_frame"], 0)

    async def test_cancelled_eye_worker_cannot_emit_obsolete_entry(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        await self.tick()
        entered, gate = threading.Event(), threading.Event()
        count = [0]
        original = self.track.motion_bank.blend
        def slow_first(*args):
            count[0] += 1
            if count[0] == 1:
                entered.set()
                gate.wait(timeout=3)
            return original(*args)
        try:
            with patch.object(self.track.motion_bank, "blend", slow_first):
                pending = asyncio.create_task(self.tick())
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.track.end_live()
                await self.track._motion_task
                gate.set()
                await pending
        finally:
            gate.set()
        self.assertIsNone(self.track._motion_entry)
        self.assertNotEqual(self.track._emitted_motion["mode"], "speech_entry_body_bridge")
        self.assertIsNone(self.clock.first_live_video_rtp_seconds)
        self.assertIsNone(self.clock.first_tts_transport_pts_seconds)


class EyeReturnOwnershipTest(unittest.IsolatedAsyncioTestCase):
    async def test_arbitrary_return_binds_emitted_anchor_and_wraps_incoming_source(self):
        manifest = eye_fixture()
        manifest["edges"][TALK][IDLE][7]["target_frame"] = 40
        bank = MotionBank(manifest)
        track = MotionPlaybackMixin()
        track._current_idle_video_path = "source-idle.mp4"
        track._current_idle_pose_id = IDLE
        track._closed = False
        track._output_fps = 20
        track._rtp_frame_index = 9
        track._completion_idle_switch = None
        track._stop_pending_idle_switches = lambda: None
        track.configure_motion_bank(bank)
        anchor = frame(184)
        track._last_live_frame = anchor
        track._note_motion_output({"pose_id": TALK, "source_frame": 7})
        track._popped_motion = {"pose_id": SMILE, "source_frame": 30}
        applied, calls = {}, []
        track._apply_idle_switch = lambda decoder, **kw: applied.update(decoder=decoder, **kw)
        def spy(old, new, progress, old_pose, old_index, new_pose, new_index):
            calls.append((old.copy(), new.copy(), old_pose, old_index, new_pose, new_index))
            return new.copy()
        try:
            with patch("scripts.webrtc_tracks.IdleVideoStreamTrack", Decoder), patch.object(bank, "blend", spy):
                self.assertTrue(track._begin_motion_return())
                await track._motion_task
            self.assertEqual([call[-1] for call in calls], [40, 0, 1, 2, 3, 5])
            for call in calls:
                self.assertEqual(call[2:5], (TALK, 7, IDLE))
                np.testing.assert_array_equal(call[0], anchor.to_ndarray(format="bgr24"))
                np.testing.assert_array_equal(call[1], frame(call[-1] * 4).to_ndarray(format="bgr24"))
            self.assertFalse(track._motion_settled.is_set())
            for _ in range(6):
                track._note_motion_idle_frame()
            self.assertTrue(track._motion_settled.is_set())
            self.assertEqual(track._emitted_motion["source_frame"], 5)
        finally:
            if "decoder" in applied:
                applied["decoder"].stop()


class EyeProfilePublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="motion-eye-attachment-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = eye_fixture()
        profile = self.manifest.pop("eye_blend")
        self.manifest.update(status="reviewed", review={"accepted": True, "reviewer": "prior parent"})
        for pose, source in self.manifest["sources"].items():
            path = self.root / (pose + ".mp4")
            path.write_bytes(("Synthetic integrity fixture: " + pose).encode())
            source.update(path=str(path), sha256=file_hash(path))
        self.parent = self.root / "parent-atlas.json"
        self.parent.write_text(json.dumps(self.manifest) + "\n")
        self.measurements = {
            "version": 1, "method": "incoming_roi_v1",
            "atlas_sha256": file_hash(self.parent),
            "source_hashes": {pose: source["sha256"] for pose, source in self.manifest["sources"].items()},
            "frames": profile["frames"],
        }
        self.measurement_path = self.root / "measurements.json"
        self.measurement_path.write_text(json.dumps(self.measurements) + "\n")
        self.output = self.root / "candidate" / "motion-atlas.json"

    def attach(self):
        from scripts.attach_motion_eye_profile import attach
        return attach(self.parent, self.measurement_path, self.output)

    def test_parent_atlas_hash_mismatch_cannot_publish(self):
        self.measurements["atlas_sha256"] = "0" * 64
        self.measurement_path.write_text(json.dumps(self.measurements))
        with self.assertRaisesRegex(ValueError, "different parent atlas"):
            self.attach()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("motion-registration.json").exists())

    def test_measurement_source_hash_mismatch_cannot_publish(self):
        self.measurements["source_hashes"][TALK] = "0" * 64
        self.measurement_path.write_text(json.dumps(self.measurements))
        with self.assertRaisesRegex(ValueError, "do not belong to this source bank"):
            self.attach()
        self.assertFalse(self.output.exists())

    def test_source_file_mutation_cannot_publish(self):
        Path(self.manifest["sources"][SMILE]["path"]).write_bytes(b"modified source bytes")
        with self.assertRaisesRegex(ValueError, "Source video changed"):
            self.attach()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("motion-registration.json").exists())

    def test_output_or_registration_collision_cannot_overwrite_existing_bytes(self):
        from scripts.attach_motion_eye_profile import attach
        for name in ("motion-atlas.json", "motion-registration.json"):
            with self.subTest(existing=name):
                self.output.parent.mkdir(exist_ok=True)
                collision = self.output.with_name(name)
                collision.write_bytes(b"existing artifact must survive")
                try:
                    with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                        self.attach()
                    self.assertEqual(collision.read_bytes(), b"existing artifact must survive")
                    other = self.output.with_name("motion-registration.json" if name == "motion-atlas.json" else "motion-atlas.json")
                    self.assertFalse(other.exists())
                finally:
                    collision.unlink()
        parent_bytes = self.parent.read_bytes()
        with self.assertRaises(ValueError):
            attach(self.parent, self.measurement_path, self.parent)
        self.assertEqual(self.parent.read_bytes(), parent_bytes)

    def test_new_candidate_preserves_sources_and_parent_but_clears_review(self):
        parent_bytes = self.parent.read_bytes()
        source_bytes = {pose: Path(source["path"]).read_bytes()
                        for pose, source in self.manifest["sources"].items()}
        result = self.attach()
        candidate = json.loads(self.output.read_text())
        registration = json.loads(self.output.with_name("motion-registration.json").read_text())
        self.assertEqual(candidate["status"], "candidate_requires_recorded_review")
        self.assertNotIn("review", candidate)
        self.assertEqual(candidate["sources"], self.manifest["sources"])
        self.assertEqual(candidate["eye_blend"]["source_hashes"], self.measurements["source_hashes"])
        self.assertEqual(self.parent.read_bytes(), parent_bytes)
        for pose, source in self.manifest["sources"].items():
            self.assertEqual(Path(source["path"]).read_bytes(), source_bytes[pose])
        self.assertEqual(registration["source_hashes"], self.measurements["source_hashes"])
        self.assertEqual(registration["atlas_sha256"], file_hash(self.output))
        self.assertEqual(result["atlas_sha256"], file_hash(self.output))
        self.assertEqual(result["routing_sha256"], MotionBank(candidate).routing_sha256)
        self.assertNotEqual(result["routing_sha256"], MotionBank(self.manifest).routing_sha256)
        self.assertEqual(candidate["eye_blend_provenance"], {
            "parent_atlas_sha256": file_hash(self.parent),
            "measurements_sha256": file_hash(self.measurement_path)})


if __name__ == "__main__":
    unittest.main()
