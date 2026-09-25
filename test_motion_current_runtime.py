"""CPU contracts for source-bound, current-phoneme body transitions.

Synthetic colors distinguish generated faces from raw bodies. These tests cover
runtime ownership and ordering; they do not establish visual acceptance.
"""
import copy
import json
import tempfile
from concurrent.futures import Future
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.hls_gpu_scheduler import HLSGPUStreamScheduler, HLSStreamJob
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, file_hash
from test_motion_eye_runtime import eye_fixture
from test_motion_transitions import fixture


def current_fixture():
    manifest = eye_fixture()
    manifest["current_phoneme"] = {
        "method": "current_similarity_v1",
        "source_hashes": {pose: source["sha256"]
                          for pose, source in manifest["sources"].items()},
    }
    return manifest


def pixels(value):
    return np.full((64, 64, 3), value, np.uint8)


def layer(value):
    raw = pixels(value)
    raw.setflags(write=False)
    alpha = np.full((24, 20), value, np.uint8)
    alpha.setflags(write=False)
    return {"raw": raw, "alpha": {"bounds": (20, 32, 40, 56), "values": alpha}}


class DeferredExecutor:
    """Resolve submitted CPU compositions in a chosen, deterministic order."""
    def __init__(self):
        self.work = []

    def submit(self, function):
        future = Future()
        self.work.append((function, future))
        return future

    def finish(self, index):
        function, future = self.work[index]
        try:
            future.set_result(function())
        except Exception as exc:
            future.set_exception(exc)


class FakeAvatar:
    def __init__(self):
        self.calls = []

    def compose_frame(self, generated, source_index, **kwargs):
        self.calls.append((source_index, dict(kwargs)))
        composed = np.asarray(generated).copy()
        if kwargs.get("return_layers"):
            return {"composed": composed, **layer(source_index)}
        return composed


class FakeRouter:
    def __init__(self, bank, sequence):
        self.motion_bank = bank
        self.sequence = sequence

    def snapshots_for_range(self, start, count, fps):
        return [types.SimpleNamespace(
            pose_id=pose, effective_render_key=pose,
            origin_generation_frame=0, is_queued=True,
            crossfade_frames=3, source_index=index,
        ) for pose, index in self.sequence[start:start + count]]

    def read_background_frames(self, snapshot, start, count):
        return [None] * count

    def source_frame_index(self, snapshot, generation_index):
        return snapshot.source_index


def make_job(bank, sequence=()):
    avatar = FakeAvatar()
    session = types.SimpleNamespace(live_pose_router=FakeRouter(bank, sequence))
    return HLSStreamJob(
        request_id="current_phoneme_cpu", session_id="current_phoneme_cpu",
        session=session, avatar=avatar, pose_avatars={}, audio_path="unused.wav",
        chunk_output_dir=Path("/tmp/unused-current-phoneme"), generation_fps=20,
        batch_size=2, conditioning_chunks=[], conditioning_ready_frames=0,
        conditioning_complete=True, total_frames=20, frames_per_chunk=10,
        startup_chunk_frames=10, startup_chunk_count=1, total_chunks=2,
        start_offset_frames=0, cancel_event=threading.Event(),
        completion_future=None, main_loop=None, output_mode="webrtc",
    )


def make_scheduler(job):
    scheduler = HLSGPUStreamScheduler.__new__(HLSGPUStreamScheduler)
    scheduler.webrtc_pose_crossfade_frames = 2
    scheduler.compose_executor = DeferredExecutor()
    scheduler.jobs = {job.request_id: job}
    scheduler.condition = threading.Condition()
    # These tests exercise composition/append ownership without starting a GPU
    # loop or finalizing a deliberately incomplete synthetic generation.
    scheduler._finalize_ready_jobs = Mock()
    scheduler._finalize_cancelled_jobs = Mock()
    return scheduler


class CurrentProfileValidationTest(unittest.TestCase):
    def test_explicit_profile_requires_existing_source_bound_eyes(self):
        good = current_fixture()
        self.assertEqual(MotionBank(good).current_phoneme, good["current_phoneme"])
        mutations = {
            "unsupported method": lambda m: m["current_phoneme"].update(method="guess"),
            "stale source": lambda m: m["current_phoneme"]["source_hashes"].update({TALK: "stale"}),
            "missing source": lambda m: m["current_phoneme"]["source_hashes"].pop(SMILE),
            "extra source": lambda m: m["current_phoneme"]["source_hashes"].update(extra="hash"),
            "missing eyes": lambda m: m.pop("eye_blend"),
            "different source dimensions": lambda m: m["sources"][TALK].update(width=63),
            "incomplete eyes": lambda m: m["eye_blend"]["frames"][TALK].pop(),
            "unknown setting": lambda m: m["current_phoneme"].update(unbound_option=True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                manifest = copy.deepcopy(good)
                mutate(manifest)
                with self.assertRaises(ValueError):
                    MotionBank(manifest)

    def test_omission_keeps_original_and_eye_only_profiles_separate(self):
        baseline = MotionBank(fixture())
        eyes = MotionBank(eye_fixture())
        candidate = MotionBank(current_fixture())
        self.assertIsNone(baseline.current_phoneme)
        self.assertIsNone(eyes.current_phoneme)
        self.assertNotEqual(candidate.routing_sha256, eyes.routing_sha256)
        self.assertNotEqual(candidate.routing_sha256, baseline.routing_sha256)

    def test_profile_requires_exact_original_frame_indices(self):
        bank = MotionBank(current_fixture())
        raw = layer(20)
        for old_index, new_index in ((None, 0), (0, None), (-1, 0), (0, 41), (1.5, 0)):
            with self.subTest(indices=(old_index, new_index)), self.assertRaises(ValueError):
                bank.blend_current(raw["raw"], raw["raw"], pixels(150),
                                   raw["alpha"], raw["alpha"], .5,
                                   IDLE, old_index, TALK, new_index)


class CurrentSchedulerOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.bank = MotionBank(current_fixture())
        self.job = make_job(self.bank)
        self.scheduler = make_scheduler(self.job)

    def test_freezes_previous_raw_body_alpha_and_original_indices_across_batches(self):
        images = [pixels(value) for value in (110, 120, 130, 140, 150)]
        layers = [layer(value) for value in (10, 20, 30, 40, 50)]
        calls = []

        def blend(old_raw, new_raw, composed, old_alpha, new_alpha,
                  progress, old_pose, old_index, new_pose, new_index):
            calls.append((old_raw.copy(), new_raw.copy(), composed.copy(),
                          old_alpha["values"].copy(), new_alpha["values"].copy(),
                          old_pose, old_index, new_pose, new_index))
            return composed.copy()

        with patch.object(self.bank, "blend_current", blend), patch.object(
                self.bank, "blend", side_effect=AssertionError("Composed-old path ran")):
            first = self.scheduler._apply_webrtc_pose_crossfade(
                self.job, images[:3], [IDLE, IDLE, TALK], [0, 0, 3],
                [11, 12, 29], raw_layers=layers[:3])
            second = self.scheduler._apply_webrtc_pose_crossfade(
                self.job, images[3:], [TALK, TALK], [3, 3], [30, 32],
                raw_layers=layers[3:])
        self.assertEqual(len(first) + len(second), 5)
        self.assertEqual(len(calls), 3)
        for offset, source_index in enumerate((29, 30, 32), start=2):
            call = calls[offset - 2]
            np.testing.assert_array_equal(call[0], layers[1]["raw"])
            np.testing.assert_array_equal(call[1], layers[offset]["raw"])
            np.testing.assert_array_equal(call[2], images[offset])
            np.testing.assert_array_equal(call[3], layers[1]["alpha"]["values"])
            np.testing.assert_array_equal(call[4], layers[offset]["alpha"]["values"])
            self.assertEqual(call[5:], (IDLE, 12, TALK, source_index))
        self.assertEqual(self.job.webrtc_last_source_frame, 32)
        self.assertEqual(self.job.webrtc_pose_crossfade_frames_applied, 3)
        np.testing.assert_array_equal(self.job.webrtc_last_raw_pose_frame, layers[-1]["raw"])
        self.assertIsNone(self.job.webrtc_pose_crossfade_raw_anchor)
        self.assertIsNone(self.job.webrtc_pose_crossfade_alpha_anchor)
        for original, output in zip(images, first + second):
            np.testing.assert_array_equal(original, output)

    def test_missing_layers_fail_before_starting_a_transition(self):
        for payload in (None, [], [None], [{"raw": pixels(1)}]):
            with self.subTest(payload=type(payload).__name__):
                with self.assertRaisesRegex(ValueError, "Current phoneme composition requires"):
                    self.scheduler._apply_webrtc_pose_crossfade(
                        self.job, [pixels(101)], [IDLE], [0], [11], raw_layers=payload)
                self.assertIsNone(self.job.webrtc_last_pose_id)
                self.assertIsNone(self.job.webrtc_last_raw_pose_frame)

    def test_missing_indices_fail_even_when_raw_layers_exist(self):
        with self.assertRaisesRegex(ValueError, "every source index"):
            self.scheduler._apply_webrtc_pose_crossfade(
                self.job, [pixels(101)], [IDLE], [0], raw_layers=[layer(1)])
        self.assertIsNone(self.job.webrtc_last_pose_id)

    def test_invalid_second_layer_cannot_advance_first_frame_history(self):
        invalid = [
            ({"raw": pixels(2).astype(np.float32), "alpha": layer(2)["alpha"]}, IDLE, 12),
            ({"raw": pixels(2)[:32], "alpha": layer(2)["alpha"]}, IDLE, 12),
            ({"raw": pixels(2), "alpha": {"bounds": (20, 32, 40, 56),
                                           "values": np.zeros((2, 2), np.uint8)}}, IDLE, 12),
            (layer(2), "unknown_pose", 12),
            (layer(2), IDLE, None),
            (layer(2), IDLE, True),
            (layer(2), IDLE, 41),
        ]
        for bad_layer, pose, index in invalid:
            with self.subTest(pose=pose, index=index, shape=bad_layer["raw"].shape):
                job = make_job(self.bank)
                with self.assertRaises(ValueError):
                    self.scheduler._apply_webrtc_pose_crossfade(
                        job, [pixels(111), pixels(112)], [IDLE, pose], [0, 0], [11, index],
                        raw_layers=[layer(11), bad_layer])
                self.assertIsNone(job.webrtc_last_pose_id)
                self.assertIsNone(job.webrtc_last_raw_pose_frame)
                self.assertEqual(job.webrtc_pose_crossfade_frames_applied, 0)

    def test_missing_crossfade_counts_do_not_silently_disable_transition(self):
        for counts in (None, [], [0, 3]):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "every crossfade length"):
                self.scheduler._apply_webrtc_pose_crossfade(
                    self.job, [pixels(111)], [IDLE], counts, [11], raw_layers=[layer(11)])
        self.assertIsNone(self.job.webrtc_last_pose_id)

    def test_no_transition_preserves_current_composed_frame_zero_exactly(self):
        images = [pixels(131), pixels(147)]
        with patch.object(self.bank, "blend_current", side_effect=AssertionError("Unexpected transition")):
            actual = self.scheduler._apply_webrtc_pose_crossfade(
                self.job, images, [IDLE, IDLE], [0, 0], [11, 12],
                raw_layers=[layer(11), layer(12)])
        for expected, output in zip(images, actual):
            np.testing.assert_array_equal(expected, output)
        self.assertEqual(self.job.webrtc_pose_crossfade_count, 0)


class CurrentWorkerOrderingTest(unittest.TestCase):
    def setUp(self):
        self.bank = MotionBank(current_fixture())
        self.job = make_job(self.bank, [(IDLE, 12), (TALK, 29), (TALK, 30)])
        self.scheduler = make_scheduler(self.job)
        self.published = []
        self.job.frame_batch_callback = lambda frames, start, total: self.published.append(
            (start, [frame.copy() for frame in frames]))

    def test_out_of_order_completion_keeps_layers_until_ordered_append(self):
        self.scheduler._dispatch_compose_batch(self.job, [pixels(112)], 0)
        self.scheduler._dispatch_compose_batch(self.job, [pixels(129), pixels(130)], 1)
        self.scheduler.compose_executor.finish(1)
        self.scheduler._drain_completed_composes()
        self.assertEqual(self.published, [])
        self.assertIsNone(self.job.webrtc_last_pose_id)
        self.assertIsNone(self.job.webrtc_last_raw_pose_frame)
        self.assertIn(1, self.job.composed_batches)
        self.assertEqual(len(self.job.composed_batches[1]["live_raw_layers"]), 2)
        self.scheduler.compose_executor.finish(0)
        calls = []

        def blend(old_raw, new_raw, composed, old_alpha, new_alpha, progress,
                  old_pose, old_index, new_pose, new_index):
            calls.append((old_raw.copy(), new_raw.copy(), old_index, new_index))
            return composed.copy()

        with patch.object(self.bank, "blend_current", blend):
            self.scheduler._drain_completed_composes()
        self.assertEqual([start for start, frames in self.published], [1, 2])
        self.assertEqual([int(f[0, 0, 0]) for _, batch in self.published for f in batch],
                         [112, 129, 130])
        self.assertEqual([(call[2], call[3]) for call in calls], [(12, 29), (12, 30)])
        for call in calls:
            np.testing.assert_array_equal(call[0], pixels(12))
        self.assertEqual(self.job.next_compose_sequence, 2)
        self.assertEqual(self.job.composed_batches, {})
        self.assertTrue(all(kwargs.get("return_layers") is True
                            for _, kwargs in self.job.avatar.calls))

    def test_cancelled_late_composition_cannot_publish_or_replace_anchor(self):
        self.scheduler._apply_webrtc_pose_crossfade(
            self.job, [pixels(110)], [IDLE], [0], [10], raw_layers=[layer(10)])
        self.scheduler._dispatch_compose_batch(self.job, [pixels(129)], 1)
        self.job.cancel_event.set()
        self.scheduler.compose_executor.finish(0)
        with patch.object(self.bank, "blend_current", side_effect=AssertionError("Cancelled blend ran")):
            self.scheduler._drain_completed_composes()
        self.assertEqual(self.published, [])
        self.assertIsNone(self.job.error_message)
        self.assertEqual(self.job.webrtc_last_pose_id, IDLE)
        self.assertEqual(self.job.webrtc_last_source_frame, 10)
        np.testing.assert_array_equal(self.job.webrtc_last_raw_pose_frame, pixels(10))
        self.assertEqual(self.job.composed_batches, {})
        self.assertEqual(self.job.webrtc_pose_crossfade_count, 0)

    def test_cancellation_during_cpu_blend_prevents_batch_publication(self):
        self.scheduler._apply_webrtc_pose_crossfade(
            self.job, [pixels(110)], [IDLE], [0], [10], raw_layers=[layer(10)])
        self.job.session.record_rendered_pose_batch = Mock()
        self.scheduler._dispatch_compose_batch(self.job, [pixels(129)], 1)
        self.scheduler.compose_executor.finish(0)

        def cancel_during_blend(old_raw, new_raw, composed, *args):
            self.job.cancel_event.set()
            return composed.copy()

        with patch.object(self.bank, "blend_current", cancel_during_blend) as blend:
            self.scheduler._drain_completed_composes()
        self.assertTrue(self.job.cancel_event.is_set())
        self.assertEqual(self.published, [])
        self.job.session.record_rendered_pose_batch.assert_not_called()
        self.assertEqual(self.job.composed_frame_idx, 0)
        self.assertIsNone(self.job.error_message)

    def test_callback_cancellation_cannot_restore_reset_session_pose_trace(self):
        self.job.session.record_rendered_pose_batch = Mock()

        def cancelled_callback(frames, start, total):
            self.job.cancel_event.set()

        self.job.frame_batch_callback = Mock(side_effect=cancelled_callback)
        self.scheduler._dispatch_compose_batch(self.job, [pixels(112)], 0)
        self.scheduler.compose_executor.finish(0)
        self.scheduler._drain_completed_composes()
        self.job.frame_batch_callback.assert_called_once()
        self.job.session.record_rendered_pose_batch.assert_not_called()
        self.assertEqual(self.job.composed_frame_idx, 0)
        self.assertEqual(self.job.chunks_appended, 0)
        self.assertIsNone(self.job.error_message)

    def test_default_and_eye_only_composers_never_request_layers(self):
        for manifest in (fixture(), eye_fixture()):
            with self.subTest(profile="eye_blend" in manifest):
                bank = MotionBank(manifest)
                job = make_job(bank, [(IDLE, 12)])
                scheduler = make_scheduler(job)
                scheduler._dispatch_compose_batch(job, [pixels(112)], 0)
                scheduler.compose_executor.finish(0)
                info = job.compose_tasks[0].result()
                self.assertEqual(job.avatar.calls, [(12, {"background_frame": None})])
                self.assertFalse(info.get("live_raw_layers"))
                np.testing.assert_array_equal(info["frames"][0], pixels(112))
                self.assertIsNone(job.webrtc_last_pose_id)

    def test_failed_layer_batch_sets_job_error_without_killing_other_jobs(self):
        # Layer validation runs at ordered append, after a worker has completed.
        # Its failure must remain a job failure rather than escape the GPU loop.
        future = Future()
        future.set_result({
            "frames": [pixels(150)], "live_pose_ids": [IDLE], "live_pose_crossfade_frames": [0],
            "live_source_frame_indices": [12], "live_raw_layers": [None],
        })
        self.job.compose_tasks[0] = future
        self.job.compose_sequence = 1
        healthy = make_job(self.bank)
        healthy.request_id = "healthy_current_phoneme_cpu"
        published = []
        healthy.frame_batch_callback = lambda frames, start, total: published.extend(frames)
        healthy_future = Future()
        healthy_future.set_result({
            "frames": [pixels(170)], "live_pose_ids": [IDLE], "live_pose_crossfade_frames": [0],
            "live_source_frame_indices": [14], "live_raw_layers": [layer(14)],
        })
        healthy.compose_tasks[0] = healthy_future
        healthy.compose_sequence = 1
        self.scheduler.jobs[healthy.request_id] = healthy
        self.scheduler._drain_completed_composes()
        self.assertEqual(len(published), 1)
        np.testing.assert_array_equal(published[0], pixels(170))
        self.assertIsNone(healthy.error_message)
        self.assertEqual(healthy.webrtc_last_source_frame, 14)
        self.assertIsNotNone(self.job.error_message)
        self.assertIn("Current phoneme", self.job.error_message)
        self.assertEqual(self.published, [])
        self.assertIsNone(self.job.webrtc_last_pose_id)

    def test_finalization_releases_last_layers_and_frozen_history(self):
        self.scheduler._apply_webrtc_pose_crossfade(
            self.job, [pixels(110), pixels(129)], [IDLE, TALK], [0, 3], [10, 29],
            raw_layers=[layer(10), layer(29)])
        self.assertIsNotNone(self.job.webrtc_last_raw_pose_frame)
        self.assertIsNotNone(self.job.webrtc_pose_crossfade_raw_anchor)
        self.job.composed_batches[1] = {"live_raw_layers": [layer(30)]}
        self.scheduler._set_request_status = Mock()
        self.scheduler._resolve_completion = Mock()
        self.scheduler._finalize_job(self.job, "cancelled")
        self.assertTrue(self.job.finalized)
        self.assertNotIn(self.job.request_id, self.scheduler.jobs)
        for name in ("webrtc_last_raw_pose_frame", "webrtc_last_pose_alpha",
                     "webrtc_pose_crossfade_raw_anchor", "webrtc_pose_crossfade_alpha_anchor"):
            self.assertIsNone(getattr(self.job, name), name)
        self.assertEqual(self.job.composed_batches, {})


class CurrentProfileAttachmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="motion-current-attachment-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = eye_fixture()
        self.manifest.update(status="reviewed", review={"accepted": True, "reviewer": "parent"})
        for pose, source in self.manifest["sources"].items():
            path = self.root / (pose + ".mp4")
            path.write_bytes(("Synthetic source integrity fixture: " + pose).encode())
            source.update(path=str(path), sha256=file_hash(path))
        self.manifest["eye_blend"]["source_hashes"] = {
            pose: source["sha256"] for pose, source in self.manifest["sources"].items()}
        self.parent = self.root / "parent" / "motion-atlas.json"
        self.parent.parent.mkdir()
        self.parent.write_text(json.dumps(self.manifest) + "\n")
        self.output = self.root / "candidate" / "motion-atlas.json"

    def attach(self):
        from scripts.attach_motion_current_profile import attach
        return attach(self.parent, self.output)

    def test_fresh_candidate_binds_sources_and_preserves_parent_bytes(self):
        parent_bytes = self.parent.read_bytes()
        sources = {pose: Path(source["path"]).read_bytes()
                   for pose, source in self.manifest["sources"].items()}
        self.parent.chmod(0o444)
        for source in self.manifest["sources"].values():
            Path(source["path"]).chmod(0o444)
        result = self.attach()
        candidate = json.loads(self.output.read_text())
        registration = json.loads(self.output.with_name("motion-registration.json").read_text())
        hashes = self.manifest["eye_blend"]["source_hashes"]
        self.assertEqual(candidate["status"], "candidate_requires_recorded_review")
        self.assertNotIn("review", candidate)
        self.assertEqual(candidate["current_phoneme"], {
            "method": "current_similarity_v1", "source_hashes": hashes})
        self.assertEqual(candidate["current_phoneme_provenance"], {
            "parent_atlas_sha256": file_hash(self.parent)})
        self.assertEqual(candidate["eye_blend"], self.manifest["eye_blend"])
        self.assertEqual(candidate["sources"], self.manifest["sources"])
        self.assertEqual(self.parent.read_bytes(), parent_bytes)
        for pose, source in self.manifest["sources"].items():
            self.assertEqual(Path(source["path"]).read_bytes(), sources[pose])
        self.assertEqual(registration["source_hashes"], hashes)
        self.assertEqual(registration["atlas_sha256"], file_hash(self.output))
        self.assertEqual(result["atlas_sha256"], file_hash(self.output))
        self.assertEqual(result["routing_sha256"], MotionBank(candidate).routing_sha256)
        self.assertNotEqual(result["routing_sha256"], MotionBank(self.manifest).routing_sha256)

    def test_changed_source_cannot_publish_candidate_or_registration(self):
        parent_bytes = self.parent.read_bytes()
        Path(self.manifest["sources"][TALK]["path"]).write_bytes(b"changed source bytes")
        with self.assertRaisesRegex(ValueError, "Source video changed"):
            self.attach()
        self.assertEqual(self.parent.read_bytes(), parent_bytes)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("motion-registration.json").exists())

    def test_collisions_or_missing_eye_geometry_cannot_publish(self):
        from scripts.attach_motion_current_profile import attach
        self.output.parent.mkdir()
        for name in ("motion-atlas.json", "motion-registration.json"):
            with self.subTest(existing=name):
                collision = self.output.with_name(name)
                collision.write_bytes(b"existing bytes must survive")
                try:
                    with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                        self.attach()
                    self.assertEqual(collision.read_bytes(), b"existing bytes must survive")
                    other = self.output.with_name("motion-registration.json"
                             if name == "motion-atlas.json" else "motion-atlas.json")
                    self.assertFalse(other.exists())
                finally:
                    collision.unlink()
        parent_bytes = self.parent.read_bytes()
        with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
            attach(self.parent, self.parent)
        self.assertEqual(self.parent.read_bytes(), parent_bytes)
        self.manifest.pop("eye_blend")
        self.parent.write_text(json.dumps(self.manifest) + "\n")
        with self.assertRaisesRegex(ValueError, "eye geometry first"):
            self.attach()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("motion-registration.json").exists())


if __name__ == "__main__":
    unittest.main()
