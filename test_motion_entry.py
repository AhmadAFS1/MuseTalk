"""CPU checks for moving-idle prebuffer and phoneme-preserving speech entry."""
import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import av
import numpy as np

from scripts.motion_transitions import IDLE, SMILE, MotionBank
from scripts.webrtc_tracks import SwitchableVideoStreamTrack, VideoSyncClock, SilenceAudioStreamTrack
from test_motion_transitions import fixture


def frame(value):
    return av.VideoFrame.from_ndarray(np.full((64, 64, 3), value, np.uint8), format="bgr24")


class Decoder:
    instances = []
    decode_gate = None
    fail_worker = False

    def __init__(self, path, fps=24, decode_threads=0):
        self.index = -1
        self.closed = False
        self.fps = fps
        self.worker = bool(decode_threads)
        self.path = path
        self.instances.append(self)

    def read_frame(self):
        if self.worker and self.decode_gate is not None:
            self.decode_gate.wait(timeout=3)
        if self.worker and self.fail_worker:
            raise RuntimeError("test source decode failure")
        self.index = (self.index + 1) % 41
        return frame(self.index * 4)

    def stop(self):
        self.closed = True

    def get_timing(self):
        return {"source_frame_index": max(0, self.index), "source_fps": self.fps,
                "source_frame_count": 41}

    def next_frame_starts_cycle(self):
        return self.index == 40

    def last_read_started_cycle(self):
        return self.index == 0


class MovingIdleEntryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Decoder.instances = []
        Decoder.decode_gate = None
        Decoder.fail_worker = False
        self.decoder_patch = patch("scripts.webrtc_tracks.IdleVideoStreamTrack", Decoder)
        self.decoder_patch.start()
        self.clock = VideoSyncClock(20, strict_fifo=True)
        self.track = SwitchableVideoStreamTrack(
            "idle.mp4", source_fps=20, output_fps=20, idle_source_fps=24,
            sync_clock=self.clock, prebuffer_seconds=.1, adaptive_fps=False,
            idle_pose_id=IDLE,
        )
        self.track.configure_motion_bank(MotionBank(fixture()))
        self.audio = SilenceAudioStreamTrack(sync_clock=self.clock)
        # The actual audio gate only needs a source to be armed. No waveform is
        # consumed in these tests; first-source frame ownership is checked below.
        self.audio._armed_source = object()
        self.audio._source_sync_clock = self.clock

    async def asyncTearDown(self):
        if Decoder.decode_gate is not None:
            Decoder.decode_gate.set()
        for name in ("_motion_entry_task", "_motion_task"):
            task = getattr(self.track, name, None)
            if task is not None:
                await task
        self.track.stop()
        self.decoder_patch.stop()

    async def tick(self):
        # Exercise production recv selection/stamping without real-time sleeps.
        self.track._last_ts = time.monotonic() - 1
        if not getattr(self, "manual_audio_clock", False):
            self.clock.publish_audio_transport_next_pts(self.track._rtp_frame_index / self.track._output_fps)
        return await self.track.recv()

    async def queue_turn(self):
        owner = self.track.start_live()
        self.clock.release_playout()
        self.clock.publish_audio_transport_next_pts(0)
        frames = [frame(180), frame(200)]
        metadata = [{"pose_id": IDLE, "source_frame": i + 3,
                     "generation_frame": i, "mode": "live"} for i in range(2)]
        for image, meta in zip(frames, metadata):
            await self.track._push_video_frame(image, time.monotonic(), 0, generation_id=owner, metadata=meta)
        return owner, frames

    async def test_prebuffer_idle_advances_even_when_scheduler_requests_hold(self):
        await self.tick()
        timing = self.track.capture_idle_sync_timing(20, hold=True)
        self.assertFalse(timing["hold_enabled"])
        self.track.start_live()
        values = []
        for _ in range(8):
            await self.tick()
            values.append(self.track._idle.index)
        self.assertEqual(len(set(values)), 8)
        self.assertFalse(self.clock.started.is_set())
        self.assertFalse(self.track._idle_sync_hold_active)

    async def test_entry_keeps_all_lip_frames_and_audio_gated_until_first_phoneme(self):
        await self.tick()
        _, generated = await self.queue_turn()
        await self.track._motion_entry_task
        ids = []
        for _ in range(6):
            output = await self.tick()
            ids.append(self.track._idle.index)
            self.assertEqual(self.track._queue.qsize(), 2)
            self.assertFalse(self.clock.started.is_set())
            self.assertFalse(self.audio._source_is_due(time.monotonic()))
            self.assertEqual(self.track._emitted_motion["mode"], "speech_entry_body_bridge")
        self.assertEqual(len(set(ids)), 6)
        expected = frame(3 * 4).reformat(format="yuv420p").to_ndarray(format="bgr24")
        np.testing.assert_array_equal(output.to_ndarray(format="bgr24"), expected)
        first_live = await self.tick()
        self.assertIs(first_live, generated[0])
        self.assertEqual(self.track._emitted_motion["generation_frame"], 0)
        self.assertEqual(self.track._queue.qsize(), 1)
        self.assertTrue(self.audio._source_is_due(time.monotonic()))
        self.assertEqual(self.track._motion_entries[-1]["first_live_generation_frame"], 0)

    async def test_target_decode_does_not_freeze_idle_and_cancelled_worker_cannot_install(self):
        Decoder.decode_gate = threading.Event()
        await self.queue_turn()
        for _ in range(5):
            await self.tick()
        self.assertGreaterEqual(self.track._idle.index, 4)
        task = self.track._motion_entry_task
        self.track.end_live()
        Decoder.decode_gate.set()
        await task
        self.assertIsNone(self.track._motion_entry)
        self.assertIsNone(self.track._last_live_frame)
        self.assertEqual(self.track._queue.qsize(), 0)
        self.assertTrue(all(d.closed for d in Decoder.instances if d.worker))
        self.assertIsNone(self.clock.first_live_video_rtp_seconds)

    async def test_abort_mid_entry_uses_displayed_bridge_then_next_turn_starts_at_zero(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        first_bridge = await self.tick()
        self.assertIs(self.track._motion_entry_last_frame, first_bridge)
        self.track.end_live()
        await self.track._motion_task
        self.assertEqual(self.track._motion_returns[-1]["source_output_frame"], 0)
        for _ in range(6):
            await self.tick()
        self.assertTrue(self.track._motion_settled.is_set())
        self.assertEqual(self.track._motion_returns[-1]["status"], "completed")
        _, new_frames = await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.assertIs(await self.tick(), new_frames[0])
        self.assertEqual(self.track._emitted_motion["generation_frame"], 0)

    async def test_abort_while_flow_worker_runs_never_emits_cancelled_entry(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        gate = threading.Event()
        entered = threading.Event()
        def slow_blend(old, new, progress):
            entered.set()
            gate.wait(timeout=3)
            return new
        with patch("scripts.webrtc_motion_playback.flow_blend", slow_blend):
            pending = asyncio.create_task(self.tick())
            await asyncio.to_thread(entered.wait, 1)
            self.track.end_live()
            gate.set()
            await pending
        self.assertIsNone(self.track._motion_entry)
        self.assertNotEqual(self.track._emitted_motion["mode"], "speech_entry_body_bridge")
        self.assertIsNone(self.clock.first_live_video_rtp_seconds)

    async def test_aborted_flow_exception_emits_installed_recovery_frame_with_its_metadata(self):
        from scripts.motion_transitions import flow_blend as actual_blend
        await self.queue_turn()
        await self.track._motion_entry_task
        await self.tick()
        gate, entered = threading.Event(), threading.Event()
        calls = [0]
        def fail_first_blend(old, new, progress):
            calls[0] += 1
            if calls[0] == 1:
                entered.set()
                gate.wait(timeout=3)
                raise RuntimeError("late cancelled flow failure")
            return actual_blend(old, new, progress)
        with patch("scripts.webrtc_motion_playback.flow_blend", fail_first_blend):
            pending = asyncio.create_task(self.tick())
            await asyncio.to_thread(entered.wait, 1)
            self.track.end_live()
            await self.track._motion_task
            expected_frame = self.track._idle_transition_frames[0]
            expected_metadata = dict(self.track._motion_transition_ids[0])
            gate.set()
            actual_frame = await pending
        self.assertIs(actual_frame, expected_frame)
        self.assertEqual(self.track._emitted_motion["source_frame"], expected_metadata["source_frame"])
        self.assertEqual(self.track._emitted_motion["mode"], "mouth_release_return")
        self.assertFalse(self.track._motion_entry_failure.is_set())
        self.assertIsNone(self.clock.first_live_video_rtp_seconds)

    async def test_decode_failure_notifies_owner_and_keeps_idle_moving_without_audio(self):
        Decoder.fail_worker = True
        owner, _ = await self.queue_turn()
        await self.track._motion_entry_task
        failure = await self.track.wait_for_motion_entry_failure(owner)
        self.assertEqual(failure["status"], "failed")
        self.assertIn("test source decode failure", failure["error"])
        self.assertIsNone(await self.track.wait_for_motion_entry_failure(owner + 1))
        indices = []
        for _ in range(4):
            await self.tick()
            indices.append(self.track._idle.index)
            self.assertFalse(self.audio._source_is_due(time.monotonic()))
        self.assertEqual(len(set(indices)), 4)
        self.assertEqual(self.track._queue.qsize(), 2)
        self.track.end_live()
        Decoder.fail_worker = False
        await self.queue_turn()
        await self.track._motion_entry_task
        self.assertFalse(self.track._motion_entry_failure.is_set())
        for _ in range(7):
            await self.tick()
        self.assertTrue(self.audio._source_is_due(time.monotonic()))

    async def test_reactive_initial_pose_decodes_its_own_prepared_path(self):
        self.track._current_idle_video_path = "prepared-smile.mp4"
        self.track._current_idle_pose_id = SMILE
        self.track.configure_motion_bank(MotionBank(fixture()), source_paths={
            IDLE: "prepared-idle.mp4", SMILE: "prepared-smile.mp4"})
        self.assertEqual(self.track._motion_idle_path, "prepared-idle.mp4")
        owner = self.track.start_live()
        self.clock.release_playout()
        self.clock.publish_audio_transport_next_pts(0)
        first = frame(180)
        for i in range(2):
            await self.track._push_video_frame(first, time.monotonic(), 0, generation_id=owner,
                metadata={"pose_id": SMILE, "source_frame": i+3, "generation_frame": i, "mode": "live"})
        await self.track._motion_entry_task
        self.assertEqual([d.path for d in Decoder.instances if d.worker], ["prepared-smile.mp4"])
        for _ in range(6):
            await self.tick()
        self.assertIs(await self.tick(), first)
        self.assertEqual(self.track._emitted_motion["pose_id"], SMILE)
        self.assertEqual(self.track._emitted_motion["generation_frame"], 0)

    async def test_invalid_first_frame_metadata_notifies_owner_instead_of_stalling(self):
        for metadata in (None, {"generation_frame": 1},
                         {"generation_frame": 0, "pose_id": IDLE, "source_frame": 10000}):
            owner = self.track.start_live()
            await self.track._push_video_frame(frame(180), time.monotonic(), 0,
                                              generation_id=owner, metadata=metadata)
            failure = await asyncio.wait_for(self.track.wait_for_motion_entry_failure(owner), .2)
            self.assertEqual(failure["status"], "failed")
            self.assertFalse(self.clock.started.is_set())
            self.track.end_live()

    async def test_preparation_timeout_keeps_loop_responsive_and_cannot_fail_next_generation(self):
        gate = threading.Event()
        Decoder.decode_gate = gate
        with patch("scripts.webrtc_motion_playback.MOTION_ENTRY_PREPARE_TIMEOUT_SECONDS", .04):
            owner, _ = await self.queue_turn()
            for _ in range(4):
                await self.tick()
                await asyncio.sleep(.001)
            self.assertGreaterEqual(self.track._idle.index, 3)
            failure = await asyncio.wait_for(self.track.wait_for_motion_entry_failure(owner), .5)
            self.assertIn("preparation exceeded", failure["error"])
            self.assertFalse(self.clock.started.is_set())
            old_worker = next(d for d in Decoder.instances if d.worker)
            self.track.end_live()
            Decoder.decode_gate = None
            new_owner, _ = await self.queue_turn()
            await self.track._motion_entry_task
            self.assertEqual(self.track._motion_entry["generation_id"], new_owner)
            self.assertEqual(self.track._motion_entry["status"], "ready")
            gate.set()
            deadline = time.monotonic() + .5
            while not old_worker.closed and time.monotonic() < deadline:
                await asyncio.sleep(.005)
            self.assertTrue(old_worker.closed)
            self.assertFalse(self.track._motion_entry_failure.is_set())
            self.assertEqual(self.track._motion_entry["generation_id"], new_owner)

    async def test_flow_timeout_notifies_owner_without_releasing_first_phoneme(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        gate = threading.Event()
        def stalled_blend(old, new, progress):
            gate.wait(timeout=2)
            return new
        try:
            with patch("scripts.webrtc_motion_playback.MOTION_ENTRY_FLOW_TIMEOUT_SECONDS", .03), \
                 patch("scripts.webrtc_motion_playback.flow_blend", stalled_blend):
                pending = asyncio.create_task(self.tick())
                failure = await asyncio.wait_for(self.track.wait_for_motion_entry_failure(), .5)
                await pending
                self.assertIn("flow exceeded", failure["error"])
                self.assertFalse(self.clock.started.is_set())
                self.assertEqual(self.track._queue.qsize(), 2)
                self.assertEqual(self.track._emitted_motion["mode"], "idle")
                self.track.end_live()
        finally:
            gate.set()

    async def test_audio_ahead_fills_contiguous_video_slot_and_preserves_first_phoneme(self):
        from scripts.webrtc_tracks import SyncedAudioStreamTrack
        _, generated = await self.queue_turn()
        await self.track._motion_entry_task
        self.track._rtp_frame_index = 40
        for _ in range(6):
            previous = await self.tick()
        previous_seconds = float(previous.pts * previous.time_base)
        self.assertAlmostEqual(previous_seconds, 2.25)
        self.manual_audio_clock = True
        self.audio._armed_source = None
        self.audio._timestamp = round(2.34 * self.audio.sample_rate)
        source = SyncedAudioStreamTrack("unused.wav", use_ffmpeg_convert=False, sync_clock=self.clock)
        source._audio_samples = b"\x01\x00" * (source._samples_per_frame * 2)
        source._fully_loaded = True
        self.audio.arm_source(source, sync_clock=self.clock, start_time=self.clock.playout_start_time)
        silent = await self.audio.recv()
        self.assertEqual(source._read_position, 0)
        self.assertAlmostEqual(self.clock.audio_transport_next_pts_seconds, 2.36)
        padding = await self.tick()
        padding_seconds = float(padding.pts * padding.time_base)
        self.assertAlmostEqual(padding_seconds, 2.30)
        self.assertAlmostEqual(padding_seconds - previous_seconds, .05)
        self.assertEqual(self.track._emitted_motion["mode"], "entry_transport_alignment")
        self.assertEqual(self.track._queue.qsize(), 2)
        self.assertFalse(self.clock.started.is_set())
        # Even with a future normal deadline, reconciliation must not sleep a
        # full frame and allow silent audio to get equally far ahead again.
        self.track._last_ts = time.monotonic()
        started = time.monotonic()
        first = await self.track.recv()
        self.assertLess(time.monotonic() - started, .04)
        self.assertIs(first, generated[0])
        self.assertAlmostEqual(float(first.pts * first.time_base), 2.35)
        self.assertAlmostEqual(float(first.pts * first.time_base) - padding_seconds, .05)
        self.assertEqual(self.track._emitted_motion["generation_frame"], 0)
        self.assertEqual(self.track._live_rtp_phase_correction_frames, 0)
        first_audio = await self.audio.recv()
        self.assertEqual(first_audio.pts - silent.pts, self.audio.samples)
        self.assertAlmostEqual(first_audio.pts / self.audio.sample_rate, 2.36)
        self.assertLessEqual(self.clock.get_stats()["first_live_rtp_abs_mismatch_seconds"], .05)
        self.assertIsNone(self.clock.audio_transport_rebase_target_seconds)
        self.assertEqual(self.track._motion_entries[-1]["clock_wait_frames"], 1)
        self.audio.cancel_source(source)
        source.stop()

    async def test_video_ahead_waits_for_silent_audio_without_rebasing_either_clock(self):
        _, generated = await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.manual_audio_clock = True
        video_pts = self.track._rtp_frame_index / self.track._output_fps
        self.clock.publish_audio_transport_next_pts(video_pts - .12)
        pending = asyncio.create_task(self.tick())
        await asyncio.sleep(.01)
        self.assertFalse(pending.done())
        self.assertFalse(self.clock.started.is_set())
        self.assertEqual(self.track._queue.qsize(), 2)
        self.clock.publish_audio_transport_next_pts(video_pts - .04)
        first = await asyncio.wait_for(pending, .1)
        self.assertIs(first, generated[0])
        self.assertAlmostEqual(float(first.pts * first.time_base), video_pts)
        self.assertIsNone(self.clock.audio_transport_rebase_target_seconds)
        self.assertEqual(self.track._live_rtp_phase_correction_frames, 0)
        self.assertLessEqual(abs(self.clock.first_live_audio_target_seconds
                                 - self.clock.first_live_video_rtp_seconds), .05)

    async def test_large_skew_fails_closed_without_consuming_live_frames(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.manual_audio_clock = True
        video_pts = self.track._rtp_frame_index / self.track._output_fps
        self.clock.publish_audio_transport_next_pts(video_pts + .8)
        output = await self.tick()
        failure = await self.track.wait_for_motion_entry_failure()
        self.assertIn("transport clocks did not converge", failure["error"])
        self.assertAlmostEqual(float(output.pts * output.time_base), video_pts)
        self.assertAlmostEqual(self.clock.audio_transport_next_pts_seconds, video_pts + .8)
        self.assertEqual(self.track._queue.qsize(), 2)
        self.assertFalse(self.clock.started.is_set())
        self.track.end_live()

    async def test_persistent_small_skew_is_bounded_by_raw_padding_frame_budget(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.manual_audio_clock = True
        for _ in range(11):
            self.clock.publish_audio_transport_next_pts(self.track._rtp_frame_index / self.track._output_fps + .15)
            await self.tick()
        failure = await self.track.wait_for_motion_entry_failure()
        self.assertIn("transport clocks did not converge", failure["error"])
        self.assertEqual(self.track._motion_entry["clock_wait_frames"], 10)
        self.assertEqual(self.track._queue.qsize(), 2)
        self.assertFalse(self.clock.started.is_set())
        self.track.end_live()

    async def test_missing_audio_clock_times_out_without_releasing_speech(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.manual_audio_clock = True
        self.clock.audio_transport_next_pts_seconds = None
        with patch("scripts.webrtc_motion_playback.MOTION_ENTRY_CLOCK_TIMEOUT_SECONDS", .03):
            pending = asyncio.create_task(self.tick())
            failure = await asyncio.wait_for(self.track.wait_for_motion_entry_failure(), .2)
            await pending
        self.assertIn("transport clocks did not converge", failure["error"])
        self.assertGreaterEqual(self.track._motion_entry["clock_wait_seconds"], .03)
        self.assertEqual(self.track._queue.qsize(), 2)
        self.assertFalse(self.clock.started.is_set())
        self.track.end_live()

    async def test_abort_during_clock_alignment_cancels_entry_and_following_turn_starts_cleanly(self):
        await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.manual_audio_clock = True
        self.clock.publish_audio_transport_next_pts(self.track._rtp_frame_index / self.track._output_fps - .12)
        pending = asyncio.create_task(self.tick())
        await asyncio.sleep(.01)
        self.assertFalse(pending.done())
        self.assertFalse(self.clock.started.is_set())
        self.assertEqual(self.track._queue.qsize(), 2)
        self.track.end_live()
        await pending
        await self.track._motion_task
        self.assertEqual(self.track._motion_entries[-1]["status"], "cancelled")
        self.assertNotIn("first_live_output_frame", self.track._motion_entries[-1])
        self.manual_audio_clock = False
        for _ in range(6):
            await self.tick()
        self.assertTrue(self.track._motion_settled.is_set())
        _, generated = await self.queue_turn()
        await self.track._motion_entry_task
        for _ in range(6):
            await self.tick()
        self.assertIs(await self.tick(), generated[0])
        self.assertEqual(self.track._emitted_motion["generation_frame"], 0)

    async def test_closed_track_never_installs_target_from_late_worker(self):
        Decoder.decode_gate = threading.Event()
        await self.queue_turn()
        self.track.stop()
        Decoder.decode_gate.set()
        await self.track._motion_entry_task
        self.assertIsNone(self.track._motion_entry)
        self.assertTrue(all(d.closed for d in Decoder.instances if d.worker))


if __name__ == "__main__":
    unittest.main()
