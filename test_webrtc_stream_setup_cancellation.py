"""Run the real endpoint functions with CPU-only service doubles.

Extracting function AST nodes avoids importing the API module's model/server
initializers; the request, rollback, abort, and callback implementations remain
the exact current source, rather than a second implementation of their logic.
"""
import ast
import asyncio
import io
import json
import os
import shutil
import tempfile
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, Mock

from fastapi import File, Form, HTTPException, Request, UploadFile
from scripts.pose_protocol import (PoseProtocolError, normalize_session_event,
                                   normalize_stream_metadata)
from scripts.webrtc_manager import WebRTCSession, WebRTCSessionManager


def load_endpoint_functions(namespace):
    source = Path(__file__).with_name("api_server.py")
    names = {"webrtc_stream", "webrtc_pose_event", "_webrtc_turn_can_publish",
             "_start_live_track", "_release_webrtc_playout"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)


class SetupCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.phase = "normalization"
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.thread_release = threading.Event()
        self.loop = asyncio.get_running_loop()

        async def pause(phase):
            if self.phase == phase:
                self.entered.set()
                await self.release.wait()

        async def recovery():
            await pause("recovery")

        self.track = types.SimpleNamespace(
            motion_bank=object(), start_live=Mock(return_value=1), end_live=Mock(),
            get_stats=lambda: {}, wait_for_motion_settled=recovery)

        async def stage(*args, **kwargs):
            await pause("staging")
            return {"staged": True}
        self.track.stage_completion_idle_video = stage

        class Audio:
            def __init__(inner, *args, **kwargs):
                inner.stop = Mock()
            async def prepare(inner):
                await pause("audio")
            def get_stats(inner):
                return {}
        self.audio_class = Audio
        self.transport = types.SimpleNamespace(cancel_source=Mock(), arm_source=Mock())
        self.session = WebRTCSession(
            "session", "avatar", idle_track=self.track, idle_sender=object(),
            audio_sender=object(), silence_audio_track=self.transport,
            idle_video_path="idle.mp4", pose_protocol_enabled=True,
            pose_video_paths={"neutral_resting": "idle.mp4"})
        self.sessions = WebRTCSessionManager()
        self.sessions.sessions["session"] = self.session
        self.sessions._queue_pose_locked = AsyncMock(return_value={})
        self.sessions.finish_assistant_turn = AsyncMock(return_value={})

        async def stage_pose(session, pose_sequence, *, seq, turn_id, **kwargs):
            session.last_pose_seq = seq
            session.active_turn_id = turn_id
            session.assistant_active = True
            return {"accepted": True}
        self.sessions.queue_pose_sequence = stage_pose
        self.scheduler = types.SimpleNamespace(submit_webrtc_stream=Mock(return_value=True))
        self.manager = types.SimpleNamespace(active_requests={}, request_lock=threading.Lock())

        def normalize_audio(path):
            if self.phase == "normalization":
                self.loop.call_soon_threadsafe(self.entered.set)
                if not self.thread_release.wait(5):
                    raise TimeoutError("Test did not release normalization")
            return types.SimpleNamespace(media_path=str(path), media_duration_seconds=1,
                original_duration_seconds=1, leading_silence_removed_seconds=0,
                trailing_silence_removed_seconds=0, to_dict=lambda: {})

        self.ns = dict(globals(), manager=self.manager, webrtc_session_manager=self.sessions,
            hls_stream_scheduler=self.scheduler, _require_webrtc=lambda: None,
            _env_bool=lambda name, default=False: default,
            _sample_process_resource_snapshot=lambda: {},
            _get_runtime_diagnostic_counts=lambda: {}, _get_webrtc_sync_mode=lambda: "strict_fifo",
            _get_expected_webrtc_reveal_delay=lambda delay: delay,
            prepare_webrtc_audio_timeline=normalize_audio, SyncedAudioStreamTrack=Audio,
            av=types.SimpleNamespace(open=Mock(side_effect=ValueError("No media probe in CPU test"))))
        load_endpoint_functions(self.ns)

    async def asyncTearDown(self):
        self.thread_release.set()
        self.release.set()
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    async def start_request(self):
        upload = UploadFile(filename="speech.wav", file=io.BytesIO(b"test audio"))
        return await self.ns["webrtc_stream"](
            "session", upload, reaction_intent="none", pose_id="speaking_direct",
            pose_sequence=json.dumps(["speaking_direct", "neutral_resting"]),
            pose_plan=None, turn_id="turn", seq="1", effective="next_boundary",
            mouth_mode="lip_sync", audio_start="immediate")

    async def abort(self, turn="turn", seq=2):
        request = types.SimpleNamespace(json=AsyncMock(return_value={
            "event": "assistant_turn_aborted", "turn_id": turn, "seq": seq}))
        return await self.ns["webrtc_pose_event"]("session", request)

    async def exercise_phase(self, phase, supersede=False):
        self.phase = phase
        task = asyncio.create_task(self.start_request())
        try:
            await asyncio.wait_for(self.entered.wait(), 2)
            old_owner = self.session.stream_owner
            old_token = self.session.stream_cancel_event
            self.assertIsNotNone(old_token)
            self.assertEqual(self.manager.active_requests, {})
            result = await self.abort()
            self.assertEqual(result["status"], "accepted")
            self.assertTrue(old_token.is_set())
            if supersede:
                await self.sessions.finish_reserved_stream(self.session, old_owner)
                await self.sessions.reserve_stream(self.session, "new-owner")
                new_token = self.session.stream_cancel_event
                new_audio = self.audio_class()
                self.session.audio_player = new_audio
                self.session.active_turn_id = "new-turn"
                self.track.end_live.reset_mock()
            self.thread_release.set()
            self.release.set()
            with self.assertRaises(HTTPException) as error:
                await asyncio.wait_for(task, 2)
            self.assertEqual(error.exception.status_code, 409)
            self.scheduler.submit_webrtc_stream.assert_not_called()
            self.track.start_live.assert_not_called()
            self.transport.arm_source.assert_not_called()
            self.assertFalse(list(Path("uploads/audio").glob("*")))
            if supersede:
                self.assertEqual(self.session.stream_owner, "new-owner")
                self.assertIs(self.session.stream_cancel_event, new_token)
                self.assertFalse(new_token.is_set())
                new_audio.stop.assert_not_called()
                self.assertIs(self.session.audio_player, new_audio)
                self.track.end_live.assert_not_called()
            else:
                self.assertIsNone(self.session.stream_owner)
                self.assertIsNone(self.session.stream_cancel_event)
        finally:
            self.thread_release.set()
            self.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_abort_during_normalization_never_submits(self):
        await self.exercise_phase("normalization")

    async def test_abort_during_previous_motion_recovery_never_submits(self):
        await self.exercise_phase("recovery")

    async def test_abort_during_idle_decoder_staging_never_submits(self):
        await self.exercise_phase("staging")

    async def test_abort_during_audio_decode_never_submits(self):
        await self.exercise_phase("audio")

    async def test_late_old_rollback_does_not_cancel_new_reservation(self):
        await self.exercise_phase("audio", supersede=True)

    async def test_stale_start_callback_cannot_restart_video(self):
        await self.sessions.reserve_stream(self.session, "old")
        old_token = self.session.stream_cancel_event
        await self.sessions.finish_reserved_stream(self.session, "old")
        await self.sessions.reserve_stream(self.session, "new")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            await self.ns["_start_live_track"](self.track, session=self.session,
                                               request_id="old", cancel_event=old_token)
        self.track.start_live.assert_not_called()
        self.assertFalse(self.session.stream_cancel_event.is_set())

    async def test_legacy_track_does_not_start_motion_failure_supervisor(self):
        self.phase = "none"
        self.track.motion_bank = None
        self.track.wait_for_motion_entry_failure = AsyncMock(
            side_effect=AttributeError("Legacy track has no motion failure event"))
        self.track.push_bgr_frames_batch = AsyncMock(return_value=False)
        self.track.signal_generation_complete = Mock()
        await self.start_request()
        submitted = self.scheduler.submit_webrtc_stream.call_args.kwargs
        await asyncio.to_thread(submitted["frame_batch_callback"], [object()], 1, 20)
        self.track.start_live.assert_called_once()
        self.track.push_bgr_frames_batch.assert_awaited_once()
        self.track.wait_for_motion_entry_failure.assert_not_awaited()
        submitted["generation_complete_callback"]("cancelled")
        submitted["completion_future"].set_result("cancelled")
        await asyncio.sleep(0)

    async def test_failed_motion_entry_cancels_its_owned_turn(self):
        self.phase = "none"
        failure = asyncio.Event()
        watching = asyncio.Event()
        async def wait_failure(generation_id):
            self.assertEqual(generation_id, 1)
            watching.set()
            await failure.wait()
            return {"status": "failed", "error": "decoder unavailable"}
        self.track.wait_for_motion_entry_failure = wait_failure
        self.track.push_bgr_frames_batch = AsyncMock(return_value=False)
        response = await self.start_request()
        request_id = response["request_id"]
        token = self.session.stream_cancel_event
        audio = self.session.audio_player
        submitted = self.scheduler.submit_webrtc_stream.call_args.kwargs
        await asyncio.to_thread(submitted["frame_batch_callback"], [object()], 1, 20)
        await asyncio.wait_for(watching.wait(), 2)
        failure.set()
        for _ in range(100):
            if self.session.stream_owner is None:
                break
            await asyncio.sleep(.01)
        self.assertTrue(token.is_set())
        self.assertIsNone(self.session.stream_owner)
        audio.stop.assert_called()
        self.track.end_live.assert_called()
        self.transport.arm_source.assert_not_called()
        submitted["completion_future"].set_result("cancelled")
        await asyncio.sleep(0)
        self.assertNotIn(request_id, self.manager.active_requests)

    async def test_late_generation_complete_does_not_mark_new_turn_finished(self):
        self.phase = "none"
        self.track.signal_generation_complete = Mock()
        await self.start_request()
        old_owner = self.session.stream_owner
        submitted = self.scheduler.submit_webrtc_stream.call_args.kwargs
        await self.sessions.finish_reserved_stream(self.session, old_owner)
        await self.sessions.reserve_stream(self.session, "new-owner")
        token = self.session.stream_cancel_event
        audio = self.audio_class()
        self.session.audio_player = audio
        submitted["generation_complete_callback"]("cancelled")
        await asyncio.sleep(0)
        self.track.signal_generation_complete.assert_not_called()
        self.track.end_live.assert_not_called()
        audio.stop.assert_not_called()
        self.assertFalse(token.is_set())
        self.assertEqual(self.session.stream_owner, "new-owner")
        submitted["completion_future"].set_result("cancelled")

    async def test_cancelled_release_cannot_arm_after_audio_prepare(self):
        self.phase = "audio"
        await self.sessions.reserve_stream(self.session, "request")
        token = self.session.stream_cancel_event
        task = asyncio.create_task(self.ns["_release_webrtc_playout"](
            self.audio_class(), self.transport, self.track, None, None, "request", 0, "test",
            session=self.session, cancel_event=token))
        await asyncio.wait_for(self.entered.wait(), 2)
        token.set()
        self.release.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            await task
        self.transport.arm_source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
