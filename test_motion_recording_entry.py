"""Proof parsing for the live harness's pre-speech interruption case."""
import copy
import unittest

from scripts.test_webrtc_motion_transitions import validate_entry_abort_candidate, validate_cancelled_entry


class EntryRecordingProofTest(unittest.TestCase):
    def setUp(self):
        self.status = {"track_stats": {"sync_clock": {
            "started": False, "first_live_video_rtp_seconds": None,
            "first_tts_transport_pts_seconds": None, "first_audio_packet_unix_ms": None}}}
        row = {"mode": "speech_entry_body_bridge", "progress": .333, "output_frame": 101}
        self.motion = {"last_emitted": row,
                       "trace": [{"mode": "live", "output_frame": 50}, row],
                       "entries": [{"generation_id": 3, "status": "playing", "frames_emitted": 2}]}

    def test_late_smile_fixture_satisfies_public_pose_plan_contract(self):
        from scripts.test_webrtc_motion_transitions import speech_segments
        from scripts.pose_protocol import normalize_pose_plan
        from scripts.motion_transitions import MotionBank, IDLE, TALK, SMILE
        from test_motion_transitions import fixture
        plan = normalize_pose_plan({"version": 2, "clock": "audio_progress",
            "segments": speech_segments("late"), "switch_mode": "next_boundary", "on_complete": IDLE})
        compiled = MotionBank(fixture()).plan(238, 20, IDLE, 0, plan["segments"])
        self.assertEqual([s["to"] for s in compiled["switches"]], [TALK, IDLE])
        self.assertLessEqual(compiled["switches"][-1]["frame"], 228)

    def test_actual_emitted_entry_before_audio_is_accepted(self):
        proof = validate_entry_abort_candidate(self.status, self.motion, 100)
        self.assertEqual(proof["generation_id"], 3)
        self.assertEqual(proof["displayed"]["output_frame"], 101)

    def test_completed_or_prepared_but_unemitted_entry_is_rejected(self):
        for change in ({"progress": 1}, {"mode": "idle"}, {"progress": 0}):
            motion = copy.deepcopy(self.motion)
            motion["last_emitted"].update(change)
            with self.assertRaises(AssertionError):
                validate_entry_abort_candidate(self.status, motion, 100)

    def test_any_current_turn_live_frame_or_audio_release_is_rejected(self):
        motion = copy.deepcopy(self.motion)
        motion["trace"].append({"mode": "live", "output_frame": 100})
        with self.assertRaises(AssertionError):
            validate_entry_abort_candidate(self.status, motion, 100)
        for key, value in (("started", True), ("first_live_video_rtp_seconds", 4),
                           ("first_tts_transport_pts_seconds", 4), ("first_audio_packet_unix_ms", 5)):
            status = copy.deepcopy(self.status)
            status["track_stats"]["sync_clock"][key] = value
            with self.assertRaises(AssertionError):
                validate_entry_abort_candidate(status, self.motion, 100)

    def test_cancelled_entry_must_belong_to_this_generation_and_never_release_live(self):
        self.motion["entries"][0]["status"] = "cancelled"
        entry = validate_cancelled_entry(self.motion, 100, 3)
        self.assertEqual(entry["frames_emitted"], 2)
        with self.assertRaises(AssertionError):
            validate_cancelled_entry(self.motion, 100, 4)
        for field, value in (("status", "completed"), ("frames_emitted", 0),
                             ("first_live_generation_frame", 0), ("first_live_output_frame", 102)):
            motion = copy.deepcopy(self.motion)
            motion["entries"][0][field] = value
            with self.assertRaises(AssertionError):
                validate_cancelled_entry(motion, 100, 3)
        self.motion["trace"].append({"mode": "live", "output_frame": 102})
        with self.assertRaises(AssertionError):
            validate_cancelled_entry(self.motion, 100, 3)


class RecorderTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_status_timeout_overrides_long_avatar_preparation_session_timeout(self):
        import asyncio
        import time
        import aiohttp
        from aiohttp import web
        from scripts.test_webrtc_motion_transitions import request_json
        gate = asyncio.Event()
        async def stalled_status(_request):
            await gate.wait()
            return web.json_response({"status": "unexpected"})
        app = web.Application()
        app.router.add_get("/status", stalled_status)
        runner = web.AppRunner(app, shutdown_timeout=.05)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        started = time.monotonic()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as http:
                with self.assertRaises(asyncio.TimeoutError):
                    await request_json(http, "GET", f"http://127.0.0.1:{port}/status",
                                       action="motion status", timeout_seconds=.04)
            self.assertLess(time.monotonic() - started, .5)
        finally:
            gate.set()
            await runner.cleanup()

    async def test_cleanup_timeout_retains_primary_failure_and_allows_next_cleanup(self):
        import asyncio
        from scripts.test_webrtc_motion_transitions import cleanup_step
        evidence = {"success": False, "error": {"type": "TimeoutError", "message": "status hung"}}
        async def never_returns():
            await asyncio.Event().wait()
        self.assertFalse(await cleanup_step("delete server session", never_returns(), evidence, .01))
        self.assertEqual(evidence["error"]["message"], "status hung")
        self.assertEqual(evidence["cleanup_errors"][0]["action"], "delete server session")
        self.assertEqual(evidence["cleanup_errors"][0]["type"], "TimeoutError")
        self.assertTrue(await cleanup_step("close receiver", asyncio.sleep(0), evidence, .01))
        self.assertFalse(evidence["success"])


if __name__ == "__main__":
    unittest.main()
