"""Browser reply preflight ordering against the actual session manager."""
import unittest

import test_webrtc_pose_runtime as runtime


class PoseLabReplyPreflightTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        runtime.PoseSessionManagerTest.setUp(self)
        self.track.motion_bank = object()

    async def event(self, name, turn, seq):
        return await self.manager.handle_pose_event(
            self.session, {"event": name, "turn_id": turn, "seq": seq})

    async def late_upload(self, *, plan):
        reserved, _ = await self.manager.reserve_stream(self.session, "late-upload-A")
        self.assertTrue(reserved)
        if plan:
            result = await self.manager.stage_pose_plan(
                self.session,
                {"version": 2, "clock": "audio_progress", "segments": [
                    {"at_permille": 0, "pose_id": "speaking_direct"},
                    {"at_permille": 350, "pose_id": "light_smile"},
                    {"at_permille": 700, "pose_id": "speaking_direct"}]},
                seq=2, turn_id="reply-A")
        else:
            result = await self.manager.queue_pose_sequence(
                self.session, ["speaking_direct", "neutral_resting"],
                seq=2, turn_id="reply-A")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "stale_seq")
        await self.manager.finish_reserved_stream(
            self.session, "late-upload-A", turn_id="reply-A", recover_pose=False)

    async def test_direct_abort_fences_unreserved_late_upload_for_both_plans(self):
        for plan in (False, True):
            with self.subTest(plan_v2=plan):
                self.setUp()
                preflight = await self.event("assistant_thinking", "reply-A", 1)
                self.assertTrue(preflight["accepted"])
                self.assertEqual(self.session.active_turn_id, "reply-A")
                self.assertIsNone(self.session.stream_owner)
                aborted = await self.event("assistant_turn_aborted", "reply-A", 3)
                self.assertTrue(aborted["accepted"])
                await self.late_upload(plan=plan)
                self.assertFalse(self.session.assistant_active)
                self.assertIsNone(self.session.active_stream)
                self.assertEqual(self.session.last_pose_seq, 3)

    async def test_user_b_fences_unreserved_a_and_old_abort_cannot_clear_b(self):
        for plan in (False, True):
            with self.subTest(plan_v2=plan):
                self.setUp()
                self.assertTrue((await self.event("assistant_thinking", "reply-A", 1))["accepted"])
                self.assertTrue((await self.event("user_speech_started", "user-B", 3))["accepted"])
                self.assertEqual(self.session.active_turn_id, "user-B")
                rejected = await self.event("assistant_turn_aborted", "reply-A", 4)
                self.assertFalse(rejected["accepted"])
                self.assertEqual(rejected["reason"], "turn_mismatch")
                await self.late_upload(plan=plan)
                self.assertTrue(self.session.user_speaking)
                self.assertEqual(self.session.active_turn_id, "user-B")
                self.assertEqual(self.session.last_pose_seq, 3)

    async def test_reserved_reply_survives_user_b_identity_until_owned_abort(self):
        self.assertTrue((await self.event("assistant_thinking", "reply-A", 1))["accepted"])
        reserved, _ = await self.manager.reserve_stream(self.session, "request-A")
        self.assertTrue(reserved)
        staged = await self.manager.queue_pose_sequence(
            self.session, ["speaking_direct", "neutral_resting"], seq=2, turn_id="reply-A")
        self.assertTrue(staged["accepted"])
        self.assertTrue((await self.event("user_speech_started", "user-B", 3))["accepted"])
        self.assertEqual(self.session.active_turn_id, "reply-A")
        self.assertTrue((await self.event("assistant_turn_aborted", "reply-A", 4))["accepted"])
        self.assertFalse(self.session.assistant_active)
        self.assertTrue(self.session.user_speaking)


if __name__ == "__main__":
    unittest.main()
