"""Exercise real harness cleanup before recorder construction/start succeeds."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import test_webrtc_motion_transitions as harness


class MotionRecorderCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def _failure(self, stage):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url="http://unused", output=Path(directory), fps=20)
            poses = {"poses": {"neutral_resting": {"avatar_id": "avatar"}}}
            peer = SimpleNamespace(on=lambda _event: lambda callback: callback,
                addTransceiver=Mock(), close=AsyncMock())
            recorder = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
            error = harness.helper.SmokeTestError(f"failed {stage}")
            factory = Mock(return_value=recorder)
            exchange = AsyncMock()
            if stage == "constructor":
                factory.side_effect = error
            elif stage == "offer":
                exchange.side_effect = error
            elif stage == "start":
                recorder.start.side_effect = error
            deletion = AsyncMock()
            with patch.object(harness, "request_json", AsyncMock(return_value={"session_id": "owned"})), \
                    patch.object(harness.helper, "worker_pose_manifest", return_value=poses), \
                    patch.object(harness.helper, "build_rtc_configuration_from_payload", return_value=None), \
                    patch.object(harness.helper, "RTCPeerConnection", return_value=peer), \
                    patch.object(harness.helper, "RTPMP4Recorder", factory), \
                    patch.object(harness.helper, "exchange_offer", exchange), \
                    patch.object(harness.helper, "delete_webrtc_session", deletion):
                with self.assertRaisesRegex(harness.helper.SmokeTestError, f"failed {stage}"):
                    await harness.record_case(object(), args, poses, "case", [])
            peer.close.assert_awaited_once()
            deletion.assert_awaited_once()
            self.assertEqual(deletion.await_args.args[1:], ("http://unused", "owned"))
            if stage == "constructor":
                recorder.stop.assert_not_awaited()
            else:
                recorder.stop.assert_awaited_once()
            report = json.loads((Path(directory) / "case.json").read_text())
            self.assertFalse(report["success"])
            self.assertEqual(report["error"]["message"], f"failed {stage}")

    async def test_constructor_failure_still_closes_peer_and_deletes_session(self):
        await self._failure("constructor")

    async def test_offer_failure_stops_unstarted_recorder(self):
        await self._failure("offer")

    async def test_partial_start_failure_stops_recorder(self):
        await self._failure("start")


if __name__ == "__main__":
    unittest.main()
