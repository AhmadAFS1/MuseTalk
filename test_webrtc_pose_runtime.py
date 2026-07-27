import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_media_stubs_if_needed():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        numpy = types.ModuleType("numpy")
        numpy.float32 = "float32"
        numpy.uint8 = "uint8"
        numpy.clip = lambda value, _low, _high: value
        sys.modules["numpy"] = numpy

    try:
        import aiortc  # noqa: F401
    except ModuleNotFoundError:
        aiortc = types.ModuleType("aiortc")

        class _Track:
            def __init__(self):
                self.readyState = "live"

            def stop(self):
                self.readyState = "ended"

        class _PeerConnection:
            def __init__(self, _configuration=None):
                self.connectionState = "new"

            def on(self, _event):
                return lambda callback: callback

            async def close(self):
                self.connectionState = "closed"

        class _Configuration:
            def __init__(self, iceServers=None):
                self.iceServers = iceServers or []

        class _IceServer:
            def __init__(self, urls=None, username=None, credential=None):
                self.urls = urls
                self.username = username
                self.credential = credential

        aiortc.VideoStreamTrack = _Track
        aiortc.MediaStreamTrack = _Track
        aiortc.RTCPeerConnection = _PeerConnection
        aiortc.RTCConfiguration = _Configuration
        aiortc.RTCIceServer = _IceServer
        aiortc.RTCRtpSender = object
        sys.modules["aiortc"] = aiortc

        mediastreams = types.ModuleType("aiortc.mediastreams")
        mediastreams.MediaStreamError = RuntimeError
        sys.modules["aiortc.mediastreams"] = mediastreams

    try:
        import av  # noqa: F401
    except ModuleNotFoundError:
        av = types.ModuleType("av")

        class _Frame:
            @classmethod
            def from_ndarray(cls, _array, format=None):
                return cls()

            def reformat(self, *args, **kwargs):
                return self

        av.VideoFrame = _Frame
        av.AudioFrame = _Frame
        av.open = lambda _path: None
        sys.modules["av"] = av


_install_media_stubs_if_needed()

from scripts.pose_protocol import POSE_IDS, POSE_SWITCH_MODE, normalize_pose_set
from scripts import webrtc_tracks
from scripts.webrtc_manager import WebRTCSession, WebRTCSessionManager
from scripts.webrtc_pose_router import LivePoseVideoRouter


def _pose_set():
    return normalize_pose_set(
        {
            "version": 1,
            "pose_set_id": "test_six",
            "default_pose_id": "neutral_resting",
            "switch_mode": POSE_SWITCH_MODE,
            "poses": {
                pose_id: {"avatar_id": f"avatar_{pose_id}"}
                for pose_id in POSE_IDS
            },
        }
    )


class _FakeFrame:
    def __init__(self, path, index):
        self.path = path
        self.index = index
        self.pts = None
        self.time_base = None

    def reformat(self, *args, **kwargs):
        return self


class _FakeIdleTrack:
    frame_count = 3
    instances = []

    def __init__(self, video_path, fps=None):
        self.video_path = video_path
        self.fps = fps or 30
        self.next_index = 0
        self.last_index = None
        self.started_cycle = False
        self.completed_cycles = 0
        self.stopped = False
        self.__class__.instances.append(self)

    def next_frame_starts_cycle(self):
        return self.last_index is not None and self.next_index >= self.frame_count

    def last_read_started_cycle(self):
        return self.started_cycle

    def read_frame(self):
        self.started_cycle = False
        if self.next_index >= self.frame_count:
            self.next_index = 0
            self.started_cycle = True
            self.completed_cycles += 1
        index = self.next_index
        self.last_index = index
        self.next_index += 1
        return _FakeFrame(self.video_path, index)

    def get_timing(self):
        return {
            "source_frame_index": self.last_index or 0,
            "source_frame_count": self.frame_count,
            "source_fps": self.fps,
            "completed_cycles": self.completed_cycles,
        }

    def stop(self):
        self.stopped = True


class _ManagerTrack:
    def __init__(self):
        self.current_pose = "neutral_resting"
        self.current_path = "/poses/neutral_resting.mp4"
        self.pending = []
        self.calls = []

    async def switch_idle_video(
        self,
        path,
        transition_seconds=0,
        *,
        pose_id=None,
        effective="immediate",
        reason="manual",
        replace_pending=False,
    ):
        if replace_pending:
            self.pending.clear()
        self.calls.append((pose_id, effective, reason, replace_pending))
        if effective == POSE_SWITCH_MODE:
            if not self.pending or self.pending[-1] != pose_id:
                self.pending.append(pose_id)
            return {"queued": True, "changed": False}
        self.current_pose = pose_id
        self.current_path = path
        return {"queued": False, "changed": True}

    def get_pose_status(self):
        return {
            "current_pose_id": self.current_pose,
            "current_idle_video_path": self.current_path,
            "pending_pose_ids": list(self.pending),
        }


class SwitchablePoseBoundaryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        webrtc_tracks.IdleVideoStreamTrack = _FakeIdleTrack
        _FakeIdleTrack.instances.clear()

    def tearDown(self):
        webrtc_tracks.IdleVideoStreamTrack = self.original_idle_track

    async def test_next_boundary_switch_does_not_cut_current_cycle(self):
        activated = []
        track = webrtc_tracks.SwitchableVideoStreamTrack(
            "neutral.mp4",
            source_fps=30,
            output_fps=30,
            prebuffer_seconds=0,
            adaptive_fps=False,
            idle_pose_id="neutral_resting",
            on_idle_pose_changed=lambda pose_id, path: activated.append((pose_id, path)),
        )

        first = track._advance_idle_frame(1)
        result = await track.queue_idle_video(
            "listening.mp4",
            pose_id="active_listening",
        )
        second = track._advance_idle_frame(1)
        third = track._advance_idle_frame(1)

        self.assertEqual((first.path, first.index), ("neutral.mp4", 0))
        self.assertEqual((second.path, second.index), ("neutral.mp4", 1))
        self.assertEqual((third.path, third.index), ("neutral.mp4", 2))
        self.assertTrue(result["queued"])
        self.assertEqual(track.get_pose_status()["pending_pose_ids"], ["active_listening"])
        self.assertEqual(activated, [])

        boundary = track._advance_idle_frame(1)
        self.assertEqual((boundary.path, boundary.index), ("listening.mp4", 0))
        self.assertEqual(track.get_pose_status()["current_pose_id"], "active_listening")
        self.assertEqual(track.get_pose_status()["pending_pose_ids"], [])
        self.assertEqual(activated, [("active_listening", "listening.mp4")])

    async def test_legacy_switch_remains_immediate(self):
        track = webrtc_tracks.SwitchableVideoStreamTrack(
            "neutral.mp4",
            source_fps=30,
            output_fps=30,
            prebuffer_seconds=0,
            adaptive_fps=False,
        )
        result = await track.switch_idle_video(
            "other.mp4",
            transition_seconds=0,
            pose_id="other",
        )
        self.assertTrue(result["changed"])
        self.assertEqual(track.get_pose_status()["current_idle_video_path"], "other.mp4")
        self.assertEqual(track.get_pose_status()["pending_pose_ids"], [])

    async def test_idle_and_live_sources_keep_independent_frame_rates(self):
        track = webrtc_tracks.SwitchableVideoStreamTrack(
            "neutral.mp4",
            source_fps=20,
            idle_source_fps=30,
            output_fps=20,
            prebuffer_seconds=0,
            adaptive_fps=False,
        )

        self.assertEqual(track._source_fps, 20)
        self.assertEqual(track._idle_source_fps, 30)
        self.assertEqual(track._output_fps, 20)
        self.assertEqual(track._idle.fps, 30)
        self.assertEqual(
            [track._advance_source() for _ in range(4)],
            [1, 2, 1, 2],
        )

        await track.queue_idle_video(
            "listening.mp4",
            pose_id="active_listening",
        )
        self.assertEqual(_FakeIdleTrack.instances[-1].fps, 30)

        track._idle.frame_count = 300
        track._idle.next_index = 151
        track._idle.last_index = 150
        timing = track.capture_idle_sync_timing(
            generation_fps=20,
            cycle_frames=400,
            hold=False,
        )
        self.assertEqual(timing["target_source_frame_index"], 150)
        self.assertEqual(timing["offset_frames"], 100)
        self.assertEqual(timing["idle_phase_seconds"], 5.0)


class PoseSessionManagerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pose_set = _pose_set()
        paths = {pose_id: f"/poses/{pose_id}.mp4" for pose_id in POSE_IDS}
        self.track = _ManagerTrack()
        self.session = WebRTCSession(
            session_id="session",
            avatar_id="legacy_avatar",
            idle_track=self.track,
            idle_video_path=paths["neutral_resting"],
            idle_pose_id="neutral_resting",
            pose_protocol_enabled=True,
            pose_set=pose_set,
            pose_switch_mode=POSE_SWITCH_MODE,
            pose_video_paths=paths,
            current_pose_id="neutral_resting",
            generation_avatar_id="avatar_speaking_direct",
        )
        self.manager = WebRTCSessionManager()
        self.manager.sessions[self.session.session_id] = self.session

    async def test_events_are_monotonic_and_reaction_is_once_per_turn(self):
        started = await self.manager.handle_pose_event(
            self.session,
            {
                "event": "user_speech_started",
                "turn_id": "turn_1",
                "seq": 1,
            },
        )
        stale = await self.manager.handle_pose_event(
            self.session,
            {
                "event": "user_speech_ended",
                "turn_id": "turn_1",
                "seq": 1,
            },
        )
        reaction = await self.manager.handle_pose_event(
            self.session,
            {
                "event": "assistant_reaction_ready",
                "reaction_intent": "warmth",
                "turn_id": "turn_1",
                "seq": 2,
            },
        )
        duplicate = await self.manager.handle_pose_event(
            self.session,
            {
                "event": "assistant_reaction_ready",
                "reaction_intent": "acknowledge",
                "turn_id": "turn_1",
                "seq": 3,
            },
        )

        self.assertTrue(started["accepted"])
        self.assertFalse(stale["accepted"])
        self.assertEqual(stale["reason"], "stale_seq")
        self.assertEqual(reaction["switch"]["pose_id"], "light_smile")
        self.assertTrue(duplicate["accepted"])
        self.assertTrue(duplicate["deduped"])
        self.assertEqual(
            [call[0] for call in self.track.calls],
            ["active_listening", "light_smile"],
        )

    async def test_speech_generation_resolves_speaking_pose_cache(self):
        self.assertEqual(
            self.session.avatar_id_for_pose("speaking_direct"),
            "avatar_speaking_direct",
        )
        self.assertEqual(
            self.session.generation_avatar_id,
            "avatar_speaking_direct",
        )

        result = await self.manager.queue_pose_sequence(
            self.session,
            ["speaking_direct", "neutral_resting"],
            seq=1,
            turn_id="turn_2",
            reaction_intent="none",
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["generation_avatar_id"], "avatar_speaking_direct")
        self.assertEqual(
            result["pose_status"]["queued_pose_ids"],
            ["speaking_direct", "neutral_resting"],
        )

    async def test_rendered_pose_trace_coalesces_adjacent_batches(self):
        self.session.record_rendered_pose_batch(
            ["light_smile", "light_smile", "speaking_direct"],
            0,
        )
        self.session.record_rendered_pose_batch(
            ["speaking_direct", "neutral_resting"],
            3,
        )

        status = self.session.rendered_pose_status()
        self.assertEqual(status["rendered_pose_id"], "neutral_resting")
        self.assertEqual(status["rendered_pose_frame_count"], 5)
        self.assertEqual(
            status["rendered_pose_trace"],
            [
                {
                    "pose_id": "light_smile",
                    "start_frame_index": 0,
                    "end_frame_index": 1,
                    "frame_count": 2,
                },
                {
                    "pose_id": "speaking_direct",
                    "start_frame_index": 2,
                    "end_frame_index": 3,
                    "frame_count": 2,
                },
                {
                    "pose_id": "neutral_resting",
                    "start_frame_index": 4,
                    "end_frame_index": 4,
                    "frame_count": 1,
                },
            ],
        )


class _RouterDecoder:
    fps = 30.0
    frame_count = 300

    def read_frames(self, frame_indices):
        return list(frame_indices)

    def close(self):
        pass


class LivePoseRouterAlignmentTest(unittest.TestCase):
    def test_first_pose_phase_alignment_shortens_only_first_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for pose_id in ("light_smile", "speaking_direct"):
                path = root / f"{pose_id}.mp4"
                path.touch()
                paths[pose_id] = str(path)
            router = LivePoseVideoRouter(
                paths,
                prepared_pose_ids=set(paths),
                initial_pose_id="light_smile",
                decoder_factory=lambda _path: _RouterDecoder(),
            )
            router.queue_pose_sequence(
                ["light_smile", "speaking_direct"],
                20,
            )

            aligned = router.align_first_queued_pose(150, 20)

            self.assertEqual(
                aligned,
                [
                    {
                        "pose_id": "light_smile",
                        "start_generation_frame": 0,
                        "end_generation_frame": 100,
                        "source_frame_offset": 150,
                    },
                    {
                        "pose_id": "speaking_direct",
                        "start_generation_frame": 100,
                        "end_generation_frame": 300,
                        "source_frame_offset": 0,
                    },
                ],
            )
            first = router.snapshot(0, 20)
            last_first = router.snapshot(99, 20)
            second = router.snapshot(100, 20)
            self.assertEqual(router.source_frame_index(first, 0), 150)
            self.assertEqual(router.source_frame_index(last_first, 99), 298)
            self.assertEqual(router.source_frame_index(second, 100), 0)
            router.close()


if __name__ == "__main__":
    unittest.main()
