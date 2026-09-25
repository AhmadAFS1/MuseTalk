import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.motion_transitions import IDLE, TALK, SMILE, configured_bank, file_hash
from scripts.webrtc_tracks import SwitchableVideoStreamTrack
from test_motion_transitions import fixture
from character_factory.scripts.build_realtime_character import adapt_subject, make_pose_set


class RegistryTest(unittest.TestCase):
    def test_registry_selects_each_identity_by_content_and_requires_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path_sets = []
            for name in ("first", "second"):
                folder = root/name
                folder.mkdir()
                manifest = fixture()
                manifest["status"] = "reviewed"
                paths = {}
                for pose in (IDLE, TALK, SMILE):
                    path = folder/(pose+".mp4")
                    path.write_bytes((name+pose).encode())
                    manifest["sources"][pose].update(path=str(path), sha256=file_hash(path))
                    paths[pose] = str(path)
                (folder/"motion-atlas.json").write_text(json.dumps(manifest))
                path_sets.append(paths)
            env = {"WEBRTC_MOTION_ATLAS_DIR": str(root), "WEBRTC_MOTION_ATLAS": "",
                   "WEBRTC_MOTION_ALLOW_UNREVIEWED": "0"}
            with patch.dict(os.environ, env):
                for paths in path_sets:
                    self.assertTrue(configured_bank(paths).compatible(paths))
                self.assertIsNone(configured_bank({}))
                mixed = dict(path_sets[0]); mixed[TALK] = path_sets[1][TALK]
                self.assertIsNone(configured_bank(mixed))
                file = root/"first/motion-atlas.json"
                manifest = json.loads(file.read_text())
                manifest["status"] = "candidate_requires_recorded_review"
                file.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "recorded review"):
                    configured_bank(path_sets[0])
                with patch.dict(os.environ, {"WEBRTC_MOTION_ALLOW_UNREVIEWED": "1"}):
                    self.assertIsNotNone(configured_bank(path_sets[0]))

    def test_public_pose_set_has_only_three_hash_named_physical_caches(self):
        atlas = fixture()
        poses = make_pose_set("character", atlas)["poses"]
        self.assertEqual(len(poses), 6)
        self.assertEqual(len({v["avatar_id"] for v in poses.values()}), 3)
        self.assertEqual(poses["active_listening"]["avatar_id"], poses[IDLE]["avatar_id"])

    def test_identity_adaptation_preserves_negative_and_motion_constraints(self):
        pack = {"pack_id": "test", "poses": {"idle": {"positive_prompt": "She stays seated. Her head remains level. The same female tutor.", "negative_prompt": "head bobbing, zoom"}}}
        adapted = adapt_subject(pack, "man")
        self.assertEqual(adapted["poses"]["idle"]["negative_prompt"], pack["poses"]["idle"]["negative_prompt"])
        self.assertIn("He stays seated. His head remains level.", adapted["poses"]["idle"]["positive_prompt"])
        self.assertEqual(adapted["approval_status"], "identity_adaptation_requires_review")
        self.assertIn("She", pack["poses"]["idle"]["positive_prompt"])


class MetadataTest(unittest.TestCase):
    def test_small_audio_lead_keeps_motion_video_rtp_contiguous(self):
        import types
        from unittest.mock import Mock
        for audio_target, expected in ((3.02,60), (3.08,62)):
            track = SwitchableVideoStreamTrack.__new__(SwitchableVideoStreamTrack)
            track.motion_bank = object()
            track._output_fps = 20
            track._rtp_frame_index = 60
            track._live_rtp_alignment_applied = False
            track._sync_clock = types.SimpleNamespace(
                audio_transport_next_pts_seconds=audio_target,
                note_first_live_rtp_alignment=Mock(), request_audio_transport_rebase=Mock())
            track._align_first_live_rtp_to_audio()
            self.assertEqual(track._rtp_frame_index, expected)
            self.assertLessEqual(abs(expected/20-audio_target), .05)

    def test_stale_generation_cannot_overwrite_last_popped_source(self):
        track = SwitchableVideoStreamTrack.__new__(SwitchableVideoStreamTrack)
        track._live_generation_id = 4
        track._frames_dropped = 0
        track._popped_motion = {"source_frame": 11}
        self.assertIsNone(track._unwrap_live_queue_item(("live_frame",3,object(),{"source_frame": 99})))
        self.assertEqual(track._popped_motion, {"source_frame": 11})
        frame = object()
        self.assertIs(track._unwrap_live_queue_item(("live_frame",4,frame,{"source_frame": 12})), frame)
        self.assertEqual(track._popped_motion, {"source_frame": 12})
        self.assertEqual(track._frames_dropped, 1)


class AudioPacingTest(unittest.IsolatedAsyncioTestCase):
    async def test_motion_audio_does_not_accumulate_small_scheduler_delays(self):
        from scripts.webrtc_tracks import SilenceAudioStreamTrack
        track = SilenceAudioStreamTrack(steady_pacing=True)
        track._transport_start_time = 100.0
        track._frames_sent = 10
        with patch("scripts.webrtc_tracks.time.monotonic", return_value=100.207):
            await track._pace()
        self.assertEqual(track._transport_start_time, 100.0)
        self.assertEqual(track._pace_reanchors, 0)
        track.stop()


if __name__ == "__main__": unittest.main()
