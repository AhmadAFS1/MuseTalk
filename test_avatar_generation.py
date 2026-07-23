import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.avatar_generation import (
    AvatarGenerationError,
    GenerateAvatarRequest,
    generate_avatar_assets,
    generate_kling_motion,
    load_motion_reference_preset,
)


class AvatarGenerationTest(unittest.TestCase):
    def test_segmind_uses_only_x_api_key_header(self):
        response = mock.Mock()
        response.status_code = 200
        response.headers = {"content-type": "video/mp4"}
        response.content = b"ftyp-test-video"

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "motion.mp4"
            with mock.patch.dict(os.environ, {"SEGMIND_API_KEY": "test-key"}, clear=False):
                with mock.patch("scripts.avatar_generation.requests.post", return_value=response) as post:
                    generate_kling_motion(
                        "https://example.test/avatar.jpg",
                        output_path,
                        motion_reference_video_url="https://example.test/reference.mp4",
                    )

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers, {"x-api-key": "test-key"})
        self.assertNotIn("Authorization", headers)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["image_url"], "https://example.test/avatar.jpg")
        self.assertEqual(payload["video_url"], "https://example.test/reference.mp4")
        self.assertNotIn("image", payload)
        self.assertNotIn("input_video", payload)

    def test_segmind_accepts_explicit_motion_controls(self):
        response = mock.Mock()
        response.status_code = 200
        response.headers = {"content-type": "video/mp4"}
        response.content = b"ftyp-test-video"

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "motion.mp4"
            with mock.patch.dict(os.environ, {"SEGMIND_API_KEY": "test-key"}, clear=False):
                with mock.patch("scripts.avatar_generation.requests.post", return_value=response) as post:
                    generate_kling_motion(
                        "https://example.test/avatar.jpg",
                        output_path,
                        motion_reference_video_url="https://example.test/reference.mp4",
                        keep_original_sound=False,
                        character_orientation="image",
                    )

        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["keep_original_sound"])
        self.assertEqual(payload["character_orientation"], "image")

    def test_motion_reference_preset_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"SEGMIND_MOTION_REFERENCE_PRESET_DIR": tmp},
                clear=False,
            ):
                with self.assertRaises(AvatarGenerationError) as context:
                    load_motion_reference_preset("missing")
        self.assertEqual(context.exception.code, "motion_reference_preset_not_found")

    def test_generate_avatar_assets_generates_every_preset_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            preset_dir = temp_root / "presets" / "test_set"
            preset_dir.mkdir(parents=True)
            poses = ["neutral_resting", "nod_agree", "look_away_reset"]
            for pose_id in poses:
                (preset_dir / f"{pose_id}.mp4").write_bytes(b"reference-video")
            (preset_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "id": "test_set",
                        "default_pose_id": "neutral_resting",
                        "character_orientation": "video",
                        "keep_original_sound": False,
                        "prompt": "Transfer the reference motion.",
                        "poses": [
                            {"id": pose_id, "path": f"{pose_id}.mp4", "role": "idle"}
                            for pose_id in poses
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_generate_still(prompt, output_path, provider_order):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"jpeg")
                return {"provider": "test-image", "model": "test-model"}

            def fake_generate_motion(still_url, output_path, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"raw-video")
                return {"provider": "test-segmind", "source": "bytes"}

            def fake_normalize(input_path, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"normalized-video")

            def fake_upload(path, bucket, key, content_type):
                return f"https://{bucket}.example.test/{key}"

            env = {
                "GENERATED_AVATAR_ASSET_DIR": str(temp_root / "generated"),
                "SEGMIND_MOTION_REFERENCE_PRESET_DIR": str(temp_root / "presets"),
            }
            request = GenerateAvatarRequest(
                avatar_id="preset-avatar",
                prompt="A centered FaceTime avatar",
                prepare=False,
                motion_reference_preset="test_set",
            )
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch(
                    "scripts.avatar_generation.generate_still_image",
                    side_effect=fake_generate_still,
                ), mock.patch(
                    "scripts.avatar_generation.generate_kling_motion",
                    side_effect=fake_generate_motion,
                ) as generate_motion, mock.patch(
                    "scripts.avatar_generation.normalize_video",
                    side_effect=fake_normalize,
                ), mock.patch(
                    "scripts.avatar_generation._upload_file_to_s3",
                    side_effect=fake_upload,
                ):
                    result = generate_avatar_assets(request)

        self.assertEqual(generate_motion.call_count, 3)
        self.assertEqual(result["motion_reference_preset"], "test_set")
        self.assertEqual(result["default_pose_id"], "neutral_resting")
        self.assertEqual(set(result["motion_videos"]), set(poses))
        self.assertEqual(
            result["motion_video_path"],
            result["motion_videos"]["neutral_resting"]["normalized_path"],
        )
        for call in generate_motion.call_args_list:
            self.assertFalse(call.kwargs["keep_original_sound"])
            self.assertEqual(call.kwargs["character_orientation"], "video")


if __name__ == "__main__":
    unittest.main()
