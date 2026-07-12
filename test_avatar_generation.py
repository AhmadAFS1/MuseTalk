import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.avatar_generation import generate_kling_motion


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


if __name__ == "__main__":
    unittest.main()
