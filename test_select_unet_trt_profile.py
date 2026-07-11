import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_unet_trt_profile import build_profile


class SelectUnetTrtProfileTest(unittest.TestCase):
    def test_split8_matches_restored_artifact_and_batch8_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "models/tensorrt_unet_static_bs8_20260529"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "unet_trt.ts").write_bytes(b"test-engine")
            (artifact_dir / "unet_trt_meta.json").write_text(
                json.dumps({"validation": {"passed": True}})
            )

            values, notes = build_profile(
                argparse.Namespace(
                    repo_root=root,
                    bs8_dir="models/tensorrt_unet_static_bs8_20260529",
                    bs16_dir="models/tensorrt_unet_static_bs16_20260704",
                    prefer="split8",
                )
            )

        self.assertEqual(
            values["MUSETALK_TRT_UNET_PATHS"],
            "8:models/tensorrt_unet_static_bs8_20260529/unet_trt.ts",
        )
        self.assertEqual(values["MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"], "8")
        self.assertEqual(values["HLS_SCHEDULER_MAX_BATCH"], "8")
        self.assertEqual(values["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "8")
        self.assertEqual(values["HLS_SCHEDULER_STARTUP_SLICE_SIZE"], "8")
        self.assertIn("selected: validated static batch-8 split8 profile", notes)


if __name__ == "__main__":
    unittest.main()
