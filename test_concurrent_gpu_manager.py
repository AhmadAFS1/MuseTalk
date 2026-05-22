import os
import unittest
from unittest import mock

from scripts.concurrent_gpu_manager import (
    GPUMemoryManager,
    recommended_scheduler_batch_config,
)


class GPUMemoryManagerTest(unittest.TestCase):
    def test_env_total_memory_drives_budget_and_batch_estimates(self):
        env = {
            "GPU_TOTAL_MEMORY_GB": "32",
            "GPU_RESERVED_MEMORY_GB": "8",
            "GPU_MEMORY_BATCH_GB": "32:21.5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            manager = GPUMemoryManager(max_live_generations=2)
            stats = manager.get_stats()

            self.assertEqual(stats["total_gb"], 32.0)
            self.assertEqual(stats["reserved_gb"], 8.0)
            self.assertEqual(stats["available_gb"], 24.0)
            self.assertEqual(stats["source"], "env")
            self.assertEqual(stats["batch_memory_gb"][32], 21.5)

            with manager.allocate(32):
                self.assertAlmostEqual(manager.get_stats()["current_usage_gb"], 21.5)

            self.assertEqual(manager.get_stats()["current_usage_gb"], 0.0)

    def test_recommended_scheduler_config_preserves_24gb_compatibility(self):
        config = recommended_scheduler_batch_config(24, profile="throughput_record")

        self.assertEqual(config["max_combined_batch_size"], 16)
        self.assertEqual(config["fixed_batch_sizes"], [8, 16])
        self.assertEqual(config["warmup_batches"], [8, 16])

    def test_recommended_scheduler_config_uses_32gb_headroom(self):
        config = recommended_scheduler_batch_config(32, profile="throughput_record")

        self.assertEqual(config["max_combined_batch_size"], 32)
        self.assertEqual(config["fixed_batch_sizes"], [4, 8, 16, 32])
        self.assertEqual(config["warmup_batches"], [4, 8, 16, 32])

    def test_baseline_stays_conservative_on_large_gpus(self):
        config = recommended_scheduler_batch_config(80, profile="baseline")

        self.assertEqual(config["max_combined_batch_size"], 4)
        self.assertEqual(config["fixed_batch_sizes"], [4])


if __name__ == "__main__":
    unittest.main()
