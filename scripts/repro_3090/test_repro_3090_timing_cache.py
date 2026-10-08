"""CPU-only timing-cache provenance tests; no TensorRT/GPU imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("timing_policy", Path(__file__).resolve().parents[1] / "trt_timing_cache.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class TimingCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache, self.seed = self.root / "timing.bin", self.root / "4070.bin"
        self.seed.write_bytes(b"old-4070-timings")
        self.base = {"gpu": "NVIDIA GeForce RTX 3090", "compute_capability": [8, 6], "tensorrt_version": "10.3.0",
                     "torch_version": "2.5.1+cu121", "cuda_version": "12.1",
                     "build_flags": {"builder_optimization_level": 5, "workspace_gb": 2.0}}
        self.target = policy.fingerprint(self.base)

    def select(self, strict=True):
        return policy.select(self.cache, self.seed, self.target, strict=strict, allow_seed=True)

    def test_strict_fresh_root_never_reads_4070_seed(self):
        data, provenance = self.select()
        self.assertIsNone(data)
        self.assertEqual(provenance, {"source": "empty", "strict": True})

    def test_legacy_seed_behavior_preserved(self):
        data, provenance = self.select(False)
        self.assertEqual(data, self.seed.read_bytes())
        self.assertEqual(provenance["source"], "legacy_4070_seed")

    def test_unlabelled_cache_is_rejected(self):
        self.cache.write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.select()

    def test_same_target_cache_round_trip(self):
        self.cache.write_bytes(b"native-timings")
        policy.record(self.cache, self.cache.read_bytes(), self.target)
        self.assertEqual(self.select()[0], b"native-timings")

    def test_cache_hash_and_target_must_match(self):
        self.cache.write_bytes(b"native-timings")
        policy.record(self.cache, self.cache.read_bytes(), self.target)
        self.cache.write_bytes(b"foreign-timings")
        with self.assertRaisesRegex(ValueError, "target/hash mismatch"):
            self.select()
        policy.record(self.cache, self.cache.read_bytes(), {**self.target, "gpu": "NVIDIA GeForce RTX 4070 SUPER"})
        with self.assertRaisesRegex(ValueError, "target/hash mismatch"):
            self.select()

    def test_existing_engine_manifest_cannot_be_relabelled(self):
        for key, wrong in (("gpu", "NVIDIA GeForce RTX 4070 SUPER"), ("compute_capability", [8, 9]),
                           ("tensorrt_version", "10.2.0"), ("cuda_version", "11.8")):
            prior = {**copy.deepcopy(self.base), "blocks": {"tail": {}}, key: wrong}
            with self.assertRaisesRegex(ValueError, key):
                policy.validate_resume(prior, self.base, True)
        policy.validate_resume({**self.base, "blocks": {"tail": {}}}, self.base, True)
        policy.validate_resume({}, self.base, True)


if __name__ == "__main__":
    unittest.main()
