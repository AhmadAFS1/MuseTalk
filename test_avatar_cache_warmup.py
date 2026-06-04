import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def _install_import_stubs():
    torch = types.ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def synchronize():
            return None

    torch.cuda = _FakeCuda()
    torch.float16 = "float16"
    torch.device = lambda value: value
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False, benchmark=False),
    )
    torch.set_float32_matmul_precision = lambda _value: None
    sys.modules["torch"] = torch

    face_parsing = types.ModuleType("musetalk.utils.face_parsing")
    face_parsing.FaceParsing = object
    sys.modules["musetalk.utils.face_parsing"] = face_parsing

    utils = types.ModuleType("musetalk.utils.utils")
    utils.load_all_model = lambda *args, **kwargs: (None, None, None)
    sys.modules["musetalk.utils.utils"] = utils

    audio_processor = types.ModuleType("musetalk.utils.audio_processor")
    audio_processor.AudioProcessor = object
    sys.modules["musetalk.utils.audio_processor"] = audio_processor

    transformers = types.ModuleType("transformers")
    transformers.WhisperModel = object
    sys.modules["transformers"] = transformers

    api_avatar = types.ModuleType("scripts.api_avatar")
    api_avatar.APIAvatar = object
    sys.modules["scripts.api_avatar"] = api_avatar

    gpu_manager = types.ModuleType("scripts.concurrent_gpu_manager")
    gpu_manager.GPUMemoryManager = object
    sys.modules["scripts.concurrent_gpu_manager"] = gpu_manager

    trt_runtime = types.ModuleType("scripts.trt_runtime")
    trt_runtime.load_unet_trt_backend = lambda *args, **kwargs: None
    trt_runtime.load_vae_trt_decoder = lambda *args, **kwargs: None
    sys.modules["scripts.trt_runtime"] = trt_runtime


_install_import_stubs()

from scripts.avatar_cache import AvatarCache
from scripts.avatar_manager_parallel import ParallelAvatarManager


class AvatarCacheWarmupTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.manager = ParallelAvatarManager.__new__(ParallelAvatarManager)
        self.manager.avatar_cache = AvatarCache(
            max_cached_avatars=0,
            ttl_seconds=3600,
            max_memory_mb=0,
            cleanup_interval=60,
        )
        self.manager.avatar_warm_executor = ThreadPoolExecutor(max_workers=2)
        self.manager.avatar_warmups = {}
        self.manager.avatar_warmups_lock = threading.Lock()
        self.manager.avatar_s3_store = SimpleNamespace(enabled=False)
        self.manager._avatar_latents_path = (
            lambda avatar_id: str(self.root / avatar_id / "latents.pt")
        )
        self.manager._avatar_exists = (
            lambda avatar_id: Path(self.manager._avatar_latents_path(avatar_id)).exists()
        )

    def tearDown(self):
        self.manager.avatar_warm_executor.shutdown(wait=True)
        self.manager.avatar_cache.clear_all()
        self.tmpdir.cleanup()

    def _prepare_avatar_file(self, avatar_id):
        latents_path = Path(self.manager._avatar_latents_path(avatar_id))
        latents_path.parent.mkdir(parents=True, exist_ok=True)
        latents_path.write_bytes(b"latents")

    def test_same_avatar_warmup_is_deduped(self):
        self._prepare_avatar_file("alice")
        calls = []
        load_started = threading.Event()
        release_load = threading.Event()

        def fake_get_or_load(avatar_id, batch_size):
            calls.append((avatar_id, batch_size))
            load_started.set()
            self.assertTrue(release_load.wait(timeout=2))
            avatar = SimpleNamespace(avatar_id=avatar_id, batch_size=batch_size)
            self.manager.avatar_cache.put(avatar_id, avatar, memory_usage_mb=1)
            return avatar

        self.manager._get_or_load_avatar = fake_get_or_load

        first = self.manager.warm_avatar_async("alice", batch_size=2)
        self.assertTrue(load_started.wait(timeout=2))
        second = self.manager.warm_avatar_async("alice", batch_size=4)

        self.assertTrue(second["deduped"])
        self.assertEqual(first["request_id"], second["request_id"])

        release_load.set()
        first["_future"].result(timeout=2)

        self.assertEqual(calls, [("alice", 2)])
        status = self.manager.get_avatar_cache_status("alice")
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["cached"])

    def test_different_avatar_warmups_can_run_concurrently(self):
        self._prepare_avatar_file("alice")
        self._prepare_avatar_file("bob")
        entered = set()
        entered_condition = threading.Condition()
        release_loads = threading.Event()

        def fake_get_or_load(avatar_id, batch_size):
            with entered_condition:
                entered.add(avatar_id)
                entered_condition.notify_all()
            self.assertTrue(release_loads.wait(timeout=2))
            avatar = SimpleNamespace(avatar_id=avatar_id, batch_size=batch_size)
            self.manager.avatar_cache.put(avatar_id, avatar, memory_usage_mb=1)
            return avatar

        self.manager._get_or_load_avatar = fake_get_or_load

        alice = self.manager.warm_avatar_async("alice", batch_size=2)
        bob = self.manager.warm_avatar_async("bob", batch_size=2)

        deadline = time.monotonic() + 2
        with entered_condition:
            while len(entered) < 2 and time.monotonic() < deadline:
                entered_condition.wait(timeout=0.05)

        self.assertEqual(entered, {"alice", "bob"})

        release_loads.set()
        alice["_future"].result(timeout=2)
        bob["_future"].result(timeout=2)

        self.assertEqual(self.manager.get_avatar_cache_status("alice")["status"], "ready")
        self.assertEqual(self.manager.get_avatar_cache_status("bob")["status"], "ready")


if __name__ == "__main__":
    unittest.main()
