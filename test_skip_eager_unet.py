#!/usr/bin/env python3
"""Stubbed tests for the strict eager-UNet skip in scripts/avatar_manager_parallel.py.

On strict r5 serving the checked TensorRT stagewise engines replace the eager UNet and
fallback is forbidden, so its 3.4 GB checkpoint is neither downloaded nor loaded. These
tests pin when skipping is allowed and that the loaded backend must be the expected one."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parent
STRICT = {"MUSETALK_UNET_BACKEND": "trt_stagewise", "MUSETALK_TRT_FALLBACK": "0"}


def _load_manager():
    saved = {}

    def install(name, module):
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = module

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None, synchronize=lambda: None)
    torch.float16 = "float16"
    torch.device = lambda value: SimpleNamespace(type=str(value).split(":")[0])
    install("torch", torch)
    for name, attrs in (("musetalk.utils.face_parsing", {"FaceParsing": object}),
                        ("musetalk.utils.utils", {"load_all_model": None, "VAE": object, "PositionalEncoding": object}),
                        ("musetalk.utils.audio_processor", {"AudioProcessor": object}),
                        ("transformers", {"WhisperModel": object}),
                        ("scripts.api_avatar", {"APIAvatar": object}),
                        ("scripts.concurrent_gpu_manager", {"GPUMemoryManager": object}),
                        ("scripts.trt_runtime", {"load_unet_trt_backend": None, "load_vae_trt_decoder": None})):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        install(name, module)
    sys.path.insert(0, str(REPO))
    sys.modules.pop("scripts.avatar_manager_parallel", None)
    import scripts.avatar_manager_parallel as module
    sys.modules.pop("scripts.avatar_manager_parallel", None)
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    return module


amp = _load_manager()


class SkipEagerUnetTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.checkpoint = self.root / "unet.pth"
        self.manager = SimpleNamespace(device=SimpleNamespace(type="cuda"),
                                       args=SimpleNamespace(unet_model_path=str(self.checkpoint)))

    def requested(self, env, cuda=True):
        self.manager.device = SimpleNamespace(type="cuda" if cuda else "cpu")
        keys = ["MUSETALK_SKIP_EAGER_UNET", *STRICT, "MUSETALK_UNET_STAGEWISE_VERIFY_SHA",
                "MUSETALK_UNET_STAGEWISE_PROBE_CHECK", "MUSETALK_UNET_STAGEWISE_PROBE_TOL"]
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with patch.dict(os.environ, {**clean, **env}, clear=True):
            return amp.ParallelAvatarManager._skip_eager_unet_requested(self.manager)

    def test_explicit_skip_requires_strict_checked_r5_on_cuda(self):
        self.assertTrue(self.requested({"MUSETALK_SKIP_EAGER_UNET": "1", **STRICT}))  # unset verify/probe = defaults
        for change in ({"MUSETALK_TRT_FALLBACK": "1"}, {"MUSETALK_UNET_BACKEND": "eager"},
                       {"MUSETALK_UNET_STAGEWISE_VERIFY_SHA": "0"}, {"MUSETALK_UNET_STAGEWISE_PROBE_CHECK": "0"},
                       {"MUSETALK_UNET_STAGEWISE_PROBE_TOL": "0.01"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.requested({"MUSETALK_SKIP_EAGER_UNET": "1", **STRICT, **change})
        with self.assertRaises(RuntimeError):
            self.requested({"MUSETALK_SKIP_EAGER_UNET": "1", **STRICT}, cuda=False)
        with self.assertRaises(RuntimeError):
            self.requested({"MUSETALK_SKIP_EAGER_UNET": "yes", **STRICT})
        self.assertFalse(self.requested({"MUSETALK_SKIP_EAGER_UNET": "0", **STRICT}))

    def test_auto_mode_skips_only_a_missing_checkpoint_under_strict_r5(self):
        self.assertTrue(self.requested(STRICT))  # e.g. a manual restart on a worker installed without it
        self.assertFalse(self.requested({"MUSETALK_TRT_FALLBACK": "1", "MUSETALK_UNET_BACKEND": "trt_stagewise"}))
        self.checkpoint.write_bytes(b"x")
        self.assertFalse(self.requested(STRICT))  # present: loaded exactly as before

    def test_backend_identity_is_checked_when_the_eager_unet_was_skipped(self):
        good = SimpleNamespace(name="tensorrt_unet_stagewise", variant="srccache", batch=16, dtype="float16")
        for label, backend in (("good", good), ("none", None), ("variant", SimpleNamespace(**{**vars(good), "variant": "default"})),
                               ("batch", SimpleNamespace(**{**vars(good), "batch": 8}))):
            with self.subTest(label=label):
                manager = SimpleNamespace(device="cuda", skip_eager_unet=True, unet=SimpleNamespace(model=None),
                                          unet_compiled=None, _env_enabled=lambda *a: False)
                with patch.object(amp, "load_unet_trt_backend", return_value=backend):
                    if label == "good":
                        amp.ParallelAvatarManager._activate_unet_backend(manager)
                        self.assertIs(manager.unet.model, good)
                    else:
                        with self.assertRaises(RuntimeError):
                            amp.ParallelAvatarManager._activate_unet_backend(manager)

    def test_download_and_install_check_drop_only_the_checkpoint(self):
        weights = (REPO / "download_weights.sh").read_text()
        self.assertIn('if [[ "${MUSETALK_SKIP_EAGER_UNET:-0}" == 1 ]]; then', weights)
        sys.path.insert(0, str(REPO / "scripts"))
        source = (REPO / "scripts/musetalk_install_state.py").read_text()
        self.assertIn('model_files = [f for f in model_files if f != "models/musetalkV15/unet.pth"]', source)
        boot = (REPO / "scripts/vast_bootstrap_locked.sh").read_text()
        self.assertLess(boot.index("  select_eager_unet_mode\n"), boot.index("  start_weights_prefetch\n"))


if __name__ == "__main__":
    unittest.main()
