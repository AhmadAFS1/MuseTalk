"""Stdlib-only contracts for actual manager methods; no GPU/quality proof."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock


SOURCE = Path(__file__).parent / "scripts/avatar_manager_parallel.py"
METHODS = {"_init_models", "_activate_unet_backend", "_skip_eager_unet_requested", "_env_enabled"}
STRICT = {
    "MUSETALK_SKIP_EAGER_UNET": "1", "MUSETALK_UNET_BACKEND": "trt_stagewise",
    "MUSETALK_TRT_FALLBACK": "0", "MUSETALK_UNET_STAGEWISE_VERIFY_SHA": "1",
    "MUSETALK_UNET_STAGEWISE_PROBE_CHECK": "1", "MUSETALK_UNET_STAGEWISE_PROBE_TOL": "0",
}


class FakeModule:
    dtype = "fp16"
    config = NS(in_channels=8)

    def half(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, value):
        return self


class FakeVAE:
    def __init__(self):
        self.vae = FakeModule()
        self.scaling_factor = 0.18215

    def set_decode_backend(self, backend):
        self.backend = backend

    def get_decode_backend_name(self):
        return "pytorch"

    def has_decode_backend(self):
        return False


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / "unet.json"
        self.config.write_text(json.dumps({"in_channels": 8, "out_channels": 4, "cross_attention_dim": 384}))
        self.vae = FakeVAE()
        self.eager = FakeModule()
        self.backend = NS(name="tensorrt_unet_stagewise", variant="srccache", batch=16, dtype="fp16")
        tree = ast.parse(SOURCE.read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelAvatarManager")
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
        self.assertEqual({n.name for n in methods}, METHODS)
        selected = ast.Module(body=[ast.ClassDef(name="Manager", bases=[], keywords=[], body=methods,
                                                decorator_list=[])], type_ignores=[])
        self.ns = {
            "os": os, "json": json, "time": time, "SimpleNamespace": NS,
            "torch": NS(float16="fp16", cuda=NS(is_available=lambda: True, empty_cache=mock.Mock()),
                        backends=NS(cuda=NS(matmul=NS()), cudnn=NS()),
                        set_float32_matmul_precision=mock.Mock(), tensor=mock.Mock()),
            "load_all_model": mock.Mock(return_value=(self.vae, NS(model=self.eager), FakeModule())),
            "VAE": mock.Mock(return_value=self.vae), "PositionalEncoding": mock.Mock(side_effect=lambda **kw: FakeModule()),
            "load_unet_trt_backend": mock.Mock(return_value=self.backend),
            "load_vae_trt_decoder": mock.Mock(return_value=None), "AudioProcessor": mock.Mock(),
            "WhisperModel": NS(from_pretrained=mock.Mock(return_value=FakeModule())), "FaceParsing": mock.Mock(),
        }
        exec(compile(ast.fix_missing_locations(selected), str(SOURCE), "exec"), self.ns)
        self.manager = self.ns["Manager"]()
        self.manager.device = NS(type="cuda")
        self.manager.unet_backend_name = "pytorch"
        self.manager.args = NS(unet_config=str(self.config), unet_model_path="unchanged/checkpoint.pth",
                               vae_type="sd-vae", whisper_dir="unchanged/whisper", version="v15",
                               left_cheek_width=90, right_cheek_width=90)
        self.manager.compile_models = mock.Mock()
        self.manager._warm_runtime_paths = mock.Mock()

    def initialize(self, env):
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.manager._init_models()

    def test_default_retains_original_loader_and_eager_model(self):
        self.initialize({})
        self.ns["load_all_model"].assert_called_once_with(
            unet_model_path="unchanged/checkpoint.pth", vae_type="sd-vae",
            unet_config=str(self.config), device=self.manager.device)
        self.ns["VAE"].assert_not_called()
        self.assertIs(self.manager.eager_unet_model, self.eager)
        self.assertFalse(self.manager.model_startup_profile["skip_eager_unet"])

    def test_default_pytorch_fallback_retained(self):
        self.ns["load_unet_trt_backend"].return_value = None
        self.initialize({})
        self.assertIs(self.manager.unet.model, self.eager)

    def test_opt_in_skips_eager_but_keeps_original_vae_and_warming(self):
        self.initialize(STRICT)
        self.ns["load_all_model"].assert_not_called()
        self.ns["VAE"].assert_called_once_with(model_path="models/sd-vae")
        self.ns["load_unet_trt_backend"].assert_called_once_with(device=self.manager.device)
        self.assertIs(self.manager.vae.vae, self.vae.vae)
        self.assertIs(self.manager.eager_vae_model, self.vae.vae)
        self.assertIs(self.manager.unet.model, self.backend)
        self.assertIsNone(self.manager.eager_unet_model)
        self.assertEqual(self.manager.unet_in_channels, 8)
        self.assertEqual(self.manager.unet_dtype, "fp16")
        self.assertEqual(self.ns["PositionalEncoding"].call_count, 2)
        self.manager._warm_runtime_paths.assert_called_once()
        self.assertTrue(self.manager.model_startup_profile["avatar_vae_encoder_retained"])

    def test_each_strict_setting_is_mandatory_before_loading(self):
        for key in STRICT.keys() - {"MUSETALK_SKIP_EAGER_UNET"}:
            for wrong in (None, "wrong"):
                env = dict(STRICT)
                if wrong is None:
                    env.pop(key)
                else:
                    env[key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaises(RuntimeError):
                    self.initialize(env)
        self.ns["load_all_model"].assert_not_called()
        self.ns["VAE"].assert_not_called()

    def test_invalid_opt_in_never_silently_enables_or_disables(self):
        for value in ("", "true", "yes", "2"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.initialize({**STRICT, "MUSETALK_SKIP_EAGER_UNET": value})
        self.ns["VAE"].assert_not_called()

    def test_cpu_request_fails_before_model_loading(self):
        self.manager.device = NS(type="cpu")
        with self.assertRaises(RuntimeError):
            self.initialize(STRICT)
        self.ns["VAE"].assert_not_called()

    def test_incompatible_config_fails_before_vae_or_backend(self):
        for config in ([], {}, {"in_channels": True},
                       {"in_channels": 9, "out_channels": 4, "cross_attention_dim": 384}):
            self.config.write_text(json.dumps(config))
            with self.subTest(config=config), self.assertRaises(RuntimeError):
                self.initialize(STRICT)
        self.ns["VAE"].assert_not_called()
        self.ns["load_unet_trt_backend"].assert_not_called()

    def test_unavailable_or_wrong_backend_never_falls_back_to_eager(self):
        variants = [None, NS(**{**vars(self.backend), "name": "wrong"}),
                    NS(**{**vars(self.backend), "variant": "default"}),
                    NS(**{**vars(self.backend), "batch": 8}), NS(**{**vars(self.backend), "dtype": "fp32"})]
        for backend in variants:
            self.ns["load_unet_trt_backend"].return_value = backend
            with self.subTest(backend=backend), self.assertRaises(RuntimeError):
                self.initialize(STRICT)
        self.ns["load_all_model"].assert_not_called()
        self.manager._warm_runtime_paths.assert_not_called()

    def test_backend_exception_is_propagated_not_repaired(self):
        self.ns["load_unet_trt_backend"].side_effect = OSError("synthetic corrupt plan")
        with self.assertRaises(OSError):
            self.initialize(STRICT)
        self.ns["load_all_model"].assert_not_called()
        self.manager._warm_runtime_paths.assert_not_called()


if __name__ == "__main__":
    unittest.main()
