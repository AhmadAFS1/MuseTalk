#!/usr/bin/env python3
"""Stubbed tests for musetalk.models.unet's meta-device fast load (no torch needed).

The real equivalence was measured on an RTX 3090 worker (c5-a, 2026-10-18): bit-identical
parameters and fp16 forward output, 9.95 s -> 0.97 s. These tests pin the call shape and
every fallback to the original load path."""
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parent


class FakeTensor:
    def __init__(self, meta=False):
        self.is_meta = meta


class FakeModel:
    instances = []

    def __init__(self, **config):
        self.config, self.built_on, self.loaded, self.moved = config, FakeTorch.current_device, None, []
        self.params = [FakeTensor(self.built_on == "meta")]
        FakeModel.instances.append(self)

    def load_state_dict(self, weights, assign=False):
        if assign and FakeTorch.reject_assign:
            raise TypeError("load_state_dict() got an unexpected keyword argument 'assign'")
        self.loaded = (weights, assign)
        if assign and not FakeTorch.leave_meta:
            self.params = [FakeTensor(False)]

    def parameters(self):
        return iter(self.params)

    def buffers(self):
        return iter([])

    def half(self):
        return self

    def to(self, device):
        self.moved.append(device)
        return self


class FakeTorch(types.ModuleType):
    current_device = "cpu"
    reject_assign = False
    leave_meta = False
    loads = []

    def __init__(self):
        super().__init__("torch")
        self.nn = types.SimpleNamespace(Module=object)
        self.cuda = types.SimpleNamespace(is_available=lambda: True)

    @staticmethod
    @contextlib.contextmanager
    def device(name):
        if not isinstance(name, str) or name != "meta":
            yield name
            return
        previous, FakeTorch.current_device = FakeTorch.current_device, "meta"
        try:
            yield name
        finally:
            FakeTorch.current_device = previous

    @staticmethod
    def load(path, **kwargs):
        FakeTorch.loads.append(kwargs)
        return {"weights": path}


class FastLoadTests(unittest.TestCase):
    def setUp(self):
        FakeModel.instances.clear()
        FakeTorch.loads.clear()
        FakeTorch.reject_assign = FakeTorch.leave_meta = False
        torch = FakeTorch()
        diffusers = types.ModuleType("diffusers")
        diffusers.UNet2DConditionModel = FakeModel
        self.modules = patch.dict(sys.modules, {"torch": torch, "torch.nn": torch.nn, "diffusers": diffusers,
                                                "numpy": types.ModuleType("numpy")})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        sys.path.insert(0, str(REPO))
        self.addCleanup(sys.path.remove, str(REPO))
        sys.modules.pop("musetalk.models.unet", None)
        self.unet = importlib.import_module("musetalk.models.unet")
        self.unet.PositionalEncoding = lambda d_model: None
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.config = Path(folder.name) / "musetalk.json"
        self.config.write_text(json.dumps({"in_channels": 8}))

    def build(self, env=None):
        with patch.dict(os.environ, env or {}):
            return self.unet.UNet(str(self.config), "unet.pth", device="cuda:0")

    def test_fast_path_builds_on_meta_and_assigns_the_mmapped_checkpoint(self):
        model = self.build().model
        self.assertEqual(len(FakeModel.instances), 1)
        self.assertEqual((model.built_on, model.loaded), ("meta", ({"weights": "unet.pth"}, True)))
        self.assertEqual(FakeTorch.loads, [{"map_location": None, "mmap": True}])
        self.assertEqual(model.moved, ["cuda:0"])

    def test_opt_out_and_old_torch_use_the_original_load(self):
        for label, env in (("opt-out", {"MUSETALK_UNET_FAST_LOAD": "0"}), ("old-torch", {})):
            with self.subTest(label=label):
                FakeModel.instances.clear()
                FakeTorch.loads.clear()
                FakeTorch.reject_assign = label == "old-torch"
                model = self.build(env).model
                self.assertEqual((model.built_on, model.loaded), ("cpu", ({"weights": "unet.pth"}, False)))
                self.assertEqual(FakeTorch.loads[-1], {})  # torch.load(model_path) exactly as before

    def test_tensors_left_on_meta_fall_back(self):
        FakeTorch.leave_meta = True
        model = self.build().model
        self.assertEqual(len(FakeModel.instances), 2)
        self.assertEqual(model.built_on, "cpu")


if __name__ == "__main__":
    unittest.main()
