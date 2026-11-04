"""Tiny preparation/provenance tests. No Torch, CUDA, model loads, or network."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prepare_new_identity_review_under_test", HERE / "prepare_new_identity_review.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="new-identity-prep-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.patch = mock.patch.object(prep, "REPO", self.repo)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        for rel in prep.CODE_FILES + prep.MODEL_FILES + ("models/sd-vae/diffusion_pytorch_model.bin",):
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("fixture-" + rel).encode())
        self.portraits = self.root / "portraits"
        self.portraits.mkdir()
        chars = []
        for i in range(30):
            aid = f"ni30_{i + 1:02d}"
            Image.new("RGB", (6, 4), (i * 7, i * 5, i * 3)).save(self.portraits / f"{aid}.png")
            chars.append({"id": aid, "personality": f"Fictional personality {i}", "age": 25 + i})
        self.characters = self.root / "characters.json"
        self.characters.write_text(json.dumps({"characters": chars}))
        self.audio = [self.root / "ai-assistant.mpga", self.root / "eng.wav"]
        for i, path in enumerate(self.audio):
            path.write_bytes(f"tiny-audio-{i}".encode())
        self.checkpoint = self.root / "student.pt"
        self.checkpoint.write_bytes(b"tiny checkpoint fixture")
        self.args = argparse.Namespace(characters=self.characters, portrait_root=self.portraits,
            audio=self.audio, checkpoint=self.checkpoint, checkpoint_sha256=prep.sha(self.checkpoint),
            out=self.root / "out", seed=20261104, resume=False, rebuild_incomplete=False, dry_run=False)
        self.calls = []

    def inputs(self):
        return prep.inspect_inputs(self.args)

    def factory(self):
        self.assertEqual(Path.cwd(), self.repo)
        self.assertEqual(os.environ["MUSETALK_FIXED_FACE_HEIGHT"], "0")
        def make(aid, source, cache, seed):
            self.calls.append((aid, seed))
            cache.mkdir(parents=True)
            for folder in ("full_imgs", "mask"):
                (cache / folder).mkdir()
                for i in range(2):
                    (cache / folder / f"{i:08d}.png").write_bytes(f"fixture-{folder}-{aid}".encode())
            for file in ("coords.pkl", "mask_coords.pkl", "latents.pt"):
                (cache / file).write_bytes(f"fixture-{file}-{aid}".encode())
        return make

    def execute(self):
        payload, missing = self.inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            prep.execute(self.args, payload, missing, self.factory)
        return json.loads((self.args.out / "new_avatar_manifest.json").read_text())

    def test_complete_manifest_resume_and_actual_reviewer_schema(self):
        before = Path.cwd()
        with mock.patch.dict(os.environ, {"MUSETALK_FIXED_FACE_HEIGHT": "1"}):
            manifest = self.execute()
            self.assertEqual(os.environ["MUSETALK_FIXED_FACE_HEIGHT"], "1")
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(len(manifest["avatars"]), 30)
        self.assertEqual(len(self.calls), 30)
        self.assertEqual(manifest["avatars"][0]["metadata"]["personality"], "Fictional personality 0")
        self.assertEqual(manifest["avatars"][0]["normalization"]["normalized_dimensions"], [512, 896])
        self.assertEqual(len(manifest["avatars"][0]["cache_files"]), 7)
        self.assertIn("runtime/results/v15/avatars/ni30_01", manifest["avatars"][0]["cache_dir"])
        # Execute the real pure manifest validator without importing the heavy reviewer module.
        tree = ast.parse((HERE / "review_distilled_unet.py").read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"sha256_file", "validate_new_manifest"}]
        namespace = {"Path": Path, "json": json, "hashlib": hashlib, "NEW_PROTOCOL": prep.PROTOCOL}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "reviewer_manifest_validator", "exec"), namespace)
        checked = namespace["validate_new_manifest"](self.args.out / "new_avatar_manifest.json",
            [Path(v["cache_dir"]) for v in manifest["avatars"]], [Path(v["file"]) for v in manifest["audio"]],
            self.checkpoint, 25, 10, None)
        self.assertEqual(len(checked["expected_clips"]), 60)
        self.args.resume = True
        self.execute()
        self.assertEqual(len(self.calls), 30, "verified resume must not prepare any avatar again")

    def test_resume_refuses_changed_inputs_even_with_rebuild(self):
        self.execute()
        self.args.resume = self.args.rebuild_incomplete = True
        rows = json.loads(self.characters.read_text())
        rows["characters"][0]["personality"] = "changed metadata"
        self.characters.write_text(json.dumps(rows))
        with self.assertRaisesRegex(prep.PrepError, "hashes differ"):
            self.execute()

    def test_corrupt_cache_requires_explicit_rebuild(self):
        manifest = self.execute()
        (Path(manifest["avatars"][0]["cache_dir"]) / "latents.pt").write_bytes(b"corrupt")
        self.args.resume = True
        with self.assertRaisesRegex(prep.PrepError, "incomplete/corrupt cache"):
            self.execute()
        self.args.rebuild_incomplete = True
        self.execute()
        self.assertEqual(len(self.calls), 31)

    def test_incomplete_normalization_requires_explicit_rebuild(self):
        self.execute()
        (self.args.out / "normalized/ni30_01/normalization.json").unlink()
        self.args.resume = True
        with self.assertRaisesRegex(prep.PrepError, "normalized portrait"):
            self.execute()
        self.args.rebuild_incomplete = True
        self.execute()
        self.assertEqual(len(self.calls), 30)

    def test_duplicate_portrait_id_audio_and_wrong_checkpoint_refused(self):
        original = (self.portraits / "ni30_02.png").read_bytes()
        (self.portraits / "ni30_02.png").write_bytes((self.portraits / "ni30_01.png").read_bytes())
        with self.assertRaisesRegex(prep.PrepError, "distinct portrait"):
            self.inputs()
        (self.portraits / "ni30_02.png").write_bytes(original)
        rows = json.loads(self.characters.read_text())
        rows["characters"][1]["id"] = "ni30_01"
        self.characters.write_text(json.dumps(rows))
        with self.assertRaisesRegex(prep.PrepError, "unique"):
            self.inputs()
        rows["characters"][1]["id"] = "ni30_02"
        self.characters.write_text(json.dumps(rows))
        self.audio[1].write_bytes(self.audio[0].read_bytes())
        with self.assertRaisesRegex(prep.PrepError, "audio clips must be distinct"):
            self.inputs()
        self.audio[1].write_bytes(b"different-audio")
        self.args.checkpoint_sha256 = "0" * 64
        with self.assertRaisesRegex(prep.PrepError, "checkpoint differs"):
            self.inputs()

    def test_declared_accepted_portrait_sha_is_enforced(self):
        rows = json.loads(self.characters.read_text())
        rows["characters"][0]["portrait_sha256"] = "0" * 64
        self.characters.write_text(json.dumps(rows))
        with self.assertRaisesRegex(prep.PrepError, "accepted generation SHA256"):
            self.inputs()
        rows["characters"][0]["portrait_sha256"] = "A" * 64
        self.characters.write_text(json.dumps(rows))
        with self.assertRaisesRegex(prep.PrepError, "64 lowercase hex"):
            self.inputs()
        rows["characters"][0]["portrait_sha256"] = prep.sha(self.portraits / "ni30_01.png")
        self.characters.write_text(json.dumps(rows))
        payload, _ = self.inputs()
        self.assertEqual(payload["characters"][0]["sha256"], rows["characters"][0]["portrait_sha256"])
        self.assertNotIn("torch", sys.modules)

    def test_padding_preserves_entire_input_geometry(self):
        source = self.root / "wide.png"
        Image.new("RGB", (40, 10), (250, 10, 20)).save(source)
        target = self.root / "normalized.png"
        info = prep.normalize(source, target)
        self.assertEqual(info["resized_dimensions"], [512, 128])
        self.assertEqual(info["padding_left_top"], [0, 384])
        with Image.open(target) as image:
            self.assertEqual(image.size, (512, 896))
            self.assertEqual(image.getpixel((0, 0)), (128, 128, 128))
            self.assertEqual(image.getpixel((0, 384)), (250, 10, 20))
            self.assertEqual(image.getpixel((511, 511)), (250, 10, 20))

    def test_dry_run_creates_nothing_and_imports_no_torch(self):
        self.args.dry_run = True
        self.assertNotIn("torch", sys.modules)
        payload, missing = self.inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            prep.execute(self.args, payload, missing, lambda: self.fail("dry run loaded production models"))
        self.assertFalse(self.args.out.exists())
        self.assertNotIn("torch", sys.modules)

    def test_original_constructor_runs_in_isolated_cwd_without_subclass(self):
        calls = []
        class Model:
            def eval(self): return self
            def requires_grad_(self, value): return self
        def vae(**kwargs):
            calls.append(("vae", Path.cwd(), kwargs))
            return types.SimpleNamespace(vae=Model())
        def parser(**kwargs):
            calls.append(("parser", Path.cwd(), kwargs))
            return object()
        def original(**kwargs):
            calls.append(("constructor", Path.cwd(), kwargs))
            return types.SimpleNamespace(avatar_path=f"./results/v15/avatars/{kwargs['avatar_id']}")
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda _: None),
            manual_seed=lambda _: None, inference_mode=contextlib.nullcontext,
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace()),
            set_float32_matmul_precision=lambda _: None)
        modules = {"torch": fake_torch, "scripts.api_avatar": types.SimpleNamespace(APIAvatar=original),
                   "musetalk.models.vae": types.SimpleNamespace(VAE=vae),
                   "musetalk.utils.face_parsing": types.SimpleNamespace(FaceParsing=parser),
                   "musetalk.utils": types.SimpleNamespace(preprocessing=object())}
        cache = self.root / "out/runtime/results/v15/avatars/ni30_01"
        before = Path.cwd()
        with mock.patch.dict(sys.modules, modules), prep.cwd(self.repo):
            make = prep.load_production_preparer()
            make("ni30_01", self.portraits.resolve(), cache, 123)
            self.assertEqual(Path.cwd(), self.repo)
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(calls[0][0:2], ("vae", self.repo))
        self.assertEqual(calls[1][0:2], ("parser", self.repo))
        self.assertEqual(calls[2][0:2], ("constructor", self.root / "out/runtime"))
        self.assertIsNone(calls[2][2]["unet"])
        self.assertIsNone(calls[2][2]["pe"])


if __name__ == "__main__":
    unittest.main()
