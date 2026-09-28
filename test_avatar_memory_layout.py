"""CPU checks of the lossless avatar memory layouts (scripts/api_avatar.py).

Covers MUSETALK_AVATAR_MASK_CHANNELS, MUSETALK_AVATAR_MASK_STORE,
MUSETALK_AVATAR_FRAME_STORE (+ LRU / readahead) and
MUSETALK_AVATAR_PLAN_FLOAT_ALPHA. A synthetic prepared avatar is loaded
through the real APIAvatar loader in every layout, and every cycle position is
composed and compared with the default layout. Run:

    CUDA_VISIBLE_DEVICES= python -m unittest -q test_avatar_memory_layout
"""
import os
import pickle
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import api_avatar as A  # noqa: E402
from musetalk.utils import blending  # noqa: E402

LAYOUT_ENV = (
    "MUSETALK_AVATAR_MASK_CHANNELS",
    "MUSETALK_AVATAR_MASK_STORE",
    "MUSETALK_AVATAR_FRAME_STORE",
    "MUSETALK_AVATAR_DECODED_LRU_FRAMES",
    "MUSETALK_AVATAR_PNG_READAHEAD",
    "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA",
    "MUSETALK_CHEEK_ONLY_BLEND",
)


def _write_png(path: Path, image, params=None) -> None:
    ok = cv2.imwrite(str(path), image, params or [])
    assert ok, path


def _make_avatar_dir(root: Path, avatar_id: str, unique: int = 12, height: int = 96,
                     width: int = 64, seed: int = 0) -> Path:
    """A forward+reverse prepared cycle like _process_frames writes."""
    rng = np.random.RandomState(seed)
    base = root / "results" / "v15" / "avatars" / avatar_id
    (base / "full_imgs").mkdir(parents=True)
    (base / "mask").mkdir()
    frames, masks, coords, crops = [], [], [], []
    for i in range(unique):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        x1, y1 = 10 + i % 3, 20 + i % 2
        box = [x1, y1, x1 + 36, y1 + 40]
        crop = [box[0] - 6, box[1] - 8, box[2] + 6, box[3] + 4]
        mh, mw = crop[3] - crop[1], crop[2] - crop[0]
        yy, xx = np.mgrid[0:mh, 0:mw]
        mask = np.clip(255 - (3 + i % 4) * np.hypot(yy - mh * .6 - i % 5, xx - mw / 2 + i % 3), 0, 255).astype(np.uint8)
        mask[: mh // 3] = 0
        frames.append(frame); masks.append(mask); coords.append(box); crops.append(crop)
    cycle = list(range(unique)) + list(range(unique))[::-1]
    for pos, src in enumerate(cycle):
        _write_png(base / "full_imgs" / f"{pos:08d}.png", frames[src])
        _write_png(base / "mask" / f"{pos:08d}.png", masks[src])
    with open(base / "coords.pkl", "wb") as handle:
        pickle.dump([coords[s] for s in cycle], handle)
    with open(base / "mask_coords.pkl", "wb") as handle:
        pickle.dump([crops[s] for s in cycle], handle)
    torch.save(torch.randn(len(cycle), 1, 8, 32, 32).half(), base / "latents.pt")
    (base / "avator_info.json").write_text(
        '{"avatar_id": "%s", "bbox_shift": 0, "version": "v15", "fixed_face_height": false}' % avatar_id)
    return base


def _load_avatar(avatar_id: str, env: dict):
    stub = SimpleNamespace(model_dtype=torch.float16, runtime_dtype=torch.float16)
    with patch.dict(os.environ, {k: v for k, v in env.items()}, clear=False):
        for name in LAYOUT_ENV:
            if name not in env:
                os.environ.pop(name, None)
        return A.APIAvatar(avatar_id, "", 0, 8, stub, stub, None, None,
                           SimpleNamespace(version="v15"), preparation=False)


def _faces(count: int, seed: int = 7):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(count)]


class EncodedImageCycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _png_variants(self):
        rng = np.random.RandomState(3)
        gray = rng.randint(0, 256, (17, 23), dtype=np.uint8)
        color = rng.randint(0, 256, (17, 23, 3), dtype=np.uint8)
        rgba = rng.randint(0, 256, (17, 23, 4), dtype=np.uint8)
        gray16 = rng.randint(0, 65536, (17, 23), dtype=np.uint16)
        bilevel = ((rng.rand(17, 23) > .5) * 255).astype(np.uint8)
        variants = {"gray8": (gray, None), "bgr": (color, None), "bgra": (rgba, None),
                    "gray16": (gray16, None),
                    "bilevel": (bilevel, [cv2.IMWRITE_PNG_BILEVEL, 1])}
        paths = {}
        for name, (image, params) in variants.items():
            path = self.dir / f"{name}.png"
            _write_png(path, image, params)
            paths[name] = str(path)
        return paths

    def test_color_and_plane0_decode_match_imread_for_every_png_layout(self):
        for name, path in self._png_variants().items():
            with self.subTest(png=name):
                reference = cv2.imread(path)
                color, _ = A._EncodedImageCycle.from_paths(
                    [path], kind="color", label=name, lru_items=2, readahead=0)
                np.testing.assert_array_equal(color[0], reference)
                plane, _ = A._EncodedImageCycle.from_paths(
                    [path], kind="plane0", label=name, lru_items=2, readahead=0)
                np.testing.assert_array_equal(plane[0], reference[:, :, 0])
                np.testing.assert_array_equal(A._read_mask_plane0_required(path), reference[:, :, 0])
                self.assertEqual(plane.shape_at(0), reference.shape[:2])

    def test_dedup_slots_match_decoded_dedup_buffers(self):
        base = _make_avatar_dir(self.dir, "dedup", unique=5)
        paths = sorted(str(p) for p in (base / "full_imgs").glob("*.png"))
        decoded, unique = A._read_imgs_dedup(paths, "frames", 1)
        store, store_unique = A._EncodedImageCycle.from_paths(
            paths, kind="color", label="frames", lru_items=4, readahead=0)
        self.assertEqual(unique, store_unique)
        for i in range(len(paths)):
            for j in range(len(paths)):
                self.assertEqual(decoded[i] is decoded[j], store.slot_at(i) == store.slot_at(j))
            np.testing.assert_array_equal(store[i], decoded[i])

    def test_lru_is_bounded_and_decoded_images_are_read_only(self):
        base = _make_avatar_dir(self.dir, "lru", unique=9)
        paths = sorted(str(p) for p in (base / "full_imgs").glob("*.png"))
        store, _ = A._EncodedImageCycle.from_paths(
            paths, kind="color", label="frames", lru_items=3, readahead=2)
        for index in range(len(store)):
            image = store[index]
            self.assertFalse(image.flags.writeable)
            with self.assertRaises(ValueError):
                image[0, 0, 0] = 1
        stats = store.stats()
        self.assertLessEqual(stats["lru_items"], 3)
        self.assertEqual(store.resident_nbytes(), store.encoded_nbytes() + 3 * 96 * 64 * 3)
        store.release_decoded_cache()
        self.assertEqual(store.stats()["lru_items"], 0)

    def test_concurrent_walkers_get_exact_images(self):
        base = _make_avatar_dir(self.dir, "threads", unique=10)
        paths = sorted(str(p) for p in (base / "full_imgs").glob("*.png"))
        reference = [cv2.imread(p) for p in paths]
        store, _ = A._EncodedImageCycle.from_paths(
            paths, kind="color", label="frames", lru_items=6, readahead=4)
        errors = []

        def walk(offset):
            for step in range(120):
                index = (offset + step) % len(paths)
                if not np.array_equal(store[index], reference[index]):
                    errors.append(index)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(walk, range(0, 60, 10)))
        self.assertEqual(errors, [])
        stats = store.stats()
        self.assertEqual(stats["requests"], 6 * 120)
        self.assertGreater(stats["hits"] + stats["readahead_joins"], 0)

    def test_queued_readahead_is_cancelled_instead_of_waited_on(self):
        base = _make_avatar_dir(self.dir, "cancel", unique=4)
        paths = sorted(str(p) for p in (base / "full_imgs").glob("*.png"))
        store, _ = A._EncodedImageCycle.from_paths(
            paths, kind="color", label="frames", lru_items=4, readahead=0)
        gate = threading.Event()
        blocker = ThreadPoolExecutor(max_workers=1)
        blocker.submit(gate.wait)  # the only worker is busy: queued work never starts
        with patch.object(A, "_png_decode_pool", return_value=blocker):
            store.prefetch([1])
            np.testing.assert_array_equal(store[1], cv2.imread(paths[1]))
        gate.set(); blocker.shutdown(wait=True)
        self.assertEqual(store.stats()["readahead_cancels"], 1)
        self.assertEqual(store.stats()["inflight"], 0)

    def test_rejects_truncated_png(self):
        path = self.dir / "cut.png"
        _write_png(path, np.zeros((8, 8, 3), np.uint8))
        path.write_bytes(path.read_bytes()[:-12])
        with self.assertRaises(ValueError):
            A._EncodedImageCycle.from_paths([str(path)], kind="color", label="x",
                                            lru_items=1, readahead=0)


class LeanPlanTest(unittest.TestCase):
    def test_derived_float_alpha_is_exact_and_blend_is_identical(self):
        rng = np.random.RandomState(1)
        mask = rng.randint(0, 256, (40, 44), dtype=np.uint8)
        image = rng.randint(0, 256, (80, 70, 3), dtype=np.uint8)
        face = rng.randint(0, 256, (30, 28, 3), dtype=np.uint8)
        plan = blending.prepare_image_blending_plan(image.shape, (20, 25, 48, 55), mask, (14, 18, 58, 58))
        lean = A._lean_compose_plan(plan)
        self.assertIsNone(lean.get("alpha"))
        self.assertNotIn("alpha", lean)
        np.testing.assert_array_equal(lean["alpha"], plan["alpha"])
        self.assertEqual(lean["alpha"].dtype, plan["alpha"].dtype)
        for fixed_point in (True, False):
            with patch.object(blending, "MUSETALK_BLEND_FIXED_POINT", fixed_point):
                np.testing.assert_array_equal(
                    blending.get_image_blending_with_plan(image.copy(), face, lean),
                    blending.get_image_blending_with_plan(image.copy(), face, plan))
        with self.assertRaises(KeyError):
            lean["missing"]


class FlagParsingTest(unittest.TestCase):
    def test_defaults_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in LAYOUT_ENV:
                os.environ.pop(name, None)
            flags = A.avatar_memory_layout_flags()
        self.assertEqual(flags["MUSETALK_AVATAR_MASK_CHANNELS"], 3)
        self.assertEqual(flags["MUSETALK_AVATAR_MASK_STORE"], "decoded")
        self.assertEqual(flags["MUSETALK_AVATAR_FRAME_STORE"], "decoded")
        self.assertEqual(flags["MUSETALK_AVATAR_PLAN_FLOAT_ALPHA"], 1)
        for name, value in (("MUSETALK_AVATAR_MASK_CHANNELS", "2"),
                            ("MUSETALK_AVATAR_FRAME_STORE", "jpeg"),
                            ("MUSETALK_AVATAR_PLAN_FLOAT_ALPHA", "maybe"),
                            ("MUSETALK_AVATAR_DECODED_LRU_FRAMES", "-1")):
            with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValueError):
                    A.avatar_memory_layout_flags()


class APIAvatarLayoutTest(unittest.TestCase):
    """The real loader + compose_frame in every layout vs the default layout."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = os.getcwd()
        _make_avatar_dir(Path(cls.tmp.name), "synthetic", unique=11)
        os.chdir(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    LAYOUTS = {
        "mask1": {"MUSETALK_AVATAR_MASK_CHANNELS": "1"},
        "maskpng3": {"MUSETALK_AVATAR_MASK_STORE": "png"},
        "maskpng1": {"MUSETALK_AVATAR_MASK_STORE": "png", "MUSETALK_AVATAR_MASK_CHANNELS": "1"},
        "framepng": {"MUSETALK_AVATAR_FRAME_STORE": "png"},
        "framepng_inline": {"MUSETALK_AVATAR_FRAME_STORE": "png", "MUSETALK_AVATAR_PNG_READAHEAD": "0",
                            "MUSETALK_AVATAR_DECODED_LRU_FRAMES": "1"},
        "lean": {"MUSETALK_AVATAR_PLAN_FLOAT_ALPHA": "0"},
        "all": {"MUSETALK_AVATAR_FRAME_STORE": "png", "MUSETALK_AVATAR_MASK_STORE": "png",
                "MUSETALK_AVATAR_MASK_CHANNELS": "1", "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA": "0"},
    }

    def _compose_all(self, avatar, faces, extra_env=None):
        outputs = []
        count = len(avatar.coord_list_cycle)
        background = np.full((96, 64, 3), 77, np.uint8)
        with patch.dict(os.environ, extra_env or {}):
            for index in list(range(count)) + [count, count + 1, 2 * count - 1, 3 * count + 5]:
                face = faces[index % len(faces)]
                outputs.append(avatar.compose_frame(face, index))
                layers = avatar.compose_frame(face, index, return_layers=True)
                outputs += [layers["composed"], layers["raw"], layers["alpha"]["values"],
                            np.asarray(layers["alpha"]["bounds"])]
                outputs.append(avatar.compose_frame(face, index, background_frame=background))
        return outputs

    def test_every_layout_composes_bit_identically(self):
        faces = _faces(5)
        for cheek in ("0", "1"):
            base_env = {"MUSETALK_CHEEK_ONLY_BLEND": cheek}
            reference_avatar = _load_avatar("synthetic", base_env)
            reference = self._compose_all(reference_avatar, faces)
            for name, env in self.LAYOUTS.items():
                with self.subTest(layout=name, cheek=cheek):
                    avatar = _load_avatar("synthetic", {**base_env, **env})
                    for fixed_point in (True, False):
                        with patch.object(blending, "MUSETALK_BLEND_FIXED_POINT", fixed_point):
                            expected = (reference if fixed_point else
                                        self._compose_all(reference_avatar, faces))
                            got = self._compose_all(avatar, faces)
                        self.assertEqual(len(expected), len(got))
                        for a, b in zip(expected, got):
                            np.testing.assert_array_equal(a, b)

    def test_accounting_follows_the_layout(self):
        default = _load_avatar("synthetic", {})
        lean = _load_avatar("synthetic", self.LAYOUTS["all"])
        frame_bytes = 96 * 64 * 3
        frames = lean.frame_list_cycle
        self.assertEqual(type(default)._numpy_sequence_nbytes(frames),
                         frames.encoded_nbytes() + min(24, frames.unique_count) * frame_bytes)
        mask_nbytes = type(default)._numpy_sequence_nbytes(_load_avatar("synthetic", self.LAYOUTS["mask1"]).mask_list_cycle)
        self.assertEqual(mask_nbytes * 3, type(default)._numpy_sequence_nbytes(default.mask_list_cycle))
        plans = type(default)._compose_plan_sequence_nbytes(lean._compose_plan_cycle)
        self.assertEqual(plans * 5, type(default)._compose_plan_sequence_nbytes(default._compose_plan_cycle))

    def test_cache_cleanup_releases_png_store(self):
        from avatar_cache import AvatarCache
        avatar = _load_avatar("synthetic", self.LAYOUTS["framepng"])
        frames = avatar.frame_list_cycle
        avatar.compose_frame(_faces(1)[0], 0)
        self.assertGreater(frames.stats()["lru_items"], 0)
        AvatarCache(cleanup_interval=3600)._cleanup_avatar(avatar)
        self.assertEqual(frames.stats()["lru_items"], 0)
        self.assertFalse(hasattr(avatar, "frame_list_cycle"))


if __name__ == "__main__":
    unittest.main()
