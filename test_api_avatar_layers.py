"""CPU checks of the actual APIAvatar composition method and layer ownership.

Only the method's AST is loaded to avoid model imports and the existing
class-level torch.no_grad wrapper. This executes the production method body;
no replacement implementation or model initialization is used.
"""
import ast
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from musetalk.utils import blending


def load_compose_method():
    path = Path(__file__).parent / 'scripts' / 'api_avatar.py'
    module = ast.parse(path.read_text(), filename=str(path))
    avatar = next(node for node in module.body
                  if isinstance(node, ast.ClassDef) and node.name == 'APIAvatar')
    method = next(node for node in avatar.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'compose_frame')
    namespace = {'np': np, 'cv2': cv2,
                 'get_image_blending': blending.get_image_blending,
                 'get_image_blending_with_plan': blending.get_image_blending_with_plan,
                 'prepare_image_blending_plan': blending.prepare_image_blending_plan}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compose_frame']


COMPOSE_FRAME = load_compose_method()


def make_avatar(*, fallback=False, empty=False, bbox=(2, 2, 6, 6), crop=(0, 0, 8, 8)):
    background = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    mask = (np.zeros((height, width), dtype=np.uint8) if empty else
            (np.arange(width * height).reshape(height, width) % 191 + 32).astype(np.uint8))
    plan = blending.prepare_image_blending_plan(background.shape, bbox, mask, crop)
    avatar = SimpleNamespace(
        # Mirrored frames intentionally share every underlying cache buffer.
        frame_list_cycle=[background, background], coord_list_cycle=[bbox, bbox],
        mask_list_cycle=[mask, mask], mask_coords_list_cycle=[crop, crop],
        _compose_plan_cycle=[] if fallback else [plan, plan],
    )
    avatar.compose_frame = MethodType(COMPOSE_FRAME, avatar)
    return avatar


def effective_alpha(layers):
    result = np.zeros(layers['raw'].shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = layers['alpha']['bounds']
    result[y0:y1, x0:x1] = layers['alpha']['values']
    return result


class APIAvatarLayersTest(unittest.TestCase):
    def setUp(self):
        self.avatar = make_avatar()
        self.face = np.full((3, 5, 3), (240, 11, 96), dtype=np.uint8)

    def test_default_is_existing_blending_reference_and_layers_match_default(self):
        for fixed_point in (True, False):
            for fallback in (False, True):
                with self.subTest(fixed_point=fixed_point, fallback=fallback), patch.object(
                        blending, 'MUSETALK_BLEND_FIXED_POINT', fixed_point):
                    avatar = make_avatar(fallback=fallback)
                    expected = blending.get_image_blending(
                        avatar.frame_list_cycle[0].copy(), cv2.resize(self.face, (4, 4)),
                        avatar.coord_list_cycle[0], avatar.mask_list_cycle[0],
                        avatar.mask_coords_list_cycle[0])
                    default = avatar.compose_frame(self.face, 3)
                    layers = avatar.compose_frame(self.face, 3, return_layers=True)
                    self.assertIsInstance(default, np.ndarray)
                    self.assertEqual(set(layers), {'composed', 'raw', 'alpha'})
                    np.testing.assert_array_equal(default, expected)
                    np.testing.assert_array_equal(layers['composed'], default)
                    np.testing.assert_array_equal(layers['raw'], avatar.frame_list_cycle[1])

    def test_sparse_alpha_excludes_mask_outside_actual_face_paste(self):
        layers = self.avatar.compose_frame(self.face, 0, return_layers=True)
        self.assertEqual(layers['alpha']['bounds'], (2, 2, 6, 6))
        values = layers['alpha']['values']
        np.testing.assert_array_equal(values, self.avatar.mask_list_cycle[0][2:6, 2:6])
        self.assertEqual(values.dtype, np.uint8)
        self.assertEqual(values.shape, (4, 4))
        alpha = effective_alpha(layers)
        self.assertGreater(self.avatar.mask_list_cycle[0][0, 0], 0)
        self.assertEqual(alpha[0, 0], 0)
        self.assertEqual(np.count_nonzero(alpha), 16)

    def test_shared_cache_arrays_and_writeability_flags_are_unchanged(self):
        avatar = self.avatar
        original = avatar.frame_list_cycle[0].copy()
        original_mask = avatar.mask_list_cycle[0].copy()
        plan = avatar._compose_plan_cycle[0]
        plan_arrays = {key: value.copy() for key, value in plan.items()
                       if isinstance(value, np.ndarray)}
        plan_flags = {key: value.flags.writeable for key, value in plan.items()
                      if isinstance(value, np.ndarray)}
        for cycle_index in (0, 1, 5):
            layers = avatar.compose_frame(self.face, cycle_index, return_layers=True)
            self.assertTrue(np.shares_memory(layers['raw'], avatar.frame_list_cycle[0]))
            self.assertFalse(layers['raw'].flags.writeable)
            self.assertFalse(layers['alpha']['values'].flags.writeable)
            self.assertTrue(np.shares_memory(layers['alpha']['values'], plan['alpha_u8']))
            with self.assertRaises(ValueError):
                layers['raw'][0, 0] = 0
            with self.assertRaises(ValueError):
                layers['alpha']['values'][0, 0] = 0
            layers['composed'][:] = 255
        np.testing.assert_array_equal(avatar.frame_list_cycle[0], original)
        np.testing.assert_array_equal(avatar.frame_list_cycle[1], original)
        np.testing.assert_array_equal(avatar.mask_list_cycle[0], original_mask)
        self.assertTrue(avatar.frame_list_cycle[0].flags.writeable)
        self.assertTrue(avatar.mask_list_cycle[0].flags.writeable)
        for key, value in plan_arrays.items():
            np.testing.assert_array_equal(plan[key], value)
            self.assertEqual(plan[key].flags.writeable, plan_flags[key])

    def test_alternate_background_is_owned_after_resize_channels_and_dtype_normalization(self):
        alternate = np.linspace(-30, 310, 4 * 5 * 4, dtype=np.float32).reshape(4, 5, 4)
        saved = alternate.copy()
        expected_raw = np.clip(cv2.resize(alternate, (8, 8),
                                          interpolation=cv2.INTER_LINEAR)[:, :, :3], 0, 255).astype(np.uint8)
        layers = self.avatar.compose_frame(self.face, 0, alternate, return_layers=True)
        default = self.avatar.compose_frame(self.face, 0, alternate)
        np.testing.assert_array_equal(layers['raw'], expected_raw)
        np.testing.assert_array_equal(layers['composed'], default)
        np.testing.assert_array_equal(alternate, saved)
        self.assertFalse(np.shares_memory(layers['raw'], alternate))
        self.assertFalse(np.shares_memory(layers['raw'], layers['composed']))
        self.assertFalse(layers['raw'].flags.writeable)
        alternate[:] = 123
        layers['composed'][:] = 0
        np.testing.assert_array_equal(layers['raw'], expected_raw)

    def test_alternate_same_shape_does_not_alias_callers_image(self):
        alternate = np.full((8, 8, 3), 73, dtype=np.uint8)
        layers = self.avatar.compose_frame(self.face, 0, alternate, return_layers=True)
        self.assertFalse(np.shares_memory(layers['raw'], alternate))
        alternate[:] = 4
        np.testing.assert_array_equal(layers['raw'], np.full((8, 8, 3), 73, np.uint8))

    def test_invalid_alternate_falls_back_to_exact_prepared_raw(self):
        for alternate in (np.zeros((8, 8), np.uint8), np.zeros((8, 8, 2), np.uint8)):
            with self.subTest(shape=alternate.shape):
                layers = self.avatar.compose_frame(self.face, 0, alternate, return_layers=True)
                np.testing.assert_array_equal(layers['raw'], self.avatar.frame_list_cycle[0])
                np.testing.assert_array_equal(layers['composed'], self.avatar.compose_frame(self.face, 0))
                self.assertTrue(np.shares_memory(layers['raw'], self.avatar.frame_list_cycle[0]))

    def test_no_op_plan_returns_empty_sparse_support_and_unchanged_body(self):
        avatar = make_avatar(empty=True)
        layers = avatar.compose_frame(self.face, 0, return_layers=True)
        self.assertEqual(layers['alpha']['bounds'], (0, 0, 0, 0))
        self.assertEqual(layers['alpha']['values'].shape, (0, 0))
        self.assertFalse(layers['alpha']['values'].flags.writeable)
        np.testing.assert_array_equal(layers['composed'], layers['raw'])
        np.testing.assert_array_equal(layers['composed'], avatar.compose_frame(self.face, 0))

    def test_clipped_negative_crop_exposes_global_effective_bounds(self):
        avatar = make_avatar(bbox=(-2, 1, 6, 7), crop=(-3, -2, 8, 9))
        layers = avatar.compose_frame(self.face, 1, return_layers=True)
        self.assertEqual(layers['alpha']['bounds'], (0, 1, 6, 7))
        np.testing.assert_array_equal(layers['alpha']['values'], avatar.mask_list_cycle[0][3:9, 3:9])
        expected = blending.get_image_blending(
            avatar.frame_list_cycle[0].copy(), cv2.resize(self.face, (8, 6)),
            avatar.coord_list_cycle[0], avatar.mask_list_cycle[0], avatar.mask_coords_list_cycle[0])
        np.testing.assert_array_equal(layers['composed'], expected)

    def test_default_does_not_build_layer_arrays(self):
        with patch('numpy.empty', side_effect=AssertionError('Default allocated layer alpha')):
            default = self.avatar.compose_frame(self.face.astype(np.float32), 0)
        self.assertIsInstance(default, np.ndarray)


if __name__ == '__main__':
    unittest.main()
