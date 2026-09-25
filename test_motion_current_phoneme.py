"""CPU contracts for current-phoneme bridges; no perceptual-acceptance claim."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.motion_current_phoneme import (
    current_phoneme_blend, validate_alpha, _maps, _similarity, _apply,
)
from scripts.motion_eye_blend import validate_eye_points


def points(dx=0, closed=False):
    p = np.array([[60, 70], [68, 64], [82, 64], [90, 70], [82, 76], [68, 76]], float)
    p[:, 0] += dx
    if closed:
        p[:, 1] = 70
    return {'left': p.tolist(), 'right': (p + [90, 0]).tolist()}


def alpha(value=255):
    return {'bounds': (60, 125, 195, 230),
            'values': np.full((105, 135), value, np.uint8)}


def identity_maps(old, new, a, b, t, op, np_):
    yy, xx = np.indices(old.shape[:2], np.float32)
    grid = np.dstack([xx, yy])
    return grid, grid.copy(), np.zeros(old.shape[:2], np.float32)


class CurrentPhonemeTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.old = np.full((256, 256, 3), 80, np.uint8)
        self.new = np.full_like(self.old, 100)
        self.comp = self.new.copy()
        self.comp[125:230, 60:195] = 200
        self.mask = alpha()
        self.points = points()

    def blend(self, t=.5, **kwargs):
        args = dict(old_raw=self.old, new_raw=self.new, current_composed=self.comp,
                    old_alpha=self.mask, new_alpha=self.mask, t=t,
                    old_eye_points=self.points, new_eye_points=self.points)
        args.update(kwargs)
        return current_phoneme_blend(**args)

    def test_final_endpoint_is_exact_independent_copy(self):
        for t in (1, 1.5):
            result = self.blend(t)
            np.testing.assert_array_equal(result, self.comp)
            self.assertFalse(np.shares_memory(result, self.comp))

    def test_current_core_is_full_strength_including_zero_progress(self):
        with patch('scripts.motion_current_phoneme._maps', side_effect=identity_maps):
            for t in (-1, 0, .05, .5, .95):
                result = self.blend(t)
                np.testing.assert_array_equal(result[125:230, 60:195], self.comp[125:230, 60:195])
            at_zero = self.blend(0)
        np.testing.assert_array_equal(at_zero[:50], self.old[:50])
        self.assertFalse(np.array_equal(at_zero, self.old))

    def test_soft_parser_alpha_is_applied_once(self):
        mask = alpha(128)
        composed = self.new.copy()
        composed[125:230, 60:195] = (200 * 128 + 100 * 127) // 255
        with patch('scripts.motion_current_phoneme._maps', side_effect=identity_maps):
            result = self.blend(.25, current_composed=composed, old_alpha=mask, new_alpha=mask)
        # Body85, composed150;150+(127/255)*(85-100)=142.529=>142.
        self.assertEqual(int(result[160, 120, 0]), 142)
        self.assertEqual(int(result[20, 20, 0]), 85)
        # Multiplying composed by parser alpha again would give117 instead.
        self.assertNotEqual(int(result[160, 120, 0]), 117)

    def test_old_generated_phoneme_has_no_argument_or_hidden_dependency(self):
        # Raw body color is permitted to differ; full current core cannot retain it.
        with patch('scripts.motion_current_phoneme._maps', side_effect=identity_maps):
            a = self.blend(.05)
            old_other = self.old.copy(); old_other[125:230, 60:195] = [0, 255, 0]
            b = self.blend(.05, old_raw=old_other)
        np.testing.assert_array_equal(a[125:230, 60:195], b[125:230, 60:195])

    def test_current_mouth_and_mask_move_with_incoming_head_geometry(self):
        comp = self.new.copy()
        comp[160:168, 120:128] = [240, 5, 5]
        result = self.blend(.5, current_composed=comp, old_eye_points=points(-8))
        # Eye corners put incoming face4px left. An unwarped paste would be wrong.
        np.testing.assert_array_equal(result[162, 117], [240, 5, 5])
        np.testing.assert_array_equal(result[162, 126], [100, 100, 100])

    def test_single_incoming_eye_survives_opposite_old_blink(self):
        old = self.old.copy(); new = self.new.copy()
        cv2.line(old, (60, 70), (90, 70), (0, 0, 240), 3)
        cv2.ellipse(new, (75, 70), (12, 5), 0, 0, 360, (240, 0, 0), -1)
        composed = new.copy(); composed[125:230, 60:195] = 200
        result = self.blend(0, old_raw=old, new_raw=new, current_composed=composed,
                            old_eye_points=points(closed=True))
        np.testing.assert_array_equal(result[70, 75], [240, 0, 0])
        self.assertGreaterEqual(int(result[67, 75, 0]),239)
        self.assertLessEqual(int(result[67, 75, 1:].max()),1)

    def test_eye_corner_similarity_preserves_normalized_lip_geometry(self):
        p = np.array([[60, 70], [90, 70], [150, 70], [180, 70]], np.float32)
        angle = .06; scale = 1.05
        matrix = np.array([[scale*np.cos(angle), -scale*np.sin(angle), 2],
                           [scale*np.sin(angle), scale*np.cos(angle), -4]], np.float32)
        moved = _apply(matrix, p)
        fitted = _similarity(p, moved)
        linear = fitted[:, :2]
        np.testing.assert_allclose(linear.T @ linear, np.eye(2) * scale**2, atol=1e-5)
        lips = np.array([[110,160], [125,155], [140,160], [125,175]], np.float32)
        transformed = _apply(fitted, lips)
        np.testing.assert_allclose(np.linalg.norm(transformed[1]-transformed[3])/
                                   np.linalg.norm(transformed[0]-transformed[2]),
                                   np.linalg.norm(lips[1]-lips[3])/
                                   np.linalg.norm(lips[0]-lips[2]), atol=1e-6)

    def test_sparse_empty_masks_are_valid_and_do_not_cover_body(self):
        empty = {'bounds': (0,0,0,0), 'values': np.empty((0,0),np.uint8)}
        with patch('scripts.motion_current_phoneme._maps', side_effect=identity_maps):
            result = self.blend(.5, current_composed=self.new, old_alpha=empty, new_alpha=empty)
        np.testing.assert_array_equal(result, np.full_like(self.new,90))

    def test_inputs_readonly_and_metadata_unchanged(self):
        arrays = (self.old,self.new,self.comp,self.mask['values'])
        snapshots = [x.copy() for x in arrays]
        metadata = copy.deepcopy(self.points)
        for a in arrays:a.flags.writeable=False
        result = self.blend(.3)
        self.assertTrue(result.flags.writeable)
        for a,b in zip(arrays,snapshots):np.testing.assert_array_equal(a,b)
        self.assertEqual(self.points,metadata)
        bounds, values = validate_alpha(self.mask,256,256)
        self.assertIs(values,self.mask['values'])
        self.assertIs(bounds,self.mask['bounds'])

    def test_invalid_alpha_rejected_before_endpoint(self):
        bad = [None,{}, {'bounds': (1,1,1,1), 'values':np.zeros((0,0),np.uint8)},
               {'bounds': (0,0,3,3), 'values':np.zeros((3,3),float)},
               {'bounds': (0,0,3,3), 'values':np.zeros((3,3,1),np.uint8)},
               {'bounds': (-1,0,2,3), 'values':np.zeros((3,3),np.uint8)},
               {'bounds': (0,0,257,3), 'values':np.zeros((3,257),np.uint8)},
               {'bounds': (False,0,3,3), 'values':np.zeros((3,3),np.uint8)},
               {'bounds': (0.,0,3,3), 'values':np.zeros((3,3),np.uint8)},
               {'bounds': (0,0,3,3), 'values':np.zeros((2,3),np.uint8)},
               {'bounds': (0,0,0,0), 'values':np.zeros((0,0),np.uint8), 'extra':0}]
        for mask in bad:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.blend(1,new_alpha=mask)

    def test_invalid_frames_progress_and_points(self):
        for key,value in [('old_raw',self.old.astype(float)),('new_raw',self.new[:20]),
                          ('current_composed',self.comp[:,:,0]),
                          ('old_eye_points',{}),('new_eye_points',None)]:
            with self.subTest(key=key), self.assertRaises(ValueError):self.blend(**{key:value})
        for t in [True,None,float('nan'),float('inf'),'0.5']:
            with self.subTest(t=t), self.assertRaises(ValueError):self.blend(t)

    def test_degenerate_similarity_rejected(self):
        with self.assertRaises(ValueError):
            _similarity(np.ones((4,2),np.float32),np.ones((4,2),np.float32))


if __name__ == '__main__':unittest.main()
