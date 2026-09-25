"""CPU candidate regression tests, not source-bank perceptual acceptance."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.motion_eye_blend import incoming_eye_blend, validate_eye_points, _geometry
from scripts.measure_motion_eyes import measure_motion_eyes, LEFT, RIGHT
from scripts.motion_transitions import IDLE, TALK, SMILE, file_hash


def points(dx=0, closed=False):
    a = np.array([[70, 90], [78, 84], [92, 84], [100, 90], [92, 96], [78, 96]], float)
    if closed:
        a[:, 1] = 90
    a[:, 0] += dx
    return {"left": a.tolist(), "right": (a + [90, 0]).tolist()}


class ConstantFlow:
    def setFinestScale(self, scale):
        pass

    def calc(self, a, b, unused):
        result = np.zeros((*a.shape, 2), np.float32)
        result[..., 0] = -4
        return result


class EyeBlendTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.old = np.full((256, 256, 3), 100, np.uint8)
        self.new = np.full_like(self.old, 180)
        self.points = points()

    def test_endpoints_are_exact_independent_copies(self):
        for progress, expected in ((0, self.old), (1, self.new), (-.1, self.old), (1.1, self.new)):
            actual = incoming_eye_blend(self.old, self.new, progress, self.points, self.points)
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.shares_memory(actual, expected))

    def test_closed_eyes_are_valid(self):
        closed = points(closed=True)
        validated = validate_eye_points(closed, 256, 256)
        self.assertEqual(np.ptp(validated["left"][:, 1]), 0)
        result = incoming_eye_blend(self.old, self.new, .5, closed, self.points)
        self.assertEqual(result.shape, self.old.shape)

    def test_incoming_aperture_has_no_old_lid_color(self):
        # A new blue aperture moves4px right while the old lid is red. Constant
        # backward flow puts incoming eyes2px right at alpha.5. No old-lid color
        # is allowed in the full-weight aperture, even though globalfade has it.
        old = np.zeros((256, 256, 3), np.uint8)
        new = np.zeros_like(old)
        cv2.line(old, (70, 90), (100, 90), (0, 0, 240), 3)
        cv2.ellipse(new, (89, 90), (12, 5), 0, 0, 360, (240, 0, 0), -1)
        baseline = ((old.astype(float) + new) * .5).astype(np.uint8)
        with patch('scripts.motion_eye_blend.flow_blend', return_value=baseline.copy()), \
             patch('cv2.DISOpticalFlow_create', return_value=ConstantFlow()):
            output = incoming_eye_blend(old, new, .5, points(), points(4))
        self.assertGreater(int(baseline[90, 87, 2]), 100)
        self.assertLessEqual(int(output[90, 87, 2]), 1)
        self.assertGreaterEqual(int(output[90, 87, 0]), 238)
        self.assertGreaterEqual(int(output[86, 87, 0]), 230)

    def test_exact_locality_and_no_input_mutation(self):
        old = self.old.copy(); new = self.new.copy(); metadata = copy.deepcopy(self.points)
        a = validate_eye_points(metadata, 256, 256)
        bounds, _ = _geometry(a, a, 256, 256)
        baseline = np.full_like(old, 140)
        with patch('scripts.motion_eye_blend.flow_blend', return_value=baseline.copy()):
            result = incoming_eye_blend(old, new, .5, metadata, metadata)
        x0, y0, x1, y1 = bounds
        outside = np.ones(old.shape[:2], bool); outside[y0:y1, x0:x1] = False
        np.testing.assert_array_equal(result[outside], baseline[outside])
        np.testing.assert_array_equal(old, self.old); np.testing.assert_array_equal(new, self.new)
        self.assertEqual(metadata, self.points)
        a['left'][:] = 0
        self.assertEqual(metadata, self.points)

    def test_invalid_metadata_is_rejected(self):
        variants = []
        for value in (None, [], {}, {'left': self.points['left']}, {**self.points, 'extra': []}):
            variants.append(value)
        for contour in (self.points['left'][:5], [[0, 0]] * 6, [['1', '2']] * 6,
                        [[float('nan'), 90]] * 6, [[-1, 90]] * 6,
                        [[256, 90]] * 6, [[True, False]] * 6):
            variants.append({'left': contour, 'right': self.points['right']})
        variants.append({'left': self.points['left'], 'right': self.points['left']})
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                incoming_eye_blend(self.old, self.new, .5, value, self.points)

    def test_invalid_image_and_progress_are_rejected(self):
        for old, new in ((self.old.astype(float), self.new), (self.old, self.new[:128]),
                         (self.old[:, :, 0], self.new[:, :, 0]),
                         (self.old[:16, :16], self.new[:16, :16])):
            with self.assertRaises(ValueError):
                incoming_eye_blend(old, new, .5, self.points, self.points)
        for t in (float('nan'), float('inf'), True, None, 'bad'):
            with self.subTest(t=t), self.assertRaises(ValueError):
                incoming_eye_blend(self.old, self.new, t, self.points, self.points)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / 'source.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 24, (256, 256))
        if not writer.isOpened():
            self.fail('OpenCV MP4 writer unavailable')
        for n in range(3):
            writer.write(np.full((256, 256, 3), 70 + n * 15, np.uint8))
        writer.release()
        self.atlas = self.root / 'atlas.json'
        self.bank = {'version': 1, 'fps': 24, 'sources': {pose: {'path': 'source.mp4',
                    'sha256': file_hash(self.video), 'width': 256, 'height': 256,
                    'frame_count': 3} for pose in (IDLE, TALK, SMILE)}}
        self.atlas.write_text(json.dumps(self.bank))

    @staticmethod
    def mesh_factory(**kwargs):
        class Mesh:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def process(self, image):
                landmarks = [types.SimpleNamespace(x=.5, y=.5) for _ in range(478)]
                p = points()
                for eye, indices in (('left', LEFT), ('right', RIGHT)):
                    for index, xy in zip(indices, p[eye]):
                        landmarks[index] = types.SimpleNamespace(x=xy[0]/256, y=xy[1]/256)
                return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=landmarks)])
        return Mesh()

    def test_every_pose_frame_is_source_bound(self):
        result = measure_motion_eyes(self.atlas, mesh_factory=self.mesh_factory)
        self.assertEqual(result['method'], 'incoming_roi_v1')
        self.assertEqual(result['atlas_sha256'], file_hash(self.atlas))
        self.assertEqual(set(result['frames']), {IDLE, TALK, SMILE})
        for pose in (IDLE, TALK, SMILE):
            self.assertEqual(result['source_hashes'][pose], file_hash(self.video))
            self.assertEqual(len(result['frames'][pose]), 3)
            self.assertEqual(result['sources'][pose]['frame_count'], 3)
            self.assertEqual(result['sources'][pose]['fps'], 24)
            self.assertEqual(result['frames'][pose][0], points())

    def test_wrong_hash_geometry_fps_or_frame_count_rejected(self):
        cases = [('sha256', '0'*64), ('width', 128), ('frame_count', 4)]
        for field, value in cases:
            bank = copy.deepcopy(self.bank); bank['sources'][IDLE][field] = value
            self.atlas.write_text(json.dumps(bank))
            with self.subTest(field=field), self.assertRaises(ValueError):
                measure_motion_eyes(self.atlas, mesh_factory=self.mesh_factory)
        bank = copy.deepcopy(self.bank); bank['fps'] = 20; self.atlas.write_text(json.dumps(bank))
        with self.assertRaises(ValueError):
            measure_motion_eyes(self.atlas, mesh_factory=self.mesh_factory)

    def test_missing_face_rejected(self):
        def missing(**kwargs):
            mesh = self.mesh_factory(**kwargs)
            mesh.process = lambda image: types.SimpleNamespace(multi_face_landmarks=[])
            return mesh
        with self.assertRaisesRegex(ValueError, 'Missing face'):
            measure_motion_eyes(self.atlas, mesh_factory=missing)


if __name__ == '__main__':
    unittest.main()
