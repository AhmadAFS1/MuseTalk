"""Opt-in active-speech bridge preserving one current phoneme and eye appearance.

Raw source frames determine body flow. A corner-derived similarity map replaces
that flow across both prepared face supports and eye supports. Every incoming
layer uses the same map, so the current mouth is neither pasted at an unrelated
position nor deformed toward the raw LTX mouth. This module detects no faces and
owns no caches; callers must bind eye metadata and prepared layers to sources.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping

import numpy as np

from scripts.motion_eye_blend import _geometry, validate_eye_points


def validate_alpha(alpha, height, width):
    """Validate a compact effective parser mask, returning bounds/values unchanged.

    Bounds are integer (x0,y0,x1,y1), with uint8 values of exactly that ROI shape.
    The sole empty representation is bounds=(0,0,0,0), values.shape=(0,0).
    This function allocates no full-frame mask and does not modify its inputs.
    """
    if not isinstance(alpha, Mapping) or set(alpha) != {"bounds", "values"}:
        raise ValueError("Current-phoneme alpha requires bounds and values")
    bounds, values = alpha["bounds"], alpha["values"]
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 4
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Integral)
                   for x in bounds)):
        raise ValueError("Alpha bounds require four integers")
    x0, y0, x1, y1 = bounds
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint8
            or values.ndim != 2 or values.shape != (y1 - y0, x1 - x0)):
        raise ValueError("Alpha values must be a matching two-dimensional uint8 ROI")
    if tuple(bounds) == (0, 0, 0, 0):
        return bounds, values
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Alpha bounds lie outside the frame or have empty extent")
    return bounds, values


def _full_alpha(alpha, height, width):
    bounds, values = validate_alpha(alpha, height, width)
    x0, y0, x1, y1 = bounds
    result = np.zeros((height, width), np.float32)
    result[y0:y1, x0:x1] = values.astype(np.float32) / 255.0
    return result


def _remap(array, mapping):
    import cv2
    return cv2.remap(array, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _similarity(source, target):
    """Least-squares orientation-preserving similarity; no shear/anisotropy."""
    x, y = source[:, 0], source[:, 1]
    matrix = np.zeros((2 * len(source), 4), np.float64)
    matrix[0::2] = np.stack([x, -y, np.ones(len(x)), np.zeros(len(x))], axis=1)
    matrix[1::2] = np.stack([y, x, np.zeros(len(x)), np.ones(len(x))], axis=1)
    fit, _, rank, _ = np.linalg.lstsq(matrix, target.reshape(-1), rcond=None)
    a, b, tx, ty = fit
    if rank != 4 or not np.isfinite(fit).all() or a * a + b * b <= 1e-12:
        raise ValueError("Eye corners cannot determine a nondegenerate similarity")
    return np.array([[a, -b, tx], [b, a, ty]], np.float32)


def _apply(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


def _similarity_map(matrix, grid):
    import cv2
    inverse = cv2.invertAffineTransform(matrix)
    x, y = grid[..., 0], grid[..., 1]
    return np.stack([inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2],
                     inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]], axis=2)


def _dense_maps(old, new, progress, grid):
    import cv2
    height, width = old.shape[:2]
    size = (max(32, width // 4), max(32, height // 4))
    a = cv2.cvtColor(cv2.resize(old, size), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(new, size), cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    forward = dis.calc(a, b, None)
    backward = dis.calc(b, a, None)
    scale = np.array([width / size[0], height / size[1]], np.float32)
    forward = np.clip(cv2.resize(forward, (width, height)) * scale, -32, 32)
    backward = np.clip(cv2.resize(backward, (width, height)) * scale, -32, 32)
    return grid - progress * forward, grid - (1 - progress) * backward


def _maps(old, new, old_alpha, new_alpha, progress, old_points, new_points):
    """One shared incoming map for raw body, prepared composite, mask and eyes."""
    import cv2
    height, width = old.shape[:2]
    yy, xx = np.indices((height, width), np.float32)
    grid = np.stack([xx, yy], axis=2)
    old_corners = np.concatenate([old_points[eye][[0, 3]] for eye in ("left", "right")])
    new_corners = np.concatenate([new_points[eye][[0, 3]] for eye in ("left", "right")])
    intermediate = old_corners * (1 - progress) + new_corners * progress
    left_matrix = _similarity(old_corners, intermediate)
    right_matrix = _similarity(new_corners, intermediate)
    left_similarity = _similarity_map(left_matrix, grid)
    right_similarity = _similarity_map(right_matrix, grid)
    old_eyes = {eye: _apply(left_matrix, old_points[eye]) for eye in ("left", "right")}
    new_eyes = {eye: _apply(right_matrix, new_points[eye]) for eye in ("left", "right")}
    (x0, y0, x1, y1), local_eye = _geometry(old_eyes, new_eyes, width, height)
    eye_weight = np.zeros((height, width), np.float32)
    eye_weight[y0:y1, x0:x1] = local_eye
    # All pasted-face and eye pixels use similarity; the blend to dense flow is
    # outside this support, including its soft parser and eye-mask feathers.
    support = ((_remap(old_alpha, left_similarity) > 0)
               | (_remap(new_alpha, right_similarity) > 0)
               | (eye_weight > 0)).astype(np.uint8)
    support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    distance = cv2.distanceTransform(1 - support, cv2.DIST_L2, 5)
    region = (1 - np.clip(distance / 20, 0, 1))[..., None]
    left, right = _dense_maps(old, new, progress, grid)
    left = left * (1 - region) + left_similarity * region
    right = right * (1 - region) + right_similarity * region
    return left, right, eye_weight


def current_phoneme_blend(old_raw, new_raw, current_composed, old_alpha, new_alpha,
                          t, old_eye_points, new_eye_points):
    """Bridge active speech with the current generated appearance from frame one.

    ``current_composed`` already contains the effective prepared parser alpha.
    After identical geometric remapping, the residual formula applies this alpha
    once: composed + (1-alpha) * (body - incoming_raw). Full parser-core pixels
    equal the warped current composition; eyes retain one incoming appearance.

    t>=1 returns an independent exact current-composed copy. t<=0 means current
    articulation positioned in old body geometry; it intentionally does NOT
    return the old composed image. This active-speech function must not be used
    as a generic terminal/idle fade. Existing eye-only blends remain separate.
    """
    images = (old_raw, new_raw, current_composed)
    if any(not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.ndim != 3
           or a.shape[2] != 3 for a in images):
        raise ValueError("Current-phoneme blend requires uint8 BGR frames")
    if any(a.shape != old_raw.shape for a in images) or min(old_raw.shape[:2]) < 32:
        raise ValueError("Current-phoneme frames must match and be at least 32x32")
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, numbers.Real) or not math.isfinite(t):
        raise ValueError("Current-phoneme progress must be a finite number")
    height, width = old_raw.shape[:2]
    validate_alpha(old_alpha, height, width)
    validate_alpha(new_alpha, height, width)
    old_points = validate_eye_points(old_eye_points, width, height)
    new_points = validate_eye_points(new_eye_points, width, height)
    if t >= 1:
        return current_composed.copy()
    progress = max(0.0, float(t))
    a = _full_alpha(old_alpha, height, width)
    b = _full_alpha(new_alpha, height, width)
    left, right, eye_weight = _maps(old_raw, new_raw, a, b, progress, old_points, new_points)
    incoming_raw = _remap(new_raw, right)
    body = np.clip(_remap(old_raw, left).astype(np.float32) * (1 - progress)
                   + incoming_raw.astype(np.float32) * progress, 0, 255).astype(np.uint8)
    body = np.clip(body.astype(np.float32) * (1 - eye_weight[..., None])
                   + incoming_raw.astype(np.float32) * eye_weight[..., None],
                   0, 255).astype(np.uint8)
    incoming_composed = _remap(current_composed, right)
    alpha = np.clip(_remap(b, right), 0, 1)[..., None]
    return np.clip(incoming_composed.astype(np.float32)
                   + (1 - alpha) * (body.astype(np.float32) - incoming_raw.astype(np.float32)),
                   0, 255).astype(np.uint8)
