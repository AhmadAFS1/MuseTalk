"""Opt-in candidate for one incoming eyelid appearance during a motion bridge.

Source landmarks must be verified against the source frames by the caller. This
module does not detect faces, modify routing, or claim perceptual acceptance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from scripts.motion_transitions import flow_blend


def validate_eye_points(points, width, height):
    """Return independent float32 contour arrays; reject unusable metadata."""
    if not isinstance(points, Mapping) or set(points) != {"left", "right"}:
        raise ValueError("Eye metadata requires exactly left and right contours")
    output = {}
    for eye in ("left", "right"):
        try:
            raw = np.asarray(points[eye])
            if raw.dtype.kind not in "fiu":
                raise ValueError("Eye coordinates must be numeric")
            contour = np.array(raw, dtype=np.float32, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid eye coordinates") from exc
        if contour.shape != (6, 2) or not np.isfinite(contour).all():
            raise ValueError("Each eye requires six finite pixel coordinate pairs")
        if (contour[:, 0].min() < 0 or contour[:, 0].max() >= width
                or contour[:, 1].min() < 0 or contour[:, 1].max() >= height):
            raise ValueError("Eye coordinates lie outside the source image")
        # A closed eye may be nearly collinear. Do not reject its zero aperture.
        if np.ptp(contour[:, 0]) < 2 or len(np.unique(contour, axis=0)) < 3:
            raise ValueError("Eye contour has no meaningful horizontal extent")
        output[eye] = contour
    if np.linalg.norm(output["left"].mean(0) - output["right"].mean(0)) < 4:
        raise ValueError("Eye contours are not distinct")
    return output


def _geometry(old_points, new_points, width, height):
    """Local mask with the same support/feather as the source experiment."""
    import cv2

    contours = [np.concatenate([old_points[eye], new_points[eye]])
                for eye in ("left", "right")]
    points = np.concatenate(contours)
    span = max(float(np.ptp(old_points[eye][:, 0])) for eye in ("left", "right"))
    pad = max(4, int(round(span * .20)))
    sigma = max(1., span * .08)
    extra = int(math.ceil(pad + sigma * 3 + 16))
    x0 = max(0, int(np.floor(points[:, 0].min())) - extra)
    x1 = min(width, int(np.ceil(points[:, 0].max())) + extra)
    y0 = max(0, int(np.floor(points[:, 1].min())) - extra)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + extra)
    # Extra context makes local dilation/blur see the same zeros or true image
    # boundary as the original full-image mask. Gaussian auto-kernel support for
    # float32 is about 4sigma. This deliberately overbounds that support.
    halo = pad + int(math.ceil(sigma * 4)) + 4
    mx0, my0 = max(0, x0 - halo), max(0, y0 - halo)
    mx1, my1 = min(width, x1 + halo), min(height, y1 + halo)
    mask = np.zeros((my1 - my0, mx1 - mx0), np.float32)
    for contour in contours:
        # Round in global coordinates before translation to preserve pixel ties.
        local = np.round(contour).astype(np.int32) - np.array([mx0, my0])
        cv2.fillConvexPoly(mask, cv2.convexHull(local), 1.)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    mask = mask[y0 - my0:y1 - my0, x0 - mx0:x1 - mx0]
    return (x0, y0, x1, y1), mask


def incoming_eye_blend(old, new, progress, old_points, new_points):
    """Keep the global bridge and refine eyes with warped incoming pixels only.

    Both landmark objects contain left/right six-point contours in source pixel
    coordinates. Inputs are never modified. The caller owns hash binding and
    fallback policy; invalid metadata raises ValueError rather than silently
    selecting unverified geometry.
    """
    import cv2

    if (not isinstance(old, np.ndarray) or not isinstance(new, np.ndarray)
            or old.dtype != np.uint8 or new.dtype != np.uint8
            or old.shape != new.shape or old.ndim != 3 or old.shape[2] != 3
            or min(old.shape[:2]) < 32):
        raise ValueError("Eye bridge requires matching uint8 BGR frames at least 32x32")
    if isinstance(progress, (bool, np.bool_)):
        raise ValueError("Bridge progress must be a finite number")
    try:
        t = float(progress)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Bridge progress must be a finite number") from exc
    if not math.isfinite(t):
        raise ValueError("Bridge progress must be finite")
    height, width = old.shape[:2]
    a_points = validate_eye_points(old_points, width, height)
    b_points = validate_eye_points(new_points, width, height)
    if t <= 0:
        return old.copy()
    if t >= 1:
        return new.copy()
    baseline = flow_blend(old, new, t)
    (x0, y0, x1, y1), mask = _geometry(a_points, b_points, width, height)
    a = old[y0:y1, x0:x1]
    b = new[y0:y1, x0:x1]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    backward = dis.calc(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    backward = np.clip(backward, -32, 32)
    yy, xx = np.indices(a.shape[:2], dtype=np.float32)
    incoming = cv2.remap(b, xx - (1 - t) * backward[..., 0],
                         yy - (1 - t) * backward[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    weight = mask[..., None]
    baseline[y0:y1, x0:x1] = np.clip(
        baseline[y0:y1, x0:x1].astype(np.float32) * (1 - weight)
        + incoming.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    return baseline
