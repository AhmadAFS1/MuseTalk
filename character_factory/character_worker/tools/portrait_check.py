#!/usr/bin/env python3
"""Validate an uploaded portrait before spending GPU time on it (run in the MuseTalk venv).

Checks: the file decodes as an image, is large enough for the 512x896 anchor, is roughly
portrait-shaped, and contains exactly one frontal face of a plausible size and position.
Writes a JSON report; exit 0 ok, 2 rejected, 1 tool error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f'portrait_check needs OpenCV and Pillow: {exc}')


def check(path: Path, min_width: int = 512, min_height: int = 512, recommended=(768, 1344)) -> dict:
    report = {'portrait': str(path), 'errors': [], 'warnings': []}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            width, height = im.size
            rgb = np.asarray(im)
    except Exception as exc:
        report['errors'].append(f'not a decodable image: {exc}')
        report['ok'] = False
        return report
    report.update(width=width, height=height)
    if width < min_width or height < min_height:
        report['errors'].append(f'too small: {width}x{height} (minimum {min_width}x{min_height})')
    elif width < recommended[0] or height < recommended[1]:
        report['warnings'].append(f'smaller than the recommended {recommended[0]}x{recommended[1]}; the 512x896 anchor will be upscaled')
    aspect = width / height
    if abs(aspect - 9 / 16) > 0.25 * 9 / 16:
        report['warnings'].append(f'aspect {aspect:.3f} is far from 9:16; the anchor is a centre crop, so the head may be cut')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 640 / max(width, height)
    small = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(small, scaleFactor=1.08, minNeighbors=5, minSize=(40, 40))
    s = 1 / scale if scale < 1 else 1.0
    boxes = [[int(x * s), int(y * s), int(w * s), int(h * s)] for x, y, w, h in faces]
    report['faces'] = boxes
    if len(boxes) == 0:
        report['errors'].append('no frontal face detected')
    elif len(boxes) > 1:
        # Haar sometimes doubles up on one face; only distinct boxes count.
        boxes.sort(key=lambda b: -b[2] * b[3])
        main = boxes[0]
        distinct = [b for b in boxes[1:] if abs(b[0] + b[2] / 2 - (main[0] + main[2] / 2)) > main[2] * 0.6 and b[3] > 0.4 * main[3]]
        if distinct:
            report['errors'].append(f'{1 + len(distinct)} faces detected; one person only')
    if boxes and not report['errors']:
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        face_fraction = h / height
        center_x = (x + w / 2) / width
        report.update(face_height_fraction=round(face_fraction, 4), face_center_x_fraction=round(center_x, 4))
        if not 0.12 <= face_fraction <= 0.6:
            report['errors'].append(f'face height is {face_fraction:.2f} of the image; expected a head-and-shoulders portrait (0.12-0.6)')
        if not 0.3 <= center_x <= 0.7:
            report['errors'].append(f'face centre at {center_x:.2f} of the width; the centre crop would cut it')
        mean_luma = float(gray.mean())
        if mean_luma < 40 or mean_luma > 225:
            report['warnings'].append(f'unusual exposure (mean luma {mean_luma:.0f})')
    report['ok'] = not report['errors']
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('portrait', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    try:
        report = check(args.portrait)
    except Exception as exc:
        print(json.dumps({'ok': False, 'errors': [f'tool error: {exc}']}), file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    print(text)
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
