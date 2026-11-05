#!/usr/bin/env python3
"""Validate the complete phone-avatar-bundle/1 archive without extracting it."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import tarfile

MAX_BYTES = 2 * 1024 ** 3
MAX_FRAMES = 2048


def image_dimensions(data: bytes) -> tuple[int, int]:
    if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) >= 24:
        return struct.unpack('>II', data[16:24])
    if data.startswith(b'\xff\xd8'):
        pos = 2
        while pos < len(data):
            if data[pos] != 255:
                raise ValueError('malformed JPEG marker')
            while pos < len(data) and data[pos] == 255:
                pos += 1
            marker = data[pos]
            pos += 1
            if marker in (0xD8, 0xD9):
                continue
            length = struct.unpack('>H', data[pos:pos + 2])[0]
            if length < 2:
                raise ValueError('malformed JPEG segment')
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack('>HH', data[pos + 3:pos + 7])
                return w, h
            pos += length
    raise ValueError('unsupported or incomplete image')


def positive(value, name, maximum=100000):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f'invalid {name}')
    return value


def validate(bundle: Path, sidecar: dict, avatar_id: str | None = None) -> dict:
    size = bundle.stat().st_size
    if not 0 < size <= MAX_BYTES:
        raise ValueError('bundle size outside limit')
    digest = hashlib.sha256()
    with bundle.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    if sidecar.get('sha256') != digest.hexdigest() or sidecar.get('bytes') != size:
        raise ValueError('bundle hash/size disagrees with sidecar')
    if sidecar.get('format') != 'phone-avatar-bundle/1':
        raise ValueError('unsupported bundle format')
    with tarfile.open(bundle, mode='r:') as archive:
        members = archive.getmembers()
        names = set()
        files = {}
        total = 0
        for entry in members:
            path = PurePosixPath(entry.name)
            if (entry.name in names or path.is_absolute() or '..' in path.parts or '\\' in entry.name
                    or not (entry.isfile() or entry.isdir()) or str(path) != entry.name.rstrip('/')):
                raise ValueError('unsafe or duplicate archive entry')
            names.add(entry.name)
            total += entry.size
            if total > MAX_BYTES or len(names) > MAX_FRAMES * 2 + 8:
                raise ValueError('expanded bundle outside limits')
            if entry.isfile():
                files[entry.name] = entry

        def read(name):
            if name not in files:
                raise ValueError(f'missing bundle file {name}')
            return archive.extractfile(files[name]).read()

        if files.get('avatar.json') is None or files['avatar.json'].size > 4 * 1024 ** 2:
            raise ValueError('missing/oversized avatar metadata')
        meta = json.loads(read('avatar.json'))
        n = positive(meta.get('unique_frames'), 'unique_frames', MAX_FRAMES)
        w = positive(meta.get('frame_width'), 'frame_width', 4096)
        h = positive(meta.get('frame_height'), 'frame_height', 4096)
        mw = positive(meta.get('mask_width'), 'mask_width', 4096)
        mh = positive(meta.get('mask_height'), 'mask_height', 4096)
        positive(meta.get('fps'), 'fps', 60)
        cycle = positive(meta.get('cycle_length'), 'cycle_length', MAX_FRAMES * 2)
        if not isinstance(meta.get('ping_pong'), bool) or cycle != n * (2 if meta['ping_pong'] else 1):
            raise ValueError('invalid cycle mapping')
        expected_id = avatar_id or sidecar.get('avatar_id')
        if meta.get('avatar_id') != expected_id or sidecar.get('avatar_id') != expected_id:
            raise ValueError('avatar identity mismatch')
        for key, want in [('frames', n), ('width', w), ('height', h), ('fps', meta['fps']), ('cycle_length', cycle)]:
            if sidecar.get(key) != want:
                raise ValueError(f'sidecar {key} mismatch')
        for key in ('frames', 'masks', 'coords', 'mask_coords'):
            if not isinstance(meta.get(key), list) or len(meta[key]) != n:
                raise ValueError(f'invalid {key} count')
        if len(set(meta['frames'])) != n or len(set(meta['masks'])) != n:
            raise ValueError('duplicate image references')
        latent = meta.get('latents') or {}
        if latent.get('shape') != [n, 8, 32, 32] or latent.get('dtype') != 'float16' or latent.get('file') != 'latents_f16.bin':
            raise ValueError('incompatible latent layout')
        latent_data = read('latents_f16.bin')
        if len(latent_data) != n * 8 * 32 * 32 * 2:
            raise ValueError('latent byte length mismatch')
        if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<e', latent_data)):
            raise ValueError('non-finite latent')
        wanted = {'avatar.json', 'latents_f16.bin', *meta['frames'], *meta['masks']}
        if set(files) != wanted:
            raise ValueError('unexpected/missing archive files')
        for index in range(n):
            frame, mask = meta['frames'][index], meta['masks'][index]
            if not frame.startswith('frames/') or not mask.startswith('masks/'):
                raise ValueError('invalid image path')
            mask_dims = image_dimensions(read(mask))
            if any(not 0 < value <= 4096 for value in mask_dims):
                raise ValueError('mask dimensions outside limit')
            if image_dimensions(read(frame)) != (w, h) or (index == 0 and mask_dims != (mw, mh)):
                raise ValueError('image geometry mismatch')
            for key in ('coords', 'mask_coords'):
                box = meta[key][index]
                if (not isinstance(box, list) or len(box) != 4 or
                        any(isinstance(v, bool) or not isinstance(v, int) for v in box)):
                    raise ValueError('invalid coordinates')
                x, y, x1, y1 = box
                if x1 <= x or y1 <= y:
                    raise ValueError('empty crop')
                if key == 'coords' and not (0 <= x < x1 <= w and 0 <= y < y1 <= h):
                    raise ValueError('face crop outside frame')
                if key == 'mask_coords' and ((x1 - x, y1 - y) != mask_dims
                        or x1 <= 0 or y1 <= 0 or x >= w or y >= h):
                    raise ValueError('mask crop geometry mismatch')
        return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('sidecar', type=Path)
    args = parser.parse_args()
    meta = validate(args.bundle, json.loads(args.sidecar.read_text()))
    print(json.dumps({'valid': True, 'avatar_id': meta['avatar_id'], 'frames': meta['unique_frames']}))


if __name__ == '__main__':
    main()
