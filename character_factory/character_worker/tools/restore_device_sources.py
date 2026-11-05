#!/usr/bin/env python3
"""Restore only verified, recipe-compatible accepted H3 work from a prior worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from types import SimpleNamespace

from publish_device_release import POSES, component, file_hash, write_json

SHA = re.compile(r'^[0-9a-f]{64}$')


def download(s3, bucket, ref, prefix, target, limit=1024 ** 3):
    key = ref.get('key')
    digest = ref.get('sha256')
    size = ref.get('bytes')
    if (not isinstance(key, str) or not key.startswith(prefix + '/') or '..' in PurePosixPath(key).parts
            or not SHA.fullmatch(digest or '') or not isinstance(size, int) or isinstance(size, bool)
            or not 0 < size <= limit):
        raise ValueError('invalid checkpoint object reference')
    print(f'restore {key}', flush=True)
    response = s3.get_object(Bucket=bucket, Key=key)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.partial')
    h, count = hashlib.sha256(), 0
    try:
        with tmp.open('wb') as out:
            for block in iter(lambda: response['Body'].read(1024 * 1024), b''):
                count += len(block)
                if count > size:
                    raise ValueError('checkpoint object exceeds expected size')
                h.update(block)
                out.write(block)
        if count != size or h.hexdigest() != digest:
            raise ValueError('checkpoint object hash/size mismatch')
        tmp.replace(target)
    finally:
        response['Body'].close()
        tmp.unlink(missing_ok=True)
    return target


def build_plan(spec, seed):
    workflow = Path(spec['repo']) / 'character_factory/h3_avatar_workflow'
    sys.path.insert(0, str(workflow))
    from batch_three_pose import plan
    entries = plan(SimpleNamespace(images=Path(spec['portrait']).parent, metadata=Path(spec['metadata']),
                                  output=Path(spec['out']), workspace=Path(spec['workspace']), limit=None,
                                  only=None, seed=seed))
    if len(entries) != 1:
        raise ValueError('source restore expected one planned character')
    return entries[0]


def validate_anchor(spec, anchor):
    from PIL import Image, ImageOps, ImageChops
    with Image.open(spec['portrait']) as portrait, Image.open(anchor) as saved:
        expected = ImageOps.fit(portrait.convert('RGB'), (512, 896), method=Image.Resampling.LANCZOS)
        if saved.size != expected.size or ImageChops.difference(expected, saved.convert('RGB')).getbbox() is not None:
            raise ValueError('checkpoint anchor differs from current portrait')


def validate_video(spec, source, pose):
    sys.path.insert(0, str(Path(spec['repo']) / 'character_factory/h3_avatar_workflow'))
    sys.path.insert(0, str(spec['h3_lib']))
    from common import check_video
    from certify_endpoints import decoded_endpoints
    media = check_video(source, frames=240 if pose != 'smiling' else 158, audio=pose == 'talking')
    if pose != 'talking' and any(s['codec_type'] == 'audio' for s in media['streams']):
        raise ValueError('silent checkpoint pose contains audio')
    proof = decoded_endpoints(source)
    if not proof.get('first_last_exact'):
        raise ValueError('checkpoint source endpoints differ')
    return proof


def restore(spec, s3, planner=build_plan, anchor_validator=validate_anchor, video_validator=validate_video):
    identity = spec['identity']
    for field in ('character_id', 'job_id', 'release_id'):
        component(identity[field])
    ref = spec['descriptor']
    if ref.get('bucket') != spec['bucket']:
        raise ValueError('checkpoint bucket mismatch')
    root = f'avatars/releases/{identity["release_id"]}'
    parts = str(ref.get('key') or '').split('/')
    if len(parts) != 5 or '/'.join(parts[:3]) != root or parts[-1] != 'source-checkpoint.json':
        raise ValueError('checkpoint is outside the expected release')
    component(parts[3])
    prefix = root + '/' + parts[3]
    staging = Path(spec['out']) / '_checkpoint'
    receipt_path = download(s3, spec['bucket'], ref, prefix, staging / 'receipt.json', 4 * 1024 ** 2)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('schema') != 'lingua-character-source-checkpoint/1'
            or any(receipt.get(k) != v for k, v in identity.items())
            or receipt.get('recipe_id') != spec['recipe_id'] or set(receipt.get('poses') or {}) != set(POSES)
            or receipt.get('qc', {}).get('status') != 'passed' or receipt.get('worker_job_id') != parts[3]):
        raise ValueError('checkpoint identity/recipe/QC mismatch')
    portrait_hash = file_hash(Path(spec['portrait']))
    if receipt.get('portrait', {}).get('sha256') != portrait_hash:
        raise ValueError('checkpoint portrait mismatch')
    provenance_path = download(s3, spec['bucket'], receipt['provenance'], prefix,
                               staging / 'provenance.json', 16 * 1024 ** 2)
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get('portrait_sha256') != portrait_hash or
            provenance.get('lock', {}).get('h3_workflows', {}).get('commit') != spec['h3_commit']):
        raise ValueError('checkpoint model/portrait provenance mismatch')
    h3 = provenance.get('h3') or {}
    if set(h3) != set(POSES) or any(h3[p].get('status') != 'complete' for p in POSES):
        raise ValueError('checkpoint generation records incomplete')
    seeds = {h3[p].get('seed') for p in POSES}
    if len(seeds) != 1 or not isinstance(next(iter(seeds)), int):
        raise ValueError('checkpoint sampler seed mismatch')
    seed = next(iter(seeds))
    entry = planner(spec, seed)
    if provenance.get('plan', {}).get('signature') != entry['signature']:
        raise ValueError('checkpoint recipe inputs changed')
    target = Path(entry['output'])
    anchor = download(s3, spec['bucket'], receipt['anchor'], prefix, target / 'anchor.png', 30 * 1024 ** 2)
    anchor_validator(spec, anchor)
    endpoints = set()
    for pose in POSES:
        source = download(s3, spec['bucket'], receipt['poses'][pose], prefix, target / pose / 'source.mp4')
        proof = video_validator(spec, source, pose)
        if not proof.get('first_last_exact'):
            raise ValueError('restored source endpoints differ')
        endpoints.add(proof.get('first_rgb_sha256'))
        record = {**h3[pose], 'endpoint_proof': proof, 'restored_from': ref,
                  'artifacts': {str(source.resolve()): file_hash(source), str(anchor.resolve()): file_hash(anchor)}}
        write_json(target / pose / 'h3.json', record)
    if len(endpoints) != 1 or None in endpoints:
        raise ValueError('restored poses do not share an endpoint')
    return {'identity_dir': str(target), 'seed': seed, 'source_checkpoint': ref}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    from s3_tool import client
    result = restore(json.loads(args.spec.read_text()), client())
    write_json(args.result, result)


if __name__ == '__main__':
    main()
