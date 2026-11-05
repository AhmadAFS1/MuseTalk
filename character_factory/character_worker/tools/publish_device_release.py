#!/usr/bin/env python3
"""Export local MuseTalk caches and commit an immutable, complete mobile release to S3.

Runs inside the MuseTalk venv. No Core ML model execution or GPU inference is needed.
The release JSON is the final upload, after every mandatory artifact is validated.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts' / 'ios'))
from validate_phone_bundle import validate

POSES = ('idle', 'talking', 'smiling')
SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$')


def component(value):
    if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value) or '..' in value:
        raise ValueError('invalid release identity component')
    return value


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def upload(s3, bucket, key, path):
    """Single conditional upload; retries may reuse only the same hash/size."""
    digest, size = file_hash(path), path.stat().st_size
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()
    print(f'upload {key} ({size} bytes)', flush=True)
    try:
        with path.open('rb') as f:
            s3.put_object(Bucket=bucket, Key=key, Body=f, ContentLength=size,
                          Metadata={'sha256': digest}, ChecksumSHA256=checksum, IfNoneMatch='*')
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
        if code not in ('PreconditionFailed', '412'):
            raise
    head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode='ENABLED')
    if (head.get('ContentLength') != size or head.get('Metadata', {}).get('sha256') != digest
            or head.get('ChecksumSHA256') != checksum):
        raise ValueError(f'uploaded object integrity mismatch: {key}')
    return {'key': key, 'sha256': digest, 'bytes': size}


def export(repo, cache, avatar_id, out):
    work = out / avatar_id
    subprocess.run([sys.executable, str(repo / 'scripts/ios/prepare_avatar_bundle.py'), '--avatar', str(cache),
                    '--phone-only', '--out', str(work)], check=True)
    subprocess.run([sys.executable, str(repo / 'scripts/ios/pack_phone_bundle.py'), str(work / 'avatar'),
                    '--avatar-id', avatar_id, '--out', str(out)], check=True)
    return out / f'{avatar_id}.tar', out / f'{avatar_id}.json'


def decode_images(tar):
    """Exercise the actual image decoder as well as archive/header validation."""
    from PIL import Image
    with tarfile.open(tar, 'r:') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith(('frames/', 'masks/')):
                with Image.open(io.BytesIO(archive.extractfile(member).read())) as im:
                    im.load()


def checkpoint(spec, s3, check_cancel=lambda: None):
    """Make accepted H3 work recoverable independently of MuseTalk preparation."""
    identity = spec['identity']
    for field in ('character_id', 'job_id', 'release_id'):
        component(identity[field])
    component(spec['worker_job_id'])
    if set(spec['poses']) != set(POSES) or not spec.get('qc', {}).get('pass'):
        raise ValueError('checkpoint requires all three poses and passing QC')
    prefix = f'avatars/releases/{identity["release_id"]}/{spec["worker_job_id"]}'
    out = Path(spec['out'])
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 'lingua-character-source-checkpoint/1', **identity,
               'worker_job_id': spec['worker_job_id'], 'recipe_id': spec['recipe_id'], 'poses': {}}
    for pose in POSES:
        check_cancel()
        row = spec['poses'][pose]
        source = Path(row['source'])
        if file_hash(source) != row['source_sha256']:
            raise ValueError('source hash changed before checkpoint')
        receipt['poses'][pose] = upload(s3, spec['bucket'], f'{prefix}/sources/{pose}-{row["source_sha256"]}.mp4', source)
    for name in ('portrait', 'anchor'):
        check_cancel()
        path = Path(spec[name])
        receipt[name] = upload(s3, spec['bucket'], f'{prefix}/{name}-{file_hash(path)}{path.suffix.lower()}', path)
    qc_path, provenance = out / 'source-qc.json', out / 'source-provenance.json'
    write_json(qc_path, spec['qc'])
    write_json(provenance, spec['provenance'])
    receipt['qc'] = {'status': 'passed', 'report': upload(s3, spec['bucket'], f'{prefix}/checkpoints/qc.json', qc_path)}
    receipt['provenance'] = upload(s3, spec['bucket'], f'{prefix}/checkpoints/provenance.json', provenance)
    check_cancel()
    path = out / 'source-checkpoint.json'
    write_json(path, receipt)
    ref = upload(s3, spec['bucket'], f'{prefix}/source-checkpoint.json', path)
    return {'bucket': spec['bucket'], **ref, 'release_id': identity['release_id']}


def publish(spec, s3, exporter=export, check_cancel=lambda: None, image_decoder=decode_images):
    identity = spec['identity']
    for field in ('character_id', 'job_id', 'release_id'):
        component(identity[field])
    component(spec['worker_job_id'])
    if set(spec['poses']) != set(POSES) or not spec.get('qc', {}).get('pass') or not spec.get('smoke', {}).get('pass'):
        raise ValueError('device release requires all three poses and passing QC/smoke')
    repo = Path(spec['repo'])
    out = Path(spec['out'])
    out.mkdir(parents=True, exist_ok=True)
    bucket = spec['bucket']
    prefix = f'avatars/releases/{identity["release_id"]}/{spec["worker_job_id"]}'
    manifest = {'schema': 'lingua-character-release/1', **identity, 'worker_job_id': spec['worker_job_id'],
                'renderer_abi': 'musetalk-v15-phone-v1', 'recipe_id': spec['recipe_id'], 'source_fps': 24,
                'preprocess_version': 'musetalk-v15-phone-v1', 'poses': {}, 'qc': {'status': 'passed'}}
    endpoint = None
    for pose in POSES:
        check_cancel()
        row = spec['poses'][pose]
        avatar_id = component(row['avatar_id'])
        tar, sidecar_path = exporter(repo, Path(row['cache']), avatar_id, out)
        sidecar = json.loads(sidecar_path.read_text())
        # The packer's wall-clock timestamp is not part of the archive. Omit it from
        # the immutable release sidecar so a restart can recreate identical bytes.
        sidecar.pop('created_at', None)
        write_json(sidecar_path, sidecar)
        meta = validate(tar, sidecar, avatar_id)
        image_decoder(tar)
        dims = (meta['frame_width'], meta['frame_height'])
        if endpoint is not None and endpoint != dims:
            raise ValueError('pose geometries do not agree')
        endpoint = dims
        source = Path(row['source'])
        if file_hash(source) != row['source_sha256']:
            raise ValueError('source video hash differs from preparation receipt')
        # All files are content-addressed. Retry after an interrupted upload is safe.
        check_cancel()
        bundle = upload(s3, bucket, f'{prefix}/phone/{pose}-{sidecar["sha256"]}.tar', tar)
        bundle.update({k: sidecar[k] for k in ('frames', 'fps', 'width', 'height', 'format', 'cycle_length', 'exporter_commit')})
        sidecar_ref = upload(s3, bucket, f'{prefix}/phone/{pose}-{sidecar["sha256"]}.json', sidecar_path)
        source_ref = upload(s3, bucket, f'{prefix}/sources/{pose}-{row["source_sha256"]}.mp4', source)
        source_ref.update(fps=24, frame_count=240 if pose != 'smiling' else 158,
                          duration_seconds=(240 if pose != 'smiling' else 158) / 24,
                          width=meta['frame_width'], height=meta['frame_height'])
        manifest['poses'][pose] = {'avatar_id': avatar_id, 'bundle': bundle, 'sidecar': sidecar_ref, 'source': source_ref}
    check_cancel()
    qc_path = out / 'qc.json'
    # The worker's CUDA smoke checks cache usability. It does not execute or qualify the
    # shared phone student; that evidence comes from its separate native device run.
    smoke_scope = {'scope': 'asset-cache-qc', 'renderer': 'musetalk-v15-cuda-worker', 'model_role': 'teacher',
                   'phone_student_acceptance': False}
    write_json(qc_path, {'pass': True, 'poses': spec['qc'], 'smoke': {**spec['smoke'], **smoke_scope}})
    manifest['qc']['report'] = upload(s3, bucket, f'{prefix}/qc/report.json', qc_path)
    provenance = out / 'provenance.json'
    write_json(provenance, {**spec['provenance'], 'asset_smoke': smoke_scope})
    manifest['provenance'] = upload(s3, bucket, f'{prefix}/provenance.json', provenance)
    for name in ('portrait', 'anchor'):
        path = Path(spec[name])
        digest = file_hash(path)
        manifest[name] = upload(s3, bucket, f'{prefix}/{name}-{digest}{path.suffix.lower()}', path)
    check_cancel()
    path = out / 'release.json'
    write_json(path, manifest)
    ref = upload(s3, bucket, f'{prefix}/release.json', path)
    return {'bucket': bucket, **ref, 'release_id': identity['release_id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    from s3_tool import client
    result = (checkpoint if args.checkpoint else publish)(json.loads(args.spec.read_text()), client())
    write_json(args.result, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
