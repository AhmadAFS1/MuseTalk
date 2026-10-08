#!/usr/bin/env python3
"""Copy audited caches into the owned API's separate writable cache root.

No downloads, preparation, pickle loading, GPU allocation, existing-file repair,
or production access. Every required byte is checked against the completed CPU
audit. Existing matching caches are read-only reuse; mismatches fail closed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

ROOT = Path('/workspace/MuseTalk')
SOURCE = ROOT / 'tmp/production_avatar_content_audit_1345/avatars'
TARGET = ROOT / 'results/v15/avatars'
AUDIT_SHA = '04eaf604d6d4f823f0ff3b45d193c9c21062c1ab7e63c4965d8a37da34005c88'
GPU_UUID = 'GPU-5640f670-debe-ec22-1cfb-4b1f63bc1d53'
FIXED = ('avator_info.json', 'input_video.mp4', 'latents.pt', 'coords.pkl', 'mask_coords.pkl')


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def no_links(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink_path_refused')


def inventory(base, row):
    no_links(base)
    require(base.is_dir(), 'cache_directory_missing')
    data = row['contents']
    require(data['metadata']['selected_fields'].get('video_layout') == 'single_video', 'unsupported_video_layout')
    expected = {'avator_info.json': data['metadata']['sha256'],
                'input_video.mp4': data['source_video']['sha256'],
                'latents.pt': data['latents']['sha256'],
                **{name: data['coordinate_pickles'][name]['sha256'] for name in FIXED[3:]}}
    for directory, key in (('full_imgs', 'frames'), ('mask', 'masks')):
        no_links(base / directory)
        paths = sorted((base / directory).glob('*.png'))
        count = data[key]['count']
        require(count > 0 and [p.name for p in paths] == [f'{i:08d}.png' for i in range(count)],
                'image_index_or_count_mismatch')
        digest = hashlib.sha256()
        for path in paths:
            no_links(path)
            require(path.is_file(), 'image_file_missing')
            checksum = sha(path)
            expected[path.relative_to(base).as_posix()] = checksum
            digest.update((path.name + '\0' + checksum + '\n').encode())
        require(digest.hexdigest() == data[key]['filename_content_inventory_sha256'], 'audited_image_hash_mismatch')
    for name in FIXED:
        no_links(base / name)
        require((base / name).is_file() and sha(base / name) == expected[name], 'audited_fixed_hash_mismatch')
    info = json.loads((base / 'avator_info.json').read_text())
    require(info.get('avatar_id') == row['avatar_id'] and info.get('version') == 'v15', 'cache_identity_mismatch')
    return expected


def inventory_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stage_one(row, source_root, target_root):
    require(not source_root.resolve().is_relative_to(target_root.resolve())
            and not target_root.resolve().is_relative_to(source_root.resolve()), 'cache_roots_must_be_disjoint')
    avatar = row['avatar_id']
    require(re.fullmatch('[A-Za-z0-9_-]+', avatar) is not None, 'unsafe_avatar_id')
    require(row['status'] == 'PASS', 'audit_row_not_passed')
    source, target = source_root / avatar, target_root / avatar
    before = inventory(source, row)
    no_links(target)
    existed = target.exists()
    if existed:
        require(inventory(target, row) == before, 'existing_cache_mismatch_refuse_overwrite')
    else:
        needed = sum((source / name).stat().st_size for name in before)
        require(shutil.disk_usage(target_root).free >= needed + 20 * 1024**3, 'insufficient_copy_headroom')
        target.mkdir()  # Exclusive reservation. A failure leaves a recoverable partial directory.
        for name in before:
            dest = target / name
            dest.parent.mkdir(exist_ok=True)
            with (source / name).open('rb') as src, dest.open('xb') as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
        require(inventory(target, row) == before, 'copied_cache_mismatch')
    for name in before:
        src, dest = (source / name).stat(), (target / name).stat()
        require((src.st_dev, src.st_ino) != (dest.st_dev, dest.st_ino), 'shared_inode_refused')
    require(inventory(source, row) == before, 'audit_source_changed_during_copy')
    return {'avatar_id': avatar, 'character': row['character'], 'pose': row['pose'], 'status': 'PASS',
            'operation': 'verified_existing_separate_cache' if existed else 'copied_required_files_exclusively',
            'files': len(before), 'required_bytes': sum((source / name).stat().st_size for name in before),
            'required_inventory_sha256': inventory_sha(before), 'all_required_hashes_match': True,
            'separate_inodes': True, 'source_unchanged': True, 'destination': str(target),
            'existing_files_overwritten': False}


def owned_host():
    require(socket.gethostname() == 'a830e00ce20c', 'not_owned_worker_hostname')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                  text=True, timeout=15, stderr=subprocess.DEVNULL).strip()
    require(gpu == GPU_UUID, 'not_owned_gpu_uuid')
    require(dt.datetime.now(dt.timezone.utc) < dt.datetime(2026, 10, 8, 19, tzinfo=dt.timezone.utc),
            'owned_rental_expired')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    owned_host()
    for path in (SOURCE, TARGET, args.audit, args.out):
        no_links(path)
        require(path.absolute().is_relative_to(ROOT) and '..' not in path.parts, 'outside_owned_checkout')
    require(not args.out.absolute().is_relative_to(SOURCE) and not args.out.absolute().is_relative_to(TARGET),
            'report_must_be_outside_cache_roots')
    require(sha(args.audit) == AUDIT_SHA, 'completed_audit_sha_mismatch')
    audit = json.loads(args.audit.read_text())
    require(audit['status'] == 'PASS' and audit['coverage_complete'] is True
            and audit['counts'] == {'PASS': 48, 'FAIL': 0, 'INVALID': 0}
            and len(audit['objects']) == len({r['avatar_id'] for r in audit['objects']}) == 48,
            'full48_cpu_audit_required')
    require(Path(audit['audit_root']) / 'avatars' == SOURCE, 'wrong_audited_root')
    TARGET.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(fd, 'w', encoding='utf-8')
    data = {'schema': 'owned3090_independent_api_cache_stage_v1', 'status': 'RUNNING',
            'audit_sha256': AUDIT_SHA, 'source': str(SOURCE), 'target': str(TARGET),
            'gpu_allocations': False, 'network_access': False, 'existing_files_overwritten': False,
            'cache_preparation': False, 'source_files_modified': False, 'release_ready': False,
            'rows': [], 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    def save():
        stream.seek(0)
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    try:
        save()
        for row in audit['objects']:
            result = stage_one(row, SOURCE, TARGET)
            data['rows'].append(result)
            save()
            print(json.dumps({'completed': len(data['rows']), 'avatar_id': row['avatar_id'], 'status': 'PASS'}), flush=True)
        data['status'] = 'PASS'
    except Exception as exc:
        data.update(status='INVALID', error_type=type(exc).__name__)
        if isinstance(exc, ValueError):
            data['reason'] = str(exc)
    finally:
        data.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        save()
        stream.close()
    return 0 if data['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
