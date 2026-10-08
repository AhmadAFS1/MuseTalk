#!/usr/bin/env python3
"""A2-only private evidence preservation; pack on worker, persist on operator.

No GPU imports/queries, SSH, credential serialization, source rewrites, or deletes.
Pack requires the operator to have stopped competing measurements. Cloud work is
an explicit separate action using the existing operator AWS CLI default profile.
No preservation status constitutes quality, throughput, or release acceptance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tarfile

import persist_native48_media_operator as operator
import preserve_live_diagnostic_evidence as archive_helper
import privacy_attestation as privacy

INSTANCE = '54909897'
HOSTNAME = '03f3421071ce'
GPU_UUID = 'GPU-a39e62bc-2405-e19d-ba7f-2c51647a46b2'
DEADLINE = dt.datetime(2026, 10, 8, 23, 45, tzinfo=dt.timezone.utc)
DESCRIPTOR_SHA = 'edfd2e0eba7e0e3fad15d53137d88e5ef43f927a14487e28e956b3bc88febcb1'
BASE = 'MuseTalk/docs/fps_comparisons/rtx3090_r5_20261008'
PROFILES = {
    'tracking_pair': (BASE + '/native/a2_tracking_pair_2202_tracking-parity',
                      BASE + '/harnesses/tracking-a2-inputs-v1.json',
                      BASE + '/harnesses/tracking-a2-lineage-v1.json'),
    'latent_seed123': ('experiments/latent_seed123_smoke_a2_2303',),
    'latent_seed123_six': ('experiments/latent_seed123_six_a2_2310',),
    'latent_fixed_cudnn': ('experiments/latent_fixed_cudnn_smoke_a2_2318',),
    'latent_fixed_cudnn_geometry': ('experiments/latent_fixed_cudnn_geometry_smoke_a2_2335',),
    'typed_decoder': (BASE + '/native/a2_taesd_typed_v3',
                      'MuseTalk/models/taesd/trt_native_sm86_typed_v3'),
}
SHA = re.compile(r'[0-9a-f]{64}\Z')
MAX_BYTES = 4 * 1024**3  # safely below single conditional PutObject limit
require = operator.require


def descriptor(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 8192,
            'descriptor missing/link/size')
    require(operator.sha_file(path) == DESCRIPTOR_SHA, 'A2 descriptor SHA mismatch')
    data = json.loads(path.read_text())
    require(data.get('instance_id') == INSTANCE and data.get('worker_hostname') == HOSTNAME
            and data.get('gpu_uuid') == GPU_UUID and dt.datetime.fromisoformat(data['deadline_utc']) == DEADLINE,
            'A2 descriptor identity/deadline mismatch')
    return data


def before_deadline(now=None):
    require((now or dt.datetime.now(dt.timezone.utc)) < DEADLINE, 'A2 preservation deadline passed')


def safe_source(base, name):
    path = base / name
    require(path.is_relative_to(base) and '..' not in path.parts, 'source outside workspace')
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'source ancestor symlink')
        if item == base:
            break
    require(path.exists(), 'required profile source missing')
    return path


def inventory(base, names):
    for name in names:
        safe_source(base, name)
    return archive_helper.inventory(base, names, max_file_bytes=2 * 1024**3,
                                    max_total_bytes=MAX_BYTES, max_files=12000)


def terminal_receipt(base, profile):
    source = base / PROFILES[profile][0]
    name = ('report.json' if profile == 'tracking_pair' else
            'gate/gate_taesd_trt.json' if profile == 'typed_decoder' else 'comparison.json')
    path = safe_source(base, (source / name).relative_to(base).as_posix())
    require(path.is_file() and path.stat().st_size < 16 * 1024**2, 'terminal receipt invalid')
    data = json.loads(path.read_text())
    if profile == 'typed_decoder':
        require(data.get('gate', {}).get('verdict') in {'PASS', 'FAIL'}
                and type(data.get('files')) is int and data['files'] > 0,
                'typed decoder gate not terminal')
        status = data['gate']['verdict']
        engine = data.get('engine', {})
        for kind in ('decoder', 'post'):
            plan = engine.get(kind + '_plan', '')
            require(isinstance(plan, str) and plan and Path(plan).name == plan, 'invalid typed plan name')
            actual = safe_source(base, PROFILES[profile][1] + '/' + plan)
            require(operator.sha_file(actual) == engine.get(kind + '_plan_sha256'), 'typed plan differs from gate')
    else:
        status = data.get('status', '')
        require(isinstance(status, str) and status.startswith(('PASS', 'FAIL', 'INVALID'))
                and isinstance(data.get('finished_utc'), str), 'experiment receipt not terminal')
        finished = dt.datetime.fromisoformat(data['finished_utc'])
        require(finished.tzinfo is not None and finished < DEADLINE, 'experiment completion after deadline')
        if profile == 'tracking_pair' and status == 'PASS':
            result = data['results'][0]
            for mode in ('serial', 'overlap'):
                record = result[mode]
                label = 'a2_tracking_pair_2202_tracking_' + mode
                capture = safe_source(base, PROFILES[profile][0] + '/' + label + '.json')
                require(operator.sha_file(capture) == record['report_sha256'], 'tracking capture receipt changed')
                require(len(record['avatars']) == 6, 'six tracking avatars required')
                for index, identity in enumerate(sorted(record['avatars'])):
                    require(re.fullmatch('[a-z_]+', identity), 'invalid tracking identity')
                    prefix = PROFILES[profile][0] + '/' + label + '/stream' + f'{index:02d}_' + identity
                    for kind in ('faces', 'arrays'):
                        actual = safe_source(base, prefix + '_' + kind + '.npz')
                        require(operator.sha_file(actual) == record['avatars'][identity][kind + '_file_sha256'],
                                'tracking raw payload differs from terminal receipt')
                    for kind in ('faces', 'refined'):
                        require(safe_source(base, prefix + '_' + kind + '.mp4').is_file(), 'tracking media missing')
    return {'path': path.relative_to(base).as_posix(), 'sha256': operator.sha_file(path),
            'recorded_status': status, 'preservation_does_not_upgrade_experiment_status': True}


def new_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pack_archive(base, rows, archive):
    # Low-compression archive preserves bytes and minimizes interference/time.
    payload = (json.dumps({'schema': 'private_live_trace_payload_v1',
                          'release_ready': False, 'files': rows}, sort_keys=True, indent=2) + '\n').encode()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        item = tarfile.TarInfo('private_manifest.json')
        item.size, item.mode = len(payload), 0o600
        tar.addfile(item, io.BytesIO(payload))
        for row in rows:
            path = safe_source(base, row['path'])
            require(operator.sha_file(path) == row['sha256'], 'source changed before pack')
            tar.add(path, arcname=row['path'], recursive=False)
    return hashlib.sha256(payload).hexdigest()


def pack(args):
    before_deadline()
    binding = descriptor(args.descriptor)
    require(socket.gethostname() == HOSTNAME, 'pack requires owned A2 hostname')
    require(args.measurements_stopped, 'explicit stopped-measurements acknowledgement required')
    base = Path('/workspace')
    parent = base / BASE / 'release'
    require(args.out.is_absolute() and args.out.parent.resolve() == parent.resolve()
            and not args.out.exists() and not args.out.is_symlink(), 'fresh fixed worker output required')
    terminal = terminal_receipt(base, args.profile)
    rows = inventory(base, PROFILES[args.profile])
    require(shutil.disk_usage(parent).free > sum(r['bytes'] for r in rows) * 2 + 1024**3,
            'insufficient pack disk reserve')
    before_deadline()
    args.out.mkdir(mode=0o700)
    archive = args.out / ('a2-' + args.profile + '.tar.gz')
    manifest_sha = pack_archive(base, rows, archive)
    require(archive.stat().st_size < MAX_BYTES, 'packed archive too large')
    archive_helper.verify(archive, rows, manifest_sha, max_manifest_bytes=4 * 1024**2)
    # Reject concurrent mutation, added files, or dropped files after packing.
    require(inventory(base, PROFILES[args.profile]) == rows, 'profile changed during packing')
    before_deadline()
    receipt = {'schema': 'a2_private_evidence_pack_v1', 'status': 'PASS_LOCAL_ALL_FILE_SHA',
               'profile': args.profile, 'descriptor_sha256': DESCRIPTOR_SHA, 'binding': binding,
               'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'files': rows,
               'manifest_sha256': manifest_sha, 'archive': archive.name,
               'terminal_receipt': terminal,
               'sha256': operator.sha_file(archive), 'bytes': archive.stat().st_size,
               'release_ready': False, 'cloud_mutations': False, 'gpu_identity_queried': False,
               'measurements_stopped_operator_acknowledgement': True}
    new_json(args.out / 'pack-receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'archive': archive.name, 'bytes': receipt['bytes'],
                      'receipt_sha256': operator.sha_file(args.out / 'pack-receipt.json')}))


def load_pack(path, digest):
    require(SHA.fullmatch(digest or '') and path.is_file() and not path.is_symlink()
            and path.stat().st_size < 4 * 1024**2 and operator.sha_file(path) == digest,
            'explicit pack receipt SHA mismatch')
    data = json.loads(path.read_text())
    profile = data.get('profile')
    require(data.get('schema') == 'a2_private_evidence_pack_v1'
            and data.get('status') == 'PASS_LOCAL_ALL_FILE_SHA' and profile in PROFILES
            and data.get('descriptor_sha256') == DESCRIPTOR_SHA and data.get('release_ready') is False
            and data.get('cloud_mutations') is False, 'invalid pack receipt')
    binding = data.get('binding', {})
    require(binding.get('instance_id') == INSTANCE and binding.get('worker_hostname') == HOSTNAME
            and binding.get('gpu_uuid') == GPU_UUID and binding.get('deadline_utc') == DEADLINE.isoformat(),
            'pack identity mismatch')
    finished = dt.datetime.fromisoformat(data['finished_utc'])
    require(finished.tzinfo is not None and finished < DEADLINE, 'pack completed after deadline')
    require(data.get('archive') == 'a2-' + profile + '.tar.gz'
            and SHA.fullmatch(data.get('sha256', '')) and SHA.fullmatch(data.get('manifest_sha256', ''))
            and type(data.get('bytes')) is int and 0 < data['bytes'] < MAX_BYTES, 'invalid archive pin')
    rows = data.get('files') or []
    require(0 < len(rows) <= 12000 and len({r.get('path') for r in rows}) == len(rows), 'invalid file rows')
    for row in rows:
        name = row.get('path', '')
        require(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
                and any(name == p or name.startswith(p + '/') for p in PROFILES[profile])
                and SHA.fullmatch(row.get('sha256', '')) and type(row.get('bytes')) is int
                and 0 <= row['bytes'] < 2 * 1024**3, 'invalid payload row')
    require(sum(r['bytes'] for r in rows) < MAX_BYTES, 'payload byte ceiling')
    terminal = data.get('terminal_receipt', {})
    require(any(r['path'] == terminal.get('path') and r['sha256'] == terminal.get('sha256') for r in rows)
            and terminal.get('preservation_does_not_upgrade_experiment_status') is True,
            'terminal receipt not bound into payload')
    archive = path.parent / data['archive']
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size == data['bytes']
            and operator.sha_file(archive) == data['sha256'], 'local archive SHA/size mismatch')
    archive_helper.verify(archive, rows, data['manifest_sha256'], max_manifest_bytes=4 * 1024**2)
    return data, archive


def persist(args):
    # Local operator work may finish after rental termination; pack itself must
    # have finished on the bound worker before its deadline.
    require(operator.ROOT == Path('/Users/ahmadsmacair/code/musetalk-r5-execution'), 'fixed operator checkout required')
    row, archive = load_pack(args.receipt, args.receipt_sha256)
    require(args.out.is_absolute() and args.out.parent.resolve() == operator.RELEASE.resolve()
            and not args.out.exists() and not args.out.is_symlink(), 'fresh fixed operator output required')
    fresh = args.out.with_suffix('.fresh-get.tar.gz')
    require(not fresh.exists() and not fresh.is_symlink(), 'fresh GET path exists')
    require(shutil.disk_usage(args.out.parent).free > row['bytes'] + 1024**3, 'fresh GET disk reserve')
    key = operator.PREFIX + '/' + row['sha256'] + '/' + archive.name
    report = {'schema': 'a2_private_evidence_persistence_v1', 'status': 'IN_PROGRESS',
              'pack_receipt_sha256': args.receipt_sha256, 'profile': row['profile'],
              'descriptor_sha256': DESCRIPTOR_SHA, 'bucket': privacy.BUCKET,
              'expected_owner': privacy.OWNER, 'key': key, 'sha256': row['sha256'], 'bytes': row['bytes'],
              'mode': args.mode, 'release_ready': False, 'public_git_payload_allowed': False}
    new_json(args.out, report)
    try:
        report['privacy_configuration_observation'] = privacy.produce()
        operator.save(args.out, report)
        if args.mode == 'conditional-put-verify':
            try:
                response = operator.cli('put-object', ['--key', key, '--body', str(archive),
                    '--if-none-match', '*', '--server-side-encryption', 'AES256',
                    '--checksum-algorithm', 'SHA256', '--metadata', 'sha256=' + row['sha256']
                    + ',source=a2-private-diagnostic'], timeout=600)
                report['put_outcome'] = 'CONFIRMED_CONDITIONAL_SUCCESS'
                report['put_version_id'] = response.get('VersionId')
            except operator.safe_capture.CaptureFailure as exc:
                report['put_outcome'] = 'AMBIGUOUS_OR_ERROR_READ_ONLY_RECONCILIATION_NO_REPUT'
                report['safe_put_failure'] = exc.record
                operator.save(args.out, report)
        head = operator.cli('head-object', ['--key', key])
        operator.validate_head(head, row)
        version = head['VersionId']
        if report.get('put_outcome') == 'CONFIRMED_CONDITIONAL_SUCCESS':
            require(report.get('put_version_id') == version, 'version changed after PUT')
        operator.cli('get-object', ['--key', key, '--version-id', version, str(fresh)], timeout=600)
        require(fresh.stat().st_size == row['bytes'] and operator.sha_file(fresh) == row['sha256'],
                'fresh GET archive mismatch')
        archive_helper.verify(fresh, row['files'], row['manifest_sha256'], max_manifest_bytes=4 * 1024**2)
        after = operator.cli('head-object', ['--key', key])
        operator.validate_head(after, row)
        require(after['VersionId'] == version, 'version changed during GET')
        report.update(status='PASS_PRIVATE_FRESH_GET_ALL_FILE_SHA', version_id=version,
                      payload_files=len(row['files']), finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception as exc:
        report.update(status='INVALID_NO_REPUT', failure_type=type(exc).__name__)
    operator.save(args.out, report)
    print(json.dumps({'status': report['status'], 'release_ready': False}))
    return 0 if report['status'].startswith('PASS_') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--profile', choices=PROFILES, required=True)
    p.add_argument('--descriptor', type=Path, required=True)
    p.add_argument('--measurements-stopped', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('persist')
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--receipt-sha256', required=True)
    p.add_argument('--mode', choices=('conditional-put-verify', 'reconcile-read-only'), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    return pack(args) if args.action == 'pack' else persist(args)


if __name__ == '__main__':
    raise SystemExit(main())
