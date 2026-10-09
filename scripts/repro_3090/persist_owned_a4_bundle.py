"""Fixed A4 private diagnostics: conditional S3 PUT, fresh GET and CPU restore.

Consumes only a canonically packed, explicitly SHA-bound diagnostic archive.
Never publishes a release, serializes credentials, overwrites cloud objects,
changes bucket permissions or mutates the production model/cache selection.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile

import persist_native48_media_operator as operator
import privacy_attestation as privacy
import safe_capture

ROOT = Path('/Users/ahmadsmacair/code/musetalk-r5-execution')
RELEASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release'
SHA = re.compile(r'[0-9a-f]{64}\Z')
MANIFEST = '.musetalk_trt_artifact_manifest.json'
CHECKSUMS = '.musetalk_trt_artifact_SHA256SUMS'
MODEL_ROOTS = tuple('models/tensorrt_unet_stagewise_a4_' + name + '_v1/bs16/' for name in
                    ('portable_prefix', 'portable_prefix_down0rest', 'portable_core_native_up3',
                     'fp16_mid', 'fp16_up0', 'fp16_mid_up0')) + (
                         'models/tensorrt_unet_stagewise_a4_fp16_mid_up0_v2/bs16/',)
LATENT_ROOT = 'docs/fps_comparisons/rtx3090_r5_20261008/avatars/a4_fixed_geometry_all6_v1/'
A5_ROOTS = tuple('models/' + name + '/bs16/' for name in (
    'tensorrt_unet_stagewise_sm86_r5_opt3_v3', 'tensorrt_unet_stagewise_sm86_r5_all_fp16_v1'))
A5_NAMES = {'manifest.json', 'probe_output.pt'} | {
    name + suffix for name in ('prefix','down0rest','down1','down2','down3','mid','up0','up1','up2','up3','tail')
    for suffix in ('.plan', '.int8.plan')}
require = operator.require
SAFE_FAILURE_REASONS = {
    'private object metadata/size/encryption mismatch', 'object version missing',
    'version changed after PUT', 'fresh GET integrity failed',
    'clean CPU restore payload differs', 'version moved during fresh GET',
}


def validate_manifest(data, profile, allocation='a4'):
    require(allocation in ('a4', 'a5') and (allocation != 'a5' or profile == 'engines'), 'allocation/profile mismatch')
    require(data.get('schema') == 1 and data.get('profile') == 'private-' + allocation + '-' + profile + '-diagnostic',
            'exact canonical diagnostic profile required')
    rows = data.get('files') or []
    require(0 < len(rows) <= 5000 and len({r.get('path') for r in rows}) == len(rows), 'file coverage invalid')
    for row in rows:
        name = row.get('path', '')
        require(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
                and SHA.fullmatch(row.get('sha256', '')) and type(row.get('size')) is int
                and 0 <= row['size'] < 2 * 1024**3 and 'symlink' not in row, 'unsafe payload entry')
        if profile == 'engines':
            roots = MODEL_ROOTS if allocation == 'a4' else A5_ROOTS
            names = {'manifest.json', 'probe_output.pt', 'mid.plan', 'up0.plan'} if allocation == 'a4' else A5_NAMES
            require(any(name.startswith(root) and Path(name).parent.as_posix() + '/' == root for root in roots)
                    and Path(name).name in names,
                    'unexpected engine diagnostic payload')
        else:
            require(name.startswith(LATENT_ROOT) and not any(part.startswith('.') for part in Path(name).parts),
                    'unexpected latent diagnostic payload')
    require(sum(r['size'] for r in rows) < 4 * 1024**3, 'payload exceeds private budget ceiling')
    return rows


def manifest_from_archive(path, digest, profile, allocation='a4'):
    require(SHA.fullmatch(digest or '') and operator.sha_file(path) == digest, 'explicit archive SHA mismatch')
    with tarfile.open(path, 'r:gz') as archive:
        member = archive.getmember(MANIFEST)
        require(member.isfile() and member.size < 4 * 1024**2, 'invalid canonical manifest')
        raw = archive.extractfile(member).read()
        data = json.loads(raw)
        rows = validate_manifest(data, profile, allocation)
        expected = {r['path']: r for r in rows}
        seen = set()
        for member in archive:
            require(member.isfile() and member.name not in seen, 'duplicate/nonfile bundle member')
            seen.add(member.name)
            if member.name in (MANIFEST, CHECKSUMS):
                continue
            require(member.name in expected and member.size == expected[member.name]['size'], 'unexpected archive member')
        require(seen == set(expected) | {MANIFEST, CHECKSUMS}, 'archive payload coverage differs')
    return data, rows


def restore_archive(archive, root, digest):
    # safe_capture's diagnostic vocabulary is closed. Keep the detailed step
    # in our receipt rather than inventing a stage that fails before execution.
    return safe_capture.capture([sys.executable, str(ROOT / 'scripts/trt_artifact_bundle.py'),
        '--repo-root', str(root), '--strict', '--sidecar-dir', str(root / 'sidecars'),
        'restore', '--uri', str(archive), '--expected-sha256', digest], cwd=ROOT, stage='unspecified',
        timeout_s=600, output_limit_bytes=64 * 1024)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--profile', choices=('engines', 'latents'), required=True)
    p.add_argument('--allocation', choices=('a4', 'a5'), default='a4')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--mode', choices=('conditional-put-verify', 'reconcile-read-only'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    require(operator.ROOT == ROOT and a.archive.is_file() and not a.archive.is_symlink()
            and a.archive.resolve().is_relative_to(RELEASE), 'fixed operator archive required')
    require(a.archive.name == a.allocation + '-' + a.profile + '-diagnostic.tar.gz'
            and 0 < a.archive.stat().st_size < 4 * 1024**3, 'archive identity/size invalid')
    require(a.out.parent.resolve() == RELEASE and a.out.suffix == '.json'
            and not a.out.exists() and not a.out.is_symlink(), 'fresh fixed receipt required')
    manifest, rows = manifest_from_archive(a.archive, a.sha256, a.profile, a.allocation)
    fresh = a.out.with_suffix('.fresh-get.tar.gz')
    require(not fresh.exists() and not fresh.is_symlink(), 'fresh GET destination required')
    require(shutil.disk_usage(RELEASE).free > a.archive.stat().st_size + sum(r['size'] for r in rows) + 1024**3,
            'fresh GET and clean restore disk reserve missing')
    os.umask(0o077)
    key = operator.PREFIX + '/' + a.sha256 + '/' + a.archive.name
    report = {'schema': 'owned_private_bundle_persistence_v2', 'status': 'IN_PROGRESS',
              'profile': a.profile, 'allocation': a.allocation,
              'instance_id': '54957508' if a.allocation == 'a4' else '54976782', 'bucket': privacy.BUCKET,
              'region': privacy.REGION, 'expected_owner': privacy.OWNER, 'key': key,
              'sha256': a.sha256, 'bytes': a.archive.stat().st_size, 'payload_files': rows,
              'mode': a.mode, 'quality_accepted': False, 'release_ready': False,
              'public_redistribution': False, 'production_modified': False,
              'note': 'Engine deltas additionally require the separately persisted native and portable parent bundles.'}
    with a.out.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    try:
        report['operation_stage'] = 'privacy_observation'
        report['privacy_configuration_observation'] = privacy.produce()
        operator.save(a.out, report)
        if a.mode == 'conditional-put-verify':
            report['operation_stage'] = 'conditional_put'
            try:
                response = operator.cli('put-object', ['--key', key, '--body', str(a.archive), '--if-none-match', '*',
                    '--server-side-encryption', 'AES256', '--checksum-algorithm', 'SHA256',
                    '--metadata', 'sha256=' + a.sha256 + ',source=' + a.allocation + '-private-diagnostic'], timeout=600)
                report.update(put_outcome='CONFIRMED_CONDITIONAL_SUCCESS', put_version_id=response.get('VersionId'))
            except safe_capture.CaptureFailure as exc:
                report.update(put_outcome='ERROR_OR_AMBIGUOUS_NO_REPUT', safe_put_failure=exc.record)
                operator.save(a.out, report)
        report['operation_stage'] = 'head_before_get'
        operator.save(a.out, report)
        head = operator.cli('head-object', ['--key', key]); operator.validate_head(head, report)
        version = head['VersionId']
        if report.get('put_outcome') == 'CONFIRMED_CONDITIONAL_SUCCESS':
            require(report.get('put_version_id') == version, 'version changed after PUT')
        report['operation_stage'] = 'exact_version_get'
        operator.save(a.out, report)
        operator.cli('get-object', ['--key', key, '--version-id', version, str(fresh)], timeout=600)
        require(fresh.stat().st_size == report['bytes'] and operator.sha_file(fresh) == a.sha256, 'fresh GET integrity failed')
        report['operation_stage'] = 'clean_cpu_restore'
        operator.save(a.out, report)
        restored = Path(tempfile.mkdtemp(prefix='musetalk-a4-' + a.profile + '-restore.'))
        restore_archive(fresh, restored, a.sha256)
        report['operation_stage'] = 'restored_payload_hashes'
        for row in rows:
            path = restored / row['path']
            require(path.is_file() and path.stat().st_size == row['size'] and operator.sha_file(path) == row['sha256'],
                    'clean CPU restore payload differs')
        report['operation_stage'] = 'head_after_restore'
        after = operator.cli('head-object', ['--key', key]); operator.validate_head(after, report)
        require(after['VersionId'] == version, 'version moved during fresh GET')
        report.update(status='PASS_PRIVATE_EXACT_VERSION_GET_CLEAN_CPU_RESTORE_ALL_SHA', version_id=version,
                      clean_restore_path=str(restored), operation_stage='complete',
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception as exc:
        report.update(status='INVALID_NO_REPUT', failure_type=type(exc).__name__)
        report['safe_failure_reason'] = str(exc) if type(exc) is ValueError and str(exc) in SAFE_FAILURE_REASONS else 'details_suppressed'
        if isinstance(exc, safe_capture.CaptureFailure):
            report['safe_failure_record'] = exc.record
    operator.save(a.out, report)
    print(json.dumps({key: report[key] for key in ('status', 'profile', 'sha256', 'bytes', 'release_ready')}))
    return 0 if report['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
