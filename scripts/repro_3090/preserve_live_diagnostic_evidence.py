"""Preserve fixed private loopback traces/logs; never publish credential-bearing raw data."""
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import persist_native48_media_operator as operator
import privacy_attestation as privacy

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
INPUTS = (
    'native/private_baseline_live_evidence_1710',
    'native/private_x264tuned_live_evidence_1733',
    'native/isolated_native_api_traced_1640_private.log',
    'native/isolated_native_api_x264tuned_final_1735_private.log',
    'native/private_nvenc_live_evidence_1743',
    'native/isolated_native_api_nvenc_final_1743_private.log',
)
ARCHIVE = BASE / 'release/private_live_baseline_tuned_1740.tar.gz'
RECEIPT = BASE / 'release/private_live_baseline_tuned_1740.json'
FRESH = BASE / 'release/private_live_baseline_tuned_1740_fresh_get.tar.gz'


def inventory(base, names):
    rows = []
    for name in names:
        source = base / name
        operator.require(not source.is_symlink() and source.exists(), 'fixed private input missing/link')
        files = sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in files:
            operator.require(not path.is_symlink(), 'private evidence link forbidden')
            if path.is_dir():
                continue
            operator.require(path.is_file() and 0 <= path.stat().st_size < 32 * 1024**2,
                             'private evidence file type/size')
            rows.append({'path': path.relative_to(base).as_posix(), 'bytes': path.stat().st_size,
                         'sha256': operator.sha_file(path)})
    operator.require(0 < len(rows) <= 512 and len({r['path'] for r in rows}) == len(rows),
                     'private evidence inventory missing/duplicate/too large')
    operator.require(sum(r['bytes'] for r in rows) < 64 * 1024**2, 'private archive byte ceiling')
    return rows


def pack(base, rows, archive):
    operator.require(not archive.exists() and not archive.is_symlink(), 'fresh archive required')
    manifest = {'schema': 'private_live_trace_payload_v1', 'release_ready': False, 'files': rows}
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    with tarfile.open(archive, 'x:gz') as tar:
        entry = tarfile.TarInfo('private_manifest.json')
        entry.size, entry.mode = len(payload), 0o600
        tar.addfile(entry, io.BytesIO(payload))
        for row in rows:
            path = base / row['path']
            operator.require(operator.sha_file(path) == row['sha256'], 'source changed before pack')
            tar.add(path, arcname=row['path'], recursive=False)
    return hashlib.sha256(payload).hexdigest()


def verify(archive, rows, manifest_sha):
    """Stream all members without extraction or trusting archive paths."""
    expected = {row['path']: row for row in rows}
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            operator.require(member.isfile() and member.name not in seen, 'nonfile/duplicate archive member')
            seen.add(member.name)
            stream = tar.extractfile(member)
            if member.name == 'private_manifest.json':
                operator.require(member.size < 256 * 1024, 'manifest oversized')
                operator.require(hashlib.sha256(stream.read()).hexdigest() == manifest_sha, 'manifest SHA mismatch')
                continue
            operator.require(member.name in expected and member.size == expected[member.name]['bytes'],
                             'unexpected member or size')
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
            operator.require(digest.hexdigest() == expected[member.name]['sha256'], 'private payload SHA mismatch')
    operator.require(seen == set(expected) | {'private_manifest.json'}, 'missing private payload')


def main():
    operator.require(ROOT == Path('/Users/ahmadsmacair/code/musetalk-r5-execution'), 'fixed operator checkout')
    operator.require(not any(p.exists() or p.is_symlink() for p in (ARCHIVE, RECEIPT, FRESH)), 'fresh outputs required')
    os.umask(0o077)
    rows = inventory(BASE, INPUTS)
    manifest_sha = pack(BASE, rows, ARCHIVE)
    verify(ARCHIVE, rows, manifest_sha)
    row = {'bytes': ARCHIVE.stat().st_size, 'sha256': operator.sha_file(ARCHIVE)}
    key = operator.PREFIX + '/' + row['sha256'] + '/' + ARCHIVE.name
    data = {'schema': 'private_live_diagnostic_preservation_v1', 'status': 'LOCAL_ARCHIVE_VERIFIED',
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'archive': ARCHIVE.name,
            **row, 'manifest_sha256': manifest_sha, 'payload_files': len(rows), 'key': key,
            'included_diagnostic_profiles': ['aiortc_v1', 'x264tuned_v1', 'nvenc_v1'],
            'bucket': privacy.BUCKET, 'expected_owner': privacy.OWNER, 'region': privacy.REGION,
            'raw_payload_public_git_allowed': False, 'release_ready': False,
            'cloud_mutations': 'one new conditional checksum-keyed private object only',
            'overwrite_allowed': False, 'acl_requested': False}
    with RECEIPT.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
    try:
        data['privacy_configuration_observation'] = privacy.produce()
        operator.save(RECEIPT, data)
        try:
            response = operator.cli('put-object', ['--key', key, '--body', str(ARCHIVE),
                '--if-none-match', '*', '--server-side-encryption', 'AES256', '--checksum-algorithm', 'SHA256',
                '--metadata', 'sha256=' + row['sha256'] + ',source=private-loopback-diagnostic'], timeout=180)
            data['put_outcome'] = 'CONFIRMED_CONDITIONAL_SUCCESS'
            data['put_version_id'] = response.get('VersionId')
        except operator.safe_capture.CaptureFailure as exc:
            data['put_outcome'] = 'AMBIGUOUS_OR_ERROR_RECONCILE_READ_ONLY_NO_REPUT'
            data['put_failure'] = exc.record
            operator.save(RECEIPT, data)
        head = operator.cli('head-object', ['--key', key])
        operator.validate_head(head, row)
        version = head['VersionId']
        if data['put_outcome'] == 'CONFIRMED_CONDITIONAL_SUCCESS':
            operator.require(data['put_version_id'] == version, 'version changed after PUT')
        data['version_id'] = version
        operator.cli('get-object', ['--key', key, '--version-id', version, str(FRESH)], timeout=180)
        operator.require(FRESH.stat().st_size == row['bytes'] and operator.sha_file(FRESH) == row['sha256'],
                         'fresh private GET archive mismatch')
        verify(FRESH, rows, manifest_sha)
        after = operator.cli('head-object', ['--key', key])
        operator.validate_head(after, row)
        operator.require(after['VersionId'] == version, 'private version moved during GET')
        data['status'] = 'PASS_PRIVATE_CONDITIONAL_PUT_FRESH_GET_ALL_PAYLOAD_SHA'
        data['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        operator.save(RECEIPT, data)
        print(json.dumps({k: data[k] for k in ('status', 'sha256', 'bytes', 'payload_files', 'release_ready')}))
        return 0
    except Exception as exc:
        data['status'] = 'FAIL_PRESERVED_NO_REPUT'
        data['failure_type'] = type(exc).__name__
        operator.save(RECEIPT, data)
        print(json.dumps({'status': data['status'], 'failure_type': data['failure_type']}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
