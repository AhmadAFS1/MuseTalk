#!/usr/bin/env python3
"""Persist only the three fixed rejected-native48 media archives privately.

Operator default AWS CLI identity; conditional PUT only, no overwrite/ACL/IAM
change, no credential serialization. Each object's privacy settings, metadata,
archive SHA, fresh GET and archive SHA are independently checked. A PUT error is
reconciled read-only, never replayed blindly. This is not a public release.
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

import privacy_attestation as privacy
import safe_capture

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release'
ARCHIVE_DIR = RELEASE / 'native_v1_production48_media_1616'
NAMES = {f'native-v1-production48-review-part{i}of3.tar.gz' for i in range(1, 4)}
PREFIX = 'trt-artifacts/rtx3090-r5/native-rejected-diagnostics/sha256'
SHA = re.compile('[0-9a-f]{64}\\Z')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cli(operation, arguments, timeout=60):
    command = ['aws', '--profile', 'default', '--region', privacy.REGION,
               '--endpoint-url', f'https://s3.{privacy.REGION}.amazonaws.com',
               '--no-cli-pager', '--cli-connect-timeout', '5', '--cli-read-timeout', '60',
               's3api', operation, '--bucket', privacy.BUCKET, '--expected-bucket-owner', privacy.OWNER,
               '--output', 'json', *arguments]
    clean = {**os.environ, 'AWS_PAGER': '', 'AWS_CLI_AUTO_PROMPT': 'off',
             'AWS_IGNORE_CONFIGURED_ENDPOINT_URLS': 'true', 'AWS_MAX_ATTEMPTS': '2'}
    value = safe_capture.capture(command, cwd=ROOT, env=clean, stage='unspecified',
                                 timeout_s=timeout, output_limit_bytes=64 * 1024)
    return json.loads(value)


def validate_head(head, row):
    require(head.get('ContentLength') == row['bytes'] and head.get('Metadata', {}).get('sha256') == row['sha256']
            and head.get('ServerSideEncryption') == 'AES256', 'private object metadata/size/encryption mismatch')
    require(isinstance(head.get('VersionId'), str) and head['VersionId'], 'object version missing')


def load_archives(receipt_sha):
    receipt = ARCHIVE_DIR / 'preservation_receipt.json'
    require(SHA.fullmatch(receipt_sha) and not receipt.is_symlink() and receipt.stat().st_size < 128 * 1024
            and sha_file(receipt) == receipt_sha, 'explicit pack receipt SHA mismatch')
    data = json.loads(receipt.read_text())
    require(data.get('status') == 'PASS_ALL48_MUXED_PIXEL_AND_HUMAN_PCM_INTEGRITY'
            and data.get('release_ready') is False and data.get('cloud_mutations') is False,
            'unverified or releasable pack receipt forbidden')
    rows = data.get('archives') or []
    require(len(rows) == 3 and {r.get('filename') for r in rows} == NAMES, 'exact three archives required')
    for row in rows:
        path = ARCHIVE_DIR / row['filename']
        require(not path.is_symlink() and path.is_file() and not ARCHIVE_DIR.is_symlink()
                and SHA.fullmatch(row.get('sha256', '')) and type(row.get('bytes')) is int
                and 0 < row['bytes'] < 5 * 1024**3 and path.stat().st_size == row['bytes']
                and sha_file(path) == row['sha256'], 'archive pin/type/size/SHA mismatch')
    return rows


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-receipt-sha256', required=True)
    parser.add_argument('--mode', choices=('conditional-put-verify', 'reconcile-read-only'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(ROOT == Path('/Users/ahmadsmacair/code/musetalk-r5-execution'), 'fixed operator checkout required')
    require(args.out.is_absolute() and args.out.parent.resolve() == RELEASE
            and not args.out.exists() and not args.out.is_symlink(), 'fresh fixed output receipt required')
    os.umask(0o077)
    rows = load_archives(args.pack_receipt_sha256)
    require(shutil.disk_usage(RELEASE).free > sum(r['bytes'] for r in rows) + 1024**3, 'fresh GET disk reserve missing')
    fresh = RELEASE / (args.out.stem + '_fresh_get')
    require(not fresh.exists(), 'fresh GET directory exists')
    fresh.mkdir(mode=0o700)
    data = {'schema': 'rejected_native48_operator_private_persistence_v1', 'status': 'IN_PROGRESS',
            'pack_receipt_sha256': args.pack_receipt_sha256, 'release_ready': False,
            'bucket': privacy.BUCKET, 'region': privacy.REGION, 'expected_owner': privacy.OWNER,
            'mode': args.mode, 'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'objects': [],
            'scope': 'Private diagnostic evidence only; no active bundle, public registry, model rights or visual acceptance changed',
            'cloud_mutations': False if args.mode == 'reconcile-read-only' else 'conditional_new_checksum_keyed_objects_only'}
    with args.out.open('x') as output:
        json.dump(data, output, indent=2)
        output.write('\n')
    try:
        for row in rows:
            # Actual default-identity STS, PAB, policy and ACL reads immediately
            # before each object. The model-delivery helper's assertion schema
            # is used solely as current fixed-bucket configuration evidence.
            observation = privacy.produce()
            key = PREFIX + '/' + row['sha256'] + '/' + row['filename']
            record = {'filename': row['filename'], 'sha256': row['sha256'], 'bytes': row['bytes'],
                      'key': key, 'privacy_configuration_observation': observation,
                      'put_attempted': args.mode == 'conditional-put-verify', 'acl_requested': False,
                      'overwrite_permitted': False, 'fresh_get_verified': False}
            data['objects'].append(record)
            save(args.out, data)
            if args.mode == 'conditional-put-verify':
                try:
                    response = cli('put-object', ['--key', key, '--body', str(ARCHIVE_DIR / row['filename']),
                                   '--if-none-match', '*', '--server-side-encryption', 'AES256',
                                   '--checksum-algorithm', 'SHA256', '--metadata',
                                   'sha256=' + row['sha256'] + ',source=musetalk-native48-rejected-diagnostic'], timeout=600)
                    record['put_outcome'] = 'CONFIRMED_CONDITIONAL_SUCCESS'
                    record['put_version_id'] = response.get('VersionId')
                except safe_capture.CaptureFailure as exc:
                    record['put_outcome'] = 'ERROR_OR_AMBIGUOUS_RECONCILE_READ_ONLY_NO_REPUT'
                    record['put_failure'] = exc.record
                    save(args.out, data)
            head = cli('head-object', ['--key', key])
            validate_head(head, row)
            version = head['VersionId']
            if record.get('put_outcome') == 'CONFIRMED_CONDITIONAL_SUCCESS':
                require(record.get('put_version_id') == version, 'current object version changed after PUT')
            record['version_id'] = version
            destination = fresh / row['filename']
            require(not destination.exists(), 'fresh GET output exists')
            cli('get-object', ['--key', key, '--version-id', version, str(destination)], timeout=600)
            require(destination.stat().st_size == row['bytes'] and sha_file(destination) == row['sha256'],
                    'fresh GET archive SHA/size mismatch')
            after = cli('head-object', ['--key', key])
            validate_head(after, row)
            require(after['VersionId'] == version, 'current object version changed during GET')
            record.update(fresh_get_verified=True, fresh_get_path=str(destination),
                          etag_used_as_content_sha256=False, status='PASS_PRIVATE_ARCHIVE_CONTENT_INTEGRITY')
            save(args.out, data)
            print(json.dumps({'verified_private_archives': len(data['objects']), 'expected': 3}), flush=True)
        data['status'] = 'PASS_ALL3_PRIVATE_ARCHIVE_FRESH_GET_SHA'
    except Exception as exc:
        data['status'] = 'INVALID'
        data['error_type'] = type(exc).__name__
        if isinstance(exc, ValueError):
            data['reason'] = str(exc)
        if isinstance(exc, safe_capture.CaptureFailure):
            data['safe_failure'] = exc.record
    data['finished_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(args.out, data)
    print(json.dumps({'status': data['status']}), flush=True)
    return 0 if data['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
