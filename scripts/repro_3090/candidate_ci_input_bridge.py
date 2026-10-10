#!/usr/bin/env python3
"""EC2-role-only staging and bounded presigning for the private candidate build.

The `urls` mode emits a secret payload intended ONLY for captured stdin delivery
to GitHub's secret setter. Never display it, save it, or put it in build args.
The stage receipt contains only hashes/object identity, never credentials/URLs.
"""
import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

BUCKET = 'lingua-musetalk-s3-storage'
OWNER = '211125449207'
REGION = 'us-east-1'
INPUTS = {
    'weights.tar.gz': {
        'key': 'docker-build-inputs/sha256/acff525e22a9ee80917e2ae394ed0746dbaa8cf13b0762b6a9de2104265d2d19/weights.tar.gz',
        'version_id': 'eOxBtMevZGxnA395QPPt9Lep2Z18W8k3',
        'sha256': 'acff525e22a9ee80917e2ae394ed0746dbaa8cf13b0762b6a9de2104265d2d19',
        'size_bytes': 3951486311},
    'native.tar.gz': {
        'key': 'trt-artifacts/rtx3090-r5/native-rejected-diagnostics/sha256/1f766487cf9272929988d17f9a4d6f76ee9c8c8fdde9b149174e1e02dbcafc5e/native-sm86-v1-serving-diagnostic.tar.gz',
        'version_id': '9zMITqphKo8iMe.AVek48BFjh6gLdgV2',
        'sha256': '1f766487cf9272929988d17f9a4d6f76ee9c8c8fdde9b149174e1e02dbcafc5e',
        'size_bytes': 983926034}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_head(head, entry):
    require(head['ContentLength'] == entry['size_bytes']
            and head.get('Metadata', {}).get('sha256') == entry['sha256']
            and head.get('VersionId') == entry['version_id']
            and head.get('ServerSideEncryption') == 'AES256', 'S3 input identity mismatch')


def clients():
    import boto3
    from botocore.config import Config
    session = boto3.Session(region_name=REGION)
    require(session.get_credentials().method == 'iam-role', 'EC2 IAM role required')
    config = Config(signature_version='s3v4', connect_timeout=5, read_timeout=30,
                    retries={'total_max_attempts': 2, 'mode': 'standard'},
                    ignore_configured_endpoint_urls=True, s3={'addressing_style': 'virtual'})
    identity = session.client('sts', config=config).get_caller_identity()
    require(identity['Account'] == OWNER and identity['Arn'].startswith(
            'arn:aws:sts::211125449207:assumed-role/linguaEc2role/'), 'Unexpected control-plane role')
    return session.client('s3', config=config, endpoint_url='https://s3.us-east-1.amazonaws.com')


def main():
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('stage', 'urls'))
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--existing-version-id', help='Read-only verification of a separately staged exact version')
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{64}', args.sha256), 'Exact metadata SHA required')
    s3 = clients()
    if args.mode == 'stage':
        require(not args.receipt.exists() and args.metadata and args.metadata.is_file()
                and not args.metadata.is_symlink(), 'Fresh receipt and regular metadata required')
        raw = args.metadata.read_bytes()
        require(0 < len(raw) <= 32 * 1024**2 and hashlib.sha256(raw).hexdigest() == args.sha256,
                'Metadata identity mismatch')
        key = 'docker-build-inputs/sha256/' + args.sha256 + '/musetalk-docker-metadata.tar.gz'
        if args.existing_version_id:
            version = args.existing_version_id
        else:
            result = s3.put_object(Bucket=BUCKET, Key=key, Body=raw, IfNoneMatch='*',
                                  ExpectedBucketOwner=OWNER, ServerSideEncryption='AES256',
                                  ContentType='application/gzip', Metadata={'sha256': args.sha256})
            version = result['VersionId']
        item = {'key': key, 'sha256': args.sha256, 'size_bytes': len(raw), 'version_id': version}
        verify_head(s3.head_object(Bucket=BUCKET, Key=key, VersionId=item['version_id'],
                                  ExpectedBucketOwner=OWNER), item)
        response = s3.get_object(Bucket=BUCKET, Key=key, VersionId=item['version_id'], ExpectedBucketOwner=OWNER)
        restored = response['Body'].read()
        require(restored == raw, 'Fresh metadata restore mismatch')
        try:
            with urllib.request.urlopen('https://' + BUCKET + '.s3.' + REGION + '.amazonaws.com/' + key,
                                        timeout=15) as response:
                raise ValueError('Metadata unexpectedly accessible anonymously')
        except urllib.error.HTTPError as error:
            require(error.code == 403, 'Anonymous access denial not confirmed')
        receipt = {'schema': 'musetalk_private_candidate_ci_inputs_v1', 'bucket': BUCKET,
                   'region': REGION, 'owner': OWNER, 'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'metadata_fresh_restore': True, 'anonymous_denied': True,
                   'inputs': {'musetalk-docker-metadata.tar.gz': item, **INPUTS}, 'published': False}
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'status': 'PRIVATE_METADATA_STAGED_VERIFIED', 'metadata': item,
                          'anonymous_denied': True}))
        return
    receipt = json.loads(args.receipt.read_text())
    require(receipt['schema'] == 'musetalk_private_candidate_ci_inputs_v1'
            and receipt['bucket'] == BUCKET and receipt['owner'] == OWNER
            and receipt['inputs']['musetalk-docker-metadata.tar.gz']['sha256'] == args.sha256
            and receipt['inputs']['weights.tar.gz'] == INPUTS['weights.tar.gz']
            and receipt['inputs']['native.tar.gz'] == INPUTS['native.tar.gz'], 'Input receipt mismatch')
    urls = {}
    for name, item in receipt['inputs'].items():
        params = {'Bucket': BUCKET, 'Key': item['key'], 'VersionId': item['version_id']}
        verify_head(s3.head_object(**params, ExpectedBucketOwner=OWNER), item)
        response = s3.get_object(**params, ExpectedBucketOwner=OWNER, Range='bytes=0-0')
        require(response['ResponseMetadata']['HTTPStatusCode'] == 206
                and len(response['Body'].read()) == 1, 'Versioned build input read failed')
        # ExpectedBucketOwner is used for verified SDK calls above, not a signed
        # header on this URL: the CI downloader sends plain HTTPS GET only.
        urls[name] = s3.generate_presigned_url('get_object', Params=params, ExpiresIn=3600)
    print(json.dumps(urls))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
        operation = getattr(exc, 'operation_name', '')
        safe = '/'.join(value for value in (operation, code)
                        if isinstance(value, str) and re.fullmatch('[A-Za-z0-9_]{1,64}', value))
        print('Private CI bridge failed (' + type(exc).__name__ + '; ' + safe
              + '); error body suppressed', file=sys.stderr)
        sys.exit(2)
