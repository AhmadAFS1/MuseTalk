#!/usr/bin/env python3
"""Conditionally persist the completed small48 decode proof using runtime auth.

Fixed owned worker, hash-bound private evidence, fresh operator privacy proof.
No new IAM/bucket policy, public ACL, environment logging, media upload or model
deserialization. Record a denied/ambiguous PUT rather than retry it blindly.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket

import persist_private_models as private
import privacy_attestation as privacy
import warm_isolated_api as warm

SOURCE = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release/native_v1_production48_media_1608/decoded_media_proof.json'
SHA = '4eae33de59483c6a626caf3512eaf96c6d58faac76cf26193f454cfb5d035036'
KEY = 'trt-artifacts/rtx3090-r5/native-rejected-diagnostics/sha256/' + SHA + '/native48-decoded-media-proof.json'
OUT = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release/native48_worker_write_probe_1630.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--privacy-attestation', type=Path, required=True)
    parser.add_argument('--privacy-attestation-sha256', required=True)
    args = parser.parse_args()
    private.require(socket.gethostname() == 'a830e00ce20c', 'not_owned_worker')
    private.require(dt.datetime.now(dt.timezone.utc) < warm.DEADLINE, 'owned_deadline_passed')
    private.require(not SOURCE.is_symlink() and SOURCE.is_file() and SOURCE.stat().st_size < 64 * 1024, 'invalid_source')
    raw = SOURCE.read_bytes()
    private.require(hashlib.sha256(raw).hexdigest() == SHA, 'fixed_decoded_proof_sha_mismatch')
    observation = privacy.load_bound(args.privacy_attestation, args.privacy_attestation_sha256)
    data = {'schema': 'owned_native48_private_proof_write_v1', 'status': 'IN_PROGRESS', 'release_ready': False,
            'key': KEY, 'sha256': SHA, 'bytes': len(raw), 'expected_owner': privacy.OWNER,
            'privacy_observation_sha256': args.privacy_attestation_sha256, 'put_attempted': False,
            'if_none_match': '*', 'acl_requested': False, 'iam_or_bucket_changed': False}
    with OUT.open('x') as out:
        json.dump(data, out, indent=2)
        out.write('\n')
    try:
        s3 = private.client()
        privacy.validate(observation)
        data['put_attempted'] = True
        result = s3.put_object(Bucket=privacy.BUCKET, Key=KEY, ExpectedBucketOwner=privacy.OWNER,
                               Body=raw, ContentLength=len(raw), IfNoneMatch='*', ServerSideEncryption='AES256',
                               Metadata={'sha256': SHA, 'source': 'rejected-native48-diagnostic'})
        data['version_id'] = result.get('VersionId')
        head = s3.head_object(Bucket=privacy.BUCKET, Key=KEY, ExpectedBucketOwner=privacy.OWNER)
        private.require(head.get('ContentLength') == len(raw) and head.get('Metadata', {}).get('sha256') == SHA
                        and head.get('ServerSideEncryption') == 'AES256', 'object_metadata_mismatch')
        fetched = s3.get_object(Bucket=privacy.BUCKET, Key=KEY, ExpectedBucketOwner=privacy.OWNER)
        try:
            actual, size = private.hash_stream(fetched['Body'], len(raw))
        finally:
            fetched['Body'].close()
        private.require(actual == SHA and size == len(raw), 'fresh_get_sha_mismatch')
        data['status'] = 'PASS_SMALL_PRIVATE_PROOF_CONDITIONAL_PUT_AND_FRESH_GET'
    except Exception as exc:
        data['status'] = 'INVALID_NO_AUTOMATIC_REPUT'
        data['error_type'] = type(exc).__name__
        data['http_status'] = private.sdk_status(exc)
    OUT.write_text(json.dumps(data, indent=2) + '\n')
    return 0 if data['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
