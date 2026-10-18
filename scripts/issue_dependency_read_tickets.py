#!/usr/bin/env python3
"""EC2-side read-only ticket issuer. Capture stdout privately; never log it.

No worker pip/boto3 bootstrap is needed. This module does not create instances,
upload objects, grant IAM access, or return the EC2 role's AWS credentials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from dependency_artifacts import require, validate_manifest
from install_locked_dependencies import check_contract


def issue(contract, s3, identity, expires=3600):
    check_contract(contract)
    require(type(expires) is int and 300 <= expires <= 3600,'Read-ticket lifetime must be 300–3600 seconds')
    require(identity.get('Account')==contract['owner']
            and '/linguaec2role/' in identity.get('Arn','').lower(), 'Expected EC2 role/account required')
    key=contract['prefix']+'manifest.json'
    response=s3.get_object(Bucket=contract['bucket'],Key=key,ExpectedBucketOwner=contract['owner'])
    with response['Body'] as body:raw=body.read(1024*1024+1)
    require(len(raw)==contract['manifest_bytes'] and hashlib.sha256(raw).hexdigest()==contract['manifest_sha256'],
            'Private manifest differs from independently pinned contract')
    manifest=validate_manifest(json.loads(raw))
    require(manifest['artifact_id']==contract['artifact_id'],'Wrong dependency artifact')
    def ticket(name,sha):
        object_key=contract['prefix']+name
        return {'key':object_key,'owner':contract['owner'],'sha256':sha,
                'url':s3.generate_presigned_url('get_object',Params={'Bucket':contract['bucket'],
                       'Key':object_key,'ExpectedBucketOwner':contract['owner']},ExpiresIn=expires)}
    # Verify the whole private closure (count, then every size) before signing anything.
    builds=[row for row in manifest['artifacts'] if row['source'].startswith('build:')]
    # The contract pins the reviewed private closure size (4 built sdists; 5 with the
    # runtime-only TensorRT libs repack).
    require(len(builds)==contract.get('private_wheels',4),'Different reviewed private-wheel closure')
    for row in builds:
        head=s3.head_object(Bucket=contract['bucket'],Key=contract['prefix']+row['filename'],
                            ExpectedBucketOwner=contract['owner'])
        require(head['ContentLength']==row['size_bytes'],'Required private wheel missing or different size')
    wheels={row['sha256']:ticket(row['filename'],row['sha256']) for row in builds}
    return {'manifest':ticket('manifest.json',contract['manifest_sha256']),'wheels':wheels}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--expires',type=int,default=3600)
    args=p.parse_args()
    try:
        import boto3
        from botocore.config import Config
        contract=json.loads(args.contract.read_text())
        s3=boto3.client('s3',region_name='us-east-1',config=Config(signature_version='s3v4'))
        value=issue(contract,s3,boto3.client('sts').get_caller_identity(),args.expires)
        print(json.dumps(value,separators=(',',':')))
    except Exception as error:
        print('Dependency ticket issuance failed ('+type(error).__name__+'); private details suppressed',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
