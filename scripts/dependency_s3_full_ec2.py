#!/usr/bin/env python3
"""One durable full-S3 experiment after application-a3; same bounded rental."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import dependency_optimization_ec2 as c
from dependency_wheel_publish_ec2 import checked_s3,BUCKET,OWNER
import dependency_s3_range_probe as ranges
from python_s3_venv_detached import launcher
from dependency_artifacts import normalized


def main():
    b=c.library();b.require(os.getuid()==0,'Root-owned EC2 experiment only');os.umask(0o077)
    ledger=json.loads((c.ROOT/'ledger.json').read_text());stop=dt.datetime.fromisoformat(ledger['deadline_utc']).timestamp()-900
    claim=c.ROOT/'s3-full-job.json';b.require(not claim.exists(),'Full S3 job already claimed; inspect existing handle')
    while time.time()<stop:
        prior=c.ROOT/'application-a3/receipt.json'
        if prior.exists():
            b.require(json.loads(prior.read_text())['state']=='passed-output-and-timing-captured','Cold-cache application test failed');break
        time.sleep(5)
    else:raise ValueError('Insufficient original lease; full S3 test not submitted')
    s3=checked_s3();head=s3.head_object(Bucket=BUCKET,Key=ranges.KEY,ExpectedBucketOwner=OWNER)
    b.require(head['ContentLength']==ranges.SIZE and head.get('VersionId'),'Different full archive object')
    value=s3.get_object(Bucket=BUCKET,Key=ranges.KEY.rsplit('/',1)[0]+'/manifest.json',ExpectedBucketOwner=OWNER)
    try:raw=value['Body'].read(1024*1024+1)
    finally:value['Body'].close()
    b.require(len(raw)<=1024*1024,'Oversized archive descriptor');manifest=json.loads(raw)
    b.require(manifest['archive_sha256']==ranges.SHA and manifest['archive_bytes']==ranges.SIZE,'Different archive manifest')
    reviewed=json.loads((c.ROOT/'evidence/wheelhouse-build-a2/manifest.json').read_text())
    b.require({normalized(k):v for k,v in manifest['contract']['facts']['packages'].items()}
              =={a['name']:a['version'] for a in reviewed['artifacts']},'Archive package contract differs from current wheelhouse')
    config={'key':ranges.KEY,'bytes':ranges.SIZE,'expected_sha256':ranges.SHA,'etag':head['ETag'],'version_id':head['VersionId'],
        'url':s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':ranges.KEY,'VersionId':head['VersionId'],
            'ExpectedBucketOwner':OWNER,'IfMatch':head['ETag']},ExpiresIn=1800)}
    ranges.validate_config(config)
    job={'job_id':uuid.uuid4().hex,'deadline_utc':ledger['deadline_utc'],
        'worker_source_sha256':hashlib.sha256((c.HERE/'dependency_s3_full_worker.py').read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(raw).hexdigest(),'scope':'same owned rental, full GET then four-range restore; no lease extension'}
    b.save('s3-full-job.json',job)
    payload="import sys,json;sys.path.insert(0,"+repr(c.WORKER_ROOT+'/tools-v7')+");import dependency_s3_full_worker as s;print(json.dumps({'result':s.execute("+repr(config)+","+repr(manifest)+","+repr(ledger['deadline_utc'])+")}))"
    result=json.loads(c.worker(b,launcher(c.WORKER_ROOT+'/jobs',job['job_id'],payload)))
    b.save('s3-full-dispatch.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps({'state':'failed','error_type':type(error).__name__,'detail':str(error) if type(error) is ValueError else 'Inspect protected control-plane state'}),flush=True)
        raise SystemExit(2)
