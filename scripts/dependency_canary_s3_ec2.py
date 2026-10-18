#!/usr/bin/env python3
"""Second-host S3 restore confirmation after fresh canary output acceptance."""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import dependency_fresh_canary_ec2 as canary
from dependency_artifacts import normalized, require
import dependency_s3_range_probe as ranges
from dependency_wheel_publish_ec2 import checked_s3, BUCKET, OWNER
from python_s3_venv_detached import launcher, reader


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('launch','status'));a=p.parse_args()
    require(os.getuid()==0,'Existing root-owned EC2 control plane only');os.umask(0o077)
    b=canary.configure();c=canary.c;claim=c.ROOT/'s3-second-job.json'
    if a.mode=='status':
        job=json.loads(claim.read_text());result=json.loads(c.worker(b,reader(c.WORKER_ROOT+'/jobs',job['job_id'])))
        b.save('s3-second-latest.json',result);outcome=result.get('result',{}).get('result',{})
        print(json.dumps({'job_id':job['job_id'],'transport_state':result['state'],'test_state':outcome.get('state'),
                          'packages_seconds':outcome.get('restore_to_packages_seconds'),
                          'validated_seconds':outcome.get('restore_to_matched_validated_seconds'),
                          'original_venv_restored':outcome.get('original_venv_restored'),
                          'original_API_healthy':outcome.get('original_API_healthy')}),flush=True);return
    accepted=json.loads(canary.accepted_receipt().read_text())
    require(not claim.exists(),'S3 confirmation already claimed; inspect existing handle')
    require(not (c.ROOT/'overlap-job.json').exists(),'S3 confirmation must precede overlap traffic')
    ledger=json.loads((c.ROOT/'ledger.json').read_text())
    require(dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(ledger['deadline_utc'])-dt.timedelta(seconds=1200),
            'Insufficient unchanged canary lease')
    names=('dependency_s3_full_worker.py','dependency_s3_range_probe.py','python_s3_venv_worker.py','python_s3_venv_source_library.py')
    files={n:base64.b64encode((c.HERE/n).read_bytes()).decode() for n in names}
    hashes={n:hashlib.sha256((c.HERE/n).read_bytes()).hexdigest() for n in names}
    audio=(b.BOOTSTRAP.parent/'startup-speech.wav').read_bytes()
    require(len(audio)<4*1024**2 and hashlib.sha256(audio).hexdigest()=='8628701b894babe55b8e04769363925528e50c1352d3edcebf1d31a44a65247e',
            'Different application audio')
    files['startup-speech.wav']=base64.b64encode(audio).decode()
    hashes['startup-speech.wav']=hashlib.sha256(audio).hexdigest()
    stage='''import base64,hashlib,json,os,pathlib
root=pathlib.Path(ROOT);assert not root.exists();root.mkdir(mode=0o700)
files=FILES;hashes=HASHES
for name,value in files.items():
 data=base64.b64decode(value);assert hashlib.sha256(data).hexdigest()==hashes[name]
 (root/name).write_bytes(data)
print(json.dumps({'files':hashes}))
'''
    values={'ROOT':repr(c.WORKER_ROOT+'/s3-tools'),'FILES':repr(files),'HASHES':repr(hashes)}
    stage=re.sub(r'\b(ROOT|FILES|HASHES)\b',lambda m:values[m[0]],stage)
    staged=json.loads(c.worker(b,stage));require(staged['files']==hashes,'S3 confirmation tools differ')
    s3=checked_s3();head=s3.head_object(Bucket=BUCKET,Key=ranges.KEY,ExpectedBucketOwner=OWNER)
    require(head['ContentLength']==ranges.SIZE and head.get('VersionId'),'Different full archive object')
    response=s3.get_object(Bucket=BUCKET,Key=ranges.KEY.rsplit('/',1)[0]+'/manifest.json',ExpectedBucketOwner=OWNER)
    with response['Body'] as body:raw=body.read(1024*1024+1)
    require(len(raw)<=1024*1024,'Oversized archive descriptor');manifest=json.loads(raw)
    require(manifest['archive_sha256']==ranges.SHA and manifest['archive_bytes']==ranges.SIZE,'Different archive manifest')
    reviewed=json.loads((c.HERE/'wheelhouse-manifest-a2.json').read_text())
    require({normalized(k):v for k,v in manifest['contract']['facts']['packages'].items()}
            =={x['name']:x['version'] for x in reviewed['artifacts']},'Different archive package contract')
    config={'key':ranges.KEY,'bytes':ranges.SIZE,'expected_sha256':ranges.SHA,'etag':head['ETag'],'version_id':head['VersionId'],
            'url':s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':ranges.KEY,
                   'VersionId':head['VersionId'],'ExpectedBucketOwner':OWNER,'IfMatch':head['ETag']},ExpiresIn=1800)}
    ranges.validate_config(config)
    payload=("import sys,json;from pathlib import Path;sys.path[:0]="+repr([c.WORKER_ROOT+'/s3-tools','/workspace/MuseTalk/scripts'])
        +";import dependency_s3_full_worker as s;print(json.dumps({'result':s.execute("+repr(config)+","+repr(manifest)+","+repr(ledger['deadline_utc'])
        +",root=Path("+repr(c.WORKER_ROOT)+"),artifact_cache=Path('/workspace/.cache/musetalk-dependencies/725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194'),single_get=False,application_audio=Path("
        +repr(c.WORKER_ROOT+'/s3-tools/startup-speech.wav')+"),expected_video_sha="+repr(accepted['cases'][0]['video_sha256'])+")}))")
    compile(payload,'<private-second-host-s3>','exec')
    job={'job_id':uuid.uuid4().hex,'deadline_utc':ledger['deadline_utc'],'files':hashes,
         'manifest_sha256':hashlib.sha256(raw).hexdigest(),'scope':'second-host complete ranged restore, original canonical env retained and restored'}
    b.save('s3-second-job.json',job)
    result=json.loads(c.worker(b,launcher(c.WORKER_ROOT+'/jobs',job['job_id'],payload)))
    b.save('s3-second-dispatch.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        print(json.dumps({'state':'failed','error_type':type(error).__name__,
                          'detail':str(error) if type(error) is ValueError else 'Private diagnostics suppressed'}),flush=True)
        raise SystemExit(2)
