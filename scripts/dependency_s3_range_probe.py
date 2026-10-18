#!/usr/bin/env python3
"""Read-only 128-MiB pilots of one immutable existing S3 env archive.

Not a full archive restore benchmark. No credentials/URLs are emitted or saved.
Run after other delivery trials so traffic cannot contaminate their timings.
"""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import urllib.parse
import urllib.request

from dependency_artifacts import require
from fetch_dependency_wheels import BUCKET, OWNER

KEY='python-venv/20261011/musetalk-python-s3-venv-20261011-a3/4cf134d4749f3fa5719a8df075c1f7b02b9f5a379feb7b4bea5eddcff2bd8599/venv.tar.zst'
SIZE=4509210997
SHA='d8a1fada0b0f59dc7c7df1c289e9101f2b215313f08895fd9ee92acb669195d6'
SAMPLE=128*1024**2


def validate_config(config):
    u=urllib.parse.urlsplit(config['url'])
    require(config['key']==KEY and config['bytes']==SIZE and config['expected_sha256']==SHA, 'Different archive contract')
    require(u.scheme=='https' and u.hostname in {BUCKET+'.s3.amazonaws.com',BUCKET+'.s3.us-east-1.amazonaws.com'}
            and not u.username and not u.password and u.port in (None,443) and not u.fragment
            and urllib.parse.unquote(u.path)=='/'+KEY, 'Unreviewed archive endpoint')
    q=urllib.parse.parse_qs(u.query)
    require(q.get('versionId')==[config['version_id']] and q.get('X-Amz-Algorithm')==['AWS4-HMAC-SHA256'], 'Immutable SigV4 version binding required')
    require(isinstance(config['etag'],str) and config['etag'].startswith('"') and config['etag'].endswith('"'), 'Missing immutable object identity')


def read_range(config,offset,count,opener=urllib.request.urlopen):
    require(0<=offset and 0<count<=SAMPLE and offset+count<=SIZE, 'Invalid pilot range')
    start=time.monotonic()
    headers={'Range':f'bytes={offset}-{offset+count-1}','If-Match':config['etag'],'x-amz-expected-bucket-owner':OWNER}
    with opener(urllib.request.Request(config['url'],headers=headers),timeout=20) as response:
        require(response.status==206 and response.headers.get('Content-Range')==f'bytes {offset}-{offset+count-1}/{SIZE}', 'Wrong range response')
        require(response.headers.get('ETag')==config['etag'] and response.headers.get('x-amz-version-id')==config['version_id'], 'Range object identity changed')
        require(int(response.headers.get('Content-Length','0'))==count, 'Range length differs')
        data=response.read(count+1)
    require(len(data)==count, 'Truncated/excess pilot data')
    return data,{'offset':offset,'bytes':count,'seconds':time.monotonic()-start}


def execute(config):
    require(platform.system()=='Linux' and platform.machine()=='x86_64', 'Linux probe only')
    validate_config(config)
    root=Path('/workspace/.dependency-optimization-20261015-a1')
    parent=json.loads((root/'delivery-matrix-a2/receipt.json').read_text())
    require(parent['state']=='complete', 'Do not contaminate ongoing delivery trials')
    out=root/'s3-range-pilot-a1'
    require(not out.exists(), 'Pilot already claimed; inspect its receipt')
    out.mkdir(mode=0o700)
    samples=[];trusted_prefix=None
    try:
        # Alternate order after the initial single-stream control. Same bytes,
        # fresh connections, known multipart alignment; caches remain unknown.
        for workers in [1,4,4,1,1,4]:
            started=time.monotonic()
            count=SAMPLE//workers
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts=list(pool.map(lambda i:read_range(config,i*count,count),range(workers)))
            payload=b''.join(p[0] for p in parts)
            sha=hashlib.sha256(payload).hexdigest()
            if trusted_prefix is None:trusted_prefix=sha
            require(sha==trusted_prefix, 'Single/ranged pilot bytes differ')
            row={'workers':workers,'bytes':len(payload),'seconds':time.monotonic()-started,'prefix_sha256':sha,
                 'range_receipts':[p[1] for p in parts]}
            samples.append(row)
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({'stage':'S3_RANGE_SAMPLE',**{k:v for k,v in row.items() if k!='range_receipts'}}),flush=True)
        summary={str(w):{'median_seconds':statistics.median(s['seconds'] for s in samples if s['workers']==w),
                         'samples_seconds':[s['seconds'] for s in samples if s['workers']==w]} for w in [1,4]}
        result={'state':'pilot-passed','samples':samples,'summary':summary,'version_id':config['version_id'],'etag':config['etag'],
                'total_payload_bytes':sum(s['bytes'] for s in samples),'full_archive_sha256_verified':False,
                'scope':'128 MiB prefix repeated; no full-archive timing, extraction, package or application claim'}
    except BaseException as error:
        result={'state':'failed','samples':samples,'error_type':type(error).__name__,
                'detail':str(error) if type(error) is ValueError else 'Transport details suppressed'}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def after_delivery(config,deadline):
    """A live bounded job may wait, but never competes with delivery traffic."""
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Linux probe only')
    root=Path('/workspace/.dependency-optimization-20261015-a1')
    stop=dt.datetime.fromisoformat(deadline).timestamp()-600
    print(json.dumps({'stage':'WAITING_FOR_DELIVERY_MATRIX','stop_epoch':stop}),flush=True)
    while time.time()<stop:
        receipt=root/'delivery-matrix-a2/receipt.json'
        if receipt.is_file():
            require(json.loads(receipt.read_text())['state']=='complete','Delivery prerequisite failed; no competing S3 pilot')
            return execute(config)
        time.sleep(5)
    raise ValueError('Lease reserve reached while waiting; no S3 pilot started')
