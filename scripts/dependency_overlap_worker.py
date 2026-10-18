#!/usr/bin/env python3
"""Screen bounded wheel prefetch overlap with the unchanged r5 bundle download.

Not an installation/full-boot benchmark. Two fresh destinations, identical
payloads and checksums; no extraction, engine build or application changes.
Private tickets enter in memory and are not retained in receipts.
"""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import urllib.parse
import urllib.request

from dependency_artifacts import require
import fetch_dependency_wheels as fetch

KEY = 'trt-artifacts/ampere-plus/r5-srcg50-int8/sha256-07644ff16a170ecbfb4eb3d39ff1150731d5508baff31e0216a5dc1bb371df2e/musetalk-trt-r5-ampere-plus.tar.gz'
SHA = '07644ff16a170ecbfb4eb3d39ff1150731d5508baff31e0216a5dc1bb371df2e'
SIZE = 1000524577


def engine_ticket(ticket):
    require(ticket.get('key') == KEY and ticket.get('sha256') == SHA
            and ticket.get('size') == SIZE and ticket.get('owner') == fetch.OWNER,
            'Unreviewed engine artifact')
    u = urllib.parse.urlsplit(ticket['url'])
    require(u.scheme == 'https' and u.hostname in {fetch.BUCKET+'.s3.amazonaws.com',
            fetch.BUCKET+'.s3.us-east-1.amazonaws.com'} and not u.username and not u.password
            and not u.fragment and u.port in (None,443) and urllib.parse.unquote(u.path) == '/'+KEY,
            'Unreviewed engine destination')
    return ticket['url']


def download_engine(ticket, root):
    url = engine_ticket(ticket); before = time.monotonic(); size = 0; sha = hashlib.sha256()
    partial = root / 'engine.partial'; final = root / 'engine.tar.gz'
    require(not partial.exists() and not final.exists(), 'Fresh owned engine target required')
    request = urllib.request.Request(url, headers={'x-amz-expected-bucket-owner':fetch.OWNER})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as out:
            require(response.status == 200 and int(response.headers.get('Content-Length',-1)) == SIZE,
                    'Engine response size/status differs')
            while chunk := response.read(fetch.CHUNK):
                size += len(chunk); require(size <= SIZE, 'Oversized engine payload')
                sha.update(chunk); out.write(chunk)
        require(size == SIZE and sha.hexdigest() == SHA, 'Engine size/trusted SHA256 differs')
        partial.replace(final)
    except BaseException:
        if partial.is_file() and not partial.is_symlink(): partial.unlink()
        raise
    return {'seconds':time.monotonic()-before,'bytes':size,'sha256':sha.hexdigest(),'trusted_sha256_verified':True}


def execute(root, deadline, envelope, engine):
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Matching Linux only')
    require(dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(deadline)-dt.timedelta(seconds=1200),
            'Insufficient bounded lease for overlap screen')
    require(root.is_absolute() and root.name == 'overlap-a1' and root.parent.name == '.dependency-canary-20261016-a1'
            and not root.exists() and root.parent.is_dir() and root.parent.stat().st_uid == os.getuid(),
            'Fresh owned canary overlap root required')
    require(shutil.disk_usage(root.parent).free >= 25*1024**3, 'Insufficient overlap scratch space')
    repo = Path('/workspace/MuseTalk')
    contract = json.loads((repo/'configs/dependencies/rtx3090-cp310-cu121-screening.json').read_text())
    manifest = fetch.read_trusted(Path('/workspace/.cache/musetalk-dependencies')/contract['artifact_id']/'manifest.json',
                                  contract['manifest_sha256'])
    require(manifest['artifact_id'] == contract['artifact_id'], 'Different overlap contract')
    engine_ticket(engine)
    root.mkdir(mode=0o700); cases=[]
    result={'state':'running','scope':'wheel-acquisition plus engine-download overlap only; not installation or total boot',
            'cache_state':'fresh per-case wheel/archive destinations; OS/CDN caches unknown', 'cases':cases}
    try:
        for name in ('serial','overlap'):
            case=root/name;case.mkdir(mode=0o700)
            before=time.monotonic()
            def wheels():
                return fetch.acquire(manifest,case/'wheels','full',4,repo,reserve_bytes=12*1024**3,tickets=envelope['wheels'])
            if name=='serial':
                a=wheels(); e=download_engine(engine,case)
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    wa=pool.submit(wheels); ea=pool.submit(download_engine,engine,case)
                    a=wa.result(); e=ea.result()
            value={'case':name,'seconds':time.monotonic()-before,'wheels':a,'engine':e}
            cases.append(value)
            (case/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps({'stage':'OVERLAP_CASE_PASSED','case':name,'seconds':value['seconds'],
                              'wheel_seconds':a['seconds'],'engine_seconds':e['seconds']}),flush=True)
        result.update(state='passed',screening_reduction_percent=100*(1-cases[1]['seconds']/cases[0]['seconds']),
                      limits=['one ordered screen per method; not three matched repetitions',
                              'no install/extraction/API time included; not a proven boot saving',
                              'engine remains identical and unextracted; no defaults changed'])
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
                      detail=str(error) if type(error) is ValueError else 'Protected diagnostics suppressed')
    (root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
