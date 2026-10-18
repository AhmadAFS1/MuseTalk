#!/usr/bin/env python3
"""Isolated Linux fault/reuse/opt-in tests after timed acquisition trials.

Never changes source requirements, the active source venv, or production defaults.
No S3 full-archive download. Loopback transport faults are explicitly fixtures.
"""
import datetime as dt
import hashlib
import http.server
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

from dependency_artifacts import require
import fetch_dependency_wheels as fetch
import install_locked_dependencies as locked

ROOT=Path('/workspace/.dependency-optimization-20261015-a1')


def fault_checks(out):
    rows=[]
    payload=b'wheel';calls=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            calls.append(self.path)
            if self.path=='/auth':self.send_error(403);return
            if self.path=='/retry' and calls.count('/retry')==1:self.send_error(503);return
            if self.path=='/disconnect':self.connection.close();return
            self.send_response(200);self.send_header('Content-Length','5');self.end_headers()
            self.wfile.write(b'bad' if self.path=='/truncate' else payload)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    class Opener:
        def __init__(self,path):self.path=path
        def open(self,*args,**kwargs):
            return urllib.request.urlopen('http://127.0.0.1:'+str(server.server_port)+self.path,timeout=3)
    artifact={'name':'fixture','filename':'fixture-1.0-py3-none-any.whl',
        'source':'https://files.pythonhosted.org/fixture-1.0-py3-none-any.whl',
        'sha256':hashlib.sha256(payload).hexdigest(),'size_bytes':5}
    try:
        for name,path,expected in [('auth','/auth',False),('truncated','/truncate',False),
                                   ('disconnect','/disconnect',False),('retry','/retry',True)]:
            cache=out/name;cache.mkdir();before=len(calls);start=time.monotonic()
            try:
                receipt=fetch.acquire_one(artifact,cache,retries=1,opener=Opener(path));passed=True
            except BaseException as error:receipt={'error_type':type(error).__name__};passed=False
            require(passed==expected,'Unexpected fixture outcome: '+name)
            require(bool((cache/artifact['filename']).exists())==expected,'Partial artifact was published')
            require(not list(cache.glob('.partial-*')),'Partial file left behind')
            rows.append({'case':name,'expected_success':expected,'observed_success':passed,
                'requests':len(calls)-before,'seconds':time.monotonic()-start,'receipt':receipt})
        cache=out/'corrupt';cache.mkdir();(cache/artifact['filename']).write_bytes(b'wrong')
        before=len(calls)
        try:fetch.acquire_one(artifact,cache,opener=Opener('/valid'));raise AssertionError('Corrupt wheel accepted')
        except ValueError:pass
        require(len(calls)==before,'Corrupt cache silently redownloaded')
        rows.append({'case':'corrupt-cache','rejected':True,'requests':0})
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
    return rows


def execute(envelope,deadline):
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Matching Linux only')
    stop=dt.datetime.fromisoformat(deadline).timestamp()-600
    while time.time()<stop:
        delivery=ROOT/'delivery-matrix-a2/receipt.json';pilot=ROOT/'s3-range-pilot-a1/receipt.json'
        if delivery.exists() and pilot.exists():
            require(json.loads(delivery.read_text())['state']=='complete','Delivery matrix did not pass')
            break
        time.sleep(5)
    else:raise ValueError('Lease reserve reached; no followup started')
    out=ROOT/'followup-a1';require(not out.exists(),'Followup already claimed; inspect same job')
    out.mkdir(mode=0o700);events=[];result={'state':'running','scope':'same-host opt-in/reuse/fault checks, not fresh-host canary'}
    def event(stage,**fields):
        row={'stage':stage,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**fields};events.append(row)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        event('LINUX_FAULT_CHECKS_STARTED')
        faults=out/'faults';faults.mkdir();result['transport_faults']=fault_checks(faults)
        parent=ROOT/'optin-venvs';parent.mkdir(mode=0o700)
        args=SimpleNamespace(contract=ROOT/'tools-v6/dependency-contract.json',repo_root=Path('/workspace/MuseTalk'),
            venv=parent/'uv',cache_root=ROOT/'optin-cache',check=False,repair=False,
            mode='locked-wheelhouse-uv',group='full',workers=2)
        event('OPTIN_FRESH_INSTALL_STARTED')
        result['fresh_install']=locked.run(args,lambda:envelope)
        event('OPTIN_FRESH_INSTALL_PASSED',seconds=result['fresh_install']['seconds'])
        args.check=True;reuses=[]
        for _ in range(3):
            reuses.append(locked.run(args,lambda:(_ for _ in ()).throw(AssertionError('Reuse requested tickets'))))
            require(reuses[-1]['state']=='reused','Warm reuse did not skip installation')
        result['reuse_checks']=reuses
        event('REUSE_CHECKS_PASSED',seconds=[r['seconds'] for r in reuses])
        with fetch.lock(args.venv.parent/('.'+args.venv.name+'.dependencies.lock')):
            try:locked.run(args,lambda:envelope);raise AssertionError('Concurrent install lock bypassed')
            except BlockingIOError:result['concurrent_lock']='rejected-before-mutation'
        marker=args.venv/locked.MARKER;old=marker.read_bytes();value=json.loads(old)
        try:
            value['state']='failed';locked.atomic_json(marker,value)
            try:locked.run(args,lambda:envelope);raise AssertionError('Failed stamp was trusted')
            except ValueError:result['stale_failed_stamp']='rejected'
        finally:marker.write_bytes(old)
        # Retain moved files for recovery; never uninstall from a running API.
        require(not locked.active_venv(args.venv),'Candidate unexpectedly in use')
        site=args.venv/'lib/python3.10/site-packages';backup=out/'missing-package-backup';backup.mkdir()
        meta=list(site.glob('docopt-*.dist-info'));require(len(meta)==1,'Ambiguous repair fixture')
        targets=[meta[0],site/'docopt.py']
        for p in targets:require(p.exists() and not p.is_symlink(),'Unsafe missing-package fixture')
        for p in targets:p.rename(backup/p.name)
        try:
            try:locked.run(args,lambda:envelope);raise AssertionError('Stale accepted stamp hid missing package')
            except ValueError:result['missing_package_check']='rejected'
            args.check=False;args.repair=True
            result['missing_package_repair']=locked.run(args,lambda:envelope)
            require(result['missing_package_repair']['acquisition']['remote_transferred_bytes']==0,'Repair redownloaded verified cache')
            require((site/'docopt.py').is_file(),'Repair did not restore module')
        except BaseException:
            for p in targets:
                if not p.exists() and (backup/p.name).exists():(backup/p.name).rename(p)
            raise
        event('MISSING_PACKAGE_REPAIR_PASSED',seconds=result['missing_package_repair']['seconds'],remote_bytes=0)
        result['state']='passed'
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
            detail=str(error) if type(error) is ValueError else 'Inspect protected followup diagnostics')
        event('FOLLOWUP_FAILED',error_type=type(error).__name__,detail=result['detail'])
    result['events']=events
    locked.atomic_json(out/'receipt.json',result)
    return result
