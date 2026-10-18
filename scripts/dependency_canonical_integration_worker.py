#!/usr/bin/env python3
"""Canonical wrapper, checkout identity and actual expired-ticket checks.

Isolated same-host integration, not a fresh worker canary/default change.
Waits for the full S3 experiment to finish and verifies the original API.
"""
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import benchmark_dependency_install as bench
from dependency_artifacts import require
from fetch_dependency_wheels import acquire_one
import install_locked_dependencies as locked

ROOT=Path('/workspace/.dependency-optimization-20261015-a1')


def execute(expired,deadline):
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Matching Linux only')
    stop=dt.datetime.fromisoformat(deadline).timestamp()-300
    while time.time()<stop:
        prerequisite=ROOT/'s3-full-a1/receipt.json'
        if prerequisite.exists():break
        time.sleep(3)
    else:raise ValueError('No lease reserve for canonical integration')
    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3) as response:
        require(json.load(response).get('ok') is True,'Original API not restored/healthy')
    require(time.time()+240<stop,'Insufficient remaining canonical test lease')
    out=ROOT/'canonical-integration-a1';require(not out.exists(),'Integration already claimed; inspect same handle')
    out.mkdir(mode=0o700);result={'state':'running','scope':'same-host canonical reuse and Linux source-only fixture; not a fresh canary'};events=[]
    def event(stage,**fields):
        row={'stage':stage,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**fields};events.append(row)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        tools=ROOT/'tools-v8';venv=ROOT/'optin-venvs/uv';repo=Path('/workspace/MuseTalk')
        env={k:v for k,v in os.environ.items() if not k.startswith(('AWS_','LINGUA_','PYTHON','PIP_','UV_'))}
        env.update({'REPO_ROOT':str(repo),'VENV_PATH':str(venv),'WORKSPACE':'/workspace',
            'MUSETALK_DEPENDENCY_INSTALL_MODE':'locked-wheelhouse-uv',
            'MUSETALK_DEPENDENCY_CONTRACT':str(tools/'dependency-contract.json'),
            'MUSETALK_DEPENDENCY_CACHE':str(ROOT/'optin-cache'),'MUSETALK_DEPENDENCY_WORKERS':'2',
            'STEP_LOG_ROOT':str(out/'logs'),'WEBRTC_NATIVE_VP8_DIR':str(repo/'.runtime/native_vp8'),
            'MUSETALK_RUNTIME_DIR':str(out/'runtime'),'MUSETALK_RUNTIME_CONFIG_SOURCE':'injected',
            'MUSETALK_RECIPE':'r5','MUSETALK_TAESD_TRT_BUILD':'0','MUSETALK_TAESD_TRT_STRICT':'1'})
        common=['bash',str(tools/'install_musetalk.sh'),'--repo-root',str(repo),'--venv',str(venv),
                '--matrix','cu121','--with-kokoro','--with-avatar-prep','--with-native-vp8',
                '--state-file',str(out/'install-state.json')]
        event('CANONICAL_WRAPPER_STARTED')
        result['canonical_install']=bench.command(common+['--skip-apt','--skip-weights','--selftest-out',str(out/'gpu-selftest.json')],
            'canonical-install',out,env,timeout=180)
        require(result['canonical_install']['exit_status']==0,'Canonical locked wrapper failed')
        log=(out/'canonical-install.log').read_text(errors='replace')
        require('"actual_mode": "verified-reuse"' in log,'Canonical wrapper did not reuse dependency contract')
        selftest=json.loads((out/'gpu-selftest.json').read_text())
        require(selftest.get('cuda_ok') is True and selftest.get('trt_import_ok') is True,'Canonical GPU selftest failed')
        result['canonical_check']=bench.command(common+['--check','--json'],'canonical-readonly-check',out,env,timeout=30)
        require(result['canonical_check']['exit_status']==0,'Canonical restart check failed')
        event('CANONICAL_WRAPPER_PASSED',install_seconds=result['canonical_install']['seconds'],
              readonly_check_seconds=result['canonical_check']['seconds'])
        fixture=out/'source-only-checkout';(fixture/'requirements').mkdir(parents=True);(fixture/'scripts').mkdir()
        for name in ['server.in','kokoro.in','avatar-prep.in','constraints-cu121.txt']:
            shutil.copyfile(repo/'requirements'/name,fixture/'requirements'/name)
        shutil.copyfile(repo/'scripts/musetalk_install_state.py',fixture/'scripts/musetalk_install_state.py')
        (fixture/'api_server.py').write_text('# application-only edit in isolated checkout fixture\n')
        args=SimpleNamespace(contract=tools/'dependency-contract.json',repo_root=fixture,venv=venv,
            cache_root=ROOT/'optin-cache',check=True,repair=False,mode='locked-wheelhouse-uv',group='full',workers=2)
        result['source_only_change']=locked.run(args,lambda:(_ for _ in ()).throw(AssertionError('Source-only change requested tickets')))
        require(result['source_only_change']['state']=='reused','Source-only edit invalidated dependencies')
        with (fixture/'requirements/server.in').open('a') as f:f.write('\n# dependency input fingerprint change\n')
        try:locked.run(args,None);raise AssertionError('Changed requirements fingerprint was accepted')
        except ValueError as error:
            require(str(error)=='Dependency input changed; publish/review a new contract','Unexpected changed-input failure')
        result['changed_dependency_inputs']='rejected-before-network'
        # URL was created with a one-second expiry before waiting for S3. Check
        # real AWS error semantics and the artifact fetcher's bounded failure.
        require(time.time()-expired['issued_epoch']>2,'Expired-ticket fixture not yet expired')
        request=urllib.request.Request(expired['ticket']['url'],headers={'x-amz-expected-bucket-owner':expired['ticket']['owner']})
        try:
            with urllib.request.urlopen(request,timeout=10):raise ValueError('Expired AWS ticket unexpectedly succeeded')
        except urllib.error.HTTPError as error:
            raw=error.read(65536);code=error.code;error.close();body=ET.fromstring(raw)
            result['expired_ticket']={'http_status':code,'aws_code':body.findtext('Code'),
                'expiry_confirmed':'expired' in str(body.findtext('Message','')).lower()}
            require(code==403 and result['expired_ticket']['expiry_confirmed'],'Actual AWS failure was not expiry')
        cache=out/'expired-cache';cache.mkdir()
        try:acquire_one(expired['artifact'],cache,ticket=expired['ticket']);raise AssertionError('Expired ticket silently fell back')
        except ValueError as error:require(str(error)=='Artifact HTTP failure 403','Unbounded/incorrect expired-ticket handling')
        require(not (cache/'manifest.json').exists() and not list(cache.glob('.partial-*')),'Expired transfer published output/partial')
        result['state']='passed';event('CANONICAL_AND_FAILURE_TESTS_PASSED')
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
            detail=str(error) if type(error) is ValueError else 'Inspect protected integration diagnostics')
        event('INTEGRATION_FAILED',error_type=type(error).__name__,detail=result['detail'])
    result['events']=events;(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
