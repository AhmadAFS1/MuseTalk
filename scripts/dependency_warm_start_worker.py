#!/usr/bin/env python3
"""Actual in-container Vast onstart reuse, isolated port/runtime, same worker."""
import datetime as dt
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

import benchmark_dependency_install as bench
from dependency_artifacts import require
import python_s3_venv_worker as legacy

ROOT=Path('/workspace/.dependency-optimization-20261015-a1')


def execute(deadline):
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Matching Linux only')
    stop=dt.datetime.fromisoformat(deadline).timestamp()-300
    while time.time()+180<stop:
        p=ROOT/'confirmation-a1/receipt.json'
        if p.exists():
            require(json.loads(p.read_text())['state']=='passed','Clean confirmation failed');break
        time.sleep(3)
    else:raise ValueError('No lease reserve for warm startup')
    out=ROOT/'warm-start-a1';require(not out.exists(),'Warm start already claimed; inspect same handle')
    out.mkdir(mode=0o700);legacy.ROOT=out;legacy.DEADLINE=dt.datetime.fromisoformat(deadline)
    env=legacy.worker_runtime_env();venv=ROOT/'optin-venvs/uv';repo=Path('/workspace/MuseTalk')
    env.update({'VENV_PATH':str(venv),'PORT':'8002','HOST':'127.0.0.1','AUTO_SETUP':'1',
        'SETUP_CLEAN':'0','SETUP_FULL_STACK':'1','SETUP_SKIP_APT':'1','SETUP_SKIP_WEIGHTS':'1',
        'SETUP_WEBRTC_TURN':'0','MUSETALK_INSTALLER':str(ROOT/'tools-v10/install_musetalk.sh'),
        'MUSETALK_INSTALL_STATE_FILE':str(ROOT/'canonical-integration-a1/install-state.json'),
        'MUSETALK_DEPENDENCY_INSTALL_MODE':'locked-wheelhouse-uv',
        'MUSETALK_DEPENDENCY_CONTRACT':str(ROOT/'tools-v10/dependency-contract.json'),
        'MUSETALK_DEPENDENCY_CACHE':str(ROOT/'optin-cache'),
        'MUSETALK_RUNTIME_DIR':str(out/'runtime'),'MUSETALK_RUNTIME_CONFIG_SOURCE':'injected',
        'MUSETALK_CANDIDATE_STANDALONE':'1','MUSETALK_RECIPE':'r5',
        'MUSETALK_TAESD_TRT_BUILD':'0','MUSETALK_TAESD_TRT_STRICT':'1',
        'ONSTART_LOG':str(out/'onstart.log'),'STARTUP_TIMEOUT_SECONDS':'90',
        'WEBRTC_NATIVE_VP8_DIR':str(repo/'.runtime/native_vp8')})
    env.pop('MUSETALK_AWS_SECRET_ID',None);env.pop('MUSETALK_AWS_SECRET_REGION',None)
    started=False;result={'state':'running','scope':'warm in-container onstart; existing checkout/models/engines, no provisioning/git/apt/weights/TURN/registration'}
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3) as response:
            require(json.load(response).get('ok') is True,'Original source API unavailable')
        started=True
        result['onstart']=bench.command(['bash',str(repo/'scripts/vast_onstart.sh')],
            'warm-onstart',out,env,timeout=120)
        require(result['onstart']['exit_status']==0,'Warm Vast onstart failed')
        log=(out/'warm-onstart.log').read_text(errors='replace')
        require(log.count('VAST_ONSTART COMPLETE:')==1,'Warm startup completion marker absent/duplicated')
        with urllib.request.urlopen('http://127.0.0.1:8002/health',timeout=3) as response:
            require(json.load(response).get('ok') is True,'Warm candidate API unhealthy')
        result['timing_lines']=[line for line in log.splitlines() if any(k in line for k in
            ('phase finished','install check','Install check','already complete','Skipping setup','TOTAL ELAPSED:','start-to-health'))][-30:]
        result['state']='passed'
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,detail=str(error) if type(error) is ValueError else 'Inspect protected warm-start log')
    finally:
        if started:
            stop=bench.command(['bash',str(repo/'scripts/vast_server_ctl.sh'),'stop'],'candidate-stop',out,env,timeout=45)
            result['stop']=stop
            if stop['exit_status']!=0:result['state']='failed-stop-unconfirmed'
        with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3) as response:
            result['original_API_healthy']=json.load(response).get('ok') is True
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
