#!/usr/bin/env python3
"""Final bounded Linux screens: full acquisition overlap and native link modes.

No application/model install, dependency removal, Docker, production callbacks,
or change to installer defaults. Each acquisition uses fresh destinations.
"""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from types import SimpleNamespace
import urllib.request

import benchmark_dependency_install as bench
from dependency_artifacts import require
import dependency_overlap_worker as overlap
import fetch_dependency_tool as tool
import fetch_dependency_wheels as fetch
import install_locked_dependencies as locked

ROOT = Path('/workspace/.dependency-optimization-final-20261016-a1')
SOURCE = '5cc706e90e50e93da1310628c025a84199cd8042'


def reserve(deadline, seconds):
    require(time.time() < dt.datetime.fromisoformat(deadline).timestamp()-seconds,
            'Insufficient unchanged final-screen lease')


def context(root):
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Matching Linux only')
    require(root == ROOT and root.is_dir() and root.resolve() == root
            and root.stat().st_uid == os.getuid(), 'Exact owned final-screen root required')
    contract = json.loads((root/'tools/contract.json').read_text())
    manifest = fetch.read_trusted(root/'tools/manifest.json', contract['manifest_sha256'])
    require(manifest['artifact_id'] == contract['artifact_id'], 'Different final-screen contract')
    require(shutil.disk_usage(root).free >= 35*1024**3, 'Insufficient final-screen scratch')
    return contract, manifest


def prepare(root, deadline):
    reserve(deadline, 1500)
    contract, manifest = context(root)
    out = root/'preparation'; require(not out.exists(), 'Preparation already claimed')
    out.mkdir(mode=0o700); before=time.monotonic()
    facts = {'os':platform.system(), 'machine':platform.machine(), 'glibc':platform.libc_ver(),
             'python':platform.python_version(), 'free_disk_bytes':shutil.disk_usage(root).free}
    require(facts['python'].startswith('3.10.'), 'Stock interpreter differs')
    # The canonical apt prerequisites are common fixtures, outside package A/B.
    env={**os.environ,'DEBIAN_FRONTEND':'noninteractive'}
    apt=[]
    for label,command in (
        ('apt-update',['apt-get','update','-y']),
        ('apt-prerequisites',['apt-get','install','-y','python3.10','python3.10-venv','python3.10-dev',
          'coturn','ffmpeg','git','curl','build-essential','ca-certificates',
          'libgl1','libglib2.0-0','libsm6','libxext6','libxrender1'])):
        receipt=bench.command(command,label,out,env,timeout=180)
        require(receipt['exit_status']==0,'Canonical system fixture failed: '+label)
        apt.append(receipt)
    # Stage only the exact repo-local wheel, not the multi-GB source checkout.
    local = [a for a in manifest['artifacts'] if a['source'].startswith('repo:')]
    require(len(local) == 1 and local[0]['name'] == 'mmcv', 'Different local artifact closure')
    a=local[0]; relative=Path(a['source'][5:])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe local artifact')
    path=root/'repo'/relative; path.parent.mkdir(mode=0o700, parents=True)
    url='https://raw.githubusercontent.com/AhmadAFS1/MuseTalk/'+SOURCE+'/'+relative.as_posix()
    sha=hashlib.sha256(); total=0; start=time.monotonic()
    with urllib.request.urlopen(url, timeout=45) as response, path.open('xb') as target:
        while chunk:=response.read(fetch.CHUNK):
            total+=len(chunk); require(total<=a['size_bytes'], 'Oversized local wheel')
            sha.update(chunk); target.write(chunk)
    require(total==a['size_bytes'] and sha.hexdigest()==a['sha256'], 'Local wheel differs')
    local_seconds=time.monotonic()-start
    uv=tool.fetch(root/'uv-tool')
    require(uv['binary_sha256']==contract['uv_binary_sha256'], 'Pinned tool differs')
    value={'state':'passed','facts':facts,'apt':apt,'local_wheel_bytes':total,'local_wheel_seconds':local_seconds,
           'local_wheel_sha256':sha.hexdigest(),'uv':uv,'seconds':time.monotonic()-before,
           'scope':'common fixture preparation; not a production bootstrap or hidden acquisition saving'}
    (out/'receipt.json').write_text(json.dumps(value,indent=2)+'\n'); return value


def acquire_pair(root, deadline, tickets, engine):
    reserve(deadline, 1500)
    contract, manifest=context(root)
    require(json.loads((root/'preparation/receipt.json').read_text())['state']=='passed', 'Preparation required')
    out=root/'overlap'; require(not out.exists(), 'Acquisition pair already claimed')
    out.mkdir(mode=0o700); rows=[]
    value={'state':'running','scope':'full-wheel plus unchanged r5 archive acquisition only; no installation or API',
           'cache_state':'empty wheel/archive destination per case; OS/CDN cache unknown', 'cases':rows}
    try:
        for name in ('serial','overlap'):
            reserve(deadline, 900)
            case=out/name;case.mkdir(mode=0o700);before=time.monotonic()
            def wheels():
                return fetch.acquire(manifest,case/'wheels','full',4,root/'repo',
                                     reserve_bytes=25*1024**3,tickets=tickets)
            if name=='serial':
                a=wheels(); e=overlap.download_engine(engine,case)
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    wa=pool.submit(wheels); ea=pool.submit(overlap.download_engine,engine,case)
                    a=wa.result();e=ea.result()
            require(a['cache_hits']==0,'Nonempty acquisition comparison')
            row={'case':name,'seconds':time.monotonic()-before,'wheels':a,'engine':e}
            rows.append(row);(case/'receipt.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps({'stage':'FULL_ACQUISITION_CASE_PASSED','case':name,'seconds':row['seconds']}),flush=True)
        value.update(state='passed',screening_reduction_percent=100*(1-rows[1]['seconds']/rows[0]['seconds']),
                     limits=['one ordered pair, not three matched repeats','no full-startup saving established',
                             'repo-local mmcv common staging excluded and separately recorded'])
    except BaseException as error:
        value.update(state='failed',error_type=type(error).__name__,
                     detail=str(error) if type(error) is ValueError else 'Private diagnostics suppressed')
    (out/'receipt.json').write_text(json.dumps(value,indent=2)+'\n');return value


def links(root, deadline):
    reserve(deadline, 900)
    contract, manifest=context(root)
    require(json.loads((root/'overlap/receipt.json').read_text())['state']=='passed','Completed acquisition required')
    wheels=root/'overlap/overlap/wheels';out=root/'link-screen'
    require(not out.exists(),'Link screen already claimed');out.mkdir(mode=0o700)
    rows=[];value={'state':'running','scope':'same local full wheelhouse; remote acquisition excluded', 'cases':rows}
    try:
        # First ordered screen; only repeat a promising supported link mode.
        for number in range(1,4):
            if number>1 and 100*(1-rows[1]['validated_seconds']/rows[0]['validated_seconds'])<20:
                value['repeat_decision']='not promoted: initial validated reduction below 20%';break
            modes=('copy','hardlink') if number%2 else ('hardlink','copy')
            for mode in modes:
                reserve(deadline, 300)
                trial=f'link-{mode}-{number}'
                result=bench.run(SimpleNamespace(experiment_root=root,trial=trial,manifest=root/'tools/manifest.json',
                    manifest_sha256=contract['manifest_sha256'],wheels=wheels,group='full',installer='uv',
                    compile_bytecode=True,link_mode=mode,python='python3.10',uv=root/'uv-tool/uv',
                    uv_sha256=contract['uv_binary_sha256'],source_pip=False,repo_root=root/'repo'))
                require(result['state']=='microbenchmark-passed' and result['package_consistency']['exit_status']==0,
                        'Link case failed inventory/imports/consistency/CUDA')
                case=root/trial;venv=case/'venv';cache=case/'package-cache'
                require(venv.stat().st_dev==cache.stat().st_dev,'Link cache is on another filesystem')
                sample=venv/'lib/python3.10/site-packages/torch/__init__.py'
                sha=fetch.sha256_file(sample);nlink=sample.stat().st_nlink
                require(mode!='hardlink' or nlink>1,'Requested hardlink was not realized')
                cleaned=bench.command([str(root/'uv-tool/uv'),'cache','clean','--cache-dir',str(cache)],
                                      'supported-cache-clean',case,dict(os.environ))
                require(cleaned['exit_status']==0 and fetch.sha256_file(sample)==sha,'Cache clean changed env')
                gpu=bench.command([str(venv/'bin/python'),'-B','-c',bench.GPU_PROBE],
                                  'gpu-after-cache-clean',case,dict(os.environ))
                require(gpu['exit_status']==0,'GPU execution failed after supported cache clean')
                row={'trial':trial,'mode':mode,'repetition':number,'packages_seconds':result['packages_ready_seconds'],
                     'validated_seconds':result['validated_ready_seconds'],'same_filesystem':True,
                     'sample_nlink_before_clean':nlink,'sample_nlink_after_clean':sample.stat().st_nlink,
                     'installed_file_hash_unchanged':True,'GPU_after_cache_clean':True}
                rows.append(row);(case/'link-evidence.json').write_text(json.dumps(row,indent=2)+'\n')
                owner=json.loads((case/'owned-trial.json').read_text())
                require(owner['uid']==os.getuid() and owner['trial']==trial and not locked.active_venv(venv),
                        'Cannot retire foreign or active test environment')
                for path in (venv,cache):
                    require(path.parent==case and path.resolve()==path and not path.is_symlink(),'Unsafe trial child')
                    if path.exists():shutil.rmtree(path)
                print(json.dumps({'stage':'LINK_CASE_PASSED',**row}),flush=True)
        value.update(state='passed',limits=['filesystem-specific local install screen','copy default unchanged',
                                          'no application output acceptance for a changed link policy'])
    except BaseException as error:
        value.update(state='failed',error_type=type(error).__name__,
                     detail=str(error) if type(error) is ValueError else 'Protected diagnostics retained')
    (out/'receipt.json').write_text(json.dumps(value,indent=2)+'\n');return value
