#!/usr/bin/env python3
"""Bounded full immutable-S3 transfer/restore test, only on the owned worker.

Runs after application timing. Preserves and restores the original canonical
venv. Signed URLs enter memory over stdin and never appear in logs/argv.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request

from dependency_artifacts import require, GROUPS
import benchmark_dependency_install as bench
from fetch_dependency_wheels import OWNER, sha256_file
import dependency_s3_range_probe as ranges
import python_s3_venv_worker as legacy

ROOT=Path('/workspace/.dependency-optimization-20261015-a1')
VENV=Path('/workspace/.venvs/musetalk_trt_stagewise')


def download(config,path,workers,stop):
    require(workers in {1,4} and not path.exists(),'Invalid/new full transfer required')
    started=time.monotonic();done=0
    if workers==1:
        request=urllib.request.Request(config['url'],headers={'If-Match':config['etag'],'x-amz-expected-bucket-owner':OWNER})
        with urllib.request.urlopen(request,timeout=30) as response,path.open('xb') as output:
            require(response.status==200 and response.headers.get('ETag')==config['etag']
                    and response.headers.get('x-amz-version-id')==config['version_id']
                    and int(response.headers.get('Content-Length','0'))==ranges.SIZE,'Different full object response')
            while chunk:=response.read(8*1024**2):
                require(time.time()<stop,'Full transfer reached lease reserve')
                done+=len(chunk);require(done<=ranges.SIZE,'Excess full response');output.write(chunk)
            output.flush();os.fsync(output.fileno())
        requests=1
    else:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            os.ftruncate(fd,ranges.SIZE)
            size=32*1024**2;parts=[(i,min(size,ranges.SIZE-i)) for i in range(0,ranges.SIZE,size)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(ranges.read_range,config,offset,count):offset for offset,count in parts}
                for future in as_completed(futures):
                    require(time.time()<stop,'Full range transfer reached lease reserve')
                    offset=futures.pop(future);data,receipt=future.result()
                    require(os.pwrite(fd,data,offset)==len(data),'Short full-range write');done+=len(data)
            os.fsync(fd);requests=len(parts)
        finally:os.close(fd)
    seconds=time.monotonic()-started
    require(done==ranges.SIZE and path.stat().st_size==ranges.SIZE,'Incomplete full archive')
    hashed=time.monotonic();require(sha256_file(path)==ranges.SHA,'Full archive checksum mismatch')
    return {'workers':workers,'payload_bytes':done,'requests':requests,'download_seconds':seconds,
        'decimal_MB_per_second':done/seconds/1e6,'checksum_seconds':time.monotonic()-hashed,
        'trusted_sha256_verified':True,'sha256':ranges.SHA}


def execute(config,manifest,deadline,*,root=None,artifact_cache=None,single_get=True,application_audio=None,expected_video_sha=None):
    ROOT = globals()['ROOT'] if root is None else Path(root)
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Matching Linux only')
    require(ROOT in {Path('/workspace/.dependency-optimization-20261015-a1'),
                     Path('/workspace/.dependency-canary-20261016-a1')}
            and ROOT.is_dir() and ROOT.resolve()==ROOT and ROOT.stat().st_uid==os.getuid(),
            'Unreviewed full-restore experiment root')
    require(type(single_get) is bool,'Explicit single-GET screen policy required')
    ranges.validate_config(config)
    require(platform.python_version()=='3.10.12' and platform.libc_ver()[1]=='2.35',
            'Full venv requires the matching Python/glibc base')
    require(manifest['archive_sha256']==ranges.SHA and manifest['archive_bytes']==ranges.SIZE
            and manifest['contract']['venv_path']==str(VENV),'Different archived environment contract')
    require(time.time()<dt.datetime.fromisoformat(deadline).timestamp()-900,'Insufficient lease for full restore')
    out=ROOT/'s3-full-a1';require(not out.exists(),'Full restore already claimed; inspect same job')
    out.mkdir(mode=0o700);events=[];backup=VENV.with_name(VENV.name+'.dependency-s3-backup-a1')
    require(VENV.is_dir() and not VENV.is_symlink() and not backup.exists(),'Unsafe canonical rollback paths')
    legacy.ROOT=out;legacy.DEADLINE=dt.datetime.fromisoformat(deadline)
    stop=legacy.DEADLINE.timestamp()-300;runtime=legacy.worker_runtime_env()
    runtime.update(DRAIN_TIMEOUT_SECONDS='20',STARTUP_TIMEOUT_SECONDS='90',MUSETALK_TAESD_TRT_BUILD='0',MUSETALK_TAESD_TRT_STRICT='1')
    stopped=False;published=False;result={'state':'running','scope':'same-host full dependency restore; code/system/models/engines retained'}
    def event(stage,**fields):
        row={'stage':stage,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**fields};events.append(row)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        # Supported cache maintenance; installed copy-mode environments stay intact.
        artifact=ROOT/'optin-cache/725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194' if artifact_cache is None else Path(artifact_cache)
        require(artifact in {ROOT/'optin-cache/725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194',
                             Path('/workspace/.cache/musetalk-dependencies/725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194')}
                and artifact.is_dir() and artifact.resolve()==artifact and artifact.stat().st_uid==os.getuid(),
                'Unreviewed installer cache path')
        uv=artifact/'uv-tool/uv';require(sha256_file(uv)=='db919fb9428cc8ad04377ba3c898e0ec5fbee83700da5b982b1c9539f1e1b34f','Unexpected cache tool')
        before=shutil.disk_usage(ROOT).free
        legacy.command([str(uv),'cache','clean','--cache-dir',str(artifact/'installer-cache')],log='cache-maintenance.log')
        event('OWNED_INSTALLER_CACHE_CLEANED',reclaimed_bytes=shutil.disk_usage(ROOT).free-before,
              installed_env_removed=False)
        needed=manifest['payload_file_bytes']+(2 if single_get else 1)*ranges.SIZE+3*1024**3
        require(shutil.disk_usage(ROOT).free>=needed,'Insufficient scratch for two archives + staged payload + rollback reserve')
        if single_get:
            event('SINGLE_GET_STARTED')
            result['single_get']=download(config,out/'single-get.tar.zst',1,stop)
            event('SINGLE_GET_VERIFIED',**result['single_get'])
        else:
            result['single_get']={'state':'not-repeated','reason':'second-host confirmation of previously screened four-range restore'}
        started=time.monotonic();event('FULL_RANGED_RESTORE_STARTED')
        archive=out/'range-get.tar.zst';result['ranged_get']=download(config,archive,4,stop)
        event('RANGED_GET_VERIFIED',**result['ranged_get'])
        if shutil.which('zstd') is None:
            legacy.command(['apt-get','install','-y','zstd'],timeout=60,log='zstd-setup.log')
        before=time.monotonic();legacy.audit_archive(archive,Path('/usr/bin/python3.10'))
        result['archive_audit_seconds']=time.monotonic()-before
        stage=out/'stage';stage.mkdir(mode=0o700);before=time.monotonic()
        legacy.command(['tar','--no-same-owner','-I','zstd','-xf',str(archive),'-C',str(stage)],timeout=180,log='extract.log')
        require(set(p.name for p in stage.iterdir())=={VENV.name},'Unexpected archive top level')
        result['extraction_seconds']=time.monotonic()-before
        legacy.command(['bash','scripts/vast_server_ctl.sh','stop'],env=runtime,timeout=60,log='source-stop.log')
        stopped=True;legacy.ensure_api_stopped();VENV.rename(backup)
        try:(stage/VENV.name).rename(VENV);published=True
        except BaseException:backup.rename(VENV);raise
        facts=legacy.package_facts()
        require(facts['packages']==manifest['contract']['facts']['packages'] and not facts['editable_packages'],
                'Restored package inventory differs')
        py=str(VENV/'bin/python')
        legacy.command([py,'-m','pip','check'],log='pip-check.log')
        result['restore_to_packages_seconds']=time.monotonic()-started
        modules=[m for group in GROUPS['full'] for m in bench.IMPORTS.get(group,[])]
        code='import importlib\nfor name in '+repr(modules)+':importlib.import_module(name)'
        for label in ['first-imports','second-process-imports']:
            legacy.command([py,'-c',code],env={**runtime,'CUDA_VISIBLE_DEVICES':''},log=label+'.log')
        legacy.command([py,'-c',bench.GPU_PROBE],env=runtime,log='gpu-imports-and-execution.log')
        result['restore_to_matched_validated_seconds']=time.monotonic()-started
        # This is the archive's independent full-inventory/SHA contract, not an
        # env owned by the locked-wheel installer. Use baseline READ-ONLY checks;
        # do not manufacture a locked ownership marker or run a source install.
        legacy.validate_native(env={**runtime,'MUSETALK_DEPENDENCY_INSTALL_MODE':'source-pip'})
        result['archive_native_check']='baseline read-only pins/imports plus independently verified archive inventory/SHA; no pip installation'
        result['restore_to_native_validated_seconds']=time.monotonic()-started
        legacy.command(['bash','scripts/vast_server_ctl.sh','start'],env=runtime,timeout=100,log='restored-start.log')
        with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as response:health=json.load(response)
        require(health.get('ok') is True,'Restored API unhealthy')
        result['restore_to_API_health_seconds']=time.monotonic()-started
        result['state']='restored-native-and-health-passed'
        event('RESTORE_API_HEALTH_PASSED',**{k:v for k,v in result.items() if k.startswith('restore_to_')})
        if application_audio is not None:
            audio=Path(application_audio)
            require(audio==ROOT/'s3-tools/startup-speech.wav' and audio.is_file() and not audio.is_symlink()
                    and sha256_file(audio)=='8628701b894babe55b8e04769363925528e50c1352d3edcebf1d31a44a65247e',
                    'Different/unreviewed restored application input')
            app=out/'application';app.mkdir(mode=0o700)
            code=("import sys;from pathlib import Path;sys.path.insert(0,"+repr(str(ROOT/'s3-tools'))+")"
                +";import python_s3_venv_source_library as b;b.ROOT=Path("+repr(str(app))+")"
                +";b.SMOKE_PHASE_SECONDS=120;b.smoke('http://127.0.0.1:8000',Path("+repr(str(audio))+"))")
            legacy.command([py,'-B','-c',code],env=runtime,timeout=180,log='restored-application.log')
            result['application']=json.loads((app/'video-smoke.json').read_text())
            result['restore_to_decoded_video_seconds']=time.monotonic()-started
            result['output_matches_same_host_locked']=result['application']['video_sha256']==expected_video_sha
            event('RESTORED_VIDEO_DECODED',output_matches_same_host_locked=result['output_matches_same_host_locked'])
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
            detail=str(error) if type(error) is ValueError else 'Inspect protected restore diagnostics')
        event('FULL_RESTORE_FAILED',error_type=type(error).__name__,detail=result['detail'])
    finally:
        if published:
            try:legacy.command(['bash','scripts/vast_server_ctl.sh','stop'],env=runtime,timeout=60,log='restored-stop.log')
            except BaseException:result['restored_stop']='unconfirmed; canonical rollback deferred until APIs proven stopped'
        if backup.exists():
            try:
                legacy.ensure_api_stopped()
                if VENV.exists():VENV.rename(out/'restored-venv-retained')
                backup.rename(VENV);result['original_venv_restored']=True
            except BaseException:result['original_venv_restored']=False
        if stopped and not backup.exists():
            try:
                legacy.command(['bash','scripts/vast_server_ctl.sh','start'],env=runtime,timeout=100,log='original-restart.log')
                with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as response:result['original_API_healthy']=json.load(response).get('ok') is True
            except BaseException:result['original_API_healthy']=False
        result['events']=events
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
