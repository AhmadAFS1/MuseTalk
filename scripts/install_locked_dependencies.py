#!/usr/bin/env python3
"""Opt-in locked Linux dependencies. Source-pip remains the default elsewhere.

No synchronization/removal, no engine builds, no implicit source fallback. Private
read tickets enter a protected file descriptor, never argv or a stored manifest.
Existing accepted environments are checked before any ticket/network request.

Opt-in base-image seeding (--seed-site / MUSETALK_DEPENDENCY_SEED_SITE, unset =
off): when the contract pins a seed index, artifacts whose every wheel RECORD
file is present and hash-identical in that site-packages are copied instead of
downloaded (see dependency_seed.py); anything else is acquired as before.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import benchmark_dependency_install as bench
from dependency_artifacts import offline_lock, require, selected
import fetch_dependency_tool as tool
import fetch_dependency_wheels as fetch

MARKER = '.musetalk-locked-dependencies.json'
UV_BINARY_SHA256 = 'db919fb9428cc8ad04377ba3c898e0ec5fbee83700da5b982b1c9539f1e1b34f'
RANGE_STREAMS_ENV = 'MUSETALK_DEPENDENCY_RANGE_STREAMS'
MAX_CONNECTIONS_ENV = 'MUSETALK_DEPENDENCY_MAX_CONNECTIONS'
SEED_SITE_ENV = 'MUSETALK_DEPENDENCY_SEED_SITE'
SEED_BYTECODE_ENV = 'MUSETALK_DEPENDENCY_SEED_BYTECODE'
SEED_INDEX_MAX_BYTES = 32*1024**2


def atomic_json(path, data):
    temporary=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x') as out:
        os.chmod(temporary,0o600)
        json.dump(data,out,indent=2,allow_nan=False);out.write('\n')
        out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def check_contract(contract):
    require(all(contract.get(k)==v for k,v in {'matrix':'cu121','python':'3.10','os':'Linux',
            'machine':'x86_64','minimum_glibc':'2.35','uv_version':'0.13.1'}.items()),
            'Wrong Python/OS/CUDA dependency contract')
    require(contract.get('bucket')==fetch.BUCKET and contract.get('owner')==fetch.OWNER,
            'Unreviewed artifact owner/bucket')
    require(re.fullmatch('[a-f0-9]{64}',contract.get('artifact_id','')) is not None,
            'Invalid artifact identity')
    require(contract.get('prefix')==fetch.PRIVATE_PREFIX+contract['artifact_id']+'/',
            'Unreviewed artifact prefix')
    require(re.fullmatch('[a-f0-9]{64}',contract.get('manifest_sha256','')) is not None,
            'Manifest digest must be independently pinned')
    require(contract.get('uv_binary_sha256')==UV_BINARY_SHA256,
            'Installer binary differs from reviewed version')
    if 'seed_index' in contract:
        pin=contract['seed_index']
        require(isinstance(pin,dict) and set(pin)=={'path','sha256','bytes'}
                and pin['path']=='configs/dependencies/seed-index-'+contract['artifact_id']+'.json.gz'
                and isinstance(pin['sha256'],str) and re.fullmatch('[a-f0-9]{64}',pin['sha256']) is not None
                and type(pin['bytes']) is int and 0<pin['bytes']<=SEED_INDEX_MAX_BYTES,
                'Seed index must be pinned by reviewed path, digest and size')


def check_platform():
    require(platform.system()=='Linux' and platform.machine()=='x86_64',
            'Linux x86_64 only; no macOS installation tests')
    require(sys.version_info[:2]==(3,10),'Host interpreter must be CPython 3.10')
    require(tuple(map(int,platform.libc_ver()[1].split('.'))) >= (2,35),
            'glibc below validated floor')


def wait_for_apt_prefetch(environ=None,limit=900):
    """The bootstrap may run apt (python3.10-venv etc.) in the background while
    wheels download; the venv must not be created before it finished."""
    raw=(os.environ if environ is None else environ).get('MUSETALK_APT_PREFETCH_DIR','').strip()
    if not raw:return None
    folder=Path(raw);pending=folder/'pending';start=time.monotonic()
    while pending.exists() and time.monotonic()-start<limit:time.sleep(1)
    require(not pending.exists(),'Background apt still running; refusing to create the venv')
    rc=(folder/'rc').read_text().strip() if (folder/'rc').exists() else '?'
    require(rc=='0','Background apt failed (rc '+rc+')')
    return round(time.monotonic()-start,3)


def transfer_settings(args,environ=None):
    """Ranged-download settings: flags win, then environment, then defaults
    (0 range streams = original single-stream behaviour; connection cap = workers)."""
    environ=os.environ if environ is None else environ
    def pick(flag,name):
        if flag is not None:return flag
        raw=environ.get(name,'').strip()
        if not raw:return None
        require(raw.isdigit(),'Invalid '+name)
        return int(raw)
    streams=pick(getattr(args,'range_streams',None),RANGE_STREAMS_ENV)
    streams=0 if streams is None else streams
    require(streams in (0,2,4),RANGE_STREAMS_ENV+' must be 0, 2 or 4')
    connections=pick(getattr(args,'max_connections',None),MAX_CONNECTIONS_ENV)
    if connections is None:connections=args.workers
    else:require(connections in (4,8,16),MAX_CONNECTIONS_ENV+' must be 4, 8 or 16')
    require(connections in fetch.CONNECTION_CHOICES,'Unsupported dependency connection cap')
    return streams,connections


def seed_settings(args,environ=None):
    """Base-image seeding: flag wins, then environment; unset (the default) = off.
    Bytecode copying is a separate opt-in because no wheel RECORD covers pycs."""
    environ=os.environ if environ is None else environ
    site=getattr(args,'seed_site',None)
    if site is None:site=environ.get(SEED_SITE_ENV,'').strip() or None
    if site is None:return None,False
    site=Path(site)
    require(site.is_absolute(),SEED_SITE_ENV+' must be an absolute path')
    raw=environ.get(SEED_BYTECODE_ENV,'').strip()
    require(raw in ('','0','1'),SEED_BYTECODE_ENV+' must be 0 or 1')
    return site,raw=='1'


def seed_plan(contract,manifest,repo,venv,group,site,bytecode,fresh,workers):
    """Lazy import: without a configured seed site nothing here runs."""
    if not contract.get('seed_index'):
        return {'receipt':{'state':'disabled','reason':'contract-pins-no-seed-index','site':str(site)},'staged':[]}
    if not fresh:
        return {'receipt':{'state':'skipped','reason':'existing-venv-repair','site':str(site)},'staged':[]}
    import dependency_seed as seed
    index=seed.load_index(contract,manifest,repo)
    return seed.stage(index,selected(manifest,group),site,venv,workers=workers,bytecode=bytecode,
                      index_sha256=contract['seed_index']['sha256'])


def check_inputs(manifest,repo):
    for name,sha in manifest['requirement_inputs'].items():
        require(Path(name).name==name,'Unsafe requirements fingerprint path')
        require(fetch.sha256_file(repo/'requirements'/name)==sha,
                'Dependency input changed; publish/review a new contract')


def inventory(venv):
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','VIRTUAL_ENV'))}
    env.update({'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'})
    code="import importlib.metadata as m,json,re;print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions()}))"
    out=subprocess.check_output([str(venv/'bin/python'),'-B','-c',code],env=env,cwd='/',timeout=20,text=True)
    return json.loads(out)


def inventory_diff(manifest,group,actual):
    expected={a['name']:a['version'] for a in selected(manifest,group)}
    return {name:{'expected':expected.get(name),'actual':actual.get(name)}
            for name in expected.keys()|actual.keys() if expected.get(name)!=actual.get(name)}


def existing_state_check(repo,venv,group):
    cmd=[sys.executable,'-B',str(repo/'scripts/musetalk_install_state.py'),'check',
         '--repo-root',str(repo),'--venv',str(venv),'--matrix','cu121','--skip-weights',
         '--native-vp8','0','--kokoro','0' if group=='server' else '1',
         '--avatar-prep','1' if group=='full' else '0','--legacy-int8','0','--chin-tools','0',
         '--json']
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    return proc.returncode,json.loads(proc.stdout)


def safe_venv(venv,repo):
    require(venv.is_absolute() and venv.name not in {'','.','..'} and len(venv.parts)>=4,
            'Explicit dedicated venv path required')
    require(not venv.is_symlink() and not repo.is_relative_to(venv)
            and not venv.is_relative_to(repo),'Venv must live outside replaceable checkout')
    require(venv.parent.is_dir() and venv.parent.resolve()==venv.parent,
            'Venv parent must exist and have no symlink components')
    require(venv.parent.stat().st_uid==os.getuid(),'Foreign venv parent')


def active_venv(venv):
    """Fail closed if a live process advertises this venv in executable/argv."""
    needle=str(venv).encode()+b'/'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if needle in (proc/'cmdline').read_bytes():return True
            if str((proc/'exe').resolve()).startswith(str(venv)+'/'):return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return False


def run(args,ticket_loader=None):
    check_platform()
    range_streams,max_connections=transfer_settings(args)
    seed_site,seed_bytecode=seed_settings(args)
    started=time.monotonic()
    contract=json.loads(args.contract.read_text());check_contract(contract)
    repo=args.repo_root.resolve(strict=True)
    venv=args.venv.absolute();safe_venv(venv,repo)
    cache=args.cache_root.absolute()
    require(not cache.is_symlink() and not cache.is_relative_to(repo),
            'Dependency cache must be outside replaceable checkout')
    cache.mkdir(mode=0o700,parents=True,exist_ok=True)
    require(cache.resolve()==cache and cache.stat().st_uid==os.getuid(),'Unsafe/foreign cache')
    with fetch.lock(venv.parent/('.'+venv.name+'.dependencies.lock')):
        artifact=cache/contract['artifact_id'];artifact.mkdir(mode=0o700,exist_ok=True)
        require(not artifact.is_symlink() and artifact.stat().st_uid==os.getuid(),'Unsafe artifact cache')
        manifest_path=artifact/'manifest.json'
        ticket=None
        if not manifest_path.exists():
            require(not args.check,'Manifest cache absent; dependency installation needed')
            ticket=ticket_loader()
            metadata={'name':'dependency-manifest','filename':'manifest.json','source':'build:manifest',
                      'sha256':contract['manifest_sha256'],'size_bytes':contract['manifest_bytes']}
            fetch.acquire_one(metadata,artifact,ticket=ticket['manifest'])
        require(not manifest_path.is_symlink(),'Manifest symlink rejected')
        manifest=fetch.read_trusted(manifest_path,contract['manifest_sha256'])
        require(manifest['artifact_id']==contract['artifact_id'],'Artifact ID differs')
        check_inputs(manifest,repo)
        marker=venv/MARKER
        prior=json.loads(marker.read_text()) if marker.is_file() and not marker.is_symlink() else None
        fresh=not venv.exists()
        if venv.exists():
            require(prior and prior.get('uid')==os.getuid() and prior.get('venv')==str(venv),
                    'Existing venv is not owned by this locked installer; leave it unchanged')
            require(prior.get('artifact_id')==manifest['artifact_id'] and prior.get('group')==args.group,
                    'Different dependency contract; use a new venv and retain old one for rollback')
            diff=inventory_diff(manifest,args.group,inventory(venv)) if (venv/'bin/python').exists() else {'python':'missing'}
            state_code,state_report=existing_state_check(repo,venv,args.group) if not diff else (11,None)
            if prior.get('state')=='accepted' and not diff and state_code==0:
                result={'state':'reused','requested_mode':args.mode,'actual_mode':'verified-reuse',
                        'artifact_id':manifest['artifact_id'],'group':args.group,
                        'seconds':time.monotonic()-started,'network_requests':0,'inventory_diff':{},
                        'existing_state_check':state_report['verdict']}
                return result
            require(not args.check,'Environment is not complete; explicit repair needed')
            require(args.repair,'Incomplete/mismatched environment; --repair required')
            require(not active_venv(venv),'Refusing to repair a running environment')
            # No removed or differently versioned packages: unsafe changes need
            # a separate path, not a sync operation over a recoverable environment.
            require(all(isinstance(v,dict) and v.get('actual') is None for v in diff.values()),
                    'Only missing-package repair is supported; use a new venv for version/ABI changes')
            require(bool(diff),'Broken files with unchanged metadata need a new venv; no stamp-only repair')
        else:
            require(not args.check,'Venv missing; installation needed')
            require(shutil.disk_usage(venv.parent).free>=30*1024**3,'Insufficient disk before installation')
            venv.mkdir(mode=0o700)
        require(not args.check,'Reuse check failed')
        owner={'uid':os.getuid(),'venv':str(venv),'artifact_id':manifest['artifact_id'],
               'group':args.group,'state':'installing','created_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(marker,owner)
        logs=artifact/('install-'+str(time.time_ns()));logs.mkdir(mode=0o700)
        result={'state':'installing','requested_mode':args.mode,'actual_mode':args.mode,
                'artifact_id':manifest['artifact_id'],'group':args.group,'fallback':None,'steps':[]}
        try:
            if ticket is None:ticket=ticket_loader()
            view=manifest;seeding=None;reserve=None
            if seed_site is not None:
                seeding=seed_plan(contract,manifest,repo,venv,args.group,seed_site,seed_bytecode,fresh,args.workers)
                result['seed']=seeding['receipt']
                if seeding['staged']:
                    import dependency_seed as seed
                    # Seeded artifacts are never acquired; their staged bytes already used disk.
                    view=seed.without(manifest,[e['name'] for e in seeding['staged']])
                    reserve=seeding['receipt']['installed_bytes']
            needed={a['sha256'] for a in selected(view,args.group) if a['source'].startswith('build:')}
            cached=all((artifact/'wheels'/a['filename']).is_file() for a in selected(view,args.group))
            reserve_bytes=(4 if cached else 30)*1024**3
            if reserve is not None:reserve_bytes=max(4*1024**3,reserve_bytes-reserve)
            result['acquisition']=fetch.acquire(view,artifact/'wheels',args.group,args.workers,repo,
                reserve_bytes=reserve_bytes,
                tickets={k:v for k,v in ticket['wheels'].items() if k in needed},
                range_streams=range_streams,max_connections=max_connections)
            uv=None
            if args.mode=='locked-wheelhouse-uv':
                uv=artifact/'uv-tool/uv'
                if not uv.exists():result['tool_acquisition']=tool.fetch(artifact/'uv-tool')
                require(fetch.sha256_file(uv)==contract['uv_binary_sha256'],'Installer binary hash mismatch')
            env={k:v for k,v in os.environ.items() if not k.startswith(('AWS_','PIP_','UV_','PYTHON','VIRTUAL_ENV'))}
            env.update({'PIP_CONFIG_FILE':'/dev/null','PIP_NO_INDEX':'1','PYTHONNOUSERSITE':'1',
                        'UV_PYTHON_DOWNLOADS':'never','UV_NO_PROGRESS':'1'})
            def step(cmd,name):
                value=bench.command(cmd,name,logs,env);result['steps'].append(value)
                require(value['exit_status']==0,'Locked installer step failed: '+name)
            if not (venv/'bin/python').exists():
                result['apt_wait_seconds']=wait_for_apt_prefetch()
                step([sys.executable,'-m','venv',str(venv)],'venv')
            if seeding and seeding['staged']:
                result['seed'].update(seed.publish(seeding,venv))
            py=venv/'bin/python';wheels=artifact/'wheels';package_cache=artifact/'installer-cache'
            bootstrap=logs/'bootstrap.lock';bootstrap.write_text(offline_lock(manifest,'bootstrap'))
            step(bench.install_command('pip',py,None,package_cache,wheels,bootstrap,True,'copy'),'bootstrap')
            order=['server','server+kokoro','full']
            for group in order[:order.index(args.group)+1]:
                lock=logs/(group+'.lock');lock.write_text(offline_lock(manifest,group))
                step(bench.install_command('uv' if uv else 'pip',py,uv,package_cache,wheels,lock,True,'copy'),
                     'install-'+group)
            result['inventory_diff']=inventory_diff(manifest,args.group,inventory(venv))
            require(not result['inventory_diff'],'Locked environment inventory differs')
            step([str(py),'-m','pip','check'],'pip-check')
            result['state']='accepted';result['seconds']=time.monotonic()-started
            owner.update(state='accepted',installer=args.mode)
            atomic_json(marker,owner)
        except BaseException as error:
            owner['state']='failed';atomic_json(marker,owner)
            result.update(state='failed',error_type=type(error).__name__,
                detail=str(error) if type(error) is ValueError else 'Inspect protected step logs',
                seconds=time.monotonic()-started)
        atomic_json(logs/'receipt.json',result)
        require(result['state']=='accepted','Locked dependency install failed; no implicit fallback')
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--venv',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--mode',choices=('locked-wheelhouse-pip','locked-wheelhouse-uv'),required=True)
    p.add_argument('--group',choices=('server','server+kokoro','full'),required=True)
    p.add_argument('--workers',type=int,choices=(1,2,4,8,16),default=4)
    p.add_argument('--range-streams',type=int,choices=(0,2,4),
                   help='Byte ranges per large public wheel (default: $'+RANGE_STREAMS_ENV+' or 0 = off)')
    p.add_argument('--max-connections',type=int,choices=(4,8),
                   help='Global open-connection cap (default: $'+MAX_CONNECTIONS_ENV+' or --workers)')
    p.add_argument('--seed-site',type=Path,
                   help='Base-image site-packages to seed verified artifacts from (default: $'+SEED_SITE_ENV+' or off)')
    p.add_argument('--tickets-fd',type=int,default=0)
    p.add_argument('--check',action='store_true')
    p.add_argument('--repair',action='store_true')
    args=p.parse_args()
    def tickets():
        with os.fdopen(os.dup(args.tickets_fd)) as source:raw=source.read(256001)
        require(len(raw)<=256000,'Ticket envelope too large')
        value=json.loads(raw)
        require(set(value)=={'manifest','wheels'},'Invalid ticket envelope')
        return value
    try:
        print(json.dumps(run(args,tickets)),flush=True)
    except BaseException as error:
        print(json.dumps({'state':'failed','error_type':type(error).__name__,
            'detail':str(error) if type(error) is ValueError else 'Protected diagnostics retained'}),flush=True)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
