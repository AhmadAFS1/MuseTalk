#!/usr/bin/env python3
"""One owned source-template profiling rental; execute only on the existing EC2."""
import argparse
import base64
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
LABEL = 'musetalk-dependency-profile-20261015-a2'
ROOT = Path('/home/ec2-user/.local/state')/LABEL
WORKER_ROOT = '/workspace/.dependency-profile-20261015'
SOURCE = '5cc706e90e50e93da1310628c025a84199cd8042'
OFFER = 39627668
MACHINE = 92165
HOST = 443275
LIFETIME = 7200
DISK = 100


def load_library():
    # Preserve the historical baseline helper and its evidence. Extend only its
    # observation window, retaining ownership, exactly-one-create and cleanup.
    library = HERE/'source_library.py'
    if not library.is_file():
        library = HERE/'python_s3_venv_source_library.py'
    text = library.read_text()
    old = 'while time.monotonic()-START < 2400:'
    assert text.count(old) == 1
    text = text.replace(old, 'while time.monotonic()-START < LIFETIME_SECONDS-600:')
    # Narrow the discovery query to the reviewed country, without weakening the
    # exact offer/machine and budget checks. Otherwise 100 cheap global offers
    # can hide a perfectly available US slot.
    search = '"num_gpus": {"eq": 1}, "gpu_name": {"eq": "RTX 3090"}, "allocated_storage": DISK,'
    assert text.count(search)==1
    text = text.replace(search, search+' "geolocation": {"in": ["US"]},')
    spec = importlib.util.spec_from_file_location('dependency_source', library)
    b = importlib.util.module_from_spec(spec)
    exec(compile(text, str(library), 'exec'), b.__dict__)
    b.LABEL, b.ROOT, b.ATTEMPT, b.SOURCE = LABEL, ROOT, 'profile1', SOURCE
    b.DISK, b.LIFETIME_SECONDS = DISK, LIFETIME
    original = b.acceptable
    def acceptable(row, allow_unverified=False):
        return (row.get('machine_id') == MACHINE and row.get('host_id') == HOST
                and row.get('geolocation') == 'Pennsylvania, US'
                and original(row, allow_unverified=True)
                and all(type(row.get(k)) in (int,float) and 0 < row[k] <= .30
                        for k in ('dph_total','dph_total_adj')))
    b.acceptable = acceptable
    def select(rows, offer_id, allow_unverified=False):
        selected = [r for r in rows if acceptable(r)]
        b.require(bool(selected), 'Reviewed machine unavailable/outside original caps; no replacement create')
        # Vast changes equivalent slot IDs as other slots are rented/released.
        # Bind the hardware/host/country/quote, resolving the ID before one create.
        return min(selected,key=lambda r:(r['dph_total_adj'],r['id']))
    b.select_offer = select
    return b


def timer(b, deadline):
    when = dt.datetime.fromisoformat(deadline).strftime('%Y-%m-%d %H:%M:%S UTC')
    subprocess.run(['systemd-run','--quiet','--unit='+LABEL+'-expiry','--on-calendar='+when,
                    '--timer-property=AccuracySec=1s','--property=Restart=on-failure','--property=RestartSec=30s',
                    '/home/ec2-user/lingua/venv/bin/python','-B',str(Path(__file__).resolve()),'expire'],
                   check=True, capture_output=True, timeout=15)
    b.require(subprocess.run(['systemctl','is-active',LABEL+'-expiry.timer'], capture_output=True).returncode==0,
              'Independent expiry not active')


def ssh_arguments(row):
    ports = {str(p.get('HostPort')) for p in (row.get('ports') or {}).get('22/tcp',[])}
    if 'jupyter' in str(row.get('image_runtype','')) and row.get('ssh_host'):
        # Official Vast CLI uses ssh_port+1 for Jupyter's proxy SSH endpoint.
        # This host's direct port timed out; the documented proxy was verified.
        host,port=row['ssh_host'],str(int(row['ssh_port'])+1)
    elif len(ports)==1 and row.get('public_ipaddr'):
        host, port = row['public_ipaddr'], ports.pop()
        import ipaddress
        assert ipaddress.ip_address(host).is_global
    else:
        host, port = row.get('ssh_host'), str(row.get('ssh_port',''))
    assert host and re.fullmatch(r'[A-Za-z0-9.-]+', host) and port.isdigit() and 1 <= int(port) <= 65535
    return ['ssh','-i','/home/ec2-user/.ssh/id_ed25519','-o','BatchMode=yes',
            '-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',
            '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(ROOT/'known_hosts'),
            '-p',port,'root@'+host,'python3 -B -']


def snapshot(b, row):
    offsets_file = ROOT/'collection-offsets.json'
    offsets = json.loads(offsets_file.read_text()) if offsets_file.is_file() else {}
    # Incremental bounded reads, not a long foreground SSH stream. Worker setup
    # and the sampler are independent of these short-lived SSH connections.
    code = '''import base64,gzip,json,pathlib,subprocess,time,datetime
root=pathlib.Path(WORKER_ROOT)
paths=list(root.glob('*.jsonl'))+list(root.glob('*.debug.log'))+list(root.glob('*.report.json'))
paths += [pathlib.Path(s) for s in ['/workspace/bootstrap.log','/workspace/onstart.log','/workspace/logs/musetalk/api_server_8000.log','/workspace/MuseTalk/.runtime/gpu_selftest.json','/workspace/MuseTalk/.runtime/install_state.json']]
offsets=OFFSETS
files=[];budget=3*1024**2
for p in paths:
 if not p.is_file() or p.is_symlink() or budget<=0:continue
 size=p.stat().st_size;key=str(p);start=offsets.get(key,0)
 if start>size:start=0
 with p.open('rb') as f:
  f.seek(start);data=f.read(min(budget,1024**2))
 if data:
  files.append({'path':key,'start':start,'end':start+len(data),'size':size,'data':base64.b64encode(gzip.compress(data)).decode()});budget-=len(data)
try:revision=subprocess.check_output(['git','-C','/workspace/MuseTalk','rev-parse','HEAD'],stderr=subprocess.DEVNULL,timeout=2,text=True).strip()
except Exception:revision=None
health=None
try:
 import urllib.request
 with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=1) as response:
  health={'http_status':response.status,'body':json.loads(response.read(256000))}
except Exception:pass
print(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':revision,'health':health,'files':files}))
'''.replace('WORKER_ROOT',repr(WORKER_ROOT)).replace('OFFSETS',repr(offsets))
    try:
        r = subprocess.run(ssh_arguments(row), input=code.encode(), capture_output=True, timeout=18)
        if r.returncode:
            return False
        value = json.loads(r.stdout)
        dest = ROOT/'worker-files'
        dest.mkdir(mode=0o700, exist_ok=True)
        summaries = []
        for f in value['files']:
            key = f['path']
            assert (key.startswith(WORKER_ROOT+'/') or key in {
                '/workspace/bootstrap.log','/workspace/onstart.log','/workspace/logs/musetalk/api_server_8000.log',
                '/workspace/MuseTalk/.runtime/gpu_selftest.json','/workspace/MuseTalk/.runtime/install_state.json'})
            target = dest/(hashlib.sha256(key.encode()).hexdigest()[:10]+'-'+Path(key).name)
            data = gzip.decompress(base64.b64decode(f['data']))
            assert len(data)==f['end']-f['start'] and len(data)<=1024**2
            # Known credential material is redacted before storage. Installer
            # profiling does not print env/header values or signed URLs.
            clean = b.scrub(data.decode(errors='replace')).encode()
            with target.open('ab' if f['start'] else 'wb') as out:
                out.write(clean)
            offsets[key] = f['end']
            summaries.append({k:f[k] for k in ('path','end','size')})
        b.save('collection-offsets.json', offsets)
        b.save('collection-latest.json', {'at_utc':value['at_utc'],'files':summaries,
                'total_collected_source_bytes':sum(offsets.values()),'source_revision':value['source_revision']})
        b.save('worker-latest.json', {k:v for k,v in value.items() if k!='files'})
        return value
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False


def archive_evidence(b,row):
    """Copy all logs in one compressed stream before teardown, not only tails."""
    import io
    import tarfile
    code='''import gzip,io,pathlib,sys,tarfile
root=pathlib.Path(WORKER_ROOT)
files=list(root.glob('*.jsonl'))+list(root.glob('*.debug.log'))+list(root.glob('*.report.json'))
files += [pathlib.Path(p) for p in ['/workspace/bootstrap.log','/workspace/onstart.log','/workspace/logs/musetalk/api_server_8000.log','/workspace/MuseTalk/.runtime/gpu_selftest.json','/workspace/MuseTalk/.runtime/install_state.json']]
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w') as archive:
 for p in files:
  if p.is_file() and not p.is_symlink():archive.add(str(p),arcname=p.name,recursive=False)
sys.stdout.buffer.write(gzip.compress(out.getvalue(),compresslevel=1))
'''.replace('WORKER_ROOT',repr(WORKER_ROOT))
    r=subprocess.run(ssh_arguments(row),input=code.encode(),capture_output=True,timeout=45)
    b.require(r.returncode==0 and 0<len(r.stdout)<200*1024**2,'Complete evidence stream failed')
    destination=ROOT/'complete-worker-files'
    destination.mkdir(mode=0o700,exist_ok=True)
    receipts=[]
    with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as archive:
        for member in archive:
            b.require(member.isfile() and '/' not in member.name and member.size<=200*1024**2,
                      'Unexpected evidence archive member')
            b.require(re.fullmatch(r'pip-[a-f0-9]{32}\.(?:jsonl|debug\.log|report\.json)',member.name)
                      or member.name in {'resources.jsonl','bootstrap.log','onstart.log','api_server_8000.log',
                                             'gpu_selftest.json','install_state.json'},'Unexpected evidence filename')
            data=archive.extractfile(member).read()
            (destination/member.name).write_bytes(data)
            receipts.append({'name':member.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (ROOT/'complete-worker-evidence.tar.gz').write_bytes(r.stdout)
    b.save('complete-evidence-receipt.json',{'at_utc':b.utc(),'compressed_bytes':len(r.stdout),
        'archive_sha256':hashlib.sha256(r.stdout).hexdigest(),'files':receipts,
        'scope':'private raw logs; do not publish debug URLs without sanitization'})
    b.event('COMPLETE_EVIDENCE_COLLECTED',files=len(receipts),compressed_bytes=len(r.stdout))


def freeze(b, original, template):
    env, script, receipt = original(template)
    patch = (HERE/'python_s3_venv_timing.patch').read_bytes()
    profile = (HERE/'dependency_startup_profile.py').read_bytes()
    deadline = int(time.time())+LIFETIME-30
    env.update({'MUSETALK_DEPENDENCY_PROFILE_DIR':WORKER_ROOT,'PYTHONUNBUFFERED':'1'})
    profile_payload=base64.b64encode(gzip.compress(profile)).decode()
    install = 'import base64,gzip,os,pathlib;os.umask(0o077);p=pathlib.Path('+repr(WORKER_ROOT)+');p.mkdir(parents=True,exist_ok=True,mode=0o700);(p/"profile.py").write_bytes(gzip.decompress(base64.b64decode('+repr(profile_payload)+')))'
    import shlex
    prefix = ('\npython3 -B -c '+shlex.quote(install)+'\n'
              'python3 -u -B '+WORKER_ROOT+'/profile.py observe '+str(deadline)+' >/dev/null 2>&1 &\n')
    script = script.replace('set -Eeuo pipefail','set -Eeuo pipefail'+prefix,1)
    script = script.replace('exec > >(tee -a "$BOOT_LOG") 2>&1',
              'exec > >(python3 -u -B '+WORKER_ROOT+'/profile.py stream | tee -a "$BOOT_LOG") 2>&1',1)
    marker = 'echo "[source-baseline] checkout_ready $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"'
    b.require(script.count(marker)==1,'Missing verified-checkout marker')
    # Only replace the canonical installer's exact pip entrypoints; all original
    # package arguments/pins/indexes and fail-fast exit handling remain intact.
    transform = '''import base64,gzip,pathlib
import subprocess
patch=gzip.decompress(base64.b64decode(PATCH))
subprocess.run(['git','apply','--check','-'],input=patch,check=True)
subprocess.run(['git','apply','-'],input=patch,check=True)
p=pathlib.Path('scripts/install_musetalk.sh');s=p.read_text();old='"$VENV_PY" -m pip';n=s.count(old)
assert n>=8
s=s.replace(old,'"$VENV_PY" -B '+PROFILE_SCRIPT+' pip')
p.write_text(s)
subprocess.run(['bash','-n',str(p)],check=True)
print('[dependency-profile] canonical pip entrypoints instrumented:',n,flush=True)
'''.replace('PATCH',repr(base64.b64encode(gzip.compress(patch)).decode())).replace('PROFILE_SCRIPT',repr(WORKER_ROOT+'/profile.py'))
    compile(transform, '<worker-install-instrumentation>', 'exec')
    compile(install, '<worker-profiler-staging>', 'exec')
    # Stdin-fed Python has no secret source/config; encoded to avoid shell quoting.
    transform_cmd = 'python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(base64.b64encode(transform.encode()).decode())+'))')
    script = script.replace(marker,marker+'\n'+transform_cmd,1)
    subprocess.run(['bash','-n'], input=script.encode(),check=True,capture_output=True)
    expanded_hash=hashlib.sha256(script.encode()).hexdigest()
    # Keep Vast's launch arguments short; instrumentation is compressed, not
    # published or served from an unauthenticated URL.
    payload=base64.b64encode(gzip.compress(script.encode())).decode()
    unpack='import base64,gzip,os,pathlib;os.umask(0o077);p=pathlib.Path("/workspace/.dependency-profile-launch-20261015.sh");p.parent.mkdir(exist_ok=True);p.write_bytes(gzip.decompress(base64.b64decode('+repr(payload)+')));os.execv("/bin/bash",["bash",str(p)])'
    script='#!/usr/bin/env bash\nexec python3 -B -c '+shlex.quote(unpack)+'\n'
    b.require(len(script.encode())<12000,'Compressed profiling launch still too long; do not submit')
    subprocess.run(['bash','-n'],input=script.encode(),check=True,capture_output=True)
    receipt.update(run_onstart_sha256=hashlib.sha256(script.encode()).hexdigest(),
                   expanded_onstart_sha256=expanded_hash, compressed_onstart_bytes=len(script.encode()),
                   profiler_sha256=hashlib.sha256(profile).hexdigest(),
                   dependency_timing_patch_sha256=hashlib.sha256(patch).hexdigest(),
                   profiler_scope='observational pip hooks/debug/reports, resource sampler, UTC output prefix',
                   selected_machine_id=MACHINE, selected_host_id=HOST, lifetime_seconds=LIFETIME)
    receipt['changes'].append('observational per-package profiling; no dependency/cache/index changes')
    return env,script,receipt


def run(b):
    previous=ROOT.with_name('musetalk-dependency-profile-20261015-a1')
    rejected=json.loads((previous/'cleanup.json').read_text())
    b.require(rejected.get('provider_absence_verified') is True and rejected.get('instance_id') is None,
              'Previous rejected submission not safely reconciled')
    original_freeze, original_owned, original_save = b.frozen_template, b.owned, b.save
    attached = False
    def owned(client, ledger):
        nonlocal attached
        row = original_owned(client, ledger)
        if row and row.get('actual_status')=='running' and not attached:
            import requests
            pub = Path('/home/ec2-user/.ssh/id_ed25519.pub').read_text().strip()
            b.require(pub.startswith('ssh-ed25519 ') and '\n' not in pub,'Invalid EC2 SSH public key')
            r=requests.post(client.base_url+'/api/v0/instances/'+str(row['id'])+'/ssh/',
                            headers=client._headers(),json={'ssh_key':pub},timeout=15,allow_redirects=False)
            b.require(r.status_code==200 and r.json().get('success') is True,'Per-test SSH attach failed')
            b.event('TEST_SSH_KEY_ATTACH',instance_id=str(row['id']),scope='owned instance only')
            attached=True
        return row
    def save(name,value):
        if name=='ledger.json' and value.get('state')=='initialized':
            value['deadline_utc']=(dt.datetime.fromisoformat(value['deadline_utc'])-dt.timedelta(seconds=2)).isoformat()
            value['purpose']='fresh pip dependency startup profiling; no S3 venv restore'
        return original_save(name,value)
    b.owned, b.save = owned, save
    original_bootstrap=b.bootstrap
    def bootstrap():
        client=original_bootstrap()
        original_create=client.create_instance
        def create_instance(*,offer_id,create_request):
            try:
                return original_create(offer_id=offer_id,create_request=create_request)
            except Exception as error:
                message=str(error)
                status=re.search(r'with status (\d+)',message)
                limit=re.search(r'len\(args\)\s*>\s*(\d+)',message)
                b.event('CREATE_REJECTED_OR_UNCERTAIN',error_type=type(error).__name__,
                    http_status=int(status[1]) if status else None,
                    argument_length_rejection=bool(limit),argument_length_limit=int(limit[1]) if limit else None,
                    insufficient_credit='insufficient' in message.lower() and any(s in message.lower() for s in ('balance','credit','fund')),
                    no_such_ask='no_such_ask' in message.lower(),raw_error_suppressed=True)
                raise
        client.create_instance=create_instance
        return client
    b.bootstrap=bootstrap
    b.frozen_template=lambda t:freeze(b,original_freeze,t)
    b.timer=lambda deadline:timer(b,deadline)
    b.snapshot=lambda row:snapshot(b,row)
    b.SMOKE_PHASE_SECONDS=120
    original_cleanup=b.cleanup
    def cleanup(client,ledger):
        # Collect bounded remaining evidence before deleting only the owned GPU.
        try:
            row=original_owned(client,ledger)
            if row:
                for _ in range(4):
                    if not snapshot(b,row):
                        break
                    sizes=json.loads((ROOT/'collection-latest.json').read_text())['files']
                    if not any(f['end']<f['size'] for f in sizes):
                        break
        finally:
            original_cleanup(client,ledger)
    b.cleanup=cleanup
    b.run(OFFER,allow_unverified=True)


def observe(b):
    """Reattach only the existing owned instance. This mode cannot create one."""
    b.require(os.getuid()==0 and ROOT.is_dir(),'Existing owned profiling run required')
    ledger=json.loads((ROOT/'ledger.json').read_text())
    b.require(ledger.get('state')=='created','No active owned rental')
    b.require(subprocess.run(['systemctl','is-active',LABEL+'-expiry.timer'],capture_output=True).returncode==0,
              'Original independent expiry must remain active')
    b.require(subprocess.run(['systemctl','is-active',LABEL+'-run.service'],capture_output=True).returncode!=0,
              'Original collector must be stopped before handoff')
    b.START=time.monotonic()-(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(ledger['request_started_utc'])).total_seconds()
    b.event('COLLECTOR_HANDOFF',reason='Verified Jupyter SSH proxy instead of unreachable direct mapping',
            instance_restarted=False,create_called=False,deadline_utc=ledger['deadline_utc'])
    client=b.bootstrap()
    last_inventory=0;row=None
    try:
        while (dt.datetime.fromisoformat(ledger['deadline_utc'])-dt.datetime.now(dt.timezone.utc)).total_seconds()>300:
            if row is None or time.monotonic()-last_inventory>30:
                row=b.owned(client,ledger);last_inventory=time.monotonic()
                b.require(row is not None,'Owned worker absent')
                b.save('provider-latest.json',{k:row.get(k) for k in ('id','label','machine_id','gpu_name','num_gpus',
                    'actual_status','cur_state','public_ipaddr','ports','ssh_host','ssh_port','image_runtype')})
            sample=snapshot(b,row)
            if sample:
                if not (ROOT/'first-proxy-ssh.json').exists():
                    b.save('first-proxy-ssh.json',{'at_utc':sample['at_utc'],'source_revision':sample['source_revision']})
                    b.event('SSH_READY',worker_utc=sample['at_utc'],route='Vast Jupyter proxy',
                            earlier_boot_evidence_preserved=True)
                if sample.get('source_revision'):
                    b.require(sample['source_revision']==SOURCE,'Wrong source checkout')
                health=sample.get('health') or {}
                if health.get('http_status')==200 and health.get('body',{}).get('ok') is True:
                    b.save('health.json',health)
                    b.event('HEALTH_READY',route='worker_loopback_over_owned_SSH',
                            health_not_full_call_acceptance=True)
                    # Validate the REST API over a loopback-only EC2 SSH tunnel.
                    # Do not open production firewall/listeners or register it.
                    import socket
                    with socket.socket() as sock:
                        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                    command=ssh_arguments(row)[:-1]
                    command[-1:-1]=['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{port}:127.0.0.1:8000']
                    tunnel=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    try:
                        time.sleep(2)
                        b.require(tunnel.poll() is None,'Owned API tunnel did not start')
                        b.SMOKE_PHASE_SECONDS=120
                        b.smoke('http://127.0.0.1:'+str(port),b.BOOTSTRAP.parent/'startup-speech.wav')
                    except Exception as error:
                        b.event('OPTIONAL_VIDEO_FAILED',error_type=type(error).__name__,
                                safe_detail=str(error) if type(error) is ValueError else 'Details suppressed')
                    finally:
                        tunnel.terminate()
                        try:tunnel.wait(timeout=5)
                        except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait(timeout=5)
                    b.event('PROFILE_COMPLETE',production_or_WebRTC_acceptance=False)
                    break
            time.sleep(10)
        else:
            b.event('PROFILE_OBSERVATION_DEADLINE',original_deadline_unchanged=True)
    finally:
        try:
            if row:
                snapshot(b,row)
                archive_evidence(b,row)
        finally:
            b.cleanup(client,ledger)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','expire','cleanup','status','collect','preflight','observe'])
    args=p.parse_args()
    b=load_library()
    try:
        if args.mode=='preflight':
            import requests
            client=b.bootstrap()
            rows=client.search_offers({'limit':100,'type':'ondemand','rentable':{'eq':True},
                'gpu_name':{'eq':'RTX 3090'},'num_gpus':{'eq':1},'allocated_storage':DISK,
                'geolocation':{'in':['US']},'dph_total':{'lte':.3},
                'inet_down_cost':{'lte':.0015},'inet_up_cost':{'lte':.0015}})
            offer=b.select_offer(rows,OFFER)
            r=requests.get(client.base_url+'/api/v0/template/',headers=client._headers(),
                params={'select_filters':json.dumps({'hash_id':{'eq':b.TEMPLATE}})},timeout=20)
            b.require(r.status_code==200,'Template read failed')
            templates=[t for t in r.json()['templates'] if t.get('hash_id')==b.TEMPLATE]
            b.require(len(templates)==1,'Template ambiguous')
            env,script,receipt=freeze(b,b.frozen_template,templates[0])
            pub=Path('/home/ec2-user/.ssh/id_ed25519.pub').read_text().split()[:2]
            derived=subprocess.check_output(['ssh-keygen','-y','-f','/home/ec2-user/.ssh/id_ed25519'],timeout=5,text=True).split()[:2]
            b.require(pub==derived,'SSH keypair mismatch')
            print(json.dumps({'preflight_pass':True,'create_called':False,'quote':{
                k:offer.get(k) for k in ('id','machine_id','host_id','geolocation','dph_total','dph_total_adj',
                    'inet_down','inet_up','inet_down_cost','inet_up_cost','cpu_cores_effective','cpu_ram','verification')},
                'template_receipt':receipt,'source_verified':SOURCE}))
        elif args.mode=='run':
            run(b)
        elif args.mode=='observe':
            observe(b)
        else:
            b.require(os.getuid()==0 and ROOT.is_dir() and not ROOT.is_symlink(),'Wrong profiling root')
            ledger=json.loads((ROOT/'ledger.json').read_text())
            if args.mode in {'expire','cleanup'}:
                if args.mode=='expire':
                    b.require(dt.datetime.now(dt.timezone.utc)>=dt.datetime.fromisoformat(ledger['deadline_utc']),'Expiry too early')
                if ledger['state']!='initialized':
                    b.cleanup(b.bootstrap(),ledger)
            elif args.mode=='collect':
                row=b.owned(b.bootstrap(),ledger)
                b.require(row is not None,'Owned worker absent')
                print(json.dumps({'collected':bool(snapshot(b,row))}))
            else:
                print(json.dumps({'ledger':ledger,'collection':json.loads((ROOT/'collection-latest.json').read_text())
                                  if (ROOT/'collection-latest.json').is_file() else None}))
        return 0
    except Exception as error:
        print(json.dumps({'operation_failed':True,'error_type':type(error).__name__,
                          'safe_detail':str(error) if type(error) is ValueError else 'Details suppressed'}),flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
