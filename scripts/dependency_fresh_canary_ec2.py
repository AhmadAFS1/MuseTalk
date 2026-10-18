#!/usr/bin/env python3
"""One fresh, bounded second-host canary; no production/default mutation.

Uses the existing stock template with pinned source plus a small hash-bound
private source overlay. All private read tickets stay in memory/launch env.
Exactly one create intent; observation never retries creation.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
import uuid

import dependency_optimization_ec2 as c
import dependency_wheel_publish_ec2 as publish
from dependency_artifacts import require

LABEL = 'musetalk-dependency-canary-20261016-a1'
ROOT = Path('/home/ec2-user/.local/state') / LABEL
WORKER_ROOT = '/workspace/.dependency-canary-20261016-a1'
MACHINE, HOST, LOCATION = 136634, 399360, 'United States, US'
FILES = ('scripts/install_musetalk.sh', 'scripts/musetalk_install_state.py',
         'scripts/install_locked_dependencies.py', 'scripts/dependency_artifacts.py',
         'scripts/fetch_dependency_tool.py', 'scripts/fetch_dependency_wheels.py',
         'scripts/benchmark_dependency_install.py',
         'configs/dependencies/rtx3090-cp310-cu121-screening.json')
PUBLICATION = 'fresh-canary-source-publication.json'


def source_bundle(repo):
    files = {}
    for name in FILES:
        path = repo / name
        require(path.is_file() and not path.is_symlink(), 'Missing reviewed source overlay')
        data = path.read_bytes()
        files[name] = {'sha256': hashlib.sha256(data).hexdigest(),
                       'data': base64.b64encode(data).decode()}
    raw = json.dumps({'files': files}, sort_keys=True, separators=(',', ':')).encode()
    require(len(raw) < 512000, 'Source overlay too large')
    return raw, hashlib.sha256(raw).hexdigest()


def publish_source(repo):
    """Small immutable private source artifact, not a CUDA wheel mirror."""
    from botocore.exceptions import ClientError
    raw, sha = source_bundle(repo)
    s3 = publish.checked_s3()
    contract = json.loads((repo / FILES[-1]).read_text())
    key = publish.PRIVATE_PREFIX + contract['artifact_id'] + '/bootstrap-' + sha + '.json'
    existed = False
    try:
        s3.head_object(Bucket=publish.BUCKET, Key=key, ExpectedBucketOwner=publish.OWNER)
        existed = True
    except ClientError as error:
        require(error.response['Error']['Code'] in {'404', 'NoSuchKey', 'NotFound'},
                'Source publication permission/identity unavailable')
    if not existed:
        s3.put_object(Bucket=publish.BUCKET, Key=key, ExpectedBucketOwner=publish.OWNER,
                      Body=raw, ServerSideEncryption='AES256', IfNoneMatch='*',
                      ContentType='application/json', Metadata={'sha256': sha})
    response = s3.get_object(Bucket=publish.BUCKET, Key=key, ExpectedBucketOwner=publish.OWNER)
    require(response['ContentLength'] == len(raw) and response.get('ServerSideEncryption') == 'AES256',
            'Source overlay size/encryption mismatch')
    with response['Body'] as body:
        require(hashlib.sha256(body.read(512001)).hexdigest() == sha, 'Published overlay hash differs')
    receipt = {'state': 'published-verified-private', 'key': key, 'sha256': sha,
               'bytes': len(raw), 'files': {n: v['sha256'] for n, v in json.loads(raw)['files'].items()},
               'existing_verified': existed, 'IAM_changed': False}
    target = c.HERE / PUBLICATION
    target.write_text(json.dumps(receipt, indent=2) + '\n'); os.chmod(target, 0o600)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}), flush=True)


def bootstrap_program(sha):
    require(len(sha) == 64 and all(x in '0123456789abcdef' for x in sha), 'Untrusted overlay hash')
    code = '''import base64,hashlib,json,os,pathlib,urllib.request
try:
 repo=pathlib.Path('/workspace/MuseTalk');root=pathlib.Path(WORKER_ROOT)
 assert repo.is_dir() and repo.resolve()==repo and not root.exists()
 request=urllib.request.Request(os.environ['MUSETALK_CANARY_SOURCE_TICKET'],headers={'x-amz-expected-bucket-owner':OWNER})
 with urllib.request.urlopen(request,timeout=45) as response:data=response.read(512001)
 assert len(data)<512001 and hashlib.sha256(data).hexdigest()==SHA
 bundle=json.loads(data);assert set(bundle)=={'files'} and set(bundle['files'])==set(FILES)
 decoded={}
 for name,item in bundle['files'].items():
  content=base64.b64decode(item['data'],validate=True)
  assert hashlib.sha256(content).hexdigest()==item['sha256']
  path=repo/name;assert not path.is_symlink() and path.parent.resolve()==path.parent
  decoded[name]=content
 root.mkdir(mode=0o700);backup=root/'source-original';backup.mkdir(mode=0o700)
 for name,content in decoded.items():
  path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():
   old=backup/name;old.parent.mkdir(parents=True,exist_ok=True);old.write_bytes(path.read_bytes())
  path.write_bytes(content);os.chmod(path,0o600)
 (root/'source-overlay.json').write_text(json.dumps({'sha256':SHA,'files':{n:bundle['files'][n]['sha256'] for n in FILES}}))
 print('[canary] reviewed source overlay ready',flush=True)
except BaseException:
 print('[canary] source delivery failed; private URI suppressed',flush=True)
 raise SystemExit(2)
'''
    # Replace whole placeholder identifiers, never user data/shell evaluation.
    values = {'WORKER_ROOT': repr(WORKER_ROOT), 'OWNER': repr(publish.OWNER),
              'SHA': repr(sha), 'FILES': repr(FILES)}
    code = re.sub(r'\b(WORKER_ROOT|OWNER|SHA|FILES)\b', lambda m: values[m[0]], code)
    compile(code, '<canary-bootstrap>', 'exec')
    return code


def freeze(b, template):
    env, script, receipt = b.frozen_template(template)
    source = json.loads((c.HERE / PUBLICATION).read_text())
    require(source['state'] == 'published-verified-private', 'Private source overlay not verified')
    contract = json.loads((c.HERE / 'dependency-contract.json').read_text())
    manifest = json.loads((c.HERE / 'wheelhouse-manifest-a2.json').read_text())
    require(manifest['artifact_id'] == contract['artifact_id'], 'Reviewed wheel manifest differs')
    s3 = publish.checked_s3()
    key = contract['prefix'] + 'manifest.json'
    envelope = {'manifest': {'sha256': contract['manifest_sha256'], 'owner': contract['owner'],
                            'key': key, 'url': publish.presign_get(s3, key, 3600)},
                'wheels': publish.tickets(manifest, expires=3600)}
    env.update({'MUSETALK_CANARY_SOURCE_TICKET': publish.presign_get(s3, source['key'], 3600),
                'MUSETALK_CANARY_TICKETS': json.dumps(envelope),
                'MUSETALK_DEPENDENCY_INSTALL_MODE': 'locked-wheelhouse-uv',
                'MUSETALK_DEPENDENCY_CONTRACT': '/workspace/MuseTalk/' + FILES[-1],
                'MUSETALK_DEPENDENCY_CACHE': '/workspace/.cache/musetalk-dependencies',
                'MUSETALK_DEPENDENCY_WORKERS': '4',
                'MUSETALK_RUNTIME_CONFIG_SOURCE': 'injected'})
    marker = 'echo "[source-baseline] checkout_ready $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"'
    require(script.count(marker) == 1, 'Unique frozen checkout marker required')
    injection = '\npython3 -B -c ' + shlex.quote(bootstrap_program(source['sha256'])) + '''
exec 8< <(printf '%s' "$MUSETALK_CANARY_TICKETS")
unset MUSETALK_CANARY_TICKETS MUSETALK_CANARY_SOURCE_TICKET
export MUSETALK_DEPENDENCY_TICKET_FD=8
'''
    script = script.replace(marker, marker + injection, 1)
    subprocess.run(['bash', '-n'], input=script.encode(), check=True, capture_output=True)
    require(len(script.encode()) < 12000, 'Bounded onstart size exceeded')
    receipt.update(run_onstart_sha256=hashlib.sha256(script.encode()).hexdigest(),
                   source_overlay=source, test_scope='fresh second-host stock-template opt-in canary')
    receipt['changes'].append('private hash-bound final installer overlay; four-way locked compiled uv')
    return env, script, receipt


def expiry_timer(deadline):
    when = dt.datetime.fromisoformat(deadline).strftime('%Y-%m-%d %H:%M:%S UTC')
    subprocess.run(['systemd-run', '--quiet', '--unit=' + LABEL + '-expiry', '--on-calendar=' + when,
                    '--timer-property=AccuracySec=1s', '--property=Restart=on-failure', '--property=RestartSec=30s',
                    '/home/ec2-user/lingua/venv/bin/python', '-B', str(Path(__file__).resolve()), 'expire'],
                   check=True, capture_output=True, timeout=15)
    require(subprocess.run(['systemctl', 'is-active', LABEL + '-expiry.timer'],
                           capture_output=True, timeout=10).returncode == 0, 'Canary expiry not active')


def configure():
    c.LABEL, c.ROOT, c.WORKER_ROOT = LABEL, ROOT, WORKER_ROOT
    c.MACHINE, c.HOST, c.LOCATION = MACHINE, HOST, LOCATION
    c.LIFETIME, c.DISK = 3600, 100
    c.PURPOSE = 'fresh verified US second-host canary; final locked installer on stock template'
    c.approved = lambda row, b: (row.get('machine_id') == MACHINE and row.get('host_id') == HOST
        and row.get('geolocation') == LOCATION and row.get('verification') == 'verified'
        and b.acceptable(row, allow_unverified=False))
    c.freeze, c.expiry_timer = freeze, expiry_timer
    return c.library()


def acceptance(b, attempt='a1'):
    """Fresh stock boot proof plus decoded REST output, not WebRTC/TURN."""
    ledger = json.loads((ROOT / 'ledger.json').read_text())
    require(ledger['state'] == 'created' and dt.datetime.now(dt.timezone.utc)
            < dt.datetime.fromisoformat(ledger['deadline_utc'])-dt.timedelta(seconds=600),
            'No sufficient live canary lease')
    require(attempt in {'a1','a2'}, 'Unreviewed acceptance attempt')
    if attempt=='a2':
        previous=json.loads((ROOT/'acceptance/receipt.json').read_text())
        require(previous.get('state')=='failed' and previous.get('detail')=='Canary API health deadline',
                'Only the terminal health-wait deadline permits this bounded acceptance retry')
        require(subprocess.run(['systemctl','is-active',LABEL+'-accept.service'],capture_output=True).returncode!=0,
                'Original acceptance service is still live')
    dest = ROOT / ('acceptance' if attempt=='a1' else 'acceptance-a2')
    require(not dest.exists(), 'Canary acceptance already claimed; inspect existing service')
    dest.mkdir(mode=0o700)
    client=b.bootstrap(); row=b.owned(client,ledger)
    require(row and row['actual_status']=='running', 'Canary not running')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    ssh=c.ssh_arguments(row)
    command=ssh[:-2]+['-N','-o','ExitOnForwardFailure=yes','-L',f'127.0.0.1:{port}:127.0.0.1:8000',ssh[-2]]
    tunnel=None; result={'state':'running','cases':[]}
    try:
        with (dest/'tunnel.log').open('xb') as log:
            tunnel=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
        base=f'http://127.0.0.1:{port}'
        until=time.monotonic()+600
        while time.monotonic()<until:
            require(tunnel.poll() is None,'Canary forwarding process ended')
            status,body=b.probe(base,'/health')
            if status==200 and body and body.get('ok') is True:break
            time.sleep(3)
        else:raise ValueError('Canary API health deadline')
        payload='''import json,os,pathlib,subprocess
repo=pathlib.Path('/workspace/MuseTalk');root=pathlib.Path(ROOT)
assert 'VAST_ONSTART COMPLETE:' in pathlib.Path('/workspace/onstart.log').read_text()
env=dict(os.environ);env.update({'MUSETALK_DEPENDENCY_INSTALL_MODE':'locked-wheelhouse-uv','MUSETALK_DEPENDENCY_CONTRACT':str(repo/'configs/dependencies/rtx3090-cp310-cu121-screening.json'),'MUSETALK_DEPENDENCY_CACHE':'/workspace/.cache/musetalk-dependencies'})
p=subprocess.run(['bash','scripts/install_musetalk.sh','--check','--with-avatar-prep','--json'],cwd=repo,env=env,capture_output=True,text=True,timeout=30)
check=json.loads(p.stdout);assert p.returncode==0
overlay=json.loads((root/'source-overlay.json').read_text())
for name,sha in overlay['files'].items():
 import hashlib
 assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha
receipts=list(pathlib.Path('/workspace/.cache/musetalk-dependencies').glob('*/install-*/receipt.json'))
assert len(receipts)==1
locked=json.loads(receipts[0].read_text());assert locked['state']=='accepted' and not locked['inventory_diff']
print(json.dumps({'source_overlay':overlay,'install_check':check,'locked_install':locked}))
'''.replace('pathlib.Path(ROOT)', 'pathlib.Path('+repr(WORKER_ROOT)+')')
        result['native_install_evidence']=json.loads(c.worker(b,payload))
        source=json.loads((c.HERE/PUBLICATION).read_text())
        require(result['native_install_evidence']['source_overlay']['sha256']==source['sha256'],
                'Fresh worker did not execute reviewed source overlay')
        audio=b.BOOTSTRAP.parent/'startup-speech.wav'
        require(hashlib.sha256(audio.read_bytes()).hexdigest()=='8628701b894babe55b8e04769363925528e50c1352d3edcebf1d31a44a65247e',
                'Different canary audio input')
        for n in range(2):
            case=dest/('output-'+str(n));case.mkdir(mode=0o700)
            old=b.ROOT;start=time.monotonic()
            try:
                b.ROOT=case;b.SMOKE_PHASE_SECONDS=150;b.smoke(base,audio)
            finally:b.ROOT=old
            value=json.loads((case/'video-smoke.json').read_text())
            result['cases'].append({'case':n,'avatar_warm_plus_generation_poll_download_decode_seconds':time.monotonic()-start,**value})
        result.update(state='passed',scope='fresh second-host canonical startup/check plus decoded internal REST; not direct WebRTC/TURN',
                      output_byte_identical_between_two_requests=result['cases'][0]['video_sha256']==result['cases'][1]['video_sha256'])
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
                      detail=str(error) if type(error) is ValueError else 'Protected evidence retained')
    finally:
        if tunnel:
            tunnel.terminate()
            try:tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait(timeout=5)
        (dest/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'state':result['state'],'scope':result.get('scope'),'detail':result.get('detail'),
                      'cases':len(result['cases'])}),flush=True)
    require(result['state']=='passed','Fresh canary acceptance failed; inspect same service/receipt')


def accepted_receipt():
    for name in ('acceptance-a2','acceptance'):
        path=ROOT/name/'receipt.json'
        if path.is_file() and json.loads(path.read_text()).get('state')=='passed':return path
    raise ValueError('Fresh canary output acceptance must pass first')


def overlap_screen(b, launch=False):
    from python_s3_venv_detached import launcher, reader
    control=ROOT/'overlap-job.json'
    if launch:
        accepted_receipt()
        for unit in ('-accept.service','-accept-a2.service'):
            require(subprocess.run(['systemctl','is-active',LABEL+unit],capture_output=True).returncode!=0,
                    'Acceptance service still live; avoid competing workloads')
        require(not control.exists(),'Overlap already claimed; inspect the original handle')
        ledger=json.loads((ROOT/'ledger.json').read_text())
        require(dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(ledger['deadline_utc'])-dt.timedelta(seconds=1200),
                'Insufficient canary lease')
        raw=(c.HERE/'dependency_overlap_worker.py').read_bytes();sha=hashlib.sha256(raw).hexdigest()
        stage='''import base64,hashlib,json,os,pathlib
root=pathlib.Path(ROOT);assert not root.exists();root.mkdir(mode=0o700)
data=base64.b64decode(DATA);assert hashlib.sha256(data).hexdigest()==SHA
(root/'dependency_overlap_worker.py').write_bytes(data)
print(json.dumps({'sha256':SHA}))
'''
        values={'ROOT':repr(WORKER_ROOT+'/overlap-tools'),'DATA':repr(base64.b64encode(raw).decode()),'SHA':repr(sha)}
        stage=re.sub(r'\b(ROOT|DATA|SHA)\b',lambda m:values[m[0]],stage)
        staged=json.loads(c.worker(b,stage));require(staged['sha256']==sha,'Overlap source staging differs')
        contract=json.loads((c.HERE/'dependency-contract.json').read_text())
        manifest=json.loads((c.HERE/'wheelhouse-manifest-a2.json').read_text())
        envelope={'wheels':publish.tickets(manifest,expires=1800)}
        import boto3
        from botocore.config import Config
        from dependency_overlap_worker import KEY,SHA,SIZE
        secret=json.loads(boto3.client('secretsmanager',region_name='us-east-1').get_secret_value(
                          SecretId='lingua/musetalk-worker-runtime')['SecretString'])
        session=boto3.Session(aws_access_key_id=secret['AWS_ACCESS_KEY_ID'],
                              aws_secret_access_key=secret['AWS_SECRET_ACCESS_KEY'],region_name='us-east-1')
        require(session.client('sts').get_caller_identity()['Arn'].endswith(':user/musetalk-s3-runtime'),
                'Unexpected read identity')
        s3=session.client('s3',config=Config(signature_version='s3v4'))
        require(s3.head_object(Bucket=publish.BUCKET,Key=KEY,ExpectedBucketOwner=publish.OWNER)['ContentLength']==SIZE,
                'Pinned engine object differs')
        engine={'key':KEY,'sha256':SHA,'size':SIZE,'owner':publish.OWNER,
                'url':s3.generate_presigned_url('get_object',Params={'Bucket':publish.BUCKET,'Key':KEY,
                      'ExpectedBucketOwner':publish.OWNER},ExpiresIn=1800)}
        payload=("import sys,json;from pathlib import Path;sys.path[:0]="+repr([WORKER_ROOT+'/overlap-tools','/workspace/MuseTalk/scripts'])
            +";import dependency_overlap_worker as w;print(json.dumps({'result':w.execute(Path("+repr(WORKER_ROOT+'/overlap-a1')+") ,"
            +repr(ledger['deadline_utc'])+","+repr(envelope)+","+repr(engine)+")}))")
        compile(payload,'<private-overlap-dispatch>','exec')
        job={'job_id':uuid.uuid4().hex,'source_sha256':sha,'deadline_utc':ledger['deadline_utc']}
        b.save('overlap-job.json',job)
        result=json.loads(c.worker(b,launcher(WORKER_ROOT+'/jobs',job['job_id'],payload)))
        b.save('overlap-dispatch.json',result)
        print(json.dumps(result),flush=True)
    else:
        job=json.loads(control.read_text())
        result=json.loads(c.worker(b,reader(WORKER_ROOT+'/jobs',job['job_id'])))
        b.save('overlap-latest.json',result)
        outcome=result.get('result',{}).get('result',{})
        print(json.dumps({'job_id':job['job_id'],'transport_state':result['state'],
                          'test_state':outcome.get('state'),'reduction_percent':outcome.get('screening_reduction_percent')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('publish', 'create', 'status', 'probe', 'accept', 'overlap', 'overlap-status', 'collect', 'cleanup', 'expire'))
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--accept-attempt',choices=('a1','a2'),default='a1')
    args = parser.parse_args()
    require(os.getuid() == 0, 'Existing root-owned EC2 control plane only')
    os.umask(0o077)
    b = configure()
    if args.mode == 'publish':
        require(args.repo is not None, 'Explicit reviewed source snapshot required')
        publish_source(args.repo)
    elif args.mode == 'create': c.create(b)
    elif args.mode == 'status': c.status(b)
    elif args.mode == 'probe': c.probe(b)
    elif args.mode == 'accept': acceptance(b,args.accept_attempt)
    elif args.mode in {'overlap','overlap-status'}: overlap_screen(b,launch=args.mode=='overlap')
    elif args.mode == 'collect': c.collect(b)
    else:
        ledger = json.loads((ROOT / 'ledger.json').read_text())
        if args.mode == 'expire':
            require(dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(ledger['deadline_utc']),
                    'Expiry too early')
        b.cleanup(b.bootstrap(), ledger)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if isinstance(error, SystemExit): raise
        print(json.dumps({'state': 'failed', 'error_type': type(error).__name__,
                          'detail': str(error) if type(error) is ValueError else 'Private diagnostics suppressed'}), flush=True)
        raise SystemExit(2)
