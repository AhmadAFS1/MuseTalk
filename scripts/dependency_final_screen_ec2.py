#!/usr/bin/env python3
"""One-hour final-screen worker, fresh ownership ledger and independent expiry."""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

import dependency_optimization_ec2 as c
import dependency_wheel_publish_ec2 as publish
from dependency_artifacts import require
from python_s3_venv_detached import launcher, reader

LABEL='musetalk-dependency-final-screen-20261016-a1'
ROOT=Path('/home/ec2-user/.local/state')/LABEL
WORKER_ROOT='/workspace/.dependency-optimization-final-20261016-a1'
# Previously used isolated benchmark host; not an arbitrary unverified fallback.
MACHINE,HOST,LOCATION=92165,443275,'Pennsylvania, US'
FILES=('dependency_artifacts.py','fetch_dependency_wheels.py','fetch_dependency_tool.py',
       'benchmark_dependency_install.py','install_locked_dependencies.py',
       'dependency_overlap_worker.py','dependency_final_screen_worker.py')


def freeze(b, template):
    env,unused,receipt=b.frozen_template(template)
    # No repo clone, model download or app restart: these are isolated screens.
    script='''#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
test ! -e /workspace/.dependency-optimization-final-20261016-a1
mkdir -m 700 /workspace/.dependency-optimization-final-20261016-a1
printf '[final-screen] stock fixture ready\\n'
'''
    subprocess.run(['bash','-n'],input=script.encode(),capture_output=True,check=True)
    receipt.update(run_onstart_sha256=hashlib.sha256(script.encode()).hexdigest(),
                   scope='isolated acquisition/link screens; NOT canonical application boot',
                   production_template_modified=False,setup_full_stack='not run',setup_clean='not run')
    return env,script,receipt


def expiry_timer(deadline):
    when=dt.datetime.fromisoformat(deadline).strftime('%Y-%m-%d %H:%M:%S UTC')
    subprocess.run(['systemd-run','--quiet','--unit='+LABEL+'-expiry','--on-calendar='+when,
                    '--timer-property=AccuracySec=1s','--property=Restart=on-failure','--property=RestartSec=30s',
                    '/home/ec2-user/lingua/venv/bin/python','-B',str(Path(__file__).resolve()),'expire'],
                   check=True,capture_output=True,timeout=15)
    require(subprocess.run(['systemctl','is-active',LABEL+'-expiry.timer'],capture_output=True,timeout=10).returncode==0,
            'Independent final-screen expiry unavailable')


def configure():
    c.LABEL,c.ROOT,c.WORKER_ROOT=LABEL,ROOT,WORKER_ROOT
    c.MACHINE,c.HOST,c.LOCATION=MACHINE,HOST,LOCATION
    c.LIFETIME,c.DISK=3600,100
    c.PURPOSE='final full-payload acquisition overlap and native copy/hardlink screens; no application boot'
    c.approved=lambda row,b:(row.get('machine_id')==MACHINE and row.get('host_id')==HOST
                           and row.get('geolocation')==c.LOCATION and b.acceptable(row,allow_unverified=True))
    c.freeze,c.expiry_timer=freeze,expiry_timer
    return c.library()


def stage(b):
    files={n:(c.HERE/n).read_bytes() for n in FILES}
    files.update({'contract.json':(c.HERE/'dependency-contract.json').read_bytes(),
                  'manifest.json':(c.HERE/'wheelhouse-manifest-a2.json').read_bytes()})
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    encoded={n:base64.b64encode(v).decode() for n,v in files.items()}
    source='''import base64,hashlib,json,os,pathlib
root=pathlib.Path(__ROOT__);assert root.is_dir() and not root.is_symlink() and root.stat().st_uid==os.getuid()
dest=root/'tools';assert not dest.exists();dest.mkdir(mode=0o700)
files=__FILES__;hashes=__HASHES__
for name,value in files.items():
 data=base64.b64decode(value);assert hashlib.sha256(data).hexdigest()==hashes[name]
 (dest/name).write_bytes(data)
print(json.dumps({'files':hashes}))
'''
    values={'__ROOT__':repr(WORKER_ROOT),'__FILES__':repr(encoded),'__HASHES__':repr(hashes)}
    source=re.sub(r'__ROOT__|__FILES__|__HASHES__',lambda m:values[m[0]],source)
    value=json.loads(c.worker(b,source));require(value['files']==hashes,'Staged tools differ')
    b.save('staged-tools.json',value);print(json.dumps({'state':'staged','files':len(hashes)}))


def job_source(mode, deadline, tickets=None, engine=None):
    require(mode in {'prepare','acquire','links'},'Unreviewed final-screen job')
    method={'prepare':'prepare','acquire':'acquire_pair','links':'links'}[mode]
    args='Path('+repr(WORKER_ROOT)+'),'+repr(deadline)
    if mode=='acquire':
        require(isinstance(tickets,dict) and isinstance(engine,dict),'Ephemeral acquisition tickets required')
        args+=','+repr(tickets)+','+repr(engine)
    else:require(tickets is None and engine is None,'Unexpected private job inputs')
    source='import sys,json;from pathlib import Path;sys.path.insert(0,'+repr(WORKER_ROOT+'/tools')+')'
    source+=';import dependency_final_screen_worker as w;print(json.dumps({"result":w.'+method+'('+args+')}))'
    compile(source,'<private-final-screen>','exec');return source


def job(b,mode,launch):
    claim=ROOT/(mode+'-job.json')
    if not launch:
        j=json.loads(claim.read_text());source=reader(WORKER_ROOT+'/jobs',j['job_id'])
        # A running directory is not proof of a live process. Add PID evidence.
        source += '''
import os,time
p=job/'pid.json'
if p.is_file() and not (job/'receipt.json').exists():
 pid=json.loads(p.read_text())['pid'];proc=Path('/proc')/str(pid)
 if proc.exists():
  fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
  print(json.dumps({'process':{'pid':pid,'state':fields[0],'start_ticks':fields[19]}}))
 else:print(json.dumps({'process':{'pid':pid,'absent':True}}))
'''
        lines=c.worker(b,source).decode().splitlines();values=[json.loads(x) for x in lines]
        value=values[0]
        if len(values)>1:value['process']=values[1]['process']
        b.save(mode+'-latest.json',value)
        result=value.get('result',{}).get('result',{})
        print(json.dumps({'transport_state':value['state'],'test_state':result.get('state'),
                          'process':value.get('process'),'cases':len(result.get('cases',[]))}));return
    require(not claim.exists(),'Job already claimed; inspect same handle')
    ledger=json.loads((ROOT/'ledger.json').read_text())
    require(dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(ledger['deadline_utc'])-dt.timedelta(seconds=1500),
            'Insufficient unchanged one-hour lease')
    tickets=ticket=None
    if mode=='acquire':
        s3=publish.checked_s3();manifest=json.loads((c.HERE/'wheelhouse-manifest-a2.json').read_text())
        import dependency_overlap_worker as overlap
        head=s3.head_object(Bucket=publish.BUCKET,Key=overlap.KEY,ExpectedBucketOwner=publish.OWNER)
        require(head['ContentLength']==overlap.SIZE,'Engine object differs')
        ticket={'key':overlap.KEY,'sha256':overlap.SHA,'size':overlap.SIZE,'owner':publish.OWNER,
                'url':s3.generate_presigned_url('get_object',Params={'Bucket':publish.BUCKET,
                       'Key':overlap.KEY,'ExpectedBucketOwner':publish.OWNER},ExpiresIn=3600)}
        overlap.engine_ticket(ticket)
        tickets=publish.tickets(manifest,expires=3600)
    source=job_source(mode,ledger['deadline_utc'],tickets,ticket)
    j={'job_id':uuid.uuid4().hex,'deadline_utc':ledger['deadline_utc'],'scope':mode}
    b.save(mode+'-job.json',j)
    value=json.loads(c.worker(b,launcher(WORKER_ROOT+'/jobs',j['job_id'],source)))
    b.save(mode+'-dispatch.json',value);print(json.dumps(value))


def export(b):
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None
    value={'prepared_utc':b.utc(),'goal_complete':False,
           'scope':'isolated full acquisition overlap and local native link screens; not total startup',
           'offer':read(ROOT/'offer.json'),'ledger':read(ROOT/'ledger.json'),
           'cleanup':read(ROOT/'cleanup.json'),'staged_tools':read(ROOT/'staged-tools.json'),
           'final_inventory':read(ROOT/'final-inventory.json'),
           'evidence_inventory':read(ROOT/'evidence-inventory.json'),
           'trials':{p.parent.name:read(p) for p in (ROOT/'evidence').glob('*/receipt.json')},
           'jobs':{n:read(ROOT/(n+'-latest.json')) for n in ('prepare','acquire','links')}}
    raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
    require(not re.search(r'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|X-Amz-(?:Credential|Signature|Security-Token)',
                          raw,re.I),'Credential pattern in public evidence; keep private')
    target=c.HERE/'final-screen-public-results.json';target.write_text(raw);os.chmod(target,0o600)
    os.chown(target,c.HERE.stat().st_uid,c.HERE.stat().st_gid)
    print(json.dumps({'state':'sanitized-export-ready','bytes':len(raw.encode()),
                      'sha256':hashlib.sha256(raw.encode()).hexdigest(),'trials':list(value['trials'])}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('create','status','stage','prepare','prepare-status','acquire','acquire-status',
                                 'links','links-status','collect','export','cleanup','expire'))
    args=p.parse_args();require(os.getuid()==0,'Existing root-owned EC2 control plane only');os.umask(0o077)
    b=configure()
    if args.mode=='create':c.create(b)
    elif args.mode=='status':c.status(b)
    elif args.mode=='stage':stage(b)
    elif args.mode=='collect':c.collect(b)
    elif args.mode=='export':export(b)
    elif args.mode in {'cleanup','expire'}:
        ledger=json.loads((ROOT/'ledger.json').read_text())
        if args.mode=='expire':require(dt.datetime.now(dt.timezone.utc)>=dt.datetime.fromisoformat(ledger['deadline_utc']),
                                       'Expiry too early')
        b.cleanup(b.bootstrap(),ledger)
    else:
        mode=args.mode.removesuffix('-status');job(b,mode,not args.mode.endswith('-status'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        print(json.dumps({'state':'failed','error_type':type(error).__name__,
                          'detail':str(error) if type(error) is ValueError else 'Protected diagnostics suppressed'}))
        raise SystemExit(2)
