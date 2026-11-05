#!/usr/bin/env python3
"""Build a private CPU-verified candidate. Build URLs never enter Docker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

IMAGE='ghcr.io/ahmadafs1/musetalk-rtx3090'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(args, **kw):
    return subprocess.run([str(x) for x in args],check=True,**kw)

def download(spec,target):
    assert re.fullmatch('[0-9a-f]{64}',spec['sha256']) and int(spec['bytes'])>0
    print('::add-mask::'+spec['url'],flush=True)
    h=hashlib.sha256();size=0
    try:
        with urllib.request.urlopen(spec['url'],timeout=120) as source,target.open('wb') as out:
            while block:=source.read(1024*1024):
                h.update(block);size+=len(block);out.write(block)
    except Exception:
        raise SystemExit('Private build input download failed (URL redacted)') from None
    assert size==spec['bytes'] and h.hexdigest()==spec['sha256'],'private build input hash/size differs'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',required=True);ap.add_argument('--work',required=True);a=ap.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',a.revision)
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==a.revision
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    token=os.environ['GH_TOKEN']
    req=urllib.request.Request('https://api.github.com/users/AhmadAFS1/packages/container/musetalk-rtx3090',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    package=json.load(urllib.request.urlopen(req,timeout=30));assert package['visibility']=='private','H3 code requires private image'
    inputs=json.loads(os.environ['CHARACTER_RUNTIME_PRIVATE_BUILD_INPUTS']);assert set(inputs)=={'h3','preparation_weights'}
    work=Path(a.work).resolve();work.mkdir(parents=True,exist_ok=True);ctx=work/'inputs';ctx.mkdir(exist_ok=True)
    for key,spec in inputs.items():download(spec,work/(key+'.input'))
    h3=ctx/'h3';h3.mkdir();run(['git','init','-q',h3]);run(['git','-C',h3,'fetch',work/'h3.input','HEAD']);run(['git','-C',h3,'checkout','--detach','-q','FETCH_HEAD'])
    run(['git','-C',h3,'remote','add','origin','https://github.com/AhmadAFS1/minimax-h3-workflows.git'])
    expected=json.load(open('character_factory/character_worker/locks/character_worker.lock.json'))['h3_workflows']['commit']
    assert subprocess.check_output(['git','-C',h3,'rev-parse','HEAD'],text=True).strip()==expected
    weights=ctx/'preparation-weights';weights.mkdir()
    # Archive is generated from an explicit allowlist. Reject traversal and links.
    import tarfile
    manifest=json.load(open('docker/character-runtime/preparation-models.json'))['files']
    with tarfile.open(work/'preparation_weights.input') as tf:
        members=tf.getmembers();assert {x.name for x in members}==set(manifest)
        for m in members:
            assert m.isfile() and m.size==manifest[m.name]['size_bytes']
            target=weights/m.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(m) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
            assert digest(target)==manifest[m.name]['sha256']
    run(['docker','login','ghcr.io','-u',os.environ['GITHUB_ACTOR'],'--password-stdin'],input=token.encode(),stdout=subprocess.DEVNULL)
    tag=IMAGE+':character-runtime-'+a.revision
    run(['docker','buildx','build','--platform','linux/amd64','--load','--progress','plain','--build-context','inputs='+str(ctx),
         '--build-arg','SOURCE_REVISION='+a.revision,'-f','docker/character-runtime/Dockerfile','-t',tag,'.'])
    run(['docker','run','--rm','--network','none','--entrypoint','python3',tag,
         'character_factory/character_worker/runtime_manifest.py','verify','--repo','/opt/character-runtime/app',
         '--models','docker/character-runtime/preparation-models.json'])
    cid=subprocess.check_output(['docker','create',tag],text=True).strip()
    try:run(['docker','cp',cid+':/opt/character-runtime/manifest.json',work/'runtime-manifest.json'])
    finally:run(['docker','rm',cid],stdout=subprocess.DEVNULL)
    config=json.loads(subprocess.check_output(['docker','image','inspect',tag],text=True))[0]['Config']
    assert not any(x.split('=',1)[0].startswith(('AWS_ACCESS','AWS_SECRET','AWS_SESSION','GH_TOKEN','GITHUB_TOKEN','H3_WORKFLOWS_GIT_TOKEN')) for x in config.get('Env',[]))
    run(['docker','push',tag])
    images=json.loads(subprocess.check_output(['docker','image','inspect',tag],text=True))[0]['RepoDigests'];image=next(x for x in images if x.startswith(IMAGE+'@sha256:'))
    receipt={'schema':'lingua-character-image-candidate/1','image':image,'tag':tag,'source_revision':a.revision,
             'runtime_manifest_sha256':digest(work/'runtime-manifest.json'),
             'cpu_offline_verification':'passed','gpu_qualification':'not_run','promotion_eligible':False,
             'inputs':{k:{f:v for f,v in x.items() if f!='url'} for k,x in inputs.items()}}
    (work/'candidate.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');print(json.dumps(receipt))
    with open(os.environ['GITHUB_OUTPUT'],'a') as out:out.write('image='+image+'\n')
if __name__=='__main__':main()
