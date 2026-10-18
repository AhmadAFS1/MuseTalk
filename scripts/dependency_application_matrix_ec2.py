#!/usr/bin/env python3
"""Same-host cold-process / warm-request acceptance after dependency trials.

Uses owned SSH forwarding and sequential isolated APIs. Never changes engines,
source packages, public routing, production registration or the original API.
"""
import datetime as dt
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import dependency_optimization_ec2 as c
from dependency_acceptance_ec2 import free_port

ATTEMPT='a2'
ISOLATE_NUMBA=False


def process_stop(b,case):
    payload='''import json,pathlib,os,signal,time,socket
root=pathlib.Path(ROOT);v=json.loads((root/'pid.json').read_text());pid=v['pid']
stat=pathlib.Path('/proc')/str(pid)/'stat'
if stat.exists():
 data=stat.read_text();assert data[data.rfind(')')+2:].split()[19]==v['start_ticks'] and os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM)
 until=time.monotonic()+5
 while stat.exists() and time.monotonic()<until:time.sleep(.1)
 if stat.exists():
  data=stat.read_text();assert data[data.rfind(')')+2:].split()[19]==v['start_ticks'] and os.getpgid(pid)==pid
  os.killpg(pid,signal.SIGKILL)
until=time.monotonic()+3
listening=True
while time.monotonic()<until:
 with socket.socket() as s:
  s.settimeout(.2);listening=s.connect_ex(('127.0.0.1',8001))==0
 if not listening:break
 time.sleep(.1)
assert not listening,'Owned API port still listening after stop'
print(json.dumps({'stop_sent':True,'pid':pid,'original_pid_present':stat.exists(),'port_closed':True}))
'''.replace('ROOT',repr(c.WORKER_ROOT+'/application-'+ATTEMPT+'/'+case))
    return json.loads(c.worker(b,payload))


def launch(b,case,venv):
    payload='''import json,os,pathlib,subprocess,time
root=pathlib.Path(__ROOT__);root.mkdir(mode=0o700,parents=True,exist_ok=False)
env=dict(os.environ)
for k in list(env):
 if k.startswith('LINGUA_'):env.pop(k)
env.update({'VENV_PATH':__VENV__,'PORT':'8001','HOST':'127.0.0.1','MUSETALK_RECIPE':'r5','MUSETALK_TAESD_TRT_BUILD':'0','MUSETALK_TAESD_TRT_STRICT':'1','MUSETALK_TRT_FALLBACK':'0','MUSETALK_RUNTIME_CONFIG_SOURCE':'injected','MUSETALK_RUNTIME_DIR':str(root/'runtime'),'WEBRTC_NATIVE_VP8_DIR':'/workspace/MuseTalk/.runtime/native_vp8','LINGUA_CONTROL_PLANE_BASE_URL':'','LINGUA_WORKER_TOKEN':'','LINGUA_WORKER_REGISTER_URL':'','LINGUA_WORKER_HEARTBEAT_URL':''})
if __ISOLATE_NUMBA__:
 cache=root/'numba-cache';cache.mkdir(mode=0o700);assert not list(cache.iterdir())
 env.update({'NUMBA_CACHE_DIR':str(cache),'NUMBA_DEBUG_CACHE':'1'})
with (root/'api.log').open('xb') as log:
 p=subprocess.Popen(['bash','scripts/run_musetalk_server.sh','--venv-path',__VENV__,'--host','127.0.0.1','--port','8001'],cwd='/workspace/MuseTalk',env=env,stdout=log,stderr=log,start_new_session=True)
data=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text()
v={'pid':p.pid,'start_ticks':data[data.rfind(')')+2:].split()[19],'spawn_epoch':time.time()}
(root/'pid.json').write_text(json.dumps(v));print(json.dumps(v))
'''.replace('__ROOT__',repr(c.WORKER_ROOT+'/application-'+ATTEMPT+'/'+case)).replace('__VENV__',repr(venv)).replace('__ISOLATE_NUMBA__',repr(ISOLATE_NUMBA))
    compile(payload,'<application-launch>','exec')
    return json.loads(c.worker(b,payload))


def generate(b,base,audio,out,index):
    import requests
    started=time.monotonic();name='dependency_'+uuid.uuid4().hex[:16]
    code,body=b.probe(base,f'/generate?avatar_id={b.AVATAR}&batch_size=16&fps=25&output_name={name}',
        method='POST',files={'audio_file':('speech.wav',audio,'audio/wav')},timeout=20)
    b.require(code==200 and body and re.fullmatch('[A-Za-z0-9_-]+',str(body.get('request_id',''))),'Generation rejected')
    request_id=body['request_id'];until=time.monotonic()+90
    while time.monotonic()<until:
        code,body=b.probe(base,'/generate/'+request_id+'/status')
        if code==200 and body and body.get('status')=='completed':
            b.require(body.get('result',{}).get('success') is True,'Generation failed');break
        b.require(not body or body.get('status') not in {'error','failed'},'Generation failed')
        time.sleep(.2)
    else:raise ValueError('Generation deadline')
    complete=time.monotonic()-started;download_started=time.monotonic()
    r=requests.get(base+'/generate/'+request_id+'/download',timeout=30,allow_redirects=False)
    b.require(r.status_code==200 and 0<len(r.content)<=64*1024**2,'Invalid generated video')
    video=out/('sample-'+str(index)+'.mp4');video.write_bytes(r.content)
    downloaded=time.monotonic()-download_started;decode_started=time.monotonic()
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-map','0:v:0','-map','0:a:0','-f','null','-'],capture_output=True,timeout=45,check=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
        'stream=codec_type,codec_name,width,height,nb_read_frames,avg_frame_rate,duration','-of','json',str(video)],timeout=20))
    videos=[s for s in info['streams'] if s.get('codec_type')=='video'];audios=[s for s in info['streams'] if s.get('codec_type')=='audio']
    b.require(len(videos)==1 and audios and int(videos[0].get('nb_read_frames',0))==249,'Unexpected decoded tracks/frames')
    return {'sample':index,'process_first_request':index==0,'video':video.name,
        'request_to_completed_seconds':complete,'download_seconds':downloaded,
        'downloaded_video_ready_seconds':complete+downloaded,'decode_check_seconds':time.monotonic()-decode_started,
        'completed_frames_per_second':249/complete,'throughput_scope':'REST request includes audio processing, rendering, PNG IO and MP4 encoding; not aggregate 400fps inference',
        'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'streams':info['streams']}


def ssim(a,b):
    result=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-i',str(a),'-i',str(b),
        '-filter_complex','[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]ssim',
        '-an','-f','null','-'],capture_output=True,text=True,timeout=45,check=True)
    values=re.findall(r'All:([0-9.]+)',result.stderr)
    if not values:raise ValueError('Missing decoded SSIM result')
    return float(values[-1])


def main():
    b=c.library();b.require(os.getuid()==0,'Root-owned EC2 acceptance only');os.umask(0o077)
    ledger=json.loads((c.ROOT/'ledger.json').read_text());deadline=dt.datetime.fromisoformat(ledger['deadline_utc']).timestamp()
    out=c.ROOT/('application-'+ATTEMPT);b.require(not out.exists(),'Application matrix already claimed; inspect service')
    out.mkdir(mode=0o700);tunnel=None;active=None;cases=[]
    def save(name,value):b.require('/' not in name,'Unsafe receipt name');(out/name).write_text(json.dumps(value,indent=2)+'\n')
    def event(stage,**fields):
        value={'stage':stage,'utc':b.utc(),**fields}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    result={'state':'running','cases':cases,'scope':'same-host internal REST/native acceptance, not WebRTC/TURN or production registration'}
    result['numba_cache_policy']='new empty directory for every case; debug-cache messages enabled equally' if ISOLATE_NUMBA else 'pre-existing package-local JIT caches; history differs'
    try:
        while time.time()<deadline-600:
            state=json.loads(c.worker(b,"import pathlib,json;p=pathlib.Path("+repr(c.WORKER_ROOT+'/followup-a1/receipt.json')+");print(json.dumps({'state':json.loads(p.read_text())['state'] if p.exists() else None}))"))
            if state['state']:
                b.require(state['state']=='passed','Opt-in followup failed; application matrix not started');break
            time.sleep(10)
        else:raise ValueError('Lease reserve reached before application matrix')
        row=b.owned(b.bootstrap(),ledger);b.require(row and row['actual_status']=='running','Owned worker unavailable')
        port=free_port();ssh=c.ssh_arguments(row)
        cmd=ssh[:-2]+['-N','-o','ExitOnForwardFailure=yes','-L',f'127.0.0.1:{port}:127.0.0.1:8001',ssh[-2]]
        with (out/'tunnel.log').open('xb') as log:tunnel=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
        base='http://127.0.0.1:'+str(port);audio=(b.BOOTSTRAP.parent/'startup-speech.wav').read_bytes()
        result['input_audio_sha256']=hashlib.sha256(audio).hexdigest()
        options=[('source-fresh','/workspace/.venvs/musetalk_trt_stagewise'),
                 ('uv-compiled',c.WORKER_ROOT+'/uv-compiled-screen-a3/venv'),
                 ('uv-no-compile',c.WORKER_ROOT+'/uv-nocompile-rep-1/venv'),
                 ('optin',c.WORKER_ROOT+'/optin-venvs/uv')]
        if ISOLATE_NUMBA:options=[v for v in options if v[0]!='uv-no-compile']
        for name,venv in options:
            b.require(time.time()<deadline-420,'Insufficient lease reserve for another API case')
            case=out/name;case.mkdir();active=name;started=time.monotonic()
            spawn=launch(b,name,venv);event('API_SPAWNED',case=name,**spawn)
            until=time.monotonic()+75
            while time.monotonic()<until:
                code,body=b.probe(base,'/health')
                if code==200 and body and body.get('ok') is True:break
                b.require(tunnel.poll() is None,'SSH forwarding exited');time.sleep(.25)
            else:raise ValueError('API health deadline')
            value={'case':name,'dispatch_to_observed_health_seconds':time.monotonic()-started,
                'cache_state':'fresh API process; shared model/avatar files and OS cache warm/unknown',
                'clock_scope':'EC2 monotonic including SSH launch transport and HTTP polling','samples':[]}
            warm=time.monotonic();code,body=b.probe(base,f'/avatars/{b.AVATAR}/cache/warm?wait=false&batch_size=16',method='POST',timeout=10)
            b.require(code in {200,202},'Avatar warm rejected')
            until=time.monotonic()+60
            while time.monotonic()<until:
                code,body=b.probe(base,f'/avatars/{b.AVATAR}/cache/status')
                if code==200 and body and body.get('status')=='ready':break
                time.sleep(.25)
            else:raise ValueError('Avatar warm deadline')
            value['avatar_warm_seconds']=time.monotonic()-warm
            for index in range(4):
                sample=generate(b,base,audio,case,index);value['samples'].append(sample)
                event('GENERATION_DECODED',case=name,sample=index,request_seconds=sample['request_to_completed_seconds'],video_sha256=sample['sha256'])
            payload="import pathlib,json,re;p=pathlib.Path("+repr(c.WORKER_ROOT+'/application-'+ATTEMPT+'/'+name+'/api.log')+");s=p.read_text(errors='replace');print(json.dumps({'render_samples':re.findall(r'Generated (\\d+) frames in ([0-9.]+)s \\(([0-9.]+) FPS\\)',s),'engine_lines':[line for line in s.splitlines() if any(k in line for k in ('built_now=','probe=exact','UNet backend ready'))][-8:],'numba_cache_files':len(list((p.parent/'numba-cache').rglob('*.nb*'))),'numba_cache_saved_events':s.count('[cache] data saved')}))"
            value['render_log_metrics']=json.loads(c.worker(b,payload))
            value['stop']=process_stop(b,name);active=None;cases.append(value);save('interim.json',result)
        pairs=[('source-within','source-fresh',1,'source-fresh',3),('uv-within','uv-compiled',1,'uv-compiled',3),
               ('source-vs-uv','source-fresh',1,'uv-compiled',1),('source-vs-optin','source-fresh',1,'optin',1),
               ('uv-vs-no-compile','uv-compiled',1,'uv-no-compile',1)]
        if ISOLATE_NUMBA:pairs=[v for v in pairs if v[0]!='uv-vs-no-compile']
        result['decoded_quality']=[{'pair':label,'ssim':ssim(out/a/('sample-'+str(ai)+'.mp4'),out/bn/('sample-'+str(bi)+'.mp4'))}
            for label,a,ai,bn,bi in pairs]
        result['state']='passed-output-and-timing-captured'
        result['quality_equivalence']='review within-process versus cross-environment variation before adoption'
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,detail=str(error) if type(error) is ValueError else 'Inspect protected case evidence')
        event('APPLICATION_MATRIX_FAILED',error_type=type(error).__name__,detail=result['detail'])
    finally:
        if active:
            try:result['active_stop']=process_stop(b,active)
            except BaseException:result['active_stop']='unconfirmed; owned expiry unchanged'
        if tunnel:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        save('receipt.json',result)
    print(json.dumps({'state':result['state'],'cases':len(cases)}),flush=True)
    return 0 if result['state']=='passed-output-and-timing-captured' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--isolate-numba',action='store_true')
    options=parser.parse_args()
    if options.isolate_numba:ATTEMPT='a3';ISOLATE_NUMBA=True
    raise SystemExit(main())
