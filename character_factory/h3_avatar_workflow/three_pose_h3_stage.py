"""Generate one first/last-frame H3 pose with the installed low-VRAM graph."""
from pathlib import Path
import argparse,os,shutil,socket,subprocess,sys,time,uuid
from PIL import Image,ImageOps
from common import HERE,check_video,completed,digest,dump,finish,read,sha,stamp
from three_pose_prompts import NATIVE_FRAMES,DELIVERY_FRAMES


def make_anchor(portrait: Path, anchor: Path) -> None:
    tmp=anchor.with_suffix('.new.png')
    ImageOps.fit(Image.open(portrait).convert('RGB'),(512,896),method=Image.Resampling.LANCZOS).save(tmp)
    if anchor.exists():
        if sha(anchor)!=sha(tmp):
            tmp.unlink();raise RuntimeError('Existing shared anchor differs from the portrait; use a new output directory')
        tmp.unlink()
    else:tmp.replace(anchor)


def quarantine_unrecorded(out: Path,record: Path) -> None:
    names=('raw.mp4','source_exact10.mp4','source_with_audio.mp4','source.mp4','submitted_api.json','history.json',record.name)
    present=[out/n for n in names if (out/n).exists()]
    if not present:return
    folder=out/'_incomplete_attempts'/(stamp().replace(':','').replace('-','')+'_'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True)
    for p in present:shutil.move(str(p),str(folder/p.name))


def remove_owned_comfy_copy(workspace: Path, item: dict, expected_prefix: str) -> None:
    """Keep the verified delivery, not a second MP4 in ComfyUI/output."""
    if (item.get('type') != 'output' or item.get('subfolder') != 'avatar_batch'
            or not str(item.get('filename', '')).startswith(expected_prefix)
            or Path(str(item.get('filename', ''))).name != item.get('filename')):
        return
    root=(workspace/'ComfyUI/output').resolve()
    candidate=(root/item['subfolder']/item['filename']).resolve()
    try:candidate.relative_to(root)
    except ValueError:return
    try:candidate.unlink(missing_ok=True)
    except OSError as exc:print('WARNING duplicate ComfyUI MP4 cleanup failed:',exc,flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);args=p.parse_args()
    s=read(args.spec);out=Path(s['output']);w=Path(s['workspace']);h3=w/'minimax-h3';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(h3))
    from run_h3_lowvram import completed_video,preflight,request_json,stage_image,wait_prompt
    from make_exact_10s import make
    from package_spoken_transition_bank import package
    from certify_endpoints import decoded_endpoints
    pose=s['pose'];anchor=out.parent/'anchor.png';portrait=Path(s['portrait']);make_anchor(portrait,anchor)
    graph=read(HERE/'h3_template.json')
    signature=digest(dict(spec=s['signature'],code=sha(__file__),graph=sha(HERE/'h3_template.json'),
        exact10=sha(h3/'make_exact_10s.py'),packager=sha(h3/'package_spoken_transition_bank.py')))
    record=out/'h3.json'
    if completed(record,signature):print('REUSE',s['id'],pose,flush=True);return
    quarantine_unrecorded(out,record)
    stage_name='h3_'+sha(anchor)[:16]+anchor.suffix.lower();stage_path=w/'ComfyUI/input'/stage_name
    stage_was_present=stage_path.exists();image=stage_image(anchor,w/'ComfyUI/input')
    assert image==stage_name
    graph['6']['inputs']['image']=image;graph['15']['inputs']['image']=image
    graph['7']['inputs'].update(prompt=s['prompt'],length=NATIVE_FRAMES[pose])
    graph['9']['inputs']['seed']=s['seed']
    graph['14']['inputs']['filename_prefix']=f"avatar_batch/{s['id']}_{pose}_{s['signature'][:12]}"
    dump(out/'submitted_api.json',graph)
    port=8197;base=f'http://127.0.0.1:{port}'
    try:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('127.0.0.1',port))
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if any(float(x)>1000 for x in used.splitlines()):raise RuntimeError('GPU is occupied; stop a local MuseTalk API before H3 generation')
        command=[str(w/'.venvs/comfy-h3/bin/python'),'main.py','--listen','127.0.0.1','--port',str(port),
                 '--reserve-vram','0.8','--extra-model-paths-config',str(h3/'extra_model_paths_h3.yaml'),
                 '--fast-disk','--disable-pinned-memory','--cache-none']
        start=time.monotonic();dump(record,dict(status='starting',signature=signature,started_utc=stamp(),command=command))
        with (out/'h3_server.log').open('a') as log:
            server=subprocess.Popen(command,cwd=w/'ComfyUI',stdout=log,stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    if server.poll() is not None:raise RuntimeError('H3 server exited; inspect h3_server.log')
                    try:request_json(base,'/system_stats');break
                    except OSError:time.sleep(1)
                else:raise RuntimeError('H3 startup timed out')
                preflight(base,graph)
                q=request_json(base,'/prompt',dict(prompt=graph,client_id=str(uuid.uuid4())))
                if not q.get('prompt_id') or q.get('node_errors'):raise RuntimeError(q)
                dump(record,dict(status='generating',signature=signature,prompt_id=q['prompt_id']))
                history=wait_prompt(base,q['prompt_id'],3600);dump(out/'history.json',history)
                if history.get('status',{}).get('status_str')!='success':raise RuntimeError(history.get('status'))
                raw=out/'raw.mp4';comfy_output=completed_video(base,history,raw)
                check_video(raw,frames=NATIVE_FRAMES[pose])
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:server.wait(timeout=30)
                    except subprocess.TimeoutExpired:server.kill();server.wait()
        generation_seconds=time.monotonic()-start
        artifacts=[out/'raw.mp4',out/'submitted_api.json',out/'history.json',anchor]
        if NATIVE_FRAMES[pose]==243:
            exact=out/'source_exact10.mp4';exact_details=make(raw,exact,anchor);input_video=exact;artifacts.append(exact)
        else:exact_details=None;input_video=raw
        packaged=out/'source_with_audio.mp4' if pose!='talking' else out/'source.mp4'
        packaging=package(input_video,packaged,anchor)
        if pose!='talking':
            # Idle and smiling sources must not play H3's incidental audio.
            subprocess.run(['ffmpeg','-v','error','-xerror','-y','-i',str(packaged),'-map','0:v:0','-c:v','copy','-an',
                            '-movflags','+faststart',str(out/'source.mp4')],check=True)
            artifacts.append(packaged)
        source=out/'source.mp4';proof=decoded_endpoints(source)
        media=check_video(source,frames=DELIVERY_FRAMES[pose],audio=pose=='talking')
        if pose!='talking' and any(x['codec_type']=='audio' for x in media['streams']):raise RuntimeError('Silent pose contains audio')
        if not proof['first_last_exact']:raise RuntimeError('Pose endpoints differ')
        artifacts.append(source)
        remove_owned_comfy_copy(w,comfy_output,f"{s['id']}_{pose}_{s['signature'][:12]}")
        finish(record,signature,artifacts,pose=pose,prompt=s['prompt'],seed=s['seed'],generation_seconds=generation_seconds,
               exact10=exact_details,packaging=packaging,media=media,endpoint_proof=proof,
               comfy_output=comfy_output,model_graph_sha256=sha(HERE/'h3_template.json'))
        print('READY',s['id'],pose,source,flush=True)
    finally:
        if not stage_was_present:stage_path.unlink(missing_ok=True)


if __name__=='__main__':main()
