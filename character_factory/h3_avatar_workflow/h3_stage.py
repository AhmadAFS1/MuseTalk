"""Run the fixed H3 graph in an owned, sequential low-VRAM server."""
from pathlib import Path
import os,signal,socket,subprocess,sys,time,uuid
from PIL import Image,ImageOps
from common import HERE,check_video,completed,digest,dump,finish,read,sha,spec_arg,stamp

def main():
    spec=spec_arg();w=Path(spec['workspace']);out=Path(spec['output']);h3=w/'minimax-h3'
    sys.path.insert(0,str(h3))
    from run_h3_lowvram import completed_video,preflight,request_json,stage_image,wait_prompt
    from make_exact_10s import make
    from package_spoken_transition_bank import package
    signature=digest(dict(spec=spec['signature'],code=sha(__file__),graph=sha(HERE/'h3_template.json')))
    record=out/'h3.json'
    if completed(record,signature):print('REUSE H3',spec['id']);return
    anchor=out/'anchor.png';ImageOps.fit(Image.open(spec['portrait']).convert('RGB'),(512,896),method=Image.Resampling.LANCZOS).save(anchor)
    graph=read(HERE/'h3_template.json');image=stage_image(anchor,w/'ComfyUI/input')
    graph['6']['inputs']['image']=image;graph['15']['inputs']['image']=image
    graph['7']['inputs']['prompt']=spec['motion_prompt']
    graph['14']['inputs']['filename_prefix']=f"avatar_workflow/{spec['id']}_{spec['signature'][:12]}"
    dump(out/'submitted_api.json',graph)
    # A distinct port avoids taking over an existing user's ComfyUI service.
    port=8197;base=f'http://127.0.0.1:{port}'
    with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if any(float(x)>1000 for x in used.splitlines()):raise RuntimeError('GPU is already occupied; resume once the other job finishes')
    command=[str(w/'.venvs/comfy-h3/bin/python'),'main.py','--listen','127.0.0.1','--port',str(port),
        '--reserve-vram','0.8','--extra-model-paths-config',str(h3/'extra_model_paths_h3.yaml'),
        '--fast-disk','--disable-pinned-memory','--cache-none']
    start=time.monotonic();details=dict(status='starting',signature=signature,started_utc=stamp(),command=command)
    with (out/'h3_server.log').open('a') as log:
        server=subprocess.Popen(command,cwd=w/'ComfyUI',stdout=log,stderr=subprocess.STDOUT)
        details['server_pid']=server.pid;dump(record,details)
        try:
            for _ in range(120):
                if server.poll() is not None:raise RuntimeError('H3 server exited; inspect h3_server.log')
                try:request_json(base,'/system_stats');break
                except OSError:time.sleep(1)
            else:raise RuntimeError('H3 startup timed out')
            preflight(base,graph);q=request_json(base,'/prompt',dict(prompt=graph,client_id=str(uuid.uuid4())))
            if not q.get('prompt_id') or q.get('node_errors'):raise RuntimeError(q)
            details.update(status='generating',prompt_id=q['prompt_id']);dump(record,details)
            history=wait_prompt(base,q['prompt_id'],3600);dump(out/'history.json',history)
            if history.get('status',{}).get('status_str')!='success':raise RuntimeError(history.get('status'))
            raw=out/'source_raw.mp4';details['comfy_output']=completed_video(base,history,raw)
            details['raw_validation']=check_video(raw,243)
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()
    details['generation_seconds']=time.monotonic()-start
    exact=out/'source_exact10.mp4';source=out/'source.mp4'
    details['exact10']=make(raw,exact,anchor);details['packaging']=package(exact,source,anchor)
    details['delivery_validation']=check_video(source)
    details.pop('status');details.pop('signature')
    finish(record,signature,[raw,exact,source,anchor,out/'submitted_api.json'],**details)
    print('SOURCE READY',source,flush=True)

if __name__=='__main__':main()
