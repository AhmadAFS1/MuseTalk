#!/usr/bin/env python3
"""Resumable portrait -> H3 -> native cache -> TAESD chin refinement."""
from pathlib import Path
import argparse,fcntl,json,os,re,shutil,subprocess,sys
from common import HERE,VERSION,digest,dump,read,sha,stamp

STAGES=('plan','audio','h3','prepare','track','render','validate','report','all')
GPU_STAGES={'h3','prepare','render'}

def prompt(a):
    pronoun='She' if a['gender']=='woman' else 'He';possessive='her' if a['gender']=='woman' else 'his'
    identity=a['label'].split(',')[0]
    beard='' if a['facial_hair'] in ('none','clean-shaven') else f" Maintain {possessive} {a['facial_hair']} consistently, including individual hairs and the visible lip border."
    return (f"Vertical 9:16 photorealistic FaceTime close-up. the same {identity} from the reference image, about {a['age']}, "
        f"{a['hair']}, {a['tone']}, navy crew-neck knit top, softly blurred home office. {pronoun} looks straight into the lens "
        'and speaks this exact line in a calm friendly voice: "Hi, I\'m glad you\'re here. Let me show you how this works." '
        'Natural blinking, a nearly still head held at the reference angle throughout, with only tiny natural micro-movements. '
        'Keep the head centered at a constant size, shoulders level, and gaze into the lens. Keep head pitch, yaw, and roll nearly constant: '
        'no nodding, chin lifts, head tilts, leaning, or swaying. The camera and framing remain fixed. '
        'Maintain natural speaking lip and jaw articulation, coherent teeth and lips, stable identity, unchanged clothes and room, no captions, no logos. '
        f"The {a['gender']} in <Picture 1> is the only person on camera. Keep {possessive} face, hair, and navy top consistent."+beard)

def plan(args):
    config=read(args.config);avatars=config['avatars'];ids=[a['id'] for a in avatars]
    if len(set(ids))!=len(ids) or any(not re.fullmatch(r'[a-z][a-z0-9_]{1,63}',i) for i in ids):raise ValueError('Avatar IDs must be unique lowercase safe identifiers')
    if args.only:
        unknown=set(args.only)-set(ids)
        if unknown:raise ValueError(f'Unknown IDs: {sorted(unknown)}')
        avatars=[a for a in avatars if a['id'] in args.only]
    specs=[]
    for a in avatars:
        portrait=Path(a['portrait']).expanduser()
        if not portrait.is_absolute():portrait=(args.config.parent/portrait).resolve()
        if not portrait.is_file():raise FileNotFoundError(f'Generate and save the portrait first: {portrait}')
        if a.get('portrait_review')!='passed':raise ValueError(f"Inspect and mark portrait_review=passed for {a['id']}")
        target=args.output/a['id'];target.mkdir(parents=True,exist_ok=True)
        spec=dict(a,portrait=str(portrait),portrait_sha256=sha(portrait),workspace=str(args.workspace),
            output=str(target),version=VERSION,motion_prompt=prompt(a),voice=a.get('voice','af_heart' if a['gender']=='woman' else 'am_michael'))
        if a.get('audio'):
            audio=Path(a['audio']).expanduser()
            if not audio.is_absolute():audio=args.config.parent/audio
            audio=audio.resolve();spec['audio']=str(audio);spec['custom_audio_sha256']=sha(audio)
        spec['signature']=digest(spec)
        path=target/'spec.json'
        if path.exists() and read(path)['signature']!=spec['signature']:raise RuntimeError(f'Inputs differ from existing avatar: {target}; use a new output directory')
        dump(path,spec);specs.append(path)
        local_portrait=target/'portrait.png'
        if local_portrait.exists() and sha(local_portrait)!=spec['portrait_sha256']:raise RuntimeError(f'Local portrait changed: {local_portrait}')
        if not local_portrait.exists():shutil.copy2(portrait,local_portrait)
    # A partial rerun must not hide other identities from the batch report.
    all_specs=[args.output/a['id']/'spec.json' for a in config['avatars'] if (args.output/a['id']/'spec.json').is_file()]
    dump(args.output/'batch.json',dict(version=VERSION,config=str(args.config),specs=[str(p) for p in all_specs],updated_utc=stamp()))
    print('PLANNED',len(specs),'avatars',flush=True);return specs

def run_stage(stage,specs,args):
    if stage=='report':
        subprocess.run([str(args.workspace/'SoulX-FlashHead/.venv/bin/python'),str(HERE/'report.py'),'--output',str(args.output)],check=True);return
    environment={'h3':'.venvs/comfy-h3','prepare':'.venvs/musetalk_trt_stagewise','render':'.venvs/musetalk_trt_stagewise',
        'audio':'SoulX-FlashHead/.venv','track':'SoulX-FlashHead/.venv','validate':'SoulX-FlashHead/.venv'}[stage]
    interpreter=args.workspace/environment/'bin/python'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        MUSETALK_BLEND_FIXED_POINT='1',MUSETALK_BLEND_SHRINK_MASK_BBOX='1')
    for spec in specs:
        if shutil.disk_usage(args.output).free<1024**3:raise RuntimeError('Less than 1 GiB free; no files were deleted. Free space before resuming.')
        print(stamp(),stage,spec.parent.name,flush=True)
        with (spec.parent/f'{stage}.log').open('a') as log:
            subprocess.run([str(interpreter),str(HERE/f'{stage}_stage.py'),'--spec',str(spec)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        print(stamp(),'COMPLETE',stage,spec.parent.name,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workspace',type=Path,default=HERE.parents[2]);p.add_argument('--only',nargs='+')
    p.add_argument('--stage',choices=STAGES,default='plan');a=p.parse_args()
    a.config=a.config.resolve();a.output=a.output.resolve();a.workspace=a.workspace.resolve();a.output.mkdir(parents=True,exist_ok=True)
    specs=plan(a)
    stages=STAGES[1:-1] if a.stage=='all' else (() if a.stage=='plan' else (a.stage,))
    for stage in stages:
        if stage in GPU_STAGES:
            with (a.workspace/'.h3_avatar_build_gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);run_stage(stage,specs,a)
        else:run_stage(stage,specs,a)

if __name__=='__main__':main()
