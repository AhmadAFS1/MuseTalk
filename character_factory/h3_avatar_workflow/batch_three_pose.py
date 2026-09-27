#!/usr/bin/env python3
"""Batch portraits -> three H3 pose loops -> three MuseTalk S3 caches each."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import argparse,fcntl,json,os,re,shutil,socket,subprocess,sys,time,urllib.error
from common import HERE,digest,dump,read,sha,stamp
from three_pose_prompts import VERSION,POSES,pose_prompt
from three_pose_s3_stage import stats

IMAGE_SUFFIXES={'.png','.jpg','.jpeg','.webp'}
MEDIA_NAMES=('raw.mp4','source_exact10.mp4','source_with_audio.mp4','source.mp4')


def safe_id(name: str) -> str:
    cleaned=re.sub(r'[^a-z0-9]+','_',name.lower()).strip('_')[:27].strip('_')
    return cleaned or 'avatar'


def load_metadata(path: Path|None) -> dict:
    if path is None:return {}
    value=read(path)
    if isinstance(value,dict) and isinstance(value.get('avatars'),list):
        result={}
        for row in value['avatars']:
            for key in (row.get('portrait'),row.get('generated_path')):
                if key:result[Path(key).name]=row
        return result
    if isinstance(value,dict) and isinstance(value.get('images'),dict):return value['images']
    if isinstance(value,dict):return value
    raise ValueError('Metadata must be a JSON mapping or an avatars list')


def discover(image_dir: Path, output: Path, limit: int|None) -> list[Path]:
    if not image_dir.is_dir():raise FileNotFoundError(image_dir)
    if image_dir==output or image_dir in output.parents:
        raise ValueError('Output must be outside the image folder so generated images are not ingested')
    paths=sorted(p for p in image_dir.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:paths=paths[:limit]
    if not paths:raise ValueError(f'No PNG/JPEG/WebP photos in {image_dir}')
    return paths


def plan(args) -> list[dict]:
    metadata=load_metadata(args.metadata)
    image_paths=discover(args.images,args.output,args.limit)
    jobs=[];ids=set()
    for photo in image_paths:
        image_hash=sha(photo);relative=photo.relative_to(args.images).as_posix()
        profile=metadata.get(relative,metadata.get(photo.name,metadata.get(photo.stem,{})))
        if not isinstance(profile,dict):raise ValueError(f'Invalid metadata for {relative}')
        ident=f"{safe_id(str(profile.get('id') or photo.stem))}_{image_hash[:8]}"
        # Large archives sometimes repeat a file name and its bytes in separate
        # subfolders. Give those entries distinct, deterministic cache IDs.
        if ident in ids:ident=f"{safe_id(str(profile.get('id') or photo.stem))}_{image_hash[:8]}_{digest(relative)[:8]}"
        if ident in ids:raise ValueError(f'Duplicate ID for photo: {photo}')
        ids.add(ident)
        if args.only and ident not in args.only and photo.stem not in args.only:continue
        target=args.output/ident
        prompts={pose:pose_prompt(profile,pose) for pose in POSES}
        signature=digest(dict(version=VERSION,photo=relative,image_sha256=image_hash,profile=profile,prompts=prompts,
                              graph_sha256=sha(HERE/'h3_template.json'),prompt_code_sha256=sha(HERE/'three_pose_prompts.py')))
        entry=dict(id=ident,photo=relative,portrait=str(photo),portrait_sha256=image_hash,profile=profile,
                   prompts=prompts,signature=signature,workspace=str(args.workspace),output=str(target),version=VERSION)
        old=target/'plan.json'
        if old.exists() and read(old)['signature']!=signature:
            raise RuntimeError(f'Photo, prompt or graph changed for {ident}; use a new output directory')
        target.mkdir(parents=True,exist_ok=True);dump(old,entry)
        for pose in POSES:
            pose_dir=target/pose;pose_dir.mkdir(exist_ok=True)
            spec=dict(id=ident,pose=pose,prompt=prompts[pose],seed=42,portrait=str(photo),
                      workspace=str(args.workspace),output=str(pose_dir),parent_signature=signature)
            spec['signature']=digest(spec)
            pose_path=pose_dir/'spec.json'
            if pose_path.exists() and read(pose_path)['signature']!=spec['signature']:
                raise RuntimeError(f'Pose inputs changed for {ident}/{pose}; use a new output directory')
            dump(pose_path,spec)
        jobs.append(entry)
    if args.only and not jobs:raise ValueError(f'No photos matched --only {args.only}')
    dump(args.output/'batch_plan.json',dict(version=VERSION,image_dir=str(args.images),output=str(args.output),
                                              jobs=[x['id'] for x in jobs],updated_utc=stamp()))
    print('PLANNED',len(jobs),'photos;',len(jobs)*3,'H3 poses and S3 caches',flush=True)
    return jobs


def published(entry: dict) -> dict|None:
    path=Path(entry['output'])/'published.json'
    if not path.exists():return None
    data=read(path)
    if data.get('status')!='complete' or data.get('signature')!=entry['signature']:
        raise RuntimeError(f'Published receipt differs from current photo: {path}')
    if set(data.get('poses',{}))!=set(POSES):raise RuntimeError(f'Incomplete published receipt: {path}')
    return data


def generate(entry: dict,args) -> None:
    if published(entry):print('REUSE PUBLISHED',entry['id'],flush=True);return
    for pose in args.poses:
        if shutil.disk_usage(args.output).free<1024**3:
            raise RuntimeError('Less than 1 GiB free. Use --release-local after S3 upload or a larger output volume.')
        spec=Path(entry['output'])/pose/'spec.json';log=Path(entry['output'])/pose/'h3_stage.log'
        print(stamp(),'H3',entry['id'],pose,flush=True)
        with log.open('a') as out:
            try:
                subprocess.run([str(args.workspace/'.venvs/comfy-h3/bin/python'),str(HERE/'three_pose_h3_stage.py'),
                                '--spec',str(spec)],stdout=out,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'H3 {pose} failed; inspect {log}') from exc


def checked_poses(entry: dict) -> dict:
    target=Path(entry['output']);records={}
    for pose in POSES:
        rec=target/pose/'h3.json'
        if not rec.exists():raise RuntimeError(f'Missing H3 pose: {rec}')
        data=read(rec)
        if data.get('status')!='complete':raise RuntimeError(f'Unfinished H3 pose: {rec}')
        source=target/pose/'source.mp4'
        if not source.is_file() or data['artifacts'].get(str(source.resolve()))!=sha(source):
            raise RuntimeError(f'Source differs from completed H3 record: {source}')
        records[pose]=data
    ends={x['endpoint_proof']['first_rgb_sha256'] for x in records.values()}
    if len(ends)!=1 or not all(x['endpoint_proof']['first_last_exact'] for x in records.values()):
        raise RuntimeError(f'Three poses do not share exact decoded endpoints: {entry["id"]}')
    return records


def prepare(entry: dict,args,url: str) -> None:
    if published(entry):print('REUSE PUBLISHED',entry['id'],flush=True);return
    checked_poses(entry)
    server=stats(url)
    if server['version']!=args.expected_version:raise RuntimeError(f'API cache version {server["version"]} differs from {args.expected_version}')
    if args.expected_bucket and server['bucket']!=args.expected_bucket:
        raise RuntimeError(f'API S3 bucket {server["bucket"]} differs from {args.expected_bucket}')
    for pose in POSES:
        spec=Path(entry['output'])/pose/'spec.json';log=Path(entry['output'])/pose/'prepare_s3.log'
        print(stamp(),'S3',entry['id'],pose,flush=True)
        with log.open('a') as out:
            try:
                subprocess.run([sys.executable,str(HERE/'three_pose_s3_stage.py'),'--spec',str(spec),
                                '--base-url',url,'--batch-size',str(args.batch_size),'--timeout',str(args.timeout)],
                               stdout=out,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'S3 {pose} failed; inspect {log}') from exc
    receipts={pose:read(Path(entry['output'])/pose/'s3.json') for pose in POSES}
    if any(x.get('status')!='complete' for x in receipts.values()):raise RuntimeError('At least one S3 upload lacks a completion receipt')
    source_hashes={pose:sha(Path(entry['output'])/pose/'source.mp4') for pose in POSES}
    if any(receipts[pose]['source_sha256']!=source_hashes[pose] for pose in POSES):
        raise RuntimeError('S3 receipt source hash differs from local source')
    final=dict(status='complete',signature=entry['signature'],portrait_sha256=entry['portrait_sha256'],
               finished_utc=stamp(),endpoint_rgb_sha256=checked_poses(entry)['idle']['endpoint_proof']['first_rgb_sha256'],
               poses={pose:dict(avatar_id=receipts[pose]['avatar_id'],source_sha256=source_hashes[pose],
                                s3_uri=receipts[pose]['s3_uri'],receipt_sha256=sha(Path(entry['output'])/pose/'s3.json')) for pose in POSES},
               visual_review='pending',server_s3_version=server['version'])
    dump(Path(entry['output'])/'published.json',final)
    if args.release_local:
        for pose in POSES:
            for name in MEDIA_NAMES:(Path(entry['output'])/pose/name).unlink(missing_ok=True)
        print('RELEASED LOCAL VIDEO',entry['id'],flush=True)
    print('PUBLISHED',entry['id'],flush=True)


def release_owned_api_files(entry: dict,args) -> None:
    """After the owned API stops, remove only this published identity's caches."""
    if not args.release_local:return
    receipt=published(entry)
    if not receipt:return
    base=args.workspace/'MuseTalk'
    for pose in POSES:
        avatar_id=receipt['poses'][pose]['avatar_id']
        cache=base/'results'/args.expected_version/'avatars'/avatar_id
        if cache.is_dir():shutil.rmtree(cache)
        for name in ('talking_source.mp4','idle_source.mp4'):
            (base/'uploads/videos'/f'{avatar_id}_{name}').unlink(missing_ok=True)
    print('RELEASED OWNED API CACHE',entry['id'],flush=True)


@contextmanager
def local_api(args,log_name: str):
    if os.getenv('AVATAR_S3_ENABLED','').lower() not in ('1','true','yes','on') or not os.getenv('AVATAR_S3_BUCKET'):
        raise RuntimeError('--local-api requires AVATAR_S3_ENABLED=1 and AVATAR_S3_BUCKET in its environment')
    port=args.local_api_port;base=f'http://127.0.0.1:{port}'
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port))
    env=os.environ.copy();profile=args.workspace/'MuseTalk/.runtime/musetalk_trt_local_sm89.env'
    for line in profile.read_text().splitlines():
        if line and not line.startswith('#'):
            key,value=line.split('=',1);env[key]=value
    env.update(MUSETALK_VAE_BACKEND='taesd',MUSETALK_UNET_BACKEND='trt',MUSETALK_TRT_FALLBACK='0',
               MUSETALK_BLEND_FIXED_POINT='1',MUSETALK_BLEND_SHRINK_MASK_BBOX='1',
               MUSETALK_TAESD_WARMUP_BATCHES='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=[str(args.workspace/'.venvs/musetalk_trt_stagewise/bin/python'),'api_server.py',
             '--host','127.0.0.1','--port',str(port)]
    log_path=args.output/log_name;log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a') as log:
        process=subprocess.Popen(command,cwd=args.workspace/'MuseTalk',env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(360):
                if process.poll() is not None:raise RuntimeError(f'Local MuseTalk API exited; inspect {log_path}')
                try:stats(base);break
                except (OSError,urllib.error.URLError,RuntimeError):time.sleep(1)
            else:raise RuntimeError(f'Local MuseTalk API timed out; inspect {log_path}')
            yield base
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=40)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def process_jobs(jobs,args) -> list[dict]:
    failures=[];lock_path=args.workspace/'.h3_avatar_build_gpu.lock'
    if args.stage=='prepare' and args.local_api:
        pending=[entry for entry in jobs if not published(entry)]
        if not pending:return failures
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with local_api(args,'local_api_prepare.log') as url:
                for entry in pending:
                    try:prepare(entry,args,url)
                    except Exception as exc:
                        failures.append(dict(id=entry['id'],stage='prepare',error=str(exc)))
                        if not args.continue_on_error:break
            for entry in pending:release_owned_api_files(entry,args)
        return failures
    for entry in jobs:
        try:
            if published(entry):
                print('REUSE PUBLISHED',entry['id'],flush=True)
                continue
            if args.stage in ('generate','all'):
                with lock_path.open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);generate(entry,args)
            if args.stage in ('prepare','all'):
                if args.local_api:
                    with lock_path.open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        with local_api(args,f'{entry["id"]}/local_api.log') as url:prepare(entry,args,url)
                    release_owned_api_files(entry,args)
                else:prepare(entry,args,args.prepare_url)
        except Exception as exc:
            failures.append(dict(id=entry['id'],stage=args.stage,error=str(exc)))
            print('FAILED',entry['id'],str(exc),flush=True)
            if not args.continue_on_error:break
    return failures


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata',type=Path);p.add_argument('--workspace',type=Path,default=HERE.parents[2])
    p.add_argument('--stage',choices=('plan','generate','prepare','all'),default='plan')
    p.add_argument('--poses',nargs='+',choices=POSES,default=list(POSES))
    p.add_argument('--limit',type=int);p.add_argument('--only',nargs='+')
    api=p.add_mutually_exclusive_group();api.add_argument('--prepare-url');api.add_argument('--local-api',action='store_true')
    p.add_argument('--local-api-port',type=int,default=8198);p.add_argument('--expected-version',default='v15')
    p.add_argument('--expected-bucket');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--release-local',action='store_true',help='Remove local MP4s only after all three verified S3 uploads')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args();args.images=args.images.resolve();args.output=args.output.resolve();args.workspace=args.workspace.resolve()
    if args.metadata:args.metadata=args.metadata.resolve()
    if args.stage in ('prepare','all') and not(args.local_api or args.prepare_url):p.error('prepare/all needs --local-api or --prepare-url')
    if args.release_local and args.stage not in ('prepare','all'):p.error('--release-local needs prepare/all')
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    if args.batch_size<1:p.error('--batch-size must be positive')
    if args.local_api and (os.getenv('AVATAR_S3_ENABLED','').lower() not in ('1','true','yes','on') or not os.getenv('AVATAR_S3_BUCKET')):
        p.error('--local-api needs AVATAR_S3_ENABLED=1 and AVATAR_S3_BUCKET')
    jobs=plan(args)
    failures=[] if args.stage=='plan' else process_jobs(jobs,args)
    dump(args.output/'batch_status.json',dict(version=VERSION,stage=args.stage,updated_utc=stamp(),
         total=len(jobs),published=sum(bool(published(x)) for x in jobs),failures=failures))
    print('BATCH',len(jobs),'photos;',len(failures),'failures',flush=True)
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
