"""Prepare a production MuseTalk avatar and require an S3 upload receipt."""
from pathlib import Path
import argparse,json,sys,urllib.request
from common import HERE,completed,digest,finish,read,sha


def stats(base_url: str,timeout: int=30) -> dict:
    with urllib.request.urlopen(base_url.rstrip('/')+'/stats',timeout=timeout) as response:
        data=json.load(response)
    s3=data.get('avatar_s3')
    if not isinstance(s3,dict) or not s3.get('enabled') or not s3.get('bucket'):
        raise RuntimeError('MuseTalk API has no enabled avatar S3 store; enable AVATAR_S3_ENABLED and configure its bucket')
    if not s3.get('version'):
        raise RuntimeError('MuseTalk API did not report its avatar S3 version')
    return s3


def publish_one(spec: dict,base_url: str,batch_size: int=8,timeout: int=1800) -> dict:
    out=Path(spec['output']);pose=spec['pose'];video=out/'source.mp4'
    if not video.is_file():raise FileNotFoundError(video)
    # The WebRTC pose router resolves each physical pose from its cache's
    # idle_video_path. Each cache must therefore expose its own source there.
    idle=None
    video_hash=sha(video);idle_hash=None
    avatar_id=f"{spec['id'][:39]}_{pose}_{video_hash[:10]}"
    before=stats(base_url)
    prefix=before.get('prefix','').strip('/')
    key='/'.join(x for x in (prefix,before['version'],avatar_id+'.tar.gz') if x)
    signature=digest(dict(spec=spec['signature'],video=video_hash,idle=idle_hash,avatar_id=avatar_id,
                          bucket=before['bucket'],key=key,code=sha(__file__)))
    record=out/'s3.json'
    if completed(record,signature):return read(record)
    sys.path.insert(0,str(Path(spec['workspace'])/'MuseTalk/character_factory/scripts'))
    from prepare_musetalk_avatars import prepare_one
    # An incomplete prior attempt can leave a local-only cache. Force creation
    # makes the server upload the freshly verified source again.
    result=prepare_one(base_url=base_url,avatar_id=avatar_id,video=video,idle_video=idle,
                       batch_size=batch_size,bbox_shift=0,force_recreate=True,timeout=timeout)
    after=stats(base_url)
    if (before['bucket'],before.get('prefix'),before['version'])!=(after['bucket'],after.get('prefix'),after['version']):
        raise RuntimeError('MuseTalk S3 destination changed during avatar preparation')
    success=(result.get('status')=='success' and result.get('avatar_id')==avatar_id and
             result.get('already_prepared') is not True and
             after.get('upload_successes',0)>before.get('upload_successes',0) and
             after.get('last_upload_key')==key)
    if 's3_uploaded' in result:
        success=success and result['s3_uploaded'] is True and result.get('s3_key')==key
    if not success:
        raise RuntimeError(f'Avatar preparation returned, but S3 upload was not confirmed for {avatar_id}; inspect API /stats and server logs')
    uri=f"s3://{before['bucket']}/{key}"
    finish(record,signature,[video],avatar_id=avatar_id,pose=pose,
           source_sha256=video_hash,idle_sha256=idle_hash,s3_uri=uri,server_response=result,
           upload_successes_before=before['upload_successes'],upload_successes_after=after['upload_successes'],
           object_key=key,version=before['version'])
    print('S3 READY',spec['id'],pose,uri,flush=True)
    return read(record)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--base-url',required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--timeout',type=int,default=1800);args=p.parse_args()
    publish_one(read(args.spec),args.base_url,args.batch_size,args.timeout)


if __name__=='__main__':main()
