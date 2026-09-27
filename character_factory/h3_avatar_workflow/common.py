"""Small, dependency-free workflow records and integrity helpers."""
from pathlib import Path
import hashlib,json,subprocess,time

VERSION='h3-taesd-chin100-seam-v1'
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def read(path):return json.loads(Path(path).read_text())

def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def completed(path,signature):
    path=Path(path)
    if not path.exists():return False
    r=read(path)
    if r.get('signature')!=signature:raise RuntimeError(f'Inputs changed; use a new output directory: {path}')
    if r.get('status')!='complete':return False
    for name,expected in r['artifacts'].items():
        if not Path(name).is_file() or sha(name)!=expected:raise RuntimeError(f'Completed artifact changed or missing: {name}')
    return True

def finish(path,signature,artifacts,**details):
    dump(path,dict(status='complete',signature=signature,finished_utc=stamp(),
        artifacts={str(Path(p).resolve()):sha(p) for p in artifacts},**details))

def check_video(path,frames=240,width=512,height=896,audio=True):
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
        'stream=codec_type,nb_read_frames,avg_frame_rate,width,height,duration','-of','json',str(path)],text=True))['streams']
    v=next(s for s in streams if s['codec_type']=='video')
    if (int(v['nb_read_frames']),v['avg_frame_rate'],v['width'],v['height'])!=(frames,'24/1',width,height):raise ValueError((path,v))
    if abs(float(v['duration'])-frames/24)>.002:raise ValueError((path,'duration'))
    if audio and not any(s['codec_type']=='audio' for s in streams):raise ValueError((path,'audio missing'))
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,capture_output=True)
    return dict(path=str(path),sha256=sha(path),streams=streams,full_decode=True)

def spec_arg():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();s=read(a.spec);s['spec_path']=str(a.spec.resolve());return s
