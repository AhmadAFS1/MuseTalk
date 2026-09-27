"""Pinned TAESD/TRT backend and the existing per-frame shared-memory tracker."""
from pathlib import Path
import os,sys,subprocess,gc
from multiprocessing import shared_memory
import cv2,numpy as np,torch

HERE=Path(__file__).resolve().parent

class Tracker:
    def __init__(self,workspace,log_path):
        self.shm=shared_memory.SharedMemory(create=True,size=896*512*3+478*2*4)
        self.frame=np.ndarray((896,512,3),np.uint8,buffer=self.shm.buf)
        self.points=np.ndarray((478,2),np.float32,buffer=self.shm.buf,offset=self.frame.nbytes)
        self.log=Path(log_path).open('a')
        self.proc=subprocess.Popen([str(Path(workspace)/'SoulX-FlashHead/.venv/bin/python'),str(HERE/'tracker_worker.py'),self.shm.name],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
    def command(self,command):
        self.proc.stdin.write(command+'\n');self.proc.stdin.flush();answer=self.proc.stdout.readline().strip()
        if not answer:raise RuntimeError(('tracker failed',self.proc.poll()))
        return answer
    def reset(self):assert self.command('reset')=='ready'
    def track(self,frame,face,box):
        self.frame[:]=frame;x,y,x1,y1=map(int,box)
        self.frame[y:y1,x:x1]=cv2.resize(face,(x1-x,y1-y))
        answer=self.command('frame').split();assert answer[0]=='ok'
        return self.points.copy(),float(answer[1])
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('quit\n');self.proc.stdin.flush();self.proc.wait(timeout=30)
        self.log.close();self.shm.close();self.shm.unlink()

def setup(workspace):
    muse=Path(workspace)/'MuseTalk';os.chdir(muse);sys.path[:0]=[str(muse),str(muse/'scripts')]
    for line in (muse/'.runtime/musetalk_trt_local_sm89.env').read_text().splitlines():
        if line and not line.startswith('#'):
            key,value=line.split('=',1);os.environ[key]=value
    os.environ.update(MUSETALK_TRT_FALLBACK='0',MUSETALK_VAE_BACKEND='taesd',MUSETALK_TAESD_WARMUP_BATCHES='8')
    from scripts.vae_fast_decoder import load_taesd_decoder
    from scripts.trt_runtime import load_unet_trt_backend
    decoder=load_taesd_decoder(device=torch.device('cuda:0'),runtime_dtype=torch.float16,force=True)
    assert decoder and decoder.name=='taesd' and decoder.compile_enabled
    gc.collect();torch.cuda.empty_cache()
    unet=load_unet_trt_backend(device=torch.device('cuda:0'),force=True);assert unet is not None
    return unet,decoder,.18215

def encode(path,frames,audio):
    h,w=frames[0].shape[:2]
    p=subprocess.Popen(['ffmpeg','-v','error','-xerror','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r','24','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264',
        '-threads','2','-crf','18','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-t','10','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    for f in frames:p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close();assert p.wait()==0
