"""Prototype: exact fixed-point blend replacements (numpy uint16, numba nogil). Scratch only."""
import sys, os, time, threading
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=os.path.join(os.path.dirname(os.path.abspath(__file__)),'numba_cache')
sys.path[:0]=['/workspace/MuseTalk','/workspace/MuseTalk/character_factory/h3_avatar_workflow']
import numpy as np, cv2, torch, numba
import chin
from musetalk.utils.blending import get_image_blending_with_plan
cv2.setNumThreads(1)
root='/workspace/experiments/avatar_diversity_20260927/black_woman'; N=48
cap=cv2.VideoCapture(f'{root}/source.mp4'); frames=[cap.read()[1] for _ in range(N)]; cap.release()
d=dict(cache=torch.load(f'{root}/cache.pt',map_location='cpu',weights_only=False),masks=np.load(f'{root}/masks.npz'),p=np.load(f'{root}/source_landmarks.npy'))
faces=np.load(f'{root}/faces.npz')['faces'][:N]
chin.prepare_source(d,frames)
def box(i): return tuple(map(int,d['cache']['boxes'][i]))
R=[cv2.resize(faces[i],(box(i)[2]-box(i)[0],box(i)[3]-box(i)[1])) for i in range(N)]

@numba.njit(nogil=True,cache=True)
def blend_nb(img, y0, x0, face, fy0, fx0, oy0, ox0, oh, ow, alpha):
    H,W=alpha.shape
    for y in range(H):
        for x in range(W):
            a=np.uint32(alpha[y,x])
            if a==0: continue
            iy=y0+y; ix=x0+x
            ry=y-oy0; rx=x-ox0
            inside = ry>=0 and ry<oh and rx>=0 and rx<ow
            for c in range(3):
                b=np.uint32(img[iy,ix,c])
                o=np.uint32(face[fy0+ry,fx0+rx,c]) if inside else b
                img[iy,ix,c]=np.uint8((o*a+b*(np.uint32(255)-a))//np.uint32(255))
    return img

def blend_fast(image, face, plan):
    ys,xs=plan['clip_slice']; dst=plan['overlay_dst_slice']; src=plan['face_src_slice']
    a=plan['alpha_u8'][:,:,0]
    blend_nb(image, ys.start, xs.start, face, src[0].start, src[1].start, dst[0].start, dst[1].start,
             dst[0].stop-dst[0].start, dst[1].stop-dst[1].start, a)
    return image

def blend_u16(image, face, plan):
    ys,xs=plan['clip_slice']; base=image[ys,xs]; ov=base.copy()
    dy,dx=plan['overlay_dst_slice']; sy,sx=plan['face_src_slice']; ov[dy,dx]=face[sy,sx]
    a=plan['alpha_u8'].astype(np.uint16); v=ov.astype(np.uint16)*a; v+=base.astype(np.uint16)*(255-a)
    # exact floor(v/255) for v in [0,65025]
    q=(v+1+(v>>8))>>8
    image[ys,xs]=q.astype(np.uint8); return image

# exactness of the division identity over the full domain
v=np.arange(0,65026,dtype=np.uint32); assert np.array_equal((v+1+(v>>8))>>8, v//255)
maxd={'nb':0,'u16':0}
for i in range(N):
    ref=get_image_blending_with_plan(frames[i].copy(),R[i],d['plans'][i])
    a=blend_fast(frames[i].copy(),R[i],d['plans'][i]); b=blend_u16(frames[i].copy(),R[i],d['plans'][i])
    maxd['nb']=max(maxd['nb'],int(np.abs(ref.astype(int)-a).max())); maxd['u16']=max(maxd['u16'],int(np.abs(ref.astype(int)-b).max()))
print('max abs diff vs production fixed-point blend over',N,'frames:',maxd)
def t(fn,reps=5):
    bufs=[f.copy() for f in frames]; s=time.perf_counter()
    for _ in range(reps):
        for i in range(N): fn(bufs[i],R[i],d['plans'][i])
    return round((time.perf_counter()-s)/(reps*N)*1000,4)
print('blend-only ms/frame: numpy uint32 (prod)',t(get_image_blending_with_plan),' numpy uint16',t(blend_u16),' numba nogil',t(blend_fast))
def scale(fn,w,sec=1.5):
    stop=time.perf_counter()+sec; c=[0]*w; bufs=[[f.copy() for f in frames[:8]] for _ in range(w)]
    def run(k):
        j=0
        while time.perf_counter()<stop: fn(bufs[k][j%8],R[j%8],d['plans'][j%8]); j+=1; c[k]+=1
    th=[threading.Thread(target=run,args=(k,)) for k in range(w)]; s=time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]
    return round(sum(c)/(time.perf_counter()-s))
print('thread scaling blend fps prod',{w:scale(get_image_blending_with_plan,w) for w in (1,4,8)},' numba',{w:scale(blend_fast,w) for w in (1,4,8)})
