"""Prototype: bit-exact numba (nogil) port of chin.warp_roi's per-pixel map math. Scratch only."""
import sys, os, time, threading
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=os.path.join(os.path.dirname(os.path.abspath(__file__)),'numba_cache')
sys.path[:0]=['/workspace/MuseTalk','/workspace/MuseTalk/character_factory/h3_avatar_workflow']
import numpy as np, cv2, torch, numba, chin
from chin import axes, curves, smooth, GRID, LIPS
cv2.setNumThreads(1)
WHO=sys.argv[1] if len(sys.argv)>1 else 'black_woman'
root=f'/workspace/experiments/avatar_diversity_20260927/{WHO}'; N=int(os.environ.get('NFRAMES','240'))
cap=cv2.VideoCapture(f'{root}/source.mp4'); frames=[cap.read()[1] for _ in range(N)]; cap.release()
d=dict(cache=torch.load(f'{root}/cache.pt',map_location='cpu',weights_only=False),masks=np.load(f'{root}/masks.npz'),p=np.load(f'{root}/source_landmarks.npy'))
faces=np.load(f'{root}/faces.npz')['faces'][:N]
d['g']=np.load(f'{root}/generated_landmarks.npy'); d['chin_delta']=np.load(f'{root}/chin_delta.npy')
chin.prepare_refined(d,frames)

@numba.njit(cache=True,inline='always')
def interp1(x,xp,fp,slopes):
    n=xp.shape[0]
    if x<xp[0]: return fp[0]
    if x>xp[n-1]: return fp[n-1]
    if x==xp[n-1]: return fp[n-1]
    lo=0; hi=n-1
    while hi-lo>1:
        mid=(lo+hi)//2
        if xp[mid]<=x: lo=mid
        else: hi=mid
    if xp[lo]==x: return fp[lo]
    return slopes[lo]*(x-xp[lo])+fp[lo]

@numba.njit(nogil=True,cache=True)
def maps_nb(x0,y0,H,W,c0,c1,h0,h1,d0,d1,span,grid,delta,scurve,sl_d,sl_s,lip_y,strength,mx,my):
    lip32=np.float32(lip_y); f06=np.float32(.60); f045=np.float32(.45); one32=np.float32(1); z32=np.float32(0)
    for y in range(H):
        yy=np.float32(y0+y)
        ry=np.float32(yy-c1)
        for x in range(W):
            xx=np.float32(x0+x)
            rx=np.float32(xx-c0)
            u=np.float32(np.float32(np.float32(rx*h0)+np.float32(ry*h1))/span)
            v=np.float32(np.float32(np.float32(rx*d0)+np.float32(ry*d1))/span)
            uf=np.float64(u)
            move=interp1(uf,grid,delta,sl_d)*strength
            target=interp1(uf,grid,scurve,sl_s)-move
            den=target-lip_y
            if den<.12: den=.12
            a=np.float64(np.float32(v-lip32))/den
            a=min(max(a,0.0),1.0); upper=a*a*(3.0-2.0*a)
            b=(np.float64(v)-target)/.65
            b=min(max(b,0.0),1.0); lower=1.0-b*b*(3.0-2.0*b)
            t=np.float32(np.float32(abs(u)-f06)/f045)
            t=min(max(t,z32),one32); lat=np.float32(one32-np.float32(np.float32(t*t)*np.float32(np.float32(3)-np.float32(np.float32(2)*t))))
            disp=move*np.float64(span)*upper*lower*np.float64(lat)
            mx[y,x]=np.float32(np.float64(xx)+disp*np.float64(d0))
            my[y,x]=np.float32(np.float64(yy)+disp*np.float64(d1))

def slopes(xp,fp):
    inv=1.0/(xp[1:]-xp[:-1]); return (fp[1:]-fp[:-1])*inv

def warp_fast(preserve,i,strength=1.,inplace=False):
    p,g=d['p'][i],d['g'][i]; center,horizontal,down,span=axes(p); source_curve,_=curves(p,g)
    lip_y=np.max((g[LIPS]-center)@down/span)+.07; bottom=np.max(source_curve)+.65
    corners=np.array([center+span*(u*horizontal+v*down) for u in (-1.05,1.05) for v in (lip_y,bottom)])
    H,W=preserve.shape[:2]; x0,y0=np.maximum(np.floor(corners.min(axis=0)).astype(int)-2,0); x1,y1=np.minimum(np.ceil(corners.max(axis=0)).astype(int)+3,[W,H])
    out=preserve if inplace else preserve.copy()
    if x1<=x0 or y1<=y0: return out
    delta=d['chin_delta'][i]
    if not delta.any(): return out   # exact: zero move => identity remap
    h,w=y1-y0,x1-x0; mx=np.empty((h,w),np.float32); my=np.empty((h,w),np.float32)
    maps_nb(int(x0),int(y0),h,w,center[0],center[1],horizontal[0],horizontal[1],down[0],down[1],span,GRID,delta,source_curve,
            slopes(GRID,delta),slopes(GRID,source_curve),float(lip_y),float(strength),mx,my)
    src=preserve if not inplace else preserve.copy()
    out[y0:y1,x0:x1]=cv2.remap(src,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return out

retained=[]
for i in range(N):
    x,y,x1,y1=map(int,d['cache']['boxes'][i]); m=d['refined_masks'][i].current(d['g'][i],d['chin_delta'][i])
    retained.append(chin.get_image_blending(frames[i].copy(),cv2.resize(faces[i],(x1-x,y1-y)),d['cache']['boxes'][i],m,d['cache']['cropboxes'][i]))
print('lip_y dtype (numpy',np.__version__,'):',type(np.max(np.zeros(3,np.float32))+.07).__name__)
bad=0
for i in range(N):
    ref=chin.warp_roi(retained[i],d,i,1.); fast=warp_fast(retained[i],i)
    if not np.array_equal(ref,fast): bad+=1; print('mismatch frame',i,int(np.abs(ref.astype(int)-fast).max()))
print(WHO,'frames',N,'bit-exact mismatches:',bad)
def t(fn,reps=3):
    s=time.perf_counter()
    for _ in range(reps):
        for i in range(N): fn(retained[i],i)
    return round((time.perf_counter()-s)/(reps*N)*1000,4)
print('warp ms/frame: numpy warp_roi',t(lambda r,i: chin.warp_roi(r,d,i,1.)),' numba exact',t(warp_fast))
def scale(fn,w,sec=1.5):
    stop=time.perf_counter()+sec; c=[0]*w
    def run(k):
        j=k
        while time.perf_counter()<stop: fn(retained[j%N],j%N); j+=1; c[k]+=1
    th=[threading.Thread(target=run,args=(k,)) for k in range(w)]; s=time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]
    return round(sum(c)/(time.perf_counter()-s))
print('thread fps numpy',{w:scale(lambda r,i: chin.warp_roi(r,d,i,1.),w) for w in (1,4,8)},' numba',{w:scale(warp_fast,w) for w in (1,4,8)})
