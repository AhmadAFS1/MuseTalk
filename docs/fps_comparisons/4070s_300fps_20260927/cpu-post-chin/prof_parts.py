"""Split warp_roi / RefinedMask.current into GIL-free cv2 parts vs numpy/Python parts. Scratch."""
import sys, time
sys.dont_write_bytecode=True
sys.path[:0]=['/workspace/MuseTalk','/workspace/MuseTalk/character_factory/h3_avatar_workflow']
import numpy as np, cv2, torch, chin
from chin import axes, curves, smooth, GRID, LIPS
cv2.setNumThreads(1)
root='/workspace/experiments/avatar_diversity_20260927/black_woman'; N=48
cap=cv2.VideoCapture(f'{root}/source.mp4'); frames=[cap.read()[1] for _ in range(N)]; cap.release()
d=dict(cache=torch.load(f'{root}/cache.pt',map_location='cpu',weights_only=False),masks=np.load(f'{root}/masks.npz'),p=np.load(f'{root}/source_landmarks.npy'))
faces=np.load(f'{root}/faces.npz')['faces'][:N]
chin.prepare_refined(d,frames); d['g']=np.load(f'{root}/generated_landmarks.npy'); d['chin_delta']=np.load(f'{root}/chin_delta.npy')
T={}
def tick(k,t0):
    t=time.perf_counter(); T[k]=T.get(k,0)+(t-t0); return t
def warp_parts(preserve,i):
    t=time.perf_counter()
    p,g=d['p'][i],d['g'][i]; center,horizontal,down,span=axes(p); source_curve,_=curves(p,g)
    lip_y=np.max((g[LIPS]-center)@down/span)+.07; bottom=np.max(source_curve)+.65
    corners=np.array([center+span*(u*horizontal+v*down) for u in (-1.05,1.05) for v in (lip_y,bottom)])
    H,W=preserve.shape[:2]; x0,y0=np.maximum(np.floor(corners.min(axis=0)).astype(int)-2,0); x1,y1=np.minimum(np.ceil(corners.max(axis=0)).astype(int)+3,[W,H])
    t=tick('w1_setup_scalar(py)',t)
    out=preserve.copy(); t=tick('w2_fullframe_copy',t)
    yy,xx=np.indices((y1-y0,x1-x0),np.float32); xx+=x0; yy+=y0
    rx,ry=xx-center[0],yy-center[1]; u=(rx*horizontal[0]+ry*horizontal[1])/span; v=(rx*down[0]+ry*down[1])/span
    t=tick('w3_uv_static',t)
    move=np.interp(u,GRID,d['chin_delta'][i]); t=tick('w4_interp_delta(dyn)',t)
    st=np.interp(u,GRID,source_curve); t=tick('w5_interp_source(static)',t)
    target=st-move; upper=smooth((v-lip_y)/np.maximum(target-lip_y,.12)); lower=1-smooth((v-target)/.65); t=tick('w6_upper_lower(dyn)',t)
    lateral=1-smooth((np.abs(u)-.60)/.45); t=tick('w7_lateral(static)',t)
    disp=move*span*upper*lower*lateral; mx=(xx+disp*down[0]).astype(np.float32); my=(yy+disp*down[1]).astype(np.float32); t=tick('w8_disp_maps(dyn)',t)
    out[y0:y1,x0:x1]=cv2.remap(preserve,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101); t=tick('w9_cv2_remap(nogil)',t)
    T['roi_px']=T.get('roi_px',0)+(y1-y0)*(x1-x0)
    return out
def mask_parts(m,g,delta):
    t=time.perf_counter()
    shift=np.interp(m.u,GRID,delta); t=tick('m1_interp_delta',t)
    cap=1-smooth((m.relative+shift+.45)/.40); weight=1-m.lateral+m.lateral*cap; t=tick('m2_cap_weight',t)
    hull=cv2.convexHull(np.rint(g[LIPS]-np.asarray(m.cb[:2])).astype(np.int32))
    points=np.concatenate([m.source_hull.reshape(-1,2),hull.reshape(-1,2)]); padding=m.radius+int(np.ceil(m.feather*1.1))+4
    x0,y0=np.maximum(points.min(0)-padding,0); x1,y1=np.minimum(points.max(0)+padding+1,m.mask.shape[::-1])
    lip=m.lip[y0:y1,x0:x1].copy(); cv2.fillConvexPoly(lip,hull-np.array([x0,y0],np.int32),255); t=tick('m3_hull_fill(py+cv2)',t)
    expanded=cv2.dilate(lip,m.kernel); distance=cv2.distanceTransform(255-expanded,cv2.DIST_L2,5); t=tick('m4_dilate_dist(nogil)',t)
    guard=1-smooth(distance/m.feather); t=tick('m5_guard_smooth',t)
    ys,xs=m.region; ix0,iy0=max(x0,xs.start),max(y0,ys.start); ix1,iy1=min(x1,xs.stop),min(y1,ys.stop)
    if ix1>ix0 and iy1>iy0:
        region=weight[iy0-ys.start:iy1-ys.start,ix0-xs.start:ix1-xs.start]; gg=guard[iy0-y0:iy1-y0,ix0-x0:ix1-x0]
        weight[iy0-ys.start:iy1-ys.start,ix0-xs.start:ix1-xs.start]=region+gg*(1-region)
    t=tick('m6_combine',t)
    mask=m.mask.copy(); mask[m.region]=np.rint(m.mask[m.region].astype(np.float32)*weight).astype(np.uint8); t=tick('m7_assemble_rint',t)
    T['lip_px']=T.get('lip_px',0)+(y1-y0)*(x1-x0); T['region_px']=T.get('region_px',0)+m.u.size
    return mask
reps=4
retained=[]
for i in range(N):
    x,y,x1,y1=map(int,d['cache']['boxes'][i]); m=d['refined_masks'][i].current(d['g'][i],d['chin_delta'][i])
    retained.append(chin.get_image_blending(frames[i].copy(),cv2.resize(faces[i],(x1-x,y1-y)),d['cache']['boxes'][i],m,d['cache']['cropboxes'][i]))
for r in range(reps):
    for i in range(N):
        a=warp_parts(retained[i],i); assert r or np.array_equal(a,chin.warp_roi(retained[i],d,i,1.))
        b=mask_parts(d['refined_masks'][i],d['g'][i],d['chin_delta'][i]); assert r or np.array_equal(b,d['refined_masks'][i].current(d['g'][i],d['chin_delta'][i]))
n=reps*N
for k in sorted(T):
    print(k, round(T[k]/n*1000,4) if not k.endswith('px') else int(T[k]/n))
