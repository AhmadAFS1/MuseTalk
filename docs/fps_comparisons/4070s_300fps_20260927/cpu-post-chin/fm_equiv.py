"""How far do cheaper FaceMesh variants move the 41 landmarks the chin path uses? Scratch."""
import sys,os,time; sys.dont_write_bytecode=True
import numpy as np, cv2, mediapipe as mp
WHO=sys.argv[1] if len(sys.argv)>1 else 'black_woman'; N=240
root=f'/workspace/experiments/avatar_diversity_20260927/{WHO}'
JAW=[234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454]
LIPS=[61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
cap=cv2.VideoCapture(f'{root}/source.mp4'); frames=[cap.read()[1] for _ in range(N)]; cap.release()
faces=np.load(f'{root}/faces.npz')['faces']; boxes=np.load(f'boxes_{WHO}.npy') if os.path.exists(f'boxes_{WHO}.npy') else None
saved=np.load(f'{root}/generated_landmarks.npy')
imgs=[]
for f,face,b in zip(frames,faces,boxes):
    x,y,x1,y1=map(int,b); g=f.copy(); g[y:y1,x:x1]=cv2.resize(face,(x1-x,y1-y)); imgs.append(g)
def track(imgs,refine=True,static=False,crop=None,scale=None):
    m=mp.solutions.face_mesh.FaceMesh(static_image_mode=static,max_num_faces=1,refine_landmarks=refine,min_detection_confidence=.5,min_tracking_confidence=.5)
    out=[]; t=time.perf_counter()
    for im in imgs:
        src=im if crop is None else im[crop[1]:crop[3],crop[0]:crop[2]]
        if scale: src=cv2.resize(src,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        h,w=src.shape[:2]; r=m.process(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
        pts=np.array([(p.x*w,p.y*h) for p in r.multi_face_landmarks[0].landmark],np.float32)
        if scale: pts/=scale
        if crop is not None: pts+=np.array(crop[:2],np.float32)
        out.append(pts)
    m.close(); return np.asarray(out),(time.perf_counter()-t)/len(imgs)*1000
def cmp(a,b):
    idx=JAW+LIPS; e=np.linalg.norm(a[:,idx]-b[:,idx],axis=2)
    return dict(mean_px=round(float(e.mean()),3),p99_px=round(float(np.percentile(e,99)),3),max_px=round(float(e.max()),3))
ref,tr=track(imgs); print(WHO,'full refined tracking ms',round(tr,3),'vs saved generated_landmarks (same env, fresh run):',cmp(ref,saved))
b=boxes[0].astype(int); crop=[int(max(0,b[0]-64)),int(max(0,b[1]-96)),int(min(512,b[2]+64)),int(min(896,b[3]+64))]
for name,kw in [('crop around face',dict(crop=crop)),('half-res input',dict(scale=.5)),('static_image_mode',dict(static=True)),('unrefined (no attention)',dict(refine=False))]:
    a,t=track(imgs,**kw); print(name,'ms',round(t,3),'jaw+lip deviation vs full refined tracking:',cmp(a,ref),flush=True)
