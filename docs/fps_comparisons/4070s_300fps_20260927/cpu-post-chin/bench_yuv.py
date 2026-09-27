import sys, time, threading, numpy as np, cv2, av
sys.dont_write_bytecode=True
cv2.setNumThreads(1)
cap=cv2.VideoCapture('/workspace/experiments/avatar_diversity_20260927/black_woman/source.mp4'); frames=[]
for _ in range(48):
    ok,f=cap.read(); frames.append(f)
def pyav(i): return av.VideoFrame.from_ndarray(frames[i%48],format='bgr24').reformat(format='yuv420p')
def pyav_split(i):
    fr=av.VideoFrame.from_ndarray(frames[i%48],format='bgr24'); return fr
def ocv(i): return cv2.cvtColor(frames[i%48],cv2.COLOR_BGR2YUV_I420)
def ocv_to_frame(i):
    y=cv2.cvtColor(frames[i%48],cv2.COLOR_BGR2YUV_I420); return av.VideoFrame.from_ndarray(y,format='yuv420p')
def scale(fn,w,sec=1.5):
    stop=time.perf_counter()+sec; c=[0]*w
    def run(k):
        j=k
        while time.perf_counter()<stop: fn(j); j+=1; c[k]+=1
    th=[threading.Thread(target=run,args=(k,)) for k in range(w)]; t=time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]
    return round(sum(c)/(time.perf_counter()-t),1)
for name,fn in [('pyav_from_ndarray+reformat',pyav),('pyav_from_ndarray_only',pyav_split),('cv2_BGR2YUV_I420',ocv),('cv2_I420+VideoFrame.from_ndarray(yuv420p)',ocv_to_frame)]:
    for _ in range(10): fn(0)
    t=time.perf_counter()
    for i in range(300): fn(i)
    ms=(time.perf_counter()-t)/300*1000
    print(name,'ms/frame',round(ms,3),'thread fps',{w:scale(fn,w) for w in (1,2,4,8)},flush=True)
a=pyav(0).to_ndarray(); b=cv2.cvtColor(frames[0],cv2.COLOR_BGR2YUV_I420)
d=np.abs(a.astype(int)-b.astype(int)); print('pyav vs cv2 I420: shape',a.shape,b.shape,'max',d.max(),'mean',round(d.mean(),3),'Ymax',d[:896].max(),'UVmax',d[896:].max())
