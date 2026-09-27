import av, cv2, numpy as np, time, glob
cv2.setNumThreads(1)
img = cv2.imread(sorted(glob.glob("/workspace/MuseTalk/results/v15/avatars/japanese_realtime_talking_7d94520b7f/full_imgs/*.png"))[0])
def t(fn,n=400):
    for _ in range(20): fn()
    s=time.perf_counter()
    for _ in range(n): fn()
    return round((time.perf_counter()-s)*1000/n,4)
for w,h in [(512,832),(512,896),(384,672),(480,832),(448,768)]:
    im=np.ascontiguousarray(cv2.resize(img,(w,h)))
    f=av.VideoFrame.from_ndarray(im,format="bgr24")
    print(w,h,"from_ndarray",t(lambda: av.VideoFrame.from_ndarray(im,format="bgr24")),
          "reformat",t(lambda: f.reformat(format="yuv420p")),
          "reformat_rgb24path", t(lambda: av.VideoFrame.from_ndarray(im[:,:,::-1].copy(),format="rgb24").reformat(format="yuv420p")),
          "cv2_1thr",t(lambda: cv2.cvtColor(im,cv2.COLOR_BGR2YUV_I420)),
          "cv2+from_ndarray(yuv)", t(lambda: av.VideoFrame.from_ndarray(cv2.cvtColor(im,cv2.COLOR_BGR2YUV_I420),format="yuv420p")))
