import av, cv2, numpy as np, glob
img = cv2.imread(sorted(glob.glob("/workspace/MuseTalk/results/v15/avatars/japanese_realtime_talking_7d94520b7f/full_imgs/*.png"))[0])
h, w = img.shape[:2]
a = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p").to_ndarray()  # (h*3/2, w)
b = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
Ya, Yb = a[:h].astype(int), b[:h].astype(int)
print("Y range swscale", Ya.min(), Ya.max(), "cv2", Yb.min(), Yb.max())
print("Y abs diff mean %.2f max %d" % (np.abs(Ya - Yb).mean(), np.abs(Ya - Yb).max()))
print("UV abs diff mean %.2f max %d" % (np.abs(a[h:].astype(int) - b[h:].astype(int)).mean(), np.abs(a[h:].astype(int) - b[h:].astype(int)).max()))
# round-trip both through swscale decode path (yuv420p->bgr24) and compare to source
for name, yuv in (("swscale", a), ("cv2", b)):
    back = av.VideoFrame.from_ndarray(np.ascontiguousarray(yuv), format="yuv420p").to_ndarray(format="bgr24")
    print(name, "roundtrip-vs-source mean abs err %.2f" % np.abs(back.astype(int) - img.astype(int)).mean())
