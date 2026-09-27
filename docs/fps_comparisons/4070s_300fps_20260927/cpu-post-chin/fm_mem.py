import sys,os,time; sys.dont_write_bytecode=True
import numpy as np, cv2, mediapipe as mp
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])/1024
cap=cv2.VideoCapture('/workspace/experiments/avatar_diversity_20260927/black_woman/source.mp4'); f=cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
print('base MB',round(rss()))
ms=[]
for k in range(4):
    m=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
    m.process(f); ms.append(m); print('graphs',k+1,'RSS MB',round(rss()),'threads',len(os.listdir('/proc/self/task')))
# first-frame (detection) vs tracking cost
m=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
t=time.perf_counter(); m.process(f); a=time.perf_counter()-t
t=time.perf_counter(); m.process(f); b=time.perf_counter()-t
s=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True)
s.process(f); t=time.perf_counter()
for _ in range(20): s.process(f)
c=(time.perf_counter()-t)/20
print('first frame (detector+landmarks) ms',round(a*1000,2),' tracked frame ms',round(b*1000,2),' static_image_mode per frame ms',round(c*1000,2))
