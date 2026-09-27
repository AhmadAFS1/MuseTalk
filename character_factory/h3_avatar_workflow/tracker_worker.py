"""Existing FaceMesh environment, accessed through a single shared frame slot."""
import sys, time
from multiprocessing import shared_memory, resource_tracker
import cv2, numpy as np, mediapipe as mp
cv2.setNumThreads(2)
shm=shared_memory.SharedMemory(name=sys.argv[1])
resource_tracker.unregister(shm._name, 'shared_memory')
frame=np.ndarray((896,512,3),np.uint8,buffer=shm.buf)
points=np.ndarray((478,2),np.float32,buffer=shm.buf,offset=frame.nbytes)
mesh=None
try:
    for line in sys.stdin:
        command=line.strip()
        if command=='quit':break
        if command=='reset':
            if mesh:mesh.close()
            mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
                refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
            print('ready',flush=True)
        elif command=='frame':
            start=time.perf_counter()
            result=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:raise RuntimeError('Missing generated face')
            points[:]=[(p.x*512,p.y*896) for p in result.multi_face_landmarks[0].landmark]
            print('ok',time.perf_counter()-start,flush=True)
        else:raise ValueError(command)
finally:
    if mesh:mesh.close()
    shm.close()
