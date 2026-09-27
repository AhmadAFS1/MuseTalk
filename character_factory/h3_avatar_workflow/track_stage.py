"""Source landmarks in the established, compatible FaceMesh environment."""
from pathlib import Path
import cv2,numpy as np,mediapipe as mp
from common import completed,digest,finish,sha,spec_arg

def track(path):
    cap=cv2.VideoCapture(str(path));rows=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,
            min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
        while True:
            ok,f=cap.read()
            if not ok:break
            result=mesh.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:raise RuntimeError(f'Missing face at frame {len(rows)}: {path}')
            rows.append([(p.x*f.shape[1],p.y*f.shape[0]) for p in result.multi_face_landmarks[0].landmark])
    cap.release();assert len(rows)==240
    return np.asarray(rows,np.float32)

def main():
    s=spec_arg();out=Path(s['output']);cv2.setNumThreads(2)
    sig=digest(dict(spec=s['signature'],source=sha(out/'source.mp4'),code=sha(__file__)))
    if completed(out/'tracking.json',sig):return
    points=track(out/'source.mp4');np.save(out/'source_landmarks.npy',points)
    finish(out/'tracking.json',sig,[out/'source_landmarks.npy'],frames=240,mediapipe=mp.__version__)

if __name__=='__main__':main()
