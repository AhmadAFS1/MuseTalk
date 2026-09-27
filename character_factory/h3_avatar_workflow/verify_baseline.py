"""Require the reusable module to reproduce all accepted raw output pixels."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;WORKSPACE=HERE.parents[2]
sys.path.insert(0,str(WORKSPACE/'MuseTalk'))
import cv2,numpy as np,torch
import chin
from common import dump,sha
cv2.setNumThreads(2);torch.set_num_threads(2)

def main():
    accepted=WORKSPACE/'experiments/chin_seam_refinement_20260927'
    diagnostics=json.loads((accepted/'diagnostics.json').read_text());report={}
    for who in ('japanese','latina'):
        base=WORKSPACE/'experiments/portrait_jaw_video_20260926';folder=base/f'{who}_new'
        d=dict(cache=torch.load(folder/'cache.pt',map_location='cpu',weights_only=False),masks=np.load(folder/'masks.npz'),
            p=np.load(base/'analysis'/f'{who}_new_portrait/source_landmarks.npy'),
            g=np.load(WORKSPACE/'experiments/chin_fps_validation_20260927'/who/'taesd_pipelined/generated_landmarks.npy'))
        cap=cv2.VideoCapture(str(base/who/'source.mp4'));frames=[]
        while True:
            ok,f=cap.read()
            if not ok:break
            frames.append(f)
        cap.release();assert len(frames)==240
        faces=np.load(accepted/f'{who}_faces.npy',mmap_mode='r');chin.prepare_refined(d,frames);chin.prepare(d)
        h=hashlib.sha256()
        for i,f in enumerate(frames):h.update(chin.corrected_refined(f,d,i,faces[i]).tobytes())
        assert h.hexdigest()==diagnostics['subjects'][who]['refined_raw_frames_sha256'],who
        report[who]=dict(frames=240,raw_sha256=h.hexdigest(),exact_accepted_match=True)
        print('EXACT BASELINE',who,flush=True)
    dump(HERE/'baseline_parity.json',dict(status='complete',chin_sha256=sha(HERE/'chin.py'),subjects=report,
        numpy=np.__version__,opencv=cv2.__version__))

if __name__=='__main__':main()
