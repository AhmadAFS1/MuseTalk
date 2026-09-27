import sys; sys.dont_write_bytecode=True
sys.path[:0]=['/workspace/MuseTalk','/workspace/MuseTalk/character_factory/h3_avatar_workflow']
import numpy as np, cv2, torch, chin, time
root='/workspace/experiments/avatar_diversity_20260927/black_woman'; N=24
cap=cv2.VideoCapture(f'{root}/source.mp4'); frames=[cap.read()[1] for _ in range(N)]; cap.release()
d=dict(cache=torch.load(f'{root}/cache.pt',map_location='cpu',weights_only=False),masks=np.load(f'{root}/masks.npz'),p=np.load(f'{root}/source_landmarks.npy'))
t=time.perf_counter(); chin.prepare_refined(d,frames); prep=(time.perf_counter()-t)/N
def nb(o): return sum(v.nbytes for v in vars(o).values() if isinstance(v,np.ndarray))
sm=np.mean([nb(m) for m in d['source_masks']]); rm=np.mean([nb(m) for m in d['refined_masks']])
pl=np.mean([sum(v.nbytes for v in p.values() if isinstance(v,np.ndarray)) for p in d['plans']])
m=d['refined_masks'][0]
print('per source frame MB: SourceMask',round(sm/1e6,3),'RefinedMask(incl. inherited)',round(rm/1e6,3),'plan',round(pl/1e6,3),'frame',frames[0].nbytes/1e6)
print({k:(v.dtype.str,v.shape) for k,v in vars(m).items() if isinstance(v,np.ndarray)})
print('prepare_refined seconds per source frame',round(prep,4))
