import os, sys, pickle, resource, time
from pathlib import Path
ROOT = Path("/workspace/MuseTalk"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np
import api_avatar as A
from musetalk.utils.blending import prepare_image_blending_plan
AV = ROOT/"results/v15/avatars/codex_smoke"

def cur_rss_mb():
    with open('/proc/self/statm') as fh:
        return int(fh.read().split()[1]) * 4096 / 1024**2

base = cur_rss_mb()
fulls = [str(f) for f in sorted((AV/"full_imgs").glob("*.png"))]
masks = [str(f) for f in sorted((AV/"mask").glob("*.png"))]
coords = pickle.load(open(AV/"coords.pkl","rb")); mcoords = pickle.load(open(AV/"mask_coords.pkl","rb"))

held = []
t0 = time.time()
for n in range(1, 4):                       # load the same avatar 3x = 3 avatars' worth
    frames, uf = A._read_imgs_dedup(fulls, "frames", 8)
    mks, um    = A._read_imgs_dedup(masks, "masks", 8)
    plans, cache = [], {}
    for i in range(len(frames)):
        key = (tuple(frames[i].shape), tuple(int(v) for v in coords[i]), id(mks[i]), tuple(int(v) for v in mcoords[i]))
        pl = cache.get(key)
        if pl is None:
            pl = prepare_image_blending_plan(frames[i].shape, coords[i], mks[i], mcoords[i]); cache[key] = pl
        plans.append(pl)
    held.append((frames, mks, plans))
    print(f"  after avatar {n}: RSS={cur_rss_mb():.1f} MB  (delta from base {cur_rss_mb()-base:.1f})  "
          f"unique frames {uf}/{len(frames)}  plans {len({id(p) for p in plans})}")
print(f"dedup={os.environ.get('MUSETALK_AVATAR_DEDUP_CYCLE','1')}  "
      f"marginal_per_avatar={(cur_rss_mb()-base)/3:.1f} MB  load_time={time.time()-t0:.1f}s")
