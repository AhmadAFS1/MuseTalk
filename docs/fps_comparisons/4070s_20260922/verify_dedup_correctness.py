"""Verify avatar cycle dedup: real RSS, accounting, and bit-identical composes."""
import os, sys, gc, pickle, resource
from pathlib import Path
ROOT = Path("/workspace/MuseTalk"); os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import cv2, numpy as np
import api_avatar as A
from musetalk.utils.blending import prepare_image_blending_plan, get_image_blending_with_plan

AV = ROOT/"results/v15/avatars/codex_smoke"
MB = 1024**2

def load_cycle(dedup: bool):
    os.environ["MUSETALK_AVATAR_DEDUP_CYCLE"] = "1" if dedup else "0"
    fulls = sorted((AV/"full_imgs").glob("*.png")); masks = sorted((AV/"mask").glob("*.png"))
    frames = A._read_imgs_local([str(f) for f in fulls], "frames", 8)
    mks    = A._read_imgs_local([str(f) for f in masks], "masks", 8)
    uf = um = None
    if A._avatar_dedup_enabled():
        frames, uf = A._dedup_cycle_arrays(frames, "frames")
        mks, um    = A._dedup_cycle_arrays(mks, "masks")
    coords = pickle.load(open(AV/"coords.pkl","rb")); mcoords = pickle.load(open(AV/"mask_coords.pkl","rb"))
    n = min(len(frames), len(coords), len(mks), len(mcoords))
    plans, cache = [], {}
    for i in range(n):
        key = (tuple(frames[i].shape), tuple(int(v) for v in coords[i]), id(mks[i]), tuple(int(v) for v in mcoords[i]))
        pl = cache.get(key)
        if pl is None:
            pl = prepare_image_blending_plan(frames[i].shape, coords[i], mks[i], mcoords[i]); cache[key] = pl
        plans.append(pl)
    return frames, mks, plans, coords, uf, um

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

for dedup in (False, True):
    gc.collect()
    frames, mks, plans, coords, uf, um = load_cycle(dedup)
    def np_bytes(vals):
        total, seen = 0, set()
        for v in vals:
            if not isinstance(v, np.ndarray): continue
            base = v.base if v.base is not None else v
            try: k = base.__array_interface__["data"][0]
            except Exception: k = id(base)
            if k in seen: continue
            seen.add(k); total += int(v.nbytes)
        return total
    def plan_bytes(pl):
        total, seen = 0, set()
        for q in pl:
            if not isinstance(q, dict) or id(q) in seen: continue
            seen.add(id(q))
            for k in ("alpha","alpha_u8"):
                a = q.get(k)
                if isinstance(a, np.ndarray): total += int(a.nbytes)
        return total
    acct = np_bytes(frames) + np_bytes(mks) + plan_bytes(plans)
    uniq_plans = len({id(p) for p in plans})
    print(f"\ndedup={dedup}")
    print(f"  cycle len          {len(frames)}")
    print(f"  unique frame bufs  {len({f.__array_interface__['data'][0] for f in frames})}")
    print(f"  unique plan objs   {uniq_plans}")
    print(f"  accounted host RAM {acct/MB:8.1f} MB")
    print(f"  process peak RSS   {rss_mb():8.1f} MB")
    if dedup:
        # correctness: composed output must be bit-identical to non-deduped
        res = (np.random.RandomState(0).rand(256,256,3)*255).astype(np.uint8)
        outs = []
        for i in (0, 1, 299, 300, 599):
            x1,y1,x2,y2 = coords[i]
            rr = cv2.resize(res, (x2-x1, y2-y1))
            outs.append(get_image_blending_with_plan(frames[i].copy(), rr, plans[i]))
        globals()["_dedup_outs"] = outs
    else:
        res = (np.random.RandomState(0).rand(256,256,3)*255).astype(np.uint8)
        outs = []
        for i in (0, 1, 299, 300, 599):
            x1,y1,x2,y2 = coords[i]
            rr = cv2.resize(res, (x2-x1, y2-y1))
            outs.append(get_image_blending_with_plan(frames[i].copy(), rr, plans[i]))
        globals()["_plain_outs"] = outs
    del frames, mks, plans

a, b = globals()["_plain_outs"], globals()["_dedup_outs"]
diffs = [int(np.abs(x.astype(int)-y.astype(int)).max()) for x, y in zip(a, b)]
print(f"\ncomposed-frame max abs diff (dedup vs plain), positions 0/1/299/300/599: {diffs}")
assert all(d == 0 for d in diffs), "DEDUP CHANGED OUTPUT"
print("PASS: composed frames bit-identical")

# mutation safety: compose_frame copies before blending, so shared buffers must survive
print("\nmutation-safety check: frame buffer unchanged after a compose")
os.environ["MUSETALK_AVATAR_DEDUP_CYCLE"] = "1"
frames, mks, plans, coords, _, _ = load_cycle(True)
before = frames[0].copy()
x1,y1,x2,y2 = coords[0]
rr = cv2.resize((np.random.rand(256,256,3)*255).astype(np.uint8), (x2-x1, y2-y1))
get_image_blending_with_plan(frames[0].copy(), rr, plans[0])
print(f"  source buffer untouched: {np.array_equal(before, frames[0])}")
assert np.array_equal(before, frames[0])
print("PASS")
