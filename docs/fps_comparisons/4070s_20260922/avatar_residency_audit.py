"""Exact per-avatar residency, and how much of it is duplicated cycle data."""
import hashlib, os, pickle, sys
from pathlib import Path
ROOT = Path("/workspace/MuseTalk"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import cv2, numpy as np, torch
from musetalk.utils.blending import prepare_image_blending_plan
MB = 1024**2

for name in ["codex_smoke", "shared_indian_20260915"]:
    d = ROOT/"results/v15/avatars"/name
    if not d.exists(): continue
    fulls = sorted((d/"full_imgs").glob("*.[pj][pn]g"))
    masks = sorted((d/"mask").glob("*.[pj][pn]g"))
    coords = pickle.load(open(d/"coords.pkl","rb"))
    mcoords = pickle.load(open(d/"mask_coords.pkl","rb"))
    lat = torch.load(d/"latents.pt", map_location="cpu")
    N = len(fulls)

    fh = [hashlib.md5(open(f,'rb').read()).hexdigest() for f in fulls]
    mh = [hashlib.md5(open(f,'rb').read()).hexdigest() for f in masks]
    uf, um = len(set(fh)), len(set(mh))
    palin = all(fh[i] == fh[N-1-i] for i in range(N))

    fr0 = cv2.imread(str(fulls[0])); mk0 = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    plan = prepare_image_blending_plan(fr0.shape, coords[0], mk0, mcoords[0])
    plan_each = (plan["alpha"].nbytes + plan["alpha_u8"].nbytes) if plan else 0

    cur = fr0.nbytes*N + mk0.nbytes*len(masks) + plan_each*N
    ded = fr0.nbytes*uf + mk0.nbytes*um + plan_each*uf
    latb = lat.element_size()*lat.nelement()

    print(f"\n### {name}   cycle={N} frames   palindromic={palin}")
    print(f"  unique frames {uf}/{N}   unique masks {um}/{len(masks)}")
    print(f"  {'':22s} {'current':>10s} {'deduped':>10s}")
    print(f"  {'frames (host)':22s} {fr0.nbytes*N/MB:9.1f}M {fr0.nbytes*uf/MB:9.1f}M")
    print(f"  {'masks (host)':22s} {mk0.nbytes*len(masks)/MB:9.1f}M {mk0.nbytes*um/MB:9.1f}M")
    print(f"  {'compose plans (host)':22s} {plan_each*N/MB:9.1f}M {plan_each*uf/MB:9.1f}M")
    print(f"  {'HOST TOTAL':22s} {cur/MB:9.1f}M {ded/MB:9.1f}M   ({cur/max(ded,1):.2f}x saving)")
    print(f"  {'latents (GPU)':22s} {latb/MB:9.1f}M {latb/MB:9.1f}M")
    for budget in (16, 24, 32):
        print(f"    avatars in {budget:2d}GB host RAM: {int(budget*1024**3/cur):4d}  ->  {int(budget*1024**3/ded):4d}")
