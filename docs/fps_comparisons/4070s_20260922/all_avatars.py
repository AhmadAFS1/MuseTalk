"""Per-avatar: cycle structure, dedup saving, residency. Shows what generalises and what doesn't."""
import hashlib, os, pickle, sys
from pathlib import Path
ROOT=Path("/workspace/MuseTalk"); os.chdir(ROOT); sys.path.insert(0,str(ROOT))
import cv2, numpy as np, torch
from musetalk.utils.blending import prepare_image_blending_plan
MB=1024**2
rows=[]
for d in sorted((ROOT/"results/v15/avatars").iterdir()):
    if not (d/"full_imgs").exists(): continue
    fulls=sorted((d/"full_imgs").glob("*.png")); masks=sorted((d/"mask").glob("*.png"))
    if not fulls: continue
    coords=pickle.load(open(d/"coords.pkl","rb")); mco=pickle.load(open(d/"mask_coords.pkl","rb"))
    N=len(fulls)
    fh=[hashlib.blake2b(open(f,'rb').read(),digest_size=16).digest() for f in fulls]
    uf=len(set(fh))
    fr=cv2.imread(str(fulls[0])); mk=cv2.imread(str(masks[0]),cv2.IMREAD_GRAYSCALE)
    pl=prepare_image_blending_plan(fr.shape,coords[0],mk,mco[0])
    plan_b=(pl["alpha"].nbytes+pl["alpha_u8"].nbytes) if pl else 0
    cur=fr.nbytes*N + mk.nbytes*len(masks) + plan_b*N
    ded=fr.nbytes*uf + mk.nbytes*uf + plan_b*uf
    x1,y1,x2,y2=coords[0]
    rows.append(dict(name=d.name, size=f"{fr.shape[1]}x{fr.shape[0]}", N=N, uniq=uf,
                     bbox=f"{x2-x1}x{y2-y1}", cur=cur/MB, ded=ded/MB, save=cur/max(ded,1)))
print(f"{'avatar':44s} {'frame':>9s} {'cycle':>6s} {'uniq':>5s} {'bbox':>8s} {'RAM now':>9s} {'dedup':>9s} {'save':>6s}")
for r in rows:
    print(f"{r['name']:44s} {r['size']:>9s} {r['N']:6d} {r['uniq']:5d} {r['bbox']:>8s} "
          f"{r['cur']:8.1f}M {r['ded']:8.1f}M {r['save']:5.2f}x")
tot_c=sum(r['cur'] for r in rows); tot_d=sum(r['ded'] for r in rows)
print(f"\nall {len(rows)} avatars resident: {tot_c:.0f} MB -> {tot_d:.0f} MB ({tot_c/tot_d:.2f}x)")
