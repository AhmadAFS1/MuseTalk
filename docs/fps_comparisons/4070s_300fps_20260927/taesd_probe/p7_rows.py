"""Probe 7 (CPU only): across every cached avatar, the first 256-face row the blend can read.

For each frame: alpha>0 AND inside the pasted bbox -> first full-frame row r0; cv2.resize INTER_LINEAR
reads source row floor((r - y1 + 0.5) * 256/h - 0.5) (and the next row). The row crop at 104 is
safe iff that source row >= 104 for every frame.
"""
import glob, pickle, json, os, sys
import numpy as np, cv2

AV = "/workspace/MuseTalk/results/v15/avatars"
out = {}
for d in sorted(glob.glob(AV + "/*")):
    try:
        coords = pickle.load(open(d + "/coords.pkl", "rb"))
        mco = pickle.load(open(d + "/mask_coords.pkl", "rb"))
        masks = sorted(glob.glob(d + "/mask/*.png"))
    except Exception:
        continue
    n = min(len(coords), len(mco), len(masks))
    if n == 0:
        continue
    mins, xmins, xmaxs = [], [], []
    for i in range(0, n, 2):
        x1, y1, x2, y2 = [int(v) for v in coords[i]]
        cx0, cy0, cx1, cy1 = [int(v) for v in mco[i]]
        m = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        if m is None or y2 <= y1 or x2 <= x1:
            continue
        ys, xs = np.nonzero(m)
        ys = ys + cy0; xs = xs + cx0
        inside = (ys >= y1) & (ys < y2) & (xs >= x1) & (xs < x2)
        if not inside.any():
            continue
        r0 = ys[inside].min()
        h = y2 - y1; w = x2 - x1
        src = int(np.floor((r0 - y1 + 0.5) * 256.0 / h - 0.5))
        mins.append(src)
        xmins.append(int(np.floor((xs[inside].min() - x1 + 0.5) * 256.0 / w - 0.5)))
        xmaxs.append(int(np.ceil((xs[inside].max() - x1 + 0.5) * 256.0 / w - 0.5)))
    if mins:
        out[os.path.basename(d)] = {"frames_checked": len(mins), "min_src_row_256": int(min(mins)),
                                    "median_src_row_256": float(np.median(mins)),
                                    "src_col_range_256": [int(min(xmins)), int(max(xmaxs))]}
        print(os.path.basename(d), out[os.path.basename(d)], flush=True)
glob_min = min(v["min_src_row_256"] for v in out.values())
print("GLOBAL min source row read by blend:", glob_min, "(crop at 104 safe:", glob_min >= 104, ")")
json.dump({"avatars": out, "global_min_src_row_256": glob_min},
          open(os.path.join(os.path.dirname(__file__), "p7_rows.json"), "w"), indent=2)
