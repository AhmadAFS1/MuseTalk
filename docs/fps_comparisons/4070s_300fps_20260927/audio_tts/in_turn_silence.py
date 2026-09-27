"""Fraction of in-utterance frames that sit inside long silent runs (after edge trim)."""
import json, sys, numpy as np, soundfile as sf, librosa
files = sys.argv[1:]
out = {}
for f in files:
    y, sr = librosa.load(f, sr=16000)
    w = 160  # 10 ms
    n = len(y) // w
    rms = np.sqrt((y[: n * w].reshape(n, w) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    active = db >= -45
    idx = np.flatnonzero(active)
    if idx.size == 0: continue
    a = active[idx[0]: idx[-1] + 1]  # trimmed utterance
    # runs of silence
    runs = []; cur = 0
    for v in a:
        if not v: cur += 1
        else:
            if cur: runs.append(cur); cur = 0
    total = len(a)
    res = {"utterance_s": total / 100}
    for thr in (200, 300, 500):
        res[f"silent_in_runs_ge_{thr}ms_pct"] = round(100 * sum(r for r in runs if r * 10 >= thr) / total, 1)
    res["silent_any_pct"] = round(100 * (total - a.sum()) / total, 1)
    out[f.split("/")[-1]] = res
print(json.dumps(out, indent=1))
