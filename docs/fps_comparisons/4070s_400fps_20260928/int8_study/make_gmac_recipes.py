"""Greedy "error per MAC" INT8 layer recipes (gmac_*), from the single-layer sensitivity study.

Candidates are every Conv2d/Linear the study inventoried outside down0 (it gains nothing from INT8: attention
bound) and up3 (most sensitive), minus the audio cross-attention K/V projections (their per-tensor range, calibrated on
TTS speech, clips real speech; README §3.4). Each layer's error is its best single-layer added MSE (max or MSE
activation clip, whichever is lower); layers are taken in order of error / MACs until the MAC budget is reached.
down3 and mid stay all-INT8 (as in r2). Writes recipes_gmac.json for `int8_layer_study.py --stage recipes`.
The r5 set is gmac_0.50.
usage: make_gmac_recipes.py [--cuts 0.50,0.55,0.59,0.62]
"""
import argparse, fnmatch, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=str(HERE / "study.json"))
    ap.add_argument("--cuts", default="0.50,0.55,0.59,0.62")
    ap.add_argument("--out", default=str(HERE / "recipes_gmac.json"))
    a = ap.parse_args()
    S = json.loads(Path(a.study).read_text())
    inv = {r["name"]: r for r in S["inventory"]}
    sm, ss = S["sens"]["max"], S["sens"]["mse"]

    def pick(n):
        mx, ms = sm.get(n, {}).get("mse_mean"), ss.get(n, {}).get("mse_mean")
        return ("max", mx) if (ms is None or (mx is not None and mx <= ms)) else ("mse", ms)

    cand = [n for n in inv if inv[n]["block"] not in ("down0", "up3") and not fnmatch.fnmatch(n, "*attn2.to_[kv]")
            and pick(n)[1] is not None]
    order = sorted(cand, key=lambda n: max(pick(n)[1], 0) / inv[n]["macs_frac"])
    rec = {}
    for cut in (float(c) for c in a.cuts.split(",")):
        names, cm = [], 0.0
        for n in order:
            if cm + inv[n]["macs_frac"] > cut + 1e-9:
                break
            names.append(n)
            cm += inv[n]["macs_frac"]
        names += [n for n in cand if inv[n]["block"] in ("down3", "mid") and n not in names]
        rec[f"gmac_{cut:.2f}"] = {"layers": names, "amax": "per_layer", "amax_per_layer": {n: pick(n)[0] for n in names}}
        print(f"gmac_{cut:.2f}: {len(names)} layers, {sum(inv[n]['macs_frac'] for n in names):.3f} of MACs, "
              f"summed single-layer error {sum(max(pick(n)[1], 0) for n in names):.2e}")
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
