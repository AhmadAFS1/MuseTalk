"""Pick a per-block INT8 variant combination under a UNet time budget (measured block times, fake-quant errors).

Each of down1/down2/up0/up1/up2 comes from one engine set: srcmix (FP16, error 0) or a recipe set. Time is the sum
of measured per-block CUDA-graph times (bench_stagewise_blocks.py JSON), plus the fixed blocks. Error is the sum
of per-block fake-quant MSE contributions (int8_layer_study.py recipes pb_<set>_<block>, mean of main_eval and
holdout); single-block contributions add up to within ~10% of the combined error (§3/§8 of the README).
usage: select_combo.py <bench.json> <budget_ms> [top]
"""
import itertools, json, sys
from pathlib import Path

D = Path(__file__).resolve().parent
SETS = {"fp16": "tensorrt_unet_stagewise_sm89_srcmix", "blkA8": "tensorrt_unet_stagewise_sm89_blkA_thr_8e-06",
        "g50": "tensorrt_unet_stagewise_sm89_gmac_0.50", "g55": "tensorrt_unet_stagewise_sm89_gmac_0.55",
        "g59": "tensorrt_unet_stagewise_sm89_gmac_0.59"}
VAR = ["down1", "down2", "up0", "up1", "up2"]
FIXED = ["down0rest", "down3", "mid", "up3", "tail"]


def main():
    bench = json.loads(Path(sys.argv[1]).read_text())["sets"]
    budget = float(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rec = json.loads((D / "int8_study/study.json").read_text())["recipes"]
    t = {}
    for k, name in SETS.items():
        root = next((r for r in bench if r.rstrip("/").endswith(name)), None)
        if root:
            t[k] = bench[root]["block_ms"]
    fixed = sum(t["fp16"][b] for b in FIXED if b not in ("down3", "mid")) + t["g59"]["down3"] + t["g59"]["mid"]
    err = {("fp16", b): 0.0 for b in VAR}
    for k in t:
        for b in VAR:
            r = rec.get(f"pb_{k}_{b}")
            if k != "fp16" and r:
                err[(k, b)] = 0.5 * (r["main_eval"]["mse_mean"] + r["holdout"]["mse_mean"])
    base = rec["pb_d3mid"]
    e_fixed = 0.5 * (base["main_eval"]["mse_mean"] + base["holdout"]["mse_mean"])
    rows = []
    for combo in itertools.product(*[[k for k in t if (k, b) in err and b in t[k]] for b in VAR]):
        ms = fixed + sum(t[k][b] for k, b in zip(combo, VAR))
        e = e_fixed + sum(err[(k, b)] for k, b in zip(combo, VAR))
        rows.append((e, ms, combo))
    ok = sorted(r for r in rows if r[1] <= budget)
    print(f"fixed blocks {fixed:.2f} ms, down3+mid error {e_fixed:.2e}; {len(ok)}/{len(rows)} combos within {budget} ms")
    print("| predicted mse | UNet ms | " + " | ".join(VAR) + " |\n|---|---|" + "---|" * len(VAR))
    for e, ms, c in ok[:top]:
        print(f"| {e:.2e} | {ms:.2f} | " + " | ".join(c) + " |")
    print("\nper-block (ms / mse):")
    for b in VAR:
        print(b, "  ".join(f"{k}={t[k][b]:.2f}/{err.get((k, b), float('nan')):.1e}" for k in t if b in t[k]))


if __name__ == "__main__":
    main()
