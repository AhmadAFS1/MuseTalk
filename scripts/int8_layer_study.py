"""Per-layer INT8 sensitivity and recipe study for the MuseTalk UNet (PyTorch fake-quant, no TensorRT).

Why: the r3 engine set (srcv1) quantized whole blocks (every Conv2d/Linear in down1-3/mid/up2/up3) with
modelopt INT8_DEFAULT_CFG max calibration, and fails the UNet gate by ~15x. This tool ranks individual
layers by the output error their INT8 weights (per-channel) + input activations (per-tensor) add, next to
their MACs, so a layer-selective INT8 recipe can be chosen before any engine is built.

Metric = the repo UNet gate (scripts/validate_unet_backend.py): per capture file, mae and max_abs of the
UNet output against the captured FP16 eager `pred_latents`; gate is max-over-files mae <= 0.01 and
max_abs <= 0.5. The shipping FP16 engines score ~0.0025 / ~0.4 on the same corpus.

Stages (--stage, repeatable; results merge into --out):
  inventory    candidate layers, category, block, MACs/frame, input shape
  baseline     all quantizers off (eager vs captured reference) and the srcv1 whole-block replica
  amax_mse     per layer, the input amax that minimises that layer's own output MSE (grid over max*alpha)
  sens         single-layer sensitivity (only that layer INT8), for max and mse amax
  recipes      evaluate named layer sets (--recipes JSON or built-in) on the full eval split
  errdist      where a recipe's largest latent errors sit, and the error quantiles
  wa           a recipe's error with only its weights INT8 vs only its inputs INT8 (which side dominates)
  lsq          learn each recipe layer's input amax end to end (straight-through rounding)
  recover      learn input amax + per-channel bias correction (+ optional weight amax) end to end; exportable
  tune         coordinate descent on each recipe layer's input amax
  export       write a builder recipe (build_unet_stagewise.py --int8-recipe), CPU only
Run under the GPU lease:
  scripts/box_guard.sh run --min-avail-gb 8 --label int8_study -- \
    /workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/int8_layer_study.py --stage inventory ...
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "calibration/unet_multi_avatar_20260928"
# conv_in/conv_out/time embedding: tiny or constant-folded; down0.resnets.0 is in the per-source prefix cache
EXCLUDE = ["conv_in", "conv_out", "time_embedding.*", "*time_emb_proj", "down_blocks.0.resnets.0.*"]
SRCV1_BLOCKS = ["down1", "down2", "down3", "mid", "up2", "up3"]


def block_of(name: str) -> str:
    m = re.match(r"(down|up)_blocks\.(\d+)\.", name)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    if name.startswith("mid_block."):
        return "mid"
    return "other"


def category_of(name: str) -> str:
    rules = [
        (r"\.resnets\.\d+\.conv1$", "res_conv1"), (r"\.resnets\.\d+\.conv2$", "res_conv2"),
        (r"\.resnets\.\d+\.conv_shortcut$", "res_shortcut"),
        (r"\.downsamplers\.\d+\.conv$", "downsample"), (r"\.upsamplers\.\d+\.conv$", "upsample"),
        (r"\.proj_in$", "proj_in"), (r"\.proj_out$", "proj_out"),
        (r"\.attn1\.to_[qkv]$", "attn1_qkv"), (r"\.attn1\.to_out\.0$", "attn1_out"),
        (r"\.attn2\.to_q$", "attn2_q"), (r"\.attn2\.to_[kv]$", "attn2_kv"), (r"\.attn2\.to_out\.0$", "attn2_out"),
        (r"\.ff\.net\.0\.proj$", "ff_in"), (r"\.ff\.net\.2$", "ff_out"),
    ]
    for pat, cat in rules:
        if re.search(pat, name):
            return cat
    return "other"


def excluded(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name + ".", p) for p in EXCLUDE)


def load_split(paths, device):
    out = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        out.append({"file": Path(p).name,
                    "avatar": (d.get("items") or [{}])[0].get("avatar_id", "?"),
                    "lat": d["latent_batch"].to(device, torch.float16),
                    "aud": d["audio_feature_batch"].to(device, torch.float16),
                    "t": d["timesteps"].to(device),
                    "ref": d["pred_latents"].to(device, torch.float16)})
    return out


def pick(files, n, offset=0):
    if n <= 0 or n >= len(files):
        return list(files)
    step = len(files) / n
    return [files[int(offset + i * step) % len(files)] for i in range(n)]


def splits(args):
    main = sorted(glob.glob(str(CORPUS / "unet_io_*.pt")))
    hold = sorted(glob.glob(str(CORPUS / "holdout/unet_io_*.pt")))
    calib = pick(main, args.calib_files, 0)
    rest = [f for f in main if f not in set(calib)]
    return {"calib": calib, "main_eval": pick(rest, args.main_eval_files, 1), "holdout": pick(hold, args.holdout_files, 0),
            "quick": pick(rest, args.quick_files // 2, 3) + pick(hold, args.quick_files - args.quick_files // 2, 5)}


@torch.inference_mode()
def forward(model, b):
    return model(b["lat"], b["t"], encoder_hidden_states=b["aud"]).sample


@torch.inference_mode()
def evaluate(model, batches) -> dict:
    rows = []
    for b in batches:
        out = forward(model, b).float()
        d = (out - b["ref"].float()).abs()
        rows.append({"file": b["file"], "avatar": b["avatar"], "mae": float(d.mean()), "max_abs": float(d.max()),
                     "mse": float((d * d).mean())})
    s = {"files": len(rows),
         "mae_mean": sum(r["mae"] for r in rows) / len(rows), "mae_max": max(r["mae"] for r in rows),
         "max_abs_max": max(r["max_abs"] for r in rows), "max_abs_mean": sum(r["max_abs"] for r in rows) / len(rows),
         "mse_mean": sum(r["mse"] for r in rows) / len(rows)}
    worst = max(rows, key=lambda r: r["mae"])
    s["worst_file"] = {"file": worst["file"], "avatar": worst["avatar"]}
    return s


def quant_layers(model):
    from modelopt.torch.quantization.nn import TensorQuantizer

    out = {}
    for name, m in model.named_modules():
        iq, wq = getattr(m, "input_quantizer", None), getattr(m, "weight_quantizer", None)
        if isinstance(iq, TensorQuantizer) and isinstance(wq, TensorQuantizer):
            out[name] = m
    return out


def set_enabled(layers: dict, names) -> None:
    names = set(names)
    for n, m in layers.items():
        on = n in names
        for q in (m.input_quantizer, m.weight_quantizer):
            q.enable() if on else q.disable()


def inventory(model, device) -> list[dict]:
    import torch.nn as nn

    macs, shapes = {}, {}
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and not excluded(name):
            def hook(mod, inp, out, _n=name):
                x = inp[0]
                shapes[_n] = list(x.shape[1:])
                if isinstance(mod, nn.Conv2d):
                    k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
                    macs[_n] = out[0].numel() * k
                else:
                    macs[_n] = (x[0].numel() // mod.in_features) * mod.in_features * mod.out_features
            hooks.append(m.register_forward_hook(hook))
    lat = torch.zeros(1, 8, 32, 32, device=device, dtype=torch.float16)
    aud = torch.zeros(1, 50, 384, device=device, dtype=torch.float16)
    with torch.inference_mode():
        model(lat, torch.tensor([0], device=device), encoder_hidden_states=aud)
    for h in hooks:
        h.remove()
    total = sum(macs.values())
    return [{"name": n, "block": block_of(n), "category": category_of(n), "macs": macs[n],
             "macs_frac": macs[n] / total, "input_shape": shapes[n]} for n in macs]


def quantize_all(model, calib, inv_names):
    import modelopt.torch.quantization as mtq

    cfg = {"quant_cfg": {"*weight_quantizer": {"num_bits": 8, "axis": 0},
                         "*input_quantizer": {"num_bits": 8, "axis": None},
                         "default": {"enable": False}},
           "algorithm": "max"}
    for p in EXCLUDE:
        cfg["quant_cfg"][p if p.endswith("*") else p + ".*"] = {"enable": False}

    def loop(m):
        for b in calib:
            forward(m, b)

    model = mtq.quantize(model, cfg, loop)
    wrapped = quant_layers(model)
    missing = sorted(set(inv_names) - set(wrapped))
    if missing:
        raise SystemExit(f"quantizer/inventory mismatch: missing={missing[:5]}")
    # excluded layers are wrapped too, but their quantizers stay disabled and they are never selected
    for n in set(wrapped) - set(inv_names):
        wrapped[n].input_quantizer.disable()
        wrapped[n].weight_quantizer.disable()
    return model, {n: wrapped[n] for n in inv_names}


@torch.inference_mode()
def mse_amax(model, layers, calib, alphas, n_files: int) -> dict:
    """Per layer: input amax (max * alpha) minimising the layer's own output MSE on calib inputs."""
    res = {}
    names = list(layers)
    set_enabled(layers, [])
    for i, name in enumerate(names):
        m = layers[name]
        caps = []
        h = m.register_forward_pre_hook(lambda mod, inp: caps.append(inp[0].detach().clone()))
        for b in calib[:n_files]:
            forward(model, b)
        h.remove()
        iq, wq = m.input_quantizer, m.weight_quantizer
        amax0 = float(iq.amax.float().max())
        # reference: fp16 weights and inputs (all quantizers off)
        ref_outs = [m(x).float() for x in caps]
        wq.enable()
        iq.enable()
        best = (None, 1.0)
        errs = {}
        for a in alphas:
            iq.amax = torch.tensor(amax0 * a, device=iq.amax.device, dtype=iq.amax.dtype)
            e = sum(float(((m(x).float() - r) ** 2).mean()) for x, r in zip(caps, ref_outs)) / len(caps)
            errs[a] = e
            if best[0] is None or e < best[0]:
                best = (e, a)
        iq.amax = torch.tensor(amax0, device=iq.amax.device, dtype=iq.amax.dtype)
        iq.disable()
        wq.disable()
        res[name] = {"amax_max": amax0, "alpha": best[1], "amax_mse": amax0 * best[1],
                     "out_mse_at_max": errs[1.0] if 1.0 in errs else None, "out_mse_best": best[0]}
        del caps, ref_outs
        if (i + 1) % 25 == 0:
            print(f"  amax_mse {i + 1}/{len(names)}", flush=True)
    return res


def apply_amax(layers, amax_table: dict | None, key: str):
    for n, m in layers.items():
        if amax_table and n in amax_table:
            v = amax_table[n][key]
            m.input_quantizer.amax = torch.tensor(v, device=m.input_quantizer.amax.device,
                                                  dtype=m.input_quantizer.amax.dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", action="append", required=True,
                    choices=["inventory", "baseline", "amax_mse", "sens", "recipes", "errdist", "wa", "lsq", "recover", "tune", "export"])
    ap.add_argument("--smooth-ff", type=float, default=0.0,
                    help="SmoothQuant alpha for ff.net.2 inputs, folded into GEGLU; not bit-exact, see scripts/unet_int8_smooth.py (0 = off)")
    ap.add_argument("--errdist-recipes", default="fp16", help="--stage errdist/wa: comma list of recipe names (fp16 only for errdist)")
    ap.add_argument("--lsq-recipe", default="", help="--stage lsq: recipe name to learn amax for")
    ap.add_argument("--lsq-epochs", type=int, default=4)
    ap.add_argument("--lsq-lr", type=float, default=0.01)
    ap.add_argument("--recover-recipe", default="", help="--stage recover: recipe name (result['recipes'])")
    ap.add_argument("--recover-params", default="amax,bias", help="comma list of amax, bias, wscale, lora")
    ap.add_argument("--recover-epochs", type=int, default=12)
    ap.add_argument("--recover-lr-amax", type=float, default=0.01)
    ap.add_argument("--recover-lr-bias", type=float, default=2e-4)
    ap.add_argument("--recover-lr-wscale", type=float, default=2e-3)
    ap.add_argument("--recover-smooth-alpha", type=float, default=0.0, help="per-input-channel smoothing alpha (0 = off)")
    ap.add_argument("--recover-smooth-files", type=int, default=8, help="calib files for the smoothing channel absmax")
    ap.add_argument("--recover-lr-lora", type=float, default=2e-5)
    ap.add_argument("--recover-lora-rank", type=int, default=16)
    ap.add_argument("--recover-cosine", type=int, default=1, help="cosine LR decay over the run (1) or constant (0)")
    ap.add_argument("--recover-tag", default="", help="suffix for the recovered recipe name")
    ap.add_argument("--tune-recipe", default="", help="--stage tune: recipe name to tune (result['recipes'])")
    ap.add_argument("--tune-alphas", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--tune-sweeps", type=int, default=2)
    ap.add_argument("--export-recipe", default="", help="--stage export: recipe name in the study JSON")
    ap.add_argument("--export-out", default="", help="--stage export: builder recipe JSON path")
    ap.add_argument("--out", default=str(ROOT / "docs/fps_comparisons/4070s_400fps_20260928/int8_study/study.json"))
    ap.add_argument("--calib-files", type=int, default=32, help="main-split bs8 files used for calibration")
    ap.add_argument("--main-eval-files", type=int, default=88)
    ap.add_argument("--holdout-files", type=int, default=96)
    ap.add_argument("--quick-files", type=int, default=12, help="files for single-layer sensitivity")
    ap.add_argument("--mse-files", type=int, default=3, help="calib files per layer for the amax MSE search")
    ap.add_argument("--sens-blocks", default="down0,down1,down2,down3,mid,up0,up1,up2,up3")
    ap.add_argument("--recipes", default="", help="JSON file: {name: {layers: [...] | select: {...}, amax: max|mse}}")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save():
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(out_path)

    if args.stage == ["export"]:
        # CPU only: write a builder recipe (scripts/build_unet_stagewise.py --int8-recipe) from a studied recipe
        entry = result["recipes"][args.export_recipe]
        rec = {"schema": "musetalk_unet_int8_recipe_v1", "source": str(out_path), "recipe": args.export_recipe,
               "summary": {k: entry[k] for k in ("layers", "macs_frac", "amax", "gate_pass") if k in entry},
               "eval": {s: entry[s] for s in ("main_eval", "holdout") if s in entry},
               "layers": {n: {"input_amax": v} for n, v in entry["layer_amax"].items()}}
        if entry.get("tensors"):
            # learned bias corrections / weight amax (--stage recover), applied by the builder
            rec["tensors"] = entry["tensors"]
        if "smooth_ff" in result:
            sm = result["smooth_ff"]
            rec["smooth_ff"] = {"alpha": sm["alpha"],
                                "exponents": {n: e for n, e in sm["exponents"].items() if n in entry["layer_amax"]}}
        Path(args.export_out).write_text(json.dumps(rec, indent=1))
        print(f"exported {len(rec['layers'])} layers -> {args.export_out}")
        return 0
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    from scripts.build_unet_stagewise import load_eager_unet

    t0 = time.time()
    model = load_eager_unet(device)
    sp = splits(args)
    result["splits"] = {k: [Path(f).name for f in v] for k, v in sp.items()}
    data = {k: load_split(v, device) for k, v in sp.items()}
    print(f"loaded model + corpus in {time.time() - t0:.1f}s: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()),
          flush=True)
    if args.smooth_ff > 0:
        from scripts import unet_int8_smooth as sq

        before = forward(model, data["quick"][0]).clone()
        exps = sq.smooth_exponents(model, sq.collect_ff_out_absmax(model, data["calib"], forward), args.smooth_ff)
        sq.apply_smoothing(model, exps)
        after = forward(model, data["quick"][0])
        diff = float((after.float() - before.float()).abs().max())
        result["smooth_ff"] = {"alpha": args.smooth_ff, "fp16_output_max_abs_change": diff, "exponents": exps}
        print(f"smooth_ff alpha={args.smooth_ff}: {len(exps)} layers, FP16 output max|change|={diff:g}", flush=True)

    inv = inventory(model, device)
    if "inventory" in args.stage or "inventory" not in result:
        result["inventory"] = inv
        by = {}
        for r in inv:
            by.setdefault(r["category"], 0.0)
            by[r["category"]] += r["macs_frac"]
        result["inventory_macs_by_category"] = dict(sorted(by.items(), key=lambda kv: -kv[1]))
        byb = {}
        for r in inv:
            byb.setdefault(r["block"], 0.0)
            byb[r["block"]] += r["macs_frac"]
        result["inventory_macs_by_block"] = byb
        result["inventory_total_gmac_per_frame"] = sum(r["macs"] for r in inv) / 1e9
        save()
        print(json.dumps({"by_category": result["inventory_macs_by_category"], "by_block": byb,
                          "gmac": result["inventory_total_gmac_per_frame"]}, indent=1), flush=True)
    inv_names = [r["name"] for r in inv]
    if set(args.stage) == {"inventory"}:
        return 0

    fp_quick = evaluate(model, data["quick"])
    t1 = time.time()
    model, layers = quantize_all(model, data["calib"], inv_names)
    print(f"quantized+max-calibrated {len(layers)} layers in {time.time() - t1:.1f}s", flush=True)
    set_enabled(layers, [])
    base_quick = evaluate(model, data["quick"])
    if abs(base_quick["mae_mean"] - fp_quick["mae_mean"]) > 1e-6:
        print(f"WARNING: disabled quantizers change output: {fp_quick} vs {base_quick}", flush=True)
    inv_by = {r["name"]: r for r in inv}

    if "baseline" in args.stage:
        rb = {}
        set_enabled(layers, [])
        for split in ("main_eval", "holdout"):
            rb[f"fp16_eager_{split}"] = evaluate(model, data[split])
        srcv1 = [n for n in inv_names if inv_by[n]["block"] in SRCV1_BLOCKS]
        set_enabled(layers, srcv1)
        for split in ("main_eval", "holdout"):
            rb[f"srcv1_replica_max_{split}"] = evaluate(model, data[split])
        rb["srcv1_replica_layers"] = len(srcv1)
        rb["srcv1_replica_macs_frac"] = sum(inv_by[n]["macs_frac"] for n in srcv1)
        set_enabled(layers, [])
        result["baseline"] = rb
        save()
        print(json.dumps(rb, indent=1), flush=True)

    if "amax_mse" in args.stage:
        t2 = time.time()
        alphas = [round(0.2 + 0.05 * i, 2) for i in range(17)]
        result["amax_mse"] = mse_amax(model, layers, data["calib"], alphas, args.mse_files)
        result["amax_mse_meta"] = {"alphas": alphas, "files": args.mse_files, "seconds": time.time() - t2}
        save()
        print(f"amax_mse done in {time.time() - t2:.1f}s", flush=True)

    amax_max = {n: {"amax_max": float(m.input_quantizer.amax.float().max())} for n, m in layers.items()}
    if "sens" in args.stage:
        blocks = set(args.sens_blocks.split(","))
        cand = [n for n in inv_names if inv_by[n]["block"] in blocks]
        sens = result.setdefault("sens", {})
        modes = ["max"] + (["mse"] if "amax_mse" in result else [])
        t3 = time.time()
        for mode in modes:
            apply_amax(layers, amax_max if mode == "max" else result["amax_mse"], "amax_max" if mode == "max" else "amax_mse")
            sm = sens.setdefault(mode, {})
            for i, n in enumerate(cand):
                set_enabled(layers, [n])
                r = evaluate(model, data["quick"])
                sm[n] = {"mae_mean": r["mae_mean"] - base_quick["mae_mean"], "mae_max": r["mae_max"],
                         "max_abs_max": r["max_abs_max"], "mse_mean": r["mse_mean"] - base_quick["mse_mean"]}
                if (i + 1) % 40 == 0:
                    print(f"  sens[{mode}] {i + 1}/{len(cand)} ({time.time() - t3:.0f}s)", flush=True)
                    save()
            set_enabled(layers, [])
        apply_amax(layers, amax_max, "amax_max")
        result["sens_meta"] = {"quick_base": base_quick, "fp_quick": fp_quick, "seconds": time.time() - t3}
        save()

    if "recipes" in args.stage:
        recipes = json.loads(Path(args.recipes).read_text()) if args.recipes else {}
        res = result.setdefault("recipes", {})
        for rname, spec in recipes.items():
            names = spec.get("layers")
            if names is None:
                sel = spec.get("select", {})
                names = [n for n in inv_names
                         if (not sel.get("blocks") or inv_by[n]["block"] in sel["blocks"])
                         and (not sel.get("categories") or inv_by[n]["category"] in sel["categories"])
                         and not any(fnmatch.fnmatch(n, p) for p in sel.get("exclude", []))]
            mode = spec.get("amax", "max")
            if mode != "max" and "amax_mse" not in result:
                raise SystemExit("recipe wants mse amax: run --stage amax_mse first")
            apply_amax(layers, amax_max, "amax_max")
            if mode == "mse":
                apply_amax(layers, result["amax_mse"], "amax_mse")
            elif mode == "per_layer":
                # {name: "max"|"mse"}: e.g. whichever gave the lower single-layer sensitivity
                per = spec["amax_per_layer"]
                apply_amax(layers, {n: result["amax_mse"][n] for n in names if per.get(n) == "mse"}, "amax_mse")
            set_enabled(layers, names)
            entry = {"layers": len(names), "macs_frac": sum(inv_by[n]["macs_frac"] for n in names), "amax": mode,
                     "layer_amax": {n: float(layers[n].input_quantizer.amax.float().max()) for n in names}}
            for split in spec.get("splits", ["main_eval", "holdout"]):
                entry[split] = evaluate(model, data[split])
            entry["gate_pass"] = all(entry[s]["mae_max"] <= 0.01 and entry[s]["max_abs_max"] <= 0.5
                                     for s in spec.get("splits", ["main_eval", "holdout"]))
            res[rname] = entry
            set_enabled(layers, [])
            save()
            print(f"recipe {rname}: layers={entry['layers']} macs={entry['macs_frac']:.3f} amax={mode} "
                  + " ".join(f"{s}: mae_max={entry[s]['mae_max']:.5f} max_abs={entry[s]['max_abs_max']:.3f}"
                             for s in spec.get("splits", ["main_eval", "holdout"])), flush=True)
        apply_amax(layers, amax_max, "amax_max")

    if "errdist" in args.stage:
        # Where the largest latent errors sit, and how heavy the tail is (quantiles over every element).
        out = result.setdefault("errdist", {})
        for rname in args.errdist_recipes.split(","):
            if rname == "fp16":
                names, amaxes = [], {}
            else:
                names = list(result["recipes"][rname]["layer_amax"])
                amaxes = result["recipes"][rname]["layer_amax"]
            apply_amax(layers, amax_max, "amax_max")
            for n in names:
                q = layers[n].input_quantizer
                q.amax = torch.tensor(amaxes[n], device=q.amax.device, dtype=q.amax.dtype)
            set_enabled(layers, names)
            entry = {}
            for split in ("main_eval", "holdout"):
                errs, tops = [], []
                with torch.inference_mode():
                    for b in data[split]:
                        e = (forward(model, b).float() - b["ref"].float()).abs()
                        errs.append(e.flatten().cpu())
                        v, idx = e.flatten().topk(3)
                        for val, ix in zip(v.tolist(), idx.tolist()):
                            s_, c_, y_, x_ = torch.unravel_index(torch.tensor(ix), e.shape)
                            tops.append({"file": b["file"], "avatar": b["avatar"], "err": val, "sample": int(s_),
                                         "ch": int(c_), "y": int(y_), "x": int(x_),
                                         "ref": float(b["ref"][s_, c_, y_, x_])})
                allv = torch.cat(errs)
                qs = torch.quantile(allv[torch.randperm(allv.numel())[:2_000_000]],
                                    torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999]))
                tops.sort(key=lambda r: -r["err"])
                rows = [r["y"] for r in tops[:30]]
                entry[split] = {"quantiles": dict(zip(["p50", "p90", "p99", "p99.9", "p99.99"], qs.tolist())),
                                "max": float(allv.max()), "n_gt_0.5": int((allv > 0.5).sum()),
                                "n_gt_0.25": int((allv > 0.25).sum()), "n": allv.numel(),
                                "top30_rows_y": rows, "top10": tops[:10]}
            out[rname] = entry
            set_enabled(layers, [])
            save()
            print(f"errdist {rname}: " + " | ".join(
                f"{s} q={ {k: round(v, 4) for k, v in entry[s]['quantiles'].items()} } max={entry[s]['max']:.3f} "
                f">0.5:{entry[s]['n_gt_0.5']} >0.25:{entry[s]['n_gt_0.25']}" for s in entry), flush=True)
        apply_amax(layers, amax_max, "amax_max")

    if "wa" in args.stage:
        # Which side of a recipe's INT8 error dominates: weights only (per-channel) vs activations only (per-tensor).
        out = result.setdefault("wa_split", {})
        for rname in args.errdist_recipes.split(","):
            src = result["recipes"][rname]
            names = list(src["layer_amax"])
            apply_amax(layers, amax_max, "amax_max")
            for n in names:
                q = layers[n].input_quantizer
                q.amax = torch.tensor(src["layer_amax"][n], device=q.amax.device, dtype=q.amax.dtype)
            entry = {}
            for side in ("weights_only", "acts_only", "both"):
                set_enabled(layers, names)
                for n in names:
                    if side == "weights_only":
                        layers[n].input_quantizer.disable()
                    elif side == "acts_only":
                        layers[n].weight_quantizer.disable()
                entry[side] = {s: evaluate(model, data[s]) for s in ("main_eval", "holdout")}
            out[rname] = entry
            set_enabled(layers, [])
            save()
            print(f"wa {rname}: " + " | ".join(
                f"{side}: " + " ".join(f"{s} mae_max={v[s]['mae_max']:.5f} mse={v[s]['mse_mean']:.3e} max_abs={v[s]['max_abs_max']:.3f}"
                                      for s in v) for side, v in entry.items()), flush=True)
        apply_amax(layers, amax_max, "amax_max")

    if "lsq" in args.stage:
        # Learned activation step size (LSQ) for every recipe layer, jointly, on the END-TO-END output MSE
        # against the captured FP16 reference over the calibration files. Weights stay per-channel max INT8
        # (modelopt, no grad); only one log-amax scalar per layer is learned. Straight-through rounding:
        # x_q = s * clamp(round_ste(x / s), -128, 127), s = amax / 127, which is TensorRT's symmetric INT8 Q/DQ.
        src = result["recipes"][args.lsq_recipe]
        names = list(src["layer_amax"])
        init = src["layer_amax"]
        set_enabled(layers, names)
        model.requires_grad_(False)  # only the per-layer log-amax scalars below are trained
        logs, hooks = {}, []
        for n in names:
            m = layers[n]
            m.input_quantizer.disable()  # replaced by the learnable quantizer below (weights stay quantized)
            p = torch.nn.Parameter(torch.tensor(float(init[n]), device=device).log())
            logs[n] = p

            def pre(mod, inp, _p=p):
                x = inp[0]
                s = (_p.exp() / 127.0).to(x.dtype)
                v = x / s
                q = v + (v.round() - v).detach()
                return (q.clamp(-128, 127) * s,) + tuple(inp[1:])
            hooks.append(m.register_forward_pre_hook(pre))
        opt = torch.optim.Adam(list(logs.values()), lr=args.lsq_lr)

        def evaluate_q(batches):
            with torch.inference_mode():
                return evaluate(model, batches)
        hist = [{"epoch": 0, "quick": evaluate_q(data["quick"])["mse_mean"]}]
        t5 = time.time()
        for ep in range(1, args.lsq_epochs + 1):
            tot = 0.0
            for b in data["calib"]:
                opt.zero_grad(set_to_none=True)
                out = model(b["lat"], b["t"], encoder_hidden_states=b["aud"]).sample.float()
                loss = ((out - b["ref"].float()) ** 2).mean()
                (loss * 1e4).backward()
                opt.step()
                sched.step()
                tot += float(loss)
            hist.append({"epoch": ep, "train_mse": tot / len(data["calib"]), "quick": evaluate_q(data["quick"])["mse_mean"],
                         "seconds": time.time() - t5})
            print(f"  lsq epoch {ep}: train mse {hist[-1]['train_mse']:.3e} quick mse {hist[-1]['quick']:.3e} "
                  f"({time.time() - t5:.0f}s)", flush=True)
        learned = {n: float(logs[n].detach().exp()) for n in names}
        for h in hooks:
            h.remove()
        # evaluate through the regular (modelopt) quantizers with the learned amax, i.e. exactly what is exported
        for n in names:
            q = layers[n].input_quantizer
            q.amax = torch.tensor(learned[n], device=q.amax.device, dtype=q.amax.dtype)
        set_enabled(layers, names)
        entry = {"layers": len(names), "macs_frac": src["macs_frac"], "amax": "lsq", "lsq_from": args.lsq_recipe,
                 "lsq": {"epochs": args.lsq_epochs, "lr": args.lsq_lr, "history": hist,
                         "amax_ratio_to_init": {n: learned[n] / init[n] for n in names}},
                 "layer_amax": learned}
        for split in ("main_eval", "holdout"):
            entry[split] = evaluate(model, data[split])
        entry["gate_pass"] = all(entry[s]["mae_max"] <= 0.01 and entry[s]["max_abs_max"] <= 0.5
                                 for s in ("main_eval", "holdout"))
        result.setdefault("recipes", {})[args.lsq_recipe + "_lsq"] = entry
        set_enabled(layers, [])
        apply_amax(layers, amax_max, "amax_max")
        save()
        print(f"recipe {args.lsq_recipe}_lsq: " + " ".join(
            f"{s}: mae_max={entry[s]['mae_max']:.5f} max_abs={entry[s]['max_abs_max']:.3f}" for s in ("main_eval", "holdout")),
            flush=True)

    if "recover" in args.stage:
        # End-to-end INT8 quality recovery for a recipe, trained on the calib files against the captured FP16
        # reference output (holdout and main_eval stay unseen). Learned, all exportable to TensorRT Q/DQ at the
        # same speed: each layer's input amax (LSQ, one scalar), optionally a per-output-channel correction added
        # to the layer's bias (removes the systematic part of the INT8 error) and optionally each output
        # channel's weight amax. modelopt's quantizers are switched off and replaced by an explicit fake quant
        # of the same form TensorRT runs: s = amax / 127, q = s * clamp(round(x / s), -128, 127), per-tensor
        # inputs, per-output-channel weights, straight-through rounding.
        import torch.nn.functional as F

        src = result["recipes"][args.recover_recipe]
        names = list(src["layer_amax"])
        init = src["layer_amax"]
        set_enabled(layers, [])
        model.requires_grad_(False)
        # modelopt's diffusers Attention wrapper routes SDPA through a forward-only export op (FP8SDPA) even with
        # its bmm quantizers off; off for training (forward values are unchanged: the bmm quantizers are disabled)
        from modelopt.torch.quantization.plugins import diffusers as mo_diffusers

        mo_saved_functionals = mo_diffusers._QuantAttention._functionals_to_replace
        mo_diffusers._QuantAttention._functionals_to_replace = []
        state, fwd_saved = {}, {}
        use_bias = "bias" in args.recover_params.split(",")
        use_wscale = "wscale" in args.recover_params.split(",")
        use_lora = "lora" in args.recover_params.split(",")

        def ste_q(v):
            return v + (v.round() - v).detach()

        # Optional SmoothQuant-style per-input-channel smoothing (exact in real arithmetic): the layer computes
        # W' x' with x' = x / s and W' = W * s (s along the input channels), s_c = a_c^alpha / w_c^(1-alpha), where a_c
        # is the channel's calib absmax and w_c the absmax of the weights reading it. For TensorRT the 1/s Mul sits
        # in front of the input Q/DQ (fusable into the preceding pointwise kernel) or folds into a LayerNorm/GEGLU.
        smooth = {}
        if args.recover_smooth_alpha > 0:
            chan_max, hooks = {}, []
            for n in names:
                def cap(mod, inp, _n=n):
                    x = inp[0]
                    a = x.abs().amax(dim=(0, 2, 3)) if x.dim() == 4 else x.abs().reshape(-1, x.shape[-1]).amax(0)
                    chan_max[_n] = torch.maximum(chan_max[_n], a.float()) if _n in chan_max else a.float()
                hooks.append(layers[n].register_forward_pre_hook(cap))
            for b in data["calib"][: args.recover_smooth_files]:
                forward(model, b)
            for h in hooks:
                h.remove()
            al = args.recover_smooth_alpha
            for n in names:
                w = layers[n].weight.detach().float()
                w_in = w.abs().amax(dim=0)
                w_in = w_in.amax(dim=(1, 2)) if w_in.dim() == 3 else w_in
                smooth[n] = (chan_max[n].clamp_min(1e-5) ** al / w_in.clamp_min(1e-5) ** (1 - al)).clamp(1e-3, 1e3)
        for n in names:
            m = layers[n]
            conv = isinstance(m, torch.nn.Conv2d)
            w = m.weight.detach()
            s_in = smooth.get(n)
            if s_in is not None:
                w = (w.float() * (s_in.view(1, -1, 1, 1) if conv else s_in.view(1, -1))).to(m.weight.dtype)
            red = tuple(range(1, w.dim()))
            w_amax0 = w.float().abs().amax(dim=red, keepdim=True).clamp_min(1e-8)
            a0 = float(init[n])
            if s_in is not None:
                # keep the layer's clip ratio (recipe amax / max-calibrated amax) on the smoothed input
                a0 = a0 / amax_max[n]["amax_max"] * float((chan_max[n] / s_in).max())
            la = torch.nn.Parameter(torch.tensor(a0, device=device).log())
            bd = torch.nn.Parameter(torch.zeros(w.shape[0], device=device)) if use_bias and m.bias is not None else None
            lw = torch.nn.Parameter(w_amax0.log().clone()) if use_wscale else None
            inv = None if s_in is None else (1.0 / s_in).to(m.weight.dtype).view((1, -1, 1, 1) if conv else (-1,))
            lora = None
            if use_lora:
                # quantization-aware LoRA: W' = W + B @ A (merged into the FP16 weights before export), B = 0 at start
                fan_in = w[0].numel()
                g = torch.Generator(device="cpu").manual_seed(len(state))
                lora = {"A": torch.nn.Parameter((torch.randn(args.recover_lora_rank, fan_in, generator=g)
                                                 / args.recover_lora_rank ** 0.5).to(device)),
                        "B": torch.nn.Parameter(torch.zeros(w.shape[0], args.recover_lora_rank, device=device))}
            state[n] = {"la": la, "bd": bd, "lw": lw, "w_amax0": w_amax0, "w": w, "inv_s": inv, "lora": lora}
            fwd_saved[n] = m.forward

            def fwd(x, _m=m, _st=state[n], _conv=conv):
                if _st["inv_s"] is not None:
                    x = x * _st["inv_s"]
                s = (_st["la"].exp() / 127.0).to(x.dtype)
                xq = ste_q(x / s).clamp(-128, 127) * s
                w = _st["w"]
                if _st["lora"] is not None:
                    w = (w.float() + (_st["lora"]["B"] @ _st["lora"]["A"]).view(w.shape)).to(w.dtype)
                    w_amax = w.detach().float().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-8)
                else:
                    w_amax = _st["w_amax0"]
                if _st["lw"] is not None:
                    w_amax = _st["lw"].exp()
                sw = (w_amax / 127.0).to(w.dtype)
                wq = ste_q(w / sw).clamp(-128, 127) * sw
                b = _m.bias
                if _st["bd"] is not None:
                    b = b + _st["bd"].to(b.dtype)
                if _conv:
                    return F.conv2d(xq, wq, b, _m.stride, _m.padding, _m.dilation, _m.groups)
                return F.linear(xq, wq, b)
            m.forward = fwd
        groups = [{"params": [st["la"] for st in state.values()], "lr": args.recover_lr_amax}]
        if use_bias:
            groups.append({"params": [st["bd"] for st in state.values() if st["bd"] is not None], "lr": args.recover_lr_bias})
        if use_wscale:
            groups.append({"params": [st["lw"] for st in state.values()], "lr": args.recover_lr_wscale})
        if use_lora:
            groups.append({"params": [p_ for st in state.values() for p_ in st["lora"].values()], "lr": args.recover_lr_lora})
        opt = torch.optim.Adam(groups)
        steps_total = max(1, args.recover_epochs * len(data["calib"]))
        # cosine decay to 5% (the INT8 rounding makes the loss noisy; a decaying step stops the random walk at the end)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: 0.05 + 0.95 * 0.5 * (1 + __import__("math").cos(__import__("math").pi * min(k, steps_total) / steps_total))
            if args.recover_cosine else 1.0)
        tag = (args.recover_recipe + "_rec_" + args.recover_params.replace(",", "+")
               + (f"_sq{args.recover_smooth_alpha:g}" if smooth else "") + (args.recover_tag and "_" + args.recover_tag))
        hist = [{"epoch": 0, "quick_mse": evaluate(model, data["quick"])["mse_mean"]}]
        print(f"recover {tag}: epoch 0 quick mse {hist[0]['quick_mse']:.3e}", flush=True)
        t6 = time.time()
        for ep in range(1, args.recover_epochs + 1):
            tot = 0.0
            for b in data["calib"]:
                opt.zero_grad(set_to_none=True)
                out = model(b["lat"], b["t"], encoder_hidden_states=b["aud"]).sample.float()
                loss = ((out - b["ref"].float()) ** 2).mean()
                (loss * 1e4).backward()
                opt.step()
                sched.step()
                tot += float(loss)
            hist.append({"epoch": ep, "train_mse": tot / len(data["calib"]), "quick_mse": evaluate(model, data["quick"])["mse_mean"],
                         "seconds": time.time() - t6})
            print(f"  recover epoch {ep}: train mse {hist[-1]['train_mse']:.3e} quick mse {hist[-1]['quick_mse']:.3e} "
                  f"({time.time() - t6:.0f}s)", flush=True)
        entry = {"layers": len(names), "macs_frac": src["macs_frac"], "amax": "recover", "recover_from": args.recover_recipe,
                 "recover": {"params": args.recover_params, "epochs": args.recover_epochs, "history": hist,
                             "lr": {"amax": args.recover_lr_amax, "bias": args.recover_lr_bias, "wscale": args.recover_lr_wscale,
                                    "lora": args.recover_lr_lora}, "lora_rank": args.recover_lora_rank if use_lora else 0,
                             "cosine": bool(args.recover_cosine)},
                 "layer_amax": {n: float(state[n]["la"].detach().exp()) for n in names}}
        for split in ("main_eval", "holdout"):
            entry[split] = evaluate(model, data[split])
        entry["gate_pass"] = all(entry[s]["mae_max"] <= 0.01 and entry[s]["max_abs_max"] <= 0.5 for s in ("main_eval", "holdout"))
        tensors = {"bias_delta": {n: st["bd"].detach().cpu() for n, st in state.items() if st["bd"] is not None},
                   "weight_amax": {n: st["lw"].detach().exp().cpu() for n, st in state.items() if st["lw"] is not None},
                   "smooth": {n: v.cpu() for n, v in smooth.items()},
                   "lora": {n: {k: v.detach().cpu() for k, v in st["lora"].items()} for n, st in state.items() if st["lora"] is not None}}
        if smooth:
            entry["recover"]["smooth"] = {"alpha": args.recover_smooth_alpha, "files": args.recover_smooth_files}
        if use_lora:
            rel = [float((st["lora"]["B"] @ st["lora"]["A"]).norm() / st["w"].float().norm()) for st in state.values()]
            entry["lora_rel_delta"] = {"max": max(rel), "mean": sum(rel) / len(rel)}
        if any(tensors.values()):
            tpath = out_path.parent / f"recover_{tag}.pt"
            torch.save(tensors, tpath)
            entry["tensors"] = str(tpath)
            entry["bias_delta_absmax"] = max((float(v.abs().max()) for v in tensors["bias_delta"].values()), default=0.0)
        for n in names:
            layers[n].forward = fwd_saved[n]
        mo_diffusers._QuantAttention._functionals_to_replace = mo_saved_functionals
        result.setdefault("recipes", {})[tag] = entry
        save()
        print(f"recipe {tag}: " + " ".join(f"{s}: mae_max={entry[s]['mae_max']:.5f} mse={entry[s]['mse_mean']:.3e} "
                                             f"max_abs={entry[s]['max_abs_max']:.3f}" for s in ("main_eval", "holdout")), flush=True)

    if "tune" in args.stage:
        # Coordinate descent on each recipe layer's input amax (factor of its max-calibrated amax), scored by
        # the END-TO-END quick-set output mse with every other recipe layer quantized at its current amax.
        src = result["recipes"][args.tune_recipe]
        names = list(src["layer_amax"])
        cur = dict(src["layer_amax"])
        grid = [float(a) for a in args.tune_alphas.split(",")]
        set_enabled(layers, names)

        def put(n, v):
            q = layers[n].input_quantizer
            q.amax = torch.tensor(v, device=q.amax.device, dtype=q.amax.dtype)

        for n in names:
            put(n, cur[n])
        score = evaluate(model, data["quick"])["mse_mean"]
        hist = [{"sweep": 0, "mse_mean": score}]
        t4 = time.time()
        for sweep in range(1, args.tune_sweeps + 1):
            for n in sorted(names, key=lambda k: -inv_by[k]["macs_frac"]):
                base_amax = amax_max[n]["amax_max"]
                best = (score, cur[n])
                for a in grid:
                    v = base_amax * a
                    if abs(v - cur[n]) < 1e-9 * max(1.0, abs(v)):
                        continue
                    put(n, v)
                    s = evaluate(model, data["quick"])["mse_mean"]
                    if s < best[0]:
                        best = (s, v)
                cur[n] = best[1]
                score = best[0]
                put(n, cur[n])
            hist.append({"sweep": sweep, "mse_mean": score, "seconds": time.time() - t4})
            print(f"  tune sweep {sweep}: quick mse_mean {score:.3e} ({time.time() - t4:.0f}s)", flush=True)
        entry = {"layers": len(names), "macs_frac": src["macs_frac"], "amax": "tuned", "tuned_from": args.tune_recipe,
                 "tune_history": hist, "layer_amax": cur}
        for split in ("main_eval", "holdout"):
            entry[split] = evaluate(model, data[split])
        entry["gate_pass"] = all(entry[s]["mae_max"] <= 0.01 and entry[s]["max_abs_max"] <= 0.5
                                 for s in ("main_eval", "holdout"))
        result.setdefault("recipes", {})[args.tune_recipe + "_tuned"] = entry
        set_enabled(layers, [])
        apply_amax(layers, amax_max, "amax_max")
        save()
        print(f"recipe {args.tune_recipe}_tuned: " + " ".join(
            f"{s}: mae_max={entry[s]['mae_max']:.5f} max_abs={entry[s]['max_abs_max']:.3f}" for s in ("main_eval", "holdout")),
            flush=True)
    print(f"done in {time.time() - t0:.1f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
