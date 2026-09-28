"""Pick the TRT TAESD build recipe (plan item 2.1): speed + accuracy per variant.

In-memory builds only (nothing persisted). Variants: builder opt level 3 / 5 x
weakly typed (FP16 flag) / strongly typed, at bs8; bs16 for the fastest recipe.
Accuracy is the uint8 LSB difference (repo post applied to both sides) against
today's compiled TAESD (live config: max-autotune, default inductor cache,
cudnn.benchmark on) and against eager, on real post-UNet latents from the
multi-avatar corpus. Also: exhaustive check of the fused uint8 BGR post engine
over every non-NaN fp16 bit pattern.

Run: scripts/box_guard.sh run --min-avail-gb 6 -- \
  /workspace/.venvs/musetalk_trt_stagewise/bin/python <this file>
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import vae_fast_decoder as vfd  # noqa: E402

DEV = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def smi():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def corpus_latents(n_batches: int) -> torch.Tensor:
    files = sorted((ROOT / "calibration/unet_multi_avatar_20260928").glob("unet_io_*.pt"))
    step = max(1, len(files) // n_batches)
    zs = [torch.load(f, map_location="cpu", weights_only=False)["pred_latents"].to(torch.float16)
          for f in files[::step][:n_batches]]
    return torch.cat(zs).contiguous()


def bench(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
            "min_ms": float(t.min()), "n": iters, "smi_after": smi()}


def lsb(a_u8: torch.Tensor, b_u8: torch.Tensor) -> dict:
    d = (a_u8.to(torch.int16) - b_u8.to(torch.int16)).abs()
    rows = d[:, 104:]  # NHWC: rows >= 104 (the blend / tracker-used rows)
    return {"max": int(d.max()), "mean": float(d.float().mean()), "frac_nonzero": float((d > 0).float().mean()),
            "rows104_max": int(rows.max()), "rows104_mean": float(rows.float().mean()),
            "hist": {int(k): int(v) for k, v in zip(*torch.unique(d, return_counts=True))}}


def main():
    res = {"smi_start": smi(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Z = corpus_latents(64).to(DEV)  # 512 latents, 64 files spread over the 11 avatars
    res["latents"] = int(Z.shape[0])

    compiled = vfd.TaesdVaeDecodeBackend.load(device=DEV, runtime_dtype=torch.float16)
    compiled.warmup([8])
    model = compiled.model
    eager = compiled._raw_decode

    with torch.inference_mode():
        ref_c, ref_e = [], []
        for i in range(0, Z.shape[0], 8):
            ref_c.append(vfd.repo_fast_postprocess_gpu(compiled.decode(Z[i:i + 8], 1.0, torch.float16)))
            ref_e.append(vfd.repo_fast_postprocess_gpu(eager(Z[i:i + 8])))
        ref_c = torch.cat(ref_c)
        ref_e = torch.cat(ref_e)
        res["compiled_vs_eager_lsb"] = lsb(ref_c, ref_e)
        zc = Z[:8].contiguous()
        res["compiled_bs8_ms"] = bench(lambda: compiled.decode(zc, 1.0, torch.float16))
        res["eager_bs8_ms"] = bench(lambda: eager(zc))
    print("compiled vs eager", res["compiled_vs_eager_lsb"], flush=True)

    # post engine: exhaustive over all non-NaN fp16 patterns
    t0 = time.time()
    post_plan = vfd.build_bgr_u8_post_plan(8)
    post = vfd._TrtEngine(post_plan, "post")
    res["post_build_s"] = time.time() - t0
    res["post_plan_bytes"] = len(post_plan)
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    vals = bits.view(torch.float16)
    vals = vals[~torch.isnan(vals)]
    n_el = 8 * 3 * 256 * 256
    tiled = vals.repeat((n_el + vals.numel() - 1) // vals.numel())[:n_el]
    covered = int(vals.numel())
    results = []
    with torch.inference_mode():
        for shift in (0, 12345):  # two layouts so every value lands in every channel
            img = torch.roll(tiled, shift).reshape(8, 3, 256, 256).to(DEV).contiguous()
            ref = vfd.repo_fast_postprocess_gpu(img)
            out = torch.empty((8, 256, 256, 3), device=DEV, dtype=torch.uint8)
            post.run(img.data_ptr(), out.data_ptr(), torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            results.append(int((out != ref).sum()))
        # NaN behaviour (documented only; the decoder clamps and callers reject non-finite)
        nan_img = torch.full((8, 3, 256, 256), float("nan"), device=DEV, dtype=torch.float16)
        ref = vfd.repo_fast_postprocess_gpu(nan_img)
        out = torch.empty((8, 256, 256, 3), device=DEV, dtype=torch.uint8)
        post.run(nan_img.data_ptr(), out.data_ptr(), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        res["post_nan_behaviour"] = {"repo_values": sorted(set(ref.flatten().tolist()))[:5],
                                     "trt_values": sorted(set(out.flatten().tolist()))[:5]}
        zc16 = compiled.decode(zc, 1.0, torch.float16).clone()
        res["post_engine_bs8_ms"] = bench(lambda: post.run(zc16.data_ptr(), out.data_ptr(),
                                                           torch.cuda.current_stream().cuda_stream))
        res["repo_post_bs8_ms"] = bench(lambda: vfd.repo_fast_postprocess_gpu(zc16))
    res["post_exhaustive"] = {"distinct_non_nan_fp16_values": covered, "mismatched_elements_per_layout": results}
    print("post exhaustive", res["post_exhaustive"], res["post_nan_behaviour"], flush=True)
    del post

    onnx8 = vfd.export_taesd_decoder_onnx(model, 8, DEV)
    onnx8b = vfd.export_taesd_decoder_onnx(model, 8, DEV)
    res["onnx_bytes"] = len(onnx8)
    res["onnx_export_deterministic_in_process"] = vfd._sha256_bytes(onnx8) == vfd._sha256_bytes(onnx8b)
    res["onnx_sha256"] = vfd._sha256_bytes(onnx8)
    res["variants"] = {}
    for opt, strong in ((3, False), (5, False), (3, True), (5, True)):
        name = f"bs8_opt{opt}_{'strong' if strong else 'weak'}"
        t0 = time.time()
        try:
            plan = vfd.build_taesd_decoder_plan(onnx8, opt_level=opt, strongly_typed=strong)
        except Exception as exc:  # noqa: BLE001
            res["variants"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(name, "FAILED", exc, flush=True)
            continue
        build_s = time.time() - t0
        eng = vfd._TrtEngine(plan, name)
        be = vfd.TaesdTrtBackend(eng, None, DEV, torch.float16, 8, meta={}, paths={})
        with torch.inference_mode():
            outs = [vfd.repo_fast_postprocess_gpu(be.decode(Z[i:i + 8], 1.0)) for i in range(0, Z.shape[0], 8)]
            u8 = torch.cat(outs)
            # determinism: re-run, and a shuffled batch composition
            again = torch.cat([vfd.repo_fast_postprocess_gpu(be.decode(Z[i:i + 8], 1.0))
                               for i in range(0, Z.shape[0], 8)])
            perm = torch.randperm(Z.shape[0], generator=torch.Generator().manual_seed(1)).to(DEV)
            shuffled = torch.cat([vfd.repo_fast_postprocess_gpu(be.decode(Z[perm[i:i + 8]], 1.0))
                                  for i in range(0, Z.shape[0], 8)])
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.numel(), device=DEV)
            timing = bench(lambda: be.decode(zc, 1.0))
        rec = {"build_s": build_s, "plan_bytes": len(plan), "bs8": timing,
               "ms_per_frame": timing["median_ms"] / 8,
               "vs_compiled_lsb": lsb(u8, ref_c), "vs_eager_lsb": lsb(u8, ref_e),
               "rerun_identical": bool(torch.equal(u8, again)),
               "batch_composition_invariant": bool(torch.equal(u8, shuffled[inv]))}
        res["variants"][name] = rec
        print(name, json.dumps({k: rec[k] for k in ("build_s", "ms_per_frame", "rerun_identical",
                                                    "batch_composition_invariant")}),
              "vs_compiled", {k: v for k, v in rec["vs_compiled_lsb"].items() if k != "hist"}, flush=True)
        del be, eng, plan
        (OUT / "explore_recipe.json").write_text(json.dumps(res, indent=1))

    # bs16 engine (speed reference for the sub-batch choice), opt3 weak and opt5 weak
    onnx16 = vfd.export_taesd_decoder_onnx(model, 16, DEV)
    z16 = Z[:16].contiguous()
    for opt in (3, 5):
        name = f"bs16_opt{opt}_weak"
        t0 = time.time()
        plan = vfd.build_taesd_decoder_plan(onnx16, opt_level=opt, strongly_typed=False)
        build_s = time.time() - t0
        eng = vfd._TrtEngine(plan, name)
        be = vfd.TaesdTrtBackend(eng, None, DEV, torch.float16, 16, meta={}, paths={})
        with torch.inference_mode():
            timing = bench(lambda: be.decode(z16, 1.0))
        res["variants"][name] = {"build_s": build_s, "plan_bytes": len(plan), "bs16": timing,
                                 "ms_per_frame": timing["median_ms"] / 16}
        print(name, res["variants"][name]["ms_per_frame"], flush=True)
        del be, eng, plan
    res["smi_end"] = smi()
    (OUT / "explore_recipe.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "explore_recipe.json")


if __name__ == "__main__":
    main()
