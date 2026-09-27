"""Probe 1: TAESD decode eager vs torch.compile modes at bs 8/16/24/32 (real post-UNet latents)."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True

BSS = [8, 16, 24, 32]
MODES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "eager", "eager_cl", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"]

model = load_taesd()
Z = real_latents(128).to(DEV)
raw = repo_raw_decode(model)
res = {"gpu_start": gpu_state("start"), "latents": "real pred_latents from unet-captures (128 frames)",
       "torch": torch.__version__, "results": {}}

with torch.inference_mode():
    refs = {bs: raw(Z[:bs]).float().clone() for bs in BSS}

for mode in MODES:
    torch._dynamo.reset()
    res["results"][mode] = {}
    for bs in BSS:
        z = Z[:bs].contiguous()
        mark = mode in ("max-autotune", "reduce-overhead")
        t0 = time.time()
        if mode == "eager":
            fn_ = raw
        elif mode == "eager_cl":
            model_cl = model.to(memory_format=torch.channels_last)
            fn_ = repo_raw_decode(model_cl)
            z = z.to(memory_format=torch.channels_last)
        else:
            fn_ = torch.compile(raw, mode=mode, dynamic=False)
        with torch.inference_mode():
            if mark:
                torch.compiler.cudagraph_mark_step_begin()
            out = fn_(z)
            torch.cuda.synchronize()
            compile_s = time.time() - t0
            err = float((out.float() - refs[bs]).abs().max())
            del out
            st = gpu_state(f"{mode} bs{bs}")
            r = bench_events(lambda: fn_(z), warmup=15, iters=40, mark_step=mark)
            thr = bench_throughput(lambda: fn_(z), warmup=10, iters=40, mark_step=mark)
        r.update({"ms_per_frame": r["median_ms"] / bs, "fps_equiv": 1000 * bs / r["median_ms"],
                  "pipelined": thr, "pipelined_ms_per_frame": thr["gpu_ms_per_call"] / bs,
                  "compile_or_first_call_s": round(compile_s, 1), "max_abs_vs_eager": err, "gpu_state": st})
        res["results"][mode][bs] = r
        print(f"{mode:28s} bs{bs:2d}: median {r['median_ms']:.3f} ms  ({r['ms_per_frame']:.3f} ms/f, "
              f"{r['fps_equiv']:.0f} fps)  wall {r['wall_median_ms']:.3f}  pipelined {thr['gpu_ms_per_call']:.3f} ms "
              f"compile {compile_s:.0f}s  maxabs {err:.2e}", flush=True)
    if mode == "eager_cl":
        model = model.to(memory_format=torch.contiguous_format)
        raw = repo_raw_decode(model)
    dump(f"p1_modes_{'_'.join(MODES)}.json", res)

res["gpu_end"] = gpu_state("end")
dump(f"p1_modes_{'_'.join(MODES)}.json", res)
