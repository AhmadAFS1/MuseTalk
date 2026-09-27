"""Probe 1b: inductor silently skips max-autotune templates on <68-SM GPUs (4070S has 56).
Force it (patch torch._inductor.utils.is_big_gpu) and re-time full + staged-crop TAESD."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch._dynamo, torch._inductor.utils as iu
import torch.nn as nn

iu.is_big_gpu = lambda index=0: True
torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True

model = load_taesd()
dec = model.decoder; layers = list(dec.layers)
FIRST = {6: 9, 11: 39, 16: 99}
raw = repo_raw_decode(model)


def staged(z):
    x = torch.tanh(z / 3) * 3
    offset = 0
    for i, L in enumerate(layers):
        if i in FIRST:
            cut = FIRST[i] - offset
            if cut > 0:
                x = x[:, :, cut:, :]; offset = FIRST[i]
        x = L(x)
        if isinstance(L, nn.Upsample):
            offset *= 2
    x = (x.mul(2).sub(1) / 2 + 0.5).clamp(0, 1)
    return x[:, :, USED_Y0 - offset:, :]

Z = real_latents(32).to(DEV)
res = {"gpu_start": gpu_state("start"), "results": {}}
with torch.inference_mode():
    refs = {bs: raw(Z[:bs]).float() for bs in (8, 16)}
for vname, f in (("full", raw), ("staged_exact", staged)):
    for bs in (8, 16):
        torch._dynamo.reset()
        z = Z[:bs].contiguous()
        fn_ = torch.compile(f, mode="max-autotune-no-cudagraphs", dynamic=False)
        t0 = time.time()
        with torch.inference_mode():
            y = fn_(z); torch.cuda.synchronize(); cs = time.time() - t0
            ref = refs[bs] if vname == "full" else refs[bs][:, :, USED_Y0:, :]
            err = float((y.float() - ref).abs().max())
            st = gpu_state(f"forced-autotune {vname} bs{bs}")
            r = bench_events(lambda: fn_(z), warmup=15, iters=40)
        r.update({"ms_per_frame": r["median_ms"] / bs, "compile_s": cs, "max_abs": err, "gpu_state": st})
        res["results"][f"{vname}_bs{bs}"] = r
        print(f"forced max-autotune {vname:14s} bs{bs}: {r['median_ms']:.3f} ms ({r['ms_per_frame']:.3f} ms/f) "
              f"compile {cs:.0f}s max_abs {err:.2e}", flush=True)
        dump("p1b_forced_autotune.json", res)
res["gpu_end"] = gpu_state("end")
dump("p1b_forced_autotune.json", res)
