"""Probe 1c: same as p1 max-autotune bs8/16 but cudnn.benchmark=False (api_server may not set it)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
torch.backends.cudnn.benchmark = False
model = load_taesd(); raw = repo_raw_decode(model); Z = real_latents(16).to(DEV)
res = {"gpu_start": gpu_state("start")}
for mode in ("eager", "max-autotune"):
    for bs in (8, 16):
        torch._dynamo.reset()
        fn_ = raw if mode == "eager" else torch.compile(raw, mode=mode, dynamic=False)
        z = Z[:bs].contiguous(); mark = mode != "eager"
        with torch.inference_mode():
            if mark: torch.compiler.cudagraph_mark_step_begin()
            fn_(z); torch.cuda.synchronize()
            st = gpu_state(f"{mode} bs{bs} nobench")
            r = bench_events(lambda: fn_(z), warmup=15, iters=40, mark_step=mark)
        r["ms_per_frame"] = r["median_ms"] / bs; r["gpu_state"] = st
        res[f"{mode}_bs{bs}"] = r
        print(f"cudnn.benchmark=False {mode} bs{bs}: {r['median_ms']:.3f} ms ({r['ms_per_frame']:.3f} ms/f)", flush=True)
res["gpu_end"] = gpu_state("end")
dump("p1c_nobench.json", res)
