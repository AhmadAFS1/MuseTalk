"""Probe 5: in-memory torch_tensorrt FP16 UNet at a native static bs16 (production converter settings), vs the shipping
bs8 engine called twice. Nothing is serialized to disk (timing cache only, a few MB, in the scratch dir)."""
import gc
import os
import time

import torch

from common import OUT, batch_inputs, load_captures, load_unet_fp16_lowmem, save_json, wait_clean
import torch_tensorrt

log = []
res = {"gpu_log": log}
dev = torch.device("cuda")
model, _ = load_unet_fp16_lowmem()


class UNetTRTWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.register_buffer("timesteps", torch.tensor([0], device=dev, dtype=torch.long))

    def forward(self, latent, encoder_hidden_states):
        return self.unet(latent, self.timesteps, encoder_hidden_states=encoder_hidden_states).sample


w = UNetTRTWrapper(model).eval()
lat16, aud16 = batch_inputs(16)
with torch.no_grad():
    ref16 = w(lat16, aud16)
t = time.time()
free0 = torch.cuda.mem_get_info()[0] / 2**30
trt16 = torch_tensorrt.compile(
    w, ir="dynamo",
    inputs=[torch_tensorrt.Input(shape=(16, 8, 32, 32), dtype=torch.half),
            torch_tensorrt.Input(shape=(16, 50, 384), dtype=torch.half)],
    enabled_precisions={torch.float16}, workspace_size=1 << 30, min_block_size=1,
    require_full_compilation=True, pass_through_build_failures=False,
    timing_cache_path=os.path.join(OUT, "tt16_timing_cache.bin"))
res["build_s"] = time.time() - t
res["free_vram_GB_before_build"] = free0
res["peak_alloc_GB_during_build"] = torch.cuda.max_memory_allocated() / 2**30
print("built", res["build_s"], flush=True)
del w, model
gc.collect(); torch.cuda.empty_cache()
with torch.no_grad():
    o = trt16(lat16, aud16)
    o = o[0] if isinstance(o, (list, tuple)) else o
    res["bs16_rel_l2_vs_torch"] = ((o.float() - ref16.float()).norm() / ref16.float().norm()).item()
    res["bs16_mae_vs_torch"] = (o.float() - ref16.float()).abs().mean().item()


def cuda_ms(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "min_ms": ts[0], "p90_ms": ts[int(0.9 * len(ts)) - 1]}


# shipping bs8 engine in the same process for a same-thermal-state A/B
# (host-RAM safe: the shipping bs8 engine is NOT loaded here; compare against its 2x bs8 numbers measured earlier:
#  torch_tensorrt 49.29 ms, raw TRT 48.06 ms, see trt.json / trt_raw.json)
import resource
res["rss_GB_after_build"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
with torch.no_grad():
    for rnd in range(2):
        st = wait_clean(f"bs16_native_round{rnd}", log)
        a = cuda_ms(lambda: trt16(lat16, aud16))
        res[f"round{rnd}"] = {"native_bs16": a, "ms_per_frame": a["median_ms"] / 16,
                              "speedup_vs_prod_2x8_torch_tensorrt_49.29": 49.29 / a["median_ms"],
                              "contaminated": st["contaminated"]}
        print(res[f"round{rnd}"], flush=True)
    # CUDA graph of the native bs16 module
    try:
        g = torch.cuda.CUDAGraph()
        sl, sa = lat16.clone(), aud16.clone()
        sw = torch.cuda.Stream(); sw.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sw):
            for _ in range(3):
                trt16(sl, sa)
        torch.cuda.current_stream().wait_stream(sw)
        with torch.cuda.graph(g):
            trt16(sl, sa)
        res["native_bs16_cudagraph"] = cuda_ms(lambda: g.replay())
    except Exception as ex:
        res["native_bs16_cudagraph"] = {"error": repr(ex)[:300]}
res["peak_reserved_GB"] = torch.cuda.max_memory_reserved() / 2**30
save_json("ttrt_bs16.json", res)
print(res)
print("DONE")
