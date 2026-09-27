"""Probe 3c: kernel-level breakdown of the shipping TRT UNet engine (CUPTI kernel names/durations) + layer-type counts."""
import collections
import json
import re

import torch

from common import batch_inputs, save_json, wait_clean

log = []
res = {"gpu_log": log}
import torch_tensorrt  # noqa
import tensorrt as trt

mod = torch_tensorrt.load("/workspace/MuseTalk/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts")
info = mod._run_on_acc_0.engine.__getstate__()[0]
del mod
blob = info[3]
import base64
blob = base64.b64decode(blob) if isinstance(blob, str) else blob
del info
torch.cuda.empty_cache()
rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = rt.deserialize_cuda_engine(blob)
del blob
insp = engine.create_engine_inspector()
layers = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
types = collections.Counter()
aten = collections.Counter()
for L in layers:
    m = re.match(r"\[(\w+)\]-\[([\w\.]+)\]", L)
    if L.startswith("Reformatting"):
        types["REFORMAT"] += 1
    elif m:
        types[m.group(1)] += 1
        aten[m.group(2)] += 1
    else:
        types["other:" + L[:40]] += 1
res["layer_types"] = types.most_common()
res["layer_aten_ops"] = aten.most_common(30)
print(res["layer_types"], flush=True)
print(res["layer_aten_ops"], flush=True)

ctx = engine.create_execution_context()
lat, aud = batch_inputs(8)
out = torch.empty(8, 4, 32, 32, device="cuda", dtype=torch.float16)
ctx.set_tensor_address("latent", lat.data_ptr())
ctx.set_tensor_address("encoder_hidden_states", aud.data_ptr())
ctx.set_tensor_address("output0", out.data_ptr())
s = torch.cuda.Stream()
for _ in range(10):
    ctx.execute_async_v3(s.cuda_stream)
torch.cuda.synchronize()
st = wait_clean("trt_kernel_profile", log)
from torch.profiler import ProfilerActivity, profile
N = 5
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
kev = [k for k in prof.profiler.kineto_results.events() if k.device_type() == torch.autograd.DeviceType.CUDA]
byname = collections.defaultdict(lambda: [0, 0.0])
for k in kev:
    byname[k.name()][0] += 1
    byname[k.name()][1] += k.duration_ns() / 1e6


def cat(n):
    nl = n.lower()
    if "mha" in nl or "fmha" in nl or "flash" in nl or "attention" in nl:
        return "attention_fused(mha)"
    if "fprop" in nl or "conv" in nl or "implicit" in nl or "xmma_fprop" in nl or "dgrad" in nl:
        return "conv"
    if "gemm" in nl or "matmul" in nl or "sgemm" in nl or "hgemm" in nl or "cutlass" in nl or "_mm_" in nl:
        return "gemm"
    if "reduce" in nl or "norm" in nl or "instance" in nl:
        return "norm/reduce"
    if "copy" in nl or "reformat" in nl or "shuffle" in nl or "transpose" in nl or "cat" in nl or "concat" in nl:
        return "copy/reformat"
    if "myelin" in nl or "pointwise" in nl or "elementwise" in nl or "generatednative" in nl or "__myl" in nl:
        return "fused_pointwise(myelin)"
    return "other"


cats = collections.defaultdict(lambda: [0, 0.0])
for n, (c, d) in byname.items():
    cats[cat(n)][0] += c
    cats[cat(n)][1] += d
tot = sum(d for _, d in byname.values())
res["kernel_total_ms_per_call"] = tot / N
res["kernels_per_call"] = len(kev) / N
res["category_ms_per_call"] = {k: {"ms": v[1] / N, "share": v[1] / tot, "kernels_per_call": v[0] / N}
                               for k, v in sorted(cats.items(), key=lambda x: -x[1][1])}
res["top_kernels"] = [{"name": n[:160], "calls_per_fwd": c / N, "ms_per_fwd": d / N, "cat": cat(n)}
                      for n, (c, d) in sorted(byname.items(), key=lambda x: -x[1][1])[:60]]
res["contaminated"] = st["contaminated"]
print("total kernel ms/call", tot / N, "kernels/call", len(kev) / N)
for k, v in res["category_ms_per_call"].items():
    print(f"  {k:28s} {v['ms']:7.3f} ms {100*v['share']:5.1f}%  n={v['kernels_per_call']:.0f}")
for r in res["top_kernels"][:30]:
    print(f"  {r['ms_per_fwd']:7.3f} ms x{r['calls_per_fwd']:.0f} [{r['cat']}] {r['name'][:130]}")
save_json("trt_kernels.json", res)
print("DONE")
