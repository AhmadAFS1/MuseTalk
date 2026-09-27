"""Probe 3c: what does the TRT UNet call block on (host side)? + torch_tensorrt cudagraphs mode."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from trt_runtime import _load_serialized_trt_module
import torch_tensorrt

res = {"gpu_start": gpu_state("start")}
lat, aud = real_unet_inputs(); lat, aud = lat.to(DEV), aud.to(DEV)
unet_mod = _load_serialized_trt_module(ROOT / "models/tensorrt_unet_sm89_bs8_local/unet_trt.ts", DEV)
model = load_taesd(); raw = repo_raw_decode(model)
z0 = real_latents(8).to(DEV)
sB = torch.cuda.Stream()


def unet():
    o = unet_mod(lat, aud)
    return o if isinstance(o, torch.Tensor) else o[0]


def host(f):
    h0 = time.perf_counter(); f(); return (time.perf_counter() - h0) * 1e3

with torch.inference_mode():
    for _ in range(5): unet(); raw(z0)
    torch.cuda.synchronize()
    ref = unet().clone()
    res["pre"] = gpu_state("pre")
    # 1) GPU busy with ~60ms of unrelated decode work on sB, then call unet on default stream
    torch.cuda.synchronize()
    with torch.cuda.stream(sB):
        for _ in range(10): raw(z0)
    res["unet_host_ms_while_other_stream_busy"] = host(unet)
    torch.cuda.synchronize()
    # 2) same, but other work on the SAME (default) stream
    for _ in range(10): raw(z0)
    res["unet_host_ms_while_same_stream_busy"] = host(unet)
    torch.cuda.synchronize()
    # 3) back-to-back unet calls
    a = host(unet); b = host(unet); c = host(unet)
    torch.cuda.synchronize()
    res["unet_host_ms_back_to_back"] = [a, b, c]
    print(res, flush=True)

    # 4) torch_tensorrt cudagraphs mode
    try:
        torch_tensorrt.runtime.set_cudagraphs_mode(True)
        for _ in range(5): unet()
        torch.cuda.synchronize()
        out_cg = unet().clone(); torch.cuda.synchronize()
        res["cudagraph_mode_max_abs_vs_normal"] = float((out_cg.float() - ref.float()).abs().max())
        res["cg_gpu_state"] = gpu_state("cudagraphs mode")
        res["cg_unet_events"] = bench_events(unet, warmup=10, iters=30)
        a = host(unet); b = host(unet); c = host(unet); torch.cuda.synchronize()
        res["cg_unet_host_ms_back_to_back"] = [a, b, c]
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
    except Exception as exc:
        res["cudagraph_mode_error"] = repr(exc)[:400]
    res["normal_unet_events"] = bench_events(unet, warmup=10, iters=30)
res["gpu_end"] = gpu_state("end")
print({k: v for k, v in res.items() if "gpu" not in k}, flush=True)
dump("p3c_block.json", res)
