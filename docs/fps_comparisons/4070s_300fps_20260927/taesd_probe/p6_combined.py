"""Probe 6: best-case GPU model path on this card with the levers found here.

UNet: live bs8 TRT artifact, torch_tensorrt cudagraphs mode ON vs OFF.
Decode: TRT FP16 TAESD staged exact row-crop (rows>=104) emitting BGR NHWC uint8 [B,152,256,3],
        vs the current live decode (torch.compile max-autotune TAESD full + repo post-process).
Measures sustained wall ms per bs8 batch (aggregate fps = 8000/ms), serial and 2-stream pipelined.
"""
import sys, time, threading, subprocess, io
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch._dynamo
import torch_tensorrt
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from trt_runtime import _load_serialized_trt_module
import p5_trt_taesd as P5  # builds nothing at import? (guarded below)

N = 60
res = {"gpu_start": gpu_state("start")}
lat, aud = real_unet_inputs(); lat, aud = lat.to(DEV), aud.to(DEV)
unet_mod = _load_serialized_trt_module(ROOT / "models/tensorrt_unet_sm89_bs8_local/unet_trt.ts", DEV)


def unet():
    o = unet_mod(lat, aud)
    return o if isinstance(o, torch.Tensor) else o[0]

trt_dec, _, *hold = P5.build(P5.StagedU8(), 8, "staged_u8_bs8", out_u8=True)
raw = repo_raw_decode(P5.model)
live_dec = torch.compile(raw, mode="max-autotune", dynamic=False)
pinned = torch.empty((8, 256, 256, 3), dtype=torch.uint8, pin_memory=True)
pinned_crop = torch.empty((8, 152, 256, 3), dtype=torch.uint8, pin_memory=True)


def live_post(img):
    return (img.float().mul(255).round().clamp_(0, 255).to(torch.uint8).flip(1).permute(0, 2, 3, 1).contiguous())

sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
samples = []


def sampler(flag):
    while flag["on"]:
        samples.append(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                                       "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
        time.sleep(0.5)


def timed(label, body):
    for i in range(8): body(i)
    torch.cuda.synchronize()
    flag = {"on": True}; samples.clear(); th = threading.Thread(target=sampler, args=(flag,)); th.start()
    t0 = time.perf_counter()
    for i in range(N): body(i)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / N
    flag["on"] = False; th.join()
    r = {"wall_ms_per_bs8": ms, "aggregate_fps": 8000 / ms, "smi": samples[:4]}
    res[label] = r
    print(f"{label:70s} {ms:7.3f} ms/bs8 -> {8000/ms:6.1f} fps   smi {samples[1:3]}", flush=True)
    return r

with torch.inference_mode():
    for cg in (False, True):
        torch_tensorrt.runtime.set_cudagraphs_mode(cg)
        tag = "UNet[cudagraphs]" if cg else "UNet[default]"
        for _ in range(5): unet()
        torch.cuda.synchronize()
        res[f"pre_{tag}"] = gpu_state(f"pre {tag}")

        def unet_only(i):
            with torch.cuda.stream(sA): unet()
        timed(f"{tag} only", unet_only)

        def live_serial(i):
            with torch.cuda.stream(sA):
                o = unet(); torch.compiler.cudagraph_mark_step_begin()
                pinned.copy_(live_post(live_dec(o)), non_blocking=True)
        timed(f"{tag} + LIVE decode (compiled TAESD full + post + pinned D2H), serial", live_serial)

        def new_serial(i):
            with torch.cuda.stream(sA):
                o = unet(); hold[2].copy_(o); y = trt_dec()
                pinned_crop.copy_(y, non_blocking=True)
        timed(f"{tag} + TRT TAESD staged-crop u8 + pinned D2H, serial", new_serial)

        prev = {}

        def new_pipe(i):
            with torch.cuda.stream(sA):
                o = unet(); ev = torch.cuda.Event(); ev.record(sA)
            with torch.cuda.stream(sB):
                sB.wait_event(ev); o.record_stream(sB)
                hold[2].copy_(o); y = trt_dec()
                pinned_crop.copy_(y, non_blocking=True)
        timed(f"{tag} + TRT TAESD staged-crop u8, 2-stream (decode on sB)", new_pipe)
    torch_tensorrt.runtime.set_cudagraphs_mode(False)
res["gpu_end"] = gpu_state("end")
dump("p6_combined.json", res)
