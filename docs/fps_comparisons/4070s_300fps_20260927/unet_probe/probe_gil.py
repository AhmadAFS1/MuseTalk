"""Probe 3d: host-side cost of the shipping TRT UNet call: does it hold the GIL, and how much CPU does the calling thread
burn per bs8 call when calls are issued back-to-back (launch-queue back-pressure), eager vs CUDA-graph replay."""
import threading
import time

import torch

from common import batch_inputs, save_json, wait_clean

log = []
res = {"gpu_log": log}
import torch_tensorrt  # noqa

mod = torch_tensorrt.load("/workspace/MuseTalk/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts").to("cuda").eval()
lat, aud = batch_inputs(8)
with torch.no_grad():
    for _ in range(5):
        mod(lat, aud)
torch.cuda.synchronize()

# CUDA graph of the torch_tensorrt call (worked in probe_trt)
g = torch.cuda.CUDAGraph()
sl, sa = lat.clone(), aud.clone()
sw = torch.cuda.Stream()
sw.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(sw):
    for _ in range(3):
        mod(sl, sa)
torch.cuda.current_stream().wait_stream(sw)
with torch.no_grad(), torch.cuda.graph(g):
    so = mod(sl, sa)
torch.cuda.synchronize()

stop = threading.Event()
counter = [0]


def spinner():
    c = 0
    while not stop.is_set():
        c += 1
    counter[0] = c


def py_rate(seconds=2.0):
    stop.clear(); counter[0] = 0
    t = threading.Thread(target=spinner); t.start()
    time.sleep(seconds)
    stop.set(); t.join()
    return counter[0] / seconds


def main_loop(kind, seconds=3.0, with_spinner=True):
    torch.cuda.synchronize()
    stop.clear(); counter[0] = 0
    th = threading.Thread(target=spinner) if with_spinner else None
    if th:
        th.start()
    n = 0
    t0 = time.perf_counter(); c0 = time.thread_time()
    with torch.no_grad():
        while time.perf_counter() - t0 < seconds:
            if kind == "eager":
                mod(lat, aud)
            else:
                g.replay()
            n += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0; cpu = time.thread_time() - c0
    stop.set()
    if th:
        th.join()
    return {"calls": n, "wall_s": wall, "calls_per_s": n / wall, "fps_bs8": 8 * n / wall,
            "main_thread_cpu_ms_per_call": 1e3 * cpu / n, "main_thread_cpu_util": cpu / wall,
            "spinner_rate_per_s": counter[0] / wall if th else None}


st = wait_clean("gil_probe", log)
res["contaminated"] = st["contaminated"]
res["spinner_alone_rate_per_s"] = py_rate()
res["eager_no_spinner"] = main_loop("eager", with_spinner=False)
res["eager_with_spinner"] = main_loop("eager")
res["graph_no_spinner"] = main_loop("graph", with_spinner=False)
res["graph_with_spinner"] = main_loop("graph")
res["spinner_alone_rate_per_s_after"] = py_rate()
for k, v in res.items():
    print(k, v, flush=True)
save_json("gil.json", res)
print("DONE")
