"""Probe 3b: why does the dependent two-stream pipeline serialize? + TRT UNet host cost / GIL."""
import sys, time, threading
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch._dynamo
torch.backends.cudnn.benchmark = True
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from trt_runtime import _load_serialized_trt_module

N = 30
res = {"gpu_start": gpu_state("start")}
lat, aud = real_unet_inputs(); lat, aud = lat.to(DEV), aud.to(DEV)
unet_mod = _load_serialized_trt_module(ROOT / "models/tensorrt_unet_sm89_bs8_local/unet_trt.ts", DEV)


def unet():
    o = unet_mod(lat, aud)
    return o if isinstance(o, torch.Tensor) else o[0]

model = load_taesd()
raw = repo_raw_decode(model)
dec = torch.compile(raw, mode="max-autotune-no-cudagraphs", dynamic=False)
z0 = real_latents(8).to(DEV)
sA, sB = torch.cuda.Stream(), torch.cuda.Stream()


def run(label, body, n=N):
    torch.cuda.synchronize()
    host = []
    t0 = time.perf_counter()
    for i in range(n):
        h0 = time.perf_counter(); body(i); host.append((time.perf_counter() - h0) * 1e3)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / n
    r = {"wall_ms_per_iter": wall, "host_ms_per_iter_median": float(np.median(host))}
    print(f"{label:55s} wall {wall:7.3f} ms/iter  host {r['host_ms_per_iter_median']:.3f} ms/iter", flush=True)
    res[label] = r
    return r

with torch.inference_mode():
    for _ in range(10): unet(); dec(z0)
    for _ in range(5):
        with torch.cuda.stream(sA): o = unet()
        with torch.cuda.stream(sB): dec(o)
    torch.cuda.synchronize()
    res["pre"] = gpu_state("pre")

    run("unet only (default stream)", lambda i: unet())

    def unet_sA(i):
        with torch.cuda.stream(sA): unet()
    run("unet only (sA)", unet_sA)
    run("taesd only (default)", lambda i: dec(z0))

    def dep_event(i):
        with torch.cuda.stream(sA):
            o = unet(); ev = torch.cuda.Event(); ev.record(sA)
        with torch.cuda.stream(sB):
            sB.wait_event(ev); o.record_stream(sB); dec(o)
    run("dep: event wait + record_stream", dep_event)

    def dep_no_rs(i):
        with torch.cuda.stream(sA):
            o = unet(); ev = torch.cuda.Event(); ev.record(sA)
        with torch.cuda.stream(sB):
            sB.wait_event(ev); dec(o)
    run("dep: event wait, no record_stream", dep_no_rs)

    def dep_clone(i):
        with torch.cuda.stream(sA):
            o = unet().clone(); ev = torch.cuda.Event(); ev.record(sA)
        with torch.cuda.stream(sB):
            sB.wait_event(ev); o.record_stream(sB); dec(o)
    run("dep: clone on sA then event", dep_clone)

    buf = torch.empty_like(z0)

    def dep_static(i):
        with torch.cuda.stream(sA):
            buf.copy_(unet()); ev = torch.cuda.Event(); ev.record(sA)
        with torch.cuda.stream(sB):
            sB.wait_event(ev); dec(buf)
    run("dep: static buffer copy (racy, timing only)", dep_static)

    def nodep_unet_out(i):
        with torch.cuda.stream(sA):
            o = unet()
        with torch.cuda.stream(sB):
            dec(z0)
    run("nodep: two streams, decode fixed z", nodep_unet_out)

    def dep_same_stream(i):
        with torch.cuda.stream(sA):
            dec(unet())
    run("serial on sA: dec(unet())", dep_same_stream)

    def dep_default(i):
        dec(unet())
    run("serial default stream: dec(unet())", dep_default)

    # double-buffered pipeline: decode of batch i-1 issued AFTER unet i, both launched before waiting
    prev = {"o": None, "ev": None}

    def pipe2(i):
        with torch.cuda.stream(sA):
            o = unet(); ev = torch.cuda.Event(); ev.record(sA)
        if prev["o"] is not None:
            with torch.cuda.stream(sB):
                sB.wait_event(prev["ev"]); prev["o"].record_stream(sB); dec(prev["o"])
        prev["o"], prev["ev"] = o, ev
    run("pipelined: decode(i-1) after launching unet(i)", pipe2)

    # --- GIL: does a TRT UNet call hold the GIL while enqueuing?
    counter = {"n": 0, "on": True}

    def spin():
        while counter["on"]:
            counter["n"] += 1
    th = threading.Thread(target=spin); th.start()
    time.sleep(0.5); counter["n"] = 0; t0 = time.perf_counter(); time.sleep(0.5)
    idle_rate = counter["n"] / (time.perf_counter() - t0)
    torch.cuda.synchronize(); counter["n"] = 0; t0 = time.perf_counter()
    host = 0.0
    for _ in range(20):
        h0 = time.perf_counter(); unet(); host += time.perf_counter() - h0
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    busy_rate = counter["n"] / dt
    counter["on"] = False; th.join()
    res["gil"] = {"spin_rate_idle_per_s": idle_rate, "spin_rate_during_unet_loop_per_s": busy_rate,
                  "unet_host_ms_per_call_with_spinner": host * 1e3 / 20}
    print("GIL probe:", res["gil"], flush=True)

    # host enqueue cost distribution for the TRT UNet (no sync between calls, queue not saturated)
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); h0 = time.perf_counter(); unet(); ts.append((time.perf_counter() - h0) * 1e3)
    res["unet_host_enqueue_ms_median"] = float(np.median(ts))
    print("TRT UNet host enqueue median ms:", np.median(ts), flush=True)

res["gpu_end"] = gpu_state("end")
dump("p3b_debug.json", res)
