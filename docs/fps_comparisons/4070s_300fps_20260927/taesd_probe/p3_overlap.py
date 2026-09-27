"""Probe 3: does TAESD on a second CUDA stream overlap with the live TRT UNet (bs8)?

Loads the exact live UNet artifact (models/tensorrt_unet_sm89_bs8_local/unet_trt.ts, read-only)
and real captured UNet inputs. Falls back to the PyTorch FP16 UNet if the TRT load fails.
"""
import sys, time, threading, subprocess
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
N = 40

res = {"gpu_start": gpu_state("start")}
lat, aud = real_unet_inputs()
lat, aud = lat.to(DEV), aud.to(DEV)

unet_kind = "trt"
try:
    from trt_runtime import _load_serialized_trt_module
    t0 = time.time()
    unet_mod = _load_serialized_trt_module(ROOT / "models/tensorrt_unet_sm89_bs8_local/unet_trt.ts", DEV)
    print(f"TRT UNet loaded in {time.time()-t0:.1f}s", flush=True)

    def unet():
        o = unet_mod(lat, aud)
        return o if isinstance(o, torch.Tensor) else o[0]
except Exception as exc:  # pragma: no cover
    print("TRT UNet load failed, using PyTorch FP16 UNet:", exc, flush=True)
    unet_kind = "pytorch_fp16"
    from musetalk.models.unet import UNet
    u = UNet(unet_config=str(ROOT / "models/musetalkV15/musetalk.json"),
             model_path=str(ROOT / "models/musetalkV15/unet.pth"), device=DEV)
    um = u.model.half().to(DEV).eval()
    tstep = torch.tensor([0], device=DEV)

    def unet():
        return um(lat, tstep, encoder_hidden_states=aud).sample
res["unet_kind"] = unet_kind

model = load_taesd()
raw = repo_raw_decode(model)
Z8 = real_latents(16).to(DEV)
taesd_fns = {
    "taesd_ma_bs8": (torch.compile(raw, mode="max-autotune", dynamic=False), Z8[:8].contiguous(), True),
    "taesd_ma_nocg_bs8": (torch.compile(raw, mode="max-autotune-no-cudagraphs", dynamic=False), Z8[:8].contiguous(), False),
}

sampling = {"on": False, "rows": []}


def sampler():
    while sampling["on"]:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        sampling["rows"].append(q)
        time.sleep(0.3)


def timed(label, body, n=N):
    torch.cuda.synchronize()
    sampling["on"], sampling["rows"] = True, []
    th = threading.Thread(target=sampler); th.start()
    t0 = time.perf_counter()
    body(n)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / n
    sampling["on"] = False; th.join()
    return {"wall_ms_per_iter": wall, "smi_samples": sampling["rows"][:6]}


sA = torch.cuda.Stream()
sB = torch.cuda.Stream()
with torch.inference_mode():
    for _ in range(10):
        unet()
    for k, (fn, z, cg) in taesd_fns.items():
        for _ in range(10):
            if cg: torch.compiler.cudagraph_mark_step_begin()
            fn(z)
    torch.cuda.synchronize()

    # host-side launch cost of one TRT UNet call (does it block the Python thread?)
    torch.cuda.synchronize(); t0 = time.perf_counter(); unet(); t_launch = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize(); t0 = time.perf_counter(); unet(); torch.cuda.synchronize(); t_full = (time.perf_counter() - t0) * 1e3
    res["unet_host_launch_ms"] = t_launch; res["unet_single_call_ms"] = t_full
    print(f"UNet host launch {t_launch:.2f} ms vs sync call {t_full:.2f} ms", flush=True)

    res["gpu_pre_blocks"] = gpu_state("pre-blocks")

    def unet_loop(n, stream=sA):
        with torch.cuda.stream(stream):
            for _ in range(n): unet()

    def mk_taesd_loop(fn, z, cg, stream=sB):
        def loop(n):
            with torch.cuda.stream(stream):
                for _ in range(n):
                    if cg: torch.compiler.cudagraph_mark_step_begin()
                    fn(z)
        return loop

    res["unet_alone"] = timed("unet", unet_loop)
    print("unet alone", res["unet_alone"], flush=True)
    for k, (fn, z, cg) in taesd_fns.items():
        r = {}
        r["taesd_alone"] = timed(k, mk_taesd_loop(fn, z, cg))
        serial_sum = res["unet_alone"]["wall_ms_per_iter"] + r["taesd_alone"]["wall_ms_per_iter"]

        def same_stream(n, fn=fn, z=z, cg=cg):
            with torch.cuda.stream(sA):
                for _ in range(n):
                    unet()
                    if cg: torch.compiler.cudagraph_mark_step_begin()
                    fn(z)
        r["serial_same_stream"] = timed("serial", same_stream)

        def two_stream_1thread(n, fn=fn, z=z, cg=cg):
            for _ in range(n):
                with torch.cuda.stream(sA):
                    unet()
                with torch.cuda.stream(sB):
                    if cg: torch.compiler.cudagraph_mark_step_begin()
                    fn(z)
        r["two_stream_1thread"] = timed("2s1t", two_stream_1thread)

        def two_stream_pipelined_dep(n, fn=fn, z=z, cg=cg):
            # realistic: decode(batch i) waits for unet(batch i); unet(i+1) proceeds meanwhile.
            for _ in range(n):
                with torch.cuda.stream(sA):
                    out = unet()
                    ev = torch.cuda.Event(); ev.record(sA)
                with torch.cuda.stream(sB):
                    sB.wait_event(ev)
                    out.record_stream(sB)
                    if cg: torch.compiler.cudagraph_mark_step_begin()
                    fn(out[:, :4].contiguous() if out.shape[1] != 4 else out)
        r["two_stream_pipelined_dep"] = timed("2s-dep", two_stream_pipelined_dep)

        def two_threads(n, fn=fn, z=z, cg=cg):
            ta = threading.Thread(target=unet_loop, args=(n,))
            tb = threading.Thread(target=mk_taesd_loop(fn, z, cg), args=(n,))
            ta.start(); tb.start(); ta.join(); tb.join()
        try:
            r["two_threads"] = timed("2thr", two_threads)
        except Exception as exc:
            r["two_threads"] = {"error": repr(exc)[:300], "wall_ms_per_iter": float("nan")}
        r["serial_sum_of_alone"] = serial_sum
        for kk in ("serial_same_stream", "two_stream_1thread", "two_stream_pipelined_dep", "two_threads"):
            r[kk]["saving_vs_serial_sum_pct"] = 100 * (1 - r[kk]["wall_ms_per_iter"] / serial_sum)
        res[k] = r
        print(k, {kk: (round(v["wall_ms_per_iter"], 3), round(v.get("saving_vs_serial_sum_pct", 0), 1))
                  if isinstance(v, dict) else round(v, 3) for kk, v in r.items()}, flush=True)
        dump("p3_overlap.json", res)

    # UNet vs UNet on two streams (is the UNet itself saturating the GPU?)
    def unet_2streams(n):
        for _ in range(n):
            with torch.cuda.stream(sA): unet()
            with torch.cuda.stream(sB): unet()
    r2 = timed("unet2s", unet_2streams)
    res["unet_x2_two_streams_per_pair"] = r2
    res["unet_x2_saving_vs_2x_alone_pct"] = 100 * (1 - r2["wall_ms_per_iter"] / (2 * res["unet_alone"]["wall_ms_per_iter"]))
    print("unet x2 two streams", r2["wall_ms_per_iter"], res["unet_x2_saving_vs_2x_alone_pct"], flush=True)

res["gpu_end"] = gpu_state("end")
dump("p3_overlap.json", res)
