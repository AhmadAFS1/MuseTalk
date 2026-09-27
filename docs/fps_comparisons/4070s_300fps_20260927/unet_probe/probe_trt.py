"""Probe 3: shipping TRT UNet artifact: bs8 time, 2x bs8, host enqueue overhead, 2-stream overlap, CUDA graphs,
plus raw TensorRT (same engine bytes, 2 execution contexts sharing weights) and engine layer inspection."""
import collections
import json
import time
import traceback

import torch

from common import batch_inputs, cuda_time, save_json, wait_clean

TS = "/workspace/MuseTalk/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts"
log = []
res = {"gpu_log": log}
FLOP_PER_FRAME = None
try:
    FLOP_PER_FRAME = json.load(open("eager.json"))["flops"]["per_frame_GFLOP"] * 1e9
except Exception:
    pass


def tflops(ms, frames):
    return None if FLOP_PER_FRAME is None else FLOP_PER_FRAME * frames / (ms / 1e3) / 1e12


import torch_tensorrt  # noqa: E402  (registers torch.classes.tensorrt)

t = time.time()
mod = torch_tensorrt.load(TS)
mod = mod.to("cuda").eval()
res["load_s"] = time.time() - t
print("loaded", res["load_s"], flush=True)

# find the engine objects inside the TorchScript module
engine_state = None
eng_attrs = []
for name, sm in mod.named_modules():
    for attr in dir(sm):
        pass
try:
    code = mod.code
    res["ts_code_head"] = code[:1500]
except Exception as e:
    res["ts_code_head"] = repr(e)
for name, sm in mod.named_modules():
    try:
        for an in sm._c._attributes if hasattr(sm._c, "_attributes") else []:
            pass
    except Exception:
        pass
# torch.jit ScriptModule attributes
def find_engines(m, prefix=""):
    found = []
    try:
        for an, av in m._c.__getattr__("__dict__").items():
            pass
    except Exception:
        pass
    try:
        names = [n for n in dir(m) if not n.startswith("__")]
    except Exception:
        names = []
    for n in names:
        try:
            v = getattr(m, n)
        except Exception:
            continue
        if type(v).__name__ in ("ScriptObject",) or "Engine" in type(v).__name__:
            found.append((prefix + n, v))
    for cn, c in m.named_children():
        found += find_engines(c, prefix + cn + ".")
    return found

engs = find_engines(mod)
res["engine_objects"] = [(n, type(v).__name__) for n, v in engs]
print("engines", res["engine_objects"], flush=True)
for n, v in engs:
    try:
        st = v.__getstate__()
        engine_state = st
        res["engine_state_summary"] = [(i, type(x).__name__, len(x) if hasattr(x, "__len__") else x) for i, x in enumerate(st)]
        break
    except Exception as e:
        res.setdefault("engine_state_errors", []).append(repr(e))
print("state", res.get("engine_state_summary"), flush=True)

lat8, aud8 = batch_inputs(8)
lat16, aud16 = batch_inputs(16)

with torch.no_grad():
    out = mod(lat8, aud8)
    out = out[0] if isinstance(out, (list, tuple)) else out
    res["out_shape"] = list(out.shape)

    # ---- bs8 single call ----
    st = wait_clean("trt_ts_bs8", log)
    r = cuda_time(lambda: mod(lat8, aud8), warmup=10, iters=30)
    r["ms_per_frame"] = r["median_ms"] / 8
    r["TFLOPS"] = tflops(r["median_ms"], 8)
    r["contaminated"] = st["contaminated"]
    res["ts_bs8"] = r
    print("ts bs8", r, flush=True)

    # ---- bs16 as 2 x bs8 (as MultiTrtUnetBackend does, incl. slicing + cat) ----
    def two_calls():
        o1 = mod(lat16[:8], aud16[:8])
        o2 = mod(lat16[8:], aud16[8:])
        return torch.cat([o1, o2], 0)
    r = cuda_time(two_calls, warmup=10, iters=30)
    r["ms_per_frame"] = r["median_ms"] / 16
    res["ts_bs16_as_2x8"] = r
    print("ts 2x8", r, flush=True)

    # ---- host enqueue time vs GPU time ----
    torch.cuda.synchronize()
    host = []
    for _ in range(30):
        torch.cuda.synchronize()
        t = time.perf_counter()
        mod(lat8, aud8)
        host.append((time.perf_counter() - t) * 1e3)
        torch.cuda.synchronize()
    host.sort()
    # back-to-back 30 calls: one event pair -> includes any inter-call GPU gaps
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    s.record()
    for _ in range(30):
        mod(lat8, aud8)
    e.record()
    t_enq = (time.perf_counter() - t) * 1e3
    e.synchronize()
    res["ts_host"] = {"enqueue_ms_median": host[15], "enqueue_ms_min": host[0], "enqueue_ms_max": host[-1],
                      "back_to_back_30_gpu_ms_per_call": s.elapsed_time(e) / 30,
                      "back_to_back_30_host_enqueue_ms_per_call": t_enq / 30}
    print("host", res["ts_host"], flush=True)

    # ---- two CUDA streams, same module ----
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def two_streams():
        with torch.cuda.stream(s1):
            mod(lat8, aud8)
        with torch.cuda.stream(s2):
            mod(lat16[8:], aud16[8:])
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
    r = cuda_time(two_streams, warmup=5, iters=30)
    res["ts_two_streams_same_module"] = r
    # kernel-level overlap check with profiler
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            two_streams()
        torch.cuda.synchronize()
    kev = [k for k in prof.profiler.kineto_results.events() if k.device_type() == torch.autograd.DeviceType.CUDA]
    by_stream = collections.defaultdict(list)
    for k in kev:
        by_stream[k.device_resource_id()].append((k.start_ns(), k.start_ns() + k.duration_ns()))
    streams = sorted(by_stream, key=lambda x: -len(by_stream[x]))
    ov = None
    if len(streams) >= 2:
        a, b = by_stream[streams[0]], by_stream[streams[1]]
        a.sort(); b.sort()
        i = j = 0
        ov = 0
        while i < len(a) and j < len(b):
            lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
            if hi > lo:
                ov += hi - lo
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
    res["ts_two_streams_profile"] = {
        "kernels_per_stream": {str(k): len(v) for k, v in by_stream.items()},
        "overlap_ms_total_3iters": None if ov is None else ov / 1e6,
        "busy_ms_per_stream": {str(k): sum(x[1] - x[0] for x in v) / 1e6 for k, v in by_stream.items()},
    }
    print("two streams", r, res["ts_two_streams_profile"], flush=True)

    # ---- CUDA graph capture of the torch_tensorrt call ----
    try:
        g = torch.cuda.CUDAGraph()
        static_lat, static_aud = lat8.clone(), aud8.clone()
        sw = torch.cuda.Stream()
        sw.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sw):
            for _ in range(3):
                mod(static_lat, static_aud)
        torch.cuda.current_stream().wait_stream(sw)
        with torch.cuda.graph(g):
            static_out = mod(static_lat, static_aud)
        g.replay(); torch.cuda.synchronize()
        ref = mod(static_lat, static_aud)
        so = static_out[0] if isinstance(static_out, (list, tuple)) else static_out
        rf = ref[0] if isinstance(ref, (list, tuple)) else ref
        r = cuda_time(lambda: g.replay(), warmup=10, iters=30)
        r["max_abs_vs_eager_call"] = (so.float() - rf.float()).abs().max().item()
        res["ts_cudagraph_torch"] = r
        print("cudagraph torch", r, flush=True)
    except Exception as ex:
        res["ts_cudagraph_torch"] = {"error": repr(ex)[:600]}
        print("cudagraph torch failed", repr(ex)[:300], flush=True)
    # torch_tensorrt runtime cudagraph mode
    try:
        torch_tensorrt.runtime.set_cudagraphs_mode(True)
        for _ in range(3):
            mod(lat8, aud8)
        r = cuda_time(lambda: mod(lat8, aud8), warmup=10, iters=30)
        res["ts_runtime_cudagraphs_mode"] = r
        print("runtime cg mode", r, flush=True)
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
    except Exception as ex:
        res["ts_runtime_cudagraphs_mode"] = {"error": repr(ex)[:600]}
        print("runtime cg mode failed", repr(ex)[:300], flush=True)
save_json("trt.json", res)

# ======================= raw TensorRT on the same engine bytes =======================
del mod
torch.cuda.synchronize(); torch.cuda.empty_cache()
if engine_state is None:
    print("no engine state; stop")
    raise SystemExit
import tensorrt as trt  # noqa: E402

blob = None
for x in engine_state:
    if isinstance(x, (bytes, str)) and len(x) > 10_000_000:
        blob = x if isinstance(x, bytes) else x.encode("latin-1") if False else x
        break
if isinstance(blob, str):
    # torch_tensorrt stores the serialized engine as a std::string; pybind gives str or bytes
    try:
        blob = blob.encode("latin-1")
    except Exception:
        blob = bytes(blob, "utf-8", "surrogateescape")
logger = trt.Logger(trt.Logger.WARNING)
rt = trt.Runtime(logger)
engine = rt.deserialize_cuda_engine(blob)
del blob, engine_state
res["raw_engine_ok"] = engine is not None
if engine is None:
    save_json("trt.json", res)
    raise SystemExit
names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
io = {n: (engine.get_tensor_mode(n).name, list(engine.get_tensor_shape(n)), str(engine.get_tensor_dtype(n))) for n in names}
res["raw_io"] = io
res["raw_device_memory_MB"] = engine.device_memory_size_v2 / 2**20 if hasattr(engine, "device_memory_size_v2") else engine.device_memory_size / 2**20
print("raw io", io, res["raw_device_memory_MB"], flush=True)

# inspector
try:
    insp = engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", [])
    res["inspector_num_layers"] = len(layers)
    res["inspector_sample"] = layers[:5]
    # summarise tactic names / precisions if present
    tac = collections.Counter()
    prec = collections.Counter()
    for L in layers:
        if isinstance(L, dict):
            tac[str(L.get("TacticName", L.get("LayerType", "?")))[:80]] += 1
            for inp in L.get("Inputs", []):
                prec[str(inp.get("Format/Datatype", "?"))[:40]] += 1
        else:
            tac["name_only"] += 1
    res["inspector_tactics_top"] = tac.most_common(40)
    res["inspector_input_formats"] = prec.most_common(20)
except Exception as ex:
    res["inspector_error"] = repr(ex)[:400]

ctx1 = engine.create_execution_context()
ctx2 = engine.create_execution_context()
in_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
out_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
print("in", in_names, "out", out_names, flush=True)


def bind(ctx, lat, aud):
    outs = {}
    # map inputs by shape
    for n in in_names:
        shp = list(engine.get_tensor_shape(n))
        t_ = lat if len(shp) == 4 else aud
        ctx.set_input_shape(n, tuple(t_.shape))
        ctx.set_tensor_address(n, t_.data_ptr())
    for n in out_names:
        shp = list(ctx.get_tensor_shape(n))
        o = torch.empty(shp, device="cuda", dtype=torch.float16)
        ctx.set_tensor_address(n, o.data_ptr())
        outs[n] = o
    return outs

la, aa = lat16[:8].contiguous(), aud16[:8].contiguous()
lb, ab = lat16[8:].contiguous(), aud16[8:].contiguous()
o1 = bind(ctx1, la, aa)
o2 = bind(ctx2, lb, ab)
sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
cur = torch.cuda.current_stream()


def raw1():
    ctx1.execute_async_v3(cur.cuda_stream)

st = wait_clean("raw_bs8", log)
r = cuda_time(raw1, warmup=10, iters=30)
r["TFLOPS"] = tflops(r["median_ms"], 8)
r["contaminated"] = st["contaminated"]
res["raw_bs8"] = r
host = []
for _ in range(30):
    torch.cuda.synchronize()
    t = time.perf_counter()
    raw1()
    host.append((time.perf_counter() - t) * 1e3)
torch.cuda.synchronize()
host.sort()
res["raw_host_enqueue_ms_median"] = host[15]


def raw_seq():
    ctx1.execute_async_v3(cur.cuda_stream)
    ctx2.execute_async_v3(cur.cuda_stream)
r = cuda_time(raw_seq, warmup=5, iters=30)
res["raw_2x8_sequential"] = r


def raw_par():
    sA.wait_stream(cur); sB.wait_stream(cur)
    ctx1.execute_async_v3(sA.cuda_stream)
    ctx2.execute_async_v3(sB.cuda_stream)
    cur.wait_stream(sA); cur.wait_stream(sB)
r = cuda_time(raw_par, warmup=5, iters=30)
res["raw_2x8_two_streams_two_contexts"] = r
print("raw", res["raw_bs8"], res["raw_2x8_sequential"], r, flush=True)

# overlap proof via profiler
from torch.profiler import ProfilerActivity, profile  # noqa
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        raw_par()
    torch.cuda.synchronize()
kev = [k for k in prof.profiler.kineto_results.events() if k.device_type() == torch.autograd.DeviceType.CUDA]
by_stream = collections.defaultdict(list)
for k in kev:
    by_stream[k.device_resource_id()].append((k.start_ns(), k.start_ns() + k.duration_ns()))
streams = sorted(by_stream, key=lambda x: -len(by_stream[x]))
ov = None
if len(streams) >= 2:
    a, b = sorted(by_stream[streams[0]]), sorted(by_stream[streams[1]])
    i = j = 0; ov = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            ov += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
res["raw_two_streams_profile"] = {"kernels_per_stream": {str(k): len(v) for k, v in by_stream.items()},
                                  "overlap_ms_total_3iters": None if ov is None else ov / 1e6,
                                  "busy_ms_per_stream": {str(k): sum(x[1] - x[0] for x in v) / 1e6 for k, v in by_stream.items()},
                                  "num_kernels_per_bs8_call": len(kev) / 6}

# CUDA graph of raw enqueueV3
try:
    gs = torch.cuda.Stream()
    with torch.cuda.stream(gs):
        ctx1.execute_async_v3(gs.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=gs):
        ctx1.execute_async_v3(gs.cuda_stream)
    g.replay(); torch.cuda.synchronize()
    r = cuda_time(lambda: g.replay(), warmup=10, iters=30)
    ref = o1[out_names[0]].clone()
    raw1(); torch.cuda.synchronize()
    r["max_abs_graph_vs_direct"] = (ref.float() - o1[out_names[0]].float()).abs().max().item()
    res["raw_cudagraph_bs8"] = r
    # two graphs from the two contexts, replayed on two streams
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(gs):
        ctx2.execute_async_v3(gs.cuda_stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(g2, stream=gs):
        ctx2.execute_async_v3(gs.cuda_stream)

    def gpar():
        sA.wait_stream(cur); sB.wait_stream(cur)
        with torch.cuda.stream(sA):
            g.replay()
        with torch.cuda.stream(sB):
            g2.replay()
        cur.wait_stream(sA); cur.wait_stream(sB)
    r = cuda_time(gpar, warmup=5, iters=30)
    res["raw_cudagraph_2x8_two_streams"] = r
    host = []
    for _ in range(30):
        torch.cuda.synchronize()
        t = time.perf_counter(); g.replay(); host.append((time.perf_counter() - t) * 1e3)
    torch.cuda.synchronize(); host.sort()
    res["raw_cudagraph_host_replay_ms_median"] = host[15]
    print("raw cg", res["raw_cudagraph_bs8"], r, flush=True)
except Exception as ex:
    res["raw_cudagraph_bs8"] = {"error": traceback.format_exc()[-800:]}
    print("raw cg failed", traceback.format_exc()[-500:], flush=True)

res["peak_reserved_GB"] = torch.cuda.max_memory_reserved() / 2**30
save_json("trt.json", res)
print("DONE")
