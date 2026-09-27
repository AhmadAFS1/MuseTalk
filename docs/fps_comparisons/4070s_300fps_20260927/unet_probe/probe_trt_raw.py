"""Probe 3b: raw TensorRT on the shipping engine bytes (2 contexts, shared weights), inspector, CUDA graphs."""
import collections, json, time, traceback
import torch
from common import batch_inputs, cuda_time, save_json, wait_clean
log = []
res = {"gpu_log": log}
FLOP_PER_FRAME = json.load(open("eager.json"))["flops"]["per_frame_GFLOP"] * 1e9
def tflops(ms, frames):
    return FLOP_PER_FRAME * frames / (ms / 1e3) / 1e12
import torch_tensorrt
mod = torch_tensorrt.load("/workspace/MuseTalk/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts")
st0 = mod._run_on_acc_0.engine.__getstate__()
info_list = st0[0]
res["engine_info_types"] = [(i, type(x).__name__, len(x) if hasattr(x, "__len__") else x) for i, x in enumerate(info_list)]
res["engine_info_small"] = {i: x for i, x in enumerate(info_list) if hasattr(x, "__len__") and len(x) < 400}
print(res["engine_info_types"], flush=True)
engine_state = info_list
lat16, aud16 = batch_inputs(16)
# ======================= raw TensorRT on the same engine bytes =======================
del mod
torch.cuda.synchronize(); torch.cuda.empty_cache()
if engine_state is None:
    print("no engine state; stop")
    raise SystemExit
import tensorrt as trt  # noqa: E402

blob = max((x for x in engine_state if isinstance(x, (bytes, str))), key=len)
print("blob type", type(blob), len(blob), flush=True)
if isinstance(blob, str):
    import base64
    try:
        blob = base64.b64decode(blob, validate=True)
    except Exception:
        blob = blob.encode("utf-8", "surrogateescape")
logger = trt.Logger(trt.Logger.WARNING)
rt = trt.Runtime(logger)
engine = rt.deserialize_cuda_engine(blob)
del blob, engine_state, st0, info_list
res["raw_engine_ok"] = engine is not None
if engine is None:
    save_json("trt_raw.json", res)
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
save_json("trt_raw.json", res)
print("DONE")
