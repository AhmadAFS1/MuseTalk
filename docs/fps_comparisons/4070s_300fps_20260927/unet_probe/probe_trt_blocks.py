"""Probe 4b: tiny in-memory TensorRT 10.3 engines for representative UNet blocks (real weights, real activations):
FP16 vs INT8 Q/DQ vs FP8 Q/DQ (modelopt), ONNX kept in RAM only; per-layer tactics via IEngineInspector and per-kernel
CUPTI names; plus torch_tensorrt (the production converter path) FP16 for the same blocks."""
import collections
import copy
import io
import json
import os
import sys
import time
import traceback

import torch
import torch.nn as nn

from common import OUT, batch_inputs, load_unet_fp16, save_json, wait_clean

import tensorrt as trt  # noqa: E402

ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
log = []
res = {"gpu_log": log, "blocks": {}}
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
RUNTIME = trt.Runtime(TRT_LOGGER)

unet = load_unet_fp16()
m = unet.model
dev = torch.device("cuda")
with torch.no_grad():
    t_emb = m.time_proj(torch.tensor([0], device=dev)).to(torch.float16)
    EMB = m.time_embedding(t_emb).detach()  # [1,1280] constant (MuseTalk timestep is always 0)

targets = {
    "res320_32x32": m.down_blocks[0].resnets[0],
    "res1280_8x8": m.down_blocks[2].resnets[1],
    "res2560to1280_8x8(up1)": m.up_blocks[1].resnets[0],
    "tf320_1024tok": m.down_blocks[0].attentions[0],
    "tf1280_64tok": m.up_blocks[1].attentions[0],
}
if ONLY:
    targets = {k: v for k, v in targets.items() if any(o in k for o in ONLY)}
cap = {}


def mk(name):
    def h(mod, args, kwargs):
        if name not in cap:
            x = args[0] if args else kwargs["hidden_states"]
            e = kwargs.get("encoder_hidden_states")
            cap[name] = (x.detach().clone(), None if e is None else e.detach().clone())
    return h


hs = [mod.register_forward_pre_hook(mk(n), with_kwargs=True) for n, mod in targets.items()]
lat, aud = batch_inputs(16)
with torch.no_grad():
    m(lat, torch.tensor([0], device=dev), encoder_hidden_states=aud)
for h in hs:
    h.remove()
blocks = {n: copy.deepcopy(mod).eval() for n, mod in targets.items()}
del unet, m, targets
torch.cuda.empty_cache()


class ResW(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
        self.register_buffer("emb", EMB.clone())

    def forward(self, x):
        return self.r(x, self.emb)


class TfW(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, ehs):
        return self.t(x, encoder_hidden_states=ehs, return_dict=False)[0]


def gpu_ms(fn, warmup=10, iters=30, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / reps)
    ts.sort()
    return ts[len(ts) // 2]


def export_onnx(model, args, names, opset=17):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, args, f, opset_version=opset, input_names=names, output_names=["out"],
                          do_constant_folding=True)
    b = f.getvalue()
    mp = __import__("onnx").load_from_string(b)
    ops = collections.Counter(n.op_type for n in mp.graph.node)
    return b, dict(ops.most_common())


def build_engine(onnx_bytes, flags):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_bytes):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("parse failed: " + " | ".join(errs)[:800])
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    for fl in flags:
        cfg.set_flag(fl)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t = time.time()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        raise RuntimeError("build failed")
    eng = RUNTIME.deserialize_cuda_engine(ser)
    return eng, time.time() - t, ser.nbytes


def inspect(eng):
    insp = eng.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    out = []
    for L in info.get("Layers", []):
        if isinstance(L, str):
            out.append({"name": L[:120]})
            continue
        ins = [str(i.get("Format/Datatype", "")) for i in L.get("Inputs", [])]
        outs = [str(i.get("Format/Datatype", "")) for i in L.get("Outputs", [])]
        out.append({"type": L.get("LayerType"), "tactic": str(L.get("TacticName", ""))[:150],
                    "in": ins[:3], "out": outs[:1], "name": str(L.get("Name", ""))[:100]})
    return out


def run_engine(eng, inputs):
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    bufs = {}
    ii = 0
    for n in names:
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            t = inputs[ii].contiguous(); ii += 1
            dt = eng.get_tensor_dtype(n)
            if dt == trt.DataType.FLOAT:
                t = t.float()
            bufs[n] = t
            ctx.set_tensor_address(n, t.data_ptr())
    for n in names:
        if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            dt = torch.float32 if eng.get_tensor_dtype(n) == trt.DataType.FLOAT else torch.float16
            o = torch.empty(tuple(ctx.get_tensor_shape(n)), device="cuda", dtype=dt)
            bufs[n] = o
            ctx.set_tensor_address(n, o.data_ptr())
    s = torch.cuda.Stream()
    fn = lambda: ctx.execute_async_v3(s.cuda_stream)  # noqa: E731

    def timed():
        cur = torch.cuda.current_stream()
        s.wait_stream(cur)
        fn()
        cur.wait_stream(s)
    ms = gpu_ms(timed)
    # kernel names
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            timed()
        torch.cuda.synchronize()
    kn = collections.defaultdict(float)
    for k in prof.profiler.kineto_results.events():
        if k.device_type() == torch.autograd.DeviceType.CUDA:
            kn[k.name()[:140]] += k.duration_ns() / 1e6 / 3
    timed(); torch.cuda.synchronize()
    return ms, bufs["out"].clone(), dict(sorted(kn.items(), key=lambda x: -x[1])[:12]), ctx


def calib_loop_factory(model, inputs_list):
    def loop(mm):
        with torch.no_grad():
            for args in inputs_list:
                mm(*args)
    return loop


import modelopt.torch.quantization as mtq  # noqa: E402

try:
    import torch_tensorrt  # noqa: F401
    HAVE_TTRT = True
except Exception:
    HAVE_TTRT = False

for bname, blk in blocks.items():
    x16, e16 = cap[bname]
    is_tf = bname.startswith("tf")
    for bs in (8, 16):
        key = f"{bname}_bs{bs}"
        if bs == 8:
            x, e = x16[:8].contiguous(), (None if e16 is None else e16[:8].contiguous())
        else:
            x, e = x16.contiguous(), (None if e16 is None else e16.contiguous())
        args = (x, e) if is_tf else (x,)
        names = ["x", "ehs"] if is_tf else ["x"]
        r = {"input_shape": list(x.shape)}
        base = (TfW(blk) if is_tf else ResW(blk)).half().cuda().eval()
        with torch.no_grad():
            from torch.utils.flop_counter import FlopCounterMode
            fc = FlopCounterMode(display=False)
            with fc:
                ref = base(*args)
            r["GFLOP"] = fc.get_total_flops() / 1e9
            st = wait_clean(key, log)
            r["contaminated"] = st["contaminated"]
            r["pytorch_eager_ms"] = gpu_ms(lambda: base(*args))
        # ---- torch_tensorrt FP16 (production converter path) ----
        if HAVE_TTRT:
            try:
                tt = torch_tensorrt.compile(copy.deepcopy(base), ir="dynamo", inputs=list(args),
                                            enabled_precisions={torch.float16}, min_block_size=1,
                                            timing_cache_path=os.path.join(OUT, "tt_timing_cache.bin"))
                with torch.no_grad():
                    r["torch_tensorrt_fp16_ms"] = gpu_ms(lambda: tt(*args))
                    r["torch_tensorrt_fp16_maxabs"] = (tt(*args).float() - ref.float()).abs().max().item()
                del tt
            except Exception as ex:
                r["torch_tensorrt_fp16_error"] = repr(ex)[:400]
            torch.cuda.empty_cache()
        # ---- ONNX -> TRT FP16 / INT8 QDQ / FP8 QDQ ----
        variants = [("fp16", None, [trt.BuilderFlag.FP16], None),
                    ("int8_qdq", mtq.INT8_DEFAULT_CFG, [trt.BuilderFlag.FP16, trt.BuilderFlag.INT8], None),
                    ("fp8_qdq_perchannelW", mtq.INT8_DEFAULT_CFG, [trt.BuilderFlag.FP16, trt.BuilderFlag.FP8], "perchannel"),
                    ("fp8_qdq_pertensorW", mtq.INT8_DEFAULT_CFG, [trt.BuilderFlag.FP16, trt.BuilderFlag.FP8], "pertensor")]
        # FP8 fake-quant accuracy in torch (modelopt FP8_DEFAULT_CFG; no export)
        try:
            mm = mtq.quantize(copy.deepcopy(base), mtq.FP8_DEFAULT_CFG, calib_loop_factory(None, [
                tuple(a[i:i + bs] if a is not None else None for a in ((x16, e16) if is_tf else (x16,))) for i in range(0, 16, bs)]))
            with torch.no_grad():
                o8 = mm(*args).float()
            r["fp8_fakequant_torch"] = {"maxabs": (o8 - ref.float()).abs().max().item(),
                                        "rel_l2": ((o8 - ref.float()).norm() / ref.float().norm()).item()}
            del mm
        except Exception as ex:
            r["fp8_fakequant_torch"] = {"error": repr(ex)[:300]}
        for vname, qcfg, flags, fp8mode in variants:
            v = {}
            try:
                mm = copy.deepcopy(base)
                if qcfg is not None:
                    calib = [tuple(a[i:i + bs] if a is not None else None for a in ((x16, e16) if is_tf else (x16,)))
                             for i in range(0, 16, bs)]
                    mm = mtq.quantize(mm, qcfg, calib_loop_factory(mm, calib))
                    with torch.no_grad():
                        v["fakequant_torch_maxabs_vs_fp16"] = (mm(*args).float() - ref.float()).abs().max().item()
                ob, ops = export_onnx(mm, args, names, opset=19 if fp8mode else 17)
                if fp8mode:
                    from qdq_fp8 import int8_qdq_to_fp8
                    ob, nq = int8_qdq_to_fp8(ob, per_tensor_weights=(fp8mode == "pertensor"))
                    v["qdq_nodes_converted"] = nq
                v["onnx_MB_in_ram"] = len(ob) / 2**20
                v["onnx_ops"] = ops
                del mm
                eng, bt, nbytes = build_engine(ob, flags)
                del ob
                v["build_s"] = bt
                v["engine_MB"] = nbytes / 2**20
                v["layers"] = inspect(eng)
                ms, out, kern, ctx = run_engine(eng, list(args))
                v["trt_ms"] = ms
                v["TFLOPS"] = r["GFLOP"] / ms
                v["maxabs_vs_torch_fp16"] = (out.float() - ref.float()).abs().max().item()
                v["rel_l2_vs_torch_fp16"] = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                v["top_kernels_ms"] = kern
                del ctx, eng
            except Exception:
                v["error"] = traceback.format_exc()[-1200:]
            torch.cuda.empty_cache()
            r[vname] = v
            print(key, vname, {k: v.get(k) for k in ("trt_ms", "TFLOPS", "maxabs_vs_torch_fp16", "rel_l2_vs_torch_fp16", "error")},
                  flush=True)
        print(key, "eager", r["pytorch_eager_ms"], "torch_tensorrt", r.get("torch_tensorrt_fp16_ms"), flush=True)
        res["blocks"][key] = r
        save_json("trt_blocks.json" if not ONLY else f"trt_blocks_{'_'.join(ONLY)}.json", res)
print("DONE")
