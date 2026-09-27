"""Probe 4d: whole-UNet estimate via per-top-level-block TensorRT engines built from in-RAM ONNX (never on disk):
FP16 (ONNX parser path, GroupNorm via InstanceNorm -> Myelin fused) vs INT8 Q/DQ (modelopt max calib, real activations).
Sum over blocks ~= full UNet (slightly pessimistic: block-boundary reformat/launch)."""
import copy
import gc
import io
import sys
import time
import traceback

import torch
import torch.nn as nn

from common import batch_inputs, load_captures, load_unet_fp16_lowmem, save_json, wait_clean
import tensorrt as trt
import modelopt.torch.quantization as mtq

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PRECS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fp16", "int8"]
log = []
res = {"gpu_log": log, "bs": BS, "blocks": {}}
L = trt.Logger(trt.Logger.ERROR)
RT = trt.Runtime(L)
dev = torch.device("cuda")
model, _ = load_unet_fp16_lowmem()
t0 = torch.tensor([0], device=dev)
with torch.no_grad():
    EMB = model.time_embedding(model.time_proj(t0).half()).detach()


class Down(nn.Module):
    def __init__(self, b, cross):
        super().__init__(); self.b = b; self.cross = cross; self.register_buffer("emb", EMB.clone())

    def forward(self, h, ehs):
        if self.cross:
            h, rs = self.b(hidden_states=h, temb=self.emb, encoder_hidden_states=ehs)
        else:
            h, rs = self.b(hidden_states=h, temb=self.emb)
        return tuple(rs)  # rs[-1] is h itself; returning it twice duplicates an ONNX output


class Mid(nn.Module):
    def __init__(self, b):
        super().__init__(); self.b = b; self.register_buffer("emb", EMB.clone())

    def forward(self, h, ehs):
        return self.b(h, self.emb, encoder_hidden_states=ehs)


class Up(nn.Module):
    def __init__(self, b, cross):
        super().__init__(); self.b = b; self.cross = cross; self.register_buffer("emb", EMB.clone())

    def forward(self, h, r0, r1, r2, ehs):
        if self.cross:
            return self.b(hidden_states=h, res_hidden_states_tuple=(r0, r1, r2), temb=self.emb, encoder_hidden_states=ehs)
        return self.b(hidden_states=h, res_hidden_states_tuple=(r0, r1, r2), temb=self.emb)


class Head(nn.Module):
    def __init__(self, m):
        super().__init__(); self.c = m.conv_in

    def forward(self, x, ehs):
        return self.c(x)


class Tail(nn.Module):
    def __init__(self, m):
        super().__init__(); self.n = m.conv_norm_out; self.a = m.conv_act; self.c = m.conv_out

    def forward(self, h, ehs):
        return self.c(self.a(self.n(h)))


# capture real block inputs by running the UNet body manually (mirrors UNet2DConditionModel.forward)
lat_all, aud_all, _ = load_captures()
lat_all, aud_all = lat_all.to(dev).half(), aud_all.to(dev).half()


@torch.no_grad()
def trace_inputs(lat, aud):
    ins = {}
    ins["head"] = (lat, aud)
    h = model.conv_in(lat)
    res_s = (h,)
    for i, b in enumerate(model.down_blocks):
        ins[f"down{i}"] = (h, aud)
        cross = hasattr(b, "has_cross_attention") and b.has_cross_attention
        if cross:
            h, rs = b(hidden_states=h, temb=EMB, encoder_hidden_states=aud)
        else:
            h, rs = b(hidden_states=h, temb=EMB)
        res_s += rs
    ins["mid"] = (h, aud)
    h = model.mid_block(h, EMB, encoder_hidden_states=aud)
    for i, b in enumerate(model.up_blocks):
        rs = res_s[-len(b.resnets):]
        res_s = res_s[: -len(b.resnets)]
        ins[f"up{i}"] = (h, *rs, aud)
        cross = hasattr(b, "has_cross_attention") and b.has_cross_attention
        if cross:
            h = b(hidden_states=h, res_hidden_states_tuple=rs, temb=EMB, encoder_hidden_states=aud)
        else:
            h = b(hidden_states=h, res_hidden_states_tuple=rs, temb=EMB)
    ins["tail"] = (h, aud)
    out = model.conv_out(model.conv_act(model.conv_norm_out(h)))
    return ins, out


idx = torch.arange(BS) % lat_all.shape[0]
ins, out_manual = trace_inputs(lat_all[idx], aud_all[idx])
with torch.no_grad():
    out_ref = model(lat_all[idx], t0, encoder_hidden_states=aud_all[idx]).sample
res["manual_trace_matches_forward_maxabs"] = (out_manual - out_ref).abs().max().item()
# calibration inputs: 4 other batches
cal_ins = []
for k in range(4):
    j = (torch.arange(BS) + BS * (k + 1)) % lat_all.shape[0]
    cal_ins.append(trace_inputs(lat_all[j], aud_all[j])[0])

wrappers = {"head": Head(model), "tail": Tail(model), "mid": Mid(model.mid_block)}
for i, b in enumerate(model.down_blocks):
    wrappers[f"down{i}"] = Down(b, getattr(b, "has_cross_attention", False))
for i, b in enumerate(model.up_blocks):
    wrappers[f"up{i}"] = Up(b, getattr(b, "has_cross_attention", False))
order = ["head", "down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "tail"]


def gpu_ms(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def build(ob, flags):
    b = trt.Builder(L)
    n = b.create_network(0)
    p = trt.OnnxParser(n, L)
    if not p.parse(ob):
        raise RuntimeError("parse: " + " | ".join(str(p.get_error(i)) for i in range(p.num_errors))[:600])
    c = b.create_builder_config()
    c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    for f in flags:
        c.set_flag(f)
    t = time.time()
    ser = b.build_serialized_network(n, c)
    del n, p
    if ser is None:
        raise RuntimeError("build failed")
    e = RT.deserialize_cuda_engine(ser)
    del ser
    return e, time.time() - t


def time_engine(e, args):
    ctx = e.create_execution_context()
    names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
    ii = 0
    keep = []
    outs = []
    for nme in names:
        if e.get_tensor_mode(nme) == trt.TensorIOMode.INPUT:
            t = args[ii].contiguous(); ii += 1
            keep.append(t); ctx.set_tensor_address(nme, t.data_ptr())
    for nme in names:
        if e.get_tensor_mode(nme) == trt.TensorIOMode.OUTPUT:
            o = torch.empty(tuple(ctx.get_tensor_shape(nme)), device="cuda",
                            dtype=torch.float32 if e.get_tensor_dtype(nme) == trt.DataType.FLOAT else torch.float16)
            outs.append(o); ctx.set_tensor_address(nme, o.data_ptr())
    s = torch.cuda.Stream()

    def f():
        cur = torch.cuda.current_stream(); s.wait_stream(cur)
        ctx.execute_async_v3(s.cuda_stream); cur.wait_stream(s)
    ms = gpu_ms(f)
    f(); torch.cuda.synchronize()
    return ms, outs, ctx, keep


import json as _json, os as _os
_prev = f"trt_topblocks_bs{BS}.json"
if _os.path.exists(_prev):
    _old = _json.load(open(_prev))
    res["blocks"] = {k: v for k, v in _old.get("blocks", {}).items() if all(p in v for p in PRECS)}
    res["gpu_log_prev"] = _old.get("gpu_log", [])
    print("resuming; done blocks:", list(res["blocks"]), flush=True)
st = wait_clean(f"topblocks_bs{BS}", log)
res["contaminated_start"] = st["contaminated"]
for name in order:
    if name in res["blocks"]:
        continue
    w = wrappers[name].eval()
    args = ins[name]
    r = {}
    with torch.no_grad():
        ref = w(*args)
        ref0 = ref[0] if isinstance(ref, tuple) else ref
        r["pytorch_eager_ms"] = gpu_ms(lambda: w(*args))
    nin = len(args)
    in_names = [f"i{k}" for k in range(nin)]
    for prec in PRECS:
        v = {}
        try:
            if prec == "fp16":
                mm = w
                flags = [trt.BuilderFlag.FP16]
            else:
                if name in ("head",):
                    v["skipped"] = "tiny conv_in (8ch input) kept fp16"
                    r[prec] = v
                    continue
                mm = copy.deepcopy(w)
                def _loop(q, _n=name):
                    with torch.no_grad():
                        for ci in cal_ins:
                            q(*ci[_n])
                mm = mtq.quantize(mm, mtq.INT8_DEFAULT_CFG, _loop)
                flags = [trt.BuilderFlag.FP16, trt.BuilderFlag.INT8]
            f = io.BytesIO()
            with torch.no_grad():
                torch.onnx.export(mm, args, f, opset_version=17, input_names=in_names, do_constant_folding=True)
            ob = f.getvalue(); del f
            v["onnx_MB_in_ram"] = len(ob) / 2**20
            if mm is not w:
                del mm
            gc.collect(); torch.cuda.empty_cache()
            e, bt = build(ob, flags)
            del ob; gc.collect()
            v["build_s"] = bt
            ms, outs, ctx, keep = time_engine(e, list(args))
            v["trt_ms"] = ms
            o0 = outs[0].float()
            v["out0_rel_l2_vs_torch"] = ((o0 - ref0.float()).norm() / ref0.float().norm()).item()
            del ctx, e, outs, keep
        except Exception:
            v["error"] = traceback.format_exc()[-900:]
        gc.collect(); torch.cuda.empty_cache()
        r[prec] = v
        print(name, prec, {k: v.get(k) for k in ("trt_ms", "build_s", "out0_rel_l2_vs_torch", "error", "skipped")}, flush=True)
    r["eager_ms"] = r["pytorch_eager_ms"]
    res["blocks"][name] = r
    save_json(f"trt_topblocks_bs{BS}.json", res)
for prec in PRECS:
    tot = 0.0
    ok = True
    for name in order:
        v = res["blocks"][name].get(prec, {})
        if "trt_ms" in v:
            tot += v["trt_ms"]
        elif prec != "fp16" and "trt_ms" in res["blocks"][name].get("fp16", {}):
            tot += res["blocks"][name]["fp16"]["trt_ms"]  # fall back to fp16 time for skipped/failed block
        else:
            ok = False
    res[f"sum_{prec}_ms"] = tot
    res[f"sum_{prec}_complete"] = ok
res["sum_eager_ms"] = sum(res["blocks"][n]["pytorch_eager_ms"] for n in order)
st = wait_clean(f"topblocks_bs{BS}_end", log)
res["contaminated_end"] = st["contaminated"]
print({k: v for k, v in res.items() if k.startswith("sum")})
save_json(f"trt_topblocks_bs{BS}.json", res)
print("DONE")
