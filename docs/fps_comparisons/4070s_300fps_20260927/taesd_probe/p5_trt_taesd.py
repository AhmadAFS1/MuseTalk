"""Probe 5: TensorRT FP16 TAESD (in-memory ONNX -> engine, nothing serialized to disk).

Variants: full decode and the bit-exact staged row-crop (rows >= 104 only), bs 8 and 16,
plus a variant that emits uint8 BGR NHWC directly (fused post-process).
"""
import sys, io, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch.nn as nn
import tensorrt as trt

LOG = trt.Logger(trt.Logger.WARNING)
model = load_taesd()
dec = model.decoder
layers = list(dec.layers)
FIRST = {6: 9, 11: 39, 16: 99}  # from p2 (exact first-row needed at post-upsample stage inputs)


class Full(nn.Module):
    def forward(self, z):
        return ((dec(z) / 2 + 0.5).clamp(0, 1))


class Staged(nn.Module):
    def __init__(self, crop_at=(6, 11, 16)):
        super().__init__(); self.crop_at = crop_at

    def forward(self, z):
        x = torch.tanh(z / 3) * 3
        offset = 0
        for i, L in enumerate(layers):
            if i in self.crop_at:
                cut = FIRST[i] - offset
                if cut > 0:
                    x = x[:, :, cut:, :]; offset = FIRST[i]
            x = L(x)
            if isinstance(L, nn.Upsample):
                offset *= 2
        x = (x.mul(2).sub(1) / 2 + 0.5).clamp(0, 1)
        return x[:, :, USED_Y0 - offset:, :]


class StagedU8(Staged):
    def forward(self, z):
        x = super().forward(z)
        # BGR NHWC 0..255 (as float16 rounded; TRT casts to uint8 below via output dtype)
        x = (x * 255).round().flip(1).permute(0, 2, 3, 1)
        return x


def build(module, bs, name, opt_level=3, out_u8=False):
    z = torch.randn(bs, 4, 32, 32, device=DEV, dtype=torch.float16)
    f = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(module, (z,), f, input_names=["z"], output_names=["y"], opset_version=17,
                          do_constant_folding=True)
    onnx_bytes = f.getvalue()
    builder = trt.Builder(LOG)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, LOG)
    assert parser.parse(onnx_bytes), [parser.get_error(i) for i in range(parser.num_errors)]
    if out_u8:
        y = net.get_output(0)
        net.unmark_output(y)
        c = net.add_cast(y, trt.DataType.UINT8)
        c.get_output(0).name = "y_u8"
        net.mark_output(c.get_output(0))
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    cfg.builder_optimization_level = opt_level
    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    build_s = time.time() - t0
    rt = trt.Runtime(LOG)
    eng = rt.deserialize_cuda_engine(plan)
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    out_name = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
    out_shape = tuple(eng.get_tensor_shape(out_name))
    out_dtype = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
                 trt.DataType.UINT8: torch.uint8}[eng.get_tensor_dtype(out_name)]
    zin = torch.empty(bs, 4, 32, 32, device=DEV, dtype=torch.float16)
    yout = torch.empty(out_shape, device=DEV, dtype=out_dtype)
    ctx.set_tensor_address("z", zin.data_ptr())
    ctx.set_tensor_address(out_name, yout.data_ptr())
    print(f"built {name} bs{bs}: onnx {len(onnx_bytes)/1e6:.1f} MB, plan {plan.nbytes/1e6:.1f} MB, "
          f"build {build_s:.0f}s, out {out_shape} {out_dtype}", flush=True)
    del plan

    def run(zz=None):
        if zz is not None:
            zin.copy_(zz)
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return yout
    return run, build_s, eng, ctx, zin, yout

if __name__ == '__main__':
    Z = real_latents(128).to(DEV)
    res = {"gpu_start": gpu_state("start"), "trt": trt.__version__, "results": {}}
    with torch.inference_mode():
        ref_full = {bs: Full()(Z[:bs]).float() for bs in (8, 16)}
    keep = []
    for name, mod, u8 in (("full", Full(), False), ("staged_exact_crop", Staged(), False),
                          ("staged_exact_crop_u8_bgr_nhwc", StagedU8(), True)):
        for bs in (16, 8):
            run, bs_s, *hold = build(mod, bs, name, out_u8=u8)
            keep.append(hold)
            z = Z[:bs].contiguous()
            with torch.inference_mode():
                y = run(z).clone(); torch.cuda.synchronize()
                ref = ref_full[bs]
                if name == "full":
                    err = float((y.float() - ref).abs().max())
                elif not u8:
                    err = float((y.float() - ref[:, :, USED_Y0:, :]).abs().max())
                else:
                    r8 = (ref[:, :, USED_Y0:, :] * 255).round().flip(1).permute(0, 2, 3, 1)
                    err = float((y.float() - r8).abs().max())
                st = gpu_state(f"trt {name} bs{bs}")
                r = bench_events(lambda: run(), warmup=15, iters=50)
                thr = bench_throughput(lambda: run(), warmup=10, iters=50)
            r.update({"ms_per_frame": r["median_ms"] / bs, "pipelined_ms_per_frame": thr["gpu_ms_per_call"] / bs,
                      "build_s": bs_s, "max_abs_vs_eager_pytorch": err, "gpu_state": st})
            res["results"][f"{name}_bs{bs}"] = r
            print(f"TRT {name:32s} bs{bs}: {r['median_ms']:.3f} ms ({r['ms_per_frame']:.3f} ms/f, pipelined "
                  f"{r['pipelined_ms_per_frame']:.3f}) max_abs {err:.3e}", flush=True)
            dump("p5_trt_taesd.json", res)
    res["gpu_end"] = gpu_state("end")
    dump("p5_trt_taesd.json", res)
