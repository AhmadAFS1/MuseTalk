"""Probe 9 (optional lever): TRT INT8 (implicit, entropy calibration on real latents) TAESD staged crop R=104.
In memory only. Reports speed and error vs FP16 eager on kept rows (PSNR on uint8)."""
import sys, io, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import tensorrt as trt
import p5_trt_taesd as P5

LOG = trt.Logger(trt.Logger.ERROR)
Z = real_latents(128).to(DEV)
BS = 16


class Calib(trt.IInt8EntropyCalibrator2):
    def __init__(self, data):
        super().__init__(); self.data = data; self.i = 0
        self.buf = torch.empty((BS, 4, 32, 32), device=DEV, dtype=torch.float32)

    def get_batch_size(self): return BS

    def get_batch(self, names):
        if self.i >= self.data.shape[0]: return None
        self.buf.copy_(self.data[self.i:self.i + BS].float()); self.i += BS
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self): return None

    def write_calibration_cache(self, cache): return None


def build_int8(module, bs):
    z = torch.randn(bs, 4, 32, 32, device=DEV, dtype=torch.float32)
    f = io.BytesIO()
    P5.model.float(); m32 = module
    with torch.inference_mode():
        torch.onnx.export(m32, (z,), f, input_names=["z"], output_names=["y"], opset_version=17)
    builder = trt.Builder(LOG); net = builder.create_network(0); parser = trt.OnnxParser(net, LOG)
    assert parser.parse(f.getvalue())
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = Calib(Z)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    t0 = time.time(); plan = builder.build_serialized_network(net, cfg); bsec = time.time() - t0
    eng = trt.Runtime(LOG).deserialize_cuda_engine(plan); ctx = eng.create_execution_context()
    zin = torch.empty(bs, 4, 32, 32, device=DEV, dtype=torch.float32)
    oname = eng.get_tensor_name(1)
    yout = torch.empty(tuple(eng.get_tensor_shape(oname)), device=DEV, dtype=torch.float32)
    ctx.set_tensor_address("z", zin.data_ptr()); ctx.set_tensor_address(oname, yout.data_ptr())

    def run(zz=None):
        if zz is not None: zin.copy_(zz)
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return yout
    return run, bsec, (eng, ctx, zin, yout)

res = {"gpu_start": gpu_state("start")}
with torch.inference_mode():
    ref = torch.cat([P5.Full()(Z[i:i + 32]) for i in range(0, 128, 32)])[:, :, USED_Y0:, :].float()
staged = P5.Staged()
run, bsec, hold = build_int8(staged, BS)
P5.model.half()
with torch.inference_mode():
    outs = torch.cat([run(Z[i:i + BS].float()).clone() for i in range(0, 128, BS)])
    d = (outs - ref).abs()
    mse_u8 = (((outs * 255).round() - (ref * 255).round()) ** 2).mean().item()
    psnr = 10 * np.log10(255 ** 2 / max(mse_u8, 1e-9))
    st = gpu_state("int8 staged bs16")
    r = bench_events(lambda: run(), warmup=15, iters=50)
res["int8_staged_bs16"] = {"median_ms": r["median_ms"], "ms_per_frame": r["median_ms"] / BS, "build_s": bsec,
                           "max_abs_vs_fp16_eager": float(d.max()), "mean_abs": float(d.mean()),
                           "psnr_u8_db": psnr, "gpu_state": st, "note": "fp32 I/O binding (adds I/O cost)"}
print(res["int8_staged_bs16"], flush=True)
res["gpu_end"] = gpu_state("end")
dump("p9_int8.json", res)
