"""Probe 4a: FP8 e4m3 torch._scaled_mm and INT8 torch._int_mm vs FP16 matmul at the MuseTalk UNet's real GEMM shapes."""
import torch

from common import save_json, wait_clean

torch.backends.cuda.matmul.allow_tf32 = False
log = []
res = {"gpu_log": log, "shapes": []}
dev = "cuda"
REPS = 20


def t_op(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS):
            fn()
        e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / REPS)
    ts.sort()
    return ts[15]


levels = [(1024, 320), (256, 640), (64, 1280), (16, 1280)]
shapes = []
for bs in (8, 16):
    for T, C in levels:
        M = bs * T
        shapes += [
            (f"bs{bs} T{T} C{C} qkvo/proj(1x1) C->C", M, C, C),
            (f"bs{bs} T{T} C{C} GEGLU C->8C", M, C, 8 * C),
            (f"bs{bs} T{T} C{C} FFout 4C->C", M, 4 * C, C),
        ]
    shapes.append((f"bs{bs} crossattn k/v 384->320 (M=bs*50)", bs * 50, 384, 320))
    # 3x3 conv as implicit GEMM (im2col K = Cin*9) for representative resnet convs
    shapes += [
        (f"bs{bs} conv3x3-as-GEMM 320->320 @32x32", bs * 1024, 320 * 9, 320),
        (f"bs{bs} conv3x3-as-GEMM 640->640 @16x16", bs * 256, 640 * 9, 640),
        (f"bs{bs} conv3x3-as-GEMM 1280->1280 @8x8", bs * 64, 1280 * 9, 1280),
        (f"bs{bs} conv3x3-as-GEMM 2560->1280 @8x8 (up1 cat)", bs * 64, 2560 * 9, 1280),
        (f"bs{bs} conv3x3-as-GEMM 640->320 @32x32 (up3 cat)", bs * 1024, 640 * 9, 320),
    ]

st = wait_clean("fp8_gemm", log)
res["contaminated_at_start"] = st["contaminated"]
tot = {"fp16": 0.0, "fp8": 0.0, "fp8_fast": 0.0, "int8": 0.0}
for name, M, K, N in shapes:
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.05
    flops = 2 * M * K * N
    r = {"name": name, "M": M, "K": K, "N": N, "GFLOP": flops / 1e9}
    r["fp16_ms"] = t_op(lambda: torch.matmul(a, w.t()))
    a8 = a.to(torch.float8_e4m3fn)
    w8 = w.to(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, device=dev)
    sb = torch.tensor(1.0, device=dev)
    try:
        r["fp8_ms"] = t_op(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.float16))
        r["fp8_fastacc_ms"] = t_op(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.float16,
                                                           use_fast_accum=True))
        r["fp8_cast_act_ms"] = t_op(lambda: a.to(torch.float8_e4m3fn))
    except Exception as ex:
        r["fp8_error"] = repr(ex)[:300]
    try:
        ai = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
        wi = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
        r["int8_ms"] = t_op(lambda: torch._int_mm(ai, wi.t()))
    except Exception as ex:
        r["int8_error"] = repr(ex)[:300]
    for k in ["fp16", "fp8", "fp8_fastacc", "int8"]:
        if f"{k}_ms" in r:
            r[f"{k}_TFLOPS"] = flops / (r[f"{k}_ms"] / 1e3) / 1e12
    if "fp8_ms" in r:
        r["fp8_speedup"] = r["fp16_ms"] / r["fp8_ms"]
        r["fp8_fastacc_speedup"] = r["fp16_ms"] / r["fp8_fastacc_ms"]
    if "int8_ms" in r:
        r["int8_speedup"] = r["fp16_ms"] / r["int8_ms"]
    res["shapes"].append(r)
    print(f"{name:52s} fp16 {r['fp16_TFLOPS']:6.1f}  fp8 {r.get('fp8_TFLOPS', 0):6.1f} fp8fast {r.get('fp8_fastacc_TFLOPS', 0):6.1f}"
          f"  int8 {r.get('int8_TFLOPS', 0):6.1f} TFLOPS | x{r.get('fp8_speedup', 0):.2f} / x{r.get('int8_speedup', 0):.2f}", flush=True)
    del a, w, a8, w8
# big square reference (roofline)
for n in (4096, 8192):
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    w = torch.randn(n, n, device=dev, dtype=torch.float16)
    ms16 = t_op(lambda: torch.matmul(a, w.t()))
    a8, w8 = a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, device=dev)
    ms8 = t_op(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sa, out_dtype=torch.float16))
    ms8f = t_op(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sa, out_dtype=torch.float16, use_fast_accum=True))
    ai = torch.randint(-127, 127, (n, n), device=dev, dtype=torch.int8)
    msi = t_op(lambda: torch._int_mm(ai, ai.t()))
    f = 2 * n ** 3
    res[f"square_{n}"] = {"fp16_TFLOPS": f / ms16 / 1e9, "fp8_TFLOPS": f / ms8 / 1e9, "fp8_fastacc_TFLOPS": f / ms8f / 1e9,
                          "int8_TOPS": f / msi / 1e9}
    print("square", n, res[f"square_{n}"], flush=True)
    del a, w, a8, w8, ai
st = wait_clean("fp8_gemm_end", log)
save_json("fp8_gemm.json", res)
print("DONE")
