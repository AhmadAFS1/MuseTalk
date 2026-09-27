"""Probe 2: row-cropped TAESD decode (MuseTalk only blends 256-px rows >= 104).

(a) latent-row crop: decode z[:, :, 13-h:, :] and compare kept rows to full decode.
(b) staged exact crop: run the cheap 32/64-res stages at full height, crop feature maps
    at the 128-res and 256-res stage inputs with a halo equal to the remaining receptive field.
"""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch._dynamo
import torch.nn as nn

torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True
LAT_Y0 = USED_Y0 // 8  # 13

model = load_taesd()
dec = model.decoder
layers = list(dec.layers)

# ---------------- receptive field (rows above the first kept output row) ------------
def rf_rows():
    """Walk layers backwards: how many input rows above output row r does each stage need."""
    need = 0.0  # in units of the current resolution
    trace = []
    for i in reversed(range(len(layers))):
        L = layers[i]
        if isinstance(L, nn.Conv2d):
            need += (L.kernel_size[0] // 2)
        elif type(L).__name__ == "AutoencoderTinyBlock":
            need += sum(c.kernel_size[0] // 2 for c in L.conv if isinstance(c, nn.Conv2d))
        elif isinstance(L, nn.Upsample):
            need = need / 2.0
        trace.append((i, type(L).__name__, need))
    return need, trace

rf_latent, rf_trace = rf_rows()
print(f"theoretical receptive-field radius (rows) in latent units: {rf_latent:.3f} -> "
      f"{rf_latent*8:.1f} px at 256", flush=True)

Z = real_latents(128).to(DEV)


def full_decode(z):
    return (dec(z).div(2).add(0.5) if False else ((dec(z) / 2 + 0.5).clamp(0, 1)))


def latent_crop_decode(z, h):
    y0 = max(0, LAT_Y0 - h)
    out = (dec(z[:, :, y0:, :]) / 2 + 0.5).clamp(0, 1)
    return out[:, :, (LAT_Y0 - y0) * 8:, :]


# Stage boundaries in DecoderTiny.layers (diffusers 0.30):
# 0 conv 1 relu 2-4 blk 5 up | 6 conv 7-9 blk 10 up | 11 conv 12-14 blk 15 up | 16 conv 17 blk 18 conv
def exact_rows_needed():
    """Per stage input (after each upsample), first row index needed for exact output rows >= 104."""
    r = USED_Y0
    first = {}
    for i in reversed(range(len(layers))):
        L = layers[i]
        if isinstance(L, nn.Conv2d):
            r -= L.kernel_size[0] // 2
        elif type(L).__name__ == "AutoencoderTinyBlock":
            r -= sum(c.kernel_size[0] // 2 for c in L.conv if isinstance(c, nn.Conv2d))
        elif isinstance(L, nn.Upsample):
            first[i + 1] = r  # rows needed at input of layer i+1 (post-upsample resolution)
            r = r // 2  # nearest upsample: output row k comes from input row k//2
        first.setdefault("latent", None)
    first["latent"] = r
    return first

FIRST = exact_rows_needed()
print("exact first-row needed at stage inputs:", FIRST, flush=True)


def staged_crop_decode(z, crop_at=(11, 16), extra_halo=0):
    """Exact crop: crop activations at the inputs of layers in crop_at (post-upsample)."""
    x = torch.tanh(z / 3) * 3
    offset = 0  # rows removed so far, in current-resolution units
    for i, L in enumerate(layers):
        if i in crop_at:
            want = max(0, FIRST[i] - extra_halo)
            cut = want - offset
            if cut > 0:
                x = x[:, :, cut:, :]
                offset = want
        x = L(x)
        if isinstance(L, nn.Upsample):
            offset *= 2
    x = (x.mul(2).sub(1) / 2 + 0.5).clamp(0, 1)
    keep_from = USED_Y0 - offset
    return x[:, :, keep_from:, :]


def u8(x):
    return (x.float() * 255).round().clamp(0, 255).to(torch.uint8)


def errs(a, b):
    d = (a.float() - b.float()).abs()
    du = (u8(a).int() - u8(b).int()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "p99_abs": float(torch.quantile(d.flatten()[:: max(1, d.numel() // 2_000_000)], 0.99)),
            "u8_max_lsb": int(du.max()), "u8_frac_pixels_differ": float((du > 0).float().mean()),
            "u8_frac_gt1lsb": float((du > 1).float().mean())}


res = {"gpu_start": gpu_state("start"), "rf_latent_rows": rf_latent, "rf_px_at_256": rf_latent * 8,
       "exact_first_rows": {str(k): v for k, v in FIRST.items()}, "latent_crop": {}, "staged": {}, "timing": {}}

with torch.inference_mode():
    ref = torch.cat([full_decode(Z[i:i + 32]) for i in range(0, 128, 32)])[:, :, USED_Y0:, :]
    for h in range(0, LAT_Y0 + 1):
        out = torch.cat([latent_crop_decode(Z[i:i + 32], h) for i in range(0, 128, 32)])
        e = errs(out, ref)
        res["latent_crop"][h] = e
        print(f"latent halo h={h:2d} (start latent row {LAT_Y0-h:2d}, px {(LAT_Y0-h)*8:3d}): "
              f"max_abs {e['max_abs']:.4f} mean {e['mean_abs']:.2e} u8maxLSB {e['u8_max_lsb']} "
              f"u8 differ {e['u8_frac_pixels_differ']*100:.3f}% >1LSB {e['u8_frac_gt1lsb']*100:.4f}%", flush=True)
    for name, kw in {"crop@128+256": dict(crop_at=(11, 16)),
                     "crop@64+128+256": dict(crop_at=(6, 11, 16)),
                     "crop@256only": dict(crop_at=(16,)),
                     "crop@128+256_halo-4": dict(crop_at=(11, 16), extra_halo=-4),
                     "crop@64+128+256_halo-4(aggr)": dict(crop_at=(6, 11, 16), extra_halo=-4)}.items():
        out = torch.cat([staged_crop_decode(Z[i:i + 32], **kw) for i in range(0, 128, 32)])
        e = errs(out, ref)
        res["staged"][name] = e
        print(f"staged {name:28s}: shape {tuple(out.shape)} max_abs {e['max_abs']:.2e} u8maxLSB {e['u8_max_lsb']} "
              f"differ {e['u8_frac_pixels_differ']*100:.4f}%", flush=True)

# ---------------- timing -------------------------------------------------------------
variants = {
    "full": lambda z: full_decode(z),
    "latent_h4": lambda z: latent_crop_decode(z, 4),
    "latent_h6": lambda z: latent_crop_decode(z, 6),
    "latent_h8": lambda z: latent_crop_decode(z, 8),
    "staged_exact_128_256": lambda z: staged_crop_decode(z, crop_at=(11, 16)),
    "staged_exact_64_128_256": lambda z: staged_crop_decode(z, crop_at=(6, 11, 16)),
}
modes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["eager_cl", "max-autotune-no-cudagraphs", "max-autotune"]
for mode in modes:
    torch._dynamo.reset()
    if mode == "eager_cl":
        model.to(memory_format=torch.channels_last)
    else:
        model.to(memory_format=torch.contiguous_format)
    res["timing"][mode] = {}
    for bs in (8, 16):
        z = Z[:bs].contiguous()
        if mode == "eager_cl":
            z = z.to(memory_format=torch.channels_last)
        for vname, v in variants.items():
            fn_ = v if mode.startswith("eager") else torch.compile(v, mode=mode, dynamic=False)
            mark = mode == "max-autotune"
            with torch.inference_mode():
                if mark:
                    torch.compiler.cudagraph_mark_step_begin()
                fn_(z); torch.cuda.synchronize()
                st = gpu_state(f"{mode} {vname} bs{bs}")
                r = bench_events(lambda: fn_(z), warmup=15, iters=40, mark_step=mark)
            r["ms_per_frame"] = r["median_ms"] / bs
            r["gpu_state"] = st
            res["timing"][mode][f"{vname}_bs{bs}"] = r
            print(f"{mode:28s} {vname:26s} bs{bs}: {r['median_ms']:.3f} ms ({r['ms_per_frame']:.3f} ms/f)", flush=True)
        dump("p2_rowcrop.json", res)
model.to(memory_format=torch.contiguous_format)
res["gpu_end"] = gpu_state("end")
dump("p2_rowcrop.json", res)
