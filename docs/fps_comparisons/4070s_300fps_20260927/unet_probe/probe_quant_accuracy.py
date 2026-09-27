"""Probe 4c: whole-UNet fake-quant accuracy (modelopt) on real captured UNet I/O: latent rel-L2 and decoded-pixel PSNR
(sd-vae decode) vs the FP16 UNet, for INT8 (max), INT8 SmoothQuant, INT8 conv-only, FP8 (max), FP8 GEMM-only."""
import copy
import json
import os

import torch

from common import REPO, load_captures, load_unet_fp16_lowmem, save_json

import modelopt.torch.quantization as mtq

dev = torch.device("cuda")
res = {}
model, missing = load_unet_fp16_lowmem()
res["missing_keys"] = len(missing)
lat, aud, pred = load_captures()
lat, aud, pred = lat.to(dev).half(), aud.to(dev).half(), pred.to(dev).half()
N = lat.shape[0]
cal = [(lat[i:i + 8], aud[i:i + 8]) for i in range(0, N // 2, 8)]
ev = [(lat[i:i + 8], aud[i:i + 8]) for i in range(N // 2, N, 8)]
t0 = torch.tensor([0], device=dev)


@torch.no_grad()
def run(mm, batches):
    return torch.cat([mm(l, t0, encoder_hidden_states=a).sample for l, a in batches])


ref = run(model, ev)
res["fp16_vs_captured_pred_rel_l2"] = ((ref.float() - pred[N // 2:].float()).norm() / pred[N // 2:].float().norm()).item()

from diffusers import AutoencoderKL  # noqa: E402

vae = AutoencoderKL.from_pretrained(os.path.join(REPO, "models/sd-vae"), torch_dtype=torch.float16).to(dev).eval()


@torch.no_grad()
def decode(l):
    outs = []
    for i in range(0, l.shape[0], 8):
        x = vae.decode(l[i:i + 8] / vae.config.scaling_factor).sample
        outs.append(((x.float() / 2 + 0.5).clamp(0, 1)))
    return torch.cat(outs)


ref_img = decode(ref)


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def loop(mm):
    with torch.no_grad():
        for l, a in cal:
            mm(l, t0, encoder_hidden_states=a)


def cfg_variant(base, disable_patterns=()):
    c = copy.deepcopy(base)
    for p in disable_patterns:
        c["quant_cfg"][p] = {"enable": False}
    return c


variants = {
    "int8_max_all_linear_conv": mtq.INT8_DEFAULT_CFG,
    "int8_smoothquant": mtq.INT8_SMOOTHQUANT_CFG,
    "int8_conv_only(no Linear)": None,  # built below
    "fp8_max_all": mtq.FP8_DEFAULT_CFG,
}
for name, cfg in variants.items():
    try:
        mm = copy.deepcopy(model)
        if cfg is None:
            cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
            cfg["quant_cfg"]["nn.Linear"] = {"*": {"enable": False}}
        mm = mtq.quantize(mm, cfg, loop)
        out = run(mm, ev)
        img = decode(out)
        p = psnr(img, ref_img)
        # mouth region proxy: lower half of the 256x256 face crop
        pm = psnr(img[:, :, 128:, :], ref_img[:, :, 128:, :])
        res[name] = {
            "latent_rel_l2": ((out.float() - ref.float()).norm() / ref.float().norm()).item(),
            "latent_maxabs": (out.float() - ref.float()).abs().max().item(),
            "psnr_full_mean_dB": p.mean().item(), "psnr_full_min_dB": p.min().item(),
            "psnr_lowerhalf_mean_dB": pm.mean().item(), "psnr_lowerhalf_min_dB": pm.min().item(),
        }
        print(name, res[name], flush=True)
        del mm
    except Exception as ex:
        import traceback
        res[name] = {"error": traceback.format_exc()[-800:]}
        print(name, "ERR", traceback.format_exc()[-400:], flush=True)
    torch.cuda.empty_cache()
# reference: TRT FP16 production engine error scale is recorded in the artifact meta (mae_max 0.002 on latents)
res["note"] = "eval = last 64 captured frames (same avatar), calib = first 64; PSNR is vs FP16-UNet decode, not vs ground truth"
save_json("quant_accuracy.json", res)
print("DONE")
