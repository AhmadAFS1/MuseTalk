"""SmoothQuant for the UNet feed-forward output projections (ff.net.2), folded into GEGLU.

BasicTransformerBlock.ff = [GEGLU(proj: dim -> 2*inner), Dropout, Linear(inner -> dim)]; GEGLU returns
value * gelu(gate) with value = the first `inner` rows of proj. Scaling input channel c of ff.net.2 by
1/s_c is therefore the same as scaling proj's value row c (and its bias) by 1/s_c, and ff.net.2's weight
column c by s_c. s_c is rounded to a power of two, which is exact for normal FP16 values, but weights pushed
into the subnormal range lose bits. Measured on 2026-09-28 (alpha 0.5): the FP16 UNet output changed by up to
0.0116, i.e. FP16-noise level but NOT bit-identical, and ff.net.2 INT8 error fell only ~1.5-2x. Not used by
any exported recipe (docs/fps_comparisons/4070s_400fps_20260928/README.md, section 3.6).
"""
from __future__ import annotations

import torch


def ff_out_layers(model) -> dict:
    """{ff.net.2 module name: (geglu proj Linear, ff.net.2 Linear)} for every transformer feed-forward."""
    out = {}
    for name, m in model.named_modules():
        net = getattr(m, "net", None)
        if name.endswith(".ff") and net is not None and len(net) >= 3 and hasattr(net[0], "proj"):
            out[f"{name}.net.2"] = (net[0].proj, net[2])
    return out


@torch.inference_mode()
def collect_ff_out_absmax(model, batches, forward) -> dict:
    """Per-input-channel max |x| of every ff.net.2 input over the calibration batches."""
    stats, hooks = {}, []
    for name, (_, lin) in ff_out_layers(model).items():
        def hook(mod, inp, _n=name):
            x = inp[0].detach().abs().flatten(0, -2).amax(dim=0).float()
            stats[_n] = torch.maximum(stats[_n], x) if _n in stats else x
        hooks.append(lin.register_forward_pre_hook(hook))
    try:
        for b in batches:
            forward(model, b)
    finally:
        for h in hooks:
            h.remove()
    return stats


def smooth_exponents(model, absmax: dict, alpha: float) -> dict:
    """SmoothQuant s = xmax^a / wmax^(1-a), rounded to a power of two; returns {layer: [log2 s_c]}."""
    exps = {}
    for name, (_, lin) in ff_out_layers(model).items():
        x = absmax[name].clamp(min=1e-5)
        w = lin.weight.detach().abs().amax(dim=0).float().clamp(min=1e-5)
        s = x.pow(alpha) / w.pow(1.0 - alpha)
        exps[name] = torch.round(torch.log2(s)).clamp(-12, 12).to(torch.int64).tolist()
    return exps


@torch.no_grad()
def apply_smoothing(model, exps: dict) -> None:
    layers = ff_out_layers(model)
    for name, e in exps.items():
        proj, lin = layers[name]
        s = torch.pow(2.0, torch.tensor(e, dtype=torch.float32, device=lin.weight.device))
        inner = lin.in_features
        assert proj.out_features == 2 * inner and s.numel() == inner, name
        proj.weight[:inner] /= s.to(proj.weight.dtype)[:, None]
        if proj.bias is not None:
            proj.bias[:inner] /= s.to(proj.bias.dtype)
        lin.weight *= s.to(lin.weight.dtype)[None, :]
