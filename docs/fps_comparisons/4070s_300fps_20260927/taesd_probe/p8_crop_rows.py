"""Probe 8: staged exact row-crop for arbitrary first-used row R (fixed_face_height avatars read from row 85).
TRT FP16 engines, in memory. Exactness checked in PyTorch eager (bitwise vs full decode rows >= R)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
import torch.nn as nn
import p5_trt_taesd as P5

layers = P5.layers; dec = P5.dec


def first_rows(R):
    r = R; first = {}
    for i in reversed(range(len(layers))):
        L = layers[i]
        if isinstance(L, nn.Conv2d):
            r -= L.kernel_size[0] // 2
        elif type(L).__name__ == "AutoencoderTinyBlock":
            r -= sum(c.kernel_size[0] // 2 for c in L.conv if isinstance(c, nn.Conv2d))
        elif isinstance(L, nn.Upsample):
            first[i + 1] = max(0, r); r = r // 2
    return first


class StagedR(nn.Module):
    def __init__(self, R):
        super().__init__(); self.R = R; self.first = first_rows(R)

    def forward(self, z):
        x = torch.tanh(z / 3) * 3
        offset = 0
        for i, L in enumerate(layers):
            if i in self.first:
                cut = self.first[i] - offset
                if cut > 0:
                    x = x[:, :, cut:, :]; offset = self.first[i]
            x = L(x)
            if isinstance(L, nn.Upsample):
                offset *= 2
        x = (x.mul(2).sub(1) / 2 + 0.5).clamp(0, 1)
        return x[:, :, self.R - offset:, :]

Z = real_latents(128).to(DEV)
res = {"gpu_start": gpu_state("start"), "rows": {}}
with torch.inference_mode():
    full = torch.cat([P5.Full()(Z[i:i + 32]) for i in range(0, 128, 32)])
for R in (0, 80, 85, 104, 110):
    m = StagedR(R)
    with torch.inference_mode():
        out = torch.cat([m(Z[i:i + 32]) for i in range(0, 128, 32)])
    exact = float((out.float() - full[:, :, R:, :].float()).abs().max())
    rr = {"first_rows": {str(k): v for k, v in m.first.items()}, "eager_max_abs_vs_full": exact}
    for bs in (8, 16):
        run, bsec, *hold = P5.build(m if R else P5.Full(), bs, f"R{R}")
        with torch.inference_mode():
            st = gpu_state(f"R{R} bs{bs}")
            r = bench_events(lambda: run(), warmup=15, iters=50)
        rr[f"trt_bs{bs}_ms"] = r["median_ms"]; rr[f"trt_bs{bs}_ms_per_frame"] = r["median_ms"] / bs
        rr[f"gpu_state_bs{bs}"] = st
        del run, hold
    res["rows"][R] = rr
    print(f"R={R:3d} first={m.first} eager exact max_abs={exact:.1e}  TRT bs8 {rr['trt_bs8_ms_per_frame']:.3f} "
          f"bs16 {rr['trt_bs16_ms_per_frame']:.3f} ms/f", flush=True)
    dump("p8_crop_rows.json", res)
res["gpu_end"] = gpu_state("end")
dump("p8_crop_rows.json", res)
