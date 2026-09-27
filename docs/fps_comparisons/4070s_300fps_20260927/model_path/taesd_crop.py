import os, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from diffusers import AutoencoderTiny
torch.set_num_threads(8)
m = AutoencoderTiny.from_pretrained("/workspace/MuseTalk/models/taesd", torch_dtype=torch.float32).eval()
dec = m.decoder
layers = list(dec.layers)
print("decoder params M", sum(p.numel() for p in dec.parameters())/1e6)
print([type(l).__name__ for l in layers])
# any normalization?
print("norm layers:", [n for n, x in dec.named_modules() if 'Norm' in type(x).__name__])

z = torch.randn(1, 4, 32, 32) * 1.2
with FlopCounterMode(display=False) as fc, torch.no_grad():
    full = dec(z)
print("TAESD decoder GFLOP/frame (FlopCounter):", fc.get_total_flops()/1e9)

# analytic per-resolution conv FLOPs
per_res = {}
hooks = []
def hk(name):
    def h(mod, inp, out):
        H, W = out.shape[2], out.shape[3]
        f = 2*H*W*mod.out_channels*mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1]
        per_res[f"{H}x{W}"] = per_res.get(f"{H}x{W}", 0) + f
    return h
for n, mod in dec.named_modules():
    if isinstance(mod, nn.Conv2d): hooks.append(mod.register_forward_hook(hk(n)))
with torch.no_grad(): dec(z)
for h in hooks: h.remove()
tot = sum(per_res.values())
for k, v in per_res.items(): print(f"  {k:8s} {v/1e9:7.2f} GF {100*v/tot:5.1f}%")

# backward requirement propagation: rows needed at input of each layer for output rows >= r0
def conv_depth(l):
    if isinstance(l, nn.Conv2d): return l.kernel_size[0]//2
    if type(l).__name__ == "AutoencoderTinyBlock": return 3
    return 0
def req_chain(r0):
    req = [None]*(len(layers)+1)
    req[len(layers)] = r0
    for i in range(len(layers)-1, -1, -1):
        l = layers[i]; r = req[i+1]
        if isinstance(l, nn.Upsample): r = r // 2
        else: r = max(0, r - conv_depth(l))
        req[i] = r
    return req

def cropped_decode(z, r0):
    req = req_chain(r0)
    x = torch.tanh(z / 3) * 3
    top = req[0]
    x = x[:, :, top:]
    flops = 0
    for i, l in enumerate(layers):
        if isinstance(l, nn.Upsample):
            x = l(x); top *= 2
        elif isinstance(l, nn.Conv2d) or type(l).__name__ == "AutoencoderTinyBlock":
            x = l(x)
            convs = [l] if isinstance(l, nn.Conv2d) else [c for c in l.modules() if isinstance(c, nn.Conv2d)]
            for c in convs:
                flops += 2*x.shape[2]*x.shape[3]*c.out_channels*c.in_channels*c.kernel_size[0]*c.kernel_size[1]
        else:
            x = l(x)
        # drop rows no longer needed
        need = req[i+1]
        if need > top:
            x = x[:, :, need-top:]; top = need
    return x, top, flops

with torch.no_grad():
    full = dec(z)
res = {}
for r0 in [104, 110, 128]:
    req = req_chain(r0)
    with torch.no_grad():
        out, top, fl = cropped_decode(z, r0)
    d = (out.mul(2).sub(1) - full[:, :, r0:]).abs().max().item()
    res[r0] = {"latent_rows_needed_from": req[0], "flops_GF": fl/1e9, "flops_frac_of_full": fl/tot, "max_abs_diff_vs_full": d, "out_rows": out.shape[2]}
    print(f"r0={r0}: latent rows needed from {req[0]} (of 32); cropped GF={fl/1e9:.2f} ({100*fl/tot:.1f}% of full); max|diff|={d:.3e}; out rows={out.shape[2]}")
    print("   per-layer req (at each layer input):", req)
# receptive field radius in output pixels for 1 row: rows needed at latent for output row r
print("RF: output row r depends on latent rows >= ", {r: req_chain(r)[0] for r in [64, 104, 128, 192, 255]})
json.dump({"per_res_GF": {k: v/1e9 for k, v in per_res.items()}, "total_GF": tot/1e9, "crop": res}, open("/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/throughput300/model_path/taesd_crop.json", "w"), indent=1)
