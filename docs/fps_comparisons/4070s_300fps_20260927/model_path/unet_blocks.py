import os, json, math, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch.nn as nn
from diffusers import UNet2DConditionModel
cfg = json.load(open("/workspace/MuseTalk/models/musetalkV15/musetalk.json"))
with torch.device("meta"):
    unet = UNet2DConditionModel(**cfg).half().eval()
B = 1
lat = torch.empty(B, 8, 32, 32, device="meta", dtype=torch.float16)
ehs = torch.empty(B, 50, 384, device="meta", dtype=torch.float16)
ts = torch.zeros(1, dtype=torch.long, device="meta")
# per leaf (conv/linear) flops + weight elements, grouped to "block units" = resnets.i / attentions.i / up/downsamplers / others
units = collections.OrderedDict()
def unit_of(name):
    p = name.split(".")
    if p[0] in ("down_blocks", "up_blocks") and len(p) > 3 and p[2] in ("resnets", "attentions", "downsamplers", "upsamplers"):
        return ".".join(p[:4])
    if p[0] == "mid_block" and len(p) > 2:
        return ".".join(p[:3])
    return p[0]
leaf = []
def hk(name):
    def h(mod, inp, out):
        x = inp[0]
        if isinstance(mod, nn.Conv2d):
            f = 2*out.shape[0]*out.shape[2]*out.shape[3]*mod.out_channels*mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1]
            rows = out.shape[0]*out.shape[2]*out.shape[3]
            res = out.shape[2]
        else:
            t = 1
            for d in x.shape[:-1]: t *= d
            f = 2*t*mod.in_features*mod.out_features
            rows = t
            res = int(math.isqrt(x.shape[1])) if x.dim()==3 and x.shape[1] != 50 else 0
        w = mod.weight.numel()
        leaf.append((name, f, w, rows, res))
    return h
hs = [m.register_forward_hook(hk(n)) for n, m in unet.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
# attention bmm per Attention module
attn_tok = {}
for n, m in unet.named_modules():
    if type(m).__name__ == "Attention":
        hs.append(m.to_q.register_forward_hook(lambda mod, inp, out, n=n: attn_tok.__setitem__(n, (inp[0].shape[1], out.shape[-1]))))
with torch.no_grad():
    unet(lat, ts, encoder_hidden_states=ehs)
for h in hs: h.remove()
for name, f, w, rows, res in leaf:
    u = unit_of(name)
    d = units.setdefault(u, {"flops": 0, "weights": 0, "res": 0})
    d["flops"] += f; d["weights"] += w; d["res"] = max(d["res"], res)
for n, (N, C) in attn_tok.items():
    u = unit_of(n)
    kv = N if n.endswith("attn1") else 50
    units[u]["flops"] += 4*B*N*kv*C
tot = sum(d["flops"] for d in units.values())
wt = sum(d["weights"] for d in units.values())
print(f"total GF/frame {tot/1e9:.2f}, conv/linear weights {wt/1e6:.1f}M")
print(f"{'unit':34s} {'res':>4s} {'GF/frm':>7s} {'%F':>5s} {'Wparams M':>9s} {'%W':>5s}")
for u, d in units.items():
    print(f"{u:34s} {d['res']:4d} {d['flops']/1e9:7.2f} {100*d['flops']/tot:5.1f} {d['weights']/1e6:9.2f} {100*d['weights']/wt:5.1f}")
# roofline lower bound per batch size using leaf ops: time = max(F/peak, weight_bytes/bw) summed per leaf
def bound(bs, peak_tflops, wbytes, bw=504e9, l2_resident_acts=True):
    t = 0.0
    tf = 0.0; tw = 0.0
    for name, f, w, rows, res in leaf:
        tc = bs*f/(peak_tflops*1e12); tm = w*wbytes/bw
        t += max(tc, tm); tf += tc; tw += tm
    # attention bmm (compute only)
    for n, (N, C) in attn_tok.items():
        kv = N if n.endswith("attn1") else 50
        t += bs*4*N*kv*C/(peak_tflops*1e12)
    return t*1e3, tf*1e3, tw*1e3
print("\nRoofline lower bound for conv/GEMM/attn math only (ms per batch; ms per frame) -- excludes norms/softmax/elementwise")
for label, peak, wb in [("FP16 w/ FP16-acc 142T", 142, 2), ("FP16 w/ FP32-acc 71T", 71, 2), ("FP8 w/ FP32-acc 142T (8-bit W)", 142, 1), ("INT8 284T", 284, 1)]:
    for bs in (8, 16, 24, 32):
        t, tf, tw = bound(bs, peak, wb)
        print(f"  {label:32s} bs{bs:2d}: {t:6.2f} ms/batch  {t/bs:5.3f} ms/frame  (pure-compute {tf/bs:5.3f}, weight-stream {tw:5.2f} ms/batch) -> {1000*bs/t:6.0f} fps")
# stage-level weight-bound fraction at bs8 FP16-acc
print("\nPer-resolution: FLOPs vs weight bytes; arithmetic intensity at bs8 (FLOP per weight byte, fp16) vs ridge 142e12/504e9=282")
byres = collections.defaultdict(lambda: [0, 0])
for name, f, w, rows, res in leaf:
    byres[res][0] += f; byres[res][1] += w
for r, (f, w) in sorted(byres.items(), key=lambda kv: -kv[0]):
    print(f"  res {r:3d}: {f/1e9:7.2f} GF/frame, {w/1e6:7.1f}M weights, AI@bs8={8*f/(2*w):7.1f}, AI@bs16={16*f/(2*w):7.1f}")
json.dump({u: d for u, d in units.items()}, open("/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/throughput300/model_path/unet_units.json", "w"), indent=1)
