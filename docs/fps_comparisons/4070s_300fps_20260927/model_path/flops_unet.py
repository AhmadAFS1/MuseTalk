import os, json, sys, math, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils.flop_counter import FlopCounterMode
from diffusers import UNet2DConditionModel

cfg = json.load(open("/workspace/MuseTalk/models/musetalkV15/musetalk.json"))
with torch.device("meta"):
    unet = UNet2DConditionModel(**cfg).half().eval()

# params by top-level stage
def nparams(m): return sum(p.numel() for p in m.parameters())
stages = collections.OrderedDict()
stages["conv_in+time"] = nparams(unet.conv_in) + nparams(unet.time_proj) + nparams(unet.time_embedding)
for i, b in enumerate(unet.down_blocks): stages[f"down_blocks.{i}"] = nparams(b)
stages["mid_block"] = nparams(unet.mid_block)
for i, b in enumerate(unet.up_blocks): stages[f"up_blocks.{i}"] = nparams(b)
stages["out"] = nparams(unet.conv_norm_out) + nparams(unet.conv_out)
tot = nparams(unet)
print("total params M", tot/1e6)
for k, v in stages.items(): print(f"  {k:16s} {v/1e6:8.2f} M  {100*v/tot:5.1f}%")

# attention params: attn1/attn2/ff/proj
cat = collections.Counter()
for name, p in unet.named_parameters():
    if ".attn1." in name: cat["self-attn qkvo"] += p.numel()
    elif ".attn2." in name: cat["cross-attn qkvo"] += p.numel()
    elif ".ff." in name: cat["ff (GEGLU)"] += p.numel()
    elif "proj_in" in name or "proj_out" in name: cat["transformer proj_in/out"] += p.numel()
    elif "norm" in name: cat["norms"] += p.numel()
    elif "time_emb" in name: cat["time_emb_proj (resnet)"] += p.numel()
    elif "conv" in name or "downsamplers" in name or "upsamplers" in name: cat["convs"] += p.numel()
    else: cat["other:"+name.split('.')[-2]] += p.numel()
print("param categories:")
for k, v in cat.most_common(): print(f"  {k:28s} {v/1e6:8.2f} M")

B = 1
lat = torch.empty(B, 8, 32, 32, device="meta", dtype=torch.float16)
ehs = torch.empty(B, 50, 384, device="meta", dtype=torch.float16)
ts = torch.zeros(1, dtype=torch.long, device="meta")

# Hook-based per-op accounting by resolution and op type
records = []
hooks = []
def mk(name, kind):
    def h(mod, inp, out):
        x = inp[0]
        records.append((name, kind, tuple(x.shape), tuple(out.shape) if isinstance(out, torch.Tensor) else None, mod))
    return h
for name, m in unet.named_modules():
    if isinstance(m, torch.nn.Conv2d): hooks.append(m.register_forward_hook(mk(name, "conv")))
    elif isinstance(m, torch.nn.Linear): hooks.append(m.register_forward_hook(mk(name, "linear")))

fc = FlopCounterMode(display=False, depth=None)
with torch.no_grad(), fc:
    out = unet(lat, ts, encoder_hidden_states=ehs).sample
for h in hooks: h.remove()
print("out shape", out.shape)
total_flops = fc.get_total_flops()
print("FlopCounter total GFLOP per frame (2*MAC):", total_flops/1e9)
counts = fc.get_flop_counts()
# top-level module breakdown
print("per top-level module (GFLOP/frame):")
for k in ["UNet2DConditionModel.conv_in", "UNet2DConditionModel.time_embedding"] + [f"UNet2DConditionModel.down_blocks.{i}" for i in range(4)] + ["UNet2DConditionModel.mid_block"] + [f"UNet2DConditionModel.up_blocks.{i}" for i in range(4)] + ["UNet2DConditionModel.conv_out"]:
    if k in counts:
        print(f"  {k:45s} {sum(counts[k].values())/1e9:8.2f}  {dict((str(op).split('.')[-2] if '.' in str(op) else str(op), round(v/1e9,2)) for op,v in counts[k].items())}")
print("global op breakdown:")
g = counts.get("Global", {})
for op, v in sorted(g.items(), key=lambda kv: -kv[1]): print(f"  {str(op):50s} {v/1e9:8.2f}")

# analytic per resolution x category from hooks
res_cat = collections.defaultdict(float)
def res_of(shape):
    if len(shape) == 4: return f"{shape[2]}x{shape[3]}"
    if len(shape) == 3:
        n = shape[1]
        if n == 50: return "ctx"
        s = int(math.isqrt(n)); return f"{s}x{s}"
    return "?"
for name, kind, ishape, oshape, m in records:
    if kind == "conv":
        Ho, Wo = oshape[2], oshape[3]
        kh, kw = m.kernel_size
        fl = 2 * B * Ho * Wo * m.out_channels * m.in_channels // m.groups * kh * kw
        cat_ = "conv1x1" if kh == 1 else "conv3x3"
        if "downsamplers" in name: cat_ = "conv3x3 (downsample s2)"
        if "upsamplers" in name: cat_ = "conv3x3 (upsample)"
        if "conv_shortcut" in name: cat_ = "conv1x1 shortcut"
        if "proj_in" in name or "proj_out" in name: cat_ = "transformer proj_in/out"
        # attribute resolution by output
        r = f"{Ho}x{Wo}"
    else:
        tokens = 1
        for d in ishape[:-1]: tokens *= d
        fl = 2 * tokens * m.in_features * m.out_features
        if ".attn1." in name: cat_ = "self-attn proj (qkv,o)"
        elif ".attn2.to_q" in name or ".attn2.to_out" in name: cat_ = "cross-attn q,o proj"
        elif ".attn2.to_k" in name or ".attn2.to_v" in name: cat_ = "cross-attn k,v proj (audio-only!)"
        elif ".ff." in name: cat_ = "FF GEGLU"
        elif "proj_in" in name or "proj_out" in name: cat_ = "transformer proj_in/out"
        elif "time_emb" in name: cat_ = "time_emb (const)"
        else: cat_ = "linear other"
        r = res_of(ishape) if cat_ != "cross-attn k,v proj (audio-only!)" else "ctx"
        if "time_emb" in name or "time_embedding" in name: r = "const"
    res_cat[(r, cat_)] += fl
# attention score/softmax analytic: for each attn1 at res N tokens, heads h, dim d: QK^T 2*N*N*C, AV 2*N*N*C ; attn2: 2*N*50*C*2
attn_rows = []
for name, m in unet.named_modules():
    if type(m).__name__ == "Attention":
        attn_rows.append((name, m))
# need token counts: capture via hooks on to_q
tok = {}
hooks=[]
for name, m in attn_rows:
    def h(mod, inp, out, name=name):
        tok[name] = (inp[0].shape[1], mod.heads, mod.inner_dim if hasattr(mod,'inner_dim') else mod.to_q.out_features)
    hooks.append(m.to_q.register_forward_hook(lambda mod, inp, out, name=name: tok.__setitem__(name, (inp[0].shape[1], out.shape[-1]))))
with torch.no_grad():
    unet(lat, ts, encoder_hidden_states=ehs)
for h in hooks: h.remove()
for name, (N, C) in tok.items():
    s = int(math.isqrt(N)); r = f"{s}x{s}"
    if name.endswith("attn1"):
        res_cat[(r, "self-attn QK^T+AV (bmm)")] += 2 * 2 * B * N * N * C
    else:
        res_cat[(r, "cross-attn QK^T+AV (bmm)")] += 2 * 2 * B * N * 50 * C

tot_an = sum(res_cat.values())
print(f"\nanalytic total GFLOP/frame: {tot_an/1e9:.2f}")
bycat = collections.defaultdict(float); byres = collections.defaultdict(float)
for (r, c), v in res_cat.items(): bycat[c] += v; byres[r] += v
print("by category:")
for c, v in sorted(bycat.items(), key=lambda kv: -kv[1]): print(f"  {c:36s} {v/1e9:8.2f} GF  {100*v/tot_an:5.1f}%")
print("by resolution:")
for r, v in sorted(byres.items(), key=lambda kv: -kv[1]): print(f"  {r:10s} {v/1e9:8.2f} GF  {100*v/tot_an:5.1f}%")
print("matrix res x cat (GF):")
res_list = ["32x32","16x16","8x8","4x4","ctx","const"]
cats = sorted(bycat, key=lambda c:-bycat[c])
for c in cats:
    print(f"  {c:36s} " + " ".join(f"{res_cat.get((r,c),0)/1e9:7.2f}" for r in res_list))
print("  cols:", res_list)
json.dump({"total_flops_counter": total_flops, "analytic": {f"{r}|{c}": v for (r,c),v in res_cat.items()}, "params": stages}, open("/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/throughput300/model_path/unet_flops.json","w"), indent=1)
