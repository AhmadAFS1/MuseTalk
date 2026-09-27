"""Probe 1+2: PyTorch FP16 eager UNet amortization curve, FLOPs, per-block and per-op-class GPU time."""
import collections
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from common import batch_inputs, cuda_time, gpu_state, load_unet_fp16, save_json, wait_clean

torch.backends.cudnn.benchmark = True
log = []
res = {"gpu_log": log}

unet = load_unet_fp16()
m = unet.model
dev = torch.device("cuda")
t0 = torch.tensor([0], device=dev)
res["params_M"] = sum(p.numel() for p in m.parameters()) / 1e6


def fwd(lat, aud):
    return m(lat, t0, encoder_hidden_states=aud).sample


# ---------- FLOPs ----------
with torch.no_grad():
    lat, aud = batch_inputs(8)
    fc = FlopCounterMode(display=False)
    with fc:
        fwd(lat, aud)
    total = fc.get_total_flops()
    by_op = {str(k): v for k, v in fc.get_flop_counts().get("Global", {}).items()}
    res["flops"] = {"bs8_total_GFLOP": total / 1e9, "per_frame_GFLOP": total / 8 / 1e9,
                    "by_op_GFLOP_bs8": {k: v / 1e9 for k, v in by_op.items()}}
    print("FLOPs/frame GFLOP", total / 8 / 1e9, by_op, flush=True)
FLOP_PER_FRAME = res["flops"]["bs8_total_GFLOP"] * 1e9 / 8

# ---------- amortization curve ----------
res["eager"] = {}
with torch.no_grad():
    for fmt in ["contiguous", "channels_last"]:
        if fmt == "channels_last":
            m.to(memory_format=torch.channels_last)
        for bs in [8, 16, 24, 32]:
            lat, aud = batch_inputs(bs)
            if fmt == "channels_last":
                lat = lat.to(memory_format=torch.channels_last)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            st = wait_clean(f"eager_{fmt}_bs{bs}", log)
            r = cuda_time(lambda: fwd(lat, aud), warmup=10, iters=30)
            r["ms_per_frame"] = r["median_ms"] / bs
            r["fps"] = 1000 * bs / r["median_ms"]
            r["TFLOPS"] = FLOP_PER_FRAME * bs / (r["median_ms"] / 1e3) / 1e12
            r["peak_alloc_GB"] = torch.cuda.max_memory_allocated() / 2**30
            r["activation_peak_GB"] = (torch.cuda.max_memory_allocated() - base) / 2**30
            r["contaminated"] = st["contaminated"]
            res["eager"][f"{fmt}_bs{bs}"] = r
            print(fmt, bs, r, flush=True)
        if fmt == "channels_last":
            m.to(memory_format=torch.contiguous_format)
save_json("eager.json", res)

# ---------- per-top-level-block standalone timing ----------
top = [("time_embedding", m.time_embedding), ("conv_in", m.conv_in)]
top += [(f"down_blocks.{i}", b) for i, b in enumerate(m.down_blocks)]
top += [("mid_block", m.mid_block)]
top += [(f"up_blocks.{i}", b) for i, b in enumerate(m.up_blocks)]
top += [("conv_norm_out", m.conv_norm_out), ("conv_out", m.conv_out)]

sub = []
for bn, blk in top:
    for attr in ["resnets", "attentions", "downsamplers", "upsamplers"]:
        if hasattr(blk, attr) and getattr(blk, attr) is not None:
            for j, s in enumerate(getattr(blk, attr)):
                sub.append((f"{bn}.{attr}.{j}", s))
tsub = []
for n, s in sub:
    if ".attentions." in n:
        tsub += [(n + ".norm", s.norm), (n + ".proj_in", s.proj_in), (n + ".proj_out", s.proj_out)]
        tb = s.transformer_blocks[0]
        tsub += [(n + ".tb.norm1", tb.norm1), (n + ".tb.attn1", tb.attn1), (n + ".tb.norm2", tb.norm2),
                 (n + ".tb.attn2", tb.attn2), (n + ".tb.norm3", tb.norm3), (n + ".tb.ff", tb.ff)]
        tsub += [(n + ".tb.attn1.to_q", tb.attn1.to_q), (n + ".tb.attn1.to_out", tb.attn1.to_out[0]),
                 (n + ".tb.ff.geglu_proj", tb.ff.net[0].proj), (n + ".tb.ff.out", tb.ff.net[2])]
all_mods = top + sub + tsub
captured = {}


def make_cap(name):
    def hook(mod, args, kwargs):
        if name not in captured:
            captured[name] = (args, kwargs)
    return hook


res["blocks"] = {}
for bs in [8, 16]:
    captured.clear()
    hs = [mod.register_forward_pre_hook(make_cap(n), with_kwargs=True) for n, mod in all_mods]
    lat, aud = batch_inputs(bs)
    with torch.no_grad():
        fwd(lat, aud)
    for h in hs:
        h.remove()
    st = wait_clean(f"blocks_bs{bs}", log)
    out = {"_contaminated": st["contaminated"]}
    with torch.no_grad():
        full = cuda_time(lambda: fwd(lat, aud), warmup=10, iters=30)
        out["_full_forward"] = full
        for n, mod in all_mods:
            a, k = captured[n]
            r = cuda_time(lambda: mod(*a, **k), warmup=10, iters=30)
            out[n] = {"median_ms": r["median_ms"], "min_ms": r["min_ms"]}
    tot = sum(out[n]["median_ms"] for n, _ in top)
    out["_sum_top_level_ms"] = tot
    for n, _ in top:
        out[n]["share_of_sum"] = out[n]["median_ms"] / tot
    res["blocks"][f"bs{bs}"] = out
    print(f"bs{bs} sum top {tot:.2f} vs full {full['median_ms']:.2f}", flush=True)
    for n, _ in top:
        print(f"  {n:18s} {out[n]['median_ms']:.3f} ms  {100*out[n]['share_of_sum']:.1f}%")
save_json("eager.json", res)

# ---------- profiler op-class breakdown (kernel time), attributed to top block ----------
from torch.profiler import ProfilerActivity, profile, record_function

scope_stack = []


def pre(name):
    def h(mod, args):
        rf = record_function("SCOPE::" + name)
        rf.__enter__()
        scope_stack.append(rf)
    return h


def post(name):
    def h(mod, args, out):
        rf = scope_stack.pop()
        rf.__exit__(None, None, None)
    return h


hooks = []
for n, mod in top + sub + [(x, y) for x, y in tsub if x.split(".tb.")[-1] in ("attn1", "attn2", "ff", "norm1", "norm2", "norm3")]:
    hooks.append(mod.register_forward_pre_hook(pre(n)))
    hooks.append(mod.register_forward_hook(post(n)))

CLASSIFY = {
    "aten::convolution": "conv", "aten::linear": "gemm_linear", "aten::addmm": "gemm_linear", "aten::mm": "gemm_linear",
    "aten::matmul": "gemm_linear", "aten::bmm": "attn_core", "aten::baddbmm": "attn_core",
    "aten::scaled_dot_product_attention": "attn_core", "aten::group_norm": "groupnorm", "aten::native_group_norm": "groupnorm",
    "aten::layer_norm": "layernorm", "aten::native_layer_norm": "layernorm", "aten::gelu": "elementwise",
    "aten::silu": "elementwise", "aten::add": "elementwise", "aten::mul": "elementwise", "aten::cat": "copy/cat",
    "aten::upsample_nearest2d": "upsample", "aten::copy_": "copy/cat", "aten::clone": "copy/cat",
    "aten::contiguous": "copy/cat", "aten::div": "elementwise", "aten::add_": "elementwise",
}

res["opclass"] = {}
for bs in [8, 16]:
    lat, aud = batch_inputs(bs)
    with torch.no_grad():
        for _ in range(5):
            fwd(lat, aud)
        torch.cuda.synchronize()
        st = wait_clean(f"profile_bs{bs}", log)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(3):
                fwd(lat, aud)
            torch.cuda.synchronize()
    evs = prof.events()
    agg = collections.defaultdict(float)   # (top, sub, cls) -> us
    cls_tot = collections.defaultdict(float)
    kname_tot = collections.defaultdict(float)
    tot_us = 0.0
    for ev in evs:
        if not ev.kernels:
            continue
        kus = sum(k.duration for k in ev.kernels)
        for k in ev.kernels:
            kname_tot[k.name[:90]] += k.duration
        tot_us += kus
        # walk up
        cls = None
        scopes = []
        p = ev
        while p is not None:
            nm = p.name
            if nm.startswith("SCOPE::"):
                scopes.append(nm[7:])
            elif nm in CLASSIFY:
                c = CLASSIFY[nm]
                if nm == "aten::convolution":
                    try:
                        w = p.input_shapes[1]
                        c = "conv1x1(gemm-like)" if (len(w) == 4 and w[2] == 1 and w[3] == 1) else f"conv{w[2]}x{w[3]}"
                    except Exception:
                        c = "conv?"
                cls = c  # outermost classified op wins (keeps walking)
            p = p.cpu_parent
        if cls is None:
            cls = "other:" + ev.name
        topn = scopes[-1] if scopes else "none"
        subn = scopes[-2] if len(scopes) > 1 else "-"
        # finest transformer part if any
        tbn = next((s for s in scopes if ".tb." in s), None)
        agg[(topn, tbn.split(".tb.")[-1] if tbn else (subn.split(".")[-2] if subn != "-" else "-"), cls)] += kus
        cls_tot[cls] += kus
    n_it = 3
    out = {"kernel_total_ms_per_forward": tot_us / 1e3 / n_it, "contaminated": st["contaminated"]}
    out["class_ms_per_forward"] = {k: v / 1e3 / n_it for k, v in sorted(cls_tot.items(), key=lambda x: -x[1])}
    out["class_share"] = {k: v / tot_us for k, v in sorted(cls_tot.items(), key=lambda x: -x[1])}
    per_top = collections.defaultdict(lambda: collections.defaultdict(float))
    for (tp, sp, c), v in agg.items():
        per_top[tp][f"{sp}|{c}"] += v / 1e3 / n_it
    out["per_top_part_class_ms"] = {tp: dict(sorted(d.items(), key=lambda x: -x[1])) for tp, d in per_top.items()}
    out["top_kernels_ms"] = {k: v / 1e3 / n_it for k, v in sorted(kname_tot.items(), key=lambda x: -x[1])[:40]}
    res["opclass"][f"bs{bs}"] = out
    print(f"profile bs{bs} total kernel ms/fwd {out['kernel_total_ms_per_forward']:.2f}")
    for k, v in out["class_ms_per_forward"].items():
        print(f"   {k:28s} {v:7.3f} ms {100*out['class_share'][k]:5.1f}%")
for h in hooks:
    h.remove()
res["peak_reserved_GB_process"] = torch.cuda.max_memory_reserved() / 2**30
save_json("eager.json", res)
print("DONE")
