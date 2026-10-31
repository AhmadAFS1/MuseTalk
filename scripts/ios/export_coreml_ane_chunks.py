#!/usr/bin/env python
"""Export the Apple-layout MuseTalk UNet as two Core ML chunks with fp16 boundaries.

`coremltools.models.utils.bisect_model` (scripts/ios/chunk_coreml.py) cuts the ML
Program by weight bytes and declares the boundary tensors fp32; on an iPhone 12 Pro
the Neural Engine compiler then dropped the chunks to the GPU/CPU (388 ms/frame under
ALL, ~1 s first prediction and 0.9-1.9 GB resident under CPU_AND_NE) where the
single program had run 100% on the ANE at 52 ms. This script cuts the network in
PyTorch instead, at a block boundary, and exports each half with fp16 inputs and
outputs so both chunks look to the ANE compiler exactly like the whole model did.

  chunk1(sample, audio) -> h, res_0..res_N      time embedding, conv_in, down blocks, mid block,
                                                and the first `--cut` up blocks
  chunk2(h, res_0..res_N, audio) -> pred_latents  remaining up blocks, conv_norm_out/act/out
                                                (recomputes the constant time embedding and the
                                                audio hidden states itself: 2M parameters)

Usage (repo root, .venv-ct):
  python scripts/ios/export_coreml_ane_chunks.py --apple-repo /tmp/ml-stable-diffusion --cut 0 --variants fp16,int8w --check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from export_coreml import OUT_DIR, quantize_weights  # noqa: E402
from export_coreml_ane import build_ane_unet  # noqa: E402


def _block(blk, sample, emb, hidden, res=None):
    has_attn = hasattr(blk, "attentions") and blk.attentions is not None
    if res is None:  # down block
        return blk(hidden_states=sample, temb=emb, encoder_hidden_states=hidden) if has_attn else blk(hidden_states=sample, temb=emb)
    if has_attn:
        return blk(hidden_states=sample, temb=emb, res_hidden_states_tuple=res, encoder_hidden_states=hidden)
    return blk(hidden_states=sample, temb=emb, res_hidden_states_tuple=res)


class Chunk1(torch.nn.Module):
    def __init__(self, unet, pe, cut: int):
        super().__init__()
        self.unet, self.cut = unet, cut
        self.register_buffer("pe_table", pe.pe[:, :50, :].clone())
        self.register_buffer("timestep", torch.zeros(1, dtype=torch.long))

    def forward(self, sample, audio):
        u = self.unet
        hidden = (audio + self.pe_table).transpose(1, 2).unsqueeze(2)
        emb = u.time_embedding(u.time_proj(self.timestep))
        sample = u.conv_in(sample)
        res = (sample,)
        for blk in u.down_blocks:
            sample, r = _block(blk, sample, emb, hidden)
            res += r
        sample = u.mid_block(sample, emb, encoder_hidden_states=hidden)
        for blk in u.up_blocks[:self.cut]:
            n = len(blk.resnets)
            sample = _block(blk, sample, emb, hidden, res[-n:])
            res = res[:-n]
        return (sample,) + tuple(res)


class Chunk2(torch.nn.Module):
    def __init__(self, unet, pe, cut: int, n_res: int):
        super().__init__()
        self.unet, self.cut, self.n_res = unet, cut, n_res
        self.register_buffer("pe_table", pe.pe[:, :50, :].clone())
        self.register_buffer("timestep", torch.zeros(1, dtype=torch.long))

    def forward(self, h, *rest):
        res, audio = tuple(rest[:self.n_res]), rest[self.n_res]
        u = self.unet
        hidden = (audio + self.pe_table).transpose(1, 2).unsqueeze(2)
        emb = u.time_embedding(u.time_proj(self.timestep))
        sample = h
        for blk in u.up_blocks[self.cut:]:
            n = len(blk.resnets)
            sample = _block(blk, sample, emb, hidden, res[-n:])
            res = res[:-n]
        sample = u.conv_out(u.conv_act(u.conv_norm_out(sample)))
        return sample


def main() -> None:
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--apple-repo", type=Path, required=True)
    ap.add_argument("--cut", type=int, default=0, help="up blocks included in chunk 1 (0 = cut at the mid/up boundary)")
    ap.add_argument("--variants", default="fp16,int8w")
    ap.add_argument("--attention", default="SPLIT_EINSUM")
    ap.add_argument("--unet-dir", type=Path, default=REPO_ROOT / "models" / "musetalkV15")
    ap.add_argument("--bench-inputs", type=Path, default=REPO_ROOT / "models" / "ios_bench" / "bench_inputs.pt")
    ap.add_argument("--check", action="store_true", help="Core ML chunk1->chunk2 vs the unchunked int8w model on real frames")
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    manifest_path = args.out_dir / "manifest.json"

    print(f"building Apple-layout UNet ({args.attention})", flush=True)
    unet, ref_unet, pe = build_ane_unet(args.apple_repo, args.unet_dir, args.attention)
    c1 = Chunk1(unet, pe, args.cut).eval()
    d = torch.load(args.bench_inputs, map_location="cpu", weights_only=True)
    lat, aud = d["latents"][:4].float(), d["audio"][:4].float()
    with torch.no_grad():
        outs1 = c1(lat[:1], aud[:1])
    n_res = len(outs1) - 1
    c2 = Chunk2(unet, pe, args.cut, n_res).eval()
    res_names = [f"res_{i}" for i in range(n_res)]
    print(f"cut {args.cut}: chunk1 -> h {tuple(outs1[0].shape)} + {n_res} residuals "
          f"{[tuple(r.shape[1:]) for r in outs1[1:]]}", flush=True)
    p1 = sum(p.numel() for n, p in c1.unet.named_parameters()
             if n.startswith(("time_embedding", "conv_in", "down_blocks", "mid_block"))
             or any(n.startswith(f"up_blocks.{i}.") for i in range(args.cut))) / 1e6
    p_all = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"  parameters: chunk1 {p1:.0f}M, chunk2 {p_all - p1:.0f}M (+2M shared head)", flush=True)

    # exactness of the split in fp32 torch
    with torch.no_grad():
        want = ref_unet(lat, torch.zeros(1, dtype=torch.long), encoder_hidden_states=pe(aud)).sample
        o1 = c1(lat, aud)
        got = c2(o1[0], *o1[1:], aud)
    print(f"  fp32 torch chunk1->chunk2 vs diffusers UNet: max |diff| {(want - got).abs().max():.2e}", flush=True)
    del ref_unet

    tag = f"unet_anec{args.cut}"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def record(name, path, extra):
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        m[name] = {"path": str(path), "bytes": size, **extra}
        manifest_path.write_text(json.dumps(m, indent=2) + "\n")
        print(f"  saved {path} ({size / 2**20:.0f} MB)", flush=True)

    def convert(module, example, in_specs, out_specs):
        with torch.no_grad():
            ep = torch.export.export(module, example, strict=False).run_decompositions({})
        return ct.convert(ep, inputs=in_specs, outputs=out_specs, convert_to="mlprogram",
                          minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16)

    f16 = np.float16
    # chunk 1
    t0 = time.time()
    print("chunk1: exporting", flush=True)
    m1 = convert(c1, (torch.randn(1, 8, 32, 32), torch.randn(1, 50, 384)),
                 [ct.TensorType(name="sample", shape=(1, 8, 32, 32), dtype=f16), ct.TensorType(name="audio", shape=(1, 50, 384), dtype=f16)],
                 [ct.TensorType(name="h", dtype=f16)] + [ct.TensorType(name=n, dtype=f16) for n in res_names])
    print(f"  converted in {time.time() - t0:.0f}s", flush=True)
    # chunk 2
    t0 = time.time()
    print("chunk2: exporting", flush=True)
    ex2 = (outs1[0],) + tuple(outs1[1:]) + (aud[:1],)
    in2 = [ct.TensorType(name="h", shape=tuple(outs1[0].shape), dtype=f16)]
    in2 += [ct.TensorType(name=n, shape=tuple(r.shape), dtype=f16) for n, r in zip(res_names, outs1[1:])]
    in2 += [ct.TensorType(name="audio", shape=(1, 50, 384), dtype=f16)]
    m2 = convert(c2, ex2, in2, [ct.TensorType(name="pred_latents", dtype=f16)])
    print(f"  converted in {time.time() - t0:.0f}s", flush=True)

    for variant in args.variants.split(","):
        parent = f"{tag}_{variant}_b1"
        for i, m in ((1, m1), (2, m2)):
            model = quantize_weights(m) if variant == "int8w" else m
            path = args.out_dir / f"{parent}_chunk{i}.mlpackage"
            model.save(str(path))
            spec = model.get_spec()
            record(f"{parent}_chunk{i}", path, {
                "kind": "unet_chunk", "chunk_of": parent, "chunk_index": i, "batch": 1, "variant": f"anec{args.cut}_{variant}",
                "layout": "apple-ane", "cut": args.cut,
                "inputs": [x.name for x in spec.description.input], "outputs": [x.name for x in spec.description.output]})
            if variant == "int8w":
                del model

    if not args.check:
        return
    from musetalk.models.unet import PositionalEncoding
    from scripts.validate_unet_backend import _load_capture_group, discover_captures, tensor_stats

    units = getattr(ct.ComputeUnit, args.units)
    parent = f"{tag}_int8w_b1" if "int8w" in args.variants else f"{tag}_fp16_b1"
    base_name = "unet_ane_int8w_b1" if "int8w" in args.variants else "unet_ane_fp16_b1"
    man = json.loads(manifest_path.read_text())
    full = ct.models.MLModel(man[base_name]["path"], compute_units=units)
    parts = [ct.models.MLModel(man[f"{parent}_chunk{i}"]["path"], compute_units=units) for i in (1, 2)]
    pe_t = PositionalEncoding(d_model=384).pe[:, :50, :].float()
    full_out, chain_out, refs = [], [], []
    t_full = t_chain = 0.0
    for path in discover_captures(REPO_ROOT / "calibration" / "unet_portable_bs8", 2):
        payload = _load_capture_group([path])
        n = int(payload.get("actual_batch", payload["latent_batch"].shape[0]))
        L, A, R = payload["latent_batch"].float()[:n], payload["audio_feature_batch"].float()[:n] - pe_t, payload["pred_latents"].float()[:n]
        for s in range(n):
            feed = {"sample": L[s:s + 1].numpy().astype(f16), "audio": A[s:s + 1].numpy().astype(f16)}
            t = time.perf_counter(); fo = np.asarray(full.predict(feed)["pred_latents"], dtype=np.float32); t_full += time.perf_counter() - t
            t = time.perf_counter()
            o1 = parts[0].predict(feed)
            feats = {**feed, **o1}
            names2 = [x.name for x in parts[1].get_spec().description.input]
            co = np.asarray(parts[1].predict({k: feats[k] for k in names2})["pred_latents"], dtype=np.float32)
            t_chain += time.perf_counter() - t
            full_out.append(fo); chain_out.append(co); refs.append(R[s:s + 1].numpy())
    F, C, Rf = (torch.from_numpy(np.concatenate(x)) for x in (full_out, chain_out, refs))
    n = F.shape[0]
    dd = (F - C).abs()
    sf, sc = tensor_stats(F, Rf), tensor_stats(C, Rf)
    print(f"\n[{args.units}] torch-cut chunks vs unchunked on {n} real frames: max |diff| {dd.max():.2e}, mean {dd.mean():.2e}")
    print(f"  vs capture reference: unchunked mae {sf['mae']:.5f} max_abs {sf['max_abs']:.4f}; chunked mae {sc['mae']:.5f} max_abs {sc['max_abs']:.4f}")
    print(f"  ms/frame incl. Python I/O: unchunked {t_full / n * 1000:.1f}, chunked {t_chain / n * 1000:.1f}")
    out = REPO_ROOT / "docs" / "ios_unet_bench" / f"chunk_{parent}_m5pro.json"
    out.write_text(json.dumps({"parent": parent, "cut": args.cut, "units": args.units, "frames": n,
                               "chunk_vs_full_max_abs": float(dd.max()), "chunk_vs_full_mean_abs": float(dd.mean()),
                               "unchunked_vs_ref": sf, "chunked_vs_ref": sc,
                               "ms_per_frame_unchunked": t_full / n * 1000, "ms_per_frame_chunked": t_chain / n * 1000,
                               "chunks": {f"chunk{i}": man[f"{parent}_chunk{i}"]["bytes"] for i in (1, 2)}}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
