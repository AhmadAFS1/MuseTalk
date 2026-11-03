"""Measured and rejected 2026-11-02 (ios_coreml_implementation.md 3.5): deep-feature reuse across adjacent frames (DeepCache-style, across video frames instead of denoising steps).

Full frame: chunk1 -> chunk2, and chunk2 also returns the input of up block (4 - level) ("deep").
Reuse frame: only the top `level` resolution levels run (conv_in, down blocks < level without their last
downsampler, up blocks >= 4 - level, conv_out) with the fresh sample latent and audio window, on the previous
frame's deep features.

  level 1: recompute 32x32 only (deep = up_blocks[2] output, [1, 640, 32, 32])
  level 2: recompute 32x32 and 16x16 (deep = up_blocks[1] output, [1, 1280, 16, 16])

Step 1 (torch fp32, CPU): error of reuse frames vs the server fp16 reference on the harness gate frames
(128 frames in 16 groups of 8 consecutive frames), next to the full computation's error.
Step 2 (--export): Core ML export of the shallow models (V2 attention, fp16 compute, int8 weights) and of a
chunk 2 that also outputs `deep`, then Python timing on CPU_AND_NE.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

MT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MT))
sys.path.insert(0, str(MT / "scripts" / "ios"))
from export_coreml_ane import build_ane_unet  # noqa: E402
from export_coreml_ane_chunks import Chunk1, _block  # noqa: E402

HB = MT / "ios" / "MuseTalkBench" / "HarnessBundle"


class Chunk2Deep(torch.nn.Module):
    """Chunk 2 (cut 0) that also returns the input of up block 4 - level."""

    def __init__(self, unet, pe, n_res: int, level: int):
        super().__init__()
        self.unet, self.n_res, self.level = unet, n_res, level
        self.register_buffer("pe_table", pe.pe[:, :50, :].clone())
        self.register_buffer("timestep", torch.zeros(1, dtype=torch.long))

    def forward(self, h, *rest):
        res, audio = tuple(rest[:self.n_res]), rest[self.n_res]
        u = self.unet
        hidden = (audio + self.pe_table).transpose(1, 2).unsqueeze(2)
        emb = u.time_embedding(u.time_proj(self.timestep))
        sample, deep = h, None
        for i, blk in enumerate(u.up_blocks):
            if i == 4 - self.level:
                deep = sample
            n = len(blk.resnets)
            sample = _block(blk, sample, emb, hidden, res[-n:])
            res = res[:-n]
        return u.conv_out(u.conv_act(u.conv_norm_out(sample))), deep.contiguous()


class Shallow(torch.nn.Module):
    """The top `level` resolution levels on cached deep features."""

    def __init__(self, unet, pe, level: int):
        super().__init__()
        self.unet, self.level = unet, level
        self.register_buffer("pe_table", pe.pe[:, :50, :].clone())
        self.register_buffer("timestep", torch.zeros(1, dtype=torch.long))

    def forward(self, sample, audio, deep):
        u = self.unet
        hidden = (audio + self.pe_table).transpose(1, 2).unsqueeze(2)
        emb = u.time_embedding(u.time_proj(self.timestep))
        s = u.conv_in(sample)
        res = (s,)
        for i in range(self.level):
            blk = u.down_blocks[i]
            for resnet, attn in zip(blk.resnets, blk.attentions):
                s = resnet(s, emb)
                s = attn(s, context=hidden)
                res += (s,)
            if i < self.level - 1 and blk.downsamplers is not None:
                for d in blk.downsamplers:
                    s = d(s)
                res += (s,)
        x = deep
        for blk in u.up_blocks[4 - self.level:]:
            n = len(blk.resnets)
            x = _block(blk, x, emb, hidden, res[-n:])
            res = res[:-n]
        return u.conv_out(u.conv_act(u.conv_norm_out(x)))


def gate_frames():
    lat = np.fromfile(HB / "unet_latents_f16.bin", dtype=np.float16).reshape(128, 8, 32, 32)
    aud = np.fromfile(HB / "unet_audio_f16.bin", dtype=np.float16).reshape(128, 50, 384)
    ref = np.fromfile(HB / "unet_ref_f16.bin", dtype=np.float16).reshape(128, 4, 32, 32)
    return torch.from_numpy(lat.astype(np.float32)), torch.from_numpy(aud.astype(np.float32)), torch.from_numpy(ref.astype(np.float32))


def stats(err: torch.Tensor):
    per = err.abs().flatten(1)
    return float(per.mean(1).mean()), float(per.mean(1).max()), float(per.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apple-repo", type=Path, required=True)
    ap.add_argument("--attention", default="SPLIT_EINSUM_V2")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--export", type=Path, default=None, help="directory for the Core ML models")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    unet, ref_unet, pe = build_ane_unet(args.apple_repo, MT / "models" / "musetalkV15", args.attention)
    del ref_unet
    lat, aud, ref = gate_frames()
    c1 = Chunk1(unet, pe, 0).eval()
    levels = [int(x) for x in args.levels.split(",")]

    if not args.skip_check:
        with torch.no_grad():
            outs = [c1(lat[i:i + 1], aud[i:i + 1]) for i in range(128)]
            n_res = len(outs[0]) - 1
            for level in levels:
                c2 = Chunk2Deep(unet, pe, n_res, level).eval()
                sh = Shallow(unet, pe, level).eval()
                full, deep = zip(*[c2(o[0], *o[1:], aud[i:i + 1]) for i, o in enumerate(outs)])
                full = torch.cat(full)
                # sanity: shallow on its own frame's deep must equal the full computation
                same = sh(lat[0:1], aud[0:1], deep[0])
                print(f"level {level}: shallow(own deep) vs full max |diff| {(same - full[0:1]).abs().max():.2e}", flush=True)
                idx = [i for i in range(128) if i % 8 != 0]  # frames with a predecessor in the same group of 8
                reuse = torch.cat([sh(lat[i:i + 1], aud[i:i + 1], deep[i - 1]) for i in idx])
                f_m, f_w, f_x = stats(full[idx] - ref[idx])
                r_m, r_w, r_x = stats(reuse - ref[idx])
                d_m, d_w, d_x = stats(reuse - full[idx])
                print(f"level {level}: vs server ref on {len(idx)} reuse frames: full fp32 mae {f_m:.5f} (worst frame {f_w:.5f}, "
                      f"max_abs {f_x:.3f}); reuse mae {r_m:.5f} (worst frame {r_w:.5f}, max_abs {r_x:.3f}); "
                      f"reuse vs full mae {d_m:.5f} (worst {d_w:.5f}, max_abs {d_x:.3f})", flush=True)
                # two in a row: frame i on deep from i - 2 (what a 1-full-2-reuse schedule would see)
                idx2 = [i for i in range(128) if i % 8 >= 2]
                reuse2 = torch.cat([sh(lat[i:i + 1], aud[i:i + 1], deep[i - 2]) for i in idx2])
                r2 = stats(reuse2 - ref[idx2])
                print(f"level {level}: deep from 2 frames back: reuse mae {r2[0]:.5f} (worst {r2[1]:.5f}, max_abs {r2[2]:.3f})", flush=True)

    if args.export:
        import coremltools as ct
        from export_coreml import quantize_weights
        args.export.mkdir(parents=True, exist_ok=True)
        f16 = np.float16
        with torch.no_grad():
            o = c1(lat[:1], aud[:1])
        n_res = len(o) - 1
        for level in levels:
            sh = Shallow(unet, pe, level).eval()
            c2 = Chunk2Deep(unet, pe, n_res, level).eval()
            with torch.no_grad():
                _, deep = c2(o[0], *o[1:], aud[:1])
                ep = torch.export.export(sh, (lat[:1], aud[:1], deep), strict=False).run_decompositions({})
            t0 = time.time()
            m = ct.convert(ep, inputs=[ct.TensorType(name="sample", shape=(1, 8, 32, 32), dtype=f16),
                                       ct.TensorType(name="audio", shape=(1, 50, 384), dtype=f16),
                                       ct.TensorType(name="deep", shape=tuple(deep.shape), dtype=f16)],
                           outputs=[ct.TensorType(name="pred_latents", dtype=f16)], convert_to="mlprogram",
                           minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16)
            m = quantize_weights(m)
            path = args.export / f"unet_shallow{level}_int8w_b1.mlpackage"
            m.save(str(path))
            print(f"level {level}: shallow exported in {time.time() - t0:.0f}s -> {path} (deep {tuple(deep.shape)})", flush=True)


if __name__ == "__main__":
    main()
