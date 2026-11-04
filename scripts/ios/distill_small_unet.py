#!/usr/bin/env python
"""Distil a smaller MuseTalk UNet from the full one: BK-SDM-style block removal + knowledge distillation (proof of concept).

The smaller shapes come from `export_small_unets.py` (bk_base, bk_small, bk_tiny), and start from the full model's
weights wherever the shapes match. The full UNet is the teacher. Each step pairs random avatar frames (the UNet's
input latents from server avatar folders, `latents.pt`) with random audio windows (Whisper features of any speech),
so the targets are the teacher's own outputs and no ground-truth video is needed. Loss, as in BK-SDM:
  out KD   mse(student output, teacher output)
  feat KD  mse at the output of every up block (student up block i against teacher up block i + offset)

Scored on the 128 harness gate frames against the server fp16 reference, the same test as every other UNet change
(ios_coreml_implementation.md 4.1): the full model in fp32 is mae 0.0004 there, the Neural Engine's whole error 0.013.

Usage (repo root, .venv-ct):
  python scripts/ios/distill_small_unet.py --variant bk_base --avatars '<dir>/*/latents.pt' --steps 2000 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from export_small_unets import build  # noqa: E402
from musetalk.models.unet import UNet  # noqa: E402

HB = REPO_ROOT / "ios" / "MuseTalkBench" / "HarnessBundle"


def gate_frames():
    lat = np.fromfile(HB / "unet_latents_f16.bin", dtype=np.float16).reshape(128, 8, 32, 32)
    aud = np.fromfile(HB / "unet_audio_f16.bin", dtype=np.float16).reshape(128, 50, 384)
    ref = np.fromfile(HB / "unet_ref_f16.bin", dtype=np.float16).reshape(128, 4, 32, 32)
    return (torch.from_numpy(lat.astype(np.float32)), torch.from_numpy(aud.astype(np.float32)),
            torch.from_numpy(ref.astype(np.float32)))


def whisper_windows(paths: list[Path], device) -> torch.Tensor:
    """[frames, 50, 384] audio windows at 25 fps for every clip (no positional encoding: the UNet wrapper adds it)."""
    from transformers import WhisperModel
    from musetalk.utils.audio_processor import AudioProcessor

    wdir = REPO_ROOT / "models" / "whisper"
    proc = AudioProcessor(feature_extractor_path=str(wdir))
    whisper = WhisperModel.from_pretrained(str(wdir)).to(device).eval().requires_grad_(False)
    out = []
    for p in paths:
        feats, length = proc.get_audio_feature(str(p))
        chunks = proc.get_whisper_chunk(feats, device, torch.float32, whisper, length, fps=25)
        out.append(chunks.float().cpu())
        print(f"  audio {p.name}: {chunks.shape[0]} windows", flush=True)
    del whisper
    return torch.cat(out)


class Taps:
    """Forward hooks that keep every up block's output."""

    def __init__(self, unet):
        self.out = []
        self.handles = [b.register_forward_hook(lambda m, i, o: self.out.append(o if torch.is_tensor(o) else o[0]))
                        for b in unet.up_blocks]

    def take(self):
        o, self.out = self.out, []
        return o


def score(student_step, lat, aud, ref, device, batch=16):
    student_step.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(lat), batch):
            outs.append(student_step(lat[i:i + batch].to(device), aud[i:i + batch].to(device)).float().cpu())
    student_step.train()
    e = (torch.cat(outs) - ref).abs()
    per, low = e.flatten(1).mean(1), e[:, :, 13:].flatten(1).mean(1)
    return {"mae": float(per.mean()), "mae_worst": float(per.max()), "blend_mae": float(low.mean()),
            "blend_worst": float(low.max()), "max_abs": float(e.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bk_base")
    ap.add_argument("--avatars", required=True, help="glob of server avatar latents.pt files")
    ap.add_argument("--audio", default=str(REPO_ROOT / "data" / "audio" / "*"), help="glob of speech files")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    lat_files = sorted(glob.glob(args.avatars))
    lats = torch.cat([torch.load(f, map_location="cpu", weights_only=True).reshape(-1, 8, 32, 32).float() for f in lat_files])
    print(f"{len(lat_files)} avatars, {len(lats)} frames", flush=True)
    audio_files = sorted(Path(p) for p in glob.glob(args.audio) if Path(p).suffix in (".wav", ".mpga", ".mp3", ".m4a"))
    auds = whisper_windows(audio_files, dev)
    print(f"{len(audio_files)} clips, {len(auds)} audio windows", flush=True)

    ud = REPO_ROOT / "models" / "musetalkV15"
    tw = UNet(unet_config=str(ud / "musetalk.json"), model_path=str(ud / "unet.pth"), device=torch.device("cpu"))
    teacher = tw.model.to(dev).eval().requires_grad_(False)
    pe = tw.pe.to(dev)
    step, _ = build(args.variant, ud)
    student = step.unet.to(dev).train().requires_grad_(True)
    step = step.to(dev)
    t_taps, s_taps = Taps(teacher), Taps(student)
    offset = len(teacher.up_blocks) - len(student.up_blocks)  # bk_tiny drops the innermost up stage
    tstep = torch.zeros(1, dtype=torch.long, device=dev)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    g_lat, g_aud, g_ref = gate_frames()
    s0 = score(step, g_lat, g_aud, g_ref, dev)
    s_taps.take()  # the scoring passes tapped the student too
    print(f"step 0: {json.dumps(s0)}", flush=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": 0, **s0, "variant": args.variant}) + "\n")
    best = s0["blend_mae"]
    t0 = time.time()
    for it in range(1, args.steps + 1):
        x = lats[torch.randint(len(lats), (args.batch,))].to(dev)
        a = auds[torch.randint(len(auds), (args.batch,))].to(dev)
        hidden = pe(a)
        with torch.no_grad():
            t_out = teacher(x, tstep, encoder_hidden_states=hidden).sample
            t_feats = t_taps.take()
        s_out = student(x, tstep, encoder_hidden_states=hidden).sample
        s_feats = s_taps.take()
        loss_out = F.mse_loss(s_out, t_out)
        loss_feat = sum(F.mse_loss(s, t_feats[i + offset]) for i, s in enumerate(s_feats)) / len(s_feats)
        loss = loss_out + args.feat_weight * loss_feat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if it % 25 == 0:
            el = time.time() - t0
            print(f"step {it}: loss {loss.item():.5f} (out {loss_out.item():.5f}, feat {loss_feat.item():.5f}), "
                  f"{el / it:.2f} s/step", flush=True)
        if it % args.eval_every == 0 or it == args.steps:
            s = score(step, g_lat, g_aud, g_ref, dev)
            s_taps.take()
            s.update({"step": it, "loss": loss.item(), "s_per_step": (time.time() - t0) / it})
            print(f"eval {it}: {json.dumps(s)}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(s) + "\n")
            if s["blend_mae"] < best:
                best = s["blend_mae"]
                torch.save(student.state_dict(), args.out / f"{args.variant}_best.pt")
    torch.save(student.state_dict(), args.out / f"{args.variant}_last.pt")


if __name__ == "__main__":
    main()
