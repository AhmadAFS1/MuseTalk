#!/usr/bin/env python
"""Distil a smaller MuseTalk UNet from the full one (BK-SDM block removal + knowledge distillation), sized for one
rented CUDA GPU (an RTX 3090 for 4 hours).

The smaller shapes come from `export_small_unets.py` (bk_base, bk_small, bk_tiny; bk_tiny is the one fast enough for
25 fps on the iPhone 12 Pro, ios_coreml_implementation.md 5.8) and start from the full model's weights wherever the
shapes match. The full UNet is the teacher. Each step pairs random avatar frames (the UNet's input latents) with random
audio windows (Whisper features of any speech), so the targets are the teacher's own outputs and no ground-truth video
is needed. Loss, as in BK-SDM:
  out KD   mse(student output, teacher output)
  feat KD  mean mse at the output of every up block: student up block i against teacher up block i + shift, where
           bk_tiny's shift of 1 skips the teacher's innermost (4x4) up stage, the one bk_tiny drops

Three sets, kept apart:
  train  every avatar not held out
  val    the held-out avatars (--val-avatars, or --val-count of them spread over the sorted ids): fixed (frame, audio)
         pairs scored against the teacher, the only number a checkpoint is chosen on
  test   the 128 harness gate frames against the server fp16 reference, the test every other UNet change took
         (ios_coreml_implementation.md 4.1: the full model in fp32 is mae 0.0004 there, the Neural Engine's own
         error 0.013). Reported at every eval, never selected on. The gate captures come from production avatars:
         the start record's gate_overlap names the loaded avatars whose frames are gate inputs.

Precision: on CUDA the teacher runs in fp16 (the server's precision, and the gate reference's), and the student keeps
fp32 master weights and trains under bf16 autocast; on mps/cpu both are fp32. Both nets use SDPA attention here;
evals run the student in fp32 with TF32 off. Core ML export keeps the explicit attention (build(processor="export")).

Data: --latents takes server avatar latents.pt ([2N,1,8,32,32], the forward+reverse cycle: the mirrored half is
dropped) and phone bundle latents_f16.bin ([N,8,32,32] fp16), avatar id = the parent directory (the bundle's, for a
bundle's avatar/ folder). A frame reached twice (an avatar's server folder and its phone bundle) is kept once, under
the first id. Latents and audio windows are moved to the device once; the Whisper windows are cached in
<out>/audio_windows.pt, so a resume does not load Whisper.

Budget: --max-minutes counts training time across resumes, and the LR's cosine runs over whichever of --steps and
--max-minutes ends first, so the run anneals when the rental does. <out>/ckpt.pt every --ckpt-minutes; --resume
continues from it. Outputs: <variant>_best.pt (by val) and <variant>_last.pt, state_dicts the stock
musetalk.models.unet.UNet wrapper loads strictly with <variant>.json; log.jsonl (one record per start and per eval).

Usage (repo root; .venv-ct on the Mac, the MuseTalk server venv on CUDA):
  python scripts/ios/distill_small_unet.py --variant bk_tiny --out <run> --val-count 4 --max-minutes 200 --steps 100000 \\
      --latents '<avatars>/*/latents.pt,<bundles>/*/avatar/latents_f16.bin' [--resume]
  python scripts/ios/export_small_unets.py --variants bk_tiny --weights <run>/bk_tiny_best.pt --tag r1 --time
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from export_coreml import UNetStep  # noqa: E402
from export_small_unets import VARIANTS, build, load_full, variant_config  # noqa: E402
from musetalk.models.unet import PositionalEncoding  # noqa: E402

HB = REPO_ROOT / "ios" / "MuseTalkBench" / "HarnessBundle"
AUDIO_SUFFIXES = (".wav", ".mpga", ".mp3", ".m4a", ".flac")
FRAME = 8 * 32 * 32


def expand(globs: str) -> list[Path]:
    files = []
    for g in (s.strip() for s in globs.split(",")):
        if g:
            hits = sorted(glob.glob(os.path.expanduser(g)))
            if not hits:
                print(f"  warning: {g} matched nothing", flush=True)
            files += [Path(h) for h in hits]
    return files


def load_latents(path: Path) -> torch.Tensor:
    """[N, 8, 32, 32] fp16, each frame once."""
    if path.suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float16)
        if raw.size % FRAME:
            raise SystemExit(f"{path}: {raw.nbytes} bytes is not a whole number of [8,32,32] fp16 frames")
        return torch.from_numpy(raw.reshape(-1, 8, 32, 32).copy())
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, (list, tuple)):  # older avatar folders saved the cycle as a list of [1,8,32,32]
        t = torch.cat([x.reshape(-1, 8, 32, 32) for x in t])
    t = t.reshape(-1, 8, 32, 32).half()
    n = len(t) // 2
    if len(t) % 2 == 0 and torch.equal(t[n:], t[:n].flip(0)):
        t = t[:n]
    return t.contiguous()


def load_avatars(files: list[Path]) -> dict[str, torch.Tensor]:
    """avatar id -> its frames, each frame once over all files: an avatar reached twice (its server folder and its
    phone bundle) would otherwise put a held-out avatar's frames into train under the other id."""
    owner: dict[bytes, str] = {}
    parts: dict[str, list[torch.Tensor]] = {}
    for f in files:
        aid = f.parent.parent.name if f.parent.name == "avatar" else f.parent.name
        lat = load_latents(f)
        keys = [hashlib.sha1(x.numpy().tobytes()).digest() for x in lat]
        new = [i for i, k in enumerate(keys) if k not in owner]
        if len(new) < len(keys):
            seen = ", ".join(sorted({owner[k] for k in keys if k in owner}))
            print(f"  {f}: {len(keys) - len(new)} of {len(keys)} frames already loaded as {seen}"
                  f"{', skipped' if not new else ''}", flush=True)
        for i in new:
            owner[keys[i]] = aid
        if new:
            parts.setdefault(aid, []).append(lat[new])
    return {aid: torch.cat(p) for aid, p in sorted(parts.items())}


def split(ids: list[str], val_avatars: str | None, val_count: int | None) -> tuple[list[str], list[str]]:
    if val_avatars:
        val = [s.strip() for s in val_avatars.split(",") if s.strip()]
        missing = [v for v in val if v not in ids]
        if missing:
            raise SystemExit(f"--val-avatars not among the loaded avatars: {missing}")
    else:
        k = val_count if val_count is not None else max(1, len(ids) // 10)
        if not 1 <= k < len(ids):
            raise SystemExit(f"--val-count {k} with {len(ids)} avatars: need at least one each for val and train")
        # spread over the sorted ids, so the held-out set does not cluster on one name prefix (language, batch)
        val = [ids[int((i + 0.5) * len(ids) / k)] for i in range(k)]
    train = [i for i in ids if i not in val]
    if not train:
        raise SystemExit("every avatar is held out for val")
    return train, val


def audio_windows(files: list[Path], cache: Path, device) -> torch.Tensor:
    """[windows, 50, 384] fp16 on the CPU: every clip's Whisper windows at 25 fps, without the positional encoding
    (UNetStep adds it). Cached under the clips' names and sizes."""
    key = [[f.name, f.stat().st_size] for f in files]
    if cache.exists():
        c = torch.load(cache, map_location="cpu", weights_only=True)
        if c["files"] == key:
            print(f"  audio windows from {cache}", flush=True)
            return c["windows"]
        print(f"  {cache} holds other clips: recomputing", flush=True)
    from transformers import WhisperModel
    from musetalk.utils.audio_processor import AudioProcessor

    wdir = REPO_ROOT / "models" / "whisper"
    proc = AudioProcessor(feature_extractor_path=str(wdir))
    whisper = WhisperModel.from_pretrained(str(wdir)).to(device).eval().requires_grad_(False)
    out = []
    with torch.no_grad():
        for p in files:
            feats, length = proc.get_audio_feature(str(p))
            chunks = proc.get_whisper_chunk(feats, device, torch.float32, whisper, length, fps=25)
            out.append(chunks.half().cpu())
            print(f"  audio {p.name}: {chunks.shape[0]} windows", flush=True)
    del whisper
    windows = torch.cat(out)
    torch.save({"files": key, "windows": windows}, cache)
    return windows


def gate_frames(limit: int):
    lat = np.fromfile(HB / "unet_latents_f16.bin", dtype=np.float16).reshape(128, 8, 32, 32)[:limit]
    aud = np.fromfile(HB / "unet_audio_f16.bin", dtype=np.float16).reshape(128, 50, 384)[:limit]
    ref = np.fromfile(HB / "unet_ref_f16.bin", dtype=np.float16).reshape(128, 4, 32, 32)[:limit]
    return torch.from_numpy(lat.copy()), torch.from_numpy(aud.copy()), torch.from_numpy(ref.astype(np.float32))


def gate_overlap(gate_lat: torch.Tensor, avatars: dict[str, torch.Tensor]) -> dict[str, int]:
    """avatar id -> how many gate inputs are byte-identical to one of its frames."""
    keys = {g.numpy().tobytes() for g in gate_lat}
    hits = {aid: sum(f.numpy().tobytes() in keys for f in lat) for aid, lat in avatars.items()}
    return {aid: n for aid, n in hits.items() if n}


class Taps:
    """Forward hooks that keep every up block's output while `on`, by block index."""

    def __init__(self, unet):
        self.on, self.out = True, {}
        for i, b in enumerate(unet.up_blocks):
            b.register_forward_hook(self._keeper(i))

    def _keeper(self, i: int):
        def keep(module, inputs, output):
            if self.on:
                self.out[i] = output if torch.is_tensor(output) else output[0]
        return keep

    def take(self) -> list[torch.Tensor]:
        o, self.out = self.out, {}
        return [o[i] for i in sorted(o)]


@torch.no_grad()
def predict(step, lat, aud, dtype, batch: int = 16) -> torch.Tensor:
    return torch.cat([step(lat[i:i + batch].to(dtype), aud[i:i + batch].to(dtype)).float()
                      for i in range(0, len(lat), batch)])


def stats(pred: torch.Tensor, ref: torch.Tensor, prefix: str) -> dict:
    e = (pred - ref).abs()
    per, low = e.flatten(1).mean(1), e[:, :, 13:].flatten(1).mean(1)
    return {f"{prefix}_mae": float(per.mean()), f"{prefix}_mae_worst": float(per.max()),
            f"{prefix}_blend_mae": float(low.mean()), f"{prefix}_blend_worst": float(low.max()),
            f"{prefix}_max_abs": float(e.max())}


def save(obj, path: Path) -> None:
    """Write-then-rename, so a rental that dies mid-save leaves the previous file whole."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bk_tiny", choices=[v for v in VARIANTS if v != "full"])
    ap.add_argument("--latents", required=True, help="comma-separated globs of avatar latents.pt / bundle latents_f16.bin")
    ap.add_argument("--audio", default=str(REPO_ROOT / "data" / "audio" / "*"), help="comma-separated globs of speech files")
    ap.add_argument("--exclude-audio", default="ai-assistant.mpga,eng.wav", help="basenames left out (the video review's clips)")
    ap.add_argument("--val-avatars", help="comma-separated avatar ids held out for checkpoint selection")
    ap.add_argument("--val-count", type=int, help="or hold out this many (default: a tenth of the avatars, at least one)")
    ap.add_argument("--val-pairs", type=int, default=256)
    ap.add_argument("--gate-frames", type=int, default=128, help="fewer only for a smoke test")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-minutes", type=float, default=0, help="training-time budget across resumes (0: none)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing, if the batch does not fit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if it exists")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    t_dtype = torch.float16 if cuda else torch.float32
    if cuda:
        torch.backends.cudnn.benchmark = True
        # bf16 training is unaffected; this keeps the fp32 evals fp32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path, ckpt_path = args.out / "log.jsonl", args.out / "ckpt.pt"
    ud = REPO_ROOT / "models" / "musetalkV15"
    (args.out / f"{args.variant}.json").write_text(json.dumps(variant_config(args.variant, ud), indent=2) + "\n")

    avatars = load_avatars(expand(args.latents))
    if not avatars:
        raise SystemExit("no latents")
    train_ids, val_ids = split(list(avatars), args.val_avatars, args.val_count)
    g_lat, g_aud, g_ref = gate_frames(args.gate_frames)
    overlap = gate_overlap(g_lat, avatars)
    print(f"{len(avatars)} avatars, {sum(len(v) for v in avatars.values())} frames; val: {', '.join(val_ids)}", flush=True)
    if overlap:
        print(f"  gate inputs that are also frames of the loaded avatars: {overlap}", flush=True)
    excluded = {s.strip() for s in args.exclude_audio.split(",") if s.strip()}
    audio_files = [p for p in expand(args.audio) if p.suffix.lower() in AUDIO_SUFFIXES and p.name not in excluded]
    if not audio_files:
        raise SystemExit("no audio")
    auds = audio_windows(audio_files, args.out / "audio_windows.pt", dev).to(dev)
    print(f"{len(audio_files)} clips, {len(auds)} audio windows", flush=True)

    full = load_full(ud)
    student, params = build(args.variant, ud, processor="train", full=full)
    student = student.to(dev)
    unet = student.unet.train().requires_grad_(True)
    if args.grad_ckpt:
        unet.enable_gradient_checkpointing()
    from diffusers.models.attention_processor import AttnProcessor2_0
    full.set_attn_processor(AttnProcessor2_0())
    teacher = UNetStep(full, PositionalEncoding(d_model=384)).to(dev, t_dtype).eval()
    t_taps, s_taps = Taps(teacher.unet), Taps(unet)
    shift = len(teacher.unet.up_blocks) - len(unet.up_blocks)

    train_lat = torch.cat([avatars[i] for i in train_ids]).to(dev)
    vgen = torch.Generator().manual_seed(args.seed + 1)
    v_lat, v_aud = [], []
    for i in range(args.val_pairs):  # an equal share of pairs per val avatar
        frames = avatars[val_ids[i % len(val_ids)]]
        v_lat.append(frames[torch.randint(len(frames), (1,), generator=vgen)])
        v_aud.append(auds[torch.randint(len(auds), (1,), generator=vgen).to(dev)])
    v_lat, v_aud = torch.cat(v_lat).to(dev), torch.cat(v_aud)
    g_lat, g_aud, g_ref = g_lat.to(dev), g_aud.to(dev), g_ref.to(dev)
    t_taps.on = False
    v_ref = predict(teacher, v_lat, v_aud, t_dtype)
    teacher_gate = stats(predict(teacher, g_lat, g_aud, t_dtype), g_ref, "gate")
    t_taps.on = True
    print(f"teacher ({t_dtype}) on the gate: mae {teacher_gate['gate_mae']:.5f}, max_abs {teacher_gate['gate_max_abs']:.4f}",
          flush=True)

    def evaluate() -> dict:
        student.eval()
        s_taps.on = False
        r = {**stats(predict(student, v_lat, v_aud, torch.float32), v_ref, "val"),
             **stats(predict(student, g_lat, g_aud, torch.float32), g_ref, "gate")}
        s_taps.on = True
        student.train()
        return r

    budget = args.max_minutes * 60
    step, elapsed0, best = 0, 0.0, math.inf
    t_start = time.time()

    def clock() -> float:
        return elapsed0 + time.time() - t_start

    def lr_scale(s: int) -> float:
        warm = min(1.0, (s + 1) / args.warmup) if args.warmup else 1.0
        p = min(1.0, max(s / args.steps, clock() / budget if budget else 0.0))
        return warm * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    train_params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0, **({"fused": True} if cuda else {}))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    gen = torch.Generator().manual_seed(args.seed)
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ck["variant"] != args.variant or ck["val_avatars"] != val_ids:
            raise SystemExit(f"{ckpt_path} is {ck['variant']} with val {ck['val_avatars']}, not {args.variant} with {val_ids}")
        unet.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        if cuda and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        step, elapsed0, best = ck["step"], ck["train_seconds"], ck["best_val"]
        t_start = time.time()
        print(f"resumed from {ckpt_path} at step {step}, {elapsed0 / 60:.1f} min trained, best val mae {best:.5f}", flush=True)
    elif args.resume:
        print(f"no {ckpt_path}: starting at step 0", flush=True)

    def log(rec: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def weights() -> dict:
        return {k: v.detach().cpu() for k, v in unet.state_dict().items()}

    def checkpoint() -> None:
        save({"variant": args.variant, "val_avatars": val_ids, "step": step, "train_seconds": clock(), "best_val": best,
              "student": weights(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "gen": gen.get_state(), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if cuda else None}, ckpt_path)

    log({"event": "start", "step": step, "variant": args.variant, "params_m": round(params / 1e6, 1), "device": str(dev),
         "train_avatars": train_ids, "val_avatars": val_ids, "train_frames": len(train_lat), "val_pairs": len(v_lat),
         "audio_clips": [p.name for p in audio_files], "audio_windows": len(auds), "gate_frames": len(g_lat),
         "gate_overlap": overlap, "teacher_dtype": str(t_dtype), **{f"teacher_{k}": v for k, v in teacher_gate.items()},
         "args": {k: str(v) for k, v in vars(args).items()}})

    def report(rec: dict) -> None:
        print(f"eval {rec['step']}: val mae {rec['val_mae']:.5f} (blend {rec['val_blend_mae']:.5f}) | gate mae "
              f"{rec['gate_mae']:.5f} worst {rec['gate_mae_worst']:.5f} blend {rec['gate_blend_mae']:.5f} max_abs "
              f"{rec['gate_max_abs']:.4f}{' | best' if rec['best'] else ''}", flush=True)

    if step == 0:
        rec = {"step": 0, "loss": None, **evaluate(), "elapsed_min": 0.0}
        rec["best"] = rec["val_mae"] < best
        best = min(best, rec["val_mae"])
        save(weights(), args.out / f"{args.variant}_best.pt")
        log(rec)
        report(rec)

    run = torch.zeros(3, device=dev)  # loss, out, feat summed since the last eval
    run_n, paired, log_t, log_step, last_ckpt = 0, False, time.time(), step, time.time()
    while step < args.steps and not (budget and clock() >= budget):
        step += 1
        x = train_lat[torch.randint(len(train_lat), (args.batch,), generator=gen).to(dev)]
        a = auds[torch.randint(len(auds), (args.batch,), generator=gen).to(dev)]
        with torch.no_grad():
            t_out = teacher(x.to(t_dtype), a.to(t_dtype)).float()
            t_feats = t_taps.take()
        with torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext():
            s_out = student(x.float(), a.float())
        s_feats = s_taps.take()
        if not paired:
            pairs = [(tuple(f.shape), tuple(t_feats[i + shift].shape)) for i, f in enumerate(s_feats)]
            if any(sp != tp for sp, tp in pairs):
                raise SystemExit(f"up-block features do not pair (student, teacher): {pairs}")
            paired = True
        loss_out = F.mse_loss(s_out.float(), t_out)
        loss_feat = sum(F.mse_loss(f.float(), t_feats[i + shift].float()) for i, f in enumerate(s_feats)) / len(s_feats)
        loss = loss_out + args.feat_weight * loss_feat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        run += torch.stack([loss.detach(), loss_out.detach(), loss_feat.detach()])
        run_n += 1
        stopping = step >= args.steps or bool(budget and clock() >= budget)
        evaluating = step % args.eval_every == 0 or stopping
        if evaluating or step % args.log_every == 0:
            m = (run / run_n).tolist()
            if not all(math.isfinite(v) for v in m):
                raise SystemExit(f"step {step}: loss is not finite ({m}); ckpt.pt is the last good state")
            sps = (time.time() - log_t) / (step - log_step)
            print(f"step {step}: loss {m[0]:.5f} (out {m[1]:.5f}, feat {m[2]:.5f}), lr {sched.get_last_lr()[0]:.2e}, "
                  f"{sps:.2f} s/step, {clock() / 60:.1f} min", flush=True)
            if evaluating:
                rec = {"step": step, "loss": m[0], "loss_out": m[1], "loss_feat": m[2], "lr": sched.get_last_lr()[0],
                       **evaluate(), "s_per_step": sps, "elapsed_min": clock() / 60}
                rec["best"] = rec["val_mae"] < best
                if rec["best"]:
                    best = rec["val_mae"]
                    save(weights(), args.out / f"{args.variant}_best.pt")
                log(rec)
                report(rec)
                run.zero_()
                run_n = 0
            log_t, log_step = time.time(), step
        if stopping:
            break
        if time.time() - last_ckpt >= args.ckpt_minutes * 60:
            if not torch.isfinite(run).all():
                raise SystemExit(f"step {step}: loss is not finite; ckpt.pt is the last good state")
            checkpoint()
            last_ckpt = time.time()
    checkpoint()
    save(weights(), args.out / f"{args.variant}_last.pt")
    print(f"step {step}, {clock() / 60:.1f} min trained, best val mae {best:.5f}: {args.out / (args.variant + '_best.pt')}",
          flush=True)


if __name__ == "__main__":
    main()
