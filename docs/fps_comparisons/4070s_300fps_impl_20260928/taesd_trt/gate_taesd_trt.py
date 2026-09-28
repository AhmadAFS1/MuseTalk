"""G-TAESD gate + exactness checks for the TensorRT TAESD backend (plan item 2.1).

Loads, in one process with the live torch flags (tf32, cudnn.benchmark, default
TorchInductor cache so compiled TAESD picks the live server's kernels):
  - today's compiled TAESD   (scripts/vae_fast_decoder.py TaesdVaeDecodeBackend, max-autotune)
  - eager TAESD               (the same module's _raw_decode, uncompiled)
  - the persisted TRT TAESD   (load_taesd_trt_backend: builds on first use, verifies the probe)
and decodes every stored post-UNet latent of the multi-avatar corpus (main + holdout).

Reports (uint8 LSB after the repo fast postprocess, full 256x256 face and rows >= 104):
  G-TAESD      TRT vs compiled: max <= 3 LSB, mean <= 0.2 LSB
  noise floor  compiled vs eager, and TRT vs eager
  bit-exact    fused TRT uint8 post vs repo post applied to the TRT engine's fp16 output
  batching     partial batches (padding), bs16 -> 2 x bs8, batch-composition invariance, reruns
  vae.py       VAE.decode_latents with the fused path == non-fused path == reference arrays;
               with the compiled backend attached it equals the compiled reference (default path)
Writes gate_taesd_trt.json (+ worst-frame PNG strip) next to this file and records the gate
verdict in the engine's meta JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import vae_fast_decoder as vfd  # noqa: E402

DEV = torch.device("cuda:0")
ROW0 = 104
G_MAX, G_MEAN = 3, 0.2


class LsbStats:
    """Streaming |a-b| statistics over uint8 NHWC frames."""

    def __init__(self):
        self.hist = np.zeros(256, dtype=np.int64)
        self.hist_rows = np.zeros(256, dtype=np.int64)
        self.frame_max = []
        self.frame_mean = []
        self.frame_rows_max = []
        self.frame_ids = []

    def add(self, a: torch.Tensor, b: torch.Tensor, ids):
        d = (a.to(torch.int16) - b.to(torch.int16)).abs()
        self.hist += torch.bincount(d.flatten().to(torch.int64), minlength=256).cpu().numpy()[:256]
        rows = d[:, ROW0:]
        self.hist_rows += torch.bincount(rows.flatten().to(torch.int64), minlength=256).cpu().numpy()[:256]
        self.frame_max += d.flatten(1).amax(1).tolist()
        self.frame_mean += d.flatten(1).float().mean(1).tolist()
        self.frame_rows_max += rows.flatten(1).amax(1).tolist()
        self.frame_ids += ids

    @staticmethod
    def _summ(h):
        n = h.sum()
        vals = np.arange(256)
        nz = np.nonzero(h)[0]
        return {"max": int(nz.max()) if nz.size else 0, "mean": float((h * vals).sum() / n),
                "frac_nonzero": float(h[1:].sum() / n),
                "hist": {int(v): int(h[v]) for v in nz}}

    def summary(self):
        worst = int(np.argmax(self.frame_mean))
        return {"frames": len(self.frame_ids), "full": self._summ(self.hist), "rows104": self._summ(self.hist_rows),
                "worst_frame_mean": {"id": self.frame_ids[worst], "mean": self.frame_mean[worst],
                                     "max": self.frame_max[worst]},
                "frame_mean_p99": float(np.percentile(self.frame_mean, 99)),
                "frames_with_max_ge3": int(sum(1 for m in self.frame_max if m >= 3))}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "calibration/unet_multi_avatar_20260928"))
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--no-record", action="store_true", help="do not write the verdict into the engine meta")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    t_start = time.time()
    res = {"schema": "gate_taesd_trt_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MUSETALK_", "TORCHINDUCTOR", "TRITON"))},
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    compiled = vfd.TaesdVaeDecodeBackend.load(device=DEV, runtime_dtype=torch.float16)
    compiled.warmup([8])
    eager = compiled._raw_decode
    trt = vfd.load_taesd_trt_backend(DEV, torch.float16, model=compiled.model)
    trt.warmup([8])
    res["engine"] = {k: trt.meta.get(k) for k in ("key", "fingerprint", "decoder_plan", "decoder_plan_sha256",
                                                  "decoder_plan_bytes", "post_plan", "post_plan_sha256",
                                                  "post_plan_bytes", "probe", "build")}
    res["probe_at_load"] = trt.probe_hashes()

    files = []
    for split, d in (("main", Path(args.corpus)), ("holdout", Path(args.corpus) / "holdout")):
        fs = sorted(d.glob("unet_io_*.pt"))
        if args.limit_files:
            fs = fs[: args.limit_files]
        files += [(split, f) for f in fs]

    stats = {"trt_vs_compiled": {}, "compiled_vs_eager": {}, "trt_vs_eager": {}}
    for key in stats:
        for split in ("main", "holdout", "all"):
            stats[key][split] = LsbStats()
    per_avatar = {}
    fp16_max_abs = {"trt_vs_compiled": 0.0, "compiled_vs_eager": 0.0}
    fused_mismatch_bytes = 0
    fused_frames = 0
    rerun_mismatch = 0
    psnr_min = {"trt_vs_compiled": float("inf"), "compiled_vs_eager": float("inf")}
    worst = {"score": -1}
    all_latents = []
    with torch.inference_mode():
        for split, f in files:
            payload = torch.load(f, map_location="cpu", weights_only=False)
            z = payload["pred_latents"].to(DEV, torch.float16).contiguous()
            items = payload.get("items") or [{}]
            avatar = items[0].get("avatar_id", "?")
            ids = [f"{f.name}#{j}" for j in range(z.shape[0])]
            if len(all_latents) < 64:
                all_latents.append(z)
            c16 = compiled.decode(z, 1.0, torch.float16).clone()
            e16 = eager(z)
            t16 = trt.decode(z, 1.0)
            fused = trt.decode_bgr_u8(z)
            fused_again = trt.decode_bgr_u8(z)
            c8, e8, t8 = (vfd.repo_fast_postprocess_gpu(x) for x in (c16, e16, t16))
            fused_mismatch_bytes += int((fused != t8).sum())
            rerun_mismatch += int((fused_again != fused).sum())
            fused_frames += z.shape[0]
            fp16_max_abs["trt_vs_compiled"] = max(fp16_max_abs["trt_vs_compiled"],
                                                  float((t16.float() - c16.float()).abs().max()))
            fp16_max_abs["compiled_vs_eager"] = max(fp16_max_abs["compiled_vs_eager"],
                                                    float((c16.float() - e16.float()).abs().max()))
            psnr_min["trt_vs_compiled"] = min(psnr_min["trt_vs_compiled"], psnr(t8, c8))
            psnr_min["compiled_vs_eager"] = min(psnr_min["compiled_vs_eager"], psnr(c8, e8))
            for key, (a, b) in (("trt_vs_compiled", (t8, c8)), ("compiled_vs_eager", (c8, e8)),
                                ("trt_vs_eager", (t8, e8))):
                stats[key][split].add(a, b, ids)
                stats[key]["all"].add(a, b, ids)
            pa = per_avatar.setdefault(avatar, {"split": split, "trt_vs_compiled": LsbStats()})
            pa["trt_vs_compiled"].add(t8, c8, ids)
            d = (t8.to(torch.int16) - c8.to(torch.int16)).abs().flatten(1).float().mean(1)
            j = int(d.argmax())
            if float(d[j]) > worst["score"]:
                worst = {"score": float(d[j]), "id": ids[j], "avatar": avatar,
                         "compiled": c8[j].cpu().numpy(), "trt": t8[j].cpu().numpy()}

        # ---- batching / padding / composition checks on 64 real batches' worth of latents
        Z = torch.cat(all_latents)[:512].contiguous()
        full = torch.cat([trt.decode_bgr_u8(Z[i:i + 8]) for i in range(0, Z.shape[0], 8)])
        full16 = torch.cat([trt.decode(Z[i:i + 8], 1.0) for i in range(0, Z.shape[0], 8)])
        batching = {}
        for n in (1, 3, 5, 7, 9, 13, 16, 17, 24, 64):
            u8 = trt.decode_bgr_u8(Z[:n])
            f16 = trt.decode(Z[:n], 1.0)
            batching[f"n{n}"] = {"u8_equal": bool(torch.equal(u8, full[:n])),
                                 "fp16_equal": bool(torch.equal(f16, full16[:n]))}
        # composition: each latent at every position of a batch built from other latents
        perm = torch.randperm(Z.shape[0], generator=torch.Generator().manual_seed(7)).to(DEV)
        shuffled = torch.cat([trt.decode_bgr_u8(Z[perm[i:i + 8]]) for i in range(0, Z.shape[0], 8)])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=DEV)
        batching["shuffled_composition_equal"] = bool(torch.equal(shuffled[inv], full))
        # a single latent replicated at each of the 8 slots, with different neighbours
        slot_ok = True
        for slot in range(8):
            batch = Z[8:16].clone()
            batch[slot] = Z[0]
            slot_ok &= bool(torch.equal(trt.decode_bgr_u8(batch)[slot], full[0]))
        batching["every_slot_position_equal"] = slot_ok
        # fp32 input (callers may pass float): cast to fp16 first, same as the compiled path
        batching["fp32_input_equal"] = bool(torch.equal(trt.decode_bgr_u8(Z[:8].float()), full[:8]))
        # compiled TAESD composition invariance (noise-floor context)
        c_full = torch.cat([vfd.repo_fast_postprocess_gpu(compiled.decode(Z[i:i + 8], 1.0, torch.float16))
                            for i in range(0, Z.shape[0], 8)])
        c_shuf = torch.cat([vfd.repo_fast_postprocess_gpu(compiled.decode(Z[perm[i:i + 8]], 1.0, torch.float16))
                            for i in range(0, Z.shape[0], 8)])
        batching["compiled_shuffled_composition_equal"] = bool(torch.equal(c_shuf[inv], c_full))

        # ---- vae.py dispatch: VAE.decode_latents with the TRT backend (fused and not) and compiled
        from musetalk.models.vae import VAE

        vae = VAE(model_path="./models/sd-vae")
        vae.vae = vae.vae.half().to(DEV).eval()
        vae.runtime_dtype = vae.vae.dtype
        zb = Z[:8].contiguous()
        vae.set_decode_backend(trt)
        trt.fused_post_enabled = True
        a_fused = vae.decode_latents(zb)
        trt.fused_post_enabled = False
        a_plain = vae.decode_latents(zb)
        trt.fused_post_enabled = True
        vae.set_decode_backend(compiled)
        a_compiled = vae.decode_latents(zb)
        vae_checks = {
            "backend_name_trt": trt.name,
            "fused_equals_nonfused": bool(np.array_equal(a_fused, a_plain)),
            "fused_equals_reference": bool(np.array_equal(a_fused, full[:8].cpu().numpy())),
            "fused_dtype_shape": [str(a_fused.dtype), list(a_fused.shape), bool(a_fused.flags["C_CONTIGUOUS"])],
            "nonfused_dtype_shape": [str(a_plain.dtype), list(a_plain.shape), bool(a_plain.flags["C_CONTIGUOUS"])],
            "compiled_backend_equals_compiled_reference": bool(np.array_equal(a_compiled, c_full[:8].cpu().numpy())),
        }
        del vae

    s_all = stats["trt_vs_compiled"]["all"].summary()
    gate = {
        "G_TAESD_full_max": s_all["full"]["max"], "G_TAESD_full_mean": s_all["full"]["mean"],
        "G_TAESD_rows104_max": s_all["rows104"]["max"], "G_TAESD_rows104_mean": s_all["rows104"]["mean"],
        "thresholds": {"max_lsb": G_MAX, "mean_lsb": G_MEAN},
        "fused_post_mismatched_bytes": fused_mismatch_bytes, "fused_frames": fused_frames,
        "rerun_mismatched_bytes": rerun_mismatch,
    }
    passed_taesd = (s_all["full"]["max"] <= G_MAX and s_all["full"]["mean"] <= G_MEAN
                    and s_all["rows104"]["max"] <= G_MAX and s_all["rows104"]["mean"] <= G_MEAN)
    passed_exact = (fused_mismatch_bytes == 0 and rerun_mismatch == 0
                    and all(v["u8_equal"] and v["fp16_equal"] for k, v in batching.items() if k.startswith("n"))
                    and batching["shuffled_composition_equal"] and batching["every_slot_position_equal"]
                    and vae_checks["fused_equals_nonfused"] and vae_checks["fused_equals_reference"]
                    and vae_checks["compiled_backend_equals_compiled_reference"])
    gate["G_TAESD"] = "PASS" if passed_taesd else "FAIL"
    gate["bit_exact_checks"] = "PASS" if passed_exact else "FAIL"
    gate["verdict"] = "PASS" if (passed_taesd and passed_exact) else "FAIL"
    res.update({
        "files": len(files), "frames": fused_frames,
        "gate": gate,
        "lsb": {k: {s: v.summary() for s, v in d.items()} for k, d in stats.items()},
        "per_avatar_trt_vs_compiled": {a: {"split": v["split"], **{kk: vv for kk, vv in
                                                                   v["trt_vs_compiled"].summary().items()
                                                                   if kk in ("frames", "full", "rows104",
                                                                             "frames_with_max_ge3")}}
                                       for a, v in sorted(per_avatar.items())},
        "fp16_max_abs": fp16_max_abs, "psnr_min_db": psnr_min,
        "batching": batching, "vae_decode_latents": vae_checks,
        "seconds": time.time() - t_start,
        "tags": "[M] measured by this run",
    })
    # drop verbose per-avatar histograms
    for a in res["per_avatar_trt_vs_compiled"].values():
        for k in ("full", "rows104"):
            a[k] = {kk: vv for kk, vv in a[k].items() if kk != "hist"}
    (OUT / "gate_taesd_trt.json").write_text(json.dumps(res, indent=1))
    try:
        import cv2

        c, t = worst["compiled"], worst["trt"]
        diff = np.clip(np.abs(c.astype(np.int16) - t.astype(np.int16)) * 40, 0, 255).astype(np.uint8)
        strip = np.concatenate([c, t, diff], axis=1)
        strip = cv2.resize(strip, (strip.shape[1] * 2, strip.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        for x, label in ((10, "compiled TAESD"), (522, "TRT TAESD"), (1034, "|diff| x40")):
            cv2.putText(strip, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / "gate_worst_frame.png"), strip)
        res["worst_frame_png"] = {"file": "gate_worst_frame.png", "id": worst["id"], "avatar": worst["avatar"],
                                  "mean_lsb": worst["score"]}
        (OUT / "gate_taesd_trt.json").write_text(json.dumps(res, indent=1))
    except Exception as exc:  # noqa: BLE001
        print("png failed", exc)
    if not args.no_record:
        meta_path = trt.paths["meta"]
        meta = json.loads(meta_path.read_text())
        meta["gate"] = {"verdict": gate["verdict"], "report": str((OUT / "gate_taesd_trt.json").relative_to(ROOT)),
                        "created_utc": res["created_utc"], "frames": fused_frames,
                        **{k: gate[k] for k in ("G_TAESD_full_max", "G_TAESD_full_mean", "G_TAESD_rows104_max",
                                                "G_TAESD_rows104_mean", "fused_post_mismatched_bytes")}}
        vfd._atomic_write(meta_path, json.dumps(meta, indent=1).encode())
    print(json.dumps({"gate": gate, "noise_floor_compiled_vs_eager": {
        k: res["lsb"]["compiled_vs_eager"]["all"][k] for k in ("full", "rows104")},
        "trt_vs_eager": {k: {kk: vv for kk, vv in res["lsb"]["trt_vs_eager"]["all"][k].items() if kk != "hist"}
                         for k in ("full", "rows104")},
        "fp16_max_abs": fp16_max_abs, "psnr_min_db": psnr_min, "batching": batching,
        "vae": vae_checks}, indent=1, default=str))
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
