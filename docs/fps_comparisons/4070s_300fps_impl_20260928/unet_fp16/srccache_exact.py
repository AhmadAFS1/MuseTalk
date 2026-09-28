"""Exactness gate for the source-prefix cache (variant "srccache").

forward_cached(precompute_prefix(latents)) must equal forward(latents) bit for bit on real corpus
batches, including a partial (padded) batch. Also reports (not gated) the difference against the
default FP16 stagewise engine set, i.e. the FP16-noise cost of moving the down0 engine boundary.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from pathlib import Path

ROOT = Path("/workspace/MuseTalk-perf300")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.chdir(ROOT)
import torch  # noqa: E402
import unet_stagewise_trt as sw  # noqa: E402

DEV = torch.device("cuda:0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline-root", default="models/tensorrt_unet_stagewise_sm89")
    ap.add_argument("--files", type=int, default=16)
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "calibration/unet_multi_avatar_20260928/unet_io_*.pt")))[::11][: a.files]
    lat = torch.cat([torch.load(f, map_location="cpu")["latent_batch"] for f in files]).half().to(DEV)
    aud = torch.cat([torch.load(f, map_location="cpu")["audio_feature_batch"] for f in files]).half().to(DEV)
    lat, aud = lat[:-3], aud[:-3]  # force a partial final chunk
    be = sw.StagewiseTrtUnetBackend.load(Path(a.root) / "bs16", device=DEV)
    assert be.variant == "srccache", be.variant
    with torch.inference_mode():
        out_fwd = be(lat, None, encoder_hidden_states=aud).sample.clone()
        h0, a0 = be.precompute_prefix(lat)
        out_cached = be.forward_cached(h0, a0, encoder_hidden_states=aud).sample.clone()
        # cached rows gathered out of order (as a multi-stream scheduler would)
        perm = torch.randperm(lat.shape[0], generator=torch.Generator().manual_seed(7)).to(DEV)
        out_perm = be.forward_cached(h0[perm], a0[perm], encoder_hidden_states=aud[perm]).sample.clone()
    exact = bool(torch.equal(out_fwd, out_cached))
    exact_perm = bool(torch.equal(out_fwd[perm], out_perm))
    del be
    torch.cuda.empty_cache()
    base = sw.StagewiseTrtUnetBackend.load(Path(a.baseline_root) / "bs16", device=DEV)
    with torch.inference_mode():
        out_base = base(lat, None, encoder_hidden_states=aud).sample
    d = (out_fwd.float() - out_base.float()).abs()
    res = dict(root=a.root, frames=int(lat.shape[0]), cached_equals_forward=exact, permuted_rows_exact=exact_perm,
               vs_default_fp16=dict(mae=float(d.mean()), max_abs=float(d.max())),
               prefix_cache_bytes_per_frame=int(h0[0].numel() * h0.element_size() + a0[0].numel() * a0.element_size()),
               verdict="PASS" if (exact and exact_perm) else "FAIL", tags="[M]")
    out = Path(__file__).resolve().parent / f"srccache_exact_{Path(a.root).name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(("PASS " if res["verdict"] == "PASS" else "FAIL ") + json.dumps(res), flush=True)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
