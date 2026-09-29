"""forward() vs forward_cached() sustained ms/frame for a srccache engine set (real corpus inputs, bs16)."""
import glob, json, os, sys, time
from pathlib import Path
ROOT = Path("/workspace/MuseTalk-perf300"); sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]; os.chdir(ROOT)
import torch
import unet_stagewise_trt as sw
DEV = torch.device("cuda:0"); root = Path(sys.argv[1]); secs = float(sys.argv[2]) if len(sys.argv) > 2 else 45
files = sorted(glob.glob(str(ROOT / "calibration/unet_multi_avatar_20260928/unet_io_*.pt")))[:16]
lat = torch.cat([torch.load(f, map_location="cpu")["latent_batch"] for f in files]).half().to(DEV)
aud = torch.cat([torch.load(f, map_location="cpu")["audio_feature_batch"] for f in files]).half().to(DEV)
be = sw.StagewiseTrtUnetBackend.load(root / "bs16", device=DEV)
with torch.inference_mode():
    h0, a0 = be.precompute_prefix(lat)
    res = {}
    for mode in ("forward", "forward_cached", "forward", "forward_cached"):
        n = 0; torch.cuda.synchronize(); t0 = time.perf_counter(); j = 0
        while time.perf_counter() - t0 < secs:
            s = (j * 16) % (lat.shape[0] - 15)
            if mode == "forward":
                be(lat[s:s + 16], None, encoder_hidden_states=aud[s:s + 16])
            else:
                be.forward_cached(h0[s:s + 16], a0[s:s + 16], encoder_hidden_states=aud[s:s + 16])
            n += 16; j += 1
            if j % 8 == 0: torch.cuda.synchronize()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        res.setdefault(mode, []).append(1000 * dt / n)
out = {k: min(v) for k, v in res.items()}; out["saving_pct"] = 100 * (1 - out["forward_cached"] / out["forward"]); out["root"] = str(root)
print(json.dumps(out)); (Path(__file__).parent / f"srccache_speed_{root.name}.json").write_text(json.dumps(out, indent=1))
