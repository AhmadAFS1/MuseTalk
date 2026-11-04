"""Measured and rejected 2026-11-03 (ios_coreml_implementation.md 3.5): the UNet on the lower part of the latent only.

MuseTalk blends only the lower face (get_image_prepare_material: upper_boundary_ratio 0.5 of the expanded crop, feathered,
so from about latent row 13 down). Run the UNet on rows `top`..31 (top divisible by 8) and score the blended rows against
the server fp16 reference on the 128 harness gate frames, torch fp32 on the CPU.

Usage (repo root): .venv-ct/bin/python scripts/ios/check_crop_unet.py . 8,16
"""
import sys, time
from pathlib import Path
import numpy as np, torch
MT = Path(sys.argv[1]); sys.path.insert(0, str(MT))
from musetalk.models.unet import UNet
HB = MT / "ios/MuseTalkBench/HarnessBundle"; UD = MT / "models/musetalkV15"
lat = torch.from_numpy(np.fromfile(HB / "unet_latents_f16.bin", dtype=np.float16).reshape(128, 8, 32, 32).astype(np.float32))
aud = torch.from_numpy(np.fromfile(HB / "unet_audio_f16.bin", dtype=np.float16).reshape(128, 50, 384).astype(np.float32))
ref = torch.from_numpy(np.fromfile(HB / "unet_ref_f16.bin", dtype=np.float16).reshape(128, 4, 32, 32).astype(np.float32))
w = UNet(unet_config=str(UD / "musetalk.json"), model_path=str(UD / "unet.pth"), device=torch.device("cpu"))
m = w.model.eval().requires_grad_(False)
torch.set_num_threads(8)
def run(top):
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, 128, 8):
            outs.append(m(lat[i:i + 8, :, top:, :], torch.zeros(1, dtype=torch.long), encoder_hidden_states=w.pe(aud[i:i + 8])).sample)
    return torch.cat(outs), time.time() - t0
full, tf = run(0)
print(f"full 32 rows: {tf:.1f} s", flush=True)
for top in [int(a) for a in sys.argv[2].split(",")]:
    crop, tc = run(top)
    print(f"crop from row {top} ({32 - top} rows): {tc:.1f} s on the CPU (full {tf:.1f} s)")
    for r0 in [r for r in (12, 14, 16) if r >= top]:
        e_full = (full[:, :, r0:] - ref[:, :, r0:]).abs().flatten(1)
        e_crop = (crop[:, :, r0 - top:] - ref[:, :, r0:]).abs().flatten(1)
        d = (crop[:, :, r0 - top:] - full[:, :, r0:]).abs().flatten(1)
        print(f"  rows {r0}-31 vs server ref: full mae {e_full.mean(1).mean():.5f} (worst {e_full.mean(1).max():.5f}, max_abs {e_full.max():.3f}) | "
              f"crop mae {e_crop.mean(1).mean():.5f} (worst {e_crop.mean(1).max():.5f}, max_abs {e_crop.max():.3f}) | crop vs full mae {d.mean(1).mean():.5f}")
