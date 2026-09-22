"""Validate the TAESD backend through the real load_vae_trt_decoder path."""
import os, sys, time, json
from pathlib import Path
ROOT = Path("/workspace/MuseTalk"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np, torch

device = torch.device("cuda"); BS, USED_Y0 = 8, 104
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

vae, unet, pe = load_all_model(device=device)
unet.model = unet.model.half().to(device).eval()
vae.vae = vae.vae.half().to(device).eval(); vae.runtime_dtype = torch.float16
sf, m = vae.scaling_factor, vae.vae

# real post-UNet latents
apx = AudioProcessor(feature_extractor_path="./models/whisper")
wh = WhisperModel.from_pretrained("./models/whisper").to(device=device, dtype=torch.float16).eval()
feats, ll = apx.get_audio_feature("data/audio/yongen.wav", weight_dtype=torch.float16)
chunks = apx.get_whisper_chunk(feats, device, torch.float16, wh, ll, fps=25)
alat = torch.load(ROOT/"results/v15/avatars/codex_smoke/latents.pt", map_location="cpu").squeeze(1)
START = 40
aud = pe(torch.stack([chunks[START+i] for i in range(BS)]).to(device, torch.float16)).to(torch.float16)
with torch.inference_mode():
    z = unet.model(alat[START:START+BS].to(device, torch.float16), torch.tensor([0], device=device),
                   encoder_hidden_states=aud).sample.to(torch.float16)
del wh; torch.cuda.empty_cache()

with torch.inference_mode():
    ref = ((m.decode((1/sf)*z).sample)/2 + 0.5).clamp(0,1).float()
print(f"reference SD-VAE  shape={tuple(ref.shape)} range=[{ref.min():.3f},{ref.max():.3f}]")

# ---- load through the REAL integration point
os.environ["MUSETALK_VAE_BACKEND"] = "taesd"
os.environ["MUSETALK_TAESD_COMPILE"] = os.getenv("MUSETALK_TAESD_COMPILE", "1")
os.environ["MUSETALK_TAESD_WARMUP_BATCHES"] = "8"
import logging; logging.basicConfig(level=logging.INFO)
from trt_runtime import load_vae_trt_decoder
be = load_vae_trt_decoder(device=device, scaling_factor=sf, vae_module=m)
assert be is not None, "backend did not load"
print(f"backend loaded: name={be.name}")

vae.set_decode_backend(be)
assert vae.has_decode_backend() and vae.get_decode_backend_name() == "taesd"

with torch.inference_mode():
    out = vae.decode_latents_tensor(z)
print(f"backend output    shape={tuple(out.shape)} dtype={out.dtype} range=[{out.min():.3f},{out.max():.3f}]")
assert out.shape == ref.shape, f"shape mismatch {out.shape} vs {ref.shape}"
assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0, "output not in [0,1]"

d  = (out.float()-ref).abs()
du = (out.float()[:,:,USED_Y0:,:]-ref[:,:,USED_Y0:,:]).abs()
print(f"\nvs SD-VAE FP16 reference:")
print(f"  full frame : mae={d.mean():.5f}  max_abs={d.max():.4f}")
print(f"  blended ROI: mae={du.mean():.5f}  max_abs={du.max():.4f}")

# full decode_latents() path (numpy uint8 BGR NHWC)
with torch.inference_mode():
    arr = vae.decode_latents(z)
print(f"\ndecode_latents() -> {type(arr).__name__} {arr.shape} {arr.dtype} "
      f"range=[{arr.min()},{arr.max()}]")
assert arr.shape == (BS,256,256,3) and arr.dtype == np.uint8

def bench(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    return float(np.mean(ts))*1000

with torch.inference_mode():
    t_taesd = bench(lambda: vae.decode_latents_tensor(z))
    vae.clear_decode_backend()
    t_sdvae = bench(lambda: vae.decode_latents_tensor(z), iters=10, warmup=4)
print(f"\nspeed @bs8: SD-VAE {t_sdvae:.2f} ms -> TAESD {t_taesd:.2f} ms  ({t_sdvae/t_taesd:.1f}x)")

json.dump({"mae_full":float(d.mean()),"max_abs_full":float(d.max()),
           "mae_roi":float(du.mean()),"max_abs_roi":float(du.max()),
           "sdvae_ms":t_sdvae,"taesd_ms":t_taesd,"speedup":t_sdvae/t_taesd},
          open("/tmp/claude-0/-workspace/e3b6c3cf-c47c-4de6-9013-fbab0c27bedf/scratchpad/taesd_backend_validation.json","w"), indent=2)
print("\nALL CONTRACT ASSERTIONS PASSED")
