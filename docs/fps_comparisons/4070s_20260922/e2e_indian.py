"""End-to-end SD-VAE vs TAESD on the shared_indian avatar, via the real backend loader."""
import json, os, pickle, subprocess, sys, time
from collections import OrderedDict
from pathlib import Path
ROOT = Path("/workspace/MuseTalk"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import cv2, numpy as np, torch

device = torch.device("cuda"); BS, FPS = 8, 25
NFRAMES = 200
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("--avatar", default="shared_indian_20260915"); _A=_ap.parse_args()
AV = ROOT/"results/v15/avatars"/_A.avatar
OUT = Path("/tmp/claude-0/-workspace/e3b6c3cf-c47c-4de6-9013-fbab0c27bedf/scratchpad/e2e_multi/"+_A.avatar)
OUT.mkdir(exist_ok=True, parents=True)
AUDIO = "data/audio/yongen.wav"

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import prepare_image_blending_plan, get_image_blending_with_plan
from transformers import WhisperModel

vae, unet, pe = load_all_model(device=device)
unet.model = unet.model.half().to(device).eval()
vae.vae = vae.vae.half().to(device).eval(); vae.runtime_dtype = torch.float16
sf, m = vae.scaling_factor, vae.vae
ts_ = torch.tensor([0], device=device)

apx = AudioProcessor(feature_extractor_path="./models/whisper")
wh = WhisperModel.from_pretrained("./models/whisper").to(device=device, dtype=torch.float16).eval()
feats, ll = apx.get_audio_feature(AUDIO, weight_dtype=torch.float16)
chunks = apx.get_whisper_chunk(feats, device, torch.float16, wh, ll, fps=FPS)
del wh; torch.cuda.empty_cache()

alat = torch.load(AV/"latents.pt", map_location="cpu").squeeze(1)
coords = pickle.load(open(AV/"coords.pkl","rb")); mcoords = pickle.load(open(AV/"mask_coords.pkl","rb"))
fulls = sorted((AV/"full_imgs").glob("*.png")); masks = sorted((AV/"mask").glob("*.png"))
ncyc = len(fulls); NF = min(NFRAMES, len(chunks))
print(f"avatar cycle={ncyc} latents={tuple(alat.shape)} frames={NF}", flush=True)

bg, plans, bbs = [], [], []
for i in range(ncyc):
    fr = cv2.imread(str(fulls[i])); mk = cv2.imread(str(masks[i % len(masks)]), cv2.IMREAD_GRAYSCALE)
    bg.append(fr); bbs.append(coords[i])
    plans.append(prepare_image_blending_plan(fr.shape, coords[i], mk, mcoords[i % len(mcoords)]))

# UNet TRT
from trt_runtime import _ensure_torch_tensorrt_registered, _load_serialized_trt_module, load_vae_trt_decoder
_ensure_torch_tensorrt_registered()
tmod = _load_serialized_trt_module(ROOT/"models/tensorrt_unet_sm89_bs8_local/unet_trt.ts", device)
unet_fn = lambda l, a: tmod(l, a)
print("UNet: TensorRT FP16 bs8", flush=True)

def dec_sdvae(z):
    return ((m.decode((1/sf)*z).sample)/2 + 0.5).clamp(0,1)

os.environ["MUSETALK_VAE_BACKEND"] = "taesd"
os.environ["MUSETALK_TAESD_WARMUP_BATCHES"] = "8"
taesd_be = load_vae_trt_decoder(device=device, scaling_factor=sf, vae_module=m)
assert taesd_be is not None and taesd_be.name == "taesd"
dec_taesd = lambda z: taesd_be.decode(latents=z, scaling_factor=sf, output_dtype=torch.float16)
print("VAE alt backend:", taesd_be.name, flush=True)

RES = OrderedDict()
for key, label, decfn in [("A_sdvae","SD-VAE FP16 (reference)",dec_sdvae),
                          ("B_taesd","TAESD",dec_taesd)]:
    frames=[]; t_unet=t_vae=t_post=t_comp=0.0
    torch.cuda.synchronize(); wall0=time.time()
    for s in range(0, NF, BS):
        n=min(BS, NF-s)
        aud = pe(torch.stack([chunks[s+i] for i in range(n)]).to(device, torch.float16)).to(torch.float16)
        li  = alat[[(s+i)%len(alat) for i in range(n)]].to(device, torch.float16)
        if n<BS:
            aud=torch.cat([aud,aud[:1].repeat(BS-n,1,1)]); li=torch.cat([li,li[:1].repeat(BS-n,1,1,1)])
        with torch.inference_mode():
            torch.cuda.synchronize(); a=time.perf_counter()
            z = unet_fn(li, aud).to(torch.float16)
            torch.cuda.synchronize(); b=time.perf_counter()
            img = decfn(z)
            torch.cuda.synchronize(); c=time.perf_counter()
            u8 = (img.detach().float().mul(255).round().clamp_(0,255).to(torch.uint8)
                  .flip(1).permute(0,2,3,1).contiguous().cpu().numpy())
            d=time.perf_counter()
        t_unet+=b-a; t_vae+=c-b; t_post+=d-c
        e=time.perf_counter()
        for i in range(n):
            ci=(s+i)%ncyc; x1,y1,x2,y2=bbs[ci]
            rr=cv2.resize(u8[i],(x2-x1,y2-y1))
            frames.append(get_image_blending_with_plan(bg[ci].copy(), rr, plans[ci]))
        t_comp+=time.perf_counter()-e
    wall=time.time()-wall0; nb=(NF+BS-1)//BS
    RES[key]={"label":label,"e2e_fps":round(len(frames)/wall,1),
              "unet_ms":round(t_unet/nb*1000,2),"vae_ms":round(t_vae/nb*1000,2),
              "post_ms":round(t_post/nb*1000,2),"compose_ms":round(t_comp/nb*1000,2),
              "gpu_ms":round((t_unet+t_vae+t_post)/nb*1000,2)}
    r=RES[key]; print(f"{label:28s} unet={r['unet_ms']}ms vae={r['vae_ms']}ms "
                      f"compose={r['compose_ms']}ms | e2e {r['e2e_fps']} fps", flush=True)
    raw=OUT/f"{key}_raw.mp4"
    vw=cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames: vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(raw),"-i",AUDIO,
                    "-c:v","libx264","-crf","16","-preset","medium","-pix_fmt","yuv420p",
                    "-c:a","aac","-shortest",str(OUT/f"{key}.mp4")], check=False)
    raw.unlink(missing_ok=True)

json.dump(RES, open(OUT/"results.json","w"), indent=2)
print(f"\nspeedup: {RES['B_taesd']['e2e_fps']/RES['A_sdvae']['e2e_fps']:.2f}x   videos in {OUT}")
