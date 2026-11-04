"""Measured and rejected 2026-11-03 (ios_coreml_implementation.md 3.5). Token downsampling (ToDo) in the MuseTalk UNet's self-attention: keys and values from a 2x2 average-pooled copy of
the hidden states, at the levels with `n` tokens. torch fp32 CPU, the 128 harness gate frames vs the server fp16 reference.

Usage (repo root): .venv-ct/bin/python scripts/ios/check_token_downsampling.py .
"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
MT = Path(sys.argv[1]); sys.path.insert(0, str(MT))
from musetalk.models.unet import UNet
HB = MT / "ios/MuseTalkBench/HarnessBundle"; UD = MT / "models/musetalkV15"
lat = torch.from_numpy(np.fromfile(HB / "unet_latents_f16.bin", dtype=np.float16).reshape(128, 8, 32, 32).astype(np.float32))
aud = torch.from_numpy(np.fromfile(HB / "unet_audio_f16.bin", dtype=np.float16).reshape(128, 50, 384).astype(np.float32))
ref = torch.from_numpy(np.fromfile(HB / "unet_ref_f16.bin", dtype=np.float16).reshape(128, 4, 32, 32).astype(np.float32))
w = UNet(unet_config=str(UD / "musetalk.json"), model_path=str(UD / "unet.pth"), device=torch.device("cpu"))
m = w.model.eval().requires_grad_(False); torch.set_num_threads(8)

class ToDo:
    def __init__(self, tokens, factor): self.tokens, self.factor = tokens, factor
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **k):
        residual = hidden_states
        b, n, c = hidden_states.shape
        kv = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is None and n in self.tokens:
            s = int(n ** 0.5)
            kv = F.avg_pool2d(hidden_states.transpose(1, 2).reshape(b, c, s, s), self.factor).flatten(2).transpose(1, 2)
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(kv)); val = attn.head_to_batch_dim(attn.to_v(kv))
        out = torch.bmm(attn.get_attention_scores(q, key, None), val)
        out = attn.to_out[1](attn.to_out[0](attn.batch_to_head_dim(out)))
        if attn.residual_connection: out = out + residual
        return out / attn.rescale_output_factor

def run():
    outs = []
    with torch.no_grad():
        for i in range(0, 128, 8):
            outs.append(m(lat[i:i + 8], torch.zeros(1, dtype=torch.long), encoder_hidden_states=w.pe(aud[i:i + 8])).sample)
    return torch.cat(outs)
def score(x, tag):
    e = (x - ref).abs(); per = e.flatten(1).mean(1); low = e[:, :, 13:].flatten(1).mean(1)
    print(f"{tag}: mae {per.mean():.5f} (worst {per.max():.5f}), blended rows 13-31 {low.mean():.5f} (worst {low.max():.5f}), max_abs {e.max():.3f}", flush=True)
attn1 = [mod for name, mod in m.named_modules() if name.endswith("attn1")]
print(len(attn1), "self-attention modules")
default = [a.processor for a in attn1]
score(run(), "full (fp32)")
for tokens, factor, tag in [({1024}, 2, "ToDo 2x2 at 32x32"), ({1024, 256}, 2, "ToDo 2x2 at 32x32 and 16x16")]:
    for a in attn1: a.set_processor(ToDo(tokens, factor))
    score(run(), tag)
for a, p in zip(attn1, default): a.set_processor(p)
