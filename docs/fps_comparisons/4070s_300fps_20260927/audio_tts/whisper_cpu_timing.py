"""CPU-only timing of MuseTalk's Whisper feature path (no GPU touched).

Measures: import cost, librosa.load (cold/warm, 24k->16k resample), HF
WhisperFeatureExtractor (30 s padded mel), whisper-tiny encoder on CPU at
several thread counts / batch sizes, and the feature error of a shorter
(non-30 s) encoder window vs the trained 30 s window.
"""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
t0 = time.perf_counter()
import numpy as np
import torch
t_torch = time.perf_counter() - t0
t0 = time.perf_counter()
import librosa
t_librosa = time.perf_counter() - t0
t0 = time.perf_counter()
from transformers import AutoFeatureExtractor, WhisperModel
t_tf = time.perf_counter() - t0
assert not torch.cuda.is_available()

MT = "/workspace/MuseTalk"
WAV24 = "/workspace/experiments/chinese_bob_webrtc_20260927/audio/turn_one.wav"  # 7.325 s Kokoro 24 kHz
res = {"imports_s": {"torch": t_torch, "librosa": t_librosa, "transformers": t_tf}}

# --- librosa.load cold / warm ---
t0 = time.perf_counter(); y, sr = librosa.load(WAV24, sr=16000); res["librosa_load_cold_s"] = time.perf_counter() - t0
ts = []
for _ in range(5):
    t0 = time.perf_counter(); y, sr = librosa.load(WAV24, sr=16000); ts.append(time.perf_counter() - t0)
res["librosa_load_warm_s"] = min(ts)
res["audio_s"] = len(y) / 16000

# --- HF feature extractor ---
t0 = time.perf_counter(); fe = AutoFeatureExtractor.from_pretrained(f"{MT}/models/whisper"); res["fe_load_s"] = time.perf_counter() - t0
t0 = time.perf_counter(); feats = fe([y], return_tensors="pt", sampling_rate=16000).input_features; res["fe_first_call_s"] = time.perf_counter() - t0
ts = []
for _ in range(5):
    t0 = time.perf_counter(); feats = fe([y], return_tensors="pt", sampling_rate=16000).input_features; ts.append(time.perf_counter() - t0)
res["fe_warm_s"] = min(ts)
res["mel_shape"] = list(feats.shape)

# --- whisper encoder on CPU ---
t0 = time.perf_counter(); wm = WhisperModel.from_pretrained(f"{MT}/models/whisper").eval(); res["whisper_load_s"] = time.perf_counter() - t0
enc = wm.encoder
res["encoder_params_M"] = sum(p.numel() for p in enc.parameters()) / 1e6
res["attn_impl"] = getattr(wm.config, "_attn_implementation", None)
enc_t = {}
with torch.inference_mode():
    for threads in (1, 4, 8, 16):
        torch.set_num_threads(threads)
        for bs in (1, 4):
            x = feats.repeat(bs, 1, 1)
            enc(x, output_hidden_states=True)  # warm
            ts = []
            for _ in range(3 if threads > 1 else 2):
                t0 = time.perf_counter(); hs = enc(x, output_hidden_states=True).hidden_states; ts.append(time.perf_counter() - t0)
            enc_t[f"t{threads}_bs{bs}"] = min(ts)
res["encoder_cpu_fp32_s"] = enc_t

# --- shorter encoder window vs trained 30 s window (feature error) ---
torch.set_num_threads(8)
def encode_window(mel, n_frames):
    """HF WhisperEncoder forward over the first n_frames mel frames (no 3000 check)."""
    m = mel[..., :n_frames]
    h = torch.nn.functional.gelu(enc.conv1(m))
    h = torch.nn.functional.gelu(enc.conv2(h)).permute(0, 2, 1)
    h = h + enc.embed_positions.weight[: h.shape[1]]
    states = [h]
    for layer in enc.layers:
        h = layer(h, None, layer_head_mask=None)[0]
        states.append(h)
    states[-1] = enc.layer_norm(h)
    return torch.stack(states, dim=2)  # [B, T, 5, 384]
with torch.inference_mode():
    full = torch.stack(enc(feats, output_hidden_states=True).hidden_states, dim=2)
    ref = encode_window(feats, 3000)
    res["reimpl_vs_hf_maxabs"] = float((ref - full).abs().max())
    valid = int(res["audio_s"] * 50)
    win = {}
    for secs in (8, 10, 15, 20):
        nf = secs * 100
        t0 = time.perf_counter(); short = encode_window(feats, nf); dt = time.perf_counter() - t0
        k = min(valid, short.shape[1])
        d = (short[:, :k] - full[:, :k])
        rel = float(d.norm() / full[:, :k].norm())
        per_layer = [float((d[:, :, i].norm() / full[:, :k, i].norm())) for i in range(5)]
        win[f"{secs}s"] = {"cpu_s_t8": dt, "rel_err": rel, "rel_err_per_layer": per_layer,
                           "cos_last": float(torch.nn.functional.cosine_similarity(short[:, :k, 4].flatten(), full[:, :k, 4].flatten(), dim=0))}
    res["short_window_vs_30s"] = win
    # incremental / streaming: encode first 2 s alone (padded to 30 s like a real 2 s turn) vs full-turn features
    y2 = y[: 2 * 16000]
    f2 = fe([y2], return_tensors="pt", sampling_rate=16000).input_features
    part = torch.stack(enc(f2, output_hidden_states=True).hidden_states, dim=2)
    k = 2 * 50 - 4  # ignore last 2 frames' boundary
    d = part[:, :k] - full[:, :k]
    res["prefix2s_vs_full_rel_err"] = float(d.norm() / full[:, :k].norm())
    res["prefix2s_vs_full_rel_err_last_layer"] = float(d[:, :, 4].norm() / full[:, :k, 4].norm())
print(json.dumps(res, indent=2))
