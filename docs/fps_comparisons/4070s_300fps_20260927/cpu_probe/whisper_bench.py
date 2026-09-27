#!/usr/bin/env python3
"""Whisper-tiny feature path for 8 s of audio, replicating musetalk AudioProcessor.

Stages: librosa.load(sr=16000) -> WhisperFeatureExtractor (log-mel, padded to 30 s)
-> WhisperModel.encoder(output_hidden_states=True) -> stack/trim/pad -> audio prompts.
"""
import json, os, sys, time, argparse
sys.path.insert(0, "/workspace/MuseTalk")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor

WAV = "/workspace/MuseTalk/data/audio/yongen.wav"
WD = "/workspace/MuseTalk/models/whisper"
dtype = torch.float16 if a.device == "cuda" else torch.float32
proc = AudioProcessor(feature_extractor_path=WD)
whisper = WhisperModel.from_pretrained(WD).to(device=a.device, dtype=dtype).eval()
whisper.requires_grad_(False)

import librosa
res = {"device": a.device, "torch_threads": a.threads, "dtype": str(dtype), "stages": []}
with torch.no_grad():
    for rep in range(a.reps + 1):
        t0 = time.perf_counter()
        y, sr = librosa.load(WAV, sr=16000)
        t1 = time.perf_counter()
        feats, n = proc.get_audio_feature(WAV, weight_dtype=dtype)  # includes a 2nd librosa.load (as prod)
        t2 = time.perf_counter()
        chunks = proc.get_whisper_chunk(feats, a.device, dtype, whisper, n, fps=20)
        if a.device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        if rep == 0:
            continue  # warmup
        res["stages"].append({"librosa_load_ms": (t1 - t0) * 1e3, "get_audio_feature_ms(load+mel)": (t2 - t1) * 1e3,
                              "whisper_encode+prompts_ms": (t3 - t2) * 1e3, "prod_total_ms": (t3 - t1) * 1e3})
    res["prompts_shape"] = list(chunks.shape)
    if a.device == "cuda":
        res["max_mem_alloc_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
med = {k: round(sorted(s[k] for s in res["stages"])[len(res["stages"]) // 2], 1) for k in res["stages"][0]}
res["median"] = med
del res["stages"]
print(json.dumps(res))
