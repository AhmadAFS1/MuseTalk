"""CPU-only Kokoro-82M timing at several torch thread counts (no GPU, offline)."""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
import torch
torch.set_num_threads(threads)
t0 = time.perf_counter()
from kokoro import KPipeline
pipe = KPipeline(lang_code="a", device="cpu", repo_id="hexgrad/Kokoro-82M")
load_s = time.perf_counter() - t0
TEXT = ("That sounds lovely tell me the best part of your day and then I have a "
        "little story for you if you'd like to hear it.")
def synth():
    n = 0
    for r in pipe(TEXT, voice="af_heart", speed=1.0):
        a = r.audio if getattr(r, "audio", None) is not None else r[-1]
        n += int(a.numel() if hasattr(a, "numel") else len(a))
    return n / 24000.0
t0 = time.perf_counter(); audio_s = synth(); first_s = time.perf_counter() - t0
ts = []
for _ in range(3):
    t0 = time.perf_counter(); audio_s = synth(); ts.append(time.perf_counter() - t0)
print(json.dumps({"threads": threads, "load_s": load_s, "first_call_s": first_s,
                  "audio_s": audio_s, "warm_s": min(ts), "warm_rtf": min(ts) / audio_s,
                  "thread_seconds_per_audio_second": threads * min(ts) / audio_s}))
