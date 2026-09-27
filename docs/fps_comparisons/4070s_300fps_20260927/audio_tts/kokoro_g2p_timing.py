"""CPU/GIL cost of Kokoro's G2P front-end alone (model=False), which stays on CPU even with GPU TTS."""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["HF_HUB_OFFLINE"] = "1"
import torch; torch.set_num_threads(1)
from kokoro import KPipeline
p = KPipeline(lang_code="a", model=False, repo_id="hexgrad/Kokoro-82M")
TEXT = ("That sounds lovely tell me the best part of your day and then I have a "
        "little story for you if you'd like to hear it.")
list(p(TEXT))
ts = []
for _ in range(5):
    t0 = time.process_time(); w0 = time.perf_counter(); r = list(p(TEXT)); ts.append((time.perf_counter() - w0, time.process_time() - t0))
print(json.dumps({"g2p_wall_s": min(t[0] for t in ts), "g2p_cpu_s": min(t[1] for t in ts), "chunks": len(r), "audio_s_equiv": 6.425}))
