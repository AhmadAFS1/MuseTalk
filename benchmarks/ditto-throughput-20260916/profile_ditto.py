"""Unpaced diagnostic; no changes to installed Ditto code or model settings."""
import sys, time, json, statistics, collections
from pathlib import Path
import numpy as np
import librosa
import torch
sys.path.insert(0, '/workspace/ditto-talkinghead')
import stream_pipeline_online as pipe
from core.utils.tensorrt_utils import TRTWrapper

stats = collections.defaultdict(list)
enabled = False
for method in ('setup', 'infer'):
    original = getattr(TRTWrapper, method)
    def wrapped(self, *a, _original=original, _method=method, **kw):
        t = time.perf_counter()
        result = _original(self, *a, **kw)
        if enabled:
            stats[Path(self.model).name + ':' + _method].append(time.perf_counter()-t)
        return result
    setattr(TRTWrapper, method, wrapped)

class Sink:
    def __init__(self, *a, **kw): self.times = []
    def __call__(self, *a, **kw): self.times.append(time.perf_counter())
    def close(self): pass
pipe.VideoWriterByImageIO = Sink
root = Path('/workspace/benchmarks/same-avatar')
mode = sys.argv[1]
sdk = pipe.StreamSDK('./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt' + ('_online' if mode=='online' else '') + '.pkl', './checkpoints/ditto_trt_Ampere_Plus')
audio, _ = librosa.load(root/'audio.wav', sr=16000)
rows = []
for rep in range(2):
    sdk.setup(str(root/'shared.png'), '/tmp/ditto-profile.mp4')
    sdk.setup_Nd(250)
    stats.clear()
    enabled = True
    t = time.perf_counter()
    if mode == 'online':
        padded = np.concatenate([np.zeros(1920, dtype=np.float32), audio])
        for i in range(0, len(padded), 3200):
            chunk = padded[i:i+6480]
            sdk.run_chunk(np.pad(chunk, (0, 6480-len(chunk))), (3,5,2))
    else:
        sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(audio))
    sdk.close()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-t
    enabled = False
    times = sdk.writer.times
    row = dict(rep=rep, frames=len(times), elapsed_s=elapsed, fps=len(times)/elapsed,
               first_frame_s=times[0]-t, output_span_fps=(len(times)-1)/(times[-1]-times[0]),
               stages={k:dict(calls=len(v), total_s=sum(v), mean_ms=1000*statistics.mean(v)) for k,v in stats.items()})
    rows.append(row)
isolated = {}
for label, model in [('warp', sdk.warp_f3d.warp_net.model), ('decoder', sdk.decode_f3d.decoder.model), ('motion_step', sdk.audio2motion.lmdm.model), ('hubert', sdk.wav2feat.w2f.hubert.model)]:
    durations = []
    for i in range(25):
        t = time.perf_counter()
        model.infer()
        torch.cuda.synchronize()
        if i >= 5: durations.append(time.perf_counter()-t)
    isolated[label] = dict(mean_ms=1000*statistics.mean(durations),
                          note='Preallocated buffers; synchronous transfers included, setup excluded.',
                          tensors={k:dict(shape=list(v[0].shape), bytes=v[2]) for k,v in model.buffer.items()})
out = root/'resource-usage'/('profile-'+mode+'.json')
out.write_text(json.dumps(dict(mode=mode, note='Unpaced, no encoding or transport. Concurrent host call durations overlap; do not sum as GPU time.', runs=rows, isolated=isolated), indent=2))
print(out)
print(json.dumps(rows[-1], indent=2))
