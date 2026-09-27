import av, fractions, time, json, numpy as np, threading, os
from concurrent.futures import ThreadPoolExecutor
IDLE = "/workspace/experiments/chinese_bob_webrtc_20260927/h3_batch/chinese_bob_pink_bedroom_fd26f914/idle/source.mp4"
c = av.open(IDLE); it = c.decode(c.streams.video[0]); frames = [next(it).reformat(format="yuv420p") for _ in range(48)]; c.close()
W, H = frames[0].width, frames[0].height
def mk(name="h264_nvenc", opts=None):
    cc = av.CodecContext.create(name, "w")
    cc.width, cc.height, cc.pix_fmt = W, H, "yuv420p"
    cc.bit_rate = 2_000_000
    cc.framerate = fractions.Fraction(20, 1); cc.time_base = fractions.Fraction(1, 20)
    cc.options = opts or {"preset": "p1", "tune": "ull", "bf": "0", "rc": "cbr", "delay": "0", "zerolatency": "1", "g": "40"}
    cc.open()
    return cc
res = {}
# session limit
encs = []
for k in range(1, 21):
    try:
        encs.append(mk())
        # push one frame to be sure session is live
        f = frames[0]; f.pts = 0; f.time_base = fractions.Fraction(1, 20); list(encs[-1].encode(f))
    except Exception as e:
        res["nvenc_session_open_failed_at"] = k; res["nvenc_error"] = str(e)[:200]; break
res["nvenc_sessions_opened"] = len(encs)
print("opened", len(encs), res.get("nvenc_error"), flush=True)
# per-frame latency single session
cc = encs[0]; n = 200
t0 = time.perf_counter(); c0 = time.process_time(); outb = 0
for i in range(n):
    f = frames[i % 48]; f.pts = i + 1; f.time_base = fractions.Fraction(1, 20)
    for p in cc.encode(f): outb += len(bytes(p))
res["nvenc_1sess_wall_ms_per_frame"] = (time.perf_counter() - t0) * 1000 / n
res["nvenc_1sess_cpu_ms_per_frame"] = (time.process_time() - c0) * 1000 / n
res["nvenc_bytes_per_frame"] = outb / n
# parallel throughput across all open sessions (<= 15)
use = encs[:min(len(encs), 15)]
def run(idx):
    e = use[idx]
    for i in range(100):
        f = frames[(i + idx) % 48]; f.pts = 1000 + i; f.time_base = fractions.Fraction(1, 20)
        for p in e.encode(f): bytes(p)
ex = ThreadPoolExecutor(len(use)); t0 = time.perf_counter(); c0 = time.process_time()
list(ex.map(run, range(len(use))))
dt = time.perf_counter() - t0
res["nvenc_parallel_sessions"] = len(use)
res["nvenc_parallel_aggregate_fps"] = len(use) * 100 / dt
res["nvenc_parallel_cpu_ms_per_frame"] = (time.process_time() - c0) * 1000 / (len(use) * 100)
print(json.dumps(res, indent=1))
json.dump(res, open("bench_nvenc.json", "w"), indent=1)
