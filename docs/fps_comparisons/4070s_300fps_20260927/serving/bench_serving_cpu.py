"""Micro-benchmarks for MuseTalk live WebRTC serving-path CPU costs.

Read-only w.r.t. /workspace: reads one idle source mp4, writes only a small
JSON next to this script. Does not touch the running api_server.
"""
import fractions
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_serving_cpu.json")
IDLE = ("/workspace/experiments/chinese_bob_webrtc_20260927/h3_batch/"
        "chinese_bob_pink_bedroom_fd26f914/idle/source.mp4")
results = {"host_cpus": os.cpu_count()}


def cpu_now():
    return time.process_time()  # all threads of this process


def thread_count():
    return len(os.listdir("/proc/self/task"))


def timed(label, fn, n):
    w0, c0 = time.perf_counter(), cpu_now()
    for i in range(n):
        fn(i)
    w1, c1 = time.perf_counter(), cpu_now()
    r = {"n": n, "wall_ms_per": (w1 - w0) * 1000 / n, "cpu_ms_per": (c1 - c0) * 1000 / n}
    results[label] = r
    print(label, json.dumps(r), flush=True)
    return r


def parallel(label, make_worker, threads, n_per):
    workers = [make_worker(t) for t in range(threads)]
    barrier = threading.Barrier(threads + 1)

    def run(w):
        barrier.wait()
        for i in range(n_per):
            w(i)

    ex = ThreadPoolExecutor(max_workers=threads)
    futs = [ex.submit(run, w) for w in workers]
    time.sleep(0.05)
    w0, c0 = time.perf_counter(), cpu_now()
    barrier.wait()
    for f in futs:
        f.result()
    w1, c1 = time.perf_counter(), cpu_now()
    ex.shutdown()
    total = threads * n_per
    r = {"threads": threads, "total_frames": total,
         "aggregate_fps": total / (w1 - w0),
         "cpu_ms_per_frame": (c1 - c0) * 1000 / total,
         "cores_busy": (c1 - c0) / (w1 - w0)}
    results[label] = r
    print(label, json.dumps(r), flush=True)
    return r


# ---------------------------------------------------------------- 1. idle decode
def decode_frames(path, n, threads=0):
    c = av.open(path)
    s = c.streams.video[0]
    if threads:
        s.thread_type = "AUTO"
        s.codec_context.thread_count = threads
    it = c.decode(s)
    out = []
    w0, c0 = time.perf_counter(), cpu_now()
    for _ in range(n):
        out.append(next(it).reformat(format="yuv420p"))
    w1, c1 = time.perf_counter(), cpu_now()
    c.close()
    return out, (w1 - w0) * 1000 / n, (c1 - c0) * 1000 / n

yuv, w, c = decode_frames(IDLE, 96)
results["idle_decode_default_threads"] = {"wall_ms_per": w, "cpu_ms_per": c}
print("idle_decode_default_threads", w, c, flush=True)
_, w, c = decode_frames(IDLE, 96, threads=16)
results["idle_decode_16_threads"] = {"wall_ms_per": w, "cpu_ms_per": c}
print("idle_decode_16_threads", w, c, flush=True)
_, w, c = decode_frames(IDLE, 96, threads=1)
results["idle_decode_1_thread"] = {"wall_ms_per": w, "cpu_ms_per": c}
print("idle_decode_1_thread", w, c, flush=True)

bgr = [f.to_ndarray(format="bgr24") for f in yuv[:48]]
H, W = bgr[0].shape[:2]
results["frame_shape"] = [H, W, 3]

# ----------------------------------------------------- 2. BGR -> yuv420p (push)
def conv(i):
    av.VideoFrame.from_ndarray(bgr[i % 48], format="bgr24").reformat(format="yuv420p")

timed("bgr_to_yuv420p_pyav_1thread", conv, 200)
parallel("bgr_to_yuv420p_pyav_8threads", lambda t: conv, 8, 100)

def conv_cv2(i):
    y = cv2.cvtColor(bgr[i % 48], cv2.COLOR_BGR2YUV_I420)
    av.VideoFrame.from_ndarray(y, format="yuv420p")

timed("bgr_to_yuv420p_cv2_1thread", conv_cv2, 200)

# --------------------------------------------------- 3. per-frame numpy copies
timed("numpy_full_frame_copy", lambda i: bgr[i % 48].copy(), 400)

# compose proxy: prepared frame copy + resize 256->bbox + ROI alpha blend
face = np.random.randint(0, 255, (256, 256, 3), np.uint8)
alpha = np.random.randint(0, 255, (300, 300, 1), np.uint8)

def compose_proxy(i):
    ori = bgr[i % 48].copy()
    r = cv2.resize(face, (260, 300))
    roi = ori[300:600, 120:380]
    a = alpha[:, :260].astype(np.uint16)
    roi[:] = ((r.astype(np.uint16) * a + roi.astype(np.uint16) * (255 - a)) // 255).astype(np.uint8)

timed("compose_proxy_copy_resize_blend", compose_proxy, 200)

# ------------------------------------------------ 4. aiortc H264 (libx264) encode
from aiortc.codecs.h264 import H264Encoder
from aiortc.codecs.opus import OpusEncoder

yuv_frames = [av.VideoFrame.from_ndarray(b, format="bgr24").reformat(format="yuv420p") for b in bgr]
TB = fractions.Fraction(1, 90000)

def make_aiortc_encoder(_t=0):
    enc = H264Encoder()
    state = {"i": 0}

    def w(i):
        f = yuv_frames[i % 48]
        f.pts = state["i"] * 4500
        f.time_base = TB
        state["i"] += 1
        payloads, ts = enc.encode(f, force_keyframe=(state["i"] == 1))
        state.setdefault("pk", 0)
        state["pk"] += len(payloads)
        state.setdefault("bytes", 0)
        state["bytes"] += sum(len(p) for p in payloads)
    w.state = state
    w.enc = enc
    return w

tc0 = thread_count()
w0 = make_aiortc_encoder()
for i in range(10):
    w0(i)  # warm
results["aiortc_libx264_threads_spawned_per_encoder"] = thread_count() - tc0
r = timed("aiortc_libx264_encode_1enc", w0, 120)
results["aiortc_libx264_packets_per_frame"] = w0.state["pk"] / w0.state["i"]
results["aiortc_libx264_bytes_per_frame"] = w0.state["bytes"] / w0.state["i"]
print("threads spawned per aiortc encoder", results["aiortc_libx264_threads_spawned_per_encoder"],
      "pkts/frame", results["aiortc_libx264_packets_per_frame"], flush=True)

for n in (4, 8, 15):
    ws = []
    def mk(t):
        w = make_aiortc_encoder()
        for i in range(5):
            w(i)
        ws.append(w)
        return w
    parallel(f"aiortc_libx264_parallel_{n}enc", mk, n, 60)
    del ws

# libx264 direct with cheaper settings
def make_x264(preset, threads, tune="zerolatency"):
    cc = av.CodecContext.create("libx264", "w")
    cc.width, cc.height = W, H
    cc.pix_fmt = "yuv420p"
    cc.bit_rate = 1_500_000
    cc.framerate = fractions.Fraction(20, 1)
    cc.time_base = fractions.Fraction(1, 20)
    opts = {"preset": preset, "tune": tune, "profile": "baseline", "level": "31"}
    if threads is not None:
        opts["threads"] = str(threads)
    cc.options = opts
    st = {"i": 0}

    def w(i):
        f = yuv_frames[i % 48]
        f.pts = st["i"]
        f.time_base = fractions.Fraction(1, 20)
        st["i"] += 1
        for p in cc.encode(f):
            bytes(p)
    return w

for preset, thr in (("medium", None), ("veryfast", None), ("ultrafast", None), ("veryfast", 1), ("ultrafast", 1), ("ultrafast", 2)):
    w = make_x264(preset, thr)
    for i in range(5):
        w(i)
    timed(f"x264_{preset}_threads{thr}_1enc", w, 100)
parallel("x264_ultrafast_threads1_parallel_15enc", lambda t: make_x264("ultrafast", 1), 15, 60)
parallel("x264_veryfast_threads1_parallel_15enc", lambda t: make_x264("veryfast", 1), 15, 60)

# ---------------------------------------------------------------- 5. Opus
from av import AudioFrame
opus = OpusEncoder()
af_pts = {"i": 0}

def opus_w(i):
    fr = AudioFrame(format="s16", layout="stereo", samples=960)
    for p in fr.planes:
        p.update(np.zeros(960 * 2, np.int16).tobytes())
    fr.sample_rate = 48000
    fr.pts = af_pts["i"] * 960
    fr.time_base = fractions.Fraction(1, 48000)
    af_pts["i"] += 1
    opus.encode(fr)

timed("aiortc_opus_encode_20ms", opus_w, 500)

# ------------------------------------------------------ 6. RTP serialize + SRTP
from aiortc.rtp import RtpPacket
try:
    from pylibsrtp import Policy, Session
    key = os.urandom(30)
    pol = Policy(key=key, ssrc_type=Policy.SSRC_ANY_OUTBOUND)
    srtp = Session(policy=pol)
except Exception as exc:  # pragma: no cover
    srtp = None
    results["srtp_error"] = str(exc)

payload = os.urandom(1200)
seq = {"i": 0}

def rtp_w(i):
    pkt = RtpPacket(payload_type=96, sequence_number=seq["i"] & 0xFFFF, timestamp=i * 4500)
    seq["i"] += 1
    pkt.ssrc = 1234
    pkt.payload = payload
    pkt.marker = 0
    data = pkt.serialize({})
    if srtp is not None:
        srtp.protect(data)

timed("rtp_serialize_plus_srtp_per_packet", rtp_w, 5000)

# ------------------------------------------------ 7. asyncio cross-thread handoff
import asyncio
loop = asyncio.new_event_loop()
t = threading.Thread(target=loop.run_forever, daemon=True)
t.start()

async def noop(x):
    return x

def handoff(i):
    asyncio.run_coroutine_threadsafe(noop(i), loop).result()

timed("run_coroutine_threadsafe_roundtrip_idle_loop", handoff, 2000)
loop.call_soon_threadsafe(loop.stop)

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2)
print("wrote", OUT)
