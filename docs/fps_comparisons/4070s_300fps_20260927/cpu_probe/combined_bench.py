#!/usr/bin/env python3
"""Combined CPU-side load at N x 20 fps in ONE process (server-like threads):
compose_frame (real avatar plan) [+ chin compose surrogate] -> colorconv -> encode (+RTP payloadize)
-> RTP serialize + SRTP protect + UDP sendto per packet, plus Opus audio (50 pkt/s/stream).
FaceMesh trackers (if requested) run as separate paced processes from fm_bench.py, started by the
wrapper shell. Machine-wide CPU is sampled from /proc/stat. GPU is NOT used (faces are synthetic).
"""
import argparse, fractions, functools, json, os, socket, statistics, sys, threading, time
import torch
torch.load = functools.partial(torch.load, map_location="cpu")
sys.path.insert(0, "/workspace/MuseTalk"); sys.path.insert(0, "/workspace/MuseTalk/scripts")
import numpy as np, cv2
cv2.setNumThreads(1)
import av
from benchmark_compose_frame import load_avatar

ap = argparse.ArgumentParser()
ap.add_argument("--streams", type=int, default=15)
ap.add_argument("--secs", type=float, default=20)
ap.add_argument("--codec", default="vp8_native")  # vp8_native | vp8_aiortc | x264_aiortc | x264_ultrafast
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--cc", default="pyav")  # pyav (current server) | cv2
ap.add_argument("--chin_surrogate", type=int, default=0, help="extra compose calls/frame to stand in for aligned compose (+1.5ms)")
ap.add_argument("--bitrate", type=int, default=2_500_000)
ap.add_argument("--switch_ms", type=float, default=0.0)
ap.add_argument("--unpaced", action="store_true")
a = ap.parse_args()
if a.switch_ms:
    sys.setswitchinterval(a.switch_ms / 1000.0)

avatar = load_avatar("japanese_realtime_talking_7d94520b7f", 8, "v15")
ncyc = len(avatar.coord_list_cycle)
VT = fractions.Fraction(1, 90000)


def proc_stat():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def make_video_encoder():
    if a.codec == "vp8_native":
        os.environ.setdefault("WEBRTC_NATIVE_VP8_MAX_BITRATE_BPS", "3500000")
        os.environ.setdefault("WEBRTC_NATIVE_VP8_BITRATE_FLOOR_BPS", str(a.bitrate))
        from scripts.webrtc_native_vp8 import load_native_encoder
        cls = load_native_encoder("/workspace/MuseTalk/.runtime/native_vp8")
        if a.threads:
            sys.modules["aiortc.codecs._musetalk_native_vpx_111"].number_of_threads = lambda p, c, _t=a.threads: _t
        e = cls(); e.target_bitrate = a.bitrate; return e
    if a.codec == "vp8_aiortc":
        import aiortc.codecs.vpx as vpx
        vpx.MAX_BITRATE = max(vpx.MAX_BITRATE, a.bitrate)
        e = vpx.Vp8Encoder(); e.target_bitrate = a.bitrate; return e
    if a.codec in ("x264_aiortc", "x264_ultrafast"):
        import aiortc.codecs.h264 as h264
        h264.MAX_BITRATE = max(h264.MAX_BITRATE, a.bitrate)
        e = h264.H264Encoder(); e.target_bitrate = a.bitrate
        if a.codec == "x264_ultrafast":
            orig = e._encode_frame
            def patched(frame, force_keyframe, _e=e, _orig=orig):
                if _e.codec is None:
                    c = av.CodecContext.create("libx264", "w"); c.width, c.height = frame.width, frame.height
                    c.bit_rate = _e.target_bitrate; c.pix_fmt = "yuv420p"; c.framerate = fractions.Fraction(20, 1)
                    c.time_base = fractions.Fraction(1, 20); c.options = {"preset": "ultrafast", "tune": "zerolatency", "level": "31"}
                    c.profile = "Baseline"; c.thread_count = max(1, a.threads); _e.codec = c
                return _orig(frame, force_keyframe)
            e._encode_frame = patched
        return e
    raise ValueError(a.codec)


from aiortc.rtp import RtpPacket, HeaderExtensionsMap
from aiortc.rtcrtpparameters import RTCRtpParameters, RTCRtpHeaderExtensionParameters
from aiortc import clock
from aiortc.codecs.opus import OpusEncoder
from pylibsrtp import Policy, Session
hmap = HeaderExtensionsMap()
hmap.configure(RTCRtpParameters(headerExtensions=[
    RTCRtpHeaderExtensionParameters(id=1, uri="urn:ietf:params:rtp-hdrext:sdes:mid"),
    RTCRtpHeaderExtensionParameters(id=3, uri="http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time")]))
sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sink.bind(("127.0.0.1", 0))
sink.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
dest = sink.getsockname()
stop_drain = False


def drain():
    sink.settimeout(0.2)
    while not stop_drain:
        try:
            sink.recv(65536)
        except OSError:
            pass


stats = []


def stream(sid, t0):
    enc = make_video_encoder()
    aenc = OpusEncoder()
    tx = Session(policy=Policy(key=os.urandom(30), ssrc_type=Policy.SSRC_ANY_OUTBOUND,
                               srtp_profile=Policy.SRTP_PROFILE_AES128_CM_SHA1_80))
    us = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    face = np.random.default_rng(sid).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    pcm = (np.random.default_rng(sid).integers(-3000, 3000, (1, 960 * 2))).astype(np.int16)
    seq = 0; apts = 0; lat = []; late = 0; nbytes = 0
    nframes = int(a.secs * 20) if not a.unpaced else 10**9
    for k in range(nframes):
        due = t0 + k * 0.05
        now = time.perf_counter()
        if a.unpaced:
            if now > t0 + a.secs:
                nframes = k
                break
        elif due > now:
            time.sleep(due - now)
        elif now - due > 0.05:
            late += 1
        s = time.perf_counter()
        idx = (sid * 37 + k) % ncyc
        composed = avatar.compose_frame(face, idx)
        for _ in range(a.chin_surrogate):
            avatar.compose_frame(face, idx)
        if a.cc == "pyav":
            vf = av.VideoFrame.from_ndarray(composed, format="bgr24").reformat(format="yuv420p")
        else:
            vf = av.VideoFrame.from_ndarray(cv2.cvtColor(composed, cv2.COLOR_BGR2YUV_I420), format="yuv420p")
        vf.pts = k * 4500; vf.time_base = VT; vf.duration = 4500; vf.pict_type = 0
        payloads, ts = enc.encode(vf)
        for i, p in enumerate(payloads):
            pk = RtpPacket(payload_type=96, sequence_number=seq & 0xFFFF, timestamp=ts)
            pk.ssrc = sid; pk.payload = p; pk.marker = int(i == len(payloads) - 1)
            pk.extensions.abs_send_time = (clock.current_ntp_time() >> 14) & 0xFFFFFF; pk.extensions.mid = "0"
            d = tx.protect(pk.serialize(hmap)); us.sendto(d, dest); nbytes += len(d); seq += 1
        # audio: 2.5 Opus packets per video frame on average
        for _ in range(3 if k % 2 else 2):
            af = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo"); af.sample_rate = 48000
            af.pts = apts; af.time_base = fractions.Fraction(1, 48000); apts += 960
            ap_, ats = aenc.encode(af)
            for p in ap_:
                pk = RtpPacket(payload_type=111, sequence_number=seq & 0xFFFF, timestamp=ats)
                pk.ssrc = 10000 + sid; pk.payload = p; pk.extensions.mid = "1"
                us.sendto(tx.protect(pk.serialize(hmap)), dest); seq += 1
        lat.append((time.perf_counter() - s) * 1000)
    stats.append({"lat": lat, "late": late, "bytes": nbytes, "frames": len(lat), "end": time.perf_counter()})


dt = threading.Thread(target=drain, daemon=True); dt.start()
for w in range(30):  # warm compose plans
    avatar.compose_frame(np.zeros((256, 256, 3), np.uint8), w)
t0 = time.perf_counter() + 1.0
ths = [threading.Thread(target=stream, args=(s, t0)) for s in range(a.streams)]
for t in ths: t.start()
time.sleep(max(0, t0 - time.perf_counter()) + 2.0)  # skip encoder warmup/first keyframes
tot0, idle0 = proc_stat(); r0 = os.times(); w0 = time.perf_counter()
time.sleep(a.secs - 4.0)
tot1, idle1 = proc_stat(); r1 = os.times(); w1 = time.perf_counter()
for t in ths: t.join()
stop_drain = True
lat = [x for s in stats for x in s["lat"]]
ncpu = os.cpu_count()
out = {"switch_ms": a.switch_ms or 5.0, "streams": a.streams, "codec": a.codec, "codec_threads": a.threads, "cc": a.cc, "chin_surrogate": a.chin_surrogate,
       "target_agg_fps": ("unpaced" if a.unpaced else 20 * a.streams),
       "achieved_agg_fps": round(sum(s["frames"] for s in stats) / (max(s["end"] for s in stats) - t0), 1),
       "late_frames": sum(s["late"] for s in stats),
       "frame_work_ms_mean": round(statistics.fmean(lat), 2), "frame_work_ms_p95": round(sorted(lat)[int(.95 * len(lat)) - 1], 2),
       "frame_work_ms_p99": round(sorted(lat)[int(.99 * len(lat)) - 1], 2), "frame_work_ms_max": round(max(lat), 2),
       "this_proc_cores": round(((r1.user - r0.user) + (r1.system - r0.system)) / (w1 - w0), 2),
       "machine_cores_busy": round(ncpu * (1 - (idle1 - idle0) / (tot1 - tot0)), 2),
       "video_kbps_per_stream": round(sum(s["bytes"] for s in stats) * 8 / (a.secs * a.streams) / 1000, 0)}
print(json.dumps(out))
