"""CPU cost of aiortc's per-packet Opus audio path (20 ms mono 48 kHz), no network."""
import time, json, fractions
import numpy as np
from av import AudioFrame
from aiortc.codecs import get_encoder
from aiortc.rtcrtpparameters import RTCRtpCodecParameters
from aiortc import rtp
codec = RTCRtpCodecParameters(mimeType="audio/opus", clockRate=48000, channels=2, payloadType=96)
enc = get_encoder(codec)
def frame(i, silent):
    pcm = np.zeros((1, 960), dtype=np.int16) if silent else (np.random.randn(1, 960) * 3000).astype(np.int16)
    f = AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    f.sample_rate = 48000; f.pts = i * 960; f.time_base = fractions.Fraction(1, 48000)
    return f
out = {}
for silent in (True, False):
    frames = [frame(i, silent) for i in range(1000)]
    for f in frames[:50]: enc.encode(f)
    t0 = time.perf_counter()
    for f in frames[50:]: enc.encode(f)
    enc_ms = (time.perf_counter() - t0) / 950 * 1000
    t0 = time.perf_counter()
    for f in frames[50:]: rtp.compute_audio_level_dbov(f)
    lvl_ms = (time.perf_counter() - t0) / 950 * 1000
    out["silence" if silent else "speech"] = {"opus_encode_ms_per_packet": enc_ms, "audio_level_ms_per_packet": lvl_ms}
for k, v in out.items():
    per_stream_ms_per_s = (v["opus_encode_ms_per_packet"] + v["audio_level_ms_per_packet"]) * 50
    v["cpu_ms_per_stream_second"] = per_stream_ms_per_s
    v["cpu_ms_per_second_at_20_streams"] = per_stream_ms_per_s * 20
print(json.dumps(out, indent=2))
