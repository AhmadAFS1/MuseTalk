#!/usr/bin/env python
"""CPU test for WEBRTC_H264_IMPL (plan item 1.8), no CUDA (NVENC is faked).

  H1 default (unset / aiortc): aiortc.codecs.H264Encoder is untouched, MAX_BITRATE
     is set exactly like the old enable_h264_nvenc(), status/log say libx264.
  H2 x264tuned: the registered encoder is the override subclass; it opens
     libx264 with the requested preset and 1 thread, produces decodable RTP
     payloads, and still goes through a timing wrapper patched onto the base
     class (api_server's patch_h264_encode_timing).
  H3 nvenc semaphore (fake h264_nvenc codec): capacity 2 -> encoders 1-2 get
     NVENC, encoder 3 falls back to x264tuned; dropping one encoder frees its
     slot; an NVENC open failure falls back and releases the slot.
  H4 invalid WEBRTC_H264_IMPL fails loudly.
Each scenario runs in its own interpreter so module patches never leak.
Writes test_h264_override.json next to this file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
OUT = Path(__file__).with_name("test_h264_override.json")


def _frames(count=6):
    from fractions import Fraction
    import av
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (896, 512, 3), dtype=np.uint8)
    out = []
    for i in range(count):
        img = base.copy()
        img[:32] = (i * 20) % 256
        frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
        frame.pts, frame.time_base = i * 4500, Fraction(1, 90000)
        out.append(frame)
    return out


def x264_sei(data: bytes) -> dict:
    """Options string x264 embeds in its SEI (proof of the preset actually used)."""
    start = data.find(b"x264 - core")
    if start < 0:
        return {}
    text = data[start:start + 2000].split(b"\x00")[0].decode("latin1", "replace")
    opts = text.split("options: ", 1)[-1].split()
    return {k: v for k, v in (o.split("=", 1) for o in opts if "=" in o)}


def scenario_default() -> dict:
    import aiortc.codecs as codecs
    import aiortc.codecs.h264 as h264
    from scripts.webrtc_h264_override import h264_status, install_h264_encoder
    original = codecs.H264Encoder
    status = install_h264_encoder(6_000_000, label="test")
    enc = codecs.H264Encoder()
    enc.target_bitrate = 2_000_000
    payloads = [enc.encode(f)[0] for f in _frames(3)]
    sei = x264_sei(b"".join(h264.h264_depayload(p) for p in payloads[0]))
    ok = (codecs.H264Encoder is original is h264.H264Encoder and h264.MAX_BITRATE == 6_000_000
          and status["impl"] == "aiortc" and not status["installed"]
          and enc.codec.name == "libx264" and all(payloads) and sei.get("subme") == "7")
    return {"ok": ok, "status": h264_status(), "codec": enc.codec.name,
            "x264_sei": {k: sei.get(k) for k in ("subme", "me", "ref", "threads", "sliced_threads")}}


def scenario_x264tuned() -> dict:
    import aiortc.codecs as codecs
    import aiortc.codecs.h264 as h264
    from aiortc.jitterbuffer import JitterFrame
    from scripts.webrtc_h264_override import install_h264_encoder
    install_h264_encoder(6_000_000, label="test")
    calls = {"n": 0}
    base_encode = h264.H264Encoder.encode

    def timed(self, frame, *a, **k):  # api_server's timing patch wraps the BASE class
        calls["n"] += 1
        return base_encode(self, frame, *a, **k)

    h264.H264Encoder.encode = timed
    enc = codecs.H264Encoder()
    dec = h264.H264Decoder()
    decoded = 0
    sei = {}
    for i, frame in enumerate(_frames(6)):
        payloads, ts = enc.encode(frame, force_keyframe=(i == 0))
        data = b"".join(h264.h264_depayload(p) for p in payloads)
        sei = sei or x264_sei(data)
        decoded += len(dec.decode(JitterFrame(data, ts))) if data else 0
    # veryfast = subme 2, me hex, ref 1; one thread, no sliced threads.
    ok = (codecs.H264Encoder is not h264.H264Encoder and type(enc).__name__ == "MuseTalkH264Encoder"
          and enc.codec.name == "libx264" and sei.get("subme") == "2" and sei.get("ref") == "1"
          and sei.get("threads") == "1" and enc.codec.thread_count == 1 and calls["n"] == 6
          and decoded >= 5)
    return {"ok": ok, "codec": enc.codec.name,
            "x264_sei": {k: sei.get(k) for k in ("subme", "me", "ref", "threads", "sliced_threads")},
            "thread_count": enc.codec.thread_count, "timing_wrapper_calls": calls["n"],
            "decoded_frames": decoded}


def scenario_nvenc_fake() -> dict:
    import gc
    import av
    import aiortc.codecs.h264 as h264
    from scripts import webrtc_h264_override as ov

    class FakeNvenc:
        name = "h264_nvenc"
        fail = False

        def __init__(self):
            self.width = self.height = 0
            self.bit_rate = 1
            self.options = {}

        def open(self):
            if FakeNvenc.fail:
                raise RuntimeError("fake: OpenEncodeSessionEx failed: out of memory (10)")

        def encode(self, frame):
            return []

    class FakeCodecContext:
        @staticmethod
        def create(name, mode):
            return FakeNvenc() if name == "h264_nvenc" else av.CodecContext.create(name, mode)

    class FakeAV:
        CodecContext = FakeCodecContext
        video = av.video

    slots = ov.NvencSessionSlots(2)
    cls = ov.build_encoder_class(h264, FakeAV, "nvenc", slots)
    frames = _frames(1)
    encoders = [cls() for _ in range(3)]
    for e in encoders:
        list(e._encode_frame(frames[0], True))
    names = [e.codec.name for e in encoders]
    in_use_3 = slots.in_use
    del encoders[0]
    gc.collect()
    freed = slots.in_use
    e4 = cls()
    list(e4._encode_frame(frames[0], True))
    FakeNvenc.fail = True
    e5_slots_before = slots.in_use
    del encoders[0]  # frees a slot so e5 tries NVENC and fails to open
    gc.collect()
    e5 = cls()
    list(e5._encode_frame(frames[0], True))
    stats = slots.stats()
    ok = (names == ["h264_nvenc", "h264_nvenc", "libx264"] and in_use_3 == 2 and freed == 1
          and e4.codec.name == "h264_nvenc" and e5.codec.name == "libx264"
          and stats["open_failures"] == 1 and stats["peak"] == 2 and slots.in_use == 1
          and stats["denied"] >= 1)
    return {"ok": ok, "first_three": names, "in_use_after_three": in_use_3,
            "in_use_after_drop": freed, "e4": e4.codec.name, "e5_after_open_failure": e5.codec.name,
            "e5_slots_before": e5_slots_before, "slots": stats}


def scenario_invalid() -> dict:
    from scripts.webrtc_h264_override import install_h264_encoder
    try:
        install_h264_encoder(6_000_000, label="test")
    except RuntimeError as exc:
        return {"ok": "Unsupported WEBRTC_H264_IMPL" in str(exc), "error": str(exc)}
    return {"ok": False, "error": "no exception"}


SCENARIOS = {
    "H1_default_untouched": (scenario_default, {}),
    "H2_x264tuned_preset_threads_timing": (scenario_x264tuned, {
        "WEBRTC_H264_IMPL": "x264tuned", "WEBRTC_H264_X264_PRESET": "veryfast",
        "WEBRTC_H264_X264_THREADS": "1"}),
    "H3_nvenc_semaphore_fallback": (scenario_nvenc_fake, {
        "WEBRTC_H264_IMPL": "nvenc", "WEBRTC_H264_X264_PRESET": "veryfast",
        "WEBRTC_H264_X264_THREADS": "1"}),
    "H4_invalid_impl_fails": (scenario_invalid, {"WEBRTC_H264_IMPL": "bogus"}),
}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        print("RESULT " + json.dumps(SCENARIOS[sys.argv[2]][0](), default=str), flush=True)
        return 0
    results = []
    for name, (_fn, env_over) in SCENARIOS.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith(("WEBRTC_H264", "WEBRTC_NVENC"))}
        env.update(env_over)
        proc = subprocess.run([sys.executable, __file__, "--scenario", name], env=env,
                              cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        detail = json.loads(line[7:]) if line else {"ok": False, "stderr": proc.stderr[-1500:]}
        logs = [l for l in proc.stdout.splitlines() if "H.264" in l]
        ok = bool(detail.pop("ok", False)) and proc.returncode == 0
        results.append({"test": name, "result": "pass" if ok else "fail", "logs": logs, **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:300]}")
        for l in logs:
            print(f"    log: {l}")
    passed = all(r["result"] == "pass" for r in results)
    OUT.write_text(json.dumps({"suite": "h264_override", "passed": passed, "results": results},
                              indent=2, default=str))
    print(f"{'PASS' if passed else 'FAIL'} h264_override -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
