"""Working H.264 encoder selection for aiortc 1.14 (300 fps plan item 1.8).

aiortc 1.14's ``H264Encoder._encode_frame`` always opens ``libx264`` with the
x264 default preset (medium), ``tune=zerolatency``, ``level=31``, profile
Baseline and FFmpeg's automatic thread count. The previous
``api_server.enable_h264_nvenc`` patched a ``create_encoder_context`` hook that
aiortc 1.14 no longer has, so NVENC was never used while the log said
"encoder set to h264_nvenc".

WEBRTC_H264_IMPL
  aiortc     (default) leave aiortc's encoder untouched = today's behaviour.
  x264tuned  libx264, same level/profile/tune/rate control, with
             WEBRTC_H264_X264_PRESET (default veryfast) and
             WEBRTC_H264_X264_THREADS (default 1).
  nvenc      h264_nvenc (WEBRTC_H264_NVENC_PRESET/TUNE, CBR, no B-frames,
             forced IDR on keyframe requests) while a process-wide semaphore of
             WEBRTC_NVENC_MAX_SESSIONS (default 12; the driver limit on this
             GeForce card) has a slot; beyond it, or if NVENC fails to open,
             that encoder uses x264tuned. Every open is logged truthfully.

The override replaces ``aiortc.codecs.H264Encoder`` (what ``get_encoder``
instantiates) with a subclass; packetization, RTP timestamps, bitrate
adaptation and the encode-timing wrapper are inherited unchanged.
"""
from __future__ import annotations

import fractions
import threading
import weakref
from typing import Optional

from scripts.webrtc_media_flags import env_int, env_str

VALID_IMPLS = ("aiortc", "x264tuned", "nvenc")


class NvencSessionSlots:
    """Process-wide cap on concurrently open NVENC encoders."""

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._lock = threading.Lock()
        self._in_use = 0
        self.peak = 0
        self.denied = 0
        self.opened = 0
        self.open_failures = 0

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_use >= self.capacity:
                self.denied += 1
                return False
            self._in_use += 1
            self.peak = max(self.peak, self._in_use)
            return True

    def release(self) -> None:
        with self._lock:
            self._in_use = max(0, self._in_use - 1)

    @property
    def in_use(self) -> int:
        return self._in_use

    def stats(self) -> dict:
        return {"capacity": self.capacity, "in_use": self._in_use, "peak": self.peak,
                "denied": self.denied, "opened": self.opened,
                "open_failures": self.open_failures}


_status: dict = {"impl": "aiortc", "installed": False}
_slots: Optional[NvencSessionSlots] = None


def selected_impl() -> str:
    impl = env_str("WEBRTC_H264_IMPL", "aiortc").lower()
    if impl not in VALID_IMPLS:
        raise RuntimeError(f"Unsupported WEBRTC_H264_IMPL={impl!r}; use one of {VALID_IMPLS}")
    return impl


def x264_settings() -> dict:
    return {"preset": env_str("WEBRTC_H264_X264_PRESET", "veryfast"),
            "threads": env_int("WEBRTC_H264_X264_THREADS", 1, minimum=0, maximum=64)}


def nvenc_settings() -> dict:
    return {"preset": env_str("WEBRTC_H264_NVENC_PRESET", "p2"),
            "tune": env_str("WEBRTC_H264_NVENC_TUNE", "ll"),
            "max_sessions": env_int("WEBRTC_NVENC_MAX_SESSIONS", 12, minimum=0, maximum=64)}


def open_x264(av, width: int, height: int, bit_rate: int, frame_rate: int,
              preset: str, threads: int):
    """libx264 with aiortc's options plus an explicit preset/thread count."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width = width
    codec.height = height
    codec.bit_rate = bit_rate
    codec.pix_fmt = "yuv420p"
    codec.framerate = fractions.Fraction(frame_rate, 1)
    codec.time_base = fractions.Fraction(1, frame_rate)
    codec.options = {"level": "31", "tune": "zerolatency", "preset": preset}
    codec.profile = "Baseline"
    if threads:
        codec.thread_count = int(threads)
    return codec


def open_nvenc(av, width: int, height: int, bit_rate: int, frame_rate: int,
               preset: str, tune: str):
    codec = av.CodecContext.create("h264_nvenc", "w")
    codec.width = width
    codec.height = height
    codec.bit_rate = bit_rate
    codec.pix_fmt = "yuv420p"
    codec.framerate = fractions.Fraction(frame_rate, 1)
    codec.time_base = fractions.Fraction(1, frame_rate)
    codec.options = {
        "preset": preset, "tune": tune, "rc": "cbr", "bf": "0", "delay": "0",
        "zerolatency": "1", "forced-idr": "1", "profile": "baseline",
        "maxrate": str(bit_rate), "bufsize": str(bit_rate),
        "g": str(frame_rate * 100),
    }
    codec.open()  # fail here (not mid-stream) when the driver refuses a session
    return codec


def build_encoder_class(h264, av, impl: str, slots: Optional[NvencSessionSlots]):
    x264 = x264_settings()
    nvenc = nvenc_settings()
    base = h264.H264Encoder

    class MuseTalkH264Encoder(base):
        musetalk_impl = impl

        def __init__(self) -> None:
            super().__init__()
            self._musetalk_codec_name: Optional[str] = None
            self._musetalk_slot_held = False
            self._musetalk_slot_finalizer = None
            self._musetalk_logged = False

        def _musetalk_release_slot(self) -> None:
            finalizer = self._musetalk_slot_finalizer
            self._musetalk_slot_finalizer = None
            self._musetalk_slot_held = False
            if finalizer is not None:
                finalizer()

        def _musetalk_open(self, frame):
            rate = h264.MAX_FRAME_RATE
            if impl == "nvenc" and slots is not None:
                if not self._musetalk_slot_held and slots.try_acquire():
                    self._musetalk_slot_held = True
                    self._musetalk_slot_finalizer = weakref.finalize(self, slots.release)
                if self._musetalk_slot_held:
                    try:
                        codec = open_nvenc(av, frame.width, frame.height, self.target_bitrate,
                                           rate, nvenc["preset"], nvenc["tune"])
                        slots.opened += 1
                        self._musetalk_note("h264_nvenc", f"slot={slots.in_use}/{slots.capacity}")
                        return codec
                    except Exception as exc:
                        slots.open_failures += 1
                        self._musetalk_release_slot()
                        print(f"⚠️ WebRTC H.264 NVENC open failed ({exc}); "
                              "this encoder falls back to x264tuned", flush=True)
            codec = open_x264(av, frame.width, frame.height, self.target_bitrate, rate,
                              x264["preset"], x264["threads"])
            reason = ""
            if impl == "nvenc":
                reason = (f" (nvenc fallback: {slots.in_use}/{slots.capacity} sessions in use)"
                          if slots is not None else " (nvenc fallback)")
            self._musetalk_note("libx264",
                                f"preset={x264['preset']} threads={x264['threads'] or 'auto'}"
                                f"{reason}")
            return codec

        def _musetalk_note(self, codec_name: str, detail: str) -> None:
            changed = codec_name != self._musetalk_codec_name
            self._musetalk_codec_name = codec_name
            if changed or not self._musetalk_logged:
                self._musetalk_logged = True
                print(f"🎞️ WebRTC H.264 encoder opened impl={impl} codec={codec_name} "
                      f"{detail} bitrate={self.target_bitrate}", flush=True)

        def _encode_frame(self, frame, force_keyframe):
            # Same control flow as aiortc 1.14 H264Encoder._encode_frame; only
            # the codec construction differs.
            if self.codec and (
                frame.width != self.codec.width
                or frame.height != self.codec.height
                or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
            ):
                self.buffer_data = b""
                self.buffer_pts = None
                self.codec = None

            if force_keyframe:
                frame.pict_type = av.video.frame.PictureType.I
            else:
                frame.pict_type = av.video.frame.PictureType.NONE

            if self.codec is None:
                self.codec = self._musetalk_open(frame)

            data_to_send = b""
            for package in self.codec.encode(frame):
                data_to_send += bytes(package)

            if data_to_send:
                yield from self._split_bitstream(data_to_send)

    MuseTalkH264Encoder.__name__ = "MuseTalkH264Encoder"
    MuseTalkH264Encoder.__qualname__ = "MuseTalkH264Encoder"
    return MuseTalkH264Encoder


def install_h264_encoder(max_bitrate: int, label: str = "api_server") -> dict:
    """Apply WEBRTC_H264_IMPL before any peer connection exists; log the truth."""
    global _status, _slots
    import av
    import aiortc.codecs as codecs
    import aiortc.codecs.h264 as h264

    impl = selected_impl()
    # Unchanged from the previous patch: aiortc clamps target_bitrate to this.
    h264.MAX_BITRATE = int(max_bitrate)
    if impl == "aiortc":
        if codecs.H264Encoder is not h264.H264Encoder:
            codecs.H264Encoder = h264.H264Encoder
        _status = {"impl": "aiortc", "codec": "libx264", "installed": False,
                   "max_bitrate": int(max_bitrate),
                   "detail": "aiortc 1.14 built-in: preset=medium tune=zerolatency "
                             "level=31 profile=Baseline threads=auto"}
        print(f"🎞️ [{label}] WebRTC H.264 impl=aiortc codec=libx264 (aiortc 1.14 built-in: "
              f"preset=medium tune=zerolatency profile=Baseline threads=auto) "
              f"max_bitrate={int(max_bitrate)}; WEBRTC_H264_ENCODER/NVENC settings are not "
              "used by this impl", flush=True)
        return dict(_status)

    if impl == "nvenc":
        _slots = NvencSessionSlots(nvenc_settings()["max_sessions"])
    encoder_class = build_encoder_class(h264, av, impl, _slots)
    codecs.H264Encoder = encoder_class
    x264 = x264_settings()
    _status = {"impl": impl, "installed": True, "max_bitrate": int(max_bitrate),
               "x264": x264,
               "nvenc": dict(nvenc_settings()) if impl == "nvenc" else None}
    print(f"🎞️ [{label}] WebRTC H.264 impl={impl} "
          + (f"codec=h264_nvenc preset={nvenc_settings()['preset']} "
             f"tune={nvenc_settings()['tune']} max_sessions={_slots.capacity} "
             f"fallback=libx264 preset={x264['preset']} threads={x264['threads']} "
             if impl == "nvenc" else
             f"codec=libx264 preset={x264['preset']} threads={x264['threads']} ")
          + f"max_bitrate={int(max_bitrate)}", flush=True)
    return dict(_status)


def h264_status() -> dict:
    status = dict(_status)
    if _slots is not None:
        status["nvenc_sessions"] = _slots.stats()
    return status
