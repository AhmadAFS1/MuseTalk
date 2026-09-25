"""Opt-in pinned native VP8 encoder; keep installed aiortc/PyAV unchanged.

The native profile selects VP8 before offer processing and changes only its
encoder registry entry. RTP/feedback, audio/H264 implementations, and the
receiver decoder remain upstream.
"""
import ctypes
from fractions import Fraction
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys
import threading

from scripts.install_native_vp8 import DEFAULT_DIRECTORY, validate_install

SUPPORTED_PACKAGES = {"aiortc": "1.14.0", "av": "16.1.0", "cffi": "2.1.1"}
VIDEO_TIME_BASE = Fraction(1, 90000)
_loaded_class = None
_loaded_directory = None


def validate_native_offer(sdp, description_type="offer"):
    """Reject unsupported native-profile offers before changing peer state."""
    from aiortc.sdp import SessionDescription
    if description_type != "offer":
        raise ValueError("Native VP8 profile requires an SDP offer")
    try:
        media = SessionDescription.parse(sdp).media
    except Exception as exc:
        raise ValueError("Native VP8 profile received an invalid SDP offer") from exc
    videos = [entry for entry in media if entry.kind == "video"]
    if (len(videos) != 1 or videos[0].port == 0
            or videos[0].direction not in ("recvonly", "sendrecv")):
        raise ValueError("Native VP8 profile requires exactly one receiving video m-line")
    if not any(codec.mimeType.lower() == "video/vp8" and codec.clockRate == 90000
               for codec in videos[0].rtp.codecs):
        raise ValueError("Native VP8 profile requires client VP8 support; H264-only offers are unsupported")


def prefer_native_vp8(pc):
    from aiortc import RTCRtpSender
    capabilities = RTCRtpSender.getCapabilities("video").codecs
    preferred = [codec for codec in capabilities
                 if codec.mimeType.lower() in ("video/vp8", "video/rtx")]
    if not any(codec.mimeType.lower() == "video/vp8" for codec in preferred):
        raise RuntimeError("Native VP8 profile has no local VP8 capability")
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(preferred)


def validate_runtime():
    """Reject untested Python/package interfaces before executing native files."""
    if (platform.system() != "Linux" or platform.machine() != "x86_64"
            or platform.python_implementation() != "CPython"
            or sys.version_info[:2] != (3, 10)):
        raise RuntimeError("Native VP8 requires the validated Linux x86_64 CPython 3.10 runtime")
    libc, version = platform.libc_ver()
    try:
        glibc = tuple(int(value) for value in version.split('.')[:2])
    except ValueError:
        glibc = ()
    if libc != "glibc" or glibc < (2, 17):
        raise RuntimeError("Native VP8 requires glibc >= 2.17 for the pinned manylinux wheel")
    versions = {}
    for name, expected in SUPPORTED_PACKAGES.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Native VP8 dependency is missing: {name}=={expected}") from exc
        if versions[name] != expected:
            raise RuntimeError(f"Native VP8 requires validated {name}=={expected}; found {versions[name]}")
    return versions


def _load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Native VP8 refuses to replace already-loaded module {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load verified native VP8 artifact: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _native_encoder_class(upstream):
    class NativeVp8Encoder(upstream.Vp8Encoder):
        """Serialize native ownership and require exact outbound frame timing."""
        def __init__(self):
            self._native_lock = threading.RLock()
            self._closed = False
            self._native_context_creations = 0
            self._native_bitrate_reconfigurations = 0
            self._native_frames_encoded = 0
            super().__init__()

        def encode(self, frame, force_keyframe=False):
            with self._native_lock:
                if self._closed:
                    raise RuntimeError("Native VP8 encoder is closed")
                if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                    raise ValueError("Native VP8 requires a frame PTS and positive time base")
                if frame.duration <= 0:
                    raise ValueError("Native VP8 requires the outbound track's positive frame duration")
                original_pts, original_base = frame.pts, frame.time_base
                native_pts = int(Fraction(frame.pts) * frame.time_base / VIDEO_TIME_BASE)
                duration = round(Fraction(frame.duration) * frame.time_base / VIDEO_TIME_BASE)
                if duration <= 0:
                    raise ValueError("Native VP8 frame duration is shorter than one 90 kHz tick")
                self.timestamp_increment = duration
                # Upstream's native codec consumes raw frame.pts in its fixed
                # 90 kHz time base. Preserve the caller's metadata after encode.
                frame.pts, frame.time_base = native_pts, VIDEO_TIME_BASE
                # Keep the previous CFFI object alive across encode: a freshly
                # allocated context can reuse an address after a resize.
                previous_codec = self.codec
                previous_bitrate = self.cfg.rc_target_bitrate
                try:
                    result = super().encode(frame, force_keyframe)
                    if self.codec is not previous_codec:
                        self._native_context_creations += 1
                    elif self.cfg.rc_target_bitrate != previous_bitrate:
                        self._native_bitrate_reconfigurations += 1
                    self._native_frames_encoded += 1
                    return result
                finally:
                    frame.pts, frame.time_base = original_pts, original_base

        def pack(self, packet):
            with self._native_lock:
                if self._closed:
                    raise RuntimeError("Native VP8 encoder is closed")
                return super().pack(packet)

        def close(self):
            # A pending executor call owns a bound-method reference, so sender
            # teardown cannot finalize it mid-encode. The lock also makes an
            # explicit close wait until any active encode has returned.
            with self._native_lock:
                if self._closed:
                    return
                self._closed = True
                codec = getattr(self, "codec", None)
                self.codec = None
                if codec:
                    upstream._vpx_assert(upstream.lib.vpx_codec_destroy(codec))

        def __del__(self):
            try:
                self.close()
            except Exception:
                # Explicit close reports errors; interpreter finalization must
                # not invoke the upstream destructor a second time.
                pass

    NativeVp8Encoder.__name__ = "NativeVp8Encoder"
    NativeVp8Encoder.__qualname__ = "NativeVp8Encoder"
    return NativeVp8Encoder


def load_native_encoder(directory=DEFAULT_DIRECTORY):
    global _loaded_class, _loaded_directory
    validate_runtime()
    directory = Path(directory).expanduser().resolve()
    validate_install(directory)
    if _loaded_class is not None:
        if directory != _loaded_directory:
            raise RuntimeError("Native VP8 cannot change its artifact directory in a running process")
        return _loaded_class
    import aiortc.codecs  # Relative source imports use the installed 1.14 APIs.
    binding_path = directory / "aiortc/codecs/_vpx.abi3.so"
    _load_module("aiortc.codecs._vpx", binding_path)
    library = ctypes.CDLL(str(binding_path))
    library.vpx_codec_version_str.restype = ctypes.c_char_p
    if library.vpx_codec_version_str() != b"v1.13.1":
        raise RuntimeError("Native VP8 binding does not expose the pinned libvpx v1.13.1")
    upstream = _load_module("aiortc.codecs._musetalk_native_vpx_111",
                            directory / "aiortc/codecs/vpx.py")
    selected = _native_encoder_class(upstream)
    selected.native_directory = str(directory)
    selected.native_libvpx_version = "v1.13.1"
    _loaded_class, _loaded_directory = selected, directory
    return selected


def _probe_encoder(encoder_class):
    """Fail startup on an unusable ABI or in-place reconfiguration path."""
    import av
    encoder = encoder_class()
    try:
        frame = av.VideoFrame(64, 64, "yuv420p")
        for index, plane in enumerate(frame.planes):
            plane.update(bytes([40 if index == 0 else 128]) * plane.buffer_size)
        frame.pts, frame.time_base, frame.duration = 0, VIDEO_TIME_BASE, 4500
        first, timestamp = encoder.encode(frame)
        context = encoder.codec
        encoder.target_bitrate = 750000
        frame.pts = 4500
        second, next_timestamp = encoder.encode(frame)
        if (not first or not second or timestamp != 0 or next_timestamp != 4500
                or encoder.codec is not context or encoder.cfg.rc_target_bitrate != 750):
            raise RuntimeError("Native VP8 startup encode/reconfiguration probe failed")
    finally:
        encoder.close()


def configure_vp8_encoder(process_label="process"):
    """Apply only before media sessions start; explicit failures never fall back."""
    mode = os.environ.get("WEBRTC_VP8_ENCODER", "pyav").strip().lower()
    if mode == "pyav":
        codecs = sys.modules.get("aiortc.codecs")
        if _loaded_class is not None and getattr(codecs, "Vp8Encoder", None) is _loaded_class:
            raise RuntimeError("Cannot change native VP8 to PyAV in a running process")
        return {"encoder": "pyav", "opt_in": False}
    if mode != "native":
        raise RuntimeError(f"Unsupported WEBRTC_VP8_ENCODER={mode!r}; use pyav or native")
    directory = os.environ.get("WEBRTC_NATIVE_VP8_DIR") or DEFAULT_DIRECTORY
    encoder_class = load_native_encoder(directory)
    import aiortc.codecs as codecs
    from aiortc.codecs.vpx import Vp8Encoder as stock_encoder
    if codecs.Vp8Encoder not in (stock_encoder, encoder_class):
        raise RuntimeError("Native VP8 refuses to replace another encoder registry override")
    _probe_encoder(encoder_class)
    codecs.Vp8Encoder = encoder_class
    summary = {"encoder": "native", "opt_in": True,
               "directory": encoder_class.native_directory,
               "libvpx": encoder_class.native_libvpx_version,
               "packages": dict(SUPPORTED_PACKAGES)}
    print(f"[{process_label}] VP8 encoder=native libvpx=v1.13.1 "
          f"directory={encoder_class.native_directory} startup_probe=passed", flush=True)
    return summary
