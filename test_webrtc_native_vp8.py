"""Opt-in configuration gates and real native VP8 sender/ownership checks."""
import asyncio
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
from aiortc import RTCRtpSender, VideoStreamTrack
import aiortc.codecs as codecs
from aiortc.codecs.vpx import Vp8Decoder, vp8_depayload
from aiortc.jitterbuffer import JitterFrame
from aiortc.rtp import (RTCP_PSFB_APP, RTCP_PSFB_FIR, RTCP_PSFB_PLI,
                       RtcpPsfbPacket, pack_remb_fci)
from scripts import webrtc_native_vp8 as native
from scripts import webrtc_tracks


def frame(index=0, width=512, height=832, fps=20):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
    pixels[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
    pixels[:, :, 2] = 120
    value = av.VideoFrame.from_ndarray(pixels, format='rgb24').reformat(format='yuv420p')
    value.pts = round(index * 90000 / fps)
    value.time_base = Fraction(1, 90000)
    value.duration = round(90000 / fps)
    return value


class NativeConfigurationTest(unittest.TestCase):
    def test_default_does_not_load_artifacts_or_change_registry(self):
        original = codecs.Vp8Encoder
        with patch.dict(os.environ, {'WEBRTC_VP8_ENCODER': 'pyav'}), \
                patch.object(native, 'load_native_encoder') as loader:
            self.assertEqual(native.configure_vp8_encoder(), {'encoder': 'pyav', 'opt_in': False})
            loader.assert_not_called()
        self.assertIs(codecs.Vp8Encoder, original)

    def test_unknown_selection_fails_without_mutating_registry(self):
        original = codecs.Vp8Encoder
        with patch.dict(os.environ, {'WEBRTC_VP8_ENCODER': 'typo'}):
            with self.assertRaisesRegex(RuntimeError, 'Unsupported WEBRTC_VP8_ENCODER'):
                native.configure_vp8_encoder()
        self.assertIs(codecs.Vp8Encoder, original)

    def test_explicit_native_missing_install_fails_without_fallback(self):
        original = codecs.Vp8Encoder
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'WEBRTC_VP8_ENCODER': 'native', 'WEBRTC_NATIVE_VP8_DIR': directory}):
            with self.assertRaisesRegex(RuntimeError, 'file missing'):
                native.configure_vp8_encoder()
        self.assertIs(codecs.Vp8Encoder, original)

    def test_unsupported_platform_and_package_versions_fail_before_loading(self):
        with patch.object(native.platform, 'machine', return_value='aarch64'):
            with self.assertRaisesRegex(RuntimeError, 'Linux x86_64 CPython 3.10'):
                native.validate_runtime()
        versions = {**native.SUPPORTED_PACKAGES, 'av': '17.0.0'}
        with patch.object(native.importlib.metadata, 'version', side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, 'av==16.1.0; found 17.0.0'):
                native.validate_runtime()

    def test_exact_native_duration_is_opt_in_and_replaces_source_duration(self):
        value = frame(fps=24)
        with patch.object(webrtc_tracks, 'WEBRTC_NATIVE_VP8', False):
            webrtc_tracks._set_video_transport_metadata(value, 0, 20)
        self.assertEqual(value.duration, 3750)
        with patch.object(webrtc_tracks, 'WEBRTC_NATIVE_VP8', True):
            for fps, duration in ((20, 4500), (24, 3750), (30, 3000)):
                webrtc_tracks._set_video_transport_metadata(value, 0, fps)
                self.assertEqual(value.duration, duration)


@unittest.skipUnless((native.DEFAULT_DIRECTORY / 'installation.json').is_file(),
                     'Install the pinned optional native VP8 files to run native codec gates')
class RealNativeEncoderTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder_class = native.load_native_encoder()

    async def test_startup_probe_failure_does_not_select_native_and_success_is_idempotent(self):
        original = codecs.Vp8Encoder
        try:
            with patch.dict(os.environ, {'WEBRTC_VP8_ENCODER': 'native'}), \
                    patch.object(native, '_probe_encoder', side_effect=RuntimeError('bad ABI')):
                with self.assertRaisesRegex(RuntimeError, 'bad ABI'):
                    native.configure_vp8_encoder()
                self.assertIs(codecs.Vp8Encoder, original)
            with patch.dict(os.environ, {'WEBRTC_VP8_ENCODER': 'native'}):
                first = native.configure_vp8_encoder('test')
                second = native.configure_vp8_encoder('test')
                self.assertEqual(first, second)
                self.assertIs(codecs.Vp8Encoder, self.encoder_class)
        finally:
            codecs.Vp8Encoder = original

    async def test_real_sender_pli_fir_resize_remb_and_current_decoder(self):
        class Source(VideoStreamTrack):
            def __init__(self):
                super().__init__()
                self.index = 0
            async def recv(self):
                value = frame(self.index, width=480 if self.index in (6, 7) else 512)
                self.index += 1
                return value
        original = codecs.Vp8Encoder
        source = Source()
        encoder = None
        try:
            codecs.Vp8Encoder = self.encoder_class
            self.assertIsInstance(codecs.get_encoder(codecs.CODECS['audio'][0]), codecs.OpusEncoder)
            self.assertIsInstance(codecs.get_encoder(next(c for c in codecs.CODECS['video'] if c.name == 'H264')), codecs.H264Encoder)
            sender = RTCRtpSender(source, SimpleNamespace(state='new'))
            selected = next(c for c in codecs.CODECS['video'] if c.name == 'VP8')
            decoder = Vp8Decoder()
            tags, context = [], None
            for index in range(10):
                if index in (2, 4):
                    await sender._handle_rtcp_packet(RtcpPsfbPacket(
                        fmt=RTCP_PSFB_PLI if index == 2 else RTCP_PSFB_FIR,
                        ssrc=123, media_ssrc=sender._ssrc))
                if index == 3:
                    await sender._handle_rtcp_packet(RtcpPsfbPacket(
                        fmt=RTCP_PSFB_APP, ssrc=123, media_ssrc=sender._ssrc,
                        fci=pack_remb_fci(800000, [sender._ssrc])))
                encoded = await sender._next_encoded_frame(selected)
                encoder = sender._RTCRtpSender__encoder
                self.assertIsInstance(encoder, self.encoder_class)
                context = encoder.codec if context is None else context
                if index < 6:
                    self.assertIs(encoder.codec, context)
                if index >= 3:
                    self.assertEqual(encoder.cfg.rc_target_bitrate, 800)
                self.assertEqual(encoder._native_context_creations,
                                 1 if index < 6 else 2 if index < 8 else 3)
                self.assertEqual(encoder._native_bitrate_reconfigurations, int(index >= 3))
                self.assertEqual(encoder._native_frames_encoded, index + 1)
                raw = b''.join(vp8_depayload(packet) for packet in encoded.payloads)
                tags.append(raw[0] & 1)
                decoded = decoder.decode(JitterFrame(raw, encoded.timestamp))
                self.assertEqual(len(decoded), 1)
                self.assertEqual((decoded[0].width, decoded[0].height),
                                 (480 if index in (6, 7) else 512, 832))
                self.assertEqual(encoded.timestamp, index * 4500)
                self.assertEqual(encoder.timestamp_increment, 4500)
                self.assertLessEqual(max(map(len, encoded.payloads)), 1300)
            self.assertEqual(tags, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        finally:
            if encoder is not None:
                encoder.close()
            source.stop()
            codecs.Vp8Encoder = original

    async def test_timing_normalization_and_missing_duration_rejection(self):
        encoder = self.encoder_class()
        try:
            value = frame()
            value.duration = 0
            with self.assertRaisesRegex(ValueError, 'positive frame duration'):
                encoder.encode(value)
            self.assertIsNone(encoder.codec)
            value.pts, value.time_base, value.duration = 2, Fraction(1, 24), 1
            payloads, timestamp = encoder.encode(value)
            self.assertTrue(payloads)
            self.assertEqual(timestamp, 7500)
            self.assertEqual(encoder.timestamp_increment, 3750)
            self.assertEqual((value.pts, value.time_base, value.duration), (2, Fraction(1, 24), 1))
        finally:
            encoder.close()
        encoder.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            encoder.encode(value)

    async def test_close_waits_for_inflight_encode_and_remains_idempotent(self):
        from concurrent.futures import ThreadPoolExecutor
        encoder = self.encoder_class()
        base = self.encoder_class.__mro__[1]
        original_encode = base.encode
        entered, release, closing = threading.Event(), threading.Event(), threading.Event()
        def held_encode(instance, value, force_keyframe=False):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test did not release encode')
            return original_encode(instance, value, force_keyframe)
        def close():
            closing.set()
            encoder.close()
        try:
            with patch.object(base, 'encode', held_encode), ThreadPoolExecutor(2) as pool:
                encoding = pool.submit(encoder.encode, frame())
                self.assertTrue(entered.wait(1))
                stopping = pool.submit(close)
                self.assertTrue(closing.wait(1))
                self.assertFalse(stopping.done())
                release.set()
                self.assertTrue(encoding.result(timeout=2)[0])
                stopping.result(timeout=2)
            self.assertIsNone(encoder.codec)
            encoder.close()
        finally:
            release.set()
            encoder.close()

    async def test_native_workers_close_with_retained_encoders_and_gc_disabled(self):
        script = '''
import gc, json
from scripts.webrtc_native_vp8 import load_native_encoder
from test_webrtc_native_vp8 import frame
from test_idle_video_decoder_lifecycle import _native_task_ids, _wait_for_task_set
Encoder = load_native_encoder()
gc.disable()
baseline = _native_task_ids()
retained, observations = [], []
for cycle in range(30):
    encoder = Encoder()
    encoder.encode(frame())
    assert len(_native_task_ids()) >= len(baseline) + 1
    encoder.close()
    encoder.close()
    retained.append(encoder)
    assert encoder.codec is None and encoder._closed
    observations.append(_wait_for_task_set(baseline))
assert not gc.isenabled()
print(json.dumps({'cycles': len(retained), 'baseline': len(baseline),
                  'final': len(_native_task_ids()), 'gc_enabled': gc.isenabled()}))
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=20,
            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        proof = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(proof['cycles'], 30)
        self.assertEqual(proof['baseline'], proof['final'])
        self.assertFalse(proof['gc_enabled'])


if __name__ == '__main__':
    unittest.main()
