"""Actual decoded PCM proof: only complete exact silence may bypass mouth synthesis."""
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.webrtc_exact_silence import is_exact_silence


def write_wav(path, values, sample_rate=24000):
    values = np.asarray(values)
    channels = values.shape[1] if values.ndim == 2 else 1
    bits = values.dtype.itemsize * 8
    code = 3 if values.dtype.kind == 'f' else 1
    payload = values.astype(values.dtype.newbyteorder('<')).tobytes()
    fmt = struct.pack('<HHIIHH', code, channels, sample_rate,
                      sample_rate * channels * bits // 8, channels * bits // 8, bits)
    body = b'WAVEfmt ' + struct.pack('<I', len(fmt)) + fmt
    body += b'data' + struct.pack('<I', len(payload)) + payload
    path.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)
    return path


class ExactSilenceDetectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_nonempty_pcm_zero_all_channels_and_unsigned_zero(self):
        cases = [('mono', np.zeros(4800, np.int16)),
                 ('stereo', np.zeros((4800, 2), np.float32)),
                 ('unsigned', np.full(4800, 128, np.uint8)),
                 ('float64', np.zeros(4800, np.float64))]
        for name, samples in cases:
            with self.subTest(name=name):
                self.assertTrue(is_exact_silence(write_wav(self.root/(name+'.wav'), samples)))

    def test_one_nonzero_sample_even_at_end_never_bypasses(self):
        for dtype, value in ((np.int16, 1), (np.float32, 1e-20),
                             (np.float32, np.nextafter(np.float32(0), np.float32(1))),
                             (np.float64, 1e-200)):
            with self.subTest(dtype=dtype, value=value):
                samples = np.zeros((9000, 2), dtype=dtype)
                samples[-1, 1] = value
                self.assertFalse(is_exact_silence(write_wav(self.root/'nonzero.wav', samples)))

    def test_antiphase_channels_do_not_cancel_into_silence(self):
        samples = np.tile(np.array([1, -1], np.int16), (4800, 1))
        self.assertFalse(is_exact_silence(write_wav(self.root/'stereo.wav', samples)))

    def test_empty_invalid_and_missing_files_do_not_bypass(self):
        empty = write_wav(self.root/'empty.wav', np.zeros(0, np.int16))
        invalid = self.root/'invalid.wav'; invalid.write_bytes(b'not audio')
        for path in (empty, invalid, self.root/'missing.wav'):
            with self.subTest(path=path):
                self.assertFalse(is_exact_silence(path))

    def test_cancelled_inspection_never_claims_positive_detection(self):
        event = threading.Event(); event.set()
        with patch('av.open', side_effect=AssertionError('Cancelled detector opened media')):
            self.assertFalse(is_exact_silence(self.root/'ignored.wav', cancel_event=event))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for stream-selection check')
    def test_multiple_audio_streams_cannot_hide_nonzero_default_selection(self):
        zero = write_wav(self.root/'mono.wav', np.zeros(4800, np.int16))
        nonzero = write_wav(self.root/'stereo.wav', np.ones((4800, 2), np.int16))
        mixed = self.root/'two-streams.mka'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
                        '-i',str(zero),'-i',str(nonzero),'-map','0:a','-map','1:a',
                        '-c:a','pcm_s16le',str(mixed)],check=True,capture_output=True)
        self.assertFalse(is_exact_silence(mixed))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for lossy-container check')
    def test_mp3_zero_decode_and_pcm_normalization_remain_exact_zero(self):
        wav = write_wav(self.root/'source.wav', np.zeros(4800, np.int16))
        mp3 = self.root/'silent.mp3'; normalized = self.root/'normalized.wav'
        for source, target in ((wav, mp3), (mp3, normalized)):
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(source), str(target)], check=True, capture_output=True)
        self.assertTrue(is_exact_silence(mp3))
        self.assertTrue(is_exact_silence(normalized))


if __name__ == '__main__':
    unittest.main()
