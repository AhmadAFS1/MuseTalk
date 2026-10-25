#!/usr/bin/env python3
"""Glitch-injection tests for qc/video_qc.py.

Takes one accepted reference clip, injects a specific defect, and asserts that the
named check fails while a clean re-encode of the same clip passes. Needs numpy,
OpenCV and ffmpeg (run with the MuseTalk server venv):

  /workspace/.venvs/musetalk_trt_stagewise/bin/python -m unittest tests/test_video_qc.py -v

Set QC_REFERENCE=/path/to/talking.mp4 and QC_OTHER=/path/to/other_identity.mp4 to
use different sources; the defaults are the saved reference deliveries.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QC = HERE.parent / 'qc'
import sys  # noqa: E402
sys.path.insert(0, str(QC))
import video_qc  # noqa: E402

REFERENCE = Path(os.environ.get('QC_REFERENCE', '/workspace/minimax-h3-workflows/references/chinese-bob-talking/video.mp4'))
OTHER = Path(os.environ.get('QC_OTHER', '/workspace/minimax-h3-workflows/references/accepted-japanese/video.mp4'))
ANCHOR = Path(os.environ.get('QC_ANCHOR', '/workspace/minimax-h3-workflows/assets/chinese-bob-talking.png'))


def decode(path: Path) -> np.ndarray:
    probe = video_qc.ffprobe(path)
    v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    w, h = int(v['width']), int(v['height'])
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()


def encode(frames: np.ndarray, dest: Path, fps: int = 24, audio_from: Path | None = None) -> None:
    """Intra-only fixed-QP H.264 so identical input frames decode identically (first == last)."""
    n, h, w, _ = frames.shape
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    if audio_from is not None:
        cmd += ['-i', str(audio_from), '-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'aac', '-shortest']
    else:
        cmd += ['-an']
    cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '1', '-qp', '12', '-x264-params', 'scenecut=0',
            '-movflags', '+faststart', str(dest)]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode()[-800:])


class GlitchDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REFERENCE.is_file() or not OTHER.is_file():
            raise unittest.SkipTest(f'reference clips missing: {REFERENCE} / {OTHER}')
        cls.tmp = Path(tempfile.mkdtemp(prefix='video_qc_test_'))
        cls.frames = decode(REFERENCE)
        cls.other = decode(OTHER)
        cls.profiles = video_qc.load_profiles(QC / 'profiles.json')
        cls.reports = {}

    def run_qc(self, name: str, frames: np.ndarray, pose: str = 'talking', audio: bool = True, expected_frames=None):
        dest = self.tmp / f'{name}.mp4'
        encode(frames, dest, audio_from=REFERENCE if audio else None)
        thresholds = video_qc.resolve_thresholds(self.profiles, 'strict', pose)
        expected = expected_frames if expected_frames is not None else video_qc.EXPECTED_FRAMES[pose]
        report = video_qc.analyze_video(dest, pose, thresholds, ANCHOR if ANCHOR.is_file() else None, expected,
                                        'auto', self.tmp / f'{name}.jpg', keep_series=False)
        (self.tmp / f'{name}.json').write_text(json.dumps(report, indent=1))
        self.reports[name] = report
        return report

    def failed(self, report: dict) -> set:
        return {c['name'] for c in report['checks'] if not c['pass']}

    def test_clean_reencode_passes(self):
        frames = self.frames.copy()
        frames[-1] = frames[0]
        report = self.run_qc('clean', frames)
        self.assertEqual(report['hard_failures'], [], report['hard_failures'])
        self.assertTrue(report['endpoint_proof']['first_last_exact'])

    def test_frozen_run_detected(self):
        frames = self.frames.copy()
        frames[100:112] = frames[99]
        report = self.run_qc('frozen', frames)
        self.assertIn('frozen.runs', report['hard_failures'])

    def test_black_frame_detected(self):
        frames = self.frames.copy()
        frames[120] = 0
        report = self.run_qc('black', frames)
        self.assertIn('blank.frames', report['hard_failures'])
        self.assertTrue({'jump.frames', 'luma.steps'} & set(report['hard_failures']))

    def test_flash_detected(self):
        frames = self.frames.copy()
        frames[130] = np.clip(frames[130].astype(np.int16) + 70, 0, 255).astype(np.uint8)
        report = self.run_qc('flash', frames)
        self.assertIn('luma.steps', report['hard_failures'])

    def test_identity_jump_detected(self):
        frames = self.frames.copy()
        n = min(len(frames), len(self.other))
        frames[110:134] = self.other[110:134] if n >= 134 else self.other[:24]
        report = self.run_qc('jump', frames)
        self.assertTrue({'jump.frames', 'face.consecutive_jump'} & set(report['hard_failures']), report['hard_failures'])

    def test_face_loss_detected(self):
        frames = self.frames.copy()
        box = video_qc.HaarFaces().detect(frames[0])[0]
        self.assertIsNotNone(box, 'reference frame 0 must contain a detectable face')
        x, y, w, h = box
        pad = int(0.35 * h)
        y0, y1 = max(0, y - pad), min(frames.shape[1], y + h + pad)
        x0, x1 = max(0, x - pad), min(frames.shape[2], x + w + pad)
        fill = frames[0, 5:25, 5:25].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        frames[140:152, y0:y1, x0:x1] = fill
        report = self.run_qc('faceloss', frames)
        self.assertTrue({'face.max_missing_run', 'face.detection_fraction'} & set(report['hard_failures']), report['hard_failures'])

    def test_missing_audio_detected(self):
        report = self.run_qc('noaudio', self.frames, audio=False)
        self.assertIn('audio.presence', report['hard_failures'])

    def test_unexpected_audio_on_idle_detected(self):
        report = self.run_qc('idle_with_audio', self.frames, pose='idle', audio=True)
        self.assertIn('audio.presence', report['hard_failures'])

    def test_wrong_frame_count_detected(self):
        report = self.run_qc('short', self.frames[:-5])
        self.assertIn('contract.frames', report['hard_failures'])

    def test_broken_loop_detected(self):
        frames = self.frames.copy()
        frames[-1] = frames[200]
        report = self.run_qc('loop', frames)
        self.assertIn('endpoints.first_last_exact', report['hard_failures'])

    def test_wrong_anchor_detected(self):
        frames = self.frames.copy()
        frames[-1] = frames[0]
        dest = self.tmp / 'wrong_anchor.mp4'
        encode(frames, dest, audio_from=REFERENCE)
        other_anchor = Path('/workspace/minimax-h3-workflows/assets/accepted-japanese.png')
        if not other_anchor.is_file():
            self.skipTest('other anchor missing')
        thresholds = video_qc.resolve_thresholds(self.profiles, 'strict', 'talking')
        report = video_qc.analyze_video(dest, 'talking', thresholds, other_anchor, 240, 'auto', None, keep_series=False)
        self.assertIn('anchor.first_frame_ssim', report['hard_failures'])

    def test_endpoint_set_mismatch(self):
        a = self.frames.copy(); a[-1] = a[0]
        b = self.other[:len(a)].copy(); b[-1] = b[0]
        pa, pb = self.tmp / 'set_a.mp4', self.tmp / 'set_b.mp4'
        encode(a, pa); encode(b, pb)
        result = video_qc.endpoint_set([pa, pb])
        self.assertFalse(result['pass'])
        same = video_qc.endpoint_set([pa, pa])
        self.assertTrue(same['pass'])

    @classmethod
    def tearDownClass(cls):
        summary = {k: {'pass': v['pass'], 'hard': v['hard_failures'], 'soft': v['warnings']} for k, v in cls.reports.items()}
        (cls.tmp / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('\nglitch test artifacts:', cls.tmp)


if __name__ == '__main__':
    unittest.main()
