#!/usr/bin/env python3
"""Technical and glitch QC for H3 pose clips and MuseTalk smoke renders.

Runs in any interpreter that has numpy and OpenCV (the MuseTalk server venv). If
`mediapipe` is importable (the `--with-chin-tools` venv), FaceMesh landmarks add
mouth-gap and eye checks; otherwise a Haar cascade supplies face boxes and a
mouth-region motion proxy stands in for the landmark checks, and the report says so.

Checks (see qc/profiles.json for thresholds, per pose):
  contract   geometry, fps, frame count, duration, full decode (ffmpeg -xerror)
  endpoints  first == last decoded RGB frame (the shared-anchor loop contract)
  frozen     runs of near-identical consecutive interior frames
  jump       single-frame content jumps far above the clip's own median motion
  luma_step  global brightness steps (flashes)
  blank      black / white / flat frames
  flicker    high-frequency alternation of global brightness
  face       a face in (almost) every frame, stable centre and size, no teleporting
  background a region outside the face that drifts away from frame 0 (scene morphing)
  anchor     SSIM of frame 0 against the conditioning anchor and SSIM of mid-clip frames
  mouth      closed for idle/smiling, moving for talking/smoke
  audio      present/absent as the pose requires; level and clipping when present

Exit codes: 0 pass, 2 hard failure, 1 tool error. Technical passes are not visual
approval; the report records values so thresholds can be tuned without code changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - the venv check in the installer guards this
    raise SystemExit(f'video_qc needs OpenCV (cv2): {exc}')

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILES = HERE / 'profiles.json'
EXPECTED_FRAMES = {'idle': 240, 'talking': 240, 'smiling': 158, 'smoke': None}
OPEN_GAP_RATIO = 0.03          # lip gap / inter-ocular distance that counts as "open"
MOUTH_ROI = (0.22, 0.78, 0.58, 0.97)   # x0, x1, y0, y1 fractions of the face box
CONTACT_TILE_WIDTH = 192


# ----------------------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_type,codec_name,nb_read_frames,avg_frame_rate,r_frame_rate,width,height,duration,pix_fmt,sample_rate,channels',
        '-show_entries', 'format=duration', '-of', 'json', str(path)], text=True)
    return json.loads(out)


def full_decode_ok(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                          capture_output=True, text=True)
    return proc.returncode == 0, proc.stderr.strip()[-800:]


def load_profiles(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_thresholds(profiles: dict, profile: str, pose: str) -> dict:
    if profile not in profiles:
        raise ValueError(f'Unknown QC profile {profile!r}; available: {sorted(k for k in profiles if not k.startswith("_"))}')
    block = profiles[profile]
    merged = json.loads(json.dumps(block.get('_default', {})))
    for key, value in block.get(pose, {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (s_a + s_b + c2)
    return float((num / den).mean())


def small_gray(frame_rgb: np.ndarray, width: int = 256) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = width / w
    return cv2.resize(gray, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


class FrameSource:
    """Stream rgb24 frames from ffmpeg so a 10 s 512x896 clip never sits whole in memory."""

    def __init__(self, path: Path, width: int, height: int):
        self.path, self.width, self.height = path, width, height
        self.frame_bytes = width * height * 3
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'rawvideo',
                                      '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()

    def __iter__(self):
        assert self.proc.stdout is not None
        while True:
            data = self.proc.stdout.read(self.frame_bytes)
            if not data:
                break
            if len(data) != self.frame_bytes:
                self.proc.kill()
                raise RuntimeError('Partial decoded frame')
            yield data, np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)
        err = self.proc.stderr.read().decode(errors='replace') if self.proc.stderr else ''
        code = self.proc.wait()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()
        if code:
            raise RuntimeError(f'ffmpeg decode failed: {err[-800:]}')


# ----------------------------------------------------------------------------- face backends

class HaarFaces:
    name = 'haar'

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.cascade.empty():
            raise RuntimeError('Haar cascade not found in this OpenCV build')
        self.previous = None

    def detect(self, frame_rgb: np.ndarray):
        """Return (box, landmarks) with box = (x, y, w, h) in frame pixels, landmarks None."""
        h, w = frame_rgb.shape[:2]
        gray = small_gray(frame_rgb, 320)
        scale = w / gray.shape[1]
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36))
        if len(faces) == 0:
            return None, None
        boxes = [(int(x * scale), int(y * scale), int(fw * scale), int(fh * scale)) for x, y, fw, fh in faces]
        if self.previous is not None and len(boxes) > 1:
            px, py = self.previous[0] + self.previous[2] / 2, self.previous[1] + self.previous[3] / 2
            boxes.sort(key=lambda b: (b[0] + b[2] / 2 - px) ** 2 + (b[1] + b[3] / 2 - py) ** 2)
        else:
            boxes.sort(key=lambda b: -b[2] * b[3])
        self.previous = boxes[0]
        return boxes[0], None

    def close(self):
        pass


class MediaPipeFaces:
    name = 'mediapipe'

    def __init__(self):
        import mediapipe as mp  # noqa: F401 - import checked by caller
        self.mp = mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.fallback = HaarFaces()

    def detect(self, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        result = self.mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            box, _ = self.fallback.detect(frame_rgb)
            return box, None
        pts = np.array([(p.x * w, p.y * h) for p in result.multi_face_landmarks[0].landmark], np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), pts

    def close(self):
        self.mesh.close()


def make_face_backend(prefer: str = 'auto'):
    if prefer in ('auto', 'mediapipe'):
        try:
            import mediapipe  # noqa: F401
            return MediaPipeFaces()
        except Exception:
            if prefer == 'mediapipe':
                raise
    return HaarFaces()


# ----------------------------------------------------------------------------- analysis

def analyze_video(video: Path, pose: str, thresholds: dict, anchor: Path | None, expected_frames: int | None,
                  face_backend: str = 'auto', contact: Path | None = None, keep_series: bool = True) -> dict:
    started = time.monotonic()
    checks: list[dict] = []
    report: dict = {'video': str(video), 'sha256': sha256_file(video), 'pose': pose, 'checks': checks}

    def add(name, severity, passed, value=None, threshold=None, detail=None):
        checks.append({'name': name, 'severity': severity, 'pass': bool(passed), 'value': value,
                       'threshold': threshold, 'detail': detail})

    # ---- container / contract
    probe = ffprobe(video)
    streams = probe.get('streams', [])
    vstreams = [s for s in streams if s.get('codec_type') == 'video']
    astreams = [s for s in streams if s.get('codec_type') == 'audio']
    if len(vstreams) != 1:
        add('contract.video_stream', 'hard', False, len(vstreams), 1, 'expected exactly one video stream')
        report['pass'] = False
        return finish(report, started)
    v = vstreams[0]
    width, height = int(v['width']), int(v['height'])
    frames = int(v.get('nb_read_frames') or 0)
    fps_text = v.get('avg_frame_rate') or v.get('r_frame_rate')
    num, den = (fps_text.split('/') + ['1'])[:2]
    fps = float(num) / float(den or 1)
    duration = float(v.get('duration') or probe.get('format', {}).get('duration') or 0)
    report.update(width=width, height=height, frames=frames, fps=fps_text, duration=duration,
                  audio_streams=len(astreams))
    contract = thresholds.get('contract', {})
    if contract.get('width') is not None:
        add('contract.geometry', 'hard', (width, height) == (contract['width'], contract['height']),
            [width, height], [contract['width'], contract['height']])
    if contract.get('fps') is not None:
        add('contract.fps', 'hard', fps_text == contract['fps'], fps_text, contract['fps'])
    if expected_frames is not None:
        add('contract.frames', 'hard', frames == expected_frames, frames, expected_frames)
        tol = contract.get('duration_tolerance_s')
        if tol is not None and fps > 0:
            add('contract.duration', 'hard', abs(duration - expected_frames / fps) <= tol,
                round(duration, 4), round(expected_frames / fps, 4), f'tolerance {tol}s')
    ok, err = full_decode_ok(video)
    add('contract.full_decode', 'hard', ok, None, None, err or None)
    if not ok or frames < 2:
        report['pass'] = False
        return finish(report, started)

    # ---- streaming frame analysis
    faces = make_face_backend(face_backend)
    report['face_backend'] = faces.name
    anchor_small = None
    if anchor is not None and anchor.is_file():
        anchor_img = cv2.cvtColor(cv2.imread(str(anchor)), cv2.COLOR_BGR2RGB)
        if anchor_img.shape[:2] != (height, width):
            anchor_img = cv2.resize(anchor_img, (width, height), interpolation=cv2.INTER_AREA)
        anchor_small = small_gray(anchor_img)
    handles = int(thresholds.get('handles', 0) or 0)
    sample_at = sorted({0, frames - 1, *[int(round(k * (frames - 1) / 7)) for k in range(1, 7)]})
    ssim_at = sorted({0, frames // 4, frames // 2, (3 * frames) // 4, frames - 1})

    mean_luma, std_luma, diffs = [], [], []
    boxes: list = []
    mouth_gap_ratio: list = []      # None when no landmarks
    eye_open: list = []
    mouth_energy: list = []
    face_energy: list = []
    background_diff: list = []
    anchor_ssim: dict = {}
    tiles: dict = {}
    first_bytes = last_bytes = None
    first_gray = prev_gray = None
    prev_mouth = None
    prev_face = None
    first_face_box = None

    for index, (raw, frame) in enumerate(FrameSource(video, width, height)):
        if index == 0:
            first_bytes = raw
        last_bytes = raw
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        g32 = gray.astype(np.float32)
        mean_luma.append(float(g32.mean()))
        std_luma.append(float(g32.std()))
        if prev_gray is not None:
            diffs.append(float(np.abs(g32 - prev_gray).mean()))
        box, pts = faces.detect(frame)
        boxes.append(box)
        if first_face_box is None and box is not None:
            first_face_box = box
        if pts is not None:
            eye_distance = float(np.linalg.norm(pts[33] - pts[263])) or 1.0
            mouth_gap_ratio.append(float(abs(pts[13][1] - pts[14][1]) / eye_distance))
            eye_open.append(float((abs(pts[159][1] - pts[145][1]) + abs(pts[386][1] - pts[374][1])) / 2 / eye_distance))
        else:
            mouth_gap_ratio.append(None)
            eye_open.append(None)
        # mouth-region motion proxy (works with either backend)
        ref_box = box or first_face_box
        if ref_box is not None:
            x, y, bw, bh = ref_box
            mx0, mx1 = int(x + MOUTH_ROI[0] * bw), int(x + MOUTH_ROI[1] * bw)
            my0, my1 = int(y + MOUTH_ROI[2] * bh), int(y + MOUTH_ROI[3] * bh)
            mx0, my0 = max(0, mx0), max(0, my0)
            mx1, my1 = min(width, max(mx0 + 1, mx1)), min(height, max(my0 + 1, my1))
            roi = g32[my0:my1, mx0:mx1]
            fx0, fy0 = max(0, x), max(0, y)
            fx1, fy1 = min(width, max(fx0 + 1, x + bw)), min(height, max(fy0 + 1, y + bh))
            face_roi = g32[fy0:fy1, fx0:fx1]
            if prev_mouth is not None and prev_mouth.shape == roi.shape and prev_face is not None and prev_face.shape == face_roi.shape:
                mouth_energy.append(float(np.abs(roi - prev_mouth).mean()))
                face_energy.append(float(np.abs(face_roi - prev_face).mean()))
            prev_mouth, prev_face = roi, face_roi
        if first_gray is None:
            first_gray = g32
            first_small = small_gray(frame)
        if first_face_box is not None:
            # background = everything outside a generously expanded first face box
            x, y, bw, bh = first_face_box
            cx, cy = x + bw / 2, y + bh / 2
            ex0, ex1 = int(max(0, cx - 1.1 * bw)), int(min(width, cx + 1.1 * bw))
            ey0, ey1 = int(max(0, cy - 1.0 * bh)), int(min(height, cy + 1.6 * bh))
            mask = np.ones((height, width), bool)
            mask[ey0:ey1, ex0:ex1] = False
            if mask.any():
                background_diff.append(float(np.abs(g32 - first_gray)[mask].mean()))
        if anchor_small is not None and index in ssim_at:
            anchor_ssim[str(index)] = round(ssim(small_gray(frame), anchor_small), 4)
        if index in sample_at:
            tiles[index] = frame.copy()
        prev_gray = g32
    faces.close()
    decoded = len(mean_luma)
    report['decoded_frames'] = decoded
    if decoded != frames:
        add('contract.decoded_frames', 'hard', False, decoded, frames, 'ffprobe and decode disagree')

    # ---- endpoints
    first_sha = hashlib.sha256(first_bytes).hexdigest()
    last_sha = hashlib.sha256(last_bytes).hexdigest()
    report['endpoint_proof'] = {'first_rgb_sha256': first_sha, 'last_rgb_sha256': last_sha,
                                'first_last_exact': first_bytes == last_bytes}
    if handles > 0:
        add('endpoints.first_last_exact', 'hard', first_bytes == last_bytes, first_sha[:16], last_sha[:16])

    lo, hi = handles, max(handles + 1, decoded - handles)      # interior frame window [lo, hi)
    d = np.asarray(diffs, np.float64)                           # d[i] is frame i+1 vs frame i
    interior_d = d[max(0, lo - 1):hi - 1] if d.size else d
    median_d = float(np.median(interior_d)) if interior_d.size else 0.0
    report['motion'] = {'median_frame_diff': round(median_d, 4),
                        'max_frame_diff': round(float(interior_d.max()), 4) if interior_d.size else None,
                        'p95_frame_diff': round(float(np.percentile(interior_d, 95)), 4) if interior_d.size else None}

    # ---- frozen runs
    fz = thresholds.get('frozen', {})
    frozen_runs, run, run_start = [], 0, None
    for i, value in enumerate(interior_d):
        if value < fz.get('diff_floor', 0.35):
            if run == 0:
                run_start = i + max(0, lo - 1)
            run += 1
        else:
            if run >= fz.get('max_run_frames', 5):
                frozen_runs.append([run_start, run])
            run = 0
    if run >= fz.get('max_run_frames', 5):
        frozen_runs.append([run_start, run])
    add('frozen.runs', fz.get('severity', 'hard'), not frozen_runs, frozen_runs,
        {'diff_floor': fz.get('diff_floor'), 'max_run_frames': fz.get('max_run_frames')},
        'interior runs of near-identical consecutive frames [start_frame, length]')

    # ---- jumps
    jp = thresholds.get('jump', {})
    jump_limit = max(jp.get('ratio_to_median', 6.0) * median_d, jp.get('abs_floor', 14.0))
    jumps = [[int(i + max(0, lo - 1) + 1), round(float(v), 3)] for i, v in enumerate(interior_d) if v > jump_limit]
    add('jump.frames', jp.get('severity', 'hard'), not jumps, jumps, round(jump_limit, 3),
        'frames whose change from the previous frame exceeds the limit [frame, mean_abs_diff]')

    # ---- luma steps, blank frames, flicker
    ml = np.asarray(mean_luma, np.float64)
    sl = np.asarray(std_luma, np.float64)
    ls = thresholds.get('luma_step', {})
    steps = np.abs(np.diff(ml[lo:hi])) if hi - lo > 1 else np.zeros(0)
    big_steps = [[int(i + lo + 1), round(float(v), 2)] for i, v in enumerate(steps) if v > ls.get('max_step', 12.0)]
    add('luma.steps', ls.get('severity', 'hard'), not big_steps, big_steps, ls.get('max_step'),
        'global brightness steps between consecutive frames [frame, delta]')
    bl = thresholds.get('blank', {})
    blank = [int(i) for i in range(lo, hi)
             if ml[i] < bl.get('min_mean_luma', 10) or ml[i] > bl.get('max_mean_luma', 245) or sl[i] < bl.get('min_std', 4)]
    add('blank.frames', bl.get('severity', 'hard'), not blank, blank,
        {k: bl.get(k) for k in ('min_mean_luma', 'max_mean_luma', 'min_std')}, 'black, white or flat frames')
    fl = thresholds.get('flicker', {})
    if steps.size > 4:
        signed = np.diff(ml[lo:hi])
        amp = fl.get('min_amplitude', 1.5)
        flips = sum(1 for a, b in zip(signed[:-1], signed[1:]) if abs(a) > amp and abs(b) > amp and (a > 0) != (b > 0))
        flip_fraction = flips / max(1, len(signed) - 1)
    else:
        flip_fraction = 0.0
    add('flicker.sign_flip_fraction', fl.get('severity', 'soft'), flip_fraction <= fl.get('max_sign_flip_fraction', 0.45),
        round(flip_fraction, 4), fl.get('max_sign_flip_fraction'))

    # ---- face
    fc = thresholds.get('face', {})
    detected = [b for b in boxes if b is not None]
    det_fraction = len(detected) / max(1, decoded)
    add('face.detection_fraction', fc.get('severity', 'hard'), det_fraction >= fc.get('min_detection_fraction', 0.985),
        round(det_fraction, 4), fc.get('min_detection_fraction'), f'backend={faces.name}')
    missing_run = best = 0
    for b in boxes:
        missing_run = missing_run + 1 if b is None else 0
        best = max(best, missing_run)
    add('face.max_missing_run', fc.get('severity', 'hard'), best <= fc.get('max_missing_run', 6), best, fc.get('max_missing_run'))
    face_stats = {}
    if len(detected) >= 2:
        centers = np.array([(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in detected], np.float64)
        sizes = np.array([b[3] for b in detected], np.float64)
        med_center = np.median(centers, axis=0)
        med_size = float(np.median(sizes)) or 1.0
        center_drift = float(np.linalg.norm(centers - med_center, axis=1).max() / height)
        size_drift = float(np.abs(sizes - med_size).max() / med_size)
        # consecutive jump: only between adjacent detected frames
        pairs = [(i, i + 1) for i in range(decoded - 1) if boxes[i] is not None and boxes[i + 1] is not None]
        jumps_px = [math.dist((boxes[i][0] + boxes[i][2] / 2, boxes[i][1] + boxes[i][3] / 2),
                              (boxes[j][0] + boxes[j][2] / 2, boxes[j][1] + boxes[j][3] / 2)) for i, j in pairs]
        max_jump = float(max(jumps_px) / height) if jumps_px else 0.0
        face_stats = {'median_center': [round(float(med_center[0]), 1), round(float(med_center[1]), 1)],
                      'median_height_px': round(med_size, 1), 'center_drift_fraction': round(center_drift, 4),
                      'size_drift_fraction': round(size_drift, 4), 'max_consecutive_jump_fraction': round(max_jump, 4)}
        add('face.center_drift', fc.get('severity', 'hard'), center_drift <= fc.get('max_center_drift_fraction', 0.12),
            round(center_drift, 4), fc.get('max_center_drift_fraction'), 'max distance of face centre from its median, / frame height')
        add('face.size_drift', fc.get('severity', 'hard'), size_drift <= fc.get('max_size_drift_fraction', 0.35),
            round(size_drift, 4), fc.get('max_size_drift_fraction'), 'max |face height - median| / median')
        add('face.consecutive_jump', fc.get('severity', 'hard'), max_jump <= fc.get('max_consecutive_jump_fraction', 0.08),
            round(max_jump, 4), fc.get('max_consecutive_jump_fraction'), 'largest centre move between adjacent frames, / frame height')
    report['face'] = face_stats

    # ---- background drift
    bg = thresholds.get('background', {})
    if background_diff:
        bgd = np.asarray(background_diff[lo:hi] or background_diff)
        add('background.max_mean_abs_diff_vs_first', bg.get('severity', 'soft'), float(bgd.max()) <= bg.get('max_mean_abs_diff_vs_first', 26.0),
            round(float(bgd.max()), 3), bg.get('max_mean_abs_diff_vs_first'), 'outside the expanded first-frame face box')

    # ---- anchor similarity
    an = thresholds.get('anchor', {})
    if anchor_ssim:
        report['anchor_ssim'] = anchor_ssim
        first_ssim = anchor_ssim.get('0')
        mids = [v for k, v in anchor_ssim.items() if k not in ('0', str(decoded - 1))]
        add('anchor.first_frame_ssim', an.get('severity', 'hard'), first_ssim is None or first_ssim >= an.get('min_ssim_first_frame', 0.97),
            first_ssim, an.get('min_ssim_first_frame'), 'frame 0 should be the anchor itself after packaging')
        if mids:
            add('anchor.mid_frames_min_ssim', an.get('severity', 'hard'), min(mids) >= an.get('min_ssim_mid_frames', 0.35),
                round(min(mids), 4), an.get('min_ssim_mid_frames'), 'identity/scene stability proxy, not a face-recognition score')

    # ---- mouth
    mo = thresholds.get('mouth', {})
    gaps = [g for g in mouth_gap_ratio[lo:hi] if g is not None]
    energy = np.asarray(mouth_energy[max(0, lo - 1):hi - 1] or mouth_energy, np.float64)
    f_energy = np.asarray(face_energy[max(0, lo - 1):hi - 1] or face_energy, np.float64)
    ratio = energy / np.maximum(f_energy, 0.2) if f_energy.size == energy.size and energy.size else np.zeros(0)
    mouth_stats = {'backend': 'landmarks' if gaps else ('motion_proxy' if energy.size else 'none')}
    if gaps:
        g = np.asarray(gaps)
        open_fraction = float((g > OPEN_GAP_RATIO).mean())
        mouth_stats.update(open_fraction=round(open_fraction, 4), p95_gap_ratio=round(float(np.percentile(g, 95)), 4),
                           gap_std_ratio=round(float(g.std()), 4))
    if energy.size:
        mouth_stats.update(motion_energy_mean=round(float(energy.mean()), 4), motion_energy_p95=round(float(np.percentile(energy, 95)), 4),
                           face_energy_mean=round(float(f_energy.mean()), 4) if f_energy.size else None)
    if ratio.size:
        mouth_stats.update(motion_ratio_mean=round(float(ratio.mean()), 4), motion_ratio_p95=round(float(np.percentile(ratio, 95)), 4))
    mode = mo.get('mode')
    sev = mo.get('severity', 'soft')
    if mode == 'closed':
        if gaps:
            add('mouth.closed.open_fraction', sev, mouth_stats['open_fraction'] <= mo.get('max_open_fraction', 0.15),
                mouth_stats['open_fraction'], mo.get('max_open_fraction'))
            add('mouth.closed.p95_gap_ratio', sev, mouth_stats['p95_gap_ratio'] <= mo.get('max_p95_gap_ratio', 0.045),
                mouth_stats['p95_gap_ratio'], mo.get('max_p95_gap_ratio'))
        if energy.size and mo.get('max_motion_energy') is not None:
            add('mouth.closed.motion_energy', sev, mouth_stats['motion_energy_mean'] <= mo['max_motion_energy'],
                mouth_stats['motion_energy_mean'], mo['max_motion_energy'], 'mean abs change in the mouth region per frame')
        if ratio.size and mo.get('max_motion_ratio') is not None:
            add('mouth.closed.motion_ratio', sev, mouth_stats['motion_ratio_mean'] <= mo['max_motion_ratio'],
                mouth_stats['motion_ratio_mean'], mo['max_motion_ratio'], 'mouth-region motion relative to whole-face motion; head motion cancels out')
    elif mode == 'moving':
        if gaps:
            add('mouth.moving.open_fraction', sev, mouth_stats['open_fraction'] >= mo.get('min_open_fraction', 0.12),
                mouth_stats['open_fraction'], mo.get('min_open_fraction'))
            add('mouth.moving.gap_std_ratio', sev, mouth_stats['gap_std_ratio'] >= mo.get('min_gap_std_ratio', 0.012),
                mouth_stats['gap_std_ratio'], mo.get('min_gap_std_ratio'))
        if energy.size and mo.get('min_motion_energy') is not None:
            add('mouth.moving.motion_energy', sev, mouth_stats['motion_energy_mean'] >= mo['min_motion_energy'],
                mouth_stats['motion_energy_mean'], mo['min_motion_energy'], 'a talking plate must actually articulate')
        if ratio.size and mo.get('min_motion_ratio') is not None:
            add('mouth.moving.motion_ratio', sev, mouth_stats['motion_ratio_mean'] >= mo['min_motion_ratio'],
                mouth_stats['motion_ratio_mean'], mo['min_motion_ratio'], 'mouth-region motion relative to whole-face motion')
    report['mouth'] = mouth_stats
    eyes = [e for e in eye_open[lo:hi] if e is not None]
    if eyes:
        e = np.asarray(eyes)
        median_open = float(np.median(e)) or 1e-6
        closed = e < 0.55 * median_open
        longest = cur = 0
        for c in closed:
            cur = cur + 1 if c else 0
            longest = max(longest, cur)
        report['eyes'] = {'median_openness_ratio': round(median_open, 4), 'longest_closed_run_frames': int(longest)}
        add('eyes.longest_closed_run', 'soft', longest <= int(round(fps)), int(longest), int(round(fps)),
            'eyes near-closed for more than about one second reads as a sleepy render')

    # ---- audio
    au = thresholds.get('audio', {})
    expected_audio = au.get('expected')
    if expected_audio is not None:
        add('audio.presence', au.get('severity', 'hard'), bool(astreams) == bool(expected_audio), len(astreams), 1 if expected_audio else 0)
    if astreams and expected_audio:
        a = astreams[0]
        a_dur = float(a.get('duration') or 0)
        tol = au.get('duration_tolerance_s', 0.05)
        add('audio.duration', au.get('severity', 'hard'), abs(a_dur - duration) <= tol, round(a_dur, 4), round(duration, 4), f'tolerance {tol}s')
        try:
            pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video), '-vn', '-ac', '1', '-ar', '16000',
                                           '-f', 's16le', '-'], stderr=subprocess.DEVNULL)
            samples = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
            rms_dbfs = 20 * math.log10(rms) if rms > 0 else -120.0
            clip_fraction = float((np.abs(samples) >= 0.999).mean()) if samples.size else 0.0
            report['audio'] = {'rms_dbfs': round(rms_dbfs, 2), 'clip_fraction': round(clip_fraction, 6), 'samples': int(samples.size)}
            add('audio.rms_dbfs', au.get('severity', 'hard'), rms_dbfs >= au.get('min_rms_dbfs', -42.0), round(rms_dbfs, 2), au.get('min_rms_dbfs'), 'silent or near-silent audio')
            add('audio.clip_fraction', au.get('severity', 'hard'), clip_fraction <= au.get('max_clip_fraction', 0.001), round(clip_fraction, 6), au.get('max_clip_fraction'))
        except subprocess.CalledProcessError as exc:
            add('audio.decode', au.get('severity', 'hard'), False, None, None, str(exc))

    # ---- contact sheet
    flagged = sorted({int(j[0]) for j in jumps} | {int(b) for b in blank} | {int(s[0]) for s in big_steps} | {int(r[0]) for r in frozen_runs})
    if contact is not None:
        write_contact(contact, tiles, flagged, video, width, height, pose)
        report['contact_sheet'] = str(contact)
    if keep_series:
        report['series'] = {'frame_diff': [round(float(x), 3) for x in d.tolist()],
                            'mean_luma': [round(float(x), 2) for x in ml.tolist()],
                            'face_present': [b is not None for b in boxes],
                            'mouth_gap_ratio': [None if g is None else round(g, 4) for g in mouth_gap_ratio],
                            'mouth_motion_energy': [round(float(x), 3) for x in mouth_energy],
                            'face_motion_energy': [round(float(x), 3) for x in face_energy]}
    report['flagged_frames'] = flagged
    return finish(report, started)


def finish(report: dict, started: float) -> dict:
    hard = [c['name'] for c in report['checks'] if c['severity'] == 'hard' and not c['pass']]
    soft = [c['name'] for c in report['checks'] if c['severity'] == 'soft' and not c['pass']]
    report['hard_failures'] = hard
    report['warnings'] = soft
    report['pass'] = not hard
    report['seconds'] = round(time.monotonic() - started, 2)
    report['tool'] = {'name': 'video_qc', 'version': 1, 'opencv': cv2.__version__, 'numpy': np.__version__}
    return report


def write_contact(path: Path, tiles: dict, flagged: list, video: Path, width: int, height: int, pose: str) -> None:
    flagged_frames = []
    if flagged:
        wanted = sorted(set(flagged))[:4]
        source = FrameSource(video, width, height)
        try:
            for index, (_, frame) in enumerate(source):
                if index in wanted:
                    flagged_frames.append((index, frame.copy()))
                if index > max(wanted):
                    break
        finally:
            source.close()
    rows = []
    for label, frames in (('sampled', sorted(tiles.items())), ('flagged', flagged_frames)):
        if not frames:
            continue
        row = []
        for index, frame in frames:
            tile = cv2.resize(frame, (CONTACT_TILE_WIDTH, int(CONTACT_TILE_WIDTH * height / width)), interpolation=cv2.INTER_AREA)
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            head = np.full((24, tile.shape[1], 3), 20, np.uint8)
            cv2.putText(head, f'{label} f{index}', (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            tile = np.vstack([head, tile])
            if label == 'flagged':
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 255), 3)
            row.append(tile)
        rows.append(np.hstack(row))
    if not rows:
        return
    widest = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, widest - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.vstack(rows)
    banner = np.full((28, sheet.shape[1], 3), 35, np.uint8)
    cv2.putText(banner, f'{video.name}  pose={pose}', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([banner, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def endpoint_set(videos: list[Path]) -> dict:
    """Cross-pose contract: every clip loops on itself and all clips share one decoded boundary frame."""
    rows = {}
    for video in videos:
        probe = ffprobe(video)
        v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        w, h = int(v['width']), int(v['height'])
        first = last = None
        for raw, _ in FrameSource(video, w, h):
            if first is None:
                first = raw
            last = raw
        rows[str(video)] = {'first_rgb_sha256': hashlib.sha256(first).hexdigest(),
                            'last_rgb_sha256': hashlib.sha256(last).hexdigest(), 'first_last_exact': first == last}
    hashes = {r['first_rgb_sha256'] for r in rows.values()}
    passed = all(r['first_last_exact'] for r in rows.values()) and len(hashes) == 1
    return {'videos': rows, 'shared_endpoint_rgb_sha256': hashes.pop() if len(hashes) == 1 else None, 'pass': passed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', type=Path)
    p.add_argument('--pose', choices=sorted(EXPECTED_FRAMES), default='talking')
    p.add_argument('--profile', default='strict')
    p.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    p.add_argument('--anchor', type=Path)
    p.add_argument('--expected-frames', type=int, help='Override the pose default (idle/talking 240, smiling 158)')
    p.add_argument('--no-expected-frames', action='store_true', help='Skip the frame-count contract')
    p.add_argument('--face-backend', choices=('auto', 'haar', 'mediapipe'), default='auto')
    p.add_argument('--out', type=Path)
    p.add_argument('--contact', type=Path)
    p.add_argument('--no-series', action='store_true')
    p.add_argument('--set', nargs='+', type=Path, help='Check that these clips share one exact decoded endpoint')
    args = p.parse_args()
    try:
        if args.set:
            result = endpoint_set(args.set)
        else:
            if not args.video:
                p.error('--video or --set is required')
            thresholds = resolve_thresholds(load_profiles(args.profiles), args.profile, args.pose)
            expected = None if args.no_expected_frames else (args.expected_frames or EXPECTED_FRAMES.get(args.pose))
            result = analyze_video(args.video, args.pose, thresholds, args.anchor, expected, args.face_backend,
                                   args.contact, keep_series=not args.no_series)
            result['profile'] = args.profile
    except Exception as exc:  # tool errors are distinct from QC failures
        print(json.dumps({'error': str(exc), 'pass': False}), file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    summary = {k: result.get(k) for k in ('pass', 'hard_failures', 'warnings', 'face_backend', 'frames', 'seconds')}
    print(json.dumps(summary if args.out else result, indent=None if args.out else 2))
    return 0 if result.get('pass') else 2


if __name__ == '__main__':
    raise SystemExit(main())
