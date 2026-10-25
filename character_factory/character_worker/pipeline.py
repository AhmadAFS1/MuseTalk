#!/usr/bin/env python3
"""One character job on a worker: portrait -> three H3 pose loops -> QC -> MuseTalk caches in S3.

Standard library only; every heavy step is a subprocess in its established environment:
  batch_three_pose.py (system python3) owns a ComfyUI server in the comfy-h3 venv for H3,
  the MuseTalk server venv runs POST /avatars/prepare (production cache + S3 upload),
  qc/video_qc.py runs in the chin-tools venv (MediaPipe) when present, else the server venv.

Stages and what they produce under the job directory:
  ingest    portraits/<character_id>.<ext>, portrait_check.json, metadata.json
  generate  attempts/<n>/<identity>/{idle,talking,smiling}/source.mp4 (+ h3.json records)
  qc        attempts/<n>/qc/<pose>.json, <pose>.jpg, endpoint_set.json
  prepare   attempts/<n>/<identity>/{pose}/s3.json, published.json   (owned MuseTalk API)
  verify    s3_verification.json (HEAD on every cache object)
  smoke     smoke/smoke.mp4 + smoke/qc.json (a real lip-sync render from the talking cache)
  records   records_publication.json (best effort upload of the job records to S3)
  release   local media removed after verified publication
  result    result.json, then the callback
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKFLOW = REPO / 'character_factory' / 'h3_avatar_workflow'
LOCK = json.loads((HERE / 'locks' / 'character_worker.lock.json').read_text())
POSES = ('idle', 'talking', 'smiling')
SAFE_ID = re.compile(r'^[a-z][a-z0-9_]{1,39}$')
IMAGE_MAGIC = ((b'\x89PNG\r\n\x1a\n', '.png'), (b'\xff\xd8\xff', '.jpg'), (b'RIFF', '.webp'))
MAX_PORTRAIT_BYTES = 30 * 1024 * 1024


class JobCancelled(Exception):
    pass


class StageFailed(Exception):
    def __init__(self, stage: str, message: str, detail=None):
        super().__init__(f'{stage}: {message}')
        self.stage, self.message, self.detail = stage, message, detail


def stamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def parse_env_file(path: Path) -> dict:
    """KEY=VALUE / export KEY=VALUE lines; quotes stripped; never executed."""
    out = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        out[key] = value
    return out


# ----------------------------------------------------------------------------- runtime

class Runtime:
    """Paths, interpreters and environment for one worker host."""

    def __init__(self, workspace: Path | None = None, root: Path | None = None):
        self.workspace = Path(os.environ.get('CHARACTER_WORKER_WORKSPACE') or workspace or '/workspace').resolve()
        self.root = Path(os.environ.get('CHARACTER_WORKER_ROOT') or root or (self.workspace / 'character_worker')).resolve()
        self.repo = Path(os.environ.get('CHARACTER_WORKER_REPO') or REPO).resolve()
        self.musetalk_python = self.workspace / '.venvs' / 'musetalk_trt_stagewise' / 'bin' / 'python'
        self.chin_python = self.workspace / '.venvs' / 'musetalk_chin_tools' / 'bin' / 'python'
        self.comfy_python = self.workspace / '.venvs' / 'comfy-h3' / 'bin' / 'python'
        self.h3_dir = self.workspace / 'minimax-h3'
        self.runtime_env_file = Path(os.environ.get('CHARACTER_WORKER_RUNTIME_ENV') or LOCK['worker']['runtime_env_file'])
        self.gpu_lock = self.workspace / '.h3_avatar_build_gpu.lock'
        self.musetalk_port = int(os.environ.get('CHARACTER_WORKER_MUSETALK_PORT') or LOCK['worker']['musetalk_private_port'])
        self._qc_python: Path | None = None

    def env(self, strip_control_plane: bool = False) -> dict:
        env = dict(os.environ)
        env.update(parse_env_file(self.runtime_env_file))
        env.setdefault('HF_HUB_OFFLINE', '1')
        env.setdefault('TRANSFORMERS_OFFLINE', '1')
        env.setdefault('PYTHONUNBUFFERED', '1')
        env.setdefault('WORKSPACE', str(self.workspace))
        if strip_control_plane:
            for key in [k for k in env if k.startswith('LINGUA_') or k in ('WORKER_ID',)]:
                env.pop(key, None)
        return env

    def s3_bucket(self) -> str | None:
        return self.env().get('AVATAR_S3_BUCKET') or None

    def qc_python(self) -> Path:
        if self._qc_python is None:
            candidate = self.chin_python
            if candidate.is_file():
                probe = subprocess.run([str(candidate), '-c', 'import mediapipe, cv2, numpy'], capture_output=True, timeout=120)
                if probe.returncode == 0:
                    self._qc_python = candidate
            if self._qc_python is None:
                self._qc_python = self.musetalk_python
        return self._qc_python

    def preflight(self) -> list[str]:
        problems = []
        for label, path in (('MuseTalk venv python', self.musetalk_python), ('comfy-h3 venv python', self.comfy_python),
                            ('H3 helper dir', self.h3_dir / 'run_h3_lowvram.py'), ('H3 model path config', self.h3_dir / 'extra_model_paths_h3.yaml'),
                            ('batch runner', WORKFLOW / 'batch_three_pose.py'), ('ffmpeg', Path(shutil.which('ffmpeg') or '/nonexistent'))):
            if not path.exists():
                problems.append(f'missing {label}: {path}')
        if not self.s3_bucket():
            problems.append('AVATAR_S3_BUCKET is not set (runtime env); caches cannot be published')
        return problems


# ----------------------------------------------------------------------------- job record

class Job:
    def __init__(self, job_dir: Path):
        self.dir = Path(job_dir)
        self.path = self.dir / 'job.json'
        self.data = read_json(self.path)
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.process: subprocess.Popen | None = None

    @property
    def id(self) -> str:
        return self.data['job_id']

    @property
    def request(self) -> dict:
        return self.data['request']

    def update(self, **fields) -> None:
        with self.lock:
            self.data.update(fields)
            self.data['updated_utc'] = stamp()
            write_json(self.path, self.data)

    def log(self, message: str) -> None:
        line = f'[{stamp()}] {message}'
        with (self.dir / 'pipeline.log').open('a') as f:
            f.write(line + '\n')

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def set_stage(self, stage: str, **extra) -> None:
        progress = dict(self.data.get('progress') or {})
        progress.update(extra)
        self.update(stage=stage, progress=progress)
        self.log(f'stage={stage} {json.dumps(extra) if extra else ""}')


def run_logged(job: Job, cmd: list, *, cwd: Path, env: dict, timeout: int, log_name: str, on_line=None) -> int:
    """Run a subprocess in its own process group, tee output to the job log, honour cancel and timeout."""
    job.check_cancel()
    log_path = job.dir / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log(f'run ({log_name}): {" ".join(str(c) for c in cmd)}')
    with log_path.open('a') as log:
        log.write(f'\n# {stamp()} {" ".join(str(c) for c in cmd)}\n')
        log.flush()
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        job.process = proc
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if on_line:
                    try:
                        on_line(line.rstrip('\n'))
                    except Exception:
                        pass
                if job.cancel_event.is_set() or time.monotonic() > deadline:
                    timed_out = time.monotonic() > deadline
                    break
            if proc.poll() is None and (job.cancel_event.is_set() or timed_out):
                terminate(proc)
            proc.wait()
        finally:
            job.process = None
    if job.cancel_event.is_set():
        raise JobCancelled()
    if timed_out:
        raise StageFailed(log_name, f'timed out after {timeout}s')
    return proc.returncode


def terminate(proc: subprocess.Popen, grace: int = 60) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


# ----------------------------------------------------------------------------- stages

def ingest_portrait(rt: Runtime, job: Job) -> Path:
    job.set_stage('ingest')
    spec = job.request['portrait']
    name = job.request['character_id']
    data: bytes
    if spec.get('base64'):
        data = base64.b64decode(spec['base64'], validate=False)
    elif spec.get('url'):
        req = urllib.request.Request(spec['url'], headers={'User-Agent': 'character-worker/1'})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read(MAX_PORTRAIT_BYTES + 1)
    elif spec.get('s3_uri'):
        tmp = job.dir / 'portraits' / 'download.bin'
        rc = run_logged(job, [rt.musetalk_python, HERE / 'tools' / 's3_tool.py', 'get', spec['s3_uri'], tmp],
                        cwd=job.dir, env=rt.env(), timeout=600, log_name='ingest.log')
        if rc:
            raise StageFailed('ingest', f'could not download {spec["s3_uri"]}')
        data = tmp.read_bytes()
        tmp.unlink()
    elif spec.get('path'):
        data = Path(spec['path']).read_bytes()
    else:
        raise StageFailed('ingest', 'portrait needs one of base64, url, s3_uri')
    if not data or len(data) > MAX_PORTRAIT_BYTES:
        raise StageFailed('ingest', f'portrait is empty or larger than {MAX_PORTRAIT_BYTES} bytes')
    ext = next((e for magic, e in IMAGE_MAGIC if data.startswith(magic)), None)
    if ext is None:
        raise StageFailed('ingest', 'portrait is not PNG, JPEG or WebP')
    portraits = job.dir / 'portraits'
    portraits.mkdir(parents=True, exist_ok=True)
    for stale in portraits.iterdir():
        stale.unlink()
    portrait = portraits / f'{name}{ext}'
    portrait.write_bytes(data)
    report_path = job.dir / 'portrait_check.json'
    rc = run_logged(job, [rt.musetalk_python, HERE / 'tools' / 'portrait_check.py', portrait, '--out', report_path],
                    cwd=job.dir, env=rt.env(), timeout=300, log_name='ingest.log')
    report = read_json(report_path) if report_path.is_file() else {'ok': False, 'errors': ['portrait check produced no report']}
    job.update(portrait={'file': str(portrait), 'sha256': sha256_file(portrait), 'bytes': len(data), 'check': report})
    if rc or not report.get('ok'):
        raise StageFailed('ingest', 'portrait rejected: ' + '; '.join(report.get('errors') or ['unknown']), report)
    profile = dict(job.request.get('metadata') or {})
    profile['id'] = name
    write_json(job.dir / 'metadata.json', {'images': {portrait.name: profile}})
    return portrait


def batch_cmd(rt: Runtime, job: Job, out: Path, stage: str, extra: list) -> list:
    return ['python3', WORKFLOW / 'batch_three_pose.py', '--images', job.dir / 'portraits', '--metadata', job.dir / 'metadata.json',
            '--output', out, '--workspace', rt.workspace, '--stage', stage, *extra]


def identity_dir(out: Path) -> Path:
    plan = read_json(out / 'batch_plan.json')
    jobs = plan.get('jobs') or []
    if len(jobs) != 1:
        raise StageFailed('generate', f'expected one identity in {out}, found {jobs}')
    return out / jobs[0]


def generate(rt: Runtime, job: Job, attempt: int, seed: int, poses: list) -> Path:
    out = job.dir / 'attempts' / f'{attempt:02d}'
    out.mkdir(parents=True, exist_ok=True)
    job.set_stage('generate', attempt=attempt, seed=seed, pose=None)
    options = job.request.get('options') or {}
    timeout = int(options.get('generate_timeout_seconds') or 6000)
    rc = run_logged(job, batch_cmd(rt, job, out, 'plan', ['--seed', seed]), cwd=WORKFLOW, env=rt.env(), timeout=600,
                    log_name=f'attempts/{attempt:02d}/batch.log')
    if rc:
        raise StageFailed('generate', 'planning failed; see batch.log')

    def on_line(line: str):
        m = re.search(r'\bH3 (\S+) (idle|talking|smiling)\b', line)
        if m:
            job.set_stage('generate', attempt=attempt, seed=seed, pose=m.group(2))

    rc = run_logged(job, batch_cmd(rt, job, out, 'generate', ['--seed', seed, '--poses', *poses]), cwd=WORKFLOW,
                    env=rt.env(), timeout=timeout, log_name=f'attempts/{attempt:02d}/batch.log', on_line=on_line)
    status = read_json(out / 'batch_status.json') if (out / 'batch_status.json').is_file() else {}
    if rc or status.get('failures'):
        raise StageFailed('generate', 'H3 generation failed: ' + json.dumps(status.get('failures') or rc)[:1200], status)
    ident = identity_dir(out)
    for pose in poses:
        rec = ident / pose / 'h3.json'
        if not rec.is_file() or read_json(rec).get('status') != 'complete':
            raise StageFailed('generate', f'{pose} has no complete H3 record')
    return ident


def qc_poses(rt: Runtime, job: Job, attempt: int, ident: Path, poses: list) -> dict:
    job.set_stage('qc', attempt=attempt)
    options = job.request.get('options') or {}
    profile = options.get('qc_profile') or 'strict'
    qc_dir = ident.parent / 'qc'
    qc_dir.mkdir(exist_ok=True)
    python = rt.qc_python()
    reports, hard = {}, {}
    for pose in poses:
        video = ident / pose / 'source.mp4'
        out = qc_dir / f'{pose}.json'
        rc = run_logged(job, [python, HERE / 'qc' / 'video_qc.py', '--video', video, '--pose', pose, '--profile', profile,
                              '--anchor', ident / 'anchor.png', '--out', out, '--contact', qc_dir / f'{pose}.jpg'],
                        cwd=HERE, env=rt.env(), timeout=900, log_name=f'attempts/{attempt:02d}/qc.log')
        if rc == 1 or not out.is_file():
            raise StageFailed('qc', f'video_qc tool error on {pose} (exit {rc}); see qc.log')
        report = read_json(out)
        report.pop('series', None)
        reports[pose] = report
        hard[pose] = report.get('hard_failures', [])
    set_result = None
    if len(poses) > 1:
        set_out = qc_dir / 'endpoint_set.json'
        rc = run_logged(job, [python, HERE / 'qc' / 'video_qc.py', '--set', *[ident / p / 'source.mp4' for p in poses], '--out', set_out],
                        cwd=HERE, env=rt.env(), timeout=600, log_name=f'attempts/{attempt:02d}/qc.log')
        set_result = read_json(set_out) if set_out.is_file() else {'pass': False, 'error': f'exit {rc}'}
    passed = all(not h for h in hard.values()) and (set_result is None or set_result.get('pass'))
    summary = {'attempt': attempt, 'profile': profile, 'pass': passed, 'face_backend': next(iter(reports.values()), {}).get('face_backend'),
               'hard_failures': {p: h for p, h in hard.items() if h},
               'warnings': {p: r.get('warnings', []) for p, r in reports.items() if r.get('warnings')},
               'endpoint_set': {k: v for k, v in (set_result or {}).items() if k != 'videos'},
               'reports': {p: str(qc_dir / f'{p}.json') for p in poses}, 'contact_sheets': {p: str(qc_dir / f'{p}.jpg') for p in poses}}
    write_json(qc_dir / 'summary.json', summary)
    job.log(f'qc attempt {attempt}: pass={passed} hard={summary["hard_failures"]} warnings={summary["warnings"]}')
    return summary


class MuseTalkServer:
    """An owned MuseTalk API on a private port, alive only for the prepare + smoke stages."""

    def __init__(self, rt: Runtime, job: Job):
        self.rt, self.job = rt, job
        self.port = rt.musetalk_port
        self.base = f'http://127.0.0.1:{self.port}'
        self.proc: subprocess.Popen | None = None
        self.log_path = job.dir / 'musetalk_api.log'

    def stats(self, timeout: int = 20) -> dict:
        with urllib.request.urlopen(self.base + '/stats', timeout=timeout) as resp:
            return json.load(resp)

    def s3(self) -> dict:
        s3 = self.stats().get('avatar_s3') or {}
        if not s3.get('enabled') or not s3.get('bucket') or not s3.get('version'):
            raise RuntimeError(f'MuseTalk API has no enabled avatar S3 store: {s3}')
        return s3

    def start(self, timeout: int = 1200) -> dict:
        env = self.rt.env(strip_control_plane=True)
        env.update(AVATAR_S3_ENABLED='1', HOST='127.0.0.1', PORT=str(self.port), WEBRTC_RELAY_ENABLED='0',
                   WEBRTC_TURN_AUTOSTART='0', SETUP_WEBRTC_TURN='0', MUSETALK_LOAD_TEST_TELEMETRY='0')
        if not env.get('AVATAR_S3_BUCKET'):
            raise StageFailed('prepare', 'AVATAR_S3_BUCKET is not set; refusing to prepare caches that cannot be published')
        cmd = ['bash', self.rt.repo / 'scripts' / 'run_musetalk_server.sh', '--host', '127.0.0.1', '--port', str(self.port),
               '--repo-root', self.rt.repo, '--venv-path', self.rt.musetalk_python.parents[1]]
        self.job.log(f'starting MuseTalk API on {self.base}')
        with self.log_path.open('a') as log:
            log.write(f'\n# {stamp()} start\n')
            self.proc = subprocess.Popen([str(c) for c in cmd], cwd=str(self.rt.repo), env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            self.job.check_cancel()
            if self.proc.poll() is not None:
                raise StageFailed('prepare', f'MuseTalk API exited with {self.proc.returncode}; see musetalk_api.log', self.tail())
            try:
                return self.s3()
            except (OSError, urllib.error.URLError, RuntimeError, ValueError) as exc:
                last = exc
                time.sleep(3)
        self.stop()
        raise StageFailed('prepare', f'MuseTalk API not ready after {timeout}s ({last}); see musetalk_api.log', self.tail())

    def tail(self, lines: int = 40) -> str:
        try:
            return '\n'.join(self.log_path.read_text(errors='replace').splitlines()[-lines:])
        except OSError:
            return ''

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.job.log('stopping MuseTalk API')
            terminate(self.proc, grace=90)
        self.proc = None


def prepare(rt: Runtime, job: Job, attempt: int, ident: Path, server: MuseTalkServer) -> dict:
    job.set_stage('prepare', attempt=attempt, pose=None)
    out = ident.parent
    options = job.request.get('options') or {}
    bucket = rt.s3_bucket()
    batch_size = int(options.get('batch_size') or LOCK['musetalk']['prepare_batch_size'])

    def on_line(line: str):
        m = re.search(r'\bS3 (\S+) (idle|talking|smiling)\b', line)
        if m:
            job.set_stage('prepare', attempt=attempt, pose=m.group(2))

    rc = run_logged(job, batch_cmd(rt, job, out, 'prepare', ['--prepare-url', server.base, '--expected-bucket', bucket,
                                                             '--expected-version', LOCK['musetalk']['avatar_cache_version'],
                                                             '--batch-size', batch_size, '--timeout', 1800,
                                                             '--seed', (job.data.get('progress') or {}).get('seed', 42)]),
                    cwd=WORKFLOW, env=rt.env(strip_control_plane=True), timeout=int(options.get('prepare_timeout_seconds') or 3600),
                    log_name=f'attempts/{attempt:02d}/batch.log', on_line=on_line)
    published_path = ident / 'published.json'
    status = read_json(out / 'batch_status.json') if (out / 'batch_status.json').is_file() else {}
    if rc or status.get('failures') or not published_path.is_file():
        raise StageFailed('prepare', 'MuseTalk preparation/publication failed: ' + json.dumps(status.get('failures') or rc)[:1200], status)
    published = read_json(published_path)
    if published.get('status') != 'complete':
        raise StageFailed('prepare', 'published.json is not complete', published)
    return published


def verify_s3(rt: Runtime, job: Job, published: dict) -> dict:
    job.set_stage('verify')
    rows = {}
    for pose, info in published['poses'].items():
        out = job.dir / 'verify' / f'{pose}.json'
        rc = run_logged(job, [rt.musetalk_python, HERE / 'tools' / 's3_tool.py', 'head', info['s3_uri']], cwd=job.dir, env=rt.env(),
                        timeout=120, log_name='verify/s3_head.log')
        head = {}
        text = (job.dir / 'verify' / 's3_head.log').read_text(errors='replace').strip().splitlines()
        for line in reversed(text):
            if line.startswith('{') and info['s3_uri'] in line:
                head = json.loads(line)
                break
        ok = rc == 0 and head.get('exists') and (head.get('size') or 0) > 0 and head.get('metadata', {}).get('avatar-id', info['avatar_id']) == info['avatar_id']
        rows[pose] = {'s3_uri': info['s3_uri'], 'avatar_id': info['avatar_id'], 'head': head, 'ok': bool(ok)}
        write_json(out, rows[pose])
    result = {'pass': all(r['ok'] for r in rows.values()), 'poses': rows, 'checked_utc': stamp()}
    write_json(job.dir / 's3_verification.json', result)
    if not result['pass']:
        raise StageFailed('verify', 'an S3 cache object is missing or empty', result)
    return result


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = f'----characterworker{uuid.uuid4().hex}'
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, path in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{Path(path).name}"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n'.encode())
        parts.append(Path(path).read_bytes())
        parts.append(b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


def smoke_render(rt: Runtime, job: Job, server: MuseTalkServer, published: dict) -> dict:
    """Generate a few seconds of real lip-sync from the talking cache and QC the result."""
    job.set_stage('smoke')
    options = job.request.get('options') or {}
    smoke_dir = job.dir / 'smoke'
    smoke_dir.mkdir(exist_ok=True)
    audio_src = Path(options.get('smoke_audio') or (rt.repo / 'data' / 'audio' / 'eng.wav'))
    audio = smoke_dir / 'smoke_audio.wav'
    seconds = float(options.get('smoke_seconds') or 5.0)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(audio_src), '-t', str(seconds), '-ac', '1', '-ar', '16000',
                    '-c:a', 'pcm_s16le', str(audio)], check=True)
    talking = published['poses'].get('talking') or next(iter(published['poses'].values()))
    avatar_id = talking['avatar_id']
    body, ctype = multipart({}, {'audio_file': audio})
    query = urllib.parse.urlencode({'avatar_id': avatar_id, 'batch_size': int(options.get('batch_size') or 8), 'fps': 24})
    req = urllib.request.Request(f'{server.base}/generate?{query}', data=body, headers={'Content-Type': ctype}, method='POST')
    with urllib.request.urlopen(req, timeout=120) as resp:
        queued = json.load(resp)
    request_id = queued.get('request_id')
    if not request_id:
        raise StageFailed('smoke', f'generate did not queue: {queued}')
    deadline = time.monotonic() + int(options.get('smoke_timeout_seconds') or 900)
    status = {}
    while time.monotonic() < deadline:
        job.check_cancel()
        with urllib.request.urlopen(f'{server.base}/generate/{request_id}/status', timeout=60) as resp:
            status = json.load(resp)
        if status.get('status') in ('completed', 'failed', 'error'):
            break
        time.sleep(2)
    if status.get('status') != 'completed' or not (status.get('result') or {}).get('success', True):
        raise StageFailed('smoke', f'smoke render did not complete: {status}')
    video = smoke_dir / 'smoke.mp4'
    with urllib.request.urlopen(f'{server.base}/generate/{request_id}/download', timeout=300) as resp, video.open('wb') as f:
        shutil.copyfileobj(resp, f)
    out = smoke_dir / 'qc.json'
    rc = run_logged(job, [rt.qc_python(), HERE / 'qc' / 'video_qc.py', '--video', video, '--pose', 'smoke', '--profile',
                          options.get('qc_profile') or 'strict', '--no-expected-frames', '--out', out, '--contact', smoke_dir / 'contact.jpg'],
                    cwd=HERE, env=rt.env(), timeout=900, log_name='smoke/qc.log')
    report = read_json(out) if out.is_file() else {'pass': False, 'hard_failures': [f'video_qc exit {rc}']}
    report.pop('series', None)
    summary = {'avatar_id': avatar_id, 'request_id': request_id, 'video': str(video), 'audio': str(audio), 'audio_seconds': seconds,
               'pass': bool(report.get('pass')), 'hard_failures': report.get('hard_failures', []), 'warnings': report.get('warnings', []),
               'frames': report.get('frames'), 'mouth': report.get('mouth'), 'face': report.get('face'), 'qc_report': str(out)}
    write_json(smoke_dir / 'summary.json', summary)
    job.log(f'smoke render: pass={summary["pass"]} hard={summary["hard_failures"]}')
    return summary


def publish_records(rt: Runtime, job: Job) -> dict:
    """Best effort: the runtime S3 key may only cover avatars/*; a denied upload is reported, not fatal."""
    job.set_stage('records')
    bucket = rt.s3_bucket()
    prefix = (rt.env().get('CHARACTER_RECORDS_S3_PREFIX') or 'character-factory/jobs').strip('/')
    if not bucket:
        return {'skipped': 'no bucket'}
    staging = job.dir / '_records'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    wanted = {'.json', '.jpg', '.log', '.png', '.webp', '.txt'}
    for path in job.dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in wanted and '_records' not in path.parts and path.name != 'job.json':
            if path.suffix.lower() in ('.png', '.webp') and path.parent.name != 'portraits':
                continue
            dest = staging / path.relative_to(job.dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    uri = f's3://{bucket}/{prefix}/{job.request["character_id"]}/{job.id}/'
    rc = run_logged(job, [rt.musetalk_python, HERE / 'tools' / 's3_tool.py', 'put-dir', staging, uri], cwd=job.dir, env=rt.env(),
                    timeout=900, log_name='records/put.log')
    shutil.rmtree(staging, ignore_errors=True)
    result = {'uri': uri, 'ok': rc == 0}
    if rc:
        result['warning'] = 'record upload failed (see records/put.log); the runtime IAM policy may not allow this prefix'
    write_json(job.dir / 'records_publication.json', result)
    return result


def release_local(rt: Runtime, job: Job, published: dict | None) -> dict:
    job.set_stage('release')
    removed = []
    for path in (job.dir / 'attempts').rglob('*.mp4'):
        removed.append(str(path))
        path.unlink(missing_ok=True)
    for path in (job.dir / 'attempts').rglob('_incomplete_attempts'):
        shutil.rmtree(path, ignore_errors=True)
    if published:
        version = LOCK['musetalk']['avatar_cache_version']
        for info in published['poses'].values():
            cache = rt.repo / 'results' / version / 'avatars' / info['avatar_id']
            if cache.is_dir():
                shutil.rmtree(cache, ignore_errors=True)
                removed.append(str(cache))
            for upload in (rt.repo / 'uploads' / 'videos').glob(f'{info["avatar_id"]}_*'):
                upload.unlink(missing_ok=True)
                removed.append(str(upload))
    job.log(f'released {len(removed)} local files/dirs')
    return {'removed': len(removed)}


def callback(job: Job, result: dict) -> None:
    url = job.request.get('callback_url')
    if not url:
        return
    body = json.dumps(result, default=str).encode()
    headers = {'Content-Type': 'application/json', 'X-Character-Job': job.id}
    token = job.request.get('callback_token') or os.environ.get('CHARACTER_WORKER_CALLBACK_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    for attempt in range(1, 5):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as resp:
                job.log(f'callback {url} -> {resp.status}')
                job.update(callback={'url': url, 'status': resp.status, 'attempt': attempt, 'sent_utc': stamp()})
                return
        except Exception as exc:
            job.log(f'callback attempt {attempt} failed: {exc}')
            time.sleep(min(30, 3 * attempt))
    job.update(callback={'url': url, 'status': 'failed', 'sent_utc': stamp()})


# ----------------------------------------------------------------------------- driver

def host_facts() -> dict:
    facts = {'instance_id': os.environ.get('VAST_INSTANCE_ID') or os.environ.get('CONTAINER_ID'),
             'public_ip': os.environ.get('PUBLIC_IPADDR'), 'hostname': os.uname().nodename}
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                      text=True, timeout=10).strip().splitlines()
        facts['gpu'] = out[0] if out else None
    except Exception:
        facts['gpu'] = None
    return facts


def run_job(rt: Runtime, job: Job) -> dict:
    """Execute every stage; returns the result dict that is also written to result.json."""
    request = job.request
    options = request.get('options') or {}
    poses = list(request.get('poses') or POSES)
    base_seed = int(options.get('seed', 42))
    max_attempts = max(1, int(options.get('max_attempts', 2)))
    started = time.monotonic()
    result = {'job_id': job.id, 'character_id': request['character_id'], 'recipe': LOCK['recipe']['name'], 'poses_requested': poses,
              'worker': host_facts(), 'started_utc': stamp(), 'attempts': [], 'status': 'running'}
    job.update(status='running', started_utc=result['started_utc'], result=None, error=None)
    published = None
    server = None
    try:
        problems = rt.preflight()
        if problems:
            raise StageFailed('preflight', '; '.join(problems))
        portrait = ingest_portrait(rt, job)
        result['portrait_sha256'] = sha256_file(portrait)
        ident = None
        qc_summary = None
        for attempt in range(1, max_attempts + 1):
            seed = base_seed + attempt - 1
            ident = generate(rt, job, attempt, seed, poses)
            qc_summary = qc_poses(rt, job, attempt, ident, poses)
            h3 = {p: read_json(ident / p / 'h3.json') for p in poses}
            result['attempts'].append({'attempt': attempt, 'seed': seed, 'identity': ident.name, 'qc': qc_summary,
                                       'h3_seconds': {p: h3[p].get('generation_seconds') for p in poses},
                                       'source_sha256': {p: sha256_file(ident / p / 'source.mp4') for p in poses}})
            if qc_summary['pass']:
                break
            job.log(f'attempt {attempt} failed QC; {"rerolling with a new seed" if attempt < max_attempts else "no attempts left"}')
            if attempt < max_attempts:
                release_local(rt, job, None) if options.get('release_failed_attempts', True) else None
        assert ident is not None and qc_summary is not None
        if not qc_summary['pass']:
            result.update(status='qc_failed', error='generated clips failed QC on every attempt', qc=qc_summary)
            return finish(rt, job, result, started, published, options)
        result['identity'] = ident.name
        result['qc'] = qc_summary
        server = MuseTalkServer(rt, job)
        with open(rt.gpu_lock, 'a') as lock_file:
            import fcntl
            job.log('waiting for the GPU lock')
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            s3 = server.start()
            result['musetalk_s3'] = {k: s3.get(k) for k in ('bucket', 'prefix', 'version')}
            published = prepare(rt, job, len(result['attempts']), ident, server)
            result['published'] = {'endpoint_rgb_sha256': published.get('endpoint_rgb_sha256'), 'poses': published['poses'],
                                   'finished_utc': published.get('finished_utc')}
            if options.get('smoke_render', True):
                result['smoke_render'] = smoke_render(rt, job, server, published)
            server.stop()
        result['s3_verification'] = verify_s3(rt, job, published)
        smoke = result.get('smoke_render')
        if smoke and not smoke['pass'] and options.get('smoke_render_required', True):
            result.update(status='qc_failed', error='the published caches rendered but the smoke render failed QC; caches remain in S3 for inspection')
            return finish(rt, job, result, started, published, options)
        if options.get('publish_records', True):
            result['records'] = publish_records(rt, job)
        result['status'] = 'done'
        return finish(rt, job, result, started, published, options)
    except JobCancelled:
        result.update(status='cancelled', error='cancelled by request')
        return finish(rt, job, result, started, published, options)
    except StageFailed as exc:
        job.log(f'FAILED {exc.stage}: {exc.message}')
        result.update(status='failed', error=str(exc), failed_stage=exc.stage, failure_detail=exc.detail)
        return finish(rt, job, result, started, published, options)
    except Exception as exc:  # unexpected: record the traceback for the operator
        import traceback
        job.log('FAILED unexpected: ' + traceback.format_exc())
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', failed_stage=job.data.get('stage'))
        return finish(rt, job, result, started, published, options)
    finally:
        if server is not None:
            server.stop()


def finish(rt: Runtime, job: Job, result: dict, started: float, published, options: dict) -> dict:
    result['finished_utc'] = stamp()
    result['seconds'] = round(time.monotonic() - started, 1)
    try:
        if options.get('release_local', True) and result['status'] in ('done', 'qc_failed'):
            result['release'] = release_local(rt, job, published if result['status'] == 'done' else None)
    except Exception as exc:
        result['release'] = {'error': str(exc)}
    write_json(job.dir / 'result.json', result)
    job.update(status=result['status'], stage='finished', finished_utc=result['finished_utc'], result=result, error=result.get('error'))
    callback(job, result)
    return result


class FakePipeline:
    """Used by tests and by CHARACTER_WORKER_FAKE_PIPELINE=1: no GPU, no S3, deterministic timing."""

    def __init__(self, seconds: float = 0.5):
        self.seconds = seconds

    def run_job(self, rt: Runtime, job: Job) -> dict:
        request = job.request
        options = request.get('options') or {}
        outcome = options.get('fake_outcome', 'done')
        started = time.monotonic()
        job.update(status='running', started_utc=stamp())
        result = {'job_id': job.id, 'character_id': request['character_id'], 'recipe': LOCK['recipe']['name'], 'fake': True,
                  'started_utc': stamp(), 'attempts': [], 'status': 'running', 'worker': host_facts()}
        try:
            for stage in ('ingest', 'generate', 'qc', 'prepare', 'verify', 'smoke', 'records'):
                job.check_cancel()
                job.set_stage(stage, attempt=1)
                time.sleep(float(options.get('fake_seconds', self.seconds)) / 7)
                if outcome == 'failed' and stage == 'generate':
                    raise StageFailed('generate', 'fake H3 failure')
                if outcome == 'qc_failed' and stage == 'qc':
                    result.update(status='qc_failed', error='fake QC failure')
                    return finish(rt, job, result, started, None, {'release_local': False})
            ident = f"{request['character_id']}_{hashlib.sha256(request['character_id'].encode()).hexdigest()[:8]}"
            poses = list(request.get('poses') or POSES)
            result['identity'] = ident
            result['published'] = {'endpoint_rgb_sha256': 'f' * 64, 'poses': {
                p: {'avatar_id': f'{ident}_{p}_{"0" * 10}', 's3_uri': f's3://fake-bucket/avatars/v15/{ident}_{p}_{"0" * 10}.tar.gz',
                    'source_sha256': '0' * 64} for p in poses}}
            result['qc'] = {'pass': True, 'attempt': 1, 'profile': options.get('qc_profile', 'strict')}
            result['smoke_render'] = {'pass': True, 'fake': True}
            result['status'] = 'done'
            return finish(rt, job, result, started, None, {'release_local': False})
        except JobCancelled:
            result.update(status='cancelled', error='cancelled by request')
            return finish(rt, job, result, started, None, {'release_local': False})
        except StageFailed as exc:
            result.update(status='failed', error=str(exc), failed_stage=exc.stage)
            return finish(rt, job, result, started, None, {'release_local': False})


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description='character worker pipeline helpers')
    p.add_argument('--preflight', action='store_true', help='print host/runtime problems (empty output and exit 0 when ready)')
    p.add_argument('--run-job-dir', type=Path, help='run one job directory that already holds job.json (operator use)')
    args = p.parse_args()
    rt = Runtime()
    if args.preflight:
        problems = rt.preflight()
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    if args.run_job_dir:
        result = run_job(rt, Job(args.run_job_dir))
        print(json.dumps({k: result.get(k) for k in ('job_id', 'status', 'error', 'seconds')}))
        return 0 if result.get('status') == 'done' else 2
    p.print_help()
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
