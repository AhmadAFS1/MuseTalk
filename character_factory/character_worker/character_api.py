#!/usr/bin/env python3
"""Character-creation worker API: a standard-library HTTP server with a one-at-a-time GPU job queue.

Starts in under a second on the system python so a booting Vast instance can report its
install phase on /health long before the venvs exist. The heavy work runs in pipeline.py.

  GET  /health                               readiness, phase, queue depth, versions (no auth)
  POST /createCharacter                      JSON body -> 202 {job_id, character_id, status, position}
  GET  /characters                           recent jobs
  GET  /characters/{job_id}                  job record (status, stage, progress, result, error)
  POST /characters/{job_id}/cancel           cancel a queued job, or interrupt a running one
  GET  /characters/{job_id}/artifacts        files in the job directory
  GET  /characters/{job_id}/artifacts/{path} download one (qc json, contact sheets, logs, smoke.mp4)
  GET  /queue                                queued and running jobs
  POST /worker/drain | /worker/undrain       stop/resume accepting new jobs

Auth: `Authorization: Bearer <CHARACTER_WORKER_TOKEN>` on everything except /health when a
token is configured (environment or the runtime env file written by the installer).

createCharacter body:
  {"character_id": "en_05_amara",                      # optional, [a-z][a-z0-9_]{1,39}
   "portrait": {"base64": "..."} | {"url": "https://..."} | {"s3_uri": "s3://bucket/key"},
   "metadata": {"label": "Black British woman", "age": 28, "gender": "woman", "hair": "...",
                "tone": "...", "facial_hair": "none", "wardrobe": "...", "background": "...",
                "pronouns": "she" | {"subject": "she", ...}, "subject": "..."},
   "poses": ["idle", "talking", "smiling"],
   "options": {"seed": 42, "max_attempts": 2, "qc_profile": "strict", "smoke_render": true,
               "smoke_render_required": true, "release_local": true, "publish_records": true,
               "batch_size": 8},
   "callback_url": "https://.../hooks/character", "callback_token": "..."}
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import mimetypes
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline  # noqa: E402

SERVICE = 'character-worker'
API_VERSION = 1
MAX_BODY = 48 * 1024 * 1024
ARTIFACT_SUFFIXES = {'.json', '.jpg', '.jpeg', '.png', '.webp', '.mp4', '.log', '.txt', '.wav'}
ACTIVE = ('queued', 'running')
TERMINAL = ('done', 'failed', 'qc_failed', 'cancelled')
QC_PROFILES = sorted(k for k in json.loads((HERE / 'qc' / 'profiles.json').read_text()) if not k.startswith('_'))
OPTION_TYPES = {'seed': int, 'max_attempts': int, 'qc_profile': str, 'smoke_render': bool, 'smoke_render_required': bool,
                'release_local': bool, 'release_failed_attempts': bool, 'publish_records': bool, 'batch_size': int,
                'generate_timeout_seconds': int, 'prepare_timeout_seconds': int, 'smoke_timeout_seconds': int,
                'smoke_seconds': (int, float), 'fake_outcome': str, 'fake_seconds': (int, float)}
ACCEPTED_PROMPT_FIELDS = ('label', 'age', 'gender', 'hair', 'tone', 'facial_hair')


class BadRequest(ValueError):
    pass


def safe_character_id(raw: str | None, fallback: str | None) -> str:
    text = (raw or fallback or 'character').lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    if text and text[0].isdigit():
        text = 'c_' + text
    text = text[:40].rstrip('_')
    if not pipeline.SAFE_ID.fullmatch(text or ''):
        raise BadRequest(f'character_id {raw!r} is not usable; use [a-z][a-z0-9_]{{1,39}}')
    return text


def validate_request(body, allow_local_paths: bool = False, fake: bool = False) -> tuple[dict, list]:
    if not isinstance(body, dict):
        raise BadRequest('body must be a JSON object')
    warnings = []
    portrait = body.get('portrait')
    if not isinstance(portrait, dict) or not any(portrait.get(k) for k in ('base64', 'url', 's3_uri', 'path')):
        raise BadRequest('portrait must be an object with base64, url or s3_uri')
    if portrait.get('path') and not allow_local_paths:
        raise BadRequest('portrait.path is only accepted when CHARACTER_WORKER_ALLOW_LOCAL_PATHS=1')
    if portrait.get('url'):
        scheme = urlparse(portrait['url']).scheme
        if scheme not in ('http', 'https'):
            raise BadRequest('portrait.url must be http(s)')
    if portrait.get('s3_uri') and not str(portrait['s3_uri']).startswith('s3://'):
        raise BadRequest('portrait.s3_uri must start with s3://')
    if portrait.get('base64'):
        try:
            base64.b64decode(str(portrait['base64'])[:64] + '=' * 3, validate=False)
        except Exception:
            raise BadRequest('portrait.base64 is not base64')
    metadata = body.get('metadata') or {}
    if not isinstance(metadata, dict):
        raise BadRequest('metadata must be an object')
    if 'age' in metadata:
        try:
            metadata['age'] = int(metadata['age'])
        except (TypeError, ValueError):
            raise BadRequest('metadata.age must be an integer')
    missing = [k for k in ACCEPTED_PROMPT_FIELDS if not metadata.get(k) and k != 'facial_hair']
    if missing or 'facial_hair' not in metadata:
        warnings.append('metadata lacks ' + ', '.join(sorted(set(missing) | ({'facial_hair'} if 'facial_hair' not in metadata else set())))
                        + '; the generic identity prompt will be used instead of the accepted talking prompt builder')
    pronouns = metadata.get('pronouns')
    if isinstance(pronouns, str) and pronouns.lower() not in ('she', 'he', 'they'):
        raise BadRequest('metadata.pronouns must be she, he or they (or an object with subject/object/possessive)')
    character_id = safe_character_id(body.get('character_id'), metadata.get('display_name') or metadata.get('label'))
    poses = body.get('poses') or list(pipeline.POSES)
    if not isinstance(poses, list) or not poses or any(p not in pipeline.POSES for p in poses):
        raise BadRequest(f'poses must be a non-empty subset of {list(pipeline.POSES)}')
    poses = [p for p in pipeline.POSES if p in poses]
    if 'talking' not in poses:
        warnings.append('talking is not requested; the smoke render is skipped because it needs the talking cache')
    options = body.get('options') or {}
    if not isinstance(options, dict):
        raise BadRequest('options must be an object')
    clean = {}
    for key, value in options.items():
        if key not in OPTION_TYPES:
            raise BadRequest(f'unknown option {key!r}; known: {sorted(OPTION_TYPES)}')
        expected = OPTION_TYPES[key]
        if expected is bool and not isinstance(value, bool):
            raise BadRequest(f'option {key} must be true/false')
        if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise BadRequest(f'option {key} must be an integer')
        if expected is str and not isinstance(value, str):
            raise BadRequest(f'option {key} must be a string')
        if isinstance(expected, tuple) and not isinstance(value, expected):
            raise BadRequest(f'option {key} must be a number')
        clean[key] = value
    if 'qc_profile' in clean and clean['qc_profile'] not in QC_PROFILES:
        raise BadRequest(f'qc_profile must be one of {QC_PROFILES}')
    if 'max_attempts' in clean and not 1 <= clean['max_attempts'] <= 4:
        raise BadRequest('max_attempts must be between 1 and 4')
    if 'batch_size' in clean and not 1 <= clean['batch_size'] <= 32:
        raise BadRequest('batch_size must be between 1 and 32')
    if 'seed' in clean and not 0 <= clean['seed'] < 2 ** 63:
        raise BadRequest('seed out of range')
    if not fake and ('fake_outcome' in clean or 'fake_seconds' in clean):
        raise BadRequest('fake_* options are only accepted by a fake-pipeline worker')
    if 'talking' not in poses:
        clean['smoke_render'] = False
    callback_url = body.get('callback_url')
    if callback_url is not None:
        if not isinstance(callback_url, str) or urlparse(callback_url).scheme not in ('http', 'https'):
            raise BadRequest('callback_url must be http(s)')
    request = {'character_id': character_id, 'portrait': portrait, 'metadata': metadata, 'poses': poses, 'options': clean,
               'callback_url': callback_url, 'callback_token': body.get('callback_token'),
               'client': {k: body.get(k) for k in ('requested_by', 'external_id', 'notes') if body.get(k) is not None}}
    return request, warnings


class JobManager:
    def __init__(self, runtime: pipeline.Runtime, runner, max_queue: int = 20, status_file: Path | None = None):
        self.rt = runtime
        self.runner = runner
        self.max_queue = max_queue
        self.jobs_dir = runtime.root / 'jobs'
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = status_file or (runtime.root / 'status.json')
        self.jobs: dict[str, pipeline.Job] = {}
        self.queue: collections.deque = collections.deque()
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.current: pipeline.Job | None = None
        self.draining = False
        self.started = time.time()
        self.last_finished = time.time()
        self.counters = collections.Counter()
        self._gpu_cache = (0.0, None)
        self._recover()
        self.thread = threading.Thread(target=self._loop, name='job-runner', daemon=True)
        self.thread.start()

    # -- persistence -------------------------------------------------------------
    def _recover(self):
        for job_file in sorted(self.jobs_dir.glob('*/job.json')):
            try:
                job = pipeline.Job(job_file.parent)
            except Exception:
                continue
            self.jobs[job.id] = job
            if job.data.get('status') == 'running':
                job.update(status='failed', error='worker process restarted during the job', stage='interrupted',
                           finished_utc=pipeline.stamp())
                self.counters['failed'] += 1
            elif job.data.get('status') == 'queued':
                self.queue.append(job.id)
            else:
                self.counters[job.data.get('status', 'unknown')] += 1
        if self.queue:
            self.wake.set()

    # -- public ------------------------------------------------------------------
    def submit(self, request: dict) -> tuple[dict, int]:
        with self.lock:
            if self.draining:
                raise PermissionError('worker is draining')
            if len(self.queue) >= self.max_queue:
                raise OverflowError(f'queue is full ({self.max_queue})')
            job_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True)
            persisted = json.loads(json.dumps(request))
            portrait = persisted['portrait']
            if portrait.get('base64'):
                # keep job.json small: the bytes go to disk once, the record points at them
                data = base64.b64decode(portrait['base64'])
                ext = next((e for magic, e in pipeline.IMAGE_MAGIC if data.startswith(magic)), '.bin')
                upload = job_dir / f'upload{ext}'
                upload.write_bytes(data)
                persisted['portrait'] = {'path': str(upload), 'uploaded_bytes': len(data), 'filename': portrait.get('filename')}
            record = {'job_id': job_id, 'character_id': request['character_id'], 'status': 'queued', 'stage': 'queued',
                      'created_utc': pipeline.stamp(), 'updated_utc': pipeline.stamp(), 'request': persisted,
                      'progress': {}, 'result': None, 'error': None, 'api_version': API_VERSION}
            pipeline.write_json(job_dir / 'job.json', record)
            job = pipeline.Job(job_dir)
            self.jobs[job_id] = job
            self.queue.append(job_id)
            position = len(self.queue)
            self.wake.set()
            return job.data, position

    def get(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        with job.lock:
            data = json.loads(json.dumps(job.data, default=str))
        if data['status'] == 'queued':
            data['position'] = self.position(job_id)
        return data

    def position(self, job_id: str) -> int | None:
        with self.lock:
            try:
                return list(self.queue).index(job_id) + 1
            except ValueError:
                return None

    def list(self, limit: int = 50) -> list:
        with self.lock:
            ids = sorted(self.jobs, reverse=True)[:limit]
        out = []
        for job_id in ids:
            data = self.get(job_id) or {}
            out.append({k: data.get(k) for k in ('job_id', 'character_id', 'status', 'stage', 'created_utc', 'finished_utc', 'error', 'position')})
        return out

    def cancel(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        with self.lock:
            status = job.data.get('status')
            if status == 'queued':
                try:
                    self.queue.remove(job_id)
                except ValueError:
                    pass
                job.cancel_event.set()
                job.update(status='cancelled', stage='cancelled', error='cancelled while queued', finished_utc=pipeline.stamp())
                self.counters['cancelled'] += 1
                return job.data
            if status == 'running':
                job.cancel_event.set()
                proc = job.process
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                return {**job.data, 'cancel_requested': True}
            return job.data

    def set_draining(self, value: bool) -> dict:
        with self.lock:
            self.draining = value
        return self.worker_state()

    def artifacts(self, job_id: str) -> list:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        rows = []
        for path in sorted(job.dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
                rows.append({'path': path.relative_to(job.dir).as_posix(), 'bytes': path.stat().st_size})
        return rows

    def artifact_path(self, job_id: str, relative: str) -> Path:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        target = (job.dir / relative).resolve()
        if job.dir.resolve() not in target.parents or not target.is_file() or target.suffix.lower() not in ARTIFACT_SUFFIXES:
            raise FileNotFoundError(relative)
        return target

    # -- status ------------------------------------------------------------------
    def boot_status(self) -> dict:
        try:
            return json.loads(self.status_file.read_text())
        except (OSError, ValueError):
            return {'phase': 'unknown', 'ready': False, 'failed': False, 'message': 'no status file'}

    def gpu(self) -> dict | None:
        now = time.time()
        if now - self._gpu_cache[0] < 30:
            return self._gpu_cache[1]
        info = None
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version,compute_cap',
                                           '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
            if out:
                name, total, used, driver, cap = [x.strip() for x in out[0].split(',')]
                info = {'name': name, 'memory_total_mib': int(float(total)), 'memory_used_mib': int(float(used)),
                        'driver': driver, 'compute_cap': cap}
        except Exception:
            info = None
        self._gpu_cache = (now, info)
        return info

    def worker_state(self) -> dict:
        boot = self.boot_status()
        if os.environ.get('CHARACTER_WORKER_READY_OVERRIDE') == '1':
            boot = {**boot, 'ready': True, 'failed': False, 'phase': boot.get('phase') or 'ready'}
        with self.lock:
            current = self.current.id if self.current else None
            queue_depth = len(self.queue)
            draining = self.draining
        ready = bool(boot.get('ready')) and not boot.get('failed')
        versions = {}
        try:
            versions = json.loads((self.rt.root / 'versions.json').read_text())
        except (OSError, ValueError):
            pass
        return {'service': SERVICE, 'api_version': API_VERSION, 'status': 'failed' if boot.get('failed') else ('ready' if ready else 'booting'),
                'ready': ready, 'accepting': ready and not draining and queue_depth < self.max_queue, 'draining': draining,
                'boot': boot, 'busy': current is not None, 'current_job': current, 'queue_depth': queue_depth,
                'idle_seconds': 0 if current else round(time.time() - self.last_finished, 1), 'uptime_seconds': round(time.time() - self.started, 1),
                'counters': dict(self.counters), 'versions': versions, 'gpu': self.gpu(), 'host': pipeline.host_facts(),
                'recipe': pipeline.LOCK['recipe']['name'], 'max_queue': self.max_queue, 'time_utc': pipeline.stamp()}

    # -- runner ------------------------------------------------------------------
    def _loop(self):
        while True:
            self.wake.wait(timeout=5)
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    job_id = self.queue.popleft()
                    job = self.jobs[job_id]
                    if job.cancel_event.is_set() or job.data.get('status') != 'queued':
                        continue
                    self.current = job
                try:
                    result = self.runner(self.rt, job)
                    self.counters[result.get('status', 'unknown')] += 1
                except Exception as exc:  # the runner reports its own failures; this is a last resort
                    job.update(status='failed', error=f'runner crashed: {type(exc).__name__}: {exc}', finished_utc=pipeline.stamp())
                    self.counters['failed'] += 1
                finally:
                    with self.lock:
                        self.current = None
                        self.last_finished = time.time()


# ----------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = f'{SERVICE}/{API_VERSION}'
    manager: JobManager
    token_sources: list
    allow_local_paths = False
    fake = False

    def log_message(self, fmt, *args):  # one compact line per request
        sys.stderr.write(f'[{pipeline.stamp()}] {self.address_string()} {fmt % args}\n')

    # -- helpers
    def send_json(self, code: int, payload, extra_headers: dict | None = None):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def token(self) -> str | None:
        for source in self.token_sources:
            value = source()
            if value:
                return value
        return None

    def authorized(self) -> bool:
        expected = self.token()
        if not expected:
            return True
        header = self.headers.get('Authorization', '')
        return header.startswith('Bearer ') and header[7:].strip() == expected

    def read_json(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY:
            raise BadRequest(f'body larger than {MAX_BODY} bytes')
        raw = self.rfile.read(length) if length else b''
        if not raw:
            raise BadRequest('empty body')
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise BadRequest(f'invalid JSON: {exc}')

    def route(self):
        path = urlparse(self.path).path.rstrip('/') or '/'
        parts = path.strip('/').split('/')
        return path, parts

    # -- verbs
    def do_GET(self):
        path, parts = self.route()
        if path in ('/health', '/status', '/'):
            state = self.manager.worker_state()
            return self.send_json(200 if not state['boot'].get('failed') else 503, state)
        if not self.authorized():
            return self.send_json(401, {'error': 'unauthorized'})
        if path == '/queue':
            with self.manager.lock:
                queued = list(self.manager.queue)
                current = self.manager.current.id if self.manager.current else None
            return self.send_json(200, {'running': current, 'queued': queued, 'draining': self.manager.draining})
        if path == '/characters':
            return self.send_json(200, {'jobs': self.manager.list()})
        if len(parts) >= 2 and parts[0] == 'characters':
            job_id = parts[1]
            if len(parts) == 2:
                data = self.manager.get(job_id)
                return self.send_json(200, data) if data else self.send_json(404, {'error': 'unknown job', 'job_id': job_id})
            if parts[2] == 'artifacts':
                try:
                    if len(parts) == 3:
                        return self.send_json(200, {'job_id': job_id, 'artifacts': self.manager.artifacts(job_id)})
                    target = self.manager.artifact_path(job_id, '/'.join(parts[3:]))
                except KeyError:
                    return self.send_json(404, {'error': 'unknown job', 'job_id': job_id})
                except FileNotFoundError:
                    return self.send_json(404, {'error': 'no such artifact'})
                ctype = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(target.stat().st_size))
                self.end_headers()
                with target.open('rb') as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return None
        return self.send_json(404, {'error': 'not found', 'path': path})

    def do_POST(self):
        path, parts = self.route()
        if not self.authorized():
            return self.send_json(401, {'error': 'unauthorized'})
        try:
            if path in ('/createCharacter', '/characters'):
                state = self.manager.worker_state()
                if not state['ready']:
                    return self.send_json(503, {'error': 'worker not ready', 'boot': state['boot']})
                body = self.read_json()
                request, warnings = validate_request(body, self.allow_local_paths, self.fake)
                try:
                    record, position = self.manager.submit(request)
                except PermissionError as exc:
                    return self.send_json(503, {'error': str(exc)})
                except OverflowError as exc:
                    return self.send_json(429, {'error': str(exc)})
                return self.send_json(202, {'job_id': record['job_id'], 'character_id': record['character_id'], 'status': record['status'],
                                            'position': position, 'warnings': warnings, 'poll': f"/characters/{record['job_id']}"})
            if len(parts) == 3 and parts[0] == 'characters' and parts[2] == 'cancel':
                try:
                    return self.send_json(200, self.manager.cancel(parts[1]))
                except KeyError:
                    return self.send_json(404, {'error': 'unknown job', 'job_id': parts[1]})
            if path == '/worker/drain':
                return self.send_json(200, self.manager.set_draining(True))
            if path == '/worker/undrain':
                return self.send_json(200, self.manager.set_draining(False))
        except BadRequest as exc:
            return self.send_json(400, {'error': str(exc)})
        return self.send_json(404, {'error': 'not found', 'path': path})


def make_server(host: str, port: int, manager: JobManager, token_sources: list, allow_local_paths: bool, fake: bool) -> ThreadingHTTPServer:
    handler = type('BoundHandler', (Handler,), {'manager': manager, 'token_sources': token_sources,
                                                 'allow_local_paths': allow_local_paths, 'fake': fake})
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def token_sources_for(runtime: pipeline.Runtime) -> list:
    return [lambda: os.environ.get('CHARACTER_WORKER_TOKEN'),
            lambda: pipeline.parse_env_file(runtime.runtime_env_file).get('CHARACTER_WORKER_TOKEN')]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default=os.environ.get('CHARACTER_WORKER_HOST', '0.0.0.0'))
    p.add_argument('--port', type=int, default=int(os.environ.get('CHARACTER_WORKER_PORT', pipeline.LOCK['worker']['port'])))
    p.add_argument('--root', type=Path, help='worker state dir (default $CHARACTER_WORKER_ROOT or /workspace/character_worker)')
    p.add_argument('--workspace', type=Path, help='workspace with the venvs and H3 install (default /workspace)')
    p.add_argument('--max-queue', type=int, default=int(os.environ.get('CHARACTER_WORKER_MAX_QUEUE', 20)))
    p.add_argument('--fake', action='store_true', default=os.environ.get('CHARACTER_WORKER_FAKE_PIPELINE') == '1',
                   help='run the fake pipeline (no GPU, no S3); for tests and orchestrator rehearsals')
    p.add_argument('--fake-seconds', type=float, default=2.0)
    p.add_argument('--allow-local-paths', action='store_true', default=os.environ.get('CHARACTER_WORKER_ALLOW_LOCAL_PATHS') == '1')
    args = p.parse_args()
    runtime = pipeline.Runtime(workspace=args.workspace, root=args.root)
    runtime.root.mkdir(parents=True, exist_ok=True)
    runner = pipeline.FakePipeline(args.fake_seconds).run_job if args.fake else pipeline.run_job
    if args.fake:
        os.environ.setdefault('CHARACTER_WORKER_READY_OVERRIDE', '1')
    manager = JobManager(runtime, runner, max_queue=args.max_queue)
    sources = token_sources_for(runtime)
    if not any(s() for s in sources):
        sys.stderr.write('WARNING: no CHARACTER_WORKER_TOKEN configured; the API accepts unauthenticated requests\n')
    server = make_server(args.host, args.port, manager, sources, args.allow_local_paths, args.fake)
    sys.stderr.write(f'[{pipeline.stamp()}] {SERVICE} listening on {args.host}:{args.port} root={runtime.root} fake={args.fake}\n')

    def stop(signum, _frame):
        sys.stderr.write(f'[{pipeline.stamp()}] signal {signum}: shutting down\n')
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
