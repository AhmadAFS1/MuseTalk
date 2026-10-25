#!/usr/bin/env python3
"""charfactory: queue character jobs and drive Vast.ai workers from an EC2 box (or any Linux host).

  charfactory.py offers [--query Q] [--limit N]          what the offer query matches right now (read-only)
  charfactory.py queue --portrait FILE [--character-id ID] [--metadata FILE|JSON] [--poses ...] [--option k=v]...
  charfactory.py status [JOB_ID] | list | workers | events [--limit N]
  charfactory.py tick                                     one orchestration pass
  charfactory.py run [--interval S]                       loop forever
  charfactory.py serve [--host H] [--port P] [--interval S]   HTTP API for the Lingua backend + the loop
  charfactory.py cancel JOB_ID
  charfactory.py destroy-worker INSTANCE_ID | destroy-all --yes

Configuration comes from CHARFACTORY_* environment variables (see config.example.env); the
Vast API key from VAST_API_KEY or ~/.config/vastai/vast_api_key.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))   # engine/job_store/vast_client are plain modules in this directory
from job_store import JobStore  # noqa: E402
from engine import DEFAULT_QUERY, Config, Orchestrator, WorkerClient  # noqa: E402
from vast_client import VastClient, VastError, driver_at_least, parse_query  # noqa: E402
import character_api  # noqa: E402  (request validation shared with the worker)
import pipeline  # noqa: E402

IMAGE_MAGIC = pipeline.IMAGE_MAGIC


def load_config(args) -> Config:
    cfg = Config.from_env()
    if getattr(args, 'template_hash', None):
        cfg.template_hash = args.template_hash
    if getattr(args, 'dry_run', False):
        cfg.dry_run = True
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def build(args, need_vast: bool = True):
    cfg = load_config(args)
    store = JobStore(os.environ.get('CHARFACTORY_DB') or cfg.data_dir / 'charfactory.db')
    vast = VastClient() if need_vast else None
    orch = Orchestrator(cfg, store, vast, WorkerClient(cfg.worker_token, cfg.health_timeout_s))
    return cfg, store, vast, orch


def parse_option(text: str):
    key, _, raw = text.partition('=')
    if not key or not raw:
        raise argparse.ArgumentTypeError('--option needs key=value')
    value: object = raw
    if raw.lower() in ('true', 'false'):
        value = raw.lower() == 'true'
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                pass
    return key, value


def store_portrait(cfg: Config, job_id: str, data: bytes, filename: str | None) -> Path:
    ext = next((e for magic, e in IMAGE_MAGIC if data.startswith(magic)), None)
    if ext is None:
        raise ValueError('portrait is not PNG, JPEG or WebP')
    folder = cfg.data_dir / 'portraits'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{job_id}{ext}'
    path.write_bytes(data)
    return path


def enqueue(cfg: Config, store: JobStore, body: dict) -> tuple[dict, list]:
    """Validate like the worker does, persist the portrait bytes, store the job."""
    request, warnings = character_api.validate_request(body, allow_local_paths=False, fake=False)
    job_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    portrait_path = None
    if request['portrait'].get('base64'):
        portrait_path = str(store_portrait(cfg, job_id, base64.b64decode(request['portrait']['base64']), request['portrait'].get('filename')))
        request['portrait'] = {'filename': request['portrait'].get('filename'), 'stored': True}
    job = store.add_job(job_id, request, portrait_path, priority=int(body.get('priority') or 0))
    return job, warnings


# ----------------------------------------------------------------------------- commands

def cmd_offers(args):
    cfg = load_config(args)
    vast = VastClient()
    query = args.query or cfg.offer_query
    offers = vast.search_offers(parse_query(query), limit=args.limit, storage_gb=cfg.disk_gb)
    orch = Orchestrator(cfg, JobStore(':memory:'), vast)
    print(f'query: {query}\nclient filters: dph<={cfg.max_dph} disk>={cfg.disk_gb} driver>={cfg.min_driver_version} cpu_ram>={cfg.min_cpu_ram_mb} gpu_ram>={cfg.min_gpu_ram_mb}')
    print(f'{"ok":2} {"offer":>10} {"gpu":16} {"vram":>6} {"ram":>7} {"disk":>6} {"driver":>10} {"down":>7} {"rel":>5} {"$/h":>7} {"cc":>5} geo')
    for o in offers:
        ok = 'Y' if orch.offer_ok(o) else '-'
        print(f'{ok:2} {o.get("id"):>10} {str(o.get("gpu_name")):16} {int(o.get("gpu_ram") or 0):>6} {int(o.get("cpu_ram") or 0):>7} {int(o.get("disk_space") or 0):>6} '
              f'{str(o.get("driver_version")):>10} {float(o.get("inet_down") or 0):>7.0f} {float(o.get("reliability2") or 0):>5.3f} {float(o.get("dph_total") or 0):>7.3f} '
              f'{str(o.get("compute_cap")):>5} {o.get("geolocation")}')
    print(f'{sum(1 for o in offers if orch.offer_ok(o))} of {len(offers)} offers pass the client filters')


def cmd_queue(args):
    cfg, store, _, _ = build(args, need_vast=False)
    data = Path(args.portrait).read_bytes()
    metadata = {}
    if args.metadata:
        text = Path(args.metadata).read_text() if Path(args.metadata).is_file() else args.metadata
        metadata = json.loads(text)
    body = {'character_id': args.character_id, 'portrait': {'base64': base64.b64encode(data).decode(), 'filename': Path(args.portrait).name},
            'metadata': metadata, 'poses': args.poses, 'options': dict(args.option or []), 'priority': args.priority}
    if args.callback_url:
        body['callback_url'] = args.callback_url
    job, warnings = enqueue(cfg, store, body)
    print(json.dumps({'job_id': job['job_id'], 'character_id': job['character_id'], 'status': job['status'], 'warnings': warnings}, indent=2))


def cmd_status(args):
    _, store, _, _ = build(args, need_vast=False)
    if args.job_id:
        job = store.get_job(args.job_id)
        print(json.dumps(job, indent=2, default=str) if job else f'unknown job {args.job_id}')
        return
    print(json.dumps(store.summary(), indent=2))


def cmd_list(args):
    _, store, _, _ = build(args, need_vast=False)
    for job in store.list_jobs(limit=args.limit):
        print(f"{job['job_id']}  {job['status']:10} {job.get('stage') or '':12} {job['character_id']:24} worker={job.get('worker_id') or '-':>12} "
              f"attempts={job.get('attempts')} {('error=' + str(job.get('error'))[:60]) if job.get('error') else ''}")


def cmd_workers(args):
    _, store, _, _ = build(args, need_vast=False)
    for w in store.list_workers():
        health = w.get('last_health') or {}
        boot = health.get('boot') or {}
        print(f"{w['worker_id']:>14} {w['status']:12} {str(w.get('gpu_name')):16} ${float(w.get('dph') or 0):.3f}/h {w.get('base_url') or '-':28} "
              f"job={w.get('busy_job_id') or '-'} phase={boot.get('phase') or '-'} {str(w.get('notes') or '')[:60]}")


def cmd_events(args):
    _, store, _, _ = build(args, need_vast=False)
    for e in reversed(store.events(args.limit)):
        print(f"{e['ts_utc']} {e['kind']:22} {e.get('job_id') or '':26} {e.get('worker_id') or '':14} {e['message']}")


def cmd_tick(args):
    cfg, store, vast, orch = build(args)
    for problem in cfg.validate():
        print('WARNING:', problem, file=sys.stderr)
    print(json.dumps(orch.tick(), indent=2))


def cmd_run(args):
    cfg, store, vast, orch = build(args)
    for problem in cfg.validate():
        print('WARNING:', problem, file=sys.stderr)
    orch.run_forever(args.interval)


def cmd_cancel(args):
    _, store, vast, orch = build(args)
    print(json.dumps(orch.cancel_job(args.job_id), indent=2, default=str))


def cmd_destroy_worker(args):
    _, store, vast, orch = build(args)
    w = store.get_worker(args.instance_id) or {'worker_id': args.instance_id, 'label': '?', 'created_utc': None}
    orch.destroy_worker(w, 'operator request', drain=True)
    if w.get('busy_job_id'):
        orch._requeue(w['busy_job_id'], 'operator destroyed the worker')
    print('destroyed', args.instance_id)


def cmd_destroy_all(args):
    if not args.yes:
        raise SystemExit('refusing without --yes')
    _, store, vast, orch = build(args)
    print('destroyed', orch.shutdown_all(), 'workers')


# ----------------------------------------------------------------------------- serve

class ApiHandler(BaseHTTPRequestHandler):
    cfg: Config
    store: JobStore
    orch: Orchestrator
    token: str | None

    def log_message(self, fmt, *args):
        sys.stderr.write(f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {self.address_string()} {fmt % args}\n')

    def send_json(self, code, payload):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if not self.token:
            return True
        header = self.headers.get('Authorization', '')
        return header.startswith('Bearer ') and header[7:].strip() == self.token

    def do_GET(self):
        path = urlparse(self.path).path.rstrip('/') or '/'
        parts = path.strip('/').split('/')
        if path in ('/health', '/'):
            return self.send_json(200, {'service': 'charfactory', 'summary': self.store.summary(), 'template_hash': self.cfg.template_hash,
                                        'max_workers': self.cfg.max_workers, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        if not self.authorized():
            return self.send_json(401, {'error': 'unauthorized'})
        if path == '/characters':
            return self.send_json(200, {'jobs': [{k: j.get(k) for k in ('job_id', 'character_id', 'status', 'stage', 'worker_id', 'created_utc', 'finished_utc', 'error')}
                                                 for j in self.store.list_jobs(limit=100)]})
        if len(parts) == 2 and parts[0] == 'characters':
            job = self.store.get_job(parts[1])
            return self.send_json(200, job) if job else self.send_json(404, {'error': 'unknown job'})
        if path == '/workers':
            return self.send_json(200, {'workers': self.store.list_workers()})
        if path == '/events':
            return self.send_json(200, {'events': self.store.events(100)})
        return self.send_json(404, {'error': 'not found'})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip('/')
        parts = path.strip('/').split('/')
        if not self.authorized():
            return self.send_json(401, {'error': 'unauthorized'})
        length = int(self.headers.get('Content-Length') or 0)
        if length > 48 * 1024 * 1024:
            return self.send_json(413, {'error': 'body too large'})
        raw = self.rfile.read(length) if length else b''
        if path in ('/createCharacter', '/characters'):
            try:
                body = json.loads(raw or b'{}')
                job, warnings = enqueue(self.cfg, self.store, body)
            except (ValueError, character_api.BadRequest) as exc:
                return self.send_json(400, {'error': str(exc)})
            return self.send_json(202, {'job_id': job['job_id'], 'character_id': job['character_id'], 'status': job['status'], 'warnings': warnings,
                                        'poll': f"/characters/{job['job_id']}"})
        if len(parts) == 3 and parts[0] == 'characters' and parts[2] == 'cancel':
            job = self.orch.cancel_job(parts[1])
            return self.send_json(200, job) if job else self.send_json(404, {'error': 'unknown job'})
        return self.send_json(404, {'error': 'not found'})


def cmd_serve(args):
    cfg, store, vast, orch = build(args)
    for problem in cfg.validate():
        print('WARNING:', problem, file=sys.stderr)
    handler = type('BoundApi', (ApiHandler,), {'cfg': cfg, 'store': store, 'orch': orch, 'token': os.environ.get('CHARFACTORY_API_TOKEN')})
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    threading.Thread(target=orch.run_forever, args=(args.interval,), daemon=True, name='orchestrator').start()
    print(f'charfactory serving on {args.host}:{args.port}; orchestrator tick every {args.interval}s', flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--template-hash', help='override CHARFACTORY_TEMPLATE_HASH')
    p.add_argument('--dry-run', action='store_true', help='never rent or destroy instances')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('offers'); s.add_argument('--query'); s.add_argument('--limit', type=int, default=30); s.set_defaults(func=cmd_offers)
    s = sub.add_parser('queue'); s.add_argument('--portrait', required=True); s.add_argument('--character-id'); s.add_argument('--metadata')
    s.add_argument('--poses', nargs='+', default=['idle', 'talking', 'smiling']); s.add_argument('--option', type=parse_option, action='append')
    s.add_argument('--priority', type=int, default=0); s.add_argument('--callback-url'); s.set_defaults(func=cmd_queue)
    s = sub.add_parser('status'); s.add_argument('job_id', nargs='?'); s.set_defaults(func=cmd_status)
    s = sub.add_parser('list'); s.add_argument('--limit', type=int, default=50); s.set_defaults(func=cmd_list)
    s = sub.add_parser('workers'); s.set_defaults(func=cmd_workers)
    s = sub.add_parser('events'); s.add_argument('--limit', type=int, default=50); s.set_defaults(func=cmd_events)
    s = sub.add_parser('tick'); s.set_defaults(func=cmd_tick)
    s = sub.add_parser('run'); s.add_argument('--interval', type=int, default=None); s.set_defaults(func=cmd_run)
    s = sub.add_parser('serve'); s.add_argument('--host', default='0.0.0.0'); s.add_argument('--port', type=int, default=8020)
    s.add_argument('--interval', type=int, default=20); s.set_defaults(func=cmd_serve)
    s = sub.add_parser('cancel'); s.add_argument('job_id'); s.set_defaults(func=cmd_cancel)
    s = sub.add_parser('destroy-worker'); s.add_argument('instance_id'); s.set_defaults(func=cmd_destroy_worker)
    s = sub.add_parser('destroy-all'); s.add_argument('--yes', action='store_true'); s.set_defaults(func=cmd_destroy_all)
    args = p.parse_args()
    try:
        args.func(args)
    except VastError as exc:
        print(f'Vast API error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
