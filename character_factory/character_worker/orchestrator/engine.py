#!/usr/bin/env python3
"""EC2-side orchestrator engine for character creation on Vast.ai workers (driven by charfactory.py).

One `tick()` does, in order:
  reconcile_workers  compare the store with Vast; adopt instances we created but lost track of;
                     poll /health; promote booting workers to ready; destroy and requeue on failure
  dispatch           hand queued jobs to idle ready workers (POST /createCharacter); rent a new
                     worker from the template hash when none is free and max_workers allows
  poll_running       GET /characters/{id} on the worker; store results; fire the result webhook
  retire_idle        drain + destroy workers idle past the TTL or older than the lifetime cap

Everything is idempotent against a restart: state lives in SQLite (job_store.py); a Vast create
call that times out after renting is recovered by the instance label on the next tick.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_store import JOB_TERMINAL, WORKER_ACTIVE, JobStore, stamp  # noqa: E402
from vast_client import VastClient, VastError, driver_at_least, mapped_port, parse_query  # noqa: E402

DEFAULT_QUERY = ('num_gpus=1 gpu_ram>=12 cpu_ram>=30 disk_space>=200 driver_version>=580.65.6 cuda_vers>=13.0 '
                 'inet_down>=500 reliability>0.98 compute_cap>=890 compute_cap<1000 rented=False')


def _cast(kind, value: str):
    if kind is bool:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    if kind is int:
        return int(float(value))
    if kind is float:
        return float(value)
    if kind is Path:
        return Path(value).expanduser()
    return value


@dataclass
class Config:
    template_hash: str = ''
    offer_query: str = DEFAULT_QUERY
    disk_gb: float = 200.0
    max_workers: int = 2
    max_dph: float = 1.5
    worker_port: int = 8010
    worker_token: str | None = None
    ready_timeout_s: int = 3600
    job_timeout_s: int = 10800
    idle_ttl_s: int = 900
    max_worker_lifetime_s: int = 43200
    health_timeout_s: int = 8
    provision_retries: int = 2
    submit_failures_max: int = 3
    unreachable_polls_max: int = 10
    pending_timeout_s: int = 900
    max_boot_failures: int = 3          # consecutive workers that never became ready before renting pauses
    provision_backoff_s: int = 1800     # how long renting stays paused after that streak
    result_webhook: str | None = None
    result_webhook_token: str | None = None
    label_prefix: str = 'charfactory'
    poll_interval_s: int = 20
    min_driver_version: str = '580.65'
    min_cpu_ram_mb: int = 29000
    min_gpu_ram_mb: int = 12000
    data_dir: Path = Path('~/.charfactory').expanduser()
    extra_env: dict = field(default_factory=dict)
    dry_run: bool = False

    @classmethod
    def from_env(cls, environ=os.environ) -> 'Config':
        cfg = cls()
        for f in fields(cls):
            key = 'CHARFACTORY_' + f.name.upper()
            if key in environ and environ[key] != '':
                if f.name == 'extra_env':
                    cfg.extra_env = json.loads(environ[key])
                else:
                    setattr(cfg, f.name, _cast(f.type if not isinstance(f.type, str) else {'bool': bool, 'int': int, 'float': float, 'Path': Path}.get(f.type.split(' ')[0], str), environ[key]))
        return cfg

    def validate(self) -> list:
        problems = []
        if not self.template_hash:
            problems.append('CHARFACTORY_TEMPLATE_HASH is required (the Vast template that runs vast_onstart_character.sh)')
        if not self.worker_token:
            problems.append('CHARFACTORY_WORKER_TOKEN is empty: the workers accept unauthenticated requests')
        return problems


class WorkerClient:
    """HTTP client for the worker API (character_api.py)."""

    def __init__(self, token: str | None, timeout: int = 8):
        self.token, self.timeout = token, timeout

    def _call(self, method: str, url: str, body=None, timeout: int | None = None):
        data = None if body is None else json.dumps(body, default=str).encode()
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, json.loads(resp.read() or b'null')
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read() or b'null')
            except ValueError:
                payload = None
            return exc.code, payload

    def health(self, base: str) -> dict:
        status, data = self._call('GET', f'{base}/health')
        if status >= 500 and not isinstance(data, dict):
            raise RuntimeError(f'health HTTP {status}')
        return data or {}

    def create(self, base: str, payload: dict):
        return self._call('POST', f'{base}/createCharacter', payload, timeout=120)

    def job(self, base: str, remote_id: str):
        return self._call('GET', f'{base}/characters/{remote_id}')

    def cancel(self, base: str, remote_id: str):
        return self._call('POST', f'{base}/characters/{remote_id}/cancel')

    def drain(self, base: str):
        return self._call('POST', f'{base}/worker/drain')


def age_seconds(iso: str | None, now: float) -> float:
    if not iso:
        return 0.0
    try:
        return now - time.mktime(time.strptime(iso, '%Y-%m-%dT%H:%M:%SZ')) + (time.mktime(time.gmtime(0)) - 0)
    except ValueError:
        return 0.0


def _utc_epoch(iso: str) -> float:
    import calendar
    return calendar.timegm(time.strptime(iso, '%Y-%m-%dT%H:%M:%SZ'))


class Orchestrator:
    def __init__(self, cfg: Config, store: JobStore, vast, workers: WorkerClient | None = None, now=time.time, log=None):
        self.cfg, self.store, self.vast = cfg, store, vast
        self.workers = workers or WorkerClient(cfg.worker_token, cfg.health_timeout_s)
        self.now = now
        self.log = log or (lambda message: print(f'[{stamp()}] {message}', flush=True))

    # ------------------------------------------------------------------ helpers
    def _age(self, iso: str | None) -> float:
        return self.now() - _utc_epoch(iso) if iso else 0.0

    def _event(self, kind: str, message: str, job_id=None, worker_id=None):
        self.store.event(kind, message, job_id=job_id, worker_id=worker_id)
        self.log(f'{kind} {f"job={job_id} " if job_id else ""}{f"worker={worker_id} " if worker_id else ""}{message}')

    # ------------------------------------------------------------------ tick
    def tick(self) -> dict:
        for phase in (self.reconcile_workers, self.dispatch, self.poll_running, self.retire_idle):
            try:
                phase()
            except Exception as exc:  # keep the loop alive; the next tick retries
                self._event('tick.error', f'{phase.__name__}: {type(exc).__name__}: {exc}')
        return self.store.summary()

    def run_forever(self, interval: int | None = None):
        interval = interval or self.cfg.poll_interval_s
        while True:
            self.tick()
            time.sleep(interval)

    # ------------------------------------------------------------------ workers
    def reconcile_workers(self):
        try:
            instances = {str(i['id']): i for i in self.vast.list_instances()}
        except VastError as exc:
            self._event('vast.error', f'list_instances: {exc}')
            return
        prefix = self.cfg.label_prefix + '-'
        for inst in instances.values():
            label = inst.get('label') or ''
            if label.startswith(prefix) and not self.store.get_worker(str(inst['id'])):
                pending = self.store.find_worker_by_label(label)
                if pending and str(pending['worker_id']).startswith('pending-'):
                    self.store.rename_worker(pending['worker_id'], str(inst['id']))
                    self.store.update_worker(str(inst['id']), status='booting', instance=inst, notes='adopted after a create timeout')
                    self._event('worker.adopted', label, worker_id=str(inst['id']))
                elif not pending:
                    self.store.add_worker(str(inst['id']), label, status='booting', instance=inst, notes='adopted: created outside this store')
                    self._event('worker.adopted', label + ' (unknown origin)', worker_id=str(inst['id']))
        for w in self.store.list_workers(WORKER_ACTIVE):
            wid = w['worker_id']
            if str(wid).startswith('pending-'):
                if self._age(w['created_utc']) > self.cfg.pending_timeout_s:
                    self.store.update_worker(wid, status='failed', notes='create never materialised')
                    self._event('worker.failed', 'pending create timed out', worker_id=wid)
                continue
            inst = instances.get(str(wid))
            if inst is None:
                self._fail_worker(w, 'instance no longer exists on Vast', destroy=False)
                continue
            self.store.update_worker(wid, instance={k: inst.get(k) for k in ('id', 'actual_status', 'cur_state', 'intended_status', 'status_msg', 'gpu_name',
                                                                                'public_ipaddr', 'ports', 'machine_id', 'dph_total', 'label', 'start_date')},
                                     last_seen_utc=stamp(), gpu_name=inst.get('gpu_name'), dph=inst.get('dph_total'))
            actual, cur, intended = inst.get('actual_status'), inst.get('cur_state'), inst.get('intended_status')
            if actual in ('exited', 'error', 'offline') or cur == 'stopped' or intended == 'stopped':
                self._fail_worker(w, f'instance state {actual}/{cur}/{intended}: {str(inst.get("status_msg") or "")[:200]}')
                continue
            if w['status'] not in ('busy',) and self._age(w['created_utc']) > self.cfg.max_worker_lifetime_s:
                self.destroy_worker(w, 'max lifetime reached', drain=True)
                continue
            endpoint = mapped_port(inst, self.cfg.worker_port)
            if endpoint is None:
                self._boot_wait(w, f'no port mapping yet (status {actual})')
                continue
            base = f'http://{endpoint[0]}:{endpoint[1]}'
            if base != w.get('base_url'):
                self.store.update_worker(wid, public_ip=endpoint[0], public_port=endpoint[1], base_url=base)
            try:
                health = self.workers.health(base)
            except Exception as exc:
                self._boot_wait(w, f'health unreachable: {exc}')
                continue
            self.store.update_worker(wid, last_health=health)
            boot = health.get('boot') or {}
            if boot.get('failed'):
                self._fail_worker(w, f"worker boot failed: {boot.get('phase')}: {boot.get('message')}")
                continue
            if health.get('draining') and not health.get('busy') and w['status'] != 'busy':
                self.destroy_worker(w, 'worker reports draining and is idle')
                continue
            if health.get('ready'):
                if w['status'] in ('provisioning', 'booting'):
                    self.store.update_worker(wid, status='ready', ready_utc=stamp(), idle_since_utc=stamp(), notes='ready', failures=0)
                    self._event('worker.ready', f"{health.get('gpu', {}) and health['gpu'].get('name')} at {base} after {int(self._age(w['created_utc']))}s", worker_id=wid)
            else:
                self._boot_wait(w, f"booting: {boot.get('phase')}: {boot.get('message')}")

    def _boot_wait(self, w: dict, note: str):
        if w['status'] in ('provisioning', 'booting') and self._age(w['created_utc']) > self.cfg.ready_timeout_s:
            self._fail_worker(w, f'not ready after {self.cfg.ready_timeout_s}s; last: {note}')
        else:
            self.store.update_worker(w['worker_id'], notes=note[:500])

    def _fail_worker(self, w: dict, reason: str, destroy: bool = True):
        self._event('worker.failed', reason, worker_id=w['worker_id'])
        if destroy:
            self.destroy_worker(w, reason)
        else:
            self.store.update_worker(w['worker_id'], status='dead', notes=reason[:500], busy_job_id=None)
        if w.get('busy_job_id'):
            self._requeue(w['busy_job_id'], reason)

    def _requeue(self, job_id: str, reason: str):
        job = self.store.get_job(job_id)
        if not job or job['status'] in JOB_TERMINAL:
            return
        attempts = int(job.get('attempts') or 0) + 1
        if attempts > self.cfg.provision_retries:
            self.store.update_job(job_id, status='failed', error=f'worker failed {attempts} times; last: {reason}', finished_utc=stamp(), attempts=attempts)
            self._event('job.failed', f'gave up after {attempts} worker failures', job_id=job_id)
            self._webhook(job_id)
        else:
            self.store.update_job(job_id, status='queued', worker_id=None, remote_job_id=None, attempts=attempts, stage='queued')
            self._event('job.requeued', f'attempt {attempts}: {reason}', job_id=job_id)

    def destroy_worker(self, w: dict, reason: str, drain: bool = False):
        wid = w['worker_id']
        if drain and w.get('base_url'):
            try:
                self.workers.drain(w['base_url'])
            except Exception:
                pass
        if not str(wid).startswith('pending-') and not self.cfg.dry_run:
            try:
                self.vast.destroy_instance(int(wid))
            except (VastError, ValueError) as exc:
                self._event('vast.error', f'destroy {wid}: {exc}', worker_id=wid)
        self.store.update_worker(wid, status='dead', notes=reason[:500], busy_job_id=None)
        self._event('worker.destroyed', reason, worker_id=wid)

    BOOT_FAILURE_NOTES = ('worker boot failed', 'not ready after', 'instance state', 'create never materialised')

    def _boot_failure_streak(self) -> tuple[int, str | None]:
        """Consecutive most-recent workers that died without ever becoming ready, and when the last one died."""
        streak, last = 0, None
        for w in self.store.list_workers():            # newest first
            if w.get('ready_utc'):
                break
            if w['status'] in ('dead', 'failed') and str(w.get('notes') or '').startswith(self.BOOT_FAILURE_NOTES):
                streak += 1
                last = last or w.get('updated_utc')
            elif w['status'] in ('dead', 'failed'):
                continue
            else:
                break                                   # a live worker interrupts the streak
        return streak, last

    def provision_worker(self):
        if self.cfg.dry_run:
            self._event('provision.dry_run', 'would rent a worker now')
            return None
        streak, last = self._boot_failure_streak()
        if streak >= self.cfg.max_boot_failures and last and self._age(last) < self.cfg.provision_backoff_s:
            self._event('provision.paused', f'{streak} consecutive workers never became ready; renting paused for '
                                            f'{int(self.cfg.provision_backoff_s - self._age(last))}s more (fix the template/boot, or destroy-all and requeue)')
            return None
        try:
            offers = self.vast.search_offers(parse_query(self.cfg.offer_query), limit=50, storage_gb=self.cfg.disk_gb)
        except VastError as exc:
            self._event('vast.error', f'search_offers: {exc}')
            return None
        candidates = [o for o in offers if self.offer_ok(o)]
        if not candidates:
            self._event('provision.no_offers', f'{len(offers)} offers matched the query, none passed the client filters (max ${self.cfg.max_dph}/h, '
                                               f'driver >= {self.cfg.min_driver_version}, {self.cfg.disk_gb} GB disk)')
            return None
        offer = sorted(candidates, key=lambda o: float(o.get('dph_total') or 0))[0]
        label = f'{self.cfg.label_prefix}-{uuid.uuid4().hex[:10]}'
        pending_id = f'pending-{label}'
        self.store.add_worker(pending_id, label, status='provisioning', offer={k: offer.get(k) for k in (
            'id', 'gpu_name', 'gpu_ram', 'cpu_ram', 'disk_space', 'driver_version', 'cuda_max_good', 'inet_down', 'reliability2',
            'dph_total', 'geolocation', 'machine_id', 'compute_cap')})
        try:
            instance_id = self.vast.create_instance(offer['id'], template_hash=self.cfg.template_hash, disk_gb=self.cfg.disk_gb, label=label,
                                                    env=self.cfg.extra_env or None)
        except VastError as exc:
            if exc.status and 400 <= exc.status < 500 and exc.status != 429:
                self.store.update_worker(pending_id, status='failed', notes=f'create rejected: {exc}'[:500])
                self._event('provision.rejected', str(exc), worker_id=pending_id)
            else:
                self.store.update_worker(pending_id, notes=f'create unconfirmed: {exc}'[:500])
                self._event('provision.unconfirmed', f'{exc}; will adopt by label {label} if it exists', worker_id=pending_id)
            return None
        self.store.rename_worker(pending_id, str(instance_id))
        self.store.update_worker(str(instance_id), status='booting')
        self._event('worker.created', f"instance {instance_id} {offer.get('gpu_name')} ${float(offer.get('dph_total') or 0):.3f}/h {offer.get('geolocation')}", worker_id=str(instance_id))
        return str(instance_id)

    def offer_ok(self, offer: dict) -> bool:
        return (float(offer.get('dph_total') or 1e9) <= self.cfg.max_dph and float(offer.get('disk_space') or 0) >= self.cfg.disk_gb
                and driver_at_least(offer.get('driver_version'), self.cfg.min_driver_version)
                and float(offer.get('cpu_ram') or 0) >= self.cfg.min_cpu_ram_mb and float(offer.get('gpu_ram') or 0) >= self.cfg.min_gpu_ram_mb
                and not offer.get('rented'))

    # ------------------------------------------------------------------ jobs
    def dispatch(self):
        queued = self.store.list_jobs(['queued'])
        if not queued:
            return
        ready = [w for w in self.store.list_workers(['ready']) if not w.get('busy_job_id')]
        submitted = 0
        for job in queued:
            if not ready:
                break
            worker = ready.pop(0)
            if self._submit(job, worker):
                submitted += 1
        remaining = len(queued) - submitted
        if remaining <= 0:
            return
        active = self.store.list_workers(WORKER_ACTIVE)
        coming = [w for w in active if w['status'] in ('provisioning', 'booting')]
        if len(active) < self.cfg.max_workers and len(coming) < remaining:
            self.provision_worker()

    def _payload(self, job: dict) -> dict:
        request = json.loads(json.dumps(job['request']))
        if job.get('portrait_path'):
            path = Path(job['portrait_path'])
            request['portrait'] = {'base64': base64.b64encode(path.read_bytes()).decode(), 'filename': path.name}
        request.setdefault('requested_by', 'charfactory')
        request['external_id'] = job['job_id']
        return request

    def _submit(self, job: dict, worker: dict) -> bool:
        wid, job_id = worker['worker_id'], job['job_id']
        self.store.update_job(job_id, status='submitting', worker_id=wid)
        try:
            status, resp = self.workers.create(worker['base_url'], self._payload(job))
        except Exception as exc:
            failures = int(worker.get('failures') or 0) + 1
            self.store.update_job(job_id, status='queued', worker_id=None)
            self.store.update_worker(wid, failures=failures, notes=f'submit error: {exc}'[:500])
            self._event('job.submit_error', f'{exc} (worker failure {failures})', job_id=job_id, worker_id=wid)
            if failures >= self.cfg.submit_failures_max:
                self._fail_worker(worker, f'{failures} submit failures')
            return False
        if status == 202 and isinstance(resp, dict) and resp.get('job_id'):
            self.store.update_job(job_id, status='running', remote_job_id=resp['job_id'], submitted_utc=stamp(), worker_id=wid, stage='submitted')
            self.store.update_worker(wid, status='busy', busy_job_id=job_id, idle_since_utc=None, failures=0)
            self._event('job.submitted', f"remote {resp['job_id']} warnings={resp.get('warnings') or []}", job_id=job_id, worker_id=wid)
            return True
        if status == 400:
            self.store.update_job(job_id, status='failed', error=f"worker rejected the request: {(resp or {}).get('error')}", finished_utc=stamp(), worker_id=None)
            self._event('job.failed', f'rejected: {(resp or {}).get("error")}', job_id=job_id)
            self._webhook(job_id)
            return False
        self.store.update_job(job_id, status='queued', worker_id=None)
        self.store.update_worker(wid, notes=f'submit HTTP {status}: {str(resp)[:200]}')
        self._event('job.submit_retry', f'HTTP {status} {str(resp)[:200]}', job_id=job_id, worker_id=wid)
        return False

    def poll_running(self):
        for job in self.store.list_jobs(['running']):
            job_id = job['job_id']
            worker = self.store.get_worker(job['worker_id']) if job.get('worker_id') else None
            if not worker or not worker.get('base_url'):
                continue
            wid = worker['worker_id']
            if self._age(job.get('submitted_utc')) > self.cfg.job_timeout_s:
                try:
                    self.workers.cancel(worker['base_url'], job['remote_job_id'])
                except Exception:
                    pass
                self.store.update_job(job_id, status='failed', error=f'timed out after {self.cfg.job_timeout_s}s on the worker', finished_utc=stamp())
                self._event('job.failed', 'timeout', job_id=job_id, worker_id=wid)
                self._webhook(job_id)
                self._fail_worker({**worker, 'busy_job_id': None}, 'job timeout; worker state unknown')
                continue
            try:
                status, data = self.workers.job(worker['base_url'], job['remote_job_id'])
            except Exception as exc:
                failures = int(worker.get('failures') or 0) + 1
                self.store.update_worker(wid, failures=failures, notes=f'poll error: {exc}'[:500])
                if failures >= self.cfg.unreachable_polls_max:
                    self._fail_worker(worker, f'unreachable for {failures} polls')
                continue
            self.store.update_worker(wid, failures=0)
            if status == 404:
                self.store.update_job(job_id, status='failed', error='worker does not know the job (restarted?)', finished_utc=stamp())
                self.store.update_worker(wid, status='ready', busy_job_id=None, idle_since_utc=stamp())
                self._event('job.failed', 'remote job unknown', job_id=job_id, worker_id=wid)
                self._webhook(job_id)
                continue
            if not isinstance(data, dict):
                continue
            remote_status = data.get('status')
            self.store.update_job(job_id, stage=data.get('stage'), progress=data.get('progress'))
            if remote_status in JOB_TERMINAL:
                self.store.update_job(job_id, status=remote_status, result=data.get('result'), error=data.get('error'), finished_utc=stamp(),
                                      stage=data.get('stage'))
                self.store.update_worker(wid, status='ready', busy_job_id=None, idle_since_utc=stamp())
                self._event(f'job.{remote_status}', (data.get('error') or 'ok')[:300], job_id=job_id, worker_id=wid)
                self._webhook(job_id)

    def cancel_job(self, job_id: str) -> dict | None:
        job = self.store.get_job(job_id)
        if not job:
            return None
        if job['status'] in ('queued', 'submitting'):
            self.store.update_job(job_id, status='cancelled', error='cancelled before dispatch', finished_utc=stamp())
            self._event('job.cancelled', 'before dispatch', job_id=job_id)
        elif job['status'] == 'running':
            worker = self.store.get_worker(job['worker_id'])
            if worker and worker.get('base_url'):
                try:
                    self.workers.cancel(worker['base_url'], job['remote_job_id'])
                except Exception as exc:
                    self._event('job.cancel_error', str(exc), job_id=job_id)
        return self.store.get_job(job_id)

    def retire_idle(self):
        queued = self.store.list_jobs(['queued'])
        for w in self.store.list_workers(['ready']):
            if w.get('busy_job_id'):
                continue
            idle = self._age(w.get('idle_since_utc') or w.get('ready_utc') or w.get('created_utc'))
            if (not queued and idle > self.cfg.idle_ttl_s):
                self.destroy_worker(w, f'idle for {int(idle)}s', drain=True)
            elif self._age(w['created_utc']) > self.cfg.max_worker_lifetime_s:
                self.destroy_worker(w, 'max lifetime reached', drain=True)

    def shutdown_all(self, reason: str = 'operator shutdown') -> int:
        count = 0
        for w in self.store.list_workers(WORKER_ACTIVE):
            self.destroy_worker(w, reason, drain=True)
            if w.get('busy_job_id'):
                self._requeue(w['busy_job_id'], reason)
            count += 1
        return count

    # ------------------------------------------------------------------ webhook
    def _webhook(self, job_id: str):
        url = self.cfg.result_webhook
        if not url:
            return
        job = self.store.get_job(job_id)
        payload = {'event': 'character.job', 'job': job}
        headers = {'Content-Type': 'application/json', 'X-Character-Job': job_id}
        if self.cfg.result_webhook_token:
            headers['Authorization'] = f'Bearer {self.cfg.result_webhook_token}'
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(url, data=json.dumps(payload, default=str).encode(), headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=20) as resp:
                    self.store.update_job(job_id, webhook_status=f'{resp.status} @ {stamp()}')
                    return
            except Exception as exc:
                self.store.update_job(job_id, webhook_status=f'error attempt {attempt}: {exc}'[:300])
                time.sleep(min(5, attempt))
