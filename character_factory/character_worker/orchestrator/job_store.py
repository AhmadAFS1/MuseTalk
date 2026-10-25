#!/usr/bin/env python3
"""SQLite state for the orchestrator: character jobs, Vast workers and an event log."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

JOB_ACTIVE = ('queued', 'submitting', 'running')
JOB_TERMINAL = ('done', 'failed', 'qc_failed', 'cancelled')
WORKER_ACTIVE = ('provisioning', 'booting', 'ready', 'busy', 'draining')
JSON_FIELDS = {'request', 'result', 'offer', 'instance', 'last_health', 'progress'}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, character_id TEXT, status TEXT, stage TEXT, request TEXT, portrait_path TEXT,
  worker_id TEXT, remote_job_id TEXT, result TEXT, progress TEXT, error TEXT, attempts INTEGER DEFAULT 0, priority INTEGER DEFAULT 0,
  created_utc TEXT, updated_utc TEXT, submitted_utc TEXT, finished_utc TEXT, webhook_status TEXT
);
CREATE TABLE IF NOT EXISTS workers (
  worker_id TEXT PRIMARY KEY, label TEXT, status TEXT, offer TEXT, instance TEXT, public_ip TEXT, public_port INTEGER, base_url TEXT,
  created_utc TEXT, updated_utc TEXT, ready_utc TEXT, last_health TEXT, last_seen_utc TEXT, busy_job_id TEXT, idle_since_utc TEXT,
  failures INTEGER DEFAULT 0, dph REAL, gpu_name TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, kind TEXT, job_id TEXT, worker_id TEXT, message TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, priority DESC, created_utc);
CREATE INDEX IF NOT EXISTS workers_status ON workers(status);
"""


def stamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _encode(fields: dict) -> dict:
    return {k: (json.dumps(v, default=str) if k in JSON_FIELDS and v is not None and not isinstance(v, str) else v) for k, v in fields.items()}


def _decode(row) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    for key in JSON_FIELDS:
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except ValueError:
                pass
    return data


class JobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.executescript(SCHEMA)

    # ---- jobs
    def add_job(self, job_id: str, request: dict, portrait_path: str | None, priority: int = 0) -> dict:
        now = stamp()
        with self.lock:
            self.conn.execute('INSERT INTO jobs (job_id, character_id, status, stage, request, portrait_path, priority, created_utc, updated_utc, attempts) '
                              'VALUES (?,?,?,?,?,?,?,?,?,0)',
                              (job_id, request['character_id'], 'queued', 'queued', json.dumps(request, default=str), portrait_path, priority, now, now))
        self.event('job.queued', f"character {request['character_id']}", job_id=job_id)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        with self.lock:
            return _decode(self.conn.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone())

    def list_jobs(self, statuses=None, limit: int = 200) -> list:
        with self.lock:
            if statuses:
                marks = ','.join('?' * len(statuses))
                rows = self.conn.execute(f'SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY priority DESC, created_utc ASC LIMIT ?',
                                         (*statuses, limit)).fetchall()
            else:
                rows = self.conn.execute('SELECT * FROM jobs ORDER BY created_utc DESC LIMIT ?', (limit,)).fetchall()
        return [_decode(r) for r in rows]

    def update_job(self, job_id: str, **fields) -> None:
        fields['updated_utc'] = stamp()
        enc = _encode(fields)
        assignments = ', '.join(f'{k}=?' for k in enc)
        with self.lock:
            self.conn.execute(f'UPDATE jobs SET {assignments} WHERE job_id=?', (*enc.values(), job_id))

    # ---- workers
    def add_worker(self, worker_id: str, label: str, status: str = 'provisioning', offer: dict | None = None, **fields) -> dict:
        now = stamp()
        base = {'worker_id': worker_id, 'label': label, 'status': status, 'offer': offer, 'created_utc': now, 'updated_utc': now,
                'dph': (offer or {}).get('dph_total'), 'gpu_name': (offer or {}).get('gpu_name'), 'failures': 0}
        base.update(fields)
        enc = _encode(base)
        with self.lock:
            self.conn.execute(f'INSERT INTO workers ({",".join(enc)}) VALUES ({",".join("?" * len(enc))})', tuple(enc.values()))
        self.event('worker.added', f'{label} status={status}', worker_id=worker_id)
        return self.get_worker(worker_id)

    def rename_worker(self, old_id: str, new_id: str) -> None:
        with self.lock:
            self.conn.execute('UPDATE workers SET worker_id=?, updated_utc=? WHERE worker_id=?', (new_id, stamp(), old_id))
            self.conn.execute('UPDATE jobs SET worker_id=? WHERE worker_id=?', (new_id, old_id))

    def get_worker(self, worker_id: str) -> dict | None:
        with self.lock:
            return _decode(self.conn.execute('SELECT * FROM workers WHERE worker_id=?', (worker_id,)).fetchone())

    def find_worker_by_label(self, label: str) -> dict | None:
        with self.lock:
            return _decode(self.conn.execute('SELECT * FROM workers WHERE label=?', (label,)).fetchone())

    def list_workers(self, statuses=None) -> list:
        with self.lock:
            if statuses:
                marks = ','.join('?' * len(statuses))
                rows = self.conn.execute(f'SELECT * FROM workers WHERE status IN ({marks}) ORDER BY created_utc', tuple(statuses)).fetchall()
            else:
                rows = self.conn.execute('SELECT * FROM workers ORDER BY created_utc DESC LIMIT 200').fetchall()
        return [_decode(r) for r in rows]

    def update_worker(self, worker_id: str, **fields) -> None:
        fields['updated_utc'] = stamp()
        enc = _encode(fields)
        assignments = ', '.join(f'{k}=?' for k in enc)
        with self.lock:
            self.conn.execute(f'UPDATE workers SET {assignments} WHERE worker_id=?', (*enc.values(), worker_id))

    # ---- events
    def event(self, kind: str, message: str, job_id: str | None = None, worker_id: str | None = None) -> None:
        with self.lock:
            self.conn.execute('INSERT INTO events (ts_utc, kind, job_id, worker_id, message) VALUES (?,?,?,?,?)',
                              (stamp(), kind, job_id, worker_id, message[:2000]))

    def events(self, limit: int = 100) -> list:
        with self.lock:
            rows = self.conn.execute('SELECT * FROM events ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        with self.lock:
            jobs = dict(self.conn.execute('SELECT status, COUNT(*) FROM jobs GROUP BY status').fetchall())
            workers = dict(self.conn.execute('SELECT status, COUNT(*) FROM workers GROUP BY status').fetchall())
        return {'jobs': jobs, 'workers': workers}
