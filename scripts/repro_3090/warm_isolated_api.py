#!/usr/bin/env python3
"""Warm only the fixed isolated A1 API plan; record actual cache residency.

No credentials, cloud operations, server restart, avatar preparation, or scored
load. This explicitly mutates process-local caches before a timed client run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/workspace/MuseTalk')
PLAN = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native/isolated_live_v1_plan.json'
PLAN_SHA = '40702fd5101ef6dc8839b4f3eb0b04120a753f18bcee12ed3218a1d0f59fda64'
BASE = 'http://127.0.0.1:8300'
GPU_UUID = 'GPU-5640f670-debe-ec22-1cfb-4b1f63bc1d53'
DEADLINE = dt.datetime(2026, 10, 8, 19, tzinfo=dt.timezone.utc)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def request(path, method='GET'):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=b'' if method == 'POST' else None)
    with urllib.request.urlopen(req, timeout=75) as response:
        raw = response.read(4 * 1024**2 + 1)
        require(len(raw) <= 4 * 1024**2, 'API response too large')
        return response.status, json.loads(raw)


def isolated_state(state):
    require(state.get('control_plane_requested') is False
            and state.get('control_plane_enabled') is False
            and state.get('registered') is False
            and state.get('local_ready') is True, 'API is not healthy and isolated')
    require(state.get('internal_port') == 8300 and state.get('instance_id') == '', 'wrong API identity')


def ready_cache(row, avatar_id):
    require(row.get('avatar_id') == avatar_id and row.get('status') == 'ready'
            and row.get('cached') is True and row.get('disk_prepared') is True
            and row.get('s3_enabled') is False and row.get('s3_restore_required') is False,
            'avatar not locally resident and ready')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'a830e00ce20c', 'not owned A1 host')
    require(dt.datetime.now(dt.timezone.utc) < DEADLINE, 'A1 deadline passed')
    require(args.out.is_absolute() and args.out.parent.resolve() == PLAN.parent
            and not args.out.exists() and not args.out.is_symlink(), 'fresh fixed report path required')
    require(not PLAN.is_symlink() and hashlib.sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA,
            'plan SHA mismatch')
    plan = json.loads(PLAN.read_text())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                  timeout=10, text=True).strip()
    require(gpu == GPU_UUID, 'GPU identity mismatch')
    cpu_quota = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    _, worker = request('/worker/state')
    isolated_state(worker)
    _, live_before = request('/webrtc/sessions/stats?view=lifetime&ring=256')
    server = live_before.get('server') or {}
    pid = server.get('pid')
    require(type(pid) is int and b'api_server.py' in Path(f'/proc/{pid}/cmdline').read_bytes(),
            'API PID is not local')
    require(live_before.get('total_sessions') == 0 and server.get('lifetime_counters') is True,
            'warmup must precede sessions with lifetime telemetry enabled')
    avatars = plan['avatars']
    require(len(avatars) == len(set(avatars)) == 16, 'fixed16-avatar plan required')
    data = {'schema': 'owned3090_isolated_api_warm_v1', 'status': 'IN_PROGRESS',
            'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'plan_sha256': PLAN_SHA, 'api_pid': pid, 'gpu_uuid': gpu, 'cpu_max': cpu_quota,
            'worker_before': worker, 'live_before': live_before, 'warmups': [],
            'startup_acceptance': False, 'release_ready': False,
            'note': 'Local existing-cache warm only, not fresh-instance or EC2 readiness. Cache warm does not run a speech inference batch.'}
    # Exclusive reservation means a failed retry never overwrites its evidence.
    with args.out.open('x') as output:
        json.dump(data, output, indent=2)
        output.write('\n')
    try:
        for avatar_id in avatars:
            require(dt.datetime.now(dt.timezone.utc) < DEADLINE, 'A1 deadline passed')
            start = time.monotonic()
            code, row = request(f'/avatars/{avatar_id}/cache/warm?batch_size=16&wait=true&timeout_seconds=60', 'POST')
            data['warmups'].append({'avatar_id': avatar_id, 'http_status': code,
                                    'operator_elapsed_seconds': time.monotonic() - start, 'response': row})
            require(code == 200, 'cache warm did not finish')
            ready_cache(row, avatar_id)
            idle = row.get('idle_frame_cache') or {}
            require(idle.get('ready') is True, 'idle-frame warm did not finish')
            print(json.dumps({'avatar_id': avatar_id, 'status': row['status'], 'completed': len(data['warmups']),
                              'idle_ready': idle.get('ready'), 'seconds': round(time.monotonic() - start, 3)}), flush=True)
        data['residency_after_all_warms'] = []
        for avatar_id in avatars:
            _, row = request(f'/avatars/{avatar_id}/cache/status')
            data['residency_after_all_warms'].append(row)
            ready_cache(row, avatar_id)
        _, stats = request('/stats')
        data['stats_after'] = {key: stats.get(key) for key in
                               ('gpu', 'cache', 'hls_scheduler', 'active_requests')}
        _, live = request('/webrtc/sessions/stats?view=lifetime&ring=256')
        data['live_after'] = live
        require(live['server']['idle_frame_cache']['pending_builds'] == 0, 'idle warm still pending')
        require(live.get('total_sessions') == 0 and stats.get('active_requests') == 0,
                'unexpected sessions or active requests')
        isolated_state(request('/worker/state')[1])
        data['status'] = 'PASS_SELECTED16_PROCESS_CACHE_RESIDENCY'
    except Exception as exc:
        data['status'] = 'INVALID'
        data['error_type'] = type(exc).__name__
        if isinstance(exc, ValueError):
            data['reason'] = str(exc)
    data['finished_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    # The only rewritten file is this helper's exclusively-created receipt.
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': data['status'], 'warmups': len(data['warmups'])}), flush=True)
    return 0 if data['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
