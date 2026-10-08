#!/usr/bin/env python3
"""Execute one fixed, gated, same-worker live stage after active cache checks.

Never changes server settings, starts a server, registers a worker, or rents.
The active stage's idle clips must remain resident, not just once be warmed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import warm_isolated_api as warm

STAGES = ('s0_n1', 's0_n3', 'ramp_n5', 'ramp_n10', 'ramp_n15')


def require(value, reason):
    warm.require(value, reason)


def require_untraced_api(pid, proc_root=Path('/proc')):
    """Refuse timing evidence while any surviving API thread has a tracer."""
    task_root = proc_root / str(pid) / 'task'
    threads = list(task_root.iterdir())
    require(bool(threads), 'API thread inventory missing')
    checked = 0
    for thread in threads:
        try:
            status = (thread / 'status').read_text()
        except FileNotFoundError:
            # A thread may exit during the read; the main process may not.
            require(thread.name != str(pid), 'API process exited during preflight')
            continue
        fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
        require(fields.get('TracerPid', '').strip() == '0', 'API thread traced; scored timing forbidden')
        checked += 1
    require(checked > 0 and (task_root / str(pid)).is_dir(), 'API process missing during tracer check')
    return {'status': 'PASS_ALL_SURVIVING_API_THREADS_UNTRACED', 'threads_checked': checked}


def stage_argv(plan, stage, attempt):
    argv = list(plan['client_stage_argv'][stage])
    require(attempt == 'h264_offer_v2', 'explicit fixed codec-attempt ID required')
    label = stage + '_' + attempt
    argv[argv.index('--label') + 1] = label
    base = Path(plan['server_env']['MUSETALK_RUNTIME_DIR']) / 'client_attempts' / attempt / stage
    for option, folder in (('--out-dir', 'lt2'), ('--trace-dir', 'traces')):
        argv[argv.index(option) + 1] = str(base / folder)
    return argv


def previous_summary(plan, stage, attempt):
    previous = STAGES[STAGES.index(stage) - 1]
    argv = stage_argv(plan, previous, attempt)
    folder = Path(argv[argv.index('--out-dir') + 1])
    label = argv[argv.index('--label') + 1]
    level = int(argv[argv.index('--levels') + 1])
    file = folder / f'{label}_n{level:02d}.json'
    require(file.is_file() and not file.is_symlink(), 'exact previous-stage summary required')
    value = json.loads(file.read_text())
    require(value.get('level') == level, 'previous-stage level mismatch')
    # Canonical coordinator's summary file shape is checked before launching
    # any next level; do not infer PASS from an exit code.
    return file, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--attempt', choices=('h264_offer_v2',), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'a830e00ce20c', 'not owned A1 host')
    require(dt.datetime.now(dt.timezone.utc) < warm.DEADLINE, 'A1 deadline passed')
    require(hashlib.sha256(warm.PLAN.read_bytes()).hexdigest() == warm.PLAN_SHA, 'plan SHA mismatch')
    require(args.out.is_absolute() and args.out.parent.resolve() == warm.PLAN.parent
            and not args.out.exists() and not args.out.is_symlink(), 'fresh fixed receipt path required')
    plan = json.loads(warm.PLAN.read_text())
    argv = stage_argv(plan, args.stage, args.attempt)
    level = int(argv[argv.index('--levels') + 1])
    selected = plan['avatars'][:level]
    require(len(selected) == level, 'missing unique avatars')
    for option in ('--out-dir', '--trace-dir'):
        require(not Path(argv[argv.index(option) + 1]).exists(), 'client output already exists')
    prior = None
    if args.stage != STAGES[0]:
        file, value = previous_summary(plan, args.stage, args.attempt)
        # Shape is deliberately strict; unsupported shape is a gate, not PASS.
        require(value.get('verdict') == 'PASS' and value.get('strict_P1_P2_P3', {}).get('status') == 'PASS'
                and value.get('strict_additional_evidence', {}).get('status') == 'EVIDENCE_COMPLETE',
                'previous stage did not strictly pass')
        prior = {'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    _, worker = warm.request('/worker/state')
    warm.isolated_state(worker)
    live = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
    pid = live['server']['pid']
    require(type(pid) is int and b'api_server.py' in Path(f'/proc/{pid}/cmdline').read_bytes(), 'not local API PID')
    untraced = require_untraced_api(pid)
    require(live.get('active_streams') == 0, 'live streams already active')
    # Read only these allowlisted, non-secret values; never serialize environ.
    env = dict(entry.split(b'=', 1) for entry in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in entry)
    keys = ('MUSETALK_UNET_STAGEWISE_CACHE_DIR', 'MUSETALK_TAESD_TRT_DIR', 'MUSETALK_TAESD_BACKEND',
            'MUSETALK_TAESD_TRT_BUILD', 'LINGUA_CONTROL_PLANE_ENABLED', 'AVATAR_S3_ENABLED', 'HOST', 'PORT')
    require(all(env.get(k.encode()) == plan['server_env'][k].encode() for k in keys), 'actual API environment differs from fixed plan')
    del env
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                       text=True, timeout=10).strip().splitlines()
    require(gpu_pids == [str(pid)], 'foreign or missing GPU process')
    rows = []
    for avatar_id in selected:
        code, row = warm.request(f'/avatars/{avatar_id}/cache/warm?batch_size=16&wait=true&timeout_seconds=60', 'POST')
        require(code == 200 and (row.get('idle_frame_cache') or {}).get('ready') is True, 'active-stage warm unfinished')
        warm.ready_cache(row, avatar_id)
        rows.append(row)
    before = time.monotonic()
    live = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
    after = time.monotonic()
    require(before <= live['server']['monotonic'] <= after and live['server']['lifetime_counters'] is True,
            'offhost or lifetime telemetry absent')
    idle = live['server']['idle_frame_cache']
    require(idle.get('pending_builds') == 0 and all(idle.get('per_avatar', {}).get(a, {}).get('clips', 0) >= 1
                                                 for a in selected), 'active idle cache evicted or unfinished')
    require(warm.request('/stats')[1].get('active_requests') == 0, 'background requests active')
    receipt = {'schema': 'owned3090_live_stage_preflight_v1', 'status': 'PRECHECK_PASS_CLIENT_STARTING',
               'stage': args.stage, 'level': level, 'api_pid': pid, 'plan_sha256': warm.PLAN_SHA,
               'client_attempt': args.attempt, 'client_argv': argv,
               'adapter_sha256': hashlib.sha256((warm.ROOT / 'scripts/repro_3090/live_client_evidence.py').read_bytes()).hexdigest(),
               'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'previous_strict_pass': prior,
               'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip(), 'warmups': rows,
               'api_tracer_check': untraced,
               'idle_before': idle, 'clock_domain': 'same_worker', 'release_ready': False,
               'scope': 'local software-H264 diagnostic only; no EC2/TURN/browser acceptance'}
    with args.out.open('x') as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write('\n')
    clean = {'PATH': '/usr/local/nvidia/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
             'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    return subprocess.call(argv, cwd=warm.ROOT, env=clean)


if __name__ == '__main__':
    raise SystemExit(main())
