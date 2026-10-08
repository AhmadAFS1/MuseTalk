#!/usr/bin/env python3
"""Start the fixed A1 API plan under model-path-only file-access tracing.

Separate non-scored startup diagnostic; never trace file content or credentials.
Requires old API stopped, port free and no GPU workload. A detached tracer can
later be stopped alone; it must be detached before any scored client workload.
"""
import hashlib
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess

import warm_isolated_api as warm

TRACE = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/native_api_model_fileopens_1638.log'
RECEIPT = TRACE.with_suffix('.start.json')
PATHS = [warm.ROOT / name for name in (
    'models/musetalkV15/unet.pth', 'models/face-parse-bisent/79999_iter.pth',
    'models/face-parse-bisent/resnet18-5c106cde.pth', 'models/syncnet/latentsync_syncnet.pt',
    'models/face_detection/s3fd.pth', 'models/auxiliary/s3fd-619a316812.pth')]


def main():
    warm.require(socket.gethostname() == 'a830e00ce20c', 'not owned A1')
    warm.require((warm.DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() > 600, 'A1 deadline too close')
    warm.require(hashlib.sha256(warm.PLAN.read_bytes()).hexdigest() == warm.PLAN_SHA, 'plan SHA mismatch')
    warm.require(not TRACE.exists() and not RECEIPT.exists() and TRACE.parent.is_dir(), 'fresh trace outputs required')
    warm.require(all(p.is_file() and not p.is_symlink() for p in PATHS), 'expected model path missing or symlink')
    with socket.socket() as check:
        check.settimeout(2)
        warm.require(check.connect_ex(('127.0.0.1', 8300)) != 0, 'old API still listening')
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                  text=True, timeout=10).strip()
    warm.require(not pids, 'GPU workload still active')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                  text=True, timeout=10).strip()
    warm.require(uuid == warm.GPU_UUID, 'owned GPU changed')
    plan = json.loads(warm.PLAN.read_text())
    argv = ['/usr/bin/strace', '-DDD', '-I', '1', '-f', '-e', 'trace=openat,open,newfstatat,statx,access',
            '-s', '512', '-o', str(TRACE)]
    for path in PATHS:
        argv += ['-P', str(path)]
    argv += plan['server_argv']
    receipt = {'schema': 'owned_native_api_model_access_trace_start_v1', 'status': 'TRACED_START_REQUESTED',
               'plan_sha256': warm.PLAN_SHA, 'trace': str(TRACE), 'model_paths': list(map(str, PATHS)),
               'file_contents_traced': False, 'credentials_traced': False, 'scored_startup': False,
               'fresh_instance': False, 'release_ready': False,
               'requirement': 'Stop only the exact detached strace process after diagnostic paths; verify TracerPid0 on API before scored live. No absence-of-use claim beyond observed operations.'}
    with RECEIPT.open('x') as out:
        json.dump(receipt, out, indent=2)
        out.write('\n')
    os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
