"""Full file-syscall startup diagnostic with a known positive control, never scored timing."""
import datetime as dt
import json
import os
from pathlib import Path
import socket
import sys

import warm_isolated_api as warm
import start_fixed_isolated_api as starter

TRACE = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/native_api_fullfile_trace_v2_1802.log'
RECEIPT = TRACE.with_suffix('.start.json')
CONTROL = TRACE.with_suffix('.control.json')
SCRIPT = warm.ROOT / 'scripts/repro_3090/start_fullfile_traced_api.py'
PYTHON = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'


def trace_argv():
    # No -P path filters: the previous path-filtered trace missed a known open.
    return ['/usr/bin/strace', '-f', '-e', 'trace=%file', '-s', '512', '-o', str(TRACE),
            PYTHON, str(SCRIPT), '--inside-trace']


def main():
    warm.require(socket.gethostname() == 'a830e00ce20c', 'not owned A1')
    warm.require((warm.DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() > 600, 'deadline too close')
    warm.select_profile('x264tuned_v1')
    if sys.argv[1:] == ['--inside-trace']:
        fields = dict(row.split(':', 1) for row in Path('/proc/self/status').read_text().splitlines() if ':' in row)
        warm.require(int(fields['TracerPid']) > 0, 'positive control is not traced')
        source = warm.ROOT / 'models/musetalkV15/unet.pth'
        warm.require(source.is_file() and not source.is_symlink(), 'positive-control model missing/link')
        with source.open('rb') as handle:
            warm.require(len(handle.read(1)) == 1, 'positive control failed')
        with CONTROL.open('x') as handle:
            json.dump({'schema': 'fullfile_trace_positive_control_v1', 'pid': os.getpid(),
                       'tracer_pid': int(fields['TracerPid']), 'model': str(source),
                       'read_bytes': 1, 'status': 'POSITIVE_CONTROL_READ_COMPLETED',
                       'scope': 'Explicit one-byte control before API exec; not evidence API needs this model'}, handle, indent=2)
        sys.argv = [str(SCRIPT), '--profile', 'x264tuned_v1']
        return starter.main()
    warm.require(not sys.argv[1:], 'unsupported trace arguments')
    warm.require(not any(p.exists() or p.is_symlink() for p in (TRACE, RECEIPT, CONTROL)), 'fresh trace outputs required')
    with socket.socket() as check:
        warm.require(check.connect_ex(('127.0.0.1', 8300)) != 0, 'API still listening')
    os.umask(0o077)
    data = {'schema': 'owned3090_fullfile_trace_start_v2', 'status': 'TRACED_START_REQUESTED',
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'profile': 'x264tuned_v1',
            'plan_sha256': warm.PLAN_SHA, 'trace': str(TRACE), 'positive_control_receipt': str(CONTROL),
            'trace_argv': trace_argv(), 'file_contents_traced': False, 'path_filters_used': False,
            'scored_timing': False, 'fresh_instance': False, 'release_ready': False,
            'scope': 'Observed startup and explicitly invoked paths only; absence cannot establish all-feature independence',
            'lifecycle': 'Stop exact owned idle API first; foreground tracer exits with it. No scored client while traced.'}
    with RECEIPT.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': data['status'], 'release_ready': False}), flush=True)
    os.execv('/usr/bin/strace', trace_argv())


if __name__ == '__main__':
    raise SystemExit(main())
