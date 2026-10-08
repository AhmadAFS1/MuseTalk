"""Gracefully stop only the fixed owned idle diagnostic API, never production."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import warm_isolated_api as warm

PID = 466043
TRACER = 465872
OUTPUT = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/fullfile_trace_v2_stop_1820.json'


def main():
    assert socket.gethostname() == 'a830e00ce20c'
    assert not OUTPUT.exists() and not OUTPUT.is_symlink()
    warm.isolated_state(warm.request('/worker/state')[1])
    live = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
    assert live['server']['pid'] == PID and live['active_streams'] == 0
    assert warm.request('/stats')[1]['active_requests'] == 0
    command = Path(f'/proc/{PID}/cmdline').read_bytes()
    assert b'api_server.py' in command and Path(f'/proc/{PID}/cwd').resolve() == warm.ROOT
    status = dict(row.split(':', 1) for row in Path(f'/proc/{PID}/status').read_text().splitlines() if ':' in row)
    assert int(status['TracerPid']) == TRACER
    data = {'schema': 'owned_fullfile_trace_stop_v1', 'status': 'STOP_REQUESTED',
            'api_pid': PID, 'tracer_pid': TRACER, 'active_streams_before': 0,
            'active_requests_before': 0, 'signal': 'SIGTERM',
            'api_cmdline_sha256': hashlib.sha256(command).hexdigest(),
            'release_ready': False, 'scored_timing': False}
    with OUTPUT.open('x') as handle:
        json.dump(data, handle, indent=2)
    os.kill(PID, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while Path(f'/proc/{PID}').exists() and time.monotonic() < deadline:
        time.sleep(.25)
    data['api_process_gone'] = not Path(f'/proc/{PID}').exists()
    with socket.socket() as connection:
        data['api_port_closed'] = connection.connect_ex(('127.0.0.1', 8300)) != 0
    data['status'] = 'PASS_OWNED_API_STOPPED' if data['api_process_gone'] and data['api_port_closed'] else 'FAILED_STOP'
    OUTPUT.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)
    return 0 if data['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
