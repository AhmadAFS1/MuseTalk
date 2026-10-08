"""Start only a SHA-pinned diagnostic API after the previous API is gone."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess

import warm_isolated_api as warm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(warm.PLAN_PROFILES), required=True)
    args = parser.parse_args()
    warm.require(socket.gethostname() == 'a830e00ce20c', 'not owned A1')
    warm.require((warm.DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() > 600, 'deadline too close')
    warm.select_profile(args.profile)
    plan = json.loads(warm.PLAN.read_text())
    warm.require(plan['server_env']['HOST'] == '127.0.0.1'
                 and plan['server_env']['PORT'] == '8300'
                 and plan['server_env']['LINGUA_CONTROL_PLANE_ENABLED'] == '0'
                 and plan['server_env']['AVATAR_S3_ENABLED'] == '0', 'isolation contract differs')
    with socket.socket() as check:
        check.settimeout(2)
        warm.require(check.connect_ex(('127.0.0.1', 8300)) != 0, 'old API still listening')
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                  text=True, timeout=10).strip()
    warm.require(not pids, 'GPU workload still active')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                  text=True, timeout=10).strip()
    warm.require(uuid == warm.GPU_UUID, 'owned GPU changed')
    argv = plan['server_argv']
    warm.require(argv[:2] == ['env', '-i'], 'clean server environment required')
    print(json.dumps({'profile': args.profile, 'plan_sha256': warm.PLAN_SHA,
                      'control_plane_enabled': False, 'release_ready': False}), flush=True)
    os.execvp(argv[0], argv)


if __name__ == '__main__':
    main()
