#!/usr/bin/env python3
"""Exercise the image's default PID 1 using Docker with no GPU or network."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


WORKER = '/opt/character-runtime/app/character_factory/character_worker'

STARTUP_FIXTURE = """#!/bin/bash
set -eu
mkdir -p "$CHARACTER_WORKER_ROOT"
/usr/bin/python3 "$REPO_ROOT/character_factory/character_worker/character_api.py" &
echo $! > "$CHARACTER_WORKER_ROOT/character_api.pid"
sleep 0.25 &
echo $! > /qa/orphan-pid
case "$ENTRYPOINT_QA_MODE" in
  ready) touch /qa/startup-returning ;;
  failed) touch /qa/startup-failed; exit 7 ;;
  booting)
    trap 'touch /qa/startup-signaled; exit 143' TERM INT
    touch /qa/startup-booting
    while sleep 1; do :; done ;;
  *) exit 99 ;;
esac
"""

API_FIXTURE = """import os
from pathlib import Path
import signal
import subprocess
import sys
import time

role = 'model' if len(sys.argv) > 1 else 'api'

def stop(signum, _frame):
    Path('/qa/' + role + '-signal').write_text(str(signum))
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if role == 'api':
    subprocess.Popen([sys.executable, __file__, 'model'], start_new_session=True)
Path('/qa/' + role + '-ready').write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
"""


def run(*args, check=True, timeout=20):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{args[0]} {args[1]} exited {result.returncode}: {result.stderr[-1000:]}')
    return result


def wait_for(test, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if test():
            return
        time.sleep(0.2)
    raise TimeoutError('entrypoint observation did not arrive')


def exercise_fixture(image, pins, mode, *, optional_ssh=False):
    name = 'character-entrypoint-qa-' + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        startup, api, output = root / 'startup.sh', root / 'api.py', root / 'result'
        startup.write_text(STARTUP_FIXTURE)
        api.write_text(API_FIXTURE)
        output.mkdir()
        mounts = ['-v', f'{startup}:{WORKER}/vast_onstart_character.sh:ro',
                  '-v', f'{api}:{WORKER}/character_api.py:ro',
                  '-v', f'{output}:/qa']
        env = ['-e', f'ENTRYPOINT_QA_MODE={mode}']
        if optional_ssh:
            key = root / 'qa-key'
            run('ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key))
            mounts += ['-v', f'{key}:/qa-client-key:ro']
            env += ['-e', 'CHARACTER_WORKER_ENABLE_SSH=1',
                    '-e', 'SSH_PUBLIC_KEY=' + key.with_suffix('.pub').read_text().strip(),
                    '-e', 'CHARACTER_WORKER_TLS_KEY_B64=fixture-private-value']
        try:
            run('docker', 'run', '-d', '--name', name, '--network', 'none', *pins, *mounts, *env, image)
            marker = {'ready': 'startup-returning', 'booting': 'startup-booting', 'failed': 'startup-failed'}[mode]
            wait_for(lambda: all((output / part).exists() for part in ('api-ready', 'model-ready', marker)))
            if optional_ssh:
                def ssh_ok():
                    result = run('docker', 'exec', name, 'ssh', '-i', '/qa-client-key', '-o', 'BatchMode=yes',
                                 '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=no',
                                 '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=2', 'root@127.0.0.1',
                                 'test -z "$CHARACTER_WORKER_TLS_KEY_B64" && echo key-only', check=False)
                    return result.returncode == 0 and result.stdout.strip() == 'key-only'
                wait_for(ssh_ok)
            if mode == 'ready':
                orphan = (output / 'orphan-pid').read_text().strip()
                wait_for(lambda: run('docker', 'exec', name, 'test', '!', '-e', '/proc/' + orphan,
                                     check=False).returncode == 0)
                # An API crash after successful onstart must fail the container and stop detached model work.
                run('docker', 'exec', name, 'kill', '-KILL', (output / 'api-ready').read_text().strip())
                expected_exit, expected_signal = 1, '15'
            elif mode == 'booting':
                run('docker', 'kill', '--signal', 'INT', name)
                expected_exit, expected_signal = 130, '2'
            else:
                wait_for(lambda: 'authoritative startup failed with status 7' in run('docker', 'logs', name).stdout)
                assert run('docker', 'inspect', '-f', '{{.State.Running}}', name).stdout.strip() == 'true'
                assert not (output / 'api-signal').exists()
                run('docker', 'stop', '-t', '12', name)
                expected_exit, expected_signal = 143, '15'
            assert int(run('docker', 'wait', name).stdout.strip()) == expected_exit
            assert (output / 'model-signal').read_text() == expected_signal
            if mode != 'ready':
                assert (output / 'api-signal').read_text() == expected_signal
            if mode == 'booting':
                assert (output / 'startup-signaled').exists()
        finally:
            result = run('docker', 'logs', name, check=False)
            print((result.stdout + result.stderr)[-2000:])
            run('docker', 'rm', '-f', name, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    source = run('docker', 'image', 'inspect', '-f', '{{index .Config.Labels "org.opencontainers.image.revision"}}',
                 args.image).stdout.strip()
    pins = ['-e', 'MUSETALK_REF=' + source,
            '-e', 'CHARACTER_WORKER_RUNTIME_MANIFEST_SHA256=' + args.manifest_sha,
            '-e', 'CHARACTER_WORKER_IMAGE_DIGEST=' + args.image]
    refusal = run('docker', 'run', '--rm', '--network', 'none', args.image, check=False)
    assert refusal.returncode == 64 and 'immutable deployment pins are missing or invalid' in refusal.stdout
    exact = run('docker', 'run', '--rm', '--network', 'none', *pins, args.image, check=False, timeout=35)
    assert exact.returncode != 0
    assert 'CHARACTER_WORKER_ONSTART BEGIN' in exact.stdout and 'CHARACTER_WORKER_ONSTART FAILED' in exact.stdout
    assert 'authoritative startup failed' in exact.stdout
    exercise_fixture(args.image, pins, 'ready', optional_ssh=True)
    exercise_fixture(args.image, pins, 'booting')
    exercise_fixture(args.image, pins, 'failed')
    args.report.write_text(json.dumps({
        'schema': 'lingua-character-entrypoint-cpu-qualification/1',
        'image': args.image, 'source_revision': source, 'runtime_manifest_sha256': args.manifest_sha,
        'default_entrypoint_unmounted_missing_pins': 'refused',
        'default_entrypoint_unmounted_no_gpu': 'authoritative preflight refused',
        'fixture_boundary': 'mount startup and API fixtures only for lifecycle checks',
        'default_pid1_lifecycle': {
            'optional_key_only_ssh': 'passed',
            'api_exit_after_successful_startup': 'passed',
            'orphan_reaped': 'passed',
            'interrupt_during_startup': 'passed',
            'boot_failure_retained_for_observation': 'passed',
            'detached_model_groups_signaled': 'passed',
        },
        'network': 'none; localhost SSH only', 'gpu_qualification': 'not_run',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
