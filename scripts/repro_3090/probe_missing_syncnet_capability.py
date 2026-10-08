"""Temporarily quarantine one owned A1 checkpoint, verify actual API features, restore it.

This is not canonical installer, TTS, cold-start, production or release acceptance.
"""
import asyncio
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import time

import probe_traced_api_speech as speech
import probe_traced_avatar_prepare as prep
import warm_isolated_api as warm

SOURCE = warm.ROOT / 'models/syncnet/latentsync_syncnet.pt'
SHA = '38fa63bad3ed2332f647c40a5dc616cb0e233db8579f698f62af4c41965c4da5'
BACKUP = warm.ROOT / 'tmp/missing_syncnet_probe_v2_1841/latentsync_syncnet.pt'
BASE = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup'
OUTPUT = BASE / 'missing_syncnet_capability_v2_1841.json'
LOG = BASE / 'missing_syncnet_capability_v2_1841_private_api.log'
AVATAR = 'startup_missing_syncnet_black_man_v2_1841'
TARGET = warm.ROOT / 'results/v15/avatars' / AVATAR


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


async def prepare():
    import aiohttp
    assert not SOURCE.exists() and not TARGET.exists()
    assert sha_file(prep.SOURCE) == prep.SOURCE_SHA
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as http:
        form = aiohttp.FormData()
        form.add_field('video_file', prep.SOURCE.read_bytes(), filename='source_exact10.mp4', content_type='video/mp4')
        async with http.post(warm.BASE + '/avatars/prepare',
            params={'avatar_id': AVATAR, 'batch_size': 16, 'bbox_shift': 0, 'force_recreate': 'false'}, data=form) as response:
            body = await response.json()
            assert response.status == 200 and body['status'] == 'success' and body['avatar_id'] == AVATAR
            assert body['already_prepared'] is False and body['s3_uploaded'] is False
    assert not SOURCE.exists() and sha_file(prep.SOURCE) == prep.SOURCE_SHA
    return {'http_status': 200, 'avatar_id': AVATAR, **prep.verify_artifacts(TARGET),
            'source_unchanged': True, 'checkpoint_missing_before_after': True}


def main():
    assert socket.gethostname() == 'a830e00ce20c'
    assert (warm.DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() > 900
    assert not any(p.exists() or p.is_symlink() for p in (OUTPUT, LOG, BACKUP.parent, TARGET))
    assert SOURCE.is_file() and not SOURCE.is_symlink() and SOURCE.stat().st_size == 1488019828
    assert sha_file(SOURCE) == SHA
    with socket.socket() as check:
        assert check.connect_ex(('127.0.0.1', 8300)) != 0
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == warm.GPU_UUID
    warm.select_profile('x264tuned_v1')
    plan = json.loads(warm.PLAN.read_text())
    assert plan['server_env']['LINGUA_CONTROL_PLANE_ENABLED'] == plan['server_env']['AVATAR_S3_ENABLED'] == '0'
    assert plan['server_env']['HOST'] == '127.0.0.1' and plan['server_env']['PORT'] == '8300'
    argv = plan['server_argv']
    guard = argv.index('scripts/box_guard.sh') - 1
    expected = ['bash', 'scripts/box_guard.sh', 'run', '--wait-min', '0', '--min-avail-gb',
                '14', '--label', 'owned3090_native_live', '--']
    assert argv[guard:guard + len(expected)] == expected
    assert argv[guard + len(expected):] == ['taskset', '-c', '0-13', 'bash',
             'scripts/run_musetalk_server.sh', '--host', '127.0.0.1', '--port', '8300']
    # The parent already holds the identical GPU lease. Remove only the nested
    # guard wrapper, retaining the clean env, affinity and canonical launcher.
    launch = argv[:guard] + argv[guard + len(expected):]
    os.umask(0o077)
    def interrupted(signum, frame):
        raise InterruptedError(f'owned probe received signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    data = {'schema': 'owned_missing_syncnet_capability_v1', 'status': 'IN_PROGRESS',
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'checkpoint_sha256': SHA,
            'backup': str(BACKUP), 'profile_sha256': warm.PLAN_SHA, 'release_ready': False,
            'scored_timing': False, 'installer_contract_changed': False,
            'nested_guard_removed_under_outer_owned_lease': True,
            'prior_trial': 'missing_syncnet_capability_1835.json; failed only nested lease timeout, checkpoint restored',
            'scope': 'Only isolated direct API startup, cached human-speech serving and fresh canonical preparation; TTS disabled.'}
    with OUTPUT.open('x') as handle:
        json.dump(data, handle, indent=2)
    api = None
    try:
        BACKUP.parent.mkdir(mode=0o700)
        SOURCE.rename(BACKUP)
        assert not SOURCE.exists() and BACKUP.is_file()
        with LOG.open('x') as log:
            api = subprocess.Popen(launch, cwd=warm.ROOT, stdout=log, stderr=subprocess.STDOUT)
            data['api_pid'] = api.pid
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                assert api.poll() is None, 'owned API exited before ready'
                try:
                    warm.isolated_state(warm.request('/worker/state')[1])
                    live = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
                    assert live['server']['pid'] == api.pid and live['active_streams'] == 0
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            else:
                raise TimeoutError('owned API readiness timeout')
            data['api_ready_with_checkpoint_absent'] = not SOURCE.exists()
            assert data['api_ready_with_checkpoint_absent']
            OUTPUT.write_text(json.dumps(data, indent=2) + '\n')
            asyncio.run(speech.run(BASE / 'missing_syncnet_speech_v2_1841.json', missing_syncnet=True))
            data['speech_report'] = 'missing_syncnet_speech_v2_1841.json'
            data['fresh_preparation'] = asyncio.run(prepare())
            assert warm.request('/stats')[1]['active_requests'] == 0
            assert warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]['active_streams'] == 0
            data['status'] = 'PASS_EXERCISED_API_CAPABILITIES_WITH_SYNCNET_ABSENT'
    except Exception as exc:
        data['status'] = 'FAILED_CAPABILITY_PROBE'
        data['exception'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if api is not None and api.poll() is None:
            api.terminate()
            try:
                api.wait(timeout=30)
            except subprocess.TimeoutExpired:
                api.kill()
                api.wait(timeout=10)
                data['owned_api_force_killed'] = True
        data['api_stopped'] = api is None or api.poll() is not None
        if BACKUP.exists():
            assert not SOURCE.exists() and not SOURCE.is_symlink(), 'cannot restore over unexpected checkpoint'
            BACKUP.rename(SOURCE)
        data['checkpoint_restored_sha256'] = sha_file(SOURCE)
        data['checkpoint_restored'] = data['checkpoint_restored_sha256'] == SHA
        data['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        OUTPUT.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)
    return 0 if data['status'].startswith('PASS') and data['checkpoint_restored'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
