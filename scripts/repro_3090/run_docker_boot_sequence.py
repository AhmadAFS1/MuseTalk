#!/usr/bin/env python3
"""Bounded one-shot CI -> EC2-owned 3090 -> health/avatar/video test.

Not a recurring automation or production rollout. Waits for exactly the already
dispatched build, never retries instance creation, and records safe progress.
"""
import datetime as dt
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import docker_boot_lease_ec2 as lease
import fetch_full_candidate_evidence as full
from fetch_dependency_ci_evidence import read_api, require

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/docker_boot_test_20261010'
TOOLS = '/home/ec2-user/.local/state/musetalk-docker-boot-tools-20261010-c06624d'
REMOTE_RUN = '/home/ec2-user/.local/state/' + lease.LABEL


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def progress(stage, **safe):
    value = {'stage': stage, 'at_utc': now(), **safe}
    with (OUT / 'progress.jsonl').open('a') as output:
        output.write(json.dumps(value) + '\n')
    with (OUT / 'progress.md').open('a') as output:
        output.write('\n## ' + value['at_utc'] + ' — ' + stage + '\n\n')
        output.write('```json\n' + json.dumps(safe, indent=2) + '\n```\n')
    print(json.dumps(value), flush=True)


def remote(mode):
    command = '/home/ec2-user/lingua/venv/bin/python -B ' + TOOLS + '/docker_boot_lease_ec2.py ' + mode
    command += ' --run-dir ' + REMOTE_RUN
    if mode == 'preflight':
        command += ' --verification ' + TOOLS + '/verified-full-image.json'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'my-ec2',
                             'sudo -n ' + command], capture_output=True, timeout=90)
    require(result.returncode == 0 and len(result.stdout) < 100000, 'EC2 phase failed; inspect safe ledger, never repeat create')
    return json.loads(result.stdout)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Worker redirects not allowed')


def request(base, path, *, data=None, content_type='application/json', binary=False):
    require(path.startswith('/') and base.startswith('http://'), 'Exact observed worker HTTP endpoint required')
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': content_type})
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=25) as response:
            raw = response.read(64 * 1024**2 + 1)
            require(len(raw) <= 64 * 1024**2, 'Worker response exceeds test limit')
            return response.status, raw if binary else json.loads(raw)
    except urllib.error.HTTPError as error:
        return error.code, None  # Provider/API errors can contain sensitive text.
    except (urllib.error.URLError, TimeoutError):
        return 0, None


def endpoint(provider):
    ip = str(provider.get('public_ipaddr', ''))
    require(ipaddress.ip_address(ip).version == 4 and ipaddress.ip_address(ip).is_global, 'Public owned IPv4 required')
    maps = provider.get('ports', {}).get('8000/tcp', [])
    if len(maps) != 1:
        return None
    port = str(maps[0].get('HostPort', ''))
    require(port.isdigit() and 1 <= int(port) <= 65535, 'Valid owned mapped API port required')
    return 'http://' + ip + ':' + port


def wait_status(base, path, deadline, expected):
    while time.monotonic() < deadline:
        code, body = request(base, path)
        if code == 200 and body and body.get('status') == expected:
            return body
        require(not body or body.get('status') not in {'failed', 'error'}, 'Worker app test failed')
        time.sleep(10)
    raise ValueError('Worker app test deadline exceeded')


def main():
    require(not OUT.exists(), 'Sequence output already exists; do not repeat creation')
    OUT.mkdir(parents=True)
    (OUT / 'progress.md').write_text('# Private Docker boot test progress\n\n'
        'One existing CI run, then at most one owned RTX 3090. No production rollout. '
        'Provider image-cache state is unknown. Health is not a usable-call verdict.\n')
    progress('WAITING_FOR_FULL_IMAGE', run_id=full.RUN, ci_url='https://github.com/AhmadAFS1/MuseTalk/actions/runs/' + str(full.RUN),
             source_revision=full.SOURCE_REV, gpu_rented=False)
    secret_before = json.loads(read_api('actions/secrets/MUSETALK_PRIVATE_BUILD_INPUTS'))
    require(secret_before['name'] == 'MUSETALK_PRIVATE_BUILD_INPUTS', 'Expected ephemeral CI secret required')
    deadline = time.monotonic() + 2 * 3600
    while time.monotonic() < deadline:
        run = json.loads(read_api(f'actions/runs/{full.RUN}'))
        require(run['head_sha'] == full.REQUEST_REV, 'CI request identity changed')
        if run['status'] == 'completed':
            require(run['conclusion'] == 'success', 'Full-image CI failed; no GPU rented')
            break
        time.sleep(45)
    else:
        raise ValueError('CI wait limit exceeded; no GPU rented')
    evidence = OUT / 'full-image-evidence'
    subprocess.run([sys.executable, '-B', str(Path(full.__file__)), '--out', str(evidence)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proof = json.loads((evidence / 'verified-full-image.json').read_text())
    lease.validate_verification(proof)
    progress('FULL_PRIVATE_IMAGE_VERIFIED', image=proof['image'], independent_pull='PASS', offline_cpu_check='PASS',
             compressed_layer_bytes=proof['compressed_layer_bytes'], gpu_rented=False)
    # Remove only the ephemeral secret installed for this exact build after
    # success proves every input has been consumed. No token values are read.
    secret_after = json.loads(read_api('actions/secrets/MUSETALK_PRIVATE_BUILD_INPUTS'))
    require(secret_after == secret_before, 'CI secret was changed externally; do not delete it')
    deletion = subprocess.run(['ssh', '-o', 'BatchMode=yes', '3-way-head-talk', '/usr/bin/gh', 'api',
                               '--method', 'DELETE', 'repos/AhmadAFS1/MuseTalk/actions/secrets/MUSETALK_PRIVATE_BUILD_INPUTS'],
                              capture_output=True, timeout=45)
    require(deletion.returncode == 0, 'Ephemeral secret removal failed')
    progress('EPHEMERAL_CI_INPUT_SECRET_REMOVED', credential_values_read=False)
    proof_transfer = '/tmp/musetalk-candidate-2057.GBx5Fc/verified-full-image.json'
    subprocess.run(['scp', '-q', str(evidence / 'verified-full-image.json'), 'my-ec2:' + proof_transfer], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', 'my-ec2', 'sudo -n install -m 600 '
                    + proof_transfer + ' ' + TOOLS + '/verified-full-image.json'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    preflight = remote('preflight')
    progress('RTX3090_PREFLIGHT', **preflight)
    created = remote('create')  # Exactly one invocation. Any exception stops; no automatic retry.
    progress('RTX3090_CREATED', **created)
    ready_deadline = time.monotonic() + 1800
    base = None
    while time.monotonic() < ready_deadline:
        provider = remote('sample')
        require(str(provider.get('id')) == created['instance_id'], 'Test instance changed')
        require(provider.get('actual_status') not in {'error', 'failed', 'invalid'}, 'Owned provider boot failed')
        if provider.get('public_ipaddr') and provider.get('ports'):
            base = endpoint(provider)
            if base:
                code, health = request(base, '/health')
                if code == 200 and health and health.get('ok') is True and health.get('service') == 'musetalk':
                    break
        time.sleep(15)
    else:
        raise ValueError('Health not ready within 30 minutes; bounded expiry remains armed')
    # request timestamp is the authoritative EC2 clock, not post-response time.
    ledger = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'my-ec2',
                             'sudo -n /bin/cat ' + REMOTE_RUN + '/startup-ledger.json'], capture_output=True, timeout=30)
    require(ledger.returncode == 0, 'Creation timing ledger unavailable')
    events = json.loads(ledger.stdout)['events']
    started = dt.datetime.fromisoformat(next(e['utc'] for e in events if e['name'] == 'request_started'))
    health_at = dt.datetime.now(dt.timezone.utc)
    progress('HEALTH_READY', instance_id=created['instance_id'], image=created['image'],
             request_to_health_seconds=(health_at - started).total_seconds(),
             observation_interval_seconds=15, cache_state='unknown', health_not_call_acceptance=True)
    avatar = 'zh_01_mei'
    code, warmed = request(base, '/avatars/' + avatar + '/cache/warm?wait=false&batch_size=16', data=b'')
    require(code in {200, 202} and warmed and warmed.get('status') != 'failed', 'Existing avatar restore/warm rejected')
    warm = wait_status(base, '/avatars/' + avatar + '/cache/status', time.monotonic() + 1200, 'ready')
    progress('AVATAR_READY', avatar_id=avatar, request_to_avatar_ready_seconds=(dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
             limitation='One existing production avatar, not fleet/48-avatar warm readiness')
    audio_path = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/quality/review_fixtures/black_woman/speech.wav'
    require(audio_path.is_file() and not audio_path.is_symlink() and audio_path.stat().st_size <= 4 * 1024**2, 'Bounded existing speech fixture required')
    audio = audio_path.read_bytes()
    boundary = 'musetalk-test-' + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio_file"; filename="speech.wav"\r\n'
            'Content-Type: audio/wav\r\n\r\n').encode() + audio + f'\r\n--{boundary}--\r\n'.encode()
    code, generated = request(base, '/generate?avatar_id=' + avatar + '&batch_size=16&fps=25', data=body,
                              content_type='multipart/form-data; boundary=' + boundary)
    require(code == 200 and generated and re.fullmatch('[A-Za-z0-9_-]{1,128}', str(generated.get('request_id', ''))), 'Video generation not accepted')
    request_id = generated['request_id']
    result = wait_status(base, '/generate/' + request_id + '/status', time.monotonic() + 600, 'completed')
    require(result.get('result', {}).get('success') is True, 'Video output generation failed')
    code, video = request(base, '/generate/' + request_id + '/download', binary=True)
    require(code == 200 and video, 'Generated output unavailable')
    video_path = OUT / 'startup-video-smoke.mp4'
    with video_path.open('xb') as output:
        output.write(video)
    subprocess.run(['/opt/homebrew/bin/ffmpeg', '-nostdin', '-loglevel', 'error', '-i', str(video_path),
                    '-map', '0:v:0', '-frames:v', '1', '-f', 'null', '-'], check=True, capture_output=True, timeout=30)
    probe = subprocess.run(['/opt/homebrew/bin/ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate,duration', '-of', 'json', str(video_path)],
                           check=True, capture_output=True, timeout=30)
    info = json.loads(probe.stdout)['streams'][0]
    require(int(info['nb_read_frames']) > 0 and info['width'] > 0 and info['height'] > 0, 'Decoded video not proved')
    progress('VIDEO_SMOKE_PASS', instance_id=created['instance_id'], avatar_id=avatar,
             request_to_decodable_output_seconds=(dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
             video={'sha256': hashlib.sha256(video).hexdigest(), 'size_bytes': len(video), **info},
             input_audio={'sha256': hashlib.sha256(audio).hexdigest(), 'size_bytes': len(audio)},
             resource_deadline_utc=created['deadline_utc'],
             limitation='REST generation/download and local frame decode only; not live WebRTC/RTP, 400 FPS, quality parity or production rollout')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.is_dir():
            progress('STOPPED_REQUIRES_INSPECTION', error_type=type(exc).__name__,
                     failure_code=str(exc) if type(exc) is ValueError else 'details_suppressed',
                     create_retry_forbidden=True, note='No raw provider/API error body recorded. Any created worker retains its bound four-hour expiry.')
        raise SystemExit(2)
