#!/usr/bin/env python3
"""Prepare, never start, an isolated owned-3090 native live experiment.

The generated commands require operator review and prior native engine preflight.
This helper performs no SSH, GPU, server, cloud, or control-plane operation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = Path('/workspace/MuseTalk')
PYTHON = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'
AUDIO = str(WORKER_ROOT / 'experiments/throughput300_candidate/audio_corpus/33_long_eng.wav')
AUDIO_SHA = '654dcbce843d70451d1123f7649a58bee11bb9dec9a7e835c05b1e367efb2078'
OBSERVED_CPU_QUOTA = 18.43199


def require(ok, message):
    if not ok:
        raise ValueError(message)


def literal_env(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        require(sep and re.fullmatch('[A-Z][A-Z0-9_]*', key), 'invalid literal env key')
        require(not re.search('TOKEN|SECRET|PASSWORD|ACCESS_KEY|CREDENTIAL', key), 'secret env key forbidden')
        require(not any(c in value for c in '$`\n\r'), 'shell expansion forbidden')
        require(key not in result, 'duplicate env key')
        result[key] = value
    return result


def cpu_ids(text):
    result = set()
    for part in text.split(','):
        require(re.fullmatch(r'\d+(?:-\d+)?', part), 'invalid CPU set')
        start, _, stop = part.partition('-')
        lo, hi = int(start), int(stop or start)
        require(lo <= hi <= 95, 'CPU outside observed affinity 0-95')
        result.update(range(lo, hi + 1))
    return result


def absolute_worker_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(WORKER_ROOT),
            'path must be explicit beneath owned worker checkout')
    require(not any(c in str(path) for c in '\n\r:$`'), 'unsafe worker path')
    return str(path)


def prepare(engine_root, taesd_dir, taesd_key, runtime_dir, avatars, codec,
            server_cpus='0-13', client_cpus='14-17'):
    server_ids, client_ids = cpu_ids(server_cpus), cpu_ids(client_cpus)
    require(server_ids and client_ids and not server_ids & client_ids, 'CPU sets must be nonempty and disjoint')
    require(len(server_ids | client_ids) <= int(OBSERVED_CPU_QUOTA), 'CPU selection exceeds observed quota floor')
    require(codec in ('aiortc', 'x264tuned', 'nvenc'), 'explicit H264 implementation required')
    require(re.fullmatch('[a-f0-9]{20}', taesd_key), 'actual TAESD fingerprint key required')
    require(avatars and len(set(avatars)) == len(avatars) and all(re.fullmatch('[A-Za-z0-9_-]+', a) for a in avatars),
            'unique explicit API cache avatar IDs required')
    runtime_dir = absolute_worker_path(runtime_dir)
    values = {}
    for filename in ('experiments/live15_r5/common.env', 'experiments/live15_r5/serve.env',
                     'experiments/live15_r5/loopfix.env', 'scripts/repro_3090/profiles/native.env'):
        values.update(literal_env(ROOT / filename))
    values.update({
        'PATH': '/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'LD_LIBRARY_PATH': '/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64',
        'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
        'REPO_ROOT': str(WORKER_ROOT), 'WORKSPACE': '/workspace',
        'VENV_PATH': str(Path(PYTHON).parent.parent), 'HOST': '127.0.0.1', 'PORT': '8300',
        'MUSETALK_RECIPE': 'fast300', 'MUSETALK_RUNTIME_DIR': runtime_dir,
        'MUSETALK_ENV_OVERRIDES_FILE': '/dev/null', 'LINGUA_CONTROL_PLANE_ENV_FILE': '/dev/null',
        'LINGUA_CONTROL_PLANE_ENABLED': '0', 'LINGUA_WORKER_CALLBACK_REQUIRED': '0',
        'AVATAR_S3_ENABLED': '0', 'WEBRTC_STUN_URLS': '', 'WEBRTC_RELAY_ENABLED': '0',
        'WEBRTC_TURN_AUTOSTART': '0', 'MUSETALK_DISABLE_LOCAL_TTS': '1',
        'MUSETALK_UNET_STAGEWISE_CACHE_DIR': absolute_worker_path(engine_root),
        'MUSETALK_UNET_STAGEWISE_VERIFY_SHA': '1', 'MUSETALK_UNET_STAGEWISE_PROBE_CHECK': '1',
        'MUSETALK_UNET_STAGEWISE_PROBE_TOL': '0', 'MUSETALK_TAESD_TRT_DIR': absolute_worker_path(taesd_dir),
        'WEBRTC_LIFETIME_COUNTERS': '1', 'WEBRTC_LIFETIME_SEND_RING': '256', 'HLS_GPU_EVENT_TIMING': '1',
        'WEBRTC_VP8_ENCODER': 'pyav', 'WEBRTC_H264_IMPL': codec,
        'WEBRTC_H264_X264_THREADS': '1', 'WEBRTC_H264_X264_PRESET': 'veryfast',
        'WEBRTC_H264_NVENC_PRESET': 'p2', 'WEBRTC_H264_NVENC_TUNE': 'll', 'WEBRTC_NVENC_MAX_SESSIONS': '12',
        'MUSETALK_TORCH_INTRAOP_THREADS': '4', 'MUSETALK_CV2_THREADS': '1',
        'MUSETALK_IDLE_DECODE_THREADS': '1', 'MUSETALK_FFMPEG_EXECUTOR_WORKERS': '4',
        'HLS_PREP_WORKERS': '4', 'HLS_COMPOSE_WORKERS': '4', 'HLS_ENCODE_WORKERS': '4',
        'MUSETALK_AVATAR_LOAD_WORKERS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '1',
    })
    command = ['env', '-i', *(f'{k}={v}' for k, v in sorted(values.items())),
               'bash', 'scripts/box_guard.sh', 'run', '--wait-min', '0', '--min-avail-gb', '14',
               '--label', 'owned3090_native_live', '--', 'taskset', '-c', server_cpus,
               'bash', 'scripts/run_musetalk_server.sh', '--host', '127.0.0.1', '--port', '8300']
    common = [PYTHON, 'scripts/repro_3090/live_client_evidence.py', '--base-url', 'http://127.0.0.1:8300',
              '--avatar-ids', ','.join(avatars), '--audio-dir', str(Path(AUDIO).parent), '--audio-class', 'any',
              '--musetalk-fps', '20', '--playback-fps', '20', '--batch-size', '16', '--chunk-duration', '1',
              '--ring', '256', '--poll-interval-s', '1', '--turn-gap-s', '1', '--settle-s', '10',
              '--warmup-s', '10', '--cooldown-s', '30', '--peers-per-shard', '1',
              '--client-cpus', client_cpus, '--ignore-ice-servers', '--abort-below-gb', '14']
    stages = {}
    # Fixed list is intentional: --chain overrides --wav-list in the old rig.
    for label, levels, seconds, minimum, repeats, stagger in (
            ('s0_n1', '1', None, '10', 2, '0'), ('s0_n3', '3', None, '10', 2, '0'),
            ('ramp_n5', '5', '300', '240', 8, '5'), ('ramp_n10', '10', '300', '240', 8, '5'),
            ('ramp_n15', '15', '300', '240', 8, '5'),
            ('soak', 'SELECT_RAMP_PASSING_N', '3630', '3600', 64, '20')):
        args = common + ['--label', label, '--levels', levels, '--min-steady-s', minimum,
                         '--wav-list', ','.join([AUDIO] * repeats), '--stagger-s', stagger,
                         '--out-dir', runtime_dir + '/' + label + '/lt2',
                         '--trace-dir', runtime_dir + '/' + label + '/traces']
        if seconds:
            args += ['--duration-s', seconds, '--stagger-join']
        stages[label] = args
    return {
        'schema': 'owned3090_isolated_live_plan_v1', 'status': 'PLAN_ONLY', 'release_ready': False,
        'instance_id': '54798270', 'ssh_alias': 'musetalk-3090-build-54798270',
        'source_scope': 'local loopback, not EC2/TURN or browser acceptance', 'cwd': str(WORKER_ROOT),
        'thread_policy': 'Explicit conservative starting configuration, not measured optimal tuning: server14 logical IDs plus client4 under18.43199 shared CPU quota; disjoint IDs are not dedicated physical cores.',
        'observed_cpu_quota': OBSERVED_CPU_QUOTA, 'observed_cpu_affinity': '0-95',
        'server_cpus': server_cpus, 'client_cpus': client_cpus, 'cpu_allocation_must_be_rechecked_before_run': True,
        'production_ports_untouched': {'api_external': 25659, 'turn_tcp_external': 25884, 'turn_udp_external': 25609},
        'bind': '127.0.0.1:8300', 'server_env': values, 'server_argv': command,
        'server_command': shlex.join(command), 'client_stage_argv': stages,
        'expected_taesd_key': taesd_key, 'engine_preflight': 'REQUIRED before server; no build or fallback allowed',
        'client_level_gate': 'Each invocation runs one level. Run 1 then3 then5 then10 then15 only after the preceding level passes. Replace SELECT_RAMP_PASSING_N with the highest capacity proven with headroom before soak.',
        'audio': {'path': AUDIO, 'sha256': AUDIO_SHA, 'duration_seconds': 60, 'sample_rate': 16000, 'channels': 1,
                  'provenance': 'docs/fps_comparisons/4070s_400fps_20260928/README.md section 3.4 C_repo_eng: real human recording',
                  'same_wav_every_stream_and_turn': True, 'whole_corpus_is_human': False},
        'avatars': avatars, 'workload': 'explicit restored API cache IDs; not automatically canonical six or complete production48',
        'avatar_coverage': {
            'canonical_offline_six': ['black_man_short_beard', 'black_woman', 'east_asian_man_goatee',
                                     'middle_eastern_man_full_beard', 'south_asian_woman', 'white_man_clean_shaven'],
            'production_inventory': '16 character identities x3 original pose caches =48; cache audit and live pose-transition coverage are separate',
            'live_assignment': 'Explicit API avatar IDs round-robin across concurrent sessions; repeated sessions do not expand unique-avatar coverage.',
            'composition_gap': 'Current API uses saved-mask standard blend, not the six-fixture refined chin path; live P1-P3 cannot establish full offline recipe parity.',
            'idle_cache_capacity_warning': '2400MiB matches the old15-avatar test, not all48 poses. Prewarming48 can evict earlier clips; separately size from actual clips and report evictions before claiming fully warm production coverage.',
        },
        'codec': {'requested': codec, 'wire_codec_evidence': 'wrapper counts received RTP payload types against negotiated codec map',
                  'nvenc_fallback_permitted_by_server': codec == 'nvenc', 'actual_encoder_log_review_required': True},
        'prerequisites': ['GPU health recovered and native preflight passed', 'port8300 free; no foreign GPU workload',
                          'all explicit API caches already local and immutable-input hashes verified',
                          'warm every selected avatar with batch16/wait=true before timing',
                          'worker/state control_plane_requested=false; lifetime counters true',
                          'TAESD key/plan and UNet native engine directory confirmed in actual startup log'],
        'offhost': {'status': 'INVALID_FOR_EXISTING_SCORER', 'reason': 'No subtraction or timestamp join across independent monotonic clocks.',
                    'http_ssh_forward_only': 'SSH forwarding8300 carries signaling only, not ICE/RTP media.'},
        'credentials': 'No AWS/control-plane keys in plan or server env. Restore caches beforehand via approved memory-only runtime credential bridge; running server cannot inherit later child credentials. Later bridge children can audit/fetch independently, but never during scored timing because downloads/hash CPU/IO would contaminate it.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('engine-root', 'taesd-dir', 'taesd-key', 'runtime-dir', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--avatar-id', action='append', required=True)
    p.add_argument('--h264-impl', choices=('aiortc', 'x264tuned', 'nvenc'), required=True)
    p.add_argument('--server-cpus', default='0-13')
    p.add_argument('--client-cpus', default='14-17')
    a = p.parse_args()
    plan = prepare(a.engine_root, a.taesd_dir, a.taesd_key, a.runtime_dir, a.avatar_id, a.h264_impl,
                   a.server_cpus, a.client_cpus)
    with Path(a.out).open('x') as out:
        json.dump(plan, out, indent=2)
        out.write('\n')
    print('PLAN_ONLY; no server or client started')


if __name__ == '__main__':
    main()
