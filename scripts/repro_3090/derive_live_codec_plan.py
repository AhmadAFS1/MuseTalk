"""Derive an isolated x264-thread trial without rewriting its failed baseline."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import shlex

from live_isolated import require

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native'
PARENT_SHA = '40702fd5101ef6dc8839b4f3eb0b04120a753f18bcee12ed3218a1d0f59fda64'
OUTPUT = FOLDER / 'isolated_live_x264tuned_v1_plan.json'
TUNED_SHA = '2e9f78eafd39ada6a48982a49c49316314cf7282697b2bc96405a0b41bfa5488'


def derive(parent):
    require(parent['server_env']['WEBRTC_H264_IMPL'] == 'aiortc', 'unexpected baseline codec')
    require(parent['server_env']['WEBRTC_H264_X264_THREADS'] == '1'
            and parent['server_env']['WEBRTC_H264_X264_PRESET'] == 'veryfast', 'baseline tuned settings differ')
    data = copy.deepcopy(parent)
    old = parent['server_env']['MUSETALK_RUNTIME_DIR']
    new = '/workspace/MuseTalk/experiments/owned3090_native_live_x264tuned_v1_1705'
    data['server_env']['WEBRTC_H264_IMPL'] = 'x264tuned'
    data['server_env']['MUSETALK_RUNTIME_DIR'] = new
    data['server_argv'] = [arg.replace(old, new) if old in arg else
                           ('WEBRTC_H264_IMPL=x264tuned' if arg == 'WEBRTC_H264_IMPL=aiortc' else arg)
                           for arg in parent['server_argv']]
    data['server_command'] = shlex.join(data['server_argv'])
    data['client_stage_argv'] = {stage: [arg.replace(old, new) for arg in argv]
                                 for stage, argv in parent['client_stage_argv'].items()}
    data['trial'] = {'parent_plan_sha256': PARENT_SHA, 'single_runtime_lever': 'WEBRTC_H264_IMPL:aiortc->x264tuned',
                     'other_runtime_env_unchanged_except_output_root': True,
                     'all_client_and_score_thresholds_unchanged': True,
                     'quality_rejected_native_diagnostic_only': True}
    require({k for k in parent['server_env'] if parent['server_env'][k] != data['server_env'][k]}
            == {'WEBRTC_H264_IMPL', 'MUSETALK_RUNTIME_DIR'}, 'unintended runtime change')
    return data


def derive_nvenc(parent):
    require(parent['server_env']['WEBRTC_H264_IMPL'] == 'x264tuned', 'unexpected NVENC parent codec')
    data = copy.deepcopy(parent)
    old = parent['server_env']['MUSETALK_RUNTIME_DIR']
    new = '/workspace/MuseTalk/experiments/owned3090_native_live_nvenc_v1_1735'
    data['server_env']['WEBRTC_H264_IMPL'] = 'nvenc'
    data['server_env']['MUSETALK_RUNTIME_DIR'] = new
    data['server_argv'] = [arg.replace(old, new) if old in arg else
                          ('WEBRTC_H264_IMPL=nvenc' if arg == 'WEBRTC_H264_IMPL=x264tuned' else arg)
                          for arg in parent['server_argv']]
    data['server_command'] = shlex.join(data['server_argv'])
    data['client_stage_argv'] = {stage: [arg.replace(old, new) for arg in argv]
                               for stage, argv in parent['client_stage_argv'].items()}
    data['trial'] = {'parent_plan_sha256': TUNED_SHA, 'single_runtime_lever': 'WEBRTC_H264_IMPL:x264tuned->nvenc',
                     'other_runtime_env_unchanged_except_output_root': True,
                     'all_client_and_score_thresholds_unchanged': True,
                     'quality_rejected_native_diagnostic_only': True,
                     'nvenc_advance_requires': 'preceding strict PASS plus aggregate slots peak/opened at least level, zero denied/open failures; actual encoder-open log still required'}
    require({k for k in parent['server_env'] if parent['server_env'][k] != data['server_env'][k]}
            == {'WEBRTC_H264_IMPL', 'MUSETALK_RUNTIME_DIR'}, 'unintended NVENC runtime change')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('x264tuned_v1', 'nvenc_v1'), default='x264tuned_v1')
    args = parser.parse_args()
    is_nvenc = args.profile == 'nvenc_v1'
    source = OUTPUT if is_nvenc else FOLDER / 'isolated_live_v1_plan.json'
    source_sha = TUNED_SHA if is_nvenc else PARENT_SHA
    destination = FOLDER / 'isolated_live_nvenc_v1_plan.json' if is_nvenc else OUTPUT
    require(not source.is_symlink() and hashlib.sha256(source.read_bytes()).hexdigest() == source_sha,
            'fixed parent plan SHA mismatch')
    data = (derive_nvenc if is_nvenc else derive)(json.loads(source.read_text()))
    with destination.open('x') as out:
        json.dump(data, out, indent=2)
        out.write('\n')
    print(json.dumps({'plan': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                      'server_started': False, 'release_ready': False}))


if __name__ == '__main__':
    main()
