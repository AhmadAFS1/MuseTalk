"""Exercise canonical one-turn speech under file tracing; no scored cadence/startup claim."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket

import warm_isolated_api as warm
from live_client_evidence import install_h264_offer

ROOT = warm.ROOT
AUDIO = ROOT / 'experiments/throughput300_candidate/audio_corpus/33_long_eng.wav'
AVATAR = 'en_01_maya_abe451d3_talking_f6cf8be7fb'


async def run(output, *, missing_syncnet=False):
    import aiohttp
    import aiortc
    assert socket.gethostname() == 'a830e00ce20c'
    assert output.is_absolute() and output.parent == ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup'
    assert not output.exists() and not output.is_symlink()
    assert hashlib.sha256(AUDIO.read_bytes()).hexdigest() == '654dcbce843d70451d1123f7649a58bee11bb9dec9a7e835c05b1e367efb2078'
    warm.select_profile('x264tuned_v1')
    warm.isolated_state(warm.request('/worker/state')[1])
    before = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
    assert before['active_streams'] == 0
    pid = before['server']['pid']
    assert b'api_server.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    fields = dict(s.split(':', 1) for s in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in s)
    if missing_syncnet:
        assert not (ROOT / 'models/syncnet/latentsync_syncnet.pt').exists()
        assert int(fields['TracerPid']) == 0
        tracer_pid = 0
    else:
        control = json.loads((ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/native_api_fullfile_trace_v2_1802.control.json').read_text())
        assert int(fields['TracerPid']) == control['tracer_pid'] > 0
        tracer_pid = control['tracer_pid']
    code, warmed = warm.request(f'/avatars/{AVATAR}/cache/warm?batch_size=16&wait=true&timeout_seconds=60', 'POST')
    assert code == 200
    warm.ready_cache(warmed, AVATAR)
    spec = importlib.util.spec_from_file_location('canonical_unscored_client', ROOT / 'load_test_webrtc_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(['--base-url', warm.BASE, '--levels', '1', '--wav-list', str(AUDIO),
                             '--ignore-ice-servers'])
    settings = vars(args)
    settings['post_retries'] = 0
    install_h264_offer(aiortc.RTCPeerConnection, aiortc.RTCRtpSender, aiortc.__version__)
    data = {'schema': 'missing_syncnet_unscored_speech_v1' if missing_syncnet else 'filetrace_unscored_human_speech_probe_v1', 'status': 'IN_PROGRESS',
            'api_pid': pid, 'tracer_pid': tracer_pid, 'missing_syncnet_test': missing_syncnet, 'scored_timing': False,
            'scope': 'One canonical human-WAV turn with SyncNet checkpoint absent, no tracing; no TTS, all-feature, live cadence or startup acceptance' if missing_syncnet else 'One canonical human-WAV turn under full file tracing; no all-feature dependency, live cadence or startup acceptance',
            'avatar_id': AVATAR, 'audio_sha256': hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
            'release_ready': False}
    os.umask(0o077)
    with output.open('x') as handle:
        json.dump(data, handle, indent=2)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as http:
        client = module.StreamClient(settings, {'index': 0, 'avatar_id': AVATAR, 'user_id': 'filetrace_v2_1805'}, http)
        try:
            await client.setup()
            turn = await client.post_turn(str(AUDIO), 0)
            assert turn['status'] == 200, 'speech request not accepted'
            data['turn_http_status'] = turn['status']
            print(json.dumps({'phase': 'SPEECH_ACCEPTED_UNSCORED', 'api_pid': pid}), flush=True)
            for tick in range(13):
                await asyncio.sleep(5)
                if tick in (3, 7, 12):
                    print(json.dumps({'phase': 'CONSUMING_UNSCORED', 'video_frames': len(client.video_t),
                                      'audio_frames': client.audio_frames}), flush=True)
            data.update(video_frames=len(client.video_t), audio_frames=client.audio_frames,
                        client_error_count=len(client.errors))
            assert data['video_frames'] > 0 and data['audio_frames'] > 0 and not client.errors
            data['status'] = 'PASS_OBSERVED_SPEECH_PATH_NOT_TIMING_ACCEPTANCE'
            if missing_syncnet:
                assert not (ROOT / 'models/syncnet/latentsync_syncnet.pt').exists()
                data['status'] = 'PASS_SPEECH_WITH_MISSING_SYNCNET_NOT_TIMING_ACCEPTANCE'
        finally:
            client.stop.set()
            if client.session_id:
                async with http.delete(f'{warm.BASE}/webrtc/sessions/{client.session_id}') as response:
                    data['delete_http_status'] = response.status
            if client.pc:
                await client.pc.close()
            for task in client.tasks:
                task.cancel()
            await asyncio.gather(*client.tasks, return_exceptions=True)
            data['active_streams_after'] = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]['active_streams']
            output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args().out)))
