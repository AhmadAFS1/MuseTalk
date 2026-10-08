"""Create only a fresh diagnostic avatar under file tracing, with canonical source unchanged."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import socket

import warm_isolated_api as warm

SOURCE = Path('/workspace/experiments/avatar_diversity_20260927/black_man_short_beard/source_exact10.mp4')
SOURCE_SHA = '064183b0d8f80105159c687858566ad271fd768037773f8cfaacb3ad3452837e'
AVATAR = 'startup_trace_v2_black_man_1810'
TARGET = warm.ROOT / 'results/v15/avatars' / AVATAR


def verify_artifacts(target):
    # Keep the canonical API's historical spelling; do not rename its output.
    required = ['latents.pt', 'coords.pkl', 'mask_coords.pkl', 'avator_info.json']
    artifacts = []
    for name in required:
        path = target / name
        assert path.is_file() and not path.is_symlink(), name
        artifacts.append({'name': name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    frames = len(list((target / 'full_imgs').glob('*.png')))
    masks = len(list((target / 'mask').glob('*.png')))
    assert frames == masks == 480, (frames, masks)
    return {'artifacts': artifacts, 'frame_pngs': frames, 'mask_pngs': masks,
            'source_frames': 240,
            'cache_cycle_policy': 'Canonical API forward plus reversed source cycle, no source-video modification'}


async def run(output):
    import aiohttp
    assert socket.gethostname() == 'a830e00ce20c'
    assert output.is_absolute() and output.parent == warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup'
    assert not output.exists() and not output.is_symlink()
    assert SOURCE.stat().st_size == 15293023 and hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    assert not TARGET.exists() and not TARGET.is_symlink()
    warm.isolated_state(warm.request('/worker/state')[1])
    live = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
    assert live['active_streams'] == 0 and warm.request('/stats')[1]['active_requests'] == 0
    pid = live['server']['pid']
    assert b'api_server.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    fields = dict(s.split(':', 1) for s in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in s)
    assert int(fields['TracerPid']) > 0
    data = {'schema': 'traced_canonical_avatar_prepare_probe_v1', 'status': 'IN_PROGRESS',
            'source': str(SOURCE), 'source_sha256': SOURCE_SHA, 'new_avatar_id': AVATAR,
            'target': str(TARGET), 'api_pid': pid, 'scored_timing': False,
            'release_ready': False, 's3_enabled': False,
            'scope': 'Fresh diagnostic API preparation only; not replacement of any published avatar or latent-parity acceptance'}
    with output.open('x') as handle:
        json.dump(data, handle, indent=2)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as http:
        form = aiohttp.FormData()
        form.add_field('video_file', SOURCE.read_bytes(), filename='source_exact10.mp4', content_type='video/mp4')
        async with http.post(warm.BASE + '/avatars/prepare', params={'avatar_id': AVATAR, 'batch_size': 16,
                              'bbox_shift': 0, 'force_recreate': 'false'}, data=form) as response:
            body = await response.json()
            data['http_status'] = response.status
            data['response'] = {k: body.get(k) for k in ('status', 'avatar_id', 'already_prepared', 's3_uploaded', 'video_layout')}
            assert response.status == 200 and body['status'] == 'success' and body['avatar_id'] == AVATAR
            assert body['already_prepared'] is False and body['s3_uploaded'] is False
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    data.update(verify_artifacts(TARGET))
    data['source_unchanged'] = True
    data['status'] = 'PASS_OBSERVED_FRESH_PREP_PATH_NOT_LATENT_PARITY'
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args().out)))
