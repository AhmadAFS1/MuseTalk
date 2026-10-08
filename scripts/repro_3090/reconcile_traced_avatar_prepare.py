"""Read-only verification after the original probe's wrong metadata filename assertion.

Never repeats POST, changes a cache, or overwrites the original failed probe evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket

import probe_traced_avatar_prepare as prep
import warm_isolated_api as warm

ORIGINAL = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/fullfile_trace_v2_avatar_prepare_probe_1811.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    output = parser.parse_args().out
    assert socket.gethostname() == 'a830e00ce20c'
    assert output.is_absolute() and output.parent == ORIGINAL.parent
    assert not output.exists() and not output.is_symlink()
    warm.isolated_state(warm.request('/worker/state')[1])
    live = warm.request('/webrtc/sessions/stats?view=lifetime&ring=256')[1]
    assert live['server']['pid'] == 466043 and live['active_streams'] == 0
    assert warm.request('/stats')[1]['active_requests'] == 0
    data = {'schema': 'traced_avatar_prepare_reconciliation_v1', 'status': 'IN_PROGRESS',
            'original_probe': {'path': str(ORIGINAL),
                'sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                'persisted_status': json.loads(ORIGINAL.read_text())['status'],
                'process_exit_code': 1,
                'observed_failure': 'AssertionError: expected avatar_info.json; canonical API writes avator_info.json',
                'original_response_body_persisted': False},
            'api_pid': 466043, 'avatar_id': prep.AVATAR,
            'source_sha256': prep.SOURCE_SHA, 'scored_timing': False, 'release_ready': False,
            'prep_repeated': False, 'cache_mutations': False,
            'scope': 'Completed local cache verified separately; original client assertion failure retained. Not latent-parity or startup acceptance.'}
    with output.open('x') as handle:
        json.dump(data, handle, indent=2)
    try:
        assert hashlib.sha256(prep.SOURCE.read_bytes()).hexdigest() == prep.SOURCE_SHA
        data.update(prep.verify_artifacts(prep.TARGET))
        assert hashlib.sha256((prep.TARGET / 'input_video.mp4').read_bytes()).hexdigest() == prep.SOURCE_SHA
        data['input_video_byte_exact'] = True
        data['cache_status_http'], row = warm.request(f'/avatars/{prep.AVATAR}/cache/status')
        data['cache_status'] = {k: row.get(k) for k in ('avatar_id', 'status', 'disk_prepared', 's3_enabled')}
        assert row['disk_prepared'] is True and row['s3_enabled'] is False
        data['status'] = 'PASS_COMPLETED_CACHE_VERIFICATION_ORIGINAL_HARNESS_FAILED'
    except Exception as exc:
        data['status'] = 'FAILED_RECONCILIATION'
        data['exception'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
