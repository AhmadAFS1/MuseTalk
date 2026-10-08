"""One bounded encoder-open diagnostic on idle owned A1 under its GPU lease."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/workspace/MuseTalk')
sys.path.insert(0, str(ROOT))
from scripts.webrtc_h264_override import open_nvenc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'a830e00ce20c'
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime(2026, 10, 8, 19, tzinfo=dt.timezone.utc)
    assert args.out.is_absolute() and args.out.parent == ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native'
    assert not args.out.exists() and not args.out.is_symlink()
    with socket.socket() as check:
        assert check.connect_ex(('127.0.0.1', 8300)) != 0, 'API still listening'
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    assert uuid == 'GPU-5640f670-debe-ec22-1cfb-4b1f63bc1d53'
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10).strip()
    assert not pids, 'foreign GPU work'
    import av
    av.logging.set_level(av.logging.DEBUG)
    data = {'schema': 'owned3090_nvenc_open_diagnostic_v1', 'status': 'IN_PROGRESS',
            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'gpu_uuid': uuid,
            'av_version': av.__version__, 'av_library_versions': av.library_versions,
            'driver_capabilities_environment': os.getenv('NVIDIA_DRIVER_CAPABILITIES'),
            'native_quality_status': 'REJECTED_UNCHANGED', 'release_ready': False,
            'library_paths': {name: str((Path('/usr/lib/x86_64-linux-gnu') / name).resolve())
                              for name in ('libnvidia-encode.so.1', 'libnvcuvid.so.1', 'libcuda.so.1')}}
    with av.logging.Capture(local=True) as logs:
        try:
            codec = open_nvenc(av, 512, 896, 1000000, 30, 'p2', 'll')
            frame = av.VideoFrame(512, 896, 'yuv420p')
            for i, plane in enumerate(frame.planes):
                plane.update(bytes([16 if i == 0 else 128]) * plane.buffer_size)
            frame.pts = 0
            packets = list(codec.encode(frame)) + list(codec.encode(None))
            assert packets, 'encoder produced no packet'
            data.update(status='PASS_SINGLE_OPEN_AND_BLACK_FRAME_NOT_LIVE_ACCEPTANCE', packet_count=len(packets))
        except Exception as exc:
            data.update(status='FAIL_SINGLE_NVENC_OPEN', exception_type=type(exc).__name__,
                        exception_message=str(exc)[:1024])
    data['ffmpeg_log_tail'] = [{'level': row[0], 'component': row[1], 'message': row[2][:2048]} for row in logs[-20:]]
    data['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    os.umask(0o077)
    with args.out.open('x') as output:
        json.dump(data, output, indent=2)
        output.write('\n')
    print(json.dumps(data))
    return 0 if data['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
