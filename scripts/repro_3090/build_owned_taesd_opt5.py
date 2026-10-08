"""Fresh same-FP16 native decoder opt5 diagnostic, then the unchanged complete gate.

Invoke only inside box_guard/watch. No recipe/default selection or old artifact mutation.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/workspace/MuseTalk')
OUTPUT = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native/taesd_opt5_v2_1827'
ENGINE = ROOT / 'models/taesd/trt_native_sm86_opt5_v2_1827'


def main():
    assert socket.gethostname() == 'a830e00ce20c'
    assert (dt.datetime(2026, 10, 8, 19, tzinfo=dt.timezone.utc) - dt.datetime.now(dt.timezone.utc)).total_seconds() > 1200
    assert not OUTPUT.exists() and not OUTPUT.is_symlink()
    assert not ENGINE.exists() and not ENGINE.is_symlink()
    with socket.socket() as check:
        assert check.connect_ex(('127.0.0.1', 8300)) != 0
    assert subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                   text=True, timeout=10).strip() == ''
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=10).strip() == 'GPU-5640f670-debe-ec22-1cfb-4b1f63bc1d53'
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'MUSETALK_TAESD_TRT_DIR': str(ENGINE), 'MUSETALK_TAESD_TRT_OPT_LEVEL': '5',
        'MUSETALK_TAESD_TRT_BATCH': '8', 'MUSETALK_TAESD_TRT_HW_COMPAT': 'none',
        'MUSETALK_TAESD_TRT_STRONGLY_TYPED': '0', 'MUSETALK_TAESD_TRT_STRICT': '1',
        'MUSETALK_TAESD_TRT_BUILD': '0', 'MUSETALK_TAESD_WARMUP_BATCHES': '8',
        'REPRO_GATE_OUT': str(OUTPUT / 'gate')})
    OUTPUT.mkdir()
    report = OUTPUT / 'build.json'
    data = {'schema': 'owned_native_taesd_opt5_diagnostic_v1', 'status': 'IN_PROGRESS',
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'engine_dir': str(ENGINE),
            'single_changed_lever': 'builder optimization level 3 to 5; same graph, FP16, batch8 and native none compatibility',
            'quality_acceptance': False, 'release_ready': False, 'default_selection_changed': False}
    with report.open('x') as handle:
        json.dump(data, handle, indent=2)
    try:
        import torch
        from scripts import vae_fast_decoder as vfd
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda:0')
        backend = vfd.TaesdVaeDecodeBackend.load(device=device, runtime_dtype=torch.float16)
        meta = vfd.build_taesd_trt_engines(backend.model, device, batch=8, engine_dir=ENGINE,
                    opt_level=5, strongly_typed=False, force=False, hw_compat='none')
        fingerprint = meta['fingerprint']
        assert fingerprint['onnx_sha256'] == '466225e995f0e70a194eccd8136f3c08fe2b5ca4c93f888c522ea2070a2044bb'
        assert fingerprint['opt_level'] == 5 and fingerprint['precision'] == 'fp16' and fingerprint['batch'] == 8
        assert fingerprint['compute_capability'] == '8.6' and meta['build']['hardware_compatibility_level'] == 'none'
        for kind in ('decoder', 'post'):
            path = ENGINE / meta[kind + '_plan']
            assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == meta[kind + '_plan_sha256']
        assert meta['probe']['fused_vs_repo_post_mismatched_bytes'] == 0
        data['meta'] = meta
        data['status'] = 'PASS_BUILD_IDENTITY_ONLY_COMPLETE_QUALITY_GATE_PENDING'
    except Exception as exc:
        data['status'] = 'FAILED_BUILD_IDENTITY'
        data['exception'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'build_status': data['status'], 'key': meta['key'], 'release_ready': False}), flush=True)
    # exec discards the builder's CUDA allocations before canonical gate startup.
    gate = [sys.executable, str(ROOT / 'scripts/repro_400fps/gate_taesd_trt.py'),
            '--no-record', '--corpus', str(ROOT / 'calibration/unet_multi_avatar_20260928')]
    os.execve(sys.executable, gate, os.environ.copy())


if __name__ == '__main__':
    main()
