"""Explicit A3-only diagnostic build; invoke inside box_guard/watch, never at boot."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

ROOT = Path('/workspace/MuseTalk')
OUT = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native/a3_taesd_fp32_0110'
ENGINE = ROOT / 'models/taesd/trt_native_sm86_fp32_final_conv_a3_0110'
DEADLINE = dt.datetime(2026, 10, 9, 2, 45, tzinfo=dt.timezone.utc)
RUNTIME_SHA = 'e348ceb6cc5ca8b3255716da07ca88c04bf46b4a499f04a66f8a7bc540cfc31b'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def capture(command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def checked_module(path, digest, name):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink_source')
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == digest, 'source_hash_mismatch')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(body, str(path), 'exec'), module.__dict__)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(args.execute, 'explicit_execution_required')
    require(socket.gethostname() == '1e7c09cffcb3', 'wrong_owned_hostname')
    require((DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() > 1200, 'cleanup_margin_required')
    require(capture(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']) ==
            'GPU-ea6411bc-775f-6685-f1a4-28b6b4011a3d', 'wrong_owned_gpu')
    require(not capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']), 'gpu_not_isolated')
    require(bool(os.environ.get('BOX_GUARD_LEASE_FILE')), 'canonical_gpu_guard_required')
    for port in (8000, 8300):
        with socket.socket() as check:
            require(check.connect_ex(('127.0.0.1', port)) != 0, 'owned_server_still_running')
    for path in (OUT, ENGINE):
        require(not path.exists() and not path.is_symlink(), 'fresh_output_required')
        require(path.parent.is_dir() and not any(p.is_symlink() for p in path.parents), 'safe_parent_required')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'MUSETALK_TAESD_TRT_BUILD': '0', 'MUSETALK_TAESD_TRT_STRICT': '1',
        'MUSETALK_TAESD_COMPILE': '0'})
    runtime = checked_module(ROOT / 'scripts/repro_3090/taesd_fp32_candidate_runtime.py',
                             RUNTIME_SHA, '_owned_fp32_runtime')
    runtime.code_identities()
    OUT.mkdir(mode=0o700)
    data = {'schema': 'owned_a3_taesd_fp32_build_v1', 'status': 'IN_PROGRESS',
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'instance_id': 54939993,
        'hostname': socket.gethostname(), 'gpu_uuid': 'GPU-ea6411bc-775f-6685-f1a4-28b6b4011a3d',
        'source_revision': capture(['git', 'rev-parse', 'HEAD']),
        'tracked_source_status': capture(['git', 'status', '--porcelain', '-uno']),
        'builder_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_source_sha256': RUNTIME_SHA, 'engine_dir': str(ENGINE),
        'observed_gpu': capture(['nvidia-smi', '--query-gpu=name,uuid,compute_cap,memory.total,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu', '--format=csv,noheader']),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'quality_accepted': False, 'performance_measured': False,
        'default_selection_changed': False, 'release_ready': False}
    report = OUT / 'build.json'
    runtime._write_new(report, runtime.json_bytes(data))
    try:
        torch, trt, vfd, actual_runtime = runtime._gpu_dependencies('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = vfd.TaesdVaeDecodeBackend.load(device, torch.float16).model
        source = vfd.export_taesd_decoder_onnx(model, 8, device)
        require(runtime.sha(source) == runtime.SOURCE_SHA256, 'canonical_export_mismatch')
        transformer = checked_module(ROOT / 'scripts/repro_3090/taesd_final_conv_fp32.py',
                                     runtime.TRANSFORMER_SHA256, '_owned_fp32_transformer')
        candidate, proof = transformer.transform(source, runtime.SOURCE_SHA256)
        proof_bytes = runtime.json_bytes(proof)
        for name, body in (('source.onnx', source), ('candidate.onnx', candidate),
                           ('transform-proof.json', proof_bytes)):
            runtime._write_new(OUT / name, body)
        del model
        torch.cuda.empty_cache()
        data['runtime'] = actual_runtime
        data['mutation'] = proof
        data['candidate'] = runtime.build_candidate(source_path=OUT / 'source.onnx',
            candidate_path=OUT / 'candidate.onnx', proof_path=OUT / 'transform-proof.json',
            expected_candidate_sha256=runtime.sha(candidate), expected_proof_sha256=runtime.sha(proof_bytes),
            output_dir=ENGINE, device=device, enable_build=True)
        data['status'] = 'BUILD_AND_PROBE_COMPLETE_QUALITY_AND_PERFORMANCE_PENDING'
    except BaseException as exc:
        data['status'] = 'FAILED_BUILD'
        data['error_type'] = type(exc).__name__
        raise
    finally:
        data['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        with report.open('wb') as handle:
            handle.write(runtime.json_bytes(data))
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({'status': data['status'], **data['candidate']}), flush=True)


if __name__ == '__main__':
    main()
