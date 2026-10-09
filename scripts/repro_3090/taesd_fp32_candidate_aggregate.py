"""Default-off, serial-tracking T/SUST diagnostics for an explicit FP32 candidate.

Reuses verified candidate-child routing and the unchanged aggregate validator.
No GPU imports at import time, no engine build/fallback, no quality acceptance.
The caller owns box_guard + single-leaf watch, input lineage, and resource expiry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
CHILD_SHA = '7eef834dd47846ab7083fae06ac2432df9e4a723509d20625a03d351010d7f8f'
REPORT_SHA = 'a102cb4ebdf7ed2f5e7828f3c061a2a43a6ed870a6a2e7fb72767f51b573ef8d'
RUNTIME_SHA = 'e348ceb6cc5ca8b3255716da07ca88c04bf46b4a499f04a66f8a7bc540cfc31b'
RENDER_SHA = 'df4e290b33d752be82d6d2ab738bc5d3e21aa439ffd05f1c3c852a8af1fd4a29'
REPEATS = {'T': 2, 'SUST': 5}
PREFIXES = ('MUSETALK_', 'HLS_', 'WEBRTC_', 'REPRO_', 'LIVE15_', 'OMP_', 'MKL_',
            'OPENBLAS_', 'TORCHINDUCTOR_', 'TRITON_', 'PYTORCH_')


class Rejected(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Rejected(reason)


def checked_module(path, digest, name):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe_helper_path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper_hash_mismatch')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def support():
    child = checked_module(ROOT / 'scripts/repro_3090/taesd_fp32_candidate_child.py', CHILD_SHA, '_aggregate_checked_child')
    require(child.RUNTIME_SHA == RUNTIME_SHA and child.TARGETS['render'][1] == RENDER_SHA, 'routing_pins_changed')
    report = checked_module(ROOT / 'scripts/repro_3090/report.py', REPORT_SHA, '_aggregate_checked_report')
    return child, report


def aggregate_arguments(stage, output, label):
    require(stage in REPEATS, 'unsupported_stage')
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,119}', label), 'invalid_label')
    # Exactly runner.aggregate_command's six-stream profile, with overlap OFF.
    # Do not add flags changing renderer batch/depth/thread/chin defaults.
    return ['--backend', 'stagewise16_taesdtrt', '--streams', '6', '--loops', '24',
            '--repeats', str(REPEATS[stage]), '--min-timed-s', '60', '--thermal-warmup-s', '120',
            '--out-root', str(output), '--label', label]


def profile_values(raw):
    values = {}
    for line in raw.decode('utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        require(sep and re.fullmatch('(?:MUSETALK|HLS|WEBRTC|HF|TRANSFORMERS|OMP|MKL|OPENBLAS)_[A-Z0-9_]+', key), 'unsafe_profile_key')
        require(not re.search('TOKEN|SECRET|PASSWORD|ACCESS_KEY|CREDENTIAL|AUTH', key), 'profile_secret_forbidden')
        require(key not in values and not any(c in value for c in ('$', '`', '\n')), 'duplicate_or_expanding_profile')
        values[key] = value.strip('"').strip("'")
    return values


@contextmanager
def environment(values):
    before = dict(os.environ)
    try:
        for key in tuple(os.environ):
            if key.startswith(PREFIXES):
                os.environ.pop(key)
        os.environ.update(values)
        yield
    finally:
        os.environ.clear(); os.environ.update(before)


def new_file(path, raw):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600), 'wb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())


def validate_measurement(report, data, stage, engine_root, engine_manifest, meta):
    require(data['args']['loops'] == 24 and data['args']['repeats'] == REPEATS[stage]
            and len(data['repeats']) == REPEATS[stage], 'exact_aggregate_windows_required')
    require(data['args'].get('tracking_overlap', False) is False, 'overlap_requires_separate_candidate_parity')
    report.loaded_backend(data, engine_root, meta['key'], meta['decoder_plan_sha256'], engine_manifest)
    return report.aggregate(data, stage, 400)


def launch(*, stage, manifest_path, manifest_sha256, candidate_key, profile_path, profile_sha256,
           engine_root, engine_manifest_sha256, receipt_path, output_dir, label,
           accepted_root='/workspace/experiments/avatar_diversity_20260927', enable=False):
    require(enable is True, 'explicit_enable_required')
    child, report = support()
    manifest, profile, engine, receipt, out, accepted = map(child.safe_path,
        (manifest_path, profile_path, engine_root, receipt_path, output_dir, accepted_root))
    argv = aggregate_arguments(stage, out, label)
    require(out.is_dir() and receipt.parent.is_dir() and not receipt.exists(), 'fresh_receipt_and_existing_output_required')
    raw_report, run_dir, effective = out / (label + '.json'), out / label, out / (label + '.effective.env')
    require(not any(p.exists() or p.is_symlink() for p in (raw_report, run_dir, effective)), 'fresh_run_paths_required')
    require(accepted.is_dir(), 'accepted_inputs_unavailable')
    runtime = child.verified_runtime()
    meta = runtime.parse_json(child.checked(manifest, manifest_sha256))
    require(meta.get('schema') == runtime.SCHEMA and meta.get('status') == 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED', 'candidate_manifest_required')
    require(re.fullmatch('[0-9a-f]{20}', candidate_key) and meta.get('key') == candidate_key, 'explicit_candidate_key_mismatch')
    require(manifest.name == 'taesd_trt_' + candidate_key + '.json', 'candidate_manifest_name_mismatch')
    require(all(runtime.valid_digest(meta.get(k)) for k in ('decoder_plan_sha256', 'post_plan_sha256')), 'candidate_plan_digests_required')
    native_manifest = runtime.parse_json(child.checked(engine / 'bs16/manifest.json', engine_manifest_sha256))
    require(native_manifest.get('complete') is True and native_manifest.get('batch') == 16
            and native_manifest.get('variant') == 'srccache', 'complete_bs16_source_cache_required')
    values = profile_values(child.checked(profile, profile_sha256))
    # Same runner bindings; compiled-reference-only mode must not enter throughput.
    values.update(MUSETALK_UNET_BACKEND='trt_stagewise', MUSETALK_UNET_STAGEWISE_BATCH='16',
        MUSETALK_UNET_STAGEWISE_CACHE_DIR=str(engine), MUSETALK_TRT_FALLBACK='0',
        MUSETALK_UNET_STAGEWISE_VERIFY_SHA='1', MUSETALK_UNET_STAGEWISE_PROBE_CHECK='1',
        MUSETALK_UNET_STAGEWISE_PROBE_TOL='0', MUSETALK_VAE_BACKEND='taesd',
        MUSETALK_TAESD_BACKEND='trt', MUSETALK_TAESD_TRT_BUILD='0', MUSETALK_TAESD_TRT_STRICT='1',
        MUSETALK_TAESD_TRT_BATCH='8', MUSETALK_TAESD_WARMUP_BATCHES='8',
        MUSETALK_TAESD_TRT_DIR=str(manifest.parent), MUSETALK_TAESD_COMPILE='0',
        MUSETALK_TAESD_TRT_FUSED_POST='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    target = ROOT / child.TARGETS['render'][0]
    target_bytes = child.checked(target, RENDER_SHA)
    vfd_bytes = child.checked(ROOT / 'scripts/vae_fast_decoder.py', child.VFD_SHA)
    new_file(effective, ''.join(f'{k}={v}\n' for k, v in sorted(values.items())).encode())
    values.update(MUSETALK_REPRO_RUNTIME_ENV=str(effective), MUSETALK_REPRO_ACCEPTED=str(accepted),
                  MUSETALK_REPRO_WORKSPACE=str(accepted.parents[1]))
    state = dict(schema='taesd_fp32_candidate_aggregate_v1', status='INVALID', returncode=2,
        stage=stage, target_returncode=None, candidate_key=candidate_key, manifest_sha256=manifest_sha256,
        decoder_plan_sha256=meta['decoder_plan_sha256'], post_plan_sha256=meta['post_plan_sha256'],
        candidate_calls=0, candidate_attempts=0, invocations=[], target_args=argv,
        target_sha256=RENDER_SHA, child_sha256=CHILD_SHA, runtime_sha256=RUNTIME_SHA, report_sha256=REPORT_SHA,
        profile_sha256=profile_sha256, effective_profile_sha256=hashlib.sha256(effective.read_bytes()).hexdigest(),
        engine_manifest_sha256=engine_manifest_sha256, tracking_overlap=False, candidate_diagnostic_only=True,
        quality_accepted=False, performance_accepted=False, release_ready=False,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(state, handle); handle.flush(); os.fsync(handle.fileno())
        try:
            with child.process_context(target, argv, out, 'render'), environment(values), \
                 child.scoped_candidate(runtime, manifest, manifest_sha256, meta, vfd_bytes, state):
                try:
                    child.execute_target(target_bytes, target)
                    state['target_returncode'] = 0
                except SystemExit as exc:
                    state['target_returncode'] = exc.code if type(exc.code) is int else (0 if exc.code is None else 1)
            require(state['candidate_calls'] >= 1 and state['candidate_calls'] == state['candidate_attempts'], 'candidate_routing_incomplete')
            if state['target_returncode'] != 0:
                state.update(status='CHILD_FAILED', returncode=state['target_returncode'])
            else:
                data = report.child_result(0, [raw_report])[0]
                result = validate_measurement(report, data, stage, engine, native_manifest, meta)
                state.update(status=result['status'], returncode=report.EXIT[result['status']], aggregate=result,
                             raw_report_sha256=report.sha256(raw_report))
        except BaseException as exc:
            state.update(status='INVALID', returncode=130 if isinstance(exc, KeyboardInterrupt) else 2,
                         error_type=type(exc).__name__)
        finally:
            state['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            handle.seek(0); json.dump(state, handle, indent=2, sort_keys=True); handle.write('\n')
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
    return state['returncode']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--enable', action='store_true')
    parser.add_argument('--stage', choices=sorted(REPEATS), required=True)
    for name in ('manifest', 'manifest-sha256', 'candidate-key', 'profile', 'profile-sha256',
                 'engine-root', 'engine-manifest-sha256', 'receipt', 'output-dir', 'label'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--accepted-root', default='/workspace/experiments/avatar_diversity_20260927')
    args = vars(parser.parse_args(argv))
    for cli, api in (('manifest', 'manifest_path'), ('profile', 'profile_path'), ('receipt', 'receipt_path')):
        args[api] = args.pop(cli)
    return launch(**args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Rejected, ValueError, OSError):
        print('candidate aggregate INVALID: contract rejected', file=sys.stderr)
        raise SystemExit(2)
