"""Default-off A3 mixed-tail diagnostic; copy plans, repository finalize-only, no builds."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import types

ROOT = Path('/workspace/MuseTalk')
NATIVE = ROOT / 'models/tensorrt_unet_stagewise_sm86_r5_v1/bs16'
PORTABLE = ROOT / 'models/tensorrt_unet_stagewise_ampere_plus_r5/bs16'
ENGINE = ROOT / 'models/tensorrt_unet_stagewise_mixed_native_portable_tail_a3_v1'
OUT = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native/a3_mixed_unet_tail_v1'
DEADLINE = dt.datetime(2026, 10, 9, 2, 45, tzinfo=dt.timezone.utc)
UUID = 'GPU-ea6411bc-775f-6685-f1a4-28b6b4011a3d'
NATIVE_SHA = 'f66b46ca38d0e34af69ee5c01be52d93cc3d2f1ba3ac7b8a68ae0426c3658316'
PORTABLE_SHA = '89eb27dc385e52fcae1602cc65f2c2a1e136383e5072ccf31b792978e84be24b'
TAIL_ONNX = '8219d8a31bf256b0c621b5f4662c2148613818e985a0a712bae0e2e03e4eb527'
BLOCKS = {'prefix', 'down0rest', 'down1', 'down2', 'down3', 'mid', 'up0', 'up1', 'up2', 'up3', 'tail'}
CODE = {'trt_timing_cache': '7fb2294e37f8e31470a86728f22a25d926ce87ea7f3751f5339a044496ce2e75',
        'unet_stagewise_trt': '4ab523ab9337f1ce903a52acf7a3f06e7423669cbf2df2b3846b8ffc374b6e9c',
        'build_unet_stagewise': '1f0c0c54f2d94dc7358a7b79f4e0176dfdb747b78140552fad35c34d32917d49'}
MODELS = {'unet.pth': '7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007',
          'musetalk.json': '5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe(path):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe_path')
    return path


def file_sha(path):
    path = safe(path); result = hashlib.sha256()
    require(path.is_file(), 'regular_file_required')
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK), 'rb') as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), 'regular_file_required')
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def checked_bytes(path, digest):
    path = safe(path); require(path.is_file(), 'regular_file_required')
    body = path.read_bytes()
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
            and hashlib.sha256(body).hexdigest() == digest, 'source_sha256_mismatch')
    return body


def write_json(path, value):
    with safe(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2); handle.write('\n')


def prepare(native_dir, portable_dir, target_root, native_sha=NATIVE_SHA, portable_sha=PORTABLE_SHA):
    native = json.loads(checked_bytes(Path(native_dir) / 'manifest.json', native_sha))
    portable = json.loads(checked_bytes(Path(portable_dir) / 'manifest.json', portable_sha))
    target_root = safe(target_root); require(not target_root.exists(), 'fresh_output_required')
    for key, value in {'schema': 'musetalk_unet_stagewise_trt_v1', 'batch': 16, 'variant': 'srccache',
                       'timestep': 0, 'complete': True, 'tensorrt_version': '10.3.0', 'torch_version': '2.5.1+cu121'}.items():
        require(native.get(key) == portable.get(key) == value, 'source_manifest_schema_or_runtime')
    require(native['spec'] == portable['spec'] and set(native['blocks']) == set(portable['blocks']) == BLOCKS, 'source_block_spec')
    require(native['gpu'] == 'NVIDIA GeForce RTX 3090' and native['compute_capability'] == [8, 6]
            and native.get('hardware_compatibility_level', 'none') == 'none', 'native_target_identity')
    require(portable['gpu'] == 'NVIDIA GeForce RTX 4070 SUPER' and portable['compute_capability'] == [8, 9]
            and portable.get('hardware_compatibility_level') == 'ampere_plus', 'portable_target_identity')
    require(native['int8_calibration']['recipe_sha256_16'] == portable['int8_calibration']['recipe_sha256_16'] == 'f6f90777264b5302'
            and native['int8_calibration']['files'] == portable['int8_calibration']['files'], 'source_recipe_identity')
    require(native['blocks']['tail']['onnx_sha256'] == portable['blocks']['tail']['onnx_sha256'] == TAIL_ONNX,
            'tail_source_graph_mismatch')
    selections, filenames = {}, set()
    for name in sorted(BLOCKS):
        origin, origin_dir, origin_sha = (portable, Path(portable_dir), portable_sha) if name == 'tail' else (native, Path(native_dir), native_sha)
        entry = copy.deepcopy(origin['blocks'][name]); filename = entry['engine_file']
        require(isinstance(filename, str) and Path(filename).name == filename and filename.endswith('.plan')
                and filename not in filenames, 'unsafe_or_duplicate_plan_filename')
        filenames.add(filename)
        level = entry['build_flags'].get('hardware_compatibility_level', 'none')
        require(level == origin.get('hardware_compatibility_level', 'none'), 'plan_compatibility_mismatch')
        require(name != 'tail' or entry['build_flags']['precision'] == 'fp16', 'tail_precision_mismatch')
        require(file_sha(origin_dir / filename) == entry['engine_sha256'], 'source_plan_sha256_mismatch')
        selections[name] = origin, origin_dir, origin_sha, entry
    manifest = copy.deepcopy(native)
    for key in ('probe', 'runtime', 'missing_blocks', 'finalized_utc', 'total_engine_mib', 'build_log', 'timing_cache_input'):
        manifest.pop(key, None)
    manifest.update(complete=False, hardware_compatibility_level='none', build_log=[],
        engine_origin='mixed_literal_existing_plans_no_export_no_build', supported_compute_capability=[8, 6],
        build_flags_scope='finalization_invocation_only; actual imported plan flags remain per-block',
        quality_accepted=False, performance_measured=False, release_ready=False, block_provenance={})
    target_root.mkdir(mode=0o700); target = target_root / 'bs16'; target.mkdir(mode=0o700)
    for name, (origin, origin_dir, origin_sha, entry) in selections.items():
        filename = entry['engine_file']
        digest = hashlib.sha256()
        with safe(origin_dir / filename).open('rb') as source, (target / filename).open('xb') as destination:
            for block in iter(lambda: source.read(8 << 20), b''):
                destination.write(block); digest.update(block)
        require(digest.hexdigest() == entry['engine_sha256'], 'source_plan_sha256_mismatch')
        manifest['blocks'][name] = entry
        manifest['block_provenance'][name] = {'source_directory': str(origin_dir), 'source_manifest_sha256': origin_sha,
            'source_gpu': origin['gpu'], 'source_compute_capability': origin['compute_capability'],
            'hardware_compatibility_level': origin.get('hardware_compatibility_level', 'none'),
            'engine_file': filename, 'engine_sha256': entry['engine_sha256'], 'onnx_sha256': entry['onnx_sha256']}
    write_json(target / 'manifest.json', manifest)
    return manifest


def capture(command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False); parser.add_argument('--execute', action='store_true')
    require(parser.parse_args(argv).execute, 'explicit_execution_required')
    require(socket.gethostname() == '1e7c09cffcb3', 'wrong_owned_hostname')
    require((DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() > 600, 'cleanup_margin_required')
    require(capture(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']) == UUID, 'wrong_owned_gpu')
    require(not capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']), 'gpu_not_isolated')
    require(bool(os.environ.get('BOX_GUARD_LEASE_FILE')), 'canonical_guard_required')
    for path in (OUT, ENGINE):
        require(not safe(path).exists() and path.parent.is_dir(), 'fresh_output_required')
    bodies = {name: checked_bytes(ROOT / f'scripts/{name}.py', digest) for name, digest in CODE.items()}
    for name, digest in MODELS.items():
        require(file_sha(ROOT / 'models/musetalkV15' / name) == digest, 'model_identity_mismatch')
    OUT.mkdir(mode=0o700)
    report = {'schema': 'owned_a3_mixed_unet_tail_v1', 'status': 'PREPARING', 'instance_id': 54939993,
              'gpu_uuid': UUID, 'engine_directory': str(ENGINE / 'bs16'), 'source_code_sha256': CODE,
              'native_manifest_sha256': NATIVE_SHA, 'portable_manifest_sha256': PORTABLE_SHA,
              'quality_accepted': False, 'performance_measured': False, 'default_selection_changed': False, 'release_ready': False}
    write_json(OUT / 'assembly.json', report)
    try:
        prepared = prepare(NATIVE, PORTABLE, ENGINE)
        os.chdir(ROOT); sys.path.insert(0, str(ROOT))
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUSETALK_UNET_STAGEWISE_VERIFY_SHA='1')
        import scripts
        for name, body in bodies.items():
            module = types.ModuleType('scripts.' + name); module.__file__ = str(ROOT / f'scripts/{name}.py')
            sys.modules[module.__name__] = module; setattr(scripts, name, module)
            exec(compile(body, module.__file__, 'exec'), module.__dict__)
        builder, sw = sys.modules['scripts.build_unet_stagewise'], sys.modules['scripts.unet_stagewise_trt']
        def forbidden(*args, **kwargs):
            raise ValueError('export_or_engine_build_forbidden')
        sw.export_block_onnx = sw.build_engine_from_onnx = forbidden
        args = types.SimpleNamespace(root=str(ENGINE), opt_level=5, workspace_gb=2.0, no_timing_cache=True,
            hardware_compat='none', strict_timing_cache=True, variant='srccache', timing_cache='',
            blocks='__FINALIZE_ONLY__', int8_blocks='', int8_recipe='', max_minutes=0, force=False, second_build=False)
        require(not (set(args.blocks.split(',')) & set(sw.block_order(args.variant))), 'nonempty_build_selector')
        torch = sys.modules['torch']; device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
        model = builder.load_eager_unet(device)
        final = builder.build_batch(args, 16, model, device)
        require(final['complete'] and final['blocks'] == prepared['blocks'], 'finalization_or_plan_metadata_changed')
        require(final['probe']['deterministic_run_to_run'] and final['probe']['graph_equals_direct_enqueue'], 'probe_invariants_failed')
        for entry in final['blocks'].values():
            require(file_sha(ENGINE / 'bs16' / entry['engine_file']) == entry['engine_sha256'], 'final_plan_changed')
        report.update(status='FINALIZED_QUALITY_AND_PERFORMANCE_UNTESTED', probe=final['probe'],
                      manifest_sha256=file_sha(ENGINE / 'bs16/manifest.json'), block_provenance=final['block_provenance'])
    except BaseException as exc:
        report.update(status='FAILED', error_type=type(exc).__name__); raise
    finally:
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (OUT / 'assembly.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
