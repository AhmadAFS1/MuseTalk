"""Default-off, exactly bound native FP16 precision-restoration diagnosis.

This is a new candidate, not the unchanged selective-INT8 release. All imported
plan bytes and source provenance remain explicit; no existing engine is edited.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import types

ROOT = Path('/workspace/MuseTalk')
ENGINE = ROOT / 'models/tensorrt_unet_stagewise_fp16_down3_a3_v1'
OUT = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native/a3_fp16_down3_v1'
HELPER_SHA = 'd5ed578efe8f9c7df364dc3726cb133ec6b380d6c56a990059f9025016f0f620'
RESTORATIONS = {('down3',), ('mid',), ('up0',), ('mid', 'up0')}
BOUND_SHA = 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9'
LINEAGE_SHA = 'd5036b697c06efedcf95c9e81b99efd61129f0f07be50572f03296fabfc3e1ae'


def builder_options(blocks=('down3',)):
    if blocks not in RESTORATIONS:
        raise ValueError('only fixed precision-restoration matrix permitted')
    return types.SimpleNamespace(root=str(ENGINE), opt_level=5, workspace_gb=2.0, no_timing_cache=True,
        hardware_compat='none', strict_timing_cache=True, variant='srccache', timing_cache='',
        blocks=','.join(blocks), int8_blocks='', int8_recipe='', max_minutes=5 * len(blocks), force=True, second_build=False)


def candidate_manifest(native, blocks=('down3',)):
    if blocks not in RESTORATIONS:
        raise ValueError('only fixed precision-restoration matrix permitted')
    manifest = copy.deepcopy(native)
    for key in ('probe', 'runtime', 'missing_blocks', 'finalized_utc', 'total_engine_mib', 'build_log', 'timing_cache_input'):
        manifest.pop(key, None)
    for block in blocks:
        manifest['blocks'].pop(block)
    manifest.update(complete=False, build_log=[], hardware_compatibility_level='none',
        engine_origin='literal_native_plans_plus_fresh_fp16_' + '_'.join(blocks),
        precision_restorations={block: 'int8_qdq_to_fp16' for block in blocks},
        int8_calibration_scope='Original calibration provenance applies to imported INT8 blocks only; named restorations are freshly FP16.',
        build_flags_scope='Final invocation; imported blocks retain their actual original per-block build flags.',
        quality_accepted=False, performance_measured=False, release_ready=False)
    return manifest


def capture(command):
    return subprocess.check_output(command, text=True, timeout=20).strip()


def main(argv=None):
    global ENGINE, OUT
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--blocks', choices=('down3', 'mid', 'up0', 'mid,up0'), default='down3')
    p.add_argument('--owned-target-json', type=Path)
    p.add_argument('--owned-target-sha256')
    p.add_argument('--engine-root', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--successor-inputs', type=Path)
    p.add_argument('--successor-envelope', type=Path)
    p.add_argument('--lineage-receipt', type=Path)
    p.add_argument('--lineage-receipt-sha256')
    a = p.parse_args(argv)
    if not a.execute:
        p.error('explicit --execute required')
    if bool(a.owned_target_json) != bool(a.owned_target_sha256):
        p.error('exact owned descriptor/hash required together')
    blocks = tuple(a.blocks.split(','))
    deadline = dt.datetime(2026, 10, 9, 2, 45, tzinfo=dt.timezone.utc)
    owned, verified = None, None
    host, uuid = '1e7c09cffcb3', 'GPU-ea6411bc-775f-6685-f1a4-28b6b4011a3d'
    if a.owned_target_json:
        if not all((a.engine_root, a.out, a.successor_inputs, a.successor_envelope, a.lineage_receipt, a.lineage_receipt_sha256)):
            p.error('fresh scoped outputs and preregistered input lineage required')
        modules = {}
        for name, digest in (('watch_owned_single_leaf_target.py', BOUND_SHA), ('preregister_metadata_quality_lineage.py', LINEAGE_SHA)):
            path = ROOT / 'scripts/repro_3090' / name
            raw = path.read_bytes()
            if path.is_symlink() or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('checked allocation/lineage source changed')
            module = types.ModuleType('_checked_' + path.stem); module.__file__ = str(path)
            exec(compile(raw, str(path), 'exec'), module.__dict__); modules[name] = module
        owned, deadline = modules['watch_owned_single_leaf_target.py'].binding(a.owned_target_json, a.owned_target_sha256)
        host, uuid = owned['worker_hostname'], owned['gpu_uuid']
        if socket.gethostname() != host:
            raise ValueError('wrong exact owned host')
        base = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
        verified = modules['preregister_metadata_quality_lineage.py'].verify_preregistration(
            parent_inputs=base / 'harnesses/quality-inputs-v1.json', parent_envelope=base / 'quality/reference-envelope-v2.json',
            successor_inputs=a.successor_inputs, successor_envelope=a.successor_envelope,
            receipt_path=a.lineage_receipt, expected_receipt_sha256=a.lineage_receipt_sha256)
        ENGINE, OUT = a.engine_root, a.out
    elif blocks != ('down3',) or any((a.engine_root, a.out, a.successor_inputs, a.successor_envelope, a.lineage_receipt, a.lineage_receipt_sha256)):
        p.error('new precision variants require an exact new-allocation binding')
    if socket.gethostname() != host:
        raise ValueError('wrong owned host')
    if (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 600:
        raise ValueError('cleanup margin required')
    if not os.environ.get('BOX_GUARD_LEASE_FILE') or not Path(os.environ['BOX_GUARD_LEASE_FILE'] + '.holder').is_file():
        raise ValueError('canonical guard required')
    if capture(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']) != uuid:
        raise ValueError('wrong owned GPU')
    if capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        raise ValueError('GPU not isolated')
    helper_path = ROOT / 'scripts/repro_3090/assemble_owned_unet_tail_a3.py'
    raw = helper_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('checked helper changed')
    h = types.ModuleType('_checked_a3_plan_helpers'); h.__file__ = str(helper_path)
    exec(compile(raw, str(helper_path), 'exec'), h.__dict__)
    native = json.loads(h.checked_bytes(h.NATIVE / 'manifest.json', h.NATIVE_SHA))
    h.require(native['complete'] and native['batch'] == 16 and native['variant'] == 'srccache'
              and set(native['blocks']) == h.BLOCKS, 'source manifest incomplete')
    h.require(native['gpu'] == 'NVIDIA GeForce RTX 3090' and native['compute_capability'] == [8, 6], 'source GPU mismatch')
    for path in (OUT, ENGINE):
        h.require(not h.safe(path).exists(), 'fresh output required')
    h.require(ENGINE.resolve().is_relative_to(ROOT / 'models') and ENGINE.resolve() != ROOT / 'models'
              and OUT.resolve().is_relative_to(ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native')
              and OUT.resolve() != ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native', 'scoped output required')
    bodies = {name: h.checked_bytes(ROOT / ('scripts/' + name + '.py'), digest) for name, digest in h.CODE.items()}
    for name, digest in h.MODELS.items():
        h.require(h.file_sha(ROOT / 'models/musetalkV15' / name) == digest, 'model changed')
    manifest = candidate_manifest(native, blocks)
    ENGINE.mkdir(mode=0o700); target = ENGINE / 'bs16'; target.mkdir(mode=0o700); OUT.mkdir(mode=0o700)
    for name, entry in manifest['blocks'].items():
        filename = entry['engine_file']
        h.require(Path(filename).name == filename and filename.endswith('.plan'), 'unsafe filename')
        h.require(h.file_sha(h.NATIVE / filename) == entry['engine_sha256'], 'source plan changed')
        shutil.copyfile(h.NATIVE / filename, target / filename)
        h.require(h.file_sha(target / filename) == entry['engine_sha256'], 'copied plan changed')
    h.write_json(target / 'manifest.json', manifest)
    report = {'schema': 'owned_fp16_precision_restoration_v2', 'status': 'BUILDING', 'instance_id': owned['instance_id'] if owned else '54939993',
        'owned_target': owned, 'input_lineage': verified, 'resource_deadline_utc': deadline.isoformat(),
        'source_manifest_sha256': h.NATIVE_SHA, 'source_code_sha256': h.CODE, 'engine_directory': str(target),
        'precision_restorations': manifest['precision_restorations'], 'quality_accepted': False,
        'performance_measured': False, 'default_selection_changed': False, 'release_ready': False,
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    h.write_json(OUT / 'build.json', report)
    try:
        os.chdir(ROOT); sys.path.insert(0, str(ROOT))
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUSETALK_UNET_STAGEWISE_VERIFY_SHA='1')
        import scripts
        for name, body in bodies.items():
            module = types.ModuleType('scripts.' + name); module.__file__ = str(ROOT / ('scripts/' + name + '.py'))
            sys.modules[module.__name__] = module; setattr(scripts, name, module)
            exec(compile(body, module.__file__, 'exec'), module.__dict__)
        builder = sys.modules['scripts.build_unet_stagewise']; torch = sys.modules['torch']
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:0'); model = builder.load_eager_unet(device)
        final = builder.build_batch(builder_options(blocks), 16, model, device)
        h.require(final['complete'] and set(final['blocks']) == h.BLOCKS, 'incomplete fresh engine')
        for name, entry in manifest['blocks'].items():
            h.require(final['blocks'][name] == entry, 'imported plan metadata changed')
        restored = {block: final['blocks'][block] for block in blocks}
        for block, entry in restored.items():
            h.require(entry['build_flags']['precision'] == 'fp16' and entry['engine_file'] == block + '.plan', 'precision not restored')
        h.require(final['probe']['deterministic_run_to_run'] and final['probe']['graph_equals_direct_enqueue'], 'probe invariants failed')
        for entry in final['blocks'].values():
            h.require(h.file_sha(target / entry['engine_file']) == entry['engine_sha256'], 'final plan changed')
        report.update(status='BUILT_AND_PROBED_QUALITY_PERFORMANCE_UNTESTED', probe=final['probe'],
            restored_blocks=restored, manifest_sha256=h.file_sha(target / 'manifest.json'))
    except BaseException as exc:
        report.update(status='FAILED', error_type=type(exc).__name__); raise
    finally:
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (OUT / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
