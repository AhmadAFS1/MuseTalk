"""Default-off, owned single-leaf finalization of a preregistered plan overlay.

The outer caller must use box_guard and watch_owned_single_leaf_target. All
878 quality inputs and metadata-only ancestry are reverified before GPU work.
No graph export/build, default-selection change, or quality approval occurs.
"""
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
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
PINS = {'unet_plan_overlay.py': '3c84b305e9906151dadc7d4985f6ee94ac641924d48684251a7f8678d7deaf1f',
        'watch_owned_single_leaf_target.py': 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9',
        'preregister_metadata_quality_lineage.py': 'd5036b697c06efedcf95c9e81b99efd61129f0f07be50572f03296fabfc3e1ae'}


def checked_module(name):
    path = ROOT / 'scripts/repro_3090' / name
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise ValueError('checked diagnostic source changed')
    module = types.ModuleType('_checked_' + path.stem); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--owned-target-json', type=Path, required=True)
    p.add_argument('--owned-target-sha256', required=True)
    p.add_argument('--variant', choices=('portable_prefix', 'portable_prefix_down0rest', 'portable_core_native_up3'), required=True)
    p.add_argument('--engine-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--successor-inputs', type=Path, required=True)
    p.add_argument('--successor-envelope', type=Path, required=True)
    p.add_argument('--lineage-receipt', type=Path, required=True)
    p.add_argument('--lineage-receipt-sha256', required=True)
    a = p.parse_args(argv)
    if not a.execute:
        raise ValueError('explicit execution required')
    target, deadline = checked_module('watch_owned_single_leaf_target.py').binding(a.owned_target_json, a.owned_target_sha256)
    if socket.gethostname() != target['worker_hostname'] or (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 600:
        raise ValueError('owned host or cleanup margin mismatch')
    holder = Path(os.environ.get('BOX_GUARD_LEASE_FILE', '') + '.holder')
    if not os.environ.get('BOX_GUARD_LEASE_FILE') or not holder.is_file():
        raise ValueError('canonical guard required')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10).strip()
    if uuid != target['gpu_uuid'] or apps:
        raise ValueError('owned GPU identity or initial isolation mismatch')
    overlay = checked_module('unet_plan_overlay.py'); h = overlay.checked_helpers()
    for path, parent in ((a.engine_root, ROOT / 'models'), (a.out, BASE / 'native')):
        h.require(h.safe(path).is_relative_to(parent) and path != parent and not path.exists(), 'fresh scoped output required')
    lineage = checked_module('preregister_metadata_quality_lineage.py')
    verified = lineage.verify_preregistration(parent_inputs=BASE / 'harnesses/quality-inputs-v1.json',
        parent_envelope=BASE / 'quality/reference-envelope-v2.json', successor_inputs=a.successor_inputs,
        successor_envelope=a.successor_envelope, receipt_path=a.lineage_receipt,
        expected_receipt_sha256=a.lineage_receipt_sha256)
    bodies = {name: h.checked_bytes(ROOT / ('scripts/' + name + '.py'), sha) for name, sha in h.CODE.items()}
    for name, sha in h.MODELS.items():
        h.require(h.file_sha(ROOT / 'models/musetalkV15' / name) == sha, 'model changed')
    a.out.mkdir(mode=0o700)
    record = dict(schema='owned_preregistered_unet_overlay_v1', instance_id=target['instance_id'],
        gpu_uuid=uuid, owned_target_sha256=a.owned_target_sha256, variant=a.variant,
        source_pins=PINS, builder_source_pins=h.CODE, input_lineage=verified, status='PREPARING',
        quality_accepted=False, performance_measured=False, default_selection_changed=False, release_ready=False)
    h.write_json(a.out / 'assembly.json', record)
    try:
        prepared = overlay.prepare(h.NATIVE, h.PORTABLE, a.engine_root, variant=a.variant)
        os.chdir(ROOT); sys.path.insert(0, str(ROOT))
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUSETALK_UNET_STAGEWISE_VERIFY_SHA='1')
        import scripts
        for name, body in bodies.items():
            module = types.ModuleType('scripts.' + name); module.__file__ = str(ROOT / ('scripts/' + name + '.py'))
            sys.modules[module.__name__] = module; setattr(scripts, name, module)
            exec(compile(body, module.__file__, 'exec'), module.__dict__)
        builder, sw = sys.modules['scripts.build_unet_stagewise'], sys.modules['scripts.unet_stagewise_trt']
        def forbidden(*_args, **_kwargs):
            raise ValueError('engine build or graph export forbidden')
        sw.export_block_onnx = sw.build_engine_from_onnx = forbidden
        options = types.SimpleNamespace(root=str(a.engine_root), opt_level=5, workspace_gb=2.0, no_timing_cache=True,
            hardware_compat='none', strict_timing_cache=True, variant='srccache', timing_cache='',
            blocks='__FINALIZE_ONLY__', int8_blocks='', int8_recipe='', max_minutes=0, force=False, second_build=False)
        h.require(not set(options.blocks.split(',')) & set(sw.block_order(options.variant)), 'build selector not empty')
        torch = sys.modules['torch']; device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
        model = builder.load_eager_unet(device)
        final = builder.build_batch(options, 16, model, device)
        h.require(final['complete'] and final['blocks'] == prepared['blocks'], 'imported plans or metadata changed')
        h.require(final['probe']['deterministic_run_to_run'] and final['probe']['graph_equals_direct_enqueue'], 'hard probe failure')
        for entry in final['blocks'].values():
            h.require(h.file_sha(a.engine_root / 'bs16' / entry['engine_file']) == entry['engine_sha256'], 'final plan changed')
        record.update(status='FINALIZED_QUALITY_PERFORMANCE_UNTESTED', probe=final['probe'],
            block_provenance=final['block_provenance'], manifest_sha256=h.file_sha(a.engine_root / 'bs16/manifest.json'))
    except BaseException as exc:
        record.update(status='FAILED', error_type=type(exc).__name__); raise
    finally:
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (a.out / 'assembly.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
