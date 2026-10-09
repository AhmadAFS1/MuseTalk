"""One default-off native opt3 build with unchanged graph/precision and lineage.

Run only inside the canonical lease and versioned single-CUDA-leaf monitor.
Stops at the first ONNX/precision mismatch, even when the builder returns zero.
No quality, throughput or release acceptance is issued by a successful build.
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

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
ENGINE = ROOT / 'models/tensorrt_unet_stagewise_sm86_r5_opt3_v2'
BASELINE = ROOT / 'models/tensorrt_unet_stagewise_sm86_r5_v1/bs16/manifest.json'
BASELINE_SHA = 'f66b46ca38d0e34af69ee5c01be52d93cc3d2f1ba3ac7b8a68ae0426c3658316'
PREREG_SHA = '2e88373643baa035e4491119ade134551de3e22bac75c3e9184bfa9d7f8585bf'
RECIPE = ROOT / 'docs/fps_comparisons/4070s_400fps_20260928/int8_study/recipe_gmac_0.50.json'
RECIPE_SHA = 'f6f90777264b53027af586a3991598c9f1c89bd7761eaecbe2084c7d119dad12'
PINS = {
    'scripts/build_unet_stagewise.py': '1f0c0c54f2d94dc7358a7b79f4e0176dfdb747b78140552fad35c34d32917d49',
    'scripts/unet_stagewise_trt.py': '4ab523ab9337f1ce903a52acf7a3f06e7423669cbf2df2b3846b8ffc374b6e9c',
    'scripts/repro_3090/watch_owned_single_leaf_target.py': 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9',
    'scripts/repro_3090/preregister_metadata_quality_lineage.py': 'd5036b697c06efedcf95c9e81b99efd61129f0f07be50572f03296fabfc3e1ae',
}
STAGES = (('down0rest', 'up3', 'tail'),
          ('down1', 'down2', 'down3', 'mid', 'up0', 'up1', 'up2'), ('prefix',))
BLOCKS = set().union(*map(set, STAGES))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(path, expected):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe pinned path')
    require(sha(path) == expected, 'pinned source or input changed')
    return path.read_bytes()


def imported(relative):
    path = ROOT / relative
    raw = checked(path, PINS[relative])
    module = types.ModuleType('_checked_' + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def options(stage):
    require(stage in STAGES, 'unregistered stage')
    return types.SimpleNamespace(root=str(ENGINE), opt_level=3, workspace_gb=2.0,
        no_timing_cache=False, hardware_compat='none', strict_timing_cache=True,
        variant='srccache', timing_cache='', blocks=','.join(stage), int8_blocks='',
        int8_recipe=str(RECIPE) if stage == STAGES[1] else '', max_minutes=20,
        force=False, second_build=False, calib_dir=str(ROOT / 'calibration/unet_multi_avatar_20260928'),
        calib_batches=8)


def assert_stage(manifest, baseline, wanted, final=False):
    require(set(manifest['blocks']) == set(wanted), 'incomplete or unregistered blocks')
    for key in ('batch', 'gpu', 'compute_capability', 'tensorrt_version', 'torch_version',
                'cuda_version', 'spec', 'variant', 'unet_weights', 'unet_config', 'timestep'):
        require(manifest[key] == baseline[key], 'baseline runtime/graph field differs:' + key)
    require(manifest.get('hardware_compatibility_level', 'none') == 'none', 'not a native build')
    expected_flags = {**baseline['build_flags'], 'builder_optimization_level': 3}
    require(manifest['build_flags'] == expected_flags, 'global build policy changed')
    for name in wanted:
        got, old = manifest['blocks'][name], baseline['blocks'][name]
        require(got['onnx_sha256'] == old['onnx_sha256'], 'same-graph experiment invalid:' + name)
        require(got['build_flags'] == {**old['build_flags'], 'builder_optimization_level': 3},
                'precision or other block policy changed:' + name)
        for key in ('inputs', 'outputs', 'engine_file'):
            require(got[key] == old[key], 'block interface changed:' + name)
    if set(STAGES[1]).issubset(wanted):
        require(manifest['int8_calibration']['files'] == baseline['int8_calibration']['files']
                and manifest['int8_calibration']['batches'] == 8, 'main calibration selection changed')
    if final:
        require(manifest.get('complete') is True and set(wanted) == BLOCKS, 'chain not finalized')
        require(manifest['probe']['deterministic_run_to_run'] is True
                and manifest['probe']['graph_equals_direct_enqueue'] is True, 'probe hard invariant failed')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--owned-target-json', type=Path, required=True)
    p.add_argument('--owned-target-sha256', required=True)
    p.add_argument('--successor-inputs', type=Path, required=True)
    p.add_argument('--successor-envelope', type=Path, required=True)
    p.add_argument('--lineage-receipt', type=Path, required=True)
    p.add_argument('--lineage-receipt-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    require(a.execute, 'explicit execution required')
    require(ROOT == Path('/workspace/MuseTalk'), 'owned worker path required')
    owned, deadline = imported('scripts/repro_3090/watch_owned_single_leaf_target.py').binding(
        a.owned_target_json, a.owned_target_sha256)
    require(socket.gethostname() == owned['worker_hostname'], 'wrong owned host')
    require((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() > 1800, 'build and persistence margin required')
    lease = os.environ.get('BOX_GUARD_LEASE_FILE')
    require(lease and Path(lease + '.holder').is_file(), 'canonical guard required')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=10).strip() == owned['gpu_uuid'], 'wrong GPU')
    lineage = imported('scripts/repro_3090/preregister_metadata_quality_lineage.py').verify_preregistration(
        parent_inputs=BASE / 'harnesses/quality-inputs-v1.json',
        parent_envelope=BASE / 'quality/reference-envelope-v2.json', successor_inputs=a.successor_inputs,
        successor_envelope=a.successor_envelope, receipt_path=a.lineage_receipt,
        expected_receipt_sha256=a.lineage_receipt_sha256)
    checked(BASE / 'quality/next_same_precision_opt3_preregistered_0610.json', PREREG_SHA)
    checked(RECIPE, RECIPE_SHA)
    baseline = json.loads(checked(BASELINE, BASELINE_SHA))
    require(baseline['complete'] is True and set(baseline['blocks']) == BLOCKS, 'baseline incomplete')
    for relative, digest in PINS.items():
        checked(ROOT / relative, digest)
    require(not ENGINE.exists() and not ENGINE.is_symlink(), 'fresh candidate required; no automatic resume')
    require(a.out.is_absolute() and a.out.parent.resolve() == BASE / 'native'
            and not any(p.is_symlink() for p in (a.out, *a.out.parents)) and not a.out.exists(), 'fresh scoped report required')
    a.out.mkdir(mode=0o700)
    report = dict(schema='owned_same_precision_opt3_build_v1', status='BUILDING', owned_target=owned,
        input_lineage=lineage, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        preregistration_sha256=PREREG_SHA, baseline_manifest_sha256=BASELINE_SHA, stages=[],
        quality_accepted=False, performance_accepted=False, release_ready=False)
    try:
        os.chdir(ROOT); sys.path.insert(0, str(ROOT))
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUSETALK_UNET_STAGEWISE_VERIFY_SHA='1')
        from scripts import build_unet_stagewise as builder
        torch = builder.torch
        require(torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1', 'pinned runtime required')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:0')
        model = builder.load_eager_unet(device)
        wanted = set()
        for stage in STAGES:
            require((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() > 600, 'cleanup margin reached')
            manifest = builder.build_batch(options(stage), 16, model, device)
            wanted.update(stage)
            assert_stage(manifest, baseline, wanted, final=stage == STAGES[-1])
            for entry in manifest['blocks'].values():
                require(sha(ENGINE / 'bs16' / entry['engine_file']) == entry['engine_sha256'], 'plan changed')
            report['stages'].append(dict(blocks=list(stage), onnx_hashes_match=True,
                completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        report.update(status='BUILT_PROBED_ALL11_SAME_GRAPH_QUALITY_PERFORMANCE_UNTESTED',
                      manifest_sha256=sha(ENGINE / 'bs16/manifest.json'), probe=manifest['probe'])
    except BaseException as exc:
        report.update(status='INVALID_BUILD', error_type=type(exc).__name__,
                      reason=str(exc) if isinstance(exc, ValueError) else 'build_or_probe_failure')
        raise
    finally:
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (a.out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
