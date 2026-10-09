"""Explicit native-UNet/portable-decoder comparison, not an all-native release.

The original quality helper remains on disk byte-identical. Authenticate it,
adapt exactly one decoder-role check in memory, and disclose this adapter.
All 698 metric computations, directions, bounds and hard invariants remain
the original code. The independently selected decoder is never relabelled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE_SHA = 'a5a98185dad4db0d1b89c2d3e04b8832721d99461cca7381605b62b035697e55'
ROLE_CHECK = '    checks.require(env["taesd"]["fingerprint"].get("hardware_compatibility_level", "none") == compatibility, "TAESD role mismatch")'
ROLE_REPLACEMENT = '    decoder_role(env, role, compatibility)'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def checked(path, expected):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink artifact required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'explicit SHA256 required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'artifact changed')
    return raw


def adapted(raw):
    require(hashlib.sha256(raw).hexdigest() == CORE_SHA, 'original quality helper changed')
    text = raw.decode()
    require(text.count(ROLE_CHECK) == 1, 'unique decoder-role check required')
    return text.replace(ROLE_CHECK, ROLE_REPLACEMENT).encode()


def decoder_role(env, role, compatibility, selection):
    actual = env['taesd']['fingerprint'].get('hardware_compatibility_level', 'none')
    if role == 'reference':
        require(actual == compatibility, 'reference decoder role changed')
        return
    require(role == 'candidate' and compatibility == 'none', 'native candidate UNet required')
    require(actual == 'ampere_plus', 'explicit portable decoder required; no hardware relabelling')
    require(env['engines'][0]['manifest_sha256'] == selection['engine_manifest_sha256'], 'different candidate UNet')
    require(env['taesd'] == selection['taesd'], 'different selected decoder identity')


def core_for(selection):
    path = HERE / 'quality_envelope.py'
    raw = checked(path, CORE_SHA)
    module = types.ModuleType('_reviewed_mixed_quality')
    module.__file__ = str(path)
    sys.path.insert(0, str(HERE))
    exec(compile(adapted(raw), str(path), 'exec'), module.__dict__)
    module.decoder_role = lambda env, role, compatibility: decoder_role(env, role, compatibility, selection)
    return module


def verify_native_manifest(manifest):
    require(manifest['complete'] is True and manifest.get('hardware_compatibility_level', 'none') == 'none', 'complete native UNet required')
    require(set(manifest['blocks']) == {'prefix','down0rest','down1','down2','down3','mid','up0','up1','up2','up3','tail'}
            and manifest['batch'] == 16 and manifest['variant'] == 'srccache'
            and manifest['gpu'] == 'NVIDIA GeForce RTX 3090' and manifest['compute_capability'] == [8, 6]
            and manifest['probe']['graph_equals_direct_enqueue'] is True
            and manifest['probe']['deterministic_run_to_run'] is True, 'native chain or hard invariant changed')


def verify_selection(selection):
    require(selection['schema'] == 'native_unet_portable_decoder_selection_v1', 'unknown selection schema')
    engine_root = Path(selection['engine_root'])
    require(engine_root.parent == ROOT / 'models', 'scoped candidate engine root required')
    manifest = json.loads(checked(engine_root / 'bs16/manifest.json', selection['engine_manifest_sha256']))
    verify_native_manifest(manifest)
    for block in manifest['blocks'].values():
        require(Path(block['engine_file']).name == block['engine_file'], 'unsafe plan name')
        checked(engine_root / 'bs16' / block['engine_file'], block['engine_sha256'])
    directory = Path(selection['taesd_dir'])
    require(directory == ROOT / 'models/taesd/trt', 'original portable decoder directory required')
    identity = selection['taesd']
    require(identity['key'] == '512bfd629a5e1f4f2e40', 'original portable decoder key required')
    meta = json.loads(checked(directory / ('taesd_trt_' + identity['key'] + '.json'), identity['meta_sha256']))
    require(identity == {'key': meta['key'], 'meta_sha256': identity['meta_sha256'],
                        'decoder_plan_sha256': meta['decoder_plan_sha256'], 'fingerprint': meta['fingerprint']}, 'decoder metadata mismatch')
    require(meta['fingerprint'].get('hardware_compatibility_level') == 'ampere_plus'
            and meta['fingerprint']['batch'] == 8
            and 'gpu' not in meta['fingerprint'] and 'compute_capability' not in meta['fingerprint'], 'portable provenance must be truthful')
    for kind in ('decoder', 'post'):
        require(Path(meta[kind + '_plan']).name == meta[kind + '_plan'], 'unsafe decoder plan name')
        checked(directory / meta[kind + '_plan'], meta[kind + '_plan_sha256'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    for name in ('candidate', 'inputs', 'envelope', 'selection', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--envelope-sha256', required=True)
    p.add_argument('--selection-sha256', required=True)
    a = p.parse_args(argv)
    require(a.execute, 'explicit mixed-candidate execution required')
    require(not a.out.exists() and a.out.is_absolute()
            and not any(x.is_symlink() for x in (a.out, *a.out.parents)), 'fresh nonsymlink output required')
    result = {'status': 'INVALID', 'quality_accepted': False, 'release_ready': False}
    try:
        selection = json.loads(checked(a.selection, a.selection_sha256))
        verify_selection(selection)
        core = core_for(selection)
        core.checks.verify_files(core.checks.read(a.inputs), a.inputs.resolve().parent)
        envelope = json.loads(checked(a.envelope, a.envelope_sha256))
        candidate = core.load_run(a.candidate, a.inputs, 'candidate')
        result = core.compare(envelope, candidate, core.checks.read(core.POLICY))
        result.update(frozen_envelope=core.evidence(a.envelope), selection=core.evidence(a.selection),
            artifact_composition='NATIVE_UNET_WITH_ORIGINAL_PORTABLE_TAESD_NOT_ALL_NATIVE',
            comparison_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_metric_helper_sha256=CORE_SHA, source_adaptation='one decoder-role check only; all metric/bound/invariant code unchanged',
            quality_accepted=False, release_ready=False)
    except Exception as exc:
        result = dict(status='INVALID', quality_accepted=False, release_ready=False,
            error_type=type(exc).__name__, reason=str(exc) if isinstance(exc, ValueError) else 'invalid_quality_evidence')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps({'out': str(a.out), 'status': result.get('quality_parity_with_reference', result.get('status'))}))
    return 2 if result.get('status') == 'INVALID' else 1 if result.get('quality_parity_with_reference') == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
