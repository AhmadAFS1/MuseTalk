"""CPU-only assembly of a fixed, preregistered native/portable UNet diagnosis.

No export, engine build, GPU import, finalization, runtime/default change or
quality approval. The copied manifest is deliberately incomplete: a fresh
on-device graph/direct/determinism probe is required before it can be loaded.
Different source graph hashes are preserved, never claimed semantically equal.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import types

HELPER_SHA = 'd5ed578efe8f9c7df364dc3726cb133ec6b380d6c56a990059f9025016f0f620'
BLOCKS = frozenset(('prefix', 'down0rest', 'down1', 'down2', 'down3', 'mid', 'up0', 'up1', 'up2', 'up3', 'tail'))
VARIANTS = {
    'portable_prefix': frozenset(('prefix',)),
    'portable_prefix_down0rest': frozenset(('prefix', 'down0rest')),
    'portable_core_native_up3': BLOCKS - {'up3'},
}


def checked_helpers():
    path = Path(__file__).resolve().with_name('assemble_owned_unet_tail_a3.py')
    if path.is_symlink():
        raise ValueError('helper symlink forbidden')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('reviewed assembly helper changed')
    module = types.ModuleType('_checked_unet_overlay_helpers')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def prepare(native_dir, portable_dir, target_root, *, variant, native_sha=None, portable_sha=None):
    if variant not in VARIANTS:
        raise ValueError('only preregistered diagnostic variants permitted')
    h = checked_helpers()
    native_dir, portable_dir, target_root = map(h.safe, (native_dir, portable_dir, target_root))
    native_sha = native_sha or h.NATIVE_SHA
    portable_sha = portable_sha or h.PORTABLE_SHA
    import json
    native = json.loads(h.checked_bytes(native_dir / 'manifest.json', native_sha))
    portable = json.loads(h.checked_bytes(portable_dir / 'manifest.json', portable_sha))
    h.require(not target_root.exists(), 'fresh_output_required')
    for key, value in {'schema': 'musetalk_unet_stagewise_trt_v1', 'batch': 16,
                       'variant': 'srccache', 'timestep': 0, 'complete': True,
                       'tensorrt_version': '10.3.0', 'torch_version': '2.5.1+cu121'}.items():
        h.require(native.get(key) == portable.get(key) == value, 'source_manifest_contract')
    h.require(native['spec'] == portable['spec'] and
              set(native['blocks']) == set(portable['blocks']) == BLOCKS, 'source_block_spec')
    h.require(native['gpu'] == 'NVIDIA GeForce RTX 3090' and native['compute_capability'] == [8, 6]
              and native.get('hardware_compatibility_level', 'none') == 'none', 'native_identity')
    h.require(portable['gpu'] == 'NVIDIA GeForce RTX 4070 SUPER' and portable['compute_capability'] == [8, 9]
              and portable.get('hardware_compatibility_level') == 'ampere_plus', 'portable_identity')
    h.require(native['int8_calibration']['recipe_sha256_16'] == portable['int8_calibration']['recipe_sha256_16']
              == 'f6f90777264b5302' and native['int8_calibration']['files'] == portable['int8_calibration']['files'],
              'source_calibration_contract')
    selected, filenames = {}, set()
    for block in sorted(BLOCKS):
        ne, pe = native['blocks'][block], portable['blocks'][block]
        h.require(ne['inputs'] == pe['inputs'] and ne['outputs'] == pe['outputs']
                  and ne['build_flags']['precision'] == pe['build_flags']['precision'], 'block_interface_or_precision')
        origin, directory, digest = ((portable, portable_dir, portable_sha) if block in VARIANTS[variant]
                                     else (native, native_dir, native_sha))
        entry = copy.deepcopy(origin['blocks'][block])
        filename = entry['engine_file']
        h.require(isinstance(filename, str) and Path(filename).name == filename and filename.endswith('.plan')
                  and filename not in filenames, 'unsafe_or_duplicate_plan_filename')
        filenames.add(filename)
        h.require(entry['build_flags'].get('hardware_compatibility_level', 'none')
                  == origin.get('hardware_compatibility_level', 'none'), 'block_compatibility')
        h.require(h.file_sha(directory / filename) == entry['engine_sha256'], 'source_plan_sha256_mismatch')
        selected[block] = origin, directory, digest, entry
    manifest = copy.deepcopy(native)
    for key in ('probe', 'runtime', 'missing_blocks', 'finalized_utc', 'total_engine_mib', 'build_log', 'timing_cache_input'):
        manifest.pop(key, None)
    manifest.update(complete=False, hardware_compatibility_level='none', build_log=[],
        engine_origin='literal_preregistered_plan_overlay_no_build_no_export', overlay_variant=variant,
        supported_compute_capability=[8, 6],
        build_flags_scope='Per-block original provenance; finalization has not run.',
        quality_accepted=False, performance_measured=False, release_ready=False, block_provenance={})
    target_root.mkdir(mode=0o700)
    target = target_root / 'bs16'; target.mkdir(mode=0o700)
    for block, (origin, directory, digest, entry) in selected.items():
        filename = entry['engine_file']; actual = hashlib.sha256()
        with (directory / filename).open('rb') as source, (target / filename).open('xb') as destination:
            for chunk in iter(lambda: source.read(8 << 20), b''):
                actual.update(chunk); destination.write(chunk)
        h.require(actual.hexdigest() == entry['engine_sha256'], 'copied_plan_sha256_mismatch')
        manifest['blocks'][block] = entry
        manifest['block_provenance'][block] = {
            'source_directory': str(directory), 'source_manifest_sha256': digest,
            'source_gpu': origin['gpu'], 'source_compute_capability': origin['compute_capability'],
            'hardware_compatibility_level': origin.get('hardware_compatibility_level', 'none'),
            'engine_file': filename, 'engine_sha256': entry['engine_sha256'], 'onnx_sha256': entry['onnx_sha256']}
    h.write_json(target / 'manifest.json', manifest)
    return manifest
