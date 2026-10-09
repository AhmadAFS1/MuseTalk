"""Default-off, isolated final-Conv FP32 candidate API; import-time stdlib only.

No CLI execution, canonical monkeypatch, loader registration, fallback, or
autobuild exists. The future caller must explicitly authorize each GPU build or
load and run it under the canonical owned-worker box_guard/watch protocol.
This module does not route runner/gate/capture children or assert quality/FPS.
Build output is a new private directory; incomplete outputs are never reused.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import types

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = '466225e995f0e70a194eccd8136f3c08fe2b5ca4c93f888c522ea2070a2044bb'
TRANSFORMER_SHA256 = '4558bd6d71e45bd5a9bc9930b40fd7aaedf6fbb32f3786ef096550e01075e7c3'
CANONICAL_VFD_SHA256 = 'aea1eb059085c1799c6162d516d028d385be107c36da8a105fa1f838eb5e9882'
RECIPE = 'taesd_final_conv_fp32_island_v1'
POST_RECIPE = 'bgr_u8_nhwc_post_fp32_rne_v1'
SCHEMA = 'taesd_fp32_candidate_engine_v1'
RUNTIME = {'torch': '2.5.1+cu121', 'cuda': '12.1', 'tensorrt': '10.3.0',
           'gpu': 'NVIDIA GeForce RTX 3090', 'compute_capability': '8.6'}
INPUT_SHAPE = [8, 4, 32, 32]
OUTPUT_SHAPE = [8, 3, 256, 256]
PREFIX = '__musetalk_taesd_final_conv_fp32_v1_'
BUILD_SETTINGS = {'network': 'STRONGLY_TYPED', 'tf32_requested': False, 'tf32_readback': False,
                  'fp16_builder_flag_readback': False, 'optimization_level': 3,
                  'workspace_bytes': 1 << 30, 'hardware_compatibility_level': 'none',
                  'timing_cache_initial_bytes': 0}


class CandidateRejected(ValueError):
    """Fixed diagnostic vocabulary only; never parser/model contents."""


def require(value, reason):
    if not value:
        raise CandidateRejected(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def invalid(_value):
        raise CandidateRejected('nonfinite_json_value')
    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    except CandidateRejected:
        raise
    except Exception:
        raise CandidateRejected('invalid_json') from None


def regular_bytes(path, limit):
    """Explicit regular file, bounded size, no symlink in its supplied path."""
    path = Path(path)
    require(path.is_absolute(), 'absolute_path_required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink_path_forbidden')
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, 'invalid_file_type_or_size')
            data = handle.read(limit + 1)
            require(len(data) == info.st_size and len(data) <= limit, 'file_changed_or_oversized')
            return data
    except CandidateRejected:
        raise
    except OSError:
        raise CandidateRejected('required_file_unavailable') from None


def checked_bytes(path, digest, limit):
    require(valid_digest(digest), 'explicit_sha256_required')
    data = regular_bytes(path, limit)
    require(sha(data) == digest, 'artifact_sha256_mismatch')
    return data


def code_identities():
    transformer = ROOT / 'scripts/repro_3090/taesd_final_conv_fp32.py'
    canonical = ROOT / 'scripts/vae_fast_decoder.py'
    checked_bytes(transformer, TRANSFORMER_SHA256, 1 << 20)
    checked_bytes(canonical, CANONICAL_VFD_SHA256, 1 << 20)
    return {'transformer_sha256': TRANSFORMER_SHA256, 'canonical_vfd_sha256': CANONICAL_VFD_SHA256,
            'runtime_helper_sha256': sha(regular_bytes(Path(__file__).absolute(), 1 << 20))}


def _verified_module(path, digest, name):
    source = checked_bytes(path, digest, 1 << 20)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    # Execute the exact checked bytes, not a second filesystem read or .pyc.
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def verify_lineage(*, source_path, candidate_path, proof_path,
                   expected_candidate_sha256, expected_proof_sha256):
    """CPU-only actual proof regeneration; no trust in claimed receipt flags."""
    code = code_identities()
    source = checked_bytes(source_path, SOURCE_SHA256, 64 << 20)
    candidate = checked_bytes(candidate_path, expected_candidate_sha256, 64 << 20)
    proof_bytes = checked_bytes(proof_path, expected_proof_sha256, 2 << 20)
    proof = parse_json(proof_bytes)
    helper = _verified_module(ROOT / 'scripts/repro_3090/taesd_final_conv_fp32.py',
                              TRANSFORMER_SHA256, '_verified_taesd_fp32_transform')
    try:
        rebuilt, rebuilt_proof = helper.transform(source, SOURCE_SHA256)
    except Exception:
        raise CandidateRejected('transformation_regeneration_failed') from None
    require(candidate == rebuilt, 'candidate_not_exact_regenerated_graph')
    require(json_bytes(proof) == json_bytes(rebuilt_proof), 'proof_not_exact_regenerated_receipt')
    require(proof.get('recipe') == RECIPE and proof.get('source_sha256') == SOURCE_SHA256
            and proof.get('transformed_sha256') == expected_candidate_sha256, 'proof_identity_mismatch')
    return {'source': source, 'candidate': candidate, 'proof_bytes': proof_bytes, 'proof': proof,
            'candidate_sha256': expected_candidate_sha256, 'proof_sha256': expected_proof_sha256, 'code': code}


def fingerprint(lineage, runtime):
    require(runtime == RUNTIME, 'incompatible_runtime_or_device')
    require(valid_digest(lineage['candidate_sha256']) and valid_digest(lineage['proof_sha256']), 'invalid_lineage_digests')
    require(set(lineage['code']) == {'transformer_sha256', 'canonical_vfd_sha256', 'runtime_helper_sha256'}
            and all(valid_digest(v) for v in lineage['code'].values()), 'invalid_code_identity')
    require(lineage['code']['transformer_sha256'] == TRANSFORMER_SHA256
            and lineage['code']['canonical_vfd_sha256'] == CANONICAL_VFD_SHA256, 'unpinned_source_helper')
    return {'recipe': RECIPE, 'post_recipe': POST_RECIPE, **runtime, **lineage['code'],
            'source_onnx_sha256': SOURCE_SHA256, 'onnx_sha256': lineage['candidate_sha256'],
            'transformation_receipt_sha256': lineage['proof_sha256'], 'batch': 8,
            'latent_shape': INPUT_SHAPE, 'image_shape': OUTPUT_SHAPE,
            'precision': 'fp16_with_final_conv_fp32', 'input_dtype': 'fp16', 'output_dtype': 'fp16',
            'strongly_typed': True, 'tf32': False, 'opt_level': 3,
            'hardware_compatibility_level': 'none', 'workspace_bytes': 1 << 30}


def key_for(fp):
    # Match canonical key encoding, with a DIFFERENT, truthful candidate fingerprint.
    return sha(json.dumps(fp, sort_keys=True, allow_nan=False).encode())[:20]


def _gpu_dependencies(device):
    """Heavy imports/driver interaction occur ONLY after explicit API opt-in."""
    require(str(device) == 'cuda:0', 'explicit_cuda0_required')
    import torch
    import tensorrt as trt
    require(torch.__version__ == RUNTIME['torch'] and torch.version.cuda == RUNTIME['cuda']
            and trt.__version__ == RUNTIME['tensorrt'], 'incompatible_runtime_version')
    require(torch.cuda.device_count() == 1, 'exactly_one_visible_gpu_required')
    runtime = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'tensorrt': trt.__version__,
               'gpu': torch.cuda.get_device_name(device),
               'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(device)))}
    require(runtime == RUNTIME, 'incompatible_runtime_or_device')
    vfd = _verified_module(ROOT / 'scripts/vae_fast_decoder.py', CANONICAL_VFD_SHA256,
                           '_verified_canonical_taesd_for_fp32_candidate')
    require(vfd.TAESD_TRT_POST_RECIPE == POST_RECIPE, 'canonical_post_recipe_changed')
    return torch, trt, vfd, runtime


def configure_builder(trt, config):
    """Exact TensorRT 10.3 API; types come from the strongly typed ONNX graph."""
    config.clear_flag(trt.BuilderFlag.TF32)
    require(config.get_flag(trt.BuilderFlag.TF32) is False, 'tf32_disable_readback_failed')
    require(config.get_flag(trt.BuilderFlag.FP16) is False, 'fp16_builder_flag_forbidden')
    config.builder_optimization_level = 3
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
    require(config.builder_optimization_level == 3, 'optimization_level_readback_failed')
    require(config.hardware_compatibility_level == trt.HardwareCompatibilityLevel.NONE, 'native_compatibility_readback_failed')
    cache = config.create_timing_cache(b'')
    require(config.set_timing_cache(cache, ignore_mismatch=False) is True, 'fresh_timing_cache_rejected')
    return dict(BUILD_SETTINGS)


def build_decoder_plan(trt, vfd, candidate_bytes, proof):
    builder = trt.Builder(vfd._trt_logger())
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, vfd._trt_logger())
    require(parser.parse(candidate_bytes) is True, 'candidate_onnx_parse_failed')
    require(network.num_inputs == network.num_outputs == 1, 'wrong_decoder_io_count')
    for tensor, shape in ((network.get_input(0), INPUT_SHAPE), (network.get_output(0), OUTPUT_SHAPE)):
        require(tensor.dtype == trt.DataType.HALF and list(tensor.shape) == shape, 'wrong_decoder_fp16_boundary')
    # Prove parser-retained island types before optimization/build.
    tensors = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor is not None:
                tensors[tensor.name] = tensor
    original_output = proof['target']['original_output_names'][0]
    for name, dtype in ((PREFIX + 'activation_fp32', trt.DataType.FLOAT),
                        (PREFIX + 'conv_fp32', trt.DataType.FLOAT), (original_output, trt.DataType.HALF)):
        require(name in tensors and tensors[name].dtype == dtype, 'parsed_island_dtype_not_proven')
    config = builder.create_builder_config()
    settings = configure_builder(trt, config)
    serialized = builder.build_serialized_network(network, config)
    require(serialized is not None, 'decoder_build_failed')
    plan = bytes(serialized)
    require(bool(plan), 'empty_decoder_plan')
    cache = bytes(config.get_timing_cache().serialize())
    require(bool(cache), 'empty_timing_cache')
    return plan, cache, settings


def validate_probe(probe):
    require(isinstance(probe, dict) and set(probe) == {'probe_seed', 'probe_shape', 'fp16_sha256',
            'u8_repo_post_sha256', 'u8_fused_sha256', 'fused_vs_repo_post_mismatched_bytes'}, 'probe_schema_mismatch')
    require(type(probe['probe_seed']) is int and probe['probe_seed'] == 20260928
            and isinstance(probe['probe_shape'], list) and all(type(x) is int for x in probe['probe_shape'])
            and probe['probe_shape'] == INPUT_SHAPE, 'probe_input_mismatch')
    require(type(probe['fused_vs_repo_post_mismatched_bytes']) is int
            and probe['fused_vs_repo_post_mismatched_bytes'] == 0, 'fused_post_probe_mismatch')
    require(all(valid_digest(probe[k]) for k in ('fp16_sha256', 'u8_repo_post_sha256', 'u8_fused_sha256')),
            'invalid_probe_digest')
    require(probe['u8_repo_post_sha256'] == probe['u8_fused_sha256'], 'fused_post_probe_hash_mismatch')


def _backend(torch, vfd, device, decoder, post, meta, directory, model=None):
    backend = vfd.TaesdTrtBackend(vfd._TrtEngine(decoder, 'candidate final-conv fp32 decoder'),
        vfd._TrtEngine(post, 'canonical taesd bgr_u8 post'), device, torch.float16, 8,
        meta=meta, paths={'decoder': directory / meta['decoder_plan'], 'post': directory / meta['post_plan'],
                         'meta': directory / f"taesd_trt_{meta['key']}.json"}, model=model)
    require(backend.fused_post_enabled is True, 'canonical_fused_post_must_be_enabled')
    return backend


def _write_new(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def build_candidate(*, source_path, candidate_path, proof_path, expected_candidate_sha256,
                    expected_proof_sha256, output_dir, device, enable_build=False):
    """Explicit GPU action when authorized; never called by import/CLI/loader."""
    require(enable_build is True, 'candidate_build_not_explicitly_enabled')
    directory = Path(output_dir)
    require(directory.is_absolute() and not directory.exists() and not directory.is_symlink(), 'fresh_absolute_output_required')
    require(directory.parent.is_dir() and not any(p.is_symlink() for p in directory.parents), 'safe_existing_output_parent_required')
    lineage = verify_lineage(source_path=source_path, candidate_path=candidate_path, proof_path=proof_path,
        expected_candidate_sha256=expected_candidate_sha256, expected_proof_sha256=expected_proof_sha256)
    torch, trt, vfd, runtime = _gpu_dependencies(device)
    fp = fingerprint(lineage, runtime)
    key = key_for(fp)
    directory.mkdir(mode=0o700, exist_ok=False)
    source_files = {'source.onnx': lineage['source'], 'candidate.onnx': lineage['candidate'],
                    'transform-proof.json': lineage['proof_bytes']}
    for name, data in source_files.items():
        _write_new(directory / name, data)
    # Failures leave a partial fresh directory WITHOUT a loadable manifest.
    decoder, cache, settings = build_decoder_plan(trt, vfd, lineage['candidate'], lineage['proof'])
    require(settings == BUILD_SETTINGS, 'builder_settings_mismatch')
    post = bytes(vfd.build_bgr_u8_post_plan(8, hw_compat='none'))
    require(bool(post), 'empty_post_plan')
    meta = {'schema': SCHEMA, 'key': key, 'fingerprint': fp,
        'decoder_plan': f'taesd_trt_{key}.decoder.plan', 'decoder_plan_sha256': sha(decoder), 'decoder_plan_bytes': len(decoder),
        'post_plan': f'taesd_trt_{key}.post_bgr_u8.plan', 'post_plan_sha256': sha(post), 'post_plan_bytes': len(post),
        'source_artifacts': {name: {'sha256': sha(data), 'bytes': len(data)} for name, data in source_files.items()},
        'build': {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'runtime': runtime,
                  'settings': settings, 'timing_cache': 'candidate_timing.cache',
                  'timing_cache_sha256': sha(cache), 'timing_cache_bytes': len(cache)},
        'status': 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED', 'quality_accepted': False,
        'full_gate_captured': False, 'performance_measured': False, 'default_selection_changed': False, 'release_ready': False}
    backend = _backend(torch, vfd, device, decoder, post, meta, directory)
    first = backend.probe_hashes()
    validate_probe(first)
    second = backend.probe_hashes()
    validate_probe(second)
    require(first == second, 'build_probe_not_deterministic')
    meta['probe'] = first
    for name, data in ((meta['decoder_plan'], decoder), (meta['post_plan'], post), ('candidate_timing.cache', cache)):
        _write_new(directory / name, data)
    encoded = json_bytes(meta)
    manifest = directory / f'taesd_trt_{key}.json'
    _write_new(manifest, encoded)  # final commit marker; partial builds cannot load
    return {'manifest_path': str(manifest), 'manifest_sha256': sha(encoded), 'key': key,
            'quality_accepted': False, 'full_gate_captured': False, 'release_ready': False}


def load_candidate(*, manifest_path, expected_manifest_sha256, device, model=None, enable=False):
    """Explicit hash-bound GPU loader; NO build, lookup, fallback, or registration."""
    require(enable is True, 'candidate_loader_not_explicitly_enabled')
    manifest = Path(manifest_path)
    raw = checked_bytes(manifest, expected_manifest_sha256, 2 << 20)
    meta = parse_json(raw)
    required = {'schema', 'key', 'fingerprint', 'decoder_plan', 'decoder_plan_sha256', 'decoder_plan_bytes',
        'post_plan', 'post_plan_sha256', 'post_plan_bytes', 'source_artifacts', 'build', 'probe', 'status',
        'quality_accepted', 'full_gate_captured', 'performance_measured', 'default_selection_changed', 'release_ready'}
    require(isinstance(meta, dict) and set(meta) == required and meta['schema'] == SCHEMA, 'candidate_manifest_schema_mismatch')
    require(meta['status'] == 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED'
            and all(meta[k] is False for k in ('quality_accepted', 'full_gate_captured', 'performance_measured',
                                             'default_selection_changed', 'release_ready')), 'candidate_manifest_claims_invalid')
    directory = manifest.parent
    source_artifacts = meta['source_artifacts']
    require(set(source_artifacts) == {'source.onnx', 'candidate.onnx', 'transform-proof.json'}, 'source_artifact_schema_mismatch')
    for name, row in source_artifacts.items():
        require(set(row) == {'sha256', 'bytes'} and type(row['bytes']) is int and row['bytes'] > 0, 'source_artifact_record_invalid')
        data = checked_bytes(directory / name, row['sha256'], 64 << 20)
        require(len(data) == row['bytes'], 'source_artifact_size_mismatch')
    require(source_artifacts['source.onnx']['sha256'] == SOURCE_SHA256, 'unpinned_original_graph')
    lineage = verify_lineage(source_path=directory / 'source.onnx', candidate_path=directory / 'candidate.onnx',
        proof_path=directory / 'transform-proof.json', expected_candidate_sha256=source_artifacts['candidate.onnx']['sha256'],
        expected_proof_sha256=source_artifacts['transform-proof.json']['sha256'])
    fp = fingerprint(lineage, RUNTIME)
    require(json_bytes(meta['fingerprint']) == json_bytes(fp) and meta['key'] == key_for(fp), 'candidate_fingerprint_or_key_mismatch')
    require(manifest.name == f"taesd_trt_{meta['key']}.json", 'candidate_manifest_filename_mismatch')
    build = meta['build']
    require(set(build) == {'created_utc', 'runtime', 'settings', 'timing_cache', 'timing_cache_sha256', 'timing_cache_bytes'}
            and build['runtime'] == RUNTIME and json_bytes(build['settings']) == json_bytes(BUILD_SETTINGS)
            and build['timing_cache'] == 'candidate_timing.cache', 'recorded_builder_contract_mismatch')
    try:
        created = dt.datetime.fromisoformat(build['created_utc'])
        require(created.tzinfo is not None and created.utcoffset() == dt.timedelta(0), 'build_timestamp_not_utc')
    except (ValueError, TypeError):
        raise CandidateRejected('invalid_build_timestamp') from None
    cache = checked_bytes(directory / 'candidate_timing.cache', build['timing_cache_sha256'], 128 << 20)
    require(type(build['timing_cache_bytes']) is int and len(cache) == build['timing_cache_bytes'], 'timing_cache_size_mismatch')
    plans = {}
    for kind, suffix in (('decoder', 'decoder.plan'), ('post', 'post_bgr_u8.plan')):
        name = f"taesd_trt_{meta['key']}.{suffix}"
        require(meta[kind + '_plan'] == name, 'plan_path_not_exact_candidate_basename')
        plans[kind] = checked_bytes(directory / name, meta[kind + '_plan_sha256'], 128 << 20)
        require(type(meta[kind + '_plan_bytes']) is int and len(plans[kind]) == meta[kind + '_plan_bytes'], 'plan_size_mismatch')
    validate_probe(meta['probe'])
    # No heavy imports/GPU action until all static artifacts have passed.
    torch, _trt, vfd, runtime = _gpu_dependencies(device)
    require(runtime == meta['build']['runtime'], 'loaded_runtime_changed')
    if model is not None:
        original = vfd.export_taesd_decoder_onnx(model, 8, device)
        require(sha(original) == SOURCE_SHA256, 'reference_model_export_mismatch')
    backend = _backend(torch, vfd, device, plans['decoder'], plans['post'], meta, directory, model)
    actual = backend.probe_hashes()
    validate_probe(actual)
    require(actual == meta['probe'], 'loaded_probe_not_exact')
    again = backend.probe_hashes()
    validate_probe(again)
    require(again == actual, 'loaded_probe_not_deterministic')
    return backend
