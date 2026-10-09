"""CPU contracts only: synthetic bytes and mocked CUDA/TRT; never build an engine."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taesd_fp32_candidate_runtime as runtime

PROBE = {'probe_seed': 20260928, 'probe_shape': [8, 4, 32, 32], 'fp16_sha256': 'a' * 64,
         'u8_repo_post_sha256': 'b' * 64, 'u8_fused_sha256': 'b' * 64, 'fused_vs_repo_post_mismatched_bytes': 0}


class Config:
    def __init__(self):
        self.flags = {'TF32': True, 'FP16': False}
        self.calls = []
    def clear_flag(self, flag):
        self.calls.append(('clear', flag)); self.flags[flag] = False
    def get_flag(self, flag):
        self.calls.append(('get', flag)); return self.flags[flag]
    def set_memory_pool_limit(self, pool, size):
        self.calls.append(('pool', pool, size))
    def create_timing_cache(self, data):
        self.calls.append(('cache', data)); return 'fresh-cache'
    def set_timing_cache(self, cache, ignore_mismatch):
        self.calls.append(('attach-cache', cache, ignore_mismatch)); return True
    def get_timing_cache(self):
        return NS(serialize=lambda: b'synthetic-cache')


def trt_constants():
    return NS(BuilderFlag=NS(TF32='TF32', FP16='FP16'), MemoryPoolType=NS(WORKSPACE='WORKSPACE'),
              HardwareCompatibilityLevel=NS(NONE='NONE'), DataType=NS(HALF='HALF', FLOAT='FLOAT'),
              NetworkDefinitionCreationFlag=NS(STRONGLY_TYPED=1))


class PureContracts(unittest.TestCase):
    def test_exact_production_source_pins(self):
        self.assertEqual(runtime.SOURCE_SHA256, '466225e995f0e70a194eccd8136f3c08fe2b5ca4c93f888c522ea2070a2044bb')
        identities = runtime.code_identities()
        self.assertEqual(identities['transformer_sha256'], runtime.TRANSFORMER_SHA256)
        self.assertEqual(identities['canonical_vfd_sha256'], runtime.CANONICAL_VFD_SHA256)

    def test_import_does_not_load_torch_trt_onnx_or_register_loader(self):
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import taesd_fp32_candidate_runtime;'
                'assert not any(k in sys.modules for k in ("torch","tensorrt","onnx","scripts.vae_fast_decoder"))')
        result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).parent)],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0)

    def test_default_off_before_any_file_or_dependency_action(self):
        with patch.object(runtime, '_gpu_dependencies') as gpu, patch.object(runtime, 'verify_lineage') as lineage:
            with self.assertRaisesRegex(runtime.CandidateRejected, 'not_explicitly_enabled'):
                runtime.build_candidate(source_path=None, candidate_path=None, proof_path=None,
                    expected_candidate_sha256=None, expected_proof_sha256=None, output_dir=None, device=None)
            with self.assertRaisesRegex(runtime.CandidateRejected, 'not_explicitly_enabled'):
                runtime.load_candidate(manifest_path=None, expected_manifest_sha256=None, device=None)
        gpu.assert_not_called(); lineage.assert_not_called()

    def test_verified_module_executes_only_checked_bytes_without_importer_read(self):
        source = b'checked_value = 42\n'
        path = Path('/nonexistent/verified_candidate_fixture.py')
        with patch.object(runtime, 'checked_bytes', return_value=source) as checked, \
             patch('importlib.util.spec_from_file_location', side_effect=AssertionError('importer read')), \
             patch('builtins.open', side_effect=AssertionError('second file read')), \
             patch('os.open', side_effect=AssertionError('second file read')):
            module = runtime._verified_module(path, runtime.sha(source), '_private_checked_fixture')
        checked.assert_called_once_with(path, runtime.sha(source), 1 << 20)
        self.assertEqual(module.checked_value, 42)
        self.assertEqual(module.__file__, str(path))
        self.assertEqual(module.__package__, '')
        self.assertNotIn('_private_checked_fixture', sys.modules)

    def test_tf32_clear_readback_native_opt3_and_empty_cache(self):
        config = Config()
        self.assertEqual(runtime.configure_builder(trt_constants(), config), runtime.BUILD_SETTINGS)
        self.assertEqual(config.calls[:2], [('clear', 'TF32'), ('get', 'TF32')])
        self.assertIn(('cache', b''), config.calls)
        self.assertIn(('attach-cache', 'fresh-cache', False), config.calls)
        self.assertEqual(config.hardware_compatibility_level, 'NONE')
        self.assertEqual(config.builder_optimization_level, 3)

    def test_tf32_refusal_or_fp16_flag_fails_closed(self):
        config = Config(); config.clear_flag = lambda flag: None
        with self.assertRaisesRegex(runtime.CandidateRejected, 'tf32_disable'):
            runtime.configure_builder(trt_constants(), config)
        config = Config(); config.flags['FP16'] = True
        with self.assertRaisesRegex(runtime.CandidateRejected, 'fp16_builder_flag'):
            runtime.configure_builder(trt_constants(), config)

    def test_builder_uses_strong_types_and_verifies_parser_boundaries(self):
        trt = trt_constants(); config = Config()
        tensors = [NS(name=runtime.PREFIX + 'activation_fp32', dtype='FLOAT'),
                   NS(name=runtime.PREFIX + 'conv_fp32', dtype='FLOAT'), NS(name='decoded', dtype='HALF')]
        network = NS(num_inputs=1, num_outputs=1, num_layers=3,
            get_input=lambda i: NS(dtype='HALF', shape=(8, 4, 32, 32)),
            get_output=lambda i: NS(dtype='HALF', shape=(8, 3, 256, 256)),
            get_layer=lambda i: NS(num_outputs=1, get_output=lambda j: tensors[i]))
        builder = NS(create_network=Mock(return_value=network), create_builder_config=lambda: config,
                     build_serialized_network=Mock(return_value=b'synthetic-plan'))
        trt.Builder = lambda log: builder
        trt.OnnxParser = lambda net, log: NS(parse=lambda raw: raw == b'synthetic-onnx')
        vfd = NS(_trt_logger=lambda: None)
        proof = {'target': {'original_output_names': ['decoded']}}
        plan, cache, settings = runtime.build_decoder_plan(trt, vfd, b'synthetic-onnx', proof)
        self.assertEqual(plan, b'synthetic-plan'); self.assertEqual(cache, b'synthetic-cache')
        builder.create_network.assert_called_once_with(2)
        tensors[1].dtype = 'HALF'
        with self.assertRaisesRegex(runtime.CandidateRejected, 'island_dtype'):
            runtime.build_decoder_plan(trt, vfd, b'synthetic-onnx', proof)

    def test_probe_requires_exact_shapes_hashes_and_fused_post(self):
        runtime.validate_probe(PROBE)
        variants = [{**PROBE, 'fused_vs_repo_post_mismatched_bytes': 1},
                    {**PROBE, 'u8_fused_sha256': 'c' * 64}, {**PROBE, 'probe_shape': [8., 4, 32, 32]},
                    {**PROBE, 'fp16_sha256': 'bad'}, {**PROBE, 'extra': True}]
        for probe in variants:
            with self.assertRaises(runtime.CandidateRejected):
                runtime.validate_probe(probe)

    def test_json_duplicates_and_nonfinite_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(runtime.CandidateRejected):
                runtime.parse_json(raw)

    def test_wrong_device_or_runtime_never_imports_canonical_vfd(self):
        torch = NS(__version__=runtime.RUNTIME['torch'], version=NS(cuda='12.1'),
                   cuda=NS(device_count=lambda: 1, get_device_name=lambda d: 'RTX 4090',
                           get_device_capability=lambda d: (8, 9)))
        with patch.dict(sys.modules, {'torch': torch, 'tensorrt': NS(__version__='10.3.0')}), \
             patch.object(runtime, '_verified_module') as module:
            with self.assertRaisesRegex(runtime.CandidateRejected, 'incompatible_runtime_or_device'):
                runtime._gpu_dependencies('cuda:0')
        module.assert_not_called()


class SyntheticArtifactContracts(unittest.TestCase):
    """Explicit synthetic source pin and transformer mocks; NOT actual model proof."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source, self.candidate = b'synthetic original ONNX fixture', b'synthetic transformed ONNX fixture'
        pin = patch.object(runtime, 'SOURCE_SHA256', runtime.sha(self.source)); pin.start(); self.addCleanup(pin.stop)
        self.proof = {'recipe': runtime.RECIPE, 'source_sha256': runtime.SOURCE_SHA256,
                      'transformed_sha256': runtime.sha(self.candidate),
                      'target': {'original_output_names': ['decoded']}}
        self.paths = {}
        for name, data in [('source', self.source), ('candidate', self.candidate), ('proof', runtime.json_bytes(self.proof))]:
            path = self.root / name; path.write_bytes(data); self.paths[name] = path
        helper = NS(transform=Mock(side_effect=lambda s, h: (self.candidate, copy.deepcopy(self.proof))))
        self.helper = helper
        self.module_patch = patch.object(runtime, '_verified_module', return_value=helper)
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        self.probes = [copy.deepcopy(PROBE)]
        self.fused = True
        self.backend = None
        def backend(*args, **kwargs):
            obj = NS(meta=kwargs['meta'], fused_post_enabled=self.fused,
                     probe_hashes=lambda: copy.deepcopy(self.probes.pop(0) if len(self.probes) > 1 else self.probes[0]))
            self.backend = obj
            return obj
        self.vfd = NS(_TrtEngine=lambda data, label: NS(data=data), TaesdTrtBackend=backend,
                      build_bgr_u8_post_plan=Mock(return_value=b'synthetic-post'),
                      export_taesd_decoder_onnx=Mock(return_value=self.source))
        self.gpu_patch = patch.object(runtime, '_gpu_dependencies', return_value=(NS(float16='float16'), trt_constants(), self.vfd, dict(runtime.RUNTIME)))
        self.gpu = self.gpu_patch.start(); self.addCleanup(self.gpu_patch.stop)
        plan_patch = patch.object(runtime, 'build_decoder_plan', return_value=(b'synthetic-decoder', b'synthetic-cache', dict(runtime.BUILD_SETTINGS)))
        self.builder = plan_patch.start(); self.addCleanup(plan_patch.stop)

    def lineage_args(self):
        return dict(source_path=self.paths['source'], candidate_path=self.paths['candidate'], proof_path=self.paths['proof'],
                    expected_candidate_sha256=runtime.sha(self.candidate), expected_proof_sha256=runtime.sha(self.paths['proof'].read_bytes()))

    def build(self):
        return runtime.build_candidate(**self.lineage_args(), output_dir=self.root / 'fresh', device='cuda:0', enable_build=True)

    def load(self, record, **kwargs):
        return runtime.load_candidate(manifest_path=record['manifest_path'], expected_manifest_sha256=record['manifest_sha256'],
                                      device='cuda:0', enable=True, **kwargs)

    def mutate_meta(self, record, fn):
        path = Path(record['manifest_path']); meta = runtime.parse_json(path.read_bytes()); fn(meta)
        raw = runtime.json_bytes(meta); path.write_bytes(raw)
        return {**record, 'manifest_sha256': runtime.sha(raw)}

    def test_lineage_rebuilds_actual_bytes_and_receipt(self):
        result = runtime.verify_lineage(**self.lineage_args())
        self.helper.transform.assert_called_once_with(self.source, runtime.SOURCE_SHA256)
        self.assertEqual(result['candidate'], self.candidate)
        self.paths['candidate'].write_bytes(b'different')
        with self.assertRaisesRegex(runtime.CandidateRejected, 'artifact_sha256'):
            runtime.verify_lineage(**self.lineage_args())

    def test_proof_cannot_claim_a_different_transformation(self):
        self.paths['proof'].write_bytes(runtime.json_bytes({**self.proof, 'claim': 'unsupported'}))
        with self.assertRaisesRegex(runtime.CandidateRejected, 'proof_not_exact'):
            runtime.verify_lineage(**self.lineage_args())

    def test_transformed_bytes_must_equal_regenerated_not_just_hash(self):
        self.helper.transform.side_effect = lambda s, h: (b'different returned bytes', self.proof)
        with self.assertRaisesRegex(runtime.CandidateRejected, 'not_exact_regenerated_graph'):
            runtime.verify_lineage(**self.lineage_args())

    def test_mock_build_load_exact_fingerprint_no_autobuild(self):
        record = self.build()
        meta = runtime.parse_json(Path(record['manifest_path']).read_bytes())
        self.assertEqual(meta['fingerprint']['precision'], 'fp16_with_final_conv_fp32')
        self.assertFalse(meta['fingerprint']['tf32']); self.assertTrue(meta['fingerprint']['strongly_typed'])
        self.assertNotEqual(meta['key'], '6a88814164891fd4cd7a')
        self.assertEqual(meta['key'], runtime.key_for(meta['fingerprint']))
        self.assertFalse(record['quality_accepted'] or record['full_gate_captured'] or record['release_ready'])
        self.builder.reset_mock()
        loaded = self.load(record, model='synthetic-reference-model')
        self.builder.assert_not_called()
        self.vfd.export_taesd_decoder_onnx.assert_called_once_with('synthetic-reference-model', 8, 'cuda:0')
        self.assertEqual(loaded.meta['probe'], PROBE)
        self.assertEqual(Path(record['manifest_path']).stat().st_mode & 0o777, 0o600)

    def test_fresh_directory_only_and_no_force_rebuild(self):
        self.build(); self.gpu.reset_mock()
        with self.assertRaisesRegex(runtime.CandidateRejected, 'fresh_absolute_output'):
            self.build()
        self.gpu.assert_not_called()

    def test_wrong_manifest_digest_stops_before_gpu(self):
        record = self.build(); self.gpu.reset_mock()
        with self.assertRaisesRegex(runtime.CandidateRejected, 'artifact_sha256'):
            self.load({**record, 'manifest_sha256': '0' * 64})
        self.gpu.assert_not_called()

    def test_corrupt_plan_and_unknown_paths_stop_before_gpu(self):
        record = self.build(); self.gpu.reset_mock()
        record = self.mutate_meta(record, lambda m: m.update(decoder_plan='../foreign.plan'))
        with self.assertRaisesRegex(runtime.CandidateRejected, 'plan_path'):
            self.load(record)
        self.gpu.assert_not_called()

    def test_corrupt_plan_hash_never_deserializes(self):
        record = self.build(); self.gpu.reset_mock()
        meta = runtime.parse_json(Path(record['manifest_path']).read_bytes())
        (Path(record['manifest_path']).parent / meta['decoder_plan']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(runtime.CandidateRejected, 'artifact_sha256'):
            self.load(record)
        self.gpu.assert_not_called()

    def test_mislabeled_precision_and_changed_helper_or_tf32_rejected(self):
        record = self.build(); original = Path(record['manifest_path']).read_bytes()
        for key, value in [('precision', 'fp16'), ('runtime_helper_sha256', '0' * 64), ('tf32', 0), ('strongly_typed', False)]:
            Path(record['manifest_path']).write_bytes(original)
            modified = self.mutate_meta(record, lambda m: m['fingerprint'].update({key: value}))
            self.gpu.reset_mock()
            with self.assertRaisesRegex(runtime.CandidateRejected, 'fingerprint_or_key'):
                self.load(modified)
            self.gpu.assert_not_called()

    def test_unverified_build_settings_and_false_quality_claim_rejected(self):
        record = self.build(); original = Path(record['manifest_path']).read_bytes()
        for modify in (lambda m: m['build']['settings'].update(tf32_readback=True), lambda m: m.update(quality_accepted=True)):
            Path(record['manifest_path']).write_bytes(original)
            modified = self.mutate_meta(record, modify)
            with self.assertRaises(runtime.CandidateRejected):
                self.load(modified)

    def test_failed_or_nondeterministic_probe_leaves_no_manifest(self):
        self.probes = [copy.deepcopy(PROBE), {**PROBE, 'fp16_sha256': 'c' * 64}]
        with self.assertRaisesRegex(runtime.CandidateRejected, 'build_probe_not_deterministic'):
            self.build()
        self.assertEqual(list((self.root / 'fresh').glob('taesd_trt_*.json')), [])

    def test_loader_requires_exact_recorded_probe(self):
        record = self.build(); self.probes = [{**PROBE, 'fp16_sha256': 'c' * 64}]
        with self.assertRaisesRegex(runtime.CandidateRejected, 'loaded_probe_not_exact'):
            self.load(record)

    def test_reference_model_change_and_disabled_fused_post_rejected(self):
        record = self.build()
        self.vfd.export_taesd_decoder_onnx.return_value = b'different model'
        with self.assertRaisesRegex(runtime.CandidateRejected, 'reference_model_export'):
            self.load(record, model='different')
        self.fused = False
        with self.assertRaisesRegex(runtime.CandidateRejected, 'fused_post_must'):
            self.load(record)

    def test_symlink_source_and_missing_manifest_never_gpu(self):
        alias = self.root / 'alias'; alias.symlink_to(self.paths['source'])
        args = {**self.lineage_args(), 'source_path': alias}
        with self.assertRaisesRegex(runtime.CandidateRejected, 'symlink'):
            runtime.verify_lineage(**args)
        with self.assertRaisesRegex(runtime.CandidateRejected, 'required_file_unavailable'):
            self.load({'manifest_path': str(self.root / 'absent.json'), 'manifest_sha256': 'f' * 64})
        self.gpu.assert_not_called()


HAS_ONNX = importlib.util.find_spec('onnx') is not None and importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_ONNX, 'ONNX/NumPy unavailable; real graph proof regeneration not exercised locally')
class OptionalOnnxLineageTest(unittest.TestCase):
    def test_synthetic_actual_onnx_proof_regenerated_without_gpu(self):
        import taesd_final_conv_fp32 as transform
        from test_taesd_final_conv_fp32 import RealOnnxTests
        source = transform.wire(RealOnnxTests().model())
        digest = runtime.sha(source)
        candidate, proof = transform.transform(source, digest)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'SOURCE_SHA256', digest), \
             patch.object(runtime, '_gpu_dependencies') as gpu:
            root = Path(tmp).resolve()
            for name, data in [('source', source), ('candidate', candidate), ('proof', runtime.json_bytes(proof))]:
                (root / name).write_bytes(data)
            result = runtime.verify_lineage(source_path=root / 'source', candidate_path=root / 'candidate', proof_path=root / 'proof',
                expected_candidate_sha256=runtime.sha(candidate), expected_proof_sha256=runtime.sha((root / 'proof').read_bytes()))
        self.assertEqual(result['candidate'], candidate)
        gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
