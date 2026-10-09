"""Synthetic CPU contracts for a literal tail-plan overlay; no CUDA imports."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.repro_3090 import assemble_owned_unet_tail_a3 as helper

BLOCKS = ('prefix', 'down0rest', 'down1', 'down2', 'down3', 'mid', 'up0', 'up1', 'up2', 'up3', 'tail')
INT8 = set(BLOCKS[2:9])
TAIL = '8219d8a31bf256b0c621b5f4662c2148613818e985a0a712bae0e2e03e4eb527'
RECIPE = 'f6f90777264b5302'
SPEC = [
    ('down0rest', ['a0', 'ehs'], ['d0r0', 'd0r1', 'd0ds']),
    ('down1', ['d0ds', 'ehs'], ['d1r0', 'd1r1', 'd1ds']),
    ('down2', ['d1ds', 'ehs'], ['d2r0', 'd2r1', 'd2ds']),
    ('down3', ['d2ds'], ['d3r0', 'd3r1']), ('mid', ['d3r1', 'ehs'], ['m']),
    ('up0', ['m', 'd2ds', 'd3r0', 'd3r1'], ['u0']),
    ('up1', ['u0', 'd1ds', 'd2r0', 'd2r1', 'ehs'], ['u1']),
    ('up2', ['u1', 'd0ds', 'd1r0', 'd1r1', 'ehs'], ['u2']),
    ('up3', ['u2', 'h0', 'd0r0', 'd0r1', 'ehs'], ['u3']), ('tail', ['u3'], ['out'])]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class OverlayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.native, self.portable = self.root / 'native', self.root / 'portable'
        self.manifests = {}
        for directory, label, gpu, cap, compat in (
                (self.native, 'native', 'NVIDIA GeForce RTX 3090', [8, 6], 'none'),
                (self.portable, 'portable', 'NVIDIA GeForce RTX 4070 SUPER', [8, 9], 'ampere_plus')):
            directory.mkdir()
            flags = dict(precision='fp16', builder_flags=['FP16'], builder_optimization_level=5,
                         workspace_gb=2.0, timing_cache=True, onnx_opset=17)
            if compat != 'none': flags['hardware_compatibility_level'] = compat
            manifest = dict(schema='musetalk_unet_stagewise_trt_v1', batch=16, variant='srccache',
                timestep=0, complete=True, tensorrt_version='10.3.0', torch_version='2.5.1+cu121',
                gpu=gpu, compute_capability=cap, hardware_compatibility_level=compat,
                spec=[dict(name=b, inputs=i, outputs=o) for b, i, o in SPEC], build_flags=flags,
                int8_calibration=dict(recipe_sha256_16=RECIPE, files=['unet_io_000001_bs8.pt']),
                blocks={}, probe={'output_file': 'probe_output.pt'}, runtime={'old': True})
            for block in BLOCKS:
                raw = f'{label}:{block}:synthetic-plan'.encode()
                name = block + ('.int8.plan' if block in INT8 else '.plan')
                (directory / name).write_bytes(raw)
                row = next((s for s in manifest['spec'] if s['name'] == block),
                           dict(inputs=['x'], outputs=['h0', 'a0']))
                manifest['blocks'][block] = dict(engine_file=name, engine_sha256=sha(raw),
                    onnx_sha256=TAIL if block == 'tail' else sha(f'{label}:{block}:graph'.encode()),
                    inputs=row['inputs'], outputs=row['outputs'],
                    build_flags=dict(flags, precision='int8_qdq_recipe:' + RECIPE if block in INT8 else 'fp16'))
            self.manifests[directory] = manifest; self.write(directory, manifest)

    def write(self, directory, manifest):
        raw = json.dumps(manifest, sort_keys=True).encode()
        (directory / 'manifest.json').write_bytes(raw)
        return sha(raw)

    def prepare(self, target=None, **kwargs):
        return helper.prepare(self.native, self.portable, target or self.root / 'candidate',
            native_sha=kwargs.get('native_sha', sha((self.native / 'manifest.json').read_bytes())),
            portable_sha=kwargs.get('portable_sha', sha((self.portable / 'manifest.json').read_bytes())))

    def test_import_is_inert_without_optional_dependencies_or_file_writes(self):
        code = ('import sys;sys.path.insert(0,sys.argv[1]);'
                'from scripts.repro_3090 import assemble_owned_unet_tail_a3;'
                'assert not any(k in sys.modules for k in ("torch","tensorrt","onnx"))')
        result = subprocess.run([sys.executable, '-B', '-c', code,
            str(Path(__file__).resolve().parents[2])], cwd=self.root, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.root.iterdir()), {self.native, self.portable})

    def test_cli_is_default_off_before_host_or_gpu_checks(self):
        with patch.object(helper, 'capture', side_effect=AssertionError('external probe')), \
             patch.object(helper.socket, 'gethostname', side_effect=AssertionError('host probe')):
            with self.assertRaisesRegex(ValueError, 'explicit_execution_required'): helper.main([])
        self.assertFalse((self.root / 'candidate').exists())

    def test_exact_ten_native_one_portable_plans_and_unfinalized_provenance(self):
        before = {d: (d / 'manifest.json').read_bytes() for d in self.manifests}
        candidate = self.prepare(); destination = self.root / 'candidate' / 'bs16'
        self.assertEqual(set(candidate['blocks']), set(BLOCKS))
        for block in BLOCKS:
            source = self.portable if block == 'tail' else self.native
            entry = self.manifests[source]['blocks'][block]
            self.assertEqual(candidate['blocks'][block], entry)
            self.assertEqual((destination / entry['engine_file']).read_bytes(), (source / entry['engine_file']).read_bytes())
            self.assertFalse((destination / entry['engine_file']).is_symlink())
            provenance = candidate['block_provenance'][block]
            self.assertEqual(provenance['source_gpu'], self.manifests[source]['gpu'])
            self.assertEqual(provenance['source_manifest_sha256'], sha(before[source]))
            self.assertEqual(provenance['engine_sha256'], entry['engine_sha256'])
            self.assertEqual(provenance['hardware_compatibility_level'], self.manifests[source]['hardware_compatibility_level'])
        for key in ('complete', 'quality_accepted', 'performance_measured', 'release_ready'):
            self.assertIs(candidate[key], False)
        self.assertEqual(candidate['hardware_compatibility_level'], 'none')
        self.assertEqual(candidate['compute_capability'], [8, 6])
        self.assertNotIn('probe', candidate); self.assertNotIn('runtime', candidate)
        self.assertFalse((destination / 'probe_output.pt').exists())
        for directory, raw in before.items(): self.assertEqual((directory / 'manifest.json').read_bytes(), raw)

    def test_manifest_hash_and_already_existing_target_rejected(self):
        with self.assertRaises(ValueError): self.prepare(native_sha='0' * 64)
        with self.assertRaises(ValueError): self.prepare(portable_sha='0' * 64)
        target = self.root / 'candidate'; target.mkdir()
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(list(target.iterdir()), [])

    def test_metadata_graph_recipe_and_spec_changes_rejected_before_output(self):
        variants = [('schema', 'bad'), ('batch', 8), ('variant', 'default'), ('timestep', 1),
                    ('complete', False), ('tensorrt_version', '10.4.0'), ('torch_version', '2.6.0'),
                    ('gpu', 'NVIDIA GeForce RTX 4090'), ('compute_capability', [8, 6]),
                    ('hardware_compatibility_level', 'none')]
        for key, value in variants:
            altered = copy.deepcopy(self.manifests[self.portable]); altered[key] = value
            self.write(self.portable, altered)
            with self.subTest(key=key), self.assertRaises(ValueError): self.prepare()
            self.assertFalse((self.root / 'candidate').exists())
        for label, mutation in (
                ('tail_graph', lambda m: m['blocks']['tail'].update(onnx_sha256='0' * 64)),
                ('tail_precision', lambda m: m['blocks']['tail']['build_flags'].update(precision='int8_qdq')),
                ('block_compatibility', lambda m: m['blocks']['tail']['build_flags'].update(hardware_compatibility_level='none')),
                ('recipe', lambda m: m['int8_calibration'].update(recipe_sha256_16='0' * 16)),
                ('calibration_files', lambda m: m['int8_calibration'].update(files=['other.pt'])),
                ('chain_spec', lambda m: m['spec'][0].update(inputs=['wrong'])),
                ('missing_block', lambda m: m['blocks'].pop('up2'))):
            altered = copy.deepcopy(self.manifests[self.portable]); mutation(altered)
            self.write(self.portable, altered)
            with self.subTest(label=label), self.assertRaises(ValueError): self.prepare()
            self.assertFalse((self.root / 'candidate').exists())

    def test_plan_path_symlink_missing_and_actual_sha_fail_closed(self):
        for name in ('../tail.plan', '/tail.plan'):
            altered = copy.deepcopy(self.manifests[self.portable]); altered['blocks']['tail']['engine_file'] = name
            self.write(self.portable, altered)
            with self.assertRaises(ValueError): self.prepare()
            self.assertFalse((self.root / 'candidate').exists())
        self.write(self.portable, self.manifests[self.portable])
        path = self.portable / 'tail.plan'; raw = path.read_bytes(); path.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'candidate').exists())
        path.unlink()
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'candidate').exists())
        outside = self.root / 'outside.plan'; outside.write_bytes(raw); path.symlink_to(outside)
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'candidate').exists())

    def test_manifest_symlink_duplicate_filenames_and_malformed_digests_rejected(self):
        altered = copy.deepcopy(self.manifests[self.native])
        altered['blocks']['up3']['engine_file'] = altered['blocks']['prefix']['engine_file']
        self.write(self.native, altered)
        with self.assertRaises(ValueError): self.prepare()
        altered = copy.deepcopy(self.manifests[self.native]); altered['blocks']['up3']['engine_sha256'] = 'bad'
        self.write(self.native, altered)
        with self.assertRaises(ValueError): self.prepare()
        path = self.native / 'manifest.json'; raw = path.read_bytes(); path.unlink()
        outside = self.root / 'outside.json'; outside.write_bytes(raw); path.symlink_to(outside)
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'candidate').exists())


if __name__ == '__main__':
    unittest.main()
