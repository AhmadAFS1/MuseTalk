"""GPU-free aggregate contracts; all generated measurements are synthetic."""
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taesd_fp32_candidate_aggregate as aggregate


def fixture(repeats=2, fps=400):
    frames = 6 * 24 * 240; wall = frames / fps
    return {'status': 'complete', 'code_integrity': {'matches_accepted_render_json': True},
        'args': {'identity_list': ['black_man_short_beard', 'black_woman', 'east_asian_man_goatee',
            'middle_eastern_man_full_beard', 'south_asian_woman', 'white_man_clean_shaven'],
            'mode': 'multi', 'backend': 'stagewise16_taesdtrt', 'pack': 16, 'decode_split': 8,
            'encode': False, 'save_arrays': False, 'streams': 6, 'loops': 24, 'repeats': repeats},
        'summary': {'deterministic_per_identity': True},
        'thermal_warmup': {'seconds': 120, 'last_30s': {'n': 60, 'temperature.gpu': {'min': 69, 'max': 71}}},
        'repeats': [{'repeat': i, 'frames': frames, 'wall_s': wall, 'aggregate_fps': fps,
            'gpu': {'smi': {'n': 100}}, 'per_worker': {str(s): {'frames': 24 * 240,
                'fps': 24 * 240 / wall, 'done_after_t0_s': wall - .1} for s in range(6)}} for i in range(repeats)]}


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.child, self.report = aggregate.support()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.out = self.root / 'out'; self.out.mkdir()
        self.accepted = self.root / 'experiments/accepted'; self.accepted.mkdir(parents=True)
        self.engine = self.root / 'engine'; (self.engine / 'bs16').mkdir(parents=True)
        for relative in ('scripts/chin_multistream_render.py', 'scripts/vae_fast_decoder.py'):
            dest = self.root / relative; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((aggregate.ROOT / relative).read_bytes())
        self.profile = self.root / 'profile.env'
        self.profile.write_text('MUSETALK_TAESD_COMPILE=1\nHLS_SCHEDULER_MAX_BATCH=16\n')
        self.key = 'a' * 20
        self.meta = dict(schema='taesd_fp32_candidate_engine_v1', status='BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED',
            key=self.key, decoder_plan_sha256='b' * 64, post_plan_sha256='c' * 64)
        self.manifest = self.root / ('taesd_trt_' + self.key + '.json'); self.manifest.write_text(json.dumps(self.meta))
        self.engine_manifest = self.engine / 'bs16/manifest.json'
        self.engine_manifest.write_text(json.dumps(dict(complete=True, batch=16, variant='srccache', probe={'output_sha256': 'd' * 64})))
        self.runtime = types.SimpleNamespace(SCHEMA=self.meta['schema'], parse_json=json.loads,
            valid_digest=lambda s: isinstance(s, str) and len(s) == 64)
        self.data = fixture(); self.exitcode = 0
        self.data['backends'] = dict(unet_name='tensorrt_unet_stagewise', decoder_name='taesd_trt',
            decoder_trt_key=self.key, decoder_trt_plan_sha256='b' * 64,
            unet_describe=dict(engine_dir=str(self.engine / 'bs16'), batch=16, probe_status='exact',
                probe_validation=dict(kind='exact', expected_sha256='d' * 64, actual_sha256='d' * 64)))
        self.receipt = self.root / 'receipt.json'
        self.kwargs = dict(stage='T', manifest_path=self.manifest, manifest_sha256=self.sha(self.manifest),
            candidate_key=self.key, profile_path=self.profile, profile_sha256=self.sha(self.profile),
            engine_root=self.engine, engine_manifest_sha256=self.sha(self.engine_manifest), receipt_path=self.receipt,
            output_dir=self.out, label='cpu_fixture', accepted_root=self.accepted, enable=True)

    @staticmethod
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_fake(self, **changes):
        @contextmanager
        def route(runtime, manifest, digest, meta, vfd, state):
            self.assertIs(runtime, self.runtime); self.assertEqual(meta, self.meta)
            state['candidate_attempts'] = state['candidate_calls'] = 1
            yield
        def render(body, path):
            self.assertEqual(hashlib.sha256(body).hexdigest(), aggregate.RENDER_SHA)
            self.assertEqual(os.environ['MUSETALK_TAESD_COMPILE'], '0')
            self.assertEqual(os.environ['MUSETALK_TRT_FALLBACK'], '0')
            self.assertEqual(os.environ['HLS_SCHEDULER_MAX_BATCH'], '16')
            self.assertNotIn('MUSETALK_AMBIENT_BAD', os.environ)
            (self.out / 'cpu_fixture.json').write_text(json.dumps(self.data))
            if self.exitcode: raise SystemExit(self.exitcode)
        with patch.object(aggregate, 'ROOT', self.root), patch.object(aggregate, 'support', return_value=(self.child, self.report)), \
             patch.object(self.child, 'ROOT', self.root), patch.object(self.child, 'verified_runtime', return_value=self.runtime), \
             patch.object(self.child, 'scoped_candidate', side_effect=route), patch.object(self.child, 'execute_target', side_effect=render):
            return aggregate.launch(**{**self.kwargs, **changes})

    def record(self): return json.loads(self.receipt.read_text())

    def test_import_is_inert(self):
        code = "import runpy,sys;runpy.run_path(sys.argv[1],run_name='inert');assert not any(x in sys.modules for x in ('torch','tensorrt','onnx','numpy'))"
        subprocess.run([sys.executable, '-B', '-c', code, aggregate.__file__], check=True, timeout=10)

    def test_real_support_pins_and_defaults(self):
        self.assertEqual(self.child.RUNTIME_SHA, aggregate.RUNTIME_SHA)
        for stage, repeats in [('T', '2'), ('SUST', '5')]:
            argv = aggregate.aggregate_arguments(stage, self.out, 'test')
            self.assertEqual(argv[argv.index('--loops') + 1], '24')
            self.assertEqual(argv[argv.index('--repeats') + 1], repeats)
            self.assertEqual(argv[argv.index('--streams') + 1], '6')
            self.assertEqual(argv[argv.index('--min-timed-s') + 1], '60')
            self.assertEqual(argv[argv.index('--thermal-warmup-s') + 1], '120')
            for forbidden in ('--tracking-overlap', '--flags', '--pack', '--depth', '--encode', '--save-arrays'):
                self.assertNotIn(forbidden, argv)

    def test_disabled_before_support_or_writes(self):
        with patch.object(aggregate, 'support') as support, self.assertRaises(aggregate.Rejected):
            aggregate.launch(**{**self.kwargs, 'enable': False})
        support.assert_not_called(); self.assertFalse(self.receipt.exists())

    def test_valid_diagnostic_restores_environment_and_main(self):
        before, main = dict(os.environ), sys.modules['__main__']
        with patch.dict(os.environ, MUSETALK_AMBIENT_BAD='1'):
            self.assertEqual(self.run_fake(), 0)
            self.assertEqual(os.environ['MUSETALK_AMBIENT_BAD'], '1')
        self.assertEqual(dict(os.environ), before); self.assertIs(sys.modules['__main__'], main)
        row = self.record(); self.assertEqual(row['status'], 'PASS')
        self.assertFalse(row['quality_accepted']); self.assertFalse(row['performance_accepted']); self.assertFalse(row['release_ready'])

    def test_valid_399_96_is_failure_not_rounded_pass(self):
        backends = self.data['backends']; self.data = fixture(fps=399.96); self.data['backends'] = backends
        self.assertEqual(self.run_fake(), 1); self.assertEqual(self.record()['status'], 'FAIL')

    def test_short_window_invalid(self):
        backends = self.data['backends']; self.data = fixture(fps=1000); self.data['backends'] = backends
        self.assertEqual(self.run_fake(), 2); self.assertEqual(self.record()['status'], 'INVALID')

    def test_wrong_shared_denominator_invalid(self):
        self.data['repeats'][0]['aggregate_fps'] *= 2
        self.assertEqual(self.run_fake(), 2)

    def test_missing_frames_invalid(self):
        self.data['repeats'][0]['per_worker']['0']['frames'] -= 8
        self.assertEqual(self.run_fake(), 2)

    def test_missing_sust_windows_invalid(self):
        self.assertEqual(self.run_fake(stage='SUST'), 2)

    def test_full_sust_five_windows(self):
        backends = self.data['backends']; self.data = fixture(repeats=5); self.data['backends'] = backends
        self.assertEqual(self.run_fake(stage='SUST'), 0)
        self.assertEqual(len(self.record()['aggregate']['windows']), 5)

    def test_renderer_failure_preserved_even_with_complete_report(self):
        self.exitcode = 7
        self.assertEqual(self.run_fake(), 7)
        self.assertEqual(self.record()['status'], 'CHILD_FAILED'); self.assertNotIn('aggregate', self.record())

    def test_wrong_backend_or_candidate_rejected(self):
        self.data['backends']['decoder_trt_plan_sha256'] = 'e' * 64
        self.assertEqual(self.run_fake(), 2)

    def test_explicit_key_mismatch_before_execution(self):
        with self.assertRaisesRegex(aggregate.Rejected, 'candidate_key'):
            self.run_fake(candidate_key='f' * 20)
        self.assertFalse(self.receipt.exists())

    def test_existing_report_not_reused(self):
        (self.out / 'cpu_fixture.json').write_text('{}')
        with self.assertRaisesRegex(aggregate.Rejected, 'fresh_run'):
            self.run_fake()

    def test_profile_secrets_expansion_duplicates_rejected(self):
        for raw in (b'MUSETALK_SECRET=x', b'MUSETALK_X=$HOME', b'MUSETALK_X=1\nMUSETALK_X=2'):
            with self.subTest(raw=raw), self.assertRaises(aggregate.Rejected): aggregate.profile_values(raw)

    def test_unknown_or_reduced_cli_not_supported(self):
        with self.assertRaises(aggregate.Rejected): aggregate.aggregate_arguments('N15', self.out, 'test')
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            aggregate.main(['--loops', '1'])


if __name__ == '__main__':
    unittest.main()
