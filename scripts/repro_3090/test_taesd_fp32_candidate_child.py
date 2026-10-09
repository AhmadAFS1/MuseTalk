"""CPU-only routing contracts; fixture bytes and fake GPU API, plus real CPU spawn."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taesd_fp32_candidate_child as child


class Device:
    def __init__(self, value): self.value = value
    def __str__(self): return self.value


class PureContracts(unittest.TestCase):
    def test_real_pinned_files_match(self):
        child.checked(child.ROOT / 'scripts/vae_fast_decoder.py', child.VFD_SHA)
        for path, sha in child.TARGETS.values(): child.checked(child.ROOT / path, sha)
        runtime = child.verified_runtime()
        self.assertEqual(runtime.CANONICAL_VFD_SHA256, child.VFD_SHA)

    def test_import_is_stdlib_only(self):
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import taesd_fp32_candidate_child;'
                'assert not any(x in sys.modules for x in ("torch","tensorrt","onnx","numpy","scripts.vae_fast_decoder"))')
        result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(child.__file__).parent)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_enable_stops_before_files_or_runtime(self):
        with patch.object(child, 'verified_runtime') as api:
            with self.assertRaisesRegex(child.RoutingRejected, 'explicit_enable'):
                child.launch(target='gate', target_args=[], manifest_path=None, manifest_sha256=None,
                             receipt_path=None, output_dir=None)
        api.assert_not_called()

    def test_unknown_target_or_arbitrary_cli_rejected(self):
        with self.assertRaisesRegex(child.RoutingRejected, 'unknown_target'):
            child.launch(target='other', target_args=[], manifest_path=None, manifest_sha256=None,
                         receipt_path=None, output_dir=None, enable=True)
        with self.assertRaises(child.RoutingRejected): child.main(['--execute-arbitrary', 'anything'])

    def test_runtime_helper_pin_tampering_rejected_before_exec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'scripts/repro_3090'; directory.mkdir(parents=True)
            (directory / 'taesd_fp32_candidate_runtime.py').write_text('raise AssertionError("must not execute")\n')
            with patch.object(child, 'ROOT', root), self.assertRaisesRegex(child.RoutingRejected, 'pinned_bytes'):
                child.verified_runtime()


class RoutingContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'scripts').mkdir()
        self.out = self.root / 'output'; self.out.mkdir()
        self.receipt = self.root / 'receipt.json'
        self.key = 'a' * 20
        self.meta = {'schema': 'taesd_fp32_candidate_engine_v1', 'status': 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED',
                     'key': self.key, 'decoder_plan_sha256': 'b' * 64, 'post_plan_sha256': 'c' * 64}
        self.manifest = self.root / f'taesd_trt_{self.key}.json'
        self.manifest.write_text(json.dumps(self.meta))
        self.manifest_sha = child.digest(self.manifest.read_bytes())
        self.vfd_raw = b'import torch\ndef load_taesd_trt_backend(*args, **kwargs):\n    return "original"\n'
        (self.root / 'scripts/vae_fast_decoder.py').write_bytes(self.vfd_raw)
        self.backend = types.SimpleNamespace(name='taesd_trt', meta=copy.deepcopy(self.meta), paths={'meta': self.manifest})
        self.modules = []
        def load(**kwargs):
            self.modules.append(sys.modules[child.VFD_NAME]); return self.backend
        self.api = types.SimpleNamespace(SCHEMA=self.meta['schema'], parse_json=json.loads,
            valid_digest=lambda x: isinstance(x, str) and len(x) == 64, load_candidate=Mock(side_effect=load))
        self.torch = types.ModuleType('torch'); self.torch.device = Device; self.torch.float16 = object()
        package = types.ModuleType('scripts'); package.__path__ = [str(self.root / 'scripts')]
        for ctx in (patch.object(child, 'ROOT', self.root), patch.object(child, 'VFD_SHA', child.digest(self.vfd_raw)),
                    patch.object(child, 'verified_runtime', return_value=self.api),
                    patch.dict(sys.modules, {'scripts': package, 'torch': self.torch})):
            ctx.start(); self.addCleanup(ctx.stop)
        self.addCleanup(sys.modules.pop, child.VFD_NAME, None)
        self.args = ['--backend', 'stagewise16_taesdtrt', '--streams', '6', '--loops', '1', '--repeats', '1',
                     '--save-arrays', '--encode', '--compare-accepted', '--out-root', str(self.out), '--label', 'candidate_capture']
        self.target = self.root / 'scripts/chin_multistream_render.py'
        self.normal_source = ('from scripts import vae_fast_decoder as vfd\n'
                              'vfd.load_taesd_trt_backend(vfd.torch.device("cuda:0"), vfd.torch.float16, model=object())\n')

    def run_target(self, source=None, **changes):
        raw = (source if source is not None else self.normal_source).encode()
        selected = changes.get('target', 'render')
        target_path = self.root / child.TARGETS[selected][0]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(raw)
        args = dict(target='render', target_args=self.args, manifest_path=self.manifest, manifest_sha256=self.manifest_sha,
                    receipt_path=self.receipt, output_dir=self.out, enable=True)
        args.update(changes)
        targets = dict(child.TARGETS); targets[selected] = (child.TARGETS[selected][0], child.digest(raw))
        with patch.object(child, 'TARGETS', targets):
            return child.launch(**args)

    def record(self): return json.loads(self.receipt.read_text())

    def test_successful_explicit_route_and_final_restoration(self):
        old_env, old_argv, old_path, old_cwd, old_main = dict(os.environ), sys.argv, list(sys.path), os.getcwd(), sys.modules['__main__']
        source = self.normal_source + 'import os,sys\nos.environ["CHILD_ONLY_FIXTURE"]="1"\nsys.path.append("fixture")\n'
        self.assertEqual(self.run_target(source), 0)
        row = self.record()
        self.assertEqual((row['status'], row['returncode'], row['candidate_calls'], row['candidate_attempts']), ('complete', 0, 1, 1))
        self.assertEqual(row['candidate_key'], self.key); self.assertEqual(row['manifest_sha256'], self.manifest_sha)
        self.assertFalse(row['quality_accepted']); self.assertFalse(row['performance_accepted']); self.assertFalse(row['release_ready'])
        self.assertEqual(self.api.load_candidate.call_args.kwargs['enable'], True)
        self.assertIsNotNone(self.api.load_candidate.call_args.kwargs['model'])
        self.assertEqual(dict(os.environ), old_env); self.assertIs(sys.argv, old_argv)
        self.assertEqual(sys.path, old_path); self.assertEqual(os.getcwd(), old_cwd)
        self.assertIs(sys.modules['__main__'], old_main)
        self.assertNotIn(child.VFD_NAME, sys.modules)
        self.assertFalse(hasattr(sys.modules['scripts'], 'vae_fast_decoder'))
        self.assertEqual(self.modules[0].load_taesd_trt_backend(), 'original')

    def test_gate_style_nonzero_exit_is_preserved(self):
        self.assertEqual(self.run_target(self.normal_source + 'raise SystemExit(1)\n'), 1)
        row = self.record()
        self.assertEqual((row['status'], row['target_returncode'], row['returncode']), ('failed', 1, 1))

    def test_gate_route_full_arguments_output_env_and_strict_exit(self):
        corpus = self.root / 'corpus'; corpus.mkdir(); (corpus / 'holdout').mkdir()
        for directory, count in ((corpus, 352), (corpus / 'holdout', 96)):
            for i in range(count): (directory / f'unet_io_{i:04d}.pt').write_bytes(b'CPU fixture')
        source = self.normal_source + f'import os\nassert os.environ["REPRO_GATE_OUT"] == {str(self.out)!r}\nraise SystemExit(1)\n'
        self.assertEqual(self.run_target(source, target='gate', target_args=['--no-record', '--corpus', str(corpus)]), 1)
        self.assertEqual((self.record()['target'], self.record()['candidate_calls']), ('gate', 1))
        self.assertEqual(self.record()['workload']['holdout_files_expected'], 96)

    def test_exception_keeps_failure_and_restores_hook_environment(self):
        hooks, env = list(sys.meta_path), dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            self.run_target(self.normal_source + 'raise RuntimeError("fixture")\n')
        self.assertEqual(self.record()['status'], 'failed')
        self.assertEqual(sys.meta_path, hooks); self.assertEqual(dict(os.environ), env)
        self.assertEqual(self.modules[0].load_taesd_trt_backend(), 'original')

    def test_no_candidate_invocation_is_invalid_even_on_zero_exit(self):
        with self.assertRaisesRegex(child.RoutingRejected, 'never_invoked'):
            self.run_target('raise SystemExit(0)\n')
        self.assertEqual((self.record()['status'], self.record()['returncode']), ('invalid', 2))
        self.api.load_candidate.assert_not_called()

    def test_returned_backend_mismatch_is_invalid(self):
        self.backend.meta['decoder_plan_sha256'] = 'd' * 64
        with self.assertRaisesRegex(child.RoutingRejected, 'backend_identity'):
            self.run_target()
        self.assertEqual(self.record()['candidate_calls'], 0)
        self.assertEqual(self.record()['candidate_attempts'], 1)

    def test_wrong_device_dtype_model_and_fallback_environment_rejected(self):
        cases = [('"cuda:1"', 'vfd.torch.float16', 'object()', ''),
                 ('"cuda:0"', 'object()', 'object()', ''),
                 ('"cuda:0"', 'vfd.torch.float16', 'None', ''),
                 ('"cuda:0"', 'vfd.torch.float16', 'object()', 'import os\nos.environ["MUSETALK_TRT_FALLBACK"]="1"\n')]
        for device, dtype, model, prefix in cases:
            with self.subTest(device=device, dtype=dtype, model=model, prefix=prefix):
                source = 'from scripts import vae_fast_decoder as vfd\n' + prefix
                source += f'vfd.load_taesd_trt_backend(vfd.torch.device({device}), {dtype}, model={model})\n'
                receipt = self.root / ('contract' + str(len(list(self.root.glob('contract*')))) + '.json')
                with self.assertRaises(child.RoutingRejected): self.run_target(source, receipt_path=receipt)
        self.api.load_candidate.assert_not_called()

    def test_candidate_failure_cannot_fall_back(self):
        self.api.load_candidate.side_effect = RuntimeError('fixture load failure')
        with self.assertRaisesRegex(child.RoutingRejected, 'candidate_child_invalid'): self.run_target()
        self.assertEqual(self.record()['status'], 'invalid')
        self.assertEqual(self.record()['candidate_calls'], 0)

    def test_preloaded_vfd_is_rejected_not_modified(self):
        existing = types.ModuleType(child.VFD_NAME)
        with patch.dict(sys.modules, {child.VFD_NAME: existing}):
            with self.assertRaisesRegex(child.RoutingRejected, 'fresh_child'):
                self.run_target()
            self.assertIs(sys.modules[child.VFD_NAME], existing)

    def test_manifest_pin_tampering_stops_before_target(self):
        self.manifest.write_text('{}')
        with self.assertRaisesRegex(child.RoutingRejected, 'pinned_bytes'):
            self.run_target()
        self.assertFalse(self.receipt.exists()); self.api.load_candidate.assert_not_called()

    def test_vfd_and_target_pin_tampering_stop_before_execution(self):
        with patch.object(child, 'VFD_SHA', '0' * 64):
            with self.assertRaisesRegex(child.RoutingRejected, 'pinned_bytes'): self.run_target()
        raw = self.normal_source.encode(); self.target.write_bytes(raw)
        with patch.object(child, 'TARGETS', {'render': ('scripts/chin_multistream_render.py', '0' * 64)}):
            with self.assertRaisesRegex(child.RoutingRejected, 'pinned_bytes'):
                child.launch(target='render', target_args=self.args, manifest_path=self.manifest,
                    manifest_sha256=self.manifest_sha, receipt_path=self.receipt, output_dir=self.out, enable=True)
        self.api.load_candidate.assert_not_called()

    def test_reduced_or_tuned_render_arguments_rejected(self):
        for bad in (['--streams', '1'], ['--flags', 'MUSETALK_TAESD_BACKEND=compiled'], ['--tracking-overlap'],
                    ['--mode', 'serial'], ['--crf', '23'], ['--limit-files', '1'], ['--identities', 'black_woman']):
            with self.subTest(bad=bad), self.assertRaises(child.RoutingRejected):
                child.validate_target_args('render', self.args + bad, self.out)
        without_encode = [x for x in self.args if x != '--encode']
        with self.assertRaises(child.RoutingRejected): child.validate_target_args('render', without_encode, self.out)

    def test_gate_full_corpus_no_record_and_no_limit_required(self):
        corpus = self.root / 'corpus'; corpus.mkdir(); (corpus / 'holdout').mkdir()
        for directory, count in ((corpus, 352), (corpus / 'holdout', 96)):
            for i in range(count): (directory / f'unet_io_{i:04d}.pt').write_bytes(b'CPU fixture only')
        args = ['--no-record', '--corpus', str(corpus)]
        result = child.validate_target_args('gate', args, self.out)
        self.assertEqual(result['files_expected'], 448)
        for bad in (args + ['--limit-files', '0'], args + ['--limit-files', '1'], args[1:]):
            with self.assertRaises(child.RoutingRejected): child.validate_target_args('gate', bad, self.out)
        (corpus / 'unet_io_0000.pt').unlink()
        with self.assertRaisesRegex(child.RoutingRejected, '448'):
            child.validate_target_args('gate', args, self.out)

    def test_verified_target_exec_uses_bytes_not_second_read(self):
        with patch('builtins.open', side_effect=AssertionError('second read')), \
             patch('os.open', side_effect=AssertionError('second read')):
            child.execute_target(b'assert __name__ == "__main__"\nassert __file__ == "/missing/fixture.py"\n', Path('/missing/fixture.py'))

    def test_cli_requires_separator_and_does_not_autodiscover_manifest(self):
        base = ['--enable', '--target', 'render', '--manifest', str(self.manifest), '--manifest-sha256', self.manifest_sha,
                '--receipt', str(self.receipt), '--output-dir', str(self.out)]
        with patch.object(child, 'launch', return_value=7) as run:
            self.assertEqual(child.main(base + ['--', *self.args]), 7)
            self.assertEqual(run.call_args.kwargs['target_args'], self.args)
            with self.assertRaises(child.RoutingRejected): child.main(base)

    def test_interrupt_is_recorded_and_not_swallowed(self):
        with self.assertRaises(KeyboardInterrupt): self.run_target(self.normal_source + 'raise KeyboardInterrupt()\n')
        self.assertEqual((self.record()['status'], self.record()['returncode']), ('failed', 130))


class SpawnContract(unittest.TestCase):
    def test_real_spawn_importable_worker_runs_before_deferred_candidate_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'scripts').mkdir(); (root / 'scripts/__init__.py').write_text('')
            (root / 'worker_fixture.py').write_text('import sys\ndef main(q):\n    q.put(not any(x in sys.modules for x in ("torch","tensorrt","scripts.vae_fast_decoder")))\n')
            # The launcher remains sys.modules[__main__]; its top-level is inert.
            bootstrap = root / 'bootstrap.py'
            bootstrap.write_text(textwrap.dedent(f'''
                import sys
                from pathlib import Path
                from types import SimpleNamespace
                sys.path.insert(0, {str(Path(child.__file__).parent)!r})
                import taesd_fp32_candidate_child as child
                if __name__ == '__main__':
                    raw = b"import multiprocessing as mp\\nfrom worker_fixture import main\\nctx=mp.get_context('spawn')\\nq=ctx.Queue()\\np=ctx.Process(target=main,args=(q,))\\np.start()\\nassert q.get(timeout=10) is True\\np.join(timeout=10)\\nassert p.exitcode == 0\\nassert 'torch' not in __import__('sys').modules\\nfrom scripts import vae_fast_decoder as vfd\\nassert vfd.fixture_loaded is True\\n"
                    vfd = b"fixture_loaded=True\\ndef load_taesd_trt_backend(*args, **kwargs): pass\\n"
                    with child.scoped_candidate(SimpleNamespace(), Path('/manifest'), 'a'*64, {{}}, vfd, {{}}):
                        assert 'scripts.vae_fast_decoder' not in sys.modules
                        child.execute_target(raw, Path('/verified/target/chin_multistream_render.py'))
                    assert 'scripts.vae_fast_decoder' not in sys.modules
            '''))
            result = subprocess.run([sys.executable, '-B', str(bootstrap)], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
