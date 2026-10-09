"""CPU-only routing and failure tests, not inference/quality/FPS evidence."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FILE = Path(__file__).with_name('taesd_fp32_candidate_quality.py')
SPEC = importlib.util.spec_from_file_location('candidate_quality_under_test', FILE)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


class QualityRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.api = quality.runtime_api()
        fp = {'recipe': self.api.RECIPE, 'precision': 'fp16_with_final_conv_fp32'}
        self.identity = {'key': self.api.key_for(fp), 'decoder_plan_sha256': 'd' * 64,
                         'post_plan_sha256': 'e' * 64}
        self.meta = {'schema': self.api.SCHEMA, 'fingerprint': fp, **self.identity,
                     'status': 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED',
                     **{k: False for k in ('quality_accepted', 'full_gate_captured',
                        'performance_measured', 'default_selection_changed', 'release_ready')}}
        self.manifest = self.root / f"taesd_trt_{self.identity['key']}.json"
        self.raw = self.api.json_bytes(self.meta)
        self.manifest.write_bytes(self.raw)
        self.digest = self.api.sha(self.raw)
        self.args = SimpleNamespace(suite='quality', python='/explicit/python',
                                    taesd_key=self.identity['key'], taesd_dir=str(self.root))
        self.launcher = self.root / 'taesd_fp32_candidate_child.py'
        self.launcher.write_text('# CPU synthetic launcher fixture\n')
        self.child_digest = self.api.sha(self.launcher.read_bytes())

    def receipt(self, target='gate', rc=1):
        return {'schema': 'taesd_fp32_candidate_child_v1', 'target': target,
                'manifest_sha256': self.digest, 'candidate_key': self.identity['key'],
                'decoder_plan_sha256': self.identity['decoder_plan_sha256'],
                'post_plan_sha256': self.identity['post_plan_sha256'],
                'candidate_calls': 1, 'candidate_attempts': 1, 'returncode': rc,
                'status': 'complete' if rc == 0 else 'failed',
                'quality_accepted': False, 'performance_accepted': False, 'release_ready': False}

    def test_import_has_no_heavy_dependencies(self):
        source = FILE.read_text()
        self.assertNotIn('import torch', source)
        self.assertNotIn('import tensorrt', source)
        self.assertNotIn('subprocess.run(', source)

    def test_selection_exact_digest(self):
        self.assertEqual(quality.selection(self.api, self.manifest, self.digest), self.identity)

    def test_selection_rejects_hash_tampering(self):
        with self.assertRaises(ValueError):
            quality.selection(self.api, self.manifest, '0' * 64)

    def test_selection_rejects_claims_recipe_and_key(self):
        for mutate in (lambda m: m.update(quality_accepted=True),
                       lambda m: m.update(key='wrong'),
                       lambda m: m['fingerprint'].update(recipe='plain_fp16')):
            m = json.loads(self.raw)
            mutate(m)
            raw = self.api.json_bytes(m)
            self.manifest.write_bytes(raw)
            with self.assertRaises(ValueError):
                quality.selection(self.api, self.manifest, self.api.sha(raw))

    def test_valid_strict_failure_receipt_is_not_rewritten(self):
        path = self.root / 'receipt.json'
        path.write_bytes(self.api.json_bytes(self.receipt()))
        record = quality.validate_receipt(self.api, path, self.digest, self.identity, 'gate', 1)
        self.assertEqual(record['returncode'], 1)

    def test_rejects_invalid_missing_fallback_and_misbound_receipts(self):
        for changes in ({'candidate_calls': 0}, {'candidate_attempts': 2},
                        {'candidate_calls': True}, {'candidate_key': 'wrong'},
                        {'decoder_plan_sha256': '0' * 64}, {'manifest_sha256': '0' * 64},
                        {'returncode': 0}, {'status': 'invalid'}, {'release_ready': True}):
            with self.subTest(changes=changes):
                path = self.root / 'receipt.json'
                path.write_bytes(self.api.json_bytes({**self.receipt(), **changes}))
                with self.assertRaises(ValueError):
                    quality.validate_receipt(self.api, path, self.digest, self.identity, 'gate', 1)
        with self.assertRaises(ValueError):
            quality.validate_receipt(self.api, self.root / 'missing', self.digest, self.identity, 'gate', 1)

    def test_render_failure_cannot_be_an_original_gate_exception(self):
        path = self.root / 'receipt.json'
        path.write_bytes(self.api.json_bytes(self.receipt('render', 1)))
        with self.assertRaises(ValueError):
            quality.validate_receipt(self.api, path, self.digest, self.identity, 'render', 1)

    def test_preserves_original_guard_and_unrelated_children(self):
        calls, records = [], []
        def original(args, out, env, label, command, gb):
            calls.append((env, label, list(command), gb))
            if label == 'taesd':
                path = self.root / 'taesd.candidate_child.json'
                path.write_bytes(self.api.json_bytes(self.receipt()))
                return 1
            return 0
        runner = SimpleNamespace(child=original, checks=SimpleNamespace(Invalid=ValueError))
        env = {'UNCHANGED': 'value'}
        with quality.route_children(runner, self.api, self.manifest, self.digest,
                                    self.identity, self.launcher, self.child_digest, records):
            command = [self.args.python, 'scripts/validate_unet_backend.py', '--limit', '0']
            self.assertEqual(runner.child(self.args, self.root, env, 'unet_main', command, 8), 0)
            self.assertEqual(calls[-1], (env, 'unet_main', command, 8))
            rc = runner.child(self.args, self.root, env, 'taesd',
                              [self.args.python, 'scripts/repro_400fps/gate_taesd_trt.py',
                               '--no-record', '--corpus', '/explicit/corpus'], 8)
            self.assertEqual(rc, 1)
            self.assertEqual(calls[-1][0], env)
            self.assertEqual(calls[-1][3], 8)
            wrapped = calls[-1][2]
            self.assertEqual(wrapped[:3], [self.args.python, '-B', '-c'])
            self.assertEqual(wrapped[4:8], [str(self.launcher), self.child_digest, '--enable', '--target'])
            self.assertEqual(wrapped[wrapped.index('--') + 1:], ['--no-record', '--corpus', '/explicit/corpus'])
        self.assertIs(runner.child, original)
        self.assertEqual([r['target'] for r in records], ['gate'])

    def test_routing_restores_on_child_exception(self):
        def original(*a):
            raise RuntimeError('synthetic child failure')
        runner = SimpleNamespace(child=original)
        with self.assertRaises(RuntimeError):
            with quality.route_children(runner, self.api, self.manifest, self.digest,
                                        self.identity, self.launcher, self.child_digest, []):
                runner.child(self.args, self.root, {}, 'unet_main', ['/explicit/python', 'unet.py'], 8)
        self.assertIs(runner.child, original)

    def test_wrong_label_or_runner_identity_rejected_before_child(self):
        original = unittest.mock.Mock()
        runner = SimpleNamespace(child=original)
        for field, value in (('suite', 'aggregate'), ('taesd_key', 'wrong'), ('taesd_dir', '/wrong')):
            args = SimpleNamespace(**vars(self.args))
            setattr(args, field, value)
            with quality.route_children(runner, self.api, self.manifest, self.digest,
                                        self.identity, self.launcher, self.child_digest, []):
                with self.assertRaises(ValueError):
                    runner.child(args, self.root, {}, 'taesd',
                                 [args.python, 'scripts/repro_400fps/gate_taesd_trt.py'], 8)
        original.assert_not_called()

    def test_pinned_runner_restores_process_state_even_on_exception(self):
        before = {n: sys.modules.get(n) for n in quality.PINS}
        path, argv, env, cwd = sys.path[:], sys.argv[:], os.environ.copy(), os.getcwd()
        with self.assertRaises(RuntimeError):
            with quality.pinned_runner(self.api) as runner:
                self.assertTrue(callable(runner.main))
                sys.argv[:] = ['changed']
                os.environ['SYNTHETIC_CANDIDATE_TEST'] = 'changed'
                os.chdir(self.root)
                raise RuntimeError('synthetic scoped failure')
        self.assertEqual({n: sys.modules.get(n) for n in quality.PINS}, before)
        self.assertEqual((sys.path[:], sys.argv[:], os.environ.copy(), os.getcwd()), (path, argv, env, cwd))

    def test_pinned_runner_rejects_code_change(self):
        with patch.dict(quality.PINS, {'runner': '0' * 64}):
            with self.assertRaises(ValueError):
                with quality.pinned_runner(self.api):
                    self.fail('changed code executed')

    def test_execute_requires_optin_before_import(self):
        with patch.object(quality, 'runtime_api') as imported:
            with self.assertRaises(ValueError):
                quality.execute(manifest=self.manifest, manifest_sha256=self.digest,
                                child_sha256=self.child_digest, proof=self.root / 'proof',
                                runner_argv=['quality'])
        imported.assert_not_called()

    def test_only_quality_and_exact_gpu_allowed(self):
        with patch.object(quality, 'HERE', self.root), patch.object(quality, 'runtime_api', return_value=self.api):
            for args in (['aggregate'], ['quality', '--general-gpu'], ['quality', '--general-g'],
                         ['quality', '--gen'], ['quality', '--g'], ['quality', '--pro'],
                         ['quality', '--targetnative']):
                with self.assertRaises(ValueError):
                    quality.execute(manifest=self.manifest, manifest_sha256=self.digest,
                                    child_sha256=self.child_digest, proof=self.root / 'proof',
                                    runner_argv=args, enable=True)

    def synthetic_runner(self, *, missing_render=False, final_rc=1):
        def child(args, out, env, label, command, gb):
            target = 'gate' if label == 'taesd' else 'render'
            rc = 1 if target == 'gate' else 0
            path = self.root / (label + '.candidate_child.json')
            path.write_bytes(self.api.json_bytes(self.receipt(target, rc)))
            return rc
        runner = SimpleNamespace(child=child, checks=SimpleNamespace(Invalid=ValueError))
        def main():
            runner.child(self.args, self.root, {}, 'taesd',
                         [self.args.python, 'scripts/repro_400fps/gate_taesd_trt.py',
                          '--no-record', '--corpus', '/corpus'], 8)
            if not missing_render:
                runner.child(self.args, self.root, {}, 'quality_capture',
                             [self.args.python, 'scripts/chin_multistream_render.py'], 12)
            return final_rc
        runner.main = main
        return runner

    @contextmanager
    def synthetic_pinned(self, runner):
        saved = sys.argv[:]
        try:
            yield runner
        finally:
            sys.argv[:] = saved

    def test_execute_preserves_failure_and_records_both_real_routes(self):
        runner = self.synthetic_runner()
        original, argv = runner.child, sys.argv[:]
        proof = self.root / 'proof.json'
        with patch.object(quality, 'HERE', self.root), patch.object(quality, 'runtime_api', return_value=self.api), \
                patch.object(quality, 'pinned_runner', return_value=self.synthetic_pinned(runner)):
            rc = quality.execute(manifest=self.manifest, manifest_sha256=self.digest,
                                 child_sha256=self.child_digest, proof=proof,
                                 runner_argv=['quality'], enable=True)
        self.assertEqual(rc, 1)
        self.assertIs(runner.child, original)
        self.assertEqual(sys.argv, argv)
        data = json.loads(proof.read_bytes())
        self.assertEqual(data['status'], 'complete_original_gates_reported')
        self.assertEqual([r['target'] for r in data['children']], ['gate', 'render'])
        self.assertEqual([r['returncode'] for r in data['children']], [1, 0])
        self.assertFalse(data['quality_accepted'])
        self.assertFalse(data['release_ready'])
        self.assertEqual(proof.stat().st_mode & 0o777, 0o600)

    def test_execute_cannot_complete_with_only_gate(self):
        runner = self.synthetic_runner(missing_render=True)
        proof = self.root / 'proof.json'
        with patch.object(quality, 'HERE', self.root), patch.object(quality, 'runtime_api', return_value=self.api), \
                patch.object(quality, 'pinned_runner', return_value=self.synthetic_pinned(runner)):
            with self.assertRaisesRegex(ValueError, 'coverage_missing'):
                quality.execute(manifest=self.manifest, manifest_sha256=self.digest,
                                child_sha256=self.child_digest, proof=proof,
                                runner_argv=['quality'], enable=True)
        data = json.loads(proof.read_bytes())
        self.assertEqual(data['status'], 'invalid')
        self.assertEqual(data['returncode'], 2)

    def test_execute_refuses_existing_proof_and_unpinned_launcher(self):
        proof = self.root / 'proof.json'
        with patch.object(quality, 'HERE', self.root), patch.object(quality, 'runtime_api', return_value=self.api), \
                patch.object(quality, 'pinned_runner') as entered:
            with self.assertRaises(ValueError):
                quality.execute(manifest=self.manifest, manifest_sha256=self.digest,
                                child_sha256='0' * 64, proof=proof,
                                runner_argv=['quality'], enable=True)
            proof.write_text('existing evidence')
            with self.assertRaises(ValueError):
                quality.execute(manifest=self.manifest, manifest_sha256=self.digest,
                                child_sha256=self.child_digest, proof=proof,
                                runner_argv=['quality'], enable=True)
            self.assertEqual(proof.read_text(), 'existing evidence')
        entered.assert_not_called()

    def test_execution_time_bootstrap_uses_checked_bytes_and_rejects_change(self):
        self.launcher.write_text('raise SystemExit(7)\n')
        digest = self.api.sha(self.launcher.read_bytes())
        command = [sys.executable, '-B', '-c', quality.CHILD_BOOTSTRAP, str(self.launcher), digest]
        result = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.launcher.write_text('raise SystemExit(99)\n')
        changed = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(changed.returncode, 2, changed.stderr)

    def test_checked_byte_bootstrap_real_spawn_has_no_launcher_reimport(self):
        worker = self.root / 'bootstrap_worker.py'
        worker.write_text('import sys\ndef main(q):\n    q.put(not any(x in sys.modules for x in ("torch","tensorrt","scripts.vae_fast_decoder")))\n')
        # A second import of this source as __mp_main__ must fail; a real spawn
        # succeeds because its importable worker does not need this namespace.
        source = ('import sys\n'
                  'assert __name__ == "__main__"\n'
                  'assert not hasattr(sys.modules["__main__"], "__file__")\n'
                  f'sys.path.insert(0, {str(self.root)!r})\n'
                  'import multiprocessing as mp\nfrom bootstrap_worker import main\n'
                  'ctx=mp.get_context("spawn")\nq=ctx.Queue()\n'
                  'p=ctx.Process(target=main,args=(q,))\np.start()\n'
                  'assert q.get(timeout=10) is True\np.join(timeout=10)\nassert p.exitcode == 0\n')
        self.launcher.write_text(source)
        digest = self.api.sha(self.launcher.read_bytes())
        result = subprocess.run([sys.executable, '-B', '-c', quality.CHILD_BOOTSTRAP,
                                 str(self.launcher), digest], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
