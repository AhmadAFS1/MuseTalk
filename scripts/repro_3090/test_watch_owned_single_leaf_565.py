"""CPU contracts only; no real CUDA/NVML/ownership claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('watch565', HERE / 'watch_owned_single_leaf_565.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
DOCUMENT = {
    'instance_id': '54976782', 'label': 'musetalk-r5-3090-dev-20261009-a5',
    'worker_alias': 'musetalk-3090-build-54976782', 'worker_hostname': 'owned-host',
    'gpu_uuid': 'GPU-01234567-0123-0123-0123-0123456789ab',
    'ledger': '/home/ec2-user/.local/state/musetalk-r5-3090-dev-20261009-a5/startup-ledger.json',
    'deadline_utc': '2026-10-09T08:35:00+00:00',
}


class DriverBinding(unittest.TestCase):
    def test_exact_original_pins(self):
        for name, digest in [('watch_owned_single_leaf.py', w.WATCH_SHA),
                             ('watch_owned_single_leaf_target.py', w.BINDING_SHA)]:
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest)
        builder = HERE.parent / 'build_unet_stagewise.py'
        self.assertEqual(hashlib.sha256(builder.read_bytes()).hexdigest(), w.BUILDER_SHA)
        self.assertEqual(hashlib.sha256((HERE / 'build_owned_unet_opt3.py').read_bytes()).hexdigest(), w.OPT3_CHILD_SHA)

    def test_only_one_literal_changes(self):
        raw = (HERE / 'watch_owned_single_leaf.py').read_bytes()
        changed = w.adapted_source(raw)
        self.assertEqual(changed.count(w.NEW_COMPARISON), 1)
        self.assertEqual(changed.replace(w.NEW_COMPARISON, w.OLD_COMPARISON), raw)

    def test_changed_core_rejected(self):
        raw = (HERE / 'watch_owned_single_leaf.py').read_bytes()
        with self.assertRaises(ValueError):
            w.adapted_source(raw + b'\n')

    def test_source_digest_required(self):
        with self.assertRaises(ValueError):
            w.source(HERE / 'watch_owned_single_leaf.py', '0' * 64)

    def test_relative_source_rejected(self):
        with self.assertRaises(ValueError):
            w.source(Path('watch_owned_single_leaf.py'), w.WATCH_SHA)

    def test_core_is_stdlib_only_until_execution(self):
        with patch.dict(sys.modules, {'torch': None}):
            core = w.reviewed_watch()
        self.assertEqual(core.__file__, str(HERE / 'watch_owned_single_leaf_565.py'))
        self.assertTrue(callable(core.leaf))

    def test_admission_preserves_historical_pins(self):
        core = w.reviewed_watch()
        before = dict(core.CANONICAL_TARGETS)
        names = set(core.TARGET_NAMES)
        w.configure(core, DOCUMENT)
        self.assertEqual({k: core.CANONICAL_TARGETS[k] for k in before}, before)
        self.assertEqual(core.TARGET_NAMES, names)
        self.assertEqual(core.CANONICAL_TARGETS['build_unet_stagewise.py'], w.BUILDER_SHA)
        self.assertEqual((core.HOST, core.UUID, core.DEADLINE),
                         ('owned-host', DOCUMENT['gpu_uuid'], '2026-10-09T08:35:00Z'))

    def test_identity_requires_host_and_actual_driver(self):
        good = 'NVIDIA GeForce RTX 3090, ' + DOCUMENT['gpu_uuid'] + ', 8.6, 565.77\n'
        with patch.object(w.socket, 'gethostname', return_value='owned-host'), \
                patch.object(w.subprocess, 'check_output', return_value=good):
            w.identity(DOCUMENT)
        for bad in [good.replace('565.77', '595.91.07'), good.replace('3090', '4070'),
                    good + good]:
            with patch.object(w.socket, 'gethostname', return_value='owned-host'), \
                    patch.object(w.subprocess, 'check_output', return_value=bad):
                with self.assertRaises(ValueError):
                    w.identity(DOCUMENT)
        with patch.object(w.socket, 'gethostname', return_value='other-host'):
            with self.assertRaises(ValueError):
                w.identity(DOCUMENT)

    def test_leaf_propagates_binding(self):
        calls = []
        core = types.SimpleNamespace(CANONICAL_TARGETS={}, leaf=lambda s: calls.append(s))
        spec = {'uuid': DOCUMENT['gpu_uuid'], 'parent_pid': 123}
        with patch.object(w, 'identity'), patch.object(w, 'reviewed_watch', return_value=core):
            w.leaf_with_binding(spec, DOCUMENT)
        self.assertEqual(calls, [spec])
        self.assertEqual(core.UUID, DOCUMENT['gpu_uuid'])
        with patch.object(w, 'identity'):
            with self.assertRaises(ValueError):
                w.leaf_with_binding({'uuid': 'other'}, DOCUMENT)

    def test_child_bootstrap_hash_and_embedded_binding(self):
        raw = b'def leaf_with_binding(s,d):\n assert s["uuid"] == d["gpu_uuid"]\n assert d["instance_id"] == "54976782"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checked.py'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            with patch.object(sys, 'argv', ['child', str(path), digest,
                                          json.dumps({'uuid': DOCUMENT['gpu_uuid']})]):
                exec(w.bootstrap(DOCUMENT), {})
            with patch.object(sys, 'argv', ['child', str(path), '0' * 64, '{}']):
                with self.assertRaises(AssertionError):
                    exec(w.bootstrap(DOCUMENT), {})

    def test_explicit_optin_required(self):
        with self.assertRaises(ValueError):
            w.main(['--owned-target-json', '/not-read', '--owned-target-sha256', '0' * 64, '--'])


if __name__ == '__main__':
    unittest.main()
