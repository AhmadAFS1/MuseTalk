"""Synthetic CPU tests; never invoke nvidia-smi or a remote host."""
import copy
import contextlib
import io
import json
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_power_comparison as power


def state(watts=300, temperature=60):
    return dict(zip(power.FIELDS.split(','), [power.GPU, 'NVIDIA GeForce RTX 3090', watts,
                                             100, 350, 350, temperature, 0, 10]))


class Fake:
    def __init__(self):
        self.row = state()
        self.sets = []
        self.started = self.stopped = False
        self.setter_failed = False
        self.restore_wrong = False
        self.hot = False
        self.child_rc = 0
    def preflight(self):
        return copy.deepcopy(self.row)
    def state(self):
        result = copy.deepcopy(self.row)
        if self.hot and self.started:
            result['temperature.gpu'] = 88
        return result
    def set_power(self, value):
        self.sets.append(value)
        self.row['power.limit'] = 350 if value == 300 and self.restore_wrong else value
        return 1 if value == 350 and self.setter_failed else 0
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True
    def poll(self):
        return self.child_rc


class PowerTests(unittest.TestCase):
    def run_fake(self, backend, **kwargs):
        receipt = {}
        result = power.experiment(backend, receipt, lambda r: None,
                                  wall=lambda: power.DEADLINE - 2000, **kwargs)
        return result, receipt

    def test_fixed_command_and_no_arbitrary_cli(self):
        argv = power.command('a2_power350_unit')
        self.assertEqual(argv[argv.index('--stages') + 1], 'T')
        self.assertEqual(argv[argv.index('--loops') + 1], '24')
        self.assertEqual(argv[argv.index('--taesd-key') + 1], power.TAESD_KEY)
        self.assertEqual(argv[argv.index('--tracking-parity-report') + 1], str(power.PAIR))
        self.assertIn('--tracking-overlap', argv)
        for label in ('foreign', 'a2_power350_../bad', 'a2_power350_hello;ls'):
            with self.assertRaises(power.Refused):
                power.command(label)

    def test_success_restores_and_does_not_promote_quality(self):
        backend = Fake()
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 0)
        self.assertEqual(backend.sets, [350, 300])
        self.assertTrue(backend.stopped)
        self.assertTrue(receipt['restore']['verified'])
        self.assertFalse(receipt['release_ready'])
        self.assertFalse(receipt['sust_authorized'])
        self.assertEqual(receipt['quality_status'], 'REJECTED_UNCHANGED')

    def test_ambiguous_failed_setter_still_restores(self):
        backend = Fake()
        backend.setter_failed = True
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 2)
        self.assertEqual(backend.sets, [350, 300])
        self.assertFalse(backend.started)
        self.assertTrue(receipt['restore']['verified'])

    def test_thermal_abort_stops_and_restores(self):
        backend = Fake()
        backend.hot = True
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 2)
        self.assertEqual(receipt['failure'], 'thermal_limit')
        self.assertTrue(backend.stopped)
        self.assertTrue(receipt['restore']['verified'])

    def test_wrong_identity_never_sets_power(self):
        backend = Fake()
        backend.row['uuid'] = 'foreign'
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 2)
        self.assertEqual(backend.sets, [])

    def test_restore_requires_actual_readback(self):
        backend = Fake()
        backend.restore_wrong = True
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 2)
        self.assertFalse(receipt['restore']['verified'])
        self.assertEqual(receipt['canonical_returncode'], 0)

    def test_canonical_nonzero_preserved(self):
        backend = Fake()
        backend.child_rc = 1
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 1)
        self.assertEqual(receipt['canonical_returncode'], 1)
        self.assertTrue(receipt['restore']['verified'])

    def test_signal_before_start_never_sets(self):
        backend = Fake()
        rc, receipt = self.run_fake(backend, interrupted=lambda: True)
        self.assertEqual(rc, 2)
        self.assertEqual(backend.sets, [])

    def test_signal_after_set_restores(self):
        backend = Fake()
        rc, receipt = self.run_fake(backend, interrupted=lambda: bool(backend.sets))
        self.assertEqual(rc, 2)
        self.assertEqual(backend.sets, [350, 300])
        self.assertTrue(receipt['restore']['verified'])

    def test_runtime_abort_restores(self):
        backend = Fake()
        clock = [0]
        def mono():
            clock[0] += 100
            return clock[0]
        rc, receipt = self.run_fake(backend, mono=mono)
        self.assertEqual(rc, 2)
        self.assertEqual(backend.sets, [350, 300])

    def test_receipt_failure_cannot_skip_restore(self):
        backend = Fake()
        def save(data):
            if backend.sets:
                raise OSError('disk full')
        with self.assertRaises(OSError):
            power.experiment(backend, {}, save, wall=lambda: power.DEADLINE - 2000)
        self.assertEqual(backend.sets, [350, 300])

    def test_owned_descendant_selection_excludes_cotenants(self):
        snap = {1: (0, 1, 1), 2: (1, 2, 20), 3: (2, 3, 30), 4: (1, 4, 40), 5: (3, 5, 50)}
        self.assertEqual(power.descendants(snap, 2), {3, 5})

    def test_every_nvml_call_bounded_and_ordinary(self):
        backend = power.Backend('a2_power350_unit', Path('/unused'))
        with patch.object(power.safe_capture, 'capture', return_value='') as run:
            backend.set_power(350)
            self.assertEqual(run.call_args.args[0], ['nvidia-smi', '-i', power.GPU, '--power-limit=350'])
            self.assertEqual(run.call_args.kwargs['timeout_s'], 1)
            self.assertEqual(run.call_args.kwargs['terminate_grace_s'], 0)
            self.assertEqual(run.call_args.kwargs['reap_grace_s'], .2)
            self.assertEqual(run.call_args.kwargs['output_limit_bytes'], 8192)
            self.assertEqual(run.call_args.kwargs['stage'], 'unspecified')

    def test_hung_capture_fails_closed_with_safe_record_only(self):
        backend = power.Backend('a2_power350_unit', Path('/unused'))
        for failure, cleanup in [('TIMEOUT', 'unconfirmed_kernel_or_child_state'),
                                 ('NONZERO_EXIT', 'unconfirmed_kernel_or_child_state')]:
            record = {'stage': 'unspecified', 'failure': failure, 'cleanup': cleanup,
                      'returncode': None, 'raw_output_persisted': False}
            with patch.object(power.safe_capture, 'capture', side_effect=power.safe_capture.CaptureFailure(record)):
                with self.assertRaises(power.Refused) as raised:
                    backend.nvml('--query-gpu=uuid')
            self.assertEqual(raised.exception.safe_record, record)
            self.assertEqual(str(raised.exception), 'nvml_capture_failed')

    def test_clean_nonzero_capture_returns_only_exit_code(self):
        backend = power.Backend('a2_power350_unit', Path('/unused'))
        record = {'stage': 'unspecified', 'failure': 'NONZERO_EXIT', 'cleanup': 'leader_reaped',
                  'returncode': 3, 'raw_output_persisted': False}
        with patch.object(power.safe_capture, 'capture', side_effect=power.safe_capture.CaptureFailure(record)):
            self.assertEqual(backend.nvml('--power-limit=350'), (3, ''))

    def test_hung_setter_keeps_safe_failure_and_attempts_restore(self):
        backend = Fake()
        record = {'stage': 'unspecified', 'failure': 'TIMEOUT', 'cleanup': 'unconfirmed_kernel_or_child_state'}
        ordinary_set = backend.set_power
        def setter(value):
            result = ordinary_set(value)
            if value == 350:
                raise power.Refused('nvml_capture_failed', record)
            return result
        backend.set_power = setter
        rc, receipt = self.run_fake(backend)
        self.assertEqual(rc, 2)
        self.assertEqual(backend.sets, [350, 300])
        self.assertEqual(receipt['safe_capture_failure'], record)
        self.assertFalse(backend.started)

    def test_power_handler_includes_hangup(self):
        import inspect
        source = inspect.getsource(power.main)
        self.assertIn('(signal.SIGINT, signal.SIGTERM, signal.SIGHUP)', source)

    def test_bridge_loader_selection_preserved_lease_overrides_removed(self):
        with patch.dict(power.os.environ, {'LD_LIBRARY_PATH': '/approved/libs', 'CUDA_VISIBLE_DEVICES': '0',
                                         'BOX_GUARD_IGNORE_PAUSE': '1', 'PYTHONPATH': '/untrusted',
                                         'LD_PRELOAD': '/untrusted.so'}, clear=True):
            backend = power.Backend('a2_power350_unit', Path('/unused'))
        self.assertEqual(backend.env['LD_LIBRARY_PATH'], '/approved/libs')
        self.assertEqual(backend.env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(backend.env['PYTHONNOUSERSITE'], '1')
        for key in ('BOX_GUARD_IGNORE_PAUSE', 'PYTHONPATH', 'LD_PRELOAD'):
            self.assertNotIn(key, backend.env)

    def test_plan_requires_direct_python_child_and_never_calls_hardware(self):
        output = io.StringIO()
        with patch.object(power.Backend, 'preflight') as preflight, contextlib.redirect_stdout(output):
            self.assertEqual(power.main(['--label', 'a2_power350_unit']), 0)
        preflight.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertTrue(plan['bridge_no_shell_intermediary'])
        self.assertEqual(plan['bridge_direct_child_argv'],
                         [power.PYTHON, str(power.ROOT / 'scripts/repro_3090/a2_power_comparison.py'),
                          '--execute', '--label', 'a2_power350_unit'])


if __name__ == '__main__':
    unittest.main()
