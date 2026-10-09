"""Synthetic CPU contracts; no engine export/build or actual quality claim."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name('build_owned_unet_opt3.py')
spec = importlib.util.spec_from_file_location('opt3', path)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def fixture():
    flags = {'builder_optimization_level': 5, 'precision': 'fp16', 'timing_cache': True}
    baseline = {key: 'fixed' for key in ('gpu', 'compute_capability', 'tensorrt_version',
        'torch_version', 'cuda_version', 'spec', 'unet_weights', 'unet_config', 'timestep')}
    baseline.update(batch=16, variant='srccache', build_flags=flags, complete=True,
                    int8_calibration={'files': ['main1', 'main2'], 'batches': 8})
    baseline['blocks'] = {name: {'onnx_sha256': name + '-hash', 'inputs': ['fixed'],
        'outputs': ['fixed'], 'engine_file': name + '.plan',
        'build_flags': {**flags, 'precision': 'int8_recipe' if name in b.STAGES[1] else 'fp16'}}
        for name in b.BLOCKS}
    candidate = copy.deepcopy(baseline)
    candidate['build_flags']['builder_optimization_level'] = 3
    for row in candidate['blocks'].values(): row['build_flags']['builder_optimization_level'] = 3
    candidate['probe'] = {'deterministic_run_to_run': True, 'graph_equals_direct_enqueue': True}
    return baseline, candidate


class Contracts(unittest.TestCase):
    def test_exact_registered_options(self):
        for stage in b.STAGES:
            o = b.options(stage)
            self.assertEqual((o.opt_level, o.workspace_gb, o.hardware_compat, o.variant,
                              o.strict_timing_cache, o.no_timing_cache, o.calib_batches),
                             (3, 2.0, 'none', 'srccache', True, False, 8))
            self.assertEqual(o.int8_recipe, str(b.RECIPE) if stage == b.STAGES[1] else '')
            self.assertFalse(o.force or o.second_build)
        with self.assertRaises(ValueError): b.options(('different',))

    def test_all_stages_and_final_pass(self):
        baseline, full = fixture(); wanted = set()
        for stage in b.STAGES:
            wanted.update(stage); got = copy.deepcopy(full)
            got['blocks'] = {k: v for k, v in got['blocks'].items() if k in wanted}
            b.assert_stage(got, baseline, wanted, final=stage == b.STAGES[-1])

    def test_changed_onnx_rejected(self):
        baseline, candidate = fixture()
        candidate['blocks']['prefix']['onnx_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'same-graph'): b.assert_stage(candidate, baseline, b.BLOCKS)

    def test_precision_regression_rejected(self):
        baseline, candidate = fixture()
        candidate['blocks']['down1']['build_flags']['precision'] = 'fp16'
        with self.assertRaisesRegex(ValueError, 'precision'): b.assert_stage(candidate, baseline, b.BLOCKS)

    def test_incomplete_zero_exit_not_accepted(self):
        baseline, candidate = fixture(); candidate['blocks'].pop('prefix')
        with self.assertRaisesRegex(ValueError, 'incomplete'): b.assert_stage(candidate, baseline, b.BLOCKS, True)
        baseline, candidate = fixture(); candidate['complete'] = False
        with self.assertRaisesRegex(ValueError, 'finalized'): b.assert_stage(candidate, baseline, b.BLOCKS, True)

    def test_calibration_reselection_rejected(self):
        baseline, candidate = fixture(); candidate['int8_calibration']['files'][0] = 'holdout'
        with self.assertRaisesRegex(ValueError, 'calibration'): b.assert_stage(candidate, baseline, b.BLOCKS)

    def test_runtime_and_global_policy_rejected(self):
        for key in ('gpu', 'spec', 'variant', 'tensorrt_version'):
            baseline, candidate = fixture(); candidate[key] = 'changed'
            with self.assertRaises(ValueError): b.assert_stage(candidate, baseline, b.BLOCKS)
        baseline, candidate = fixture(); candidate['build_flags']['timing_cache'] = False
        with self.assertRaisesRegex(ValueError, 'policy'): b.assert_stage(candidate, baseline, b.BLOCKS)

    def test_final_hard_invariants_rejected(self):
        for key in ('deterministic_run_to_run', 'graph_equals_direct_enqueue'):
            baseline, candidate = fixture(); candidate['probe'][key] = False
            with self.assertRaisesRegex(ValueError, 'invariant'): b.assert_stage(candidate, baseline, b.BLOCKS, True)

    def test_execution_default_off(self):
        with self.assertRaisesRegex(ValueError, 'explicit execution'):
            b.main(['--owned-target-json', '/not-read', '--owned-target-sha256', '0'*64,
                '--successor-inputs', '/not-read', '--successor-envelope', '/not-read',
                '--lineage-receipt', '/not-read', '--lineage-receipt-sha256', '0'*64, '--out', '/not-read'])


if __name__ == '__main__': unittest.main()
