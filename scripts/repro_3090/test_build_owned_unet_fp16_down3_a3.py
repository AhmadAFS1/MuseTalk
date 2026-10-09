import contextlib
import io
import unittest
import build_owned_unet_fp16_down3_a3 as builder


class Tests(unittest.TestCase):
    def test_fixed_matrix_preserves_fp16_only_build_options(self):
        for blocks in (('mid',), ('up0',), ('mid', 'up0')):
            a = builder.builder_options(blocks)
            self.assertEqual(a.blocks, ','.join(blocks))
            self.assertEqual(a.int8_blocks, '')
            self.assertEqual(a.int8_recipe, '')
            self.assertEqual(a.hardware_compat, 'none')
            self.assertEqual(a.max_minutes, 5 * len(blocks))
            self.assertEqual(a.opt_level, 5)
            self.assertEqual(a.workspace_gb, 2.0)
            self.assertTrue(a.no_timing_cache)
        with self.assertRaises(ValueError):
            builder.builder_options(('down1',))

    def test_selected_plans_only_removed_and_original_unchanged(self):
        native = {'blocks': {b: {'bytes': b} for b in ('mid', 'up0', 'down1')}, 'probe': {'old': True}}
        for blocks in (('mid',), ('up0',), ('mid', 'up0')):
            candidate = builder.candidate_manifest(native, blocks)
            self.assertEqual(set(candidate['blocks']), set(native['blocks']) - set(blocks))
            self.assertEqual(set(candidate['precision_restorations']), set(blocks))
        self.assertEqual(set(native['blocks']), {'mid', 'up0', 'down1'})

    def test_nonlegacy_variant_requires_exact_allocation_before_io(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            builder.main(['--execute', '--blocks', 'mid'])

    def test_default_off(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            builder.main([])

    def test_only_one_fp16_block_selected(self):
        args = builder.builder_options()
        self.assertEqual(args.blocks, 'down3')
        self.assertEqual(args.int8_blocks, '')
        self.assertEqual(args.int8_recipe, '')
        self.assertEqual(args.hardware_compat, 'none')
        self.assertTrue(args.no_timing_cache)
        self.assertTrue(args.strict_timing_cache)
        self.assertEqual(args.max_minutes, 5)

    def test_prior_probe_removed_and_source_not_mutated(self):
        native = {'blocks': {'down3': {'a': 1}, 'other': {'b': 2}}, 'probe': {'old': True},
                  'runtime': {}, 'build_log': ['old'], 'complete': True}
        result = builder.candidate_manifest(native)
        self.assertIn('down3', native['blocks'])
        self.assertIn('probe', native)
        self.assertNotIn('down3', result['blocks'])
        self.assertNotIn('probe', result)
        self.assertNotIn('runtime', result)
        self.assertEqual(result['blocks']['other'], native['blocks']['other'])
        self.assertFalse(result['quality_accepted'])
        self.assertFalse(result['performance_measured'])
        self.assertFalse(result['release_ready'])


if __name__ == '__main__':
    unittest.main()
