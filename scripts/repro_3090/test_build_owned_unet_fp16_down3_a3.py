import contextlib
import io
import unittest
import build_owned_unet_fp16_down3_a3 as builder


class Tests(unittest.TestCase):
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
