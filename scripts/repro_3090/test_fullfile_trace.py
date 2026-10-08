import unittest
import start_fullfile_traced_api as helper


class FullFileTraceTests(unittest.TestCase):
    def test_all_files_and_children_no_path_filter(self):
        argv = helper.trace_argv()
        self.assertIn('-f', argv)
        self.assertIn('trace=%file', argv)
        self.assertNotIn('-P', argv)
        self.assertNotIn('-DDD', argv)
        self.assertEqual(argv[-1], '--inside-trace')
        self.assertTrue(str(helper.TRACE).startswith('/workspace/MuseTalk/docs/fps_comparisons/rtx3090_r5_20261008/startup/'))


if __name__ == '__main__':
    unittest.main()
