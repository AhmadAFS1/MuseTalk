import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_owned_legacy_serial_render_a3 as adapter


class AdapterTests(unittest.TestCase):
    def test_only_reporting_added_and_values_unchanged(self):
        timing = {'frames': 5760, 'compose_ms': 42.5}
        clips = [{'raw_refined_sha256': 'same'}]
        stats = {'timing_ms': timing, 'clips': clips, 't_last_compose': 123.456}
        source = {2: ('repdone', 2, 0, stats)}
        out = adapter.serial_reporting(source)
        self.assertEqual(set(out[2][3]) - set(stats), set(adapter.REPORT_FIELDS))
        self.assertEqual(out[2][:3], source[2][:3])
        for key in stats:
            self.assertIs(out[2][3][key], stats[key])
        self.assertEqual(stats, source[2][3])
        self.assertNotIn('tracking_overlap', stats)
        self.assertFalse(out[2][3]['tracking_overlap'])

    def test_conflicting_fields_rejected(self):
        for field, value in (('tracking_overlap', True), ('tracking_overlap', False), ('timing_semantics', 'anything')):
            with self.assertRaises(ValueError):
                adapter.serial_reporting({0: ('repdone', 0, 0, {field: value})})

    def test_bad_messages_rejected(self):
        for value in ([], {0: ('repdone', 1, 0, {})}, {0: ('hello', 0, 0, {})},
                      {0: ('repdone', 0, 0, [])}, {False: ('repdone', False, 0, {})}):
            with self.assertRaises(ValueError):
                adapter.serial_reporting(value)

    def test_wait_arguments_and_non_repdone_untouched(self):
        calls, original_messages = [], {0: ('hello', 0, {'same': True})}
        class Collector:
            def wait_for(self, kind, streams, timeout=600):
                calls.append((kind, streams, timeout))
                return original_messages
        adapter.adapt_collector(Collector)
        streams = range(6)
        result = Collector().wait_for('hello', streams, timeout=321)
        self.assertIs(result, original_messages)
        self.assertEqual(calls, [('hello', streams, 321)])

    def test_repdone_adapter_preserves_wait(self):
        class Collector:
            def wait_for(self, kind, streams, timeout=600):
                return {0: ('repdone', 0, 1, {'frames': 5760})}
        adapter.adapt_collector(Collector)
        self.assertEqual(Collector().wait_for('repdone', [0])[0][3]['frames'], 5760)

    def test_canonical_contract_no_overlap_or_capture(self):
        for stage, repeats in (('T', '2'), ('SUST', '5')):
            argv = adapter.renderer_args(stage, Path('/test'), 'label')
            for flag, expected in (('--streams', '6'), ('--loops', '24' if stage == 'T' else '20'), ('--repeats', repeats),
                                   ('--min-timed-s', '60'), ('--thermal-warmup-s', '120')):
                self.assertEqual(argv[argv.index(flag) + 1], expected)
            for flag in ('--tracking-overlap', '--encode', '--save-arrays', '--compare-accepted'):
                self.assertNotIn(flag, argv)
        with self.assertRaises(ValueError):
            adapter.renderer_args('Q', Path('/test'), 'label')

    def test_default_off_before_machine_or_gpu_access(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            adapter.main(['--stage', 'T', '--output-dir', '/test', '--label', 'label'])

    def test_helper_directory_resolved_from_inert_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / 'scripts/repro_3090'
            here.mkdir(parents=True)
            # Fake checked_source, not a renderer/GPU-quality test. The new
            # subprocess has no inherited test-runner/helper sys.path entry.
            (here / 'watch_owned_single_leaf.py').write_text(
                "def checked_source(path, digest):\n"
                " return b'class Collector:\\n def wait_for(self,*a,**k): return {}\\n'\n")
            script = ('import importlib.util, pathlib, sys; '
                      's=importlib.util.spec_from_file_location("adapter",sys.argv[1]); '
                      'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                      'n=m.load_renderer(pathlib.Path(sys.argv[2])); '
                      'assert n["__name__"]!="__main__"; assert "torch" not in sys.modules')
            subprocess.run([sys.executable, '-B', '-c', script, str(Path(adapter.__file__).resolve()), str(root)],
                           cwd=root, check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
