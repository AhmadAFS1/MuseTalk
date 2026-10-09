"""CPU-only exact monitor selection; no SSH, GPU, models or quality evaluation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quality_launcher', HERE / 'launch_owned_a3_quality_checks.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class MonitorSelection(unittest.TestCase):
    def test_historical_monitor_unchanged(self):
        prefix, pins = m.owned_watch(HERE, Path('/owned.json'), 'a'*64, '595.91.07')
        self.assertEqual(Path(prefix[0]).name, 'watch_owned_single_leaf_target.py')
        self.assertEqual(pins, {'watch_owned_single_leaf_target.py': m.BOUND_WATCH_SHA})

    def test_565_explicit_exact_source(self):
        prefix, pins = m.owned_watch(HERE, Path('/owned.json'), 'a'*64, '565.77')
        self.assertEqual(Path(prefix[0]).name, 'watch_owned_single_leaf_565.py')
        self.assertEqual(pins, {'watch_owned_single_leaf_565.py': m.WATCH565_SHA})

    def test_descriptor_and_driver_required(self):
        for doc, digest, driver in [(None, 'a'*64, '565.77'), ('/owned', None, '565.77'),
                                    ('/owned', 'a'*64, 'unknown')]:
            with self.assertRaises(ValueError): m.owned_watch(HERE, doc, digest, driver)

    def test_changed_monitor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            (here / 'watch_owned_single_leaf_565.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                m.owned_watch(here, '/owned', 'a'*64, '565.77')


if __name__ == '__main__': unittest.main()
