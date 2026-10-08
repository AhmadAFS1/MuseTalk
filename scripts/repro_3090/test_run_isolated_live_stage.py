"""CPU-only contracts for excluding tracing overhead from scored live runs."""
import tempfile
import unittest
from pathlib import Path

from run_isolated_live_stage import require_untraced_api


class TracerGuardTests(unittest.TestCase):
    def fixture(self, root, tid, tracer):
        task = root / '123' / 'task' / str(tid)
        task.mkdir(parents=True)
        (task / 'status').write_text(f'Name:\tpython\nTracerPid:\t{tracer}\n')

    def test_all_threads_untraced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 123, 0)
            self.fixture(root, 124, 0)
            self.assertEqual(require_untraced_api(123, root)['threads_checked'], 2)

    def test_child_thread_traced_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 123, 0)
            self.fixture(root, 124, 987)
            with self.assertRaises(ValueError):
                require_untraced_api(123, root)

    def test_missing_tracer_field_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 123, 0)
            (root / '123/task/123/status').write_text('Name:\tpython\n')
            with self.assertRaises(ValueError):
                require_untraced_api(123, root)


if __name__ == '__main__':
    unittest.main()
