"""CPU-only orchestration contracts; no processes, GPU or quality acceptance."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owned_quality',HERE/'run_owned_quality_a5.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Execution(unittest.TestCase):
    def test_default_off_and_quality_only(self):
        for argv in [[],['--execute','--','gpu']]:
            with self.assertRaises(ValueError):
                m.main(['--owned-target-json','/not-read','--owned-target-sha256','a'*64,*argv])

    def test_actual_source_pins(self):
        root = HERE.parents[1]
        for name,digest in m.PINS.items(): m.checked(HERE/name,digest)
        for relative,digest in m.TARGETS.values(): m.checked(root/relative,digest)

    def test_quality_adapter_does_not_change_workload_parameters(self):
        argv = m.quality_capture_arguments('/out','canonical_capture','/owned','a'*64)
        self.assertEqual(argv[:4],['--enable','--stage','Q','--output-dir'])
        self.assertNotIn('--tracking-overlap',argv)

    def test_runtime_probe_default_off(self):
        spec = importlib.util.spec_from_file_location('runtime_probe',HERE/'owned_runtime_probe.py')
        probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
        with self.assertRaises(ValueError): probe.main([])


if __name__ == '__main__': unittest.main()
