"""CPU-only bookkeeping adaptation; never downloads or imports CUDA."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owned_inputs', HERE / 'prepare_owned_cpu_inputs.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
DOC = {'worker_hostname': 'owned-host', 'gpu_uuid': 'GPU-new', 'instance_id': '12345678'}


class Contracts(unittest.TestCase):
    def test_pinned_original(self):
        raw = (HERE / 'prepare_owned_a4_cpu_inputs.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), m.AUDIT_SHA)

    def test_only_bookkeeping_literals_change(self):
        raw = (HERE / 'prepare_owned_a4_cpu_inputs.py').read_bytes()
        changed = m.adapted(raw, DOC)
        reverse = {b"'owned-host'": b"'9dad9291adec'", b"'GPU-new'": b"'GPU-050b4bc5-99f6-305e-8c7f-fd00ea44551f'",
            b"'12345678'": b"'54957508'", b"'owned_cpu_original_input_audit_v2'": b"'owned_a4_cpu_original_input_audit_v1'"}
        for new, old in reverse.items(): changed = changed.replace(new, old)
        self.assertEqual(changed, raw)

    def test_changed_audit_rejected(self):
        with self.assertRaises(ValueError): m.adapted((HERE / 'prepare_owned_a4_cpu_inputs.py').read_bytes() + b'\n', DOC)

    def test_adapted_import_preserves_input_model_pins(self):
        raw = m.adapted((HERE / 'prepare_owned_a4_cpu_inputs.py').read_bytes(), DOC)
        mod = m.load(raw, HERE / 'prepare_owned_a4_cpu_inputs.py')
        self.assertEqual(mod.PARENT_SHA, '6d3ab6ef31605c2231605e03042e82361a27112589ef6d7f6f8ff8f4b016eea5')
        self.assertEqual(mod.SYNCNET_SHA, '38fa63bad3ed2332f647c40a5dc616cb0e233db8579f698f62af4c41965c4da5')

    def test_default_off_before_source_or_host(self):
        with self.assertRaisesRegex(ValueError, 'explicit execution'):
            m.main(['--owned-target-json', '/not-read', '--owned-target-sha256', '0'*64, '--out', '/not-written'])


if __name__ == '__main__': unittest.main()
