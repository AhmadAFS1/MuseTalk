"""CPU-only allocation descriptor rejection; not CUDA ownership evidence."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import watch_owned_single_leaf_target as wrapper


class Tests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        self.path = Path(t.name).resolve() / 'target.json'
        self.now = dt.datetime(2026, 10, 9, 3, tzinfo=dt.timezone.utc)
        self.doc = dict(instance_id='54999999', label='musetalk-r5-3090-dev-test-a4',
            worker_alias='musetalk-3090-build-54999999', worker_hostname='synthetic-host',
            gpu_uuid='GPU-00000000-0000-0000-0000-000000000001',
            ledger='/home/ec2-user/.local/state/musetalk-r5-3090-dev-test-a4/startup-ledger.json',
            deadline_utc='2026-10-09T05:00:00Z')

    def write(self):
        raw = json.dumps(self.doc).encode(); self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_valid_descriptor_has_exact_bindings(self):
        doc, deadline = wrapper.binding(self.path, self.write(), now=self.now)
        self.assertEqual(doc, self.doc)
        self.assertEqual(deadline, self.now + dt.timedelta(hours=2))

    def test_protected_instance_alias_label_ledger_and_deadline_rejected(self):
        for key, value in [('instance_id', '51074906'), ('worker_alias', '3-way-head-talk'),
                           ('label', 'production'), ('ledger', '/tmp/ledger'),
                           ('deadline_utc', '2026-10-09T02:59:00Z'),
                           ('deadline_utc', '2026-10-11T05:00:00Z')]:
            original = dict(self.doc); self.doc[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                wrapper.binding(self.path, self.write(), now=self.now)
            self.doc = original

    def test_hash_and_unknown_schema_rejected(self):
        self.write()
        with self.assertRaises(ValueError): wrapper.binding(self.path, '0' * 64, now=self.now)
        self.doc['unexpected'] = 'value'
        with self.assertRaises(ValueError): wrapper.binding(self.path, self.write(), now=self.now)

    def test_default_off_before_descriptor_or_hardware(self):
        with patch.object(wrapper, 'binding', side_effect=AssertionError('descriptor read')), \
             patch.object(wrapper.socket, 'gethostname', side_effect=AssertionError('host read')):
            with self.assertRaises(ValueError):
                wrapper.main(['--owned-target-json', str(self.path), '--owned-target-sha256', '0' * 64, '--', '--enable'])

    def test_canonical_protocol_source_and_pins_preserved(self):
        watch = wrapper.reviewed_watch()
        self.assertEqual(watch.ALLOCATION_BYTES, 2 << 20)
        self.assertEqual(watch.QUERY_SECONDS, 3); self.assertEqual(watch.FINISH_SECONDS, 10)
        self.assertEqual(watch.CANONICAL_TARGETS['validate_unet_backend.py'],
                         '81b74eddf5aaff8348ac27cce67b92763e937e09309762d137062b0a23f1d7a0')


if __name__ == '__main__':
    unittest.main()
