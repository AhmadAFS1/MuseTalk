"""Synthetic finalization contract; no real GPU, TensorRT, or quality evidence."""
import copy
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_assemble_owned_unet_tail_a3 as fixtures
import run_owned_unet_overlay as target
import unet_plan_overlay as overlay


class Tests(unittest.TestCase):
    write = fixtures.OverlayTests.write

    def setUp(self):
        fixtures.OverlayTests.setUp(self)
        self.h = overlay.checked_helpers()
        self.h.NATIVE, self.h.PORTABLE = self.native, self.portable
        self.h.NATIVE_SHA = hashlib.sha256((self.native / 'manifest.json').read_bytes()).hexdigest()
        self.h.PORTABLE_SHA = hashlib.sha256((self.portable / 'manifest.json').read_bytes()).hexdigest()
        self.h.CODE, self.h.MODELS = {}, {}
        self.base = self.root / 'docs'; (self.base / 'native').mkdir(parents=True)
        (self.root / 'models').mkdir()
        self.engine, self.out = self.root / 'models/candidate', self.base / 'native/run'
        self.lease = self.root / 'lease'; self.lease.with_suffix('.holder').write_text('synthetic')
        self.uuid = 'GPU-00000000-0000-0000-0000-000000000001'
        self.binding = Mock(return_value=({'worker_hostname': socket.gethostname(), 'gpu_uuid': self.uuid,
                                         'instance_id': '54999999'}, dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)))
        self.verify = Mock(return_value={'synthetic_only': True})
        self.wrapper = types.SimpleNamespace(binding=self.binding)
        self.lineage = types.SimpleNamespace(verify_preregistration=self.verify)
        self.argv = ['--owned-target-json', str(self.root / 'target.json'), '--owned-target-sha256', '0' * 64,
                     '--variant', 'portable_prefix', '--engine-root', str(self.engine), '--out', str(self.out),
                     '--successor-inputs', str(self.root / 'inputs.json'), '--successor-envelope', str(self.root / 'envelope.json'),
                     '--lineage-receipt', str(self.root / 'receipt.json'), '--lineage-receipt-sha256', '1' * 64]
        self.sw = types.SimpleNamespace(block_order=lambda _variant: tuple(overlay.BLOCKS))
        self.options = None
        def build(options, batch, _model, _device):
            self.options = options; self.assertEqual(batch, 16)
            m = json.loads((self.engine / 'bs16/manifest.json').read_text())
            m.update(complete=True, probe={'deterministic_run_to_run': True, 'graph_equals_direct_enqueue': True})
            (self.engine / 'bs16/manifest.json').write_text(json.dumps(m))
            return m
        self.builder = types.SimpleNamespace(load_eager_unet=Mock(return_value='synthetic'), build_batch=Mock(side_effect=build))
        self.torch = types.SimpleNamespace(device=lambda _name: 'synthetic', backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace()))

    def run_fake(self, execute=True):
        modules = {'watch_owned_single_leaf_target.py': self.wrapper, 'unet_plan_overlay.py': overlay,
                   'preregister_metadata_quality_lineage.py': self.lineage}
        with patch.object(target, 'ROOT', self.root), patch.object(target, 'BASE', self.base), \
             patch.object(target, 'checked_module', side_effect=lambda name: modules[name]), \
             patch.object(target.subprocess, 'check_output', side_effect=[self.uuid, '']), \
             patch.object(overlay, 'checked_helpers', return_value=self.h), \
             patch.dict(os.environ, BOX_GUARD_LEASE_FILE=str(self.lease)), \
             patch.dict(sys.modules, {'torch': self.torch, 'scripts.build_unet_stagewise': self.builder,
                                      'scripts.unet_stagewise_trt': self.sw}), \
             patch.object(sys, 'path', list(sys.path)), patch.object(target.os, 'chdir'), \
             contextlib.redirect_stdout(io.StringIO()):
            return target.main((['--execute'] if execute else []) + self.argv)

    def test_default_off_before_owned_binding(self):
        with self.assertRaisesRegex(ValueError, 'explicit execution'): self.run_fake(execute=False)
        self.binding.assert_not_called(); self.assertFalse(self.out.exists())

    def test_exact_copy_then_finalize_only_and_no_acceptance(self):
        self.run_fake()
        self.verify.assert_called_once(); self.builder.build_batch.assert_called_once()
        self.assertEqual(self.options.blocks, '__FINALIZE_ONLY__')
        self.assertFalse(self.options.force); self.assertEqual(self.options.int8_blocks, '')
        row = json.loads((self.out / 'assembly.json').read_text())
        self.assertEqual(row['status'], 'FINALIZED_QUALITY_PERFORMANCE_UNTESTED')
        self.assertFalse(row['quality_accepted']); self.assertFalse(row['performance_measured']); self.assertFalse(row['release_ready'])
        for name, entry in json.loads((self.engine / 'bs16/manifest.json').read_text())['blocks'].items():
            source = self.portable if name == 'prefix' else self.native
            self.assertEqual((source / entry['engine_file']).read_bytes(), (self.engine / 'bs16' / entry['engine_file']).read_bytes())
        with self.assertRaisesRegex(ValueError, 'forbidden'): self.sw.build_engine_from_onnx()

    def test_bad_input_lineage_stops_before_copy_or_gpu(self):
        self.verify.side_effect = ValueError('lineage mismatch')
        with self.assertRaisesRegex(ValueError, 'lineage mismatch'): self.run_fake()
        self.assertFalse(self.engine.exists()); self.assertFalse(self.out.exists())
        self.builder.load_eager_unet.assert_not_called()

    def test_expired_cleanup_margin_stops_before_copy(self):
        doc, _deadline = self.binding.return_value
        self.binding.return_value = doc, dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=590)
        with self.assertRaisesRegex(ValueError, 'cleanup margin'): self.run_fake()
        self.verify.assert_not_called(); self.assertFalse(self.engine.exists())

    def test_failed_probe_never_becomes_quality_accepted(self):
        build = self.builder.build_batch.side_effect
        def fail(*args):
            m = build(*args); m['probe']['graph_equals_direct_enqueue'] = False; return m
        self.builder.build_batch.side_effect = fail
        with self.assertRaisesRegex(ValueError, 'hard probe'): self.run_fake()
        row = json.loads((self.out / 'assembly.json').read_text())
        self.assertEqual(row['status'], 'FAILED'); self.assertFalse(row['quality_accepted'])


if __name__ == '__main__':
    unittest.main()
