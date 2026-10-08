#!/usr/bin/env python3
"""CPU tests for actual bundle CLI payload selection; no S3/GPU access."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import trt_artifact_bundle as bundle


class PayloadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root / 'models/native/bs16/prefix.plan'
        self.plan.parent.mkdir(parents=True)
        self.plan.write_bytes(b'test-only-plan')
        self.archive = self.root / 'output/native.tar.gz'
        self.sidecars = self.root / 'output/sidecars'

    def cli(self, required_files, required_dirs):
        return subprocess.run([
            sys.executable, str(Path(bundle.__file__).resolve()),
            '--repo-root', str(self.root), '--strict',
            '--sidecar-dir', str(self.sidecars), 'create',
            '--output', str(self.archive), '--profile', 'cpu-test-only',
            '--required-files', required_files, '--required-dirs', required_dirs,
            '--optional-paths', '', '--compresslevel', '1',
        ], capture_output=True, text=True, timeout=20)

    def manifest(self):
        with tarfile.open(self.archive) as archive:
            return json.load(archive.extractfile(bundle.BUNDLE_MANIFEST)), archive.getnames()

    def test_explicit_empty_dirs_does_not_restore_legacy_defaults(self):
        # A sibling ONNX graph must not enter the explicit serving-file payload.
        self.plan.with_suffix('.onnx').write_bytes(b'build-only-graph')
        result = self.cli('models/native/bs16/prefix.plan', '')
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest, names = self.manifest()
        self.assertEqual(manifest['required_dirs'], [])
        self.assertEqual(manifest['optional_paths'], [])
        self.assertEqual([row['path'] for row in manifest['files']], ['models/native/bs16/prefix.plan'])
        self.assertEqual(set(names), {bundle.BUNDLE_MANIFEST, bundle.BUNDLE_CHECKSUMS,
                                     'models/native/bs16/prefix.plan'})

    def test_explicit_empty_files_does_not_restore_legacy_defaults(self):
        result = self.cli('', 'models/native/bs16')
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest, _ = self.manifest()
        self.assertEqual(manifest['required_files'], [])
        self.assertEqual([row['path'] for row in manifest['files']], ['models/native/bs16/prefix.plan'])

    def test_empty_payload_is_rejected_before_sidecars_or_archive(self):
        result = self.cli('', '')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No TRT artifact payload files selected', result.stderr)
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.sidecars.exists())

    def test_missing_selected_file_is_not_hidden_by_legacy_defaults(self):
        result = self.cli('models/native/bs16/missing.plan', '')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('models/native/bs16/missing.plan', result.stderr)
        self.assertNotIn('calibration/vae_decoder', result.stderr)
        self.assertFalse(self.archive.exists())

    def test_omitted_cli_options_preserve_legacy_defaults(self):
        with patch.object(sys, 'argv', ['bundle', 'create', '--output', 'unused.tar.gz']):
            args = bundle.parse_args()
        self.assertEqual(bundle._parse_csv(args.required_files), list(bundle.DEFAULT_REQUIRED_FILES))
        self.assertEqual(bundle._parse_csv(args.required_dirs), list(bundle.DEFAULT_REQUIRED_DIRS))


if __name__ == '__main__':
    unittest.main()
