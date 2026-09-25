"""Pinned native dependency installation must fail closed and preserve old files."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.install_native_vp8 import install, validate_install


class NativeVP8InstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheel = self.root / 'source.whl'
        self.relative = 'aiortc/codecs/_vpx.abi3.so'
        self.binary = b'test native bytes, never imported'
        with zipfile.ZipFile(self.wheel, 'w') as z:
            z.writestr(self.relative, self.binary)
            z.writestr('../must-not-be-extracted', b'not allowlisted')
        self.manifest = {
            'package': 'aiortc', 'version': '1.11.0', 'license': 'BSD-3-Clause',
            'wheel_url': 'https://example.invalid/not-requested',
            'wheel_sha256': hashlib.sha256(self.wheel.read_bytes()).hexdigest(),
            'wheel_size_bytes': self.wheel.stat().st_size,
            'extracted': [{'path': self.relative, 'bytes': len(self.binary),
                           'sha256': hashlib.sha256(self.binary).hexdigest()}],
        }
        self.destination = self.root / 'installed'

    def test_verified_subset_and_receipt_are_atomic_and_idempotent(self):
        result = install(self.destination, self.wheel, manifest=self.manifest)
        self.assertFalse(result['reused'])
        self.assertEqual((self.destination / self.relative).read_bytes(), self.binary)
        self.assertFalse((self.root / 'must-not-be-extracted').exists())
        self.assertEqual(json.loads((self.destination / 'installation.json').read_text())['wheel_sha256'],
                         self.manifest['wheel_sha256'])
        self.wheel.unlink()
        with patch('urllib.request.urlopen', side_effect=AssertionError('no download')):
            self.assertTrue(install(self.destination, manifest=self.manifest)['reused'])
        self.assertFalse(list(self.root.glob('.native-vp8-install-*')))

    def test_tampered_wheel_never_publishes_destination(self):
        original = self.wheel.read_bytes()
        self.wheel.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            install(self.destination, self.wheel, manifest=self.manifest)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob('.native-vp8-install-*')))

    def test_corrupt_existing_install_is_preserved_and_rejected(self):
        install(self.destination, self.wheel, manifest=self.manifest)
        path = self.destination / self.relative
        path.write_bytes(b'x' * len(self.binary))
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            install(self.destination, self.wheel, manifest=self.manifest)
        self.assertEqual(path.read_bytes(), b'x' * len(self.binary))

    def test_internal_member_digest_is_independently_checked(self):
        self.manifest['extracted'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'archive hash mismatch'):
            install(self.destination, self.wheel, manifest=self.manifest)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob('.native-vp8-install-*')))

    def test_corrupt_receipt_and_path_escape_are_rejected(self):
        install(self.destination, self.wheel, manifest=self.manifest)
        (self.destination / 'installation.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'another wheel'):
            validate_install(self.destination, self.manifest)
        self.manifest['extracted'][0]['path'] = '../escape'
        with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
            install(self.root / 'other-install', self.wheel, manifest=self.manifest)
        self.assertFalse((self.root / 'escape').exists())
        self.assertFalse((self.root / 'other-install').exists())


if __name__ == '__main__':
    unittest.main()
