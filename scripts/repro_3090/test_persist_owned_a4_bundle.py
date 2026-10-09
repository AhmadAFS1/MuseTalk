"""CPU-only private archive scope/negative cases; never call AWS or CUDA."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import persist_owned_a4_bundle as persistence


class PersistenceTests(unittest.TestCase):
    def test_restore_uses_supported_bounded_capture_stage(self):
        with mock.patch.object(persistence.safe_capture, 'capture', return_value='verified') as capture:
            result = persistence.restore_archive(Path('/fresh/archive.tar.gz'), Path('/fresh/restore'), 'a' * 64)
        self.assertEqual(result, 'verified')
        argv = capture.call_args.args[0]
        self.assertIn('--strict', argv)
        self.assertEqual(argv[argv.index('--repo-root') + 1], '/fresh/restore')
        self.assertEqual(argv[argv.index('--uri') + 1], '/fresh/archive.tar.gz')
        self.assertEqual(argv[argv.index('--expected-sha256') + 1], 'a' * 64)
        self.assertEqual(capture.call_args.kwargs['stage'], 'unspecified')
        self.assertEqual(capture.call_args.kwargs['timeout_s'], 600)
        self.assertEqual(capture.call_args.kwargs['output_limit_bytes'], 64 * 1024)

    def data(self, profile='engines'):
        name = persistence.MODEL_ROOTS[0] + 'manifest.json' if profile == 'engines' else persistence.LATENT_ROOT + 'comparison.json'
        return {'schema': 1, 'profile': 'private-a4-' + profile + '-diagnostic',
                'files': [{'path': name, 'size': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}]}

    def test_exact_engine_and_latent_scopes_are_distinct(self):
        for profile in ('engines', 'latents'):
            self.assertEqual(len(persistence.validate_manifest(self.data(profile), profile)), 1)
            with self.assertRaises(ValueError):
                persistence.validate_manifest(self.data(profile), 'latents' if profile == 'engines' else 'engines')

    def test_foreign_secret_source_and_traversal_paths_rejected(self):
        for name in ('/root/onstart.sh', '../models/test', '.runtime/secret.env',
                     'models/tensorrt_unet_stagewise_sm86_r5_v1/bs16/manifest.json',
                     persistence.MODEL_ROOTS[0] + 'unexpected.plan',
                     persistence.MODEL_ROOTS[0] + '../manifest.json'):
            data = self.data(); data['files'][0]['path'] = name
            with self.assertRaises(ValueError):
                persistence.validate_manifest(data, 'engines')

    def test_symlink_duplicate_and_bad_hash_size_rejected(self):
        for key, value in (('symlink', 'target'), ('sha256', '0'), ('size', -1), ('size', 2 * 1024**3)):
            data = self.data(); data['files'][0][key] = value
            with self.assertRaises(ValueError):
                persistence.validate_manifest(data, 'engines')
        data = self.data(); data['files'] *= 2
        with self.assertRaises(ValueError):
            persistence.validate_manifest(data, 'engines')

    def test_archive_exact_coverage_and_explicit_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.tar.gz'
            data = self.data()
            with tarfile.open(path, 'w:gz') as archive:
                for name, raw in ((persistence.MANIFEST, json.dumps(data).encode()),
                                  (persistence.CHECKSUMS, b'checksum'), (data['files'][0]['path'], b'{}')):
                    member = tarfile.TarInfo(name); member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
            digest = persistence.operator.sha_file(path)
            self.assertEqual(persistence.manifest_from_archive(path, digest, 'engines')[0], data)
            with self.assertRaises(ValueError):
                persistence.manifest_from_archive(path, '0' * 64, 'engines')


if __name__ == '__main__':
    unittest.main()
