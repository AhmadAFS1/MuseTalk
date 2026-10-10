"""Synthetic input safety checks; not native GPU or publication acceptance."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import assemble_candidate_inventory as assembly


class AssemblyTests(unittest.TestCase):
    def make_native(self, path, *, changed=False, extra=False, duplicate=False):
        payload = b'synthetic-plan'
        model = 'models/native/test.plan'
        manifest = {'schema': 1, 'files': [{'path': model,
                    'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}]}
        members = [(model, payload + b'x' if changed else payload),
                   ('.musetalk_trt_artifact_manifest.json', json.dumps(manifest).encode()),
                   ('.musetalk_trt_artifact_SHA256SUMS', b'synthetic')]
        if extra:
            members.append(('../outside', b'unexpected'))
        if duplicate:
            members.append(members[0])
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in members:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def test_actual_bytes_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.tar.gz'
            self.make_native(path)
            files, manifest = assembly.native_inventory(path, 1)
            self.assertEqual(files['models/native/test.plan']['size_bytes'], len(b'synthetic-plan'))
            self.assertEqual(len(manifest['sha256']), 64)

    def test_changed_extra_and_duplicate_members_rejected(self):
        for option in ('changed', 'extra', 'duplicate'):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'native.tar.gz'
                self.make_native(path, **{option: True})
                with self.assertRaises(ValueError):
                    assembly.native_inventory(path, 1)

    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'file'
            original.write_bytes(b'synthetic')
            link = Path(directory) / 'link'
            link.symlink_to(original)
            with self.assertRaises(ValueError):
                assembly.fingerprint(link)


if __name__ == '__main__':
    unittest.main()
