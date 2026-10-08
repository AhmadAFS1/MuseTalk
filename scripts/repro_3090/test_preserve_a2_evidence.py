import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import preserve_a2_evidence as h


class A2EvidenceTests(unittest.TestCase):
    def test_real_descriptor(self):
        path = Path(__file__).resolve().parents[2] / h.BASE.removeprefix('MuseTalk/') / 'provisioning/a2_owned_target.json'
        self.assertEqual(h.descriptor(path)['instance_id'], h.INSTANCE)

    def test_wrong_descriptor_and_link_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'descriptor.json'
            p.write_text('{}')
            with self.assertRaises(ValueError):
                h.descriptor(p)
            link = Path(d) / 'link'
            link.symlink_to(p)
            with self.assertRaises(ValueError):
                h.descriptor(link)

    def test_deadline(self):
        h.before_deadline(h.DEADLINE - dt.timedelta(seconds=1))
        for when in (h.DEADLINE, h.DEADLINE + dt.timedelta(seconds=1)):
            with self.assertRaises(ValueError):
                h.before_deadline(when)

    def test_ancestor_links_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / 'real').mkdir()
            (base / 'real/data').write_text('x')
            (base / 'link').symlink_to(base / 'real', target_is_directory=True)
            for name in ('link/data', '../outside'):
                with self.assertRaises(ValueError):
                    h.safe_source(base, name)

    def test_terminal_failure_preserved_and_in_progress_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            target = base / h.PROFILES['latent_seed123'][0]
            target.mkdir(parents=True)
            p = target / 'comparison.json'
            with self.assertRaises(ValueError):
                h.terminal_receipt(base, 'latent_seed123')
            p.write_text(json.dumps({'status': 'INVALID_INCOMPLETE', 'finished_utc': '2026-10-08T23:15:00+00:00'}))
            self.assertEqual(h.terminal_receipt(base, 'latent_seed123')['recorded_status'], 'INVALID_INCOMPLETE')
            p.write_text(json.dumps({'status': 'IN_PROGRESS'}))
            with self.assertRaises(ValueError):
                h.terminal_receipt(base, 'latent_seed123')

    def make_pack(self, base):
        profile = 'latent_seed123'
        target = base / h.PROFILES[profile][0]
        target.mkdir(parents=True)
        (target / 'cache.pt').write_bytes(b'raw payload does not deserialize')
        (target / 'comparison.json').write_text(json.dumps({'status': 'FAIL_REPEAT_CHANGED',
                                                          'finished_utc': '2026-10-08T23:15:00+00:00'}))
        rows = h.inventory(base, h.PROFILES[profile])
        archive = base / ('a2-' + profile + '.tar.gz')
        manifest_sha = h.pack_archive(base, rows, archive)
        value = {'schema': 'a2_private_evidence_pack_v1', 'status': 'PASS_LOCAL_ALL_FILE_SHA',
                 'profile': profile, 'descriptor_sha256': h.DESCRIPTOR_SHA,
                 'binding': {'instance_id': h.INSTANCE, 'worker_hostname': h.HOSTNAME,
                             'gpu_uuid': h.GPU_UUID, 'deadline_utc': h.DEADLINE.isoformat()},
                 'finished_utc': '2026-10-08T23:16:00+00:00', 'files': rows,
                 'manifest_sha256': manifest_sha, 'archive': archive.name,
                 'sha256': h.operator.sha_file(archive), 'bytes': archive.stat().st_size,
                 'terminal_receipt': h.terminal_receipt(base, profile),
                 'release_ready': False, 'cloud_mutations': False}
        receipt = base / 'pack-receipt.json'
        receipt.write_text(json.dumps(value))
        return receipt, value

    def test_pack_roundtrip_all_payload_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            receipt, value = self.make_pack(Path(d))
            actual, archive = h.load_pack(receipt, h.operator.sha_file(receipt))
            self.assertEqual(actual, value)
            self.assertTrue(archive.is_file())

    def test_tampered_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            receipt, _ = self.make_pack(Path(d))
            digest = h.operator.sha_file(receipt)
            receipt.write_text('{}')
            with self.assertRaises(ValueError):
                h.load_pack(receipt, digest)

    def test_wrong_host_late_pack_unbound_terminal_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            receipt, original = self.make_pack(Path(d))
            for change in ('host', 'late', 'terminal', 'path', 'release'):
                value = json.loads(json.dumps(original))
                if change == 'host': value['binding']['worker_hostname'] = 'A1'
                if change == 'late': value['finished_utc'] = h.DEADLINE.isoformat()
                if change == 'terminal': value['terminal_receipt']['sha256'] = '0' * 64
                if change == 'path': value['files'][0]['path'] = '../escape'
                if change == 'release': value['release_ready'] = True
                receipt.write_text(json.dumps(value))
                with self.assertRaises(ValueError, msg=change):
                    h.load_pack(receipt, h.operator.sha_file(receipt))

    def test_tampered_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            receipt, value = self.make_pack(Path(d))
            (Path(d) / value['archive']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                h.load_pack(receipt, h.operator.sha_file(receipt))

    def persist_args(self, receipt, base, mode='conditional-put-verify'):
        return SimpleNamespace(receipt=receipt, receipt_sha256=h.operator.sha_file(receipt),
                               mode=mode, out=base / 'persistence.json')

    def test_no_upload_when_privacy_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            receipt, _ = self.make_pack(base)
            with patch.object(h.operator, 'RELEASE', base), patch.object(h.privacy, 'produce', side_effect=ValueError()), \
                 patch.object(h.operator, 'cli') as cli:
                self.assertEqual(h.persist(self.persist_args(receipt, base)), 2)
                cli.assert_not_called()

    def test_conditional_upload_and_version_bound_fresh_get(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            receipt, row = self.make_pack(base)
            calls = []
            def cli(operation, arguments, **kwargs):
                calls.append((operation, arguments))
                if operation == 'put-object': return {'VersionId': 'v1'}
                if operation == 'head-object':
                    return {'VersionId': 'v1', 'ContentLength': row['bytes'],
                            'Metadata': {'sha256': row['sha256']}, 'ServerSideEncryption': 'AES256'}
                if operation == 'get-object':
                    Path(arguments[-1]).write_bytes((base / row['archive']).read_bytes())
                    return {}
                raise AssertionError(operation)
            with patch.object(h.operator, 'RELEASE', base), patch.object(h.privacy, 'produce', return_value={}), \
                 patch.object(h.operator, 'cli', side_effect=cli):
                self.assertEqual(h.persist(self.persist_args(receipt, base)), 0)
            self.assertEqual([c[0] for c in calls], ['put-object', 'head-object', 'get-object', 'head-object'])
            self.assertIn('--if-none-match', calls[0][1])
            self.assertEqual(calls[0][1][calls[0][1].index('--if-none-match') + 1], '*')
            self.assertIn('--version-id', calls[2][1])
            self.assertNotIn('--acl', calls[0][1])

    def test_reconcile_does_not_put(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            receipt, _ = self.make_pack(base)
            with patch.object(h.operator, 'RELEASE', base), patch.object(h.privacy, 'produce', return_value={}), \
                 patch.object(h.operator, 'cli', side_effect=ValueError()) as cli:
                self.assertEqual(h.persist(self.persist_args(receipt, base, 'reconcile-read-only')), 2)
                self.assertEqual(cli.call_args.args[0], 'head-object')


if __name__ == '__main__':
    unittest.main()
