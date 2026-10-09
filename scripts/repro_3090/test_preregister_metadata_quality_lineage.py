"""CPU synthetic filesystem contracts; never GPU, runtime quality or cloud proof."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preregister_metadata_quality_lineage as lineage
import quality_envelope as quality

EVIDENCE = lineage.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'


class OriginalPins(unittest.TestCase):
    def test_exact_originals_and_all_698_bounds(self):
        inputs, envelope = lineage.parents(EVIDENCE / 'harnesses/quality-inputs-v1.json',
                                           EVIDENCE / 'quality/reference-envelope-v2.json')
        self.assertEqual(len(inputs['files']), 878)
        self.assertEqual(len(lineage.METADATA), 12)
        self.assertEqual(sum(len(a['bounds']) for a in envelope['avatars'].values()) + len(envelope['global_bounds']), 698)
        rows = {r['path']: r for r in inputs['files']}
        self.assertEqual(rows[lineage.WORKER_PATH]['sha256'], lineage.WORKER_SHA)

    def test_duplicate_and_nonfinite_json_refused(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(lineage.Rejected): lineage.parse(raw)


class SyntheticFiles(unittest.TestCase):
    """Synthetic parent pins are patched only here; production pins tested above."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name).resolve() / 'repo'
        self.base = self.repo / 'docs/fps_comparisons/run/harnesses'; self.base.mkdir(parents=True)
        self.quality_dir = self.base.parent / 'quality'; self.quality_dir.mkdir()
        self.parent_inputs = self.base / 'original.json'
        self.parent_envelope = self.quality_dir / 'original-envelope.json'
        self.outputs = {'out_inputs': self.base / 'successor.json', 'out_envelope': self.quality_dir / 'successor-envelope.json',
                        'receipt_path': self.quality_dir / 'lineage.json'}
        self.rows = []
        self.payloads = {}
        for i, payload in enumerate(lineage.METADATA.values()):
            raw = ('synthetic frozen payload ' + str(i)).encode(); self.payloads[payload] = raw
            self.add_row(payload, raw)
        worker = b'synthetic historical worker fixture\n'
        self.add_row(lineage.WORKER_PATH, worker)
        patcher = patch.object(lineage, 'WORKER_SHA', lineage.sha(worker)); patcher.start(); self.addCleanup(patcher.stop)
        self.metadata_raw = {}
        for i, (metadata, payload) in enumerate(lineage.METADATA.items()):
            raw = self.payloads[payload]
            etag = lineage.sha(raw) if i % 2 else hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            original = ('a' * 40 + '\n' + etag + '\n1.0\n').encode()
            current = ('a' * 40 + '\n' + etag + '\n2.0\n').encode()
            self.add_row(metadata, original, current)
            self.metadata_raw[metadata] = current
        for i in range(853): self.add_row('../../../../fixtures/file' + str(i), b'' if i == 0 else str(i).encode())
        self.manifest = {'schema': 'repro_3090_inputs_v1', 'files': self.rows, 's3_objects': []}
        self.parent_raw = lineage.encoded(self.manifest); self.parent_inputs.write_bytes(self.parent_raw)
        self.parent_sha = lineage.sha(self.parent_raw)
        patcher = patch.object(lineage, 'PARENT_INPUT_SHA', self.parent_sha); patcher.start(); self.addCleanup(patcher.stop)
        self.envelope = json.loads((EVIDENCE / 'quality/reference-envelope-v2.json').read_bytes())
        self.envelope['input_manifest_sha256'] = self.parent_sha
        for reference in self.envelope['references']: reference['input_manifest_sha256'] = self.parent_sha
        self.envelope_raw = lineage.encoded(self.envelope); self.parent_envelope.write_bytes(self.envelope_raw)
        patcher = patch.object(lineage, 'PARENT_ENVELOPE_SHA', lineage.sha(self.envelope_raw)); patcher.start(); self.addCleanup(patcher.stop)

    def add_row(self, relative, original, current=None):
        path = (self.base / relative).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original if current is None else current)
        self.rows.append({'path': relative, 'bytes': len(original), 'sha256': lineage.sha(original)})

    def preregister(self, **kwargs):
        return lineage.preregister(parent_inputs=self.parent_inputs, parent_envelope=self.parent_envelope,
                                   **{**self.outputs, **kwargs})

    def verify(self, result):
        return lineage.verify_preregistration(parent_inputs=self.parent_inputs, parent_envelope=self.parent_envelope,
            successor_inputs=self.outputs['out_inputs'], successor_envelope=self.outputs['out_envelope'],
            receipt_path=self.outputs['receipt_path'], expected_receipt_sha256=result['receipt_sha256'])

    def test_default_audit_writes_nothing_and_all_12_etags_verified(self):
        result = self.preregister()
        self.assertEqual(result['status'], 'AUDIT_ONLY_NO_PREREGISTRATION')
        self.assertFalse(result['outputs_written'])
        self.assertFalse(any(p.exists() for p in self.outputs.values()))
        audit = result['audit']
        self.assertEqual((audit['paths'], audit['nonmetadata_exact'], audit['metadata_validated']), (878, 866, 12))
        self.assertEqual({p['etag_algorithm'] for p in audit['metadata_proofs']}, {'sha256', 'git_blob_sha1'})
        self.assertTrue(all(p['etag_content_verified'] for p in audit['metadata_proofs']))
        self.assertEqual(audit['original_metadata_recoverability'], 'UNKNOWN')

    def test_successor_verifies_and_preserves_originals_and_every_bound(self):
        result = self.preregister(execute=True)
        proof = self.verify(result)
        self.assertEqual(proof['status'], 'VERIFIED_METADATA_EQUIVALENCE_ONLY')
        self.assertFalse(proof['quality_accepted'])
        self.assertEqual(self.parent_inputs.read_bytes(), self.parent_raw)
        self.assertEqual(self.parent_envelope.read_bytes(), self.envelope_raw)
        derived = lineage.parse(self.outputs['out_envelope'].read_bytes())
        self.assertEqual(derived['input_manifest_sha256'], result['successor_input_manifest_sha256'])
        self.assertNotEqual(derived['input_manifest_sha256'], self.parent_sha)
        for name in self.envelope:
            if name not in ('status', 'frozen_utc', 'input_manifest_sha256'):
                self.assertEqual(lineage.encoded(derived[name]), lineage.encoded(self.envelope[name]), name)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in self.outputs.values()))

    def test_unchanged_compare_accepts_truthful_binding_but_no_acceptance(self):
        self.preregister(execute=True)
        derived = lineage.parse(self.outputs['out_envelope'].read_bytes())
        # Pure synthetic metric values at their pre-existing limits: NOT measured.
        candidate = {'role': 'candidate', 'started_utc': (dt.datetime.fromisoformat(derived['frozen_utc']) + dt.timedelta(seconds=1)).isoformat(),
            'input_manifest_sha256': derived['input_manifest_sha256'],
            'avatars': {ident: {'signature': row['signature'], 'metrics': {k: v['limit'] for k, v in row['bounds'].items()}}
                        for ident, row in derived['avatars'].items()},
            'global_metrics': {k: v['limit'] for k, v in derived['global_bounds'].items()},
            'strict_original_gates': {'synthetic_unchanged_failure': 'FAIL'}}
        policy = quality.checks.read(quality.POLICY)
        result = quality.compare(derived, candidate, policy)
        self.assertEqual(len(result['metrics']), 698)
        self.assertEqual(result['strict_original_gates'], candidate['strict_original_gates'])
        self.assertEqual(result['decision'], 'incomplete'); self.assertFalse(result['release_ready'])
        candidate['input_manifest_sha256'] = self.parent_sha
        with self.assertRaisesRegex(ValueError, 'inputs differ'): quality.compare(derived, candidate, policy)
        candidate['input_manifest_sha256'] = derived['input_manifest_sha256']
        candidate['started_utc'] = self.envelope['frozen_utc']
        with self.assertRaisesRegex(ValueError, 'predates'): quality.compare(derived, candidate, policy)

    def test_worker935_or_any_other_worker_change_rejected(self):
        (self.base / lineage.WORKER_PATH).resolve().write_bytes(b'reviewed overlap worker is STILL not metadata')
        with self.assertRaisesRegex(lineage.Rejected, 'nonmetadata_content_changed'): self.preregister(execute=True)
        self.assertFalse(any(p.exists() for p in self.outputs.values()))

    def test_latent_like_nonmetadata_change_rejected(self):
        (self.repo / 'fixtures/file7').write_bytes(b'changed cached latent')
        with self.assertRaisesRegex(lineage.Rejected, 'nonmetadata_content_changed'): self.preregister()

    def test_associated_payload_changed_rejected(self):
        (self.base / next(iter(lineage.METADATA.values()))).resolve().write_bytes(b'changed model')
        with self.assertRaisesRegex(lineage.Rejected, 'nonmetadata_content_changed'): self.preregister()

    def test_all_metadata_even_original_hash_match_is_validated(self):
        parent = copy.deepcopy(self.manifest)
        name = next(iter(lineage.METADATA)); raw = b'not valid metadata'
        (self.base / name).resolve().write_bytes(raw)
        for row in parent['files']:
            if row['path'] == name: row.update(bytes=len(raw), sha256=lineage.sha(raw))
        with self.assertRaisesRegex(lineage.Rejected, 'metadata_syntax'): lineage.audit(parent, self.base)

    def test_bad_metadata_etag_and_timestamp_rejected(self):
        name = next(iter(lineage.METADATA)); path = (self.base / name).resolve()
        fields = self.metadata_raw[name].decode().splitlines()
        for index, value in ((0, 'not-a-commit'), (1, '0' * 40), (2, 'nan'), (2, 'inf'), (2, '-1')):
            with self.subTest(index=index, value=value):
                changed = list(fields); changed[index] = value
                path.write_text('\n'.join(changed) + '\n')
                with self.assertRaises(lineage.Rejected): self.preregister()

    def test_missing_metadata_does_not_create_successor(self):
        (self.base / next(iter(lineage.METADATA))).resolve().unlink()
        with self.assertRaises(FileNotFoundError): self.preregister(execute=True)
        self.assertFalse(any(p.exists() for p in self.outputs.values()))

    def test_symlink_input_and_parent_escape_rejected(self):
        file = self.repo / 'fixtures/file1'; original = self.repo / 'saved'; file.rename(original); file.symlink_to(original)
        with self.assertRaisesRegex(lineage.Rejected, 'nonsymlink'): self.preregister()

    def test_duplicate_paths_and_removed_file_rejected(self):
        for mutate in (lambda x: x['files'].append(x['files'][0]), lambda x: x['files'].pop()):
            parent = copy.deepcopy(self.manifest); mutate(parent)
            with self.assertRaisesRegex(lineage.Rejected, '878'): lineage.audit(parent, self.base)

    def test_original_pin_tampering_and_helper_change_rejected(self):
        self.parent_inputs.write_bytes(self.parent_raw + b' ')
        with self.assertRaisesRegex(lineage.Rejected, 'immutable_parent_hash'): self.preregister()
        self.parent_inputs.write_bytes(self.parent_raw)
        with patch.object(lineage, 'HELPER_SHA', '0' * 64), self.assertRaisesRegex(lineage.Rejected, 'helper_changed'):
            self.preregister()

    def test_outputs_must_be_fresh_distinct_siblings(self):
        self.outputs['out_inputs'].write_text('untouched')
        with self.assertRaisesRegex(lineage.Rejected, 'fresh_distinct'): self.preregister(execute=True)
        self.assertEqual(self.outputs['out_inputs'].read_text(), 'untouched')
        self.outputs['out_inputs'].unlink()
        with self.assertRaisesRegex(lineage.Rejected, 'sibling'):
            self.preregister(execute=True, out_inputs=self.quality_dir / 'wrong-base.json')

    def test_no_commit_marker_after_partial_write(self):
        original = lineage.write_new
        def fail_second(path, raw):
            if path == self.outputs['out_envelope']: raise OSError('fixture write failure')
            return original(path, raw)
        with patch.object(lineage, 'write_new', side_effect=fail_second), self.assertRaises(OSError): self.preregister(execute=True)
        self.assertTrue(self.outputs['out_inputs'].exists()); self.assertFalse(self.outputs['receipt_path'].exists())

    def test_rehashed_envelope_widening_or_old_rebinding_still_rejected(self):
        result = self.preregister(execute=True)
        original_envelope = self.outputs['out_envelope'].read_bytes()
        original_receipt = self.outputs['receipt_path'].read_bytes()
        for mutation in ('bound', 'old_binding', 'reference_binding', 'helper', 'noise'):
            data = lineage.parse(original_envelope); receipt = lineage.parse(original_receipt)
            if mutation == 'bound': next(iter(data['global_bounds'].values()))['limit'] += 1
            elif mutation == 'old_binding': data['input_manifest_sha256'] = self.parent_sha
            elif mutation == 'reference_binding': data['references'][0]['input_manifest_sha256'] = result['successor_input_manifest_sha256']
            elif mutation == 'helper': data['helper_sha256'] = '0' * 64
            else: data['noise_method'] = 'widened noise'
            self.outputs['out_envelope'].write_bytes(lineage.encoded(data))
            receipt['successor_envelope_sha256'] = lineage.sha(lineage.encoded(data))
            self.outputs['receipt_path'].write_bytes(lineage.encoded(receipt))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(lineage.Rejected, 'beyond_metadata_binding'):
                self.verify({**result, 'receipt_sha256': lineage.sha(lineage.encoded(receipt))})

    def test_actual_files_changed_after_preregistration_rejected(self):
        result = self.preregister(execute=True)
        path = (self.base / next(iter(lineage.METADATA))).resolve()
        path.write_bytes(path.read_bytes().replace(b'2.0', b'3.0'))
        with self.assertRaisesRegex(lineage.Rejected, 'actual_inputs_changed'): self.verify(result)


if __name__ == '__main__':
    unittest.main()
