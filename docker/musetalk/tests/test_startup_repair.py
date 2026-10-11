"""Synthetic overlay policy tests: no Docker, registry or usable models."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repair_startup as repair


class StartupRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.source = self.root / 'base', self.root / 'source'
        for directory, block in ((self.base, repair.OLD_BLOCK), (self.source, repair.NEW_BLOCK)):
            path = directory / repair.PATCH_PATH
            path.parent.mkdir(parents=True)
            path.write_text('before\n' + block + '\nafter\n')
        self.manifest = {
            'source_revision': repair.BASE_REVISION, 'status': 'candidate',
            'image_visibility': 'private', 'promotion_eligible': False,
            'source_files': {repair.PATCH_PATH: repair.fingerprint(self.base / repair.PATCH_PATH)},
            'model_files': {'synthetic-model': {'sha256': 'd' * 64}},
            'archives': {'synthetic-native': {'sha256': 'e' * 64}},
            'notices': {'synthetic-notice': {'sha256': 'f' * 64}},
            'candidate_reason': 'Synthetic unresolved gates',
        }
        for name in ('verify_source', 'verify_evidence'):
            mocked = patch.object(repair.release, name)
            mocked.start()
            self.addCleanup(mocked.stop)

    def run_repair(self):
        return repair.repaired_manifest(self.base, self.source, self.manifest, 'a' * 40)

    def test_preserves_assets_and_candidate_limits_without_mutating_base(self):
        original = copy.deepcopy(self.manifest)
        changed = self.run_repair()
        self.assertEqual(self.manifest, original)
        for field in ('model_files', 'archives', 'notices', 'candidate_reason', 'promotion_eligible'):
            self.assertEqual(changed[field], original[field])
        self.assertEqual(changed['source_revision'], 'a' * 40)
        self.assertEqual(changed['startup_repair']['gpu_validation'], 'NOT_RUN')
        self.assertEqual(changed['source_files'][repair.PATCH_PATH],
                         repair.fingerprint(self.source / repair.PATCH_PATH))

    def test_rejects_extra_script_change(self):
        path = self.source / repair.PATCH_PATH
        path.write_text(path.read_text() + 'unreviewed-command\n')
        with self.assertRaisesRegex(ValueError, 'Only the reviewed'):
            self.run_repair()

    def test_rejects_wrong_base_or_promotion(self):
        for field, value in (('source_revision', 'b' * 40), ('image_visibility', 'public'),
                             ('promotion_eligible', True), ('status', 'validated')):
            original = self.manifest[field]
            self.manifest[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_repair()
            self.manifest[field] = original

    def test_unchanged_source_must_match_original_hash(self):
        name = 'api_server.py'
        self.manifest['source_files'][name] = {'sha256': 'c' * 64, 'size_bytes': 1}
        (self.source / name).write_text('unreviewed source')
        with self.assertRaises(ValueError):
            self.run_repair()
