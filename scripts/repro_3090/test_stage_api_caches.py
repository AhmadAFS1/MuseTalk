"""Synthetic CPU-only copy/integrity tests, not avatar/GPU compatibility proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_api_caches as stage


class CopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.source_root, self.target_root = root / 'source', root / 'target'
        self.target_root.mkdir()
        self.avatar_id = 'test_avatar_talking'
        self.base = self.source_root / self.avatar_id
        self.base.mkdir(parents=True)
        for name in stage.FIXED:
            (self.base / name).write_bytes(name.encode())
        (self.base / 'avator_info.json').write_text(json.dumps({'avatar_id': self.avatar_id, 'version': 'v15'}))
        for directory in ('full_imgs', 'mask'):
            (self.base / directory).mkdir()
            for i in range(2):
                (self.base / directory / f'{i:08d}.png').write_bytes(f'{directory}-{i}'.encode())
        def images(directory):
            digest = hashlib.sha256()
            for path in sorted((self.base / directory).glob('*.png')):
                digest.update((path.name + '\0' + stage.sha(path) + '\n').encode())
            return {'count': 2, 'filename_content_inventory_sha256': digest.hexdigest()}
        self.row = {'avatar_id': self.avatar_id, 'character': 'test', 'pose': 'talking', 'status': 'PASS',
                    'contents': {'metadata': {'sha256': stage.sha(self.base / 'avator_info.json'),
                                              'selected_fields': {'video_layout': 'single_video'}},
                                 'source_video': {'sha256': stage.sha(self.base / 'input_video.mp4')},
                                 'latents': {'sha256': stage.sha(self.base / 'latents.pt')},
                                 'coordinate_pickles': {name: {'sha256': stage.sha(self.base / name)} for name in stage.FIXED[3:]},
                                 'frames': images('full_imgs'), 'masks': images('mask')}}
        self.disk = patch.object(stage.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(100*1024**3, 0, 100*1024**3))
        self.disk.start()
        self.addCleanup(self.disk.stop)

    def run_copy(self, row=None):
        return stage.stage_one(row or self.row, self.source_root, self.target_root)

    def test_copy_separate_inodes_and_all_hashes_without_modifying_source(self):
        before = stage.inventory(self.base, self.row)
        result = self.run_copy()
        self.assertEqual(result['operation'], 'copied_required_files_exclusively')
        self.assertEqual(result['files'], 9)
        self.assertTrue(result['separate_inodes'])
        self.assertFalse(result['existing_files_overwritten'])
        self.assertEqual(stage.inventory(self.base, self.row), before)
        target = self.target_root / self.avatar_id
        self.assertEqual(stage.inventory(target, self.row), before)
        (target / 'latents.pt').write_bytes(b'API writes affect only the independent target')
        self.assertEqual(stage.inventory(self.base, self.row), before)

    def test_matching_existing_cache_is_verified_not_rewritten(self):
        self.run_copy()
        target = self.target_root / self.avatar_id / 'latents.pt'
        before = target.stat()
        result = self.run_copy()
        self.assertEqual(result['operation'], 'verified_existing_separate_cache')
        self.assertEqual((target.stat().st_ino, target.stat().st_mtime_ns), (before.st_ino, before.st_mtime_ns))

    def test_existing_corruption_refused_without_repair(self):
        self.run_copy()
        target = self.target_root / self.avatar_id / 'latents.pt'
        target.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            self.run_copy()
        self.assertEqual(target.read_bytes(), b'corrupt')

    def test_source_corruption_fails_before_target_creation(self):
        (self.base / 'full_imgs/00000000.png').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'audited_image_hash_mismatch'):
            self.run_copy()
        self.assertFalse((self.target_root / self.avatar_id).exists())

    def test_noncanonical_indices_refused(self):
        (self.base / 'mask/00000001.png').rename(self.base / 'mask/00000004.png')
        with self.assertRaisesRegex(ValueError, 'index_or_count'):
            self.run_copy()

    def test_symlink_input_refused_even_when_bytes_match(self):
        source = self.base / 'latents.pt'
        duplicate = self.base / 'outside'
        source.rename(duplicate)
        source.symlink_to(duplicate)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_copy()

    def test_hardlinked_existing_cache_refused(self):
        target = self.target_root / self.avatar_id
        shutil.copytree(self.base, target, copy_function=os.link)
        with self.assertRaisesRegex(ValueError, 'shared_inode'):
            self.run_copy()

    def test_low_disk_no_partial_copy(self):
        with patch.object(stage.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(0, 0, 10)):
            with self.assertRaisesRegex(ValueError, 'headroom'):
                self.run_copy()
        self.assertFalse((self.target_root / self.avatar_id).exists())

    def test_invalid_row_id_layout_and_status_refused(self):
        for key, value in [('avatar_id', '../escape'), ('status', 'INVALID')]:
            row = copy.deepcopy(self.row)
            row[key] = value
            with self.assertRaises(ValueError):
                self.run_copy(row)
        row = copy.deepcopy(self.row)
        row['contents']['metadata']['selected_fields']['video_layout'] = 'multiple_video'
        with self.assertRaisesRegex(ValueError, 'layout'):
            self.run_copy(row)

    def test_overlapping_roots_refused(self):
        for target in (self.source_root, self.base / 'target', self.source_root.parent):
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                stage.stage_one(self.row, self.source_root, target)


if __name__ == '__main__':
    unittest.main()
