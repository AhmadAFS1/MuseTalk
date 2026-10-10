import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dependency_inventory


class DependencyInventoryTests(unittest.TestCase):
    def test_installed_native_notices_and_dpkg_source_are_fingerprinted_without_auth_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / 'opt/musetalk/venv/lib/python3.10/site-packages'
            dist = site / 'synthetic-1.0.dist-info'
            dist.mkdir(parents=True)
            (dist / 'METADATA').write_text('Name: synthetic\nVersion: 1.0\nLicense-Expression: MIT\n')
            (dist / 'LICENSE').write_text('synthetic notice')
            (dist / 'THIRD_PARTY_LICENSES.txt').write_text('synthetic bundled notices')
            (dist / 'direct_url.json').write_text('synthetic auth URL must never be reported')
            (site / 'synthetic.so').write_bytes(b'synthetic native library')
            (dist / 'RECORD').write_text('synthetic-1.0.dist-info/LICENSE,,\nsynthetic-1.0.dist-info/THIRD_PARTY_LICENSES.txt,,\nsynthetic-1.0.dist-info/direct_url.json,,\nsynthetic.so,,\nmissing.so,,\n../../../escape.so,,\n')
            notice = root / 'usr/share/doc/synthetic/copyright'
            notice.parent.mkdir(parents=True)
            notice.write_text('synthetic OS notice')
            result = dependency_inventory.inventory(root, site, 'synthetic:amd64\t1.0\tsynthetic-source\t1.0\n')
            item = result['pip_distributions'][0]
            self.assertEqual(item['declared_license'], 'MIT')
            self.assertEqual(item['native_library_files'][0]['sha256'], hashlib.sha256(b'synthetic native library').hexdigest())
            self.assertEqual(len(item['installed_notice_files']), 2)
            self.assertEqual(len(item['missing_record_native_files']), 1)
            self.assertEqual(result['dpkg_packages'][0]['source_package'], 'synthetic-source')
            self.assertEqual(len(result['os_notice_files']), 1)
            self.assertFalse(result['publication_review_accepted'])
            self.assertNotIn('auth URL', json.dumps(result))

    def test_unbounded_site_root_and_invalid_dpkg_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                dependency_inventory.inventory(root, root.parent, '')
            with self.assertRaises(ValueError):
                dependency_inventory.inventory(root, root, 'invalid row')
