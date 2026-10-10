import hashlib
import io
import unittest
import zipfile

import fetch_full_candidate_evidence as fetch


class FullEvidenceTests(unittest.TestCase):
    def archive(self, names, symlink=False):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            for name in names:
                item = zipfile.ZipInfo(name)
                if symlink:
                    item.external_attr = 0o120777 << 16
                archive.writestr(item, b'synthetic')
        raw = stream.getvalue()
        return raw, {'size_in_bytes': len(raw), 'digest': 'sha256:' + hashlib.sha256(raw).hexdigest()}

    def test_exact_layout_and_digest(self):
        raw, metadata = self.archive(fetch.LAYOUT)
        files = fetch.verified_zip(raw, metadata, fetch.LAYOUT)
        self.assertEqual(set(files), set(fetch.LAYOUT.values()))
        with self.assertRaises(ValueError):
            fetch.verified_zip(raw, {**metadata, 'digest': 'sha256:' + '0' * 64}, fetch.LAYOUT)

    def test_unknown_duplicate_and_symlink_rejected(self):
        for names, link in (([*fetch.LAYOUT, '../secret'], False),
                            ([*fetch.LAYOUT, next(iter(fetch.LAYOUT))], False), (fetch.LAYOUT, True)):
            raw, metadata = self.archive(names, symlink=link)
            with self.subTest(link=link), self.assertRaises(ValueError):
                fetch.verified_zip(raw, metadata, fetch.LAYOUT)


if __name__ == '__main__':
    unittest.main()
