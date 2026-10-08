import tempfile
import unittest
from pathlib import Path
import preserve_live_diagnostic_evidence as helper


class PrivateTraceTests(unittest.TestCase):
    def test_pack_and_stream_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'data').mkdir()
            (base / 'data/trace.bin').write_bytes(b'fixed trace')
            rows = helper.inventory(base, ('data',))
            archive = base / 'archive.tar.gz'
            digest = helper.pack(base, rows, archive)
            helper.verify(archive, rows, digest)
            with self.assertRaises(ValueError):
                helper.pack(base, rows, archive)
            rows[0]['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                helper.verify(archive, rows, digest)

    def test_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'data').mkdir()
            (base / 'real').write_bytes(b'x')
            (base / 'data/link').symlink_to(base / 'real')
            with self.assertRaises(ValueError):
                helper.inventory(base, ('data',))


if __name__ == '__main__':
    unittest.main()
