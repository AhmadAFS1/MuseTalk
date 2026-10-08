"""Synthetic exact-target pruning checks; never touches installed TensorRT."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prune_diagnostic as prune


class PruningTests(unittest.TestCase):
    def test_only_exact_regular_pinned_resource_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / prune.RELATIVE
            path.parent.mkdir(parents=True)
            data = b"synthetic Windows resource"
            path.write_bytes(data)
            distribution = SimpleNamespace(version="10.3.0", locate_file=lambda _name: path)
            with patch.object(prune, "EXPECTED_BYTES", len(data)), \
                 patch.object(prune, "EXPECTED_SHA", hashlib.sha256(data).hexdigest()):
                found, item = prune.inspect(root, distribution)
                self.assertEqual(found, path)
                self.assertEqual(item["size_bytes"], len(data))
                self.assertTrue(path.exists())  # inspection is read-only
                for replacement in (b"x" * len(data), data[:-1]):
                    path.write_bytes(replacement)
                    with self.assertRaises(ValueError):
                        prune.inspect(root, distribution)
                path.write_bytes(data)
                distribution.version = "10.4.0"
                with self.assertRaises(ValueError):
                    prune.inspect(root, distribution)
                distribution.version = "10.3.0"
                elsewhere = root / "not-the-installed-package"
                elsewhere.write_bytes(data)
                distribution.locate_file = lambda _name: elsewhere
                with self.assertRaises(ValueError):
                    prune.inspect(root, distribution)
                path.unlink()
                path.symlink_to(elsewhere)
                with self.assertRaises(ValueError):
                    prune.inspect(root, distribution)

    def test_candidate_does_not_remove_linux_resources_or_become_release(self):
        self.assertIn("_win.so", prune.RELATIVE)
        source = (Path(__file__).resolve().parents[1] / "prune_diagnostic.py").read_text()
        self.assertEqual(source.count("path.unlink()"), 1)
        self.assertIn('"gpu_tested": False', source)
        self.assertIn('"promotion_eligible": False', source)


if __name__ == "__main__":
    unittest.main()
