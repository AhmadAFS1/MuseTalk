"""Synthetic CPU lineage checks; never GPU or quality acceptance."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import freeze_tracking_lineage as lineage


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.base = self.repo / "docs/fps_comparisons/run/harnesses"
        self.base.mkdir(parents=True)

    def row(self, name, raw):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": "../../../../" + name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def test_unchanged_inputs_remain_exact_and_missing_is_rejected(self):
        row = self.row("models/weight.bin", b"original")
        parent = {"files": [row], "s3_objects": []}
        actual, changes = lineage.derive(parent, self.base, self.repo)
        self.assertEqual(actual, parent)
        self.assertEqual(changes, [])
        (self.repo / "models/weight.bin").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            lineage.derive(parent, self.base, self.repo)

    def test_payload_source_and_unknown_worker_changes_fail_closed(self):
        for name in ("models/weight.bin", "character_factory/h3_avatar_workflow/chin.py", "scripts/chin_multistream/worker.py"):
            row = self.row(name, b"original")
            (self.repo / name).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "unapproved"):
                lineage.derive({"files": [row]}, self.base, self.repo)

    def test_metadata_requires_original_payload_and_valid_etag(self):
        payload = self.row("models/taesd/config.json", b"{}")
        name = "models/taesd/.cache/huggingface/download/config.json.metadata"
        before = self.row(name, b"old timestamped metadata")
        path = self.repo / name
        etag = hashlib.sha1(b"blob 2\0{}").hexdigest()
        path.write_text("1" * 40 + "\n" + etag + "\n1791496800.1\n")
        actual, changes = lineage.derive({"files": [before, payload]}, self.base, self.repo)
        self.assertEqual(len(actual["files"]), 2)
        self.assertEqual(len(changes), 1)
        self.assertEqual(actual["files"][1], payload)
        path.write_text("1" * 40 + "\n" + "2" * 40 + "\n1791496800.1\n")
        with self.assertRaisesRegex(ValueError, "etag"):
            lineage.derive({"files": [before, payload]}, self.base, self.repo)
        (self.repo / "models/taesd/config.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed model"):
            lineage.derive({"files": [before, payload]}, self.base, self.repo)

    def test_duplicate_paths_are_rejected(self):
        row = self.row("models/weight.bin", b"original")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            lineage.derive({"files": [row, row]}, self.base, self.repo)


if __name__ == "__main__":
    unittest.main()
