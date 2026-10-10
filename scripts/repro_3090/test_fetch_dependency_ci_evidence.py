"""Synthetic archive safety tests; not CI build evidence."""
import hashlib
import io
import unittest
import zipfile

import fetch_dependency_ci_evidence as fetch


class DependencyArchiveTests(unittest.TestCase):
    def archive(self, layout, *, extra=None, symlink=False):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name in layout:
                if symlink:
                    info = zipfile.ZipInfo(name)
                    info.external_attr = 0o120777 << 16
                    archive.writestr(info, b"synthetic")
                else:
                    archive.writestr(name, b"synthetic")
            if extra:
                archive.writestr(extra, b"synthetic")
        raw = output.getvalue()
        return raw, hashlib.sha256(raw).hexdigest()

    def test_old_root_and_fixed_nested_layouts_have_exact_outputs(self):
        for layout in ({name: name for name in fetch.EXPECTED},
                       fetch.LAYOUTS["7a78e4a"], fetch.LAYOUTS["e234988"]):
            raw, digest = self.archive(layout)
            result = fetch.entries(raw, layout=layout, digest=digest)
            self.assertEqual(set(result), set(layout.values()))
            self.assertTrue(all(data == b"synthetic" for data in result.values()))

    def test_publication_and_build_receipts_do_not_collide(self):
        layout = fetch.LAYOUTS["e234988"]
        self.assertEqual(len(layout), 14)
        self.assertEqual(layout["musetalk-dependency-build/reports/result.json"], "result.json")
        self.assertEqual(layout["ghcr-dependency-publication/result.json"], "publication-result.json")

    def test_extra_members_wrong_digest_and_layout_are_rejected(self):
        layout = fetch.LAYOUTS["7a78e4a"]
        raw, digest = self.archive(layout, extra="../unexpected")
        with self.assertRaisesRegex(ValueError, "file_set"):
            fetch.entries(raw, layout=layout, digest=digest)
        raw, digest = self.archive(layout)
        with self.assertRaisesRegex(ValueError, "sha_mismatch"):
            fetch.entries(raw, layout=layout, digest="0" * 64)
        with self.assertRaisesRegex(ValueError, "file_set"):
            fetch.entries(raw, digest=digest)

    def test_zip_symlinks_and_unsafe_output_paths_are_rejected(self):
        layout = {name: name for name in fetch.EXPECTED}
        raw, digest = self.archive(layout, symlink=True)
        with self.assertRaisesRegex(ValueError, "unsafe_artifact_member"):
            fetch.entries(raw, layout=layout, digest=digest)
        raw, digest = self.archive({"source": "../escape"})
        with self.assertRaisesRegex(ValueError, "unsafe_output_layout"):
            fetch.entries(raw, layout={"source": "../escape"}, digest=digest)


if __name__ == "__main__":
    unittest.main()
