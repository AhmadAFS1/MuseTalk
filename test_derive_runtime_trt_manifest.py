#!/usr/bin/env python3
"""Mock tests for scripts/derive_runtime_trt_manifest.py (fixture wheels only)."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
import derive_runtime_trt_manifest as d
from dependency_artifacts import digest, identity, validate_manifest
from fetch_dependency_wheels import sha256_file

FILENAME = "tensorrt_cu12_libs-10.3.0-py2.py3-none-manylinux_2_17_x86_64.whl"
DIST = "tensorrt_cu12_libs-10.3.0.dist-info"


def fixture_wheel(path):
    members = {
        "tensorrt_libs/__init__.py": b"# fixture\n",
        "tensorrt_libs/libnvinfer.so.10": b"runtime" * 100,
        "tensorrt_libs/libnvinfer_plugin.so.10": b"plugin" * 50,
        "tensorrt_libs/libnvinfer_builder_resource.so.10.3.0": b"builder" * 1000,
        "tensorrt_libs/libnvinfer_builder_resource_win.so.10.3.0": b"winbuilder" * 1000,
        DIST + "/METADATA": b"Metadata-Version: 2.1\nName: tensorrt-cu12-libs\nVersion: 10.3.0\n",
        DIST + "/WHEEL": b"Wheel-Version: 1.0\nTag: py2-none-manylinux_2_17_x86_64\n",
    }
    rows = [(n, d.record_hash(b), str(len(b))) for n, b in members.items()] + [(DIST + "/RECORD", "", "")]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in members.items():
            z.writestr(zipfile.ZipInfo(n, date_time=(2024, 8, 1, 0, 0, 0)), b)
        z.writestr(zipfile.ZipInfo(DIST + "/RECORD", date_time=(2024, 8, 1, 0, 0, 0)),
                   "".join(",".join(r) + "\n" for r in rows))
    return path


class RepackTests(unittest.TestCase):
    def test_repack_drops_only_builder_resources_and_rewrites_record(self):
        with tempfile.TemporaryDirectory() as folder:
            src = fixture_wheel(Path(folder) / FILENAME)
            out = Path(folder) / "out" / FILENAME
            out.parent.mkdir()
            removed = d.repack(src, out, sha256_file(src))
            self.assertEqual(sorted(r["path"] for r in removed),
                             ["tensorrt_libs/libnvinfer_builder_resource.so.10.3.0",
                              "tensorrt_libs/libnvinfer_builder_resource_win.so.10.3.0"])
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
                self.assertNotIn("tensorrt_libs/libnvinfer_builder_resource.so.10.3.0", names)
                self.assertIn("tensorrt_libs/libnvinfer.so.10", names)
                self.assertEqual(names[-1], DIST + "/RECORD")
                rows = list(csv.reader(io.StringIO(z.read(DIST + "/RECORD").decode())))
                for path, digest_text, size in rows:
                    if path.endswith("/RECORD"):
                        continue
                    data = z.read(path)
                    self.assertEqual((digest_text, int(size)), (d.record_hash(data), len(data)))
                self.assertEqual({r[0] for r in rows}, set(names))
                marker = json.loads(z.read(DIST + "/MUSETALK_RUNTIME_ONLY"))
                self.assertEqual(marker["original_sha256"], sha256_file(src))

    def test_wrong_input_hash_or_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            src = fixture_wheel(Path(folder) / FILENAME)
            with self.assertRaises(ValueError):
                d.repack(src, Path(folder) / "x.whl", "a" * 64)
            existing = Path(folder) / "exists.whl"
            existing.write_bytes(b"x")
            with self.assertRaises(ValueError):
                d.repack(src, existing, sha256_file(src))

    def test_derived_manifest_is_valid_and_marks_a_build_row(self):
        manifest = json.loads((REPO / "docs/dependency_optimization_20261015/wheelhouse-manifest-a2.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            src = fixture_wheel(Path(folder) / FILENAME)
            row = next(a for a in manifest["artifacts"] if a["name"] == "tensorrt-cu12-libs")
            row.update(sha256=sha256_file(src), size_bytes=src.stat().st_size)
            manifest["artifact_id"] = digest(identity(manifest))
            facts = {"python": "3.10.12", "system": "Linux", "machine": "x86_64", "glibc": "2.35"}
            receipt = d.derive(manifest, src, Path(folder) / "derived", facts)
            derived = json.loads((Path(folder) / "derived/manifest.json").read_text())
            validate_manifest(derived)
            self.assertNotEqual(derived["artifact_id"], manifest["artifact_id"])
            trt = next(a for a in derived["artifacts"] if a["name"] == "tensorrt-cu12-libs")
            self.assertEqual(trt["source"], "build:" + row["sha256"])
            self.assertEqual(trt["build_provenance"]["input_source"], row["source"])
            self.assertEqual(receipt["build_rows"], 5)
            self.assertIn("tensorrt-cu12-libs===10.3.0", (Path(folder) / "derived/full.lock").read_text())
            # Everything else is untouched.
            others = lambda m: [a for a in m["artifacts"] if a["name"] != "tensorrt-cu12-libs"]
            self.assertEqual(others(derived), others(manifest))


if __name__ == "__main__":
    unittest.main()
