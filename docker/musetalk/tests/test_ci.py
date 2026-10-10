"""Synthetic public-asset transport tests; no network, Docker, credentials or GPU."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci


def entry(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


class CITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.payloads = {"native.tar.gz": b"native", "weights.tar.gz": b"weights"}
        self.archives = {name: entry(value) for name, value in self.payloads.items()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_request_rejects_injection_or_floating_identity(self):
        valid = ["a" * 40, "r5-native-candidate-20261008", "b" * 64, "candidate"]
        ci.require_inputs(*valid)
        for index, value in ((0, "main"), (1, "--repo=other"), (1, "release; command"), (2, "latest"), (3, "promoted")):
            invalid = valid.copy()
            invalid[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                ci.require_inputs(*invalid)

    def test_build_passes_exact_manifest_runtime_base_not_a_floating_default(self):
        for use_runtime in (False, True):
            work = self.root / str(use_runtime)
            reports = work / "reports"
            reports.mkdir(parents=True)
            m = {"source_revision": "a" * 40, "status": "candidate",
                 "cuda_base": ci.release.CUDA_DEVEL_BASE, "apt_packages": ["python3=3.10.6-1"]}
            if use_runtime:
                m["cuda_runtime_base"] = ci.release.CUDA_RUNTIME_BASE
            expected = ci.release.CUDA_RUNTIME_BASE if use_runtime else ci.release.CUDA_DEVEL_BASE
            with self.subTest(use_runtime=use_runtime), patch.object(ci.subprocess, "run") as run, \
                 patch("builtins.print"):
                ci.build(self.root, self.root / "assets", m, work, reports)
                command = next(c.args[0] for c in run.call_args_list if c.args[0][:3] == ["docker", "buildx", "build"])
                self.assertIn("CUDA_RUNTIME_BASE=" + expected, command)
                self.assertIn("CUDA_BASE=" + ci.release.CUDA_DEVEL_BASE, command)
            result = json.loads((reports / "build-result.json").read_text())
            self.assertEqual(result["cuda_runtime_base"], expected)
            self.assertFalse(result["promotion_eligible"])

    def test_parts_are_ordered_size_and_digest_pinned(self):
        m = {"archives": self.archives}
        self.assertEqual(ci.transport_parts(m)["native.tar.gz"][0]["name"], "native.tar.gz")
        parts = [{"name": f"native.tar.gz.part{i:03d}", **entry(data)} for i, data in enumerate((b"nat", b"ive"))]
        m["github_release_assets"] = {"native.tar.gz": parts}
        ci.transport_parts(m)
        for key, value in (("name", "../native.tar.gz"), ("size_bytes", 2 * 1024**3), ("sha256", "floating")):
            bad = copy.deepcopy(m)
            bad["github_release_assets"]["native.tar.gz"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.transport_parts(bad)

    def test_assembly_verifies_parts_and_complete_archive(self):
        pieces = {"native.tar.gz.part000": b"nat", "native.tar.gz.part001": b"ive", "weights.tar.gz": b"weights"}
        m = {"archives": self.archives, "github_release_assets": {"native.tar.gz": [
            {"name": name, **entry(data)} for name, data in pieces.items() if name.startswith("native")]}}
        assets, scratch = self.root / "assets", self.root / "scratch"
        assets.mkdir()
        scratch.mkdir()
        def fetch(_tag, name, directory):
            path = directory / name
            path.write_bytes(pieces[name])
            return path
        ci.assemble_archives(m, "fixture", assets, scratch, fetch)
        self.assertEqual((assets / "native.tar.gz").read_bytes(), b"native")
        self.assertEqual(list(scratch.iterdir()), [])
        with self.assertRaises(ValueError):
            ci.assemble_archives(m, "fixture", assets, scratch, fetch)

    def test_corrupt_download_cannot_be_assembled(self):
        assets, scratch = self.root / "assets", self.root / "scratch"
        assets.mkdir()
        scratch.mkdir()
        def fetch(_tag, name, directory):
            path = directory / name
            path.write_bytes(b"wrong")
            return path
        with self.assertRaises(ValueError):
            ci.assemble_archives({"archives": self.archives}, "fixture", assets, scratch, fetch)

    def metadata(self, extra=None, manifest_changes=None):
        files = {"licenses/fixture.txt": b"Synthetic fixture; no redistribution claim"}
        evidence = {
            "evidence/quality.json": {"decision": "incomplete", "strict_original_gates": "NOT_RUN",
                                       "bundle_sha256": self.archives["native.tar.gz"]["sha256"]},
            "evidence/aggregate.json": {"gpu": "NVIDIA GeForce RTX 3090", "status": "NOT_RUN",
                                         "limitation": "Synthetic fixture; no GPU evidence",
                                         "bundle_sha256": self.archives["native.tar.gz"]["sha256"]},
        }
        if (manifest_changes or {}).get("image_visibility") == "private":
            evidence["evidence/packaging.json"] = {
                "schema": "musetalk_private_packaging_findings_v1", "scope": "private_deployment",
                "source_revision": "a" * 40, "bundle_sha256": self.archives["native.tar.gz"]["sha256"],
                "public_redistribution_reviewed": True, "blanket_use_rights_clearance": False,
                "notice_policy": "preserve_bundled_and_model_notices", "remaining_findings": ["Synthetic only"]}
            manifest_changes = {**manifest_changes, "packaging_review_file": "evidence/packaging.json"}
        files.update({name: json.dumps(data).encode() for name, data in evidence.items()})
        m = {"schema": ci.release.SCHEMA, "status": "candidate", "promotion_eligible": False,
             "candidate_reason": "Synthetic test only", "source_revision": "a" * 40,
             "cuda_base": "fixture.invalid/cuda@sha256:" + "b" * 64, "platform": "linux/amd64", "matrix": "cu121",
             "bundle_name": "rtx3090-r5-srcg50-int8", "redistribution_reviewed": True,
             "avatar_prep": True, "kokoro": False, "vp8_encoder": "native",
             "source_files": {"api_server.py": {}}, "model_files": {
                 "models/fixture/file": {"public_redistribution": True, "license_id": "fixture"}},
             "notices": {"licenses/fixture.txt": entry(files["licenses/fixture.txt"])},
             "evidence": {name: entry(files[name]) for name in evidence}, "archives": self.archives,
             "bundle_manifest_sha256": "c" * 64,
             "apt_packages": [name + "=1.0" for name in sorted(ci.release.REQUIRED_APT)],
             "quality_decision_file": "evidence/quality.json", "aggregate_acceptance_file": "evidence/aggregate.json"}
        m.update(manifest_changes or {})
        files["release.json"] = json.dumps(m).encode()
        files.update(extra or {})
        path = self.root / "metadata.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        return path

    def test_metadata_is_separately_pinned_and_exactly_allowlisted(self):
        archive = self.metadata()
        ci.read_metadata(archive, ci.release.sha256(archive), self.root / "release", "a" * 40, "candidate")
        with self.assertRaises(ValueError):
            ci.read_metadata(archive, "0" * 64, self.root / "wrong", "a" * 40, "candidate")
        archive = self.metadata({"evidence/unlisted.json": b"{}"})
        with self.assertRaises(ValueError):
            ci.read_metadata(archive, ci.release.sha256(archive), self.root / "extra", "a" * 40, "candidate")

    def test_metadata_cannot_write_source_or_escape(self):
        for extra in ({"scripts/installer.sh": b"bad"}, {"../escape": b"bad"}):
            archive = self.metadata(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                ci.read_metadata(archive, ci.release.sha256(archive), self.root / "release", "a" * 40, "candidate")
        self.assertFalse((self.root / "release").exists())

    def test_private_metadata_rejected_by_public_transport(self):
        archive = self.metadata(manifest_changes={"image_visibility": "private"})
        with self.assertRaisesRegex(ValueError, "Private"):
            ci.read_metadata(archive, ci.release.sha256(archive), self.root / "private", "a" * 40, "candidate")
        ci.read_metadata(archive, ci.release.sha256(archive), self.root / "accepted", "a" * 40,
                         "candidate", private_transport=True)

    def test_download_has_fixed_repository_and_no_shell(self):
        def complete(command, **kwargs):
            self.assertNotIn("shell", kwargs)
            self.assertEqual(command[:4], ["gh", "release", "download", "fixture"])
            self.assertEqual(command[4:6], ["--repo", "AhmadAFS1/MuseTalk"])
            (self.root / "native.tar.gz").write_bytes(b"data")
        with patch.object(ci.subprocess, "run", side_effect=complete):
            self.assertEqual(ci.download("fixture", "native.tar.gz", self.root), self.root / "native.tar.gz")


if __name__ == "__main__":
    unittest.main()
