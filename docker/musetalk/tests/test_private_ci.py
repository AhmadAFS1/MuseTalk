"""Synthetic private-input transport contracts; no cloud calls or credentials."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import private_ci
import test_ci

entry = test_ci.entry


def urls():
    query = ("?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=900&X-Amz-Signature=" + "a" * 64
             + "&X-Amz-Credential=SYNTHETIC%2F20261009%2Fus-east-1%2Fs3%2Faws4_request")
    return {name: "https://" + private_ci.S3_HOST + "/docker-build-inputs/synthetic/" + name + query
            for name in private_ci.INPUT_NAMES}


class Response(io.BytesIO):
    status = 200
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


class PrivateCITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archives = {"native.tar.gz": entry(b"native"), "weights.tar.gz": entry(b"weights")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_urls_require_exact_inputs_approved_https_host_and_bounded_signature(self):
        private_ci.validate_urls(urls())
        bad_values = [urls()["weights.tar.gz"].replace("https:", "http:"),
                      urls()["weights.tar.gz"].replace(private_ci.S3_HOST, "fixture.invalid"),
                      urls()["weights.tar.gz"].replace("X-Amz-Expires=900", "X-Amz-Expires=7200"),
                      urls()["weights.tar.gz"].replace("/docker-build-inputs/", "/not-approved/"),
                      urls()["weights.tar.gz"] + "#fragment",
                      urls()["weights.tar.gz"].replace("us-east-1%2Fs3", "us-west-2%2Fs3"),
                      urls()["weights.tar.gz"].replace("https://", "https://synthetic:secret@")]
        for value in bad_values:
            invalid = urls()
            invalid["weights.tar.gz"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                private_ci.validate_urls(invalid)
        for invalid in ({}, {**urls(), "extra": "synthetic"}):
            with self.assertRaises(ValueError):
                private_ci.validate_urls(invalid)

    def test_complete_download_requires_hash_exact_size_and_no_overwrite(self):
        data = b"synthetic object"
        digest = hashlib.sha256(data).hexdigest()
        destination = self.root / "object"
        private_ci.download("synthetic", destination, digest, len(data), opener=lambda *a, **k: Response(data))
        self.assertEqual(destination.read_bytes(), data)
        with self.assertRaises(ValueError):
            private_ci.download("synthetic", destination, digest, len(data), opener=lambda *a, **k: Response(data))
        for index, (expected_hash, expected_size, limit) in enumerate((
                ("0" * 64, len(data), 100), (digest, len(data) + 1, 100), (digest, None, 3))):
            with self.subTest(index=index), self.assertRaises(ValueError):
                private_ci.download("synthetic", self.root / str(index), expected_hash, expected_size,
                                    limit=limit, opener=lambda *a, **k: Response(data))

    def test_redirects_and_partial_http_are_rejected(self):
        with self.assertRaises(ValueError):
            private_ci.NoRedirect().redirect_request(None, None, None, None, None, None)
        response = Response(b"data")
        response.status = 206
        with self.assertRaises(ValueError):
            private_ci.download("synthetic", self.root / "partial", hashlib.sha256(b"data").hexdigest(),
                                opener=lambda *a, **k: response)

    def test_private_archive_transport_accepts_real_large_weight_size_without_public_chunks(self):
        archives = {**self.archives, "weights.tar.gz": {"sha256": "a" * 64, "size_bytes": 3951486311}}
        private_ci.validate_archives({"archives": archives})
        for key, value in (("size_bytes", private_ci.MAX_ARCHIVE + 1), ("size_bytes", True),
                           ("sha256", "floating"), ("url", "synthetic-private-url")):
            invalid = {**archives, "weights.tar.gz": {**archives["weights.tar.gz"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                private_ci.validate_archives({"archives": invalid})

    def test_source_and_manifest_contracts_fail_before_large_downloads(self):
        metadata = test_ci.CITests.metadata(self)
        calls = []
        def fetch(url, destination, digest, size=None, **kwargs):
            calls.append(destination.name)
            self.assertEqual(destination.name, private_ci.ci.METADATA_NAME)
            destination.write_bytes(metadata.read_bytes())
            return destination
        with patch.object(private_ci.subprocess, "check_output", side_effect=["a" * 40, b""]), \
             patch.object(private_ci.shutil, "disk_usage", return_value=type("Usage", (), {"free": 70 * 1024**3})()), \
             patch.object(private_ci.release, "descriptor", side_effect=ValueError("synthetic mismatch")), \
             self.assertRaises(ValueError):
            private_ci.assemble(self.root, self.root / "work", "a" * 40,
                                private_ci.release.sha256(metadata), urls(), fetch)
        self.assertEqual(calls, [private_ci.ci.METADATA_NAME])

    def test_assembly_retains_hash_receipt_but_no_presigned_urls(self):
        metadata = test_ci.CITests.metadata(self)
        payloads = {private_ci.ci.METADATA_NAME: metadata.read_bytes(), "native.tar.gz": b"native",
                    "weights.tar.gz": b"weights"}
        def fetch(url, destination, digest, size=None, **kwargs):
            data = payloads[destination.name]
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
            if size is not None:
                self.assertEqual(len(data), size)
            destination.write_bytes(data)
            return destination
        with patch.object(private_ci.subprocess, "check_output", side_effect=["a" * 40, b""]), \
             patch.object(private_ci.shutil, "disk_usage", return_value=type("Usage", (), {"free": 70 * 1024**3})()), \
             patch.object(private_ci.release, "descriptor"), patch.object(private_ci.release, "verify_model_contract"), \
             patch.object(private_ci.release, "verify_source"):
            assets, manifest, reports = private_ci.assemble(self.root, self.root / "work", "a" * 40,
                                                           private_ci.release.sha256(metadata), urls(), fetch)
        receipt = (reports / "private-inputs.json").read_text()
        self.assertNotIn("X-Amz", receipt)
        self.assertNotIn(private_ci.S3_HOST, receipt)
        self.assertFalse(json.loads(receipt)["promotion_eligible"])
        self.assertEqual((assets / "weights.tar.gz").read_bytes(), b"weights")

    def test_private_and_cloud_secrets_removed_before_build_subprocesses(self):
        arguments = ["private_ci.py", "--root", str(self.root), "--work", str(self.root / "work"),
                     "--source-revision", "a" * 40, "--metadata-sha256", "b" * 64]
        def check_assemble(*args):
            for key in (private_ci.INPUT_ENV, "AWS_SECRET_ACCESS_KEY", "GH_TOKEN", "GITHUB_TOKEN"):
                self.assertNotIn(key, private_ci.os.environ)
            return self.root, {}, self.root
        with patch.object(sys, "argv", arguments), patch.dict(private_ci.os.environ, {
                private_ci.INPUT_ENV: json.dumps(urls()), "AWS_SECRET_ACCESS_KEY": "synthetic",
                "GH_TOKEN": "synthetic", "GITHUB_TOKEN": "synthetic"}), \
             patch.object(private_ci, "assemble", side_effect=check_assemble), patch.object(private_ci.ci, "build") as build:
            private_ci.main()
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
