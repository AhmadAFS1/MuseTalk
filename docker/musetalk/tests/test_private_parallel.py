"""Synthetic transport checks; no network, credentials, Docker or model load."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release


class PrivateParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target = Path(self.tmp.name) / "payload"
        self.data = b"synthetic pinned payload"
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.entry = {"size_bytes": len(self.data), "sha256": self.digest, "source": {
            "bucket": "private-fixture", "key": "sha256/" + self.digest + "/payload",
            "region": "us-east-1", "expected_owner": "211125449207"}}
        self.head = {"ContentLength": len(self.data), "Metadata": {"sha256": self.digest}, "VersionId": "v1"}
        self.s3 = Mock()
        self.s3.head_object.return_value = self.head
        self.s3.download_file.side_effect = lambda _bucket, _key, path, **_kw: Path(path).write_bytes(self.data)
        self.boto = SimpleNamespace(client=Mock(return_value=self.s3))
        self.config = SimpleNamespace(Config=Mock(return_value="fixture-config"))
        self.transfer = SimpleNamespace(TransferConfig=Mock(return_value="fixture-transfer"))

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self):
        with patch.dict(sys.modules, {"boto3": self.boto, "botocore.config": self.config,
                                      "boto3.s3.transfer": self.transfer}), \
             patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "fixture-id", "AWS_SECRET_ACCESS_KEY": "fixture-secret",
                                     "AWS_ENDPOINT_URL": "https://untrusted.invalid"}, clear=True):
            return release.fetch_private_model(self.entry, self.target)

    def test_current_version_uses_owner_and_sha_but_no_version_permission(self):
        result = self.fetch()
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(result["version_id_requested"])
        self.assertEqual(result["observed_version_id"], "v1")
        self.assertTrue(result["head_version_unchanged"])
        for call in self.s3.head_object.call_args_list:
            self.assertEqual(call.kwargs["ExpectedBucketOwner"], "211125449207")
            self.assertNotIn("VersionId", call.kwargs)
        self.assertEqual(self.s3.download_file.call_args.kwargs["ExtraArgs"], {"ExpectedBucketOwner": "211125449207"})
        self.assertEqual(self.transfer.TransferConfig.call_args.kwargs,
                         {"max_concurrency": 4, "multipart_chunksize": 8 * 1024**2, "num_download_attempts": 1})
        self.assertTrue(self.config.Config.call_args.kwargs["ignore_configured_endpoint_urls"])
        self.s3.put_object.assert_not_called()

    def test_explicit_version_is_not_silently_downgraded(self):
        self.entry["source"]["version_id"] = "v1"
        self.assertTrue(self.fetch()["version_id_requested"])
        self.assertEqual(self.s3.download_file.call_args.kwargs["ExtraArgs"]["VersionId"], "v1")
        for call in self.s3.head_object.call_args_list:
            self.assertEqual(call.kwargs["VersionId"], "v1")
        self.head["VersionId"] = "changed"
        self.s3.download_file.reset_mock()
        with self.assertRaises(ValueError):
            self.fetch()
        self.s3.download_file.assert_not_called()

    def test_bad_content_or_changed_version_fails_closed(self):
        for payload in (b"x" * len(self.data), self.data[:-1], self.data + b"x"):
            self.s3.download_file.side_effect = lambda _b, _k, path, **_kw: Path(path).write_bytes(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.fetch()
        self.s3.download_file.side_effect = lambda _b, _k, path, **_kw: Path(path).write_bytes(self.data)
        for after in ({**self.head, "VersionId": "changed"}, {**self.head, "ContentLength": 1},
                      {**self.head, "Metadata": {"sha256": "0" * 64}}):
            self.s3.head_object.side_effect = [self.head, after]
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.fetch()

    def test_unproven_head_or_missing_credentials_stops_before_download(self):
        for head in ({**self.head, "VersionId": None}, {**self.head, "Metadata": {}}, {**self.head, "ContentLength": 1}):
            self.s3.head_object.return_value = head
            self.s3.download_file.reset_mock()
            with self.subTest(head=head), self.assertRaises(ValueError):
                self.fetch()
            self.s3.download_file.assert_not_called()
        with patch.dict(sys.modules, {"boto3": self.boto, "botocore.config": self.config,
                                      "boto3.s3.transfer": self.transfer}), patch.dict(os.environ, {}, clear=True):
            self.boto.client.reset_mock()
            with self.assertRaises(ValueError):
                release.fetch_private_model(self.entry, self.target)
            self.boto.client.assert_not_called()
        self.s3.put_object.assert_not_called()

    def test_sdk_details_are_suppressed_and_no_manual_retry(self):
        self.s3.download_file.side_effect = RuntimeError("synthetic-secret-value-and-signed-url")
        with self.assertRaises(ValueError) as raised:
            self.fetch()
        self.assertNotIn("synthetic-secret", str(raised.exception))
        self.assertEqual(self.s3.download_file.call_count, 1)
        self.s3.put_object.assert_not_called()


if __name__ == "__main__":
    unittest.main()
