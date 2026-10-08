"""Synthetic CPU checks; no network, credentials, or model deserialization."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_private_delivery as diagnostic


class ReadTests(unittest.TestCase):
    def test_prefix_exactness_version_owner_and_body_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source"
            data = b"test" * 600
            path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            entry = {"size_bytes": len(data), "sha256": sha, "source": {
                "bucket": diagnostic.pins.BUCKET, "expected_owner": diagnostic.pins.OWNER,
                "key": diagnostic.pins.PREFIX + "/" + sha + "/" + Path(diagnostic.MODEL).name}}
            for ranged in (False, True):
                body = io.BytesIO(data)
                s3 = Mock()
                s3.get_object.return_value = {"ContentLength": len(data), "VersionId": "v1", "Body": body,
                                             "ContentRange": f"bytes 0-{len(data)-1}/{len(data)}"}
                result = diagnostic.read_probe(s3, entry, "v1", path, ranged=ranged)
                self.assertEqual(result["bytes_verified"], len(data) if ranged else 1024)
                self.assertFalse(result["whole_object_verified"])
                self.assertTrue(body.closed)
                self.assertEqual(s3.get_object.call_args.kwargs["VersionId"], "v1")
                self.assertEqual(s3.get_object.call_args.kwargs["ExpectedBucketOwner"], diagnostic.pins.OWNER)
                s3.put_object.assert_not_called()
            for replacement in ({"VersionId": "wrong"}, {"ContentRange": "wrong"}, {"Body": io.BytesIO(b"bad")}):
                s3.get_object.return_value = {"ContentLength": len(data), "VersionId": "v1", "Body": io.BytesIO(data),
                                             "ContentRange": f"bytes 0-{len(data)-1}/{len(data)}", **replacement}
                with self.assertRaises(diagnostic.pins.Invalid):
                    diagnostic.read_probe(s3, entry, "v1", path, ranged=True)

    def test_normal_get_omits_version_but_checks_returned_version_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source"
            data = b"test" * 600
            path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            entry = {"size_bytes": len(data), "sha256": sha, "source": {
                "bucket": diagnostic.pins.BUCKET, "expected_owner": diagnostic.pins.OWNER,
                "key": diagnostic.pins.PREFIX + "/" + sha + "/" + Path(diagnostic.MODEL).name}}
            s3 = Mock()
            s3.get_object.return_value = {"ContentLength": len(data), "VersionId": "v1", "Body": io.BytesIO(data),
                                         "ContentRange": f"bytes 0-{len(data)-1}/{len(data)}"}
            result = diagnostic.read_probe(s3, entry, "v1", path, ranged=True, versioned=False)
            self.assertNotIn("VersionId", s3.get_object.call_args.kwargs)
            self.assertEqual(s3.get_object.call_args.kwargs["ExpectedBucketOwner"], diagnostic.pins.OWNER)
            self.assertFalse(result["version_id_requested"])
            self.assertTrue(result["response_version_matches_head"])
            s3.get_object.return_value.update(VersionId="changed", Body=io.BytesIO(data))
            with self.assertRaises(diagnostic.pins.Invalid):
                diagnostic.read_probe(s3, entry, "v1", path, ranged=True, versioned=False)
            self.assertTrue(s3.get_object.return_value["Body"].closed)

    def test_foreign_object_or_missing_version_rejected(self):
        entry = {"sha256": "a" * 64, "source": {"bucket": "foreign", "expected_owner": diagnostic.pins.OWNER,
                                                   "key": "foreign"}}
        with self.assertRaises(diagnostic.pins.Invalid):
            diagnostic.arguments(entry, "v1")
        entry["source"].update(bucket=diagnostic.pins.BUCKET,
                              key=diagnostic.pins.PREFIX + "/" + entry["sha256"] + "/" + Path(diagnostic.MODEL).name)
        with self.assertRaises(diagnostic.pins.Invalid):
            diagnostic.arguments(entry, None)


if __name__ == "__main__":
    unittest.main()
