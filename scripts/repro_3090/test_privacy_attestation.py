"""CPU-only operator-attestation tests; no AWS requests or object operations."""
import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy_attestation as attestation
import safe_capture


def fixture_read(service, operation):
    rows = {
        "get-caller-identity": {"Account": attestation.OWNER, "Arn": f"arn:aws:iam::{attestation.OWNER}:user/operator",
                                "UserId": "DO_NOT_SERIALIZE_THIS_VALUE"},
        "get-public-access-block": {"PublicAccessBlockConfiguration": {key: True for key in attestation.FLAGS}},
        "get-bucket-policy-status": {"Error": {"Code": "NoSuchBucketPolicy"}},
        "get-bucket-acl": {"Owner": {"ID": "a" * 64}, "Grants": [
            {"Grantee": {"Type": "CanonicalUser", "ID": "a" * 64}, "Permission": "FULL_CONTROL"}]},
    }
    value = rows[operation]
    basis = "canonical_cli_error_code" if operation == "get-bucket-policy-status" else "cli_stdout_utf8_stripped"
    return value, {"sha256": attestation.canonical_sha(value), "hash_basis": basis}


def valid_fixture(current=None):
    current = current or attestation.utcnow()
    return attestation.produce(read=fixture_read, clock=lambda: current)


class AttestationTests(unittest.TestCase):
    def test_producer_reads_only_fixed_settings_and_identity(self):
        read = Mock(side_effect=fixture_read)
        data = attestation.produce(read=read)
        self.assertEqual([call.args for call in read.call_args_list], [
            ("sts", "get-caller-identity"), ("s3api", "get-public-access-block"),
            ("s3api", "get-bucket-policy-status"), ("s3api", "get-bucket-acl")])
        self.assertEqual(data["bucket"], attestation.BUCKET)
        self.assertEqual(set(data["cli_reads"]), attestation.READS)
        self.assertNotIn("DO_NOT_SERIALIZE", json.dumps(data))
        self.assertNotIn("user/operator", json.dumps(data))
        self.assertFalse(data["cloud_mutations"])

    def test_cli_uses_existing_default_identity_pinned_owner_endpoint_and_timeout(self):
        with patch.object(safe_capture, "capture", return_value='{"safe": true}') as capture:
            value, evidence = attestation.cli_read("s3api", "get-public-access-block")
        self.assertEqual(value, {"safe": True})
        self.assertEqual(evidence["sha256"], hashlib.sha256(b'{"safe": true}').hexdigest())
        command = capture.call_args.args[0]
        self.assertEqual(command[command.index("--profile") + 1], "default")
        self.assertEqual(command[command.index("--endpoint-url") + 1], "https://s3.us-east-1.amazonaws.com")
        self.assertEqual(command[command.index("--expected-bucket-owner") + 1], attestation.OWNER)
        self.assertEqual(command[command.index("--bucket") + 1], attestation.BUCKET)
        self.assertEqual(capture.call_args.kwargs["timeout_s"], 30)

    def test_only_specific_absent_policy_error_is_allowed(self):
        for failure, codes, allowed in (
                ("NONZERO_EXIT", ["AWS_BUCKET_POLICY_ABSENT"], True),
                ("TIMEOUT", ["AWS_BUCKET_POLICY_ABSENT"], False),
                ("NONZERO_EXIT", ["AWS_ACCESS_DENIED"], False),
                ("NONZERO_EXIT", ["AWS_BUCKET_POLICY_ABSENT", "AWS_ACCESS_DENIED"], False)):
            error = safe_capture.CaptureFailure({"stage": "operator_privacy_read", "failure": failure,
                                                "diagnostics": [{"code": code} for code in codes]})
            with self.subTest(failure=failure, codes=codes), patch.object(safe_capture, "capture", side_effect=error):
                if allowed:
                    value, evidence = attestation.cli_read("s3api", "get-bucket-policy-status")
                    self.assertEqual(value["Error"]["Code"], "NoSuchBucketPolicy")
                    self.assertEqual(evidence["hash_basis"], "canonical_cli_error_code")
                    with self.assertRaises(attestation.Invalid):
                        attestation.cli_read("s3api", "get-public-access-block")
                else:
                    with self.assertRaises(attestation.Invalid):
                        attestation.cli_read("s3api", "get-bucket-policy-status")

    def test_producer_fails_if_any_read_is_missing(self):
        for failed in ("get-caller-identity", "get-public-access-block", "get-bucket-policy-status", "get-bucket-acl"):
            def read(service, operation):
                if operation == failed:
                    raise attestation.Invalid("operator_privacy_cli_read_failed")
                return fixture_read(service, operation)
            with self.subTest(failed=failed), self.assertRaises(attestation.Invalid):
                attestation.produce(read=read)

    def test_producer_rejects_wrong_identity_public_policy_or_acl(self):
        replacements = {
            "get-caller-identity": {"Account": "000000000000", "Arn": "untrusted"},
            "get-public-access-block": {"PublicAccessBlockConfiguration": {key: False for key in attestation.FLAGS}},
            "get-bucket-policy-status": {"PolicyStatus": {"IsPublic": True}},
            "get-bucket-acl": {"Owner": {"ID": "a" * 64}, "Grants": [
                {"Grantee": {"Type": "Group", "URI": "http://acs.amazonaws.com/groups/global/AllUsers"}, "Permission": "READ"}]},
        }
        for failed, replacement in replacements.items():
            def read(service, operation):
                value, evidence = fixture_read(service, operation)
                return (replacement, evidence) if operation == failed else (value, evidence)
            with self.subTest(failed=failed), self.assertRaises(attestation.Invalid):
                attestation.produce(read=read)

    def test_freshness_is_strictly_less_than_fifteen_minutes(self):
        start = dt.datetime(2026, 10, 8, 8, 37, tzinfo=dt.timezone.utc)
        data = valid_fixture(start)
        for seconds in (0, 899.999):
            attestation.validate(data, start + dt.timedelta(seconds=seconds))
        for seconds in (-1, 900, 901):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(attestation.Invalid, "stale_or_future"):
                attestation.validate(data, start + dt.timedelta(seconds=seconds))

    def test_all_boolean_flags_read_hashes_and_identity_are_required(self):
        original = valid_fixture()
        mutations = [lambda d: d.update(bucket="other"), lambda d: d.update(expected_owner="000000000000"),
                     lambda d: d.update(operator_account="000000000000"), lambda d: d.update(region="other"),
                     lambda d: d["cli_reads"].pop("bucket_acl"), lambda d: d["cli_reads"]["sts_identity"].update(sha256="bad"),
                     lambda d: d.update(bucket_policy="public"), lambda d: d.update(bucket_acl="unavailable"),
                     lambda d: d.update(observed_at_utc="2026-10-08T08:37:00"),
                     lambda d: d.update(expires_at_utc=d["observed_at_utc"])]
        for flag in attestation.FLAGS:
            for value in (False, "true", 1, None):
                mutations.append(lambda d, flag=flag, value=value: d["public_access_block"].update({flag: value}))
        for mutate in mutations:
            data = copy.deepcopy(original)
            mutate(data)
            with self.assertRaises(attestation.Invalid):
                attestation.validate(data)

    def test_explicit_digest_tampering_symlink_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            raw = json.dumps(valid_fixture()).encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(attestation.load_bound(path, digest)["schema"], attestation.SCHEMA)
            for bad in (None, "", "0" * 64):
                with self.assertRaises(attestation.Invalid):
                    attestation.load_bound(path, bad)
            link = Path(tmp) / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(attestation.Invalid, "symlink"):
                attestation.load_bound(link, digest)
            path.write_bytes(raw + b" ")
            with self.assertRaisesRegex(attestation.Invalid, "sha_mismatch"):
                attestation.load_bound(path, digest)
            with self.assertRaisesRegex(attestation.Invalid, "unreadable"):
                attestation.load_bound(Path(tmp) / "missing.json", digest)

    def test_main_writes_private_new_file_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            with patch.object(attestation, "produce", return_value=valid_fixture()), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(attestation.main(["--out", str(path)]), 0)
                first = path.read_bytes()
                self.assertEqual(attestation.main(["--out", str(path)]), 2)
            result = json.loads(output.getvalue().splitlines()[0])
            self.assertEqual(result["sha256"], hashlib.sha256(first).hexdigest())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
