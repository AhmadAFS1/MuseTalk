#!/usr/bin/env python3
"""Read fixed S3 privacy settings with the operator's existing default CLI identity.

No object access, upload, bucket change, IAM change, or credential serialization.
The output is an operator assertion, not an AWS-signed statement. The consuming
operator must supply its SHA256 separately through a trusted invocation channel.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

import safe_capture

BUCKET = "lingua-musetalk-s3-storage"
REGION = "us-east-1"
OWNER = "211125449207"
FLAGS = ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")
SCHEMA = "musetalk_operator_bucket_privacy_v1"
MAX_AGE_SECONDS = 15 * 60
SHA = re.compile(r"[0-9a-f]{64}\Z")
READS = {"sts_identity", "bucket_public_access_block", "bucket_policy_status", "bucket_acl"}


class Invalid(RuntimeError):
    pass


def require(ok, code):
    if not ok:
        raise Invalid(code)


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cli_read(service, operation):
    endpoint = f"https://{service}.{REGION}.amazonaws.com"
    command = ["aws", "--profile", "default", "--region", REGION, "--endpoint-url", endpoint,
               "--no-cli-pager", "--cli-connect-timeout", "5", "--cli-read-timeout", "20",
               service, operation, "--output", "json"]
    if service == "s3api":
        command[command.index("--endpoint-url") + 1] = f"https://s3.{REGION}.amazonaws.com"
        command += ["--bucket", BUCKET, "--expected-bucket-owner", OWNER]
    # Retain the existing default CLI credential chain. Ignore custom endpoint
    # configuration and never echo environment, argv, SDK errors, or credentials.
    env = {**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off",
           "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true"}
    try:
        raw = safe_capture.capture(command, cwd=Path(__file__).parent, env=env,
                                   stage="operator_privacy_read", timeout_s=30)
    except safe_capture.CaptureFailure as exc:
        codes = {item["code"] for item in exc.record.get("diagnostics", [])}
        if (operation == "get-bucket-policy-status" and exc.record.get("failure") == "NONZERO_EXIT"
                and codes == {"AWS_BUCKET_POLICY_ABSENT"}):
            # Hash the normalized CLI error code, not credential-bearing stderr.
            value = {"Error": {"Code": "NoSuchBucketPolicy"}}
            return value, {"sha256": canonical_sha(value), "hash_basis": "canonical_cli_error_code"}
        raise Invalid("operator_privacy_cli_read_failed") from None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise Invalid("operator_privacy_cli_response_invalid") from None
    require(isinstance(value, dict), "operator_privacy_cli_response_invalid")
    return value, {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "hash_basis": "cli_stdout_utf8_stripped"}


def produce(read=cli_read, clock=utcnow):
    started = clock()
    identity, identity_hash = read("sts", "get-caller-identity")
    require(identity.get("Account") == OWNER, "operator_account_is_not_bucket_owner")
    arn = identity.get("Arn")
    require(isinstance(arn, str) and re.fullmatch(r"arn:aws:(?:iam|sts)::" + OWNER + r":[A-Za-z0-9+=,.@_:/-]+", arn),
            "operator_identity_invalid")
    block, block_hash = read("s3api", "get-public-access-block")
    flags = block.get("PublicAccessBlockConfiguration", {})
    require(isinstance(flags, dict) and all(flags.get(key) is True for key in FLAGS), "operator_public_access_block_not_all_true")
    policy, policy_hash = read("s3api", "get-bucket-policy-status")
    if policy.get("Error", {}).get("Code") == "NoSuchBucketPolicy":
        policy_result = "absent"
    else:
        require(policy.get("PolicyStatus", {}).get("IsPublic") is False, "operator_bucket_policy_not_proven_nonpublic")
        policy_result = "observed_nonpublic"
    acl, acl_hash = read("s3api", "get-bucket-acl")
    owner_id = acl.get("Owner", {}).get("ID")
    grants = acl.get("Grants")
    require(isinstance(owner_id, str) and SHA.fullmatch(owner_id), "operator_acl_owner_invalid")
    require(isinstance(grants, list) and grants, "operator_acl_grants_missing")
    require(all(isinstance(g, dict) and g.get("Grantee", {}).get("Type") == "CanonicalUser"
                and g["Grantee"].get("ID") == owner_id and g.get("Permission") in {
                    "FULL_CONTROL", "READ", "WRITE", "READ_ACP", "WRITE_ACP"} for g in grants),
            "operator_acl_not_owner_only")
    data = {"schema": SCHEMA, "scope": "fixed_bucket_private_model_delivery_only", "bucket": BUCKET,
            "region": REGION, "expected_owner": OWNER, "observed_at_utc": started.isoformat(),
            "expires_at_utc": (started + dt.timedelta(seconds=MAX_AGE_SECONDS)).isoformat(),
            "operator_account": OWNER, "operator_arn_sha256": hashlib.sha256(arn.encode()).hexdigest(),
            "public_access_block": {key: True for key in FLAGS}, "bucket_policy": policy_result,
            "bucket_acl": "owner_only_canonical_user", "cloud_mutations": False,
            "cli_reads": {"bucket_acl": acl_hash, "bucket_policy_status": policy_hash,
                          "bucket_public_access_block": block_hash, "sts_identity": identity_hash}}
    validate(data, clock())
    return data


def validate(data, current=None):
    require(isinstance(data, dict) and data.get("schema") == SCHEMA, "privacy_attestation_schema")
    require(data.get("scope") == "fixed_bucket_private_model_delivery_only" and data.get("cloud_mutations") is False,
            "privacy_attestation_scope")
    require(data.get("bucket") == BUCKET and data.get("region") == REGION and data.get("expected_owner") == OWNER
            and data.get("operator_account") == OWNER, "privacy_attestation_wrong_bucket_or_owner")
    require(SHA.fullmatch(str(data.get("operator_arn_sha256", ""))), "privacy_attestation_identity_hash")
    block = data.get("public_access_block")
    require(isinstance(block, dict) and set(block) == set(FLAGS) and all(block[key] is True for key in FLAGS),
            "privacy_attestation_flags_not_all_true")
    require(data.get("bucket_policy") in {"absent", "observed_nonpublic"}
            and data.get("bucket_acl") == "owner_only_canonical_user", "privacy_attestation_public_or_missing_proof")
    reads = data.get("cli_reads")
    require(isinstance(reads, dict) and set(reads) == READS, "privacy_attestation_cli_read_proof_missing")
    for key, item in reads.items():
        require(isinstance(item, dict) and set(item) == {"sha256", "hash_basis"}
                and SHA.fullmatch(str(item.get("sha256", ""))), "privacy_attestation_cli_read_hash")
        expected = "canonical_cli_error_code" if key == "bucket_policy_status" and data["bucket_policy"] == "absent" else "cli_stdout_utf8_stripped"
        require(item.get("hash_basis") == expected, "privacy_attestation_cli_read_hash_basis")
    try:
        observed = dt.datetime.fromisoformat(data["observed_at_utc"])
        expires = dt.datetime.fromisoformat(data["expires_at_utc"])
        current = current or utcnow()
        require(observed.utcoffset() == dt.timedelta(0) and expires.utcoffset() == dt.timedelta(0)
                and current.utcoffset() is not None, "privacy_attestation_timestamp_zone")
        age = (current - observed).total_seconds()
    except (ValueError, TypeError, KeyError):
        raise Invalid("privacy_attestation_timestamp_invalid") from None
    require(expires - observed == dt.timedelta(seconds=MAX_AGE_SECONDS), "privacy_attestation_expiry_invalid")
    require(0 <= age < MAX_AGE_SECONDS and current < expires, "privacy_attestation_stale_or_future")
    return {"observed_at_utc": observed.isoformat(), "expires_at_utc": expires.isoformat(), "age_seconds": round(age, 3)}


def load_bound(path, digest, current=None):
    require(SHA.fullmatch(str(digest or "")), "privacy_attestation_explicit_sha_required")
    path = Path(path)
    require(not path.is_symlink(), "privacy_attestation_symlink")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 64 * 1024, "privacy_attestation_file_size_or_type")
            raw = handle.read(64 * 1024 + 1)
    except OSError:
        raise Invalid("privacy_attestation_unreadable") from None
    require(hashlib.sha256(raw).hexdigest() == digest, "privacy_attestation_sha_mismatch")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise Invalid("privacy_attestation_json_invalid") from None
    validate(data, current)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = produce()
        raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": "read_only_privacy_attestation_written", "sha256": hashlib.sha256(raw).hexdigest(),
                          "expires_at_utc": data["expires_at_utc"], "cloud_mutations": False}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "code": str(exc) if isinstance(exc, Invalid) else "details_suppressed",
                          "cloud_mutations": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
