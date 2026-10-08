#!/usr/bin/env python3
"""Plan, or explicitly persist five pinned private inputs without overwriting S3.

Run through the owned-worker credential bridge after benchmark work has stopped.
No model deserialization, GPU work, public ACL, bucket/IAM change, or deletion.
Private delivery does not clear unresolved model usage or redistribution rights.
Child output may be suppressed: every operation records safe facts in --out.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys

import privacy_attestation

MODEL_PATHS = {
    "models/syncnet/latentsync_syncnet.pt",
    "models/face-parse-bisent/79999_iter.pth",
    "models/face-parse-bisent/resnet18-5c106cde.pth",
    "models/auxiliary/s3fd-619a316812.pth",
    "models/face_detection/s3fd.pth",
}
BUCKET = privacy_attestation.BUCKET
REGION = privacy_attestation.REGION
OWNER = privacy_attestation.OWNER
PREFIX = "trt-artifacts/private-runtime-models/sha256"
WORKER_ROOT = "/workspace/MuseTalk"
WORKER_HOSTNAME = "a830e00ce20c"
INSTANCE_ID = "54798270"
QUALITY_MANIFEST_SHA256 = "6d3ab6ef31605c2231605e03042e82361a27112589ef6d7f6f8ff8f4b016eea5"
SHA = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_GROUPS = {"http://acs.amazonaws.com/groups/global/AllUsers",
                 "http://acs.amazonaws.com/groups/global/AuthenticatedUsers"}
PUBLIC_ACCESS_BLOCK_FLAGS = privacy_attestation.FLAGS


class Invalid(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise Invalid(code)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def hash_stream(stream, expected_size=None):
    digest = hashlib.sha256()
    count = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        count += len(chunk)
        require(expected_size is None or count <= expected_size, "content_exceeds_pinned_size")
        digest.update(chunk)
    require(expected_size is None or count == expected_size, "content_truncated")
    return digest.hexdigest(), count


def safe_model_path(root, name):
    require(name in MODEL_PATHS, "model_outside_allowlist")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "model_escapes_root")
    for parent in (path, *path.parents):
        if parent == root:
            break
        require(not parent.is_symlink(), "model_symlink_not_allowed")
    require(path.is_file(), "model_missing")
    return path


def plan(root, manifest):
    require(not manifest.is_symlink() and manifest.is_file(), "manifest_missing_or_symlink")
    require(manifest.stat().st_size <= 4 * 1024**2, "manifest_too_large")
    raw = manifest.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == QUALITY_MANIFEST_SHA256, "quality_manifest_sha_mismatch")
    data = json.loads(raw)
    require(data.get("schema") == "repro_3090_inputs_v1" and isinstance(data.get("files"), list), "manifest_schema")
    targets = {(root / name).resolve(): name for name in MODEL_PATHS}
    rows = {}
    for row in data["files"]:
        require(isinstance(row, dict) and isinstance(row.get("path"), str), "manifest_row")
        name = targets.get((manifest.parent / row["path"]).resolve())
        if name is None:
            continue
        require(name not in rows, "duplicate_model_pin")
        require(SHA.fullmatch(str(row.get("sha256", ""))) and type(row.get("bytes")) is int
                and 0 < row["bytes"] <= 4 * 1024**3, "invalid_model_pin")
        path = safe_model_path(root, name)
        require(path.stat().st_size == row["bytes"], "local_model_size_mismatch")
        with path.open("rb") as stream:
            digest, _ = hash_stream(stream, row["bytes"])
        require(digest == row["sha256"], "local_model_sha_mismatch")
        rows[name] = {
            "sha256": digest, "size_bytes": row["bytes"],
            "public_redistribution": False, "private_delivery_authorized": True,
            "source": {"type": "s3", "bucket": BUCKET, "region": REGION, "expected_owner": OWNER,
                       "key": PREFIX + "/" + digest + "/" + Path(name).name},
        }
    require(set(rows) == MODEL_PATHS, "five_private_model_pins_required")
    return dict(sorted(rows.items()))


def client():
    # Explicit credentials only. Never read an ambient profile, anonymous URL,
    # EC2 instance role, configured endpoint URL, or broad host credential file.
    require(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"), "runtime_credentials_missing")
    import boto3
    from botocore.config import Config
    result = boto3.client("s3", region_name=REGION,
                          aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                          aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                          config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=60,
                                        retries={"total_max_attempts": 3, "mode": "standard"},
                                        ignore_configured_endpoint_urls=True))
    require("IfNoneMatch" in result.meta.service_model.operation_model("PutObject").input_shape.members,
            "sdk_lacks_conditional_put")
    return result


def sdk_status(exc):
    response = getattr(exc, "response", {})
    value = response.get("ResponseMetadata", {}).get("HTTPStatusCode") if isinstance(response, dict) else None
    return value if type(value) is int else None


def public_acl(value):
    return any(g.get("Grantee", {}).get("URI") in PUBLIC_GROUPS for g in value.get("Grants", []))


def privacy_observations(s3):
    result = {"independent_privacy_proof": False,
              "basis": "No affirmative privacy proof yet; inaccessible configuration is not proof of privacy."}
    for method, key, parse in (
        ("get_public_access_block", "bucket_public_access_block", lambda r: {
            k: r.get("PublicAccessBlockConfiguration", {}).get(k) for k in
            PUBLIC_ACCESS_BLOCK_FLAGS}),
        ("get_bucket_policy_status", "bucket_policy_public", lambda r: r.get("PolicyStatus", {}).get("IsPublic")),
        ("get_bucket_acl", "bucket_public_acl_grants", public_acl),
    ):
        try:
            result[key] = {"status": "observed", "value": parse(getattr(s3, method)(Bucket=BUCKET, ExpectedBucketOwner=OWNER))}
        except Exception as exc:
            result[key] = {"status": "not_available", "http_status": sdk_status(exc)}
    block = result["bucket_public_access_block"]
    if block.get("status") == "observed" and all(block.get("value", {}).get(k) is True for k in PUBLIC_ACCESS_BLOCK_FLAGS):
        result["independent_privacy_proof"] = True
        result["basis"] = "Observed all four bucket-level S3 PublicAccessBlock flags true for the fixed expected owner."
    return result


def choose_privacy_proof(observed, attestation_path=None, attestation_sha256=None):
    """Never let a historical assertion override current contrary evidence."""
    require(bool(attestation_path) == bool(attestation_sha256), "privacy_attestation_path_and_sha_required_together")
    for key in ("bucket_policy_public", "bucket_public_acl_grants"):
        require(observed.get(key, {}).get("value") is not True, "bucket_public_access_observed")
    block = observed.get("bucket_public_access_block", {})
    if block.get("status") == "observed":
        require(all(block.get("value", {}).get(k) is True for k in PUBLIC_ACCESS_BLOCK_FLAGS),
                "live_bucket_privacy_block_not_all_true")
    attestation = None
    if attestation_path:
        try:
            attestation = privacy_attestation.load_bound(attestation_path, attestation_sha256)
        except privacy_attestation.Invalid as exc:
            raise Invalid(str(exc)) from None
    if observed.get("independent_privacy_proof") is True:
        return {"source": "live_worker_bucket_public_access_block", "attestation_used": False}, None
    require(attestation is not None, "affirmative_bucket_privacy_proof_required_before_object_access")
    return {"source": "explicit_sha_bound_operator_attestation", "attestation_used": True,
            "sha256": attestation_sha256, "observed_at_utc": attestation["observed_at_utc"],
            "expires_at_utc": attestation["expires_at_utc"],
            "trust_scope": "operator_assertion_not_aws_signed; digest_must_arrive_via_trusted_operator_invocation"}, attestation


def verify_remote(s3, entry):
    response = s3.get_object(Bucket=BUCKET, Key=entry["source"]["key"], ExpectedBucketOwner=OWNER)
    body = response["Body"]
    try:
        require(response.get("ContentLength") == entry["size_bytes"], "remote_size_mismatch")
        digest, count = hash_stream(body, entry["size_bytes"])
        require(digest == entry["sha256"], "remote_sha_mismatch")
    finally:
        body.close()
    proof = {"remote_content_verified": True, "verified_at_utc": now(), "method": "streamed_get_object_sha256",
             "sha256": digest, "size_bytes": count, "etag_used_as_hash": False}
    if response.get("VersionId"):
        require(isinstance(response["VersionId"], str) and len(response["VersionId"]) <= 1024, "invalid_remote_version_id")
        entry["source"]["version_id"] = response["VersionId"]
    try:
        args = {"Bucket": BUCKET, "Key": entry["source"]["key"], "ExpectedBucketOwner": OWNER}
        if entry["source"].get("version_id"):
            args["VersionId"] = entry["source"]["version_id"]
        acl = s3.get_object_acl(**args)
        proof["object_public_acl_grants"] = {"status": "observed", "value": public_acl(acl)}
    except Exception as exc:
        proof["object_public_acl_grants"] = {"status": "not_available", "http_status": sdk_status(exc)}
    require(proof["object_public_acl_grants"].get("value") is not True, "remote_object_has_public_acl")
    return proof


def persist_one(s3, root, name, entry):
    exists = False
    try:
        header = s3.head_object(Bucket=BUCKET, Key=entry["source"]["key"], ExpectedBucketOwner=OWNER)
        require(header.get("ContentLength") == entry["size_bytes"], "existing_remote_size_mismatch")
        exists = True
    except Invalid:
        raise
    except Exception as exc:
        # AccessDenied is not evidence an object is missing.
        if sdk_status(exc) != 404:
            raise Invalid("head_failed_object_existence_unknown") from None
    action = "reused_existing" if exists else "uploaded_missing"
    if not exists:
        with safe_model_path(root, name).open("rb") as body:
            require(os.fstat(body.fileno()).st_size == entry["size_bytes"], "local_model_changed_before_upload")
            try:
                s3.put_object(Bucket=BUCKET, Key=entry["source"]["key"], Body=body,
                              ContentLength=entry["size_bytes"], ExpectedBucketOwner=OWNER, IfNoneMatch="*",
                              ChecksumSHA256=base64.b64encode(bytes.fromhex(entry["sha256"])).decode(),
                              ContentType="application/octet-stream", ServerSideEncryption="AES256",
                              Metadata={"sha256": entry["sha256"], "delivery-scope": "private-runtime-only"})
                # ACL deliberately omitted (default private / compatible with
                # bucket-owner-enforced ACL-disabled buckets). Never public-read.
            except Exception as exc:
                if sdk_status(exc) in {409, 412}:
                    action = "concurrent_object_reconciled_by_get"
                else:
                    # No manual retry after an ambiguous outcome, no overwrite.
                    raise Invalid("conditional_upload_failed_or_ambiguous") from None
    return {"action": action, **verify_remote(s3, entry)}


def write_report(handle, report):
    handle.seek(0)
    handle.truncate()
    json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def run(args, report, handle, make_client=client):
    rows = plan(args.root, args.manifest)
    report.update(status="plan_only", local_content_verified=True, planned_external_model_files=rows,
                  planned_upload_bytes_if_all_missing=sum(e["size_bytes"] for e in rows.values()),
                  verification_download_bytes=sum(e["size_bytes"] for e in rows.values()))
    write_report(handle, report)
    if not args.execute:
        return 0
    require(args.root.resolve() == Path(WORKER_ROOT) and socket.gethostname() == WORKER_HOSTNAME, "wrong_worker_or_root")
    s3 = make_client()
    report["privacy_observations"] = privacy_observations(s3)
    write_report(handle, report)
    proof_source, attestation = choose_privacy_proof(report["privacy_observations"],
                                                   getattr(args, "privacy_attestation", None),
                                                   getattr(args, "privacy_attestation_sha256", None))
    report["privacy_proof"] = proof_source
    report["status"] = "in_progress"
    for name, entry in rows.items():
        if attestation is not None:
            try:
                # Recheck before each object sequence, not only before hashing
                # the five local files or at the start of a long upload batch.
                report["privacy_proof"]["latest_freshness_check"] = privacy_attestation.validate(attestation)
            except privacy_attestation.Invalid as exc:
                raise Invalid(str(exc)) from None
        report["active_model_path"] = name
        write_report(handle, report)
        proof = persist_one(s3, args.root, name, entry)
        report["objects"][name] = proof
        write_report(handle, report)
    require(set(report["objects"]) == MODEL_PATHS and all(p["remote_content_verified"] for p in report["objects"].values()),
            "all_five_remote_content_proofs_required")
    report.update(status="verified", external_model_files=rows, delivery_verified=True,
                  completed_at_utc=now(), active_model_path=None)
    write_report(handle, report)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(WORKER_ROOT))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--privacy-attestation", type=Path,
                        help="optional fresh operator privacy assertion; requires explicit trusted SHA256")
    parser.add_argument("--privacy-attestation-sha256")
    args = parser.parse_args(argv)
    if bool(args.privacy_attestation) != bool(args.privacy_attestation_sha256):
        parser.error("--privacy-attestation and --privacy-attestation-sha256 must be supplied together")
    logging.disable(logging.CRITICAL)
    args.root = args.root.resolve()
    args.manifest = args.manifest.absolute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Only this invocation's new report can be updated; never replace a prior run.
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    report = {"schema": "musetalk_private_model_delivery_v1", "started_at_utc": now(),
              "instance_id": INSTANCE_ID, "input_manifest_sha256": QUALITY_MANIFEST_SHA256,
              "destination_bucket": BUCKET, "destination_region": REGION, "expected_owner": OWNER,
              "destination_prefix": PREFIX, "execute_requested": args.execute,
              "status": "initializing", "delivery_verified": False, "external_model_files": {}, "objects": {},
              "redistribution_scope": "baked_model_files_only", "private_model_usage_rights": "unresolved",
              "public_redistribution_authorized": False, "public_acl_requested": False,
              "existing_objects_overwritten": False, "object_deletions": False}
    with os.fdopen(fd, "w") as handle:
        write_report(handle, report)
        try:
            code = run(args, report, handle)
        except Exception as exc:
            # Controlled own error codes only. SDK exception strings can contain
            # signed URLs, request details, or credential-bearing configuration.
            report.update(status="failed", delivery_verified=False, external_model_files={},
                          failure_code=str(exc) if isinstance(exc, Invalid) else "operation_failed_details_suppressed",
                          failed_at_utc=now())
            write_report(handle, report)
            code = 2
    print(json.dumps({"status": report["status"], "delivery_verified": report["delivery_verified"],
                      "verified_objects": len(report["objects"]), "report_written": True}))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Private-model operation failed before report creation; details suppressed", file=sys.stderr)
        raise SystemExit(2)
