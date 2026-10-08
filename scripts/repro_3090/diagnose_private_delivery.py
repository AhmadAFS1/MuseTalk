#!/usr/bin/env python3
"""Bounded CPU-only reads of the first frozen private model; never upload or repair.

Run via the owned-worker memory-only credential bridge. Concurrent cache-audit
traffic is disclosed; timings are diagnostic, not startup acceptance measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time

import persist_private_models as pins


MODEL = "models/auxiliary/s3fd-619a316812.pth"


def arguments(entry, version, *, versioned=True):
    source = entry["source"]
    pins.require(source["bucket"] == pins.BUCKET and source["expected_owner"] == pins.OWNER
                 and source["key"] == pins.PREFIX + "/" + entry["sha256"] + "/" + Path(MODEL).name,
                 "diagnostic_object_outside_fixed_scope")
    pins.require(isinstance(version, str) and 0 < len(version) <= 1024, "diagnostic_version_missing")
    request = dict(Bucket=pins.BUCKET, Key=source["key"], ExpectedBucketOwner=pins.OWNER)
    if versioned:
        request["VersionId"] = version
    return request


def read_probe(s3, entry, version, local_path, *, ranged, versioned=True):
    request = arguments(entry, version, versioned=versioned)
    count = min(1024 * 1024 if ranged else 1024, entry["size_bytes"])
    if ranged:
        request["Range"] = f"bytes=0-{count - 1}"
    start = time.monotonic()
    response = s3.get_object(**request)
    headers_s = time.monotonic() - start
    body = response["Body"]
    try:
        pins.require(response.get("VersionId") == version, "diagnostic_response_version_mismatch")
        pins.require(response.get("ContentLength") == (count if ranged else entry["size_bytes"]),
                     "diagnostic_response_length_mismatch")
        if ranged:
            pins.require(response.get("ContentRange") == f"bytes 0-{count - 1}/{entry['size_bytes']}",
                         "diagnostic_response_range_mismatch")
        data = body.read(count)
        pins.require(len(data) == count, "diagnostic_prefix_truncated")
        with local_path.open("rb") as original:
            expected = original.read(count)
        pins.require(data == expected, "diagnostic_prefix_content_mismatch")
        return {"status": "PASS_PREFIX_ONLY", "range_request": ranged, "version_id_requested": versioned,
                "response_version_matches_head": True, "bytes_verified": count,
                "prefix_sha256": hashlib.sha256(data).hexdigest(), "headers_seconds": headers_s,
                "elapsed_seconds": time.monotonic() - start, "whole_object_verified": False}
    finally:
        body.close()


def make_client():
    import boto3
    from botocore.config import Config
    pins.require(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"),
                 "runtime_credentials_missing")
    return boto3.client("s3", region_name=pins.REGION,
                        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                        config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=8,
                                      retries={"total_max_attempts": 1, "mode": "standard"},
                                      ignore_configured_endpoint_urls=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--privacy-attestation", type=Path, required=True)
    parser.add_argument("--privacy-attestation-sha256", required=True)
    parser.add_argument("--unversioned-request", action="store_true",
                        help="Compare normal GetObject access; retain HEAD/response version and full SHA checks.")
    args = parser.parse_args()
    root = Path(pins.WORKER_ROOT)
    pins.require(socket.gethostname() == pins.WORKER_HOSTNAME and os.geteuid() == 0, "wrong_owned_worker")
    pins.require(args.out.is_absolute() and args.out.parent == root / "docs/fps_comparisons/rtx3090_r5_20261008/release",
                 "diagnostic_report_outside_run")
    entry = pins.plan(root, args.manifest)[MODEL]
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        report = {"schema": "private_model_read_diagnostic_v2", "status": "RUNNING", "started_utc": pins.now(),
                  "model": MODEL, "source": entry["source"], "expected_bytes": entry["size_bytes"],
                  "expected_sha256": entry["sha256"], "gpu_used": False, "cloud_mutations": False,
                  "concurrent_cpu_network_audit": True, "startup_acceptance": False, "probes": {},
                  "version_id_requested": not args.unversioned_request}
        pins.write_report(handle, report)
        try:
            s3 = make_client()
            privacy, _ = pins.choose_privacy_proof(pins.privacy_observations(s3), args.privacy_attestation,
                                                  args.privacy_attestation_sha256)
            report["privacy_proof"] = privacy
            head = s3.head_object(Bucket=pins.BUCKET, Key=entry["source"]["key"], ExpectedBucketOwner=pins.OWNER)
            pins.require(head.get("ContentLength") == entry["size_bytes"], "diagnostic_head_size_mismatch")
            pins.require(head.get("Metadata", {}).get("sha256") == entry["sha256"], "diagnostic_head_sha_metadata_mismatch")
            version = head.get("VersionId")
            request = arguments(entry, version, versioned=not args.unversioned_request)
            report["version_id"] = version
            for label, ranged in (("full_get_first_1k", False), ("range_first_1m", True)):
                report["active_probe"] = label
                pins.write_report(handle, report)
                try:
                    report["probes"][label] = read_probe(s3, entry, version, pins.safe_model_path(root, MODEL),
                                                         ranged=ranged, versioned=not args.unversioned_request)
                except Exception as exc:
                    report["probes"][label] = {"status": "FAIL", "exception_type": type(exc).__name__,
                                                "http_status": pins.sdk_status(exc), "details_suppressed": True,
                                                "failure_code": str(exc) if isinstance(exc, pins.Invalid) else None}
                pins.write_report(handle, report)
            from boto3.s3.transfer import TransferConfig
            import importlib.metadata
            report["sdk_versions"] = {name: importlib.metadata.version(name) for name in ("boto3", "botocore")}
            report["active_probe"] = "parallel_whole_object"
            report["downloaded_bytes_observed"] = 0
            pins.write_report(handle, report)
            lock, last_write = threading.Lock(), [0.0]
            def progress(count):
                with lock:
                    report["downloaded_bytes_observed"] += count
                    if time.monotonic() - last_write[0] >= 1:
                        pins.write_report(handle, report)
                        last_write[0] = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="private-read-diagnostic-", dir=root / "tmp") as temporary:
                target = Path(temporary) / "download.bin"
                start = time.monotonic()
                s3.download_file(request.pop("Bucket"), request.pop("Key"), str(target), ExtraArgs=request,
                                 Callback=progress, Config=TransferConfig(max_concurrency=4, multipart_chunksize=8 * 1024**2,
                                                                          num_download_attempts=1))
                with target.open("rb") as stream:
                    digest, count = pins.hash_stream(stream, entry["size_bytes"])
                pins.require(digest == entry["sha256"], "diagnostic_whole_object_sha_mismatch")
                after = s3.head_object(Bucket=pins.BUCKET, Key=entry["source"]["key"], ExpectedBucketOwner=pins.OWNER)
                pins.require(after.get("VersionId") == version, "diagnostic_object_version_changed")
                pins.require(after.get("ContentLength") == entry["size_bytes"]
                             and after.get("Metadata", {}).get("sha256") == entry["sha256"],
                             "diagnostic_object_metadata_changed")
                report["probes"]["parallel_whole_object"] = {"status": "PASS_CONTENT_ONLY", "bytes": count,
                    "sha256": digest, "elapsed_seconds": time.monotonic() - start, "max_concurrency": 4,
                    "multipart_chunksize": 8 * 1024**2, "whole_object_verified": True,
                    "version_id_requested": not args.unversioned_request, "head_version_unchanged": True}
            report.update(status="DIAGNOSTIC_COMPLETE", active_probe=None, finished_utc=pins.now())
        except Exception as exc:
            report.update(status="FAILED", exception_type=type(exc).__name__, http_status=pins.sdk_status(exc),
                          details_suppressed=True, finished_utc=pins.now(),
                          failure_code=str(exc) if isinstance(exc, pins.Invalid) else None)
        pins.write_report(handle, report)
    return 0 if report["status"] == "DIAGNOSTIC_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
