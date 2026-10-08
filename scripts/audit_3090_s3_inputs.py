#!/usr/bin/env python3
"""Read-only S3 availability audit; HEAD is not a content/latent integrity test."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit


def check_object(item, profile=None):
    parsed = urlsplit(item["s3_uri"])
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        return {**item, "status": "INVALID", "reason": "invalid S3 URI"}
    command = ["aws"]
    if profile:
        command += ["--profile", profile]
    command += ["s3api", "head-object", "--bucket", parsed.netloc, "--key", parsed.path.lstrip("/"), "--output", "json"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if process.returncode:
            # Do not persist raw provider error text (may include unrelated identity/config).
            category = next((c for c in ("404", "403", "AccessDenied", "ExpiredToken", "NoSuchKey") if c in process.stderr), "request_failed")
            return {**item, "status": "INVALID", "reason": category, "exit_code": process.returncode}
        response = json.loads(process.stdout)
        actual = {"bytes": response["ContentLength"], "etag": response.get("ETag"), "metadata": response.get("Metadata", {}), "last_modified": response.get("LastModified")}
        mismatches = []
        if actual["bytes"] != item["bytes"]:
            mismatches.append("size")
        if item.get("etag") and actual["etag"] != item["etag"]:
            mismatches.append("etag")
        for key, value in item.get("metadata", {}).items():
            if actual["metadata"].get(key) != value:
                mismatches.append("metadata:" + key)
        return {**item, "actual": actual, "status": "FAIL" if mismatches else "PASS", "mismatches": mismatches}
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return {**item, "status": "INVALID", "reason": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    items = []
    for character in manifest["characters"]:
        items.append({"character": character["id"], "kind": "portrait", **character["portrait"]})
        for pose, cache in character["poses"].items():
            items.append({"character": character["id"], "kind": "pose_cache", "pose": pose, **cache})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: check_object(item, args.profile), items))
    counts = {status: sum(r["status"] == status for r in results) for status in ("PASS", "FAIL", "INVALID")}
    status = "INVALID" if counts["INVALID"] else "FAIL" if counts["FAIL"] else "PASS"
    report = {
        "schema": "rtx3090_s3_availability_v1", "status": status,
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "credential_reference": args.profile or "AWS CLI default credential chain",
        "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "HEAD availability/size/recorded ETag/metadata; no archive/latent/content validation",
        "etag_is_content_sha256": False, "counts": counts, "objects": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": status, "counts": counts, "report": str(args.out)}))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
