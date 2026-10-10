#!/usr/bin/env python3
"""Private, checksum-pinned S3 input transport for a nonpromotable full image.

Presigned URLs are one-time CI secret inputs, never files, build arguments or
diagnostics. No AWS credential or registry token is required by the builder.
This transport does not clear licenses, quality, GPU or deployment gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

import ci
import release

INPUT_ENV = "MUSETALK_PRIVATE_BUILD_INPUTS"
S3_HOST = "lingua-musetalk-s3-storage.s3.us-east-1.amazonaws.com"
INPUT_NAMES = {ci.METADATA_NAME, "weights.tar.gz", "native.tar.gz"}
MAX_ARCHIVE = 16 * 1024**3


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Private input redirects are forbidden")


def validate_urls(payload):
    release.require(isinstance(payload, dict) and set(payload) == INPUT_NAMES,
                    "Exactly three private input URLs are required")
    for name, value in payload.items():
        release.require(isinstance(value, str) and len(value) <= 8192
                        and not any(c.isspace() or ord(c) < 32 for c in value), "Invalid input URL")
        url = urllib.parse.urlsplit(value)
        release.require(url.scheme == "https" and url.hostname == S3_HOST and url.port in {None, 443}
                        and not url.username and not url.password and not url.fragment,
                        "Private inputs must use the approved regional S3 HTTPS endpoint")
        path = urllib.parse.unquote(url.path).lstrip("/")
        release.relative(path)
        release.require(path.startswith(("docker-build-inputs/", "trt-artifacts/")),
                        "Private input object prefix is not approved")
        query = urllib.parse.parse_qs(url.query, strict_parsing=True)
        release.require(query.get("X-Amz-Algorithm") == ["AWS4-HMAC-SHA256"]
                        and len(query.get("X-Amz-Signature", [])) == 1
                        and re.fullmatch(r"[0-9a-f]{64}", query["X-Amz-Signature"][0])
                        and len(query.get("X-Amz-Expires", [])) == 1
                        and query["X-Amz-Expires"][0].isdigit()
                        and 1 <= int(query["X-Amz-Expires"][0]) <= 3600,
                        "Private URLs must use bounded one-hour-or-less S3 signatures")
        release.require(len(query.get("X-Amz-Credential", [])) == 1
                        and query["X-Amz-Credential"][0].endswith("/us-east-1/s3/aws4_request"),
                        "Private input signature region/service mismatch")
    return payload


def download(url, destination, digest, size=None, *, limit=MAX_ARCHIVE, opener=None):
    release.require(release.SHA.fullmatch(str(digest)) and not destination.exists(),
                    "Pinned checksum and a fresh private download destination are required")
    release.require(size is None or type(size) is int and 0 < size <= limit,
                    "Invalid private input size")
    opener = opener or urllib.request.build_opener(NoRedirect()).open
    expected = hashlib.sha256()
    count = 0
    # Do not include URL/HTTP exception text in any outer diagnostic.
    with opener(url, timeout=60) as response:
        release.require(response.status == 200, "Private object request did not return a complete object")
        length = response.headers.get("Content-Length")
        if length is not None:
            release.require(length.isdigit() and 0 < int(length) <= limit
                            and (size is None or int(length) == size), "Private object length mismatch")
        with destination.open("xb") as target:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                count += len(block)
                release.require(count <= limit and (size is None or count <= size),
                                "Private object exceeds its pinned size")
                expected.update(block)
                target.write(block)
    release.require(count > 0 and (size is None or count == size) and expected.hexdigest() == digest,
                    "Private object size/checksum mismatch")
    return destination


def validate_archives(manifest):
    """Direct private objects have no GitHub public-asset 2GiB/chunk constraint."""
    archives = manifest.get("archives", {})
    release.require(set(archives) == {"weights.tar.gz", "native.tar.gz"}, "Exactly two private archives required")
    for entry in archives.values():
        release.require(isinstance(entry, dict) and set(entry) == {"sha256", "size_bytes"}
                        and release.SHA.fullmatch(str(entry["sha256"]))
                        and type(entry["size_bytes"]) is int and 0 < entry["size_bytes"] <= MAX_ARCHIVE,
                        "Private archive requires a bounded exact size and SHA-256")
    return archives


def assemble(root, work, revision, metadata_sha256, urls, fetch=download):
    release.require(re.fullmatch(r"[0-9a-f]{40}", revision) and release.SHA.fullmatch(metadata_sha256),
                    "Full source and metadata hashes required")
    validate_urls(urls)
    release.require(not work.exists(), "Private build work directory must be new")
    release.require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == revision,
                    "Private build source revision mismatch")
    release.require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
                    "Private build requires clean committed source")
    work.mkdir(parents=True)
    release.require(shutil.disk_usage(work).free >= ci.MIN_DISK, "Less than 60GiB free; no runner cleanup")
    reports = work / "reports"
    reports.mkdir()
    metadata = fetch(urls[ci.METADATA_NAME], work / ci.METADATA_NAME, metadata_sha256, limit=ci.MAX_METADATA)
    assets = work / "release"
    manifest = ci.read_metadata(metadata, metadata_sha256, assets, revision, "candidate",
                                validate_transport=validate_archives)
    # Fail before downloading multi-GB payloads if descriptor/license/source
    # contracts are inconsistent. Never set a review/acceptance flag here.
    release.descriptor(root, manifest)
    release.verify_model_contract(root, manifest)
    release.verify_source(root, manifest)
    for name, entry in manifest["archives"].items():
        fetch(urls[name], assets / name, entry["sha256"], entry["size_bytes"])
    result = {"schema": "musetalk_private_ci_inputs_v1", "source_revision": revision,
              "metadata_sha256": metadata_sha256, "archives": manifest["archives"],
              "channel": "candidate", "promotion_eligible": False, "private_transport": "S3_PRESIGNED",
              "limitation": "Transport checks only; reviewed manifest and all image/GPU gates remain required"}
    (reports / "private-inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    return assets, manifest, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    # Consume/remove the secret before spawning any build/check subprocess.
    urls = json.loads(os.environ.pop(INPUT_ENV, ""))
    for name in list(os.environ):
        if name.startswith(("GH_TOKEN", "GITHUB_TOKEN", "AWS_")):
            os.environ.pop(name, None)
    assets, manifest, reports = assemble(args.root.resolve(), args.work, args.source_revision,
                                         args.metadata_sha256, urls)
    urls.clear()
    if not args.assemble_only:
        ci.build(args.root.resolve(), assets, manifest, args.work, reports)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Private candidate assembly/build rejected (URLs and exception details suppressed)", file=sys.stderr)
        raise SystemExit(1)
