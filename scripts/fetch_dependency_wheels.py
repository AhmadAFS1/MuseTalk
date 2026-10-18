#!/usr/bin/env python3
"""Bounded acquisition only; trusted manifest digest required, no pip invocations.

The initial acquisition candidate can include sdists. A successful download is
not an offline-wheelhouse or native-compatibility pass.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from dependency_artifacts import require, selected, validate_manifest

CHUNK = 1024 * 1024
MAX_FILE = 4 * 1024**3
BUCKET = "lingua-musetalk-s3-storage"
OWNER = "211125449207"
PRIVATE_PREFIX = "python-venv/wheelhouse/"


def check_ticket(a, ticket):
    require(isinstance(ticket, dict) and ticket.get("sha256") == a["sha256"]
            and ticket.get("owner") == OWNER, "Missing/bad private artifact ticket")
    u = urllib.parse.urlsplit(ticket["url"])
    require(u.scheme == "https" and u.hostname in {BUCKET+".s3.amazonaws.com", BUCKET+".s3.us-east-1.amazonaws.com"}
            and not u.username and not u.password and u.port in (None, 443) and not u.fragment,
            "Private artifact ticket destination rejected")
    require(re.fullmatch(re.escape(PRIVATE_PREFIX)+r"[a-f0-9]{64}/"+re.escape(a["filename"]), ticket.get("key", ""))
            and urllib.parse.unquote(u.path) == "/"+ticket["key"]
            and ticket["key"].endswith("/"+a["filename"]), "Private artifact key mismatch")
    return ticket["url"]


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(CHUNK):
            value.update(chunk)
    return value.hexdigest()


def read_trusted(path, expected_sha):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, "Untrusted manifest digest")
    return validate_manifest(json.loads(raw))


@contextmanager
def lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(fd).st_uid == os.getuid(), "Foreign cache lock")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


class SecureRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(newurl)
        require(u.scheme == "https" and not u.username and not u.password, "Unsafe artifact redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def stream_source(a, repo_root, opener, ticket=None):
    if a["source"].startswith("repo:"):
        require(repo_root is not None, "Repo-local wheel needs an explicit checkout")
        root = Path(repo_root).resolve(strict=True)
        path = root / a["source"][5:]
        require(path.resolve(strict=True).is_relative_to(root) and not path.is_symlink(), "Repo artifact escapes checkout")
        return path.open("rb")
    headers = {"User-Agent": "MuseTalk-dependency-fetch/1"}
    if a["source"].startswith("build:"):
        url = check_ticket(a, ticket)
        headers["x-amz-expected-bucket-owner"] = OWNER
    else:
        require(ticket is None, "Unexpected override of reviewed upstream source")
        url = a["source"]
    request = urllib.request.Request(url, headers=headers)
    return opener.open(request, timeout=30)


def acquire_one(a, cache, repo_root=None, retries=2, opener=None, ticket=None):
    """Never expose URL-bearing exceptions; partials are never published."""
    cache = Path(cache)
    target = cache / a["filename"]
    start = time.monotonic()
    with lock(cache / (a["filename"] + ".lock")):
        if target.exists():
            require(target.is_file() and not target.is_symlink(), "Unsafe cached artifact")
            require(sha256_file(target) == a["sha256"], "Cached artifact hash mismatch; quarantining requires explicit recovery")
            require(a["size_bytes"] in (None, target.stat().st_size), "Cached artifact size mismatch")
            return {"name": a["name"], "state": "verified-cache-hit", "bytes": target.stat().st_size,
                    "transferred_bytes": 0, "remote_transferred_bytes": 0, "local_copied_bytes": 0,
                    "attempts": 0, "seconds": time.monotonic() - start}
        opener = opener or urllib.request.build_opener(SecureRedirect())
        transferred = 0
        attempts = 0
        for attempt in range(retries + 1):
            attempts += 1
            fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=cache)
            try:
                h = hashlib.sha256()
                size = 0
                with os.fdopen(fd, "wb") as out, stream_source(a, repo_root, opener, ticket) as source:
                    length = source.headers.get("Content-Length") if hasattr(source, "headers") else None
                    if length is not None:
                        length = int(length)
                        require(0 < length <= MAX_FILE, "Invalid source length")
                        require(a["size_bytes"] in (None, length), "Source size mismatch")
                    while chunk := source.read(CHUNK):
                        size += len(chunk)
                        transferred += len(chunk)
                        require(size <= (a["size_bytes"] or MAX_FILE), "Artifact exceeds trusted size/limit")
                        h.update(chunk)
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                require(size > 0 and a["size_bytes"] in (None, size), "Incomplete artifact size")
                require(length in (None, size), "Truncated response")
                require(h.hexdigest() == a["sha256"], "Artifact integrity mismatch")
                os.replace(temporary, target)
                return {"name": a["name"], "state": "downloaded-verified", "bytes": size,
                        "transferred_bytes": transferred,
                        "remote_transferred_bytes": 0 if a["source"].startswith("repo:") else transferred,
                        "local_copied_bytes": transferred if a["source"].startswith("repo:") else 0,
                        "attempts": attempts,
                        "seconds": time.monotonic() - start,
                        "origin": urllib.parse.urlsplit(ticket["url"] if ticket else a["source"]).hostname or "repo-local"}
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                # Auth failures and not-found are not transient. Integrity errors
                # deliberately do not enter this retry path.
                if isinstance(error, urllib.error.HTTPError):
                    status = error.code
                    with suppress(Exception):  # Python 3.9: body-less HTTPError.close() raises KeyError.
                        error.close()
                    if status not in {408, 429, 500, 502, 503, 504}:
                        raise ValueError("Artifact HTTP failure " + str(status)) from None
                if attempt == retries:
                    raise ValueError("Bounded artifact transport retries exhausted") from None
                time.sleep(min(2**attempt, 4))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)  # Exact mkstemp-owned partial only.
    raise AssertionError("Unreachable acquisition state")


def acquisition_order(rows, private_shas):
    """Ticketed private wheels first: they are tiny and their presigned URLs
    expire, so they must not queue behind gigabytes of CDN transfers. Then the
    largest known packages; unknown sizes fall back to a stable NVIDIA/Torch
    priority, not a guess reported as a measured byte count."""
    return sorted(rows, key=lambda a: (0 if a["sha256"] in private_shas else 1, -(a["size_bytes"] or 0),
                                       0 if a["name"].startswith(("tensorrt", "torch", "nvidia")) else 1, a["name"]))


def acquire(manifest, cache, group, workers, repo_root=None, reserve_bytes=30 * 1024**3, tickets=None):
    require(workers in (1, 2, 4, 8), "Bounded concurrency required")
    cache = Path(cache)
    require(not cache.is_symlink(), "Cache symlink rejected")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(cache.stat().st_uid == os.getuid(), "Foreign cache")
    with lock(cache / ".acquisition.lock"):
        marker = cache / ".artifact-id"
        if marker.exists():
            require(not marker.is_symlink() and marker.read_text() == manifest["artifact_id"], "Cache artifact identity mismatch")
        else:
            marker.write_text(manifest["artifact_id"])
        require(shutil.disk_usage(cache).free >= reserve_bytes, "Insufficient scratch disk before acquisition")
        start = time.monotonic()
        rows = selected(manifest, group)
        tickets = tickets or {}
        built = {a["sha256"]: a for a in rows if a["source"].startswith("build:")}
        require(set(tickets) == set(built), "Private ticket set differs from selected built-wheel contract")
        for sha, a in built.items():
            check_ticket(a, tickets[sha])
            require(tickets[sha]["key"] == PRIVATE_PREFIX+manifest["artifact_id"]+"/"+a["filename"], "Ticket artifact identity mismatch")
        rows = acquisition_order(rows, built)
        receipts = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(acquire_one, a, cache, repo_root, ticket=tickets.get(a["sha256"])): a["name"] for a in rows}
            for future in as_completed(futures):
                try:
                    receipts.append(future.result())
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
        return {"state": "acquisition-complete", "artifact_id": manifest["artifact_id"], "group": group,
                "workers": workers, "seconds": time.monotonic() - start,
                "transferred_bytes": sum(r["transferred_bytes"] for r in receipts),
                "remote_transferred_bytes": sum(r["remote_transferred_bytes"] for r in receipts),
                "local_copied_bytes": sum(r["local_copied_bytes"] for r in receipts),
                "cache_hits": sum(r["state"] == "verified-cache-hit" for r in receipts),
                "artifacts": sorted(receipts, key=lambda r: r["name"]),
                "native_compatibility_proven": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--group", choices=("server", "server+kokoro", "full"), default="full")
    p.add_argument("--workers", type=int, choices=(1, 2, 4, 8), required=True)
    p.add_argument("--repo-root", type=Path)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--signed-sources-stdin", action="store_true", help="Ephemeral private-wheel tickets; never saved or placed in argv")
    a = p.parse_args()
    start = time.monotonic()
    try:
        m = read_trusted(a.manifest, a.manifest_sha256)
        data = sys.stdin.read(256001) if a.signed_sources_stdin else None
        require(data is None or len(data) <= 256000, "Private ticket input too large")
        tickets = json.loads(data) if data is not None else None
        result = acquire(m, a.cache, a.group, a.workers, a.repo_root, tickets=tickets)
        result["outer_seconds"] = time.monotonic() - start
    except BaseException as error:
        # No traceback: urllib errors may include signed URLs in future transports.
        result = {"state": "failed", "error_type": type(error).__name__,
                  "detail": str(error) if type(error) is ValueError else "Details suppressed",
                  "outer_seconds": time.monotonic() - start}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    require(not a.receipt.exists(), "Receipt already exists; never silently overwrite a trial")
    a.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)
    return 0 if result["state"] == "acquisition-complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
