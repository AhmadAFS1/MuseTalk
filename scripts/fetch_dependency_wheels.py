#!/usr/bin/env python3
"""Bounded acquisition only; trusted manifest digest required, no pip invocations.

The initial acquisition candidate can include sdists. A successful download is
not an offline-wheelhouse or native-compatibility pass.

Opt-in ranged mode (--range-streams 2|4, default 0 = one stream per artifact)
splits large public wheels into byte ranges fetched over several connections.
Only reviewed public CDN hosts qualify; repo-local, private (build:) and
ticketed sources never do. The pinned manifest sha256 over the whole file is
still the only publication gate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from contextlib import contextmanager
import datetime as dt
import errno
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import sys
import tempfile
import threading
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

# Ranged mode qualifies only TensorRT libs, torch, cuDNN and cuBLAS in the
# current manifest (triton is below the threshold).
RANGED_MIN_BYTES = 256 * 1024**2
RANGE_PART = 64 * 1024**2
RANGED_HOSTS = frozenset({"pypi.nvidia.com", "download.pytorch.org", "download-r2.pytorch.org",
                          "files.pythonhosted.org"})
RANGE_STREAM_CHOICES = (0, 2, 4)
CONNECTION_CHOICES = (1, 2, 4, 8)
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# socket.timeout is TimeoutError from 3.10; listed for older interpreters.
PART_TRANSPORT_ERRORS = (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                         ssl.SSLError, http.client.IncompleteRead)


class RangeUnsupported(Exception):
    """A ranged GET was not answered with 206. Its body is never read."""


class RangeTruncated(ConnectionError):
    """A 206 body ended early. Retryable at the part level only."""


class RangeAborted(Exception):
    """A sibling part failed; stop this part without retrying."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def stream_source(a, repo_root, opener, ticket=None, byte_range=None):
    if a["source"].startswith("repo:"):
        require(byte_range is None, "Ranged reads are for reviewed public sources only")
        require(repo_root is not None, "Repo-local wheel needs an explicit checkout")
        root = Path(repo_root).resolve(strict=True)
        path = root / a["source"][5:]
        require(path.resolve(strict=True).is_relative_to(root) and not path.is_symlink(), "Repo artifact escapes checkout")
        return path.open("rb")
    headers = {"User-Agent": "MuseTalk-dependency-fetch/1"}
    if a["source"].startswith("build:"):
        require(byte_range is None, "Ranged reads are for reviewed public sources only")
        url = check_ticket(a, ticket)
        headers["x-amz-expected-bucket-owner"] = OWNER
    else:
        require(ticket is None, "Unexpected override of reviewed upstream source")
        url = a["source"]
    if byte_range is not None:
        # urllib's redirect handler carries Range to the (https-only) target.
        headers["Range"] = "bytes=%d-%d" % byte_range
    request = urllib.request.Request(url, headers=headers)
    return opener.open(request, timeout=30)


def ranged_eligible(a, ticket=None):
    """Public reviewed CDN wheels of at least RANGED_MIN_BYTES; never repo:, build: or ticketed."""
    source = a.get("source") or ""
    size = a.get("size_bytes")
    if ticket is not None or source.startswith(("repo:", "build:")) or type(size) is not int:
        return False
    u = urllib.parse.urlsplit(source)
    return (u.scheme == "https" and u.hostname in RANGED_HOSTS and u.port in (None, 443)
            and not u.username and not u.password and not u.query and not u.fragment
            and RANGED_MIN_BYTES <= size <= MAX_FILE)


class RangeIdentity:
    """Pins ETag/Last-Modified from the first validated part. Akamai (pypi.nvidia.com)
    and R2 (download-r2.pytorch.org) ignore If-Match, so preconditions are not used."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def check(self, headers):
        seen = (headers.get("ETag"), headers.get("Last-Modified"))
        with self._lock:
            if self._value is None:
                self._value = seen
            pinned = self._value
        require(pinned == seen, "Object identity changed between ranges")

    @property
    def pinned(self):
        with self._lock:
            return self._value is not None


def check_range_response(source, a, first, last, identity):
    if getattr(source, "status", None) != 206:
        raise RangeUnsupported()  # Closed by the caller without reading the body.
    headers = source.headers
    require((headers.get("Content-Range") or "").strip() == "bytes %d-%d/%d" % (first, last, a["size_bytes"]),
            "Wrong range response")
    length = (headers.get("Content-Length") or "").strip()
    require(length.isdigit() and int(length) == last - first + 1, "Range length differs")
    encoding = headers.get("Content-Encoding")
    require(encoding is None or encoding.strip().lower() in ("", "identity"), "Encoded range rejected")
    identity.check(headers)


def write_at(fd, data, offset):
    view = memoryview(data)
    while view:
        try:
            written = os.pwrite(fd, view, offset)
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT):
                raise ValueError("Insufficient disk for ranged artifact") from None
            raise
        require(written > 0, "Short artifact write")
        view = view[written:]
        offset += written


def fetch_part(a, opener, budget, fd, part, identity, retries, stop, record, on_validated=None):
    """Fetch one inclusive byte range into fd. Transient failures (including a clean
    early end of body, TLS errors and IncompleteRead) resume from the last written
    offset; 4xx/416 are hard failures. Only the HTTP request holds a budget slot."""
    first, last = part
    position = first
    record["started_utc"] = utc_now()
    started = time.monotonic()
    for attempt in range(retries + 1):
        if stop.is_set():
            raise RangeAborted()
        record["attempts"] += 1
        if attempt:
            record["resume_offsets"].append(position)
        try:
            waiting = time.monotonic()
            with budget:
                record["wait_seconds"] += time.monotonic() - waiting
                if stop.is_set():
                    raise RangeAborted()
                with stream_source(a, None, opener, byte_range=(position, last)) as source:
                    check_range_response(source, a, position, last, identity)
                    if on_validated is not None:
                        on_validated()
                    while position <= last:
                        if stop.is_set():
                            raise RangeAborted()
                        chunk = source.read(min(CHUNK, last - position + 1))
                        if not chunk:
                            break
                        require(len(chunk) <= last - position + 1, "Range body exceeds requested length")
                        write_at(fd, chunk, position)
                        position += len(chunk)
                        record["bytes"] += len(chunk)
            if position <= last:
                raise RangeTruncated()
            record["finished_utc"] = utc_now()
            record["seconds"] = time.monotonic() - started
            return
        except PART_TRANSPORT_ERRORS as error:
            # Same status policy as the single-stream loop; never expose URLs.
            if isinstance(error, urllib.error.HTTPError):
                status = error.code
                error.close()
                if status not in TRANSIENT_STATUS:
                    raise ValueError("Artifact HTTP failure " + str(status)) from None
            if attempt == retries:
                raise ValueError("Bounded range retries exhausted") from None
            time.sleep(min(2**attempt, 4))
    raise AssertionError("Unreachable range state")


def preallocate(fd, size):
    """posix_fallocate where supported; ENOSPC/EDQUOT is fatal, otherwise a sparse file."""
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is not None:
        try:
            fallocate(fd, 0, size)
            return "fallocate"
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT):
                raise ValueError("Insufficient disk for ranged artifact") from None
    os.ftruncate(fd, size)
    return "ftruncate"


def sha256_fd(fd, size):
    value = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(CHUNK, size - offset), offset)
        require(chunk, "Incomplete artifact size")
        value.update(chunk)
        offset += len(chunk)
    return value.hexdigest()


def acquire_ranged(a, cache, opener, budget, retries, streams):
    """Probe part first; fan out only once its 206 headers validate. Raises
    RangeUnsupported (with .stage/.transferred) so the caller can fall back once."""
    size = a["size_bytes"]
    parts = [(first, min(first + RANGE_PART, size) - 1) for first in range(0, size, RANGE_PART)]
    records = [{"index": i, "first_byte": first, "last_byte": last, "bytes": 0, "attempts": 0,
                "resume_offsets": [], "wait_seconds": 0.0, "started_utc": None, "finished_utc": None,
                "seconds": None} for i, (first, last) in enumerate(parts)]
    identity = RangeIdentity()
    stop = threading.Event()
    probe_ready = threading.Event()
    futures = []
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=cache)
    try:
        preallocation = preallocate(fd, size)
        pool = ThreadPoolExecutor(max_workers=streams, thread_name_prefix="range-part")
        try:
            probe = pool.submit(fetch_part, a, opener, budget, fd, parts[0], identity, retries, stop,
                                records[0], probe_ready.set)
            probe.add_done_callback(lambda _: probe_ready.set())
            probe_ready.wait()
            if probe.done():
                probe.result()  # RangeUnsupported here means nothing else was requested.
            futures = [probe] + [pool.submit(fetch_part, a, opener, budget, fd, parts[i], identity, retries,
                                             stop, records[i]) for i in range(1, len(parts))]
            for future in as_completed(futures):
                future.result()
        except BaseException as error:
            stop.set()
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True)  # In-flight parts notice stop within one chunk read.
            if isinstance(error, RangeUnsupported):
                error.stage = "after-probe" if identity.pinned else "probe"
                error.transferred = sum(r["bytes"] for r in records)
            raise
        finally:
            pool.shutdown(wait=True)
        transferred = sum(r["bytes"] for r in records)
        os.fsync(fd)
        require(os.fstat(fd).st_size == size and transferred == size, "Incomplete artifact size")
        hashing = time.monotonic()
        digest = sha256_fd(fd, size)
        hash_seconds = time.monotonic() - hashing
        require(digest == a["sha256"], "Artifact integrity mismatch")  # Full re-read; never retried.
        os.replace(temporary, cache / a["filename"])
        return {"transferred": transferred, "requests": sum(r["attempts"] for r in records),
                "parts": records, "preallocation": preallocation, "hash_seconds": hash_seconds}
    finally:
        os.close(fd)
        if os.path.exists(temporary):
            os.unlink(temporary)  # Exact mkstemp-owned partial only.


def acquire_one(a, cache, repo_root=None, retries=2, opener=None, ticket=None, budget=None, range_streams=0):
    """Never expose URL-bearing exceptions; partials are never published.

    budget bounds open connections across all artifacts (held only around HTTP
    requests). range_streams=0 keeps the original single-stream behaviour."""
    require(range_streams in RANGE_STREAM_CHOICES, "Bounded range streams required")
    cache = Path(cache)
    target = cache / a["filename"]
    start = time.monotonic()
    started_utc = utc_now()
    with lock(cache / (a["filename"] + ".lock")):
        if target.exists():
            require(target.is_file() and not target.is_symlink(), "Unsafe cached artifact")
            require(sha256_file(target) == a["sha256"], "Cached artifact hash mismatch; quarantining requires explicit recovery")
            require(a["size_bytes"] in (None, target.stat().st_size), "Cached artifact size mismatch")
            return {"name": a["name"], "state": "verified-cache-hit", "bytes": target.stat().st_size,
                    "transferred_bytes": 0, "remote_transferred_bytes": 0, "local_copied_bytes": 0,
                    "attempts": 0, "seconds": time.monotonic() - start,
                    "started_utc": started_utc, "finished_utc": utc_now()}
        opener = opener or urllib.request.build_opener(SecureRedirect())
        budget = contextlib.nullcontext() if budget is None else budget
        transferred = 0
        attempts = 0
        ranged = {}
        if range_streams and ranged_eligible(a, ticket):
            try:
                value = acquire_ranged(a, cache, opener, budget, retries, range_streams)
            except RangeUnsupported as error:
                # Exactly one fallback to the original single-stream path below.
                transferred = getattr(error, "transferred", 0)
                stage = getattr(error, "stage", "probe")
                ranged = {"transfer_mode": "single",
                          "range_fallback": "range-unsupported" if stage == "probe" else "range-unsupported-after-probe"}
            else:
                retried = sum(len(r["resume_offsets"]) for r in value["parts"])
                return {"name": a["name"], "state": "downloaded-verified", "bytes": a["size_bytes"],
                        "transferred_bytes": value["transferred"], "remote_transferred_bytes": value["transferred"],
                        "local_copied_bytes": 0,
                        "attempts": 1 + retried,  # 1 + part-level retries; range_requests counts every GET.
                        "seconds": time.monotonic() - start,
                        "origin": urllib.parse.urlsplit(a["source"]).hostname,
                        "started_utc": started_utc, "finished_utc": utc_now(),
                        "transfer_mode": "ranged", "range_streams": range_streams, "range_fallback": None,
                        "parts": len(value["parts"]), "range_part_bytes": RANGE_PART,
                        "range_requests": value["requests"], "preallocation": value["preallocation"],
                        "hash_seconds": value["hash_seconds"], "part_timings": value["parts"]}
        slot = contextlib.nullcontext() if a["source"].startswith("repo:") else budget
        for attempt in range(retries + 1):
            attempts += 1
            fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=cache)
            try:
                h = hashlib.sha256()
                size = 0
                with os.fdopen(fd, "wb") as out, slot, stream_source(a, repo_root, opener, ticket) as source:
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
                        "origin": urllib.parse.urlsplit(ticket["url"] if ticket else a["source"]).hostname or "repo-local",
                        "started_utc": started_utc, "finished_utc": utc_now(), **ranged}
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                # Auth failures and not-found are not transient. Integrity errors
                # deliberately do not enter this retry path.
                if isinstance(error, urllib.error.HTTPError):
                    status = error.code
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


def acquire(manifest, cache, group, workers, repo_root=None, reserve_bytes=30 * 1024**3, tickets=None,
            range_streams=0, max_connections=None):
    require(workers in (1, 2, 4, 8), "Bounded concurrency required")
    require(range_streams in RANGE_STREAM_CHOICES, "Bounded range streams required")
    connections = workers if max_connections is None else max_connections
    require(connections in CONNECTION_CHOICES, "Bounded connection cap required")
    # Only HTTP requests hold a slot; coordinator threads never do, so there is no
    # nested wait. Open sockets <= connections; threads <= workers * (1 + range_streams).
    budget = threading.BoundedSemaphore(connections)
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
        started_utc = utc_now()
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
            futures = {pool.submit(acquire_one, a, cache, repo_root, ticket=tickets.get(a["sha256"]), budget=budget,
                                   range_streams=range_streams): a["name"] for a in rows}
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
                "native_compatibility_proven": False,
                "range_streams": range_streams, "max_connections": connections,
                "started_utc": started_utc, "finished_utc": utc_now()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--group", choices=("server", "server+kokoro", "full"), default="full")
    p.add_argument("--workers", type=int, choices=(1, 2, 4, 8), required=True)
    p.add_argument("--range-streams", type=int, choices=(0, 2, 4), default=0,
                   help="Parallel byte ranges per large public wheel; 0 (default) keeps one stream per artifact")
    p.add_argument("--max-connections", type=int, choices=(4, 8),
                   help="Global cap on open HTTP connections (default: --workers)")
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
        result = acquire(m, a.cache, a.group, a.workers, a.repo_root, tickets=tickets,
                         range_streams=a.range_streams, max_connections=a.max_connections)
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
