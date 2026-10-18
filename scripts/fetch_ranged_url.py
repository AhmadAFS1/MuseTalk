#!/usr/bin/env python3
"""Multi-connection, sha256-verified download of one large https object.

Used by the Vast bootstrap to prefetch the r5 engine bundle from an
EC2-presigned S3 URL before the venv exists (stdlib only). The URL is read
from stdin, never argv. Ranged parts reuse fetch_dependency_wheels' tested
machinery (per-part identity checks, resume, stall detection); a server that
refuses ranges falls back to one verified stream. Publishes --out only after
the full sha256 matches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_dependency_wheels as f  # noqa: E402


def single_stream(art, out, opener):
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=out.parent)
    try:
        digest, total = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as handle, f.stream_source(art, None, opener) as source:
            while chunk := source.read(f.CHUNK):
                total += len(chunk)
                f.require(total <= art["size_bytes"], "Object larger than pinned size")
                digest.update(chunk)
                handle.write(chunk)
        f.require(total == art["size_bytes"] and digest.hexdigest() == art["sha256"], "Object integrity mismatch")
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fetch(url, sha256, size, out, streams=8, opener=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    art = {"name": "ranged-object", "version": "0", "filename": out.name, "source": url,
           "sha256": sha256, "size_bytes": size}
    opener = opener or urllib.request.build_opener(f.SecureRedirect())
    started = time.monotonic()
    try:
        result = f.acquire_ranged(art, out.parent, opener, threading.BoundedSemaphore(streams), 2, streams)
        mode = "ranged"
    except f.RangeUnsupported:
        single_stream(art, out, opener)
        result, mode = {"parts": [], "stalls_resumed": 0}, "single-stream-fallback"
    return {"state": "complete", "mode": mode, "bytes": size, "streams": streams,
            "parts": len(result.get("parts") or []), "stalls_resumed": result.get("stalls_resumed", 0),
            "seconds": round(time.monotonic() - started, 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sha256", required=True)
    p.add_argument("--bytes", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--streams", type=int, choices=(1, 2, 4, 8), default=8)
    a = p.parse_args()
    try:
        url = sys.stdin.readline().strip()
        u = urllib.parse.urlsplit(url)
        f.require(u.scheme == "https" and u.hostname and not u.username, "https URL required on stdin")
        f.require(len(a.sha256) == 64 and all(c in "0123456789abcdef" for c in a.sha256) and a.bytes > 0, "Bad pin")
        print(json.dumps(fetch(url, a.sha256, a.bytes, a.out, a.streams)), flush=True)
        return 0
    except BaseException as error:  # Never print the URL (it carries a signature).
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "detail": str(error) if type(error) is ValueError else "Details suppressed"}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
