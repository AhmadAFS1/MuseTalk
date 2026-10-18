#!/usr/bin/env python3
"""Loopback tests for scripts/fetch_ranged_url.py (no external network)."""
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import fetch_dependency_wheels as f
import fetch_ranged_url as r

PAYLOAD = b"".join(hashlib.sha256(i.to_bytes(4, "big")).digest() for i in range(4000))  # 125 KiB


class RangedUrlTests(unittest.TestCase):
    def setUp(self):
        seen, lock = [], threading.Lock()
        self.seen = seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                header = self.headers.get("Range")
                with lock:
                    seen.append((self.path, header))
                if header is None or self.path.startswith("/ignore"):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(PAYLOAD)))
                    self.end_headers()
                    self.wfile.write(PAYLOAD)
                    return
                first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", header).groups())
                body = PAYLOAD[first:last + 1]
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (first, last, len(PAYLOAD)))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", '"v1"')
                self.send_header("Last-Modified", "Sat, 17 Oct 2026 00:00:00 GMT")
                self.end_headers()
                self.wfile.write(body)

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for patcher in (patch.object(f, "RANGE_PART", 16 * 1024), patch.object(f.time, "sleep"),
                        patch.dict(os.environ, {"no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost"})):
            patcher.start()
            self.addCleanup(patcher.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.sha = hashlib.sha256(PAYLOAD).hexdigest()

    def test_ranged_download_publishes_verified_object(self):
        out = self.dir / "bundle.tar.gz"
        result = r.fetch(self.base + "/obj?X-Amz-Signature=x", self.sha, len(PAYLOAD), out, streams=4)
        self.assertEqual(out.read_bytes(), PAYLOAD)
        self.assertEqual(result["mode"], "ranged")
        self.assertEqual(result["parts"], 8)
        self.assertEqual(sum(1 for _, h in self.seen if h), 8)
        self.assertEqual(list(self.dir.glob(".partial-*")), [])

    def test_server_ignoring_range_falls_back_to_verified_single_stream(self):
        out = self.dir / "bundle.tar.gz"
        result = r.fetch(self.base + "/ignore", self.sha, len(PAYLOAD), out, streams=4)
        self.assertEqual((result["mode"], out.read_bytes()), ("single-stream-fallback", PAYLOAD))

    def test_wrong_pin_publishes_nothing(self):
        out = self.dir / "bundle.tar.gz"
        for path in ("/obj", "/ignore"):
            with self.assertRaises(ValueError):
                r.fetch(self.base + path, "a" * 64, len(PAYLOAD), out, streams=4)
            self.assertFalse(out.exists())
            self.assertEqual(list(self.dir.glob(".partial-*")), [])

    def test_cli_requires_https_and_never_echoes_url(self):
        proc = subprocess.run([sys.executable, "-B", str(Path(r.__file__)), "--sha256", self.sha, "--bytes",
                               str(len(PAYLOAD)), "--out", str(self.dir / "x")],
                              input=self.base + "/obj?X-Amz-Signature=secret\n", capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["state"], "failed")
        self.assertNotIn("secret", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
