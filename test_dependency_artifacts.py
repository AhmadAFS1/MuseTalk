"""Stdlib fixtures only: no pip, CUDA, external network or package installation."""
import datetime as dt
import errno
import hashlib
import http.client
import http.server
import io
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import dependency_artifacts as d
import fetch_dependency_wheels as f
import build_dependency_wheelhouse as b
import benchmark_dependency_install as bench
import dependency_optimization_ec2 as control
import dependency_local_matrix as matrix
import dependency_wheel_publish_ec2 as publish
import benchmark_dependency_delivery as delivery
import dependency_s3_range_probe as range_probe
import install_locked_dependencies as locked
import dependency_followup_worker as followup
import dependency_application_matrix_ec2 as application
import summarize_dependency_optimization as summary
import dependency_s3_full_worker as full_s3
import dependency_canonical_integration_worker as canonical_integration
import dependency_delivery_confirmation_worker as confirmation
import dependency_warm_start_worker as warm_start
import musetalk_install_state as install_state


def row(name="example", version="1.0", url="https://files.pythonhosted.org/example-1.0-py3-none-any.whl", content=b"wheel"):
    return {"metadata": {"name": name, "version": version}, "download_info": {
        "url": url, "archive_info": {"hashes": {"sha256": hashlib.sha256(content).hexdigest()}}}}


def manifest():
    artifacts = []
    for name, version in d.REQUIRED.items():
        a = d.artifact(row(name, version, "https://files.pythonhosted.org/" + name.replace("-", "_")
                          + "-" + version + "-py3-none-any.whl"), "bootstrap" if name in d.BOOTSTRAP else "server")
        a["size_bytes"] = 5
        artifacts.append(a)
    m = {"schema": d.SCHEMA, "contract": {"os": "Linux", "machine": "x86_64", "implementation": "cpython",
         "python": "3.10", "matrix": "cu121", "validated_glibc": "2.35"},
         "requirement_inputs": {}, "artifacts": artifacts, "source_revision_provenance": d.SOURCE}
    m["artifact_id"] = d.digest(d.identity(m))
    return m


class ManifestTests(unittest.TestCase):
    def test_name_normalization(self):
        self.assertEqual(d.normalized("torch_tensorrt"), "torch-tensorrt")

    def test_wheel_tags_and_hash(self):
        a = d.artifact(row(), "server")
        self.assertEqual(a["wheel_tags"], {"python": "py3", "abi": "none", "platform": "any"})
        self.assertEqual(a["sha256"], hashlib.sha256(b"wheel").hexdigest())

    def test_secret_and_temporary_urls_rejected(self):
        for url in ("https://user:secret@files.pythonhosted.org/a.whl", "https://files.pythonhosted.org/a.whl?token=secret",
                    "http://files.pythonhosted.org/a.whl", "https://unreviewed.example/a.whl",
                    "https://files.pythonhosted.org/a.whl#token=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                d.artifact(row(url=url), "server")

    def test_only_reviewed_repo_wheel(self):
        a = d.artifact(row("mmcv", "2.1.0", "file:///workspace/MuseTalk/" + d.MMCV), "avatar-prep")
        self.assertEqual(a["source"], "repo:" + d.MMCV)
        with self.assertRaises(ValueError):
            d.artifact(row(url="file:///etc/secret.whl"), "server")

    def test_missing_hash_rejected(self):
        r = row()
        r["download_info"]["archive_info"] = {}
        with self.assertRaises(ValueError):
            d.artifact(r, "server")

    def test_unreviewed_hash_repair_rejected_without_network(self):
        r = row()
        r["download_info"]["archive_info"] = {}
        with patch.object(d.urllib.request, "urlopen") as network, self.assertRaises(ValueError):
            d.complete_missing_hash(r)
        network.assert_not_called()

    def test_reviewed_hash_repair_checks_both_sources(self):
        r = row("Jinja2", "3.1.6", "https://download.pytorch.org/whl/jinja2-3.1.6-py3-none-any.whl")
        r["download_info"]["archive_info"] = {}
        index = {"urls": [{"filename": "jinja2-3.1.6-py3-none-any.whl", "size": 5,
                            "digests": {"sha256": hashlib.sha256(b"wheel").hexdigest()}}]}
        with patch.object(d.urllib.request, "urlopen", side_effect=[io.BytesIO(json.dumps(index).encode()), io.BytesIO(b"wheel")]):
            result = d.complete_missing_hash(r)
        self.assertIn("historical-payload-hash-unavailable", result["hash_provenance"])
        self.assertEqual(result["download_info"]["archive_info"]["hashes"]["sha256"], hashlib.sha256(b"wheel").hexdigest())

    def test_source_mismatch_during_hash_repair_fails(self):
        r = row("MarkupSafe", "3.0.3", "https://download.pytorch.org/a.whl")
        r["download_info"]["archive_info"] = {}
        index = {"urls": [{"filename": "a.whl", "size": 5, "digests": {"sha256": "0" * 64}}]}
        with patch.object(d.urllib.request, "urlopen", side_effect=[io.BytesIO(json.dumps(index).encode()), io.BytesIO(b"wheel")]):
            with self.assertRaisesRegex(ValueError, "differs from official-index"):
                d.complete_missing_hash(r)

    def test_complete_lock_has_no_direct_url(self):
        lock = d.offline_lock(manifest(), "full")
        self.assertNotIn("https:", lock)
        self.assertNotIn(" @ ", lock)
        self.assertIn("torch-tensorrt===2.5.0 --hash=sha256:", lock)

    def test_sdist_or_unknown_size_blocks_offline_lock(self):
        for key, value in (("format", "sdist"), ("size_bytes", None)):
            m = manifest()
            m["artifacts"][0][key] = value
            m["artifact_id"] = d.digest(d.identity(m))
            with self.assertRaises(ValueError):
                d.offline_lock(m, "full")

    def test_changed_contract_or_gpu_artifact_rejected(self):
        for mutation in (lambda m: m["contract"].update(python="3.11"),
                         lambda m: next(a for a in m["artifacts"] if a["name"] == "torch-tensorrt").update(version="2.5.0+cu121")):
            m = manifest()
            mutation(m)
            m["artifact_id"] = d.digest(d.identity(m))
            with self.assertRaises(ValueError):
                d.validate_manifest(m)

    def test_unrelated_source_commit_does_not_invalidate_dependencies(self):
        m = manifest()
        before = d.digest(d.identity(m))
        m["source_revision_provenance"] = "a" * 40
        self.assertEqual(before, d.digest(d.identity(m)))

    def test_external_manifest_digest_required(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "manifest.json"
            p.write_text(json.dumps(manifest()))
            with self.assertRaises(ValueError):
                f.read_trusted(p, "0" * 64)
            self.assertEqual(f.read_trusted(p, f.sha256_file(p))["schema"], 1)


class FakeResponse(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}


class FakeOpener:
    def __init__(self, data=b"wheel", length=None, failures=()):
        self.data, self.length, self.failures, self.calls = data, length, list(failures), 0

    def open(self, *args, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return FakeResponse(self.data, self.length)


class AcquisitionTests(unittest.TestCase):
    def private_ticket(self, a):
        key = f.PRIVATE_PREFIX+"a"*64+"/"+a["filename"]
        return {"url":"https://"+f.BUCKET+".s3.us-east-1.amazonaws.com/"+key+"?X-Amz-Signature=secret",
                "sha256":a["sha256"],"owner":f.OWNER,"key":key}

    def test_private_ticket_destination_and_object_identity(self):
        a = d.artifact(row(), "server")
        ticket = self.private_ticket(a)
        self.assertEqual(f.check_ticket(a, ticket), ticket["url"])
        for change in ({"owner":"wrong"},{"sha256":"0"*64},{"key":f.PRIVATE_PREFIX+"../"+a["filename"]},
                       {"url":ticket["url"].replace(".s3.us-east-1.amazonaws.com", ".evil.example")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                f.check_ticket(a, dict(ticket, **change))

    def test_built_wheel_uses_private_ticket_and_safe_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            a = d.artifact(row(), "server");a["source"]="build:"+"0"*64
            ticket = self.private_ticket(a)
            opener = FakeOpener()
            receipt = f.acquire_one(a, folder, opener=opener, ticket=ticket)
            self.assertEqual(receipt["remote_transferred_bytes"], 5)
            self.assertEqual(receipt["local_copied_bytes"], 0)
            self.assertNotIn("secret", json.dumps(receipt))

    def test_missing_private_ticket_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            a = d.artifact(row(), "server");a["source"]="build:"+"0"*64
            opener = FakeOpener()
            with self.assertRaises(ValueError):f.acquire_one(a, folder, opener=opener)
            self.assertEqual(opener.calls, 0)

    def test_verified_download_and_cache_hit_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            a = d.artifact(row(), "server")
            opener = FakeOpener()
            first = f.acquire_one(a, folder, opener=opener)
            second = f.acquire_one(a, folder, opener=opener)
            self.assertEqual(first["transferred_bytes"], 5)
            self.assertEqual(second["transferred_bytes"], 0)
            self.assertEqual(opener.calls, 1)
            self.assertEqual(list(Path(folder).glob(".partial-*")), [])

    def test_truncation_and_hash_failure_never_publish(self):
        for opener in (FakeOpener(b"bad"), FakeOpener(b"wheel", length=6)):
            with self.subTest(opener=opener), tempfile.TemporaryDirectory() as folder:
                a = d.artifact(row(), "server")
                with self.assertRaises(ValueError):
                    f.acquire_one(a, folder, opener=opener)
                self.assertFalse((Path(folder)/a["filename"]).exists())
                self.assertEqual(list(Path(folder).glob(".partial-*")), [])

    def test_size_limit_rejects_oversized_response(self):
        with tempfile.TemporaryDirectory() as folder:
            a = d.artifact(row(), "server")
            a["size_bytes"] = 4
            with self.assertRaises(ValueError):
                f.acquire_one(a, folder, opener=FakeOpener())

    def test_corrupt_cache_fails_without_redownload(self):
        with tempfile.TemporaryDirectory() as folder:
            a = d.artifact(row(), "server")
            (Path(folder)/a["filename"]).write_bytes(b"bad")
            opener = FakeOpener()
            with self.assertRaises(ValueError):
                f.acquire_one(a, folder, opener=opener)
            self.assertEqual(opener.calls, 0)

    def test_auth_failure_is_bounded_and_does_not_expose_url(self):
        with tempfile.TemporaryDirectory() as folder:
            opener = FakeOpener(failures=[urllib.error.HTTPError("https://private/?secret", 403, "Forbidden", {}, None)])
            with self.assertRaisesRegex(ValueError, "^Artifact HTTP failure 403$"):
                f.acquire_one(d.artifact(row(), "server"), folder, opener=opener)
            self.assertEqual(opener.calls, 1)

    def test_transient_failure_retries_then_succeeds(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(f.time, "sleep"):
            opener = FakeOpener(failures=[urllib.error.URLError("secret-bearing exception")])
            result = f.acquire_one(d.artifact(row(), "server"), folder, opener=opener)
            self.assertEqual(result["attempts"], 2)

    def test_transport_failure_exhaustion_suppresses_secret(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(f.time, "sleep"):
            opener = FakeOpener(failures=[urllib.error.URLError("secret")] * 3)
            with self.assertRaisesRegex(ValueError, "^Bounded artifact transport retries exhausted$"):
                f.acquire_one(d.artifact(row(), "server"), folder, opener=opener)

    def test_simultaneous_installer_lock_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"claim"
            with f.lock(p), self.assertRaises(BlockingIOError):
                with f.lock(p):
                    self.fail("Must not enter concurrent writer")

    def test_low_disk_blocks_before_any_acquisition(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(f.shutil, "disk_usage") as usage:
            usage.return_value = type("Usage", (), {"free": 1})()
            with self.assertRaisesRegex(ValueError, "Insufficient scratch disk"):
                f.acquire(manifest(), folder, "full", 2)

    def test_cache_identity_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)/"cache"
            cache.mkdir()
            (cache/".artifact-id").write_text("wrong")
            with self.assertRaisesRegex(ValueError, "Cache artifact identity mismatch"):
                f.acquire(manifest(), cache, "full", 2, reserve_bytes=0)
            link = Path(folder)/"link"
            link.symlink_to(cache, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Cache symlink rejected"):
                f.acquire(manifest(), link, "full", 2, reserve_bytes=0)

    def test_private_ticketed_wheels_are_acquired_first(self):
        rows = [{"name": "tensorrt-cu12-libs", "sha256": "a"*64, "size_bytes": 2_037_510_258},
                {"name": "docopt", "sha256": "b"*64, "size_bytes": 13_723},
                {"name": "torch", "sha256": "c"*64, "size_bytes": 780_000_000},
                {"name": "antlr4-python3-runtime", "sha256": "d"*64, "size_bytes": 144_573},
                {"name": "unknown-size", "sha256": "e"*64, "size_bytes": None}]
        order = [r["name"] for r in f.acquisition_order(rows, {"b"*64, "d"*64})]
        self.assertEqual(order, ["antlr4-python3-runtime", "docopt", "tensorrt-cu12-libs", "torch", "unknown-size"])

    def test_insecure_redirect_rejected(self):
        with self.assertRaises(ValueError):
            f.SecureRedirect().redirect_request(None, None, 302, "", {}, "http://example.test/file")


LEGACY_RECEIPT_KEYS = {"name", "state", "bytes", "transferred_bytes", "remote_transferred_bytes",
                       "local_copied_bytes", "attempts", "seconds", "origin"}
LEGACY_CACHE_KEYS = LEGACY_RECEIPT_KEYS - {"origin"}
LEGACY_ACQUISITION_KEYS = {"state", "artifact_id", "group", "workers", "seconds", "transferred_bytes",
                           "remote_transferred_bytes", "local_copied_bytes", "cache_hits", "artifacts",
                           "native_compatibility_proven"}
TIMESTAMP_KEYS = {"started_utc", "finished_utc"}
RANGE_PAYLOAD = bytes((i * 7 + 3) % 251 for i in range(60))  # 8 parts of 8 bytes (last part 4 bytes)


def pause(seconds):
    # time.sleep is patched out in these tests to skip retry backoff.
    threading.Event().wait(seconds)


class ConnectionTracker:
    def __init__(self):
        self.lock, self.current, self.peak = threading.Lock(), 0, 0

    def opened(self):
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def closed(self):
        with self.lock:
            self.current -= 1


class RangeResponse(io.BytesIO):
    """A 206/200 fixture that records how many body bytes the client read."""
    def __init__(self, data, status=206, headers=None, delay=0.0, tracker=None, fail_after=None):
        super().__init__(data)
        self.status, self.headers, self.delay, self.tracker = status, dict(headers or {}), delay, tracker
        self.read_bytes, self.fail_after = 0, fail_after
        if tracker:
            tracker.opened()

    def read(self, n=-1):
        if self.delay:
            pause(self.delay)
        if self.fail_after is not None and self.read_bytes >= self.fail_after[0]:
            raise self.fail_after[1]
        if self.fail_after is not None:
            n = min(n, self.fail_after[0] - self.read_bytes) if n >= 0 else self.fail_after[0] - self.read_bytes
        chunk = super().read(n)
        self.read_bytes += len(chunk)
        return chunk

    def close(self):
        if not self.closed and self.tracker:
            self.tracker.closed()
        super().close()


class RangeOpener:
    """Serves one payload: 206 for a single Range header, 200 otherwise. hook(n, first, last)
    may raise, return a response, or return None for the correct default response."""
    def __init__(self, data=RANGE_PAYLOAD, hook=None, etag='"fixture-etag"',
                 modified="Sat, 17 Oct 2026 00:00:00 GMT", delay=0.0, tracker=None):
        self.data, self.hook, self.etag, self.modified = data, hook, etag, modified
        self.delay, self.tracker = delay, tracker
        self.lock, self.ranges, self.responses = threading.Lock(), [], []

    @property
    def calls(self):
        return len(self.ranges)

    def open(self, request, timeout=None):
        header = request.get_header("Range")
        first = last = None
        if header is not None:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
            first, last = int(match[1]), int(match[2])
        with self.lock:
            n = len(self.ranges)
            self.ranges.append((first, last))
        response = self.hook(n, first, last) if self.hook else None
        if response is None:
            response = self.full() if first is None else self.part(first, last)
        with self.lock:
            self.responses.append(response)
        return response

    def part(self, first, last, body=None, delay=None, fail_after=None, **headers):
        body = self.data[first:last + 1] if body is None else body
        value = {"Content-Range": "bytes %d-%d/%d" % (first, last, len(self.data)),
                 "Content-Length": str(last - first + 1), "ETag": self.etag, "Last-Modified": self.modified}
        for key, change in headers.items():
            key = key.replace("_", "-")
            if change is None:
                value.pop(key, None)
            else:
                value[key] = change
        return RangeResponse(body, 206, value, self.delay if delay is None else delay, self.tracker, fail_after)

    def full(self, status=200):
        return RangeResponse(self.data, status, {"Content-Length": str(len(self.data))}, self.delay, self.tracker)


def http_error(status):
    # A body object keeps HTTPError.close() valid on Python 3.9 as well.
    return urllib.error.HTTPError("https://private/?secret", status, "No", {}, io.BytesIO())


def large(data=RANGE_PAYLOAD, host="pypi.nvidia.com", name="tensorrt-cu12-libs"):
    filename = name.replace("-", "_") + "-10.3.0-py2.py3-none-manylinux_2_17_x86_64.whl"
    return {"name": name, "version": "10.3.0", "filename": filename,
            "source": "https://" + host + "/" + name + "/" + filename,
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


class RangedAcquisitionTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("RANGED_MIN_BYTES", 16), ("RANGE_PART", 8)):
            patcher = patch.object(f, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        sleeper = patch.object(f.time, "sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = Path(folder.name)

    def assert_unpublished(self, a):
        self.assertFalse((self.cache / a["filename"]).exists())
        self.assertEqual(list(self.cache.glob(".partial-*")), [])

    def private_ticket(self, a):
        key = f.PRIVATE_PREFIX + "a" * 64 + "/" + a["filename"]
        return {"url": "https://" + f.BUCKET + ".s3.us-east-1.amazonaws.com/" + key + "?X-Amz-Signature=secret",
                "sha256": a["sha256"], "owner": f.OWNER, "key": key}

    # 1
    def test_ranged_success_publishes_verified_file(self):
        for streams in (2, 4):
            with self.subTest(streams=streams):
                a = large()
                opener = RangeOpener()
                receipt = f.acquire_one(a, self.cache, opener=opener, range_streams=streams)
                self.assertEqual((self.cache / a["filename"]).read_bytes(), RANGE_PAYLOAD)
                self.assertEqual(opener.calls, 8)
                self.assertEqual(sorted(opener.ranges), [(i, min(i + 8, 60) - 1) for i in range(0, 60, 8)])
                self.assertEqual(opener.ranges[0], (0, 7))  # Probe part first.
                self.assertEqual(list(self.cache.glob(".partial-*")), [])
                self.assertEqual((receipt["transfer_mode"], receipt["parts"], receipt["range_streams"]),
                                 ("ranged", 8, streams))
                self.assertEqual((receipt["range_requests"], receipt["attempts"], receipt["range_fallback"]), (8, 1, None))
                self.assertEqual((receipt["bytes"], receipt["transferred_bytes"]), (60, 60))
                self.assertEqual(receipt["origin"], "pypi.nvidia.com")
                self.assertTrue(all(r["finished_utc"] and r["seconds"] is not None and r["attempts"] == 1
                                    for r in receipt["part_timings"]))
                (self.cache / a["filename"]).unlink()

    # 2
    def test_ineligible_or_disabled_sources_never_send_range(self):
        repo = self.cache / "repo"
        (repo / "third_party_wheels").mkdir(parents=True)
        (repo / "third_party_wheels/local.whl").write_bytes(RANGE_PAYLOAD)
        built = large(name="private-wheel")
        built["source"] = "build:" + "0" * 64
        repo_row = large(name="mmcv")
        repo_row["source"] = "repo:third_party_wheels/local.whl"
        cases = (("below-threshold", large(RANGE_PAYLOAD[:15]), {}, 4),
                 ("unknown-host", large(host="github.com"), {}, 4),
                 ("ticketed-build", built, {"ticket": self.private_ticket(built)}, 4),
                 ("repo-local", repo_row, {"repo_root": repo}, 4),
                 ("default-off", large(), {}, 0))
        for label, a, extra, streams in cases:
            with self.subTest(label=label):
                opener = RangeOpener(RANGE_PAYLOAD[:a["size_bytes"]])
                (self.cache / label).mkdir()
                receipt = f.acquire_one(a, self.cache / label, opener=opener, range_streams=streams, **extra)
                self.assertEqual([r for r in opener.ranges if r != (None, None)], [])
                self.assertEqual(opener.calls, 0 if label == "repo-local" else 1)
                self.assertEqual(set(receipt), LEGACY_RECEIPT_KEYS | TIMESTAMP_KEYS)
                self.assertEqual(receipt["attempts"], 1)

    # 3
    def test_probe_answered_200_falls_back_once_without_reading_body(self):
        a = large()
        def hook(n, first, last):
            if n == 0:
                pause(0.05)  # A slow probe: nothing else may be requested while it is pending.
                return opener.full()
        opener = RangeOpener(hook=hook)
        receipt = f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assertEqual(opener.ranges, [(0, 7), (None, None)])
        self.assertEqual(opener.responses[0].read_bytes, 0)
        self.assertTrue(opener.responses[0].closed)
        self.assertEqual((receipt["transfer_mode"], receipt["range_fallback"], receipt["attempts"]),
                         ("single", "range-unsupported", 1))
        self.assertEqual((self.cache / a["filename"]).read_bytes(), RANGE_PAYLOAD)
        self.assertEqual(list(self.cache.glob(".partial-*")), [])

    # 4
    def test_wrong_range_headers_rejected_on_any_part(self):
        changes = (("start", lambda a, b: {"Content_Range": "bytes %d-%d/60" % (a + 1, b)}, "Wrong range response"),
                   ("end", lambda a, b: {"Content_Range": "bytes %d-%d/60" % (a, b + 1)}, "Wrong range response"),
                   ("total", lambda a, b: {"Content_Range": "bytes %d-%d/61" % (a, b)}, "Wrong range response"),
                   ("no-range", lambda a, b: {"Content_Range": None}, "Wrong range response"),
                   ("length", lambda a, b: {"Content_Length": str(b - a + 2)}, "Range length differs"),
                   ("no-length", lambda a, b: {"Content_Length": None}, "Range length differs"))
        for target in (0, 24):  # The probe part and a fanned-out part.
            for label, change, message in changes:
                with self.subTest(target=target, change=label):
                    a = large()
                    def hook(n, first, last, change=change, target=target):
                        if first == target:
                            return opener.part(first, last, **change(first, last))
                    opener = RangeOpener(hook=hook)
                    with self.assertRaisesRegex(ValueError, "^" + message + "$"):
                        f.acquire_one(a, self.cache, opener=opener, range_streams=4)
                    self.assert_unpublished(a)

    # 5
    def test_object_identity_change_between_parts_refused(self):
        for header in ("ETag", "Last_Modified"):
            with self.subTest(header=header):
                a = large()
                hook = lambda n, first, last, header=header: opener.part(first, last, **{header: "changed"}) \
                    if first == 32 else None
                opener = RangeOpener(hook=hook)
                with self.assertRaisesRegex(ValueError, "^Object identity changed between ranges$"):
                    f.acquire_one(a, self.cache, opener=opener, range_streams=4)
                self.assert_unpublished(a)

    # 6
    def test_short_or_broken_part_resumes_from_written_offset(self):
        failures = (("clean-eof", None), ("incomplete-read", http.client.IncompleteRead(b"")),
                    ("tls", ssl.SSLError("fixture")), ("reset", ConnectionResetError()))
        for label, failure in failures:
            with self.subTest(label=label):
                a = large()
                seen = []
                def hook(n, first, last, failure=failure):
                    if first == 16 and not seen:
                        seen.append(n)
                        if failure is None:
                            return opener.part(first, last, body=RANGE_PAYLOAD[16:19])
                        return opener.part(first, last, fail_after=(3, failure))
                opener = RangeOpener(hook=hook)
                receipt = f.acquire_one(a, self.cache, opener=opener, range_streams=2)
                self.assertIn((19, 23), opener.ranges)
                self.assertEqual(opener.calls, 9)
                self.assertEqual(receipt["part_timings"][2]["resume_offsets"], [19])
                self.assertEqual((receipt["attempts"], receipt["transferred_bytes"]), (2, 60))
                self.assertEqual((self.cache / a["filename"]).read_bytes(), RANGE_PAYLOAD)
                (self.cache / a["filename"]).unlink()

    # 7
    def test_part_status_policy_and_cancellation(self):
        a = large()
        failed = []
        def transient(n, first, last):
            if first == 16 and not failed:
                failed.append(n)
                raise http_error(503)
        opener = RangeOpener(hook=transient)
        receipt = f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assertEqual((opener.calls, receipt["attempts"]), (9, 2))
        self.assertEqual(receipt["part_timings"][2]["resume_offsets"], [16])
        for status in (403, 404, 416):
            with self.subTest(status=status), patch.object(f, "CHUNK", 1):
                a = large(name="nvidia-cublas-cu12")
                def hard(n, first, last, status=status):
                    if first == 16:
                        pause(0.15)  # Fails while part 1 is mid-body.
                        raise http_error(status)
                    if first:
                        return opener.part(first, last, delay=0.1)  # 1-byte reads: 0.8 s per part.
                opener = RangeOpener(hook=hard)
                with self.assertRaisesRegex(ValueError, "^Artifact HTTP failure %d$" % status) as caught:
                    f.acquire_one(a, self.cache, opener=opener, range_streams=2)
                self.assertNotIn("secret", str(caught.exception))
                self.assertNotIn("https", str(caught.exception))
                self.assertLess(opener.calls, 8)  # Queued parts cancelled, never requested.
                in_flight = [r for r in opener.responses if r.delay]
                self.assertIn((8, 15), opener.ranges)
                self.assertTrue(in_flight and all(r.read_bytes < 8 and r.closed for r in in_flight))  # Aborted mid-part.
                self.assert_unpublished(a)

    # 8
    def test_part_retries_exhausted_leaves_no_partial(self):
        a = large()
        def broken(n, first, last):
            if first == 8:
                raise urllib.error.URLError("https://private/?secret")
        opener = RangeOpener(hook=broken)
        with self.assertRaisesRegex(ValueError, "^Bounded range retries exhausted$"):
            f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assertEqual(opener.ranges.count((8, 15)), 3)
        self.assert_unpublished(a)

    # 9
    def test_corrupted_range_bytes_fail_integrity_without_retry(self):
        a = large()
        opener = RangeOpener(hook=lambda n, first, last: opener.part(first, last, body=b"X" * 8) if first == 16 else None)
        with self.assertRaisesRegex(ValueError, "^Artifact integrity mismatch$"):
            f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assertEqual(opener.calls, 8)
        self.assert_unpublished(a)

    # 10
    def test_encoded_range_rejected(self):
        a = large()
        opener = RangeOpener(hook=lambda n, first, last: opener.part(first, last, Content_Encoding="gzip") if first == 40 else None)
        with self.assertRaisesRegex(ValueError, "^Encoded range rejected$"):
            f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assert_unpublished(a)
        opener = RangeOpener(hook=lambda n, first, last: opener.part(first, last, Content_Encoding="identity"))
        self.assertEqual(f.acquire_one(a, self.cache, opener=opener, range_streams=4)["transfer_mode"], "ranged")

    # 11
    def test_connection_cap_bounds_open_responses_without_deadlock(self):
        for streams, cap in ((4, 4), (4, 1), (0, 1)):
            with self.subTest(streams=streams, cap=cap), patch.object(f, "RANGED_MIN_BYTES", 4), \
                 patch.object(f, "RANGE_PART", 2):
                tracker = ConnectionTracker()
                opener = RangeOpener(b"wheel", delay=0.002, tracker=tracker)
                outcome = {}
                def run():
                    try:
                        outcome["value"] = f.acquire(manifest(), self.cache / ("cap-%d-%d" % (streams, cap)), "full", 4,
                                                     reserve_bytes=0, range_streams=streams, max_connections=cap)
                    except BaseException as error:
                        outcome["error"] = error
                with patch.object(f.urllib.request, "build_opener", return_value=opener):
                    worker = threading.Thread(target=run, daemon=True)
                    worker.start()
                    worker.join(timeout=30)
                self.assertFalse(worker.is_alive(), "acquisition deadlocked")
                self.assertNotIn("error", outcome)
                value = outcome["value"]
                self.assertEqual((value["state"], value["max_connections"], value["range_streams"]),
                                 ("acquisition-complete", cap, streams))
                self.assertLessEqual(tracker.peak, cap)
                self.assertEqual(tracker.current, 0)
                if cap > 1:
                    self.assertGreater(tracker.peak, 1)
                modes = {r.get("transfer_mode", "single") for r in value["artifacts"]}
                self.assertEqual(modes, {"ranged"} if streams else {"single"})
                self.assertEqual(opener.calls, len(value["artifacts"]) * (3 if streams else 1))
        for streams, cap in ((3, 4), (2, 3), (True, 4)):
            with self.subTest(invalid=(streams, cap)), self.assertRaisesRegex(ValueError, "Bounded"):
                f.acquire(manifest(), self.cache / "invalid", "full", 4, reserve_bytes=0,
                          range_streams=streams, max_connections=cap)

    # 12
    def test_preallocation_enospc_is_fatal_and_unsupported_falls_back(self):
        a = large()
        opener = RangeOpener()
        with patch.object(f.os, "posix_fallocate", side_effect=OSError(errno.ENOSPC, "No space"), create=True):
            with self.assertRaisesRegex(ValueError, "^Insufficient disk for ranged artifact$"):
                f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assertEqual(opener.calls, 0)
        self.assert_unpublished(a)
        for label, replacement in (("unsupported", MagicMock(side_effect=OSError(errno.EOPNOTSUPP, "No"))),
                                   ("absent", None)):
            with self.subTest(label=label), patch.object(f.os, "posix_fallocate", replacement, create=True):
                receipt = f.acquire_one(a, self.cache, opener=RangeOpener(), range_streams=4)
                self.assertEqual((receipt["preallocation"], receipt["transfer_mode"]), ("ftruncate", "ranged"))
                self.assertEqual((self.cache / a["filename"]).read_bytes(), RANGE_PAYLOAD)
                (self.cache / a["filename"]).unlink()

    # 13
    def test_cache_hit_is_offline_and_redirect_keeps_range_https_only(self):
        a = large()
        (self.cache / a["filename"]).write_bytes(RANGE_PAYLOAD)
        opener = RangeOpener()
        receipt = f.acquire_one(a, self.cache, opener=opener, range_streams=4)
        self.assertEqual((receipt["state"], opener.calls), ("verified-cache-hit", 0))
        request = urllib.request.Request(a["source"], headers={"Range": "bytes=8-15", "User-Agent": "fixture"})
        redirected = f.SecureRedirect().redirect_request(request, None, 302, "Found", {},
                                                         "https://download-r2.pytorch.org/redirected.whl")
        self.assertEqual(redirected.get_header("Range"), "bytes=8-15")
        with self.assertRaisesRegex(ValueError, "Unsafe artifact redirect"):
            f.SecureRedirect().redirect_request(request, None, 302, "Found", {}, "http://pypi.nvidia.com/x.whl")

    # 15
    def test_receipts_keep_summarizer_fields_and_contain_no_secrets(self):
        built = large(name="private-wheel")
        built["source"] = "build:" + "0" * 64
        receipts = {
            "ranged": f.acquire_one(large(), self.cache, opener=RangeOpener(), range_streams=4),
            "fallback": f.acquire_one(large(name="torch"), self.cache, range_streams=4,
                                      opener=RangeOpener(hook=lambda n, first, last: RangeOpener().full() if n == 0 else None)),
            "private": f.acquire_one(built, self.cache, opener=RangeOpener(), ticket=self.private_ticket(built),
                                     range_streams=4),
            "single": f.acquire_one(large(name="nvidia-cudnn-cu12"), self.cache, opener=RangeOpener()),
            "cache-hit": f.acquire_one(large(), self.cache, opener=RangeOpener(), range_streams=4),
        }
        for label, receipt in receipts.items():
            with self.subTest(label=label):
                expected = LEGACY_CACHE_KEYS if label == "cache-hit" else LEGACY_RECEIPT_KEYS
                self.assertLessEqual(expected | TIMESTAMP_KEYS, set(receipt))
                text = json.dumps(receipt, allow_nan=False)
                for forbidden in ("secret", "https://", "X-Amz", "?"):
                    self.assertNotIn(forbidden, text)
                for key in TIMESTAMP_KEYS:
                    self.assertIsNotNone(dt.datetime.fromisoformat(receipt[key]).tzinfo)
        for timing in receipts["ranged"]["part_timings"]:
            self.assertLessEqual(dt.datetime.fromisoformat(timing["started_utc"]),
                                 dt.datetime.fromisoformat(timing["finished_utc"]))
        with patch.object(f, "RANGED_MIN_BYTES", 4), patch.object(f, "RANGE_PART", 2), \
             patch.object(f.urllib.request, "build_opener", return_value=RangeOpener(b"wheel")):
            value = f.acquire(manifest(), self.cache / "acquisition", "full", 2, reserve_bytes=0, range_streams=2)
        self.assertLessEqual(LEGACY_ACQUISITION_KEYS | TIMESTAMP_KEYS, set(value))
        self.assertEqual(value["max_connections"], 2)
        summarized = summary.summarize({"prepared_utc": "fixture", "trials_and_builds": {"delivery-ranged-1": {
            "trial": "ranged-1", "state": "delivery-trial-passed", "method": "wheelhouse-uv", "workers": 2,
            "packages_ready_seconds": 1, "validated_ready_seconds": 2, "acquisition": value}}})
        self.assertEqual(summarized["trials"][0]["acquisition_seconds"], value["seconds"])

    def test_cli_and_eligibility_bounds(self):
        self.assertTrue(f.ranged_eligible(large()))
        for change in ({"source": "http://pypi.nvidia.com/x.whl"}, {"source": "https://pypi.nvidia.com:8443/x.whl"},
                       {"source": "https://u:p@pypi.nvidia.com/x.whl"}, {"source": "https://pypi.nvidia.com/x.whl?t=1"},
                       {"size_bytes": None}, {"size_bytes": True}, {"size_bytes": f.MAX_FILE + 1}):
            with self.subTest(change=change):
                self.assertFalse(f.ranged_eligible(dict(large(), **change)))
        self.assertFalse(f.ranged_eligible(large(), ticket={"url": "https://example"}))
        with patch.object(f, "RANGED_MIN_BYTES", 256 * 1024**2):
            self.assertFalse(f.ranged_eligible(dict(large(), size_bytes=209_460_013)))  # triton stays single-stream
            self.assertTrue(f.ranged_eligible(dict(large(), size_bytes=2_037_510_258)))
        base = ["fetch", "--manifest", "m.json", "--manifest-sha256", "0" * 64, "--cache", str(self.cache),
                "--workers", "4", "--receipt"]
        for bad in (["--range-streams", "3"], ["--max-connections", "2"]):
            with self.subTest(bad=bad), patch.object(sys, "argv", base + [str(self.cache / "r.json")] + bad), \
                 patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                f.main()
        receipt = self.cache / "receipt.json"
        with patch.object(sys, "argv", base + [str(receipt), "--range-streams", "4", "--max-connections", "8"]), \
             patch.object(f, "read_trusted", return_value=manifest()), \
             patch.object(f, "acquire", return_value={"state": "acquisition-complete"}) as acquire, \
             patch("sys.stdout", io.StringIO()):
            self.assertEqual(f.main(), 0)
        self.assertEqual((acquire.call_args.kwargs["range_streams"], acquire.call_args.kwargs["max_connections"]), (4, 8))
        receipt.unlink()
        with patch.object(sys, "argv", base + [str(receipt)]), patch.object(f, "read_trusted", return_value=manifest()), \
             patch.object(f, "acquire", return_value={"state": "acquisition-complete"}) as acquire, \
             patch("sys.stdout", io.StringIO()):
            f.main()
        self.assertEqual((acquire.call_args.kwargs["range_streams"], acquire.call_args.kwargs["max_connections"]), (0, None))


class LoopbackRangeServerTests(unittest.TestCase):
    """Real http.server + urllib over loopback: 206s, a mid-part disconnect with
    resume, a server that ignores Range, and an ETag change between parts."""
    PART = 16 * 1024
    PAYLOAD = b"".join(hashlib.sha256(i.to_bytes(4, "big")).digest() for i in range(3200))  # 100 KiB, 7 parts
    CUT = (2 * 16 * 1024, 5000)  # Part 2 sends 5000 of its bytes once, then closes.

    def setUp(self):
        payload, cut, seen, lock = self.PAYLOAD, self.CUT, [], threading.Lock()
        self.seen = seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                mode, header = self.path.strip("/"), self.headers.get("Range")
                with lock:
                    seen.append((mode, header))
                    count = sum(m == mode for m, _ in seen)
                    cut_now = mode == "disconnect" and header is not None \
                        and header.startswith("bytes=%d-" % cut[0]) and (mode, header) not in seen[:-1]
                etag = '"v2"' if mode == "etag" and count > 1 else '"v1"'
                if header is None or mode == "ignore":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("ETag", etag)
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
                first, last = int(match[1]), int(match[2])
                body = payload[first:last + 1]
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (first, last, len(payload)))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", "Sat, 17 Oct 2026 00:00:00 GMT")
                self.end_headers()
                self.wfile.write(body[:cut[1]] if cut_now else body)
                self.wfile.flush()
                self.close_connection = True

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # Client hang-ups after an unread 200 body are expected.

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for patcher in (patch.object(f, "RANGED_MIN_BYTES", 1), patch.object(f, "RANGE_PART", self.PART),
                        patch.object(f.time, "sleep"),
                        patch.dict(os.environ, {"no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost"})):
            patcher.start()
            self.addCleanup(patcher.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = Path(folder.name)

    def opener(self, mode):
        real = urllib.request.build_opener(f.SecureRedirect())  # Built after no_proxy is set.
        port = self.server.server_port

        class Loopback:
            def open(self, request, timeout=None):
                # Same headers (Range, User-Agent) as the reviewed https request, sent to loopback.
                local = urllib.request.Request("http://127.0.0.1:%d/%s" % (port, mode),
                                               headers=dict(request.header_items()))
                return real.open(local, timeout=5)
        return Loopback()

    def artifact(self):
        return large(self.PAYLOAD, host="download-r2.pytorch.org", name="torch")

    def requests(self, mode):
        return [header for m, header in self.seen if m == mode]

    def test_real_206_ranges(self):
        a = self.artifact()
        receipt = f.acquire_one(a, self.cache, opener=self.opener("ok"), range_streams=4)
        self.assertEqual((self.cache / a["filename"]).read_bytes(), self.PAYLOAD)
        self.assertEqual((receipt["transfer_mode"], receipt["parts"], receipt["range_requests"]), ("ranged", 7, 7))
        self.assertEqual(len(self.requests("ok")), 7)
        self.assertTrue(all(h and h.startswith("bytes=") for h in self.requests("ok")))

    def test_mid_part_disconnect_resumes_from_written_offset(self):
        a = self.artifact()
        receipt = f.acquire_one(a, self.cache, opener=self.opener("disconnect"), range_streams=4)
        first, sent = self.CUT
        resumed = "bytes=%d-%d" % (first + sent, first + self.PART - 1)
        self.assertIn(resumed, self.requests("disconnect"))
        self.assertEqual(len(self.requests("disconnect")), 8)
        self.assertEqual(receipt["part_timings"][2]["resume_offsets"], [first + sent])
        self.assertEqual((receipt["attempts"], receipt["transferred_bytes"]), (2, len(self.PAYLOAD)))
        self.assertEqual((self.cache / a["filename"]).read_bytes(), self.PAYLOAD)

    def test_server_ignoring_range_falls_back_to_single_stream(self):
        a = self.artifact()
        receipt = f.acquire_one(a, self.cache, opener=self.opener("ignore"), range_streams=4)
        self.assertEqual(self.requests("ignore"), ["bytes=0-%d" % (self.PART - 1), None])
        self.assertEqual((receipt["transfer_mode"], receipt["range_fallback"]), ("single", "range-unsupported"))
        self.assertEqual((self.cache / a["filename"]).read_bytes(), self.PAYLOAD)
        self.assertEqual(list(self.cache.glob(".partial-*")), [])

    def test_etag_change_between_parts_refused(self):
        a = self.artifact()
        with self.assertRaisesRegex(ValueError, "^Object identity changed between ranges$"):
            f.acquire_one(a, self.cache, opener=self.opener("etag"), range_streams=4)
        self.assertFalse((self.cache / a["filename"]).exists())
        self.assertEqual(list(self.cache.glob(".partial-*")), [])


class LinuxHarnessTests(unittest.TestCase):
    def test_full_delivery_refuses_macos_before_claim(self):
        with patch.object(delivery.platform,"system",return_value="Darwin"), self.assertRaisesRegex(ValueError,"Linux only"):
            delivery.run(None,None)

    def test_source_control_preserves_three_indexes_and_avatar_order(self):
        commands=bench.source_commands('/venv/python',Path('/repo'),'server')
        cmd=commands[0][1]
        self.assertIn('/repo/requirements/server.in',cmd)
        self.assertIn('https://download.pytorch.org/whl/cu121',cmd)
        self.assertIn('https://pypi.nvidia.com',cmd)
        self.assertNotIn('--no-index',cmd)
        avatar=bench.source_commands('/venv/python',Path('/repo'),'full')
        self.assertEqual([n for n,c in avatar],['source-avatar-prerequisites','source-mmcv','source-chumpy','source-avatar'])
        self.assertIn('--no-build-isolation',avatar[2][1])
        self.assertIn('--no-deps',avatar[1][1])

    def test_source_report_different_hash_fails(self):
        a=d.artifact(row(), 'server')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'report.json';r=row();r['download_info']['archive_info']['hashes']['sha256']='0'*64
            p.write_text(json.dumps({'install':[r]}))
            with self.assertRaisesRegex(ValueError,'different artifact hash'):
                bench.verify_source_reports({'artifacts':[a]},[p],None,None)

    def test_source_report_requires_full_closure_coverage(self):
        a=d.artifact(row(), 'server')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'report.json';p.write_text(json.dumps({'install':[]}))
            with self.assertRaisesRegex(ValueError,'complete selected closure'):
                bench.verify_source_reports({'artifacts':[a]},[p],None,None)

    def test_build_and_benchmark_refuse_macos_before_mutation(self):
        with patch.object(b.platform, "system", return_value="Darwin"), self.assertRaisesRegex(ValueError, "Matching Linux only"):
            b.build(None, None, None, None)
        with patch.object(bench.platform, "system", return_value="Darwin"), self.assertRaisesRegex(ValueError, "Linux only"):
            bench.run(None)

    def test_wheel_metadata_checked_independently_of_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"example-1.0-py3-none-any.whl"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("example-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n")
            self.assertEqual(b.checked_wheel(p, "example", "1.0")["platform"], "any")
            with self.assertRaises(ValueError):
                b.checked_wheel(p, "different", "1.0")

    def test_traversal_in_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"example-1.0-py3-none-any.whl"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("../evil", "test")
            with self.assertRaisesRegex(ValueError, "Unsafe wheel member"):
                b.checked_wheel(p, "example", "1.0")

    def test_pip_and_uv_use_same_offline_lock_and_binary_policy(self):
        for tool in ("pip", "uv"):
            cmd = bench.install_command(tool, "/trial/python", "/tool/uv", "/trial/cache", "/wheels", "/lock", True, "copy")
            self.assertIn("--no-index", cmd)
            self.assertIn("--require-hashes", cmd)
            self.assertIn("--only-binary=:all:", cmd)
            self.assertEqual(cmd[cmd.index("-r") + 1], "/lock")
            self.assertNotIn("sync", cmd)
            if tool == "uv":
                self.assertIn("--compile-bytecode", cmd)
                self.assertIn("--offline", cmd)

    def test_bytecode_comparison_is_explicit(self):
        pip = bench.install_command("pip", "python", None, "cache", "wheels", "lock", False, "copy")
        uv = bench.install_command("uv", "python", "uv", "cache", "wheels", "lock", False, "copy")
        self.assertIn("--no-compile", pip)
        self.assertNotIn("--compile-bytecode", uv)

    def test_gpu_initializing_import_not_hidden_or_omitted(self):
        self.assertNotIn("torch_tensorrt", bench.IMPORTS["server"])
        self.assertIn("torch_tensorrt", bench.GPU_PROBE)
        self.assertIn("value==140.0", bench.GPU_PROBE)
        compile(bench.GPU_PROBE, "<gpu-probe-syntax-only>", "exec")

    def test_non_experiment_path_and_traversal_refused(self):
        for path, name in (("/workspace/MuseTalk", "test"), ("/tmp/test", "trial"),
                           ("/workspace/.dependency-optimization-test", "../bad")):
            with self.assertRaises(ValueError):
                bench.trial_path(path, name)


class RentalControlSafety(unittest.TestCase):
    def test_expiry_fixture_is_one_second_and_manifest_only(self):
        from types import SimpleNamespace
        calls=[]
        def generate(*args,**kwargs):calls.append(kwargs);return 'https://fixture/?X-Amz-Algorithm=AWS4-HMAC-SHA256'
        fake=SimpleNamespace(generate_presigned_url=generate)
        key=publish.PRIVATE_PREFIX+'725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194/manifest.json'
        publish.presign_expiry_fixture(fake,key)
        self.assertEqual(calls[0]['ExpiresIn'],1)
        with self.assertRaises(ValueError):publish.presign_expiry_fixture(fake,key.replace('manifest.json','unrelated.whl'))
        self.assertEqual(len(calls),1)

    def test_followup_dispatch_cannot_fall_through_to_cleanup(self):
        from types import SimpleNamespace
        for mode, launch in (("followup", True), ("followup-status", False)):
            with patch.object(sys, "argv", ["control", mode]), patch.object(control.os, "getuid", return_value=0), \
                 patch.object(control.os, "umask"), patch.object(control, "library", return_value=SimpleNamespace(require=d.require)), \
                 patch.object(control, "screen") as screen:
                control.main()
            self.assertEqual(screen.call_args.kwargs, {"launch": launch, "followup": True})

    def test_range_pilot_dispatch_cannot_fall_through_to_cleanup(self):
        from types import SimpleNamespace
        for mode, launch in (("s3-pilot", True), ("s3-pilot-status", False)):
            with patch.object(sys, "argv", ["control", mode]), patch.object(control.os, "getuid", return_value=0), \
                 patch.object(control.os, "umask"), patch.object(control, "library", return_value=SimpleNamespace(require=d.require)), \
                 patch.object(control, "screen") as screen:
                control.main()
            self.assertEqual(screen.call_args.kwargs, {"launch": launch, "pilot": True})

    def test_legacy_presigning_rejected_before_worker_request(self):
        from types import SimpleNamespace
        fake=SimpleNamespace(generate_presigned_url=lambda *a,**k:'https://example/?AWSAccessKeyId=secret&Signature=secret')
        with self.assertRaisesRegex(ValueError,'legacy auth'):
            publish.presign_get(fake,publish.PRIVATE_PREFIX+'a'*64+'/manifest.json',300)

    def test_preservation_manifest_is_small_source_and_output_pairs(self):
        path=Path(__file__).parent/'docs/dependency_optimization_20261015/wheelhouse-manifest-a2.json'
        rows=publish.paths_for_manifest(json.loads(path.read_text()))
        self.assertEqual(len(rows),8)
        self.assertEqual(sum(r['kind']=='built-wheel' for r in rows),4)
        self.assertTrue(all('/' not in r['name'] for r in rows))

    def test_matrix_dispatch_cannot_fall_through_to_cleanup(self):
        from types import SimpleNamespace
        for mode, launch in (("matrix", True), ("matrix-status", False)):
            with patch.object(sys, "argv", ["control", mode]), patch.object(control.os, "getuid", return_value=0), \
                 patch.object(control.os, "umask"), patch.object(control, "library", return_value=SimpleNamespace(require=d.require)), \
                 patch.object(control, "screen") as screen:
                control.main()
            screen.assert_called_once()
            self.assertEqual(screen.call_args.kwargs, {"launch": launch, "matrix": True})

    def test_disposable_matrix_refuses_macos(self):
        with patch.object(matrix.platform, "system", return_value="Darwin"), self.assertRaisesRegex(ValueError, "Matching Linux only"):
            matrix.execute("2099-01-01T00:00:00+00:00")

    def test_quote_bound_to_reviewed_hardware_and_all_caps(self):
        source = control.library()
        row = {"machine_id": control.MACHINE, "host_id": control.HOST, "geolocation": control.LOCATION,
               "gpu_name": "RTX 3090", "num_gpus": 1, "cpu_arch": "amd64", "verification": "unverified",
               "cpu_cores_effective": 8, "cpu_ram": 32222, "disk_space": 328, "inet_down": 527.7,
               "cuda_max_good": 13.2, "reliability": .99, "inet_down_cost": .000004, "inet_up_cost": .000004,
               "storage_cost": .2, "dph_total": .1878, "dph_total_adj": .1879}
        self.assertTrue(control.approved(row, source))
        for key, value in (("host_id", 0), ("machine_id", 0), ("geolocation", "Texas, US"),
                           ("dph_total_adj", .301), ("inet_up_cost", .002), ("inet_down_cost", .002),
                           ("dph_total", float("nan")), ("disk_space", 50)):
            with self.subTest(key=key):
                self.assertFalse(control.approved(dict(row, **{key: value}), source))

    def test_existing_create_namespace_never_repeated(self):
        source = control.library()
        with tempfile.TemporaryDirectory() as folder, patch.object(control, "ROOT", Path(folder)), \
             patch.object(control.os, "getuid", return_value=0), patch.object(source, "bootstrap") as provider:
            with self.assertRaisesRegex(ValueError, "never repeat create"):
                control.create(source)
            provider.assert_not_called()

    def test_launch_patch_changes_timing_and_cache_not_pins(self):
        source = control.library()
        marker = 'echo "[source-baseline] checkout_ready $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"'
        with patch.object(source, "frozen_template", return_value=({}, "#!/bin/bash\n" + marker + "\n", {"changes": []})):
            env, script, receipt = control.freeze(source, {})
        self.assertTrue(env["PIP_CACHE_DIR"].startswith(control.WORKER_ROOT + "/"))
        self.assertNotIn("pip install", script)
        self.assertEqual(receipt["lifetime_seconds"], 7200)
        self.assertLess(len(script.encode()), 12000)


class S3RangeProbeTests(unittest.TestCase):
    def test_full_restore_refuses_mac_before_reading_configuration(self):
        with patch.object(full_s3.platform,'system',return_value='Darwin'),self.assertRaisesRegex(ValueError,'Linux'):
            full_s3.execute(None,None,None)

    def test_full_transfer_requires_new_path_and_bounded_concurrency(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'already';path.write_text('fixture')
            for workers in (0,2,8,4):
                with self.subTest(workers=workers),self.assertRaises(ValueError):
                    full_s3.download(None,path,workers,0)

    def config(self):
        return {"key":range_probe.KEY,"bytes":range_probe.SIZE,"expected_sha256":range_probe.SHA,
                "version_id":"fixture-version","etag":'"fixture-etag"',
                "url":"https://"+f.BUCKET+".s3.us-east-1.amazonaws.com/"+range_probe.KEY
                    +"?versionId=fixture-version&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=not-real"}

    def response(self,data=b"12345"):
        response=FakeResponse(data)
        response.status=206
        response.headers.update({"Content-Range":f"bytes 0-4/{range_probe.SIZE}",
            "ETag":'"fixture-etag"',"x-amz-version-id":"fixture-version"})
        return response

    def test_versioned_sigv4_contract_required(self):
        config=self.config();range_probe.validate_config(config)
        for update in ({"bytes":1},{"version_id":"changed"},{"expected_sha256":"0"*64},
                       {"url":config["url"].replace(".s3.us-east-1.amazonaws.com",".evil.example")},
                       {"url":config["url"].replace("AWS4-HMAC-SHA256","legacy")}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                range_probe.validate_config(dict(config,**update))

    def test_range_identity_and_bytes_are_checked(self):
        data,receipt=range_probe.read_range(self.config(),0,5,opener=lambda *a,**k:self.response())
        self.assertEqual(data,b"12345");self.assertEqual(receipt["bytes"],5)
        for field,value in (("Content-Range","bytes 1-5/9"),("ETag",'"other"'),
                            ("x-amz-version-id","other"),("Content-Length","6")):
            response=self.response();response.headers[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                range_probe.read_range(self.config(),0,5,opener=lambda *a,**k:response)

    def test_truncated_range_and_wrong_status_rejected(self):
        for response in (self.response(b"1234"),self.response()):
            response.headers["Content-Length"]="5"
            if response.getbuffer().nbytes==5:response.status=200
            with self.assertRaises(ValueError):
                range_probe.read_range(self.config(),0,5,opener=lambda *a,**k:response)

    def test_invalid_range_fails_before_network(self):
        for offset,count in ((-1,5),(0,0),(range_probe.SIZE,5),(0,range_probe.SAMPLE+1)):
            with self.subTest(offset=offset,count=count),self.assertRaises(ValueError):
                range_probe.read_range(self.config(),offset,count,opener=lambda *a,**k:self.fail("network called"))

    def test_pilot_refuses_mac_before_any_mutation(self):
        with patch.object(range_probe.platform,"system",return_value="Darwin"),self.assertRaisesRegex(ValueError,"Linux"):
            range_probe.after_delivery(self.config(),"2099-01-01T00:00:00+00:00")


class LockedInstallerTests(unittest.TestCase):
    def locked_fixture(self,root):
        from types import SimpleNamespace
        m=manifest();repo=root/'repo';repo.mkdir();venv=root/'venvs/example';venv.mkdir(parents=True)
        cache=root/'cache';artifact=cache/m['artifact_id'];artifact.mkdir(parents=True)
        p=artifact/'manifest.json';p.write_text(json.dumps(m));config=self.contract()
        config.update(artifact_id=m['artifact_id'],prefix=f.PRIVATE_PREFIX+m['artifact_id']+'/',
                      manifest_sha256=f.sha256_file(p))
        contract=root/'contract.json';contract.write_text(json.dumps(config))
        (venv/locked.MARKER).write_text(json.dumps({'state':'accepted','uid':locked.os.getuid(),
            'venv':str(venv),'artifact_id':m['artifact_id'],'group':'server'}))
        args=SimpleNamespace(dependency_contract=str(contract),dependency_cache=str(cache))
        groups={'kokoro':(False,'flag'),'avatar_prep':(False,'flag'),'legacy_int8':(False,'flag')}
        dists={a['name']:{'version':a['version']} for a in m['artifacts']}
        return args,repo,venv,groups,dists

    def test_boot_contract_check_verifies_full_inventory_without_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();args,repo,venv,groups,dists=self.locked_fixture(root)
            before={str(p.relative_to(root)):f.sha256_file(p) for p in root.rglob('*') if p.is_file()}
            report=install_state.Report();install_state.check_locked_contract(args,repo,venv,groups,dists,report)
            self.assertEqual(report.exit_code,0)
            self.assertEqual(report.checks[-1]['check'],'locked_contract')
            after={str(p.relative_to(root)):f.sha256_file(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            dists['numpy']['version']='2.0.0'
            report=install_state.Report();install_state.check_locked_contract(args,repo,venv,groups,dists,report)
            self.assertEqual(report.exit_code,11)
            self.assertIn('Full locked package inventory differs',report.repair[-1])

    def test_boot_contract_missing_or_wrong_stamp_fails_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();args,repo,venv,groups,dists=self.locked_fixture(root)
            marker=venv/locked.MARKER;value=json.loads(marker.read_text());value['state']='failed';marker.write_text(json.dumps(value))
            report=install_state.Report();install_state.check_locked_contract(args,repo,venv,groups,dists,report)
            self.assertEqual(report.exit_code,11)
            args.dependency_cache=str(root/'nonexistent')
            report=install_state.Report();install_state.check_locked_contract(args,repo,venv,groups,dists,report)
            self.assertEqual(report.exit_code,11);self.assertFalse(Path(args.dependency_cache).exists())

    def test_source_mode_has_no_locked_check_dependency(self):
        from types import SimpleNamespace
        report=install_state.Report()
        install_state.check_locked_contract(SimpleNamespace(),None,None,None,None,report)
        self.assertEqual(report.checks,[])

    def test_summary_uses_only_complete_delivery_receipts_and_medians(self):
        def receipt(name,method,value):
            return {'trial':name,'state':'delivery-trial-passed','method':method,'workers':4,
                    'packages_ready_seconds':value-1,'validated_ready_seconds':value,'acquisition':{'seconds':2}}
        data={'prepared_utc':'fixture','trials_and_builds':{
            'delivery-source-1':receipt('source-1','source-pip',10),
            'delivery-source-2':receipt('source-2','source-pip',12),
            'delivery-uv-1':receipt('uv-1','wheelhouse-uv',5),
            'inner-delivery-uv-1':receipt('excluded-local','wheelhouse-uv',1),
            'delivery-failed':{'state':'failed'}}}
        result=summary.summarize(data)
        self.assertEqual(len(result['trials']),3)
        source=next(r for r in result['summary'] if r['method']=='source-pip')
        self.assertEqual(source['validated_ready_seconds']['median'],11)
        self.assertFalse(result['goal_complete'])

    def test_application_launch_payload_is_syntax_valid_and_isolated(self):
        from types import SimpleNamespace
        with patch.object(application.c,'worker',return_value=b'{"pid":1}') as worker:
            application.launch(SimpleNamespace(),'fixture', '/workspace/.dependency-optimization-fixture/venv')
        payload=worker.call_args.args[1]
        compile(payload,'<fixture-launch>','exec')
        self.assertIn("'VENV_PATH':",payload)
        self.assertIn("'127.0.0.1'",payload)
        self.assertIn("'MUSETALK_TAESD_TRT_BUILD':'0'",payload)
        self.assertNotIn('dependency fixture',payload)

    def test_cold_numba_cases_have_distinct_empty_caches(self):
        from types import SimpleNamespace
        with patch.object(application,'ATTEMPT','a3'),patch.object(application,'ISOLATE_NUMBA',True), \
             patch.object(application.c,'worker',return_value=b'{"pid":1}') as worker:
            application.launch(SimpleNamespace(),'fixture','/workspace/.dependency-optimization-fixture/venv')
        payload=worker.call_args.args[1];compile(payload,'<cache-fixture>','exec')
        self.assertIn('/application-a3/fixture',payload)
        self.assertIn("'NUMBA_CACHE_DIR':str(cache)",payload)
        self.assertIn('assert not list(cache.iterdir())',payload)

    def test_range_settings_from_flags_environment_and_defaults(self):
        from types import SimpleNamespace
        def args(**changes):
            return SimpleNamespace(**dict({'workers':4,'range_streams':None,'max_connections':None},**changes))
        self.assertEqual(locked.transfer_settings(args(),{}),(0,4))
        self.assertEqual(locked.transfer_settings(args(workers=2),{}),(0,2))
        env={locked.RANGE_STREAMS_ENV:'4',locked.MAX_CONNECTIONS_ENV:'8'}
        self.assertEqual(locked.transfer_settings(args(),env),(4,8))
        self.assertEqual(locked.transfer_settings(args(range_streams=2,max_connections=4),env),(2,4))
        self.assertEqual(locked.transfer_settings(args(),{locked.RANGE_STREAMS_ENV:' '}),(0,4))
        for bad in ({locked.RANGE_STREAMS_ENV:'3'},{locked.RANGE_STREAMS_ENV:'on'},{locked.RANGE_STREAMS_ENV:'-2'},
                    {locked.MAX_CONNECTIONS_ENV:'2'},{locked.MAX_CONNECTIONS_ENV:'16'}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                locked.transfer_settings(args(),bad)

    def test_canonical_installer_passes_range_settings_through(self):
        text=(Path(__file__).parent/'scripts/install_musetalk.sh').read_text()
        body=re.search(r'(?ms)^phase_packages\(\) \{\n.*?^\}',text).group(0)
        stub=('set -Eeuo pipefail\nflag_on() { [[ "$1" == 1 ]]; }\nlog() { :; }\n'
              'run_strict_step() { shift; printf "|%s" "$@"; }\n'
              'DEPENDENCY_INSTALL_MODE=locked-wheelhouse-uv KOKORO=0 AVATAR_PREP=0 PYTHON_BIN=py SCRIPT_DIR=/s\n'
              'DEPENDENCY_CONTRACT=/c REPO_ROOT=/r VENV_PATH=/v DEPENDENCY_CACHE=/d\n')
        clean={k:v for k,v in os.environ.items() if not k.startswith('MUSETALK_')}
        for env,expected,absent in (({},'|--range-streams|0','--max-connections'),
                                    ({'MUSETALK_DEPENDENCY_RANGE_STREAMS':'4','MUSETALK_DEPENDENCY_MAX_CONNECTIONS':'8'},
                                     '|--range-streams|4|--max-connections|8',None)):
            with self.subTest(env=env):
                result=subprocess.run(['/bin/bash','-c',stub+body+'\nphase_packages\n'],env=dict(clean,**env),
                                      capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertTrue(result.stdout.endswith(expected),result.stdout)
                self.assertIn('|--workers|4|--tickets-fd|0|',result.stdout)
                if absent:self.assertNotIn(absent,result.stdout)

    def contract(self):
        return json.loads((Path(__file__).parent/'configs/dependencies/rtx3090-cp310-cu121-screening.json').read_text())

    def test_reviewed_contract_and_pinned_uv_required(self):
        config=self.contract();locked.check_contract(config)
        for mutation in ({'owner':'other'},{'artifact_id':'../other'},{'prefix':'unrelated/'},
                         {'manifest_sha256':''},{'uv_binary_sha256':'0'*64}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                locked.check_contract(dict(config,**mutation))

    def test_no_mac_install_or_ticket_read(self):
        with patch.object(locked.platform,'system',return_value='Darwin'),self.assertRaisesRegex(ValueError,'Linux'):
            locked.run(None,lambda:self.fail('tickets read on Mac'))
        with patch.object(followup.platform,'system',return_value='Darwin'),self.assertRaisesRegex(ValueError,'Linux'):
            followup.execute(None,'2099-01-01T00:00:00+00:00')
        with patch.object(canonical_integration.platform,'system',return_value='Darwin'),self.assertRaisesRegex(ValueError,'Linux'):
            canonical_integration.execute(None,'2099-01-01T00:00:00+00:00')
        with patch.object(confirmation.platform,'system',return_value='Darwin'),self.assertRaisesRegex(ValueError,'Linux'):
            confirmation.execute(None,'2099-01-01T00:00:00+00:00')
        with patch.object(warm_start.platform,'system',return_value='Darwin'),self.assertRaisesRegex(ValueError,'Linux'):
            warm_start.execute('2099-01-01T00:00:00+00:00')

    def test_wrong_python_and_glibc_fail_before_io(self):
        with patch.object(locked.platform,'system',return_value='Linux'), \
             patch.object(locked.platform,'machine',return_value='x86_64'), \
             patch.object(locked.sys,'version_info',(3,11,0)),self.assertRaisesRegex(ValueError,'3.10'):
            locked.check_platform()
        with patch.object(locked.platform,'system',return_value='Linux'), \
             patch.object(locked.platform,'machine',return_value='x86_64'), \
             patch.object(locked.sys,'version_info',(3,10,12)), \
             patch.object(locked.platform,'libc_ver',return_value=('glibc','2.31')),self.assertRaisesRegex(ValueError,'glibc'):
            locked.check_platform()

    def test_changed_requirements_fails_and_unrelated_code_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'requirements').mkdir()
            p=root/'requirements/server.in';p.write_bytes(b'fixed')
            manifest={'requirement_inputs':{'server.in':hashlib.sha256(b'fixed').hexdigest()}}
            locked.check_inputs(manifest,root)
            (root/'api_server.py').write_text('unrelated code update')
            locked.check_inputs(manifest,root)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Dependency input changed'):locked.check_inputs(manifest,root)

    def test_exact_inventory_rejects_missing_extra_and_wrong_build(self):
        m=manifest();actual={a['name']:a['version'] for a in m['artifacts']}
        self.assertEqual(locked.inventory_diff(m,'full',actual),{})
        for change in ({'torch':'2.5.1+cpu'},{'extra':'1.0'},{'numpy':None}):
            self.assertTrue(locked.inventory_diff(m,'full',dict(actual,**change)))

    def test_venv_inside_checkout_or_broad_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';repo.mkdir()
            for venv in (Path('/'),Path('/workspace'),repo/'venv'):
                with self.subTest(venv=venv),self.assertRaises(ValueError):locked.safe_venv(venv,repo)

    def test_canonical_default_remains_source_pip_and_locked_bootstrap_is_not_duplicated(self):
        text=(Path(__file__).parent/'scripts/install_musetalk.sh').read_text()
        self.assertIn('DEPENDENCY_INSTALL_MODE="${MUSETALK_DEPENDENCY_INSTALL_MODE:-source-pip}"',text)
        self.assertIn('Locked dependency phase owns venv/bootstrap timing',text)
        self.assertIn('install_locked_dependencies.py',text)


if __name__ == "__main__":
    unittest.main()
