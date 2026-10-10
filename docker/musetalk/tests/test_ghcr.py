"""Private registry contracts using synthetic data only; no credentials/network."""
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghcr


class GHCRTests(unittest.TestCase):
    def test_digest_is_actual_and_unambiguous(self):
        digest = "sha256:" + "a" * 64
        self.assertEqual(ghcr.pushed_digest("tag: digest: " + digest + " size: 123"), digest)
        for value in ("latest", "digest: sha256:123", "digest: " + digest + " digest: " + digest):
            with self.assertRaises(ValueError):
                ghcr.pushed_digest(value)

    def test_private_visibility_is_required(self):
        for visibility in ("public", "internal", None):
            with patch.object(ghcr, "command", return_value=json.dumps({"name": "musetalk-rtx3090", "visibility": visibility})):
                with self.assertRaises(ValueError):
                    ghcr.private_package()
        with patch.object(ghcr, "command", return_value=json.dumps({"name": "musetalk-rtx3090", "visibility": "private"})):
            self.assertEqual(ghcr.private_package()["visibility"], "private")

    def test_anonymous_denial_needs_authorization_failure_not_outage(self):
        for code in (401, 403, 404, 429, 500):
            def opener(*args, **kwargs):
                raise urllib.error.HTTPError("https://fixture.invalid", code, "synthetic", {}, None)
            if code in (401, 403):
                self.assertTrue(ghcr.anonymous_denied("sha256:" + "a" * 64, opener))
            else:
                with self.assertRaises(ValueError):
                    ghcr.anonymous_denied("sha256:" + "a" * 64, opener)

    def test_anonymous_manifest_success_is_rejected(self):
        with self.assertRaises(ValueError):
            ghcr.anonymous_denied("sha256:" + "a" * 64,
                                 lambda *a, **k: io.BytesIO(b'{"token":"synthetic-anonymous-token"}'))

    def test_login_passes_secret_only_via_stdin(self):
        with patch.dict(ghcr.os.environ, {"GH_TOKEN": "synthetic-not-a-real-secret", "GITHUB_ACTOR": "AhmadAFS1",
                                         "GITHUB_REPOSITORY": ghcr.REPOSITORY}), patch.object(ghcr, "command") as run:
            ghcr.login()
            argv = run.call_args.args[0]
            self.assertNotIn("synthetic-not-a-real-secret", str(argv))
            self.assertEqual(run.call_args.kwargs["payload"], "synthetic-not-a-real-secret\n")

    def layer(self, name, payload):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
        stream.seek(0)
        return stream

    def test_every_layer_rejects_developer_credentials_and_secret_content(self):
        for name, content in (("root/.aws/credentials", b"synthetic"), ("opt/musetalk/app/.env", b"synthetic"),
                              ("root/.config/gh/hosts.yml", b"synthetic"),
                              ("opt/file.txt", b"-----BEGIN OPENSSH PRIVATE KEY-----"), ("../escape", b"synthetic")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ghcr.scan_layer(self.layer(name, content), "synthetic")
        self.assertEqual(ghcr.scan_layer(self.layer("opt/file.txt", b"synthetic"), "synthetic")["regular_files_scanned"], 1)

    def test_chunk_boundary_secret_is_not_missed(self):
        payload = b" " * (1024 * 1024 - 10) + b"-----BEGIN OPENSSH PRIVATE KEY-----"
        with self.assertRaises(ValueError):
            ghcr.scan_layer(self.layer("opt/file.txt", payload), "synthetic")

    def test_only_diagnostic_tags_are_supported(self):
        for tag in ("latest", "validated-" + "a" * 40, "candidate-" + "a" * 40, "--bad"):
            with patch.object(ghcr, "command") as run, self.assertRaises(ValueError):
                ghcr.push("local", tag)
            run.assert_not_called()

    def test_foreign_or_floating_independent_pull_is_rejected(self):
        for reference in (ghcr.IMAGE + ":latest", "ghcr.io/other/image@sha256:" + "a" * 64):
            with patch.object(ghcr, "command") as run, self.assertRaises(ValueError):
                ghcr.verify(reference)
            run.assert_not_called()
