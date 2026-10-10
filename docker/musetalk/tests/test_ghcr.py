"""Private registry contracts using synthetic data only; no credentials/network."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghcr


class GHCRTests(unittest.TestCase):
    def test_failure_receipt_never_reflects_exception_text_or_matching_content(self):
        secret = "ghp_" + "s" * 40
        record = ghcr.failure_record(ValueError("https://fixture.invalid?token=" + secret))
        self.assertNotIn(secret, json.dumps(record))
        self.assertNotIn("fixture.invalid", json.dumps(record))
        finding = ghcr.AuditFinding("github-token", path="opt/" + secret, layer="synthetic",
                                   file_sha256="a" * 64, file_size=123)
        record = ghcr.failure_record(finding)
        self.assertNotIn(secret, json.dumps(record))
        self.assertNotIn("path", record["finding"])
        self.assertIn("path_sha256", record["finding"])
        self.assertEqual(record["finding"]["rule"], "github-token")

    def test_failure_receipt_is_persistent_and_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            work = Path(directory) / "new"
            ghcr.report_failure(ValueError("synthetic secret body"), work)
            before = (work / "failure.json").read_text()
            self.assertEqual(json.loads(before)["exception_type"], "ValueError")
            self.assertNotIn("synthetic secret body", before)
            ghcr.report_failure(TypeError("different secret"), work)
            self.assertEqual((work / "failure.json").read_text(), before)

    def test_command_failure_exposes_only_fixed_operation_and_status(self):
        result = ghcr.subprocess.CompletedProcess([], 1, "synthetic-secret-output", "synthetic-secret-body (HTTP 403)")
        with patch.object(ghcr.subprocess, "run", return_value=result):
            with self.assertRaises(ghcr.RegistryOperationError) as caught:
                ghcr.command(["gh", "api", "synthetic-secret-url"], payload="synthetic-secret-input")
        self.assertEqual(str(caught.exception), "package API failed (exit 1 HTTP 403; output suppressed)")

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
                              ("opt/file.txt", b"-----BEGIN OPENSSH PRIVATE KEY-----\n" + b"M" * 64), ("../escape", b"synthetic")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ghcr.scan_layer(self.layer(name, content), "synthetic")
        self.assertEqual(ghcr.scan_layer(self.layer("opt/file.txt", b"synthetic"), "synthetic")["regular_files_scanned"], 1)

    def test_chunk_boundary_secret_is_not_missed(self):
        payload = b" " * (1024 * 1024 - 10) + b"-----BEGIN OPENSSH PRIVATE KEY-----\n" + b"M" * 64
        with self.assertRaises(ValueError):
            ghcr.scan_layer(self.layer("opt/file.txt", payload), "synthetic")

    def test_scanner_failure_identifies_rule_path_and_complete_file_hash_only(self):
        data = b"prefix\n-----BEGIN OPENSSH PRIVATE KEY-----\n" + b"M" * 64 + b"\nsuffix"
        with self.assertRaises(ghcr.AuditFinding) as caught:
            ghcr.scan_layer(self.layer("usr/lib/synthetic.txt", data), "synthetic")
        detail = caught.exception.detail
        self.assertEqual(detail["path"], "usr/lib/synthetic.txt")
        self.assertEqual(detail["rule"], "private-key-material")
        self.assertEqual(detail["file_sha256"], ghcr.hashlib.sha256(data).hexdigest())
        self.assertNotIn("synthetic body", json.dumps(detail))

    def test_crypto_header_literals_are_not_keys_but_binary_embedded_keys_are_rejected(self):
        header = b"-----BEGIN PRIVATE KEY-----"
        result = ghcr.scan_layer(self.layer("usr/lib/synthetic-crypto.so", b"binary\0" + header + b"\0parser"), "synthetic")
        self.assertEqual(result["pem_marker_literal_files_without_key_material"], 1)
        with self.assertRaises(ghcr.AuditFinding):
            ghcr.scan_layer(self.layer("usr/lib/synthetic-crypto.so", header + b"\n" + b"M" * 64), "synthetic")

    def test_raw_json_escaped_encrypted_and_dsa_key_material_are_rejected(self):
        for name in (b"PRIVATE KEY", b"RSA PRIVATE KEY", b"EC PRIVATE KEY", b"DSA PRIVATE KEY",
                     b"OPENSSH PRIVATE KEY", b"ENCRYPTED PRIVATE KEY"):
            key = b"-----BEGIN " + name + b"-----\n" + b"M" * 64 + b"\n-----END " + name + b"-----"
            for payload in (key, json.dumps({"synthetic": key.decode()}).encode()):
                with self.subTest(name=name, escaped=payload != key), self.assertRaises(ghcr.AuditFinding):
                    ghcr.scan_layer(self.layer("opt/synthetic.txt", payload), "synthetic")
        legacy = b"-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-256-CBC,SYNTHETIC\n\n" + b"M" * 64
        with self.assertRaises(ghcr.AuditFinding):
            ghcr.scan_layer(self.layer("opt/synthetic.txt", b" " * (1024 * 1024 - 200) + legacy), "synthetic")

    def test_layer_root_directory_marker_is_safe_but_absolute_path_is_not(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            member = tarfile.TarInfo("./")
            member.type = tarfile.DIRTYPE
            tar.addfile(member)
        stream.seek(0)
        self.assertEqual(ghcr.scan_layer(stream, "synthetic")["regular_files_scanned"], 0)
        with self.assertRaises(ValueError):
            ghcr.scan_layer(self.layer("/escape", b"synthetic"), "synthetic")

    def test_only_explicit_nonpromotable_tags_are_supported(self):
        for tag in ("latest", "validated-" + "a" * 40, "candidate-short", "--bad"):
            with patch.object(ghcr, "command") as run, self.assertRaises(ValueError):
                ghcr.push("local", tag)
            run.assert_not_called()

    def test_candidate_publication_requires_matching_manifest_build_and_baked_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "release.json"
            manifest = {"source_revision": "a" * 40, "status": "candidate", "promotion_eligible": False,
                        "cuda_base": ghcr.release.CUDA_DEVEL_BASE}
            manifest_path.write_text(json.dumps(manifest))
            receipt = {"schema": "musetalk_docker_ci_build_v1", "image": "local-candidate",
                       "source_revision": "a" * 40, "channel": "candidate", "cpu_build_check": "PASS",
                       "promotion_eligible": False}
            (root / "build-result.json").write_text(json.dumps(receipt))
            inspect = {"Id": "sha256:" + "b" * 64, "Config": {"Labels": {
                "org.opencontainers.image.revision": "a" * 40, "io.musetalk.release-channel": "candidate",
                "io.musetalk.cuda-runtime-base": ghcr.release.CUDA_DEVEL_BASE}}}
            remote = {"config": {"digest": inspect["Id"]}, "layers": [{"size": 123}]}
            with patch.object(ghcr.release, "load_manifest", return_value=manifest), \
                 patch.object(ghcr, "private_package", return_value={"visibility": "private"}), \
                 patch.object(ghcr, "anonymous_denied", return_value=True), patch.object(ghcr, "audit", return_value={}), \
                 patch.object(ghcr, "push", return_value=(ghcr.IMAGE + "@sha256:" + "c" * 64, "sha256:" + "c" * 64)) as push, \
                 patch.object(ghcr, "command", side_effect=[json.dumps([inspect]), json.dumps(manifest), json.dumps(remote)]):
                result = ghcr.publish_candidate("local-candidate", root / "audit", "a" * 40, manifest_path, root)
            self.assertTrue(result["serving_image"])
            self.assertFalse(result["production_ready"])
            self.assertFalse(result["promotion_eligible"])
            self.assertEqual(result["compressed_layer_bytes"], 123)
            push.assert_called_once_with("local-candidate", "candidate-" + "a" * 40)
            for key, value in (("cpu_build_check", "FAIL"), ("source_revision", "b" * 40)):
                bad = {**receipt, key: value}
                (root / "build-result.json").write_text(json.dumps(bad))
                with patch.object(ghcr.release, "load_manifest", return_value=manifest), \
                     patch.object(ghcr, "private_package") as privacy, self.assertRaises(ValueError):
                    ghcr.publish_candidate("local-candidate", root / "audit", "a" * 40, manifest_path, root)
                privacy.assert_not_called()

    def test_candidate_cannot_promote_or_publish_a_validated_manifest(self):
        for manifest in ({"source_revision": "a" * 40, "status": "validated", "promotion_eligible": False},
                         {"source_revision": "a" * 40, "status": "candidate", "promotion_eligible": True}):
            with patch.object(ghcr.release, "load_manifest", return_value=manifest), \
                 patch.object(ghcr, "private_package") as privacy, self.assertRaises(ValueError):
                ghcr.publish_candidate("local", Path("synthetic"), "a" * 40, Path("synthetic"), Path("synthetic"))
            privacy.assert_not_called()

    def test_foreign_or_floating_independent_pull_is_rejected(self):
        for reference in (ghcr.IMAGE + ":latest", "ghcr.io/other/image@sha256:" + "a" * 64):
            with patch.object(ghcr, "command") as run, self.assertRaises(ValueError):
                ghcr.verify(reference)
            run.assert_not_called()
