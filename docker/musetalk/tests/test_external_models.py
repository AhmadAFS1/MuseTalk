"""Private model transport tests use only synthetic bytes and mocked S3 clients."""
import copy
import ast
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release


class ExternalModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "app"
        self.root.mkdir()
        self.cache = self.base / "cache"
        self.name = "models/face_detection/s3fd.pth"
        self.data = b"SYNTHETIC private model fixture, not a usable model"
        digest = hashlib.sha256(self.data).hexdigest()
        self.entry = {"sha256": digest, "size_bytes": len(self.data), "public_redistribution": False,
                      "private_delivery_authorized": True,
                      "source": {"type": "s3", "bucket": "fixture-private-bucket", "region": "us-east-1",
                                 "expected_owner": "123456789012", "key": "private-models/sha256/" + digest + "/s3fd.pth"}}
        self.manifest = {"model_files": {}, "external_model_files": {self.name: self.entry},
                         "redistribution_scope": "baked_model_files_only", "private_model_usage_rights": "unresolved"}

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self, _entry, path):
        path.write_bytes(self.data)

    def test_only_five_explicit_private_paths_and_no_public_overlap(self):
        self.assertEqual(release.external_models(self.manifest), {self.name: self.entry})
        for name in ("models/private/unet.pth", "../secret", "api_server.py"):
            bad = {**self.manifest, "external_model_files": {name: self.entry}}
            with self.subTest(name=name), self.assertRaises(ValueError):
                release.external_models(bad)
        with self.assertRaises(ValueError):
            release.external_models({**self.manifest, "model_files": {self.name: self.entry}})

    def test_source_must_be_private_pinned_s3_not_arbitrary_url(self):
        for field, value in (("key", "floating/latest.pth"), ("bucket", "https://example.invalid"),
                             ("expected_owner", ""), ("region", "arbitrary/endpoint"), ("url", "https://example.invalid")):
            m = copy.deepcopy(self.manifest)
            m["external_model_files"][self.name]["source"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                release.external_models(m)
        for field, value in (("private_delivery_authorized", False), ("public_redistribution", True),
                             ("sha256", "floating"), ("size_bytes", 0)):
            m = copy.deepcopy(self.manifest)
            m["external_model_files"][self.name][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                release.external_models(m)
        with self.assertRaises(ValueError):
            release.external_models({**self.manifest, "private_model_usage_rights": "private fetch clears rights"})

    def test_verified_fetch_cache_and_target_are_separate_from_baked_models(self):
        with patch("builtins.print"):
            report = release.runtime_models(self.root, self.manifest, self.cache, self.fetch)
            self.assertEqual(report["downloaded_bytes"], len(self.data))
            self.assertEqual((self.root / self.name).read_bytes(), self.data)
            self.assertEqual((self.cache / self.entry["sha256"]).read_bytes(), self.data)
            with patch.object(self, "fetch", side_effect=AssertionError("must not fetch")):
                report = release.runtime_models(self.root, self.manifest, self.cache, self.fetch)
                self.assertEqual(report["already_present"], 1)
            second_root = self.base / "second-app"
            second_root.mkdir()
            report = release.runtime_models(second_root, self.manifest, self.cache,
                                            lambda *_: self.fail("cache must prevent fetch"))
            self.assertEqual(report["cache_hits"], 1)
            self.assertEqual(report["downloaded_bytes"], 0)

    def test_corrupt_fetch_never_publishes_cache_or_target(self):
        with self.assertRaises(ValueError):
            release.runtime_models(self.root, self.manifest, self.cache, lambda _e, path: path.write_bytes(b"bad"))
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertFalse((self.root / self.name).exists())

    def test_corrupt_cache_or_existing_target_fail_without_overwrite(self):
        self.cache.mkdir()
        cached = self.cache / self.entry["sha256"]
        cached.write_bytes(b"corrupt cache")
        with self.assertRaises(ValueError):
            release.runtime_models(self.root, self.manifest, self.cache, lambda *_: self.fail("must not fetch"))
        self.assertEqual(cached.read_bytes(), b"corrupt cache")
        target = self.root / self.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user-owned wrong target")
        with self.assertRaises(ValueError):
            release.runtime_models(self.root, self.manifest, self.cache, lambda *_: self.fail("must not fetch"))
        self.assertEqual(target.read_bytes(), b"user-owned wrong target")

    def test_target_or_cache_symlink_rejected(self):
        target = self.root / self.name
        target.parent.mkdir(parents=True)
        target.symlink_to(self.base / "outside")
        with self.assertRaises(ValueError):
            release.runtime_models(self.root, self.manifest, self.cache, self.fetch)

    def test_s3_sdk_uses_injected_credentials_owner_and_no_configured_endpoint(self):
        client = Mock()
        client.head_object.return_value = {"ContentLength": len(self.data), "Metadata": {"sha256": self.entry["sha256"]},
                                            "VersionId": "fixture-version"}
        client.download_file.side_effect = lambda _b, _k, path, **_kw: Path(path).write_bytes(self.data)
        boto = SimpleNamespace(client=Mock(return_value=client))
        config = SimpleNamespace(Config=Mock(return_value="config"))
        with patch.dict(sys.modules, {"boto3": boto, "botocore.config": config,
                                      "boto3.s3.transfer": SimpleNamespace(TransferConfig=Mock())}), \
             patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-id", "AWS_SECRET_ACCESS_KEY": "test-secret",
                                     "AWS_ENDPOINT_URL": "https://untrusted.invalid"}, clear=True):
            release.fetch_private_model(self.entry, self.base / "download")
        kwargs = boto.client.call_args.kwargs
        self.assertEqual(kwargs["aws_access_key_id"], "test-id")
        self.assertEqual(kwargs["region_name"], "us-east-1")
        self.assertTrue(config.Config.call_args.kwargs["ignore_configured_endpoint_urls"])
        self.assertEqual(config.Config.call_args.kwargs["signature_version"], "s3v4")
        self.assertEqual(client.head_object.call_args.kwargs["ExpectedBucketOwner"], "123456789012")
        self.assertEqual(client.download_file.call_args.kwargs["ExtraArgs"]["ExpectedBucketOwner"], "123456789012")
        self.assertEqual((self.base / "download").read_bytes(), self.data)
        client.put_object.assert_not_called()

    def test_no_anonymous_or_ambient_role_credentials(self):
        boto = SimpleNamespace(client=Mock())
        with patch.dict(sys.modules, {"boto3": boto, "botocore.config": SimpleNamespace(Config=Mock()),
                                      "boto3.s3.transfer": SimpleNamespace(TransferConfig=Mock())}), \
             patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            release.fetch_private_model(self.entry, self.base / "download")
        boto.client.assert_not_called()

    def test_sdk_failure_suppresses_request_details(self):
        client = Mock()
        client.head_object.side_effect = RuntimeError("test-secret signed URL contents")
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(client=Mock(return_value=client)),
                                      "botocore.config": SimpleNamespace(Config=Mock()),
                                      "boto3.s3.transfer": SimpleNamespace(TransferConfig=Mock())}), \
             patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-id", "AWS_SECRET_ACCESS_KEY": "test-secret"}), \
             self.assertRaises(ValueError) as caught:
            release.fetch_private_model(self.entry, self.base / "download")
        self.assertNotIn("test-secret", str(caught.exception))

    def test_full_model_declarations_still_required_in_split_image(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "musetalk_install_state.py").write_text(
            "SERVER_MODEL_FILES = ['models/server/model.bin']\nAVATAR_PREP_MODEL_FILES = ['models/face_detection/s3fd.pth']\n")
        with self.assertRaises(ValueError):
            release.verify_model_contract(self.root, self.manifest)
        release.verify_model_contract(self.root, {**self.manifest, "model_files": {"models/server/model.bin": {}}})

    def test_actual_full_api_contract_excludes_training_but_keeps_preparation(self):
        source = (Path(__file__).resolve().parents[3] / "scripts/musetalk_install_state.py").read_text()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/musetalk_install_state.py").write_text(source)
        lists = {target.id: ast.literal_eval(node.value)
                 for node in ast.parse(source).body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name) and
                 target.id in {"SERVER_MODEL_FILES", "AVATAR_PREP_MODEL_FILES"}}
        required = set(lists["SERVER_MODEL_FILES"] + lists["AVATAR_PREP_MODEL_FILES"])
        self.assertEqual(len(required), 13)
        self.assertNotIn("models/syncnet/latentsync_syncnet.pt", required)
        manifest = {"model_files": dict.fromkeys(required, {}), "external_model_files": {}}
        release.verify_model_contract(self.root, manifest)
        for missing in ("models/dwpose/dw-ll_ucoco_384.pth", "models/face_detection/s3fd.pth",
                        "models/taesd/diffusion_pytorch_model.safetensors"):
            incomplete = copy.deepcopy(manifest)
            del incomplete["model_files"][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                release.verify_model_contract(self.root, incomplete)

    def test_cpu_image_check_defers_only_declared_private_models(self):
        manifest = {**self.manifest, "notices": {}, "kokoro": False}
        with patch.object(release, "verify_source"), patch.object(release, "verify_model_contract") as contract, \
             patch.object(release, "descriptor"), patch.object(release.subprocess, "run") as run:
            release.cpu_check(self.root, manifest)
            contract.assert_called_once_with(self.root, manifest)
            self.assertIn("--skip-weights", run.call_args_list[0].args[0])
        target = self.root / self.name
        target.parent.mkdir(parents=True)
        target.write_bytes(self.data)
        with patch.object(release, "verify_source"), patch.object(release, "verify_model_contract"), \
             patch.object(release, "descriptor"), patch.object(release.subprocess, "run") as run, \
             self.assertRaises(ValueError):
            release.cpu_check(self.root, manifest)
        run.assert_not_called()

    def test_immutable_model_restore_precedes_full_install_check_and_reuses_bootstrap(self):
        source = (Path(__file__).resolve().parents[3] / "scripts/vast_onstart.sh").read_text()
        main = source.split("main() {", 1)[1]
        self.assertLess(main.index("runtime-models"), main.index("run_setup_if_needed"))
        before_models = main[:main.index("runtime-models")]
        self.assertIn("if env_flag_is_true \"$IMMUTABLE_BOOT\"; then", before_models)
        self.assertLess(before_models.rindex("bootstrap_runtime_secrets"), before_models.rindex("immutable_runtime_policy"))
        helper = source.split("bootstrap_runtime_secrets() {", 1)[1].split("configure_webrtc_turn() {", 1)[0]
        self.assertIn("(( IMMUTABLE_SECRETS_LOADED ))", helper)
        self.assertIn('IMMUTABLE_SECRETS_LOADED=1', helper)


if __name__ == "__main__":
    unittest.main()
