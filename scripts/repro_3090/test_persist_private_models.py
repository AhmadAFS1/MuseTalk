"""Synthetic CPU tests; no AWS, network, model loading, or GPU operation."""
import base64
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persist_private_models as persist
import privacy_attestation
from test_privacy_attestation import valid_fixture


class SDKFailure(Exception):
    def __init__(self, status):
        super().__init__("synthetic-credential-value must never be printed")
        self.response = {"ResponseMetadata": {"HTTPStatusCode": status}}


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "audit/harnesses/quality-inputs-v1.json"
        self.manifest.parent.mkdir(parents=True)
        self.rows = []
        self.data = {}
        for name in sorted(persist.MODEL_PATHS):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("synthetic content " + name).encode()
            path.write_bytes(data)
            self.data[name] = data
            self.rows.append({"path": os.path.relpath(path, self.manifest.parent),
                              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.save_manifest()
        self.pin = patch.object(persist, "QUALITY_MANIFEST_SHA256", hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.pin.start()

    def tearDown(self):
        self.pin.stop()
        self.tmp.cleanup()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({"schema": "repro_3090_inputs_v1", "files": self.rows}))

    def repin(self):
        self.save_manifest()
        persist.QUALITY_MANIFEST_SHA256 = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def entry(self):
        name = sorted(persist.MODEL_PATHS)[0]
        return name, persist.plan(self.root, self.manifest)[name]

    def attestation(self):
        data = valid_fixture()
        path = self.root / "privacy.json"
        path.write_text(json.dumps(data))
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), data

    def s3(self, name):
        result = Mock()
        result.head_object.return_value = {"ContentLength": len(self.data[name])}
        result.get_object.return_value = {"ContentLength": len(self.data[name]), "Body": io.BytesIO(self.data[name]),
                                          "VersionId": "fixture-version"}
        result.get_object_acl.return_value = {"Grants": []}
        return result

    def test_exact_allowlist_matches_docker_contract(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("delivery_release", Path(__file__).resolve().parents[2] / "docker/musetalk/release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(persist.MODEL_PATHS, module.EXTERNAL_MODEL_ALLOWLIST)
        rows = persist.plan(self.root, self.manifest)
        self.assertEqual(module.external_models({"model_files": {}, "external_model_files": rows,
                                                "redistribution_scope": "baked_model_files_only",
                                                "private_model_usage_rights": "unresolved"}), rows)

    def test_actual_manifest_and_all_five_local_hashes_required(self):
        rows = persist.plan(self.root, self.manifest)
        self.assertEqual(set(rows), persist.MODEL_PATHS)
        for name, entry in rows.items():
            self.assertEqual(entry["source"]["key"], persist.PREFIX + "/" + entry["sha256"] + "/" + Path(name).name)
            self.assertFalse(entry["public_redistribution"])
            self.assertTrue(entry["private_delivery_authorized"])
        with patch.object(persist, "QUALITY_MANIFEST_SHA256", "0" * 64), self.assertRaises(persist.Invalid):
            persist.plan(self.root, self.manifest)
        (self.root / sorted(persist.MODEL_PATHS)[0]).write_bytes(b"changed")
        with self.assertRaises(persist.Invalid):
            persist.plan(self.root, self.manifest)

    def test_missing_or_duplicate_pin_fails(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.repin()
        with self.assertRaisesRegex(persist.Invalid, "duplicate"):
            persist.plan(self.root, self.manifest)
        self.rows = self.rows[1:-1]
        self.repin()
        with self.assertRaisesRegex(persist.Invalid, "five"):
            persist.plan(self.root, self.manifest)

    def test_symlink_input_is_not_materialized_silently(self):
        name = sorted(persist.MODEL_PATHS)[0]
        path = self.root / name
        target = self.root / "fixture-original"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(persist.Invalid):
            persist.plan(self.root, self.manifest)

    def test_existing_remote_is_stream_hashed_and_never_uploaded(self):
        name, entry = self.entry()
        s3 = self.s3(name)
        proof = persist.persist_one(s3, self.root, name, entry)
        self.assertEqual(proof["action"], "reused_existing")
        self.assertTrue(proof["remote_content_verified"])
        self.assertFalse(proof["etag_used_as_hash"])
        self.assertEqual(entry["source"]["version_id"], "fixture-version")
        s3.put_object.assert_not_called()
        self.assertTrue(s3.get_object.return_value["Body"].closed)

    def test_missing_upload_conditional_owner_checksum_and_no_acl(self):
        name, entry = self.entry()
        s3 = self.s3(name)
        s3.head_object.side_effect = SDKFailure(404)
        proof = persist.persist_one(s3, self.root, name, entry)
        arguments = s3.put_object.call_args.kwargs
        self.assertEqual(arguments["IfNoneMatch"], "*")
        self.assertEqual(arguments["ExpectedBucketOwner"], persist.OWNER)
        self.assertEqual(arguments["ChecksumSHA256"], base64.b64encode(bytes.fromhex(entry["sha256"])).decode())
        self.assertEqual(arguments["ServerSideEncryption"], "AES256")
        self.assertNotIn("ACL", arguments)
        self.assertEqual(proof["action"], "uploaded_missing")
        self.assertTrue(proof["remote_content_verified"])

    def test_head_denial_does_not_mean_missing(self):
        name, entry = self.entry()
        s3 = self.s3(name)
        s3.head_object.side_effect = SDKFailure(403)
        with self.assertRaisesRegex(persist.Invalid, "existence_unknown"):
            persist.persist_one(s3, self.root, name, entry)
        s3.put_object.assert_not_called()

    def test_conditional_race_only_reconciles_by_get_never_overwrites(self):
        name, entry = self.entry()
        for status in (409, 412):
            s3 = self.s3(name)
            s3.head_object.side_effect = SDKFailure(404)
            s3.put_object.side_effect = SDKFailure(status)
            proof = persist.persist_one(s3, self.root, name, copy.deepcopy(entry))
            self.assertEqual(proof["action"], "concurrent_object_reconciled_by_get")
            self.assertEqual(s3.put_object.call_count, 1)
            self.assertTrue(proof["remote_content_verified"])

    def test_put_failure_suppresses_sdk_value_and_does_not_retry(self):
        name, entry = self.entry()
        s3 = self.s3(name)
        s3.head_object.side_effect = SDKFailure(404)
        s3.put_object.side_effect = SDKFailure(403)
        with self.assertRaises(persist.Invalid) as error:
            persist.persist_one(s3, self.root, name, entry)
        self.assertNotIn("synthetic-credential-value", str(error.exception))
        self.assertEqual(s3.put_object.call_count, 1)
        s3.get_object.assert_not_called()

    def test_bad_remote_hash_or_size_is_never_repaired(self):
        name, entry = self.entry()
        for replacement in (b"X" * len(self.data[name]), self.data[name][:-1]):
            s3 = self.s3(name)
            s3.get_object.return_value["Body"] = io.BytesIO(replacement)
            with self.assertRaises(persist.Invalid):
                persist.persist_one(s3, self.root, name, entry)
            s3.put_object.assert_not_called()
            self.assertTrue(s3.get_object.return_value["Body"].closed)

    def test_public_object_acl_fails_even_with_good_bytes(self):
        name, entry = self.entry()
        s3 = self.s3(name)
        s3.get_object_acl.return_value = {"Grants": [{"Grantee": {"URI": next(iter(persist.PUBLIC_GROUPS))}}]}
        with self.assertRaisesRegex(persist.Invalid, "public_acl"):
            persist.persist_one(s3, self.root, name, entry)
        s3.put_object.assert_not_called()

    def test_denied_privacy_reads_are_not_claimed_private_proof(self):
        s3 = Mock()
        for method in ("get_public_access_block", "get_bucket_policy_status", "get_bucket_acl"):
            getattr(s3, method).side_effect = SDKFailure(403)
        result = persist.privacy_observations(s3)
        self.assertFalse(result["independent_privacy_proof"])
        self.assertEqual(result["bucket_policy_public"]["status"], "not_available")
        self.assertNotIn("synthetic-credential-value", json.dumps(result))

    def test_only_complete_boolean_bucket_block_is_affirmative_proof(self):
        for config, expected in (({k: True for k in persist.PUBLIC_ACCESS_BLOCK_FLAGS}, True),
                                 ({k: False for k in persist.PUBLIC_ACCESS_BLOCK_FLAGS}, False),
                                 ({k: "true" for k in persist.PUBLIC_ACCESS_BLOCK_FLAGS}, False),
                                 ({"BlockPublicAcls": True}, False), ({}, False)):
            s3 = Mock()
            s3.get_public_access_block.return_value = {"PublicAccessBlockConfiguration": config}
            s3.get_bucket_policy_status.side_effect = SDKFailure(403)
            s3.get_bucket_acl.side_effect = SDKFailure(403)
            observation = persist.privacy_observations(s3)
            self.assertEqual(observation["independent_privacy_proof"], expected)
            s3.get_public_access_block.assert_called_once_with(Bucket=persist.BUCKET, ExpectedBucketOwner=persist.OWNER)

    def test_unproven_or_known_public_bucket_stops_before_any_objects(self):
        for observation in (
                {"independent_privacy_proof": False, "bucket_policy_public": {}, "bucket_public_acl_grants": {}},
                {"independent_privacy_proof": True, "bucket_policy_public": {"value": True}, "bucket_public_acl_grants": {}},
                {"independent_privacy_proof": True, "bucket_policy_public": {}, "bucket_public_acl_grants": {"value": True}}):
            report = {"objects": {}, "external_model_files": {}, "delivery_verified": False}
            args = SimpleNamespace(root=self.root, manifest=self.manifest, execute=True)
            with patch.object(persist, "WORKER_ROOT", str(self.root.resolve())), \
                 patch.object(persist.socket, "gethostname", return_value=persist.WORKER_HOSTNAME), \
                 patch.object(persist, "write_report"), \
                 patch.object(persist, "privacy_observations", return_value=observation), \
                 patch.object(persist, "persist_one") as object_operation, self.assertRaises(persist.Invalid):
                persist.run(args, report, io.StringIO(), make_client=Mock())
            object_operation.assert_not_called()
            self.assertEqual(report["external_model_files"], {})
            self.assertFalse(report["delivery_verified"])

    def test_sdk_has_explicit_runtime_credentials_and_bounded_safe_config(self):
        s3 = Mock()
        s3.meta.service_model.operation_model.return_value.input_shape.members = {"IfNoneMatch": {}}
        boto = SimpleNamespace(client=Mock(return_value=s3))
        config = SimpleNamespace(Config=Mock(return_value="fixture-config"))
        with patch.dict(sys.modules, {"boto3": boto, "botocore.config": config}), \
             patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "fixture-id", "AWS_SECRET_ACCESS_KEY": "fixture-secret",
                                     "AWS_ENDPOINT_URL": "https://untrusted.invalid"}, clear=True):
            self.assertIs(persist.client(), s3)
        self.assertEqual(boto.client.call_args.kwargs["aws_access_key_id"], "fixture-id")
        self.assertEqual(boto.client.call_args.kwargs["region_name"], persist.REGION)
        self.assertEqual(config.Config.call_args.kwargs["retries"]["total_max_attempts"], 3)
        self.assertTrue(config.Config.call_args.kwargs["ignore_configured_endpoint_urls"])
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(persist.Invalid):
            persist.client()

    def test_explicit_bound_attestation_allows_denied_reads_but_live_proof_wins(self):
        path, digest, data = self.attestation()
        unavailable = {"independent_privacy_proof": False, "bucket_public_access_block": {"status": "not_available"},
                       "bucket_policy_public": {}, "bucket_public_acl_grants": {}}
        source, loaded = persist.choose_privacy_proof(unavailable, path, digest)
        self.assertTrue(source["attestation_used"])
        self.assertEqual(source["sha256"], digest)
        self.assertEqual(loaded, data)
        source, loaded = persist.choose_privacy_proof({**unavailable, "independent_privacy_proof": True}, path, digest)
        self.assertFalse(source["attestation_used"])
        self.assertIsNone(loaded)
        with self.assertRaisesRegex(persist.Invalid, "sha_mismatch"):
            persist.choose_privacy_proof({**unavailable, "independent_privacy_proof": True}, path, "0" * 64)

    def test_attestation_cannot_override_any_live_public_or_false_block_observation(self):
        path, digest, _ = self.attestation()
        for observation in (
                {"bucket_policy_public": {"value": True}}, {"bucket_public_acl_grants": {"value": True}},
                {"bucket_public_access_block": {"status": "observed", "value": {key: False for key in persist.PUBLIC_ACCESS_BLOCK_FLAGS}}}):
            with self.subTest(observation=observation), self.assertRaises(persist.Invalid):
                persist.choose_privacy_proof(observation, path, digest)
        for arguments in ((path, None), (None, digest), (None, None)):
            with self.assertRaises(persist.Invalid):
                persist.choose_privacy_proof({}, *arguments)

    def test_attested_full_run_and_freshness_recheck_before_every_object(self):
        path, digest, data = self.attestation()
        for expires_after_first in (False, True):
            report = {"objects": {}, "external_model_files": {}, "delivery_verified": False}
            args = SimpleNamespace(root=self.root, manifest=self.manifest, execute=True,
                                   privacy_attestation=path, privacy_attestation_sha256=digest)
            # load_bound normally validates first. Mock only that completed
            # validation so this check isolates the per-object expiry checks.
            freshness = ([{"age_seconds": 899}, privacy_attestation.Invalid("privacy_attestation_stale_or_future")]
                         if expires_after_first else [{"age_seconds": 1}] * 5)
            with patch.object(persist, "WORKER_ROOT", str(self.root.resolve())), \
                 patch.object(persist.socket, "gethostname", return_value=persist.WORKER_HOSTNAME), \
                 patch.object(persist, "write_report"), \
                 patch.object(persist, "privacy_observations", return_value={"independent_privacy_proof": False}), \
                 patch.object(privacy_attestation, "load_bound", return_value=data), \
                 patch.object(privacy_attestation, "validate", side_effect=freshness) as validate, \
                 patch.object(persist, "persist_one", return_value={"remote_content_verified": True}) as operation:
                if expires_after_first:
                    with self.assertRaisesRegex(persist.Invalid, "stale_or_future"):
                        persist.run(args, report, io.StringIO(), make_client=Mock())
                    self.assertEqual(operation.call_count, 1)
                    self.assertEqual(report["external_model_files"], {})
                else:
                    self.assertEqual(persist.run(args, report, io.StringIO(), make_client=Mock()), 0)
                    self.assertEqual(operation.call_count, 5)
                    self.assertEqual(validate.call_count, 5)
                    self.assertTrue(report["privacy_proof"]["attestation_used"])

    def test_plan_only_writes_nonconsumable_proposal_without_sdk(self):
        output = self.root / "report.json"
        with patch.object(persist, "client") as sdk, contextlib.redirect_stdout(io.StringIO()):
            code = persist.main(["--root", str(self.root), "--manifest", str(self.manifest), "--out", str(output)])
        self.assertEqual(code, 0)
        sdk.assert_not_called()
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "plan_only")
        self.assertEqual(report["external_model_files"], {})
        self.assertEqual(set(report["planned_external_model_files"]), persist.MODEL_PATHS)
        self.assertFalse(report["delivery_verified"])
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            persist.main(["--root", str(self.root), "--manifest", str(self.manifest), "--out", str(output)])

    def test_failed_run_records_safe_error_and_no_delivery_manifest(self):
        output = self.root / "failure.json"
        with patch.object(persist, "run", side_effect=SDKFailure(403)), contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = persist.main(["--root", str(self.root), "--manifest", str(self.manifest), "--out", str(output), "--execute"])
        report = json.loads(output.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["external_model_files"], {})
        self.assertNotIn("synthetic-credential-value", output.read_text() + stdout.getvalue())

    def test_all_five_remote_proofs_required_for_consumable_manifest(self):
        report = {"objects": {}, "external_model_files": {}, "delivery_verified": False}
        handle = io.StringIO()
        args = SimpleNamespace(root=self.root, manifest=self.manifest, execute=True)
        s3 = Mock()
        proof = {"remote_content_verified": True}
        with patch.object(persist, "WORKER_ROOT", str(self.root.resolve())), \
             patch.object(persist.socket, "gethostname", return_value=persist.WORKER_HOSTNAME), \
             patch.object(persist, "write_report"), \
             patch.object(persist, "privacy_observations", return_value={"independent_privacy_proof": True, "bucket_policy_public": {}, "bucket_public_acl_grants": {}}), \
             patch.object(persist, "persist_one", return_value=proof) as persist_one:
            self.assertEqual(persist.run(args, report, handle, make_client=lambda: s3), 0)
        self.assertEqual(persist_one.call_count, 5)
        self.assertEqual(set(report["external_model_files"]), persist.MODEL_PATHS)
        self.assertTrue(report["delivery_verified"])

    def test_read_only_parallel_dispatch_never_calls_persistence(self):
        report = {"objects": {}, "external_model_files": {}, "delivery_verified": False}
        args = SimpleNamespace(root=self.root, manifest=self.manifest, execute=True, read_only_parallel=True)
        with patch.object(persist, "WORKER_ROOT", str(self.root.resolve())), \
             patch.object(persist.socket, "gethostname", return_value=persist.WORKER_HOSTNAME), \
             patch.object(persist, "write_report"), \
             patch.object(persist, "privacy_observations", return_value={"independent_privacy_proof": True}), \
             patch.object(persist, "persist_one") as mutation, \
             patch.object(persist, "verify_remote_parallel", return_value={"remote_content_verified": True}) as read:
            self.assertEqual(persist.run(args, report, io.StringIO(), make_client=Mock()), 0)
        mutation.assert_not_called()
        self.assertEqual(read.call_count, 5)
        self.assertTrue(report["delivery_verified"])
        self.assertTrue(all("version_id" not in row["source"] for row in report["external_model_files"].values()))


if __name__ == "__main__":
    unittest.main()
