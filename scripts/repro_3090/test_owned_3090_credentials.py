"""CPU-only bridge tests. No SSH, AWS, GPU, or real credential is accessed."""
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_owned_3090_with_runtime_credentials as bridge


class CredentialBridgeTests(unittest.TestCase):
    def target(self):
        label = "musetalk-r5-3090-dev-synthetic-fixture"
        return {"instance_id": "54999999", "label": label,
                "worker_alias": "musetalk-3090-build-54999999", "worker_hostname": "fixture-host",
                "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
                "ledger": f"/home/ec2-user/.local/state/{label}/startup-ledger.json",
                "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()}

    def test_new_target_preserves_secret_source_and_binds_exact_identity(self):
        original = dict(bridge.__dict__)
        with patch.dict(bridge.__dict__, original):
            target = self.target()
            bridge.apply_owned_target(target)
            self.assertEqual(bridge.INSTANCE, target["instance_id"])
            self.assertEqual(bridge.WORKER_ALIAS, target["worker_alias"])
            self.assertEqual(bridge.LEDGER, target["ledger"])
            self.assertEqual(bridge.SNAPSHOT, original["SNAPSHOT"])
            self.assertEqual(bridge.SECRET_ARN, original["SECRET_ARN"])
            self.assertEqual(bridge.BOOTSTRAP_SHA256, original["BOOTSTRAP_SHA256"])
            self.assertTrue(bridge.OWNED_TARGET_VALIDATED)
            compile(bridge.remote_code(bridge.ec2_export), "<bounded-ec2>", "exec")
            compile(bridge.remote_code(bridge.worker_run, "fixture-nonce"), "<bounded-worker>", "exec")

    def test_target_rejects_production_release_unknown_keys_paths_and_deadlines(self):
        base = self.target()
        bad = [{**base, "instance_id": "51074906"}, {**base, "label": "musetalk-r5-3090-release-fixture"},
               {**base, "worker_alias": "3-way-head-talk"}, {**base, "worker_hostname": "host;unsafe"},
               {**base, "gpu_uuid": "GPU-invented"}, {**base, "ledger": "/tmp/foreign.json"},
               {**base, "SECRET_ARN": "different"}, {**base, "deadline_utc": "2026-01-01T00:00:00+00:00"},
               {**base, "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)).isoformat()},
               {**base, "deadline_utc": "2026-10-08T23:00:00"}, {**base, "instance_id": 54999999}]
        original = dict(bridge.__dict__)
        for target in bad:
            with self.subTest(target=target), patch.dict(bridge.__dict__, original), self.assertRaises(ValueError):
                bridge.apply_owned_target(target)
        self.assertEqual(bridge.INSTANCE, original["INSTANCE"])

    def test_new_target_requires_matching_create_ledger_and_deadline(self):
        original = dict(bridge.__dict__)
        with patch.dict(bridge.__dict__, original):
            target = self.target()
            bridge.apply_owned_target(target)
            ledger = {"instance_id": target["instance_id"], "label": target["label"], "state": "created",
                      "resource_deadline_utc": bridge.DEADLINE, "control_plane_url": "http://127.0.0.1:8000",
                      "events": [{"name": "request_started"}]}
            bridge.validate_ledger(ledger)
            bridge.validate_ledger({**ledger, "state": "reconciled"})
            for bad in ({**ledger, "instance_id": "51074906"}, {**ledger, "label": "foreign"},
                        {**ledger, "state": "initialized"}, {**ledger, "resource_deadline_utc": "different"},
                        {**ledger, "control_plane_url": "https://foreign.invalid"}, {**ledger, "events": []}):
                with self.subTest(ledger=bad), self.assertRaises(ValueError):
                    bridge.validate_ledger(bad)

    def credentials(self):
        return {"AWS_ACCESS_KEY_ID": "synthetic-fixture-id", "AWS_SECRET_ACCESS_KEY": "synthetic-fixture-secret",
                "AWS_DEFAULT_REGION": "us-east-1", "MUSETALK_AWS_SECRET_ID": bridge.SECRET_ARN}

    def test_plan_does_not_connect_or_show_command_arguments(self):
        output = io.StringIO()
        with patch.object(bridge.subprocess, "Popen") as spawn, patch.object(bridge.subprocess, "run") as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(bridge.main(["--", "/bin/true", "possibly-sensitive-argument"]), 0)
        spawn.assert_not_called()
        run.assert_not_called()
        self.assertNotIn("possibly-sensitive", output.getvalue())
        self.assertFalse(json.loads(output.getvalue())["cloud_or_worker_actions"])

    def test_bootstrap_rejects_wrong_secret_unknown_key_and_missing_auth(self):
        for values in ({**self.credentials(), "MUSETALK_AWS_SECRET_ID": "different"},
                       {**self.credentials(), "LINGUA_REGISTRATION_TOKEN": "do-not-transport"},
                       {**self.credentials(), "AWS_SECRET_ACCESS_KEY": ""}):
            with self.subTest(keys=list(values)), self.assertRaises(ValueError):
                bridge.validate_bootstrap(values)

    def test_snapshot_requires_exact_hash_and_template_identity(self):
        values = self.credentials()
        script = "\n".join("export " + key + "=" + shlex.quote(value) for key, value in values.items())
        template = {"id": 408714, "onstart": script}
        with self.assertRaises(ValueError):
            bridge.parse_snapshot(template)
        with patch.object(bridge, "ONSTART_SHA256", hashlib.sha256(script.encode()).hexdigest()):
            self.assertEqual(bridge.parse_snapshot(template), values)
            with self.assertRaises(ValueError):
                bridge.parse_snapshot({**template, "id": 756005})

    def test_snapshot_duplicate_credential_assignment_fails(self):
        script = "export AWS_ACCESS_KEY_ID=first\nexport AWS_ACCESS_KEY_ID=second\n"
        with patch.object(bridge, "ONSTART_SHA256", hashlib.sha256(script.encode()).hexdigest()), self.assertRaises(ValueError):
            bridge.parse_snapshot({"id": 408714, "onstart": script})

    def test_snapshot_allows_only_identical_repeated_secret_reference(self):
        values = self.credentials()
        script = "\n".join("export " + key + "=" + shlex.quote(value) for key, value in values.items())
        script += "\nexport MUSETALK_AWS_SECRET_ID=" + shlex.quote(bridge.SECRET_ARN) + "\n"
        with patch.object(bridge, "ONSTART_SHA256", hashlib.sha256(script.encode()).hexdigest()):
            self.assertEqual(bridge.parse_snapshot({"id": 408714, "onstart": script}), values)

    def runtime_payload(self):
        return {"AWS_ACCESS_KEY_ID": "runtime-fixture-id", "AWS_SECRET_ACCESS_KEY": "runtime-fixture-secret",
                "AWS_DEFAULT_REGION": "us-east-1", "AVATAR_S3_BUCKET": "lingua-musetalk-s3-storage"}

    def test_runtime_env_drops_loader_registration_and_bootstrap_credentials(self):
        payload = self.runtime_payload()
        module = SimpleNamespace(_exports_from_payload=lambda p: dict(p))
        contaminated = {"AWS_ACCESS_KEY_ID": "bootstrap-fixture-id", "AWS_SECRET_ACCESS_KEY": "bootstrap-fixture-secret",
                        "LINGUA_WORKER_REGISTRATION_TOKEN": "fixture-token", "PYTHONPATH": "/untrusted",
                        "AWS_ENDPOINT_URL": "https://untrusted.invalid", "HTTPS_PROXY": "https://untrusted.invalid"}
        with patch.dict(os.environ, contaminated, clear=True):
            env = bridge.runtime_child_env(module, payload)
        self.assertEqual(env["AWS_ACCESS_KEY_ID"], payload["AWS_ACCESS_KEY_ID"])
        self.assertEqual(env["LINGUA_CONTROL_PLANE_ENABLED"], "0")
        for key in ("LINGUA_WORKER_REGISTRATION_TOKEN", "PYTHONPATH", "AWS_ENDPOINT_URL", "HTTPS_PROXY"):
            self.assertNotIn(key, env)
        self.assertNotIn("bootstrap-fixture-secret", json.dumps(env))

    def test_new_operational_keys_in_secret_fail_closed(self):
        module = Mock()
        with self.assertRaises(ValueError):
            bridge.runtime_child_env(module, {**self.runtime_payload(), "PID_FILE": "/stale.pid"})
        module._exports_from_payload.assert_not_called()

    def test_remote_code_compiles_and_transport_never_uses_pty_or_shell_interpolation(self):
        for fn in (bridge.ec2_export, bridge.worker_run):
            code = bridge.remote_code(fn, "nonce")
            compile(code, "<remote-memory-script>", "exec")
            args = bridge.ssh_arguments(bridge.WORKER_ALIAS, bridge.WORKER_PYTHON, code)
            self.assertIn("-T", args)
            self.assertIn("StrictHostKeyChecking=yes", args)
            self.assertEqual(shlex.split(args[-1]), [bridge.WORKER_PYTHON, "-B", "-c", code])
            self.assertNotIn("synthetic-fixture-secret", json.dumps(args))

    def test_worker_identity_rejected_before_stdin_or_secret_fetch(self):
        with patch.object(bridge.resource, "setrlimit"), patch.object(bridge.os, "geteuid", return_value=0), \
             patch.object(bridge.socket, "gethostname", return_value="different-worker"), \
             patch.object(bridge.sys, "stdin") as stdin, patch.object(bridge.subprocess, "check_output") as query:
            with self.assertRaises(ValueError):
                bridge.worker_run()
        stdin.buffer.readline.assert_not_called()
        query.assert_not_called()

    def test_bad_handshake_never_extracts_credentials(self):
        worker = Mock()
        worker.poll.return_value = 0
        with patch.object(bridge, "seconds_remaining", return_value=7200), patch.object(bridge.resource, "setrlimit"), \
             patch.object(bridge, "remote_code", return_value="synthetic fixture program"), \
             patch.object(bridge.subprocess, "Popen", return_value=worker), \
             patch.object(bridge, "read_record", return_value={"phase": "ready", "instance_id": "51074906"}), \
             patch.object(bridge.subprocess, "run") as extract:
            with self.assertRaises(ValueError):
                bridge.execute(["/bin/true"], 30)
        extract.assert_not_called()
        worker.stdin.write.assert_not_called()

    def test_denied_runtime_secret_never_starts_command_or_prints_credentials(self):
        request = {"credentials": self.credentials(), "command": ["/bin/true"],
                   "timeout_seconds": 30, "nonce": "test-nonce"}
        fixture = b"fixture-bootstrap"
        module = SimpleNamespace(_read_secret_payload=Mock(side_effect=PermissionError("synthetic-fixture-secret")))
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
        output = io.StringIO()
        with patch.object(bridge.resource, "setrlimit"), patch.object(bridge.os, "geteuid", return_value=0), \
             patch.object(bridge.socket, "gethostname", return_value=bridge.WORKER_HOSTNAME), \
             patch.object(bridge, "seconds_remaining", return_value=7200), \
             patch.object(bridge.subprocess, "check_output", return_value=bridge.GPU_UUID), \
             patch.object(bridge.Path, "is_symlink", return_value=False), \
             patch.object(bridge.Path, "read_bytes", return_value=fixture), \
             patch.object(bridge, "BOOTSTRAP_SHA256", hashlib.sha256(fixture).hexdigest()), \
             patch.object(bridge, "NONCE", "test-nonce", create=True), \
             patch.object(bridge.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode() + b"\n"))), \
             patch.object(bridge.importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(bridge.importlib.util, "module_from_spec", return_value=module), \
             patch.object(bridge.subprocess, "Popen") as spawn, \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output):
            with self.assertRaises(PermissionError):
                bridge.worker_run()
        spawn.assert_not_called()
        self.assertNotIn("synthetic-fixture", output.getvalue())

    def test_execute_transfers_credentials_only_on_stdin_and_discards_output(self):
        worker = Mock()
        worker.poll.return_value = 0
        worker.wait.return_value = 0
        transmitted = []
        def write(value):
            transmitted.append(bytes(value))
            return len(value)
        worker.stdin.write.side_effect = write
        ready = {"phase": "ready", "instance_id": bridge.INSTANCE, "nonce": "fixed-nonce"}
        done = {"phase": "finished", "instance_id": bridge.INSTANCE, "returncode": 0,
                "child_output": "suppressed", "credential_files_written_by_helper": False}
        output = io.StringIO()
        with patch.object(bridge, "seconds_remaining", return_value=7200), patch.object(bridge.resource, "setrlimit"), \
             patch.object(bridge, "remote_code", return_value="synthetic fixture program"), \
             patch.object(bridge.uuid, "uuid4", return_value=SimpleNamespace(hex="fixed-nonce")), \
             patch.object(bridge.subprocess, "Popen", return_value=worker) as spawn, \
             patch.object(bridge, "read_record", side_effect=[ready, done]), \
             patch.object(bridge.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(self.credentials()).encode())) as extract, \
             contextlib.redirect_stdout(output):
            self.assertEqual(bridge.execute(["/bin/true"], 30), 0)
        self.assertIn(b"synthetic-fixture-secret", b"".join(transmitted))
        self.assertNotIn("synthetic-fixture-secret", str(spawn.call_args) + str(extract.call_args) + output.getvalue())
        self.assertEqual(spawn.call_args.kwargs["stderr"], bridge.subprocess.DEVNULL)
        self.assertEqual(extract.call_args.kwargs["stderr"], bridge.subprocess.DEVNULL)

    def test_expired_or_near_deadline_never_connects(self):
        with patch.object(bridge, "seconds_remaining", return_value=45), patch.object(bridge.subprocess, "Popen") as spawn:
            with self.assertRaises(ValueError):
                bridge.execute(["/bin/true"], 30)
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
