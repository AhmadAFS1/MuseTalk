"""CPU process-lifecycle tests; Linux /proc data is synthetic where unavailable."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import supervise


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def proc(self, marker="owner", ticks=123, state="S"):
        proc = self.root / "proc/200"
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "environ").write_bytes((supervise.OWNER_ENV + "=" + marker + "\0").encode())
        (proc / "stat").write_text("200 (space and ) parentheses) " + " ".join([state] + ["0"] * 18 + [str(ticks)]))
        return proc.parent

    def test_proc_marker_start_ticks_and_zombies(self):
        proc = self.proc()
        self.assertEqual(supervise.identity(200, "owner", proc), 123)
        self.assertIsNone(supervise.identity(200, "other", proc))
        self.assertIsNone(supervise.identity(1, "owner", proc))
        self.proc(state="Z")
        self.assertIsNone(supervise.identity(200, "owner", proc))

    def test_stale_pidfile_does_not_own_reused_process(self):
        proc = self.proc(marker="other-container")
        path = self.root / "api.pid"
        path.write_text("200\n")
        self.assertIsNone(supervise.checked_pid_file(path, "owner", proc))
        self.proc(marker="owner", ticks=900)
        self.assertEqual(supervise.checked_pid_file(path, "owner", proc), (200, 900))
        path.write_text("not a pid\n")
        self.assertIsNone(supervise.checked_pid_file(path, "owner", proc))

    def test_supervisors_never_reuse_persistent_pid_paths(self):
        env = {"LOG_DIR": str(self.root / "persistent"), "PID_FILE": str(self.root / "stale.pid"),
               "LINGUA_CONTROL_PLANE_ENV_FILE": str(self.root / "stale.env")}
        first = supervise.Supervisor(self.root, env, self.root / "run")
        second = supervise.Supervisor(self.root, env, self.root / "run")
        self.assertNotEqual(first.api_pid, second.api_pid)
        self.assertNotEqual(first.marker, second.marker)
        self.assertEqual(Path(first.env["PID_FILE"]).parent, first.state)
        self.assertNotEqual(first.env["PID_FILE"], env["PID_FILE"])
        self.assertEqual(first.env["LINGUA_CONTROL_PLANE_ENV_FILE"], "/dev/null")
        self.assertEqual(first.env["MUSETALK_BOOTSTRAP_SECRET_DIR"], str(first.state))

    def test_pidfd_rechecks_identity_before_signal(self):
        with patch.object(supervise.os, "pidfd_open", return_value=99, create=True), \
             patch.object(supervise.signal, "pidfd_send_signal", create=True) as send, \
             patch.object(supervise.os, "close") as close, patch.object(supervise, "identity", return_value=124):
            supervise.signal_owned({200: 123}, "owner", signal.SIGTERM)
            send.assert_not_called()
            close.assert_called_once_with(99)
        with patch.object(supervise.os, "pidfd_open", return_value=99, create=True), \
             patch.object(supervise.signal, "pidfd_send_signal", create=True) as send, \
             patch.object(supervise.os, "close"), patch.object(supervise, "identity", return_value=123):
            supervise.signal_owned({200: 123}, "owner", signal.SIGTERM)
            send.assert_called_once_with(99, signal.SIGTERM, None, 0)

    def test_ctl_guard_rejects_changed_starttime(self):
        with patch.dict(os.environ, {supervise.OWNER_ENV: "owner", "MUSETALK_SUPERVISOR_IDENTITIES": json.dumps({200: 123})}), \
             patch.object(supervise, "identity", return_value=456), patch.object(supervise, "signal_owned") as send:
            self.assertEqual(supervise.process_command(200, "TERM"), 1)
            send.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open")
                         and hasattr(signal, "pidfd_send_signal"), "real Linux pidfd test")
    def test_real_linux_pidfd_only_signals_marked_process(self):
        marker = "musetalk-test-" + str(os.getpid()) + "-" + str(time.monotonic_ns())
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 env={**os.environ, supervise.OWNER_ENV: marker})
        try:
            deadline = time.monotonic() + 2
            ticks = None
            while ticks is None and time.monotonic() < deadline:
                ticks = supervise.identity(child.pid, marker)
                time.sleep(0.01)
            self.assertIsNotNone(ticks)
            supervise.signal_owned({child.pid: ticks}, marker + "wrong", signal.SIGTERM)
            self.assertIsNone(child.poll())
            supervise.signal_owned({child.pid: ticks + 1}, marker, signal.SIGTERM)
            self.assertIsNone(child.poll())
            supervise.signal_owned({child.pid: ticks}, marker, signal.SIGTERM)
            self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_terminate_boot_group_does_not_wait_for_foreground_sleep(self):
        child = subprocess.Popen(["bash", "-c", "sleep 30 & wait"], start_new_session=True)
        try:
            start = time.monotonic()
            supervise.terminate_group(child, timeout=0.2)
            self.assertIsNotNone(child.poll())
            self.assertLess(time.monotonic() - start, 1.0)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def test_signal_during_boot_interrupts_and_reaps_before_cleanup(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "vast_onstart.sh").write_text("exec sleep 30\n")
        owner = supervise.Supervisor(self.root, {**os.environ, "MUSETALK_SHUTDOWN_TIMEOUT_SECONDS": "10"}, self.root / "run")
        timer = threading.Timer(0.2, lambda: owner.receive_signal(signal.SIGTERM, None))
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        timer.start()
        try:
            with patch.object(supervise.os, "pidfd_open", side_effect=lambda *_: os.open(os.devnull, os.O_RDONLY), create=True), \
                 patch.object(supervise.signal, "pidfd_send_signal", create=True), \
                 patch.object(supervise, "owned_pids", return_value={}):
                start = time.monotonic()
                self.assertEqual(owner.run(), 143)
                self.assertLess(time.monotonic() - start, 2.0)
                self.assertIsNotNone(owner.boot.poll())
        finally:
            timer.cancel()
            for sig, handler in old.items():
                signal.signal(sig, handler)

    def test_invalid_shutdown_deadline_rejected(self):
        with self.assertRaises(ValueError):
            supervise.Supervisor(self.root, {"MUSETALK_SHUTDOWN_TIMEOUT_SECONDS": "0"}, self.root / "run")

    def test_denied_pidfd_signal_fails_before_launch_and_closes_descriptor(self):
        owner = supervise.Supervisor(self.root, {}, self.root / "run")
        with patch.object(supervise.signal, "signal"), \
             patch.object(supervise.os, "pidfd_open", return_value=99, create=True), \
             patch.object(supervise.signal, "pidfd_send_signal", side_effect=PermissionError("denied"), create=True), \
             patch.object(supervise.os, "close") as close, patch.object(supervise.subprocess, "Popen") as spawn:
            with self.assertRaises(PermissionError):
                owner.run()
            spawn.assert_not_called()
            close.assert_called_once_with(99)

    def test_hung_drain_is_interrupted_with_cleanup_budget_reserved(self):
        owner = supervise.Supervisor(self.root, {"MUSETALK_SHUTDOWN_TIMEOUT_SECONDS": "10"}, self.root / "run")
        owner.api_pid.write_text("200\n")
        stop = Mock()
        stop.wait.side_effect = subprocess.TimeoutExpired("ctl stop", timeout=3)
        with patch.object(supervise, "owned_pids", return_value={200: 123}), \
             patch.object(supervise, "checked_pid_file", return_value=(200, 123)), \
             patch.object(supervise.subprocess, "Popen", return_value=stop) as spawn, \
             patch.object(supervise.time, "monotonic", side_effect=[100, 100, 108, 110]), \
             patch.object(supervise, "terminate_group") as terminate, \
             patch.object(supervise, "signal_owned") as send:
            owner.cleanup()
            stop.wait.assert_called_once_with(timeout=3)
            terminate.assert_called_once_with(stop, timeout=1)
            self.assertEqual([call.args[2] for call in send.call_args_list], [signal.SIGTERM, signal.SIGKILL])
            self.assertEqual(json.loads(owner.env["MUSETALK_SUPERVISOR_IDENTITIES"]), {"200": 123})
            self.assertEqual(spawn.call_args.kwargs["env"]["LINGUA_CONTROL_PLANE_ENV_FILE"], "/dev/null")
            self.assertEqual(spawn.call_args.kwargs["env"]["VAST_SERVER_CTL_LOAD_TURN_ENV"], "0")

    def test_supervisor_secret_cleanup_runs_even_when_process_cleanup_fails(self):
        owner = supervise.Supervisor(self.root, {}, self.root / "run")
        secret = owner.state / "secret-test.env"
        secret.write_text("synthetic-secret")
        keep = owner.state / "unrelated.txt"
        keep.write_text("keep")
        with patch.object(owner, "cleanup_processes", side_effect=OSError("denied")), self.assertRaises(OSError):
            owner.cleanup()
        self.assertFalse(secret.exists())
        self.assertTrue(keep.exists())

    def test_ctl_freezes_image_ownership_before_env_sourcing(self):
        source = (Path(__file__).resolve().parents[3] / "scripts/vast_server_ctl.sh").read_text()
        prefix = source.split("# shellcheck source=lib/musetalk_env_layers.sh", 1)[0]
        test = self.root / "ctl-prefix.sh"
        test.write_text(prefix + '\nMUSETALK_SUPERVISOR_OWNER=stale\n')
        env = {**os.environ, "MUSETALK_SUPERVISOR_OWNER": "current", "PID_FILE": str(self.root / "api.pid"),
               "TURN_PID_FILE": str(self.root / "turn.pid"), "LINGUA_CONTROL_PLANE_ENV_FILE": str(self.root / "stale.env")}
        result = subprocess.run(["bash", str(test)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("readonly", result.stderr)

    def test_ctl_real_main_dispatch_preserves_readonly_image_pid_paths(self):
        source = (Path(__file__).resolve().parents[3] / "scripts/vast_server_ctl.sh").read_text()
        prefix = source.split("# shellcheck source=lib/musetalk_env_layers.sh", 1)[0]
        main = "main() {" + source.split("\nmain() {", 1)[1].split('\nmain "$@"', 1)[0]
        test = self.root / "ctl-main.sh"
        test.write_text(prefix + '\nload_turn_env() { :; }\n'
                        'start_server() { echo START; }\nstop_server() { echo STOP; }\n' + main + '\nmain "$@"\n')
        env = {**os.environ, "MUSETALK_SUPERVISOR_OWNER": "current", "PID_FILE": str(self.root / "api.pid"),
               "TURN_PID_FILE": str(self.root / "turn.pid")}
        for command in ("start", "stop"):
            result = subprocess.run(["bash", str(test), command], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), command.upper())

    def test_secret_file_removed_after_source_readonly_failure_or_term(self):
        source = (Path(__file__).resolve().parents[3] / "scripts/vast_onstart.sh").read_text()
        helpers = []
        for name in ("cleanup_bootstrap_secret", "report_exit", "report_signal"):
            helpers.append(name + "() {" + source.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n")
        for mode in ("source-failure", "term"):
            secret = self.root / (mode + ".env")
            secret.write_text("export AWS_SECRET_ACCESS_KEY='synthetic-secret'\nMUSETALK_SUPERVISOR_OWNER=stale\n")
            script = self.root / (mode + ".sh")
            script.write_text('set -Eeuo pipefail\nSCRIPT_NAME=test\nONSTART_MARKER_PRINTED=1\n'
                              'in_main_shell() { return 0; }\nprint_failed_marker() { :; }\n'
                              + "".join(helpers) + '\ntrap report_exit EXIT\ntrap "report_signal TERM" TERM\n'
                              + "readonly BOOTSTRAP_SECRET_TMP=" + str(secret) + "\nreadonly MUSETALK_SUPERVISOR_OWNER=current\n"
                              + ('source "$BOOTSTRAP_SECRET_TMP"\n' if mode == "source-failure" else 'kill -TERM $$\n'))
            result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
            if mode == "term":
                self.assertEqual(result.returncode, 143)
            else:
                # macOS Bash 3.2 reports some fatal readonly source errors as
                # exit 0; image/CI use Bash >=4. Cleanup must work on both.
                self.assertIn("readonly", result.stderr)
            self.assertFalse(secret.exists(), mode)
            self.assertNotIn("synthetic-secret", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
