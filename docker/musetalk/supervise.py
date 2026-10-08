#!/usr/bin/env python3
"""Linux image process owner; canonical boot/drain with bounded signal handling.

Never trust persistent numeric PID files. Every supervisor gets new PID paths
and a unique inherited owner marker. Before stopping any process, check both the
marker and Linux /proc start ticks to reject PID reuse.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid
import sys

OWNER_ENV = "MUSETALK_SUPERVISOR_OWNER"


def identity(pid, marker, proc_root=Path("/proc")):
    """Return stable start ticks only for a live, non-zombie owned process."""
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        root = Path(proc_root) / str(pid)
        values = (root / "environ").read_bytes().split(b"\0")
        if (OWNER_ENV + "=" + marker).encode() not in values:
            return None
        # comm can contain spaces or ')'; parse after its final close parenthesis.
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])  # field 22 starttime; suffix begins at field 3
    except (OSError, ValueError, IndexError):
        return None


def owned_pids(marker, proc_root=Path("/proc")):
    found = {}
    for path in Path(proc_root).iterdir():
        if path.name.isdigit():
            pid = int(path.name)
            ticks = identity(pid, marker, proc_root)
            if ticks is not None and pid != os.getpid():
                found[pid] = ticks
    return found


def checked_pid_file(path, marker, proc_root=Path("/proc")):
    try:
        raw = Path(path).read_text().strip()
        pid = int(raw) if raw.isdecimal() else -1
    except OSError:
        return None
    ticks = identity(pid, marker, proc_root)
    return (pid, ticks) if ticks is not None else None


def terminate_group(child, timeout=5.0):
    """Interrupt and reap the tracked bootstrap/control command, bounded."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=max(0.01, timeout))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=2)


def signal_owned(processes, marker, sig):
    """Bind the signal to a pidfd, closing the check→kill PID-reuse race."""
    if not processes:
        return
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Immutable Linux runtime requires pidfd signaling support")
    for pid, ticks in processes.items():
        try:
            fd = os.pidfd_open(pid, 0)
        except ProcessLookupError:
            continue
        try:
            if identity(pid, marker) == ticks:
                try:
                    signal.pidfd_send_signal(fd, sig, None, 0)
                except ProcessLookupError:
                    pass
        finally:
            os.close(fd)


class Supervisor:
    def __init__(self, root, env=None, state_parent=Path("/run/musetalk")):
        self.root = Path(root)
        self.env = dict(os.environ if env is None else env)
        self.marker = uuid.uuid4().hex
        self.env[OWNER_ENV] = self.marker
        state_parent.mkdir(parents=True, exist_ok=True)
        self.state = Path(tempfile.mkdtemp(prefix="processes-", dir=state_parent))
        self.state.chmod(0o700)
        # These do not live under the persistent LOG_DIR and cannot be inherited
        # from another container or an earlier process with a reused numeric PID.
        self.api_pid = self.state / "api.pid"
        self.turn_pid = self.state / "turn.pid"
        self.env.update(PID_FILE=str(self.api_pid), TURN_PID_FILE=str(self.turn_pid),
                        MUSETALK_BOOTSTRAP_SECRET_DIR=str(self.state), LINGUA_CONTROL_PLANE_ENV_FILE="/dev/null")
        self.stopping = 0
        self.boot = None
        self.timeout = float(self.env.get("MUSETALK_SHUTDOWN_TIMEOUT_SECONDS", "360"))
        if not 10 <= self.timeout <= 3600:
            raise ValueError("MUSETALK_SHUTDOWN_TIMEOUT_SECONDS must be 10..3600")

    def receive_signal(self, sig, _frame):
        self.stopping = sig

    def run(self):
        signal.signal(signal.SIGTERM, self.receive_signal)
        signal.signal(signal.SIGINT, self.receive_signal)
        # Probe syscall availability before launching any detached service.
        fd = os.pidfd_open(os.getpid(), 0)
        try:
            signal.pidfd_send_signal(fd, 0, None, 0)
        finally:
            os.close(fd)
        try:
            self.boot = subprocess.Popen(["bash", str(self.root / "scripts/vast_onstart.sh")],
                                         cwd=self.root, env=self.env, start_new_session=True)
            while self.boot.poll() is None and not self.stopping:
                time.sleep(0.1)
            if self.stopping:
                return 128 + self.stopping
            if self.boot.returncode:
                return self.boot.returncode
            api = checked_pid_file(self.api_pid, self.marker)
            if api is None:
                print("API PID is absent or not owned after canonical startup", flush=True)
                return 1
            turn = checked_pid_file(self.turn_pid, self.marker) if self.turn_pid.exists() else None
            if self.turn_pid.exists() and turn is None:
                print("TURN PID does not identify an owned process", flush=True)
                return 1
            while not self.stopping:
                if identity(api[0], self.marker) != api[1]:
                    print("Owned API exited; failing for provider restart", flush=True)
                    return 1
                if turn and identity(turn[0], self.marker) != turn[1]:
                    print("Owned TURN exited; failing for provider restart", flush=True)
                    return 1
                time.sleep(0.2)
            return 128 + self.stopping
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            self.cleanup_processes()
        finally:
            # Fresh task-owned state only, including a boot needing SIGKILL.
            # Credential cleanup must also run if a drain/signal syscall fails.
            for path in self.state.glob("secret-*.env"):
                if path.is_file() or path.is_symlink():
                    path.unlink()

    def cleanup_processes(self):
        deadline = time.monotonic() + self.timeout
        # On TERM during boot: interrupt/reap boot before drain, so bootstrap
        # cannot launch another API or overwrite PID files after cleanup starts.
        if self.boot is not None:
            terminate_group(self.boot, timeout=min(5, self.timeout / 3))
        processes = owned_pids(self.marker)
        for path in (self.api_pid, self.turn_pid):
            if path.exists() and checked_pid_file(path, self.marker) is None:
                # Only freshly allocated task-owned PID files may be removed.
                path.unlink()
        # A launch interrupted between spawn and PID-file write can leave an
        # orphan. Its unique marker still permits bounded, identity-safe cleanup.
        if self.api_pid.exists() or self.turn_pid.exists():
            self.env["MUSETALK_SUPERVISOR_IDENTITIES"] = json.dumps(processes)
            stop_env = {**self.env, "VAST_SERVER_CTL_LOAD_TURN_ENV": "0", "LINGUA_CONTROL_PLANE_ENV_FILE": "/dev/null"}
            stop = subprocess.Popen(["bash", str(self.root / "scripts/vast_server_ctl.sh"), "stop"],
                                    cwd=self.root, env=stop_env, start_new_session=True)
            try:
                stop.wait(timeout=max(0.1, deadline - time.monotonic() - 7))
            except subprocess.TimeoutExpired:
                print("Canonical drain exceeded image shutdown deadline; forcing owned cleanup", flush=True)
                terminate_group(stop, timeout=1)
        processes.update(owned_pids(self.marker))
        signal_owned(processes, self.marker, signal.SIGTERM)
        end = min(deadline - 1, time.monotonic() + 5)
        while time.monotonic() < end and any(identity(pid, self.marker) == ticks for pid, ticks in processes.items()):
            time.sleep(0.1)
        signal_owned(processes, self.marker, signal.SIGKILL)


def process_command(pid, action):
    marker = os.environ.get(OWNER_ENV, "")
    if not marker:
        return 1
    actual = identity(pid, marker)
    expected = json.loads(os.environ.get("MUSETALK_SUPERVISOR_IDENTITIES", "{}" )).get(str(pid), actual)
    if actual is None or actual != expected:
        return 1
    if action == "check":
        return 0
    if action not in {"TERM", "KILL"}:
        return 2
    signal_owned({pid: expected}, marker, getattr(signal, "SIG" + action))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "process":
        raise SystemExit(process_command(int(sys.argv[2]), sys.argv[3]))
    raise SystemExit(Supervisor(Path(os.environ.get("REPO_ROOT", "/opt/musetalk/app"))).run())
