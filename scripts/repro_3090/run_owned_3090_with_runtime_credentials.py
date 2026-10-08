#!/usr/bin/env python3
"""Operator-only credential bridge for one identity-verified owned experiment.

Plan-only unless --execute is present. This is not a sandbox for arbitrary code:
the operator must authorize the child command and its file/network effects.
Never use commands that dump credentials or persist their environment. Child
stdout/stderr are discarded; request explicit, non-secret result files instead.

The protected EC2 snapshot is read through SSH into operator process memory.
Only after the fixed worker proves its hostname/GPU identity on the same SSH
connection are bootstrap credentials sent on stdin. The canonical worker secret
reader obtains runtime credentials in memory; only a narrow runtime allowlist
enters the child environment. No credential/env file is created by this helper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import os
from pathlib import Path
import pwd
import re
import resource
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
import contextlib

INSTANCE = "54798270"
LABEL = "musetalk-r5-3090-dev-20261008"
EC2_ALIAS = "my-ec2"
WORKER_ALIAS = "musetalk-3090-build-54798270"
EC2_PYTHON = "/home/ec2-user/lingua/venv/bin/python"
WORKER_PYTHON = "/workspace/.venvs/musetalk_trt_stagewise/bin/python"
WORKER_ROOT = "/workspace/MuseTalk"
WORKER_HOSTNAME = "a830e00ce20c"
GPU_UUID = "GPU-5640f670-debe-ec22-1cfb-4b1f63bc1d53"
SNAPSHOT = "/home/ec2-user/.local/state/musetalk-r5-20261008-preflight-v4/template-408714.before.json"
LEDGER = "/home/ec2-user/.local/state/musetalk-r5-3090-dev-20261008/startup-ledger.json"
ONSTART_SHA256 = "14ccb8011f7b5b2cb10e77551ea32826fd729f24eef2570cea9493a086eb8c39"
BOOTSTRAP_SHA256 = "de8b1ccf14d1aa282d8855ae46ae8ba84a13fd9659a3b6c204626f747379d433"
SECRET_ARN = "arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-Dof4b8"
DEADLINE = "2026-10-08T19:00:00+00:00"
MAX_RECORD = 32768
BOOTSTRAP_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                  "AWS_DEFAULT_REGION", "MUSETALK_AWS_SECRET_ID"}
PAYLOAD_KEYS = {"AVATAR_S3_BUCKET", "AVATAR_S3_ENABLED", "AVATAR_S3_PREFIX", "AVATAR_S3_REGION",
                "AWS_ACCESS_KEY_ID", "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN"}
CHILD_KEYS = PAYLOAD_KEYS | {"TRT_ARTIFACT_S3_BUCKET", "TRT_ARTIFACT_S3_REGION"}
OWNED_TARGET_VALIDATED = False


def apply_owned_target(document):
    """Bind a new development target; never broaden secret/key/source access.

    This non-secret descriptor is not spending authorization. The EC2 ledger
    must separately prove the paid create and exactly the installed deadline.
    Release/production workers are deliberately excluded from this repair tool.
    """
    require(isinstance(document, dict) and set(document) == {
        "instance_id", "label", "worker_alias", "worker_hostname", "gpu_uuid",
        "ledger", "deadline_utc"})
    require(all(isinstance(v, str) and 0 < len(v) <= 512 and "\0" not in v
                for v in document.values()))
    instance = document["instance_id"]
    label = document["label"]
    require(re.fullmatch(r"[1-9][0-9]{6,11}", instance) is not None and instance != "51074906")
    require(re.fullmatch(r"musetalk-r5-3090-dev-[A-Za-z0-9_.-]{4,80}", label) is not None)
    require(document["worker_alias"] == "musetalk-3090-build-" + instance)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,63}", document["worker_hostname"]) is not None)
    require(re.fullmatch(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", document["gpu_uuid"]) is not None)
    require(document["ledger"] == f"/home/ec2-user/.local/state/{label}/startup-ledger.json")
    deadline = dt.datetime.fromisoformat(document["deadline_utc"].replace("Z", "+00:00"))
    require(deadline.tzinfo is not None and deadline.utcoffset() == dt.timedelta(0))
    require(120 < (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 24 * 3600)
    globals().update(INSTANCE=instance, LABEL=label, WORKER_ALIAS=document["worker_alias"],
                     WORKER_HOSTNAME=document["worker_hostname"], GPU_UUID=document["gpu_uuid"],
                     LEDGER=document["ledger"], DEADLINE=deadline.isoformat(), OWNED_TARGET_VALIDATED=True)


def require(condition):
    if not condition:
        raise ValueError("Credential bridge precondition failed; values suppressed")


def seconds_remaining():
    return (dt.datetime.fromisoformat(DEADLINE) - dt.datetime.now(dt.timezone.utc)).total_seconds()


def validate_bootstrap(values):
    require(isinstance(values, dict) and set(values) <= BOOTSTRAP_KEYS)
    require(all(isinstance(v, str) and v and "\0" not in v and len(v) <= 8192 for v in values.values()))
    require(values.get("MUSETALK_AWS_SECRET_ID") == SECRET_ARN)
    require(values.get("AWS_DEFAULT_REGION") == "us-east-1")
    require(bool(values.get("AWS_ACCESS_KEY_ID")) and bool(values.get("AWS_SECRET_ACCESS_KEY")))
    return values


def parse_snapshot(template):
    require(template.get("id") == 408714)
    script = template.get("onstart", "")
    require(hashlib.sha256(script.encode()).hexdigest() == ONSTART_SHA256)
    values = {}
    for line in script.splitlines():
        if line.strip().startswith("export "):
            for token in shlex.split(line.strip()[7:]):
                if "=" in token:
                    key, value = token.split("=", 1)
                    if key in BOOTSTRAP_KEYS:
                        # Reviewed template repeats the same secret ARN before
                        # canonical startup. Conflicting assignments still fail.
                        require(key not in values or values[key] == value)
                        values[key] = value
    values.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    return validate_bootstrap(values)


def validate_ledger(ledger):
    require(isinstance(ledger, dict))
    require(str(ledger.get("instance_id")) == INSTANCE and ledger.get("label") == LABEL)
    require(ledger.get("state") in {"created", "reconciled"})
    if OWNED_TARGET_VALIDATED:
        require(ledger.get("resource_deadline_utc") == DEADLINE)
        require(ledger.get("control_plane_url") == "http://127.0.0.1:8000")
        require(any(e.get("name") == "request_started" for e in ledger.get("events", [])))


def ec2_export():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    require(seconds_remaining() > 60)
    snapshot = Path(SNAPSHOT)
    require(not snapshot.is_symlink() and not snapshot.parent.is_symlink())
    uid = pwd.getpwnam("ec2-user").pw_uid
    require(snapshot.stat().st_uid in {0, uid} and stat.S_IMODE(snapshot.stat().st_mode) == 0o600)
    require(snapshot.parent.stat().st_uid in {0, uid} and stat.S_IMODE(snapshot.parent.stat().st_mode) == 0o700)
    require(snapshot.stat().st_size < 1024 * 1024)
    ledger_path = Path(LEDGER)
    require(not ledger_path.is_symlink() and not ledger_path.parent.is_symlink())
    require(ledger_path.stat().st_uid in {0, uid} and stat.S_IMODE(ledger_path.stat().st_mode) == 0o600)
    require(ledger_path.parent.stat().st_uid in {0, uid} and stat.S_IMODE(ledger_path.parent.stat().st_mode) == 0o700)
    require(ledger_path.stat().st_size < 1024 * 1024)
    ledger = json.loads(ledger_path.read_text())
    validate_ledger(ledger)
    values = parse_snapshot(json.loads(snapshot.read_text()))
    # This stdout is captured by the operator helper, never attached to a PTY.
    sys.stdout.write(json.dumps(values))
    sys.stdout.flush()


def runtime_child_env(module, payload):
    require(isinstance(payload, dict) and set(payload) <= PAYLOAD_KEYS)
    require(payload.get("AWS_ACCESS_KEY_ID") and payload.get("AWS_SECRET_ACCESS_KEY"))
    require(payload.get("AVATAR_S3_BUCKET") == "lingua-musetalk-s3-storage")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exports = module._exports_from_payload(payload)
    require(set(exports) <= CHILD_KEYS)
    require(all(isinstance(v, str) and "\0" not in v and len(v) <= 8192 for v in exports.values()))
    # Deliberately do not inherit credentials, registration, Python loader hooks,
    # proxy overrides, or arbitrary runtime-secret operational settings.
    env = {k: os.environ[k] for k in ("HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH",
                                    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                                    "NVIDIA_DRIVER_CAPABILITIES") if k in os.environ}
    env.update(exports)
    env.update(PATH="/workspace/.venvs/musetalk_trt_stagewise/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               AWS_EC2_METADATA_DISABLED="true", AWS_CONFIG_FILE="/dev/null",
               AWS_SHARED_CREDENTIALS_FILE="/dev/null", AWS_IGNORE_CONFIGURED_ENDPOINT_URLS="true",
               LINGUA_CONTROL_PLANE_ENABLED="0", LINGUA_WORKER_CALLBACK_REQUIRED="0")
    return env


def terminate_owned_process_group(child):
    """Give this already-created session's leader five seconds for restoration.

    No PID discovery, foreign groups, sudo, daemonization, or target-policy
    changes. Repeated termination signals cannot restart/interrupt this grace.
    Remaining members of this exact owned group are killed after its leader's
    exit as well, so a successful parent exit does not spare background members.
    """
    # poll()/wait() reap the leader and release its PID. Never call either until
    # AFTER the last group signal: its unreaped PID reserves this exact PGID.
    if child.returncode is not None:
        return
    require(hasattr(os, 'waitid') and hasattr(os, 'WNOWAIT'))
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    try:
        os.waitid(os.P_PID, child.pid, options)
    except ChildProcessError:
        return  # No unreaped child remains; ownership cannot be proved.
    previous = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if os.waitid(os.P_PID, child.pid, options) is not None:
                break
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=1)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def worker_run():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    require(os.geteuid() == 0 and socket.gethostname() == WORKER_HOSTNAME)
    require(seconds_remaining() > 60)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                  stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
    require(gpu == GPU_UUID)
    bootstrap = Path(WORKER_ROOT) / "scripts/bootstrap_aws_secrets.py"
    require(not bootstrap.is_symlink() and hashlib.sha256(bootstrap.read_bytes()).hexdigest() == BOOTSTRAP_SHA256)
    # Same-connection identity handshake happens BEFORE receiving any secret.
    print(json.dumps({"phase": "ready", "instance_id": INSTANCE, "nonce": NONCE}), flush=True)
    raw = sys.stdin.buffer.readline(MAX_RECORD + 1)
    require(raw.endswith(b"\n") and len(raw) <= MAX_RECORD)
    request = json.loads(raw)
    require(set(request) == {"credentials", "command", "timeout_seconds", "nonce"} and request["nonce"] == NONCE)
    credentials = validate_bootstrap(request["credentials"])
    command = request["command"]
    require(isinstance(command, list) and 1 <= len(command) <= 128)
    require(all(isinstance(v, str) and "\0" not in v and len(v) < 8192 for v in command))
    require(command[0].startswith("/") and request["timeout_seconds"] <= seconds_remaining() - 30)
    require(type(request["timeout_seconds"]) is int and 1 <= request["timeout_seconds"] <= 7200)
    for key in tuple(os.environ):
        if key.startswith(("AWS_", "MUSETALK_AWS_", "BOTO")) or key.upper().endswith("_PROXY"):
            os.environ.pop(key, None)
    os.environ.update(credentials)
    os.environ.update(AWS_EC2_METADATA_DISABLED="true", AWS_CONFIG_FILE="/dev/null",
                      AWS_SHARED_CREDENTIALS_FILE="/dev/null", AWS_IGNORE_CONFIGURED_ENDPOINT_URLS="true")
    logging.disable(logging.CRITICAL)
    spec = importlib.util.spec_from_file_location("approved_runtime_bootstrap", bootstrap)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
        payload = module._read_secret_payload(SECRET_ARN, "us-east-1")
        env = runtime_child_env(module, payload)
    # The SDK path above intentionally does NOT invoke _write_export_file/main.
    for key in BOOTSTRAP_KEYS:
        os.environ.pop(key, None)
    credentials.clear()
    request.pop("credentials")
    raw = b""
    require(seconds_remaining() > request["timeout_seconds"] + 30)
    child = subprocess.Popen(command, cwd=WORKER_ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    def stop(_sig, _frame):
        terminate_owned_process_group(child)
        raise SystemExit(143)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        code = child.wait(timeout=request["timeout_seconds"])
        result = {"phase": "finished", "instance_id": INSTANCE, "returncode": code,
                  "child_output": "suppressed", "credential_files_written_by_helper": False}
    except subprocess.TimeoutExpired:
        terminate_owned_process_group(child)
        result = {"phase": "timed_out", "instance_id": INSTANCE, "returncode": 124,
                  "child_output": "suppressed", "credential_files_written_by_helper": False}
    print(json.dumps(result), flush=True)


def remote_code(function, nonce=""):
    imports = "import contextlib, datetime as dt, hashlib, importlib.util, io, json, logging, os, pwd, resource, shlex, signal, socket, stat, subprocess, sys, time\nfrom pathlib import Path\n"
    constants = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (str, int, set))}
    constants["NONCE"] = nonce
    definitions = [require, seconds_remaining, validate_bootstrap, parse_snapshot, validate_ledger,
                   runtime_child_env, terminate_owned_process_group, function]
    return imports + "\n".join(f"{k}={v!r}" for k, v in constants.items()) + "\n" + "\n".join(
        inspect.getsource(f) for f in definitions) + "\ntry:\n " + function.__name__ + "()\nexcept Exception:\n sys.exit(2)\n"


def ssh_arguments(alias, executable, code, sudo=False):
    args = (["sudo", "-n"] if sudo else []) + [executable, "-B", "-c", code]
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=10", alias, shlex.join(args)]


def read_record(stream, timeout):
    deadline = time.monotonic() + timeout
    data = b""
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        require(remaining > 0 and select.select([stream], [], [], remaining)[0])
        chunk = os.read(stream.fileno(), MAX_RECORD + 1 - len(data))
        require(chunk and len(data) + len(chunk) <= MAX_RECORD)
        data += chunk
    require(data.endswith(b"\n") and data.count(b"\n") == 1)
    return json.loads(data)


def execute(command, timeout):
    require(seconds_remaining() > timeout + 60)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    nonce = uuid.uuid4().hex
    worker = subprocess.Popen(ssh_arguments(WORKER_ALIAS, WORKER_PYTHON, remote_code(worker_run, nonce)),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    stage = "worker_identity_handshake"
    try:
        ready = read_record(worker.stdout, 30)
        require(ready == {"phase": "ready", "instance_id": INSTANCE, "nonce": nonce})
        stage = "protected_ec2_snapshot"
        extracted = subprocess.run(ssh_arguments(EC2_ALIAS, EC2_PYTHON, remote_code(ec2_export), sudo=True),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   timeout=30, check=False)
        require(extracted.returncode == 0 and len(extracted.stdout) <= MAX_RECORD)
        values = validate_bootstrap(json.loads(extracted.stdout))
        request = {"credentials": values, "command": command, "timeout_seconds": timeout, "nonce": nonce}
        encoded = json.dumps(request).encode() + b"\n"
        require(len(encoded) <= MAX_RECORD)
        pending = memoryview(encoded)
        while pending:
            written = worker.stdin.write(pending)
            require(isinstance(written, int) and written > 0)
            pending = pending[written:]
        worker.stdin.close()
        values.clear()
        pending = None
        encoded = b""
        extracted = None
        stage = "runtime_secret_and_authorized_command"
        result = read_record(worker.stdout, timeout + 120)
        require(set(result) == {"phase", "instance_id", "returncode", "child_output", "credential_files_written_by_helper"})
        require(result["phase"] in {"finished", "timed_out"} and result["instance_id"] == INSTANCE)
        require(type(result["returncode"]) is int and result["child_output"] == "suppressed"
                and result["credential_files_written_by_helper"] is False)
        require(worker.wait(timeout=10) == 0)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["returncode"] == 0 else 1
    except Exception:
        print(json.dumps({"phase": "bridge_failed", "stage": stage, "instance_id": INSTANCE,
                          "details": "suppressed"}), file=sys.stderr)
        raise
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--owned-target-json", help="Non-secret descriptor for a separately provisioned, bounded development target")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.owned_target_json:
        target = Path(args.owned_target_json)
        require(not target.is_symlink() and target.is_file() and target.stat().st_size <= MAX_RECORD)
        apply_owned_target(json.loads(target.read_text()))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(command and command[0].startswith("/") and 1 <= args.timeout_seconds <= 7200)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "instance_id": INSTANCE, "destination": WORKER_ALIAS,
                          "credential_source": SNAPSHOT, "runtime_secret_reference": SECRET_ARN,
                          "deadline_utc": DEADLINE, "command_argument_count": len(command),
                          "child_output": "suppressed", "cloud_or_worker_actions": False}, sort_keys=True))
        return 0
    return execute(command, args.timeout_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Owned-worker credential command failed; details and values suppressed", file=sys.stderr)
        raise SystemExit(2)
