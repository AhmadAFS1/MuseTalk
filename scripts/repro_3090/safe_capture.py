"""Bounded preflight subprocesses; persist only fixed diagnostic vocabulary.

This is not a benchmark timer. Raw command arguments, environment values and
output never appear in CaptureFailure (including its string representation).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


OUTPUT_LIMIT_BYTES = 1024 * 1024
DIAGNOSTIC_TAIL_BYTES = 64 * 1024
_MESSAGES = {
    "CUDA_DRIVER_INITIALIZATION_FAILED": "CUDA driver initialization failed",
    "CUDA_DRIVER_TOO_OLD": "The NVIDIA driver is too old or insufficient for this CUDA runtime",
    "CUDA_DEVICES_UNREADABLE": "Torch-TensorRT could not read CUDA-capable devices",
    "NVML_INITIALIZATION_FAILED": "NVML initialization failed",
    "DRIVER_LIBRARY_VERSION_MISMATCH": "Driver/library version mismatch",
    "MISSING_MODULE": "A required Python module could not be imported",
    "AWS_ACCESS_DENIED": "AWS access was denied",
    "AWS_CREDENTIALS_UNAVAILABLE": "AWS credentials were unavailable or invalid",
    "AWS_BUCKET_POLICY_ABSENT": "The bucket has no bucket policy",
}
_NEEDLES = {
    "CUDA_DRIVER_INITIALIZATION_FAILED": (b"CUDA driver initialization failed",),
    "CUDA_DRIVER_TOO_OLD": (b"CUDA driver version is insufficient", b"The NVIDIA driver on your system is too old"),
    "CUDA_DEVICES_UNREADABLE": (b"Unable to read CUDA capable devices",),
    "NVML_INITIALIZATION_FAILED": (b"Failed to initialize NVML",),
    "DRIVER_LIBRARY_VERSION_MISMATCH": (b"Driver/library version mismatch",),
    "MISSING_MODULE": (b"ModuleNotFoundError: No module named",),
    "AWS_ACCESS_DENIED": (b"(AccessDenied)", b"(403)", b"(Forbidden)"),
    "AWS_CREDENTIALS_UNAVAILABLE": (b"Unable to locate credentials", b"(InvalidAccessKeyId)", b"(ExpiredToken)"),
    "AWS_BUCKET_POLICY_ABSENT": (b"(NoSuchBucketPolicy)",),
}
_STAGES = {"nvml_identity", "nvml_workloads", "runtime_import", "git_revision", "git_status", "s3_head", "operator_privacy_read", "unspecified"}


class CaptureFailure(subprocess.SubprocessError):
    """Safe to serialize; does not retain subprocess output or command arguments."""

    def __init__(self, record):
        self.record = record
        super().__init__(f"{record['stage']}: {record['failure']}")


def _signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
        return "sent"
    except ProcessLookupError:
        return "group_absent"
    except OSError:
        # A container may forbid signal delivery; macOS may also report EPERM
        # for an already-dead, unreaped group. Preserve the bounded wait and do
        # not turn a failed signal into a claim that cleanup succeeded.
        return "signal_delivery_failed"


def capture(command, *, cwd, env=None, stage="unspecified", timeout_s=30,
            output_limit_bytes=OUTPUT_LIMIT_BYTES, terminate_grace_s=7, reap_grace_s=1):
    """Drain both pipes without unbounded memory or an unbounded child wait.

    POSIX-only, like the Linux harness. A new session limits signals to this
    command's group. box_guard receives TERM and owns cleanup of its separately
    grouped GPU child; allow its five-second TERM grace before killing the guard.
    A kernel-uninterruptible process cannot be made killable here: such a timeout
    records cleanup as unconfirmed and must not be treated as permission to retry.
    """
    if stage not in _STAGES:
        raise ValueError("unknown capture stage")
    if (not math.isfinite(timeout_s) or timeout_s <= 0 or output_limit_bytes < 1
            or terminate_grace_s < 0 or reap_grace_s <= 0):
        raise ValueError("invalid capture bounds")
    started = time.monotonic()
    basename = Path(str(command[0])).name
    # Python executable paths vary by venv. Never echo an arbitrary executable name.
    executable = basename if basename in {"nvidia-smi", "git", "aws", "bash", "python", "python3"} else "other"
    record = {"schema": "repro_3090_preflight_failure_v1", "stage": stage, "executable": executable,
              "timeout_s": timeout_s, "output_limit_bytes_per_stream": output_limit_bytes,
              "diagnostics": [], "raw_output_persisted": False}
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        record.update(failure="EXECUTABLE_UNAVAILABLE" if isinstance(exc, FileNotFoundError) else "EXECUTION_FAILED",
                      elapsed_s=round(time.monotonic() - started, 3), returncode=None, cleanup="not_started")
        raise CaptureFailure(record) from None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    tails = {"stdout": b"", "stderr": b""}
    totals = {"stdout": 0, "stderr": 0}
    recognized = set()
    abort_at = None
    failure = None
    killed = False
    signals = {}
    for name in ("stdout", "stderr"):
        stream = getattr(process, name)
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    try:
        while True:
            now = time.monotonic()
            if failure is None and now - started >= timeout_s:
                failure = "TIMEOUT"
            if failure is not None and abort_at is None:
                abort_at = now
                signals["TERM"] = _signal_group(process, signal.SIGTERM)
            if abort_at is not None and now - abort_at >= terminate_grace_s and not killed:
                signals["KILL"] = _signal_group(process, signal.SIGKILL)
                killed = True
            if abort_at is not None and now - abort_at >= terminate_grace_s + reap_grace_s:
                break
            # Do not reap the session leader while an inherited pipe remains
            # open. Keeping its PID reserved prevents a later timeout signal
            # from targeting a recycled process-group ID.
            if not selector.get_map() and process.poll() is not None:
                break
            for key, _ in selector.select(timeout=0.05):
                try:
                    chunk = os.read(key.fileobj.fileno(), 16384)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                totals[name] += len(chunk)
                window = tails[name][-512:] + chunk
                for code, needles in _NEEDLES.items():
                    if any(needle in window for needle in needles):
                        recognized.add(code)
                tails[name] = (tails[name] + chunk)[-DIAGNOSTIC_TAIL_BYTES:]
                if name == "stdout":
                    stdout.extend(chunk[:max(0, output_limit_bytes - len(stdout))])
                if totals[name] > output_limit_bytes and failure is None:
                    failure = "OUTPUT_LIMIT"
        returncode = process.poll()
        if failure is None and returncode == 0:
            return stdout.decode("utf-8", errors="replace").strip()
        record.update(failure=failure or "NONZERO_EXIT", elapsed_s=round(time.monotonic() - started, 3),
                      returncode=returncode, output_bytes_observed=totals,
                      diagnostics=[{"code": code, "message": _MESSAGES[code]} for code in sorted(recognized)],
                      cleanup="leader_reaped" if returncode is not None else "unconfirmed_kernel_or_child_state",
                      termination_requested=abort_at is not None, kill_requested=killed,
                      signal_attempts=signals,
                      process_state_check_required_before_retry=abort_at is not None,
                      descendant_cleanup="not_certified" if abort_at is not None else "not_applicable")
        raise CaptureFailure(record)
    except BaseException as exc:
        if not isinstance(exc, CaptureFailure):
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=terminate_grace_s)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=reap_grace_s)
                except subprocess.TimeoutExpired:
                    pass
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
