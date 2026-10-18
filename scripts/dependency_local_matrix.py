#!/usr/bin/env python3
"""Bounded repetitions/bytecode comparisons, with owned disposable-env retirement.

Preserve the original source env and the uv canary env. Never remove receipts,
locks, raw logs, downloaded artifacts or an environment used by a live API.
"""
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from dependency_artifacts import require
from fetch_dependency_wheels import sha256_file


def execute(deadline):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Matching Linux only")
    root = Path("/workspace/.dependency-optimization-20261015-a1")
    require(json.loads((root/"screening-a3/receipt.json").read_text())["state"] == "complete", "Screening job not terminal/passed")
    control = root/"local-matrix-a1"
    require(not control.exists(), "Matrix already claimed; inspect current receipt/process")
    control.mkdir(mode=0o700)
    uv = root/"uv-0.13.1/uv"
    tools = root/"tools-v3"
    house = root/"wheelhouse-build-a2"
    stop = dt.datetime.fromisoformat(deadline).timestamp()-300
    events = []
    def event(stage, **fields):
        row = {"stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), **fields}
        events.append(row)
        with (control/"events.jsonl").open("a") as out:out.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
    def run(name, args):
        remaining = int(stop-time.time())
        require(remaining > 300, "Lease reserve reached; no new trial")
        start = time.monotonic()
        event("STEP_STARTED", name=name)
        with (control/(name+".log")).open("xb") as log:
            p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:status = p.wait(timeout=min(remaining, 900))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid, signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid, signal.SIGKILL);p.wait()
                raise ValueError("Matrix trial timed out: "+name) from None
        event("STEP_FINISHED", name=name, seconds=time.monotonic()-start, exit_status=status)
        require(status == 0, "Matrix step failed: "+name)
    def retire(name, allow_failed=False):
        require(name.startswith(("pip-", "uv-")) and "/" not in name and name != "uv-compiled-screen-a3", "Not a disposable trial")
        trial = root/name
        owner = json.loads((trial/"owned-trial.json").read_text())
        receipt_path = trial/"receipt.json"
        receipt = json.loads(receipt_path.read_text())
        require(owner["uid"] == os.getuid() and owner["trial"] == name and not trial.is_symlink(), "Foreign trial")
        require(receipt["state"] == "microbenchmark-passed" or
                (allow_failed and name == "pip-compiled-screen-a2" and receipt.get("detail") == "Trial step failed: first-imports"), "No terminal accepted retirement state")
        retired = trial/"retirement.json"
        require(not retired.exists(), "Trial already retired")
        before = shutil.disk_usage(root).free
        removed = []
        for leaf in ("venv", "package-cache"):
            path = trial/leaf
            require(path.is_dir() and not path.is_symlink() and path.resolve().parent == trial.resolve(), "Unsafe trial child")
            shutil.rmtree(path)  # Exactly this generated terminal trial's child.
            removed.append(str(path))
        value = {"removed": removed, "receipt_sha256": sha256_file(receipt_path), "reclaimed_free_bytes": shutil.disk_usage(root).free-before,
                 "recovery": "rebuildable from retained exact wheels/locks; original source and canary environments preserved"}
        retired.write_text(json.dumps(value, indent=2)+"\n")
        event("DISPOSABLE_TRIAL_RETIRED", trial=name, **value)
    try:
        # These prior controllers are terminal before entry; none launched an API
        # from these two pip environments. The selected uv env stays recoverable.
        retire("pip-compiled-screen-a2", allow_failed=True)
        retire("pip-compiled-screen-a3")
        run("uv-canary-cache-clean", [str(uv), "cache", "clean", "--cache-dir", str(root/"uv-compiled-screen-a3/package-cache")])
        cases = [("pip", True, 2), ("uv", True, 2), ("uv", True, 3), ("pip", True, 3),
                 ("uv", False, 1), ("pip", False, 1)]
        for tool, compiled, number in cases:
            name = tool+("-compiled-rep-" if compiled else "-nocompile-rep-")+str(number)
            args = ["python3.10", "-u", "-B", str(tools/"benchmark_dependency_install.py"), "--experiment-root", str(root),
                    "--trial", name, "--manifest", str(house/"manifest.json"), "--manifest-sha256", sha256_file(house/"manifest.json"),
                    "--wheels", str(house/"wheels"), "--group", "full", "--installer", tool, "--python", "python3.10"]
            if compiled:args.append("--compile-bytecode")
            if tool == "uv":args += ["--uv", str(uv), "--uv-sha256", sha256_file(uv), "--link-mode", "copy"]
            run(name, args)
            # Keep the no-bytecode uv candidate for a later first-API/output
            # comparison. Other repeated installs are reproducible expendables.
            if tool == "uv" and not compiled:
                run("uv-nocompile-cache-clean", [str(uv), "cache", "clean", "--cache-dir", str(root/name/"package-cache")])
            else:retire(name)
        event("LOCAL_MATRIX_COMPLETE", scope="3 matched compiled samples per installer plus bytecode screens; remote delivery still pending")
        result = {"state": "complete", "events": events}
    except BaseException as error:
        event("MATRIX_FAILED", error_type=type(error).__name__, detail=str(error) if type(error) is ValueError else "Inspect protected step log")
        result = {"state": "failed", "events": events}
    (control/"receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    return result
