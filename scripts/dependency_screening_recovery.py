#!/usr/bin/env python3
"""Explicit build-only recovery after the terminal first screening receipt.

Reuses already hash-verified source downloads, not a new VM or duplicate fetch.
The historical failed build and its sources/logs are retained unchanged.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from dependency_artifacts import require
from fetch_dependency_wheels import sha256_file


def execute(deadline, phase="build"):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Matching Linux only")
    root = Path("/workspace/.dependency-optimization-20261015-a1")
    require(phase in {"build", "imports"}, "Unreviewed recovery phase")
    old = root/("screening-a2/receipt.json" if phase == "imports" else "screening-a1/receipt.json")
    previous = json.loads(old.read_text())
    expected = "Recovery step failed: pip-compiled-screen-a2" if phase == "imports" else "Screening step failed: build-wheelhouse"
    require(previous["state"] == "failed" and previous["events"][-1].get("detail") == expected, "Different failure; investigate before recovery")
    suffix = "a3" if phase == "imports" else "a2"
    control = root/("screening-"+suffix)
    require(not control.exists(), "Recovery already claimed; inspect its receipt, never repeat")
    control.mkdir(mode=0o700)
    tools = root/("tools-v3" if phase == "imports" else "tools-v2")
    stop_epoch = dt.datetime.fromisoformat(deadline).timestamp()-300
    events = []
    def event(stage, **fields):
        value = {"stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), **fields}
        events.append(value)
        with (control/"events.jsonl").open("a") as out:
            out.write(json.dumps(value)+"\n")
        print(json.dumps(value), flush=True)
    def run(name, args):
        remaining = int(stop_epoch-time.time())
        require(remaining > 300, "Lease reserve reached")
        event("STEP_STARTED", name=name)
        start = time.monotonic()
        with (control/(name+".log")).open("xb") as log:
            p = subprocess.Popen(["python3.10", "-u", "-B", *args], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = p.wait(timeout=min(remaining, 1200))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid, signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid, signal.SIGKILL);p.wait()
                raise ValueError("Recovery step exceeded deadline: "+name) from None
        event("STEP_FINISHED", name=name, seconds=time.monotonic()-start, exit_status=status)
        require(status == 0, "Recovery step failed: "+name)
    try:
        event("RECOVERY_STARTED", original_failed_receipt_sha256=sha256_file(old),
              source_cache="all 194 downloads preserved; build re-verifies before use",
              fix=("separate GPU-required Torch-TensorRT import from CUDA-hidden imports" if phase == "imports" else
                   "upstream NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1; no package/version/source modifications"))
        wheelhouse = root/"wheelhouse-build-a2"
        if phase == "build":
            run("build-wheelhouse-a2", [str(tools/"build_dependency_wheelhouse.py"), "--manifest", str(tools/"acquisition-manifest.json"),
                "--manifest-sha256", "74cbf88526684a18bef3c8e0e8db3add6472ae58d62992c5599543049201240a",
                "--source-cache", str(root/"source-artifacts-4"), "--output", str(wheelhouse), "--python", "python3.10"])
            run("fetch-uv", [str(tools/"fetch_dependency_tool.py"), "--output", str(root/"uv-0.13.1")])
        else:
            require(json.loads((wheelhouse/"receipt.json").read_text())["state"] == "wheels-built", "No accepted wheel build to reuse")
        uv = root/"uv-0.13.1/uv"
        for tool in (("uv", "pip") if phase == "imports" else ("pip", "uv")):
            args = [str(tools/"benchmark_dependency_install.py"), "--experiment-root", str(root), "--trial", tool+"-compiled-screen-"+suffix,
                    "--manifest", str(wheelhouse/"manifest.json"), "--manifest-sha256", sha256_file(wheelhouse/"manifest.json"),
                    "--wheels", str(wheelhouse/"wheels"), "--group", "full", "--installer", tool, "--compile-bytecode", "--python", "python3.10"]
            if tool == "uv":args += ["--uv", str(uv), "--uv-sha256", sha256_file(uv), "--link-mode", "copy"]
            run(tool+"-compiled-screen-"+suffix, args)
        event("SCREENING_COMPLETE", full_delivery_and_native_application_acceptance="PENDING")
        result = {"state": "complete", "events": events}
    except BaseException as error:
        event("RECOVERY_FAILED", error_type=type(error).__name__, detail=str(error) if type(error) is ValueError else "Inspect protected step log")
        result = {"state": "failed", "events": events}
    (control/"receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deadline-utc", required=True)
    result = execute(p.parse_args().deadline_utc)
    print(json.dumps({"result": result}), flush=True)
    raise SystemExit(0 if result["state"] == "complete" else 2)
