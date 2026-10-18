#!/usr/bin/env python3
"""One durable Linux screening job; wait for baseline, then isolated comparisons.

No create/destroy, S3 upload, production mutations, environment replacement or
default changes. The EC2 lease/expiry owns rental lifetime independently.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

from dependency_artifacts import require
from fetch_dependency_wheels import sha256_file

MANIFEST_SHA = "74cbf88526684a18bef3c8e0e8db3add6472ae58d62992c5599543049201240a"


def execute(root, deadline):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux screening only")
    require(root == Path("/workspace/.dependency-optimization-20261015-a1"), "Exact owned experiment root required")
    control = root / "screening-a1"
    require(not control.exists(), "Screening already claimed; inspect its existing job")
    control.mkdir(mode=0o700)
    tools = root / "tools"
    stop_epoch = dt.datetime.fromisoformat(deadline).timestamp() - 300
    require(time.time() < stop_epoch, "Lease too short to start")
    events = []
    def event(stage, **fields):
        row = {"stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), **fields}
        events.append(row)
        with (control / "events.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    def run(name, args):
        remaining = int(stop_epoch-time.time())
        require(remaining > 300, "Lease reserve reached; no new benchmark started")
        event("STEP_STARTED", name=name)
        before = time.monotonic()
        with (control / (name + ".log")).open("xb") as log:
            p = subprocess.Popen(["python3.10", "-u", "-B", *args], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=min(remaining, 1200))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                raise ValueError("Screening step exceeded deadline: " + name) from None
        event("STEP_FINISHED", name=name, seconds=time.monotonic()-before, exit_status=code)
        require(code == 0, "Screening step failed: " + name)
    try:
        event("WAITING_FOR_BASELINE", deadline_utc=deadline)
        wait_until = min(stop_epoch-1800, time.time()+2400)
        while time.time() < wait_until:
            healthy = False
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                    health = json.loads(response.read(256000))
                healthy = health.get("ok") is True and health.get("service") == "musetalk"
            except Exception:
                pass
            log = Path("/workspace/bootstrap.log")
            complete = log.is_file() and "VAST_ONSTART COMPLETE:" in log.read_text(errors="replace")[-80000:]
            if healthy and complete:
                break
            time.sleep(5)
        else:
            raise ValueError("Source baseline has not completed; inspect logs, never restart it automatically")
        event("SOURCE_BASELINE_HEALTHY", competing_work="baseline API remains idle; no downloads overlapped with baseline")
        run("fetch-4", [str(tools/"fetch_dependency_wheels.py"), "--manifest", str(tools/"acquisition-manifest.json"),
            "--manifest-sha256", MANIFEST_SHA, "--cache", str(root/"source-artifacts-4"), "--group", "full", "--workers", "4",
            "--repo-root", "/workspace/MuseTalk", "--receipt", str(control/"fetch-4-receipt.json")])
        run("build-wheelhouse", [str(tools/"build_dependency_wheelhouse.py"), "--manifest", str(tools/"acquisition-manifest.json"),
            "--manifest-sha256", MANIFEST_SHA, "--source-cache", str(root/"source-artifacts-4"), "--output", str(root/"wheelhouse-build"), "--python", "python3.10"])
        run("fetch-uv", [str(tools/"fetch_dependency_tool.py"), "--output", str(root/"uv-0.13.1")])
        wheelhouse = root/"wheelhouse-build"
        binary = root/"uv-0.13.1/uv"
        for tool in ("pip", "uv"):
            args = [str(tools/"benchmark_dependency_install.py"), "--experiment-root", str(root),
                "--trial", tool+"-compiled-screen-1", "--manifest", str(wheelhouse/"manifest.json"),
                "--manifest-sha256", sha256_file(wheelhouse/"manifest.json"), "--wheels", str(wheelhouse/"wheels"),
                "--group", "full", "--installer", tool, "--compile-bytecode", "--python", "python3.10"]
            if tool == "uv":
                args += ["--uv", str(binary), "--uv-sha256", sha256_file(binary), "--link-mode", "copy"]
            run(tool+"-compiled-screen-1", args)
        event("SCREENING_COMPLETE", scope="two local-wheel microbenchmarks only; repeated full delivery/native acceptance pending")
        result = {"state": "complete", "events": events, "native_application_acceptance": "PENDING"}
    except BaseException as error:
        event("SCREENING_FAILED", error_type=type(error).__name__,
              detail=str(error) if type(error) is ValueError else "Inspect private step log")
        result = {"state": "failed", "events": events, "native_application_acceptance": "PENDING"}
    (control/"receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--deadline-utc", required=True)
    a = p.parse_args()
    result = execute(a.root, a.deadline_utc)
    print(json.dumps({"result": result}), flush=True)
    raise SystemExit(0 if result["state"] == "complete" else 2)
