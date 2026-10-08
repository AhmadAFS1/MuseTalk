#!/usr/bin/env python3
"""GPU-process isolation watchdog. Internal: invoke only within box_guard's session."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def descends_from(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            text = (Path("/proc") / str(pid) / "status").read_text()
        except FileNotFoundError:
            return False
        pid = next(int(line.split()[1]) for line in text.splitlines() if line.startswith("PPid:"))
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    # box_guard setsid+exec makes us the session/group leader. Never kill an operator shell group.
    holder = Path(os.environ.get("BOX_GUARD_LEASE_FILE", "/workspace/.gpu_lease") + ".holder")
    if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid() or not holder.is_file():
        p.error("watch.py must run as the isolated session leader under scripts/box_guard.sh")
    child = subprocess.Popen(command)
    with Path(args.out).open("w") as output:
        while child.poll() is None:
            try:
                raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
                pids = [int(x.strip()) for x in raw.splitlines() if x.strip()]
                foreign = [pid for pid in pids if not descends_from(pid, child.pid)]
                row = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "monotonic_s": time.monotonic(),
                       "gpu_pids": pids, "foreign_pids": foreign}
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                foreign = ["GPU observer failed"]
                row = {"error": type(exc).__name__}
            output.write(json.dumps(row) + "\n")
            output.flush()
            if foreign:
                print("INVALID: foreign GPU workload or observer failure", flush=True)
                # Terminate the owned workload and all workers; guard owns this isolated group.
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return 2
            time.sleep(1)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
