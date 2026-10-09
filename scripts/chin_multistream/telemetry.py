"""Light telemetry: per-process CPU seconds from /proc, system CPU, MemAvailable and nvidia-smi samplers."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

CLK = os.sysconf("SC_CLK_TCK")


def cgroup_cpu_snapshot(membership="/proc/self/cgroup", mountinfo="/proc/self/mountinfo"):
    """Read this process's cgroup-v2 CPU counters, never assume host-root stats.

    Missing/ambiguous/unsupported data is explicitly unavailable. No mutation,
    GPU operation, sampling thread or benchmark acceptance decision occurs here.
    """
    try:
        groups = [line.split(":", 2)[2] for line in Path(membership).read_text().splitlines()
                  if line.startswith("0::")]
        mounts = []
        for line in Path(mountinfo).read_text().splitlines():
            fields = line.split()
            separator = fields.index("-")
            if fields[separator + 1] == "cgroup2":
                mounts.append((fields[3], fields[4]))
        if len(groups) != 1 or len(mounts) != 1:
            raise ValueError("ambiguous_or_unsupported_cgroup")
        group = PurePosixPath(groups[0])
        mount_root, mount_point = map(PurePosixPath, mounts[0])
        if any(not p.is_absolute() or ".." in p.parts or "\\" in str(p)
               for p in (group, mount_root, mount_point)):
            raise ValueError("unresolved_cgroup_path")
        # In a cgroup namespace / is its own root, even when mountinfo retains
        # the host-side mount root. Otherwise require an exact descendant.
        relative = PurePosixPath(".") if group == PurePosixPath("/") else group.relative_to(mount_root)
        directory = Path(str(mount_point / relative))
        counters = {}
        allowed = {"usage_usec", "user_usec", "system_usec", "nr_periods", "nr_throttled", "throttled_usec"}
        for line in (directory / "cpu.stat").read_text().splitlines():
            key, value = line.split()
            if key in allowed:
                if key in counters or not value.isdigit():
                    raise ValueError("malformed_cpu_counter")
                counters[key] = int(value)
        if not {"usage_usec", "nr_periods", "nr_throttled", "throttled_usec"} <= counters.keys():
            raise ValueError("missing_cpu_counters")
        return {"status": "AVAILABLE", "source": str(directory / "cpu.stat"),
                "sample_monotonic_s": time.monotonic(), "counters": counters}
    except (OSError, ValueError, IndexError) as exc:
        return {"status": "UNAVAILABLE", "reason": type(exc).__name__}


def cgroup_cpu_interval(before, after):
    """Counter delta around a window, including final worker message collection.

    throttled_usec is a kernel cgroup counter, not lost wall time or proof of
    causality. Keep it distinct from the composed-frame FPS denominator.
    """
    out = {"schema": "cgroup_cpu_interval_v1", "before": before, "after": after,
           "scope": "Parent cgroup, includes all descendants; interval includes worker report collection"}
    if before.get("status") != "AVAILABLE" or after.get("status") != "AVAILABLE":
        return {**out, "status": "UNAVAILABLE"}
    if before["source"] != after["source"] or before["counters"].keys() != after["counters"].keys():
        return {**out, "status": "INVALID", "reason": "source_or_counter_set_changed"}
    elapsed = after["sample_monotonic_s"] - before["sample_monotonic_s"]
    delta = {key: after["counters"][key] - value for key, value in before["counters"].items()}
    if elapsed <= 0 or any(value < 0 for value in delta.values()):
        return {**out, "status": "INVALID", "reason": "counter_reset_or_nonpositive_interval"}
    periods = delta["nr_periods"]
    if delta["nr_throttled"] > periods:
        return {**out, "status": "INVALID", "reason": "inconsistent_period_counters"}
    return {**out, "status": "AVAILABLE", "sample_elapsed_s": elapsed, "counters_delta": delta,
            "throttled_period_fraction": delta["nr_throttled"] / periods if periods else None,
            "throttled_time_s": delta["throttled_usec"] / 1_000_000}


def proc_cpu_s(pid="self") -> float:
    """utime+stime of one process (all its threads), seconds."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return float("nan")
    fields = text[text.rindex(")") + 2:].split()
    return (int(fields[11]) + int(fields[12])) / CLK


def system_cpu():
    """(busy_jiffies, total_jiffies) from /proc/stat."""
    vals = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals[:8])
    return total - idle, total


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return float("nan")


def rss_mib(pid="self") -> dict:
    out = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key = line.split(":")[0]
            if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
                out[key] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return out


class MemSampler(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            self.samples.append(mem_available_gb())
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)
        return dict(min_gb=min(self.samples) if self.samples else None,
                    max_gb=max(self.samples) if self.samples else None, n=len(self.samples))


class SmiSampler:
    """nvidia-smi --query-gpu at a fixed interval while a timed repeat runs."""

    FIELDS = "utilization.gpu,clocks.sm,power.draw,memory.used,temperature.gpu"

    def __init__(self, interval_ms=500):
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", f"-lms={interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.lines = []
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.strip())

    def stop(self, tail_samples=None):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.thread.join(timeout=5)
        rows = []
        for line in self.lines:
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                continue
        if not rows:
            return dict(n=0)
        if tail_samples is not None:
            rows = rows[-tail_samples:]
        import statistics

        cols = list(zip(*rows))
        names = self.FIELDS.split(",")
        return dict(n=len(rows), **{names[k]: dict(median=statistics.median(c), min=min(c), max=max(c))
                                    for k, c in enumerate(cols)})
