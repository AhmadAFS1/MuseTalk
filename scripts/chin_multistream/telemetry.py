"""Light telemetry: per-process CPU seconds from /proc, system CPU, MemAvailable and nvidia-smi samplers."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")


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
