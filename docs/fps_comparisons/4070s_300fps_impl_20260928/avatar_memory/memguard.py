"""Shared-box RAM guard + /proc memory readers for the avatar-memory harnesses.

MemAvailable is min(/proc/meminfo MemAvailable, cgroup estimate), the same
definition scripts/box_guard.sh uses. Every harness here:
  * raises its own oom_score_adj to 1000 (the kernel kills it first);
  * waits before each load until MemAvailable >= floor + the expected cost;
  * runs a 4 Hz watchdog that hard-exits (code 86) below the kill floor.
"""
import os
import threading
import time

GB = 1024 ** 3
MB = 1024 ** 2


def _meminfo_kb(key: str) -> int:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return 0


def _cgroup_avail_bytes():
    try:
        maximum = open("/sys/fs/cgroup/memory.max").read().strip()
        current = int(open("/sys/fs/cgroup/memory.current").read().strip())
    except OSError:
        return None
    if maximum == "max":
        return None
    reclaimable = 0
    with open("/sys/fs/cgroup/memory.stat") as handle:
        for line in handle:
            key, value = line.split()
            if key in ("active_file", "inactive_file", "slab_reclaimable"):
                reclaimable += int(value)
    return int(maximum) - current + reclaimable


def mem_available_bytes() -> int:
    meminfo = _meminfo_kb("MemAvailable") * 1024
    cgroup = _cgroup_avail_bytes()
    return min(meminfo, cgroup) if cgroup is not None else meminfo


def mem_available_gb() -> float:
    return mem_available_bytes() / GB


def set_oom_score_adj(value: int = 1000) -> None:
    try:
        with open("/proc/self/oom_score_adj", "w") as handle:
            handle.write(str(value))
    except OSError:
        pass


def wait_for_headroom(need_gb: float, floor_gb: float, timeout_s: float = 900, poll_s: float = 5) -> bool:
    """Block until MemAvailable >= floor + need. False on timeout."""
    deadline = time.time() + timeout_s
    warned = False
    while True:
        avail = mem_available_gb()
        if avail >= floor_gb + need_gb:
            return True
        if time.time() >= deadline:
            return False
        if not warned:
            print(f"[memguard] waiting: MemAvailable {avail:.2f} GB < {floor_gb:.2f} + {need_gb:.2f} GB", flush=True)
            warned = True
        time.sleep(poll_s)


class Watchdog:
    def __init__(self, kill_below_gb: float):
        self.kill_below_gb = kill_below_gb
        self.min_seen_gb = mem_available_gb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="memguard")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(0.25):
            avail = mem_available_gb()
            self.min_seen_gb = min(self.min_seen_gb, avail)
            if avail < self.kill_below_gb:
                print(f"[memguard] WATCHDOG: MemAvailable {avail:.2f} GB < {self.kill_below_gb} GB; exiting",
                      flush=True)
                os._exit(86)


def smaps_rollup(pid="self") -> dict:
    """RSS / PSS / USS / anon in bytes from /proc/<pid>/smaps_rollup."""
    fields = {}
    with open(f"/proc/{pid}/smaps_rollup") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == "kB":
                fields[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "rss": fields.get("Rss", 0),
        "pss": fields.get("Pss", 0),
        "uss": fields.get("Private_Clean", 0) + fields.get("Private_Dirty", 0),
        "anon": fields.get("Anonymous", 0),
    }


def process_cpu_seconds() -> float:
    times = os.times()
    return times.user + times.system


def force_cpu_torch_load() -> None:
    """Harness-only: prepared latents.pt files hold CUDA tensors; with CUDA hidden,
    map them to the CPU. The server loads them onto the GPU (VRAM, not host RAM)."""
    import torch

    original = torch.load
    if getattr(original, "_memguard_cpu", False):
        return

    def load(f, *args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original(f, *args, **kwargs)

    load._memguard_cpu = True
    torch.load = load
