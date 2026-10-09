"""Opt-in wrapper; canonical renderer bytes, frame math and timing stay unchanged.

Use this script instead of chin_multistream_render.py for a separately bound
diagnostic run. Import is CPU/stdlib-only. Never update historical launcher pins
to make it pretend to be the old renderer. Counter availability is not acceptance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading

import chin_multistream_render as renderer
from chin_multistream import gpu, telemetry


_lock = threading.Lock()
_original_run_multi = renderer.run_multi
RENDERER_SHA256 = "df4e290b33d752be82d6d2ab738bc5d3e21aa439ffd05f1c3c852a8af1fd4a29"


def run_multi(args, run_dir):
    if hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest() != RENDERER_SHA256:
        raise RuntimeError("Canonical renderer bytes changed; telemetry binding refused")
    if not _lock.acquire(blocking=False):
        raise RuntimeError("CPU telemetry wrapper cannot overlap another invocation")
    original_issue = gpu.run_repeat
    original_collect = renderer.Collector.wait_for
    intervals = []
    pending = None

    def issue(*positional, **keywords):
        nonlocal pending
        if pending is not None:
            raise RuntimeError("Previous telemetry window has not collected workers")
        pending = {"before": telemetry.cgroup_cpu_snapshot(), "issued": False}
        result = original_issue(*positional, **keywords)
        pending["issued"] = True
        return result

    def collect(self, kind, streams, timeout=600.0):
        nonlocal pending
        messages = original_collect(self, kind, streams, timeout=timeout)
        if kind == "repdone":
            if pending is None or not pending["issued"]:
                raise RuntimeError("Worker collection has no matching issued telemetry window")
            after = telemetry.cgroup_cpu_snapshot()
            intervals.append(telemetry.cgroup_cpu_interval(pending["before"], after))
            pending = None
        return messages

    try:
        gpu.run_repeat = issue
        renderer.Collector.wait_for = collect
        result = _original_run_multi(args, run_dir)
        if pending is not None or len(intervals) != len(result["repeats"]):
            raise RuntimeError("CPU telemetry coverage does not match completed windows")
        for repeat, interval in zip(result["repeats"], intervals):
            repeat["cgroup_cpu"] = interval
        result["cpu_window_telemetry"] = {
            "schema": "opt_in_cpu_window_telemetry_v1",
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "renderer_bytes_changed": False,
            "frame_math_or_fps_denominator_changed": False,
            "gpu_performance_or_quality_accepted": False,
        }
        return result
    finally:
        gpu.run_repeat = original_issue
        renderer.Collector.wait_for = original_collect
        _lock.release()


def main(argv=None):
    args = renderer.parse_args(argv)
    if args.mode != "multi":
        raise ValueError("CPU-window telemetry requires multi mode")
    original = renderer.run_multi
    try:
        renderer.run_multi = run_multi
        return renderer.main(argv)
    finally:
        renderer.run_multi = original


if __name__ == "__main__":
    main()
