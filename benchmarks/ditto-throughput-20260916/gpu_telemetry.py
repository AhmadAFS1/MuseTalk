#!/usr/bin/env python3
"""Sample one NVIDIA GPU while running a benchmark command."""

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path


QUERY = (
    "index,name,memory.total,memory.used,utilization.gpu,"
    "utilization.memory,power.draw,temperature.gpu"
)


def read_gpu(index: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    return {
        "gpu_index": int(values[0]),
        "gpu_name": values[1],
        "memory_total_mib": float(values[2]),
        "memory_used_mib": float(values[3]),
        "gpu_util_pct": float(values[4]),
        "memory_util_pct": float(values[5]),
        "power_w": float(values[6]),
        "temperature_c": float(values[7]),
    }


def summarize(samples: list[dict]) -> dict:
    if not samples:
        return {"samples": 0}
    busy = [sample for sample in samples if sample["gpu_util_pct"] > 0]
    def stats(key: str, rows: list[dict]) -> dict:
        values = [row[key] for row in rows]
        return {
            "avg": round(statistics.fmean(values), 3),
            "p95": round(sorted(values)[min(len(values) - 1, int(len(values) * .95))], 3),
            "peak": round(max(values), 3),
        }
    return {
        "samples": len(samples),
        "duration_s": round(samples[-1]["t_rel_s"] - samples[0]["t_rel_s"], 3),
        "gpu_util_pct": stats("gpu_util_pct", samples),
        "gpu_busy_util_pct": stats("gpu_util_pct", busy) if busy else None,
        "busy_sample_fraction": round(len(busy) / len(samples), 4),
        "memory_used_mib": stats("memory_used_mib", samples),
        "power_w": stats("power_w", samples),
        "temperature_c": stats("temperature_c", samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=.1)
    parser.add_argument("--idle-before", type=float, default=2)
    parser.add_argument("--idle-after", type=float, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    phase = {"value": "idle_before"}
    stop = threading.Event()
    started = time.time()

    def sample_loop():
        while not stop.is_set():
            tick = time.time()
            try:
                row = read_gpu(args.gpu_index)
                row.update(timestamp=tick, t_rel_s=round(tick - started, 6), phase=phase["value"])
                samples.append(row)
            except Exception as exc:
                samples.append({"timestamp": tick, "t_rel_s": round(tick-started, 6),
                                "phase": phase["value"], "error": repr(exc)})
            stop.wait(max(0, args.interval - (time.time() - tick)))

    worker = threading.Thread(target=sample_loop, daemon=True)
    worker.start()
    time.sleep(args.idle_before)
    phase["value"] = "command"
    command_started = time.time()
    proc = subprocess.run(command)
    command_ended = time.time()
    phase["value"] = "idle_after"
    time.sleep(args.idle_after)
    stop.set()
    worker.join(timeout=5)

    valid = [sample for sample in samples if "error" not in sample]
    phases = {
        name: summarize([sample for sample in valid if sample["phase"] == name])
        for name in ("idle_before", "command", "idle_after")
    }
    report = {
        "schema": 1,
        "command": command,
        "exit_code": proc.returncode,
        "sample_interval_target_s": args.interval,
        "command_started": command_started,
        "command_ended": command_ended,
        "command_wall_s": round(command_ended - command_started, 6),
        "gpu": valid[0]["gpu_name"] if valid else None,
        "phases": phases,
        "samples": samples,
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
