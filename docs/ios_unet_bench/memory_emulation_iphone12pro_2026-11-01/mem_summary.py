#!/usr/bin/env python3 -I
"""Summarize one or more phone_<label>.log files from run_phone.sh: the [mem] trace, the lipsync result and the
MEM_RESULT line, plus whether the process ended on its own or disappeared (a jetsam kill)."""
import json
import re
import sys

MEM = re.compile(r"\[mem\] t=([\d.]+) s footprint (\d+) MB, (-?\d+) MB left \(peak (\d+)\)")


def summarize(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    out = {"log": path}
    samples = [(float(m.group(1)), int(m.group(2)), int(m.group(3))) for m in (MEM.search(l) for l in lines) if m]
    out["trace_points"] = len(samples)
    if samples:
        out["peak_footprint_mb_from_trace"] = max(s[1] for s in samples)
        out["min_headroom_mb_from_trace"] = min(s[2] for s in samples)
        out["last_trace"] = samples[-1]
    for l in lines:
        if "ballast" in l and "[mem]" in l:
            out["ballast_line"] = l.strip()
        if "emulating a" in l:
            out["limit_line"] = l.strip()
        if "memory pressure" in l:
            out.setdefault("pressure_events", []).append(l.strip())
        if "models loaded in" in l:
            out["models_loaded"] = l.strip()
        if "thermal" in l.lower() and "[mem]" not in l:
            out.setdefault("thermal_lines", []).append(l.strip())
    m = re.search(r"LIPSYNC_RESULT (\{.*\})", text)
    if m:
        r = json.loads(m.group(1))
        keep = ("shown", "frames", "stalls", "dropped_late", "skipped", "stage_infer_ms", "stage_compose_ms",
                "load_seconds", "start_latency_seconds", "seconds", "generated")
        out["lipsync"] = {k: r.get(k) for k in keep if k in r}
        out["lipsync"]["shown_fps"] = round(r.get("shown", 0) / r["seconds"], 1) if r.get("seconds") else None
    m = re.search(r"MEM_RESULT (\{.*\})", text)
    if m:
        out["mem_result"] = json.loads(m.group(1))
    m = re.search(r"== .* launch exit (\d+)", text)
    out["launch_exit"] = int(m.group(1)) if m else None
    out["ended"] = ("LIPSYNC_RESULT" in text and "exited" in text.lower()) or "MEM_RESULT" in text
    for l in lines:
        if "ERROR" in l or "killed" in l.lower() or "terminated" in l.lower() or "Locked" in l:
            out.setdefault("notable", []).append(l.strip()[:200])
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(summarize(p), indent=1))
