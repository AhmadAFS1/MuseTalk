"""One markdown table over every live15 run directory: config, level, server-side verdict and client trace verdict.
  summarize_runs.py [runs root, default tmp/live15_r5] [--json out.json]
Per level: the load tester's server-side line (PASS/FAIL, send_max = the server's largest send interval) and the
live_trace_report.py summary (streams passing P1-P3, worst anchored 1 s window, worst client gap, gaps > 100 ms).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def overrides(run: Path) -> str:
    head = (run / "README.txt").read_text(errors="replace").splitlines()[0] if (run / "README.txt").exists() else ""
    m = re.search(r"overrides=(\S+)", head)
    if not m:
        return "?"
    return "+".join(Path(p).stem for p in m.group(1).split(":") if Path(p).stem != "common")


def server_lines(run: Path) -> dict:
    out = {}
    for f in sorted(run.glob("*.out")):
        for line in f.read_text(errors="replace").splitlines():
            m = re.match(r"^(PASS|FAIL|INVALID) level N=(\d+) label=(\S+)", line)
            if not m:
                continue
            stage = m.group(3).split("_")[0]
            send = re.search(r"send_max=([\d.]+)", line)
            gen = re.search(r"generated_fps=([\d.]+)", line)
            rss = re.search(r"rss_max_mb=([\d.]+)", line)
            out[(stage, int(m.group(2)))] = {
                "server_verdict": m.group(1),
                "send_max_ms": round(float(send.group(1)) * 1000, 1) if send else None,
                "generated_fps": float(gen.group(1)) if gen else None,
                "rss_max_mb": float(rss.group(1)) if rss else None,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="tmp/live15_r5")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.root).iterdir()):
        if not run.is_dir():
            continue
        srv = server_lines(run)
        for tj in sorted(run.glob("*_n*_trace.json")):
            m = re.match(r"(\w+?)_n(\d+)_trace\.json", tj.name)
            stage, n = m.group(1), int(m.group(2))
            s = json.loads(tj.read_text())["summary"]
            row = {"run": run.name, "config": overrides(run), "stage": stage, "N": n, **srv.get((stage, n), {}),
                   "streams_pass": f"{s['streams_pass']}/{s['streams']}", "all_pass": s["all_pass"],
                   "min_anchored_1s": s["min_anchored_1s"], "worst_gap_ms": s["worst_gap_ms"],
                   "gaps_over_100ms": s["gaps_over_100ms_total"], "min_fresh": s["min_server_fresh_fraction"]}
            rows.append(row)
    cols = ["run", "config", "stage", "N", "server_verdict", "send_max_ms", "streams_pass", "all_pass",
            "min_anchored_1s", "worst_gap_ms", "gaps_over_100ms", "min_fresh", "generated_fps", "rss_max_mb"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
