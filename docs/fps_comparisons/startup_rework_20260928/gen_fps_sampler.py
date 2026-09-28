#!/usr/bin/env python3
"""Sample server-side generation progress during a WebRTC load test.

Polls GET /hls/sessions/stats (the shared GPU scheduler's job table) and keeps,
per request_id, the highest current_frame_idx seen. Frames generated per second
across all jobs is the live "fresh fps" of the GPU scheduler, independent of
the client-side delivery cadence that load_test_webrtc.py reports.

  gen_fps_sampler.py sample --url http://127.0.0.1:8000 --out X.jsonl   (runs until killed)
  gen_fps_sampler.py summarize X.jsonl [--stage-gap-s 5]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.request


def sample(url: str, out: str, interval: float) -> None:
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    with open(out, "a", encoding="utf-8") as fh:
        while not stop:
            t = time.time()
            try:
                with urllib.request.urlopen(f"{url}/hls/sessions/stats", timeout=2) as resp:
                    payload = json.load(resp)
                jobs = payload.get("scheduler", {}).get("jobs", [])
                rec = {
                    "t": round(t, 3),
                    "jobs": {
                        j["request_id"]: [j.get("current_frame_idx", 0), j.get("total_frames", 0),
                                          j.get("avg_gpu_batch_s", 0.0), j.get("avg_unet_s", 0.0),
                                          j.get("avg_vae_s", 0.0)]
                        for j in jobs
                    },
                }
            except Exception as exc:  # server busy or restarting; keep sampling
                rec = {"t": round(t, 3), "error": f"{type(exc).__name__}: {exc}"[:200]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            time.sleep(max(0.0, interval - (time.time() - t)))


def summarize(path: str, stage_gap_s: float) -> dict:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if "jobs" in r]
    # Split into stages: runs of samples with at least one job, separated by idle gaps.
    stages, current, last_busy = [], [], None
    for r in rows:
        if r["jobs"]:
            if last_busy is not None and r["t"] - last_busy > stage_gap_s and current:
                stages.append(current)
                current = []
            current.append(r)
            last_busy = r["t"]
    if current:
        stages.append(current)

    out = []
    for rows_s in stages:
        best: dict[str, int] = {}
        first_progress_t, last_progress_t = None, None
        totals: dict[str, int] = {}
        timeline = []
        for r in rows_s:
            advanced = False
            for rid, vals in r["jobs"].items():
                idx, total = int(vals[0]), int(vals[1])
                totals[rid] = total
                if idx > best.get(rid, 0):
                    best[rid] = idx
                    advanced = True
            if advanced:
                if first_progress_t is None:
                    first_progress_t = r["t"]
                last_progress_t = r["t"]
            timeline.append((r["t"], sum(best.values())))
        frames = sum(best.values())
        span = (last_progress_t - first_progress_t) if first_progress_t is not None else 0.0
        # Peak 5 s window rate.
        peak = 0.0
        j = 0
        for i in range(len(timeline)):
            while timeline[i][0] - timeline[j][0] > 5.0:
                j += 1
            dt = timeline[i][0] - timeline[j][0]
            if dt >= 4.0:
                peak = max(peak, (timeline[i][1] - timeline[j][1]) / dt)
        gpu = [v for r in rows_s for v in r["jobs"].values() if v[2]]
        out.append({
            "jobs": len(best),
            "frames_generated": frames,
            "frames_expected": sum(totals.values()),
            "generation_span_s": round(span, 2),
            "mean_gen_fps": round(frames / span, 1) if span > 0 else None,
            "peak_5s_gen_fps": round(peak, 1),
            "avg_gpu_batch_s": round(sum(v[2] for v in gpu) / len(gpu), 4) if gpu else None,
            "avg_unet_s": round(sum(v[3] for v in gpu) / len(gpu), 4) if gpu else None,
            "avg_vae_s": round(sum(v[4] for v in gpu) / len(gpu), 4) if gpu else None,
        })
    return {"source": path, "stages": out}


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--out", required=True)
    s.add_argument("--interval", type=float, default=0.5)
    m = sub.add_parser("summarize")
    m.add_argument("path")
    m.add_argument("--stage-gap-s", type=float, default=5.0)
    a = p.parse_args()
    if a.cmd == "sample":
        sample(a.url, a.out, a.interval)
    else:
        print(json.dumps(summarize(a.path, a.stage_gap_s), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
