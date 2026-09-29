"""Per-window timeline of a soak level: client gaps > 100 ms, worst anchored 1 s window, server largest send interval,
server RSS and MemAvailable, so drift over an hour is visible.
  soak_timeline.py <run dir> [--stage soak] [--n 15] [--window-s 300] [--md out.md]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

VID = np.dtype([("t", "<f8"), ("pts", "<i8")])
RING = np.dtype([("seq", "<i8"), ("t", "<f8"), ("k", "i1"), ("pts", "<i8")])


def anchored_min(t, span=1.0):
    if len(t) < 21:
        return None
    j = np.searchsorted(t, t + span, side="left")
    ok = t + span <= t[-1]
    return int((j - np.arange(len(t)))[ok].min()) if ok.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stage", default="soak")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--window-s", type=float, default=300.0)
    ap.add_argument("--md")
    a = ap.parse_args()
    R = Path(a.run)
    lv = R / "traces" / a.stage / f"n{a.n:02d}"
    lm = json.loads((lv / "level_meta.json").read_text())
    t_go, t_end = lm["t_go"], lm["t_end"]
    streams = []
    for f in sorted(lv.glob("s*_video.bin")):
        k = int(f.name[1:3])
        meta = json.loads((lv / f"s{k:02d}_meta.json").read_text())
        ring_path = lv / f"ring_{meta['session_id']}.bin"
        ring = np.frombuffer(ring_path.read_bytes(), dtype=RING)["t"] if ring_path.exists() else np.array([])
        streams.append((k, np.frombuffer(f.read_bytes(), dtype=VID)["t"], ring))
    # box.csv is epoch time; map to monotonic with the current offset (both clocks advance together)
    off = time.time() - time.monotonic()
    box = np.genfromtxt(R / "box.csv", delimiter=",", names=True)
    bt = box["t"] - off
    rows = []
    w = 0.0
    last_join = max((j["t"] for j in lm.get("joins", [])), default=t_go) - t_go
    while t_go + w < t_end:
        a0, a1 = t_go + w, min(t_go + w + a.window_s, t_end)
        gaps = 0
        worst_gap = 0.0
        amin = None
        smax = 0.0
        for k, t, ring in streams:
            s = t[(t >= a0) & (t < a1)]
            if len(s) > 21:
                d = np.diff(s)
                gaps += int((d > 0.100).sum())
                worst_gap = max(worst_gap, float(d.max()) * 1000)
                m = anchored_min(s)
                amin = m if amin is None or (m is not None and m < amin) else amin
            r = ring[(ring >= a0) & (ring < a1)]
            if len(r) > 2:
                smax = max(smax, float(np.diff(r).max()) * 1000)
        bm = (bt >= a0) & (bt < a1)
        rss = np.nanmax(box["server_rss_kb"][bm]) / 1024 if bm.any() else float("nan")
        avail = np.nanmin(box["mem_available_kb"][bm]) / 1024 if bm.any() else float("nan")
        rows.append((w, a1 - t_go, gaps, round(worst_gap, 1), amin, round(smax, 1), round(rss), round(avail)))
        w += a.window_s
    lines = [f"Soak timeline: {lv} (last join at {last_join:.0f} s after GO)", "",
             "| window s | client gaps > 100 ms | worst client gap ms | worst anchored 1 s (client) | server largest send interval ms | server RSS max MB | MemAvailable min MB |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]:.0f}-{r[1]:.0f} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
