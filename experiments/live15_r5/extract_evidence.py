"""Regenerate the evidence extracts cited in docs/fps_comparisons/live15_r5_20260929/README.md from the run dirs.
  extract_evidence.py <runs root, e.g. tmp/live15_r5> <out dir, e.g. docs/fps_comparisons/live15_r5_20260929/evidence>
Each extract names the run it comes from; runs that are missing are skipped with a note."""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
VID = np.dtype([("t", "<f8"), ("pts", "<i8")])
RING = np.dtype([("seq", "<i8"), ("t", "<f8"), ("k", "i1"), ("pts", "<i8")])


def run(name):
    d = ROOT / name
    return d if d.exists() else None


def log_lines(d):
    return (d / "api_server_8300.log").read_text(errors="replace").splitlines()


def level(d, stage, n):
    lv = d / "traces" / stage / f"n{n:02d}"
    return lv, json.loads((lv / "level_meta.json").read_text())


def client_gaps(lv, t_go, thr=0.1):
    out = []
    for f in sorted(lv.glob("s*_video.bin")):
        t = np.frombuffer(f.read_bytes(), dtype=VID)["t"]
        d = np.diff(t)
        for k in np.where(d > thr)[0]:
            out.append((float(t[k + 1] - t_go), float(d[k] * 1000), f.name[:3]))
    return sorted(out)


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / name} ({len(lines)} lines)")


# 1. gen-2 GC pauses at N=5 (GC log run) and the gap clusters they line up with
d = run("20260929T062327Z_B_gclog")
if d:
    lv, lm = level(d, "ramp", 5)
    gcs = [(float(m.group(3)) - lm["t_go"], float(m.group(2))) for m in
           (re.search(r"🧹 GC gen(2) ([\d.]+) ms .*t_mono=([\d.]+)", l) for l in log_lines(d)) if m]
    gcs = [g for g in gcs if 0 < g[0] < lm["t_end"] - lm["t_go"]]
    gaps = client_gaps(lv, lm["t_go"])
    lines = [f"run {d.name}, N=5, Arm B (serve.env), GC log on. Times are seconds after GO.",
             "gen-2 collections >= 20 ms (t, ms):"] + [f"  {t:8.1f}  {ms:6.1f}" for t, ms in gcs]
    lines += [f"client gaps > 100 ms: {len(gaps)}; within 0.3 s after a gen-2 collection >= 100 ms: "
              f"{sum(1 for t, g, s in gaps if any(0 <= t - gt <= 0.3 for gt, ms in gcs if ms >= 100))}"]
    write("01_gc_pauses_n5.txt", lines)

# 2. asyncio slow callbacks at N=5 (before the off-loop fix), by endpoint, and request latency
d = run("20260929T064641Z_B_loopdiag")
if d:
    L = log_lines(d)
    slow = [l for l in L if "🐌 asyncio Executing" in l]
    kinds = collections.Counter("BaseHTTPMiddleware call_next (HTTP handler)" if "BaseHTTPMiddleware" in l else
                                re.sub(r"^.*?(<Task \w+ name='[^']*' coro=<([\w.]+)|<Handle ([\w.<>]+)).*$",
                                       lambda m: m.group(2) or m.group(3) or "?", l) for l in slow)
    lat = collections.defaultdict(list)
    for l in L:
        if "API request done" in l:
            p = re.search(r"path=(\S+)", l).group(1)
            p = re.sub(r"/sessions/[A-Za-z0-9_-]{10,}", "/sessions/<id>", p)
            if "/cache/warm" in p:
                continue
            lat[p].append(float(re.search(r"elapsed_ms=([\d.]+)", l).group(1)))
    lines = [f"run {d.name}, N=5, asyncio slow-callback log >= 40 ms (GC freeze on, nvidia-smi still on the loop)",
             f"slow callbacks: {len(slow)}"] + [f"  {v:3d}  {k}" for k, v in kinds.most_common()]
    lines += ["request latency (ms) by endpoint:"]
    for p, v in sorted(lat.items(), key=lambda x: -len(x[1])):
        v.sort()
        lines.append(f"  {p:40s} n={len(v):5d} p50={v[len(v) // 2]:7.1f} max={v[-1]:7.1f}")
    write("02_slow_callbacks_n5.txt", lines)

# 3. loop-stall stack dumps at N=10: which threads were running while the loop waited
d = run("20260929T074547Z_B_stalldump")
if d:
    L = log_lines(d)
    dumps, cur = [], None
    for l in L:
        m = re.match(r"🩺 loop stall (\d+) ms", l)
        if m:
            cur = {"ms": int(m.group(1)), "th": []}
            dumps.append(cur)
        elif cur is not None and l.startswith("   ["):
            name = re.match(r"   \[([^\]]+)\] (\S+)", l)
            cur["th"].append((re.sub(r"_\d+$", "_N", name.group(1)), name.group(2)))
        else:
            cur = None
    per = collections.Counter()
    for x in dumps:
        for n in {t for t, _ in x["th"]}:
            per[n] += 1
    lines = [f"run {d.name}, N=10, MUSETALK_LOOP_STALL_DUMP_MS=40: {len(dumps)} stalls, ms {sorted(x['ms'] for x in dumps)}",
             "threads not parked in a wait during a stall (count of stalls):"] + \
            [f"  {v:3d}/{len(dumps)}  {k}" for k, v in per.most_common()]
    first = [l for l in L if l.startswith("🩺 loop stall") or l.startswith("   [")]
    lines += ["", "first dumps verbatim:"] + first[:40]
    write("03_stall_dumps_n10.txt", lines)

# 4. gen-2 collections vs client gaps at N=15 (before the packed FIFO and the threshold change)
d = run("20260929T081141Z_B_loopfix2_gclog")
if d:
    lv, lm = level(d, "ramp", 15)
    g2 = [(float(m.group(4)) - lm["t_go"], float(m.group(2)), int(m.group(3))) for m in
          (re.search(r"🧹 GC gen(2) ([\d.]+) ms collected=(\d+).*t_mono=([\d.]+)", l) for l in log_lines(d)) if m]
    g2 = [x for x in g2 if 0 < x[0] < lm["t_end"] - lm["t_go"]]
    gaps = client_gaps(lv, lm["t_go"])
    near = sum(1 for t, g, s in gaps if any(-0.05 <= t - gt <= 0.2 for gt, _, _ in g2))
    lines = [f"run {d.name}, N=15, GC log on (perturbs timing; used for attribution only)",
             f"gen-2 collections >= 20 ms: {len(g2)}; interval s {np.round(np.diff([x[0] for x in g2]), 1).tolist()}",
             f"durations ms {[x[1] for x in g2]}", f"objects collected {[x[2] for x in g2]}",
             f"client gaps > 100 ms: {len(gaps)}; starting within 200 ms after a gen-2 collection: {near}"]
    write("04_gen2_vs_gaps_n15.txt", lines)

# 5. what the cyclic GC frees while streaming
lines = []
for name, label in (("20260929T081939Z_B_garbage", "N=5, before the packed FIFO"),
                    ("20260929T085313Z_B_garbage15", "N=15, with the packed FIFO (sampler kept garbage, so counts accumulate; look at the increments)")):
    d = run(name)
    if d:
        lines.append(f"run {name}, {label}:")
        lines += ["  " + l[:400] for l in log_lines(d) if "garbage sample" in l and "VideoFormat" in l][:12]
if lines:
    write("05_garbage_types.txt", lines)

# 6. aiortc receiver one-frame hold: client arrival of frame N follows frame N+1's send
d = run("20260929T084351Z_B_loopfix3_lightpoll")
if d:
    lv, lm = level(d, "ramp", 15)
    T = lm["t_go"] + 220.9
    lines = [f"run {d.name}, N=15, around t = GO + 220.9 s (times relative to it)"]
    for s in (5, 9, 12):
        meta = json.loads((lv / f"s{s:02d}_meta.json").read_text())
        r = np.frombuffer((lv / f"ring_{meta['session_id']}.bin").read_bytes(), dtype=RING)
        c = np.frombuffer((lv / f"s{s:02d}_video.bin").read_bytes(), dtype=VID)
        send = {int(p): t for p, t in zip(r["pts"], r["t"])}
        lines.append(f"stream {s}:")
        for x in c[(c["t"] > T - 0.2) & (c["t"] < T + 0.15)]:
            st = send.get(int(x["pts"]))
            if st:
                lines.append(f"  pts {x['pts']:>10}  sent {st - T:+.3f}  arrived {x['t'] - T:+.3f}  "
                             f"latency {(x['t'] - st) * 1000:6.1f} ms")
    write("06_aiortc_one_frame_hold.txt", lines)

# 7. the load tester's own polling load on the server loop, before and after per-session polling
lines = []
for name in ("20260929T083012Z_B_loopfix3", "20260929T084351Z_B_loopfix3_lightpoll"):
    d = run(name)
    if d:
        v = [float(re.search(r"elapsed_ms=([\d.]+)", l).group(1)) for l in log_lines(d)
             if "API request done" in l and "path=/webrtc/sessions/stats " in l]
        st = [l for l in log_lines(d) if "API request done" in l and re.search(r"path=/webrtc/sessions/[^/ ]+/status ", l)]
        lines.append(f"run {name}: GET /webrtc/sessions/stats n={len(v)} total {sum(v) / 1000:.1f} s; "
                     f"GET /webrtc/sessions/<id>/status n={len(st)}")
if lines:
    write("07_test_polling_load.txt", lines)
