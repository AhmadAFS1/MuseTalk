"""Per-stream client-side verdicts for a live WebRTC load-test level, from load_test_webrtc_v2.py --trace-dir output.

Inputs (one level dir, e.g. <trace_dir>/n15/):
  s<k>_video.bin      per decoded client video frame: (arrival monotonic s, RTP-derived pts), little-endian <dq
  s<k>_meta.json      the stream's session id / avatar / join timing
  ring_<sid>.bin      the server's per-track send ring: (seq, send monotonic s, kind f|h|p|i, pts) <qdbq
  shard<k>_looplag.bin client event-loop stalls > 20 ms: (t, lag_ms) pairs
  server_counters.jsonl 1 Hz stitched server counters; level_meta.json (t_go, t_end, joins)
Both clocks are CLOCK_MONOTONIC on the same box, so client arrivals and server sends are directly comparable.

Criteria (defined before the run; see experiments/live15_r5/README.md):
  P1 rate:      every 1 s window that starts at a frame arrival holds >= 20 frames, i.e. t[i+19] - t[i] <= 1.000 s
  P2 fresh:     server fresh fraction over speaking slots >= 0.995, max held run <= 2; client content (f/i/p-labelled
                arrivals) >= 18 per anchored 1 s window
  P3 buffering: client max inter-arrival gap <= 120 ms; gaps > 100 ms at most 1 per stream per 10 min and none
                caused by the server's own send cadence
Each stream is scored from its first arrival + settle to the level end (or its last arrival).

Gap causes: client_loop (the shard's event loop stalled), server_send (a server send interval > 100 ms),
server_late (a send interval of 75-100 ms: aiortc's receiver emits frame N only when frame N+1's first packet
arrives, so one frame sent 25-50 ms late shows up as a > 100 ms client gap), transport (none of these).
Server cadence (reported beside P1-P3, not a pass criterion): the same anchored 1 s count over the server's own
send times, i.e. what a receiver that completes frames on their last packet (a browser) sees before the network.
usage: live_trace_report.py <level dir> [--settle-s 10] [--json out.json] [--md out.md]
"""
from __future__ import annotations

import argparse
import bisect
import json
import struct
from collections import Counter
from pathlib import Path

WRAP = 1 << 32


def read_pairs(path: Path, fmt: str):
    size = struct.calcsize(fmt)
    data = path.read_bytes()
    return [struct.unpack_from(fmt, data, i) for i in range(0, len(data) - size + 1, size)]


def pct(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(q * (len(ys) - 1) + 0.5))]


def anchored_windows(t, span=1.0, need=20):
    """For each arrival i: frames in [t_i, t_i + span). Returns (min count, windows below need, first bad t)."""
    j, low, bad, first_bad, mins = 0, None, 0, None, []
    for i in range(len(t)):
        if t[i] + span > t[-1]:
            break
        if j < i:
            j = i
        while j < len(t) and t[j] < t[i] + span:
            j += 1
        c = j - i
        low = c if low is None else min(low, c)
        if c < need:
            bad += 1
            if first_bad is None:
                first_bad = t[i]
    return low, bad, first_bad


def fixed_bins(t, t0):
    counts = Counter(int(x - t0) for x in t)
    if not counts:
        return None
    lo, hi = min(counts), max(counts)
    return min(counts.get(b, 0) for b in range(lo, hi))  if hi > lo else counts[lo]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("level_dir")
    ap.add_argument("--settle-s", type=float, default=10.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    d = Path(a.level_dir)
    meta = json.loads((d / "level_meta.json").read_text()) if (d / "level_meta.json").exists() else {}
    t_end_level = meta.get("t_end")
    lags = []
    for f in sorted(d.glob("shard*_looplag.bin")):
        raw = f.read_bytes()
        vals = struct.unpack(f"<{len(raw) // 8}d", raw)
        lags += list(zip(vals[0::2], vals[1::2]))
    lags.sort()
    lag_t = [x[0] for x in lags]
    rows = []
    for mf in sorted(d.glob("s*_meta.json")):
        m = json.loads(mf.read_text())
        k = m["stream"]
        vid = read_pairs(d / f"s{k:02d}_video.bin", "<dq") if (d / f"s{k:02d}_video.bin").exists() else []
        if len(vid) < 40:
            rows.append({"stream": k, "avatar_id": m.get("avatar_id"), "error": "too few frames", "frames": len(vid)})
            continue
        t_all = [x[0] for x in vid]
        pts_all = [x[1] for x in vid]
        w0 = t_all[0] + a.settle_s
        w1 = min(t_end_level, t_all[-1]) if t_end_level else t_all[-1]
        ring_path = d / f"ring_{m['session_id']}.bin"
        ring = read_pairs(ring_path, "<qdbq") if ring_path.exists() else []
        if ring:
            w1 = min(w1, ring[-1][1])  # the ring is polled at 1 Hz: score only the span it covers
        idx = [i for i, t in enumerate(t_all) if w0 <= t <= w1]
        t = [t_all[i] for i in idx]
        span = t[-1] - t[0]
        gaps = [(b - a_, a_) for a_, b in zip(t, t[1:])]
        low, bad, first_bad = anchored_windows(t)
        # ---- join to the server send ring by the per-stream RTP offset
        kinds, lat, matched = [], [], 0
        fresh_frac = held_run_server = None
        send_gaps_100 = 0
        send_interval_max = server_low = None
        if ring:
            rt = [r[1] for r in ring]
            by_pts = {}
            for r in ring:
                by_pts.setdefault(r[3] % WRAP, r)
            # candidate offsets from arrivals paired with the latest send before them
            cand = Counter()
            for i in range(0, min(len(t_all), 400)):
                j = bisect.bisect_right(rt, t_all[i]) - 1
                for jj in range(max(0, j - 3), j + 1):
                    cand[(pts_all[i] - ring[jj][3]) % WRAP] += 1
            offset = cand.most_common(1)[0][0] if cand else 0
            for i in idx:
                r = by_pts.get((pts_all[i] - offset) % WRAP)
                if r is None:
                    kinds.append("?")
                    continue
                matched += 1
                kinds.append(chr(r[2]))
                lat.append(t_all[i] - r[1])
            # server-side freshness over speaking slots inside the stream's window
            sp = [r for r in ring if w0 <= r[1] <= w1 and chr(r[2]) in "fh"]
            if sp:
                fresh_frac = sum(1 for r in sp if chr(r[2]) == "f") / len(sp)
                run = held_run_server = 0
                for r in sp:
                    run = run + 1 if chr(r[2]) == "h" else 0
                    held_run_server = max(held_run_server, run)
            inside = [r[1] for r in ring if w0 <= r[1] <= w1]
            send_gaps_100 = sum(1 for x, y in zip(inside, inside[1:]) if y - x > 0.100)
            send_interval_max = max((y - x for x, y in zip(inside, inside[1:])), default=None)
            server_low = anchored_windows(inside)[0] if len(inside) > 40 else None
        # content-bearing arrivals (not held repeats) per anchored 1 s window
        content_t = [t[i] for i, kd in enumerate(kinds) if kd in "fip"] if kinds else []
        c_low = anchored_windows(content_t, need=18)[0] if len(content_t) > 40 else None
        held_run_client = run = 0
        for kd in kinds:
            run = run + 1 if kd == "h" else 0
            held_run_client = max(held_run_client, run)
        # ---- gap attribution
        big = []
        for g, at in gaps:
            if g <= 0.100:
                continue
            cause = "transport"
            li = bisect.bisect_left(lag_t, at - 0.05)
            if li < len(lags) and lags[li][0] <= at + g + 0.05:
                cause = "client_loop"
            elif ring:
                j = bisect.bisect_right(rt, at + g) - 1
                window = [r[1] for r in ring if at - 0.2 <= r[1] <= at + g]
                intervals = [y - x for x, y in zip(window, window[1:])]
                if any(iv > 0.100 for iv in intervals):
                    cause = "server_send"
                elif any(iv > 0.075 for iv in intervals):
                    cause = "server_late"
            big.append({"t_rel_s": round(at - t_all[0], 2), "gap_ms": round(g * 1000, 1), "cause": cause})
        per10 = len(big) / max(span / 600.0, 1e-9)
        p1 = low is not None and low >= 20
        p2 = (fresh_frac is None or (fresh_frac >= 0.995 and (held_run_server or 0) <= 2)) and (c_low is None or c_low >= 18)
        p3 = (max(g for g, _ in gaps) <= 0.120) and per10 <= 1.0 and not any(b["cause"] == "server_send" for b in big)
        rows.append({
            "stream": k, "avatar_id": m.get("avatar_id"), "session_id": m.get("session_id"),
            "window_s": round(span, 1), "frames": len(t), "mean_fps": round((len(t) - 1) / span, 3),
            "anchored_1s_min_frames": low, "anchored_1s_windows_below_20": bad,
            "first_below_20_rel_s": round(first_bad - t_all[0], 2) if first_bad else None,
            "fixed_1s_bin_min": fixed_bins(t, t[0]),
            "gap_max_ms": round(max(g for g, _ in gaps) * 1000, 1), "gap_p99_ms": round(pct([g for g, _ in gaps], 0.99) * 1000, 1),
            "gaps_over_100ms": len(big), "gaps_over_120ms": sum(1 for b in big if b["gap_ms"] > 120),
            "gaps_over_100ms_per_10min": round(per10, 2), "gap_causes": dict(Counter(b["cause"] for b in big)),
            "worst_gaps": sorted(big, key=lambda b: -b["gap_ms"])[:5],
            "pts_join_matched": round(matched / len(idx), 4) if ring else None,
            "client_kinds": dict(Counter(kinds)) if kinds else None,
            "client_content_1s_min": c_low, "client_held_run_max": held_run_client if kinds else None,
            "server_fresh_fraction": round(fresh_frac, 5) if fresh_frac is not None else None,
            "server_held_run_max": held_run_server, "server_send_gaps_over_100ms": send_gaps_100,
            "server_send_interval_max_ms": round(send_interval_max * 1000, 1) if send_interval_max else None,
            "server_anchored_1s_min": server_low,
            "latency_ms_p50": round(pct(lat, 0.5) * 1000, 1) if lat else None,
            "latency_ms_p99": round(pct(lat, 0.99) * 1000, 1) if lat else None,
            "latency_ms_max": round(max(lat) * 1000, 1) if lat else None,
            "P1_rate": p1, "P2_fresh": p2, "P3_no_buffering": p3, "PASS": bool(p1 and p2 and p3)})
    ok = [r for r in rows if "error" not in r]
    summary = {"level_dir": str(d), "streams": len(rows), "streams_pass": sum(1 for r in ok if r["PASS"]),
               "all_pass": bool(ok) and len(ok) == len(rows) and all(r["PASS"] for r in ok),
               "min_anchored_1s": min((r["anchored_1s_min_frames"] for r in ok), default=None),
               "worst_gap_ms": max((r["gap_max_ms"] for r in ok), default=None),
               "gaps_over_100ms_total": sum(r["gaps_over_100ms"] for r in ok),
               "min_server_fresh_fraction": min((r["server_fresh_fraction"] for r in ok if r["server_fresh_fraction"] is not None), default=None),
               "max_server_held_run": max((r["server_held_run_max"] or 0 for r in ok), default=None),
               "min_pts_join": min((r["pts_join_matched"] for r in ok if r["pts_join_matched"] is not None), default=None),
               "client_loop_spikes": len(lags), "client_loop_spike_max_ms": round(max((x[1] for x in lags), default=0), 1),
               "server_cadence_min_anchored_1s": min((r["server_anchored_1s_min"] for r in ok
                                                      if r.get("server_anchored_1s_min") is not None), default=None),
               "server_send_interval_max_ms": max((r["server_send_interval_max_ms"] for r in ok
                                                   if r.get("server_send_interval_max_ms") is not None), default=None),
               "gap_causes": dict(sum((Counter(r["gap_causes"]) for r in ok), Counter()))}
    out = {"summary": summary, "streams": rows}
    lines = [f"# Live trace report: {d}", "", "```", json.dumps(summary, indent=1), "```", "",
             "| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['stream']} | {r['avatar_id']} | {r['error']} ({r['frames']} frames) |" + " |" * 13)
            continue
        lines.append(f"| {r['stream']} | {r['avatar_id']} | {r['window_s']} | {r['mean_fps']} | {r['anchored_1s_min_frames']} | {r['fixed_1s_bin_min']} | "
                     f"{r['gap_max_ms']} | {r['gaps_over_100ms']} {r['gap_causes'] or ''} | {r['server_fresh_fraction']} | {r['server_held_run_max']}/{r['client_held_run_max']} | "
                     f"{r['client_content_1s_min']} | {r['latency_ms_p50']}/{r['latency_ms_p99']} | {r['pts_join_matched']} | "
                     f"{'PASS' if r['P1_rate'] else 'FAIL'} | {'PASS' if r['P2_fresh'] else 'FAIL'} | {'PASS' if r['P3_no_buffering'] else 'FAIL'} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    if a.md:
        Path(a.md).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
