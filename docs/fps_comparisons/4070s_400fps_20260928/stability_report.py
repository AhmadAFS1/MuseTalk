"""Stability report for a long multi-stream harness run (chin_multistream_render.py --repeats N).

Per timed window (one repeat, >= 60 s): aggregate fps, each stream's fps (min / median / max), GPU clock, power
and temperature, host MemAvailable, worker and FaceMesh RSS. Across windows: aggregate trend, the slowest stream
against a real-time target, per-stream variation, memory growth (leak check), and bit-exact determinism of every
clip per avatar.
usage: stability_report.py <harness json> [--target-fps 20] [--md out.md]
"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--target-fps", type=float, default=20.0)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text())
    summ = d.get("summary", {})
    for k in ("backend", "pack", "deterministic_per_identity", "all_repeats_ge_min_timed_s", "gpu_busy_frac_median"):
        d.setdefault(k, summ.get(k))
    reps = d["repeats"]
    streams = d.get("streams", [])
    n = len(streams) or len(reps[0]["per_worker"])
    lines = [f"# Stability: {Path(a.json).stem}", "",
             f"{n} concurrent streams over {len({s['identity'] for s in streams})} avatars, {len(reps)} timed windows, "
             f"backend {d.get('backend')}, pack {d.get('pack')}.", "",
             "| Window | Wall s | Aggregate fps | Stream fps min / median / max | SM MHz | Power W | GPU °C | GPU mem MiB | MemAvailable min GB | Worker RSS max MiB | FaceMesh RSS max MiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    per_stream = defaultdict(list)
    rss_first, rss_last, fm_first, fm_last = {}, {}, {}, {}
    for r in reps:
        pw = r["per_worker"]
        fps = [pw[k]["fps"] for k in sorted(pw, key=int)]
        for k in pw:
            per_stream[int(k)].append(pw[k]["fps"])
        smi = r["gpu"].get("smi", {})
        rss = {k: pw[k].get("rss_mib", {}).get("RssAnon", 0) for k in pw}
        fm = {k: pw[k].get("facemesh_rss_mib", {}).get("RssAnon", 0) for k in pw}
        if r is reps[0]:
            rss_first, fm_first = rss, fm
        rss_last, fm_last = rss, fm
        mem = r.get("mem_available_gb", {})
        lines.append(f"| {r['repeat']} | {r['wall_s']:.1f} | {r['aggregate_fps']:.1f} | {min(fps):.2f} / {st.median(fps):.2f} / {max(fps):.2f} | "
                     f"{smi.get('clocks.sm', {}).get('median', '-')} | {smi.get('power.draw', {}).get('median', '-')} | "
                     f"{smi.get('temperature.gpu', {}).get('median', '-')} | {smi.get('memory.used', {}).get('max', '-')} | "
                     f"{mem.get('min_gb', float('nan')):.2f} | {max(v.get('VmRSS', 0) for v in [pw[k].get('rss_mib', {}) for k in pw]):.0f} | "
                     f"{max(v.get('VmRSS', 0) for v in [pw[k].get('facemesh_rss_mib', {}) for k in pw]):.0f} |")
    clip_fps, spread = [], []
    for r in reps:
        pw = r["per_worker"]
        done = [pw[k]["done_after_t0_s"] for k in pw]
        spread.append(max(done) - min(done))
        for k in pw:
            t = sorted(c["t_done"] for c in pw[k]["clips"])
            clip_fps += [240.0 / (b - a_) for a_, b in zip(t, t[1:]) if b > a_]
    clip_fps.sort()
    q = lambda p: clip_fps[min(len(clip_fps) - 1, int(p * len(clip_fps)))]
    agg = [r["aggregate_fps"] for r in reps]
    slowest = clip_fps[0] if clip_fps else min(min(v) for v in per_stream.values())
    cv = {s: (st.pstdev(v) / st.mean(v) if len(v) > 1 else 0.0) for s, v in per_stream.items()}
    growth = {k: rss_last[k] - rss_first[k] for k in rss_last}
    fm_growth = {k: fm_last[k] - fm_first[k] for k in fm_last}
    lines += ["", "## Across windows", "",
              f"- Aggregate fps: first {agg[0]:.1f}, last {agg[-1]:.1f}, min {min(agg):.1f}, median {st.median(agg):.1f}, max {max(agg):.1f}.",
              f"- Per-stream fps measured clip by clip (240 frames between consecutive clip completions, {len(clip_fps)} clips): "
              f"min {slowest:.2f}, p1 {q(0.01):.2f}, p5 {q(0.05):.2f}, median {q(0.5):.2f}, max {clip_fps[-1]:.2f}.",
              f"- Slowest clip of any stream: **{slowest:.2f} fps** against a {a.target_fps:.0f} fps real-time target "
              f"({'every stream stayed above it' if slowest >= a.target_fps else 'BELOW the target'}); aggregate needed for "
              f"{n} x {a.target_fps:.0f} fps = {n * a.target_fps:.0f}.",
              f"- Fairness: all streams finish each window within {min(spread):.2f}-{max(spread):.2f} s of each other "
              f"(the GPU issuer serves every stream's frames in one shared bs16 queue).",
              f"- Per-stream fps variation across windows (coefficient of variation): max {100 * max(cv.values()):.2f}%, "
              f"median {100 * st.median(cv.values()):.2f}%.",
              f"- Worker private memory (RssAnon) change, first to last window: {min(growth.values()):+.1f} to {max(growth.values()):+.1f} MiB; "
              f"FaceMesh helper: {min(fm_growth.values()):+.1f} to {max(fm_growth.values()):+.1f} MiB.",
              f"- MemAvailable minimum per window: {min(r.get('mem_available_gb', {}).get('min_gb', 99) for r in reps):.2f}-"
              f"{max(r.get('mem_available_gb', {}).get('min_gb', 0) for r in reps):.2f} GB.",
              f"- Bit-exact determinism (every clip of an avatar hashes identically across streams and loops): "
              f"{d.get('deterministic_per_identity')}.",
              f"- All windows timed >= min: {d.get('all_repeats_ge_min_timed_s')}; GPU busy fraction median {d.get('gpu_busy_frac_median'):.4f}.",
              f"- Run status: {d.get('status')}; shutdown note: {d.get('stop_warning') or 'none'}.",
              ]
    out = "\n".join(lines) + "\n"
    print(out)
    if a.md:
        Path(a.md).write_text(out)


if __name__ == "__main__":
    main()
