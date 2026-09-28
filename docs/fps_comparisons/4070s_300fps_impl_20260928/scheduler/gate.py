#!/usr/bin/env python
"""CPU-only gates for the scheduler GPU sequence (gpu_sequence.sh).

Reads replay_scheduler_exactness.py JSON, prints one PASS/FAIL line with the
measured numbers, writes gate_<name>.json next to this file, exits 0 on PASS.

  gate.py reproduce NAME A.json:LABEL B.json:LABEL
      two golden runs of the SAME code must be SHA-identical (every job, every
      decoded face, composed BGR frame and yuv420p frame, in order).
  gate.py exact NAME REF.json:LABEL CAND.json:LABEL[+LABEL...] [CAND2.json:...]
      each candidate run must be SHA-identical to the reference run
      (faces skipped by HLS_SKIP_GPU_FOR_RAW are excluded from the face check;
      frames and yuv are always compared).
  gate.py evsync NAME RUN.json:LABEL [--tol 0.02]
      plan item 0.2: CUDA-event UNet and VAE(+D2H) totals within tol of the
      host (sync) totals on a single stream at depth 1.
  gate.py speed NAME BASE.json:LABEL CAND.json:LABEL [--min-ratio 1.0] [--goal-ratio 1.10] [--min-window-s 60]
      unpaced null-sink throughput at every N: candidate/base >= min-ratio.
  gate.py summary
      collects every gate_*.json into gpu_sequence_summary.json (gate names starting
      with diag_ are reported but do not gate).

A LABEL of '*' means every run in the file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.replay_scheduler_exactness import compare  # noqa: E402


def load_runs(spec: str) -> list[dict]:
    path, _, labels = spec.partition(":")
    data = json.loads(Path(path).read_text())
    runs = data.get("runs", [data])
    for run in runs:
        run.setdefault("_file", path)
        run.setdefault("_provenance", data.get("provenance"))
    if not labels or labels == "*":
        return runs
    wanted = labels.split("+")
    by_label = {run.get("label"): run for run in runs}
    missing = [label for label in wanted if label not in by_label]
    if missing:
        raise KeyError(f"labels {missing} not in {path} (has {sorted(by_label)})")
    return [by_label[label] for label in wanted]


def write(name: str, verdict: bool, payload: dict, line: str) -> int:
    payload = {"gate": name, "verdict": "PASS" if verdict else "FAIL", "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **payload}
    (HERE / f"gate_{name}.json").write_text(json.dumps(payload, indent=1))
    print(f"{'PASS' if verdict else 'FAIL'} {name}: {line}", flush=True)
    return 0 if verdict else 1


def run_error(run: dict) -> str | None:
    if run.get("error"):
        return run["error"]
    bad = [k for k, v in (run.get("jobs") or {}).items() if v.get("status") != "completed" or v.get("order_errors")]
    return f"jobs not completed in order: {bad}" if bad else None


def exact_report(ref: dict, cand: dict) -> dict:
    report = compare(ref, cand, ref.get("label"), cand.get("label"))
    jobs = report["jobs"].values()
    return {
        "ref": ref.get("label"), "cand": cand.get("label"), "identical": report["identical"],
        "frames": report["frames_total"],
        "frame_mismatches": sum(j.get("frame_mismatches", 0) for j in jobs),
        "yuv_mismatches": sum(j.get("yuv_mismatches", 0) for j in jobs),
        "faces_compared": sum(j.get("faces_compared", 0) for j in jobs),
        "face_mismatches": sum(j.get("face_mismatches", 0) for j in jobs),
        "raw_no_gpu_faces": sum(j.get("raw_no_gpu_faces_b", 0) for j in jobs),
        "yuv_contract_mismatches": sum(j.get("yuv_contract_mismatches_b", 0) for j in jobs),
        "order_errors": sum(j.get("order_errors", 0) for j in jobs),
        "first_mismatch": {k: v.get("first_frame_mismatch") for k, v in report["jobs"].items()
                           if v.get("first_frame_mismatch") is not None},
        "cand_env": cand.get("env_overrides"), "cand_harness": cand.get("harness_options"),
        "cand_first_frame_latency_s": cand.get("first_frame_latency_s"),
        "cand_wall_s": cand.get("wall_s"), "ref_wall_s": ref.get("wall_s"),
        "cand_capacity_derived": (cand.get("capacity") or {}).get("derived"),
        "cand_pipeline": cand.get("scheduler_pipeline"),
        "cand_errors": run_error(cand), "ref_errors": run_error(ref),
        "per_job": report["jobs"],
    }


def cmd_exact(name: str, ref_spec: str, cand_specs: list[str]) -> int:
    refs = load_runs(ref_spec)
    if len(refs) != 1:
        raise KeyError("the reference must name exactly one run")
    ref = refs[0]
    results = []
    for spec in cand_specs:
        for cand in load_runs(spec):
            results.append(exact_report(ref, cand))
    ok = bool(results) and all(r["identical"] and not r["cand_errors"] and not r["ref_errors"] for r in results)
    line = " ".join(
        f"{r['cand']}={'ok' if r['identical'] and not r['cand_errors'] else 'DIFF'}"
        f"(frames={r['frames']} mism={r['frame_mismatches']} yuv={r['yuv_mismatches']} "
        f"faces={r['faces_compared']}/{r['face_mismatches']} raw={r['raw_no_gpu_faces']})"
        for r in results)
    return write(name, ok, {"ref": {"file": ref["_file"], "label": ref.get("label"),
                                    "provenance": ref.get("_provenance")},
                            "results": results}, f"vs {ref.get('label')}: {line}")


def cmd_evsync(name: str, spec: str, tol: float) -> int:
    runs = load_runs(spec)
    rows, ok = [], bool(runs)
    for run in runs:
        ev = run.get("event_vs_host")
        if not ev:
            rows.append({"label": run.get("label"), "error": "no event_vs_host (HLS_GPU_EVENT_TIMING=1 missing?)"})
            ok = False
            continue
        row_ok = (ev.get("stage_sync_on") and ev.get("depth") == 1
                  and all(ev[s]["rel_diff"] is not None and abs(ev[s]["rel_diff"]) <= tol for s in ("unet", "vae")))
        ok &= bool(row_ok) and not run_error(run)
        rows.append({"label": run.get("label"), "ok": bool(row_ok), **ev})
    line = " ".join(
        f"{r['label']}: unet {r['unet']['event_ms']:.1f}/{r['unet']['host_ms']:.1f}ms ({r['unet']['rel_diff']:+.3%}) "
        f"vae {r['vae']['event_ms']:.1f}/{r['vae']['host_ms']:.1f}ms ({r['vae']['rel_diff']:+.3%}) "
        f"h2d {r['h2d']['event_ms']:.1f}/{r['h2d']['host_ms']:.1f}ms"
        if "unet" in r else f"{r['label']}: {r.get('error')}" for r in rows)
    return write(name, ok, {"tol": tol, "rows": rows}, f"(tol {tol:.1%}) {line}")


def cmd_speed(name: str, base_spec: str, cand_spec: str, min_ratio: float, goal_ratio: float,
              min_window_s: float) -> int:
    base_runs, cand_runs = load_runs(base_spec), load_runs(cand_spec)
    if len(base_runs) != 1:
        raise KeyError("the base must name exactly one run")
    base = {r["n_jobs"]: r for r in base_runs[0].get("speed", [])}
    out, ok = [], True
    for cand_run in cand_runs:
        cand = {r["n_jobs"]: r for r in cand_run.get("speed", [])}
        rows = []
        for n in sorted(set(base) | set(cand)):
            b, c = base.get(n), cand.get(n)
            if b is None or c is None:
                rows.append({"n": n, "error": "missing"})
                ok = False
                continue
            ratio = c["generated_fps"] / b["generated_fps"] if b["generated_fps"] else None
            window_ok = min(b["window_s"], c["window_s"]) >= min_window_s
            row = {"n": n, "base_fps": b["generated_fps"], "cand_fps": c["generated_fps"],
                   "ratio": round(ratio, 4) if ratio else None, "window_ok": window_ok,
                   "base_util": b.get("smi", {}).get("util_pct"), "cand_util": c.get("smi", {}).get("util_pct"),
                   "base_sched_cpu": b.get("scheduler_thread_cpu_cores"),
                   "cand_sched_cpu": c.get("scheduler_thread_cpu_cores"),
                   "cand_capacity": c.get("capacity_window")}
            ok &= bool(ratio is not None and ratio >= min_ratio and window_ok)
            rows.append(row)
        ok &= not run_error_speed(cand_run) and not run_error_speed(base_runs[0])
        out.append({"cand": cand_run.get("label"), "rows": rows,
                    "goal_met": all((r.get("ratio") or 0) >= goal_ratio for r in rows)})
    line = " | ".join(
        f"{o['cand']} vs {base_runs[0].get('label')}: " + " ".join(
            f"N={r['n']} {r.get('base_fps')}->{r.get('cand_fps')} fps (x{r.get('ratio')})" for r in o["rows"])
        + f" goal(x{goal_ratio})={'met' if o['goal_met'] else 'not met'}" for o in out)
    return write(name, ok, {"min_ratio": min_ratio, "goal_ratio": goal_ratio, "min_window_s": min_window_s,
                            "base": base_runs[0].get("label"), "results": out}, line)


def run_error_speed(run: dict) -> str | None:
    return run.get("error")


def cmd_summary() -> int:
    gates = []
    for path in sorted(HERE.glob("gate_*.json")):
        data = json.loads(path.read_text())
        gates.append({"gate": data.get("gate"), "verdict": data.get("verdict"), "checked_at": data.get("checked_at"),
                      "file": path.name})
    gating = [g for g in gates if not str(g["gate"]).startswith("diag_")]
    ok = bool(gating) and all(g["verdict"] == "PASS" for g in gating)
    (HERE / "gpu_sequence_summary.json").write_text(json.dumps({"all_pass": ok, "gates": gates}, indent=1))
    for g in gates:
        tag = " (diagnostic, not gating)" if g not in gating else ""
        print(f"  {g['verdict']:4s} {g['gate']}  ({g['checked_at']}){tag}")
    print(f"{'PASS' if ok else 'FAIL'} summary: {sum(g['verdict'] == 'PASS' for g in gating)}/{len(gating)} "
          "gating gates pass", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reproduce")
    p.add_argument("name")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("exact")
    p.add_argument("name")
    p.add_argument("ref")
    p.add_argument("cands", nargs="+")
    p = sub.add_parser("evsync")
    p.add_argument("name")
    p.add_argument("run")
    p.add_argument("--tol", type=float, default=0.02)
    p = sub.add_parser("speed")
    p.add_argument("name")
    p.add_argument("base")
    p.add_argument("cand")
    p.add_argument("--min-ratio", type=float, default=1.0)
    p.add_argument("--goal-ratio", type=float, default=1.10)
    p.add_argument("--min-window-s", type=float, default=60.0)
    sub.add_parser("summary")
    args = ap.parse_args()
    try:
        return dispatch(args)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, 1, None):
            raise
        name = getattr(args, "name", args.cmd)
        return write(name, False, {"error": f"{type(exc).__name__}: {exc}"},
                     f"could not evaluate ({type(exc).__name__}: {exc})")


def dispatch(args) -> int:
    if args.cmd == "reproduce":
        return cmd_exact(args.name, args.a, [args.b])
    if args.cmd == "exact":
        return cmd_exact(args.name, args.ref, args.cands)
    if args.cmd == "evsync":
        return cmd_evsync(args.name, args.run, args.tol)
    if args.cmd == "speed":
        return cmd_speed(args.name, args.base, args.cand, args.min_ratio, args.goal_ratio, args.min_window_s)
    return cmd_summary()


if __name__ == "__main__":
    sys.exit(main())
