"""Collect the chin_multistream run JSONs in this directory into summary.json (and print a compact table)."""
import json
import statistics
import sys
from pathlib import Path

D = Path(__file__).resolve().parent


def load(name):
    p = D / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def multi_row(r):
    s = r.get("summary", {})
    reps = r.get("repeats", [])
    row = dict(label=r["label"], status=r.get("status"), backend=r["args"]["backend"], streams=r["args"]["streams"],
               loops=r["args"]["loops"], repeats=len(reps), pack=r["args"]["pack"],
               fps_per_repeat=[round(x["aggregate_fps"], 1) for x in reps],
               median_fps=round(s["median_aggregate_fps"], 1) if s.get("median_aggregate_fps") else None,
               wall_s=[round(x["wall_s"], 1) for x in reps],
               timed_ge_60s=all(x["wall_s"] >= 60 for x in reps) if reps else None,
               all_clips_match_accepted=s.get("all_clips_match_accepted"),
               clips_checked=r.get("exactness", {}).get("clips_checked"),
               clips_matching_raw=r.get("exactness", {}).get("clips_matching_raw"),
               clips_matching_faces=r.get("exactness", {}).get("clips_matching_faces"),
               deterministic_per_identity=s.get("deterministic_per_identity"),
               backends={k: r.get("backends", {}).get(k) for k in ("unet_name", "unet_class", "decoder_name")})
    if reps:
        med = lambda f: round(statistics.median(f(x) for x in reps), 3)
        row.update(
            gpu_busy_frac=med(lambda x: x["gpu"]["busy_frac_events"]),
            gpu_ms_per_frame=med(lambda x: x["gpu"]["ms_per_frame"]),
            gpu_ms_per_job=med(lambda x: x["gpu"]["ms_per_job"]["total"]),
            gpu_util_smi_median=med(lambda x: (x["gpu"]["smi"] or {}).get("utilization.gpu", {}).get("median", float("nan"))),
            gpu_thread_sync_wait_frac=med(lambda x: x["gpu_thread"]["sync_wait_ms"] / (x["wall_s"] * 1000)),
            gpu_thread_credit_wait_frac=med(lambda x: x["gpu_thread"]["credit_wait_frac"]),
            worker_idle_frac=med(lambda x: x["workers_mean_idle_frac"]),
            tracker_ms_per_frame=med(lambda x: x["workers_aggregate_ms"]["tracking_ipc_ms"] / x["frames"]),
            facemesh_ms_per_frame=med(lambda x: x["workers_aggregate_ms"]["facemesh_ms"] / x["frames"]),
            compose_ms_per_frame=med(lambda x: x["workers_aggregate_ms"]["compose_ms"] / x["frames"]),
            reset_ms_per_frame=med(lambda x: x["workers_aggregate_ms"]["reset_ms"] / x["frames"]),
            cores_harness=med(lambda x: x["cores"]["harness_total"]),
            cores_parent=med(lambda x: x["cores"]["parent"]),
            cores_workers=med(lambda x: x["cores"]["workers"]),
            cores_facemesh=med(lambda x: x["cores"]["facemesh"]),
            cores_system_busy=med(lambda x: x["cores"]["system_busy"]),
            mem_available_min_gb=min(x["mem_available_gb"]["min_gb"] for x in reps if x["mem_available_gb"]["min_gb"] is not None),
        )
        cmp = [pw.get("compare_accepted") for pw in reps[0]["per_worker"].values() if pw.get("compare_accepted")]
        if cmp:
            row["compare_accepted_first_clip"] = {pw["identity"]: pw["compare_accepted"] for pw in reps[0]["per_worker"].values()}
    if r.get("videos"):
        row["videos"] = r["videos"]
    return row


def main():
    out = dict(schema="chin_multistream_summary_v1", tags="[M] measured (each row from its own run JSON)")
    cp = load("check_prep")
    if cp:
        out["check_prep"] = dict(passes=cp["passes"], subjects={k: dict(raw_match=v["raw_match"], faces_match=v["faces_match"],
                                                                        verbatim=v.get("verbatim", {}).get("verbatim_match"))
                                                               for k, v in cp["subjects"].items()})
    rows = []
    for p in sorted(D.glob("*.json")):
        if p.name in ("summary.json", "check_prep.json"):
            continue
        r = json.loads(p.read_text())
        if r.get("schema") != "chin_multistream_v1":
            continue
        if r["args"]["mode"] == "serial":
            s = r.get("summary", {})
            out.setdefault("serial", []).append(dict(
                label=r["label"], status=r.get("status"), backend=r["args"]["backend"],
                median_fps_over_identities=s.get("median_fps_over_identities"), median_fps_all_runs=s.get("median_fps_all_runs"),
                min_fps=s.get("min_fps"), max_fps=s.get("max_fps"), all_runs_match_accepted=s.get("all_runs_match_accepted"),
                per_identity={k: dict(median_fps=v["median_fps"], runs=[round(x["warm_render_fps"], 1) for x in v["runs"]],
                                      accepted_fps=v["accepted_warm_render_fps"],
                                      match=all(x["raw_match"] and x["faces_match"] for x in v["runs"]))
                              for k, v in r.get("serial", {}).items()}))
        else:
            rows.append(multi_row(r))
    out["multi"] = rows
    (D / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for row in rows:
        print(f"{row['label']:40s} {row['backend']:22s} N={row['streams']:<3d} loops={row['loops']:<3d} fps={row['fps_per_repeat']} "
              f"exact={row['all_clips_match_accepted']} det={row['deterministic_per_identity']} "
              f"gpu_busy={row.get('gpu_busy_frac')} idle={row.get('worker_idle_frac')} cores={row.get('cores_harness')}")
    for s in out.get("serial", []):
        print(f"{s['label']:40s} serial median={s['median_fps_over_identities']} range=({s['min_fps']},{s['max_fps']}) exact={s['all_runs_match_accepted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
