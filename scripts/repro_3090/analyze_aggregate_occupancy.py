"""Read-only diagnosis of six-stream fixed-bs16 aggregate reports, not acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/fps_comparisons/rtx3090_r5_20261008"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(report):
    require(report["schema"] == "chin_multistream_v1", "Wrong report schema")
    args = report["args"]
    require(args["streams"] == 6 and args["pack"] == 16 and args["decode_split"] == 8,
            "Expected unchanged six-stream bs16/bs8 workload")
    require(not args["encode"] and not args["compare_accepted"], "Capture/comparison contaminated timing")
    require(report["backends"]["unet_describe"]["batch"] == 16, "Fixed-bs16 backend not established")
    result = []
    for row in report["repeats"]:
        frames, seconds = row["frames"], row["wall_s"]
        jobs, partial, chunks = (row["gpu"][name] for name in ("jobs", "partial_jobs", "chunks"))
        require(all(type(n) is int and n > 0 for n in (frames, jobs, chunks)), "Invalid frame/job counts")
        require(type(partial) is int and 0 <= partial <= jobs, "Invalid partial count")
        require(chunks == 2 * jobs - partial and frames == 8 * chunks
                and frames == 6 * args["loops"] * 240, "Chunk/job/frame accounting differs")
        require(math.isfinite(seconds) and seconds >= 60, "Invalid sustained duration")
        fps = frames / seconds
        require(abs(fps - row["aggregate_fps"]) < 1e-9, "Shared-wall FPS differs")
        workers = row["per_worker"]
        require(len(workers) == 6 and all(w["frames"] == frames // 6 for w in workers.values()),
                "Missing/uncompleted worker frames")
        worker_rows = []
        for key, worker in workers.items():
            timing = worker["per_frame_ms"]
            worker_rows.append({"stream": key, "identity": worker["identity"],
                                "tracking_ipc_ms_per_frame": timing["tracking_ipc_ms"],
                                "facemesh_service_ms_per_frame": timing["facemesh_ms"],
                                "composition_ms_per_frame": timing["compose_ms"],
                                "worker_idle_fraction": worker["idle_frac"]})
        result.append({"repeat": row["repeat"], "valid_frames": frames, "shared_seconds": seconds,
                       "full_recipe_fps": fps, "unchanged_400fps_gate": "PASS" if fps >= 400 else "FAIL",
                       "gpu_jobs": jobs, "partial_8frame_jobs": partial,
                       "partial_job_fraction": partial / jobs,
                       "useful_fixed_bs16_row_fraction": frames / (jobs * 16),
                       "executed_padding_rows_inferred_from_fixed_batch": jobs * 16 - frames,
                       "credit_wait_fraction": row["gpu_thread"]["credit_wait_frac"],
                       "gpu_busy_fraction_events": row["gpu"]["busy_frac_events"],
                       "workers": worker_rows})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.out.parent.resolve() == BASE / "native" and not args.out.exists()
            and not args.out.is_symlink(), "New diagnostic output under native required")
    files = [BASE / "native/native_v1_aggregate" / f"native_v1_{name}.json" for name in ("T", "SUST")]
    groups = {name: analyze(json.loads(path.read_text())) for name, path in zip(("T", "SUST"), files)}
    require(len(groups["T"]) == 2 and len(groups["SUST"]) == 5, "Expected complete T2/SUST5")
    code = [ROOT / "scripts/chin_multistream/gpu.py", ROOT / "scripts/chin_multistream/worker.py",
            ROOT / "character_factory/h3_avatar_workflow/backend.py", ROOT / "scripts/unet_stagewise_trt.py"]
    environment = BASE / "native/native_v1_aggregate/environment.json"
    data = {"schema": "rtx3090_fixed_batch_occupancy_diagnostic_v1", "status": "DIAGNOSTIC_NOT_ACCEPTANCE",
            "input_files": {p.relative_to(ROOT).as_posix(): digest(p) for p in [*files, environment]},
            "inspected_code_files": {p.relative_to(ROOT).as_posix(): digest(p) for p in code},
            "groups": groups, "native_quality_status": "REJECTED_UNCHANGED", "release_ready": False,
            "observations": [
                "All seven native full-recipe windows remain below the unchanged400FPS target.",
                "Partial jobs contain8valid rows in a fixed16-row UNet; padding rows are not valid completed output.",
                "Worker tracking/composition is serial. FaceMesh service is already included in tracking IPC elapsed time; do not add those two timers.",
                "CPU workers are nearly never idle and credits frequently starve. This identifies a scheduling/CPU investigation, not proof of a speedup or sole physical cause."],
            "next_experiment": "Test bounded overlap of canonical per-stream FaceMesh tracking with prior-frame composition, preserving tracker/frame order, three-tap filter, source hashes, frame counts and original timers/acceptance. First prove exact output against the serial path; keep default off until actual GPU/full-recipe/quality trials pass.",
            "limitations": ["No new GPU run, optimization, thermal attribution or hardware-capacity claim.",
                            "Existing GPU-only388FPS is a different workload and does not establish400full-recipeFPS.",
                            "Graph/decoder numerical non-regression must be fixed independently; scheduling cannot waive it."]}
    with args.out.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": data["status"], "windows": 7,
                      "occupancy_range": [min(r["useful_fixed_bs16_row_fraction"] for rs in groups.values() for r in rs),
                                          max(r["useful_fixed_bs16_row_fraction"] for rs in groups.values() for r in rs)]}))


if __name__ == "__main__":
    main()
