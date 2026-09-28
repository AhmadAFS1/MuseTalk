"""Host-RAM capacity table: 20 sessions x 20 distinct avatars on dedicated 30 GB / 36 GB boxes.

Inputs: ram_attribution.json (measured avatar materials per layout, this directory) and
decode_cost.json (PNG decode CPU). Everything that is not an avatar material comes from
the 300 fps plan (docs/musetalk_4070s_300fps_plan_2026-09-27.md section 3) and is tagged
with its source there: [D] documented measurement, [I] inference, [M] measured, [A] an
assumption made here. Only the avatar rows are measured by this area.

A "distinct avatar" is one identity with its three prepared poses (idle, talking,
smiling): a pose-set session keeps all three loaded (hls_gpu_scheduler
prepared_pose_avatar_ids). The talking-only rows cover sessions without pose sets.

    python capacity_table.py            # prints the table, writes capacity_table.json
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSIONS = 20
AVATARS = 20
BOXES_GB = (30.0, 36.0)

FIXED = {  # GB, independent of the avatar count
    "os_and_system_reserve": (1.0, "[A] dedicated box: kernel, page tables, sshd/systemd; no IDE or other projects"),
    "memavailable_guard": (4.0, "[plan 0.1] MemAvailable floor every run keeps"),
    "server_base": (2.2, "[plan D] process after model load, before sessions"),
    "facemesh_tracker_pool": (6 * 0.2 + SESSIONS * 0.031, "[plan I] 6 procs x 0.2 + 0.031 per stream (chin recipe)"),
    "turn_to_turn_growth": (0.7, "[plan D] 6450 -> 7165 MB over turns"),
}
PER_SESSION = {
    "non_queue": (0.15, "[plan I] 0.43 measured - 0.275 full queue"),
    "runahead_i420_cap100": (0.069, "[plan I] lever 1.9: 100 frames x 0.69 MB"),
}
PER_SESSION_UNCAPPED_QUEUE = (0.275, "[plan I] today's 400-frame run-ahead")
CHIN_MB_PER_SOURCE_FRAME = (1.1, "[plan 3.1 M/I] trimmed SourceMask+RefinedMask per unique source frame")
IDLE_CACHE_GB = (1.0, "WEBRTC_IDLE_FRAME_CACHE_MAX_MB default 1024 (LRU across all clips)")

LAYOUT_ORDER = ("baseline", "mask1_lean", "mask1_lean_maskpng", "png_all")
LAYOUT_FLAGS = {
    "baseline": "(defaults = today)",
    "mask1_lean": "MASK_CHANNELS=1 PLAN_FLOAT_ALPHA=0",
    "mask1_lean_maskpng": "MASK_CHANNELS=1 MASK_STORE=png PLAN_FLOAT_ALPHA=0",
    "png_all": "MASK_CHANNELS=1 MASK_STORE=png FRAME_STORE=png PLAN_FLOAT_ALPHA=0",
}


def main():
    attribution = json.loads((HERE / "ram_attribution.json").read_text())
    decode_path = HERE / "decode_cost.json"
    decode = json.loads(decode_path.read_text()) if decode_path.exists() else None
    png_cores = decode["estimate_300fps"] if decode else None

    fixed_gb = sum(v for v, _ in FIXED.values())
    session_gb = SESSIONS * sum(v for v, _ in PER_SESSION.values())
    session_uncapped_gb = SESSIONS * (PER_SESSION["non_queue"][0] + PER_SESSION_UNCAPPED_QUEUE[0])
    rows = []
    for layout in LAYOUT_ORDER:
        summary = attribution["configs"].get(layout)
        if summary is None:
            continue
        complete = [v for v in summary["per_identity"].values() if v["complete_pose_set"]]
        per_identity_mb = [v["server_equiv_host_mb"] for v in complete]
        talking_mb = [p["server_equiv_host_mb"] for p in summary["per_pose"].values() if p["pose"] == "talking"]
        # unique source frames per identity (layout independent) size the chin assets
        unique_frames = [ident["unique_frames"] for ident in complete]
        chin_identity_gb = CHIN_MB_PER_SOURCE_FRAME[0] * max(unique_frames) / 1024
        for label, values in (("identity (3 poses)", per_identity_mb), ("talking pose only", talking_mb)):
            mean_gb, max_gb = statistics.mean(values) / 1024, max(values) / 1024
            for chin in (False, True):
                if chin and label != "identity (3 poses)":
                    continue
                per_avatar_gb = max_gb + (chin_identity_gb if chin else 0.0)
                per_avatar_mean_gb = mean_gb + (chin_identity_gb if chin else 0.0)
                row = {"layout": layout, "flags": LAYOUT_FLAGS[layout], "unit": label, "chin_assets": chin,
                       "avatar_materials_gb_mean": round(mean_gb, 3), "avatar_materials_gb_max": round(max_gb, 3),
                       "per_avatar_budget_gb": round(per_avatar_gb, 3),
                       "per_avatar_mean_gb": round(per_avatar_mean_gb, 3),
                       "chin_assets_gb_per_identity": round(chin_identity_gb, 3) if chin else 0.0,
                       "extra_cpu_cores_at_300fps": (
                           [png_cores["cores_single_thread_cost"], png_cores["cores_contended_cost"]]
                           if layout == "png_all" and png_cores else 0),
                       "boxes": {}}
                for box in BOXES_GB:
                    for queue_label, sessions in (("runahead_cap100", session_gb), ("uncapped_queue", session_uncapped_gb)):
                        need = fixed_gb + sessions + AVATARS * per_avatar_gb
                        spare = box - need
                        max_avatars = math.floor((box - fixed_gb - sessions) / per_avatar_gb)
                        row["boxes"][f"{int(box)}GB_{queue_label}"] = {
                            "need_gb": round(need, 2), "spare_gb": round(spare, 2), "fits_20x20": spare >= 0,
                            "need_gb_at_mean_identity": round(fixed_gb + sessions + AVATARS * per_avatar_mean_gb, 2),
                            "max_distinct_avatars_at_20_sessions": max(0, max_avatars),
                            "fits_20x20_with_idle_cache": spare - IDLE_CACHE_GB[0] >= 0}
                rows.append(row)
    table = {
        "sessions": SESSIONS, "distinct_avatars": AVATARS, "boxes_gb": BOXES_GB,
        "fixed_gb": {k: {"gb": round(v, 3), "source": s} for k, (v, s) in FIXED.items()},
        "fixed_total_gb": round(fixed_gb, 3),
        "per_session_gb": {k: {"gb": v, "source": s} for k, (v, s) in PER_SESSION.items()},
        "sessions_total_gb_cap100": round(session_gb, 3), "sessions_total_gb_uncapped": round(session_uncapped_gb, 3),
        "chin_assets": {"mb_per_unique_source_frame": CHIN_MB_PER_SOURCE_FRAME[0],
                        "source": CHIN_MB_PER_SOURCE_FRAME[1]},
        "idle_frame_cache_optional_gb": {"gb": IDLE_CACHE_GB[0], "source": IDLE_CACHE_GB[1]},
        "avatar_materials_source": "[M] ram_attribution.json: steady USS per pose minus latents (VRAM in the server), max over complete identities",
        "rows": rows,
    }
    (HERE / "capacity_table.json").write_text(json.dumps(table, indent=1))
    header = (f"{'layout':20s} {'unit':20s} chin  GB/avatar mean-max | need 20x20 mean-max | " +
              " | ".join(f"{int(b)} GB: fits / max avatars" for b in BOXES_GB))
    print(f"fixed {fixed_gb:.2f} GB (incl. 1.0 reserve + 4.0 guard), 20 sessions {session_gb:.2f} GB (run-ahead cap 100)")
    print(header)
    for row in rows:
        cells = []
        for box in BOXES_GB:
            b = row["boxes"][f"{int(box)}GB_runahead_cap100"]
            cells.append(f"{'yes' if b['fits_20x20'] else 'NO ':3s} / {b['max_distinct_avatars_at_20_sessions']:3d}       ")
        first = row["boxes"][f"{int(BOXES_GB[0])}GB_runahead_cap100"]
        print(f"{row['layout']:20s} {row['unit']:20s} {'yes ' if row['chin_assets'] else 'no  '} "
              f"{row['per_avatar_mean_gb']:6.2f}-{row['per_avatar_budget_gb']:<6.2f}    | "
              f"{first['need_gb_at_mean_identity']:5.1f}-{first['need_gb']:<5.1f} GB       | " + " | ".join(cells))
    print("need includes the 1.0 GB OS reserve and the 4.0 GB MemAvailable guard; fits/max use the largest "
          "identity (conservative); +1.0 GB if WEBRTC_IDLE_FRAME_CACHE=1; +4.1 GB with today's uncapped 400-frame run-ahead")
    return 0


if __name__ == "__main__":
    sys.exit(main())
