"""Per-identity quality table across quality_ab_metrics.py runs (candidate vs accepted render), as markdown.
usage: compare_quality.py [--runs-dir DIR] <label> [<label> ...]   e.g. srcmix_taesdtrt srcblkA8 srcv1_taesdtrt
A label is looked up in --runs-dir (default: the published quality_metrics/runs), then in the published runs.
"""
import json, sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "4070s_300fps_impl_20260928/quality_metrics/runs"
IDS = ["black_man_short_beard", "black_woman", "east_asian_man_goatee", "middle_eastern_man_full_beard",
       "south_asian_woman", "white_man_clean_shaven"]


def gate(d, name):
    for g in d["gates"]:
        if g["gate"] == name:
            return g["value"]
    return None


def row(d):
    ps = d["overall"]["psnr"]
    return {"lip_corr": gate(d, "lip.aperture_corr"), "ap_delta_px": gate(d, "lip.mean_abs_delta_px"),
            "flk_mouth": gate(d, "flicker.mouth_ratio"), "flk_jaw": gate(d, "flicker.jaw_ratio"),
            "chin_err_d": d["chin"]["B"]["target_chin_abs_error_px"]["mean"] - d["chin"]["A"]["target_chin_abs_error_px"]["mean"]
            if "B" in d["chin"] else None,
            "lmk_mean": gate(d, "chin.landmark_dev_mean_px"), "lmk_p99": gate(d, "chin.landmark_dev_p99_px"),
            "psnr_face": ps["face_box"]["mean_frame_db_capped"], "psnr_mouth": ps["mouth_roi"]["mean_frame_db_capped"]}


def main():
    args = sys.argv[1:]
    dirs = [RUNS]
    if args[:1] == ["--runs-dir"]:
        dirs = [Path(args[1]), RUNS]
        args = args[2:]
    labels = args
    cols = ["lip_corr", "ap_delta_px", "flk_mouth", "flk_jaw", "chin_err_d", "lmk_mean", "lmk_p99", "psnr_face", "psnr_mouth"]
    print("| identity | run | " + " | ".join(cols) + " |")
    print("|---|---|" + "---:|" * len(cols))
    agg = {l: {c: [] for c in cols} for l in labels}
    for i in IDS:
        for l in labels:
            f = next((d / f"{i}__{l}.json" for d in dirs if (d / f"{i}__{l}.json").exists()), None)
            if f is None:
                continue
            r = row(json.loads(f.read_text()))
            for c in cols:
                if r[c] is not None:
                    agg[l][c].append(r[c])
            print(f"| {i} | {l} | " + " | ".join("-" if r[c] is None else f"{r[c]:.4g}" for c in cols) + " |")
    print("\n| run (range over identities) | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for l in labels:
        print(f"| {l} | " + " | ".join(f"{min(v):.4g}–{max(v):.4g}" if v else "-" for v in agg[l].values()) + " |")


if __name__ == "__main__":
    main()
