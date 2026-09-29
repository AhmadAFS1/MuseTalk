#!/usr/bin/env python
"""Compare two WEBRTC_PREENCODE_SHA_DIR taps (E0 gate for 1.5 / 1.7 / 1.11).

Each tap directory holds one preencode_<pid>_<track>.jsonl per track with one
line per live frame that entered the track queue: {"g": generation, "i": index
in turn, "sha256": packed-I420 SHA-256, "gf": generation frame}. For an N=1 run
with the same WAVs and a pinned idle phase (WEBRTC_TEST_IDLE_SYNC_SOURCE_FRAME),
baseline and candidate must produce the same sequence of turns and, per turn,
the same frame count and SHA sequence.

  compare_preencode_sha.py BASE_DIR CAND_DIR --out result.json
Prints PASS/FAIL with counts; exit 0 only on PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(directory: Path) -> list:
    """Turns (ordered by generation) of every track, tracks ordered by first stamp."""
    tracks = []
    for path in sorted(directory.glob("preencode_*.jsonl")):
        turns: dict = {}
        first_t = None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            first_t = entry["t"] if first_t is None else min(first_t, entry["t"])
            turns.setdefault(entry["g"], []).append((entry["i"], entry["sha256"], entry.get("gf")))
        ordered = [sorted(frames) for _g, frames in sorted(turns.items())]
        if ordered:
            tracks.append((first_t, path.name, ordered))
    tracks.sort()
    return tracks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base, cand = load(Path(args.base)), load(Path(args.cand))
    report = {"base_dir": args.base, "cand_dir": args.cand, "base_tracks": len(base),
              "cand_tracks": len(cand), "turns": [], "passed": False}
    ok = len(base) == len(cand) and len(base) > 0
    for (_, bname, bturns), (_, cname, cturns) in zip(base, cand):
        if len(bturns) != len(cturns):
            ok = False
        for index, (bt, ct) in enumerate(zip(bturns, cturns)):
            bsha = [x[1] for x in bt]
            csha = [x[1] for x in ct]
            first_diff = next((k for k, (a, b) in enumerate(zip(bsha, csha)) if a != b), None)
            equal = bsha == csha
            ok = ok and equal
            report["turns"].append({"base_track": bname, "cand_track": cname, "turn": index,
                                    "base_frames": len(bsha), "cand_frames": len(csha),
                                    "identical": equal, "first_diff_index": first_diff})
    frames = sum(t["base_frames"] for t in report["turns"])
    report["frames_compared"] = frames
    report["passed"] = bool(ok and frames > 0)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"{'PASS' if report['passed'] else 'FAIL'} preencode I420 SHA: tracks {len(base)}/{len(cand)}, "
          f"turns {len(report['turns'])}, frames {frames}, identical_turns "
          f"{sum(t['identical'] for t in report['turns'])}/{len(report['turns'])}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
