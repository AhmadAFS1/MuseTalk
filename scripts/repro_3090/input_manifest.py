#!/usr/bin/env python3
"""Freeze local canonical inputs without downloading or regenerating any source assets."""
import argparse
import json
import os
from pathlib import Path

import report

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--accepted-root", required=True)
    p.add_argument("--model-root", action="append", required=True, help="repeat for ALL runtime model/weight directories")
    p.add_argument("--extra", action="append", default=[], help="extra fixture/audio/recipe files or directories")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        p.error("refusing to replace a frozen manifest; choose a new path")
    sources = [Path(a.corpus), *(Path(a.accepted_root) / name for name in report.IDENTITIES),
               ROOT / "musetalk/utils/blending.py", ROOT / "character_factory/h3_avatar_workflow",
               *map(Path, a.model_root), *map(Path, a.extra)]
    files = set()
    for path in sources:
        report.require(path.exists(), f"missing manifest source: {path}")
        files.update(x.resolve() for x in (path.rglob("*") if path.is_dir() else [path])
                     if x.is_file() and "__pycache__" not in x.parts and x.suffix != ".pyc")
    rows = [{"path": os.path.relpath(x, out.parent), "bytes": x.stat().st_size, "sha256": report.sha256(x)} for x in sorted(files)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "repro_3090_inputs_v1", "files": rows, "s3_objects": []}, indent=2) + "\n")
    print(json.dumps({"manifest": str(out), "files": len(rows), "sha256": report.sha256(out)}))


if __name__ == "__main__":
    main()
