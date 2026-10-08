#!/usr/bin/env python3
"""Inventory or export a clean, tracked, allowlisted Docker source context.

Inventory is safe on a dirty worktree and emits facts, never an acceptance verdict.
Export requires a clean commit and a matching validated release manifest.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import release


def allowed(name):
    p = Path(name)
    if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
        return False
    if name in {"LICENSE", "api_server.py", "download_weights.sh"}:
        return True
    if name.startswith("musetalk/"):
        return p.suffix in {".py", ".json", ".txt", ".yaml", ".yml", ".tiktoken", ".npz"}
    if name.startswith("templates/"):
        return len(p.parts) == 2 and p.suffix == ".py"
    if name.startswith("scripts/"):
        return ((len(p.parts) == 2 and p.suffix in {".py", ".sh"})
                or (name.startswith("scripts/lib/") and p.suffix == ".sh")
                or name == "scripts/native_vp8_manifest.json"
                or name.startswith("scripts/native_vp8_licenses/"))
    if name.startswith("requirements/"):
        return p.suffix == ".in" or p.name in {"constraints-cu121.txt", "constraints-chin-tools.txt"}
    if name.startswith("configs/recipes/"):
        return p.suffix == ".env"
    if name.startswith("configs/trt_bundles/"):
        return p.suffix == ".json"
    if name.startswith("docker/musetalk/"):
        return len(p.parts) == 3 and p.name in {"Dockerfile", "entrypoint.sh", "release.py", "context.py", "supervise.py", "validate_image.sh", "README.md"}
    return False


def inventory(root):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    entries = {}
    for name in sorted(n for n in names if allowed(n)):
        path = release.checked_path(root, name)
        release.require(path.is_file(), "Tracked source missing")
        release.scan_text(path)
        entries[name] = {"sha256": release.sha256(path), "size_bytes": path.stat().st_size}
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    entries = inventory(a.root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=a.root))
    if a.output:
        release.require(not dirty, "Commit reviewed changes before exporting a release context")
        release.require(a.manifest, "Release manifest required for export")
        m = release.load_manifest(a.manifest)
        release.require(revision == m["source_revision"] and entries == m["source_files"], "Source manifest mismatch")
        release.require(not a.output.exists(), "Context output already exists; refusing to overwrite")
        a.output.mkdir(parents=True)
        for name in entries:
            target = a.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a.root / name, target)
        shutil.copyfile(a.root / ".dockerignore", a.output / ".dockerignore")
    print(json.dumps({"source_revision": revision, "dirty": dirty, "source_files": entries}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print("Context rejected: " + str(exc), file=sys.stderr)
        sys.exit(1)
