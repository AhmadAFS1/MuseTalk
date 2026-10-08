#!/usr/bin/env python3
"""Remove one exact Windows builder resource from an ephemeral Docker diagnostic.

Not called by the installer or serving entrypoint. No Linux TensorRT resource,
parser, plugin, bindings, model, engine, or user cache is an eligible target.
CPU import success does not prove GPU deserialization or license clearance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

RELATIVE = "lib/python3.10/site-packages/tensorrt_libs/libnvinfer_builder_resource_win.so.10.3.0"
EXPECTED_BYTES = 1397061088
EXPECTED_SHA = "e74bc4dd3bf4ecee6c8ab8ddbf22646f2de521558e8fdebd94cb8467545485b5"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(venv, distribution=None):
    venv = Path(venv)
    require(venv.is_absolute() and not venv.is_symlink(), "invalid diagnostic venv")
    path = venv / RELATIVE
    require(path.resolve().is_relative_to(venv.resolve()), "resource escaped diagnostic venv")
    for parent in (path, *path.parents):
        if parent == venv:
            break
        require(not parent.is_symlink(), "resource has symlink ancestor")
    require(path.is_file() and path.stat().st_size == EXPECTED_BYTES, "resource missing or size differs")
    distribution = distribution or importlib.metadata.distribution("tensorrt-cu12-libs")
    require(distribution.version == "10.3.0", "wrong TensorRT distribution")
    installed = Path(distribution.locate_file("tensorrt_libs/" + path.name))
    require(installed.resolve() == path.resolve(), "resource is not from selected installed distribution")
    require(sha256(path) == EXPECTED_SHA, "resource digest differs; no removal authorized")
    return path, {"path_relative_to_venv": RELATIVE, "size_bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA,
                  "distribution": "tensorrt-cu12-libs", "version": "10.3.0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux amd64 build only")
    require(args.out == Path("/opt/musetalk/runtime_pruning.json") and not args.out.exists(),
            "new fixed diagnostic report required")
    path, item = inspect(Path("/opt/musetalk/venv"))
    report = {"schema": "musetalk_dependency_pruning_diagnostic_v1", "status": "PLAN_ONLY", "item": item,
              "removed_bytes": 0, "gpu_tested": False, "promotion_eligible": False,
              "wheel_record_modified": False, "public_redistribution_reviewed": False,
              "scope": "One checksum-pinned Windows cross-build resource; all Linux runtime/builder libraries retained"}
    if args.execute:
        path.unlink()
        report.update(status="REMOVED_DIAGNOSTIC_ONLY", removed_bytes=item["size_bytes"])
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "removed_bytes": report["removed_bytes"], "gpu_tested": False}))


if __name__ == "__main__":
    main()
