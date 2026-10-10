#!/usr/bin/env python3
"""Small public pinned-base audit reproduction; no registry login or publication."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

import ghcr
import release


def archive_layout(path):
    """Record structure/compression, never exported file contents or metadata env."""
    layers = []
    with tarfile.open(path) as exported:
        manifests = json.load(exported.extractfile("manifest.json"))
        release.require(len(manifests) == 1, "Diagnostic needs exactly one exported image")
        for name in manifests[0]["Layers"]:
            release.relative(name)
            member = exported.getmember(name)
            with exported.extractfile(member) as content:
                magic = content.read(4)
            encoding = "gzip" if magic[:2] == b"\x1f\x8b" else "zstd" if magic == b"\x28\xb5\x2f\xfd" else "tar-or-other"
            layers.append({"member": name, "size_bytes": member.size, "encoding": encoding})
    return {"schema": "musetalk_docker_export_structure_v1", "layers": layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    release.require(not args.work.exists(), "Diagnostic work directory must be new")
    args.work.mkdir(parents=True)
    release.require(shutil.disk_usage(args.work).free >= 20 * 1024**3, "Less than 20GiB diagnostic disk")
    ghcr.stage("diagnostic-base-pull")
    ghcr.command(["docker", "pull", "--platform", "linux/amd64", release.CUDA_RUNTIME_BASE])
    audit_work = args.work / "runtime-base-audit"
    try:
        result = ghcr.audit(release.CUDA_RUNTIME_BASE, audit_work)
    except Exception as exc:
        ghcr.report_failure(exc, audit_work)
        archive = audit_work / "image-layer-audit.tar"
        if archive.exists():
            try:
                (audit_work / "export-layout.json").write_text(json.dumps(archive_layout(archive), indent=2) + "\n")
            except Exception as layout_error:
                ghcr.report_failure(layout_error, args.work / "layout-diagnostic")
        raise SystemExit(1)
    result.update({"diagnostic": "pinned-runtime-base-only", "published": False, "serving_accepted": False})
    (args.work / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ghcr.report_failure(exc)
        raise SystemExit(1)
