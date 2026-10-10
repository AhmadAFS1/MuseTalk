#!/usr/bin/env python3
"""Pinned public-release input assembly and build-only CI driver.

No registry publication, cloud credentials, package installation on the host, or
GPU acceptance. Only the exact public GitHub repository below supplies assets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import release

REPOSITORY = "AhmadAFS1/MuseTalk"
METADATA_NAME = "musetalk-docker-metadata.tar.gz"
MAX_METADATA = 32 * 1024**2
MAX_PART = 2 * 1024**3 - 1
MIN_DISK = 60 * 1024**3


def require_inputs(revision, tag, digest, channel):
    release.require(re.fullmatch(r"[0-9a-f]{40}", revision), "Full source commit required")
    release.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", tag), "Unsafe release tag")
    release.require(release.SHA.fullmatch(digest), "Metadata archive SHA-256 required")
    release.require(channel in {"candidate", "validated"}, "Explicit build channel required")


def read_metadata(archive, digest, assets, revision, channel, *, validate_transport=None):
    release.require(archive.stat().st_size <= MAX_METADATA, "Metadata archive too large")
    release.require(release.sha256(archive) == digest, "Metadata archive checksum mismatch")
    names = set()
    total = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = release.relative(member.name)
            if member.isdir():
                release.require(name in {"licenses", "evidence"} or name.startswith(("licenses/", "evidence/")),
                                "Unexpected metadata directory")
                continue
            release.require(member.isreg() and name not in names, "Metadata must contain unique regular files")
            release.require(name == "release.json" or name.startswith(("licenses/", "evidence/")),
                            "Unexpected metadata path")
            release.require(0 <= member.size < 10 * 1024**2, "Metadata member too large")
            total += member.size
            release.require(total <= MAX_METADATA and len(names) < 1000, "Metadata expansion too large")
            names.add(name)
    release.require("release.json" in names, "Metadata has no release manifest")
    release.require(not assets.exists(), "Refusing existing release directory")
    assets.mkdir(parents=True)
    release.extract(archive, assets, names)
    manifest = release.load_manifest(assets / "release.json")
    release.require(manifest["source_revision"] == revision and manifest["status"] == channel,
                    "Metadata source/channel differs from reviewed build request")
    release.require(names == {"release.json", *manifest["notices"], *manifest["evidence"]},
                    "Metadata contains unlisted files or omits notices/evidence")
    for name, entry in manifest["notices"].items():
        release.scan_text(release.check_file(assets, name, entry))
    release.verify_evidence(assets, manifest)
    (validate_transport or transport_parts)(manifest)  # validate before downloads
    return manifest


def transport_parts(manifest):
    """GitHub assets are <2GiB; larger archives use explicitly pinned chunks."""
    configured = manifest.get("github_release_assets", {})
    release.require(isinstance(configured, dict) and set(configured) <= set(manifest["archives"]),
                    "Unrecognized release transport archive")
    result = {}
    for archive, entry in manifest["archives"].items():
        parts = configured.get(archive, [{"name": archive, **entry}])
        release.require(isinstance(parts, list) and 1 <= len(parts) <= 32, "Invalid release part count")
        total = 0
        for index, part in enumerate(parts):
            expected = archive if len(parts) == 1 else f"{archive}.part{index:03d}"
            release.require(part.get("name") == expected, "Unexpected or out-of-order release part name")
            size = part.get("size_bytes")
            release.require(type(size) is int and 0 < size <= MAX_PART, "Release part must be positive and <2GiB")
            release.require(release.SHA.fullmatch(str(part.get("sha256", ""))), "Release part SHA-256 required")
            total += size
        release.require(total == entry.get("size_bytes") and release.SHA.fullmatch(str(entry.get("sha256", ""))),
                        "Release parts do not match canonical archive size/hash contract")
        if len(parts) == 1:
            release.require(parts[0]["sha256"] == entry["sha256"], "Single part/full archive hash differs")
        result[archive] = parts
    return result


def download(tag, name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    release.require(not destination.exists(), "Refusing release asset overwrite")
    subprocess.run(["gh", "release", "download", tag, "--repo", REPOSITORY,
                    "--pattern", name, "--dir", str(directory)], check=True)
    release.require(destination.is_file() and not destination.is_symlink(), "Release asset missing/not regular")
    return destination


def assemble_archives(manifest, tag, assets, scratch, fetch=download):
    for archive, parts in transport_parts(manifest).items():
        destination = assets / archive
        release.require(not destination.exists(), "Refusing archive overwrite")
        with destination.open("xb") as target:
            for part in parts:
                path = fetch(tag, part["name"], scratch)
                release.check_file(scratch, part["name"], part)
                with path.open("rb") as source:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                # Only delete this validated, freshly downloaded task-local part.
                path.unlink()
        release.check_file(assets, archive, manifest["archives"][archive])


def build(root, assets, manifest, work, reports):
    runtime_base = release.selected_runtime_base(manifest)
    source = work / "source-context"
    with (reports / "source-inventory.json").open("x") as output:
        subprocess.run([sys.executable, str(root / "docker/musetalk/context.py"), "--root", str(root),
                        "--manifest", str(assets / "release.json"), "--output", str(source)], stdout=output, check=True)
    python_pin = next(p.split("=", 1)[1] for p in manifest["apt_packages"] if p.startswith("python3="))
    image = "musetalk-r5-ci:" + manifest["status"] + "-" + manifest["source_revision"]
    # No build args, contexts, secret mounts or environment exports can carry CI credentials.
    subprocess.run(["docker", "buildx", "build", "--platform", "linux/amd64", "--progress", "plain",
                    "--build-context", "release=" + str(assets),
                    "--build-arg", "CUDA_BASE=" + manifest["cuda_base"],
                    "--build-arg", "CUDA_RUNTIME_BASE=" + runtime_base,
                    "--build-arg", "BOOTSTRAP_PYTHON_VERSION=" + python_pin,
                    "--build-arg", "SOURCE_REVISION=" + manifest["source_revision"],
                    "--build-arg", "RELEASE_CHANNEL=" + manifest["status"],
                    "--metadata-file", str(reports / "build-metadata.json"),
                    "--file", str(source / "docker/musetalk/Dockerfile"), "--tag", image, "--load", str(source)], check=True)
    for name, command in (
        ("image-inspect.json", ["docker", "image", "inspect", image]),
        ("image-history.jsonl", ["docker", "history", "--no-trunc", "--format", "{{json .}}", image]),
        ("installed-dependencies.json", ["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
                                         "--entrypoint", "/opt/musetalk/venv/bin/python", image,
                                         "docker/musetalk/dependency_inventory.py"]),
        ("cpu-check.log", ["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
                           "--entrypoint", "/bin/bash", image, "/opt/musetalk/app/docker/musetalk/entrypoint.sh", "check"]),
    ):
        with (reports / name).open("x") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    result = {"schema": "musetalk_docker_ci_build_v1", "image": image,
              "source_revision": manifest["source_revision"], "channel": manifest["status"],
              "cuda_build_base": manifest["cuda_base"], "cuda_runtime_base": runtime_base,
              "cpu_build_check": "PASS", "published": False, "promotion_eligible": False,
              "limitation": "No GPU/container acceptance or complete image-layer audit; local runner image is ephemeral"}
    (reports / "build-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--channel", required=True, choices=["candidate", "validated"])
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    require_inputs(args.source_revision, args.release_tag, args.metadata_sha256, args.channel)
    args.root = args.root.resolve()
    release.require(not args.work.exists(), "CI work directory must be new")
    args.work.mkdir(parents=True)
    reports = args.work / "reports"
    reports.mkdir()
    usage = shutil.disk_usage(args.work)
    (reports / "disk-before.json").write_text(json.dumps(dict(zip(("total", "used", "free"), usage))) + "\n")
    release.require(usage.free >= MIN_DISK, "Less than 60GiB free before build; replan instead of host cleanup")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip()
    release.require(revision == args.source_revision, "Checked-out source differs from reviewed request")
    scratch = args.work / "downloads"
    archive = download(args.release_tag, METADATA_NAME, scratch)
    assets = args.work / "release"
    manifest = read_metadata(archive, args.metadata_sha256, assets, revision, args.channel)
    # Descriptor mismatch fails before downloading GB-sized archives.
    release.descriptor(args.root, manifest)
    release.verify_model_contract(args.root, manifest)
    assemble_archives(manifest, args.release_tag, assets, scratch)
    if not args.assemble_only:
        build(args.root, assets, manifest, args.work, reports)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print("CI assembly/build rejected: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
