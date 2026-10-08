#!/usr/bin/env python3
"""Build actual dependencies on a pinned CUDA base without a fabricated release."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import context
import release

# Actual linux/amd64 image manifest from NVIDIA's Docker Hub tag API, 2026-10-08.
# Tag: 12.1.1-cudnn8-devel-ubuntu22.04; index21196d81... is intentionally not confused with its platform digest.
BASE = "nvidia/cuda@sha256:cc55d151af1e8e083f3210af753a5cfbcbc5455421531eb0459887026bb4699f"
TAG_METADATA_URL = "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/12.1.1-cudnn8-devel-ubuntu22.04"
APT_PROBE = r'''set -euo pipefail
apt-get update >&2
for package in "$@"; do
  version=$(apt-cache policy "$package" | awk '/Candidate:/ {print $2; exit}')
  test -n "$version"
  test "$version" != "(none)"
  printf '%s=%s\n' "$package" "$version"
done
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release.require(not args.work.exists(), "Dependency diagnostic output must be new")
    args.work.mkdir(parents=True)
    reports = args.work / "reports"
    reports.mkdir()
    usage = shutil.disk_usage(args.work)
    release.require(usage.free >= 60 * 1024**3, "Less than 60GiB free; replan without destructive runner cleanup")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    release.require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=root), "Commit reviewed source before diagnostic build")
    entries = context.inventory(root)
    source = args.work / "source-context"
    source.mkdir()
    for name in entries:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
    shutil.copyfile(root / ".dockerignore", source / ".dockerignore")
    (reports / "source-inventory.json").write_text(json.dumps({"revision": revision, "source_files": entries}, indent=2) + "\n")
    (reports / "base-identity.json").write_text(json.dumps({"base": BASE, "platform": "linux/amd64", "source": TAG_METADATA_URL,
        "disk_free_before": usage.free, "scope": "Dependency-only; no model payload, GPU acceptance, native engine or publication"}, indent=2) + "\n")
    subprocess.run(["docker", "pull", "--platform", "linux/amd64", BASE], check=True)
    # Resolve exact apt candidate versions on the real base using apt-cache,
    # before build. No credentials, host package install or mutable model inputs.
    output = subprocess.check_output(["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash", BASE,
                                      "-c", APT_PROBE, "apt-pin-probe", *sorted(release.REQUIRED_APT)], text=True)
    pins = output.splitlines()
    release.require(len(pins) == len(release.REQUIRED_APT) and all(release.APT.fullmatch(line) for line in pins), "Invalid apt candidate output")
    release.require({line.split("=", 1)[0] for line in pins} == release.REQUIRED_APT, "Apt pin package set differs")
    pin_dir = args.work / "pins"
    pin_dir.mkdir()
    (pin_dir / "apt-pins.txt").write_text("\n".join(pins) + "\n")
    (reports / "apt-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    image = "musetalk-dependencies-ci:" + revision
    subprocess.run(["docker", "buildx", "build", "--platform", "linux/amd64", "--progress", "plain", "--load",
                    "--build-context", "pins=" + str(pin_dir), "--build-arg", "CUDA_BASE=" + BASE,
                    "--build-arg", "SOURCE_REVISION=" + revision, "--metadata-file", str(reports / "build-metadata.json"),
                    "--file", str(root / "docker/musetalk/Dockerfile.dependencies"), "--tag", image, str(source)], check=True)
    for name, command in (
        ("image-inspect.json", ["docker", "image", "inspect", image]),
        ("image-history.jsonl", ["docker", "history", "--no-trunc", "--format", "{{json .}}", image]),
        ("pip-freeze.txt", ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/opt/musetalk/venv/bin/python", image, "-m", "pip", "freeze"]),
        ("dpkg-packages.txt", ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/usr/bin/dpkg-query", image, "-W"]),
    ):
        with (reports / name).open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = {"schema": "musetalk_dependency_preflight_v1", "status": "PASS", "base": BASE, "source_revision": revision,
              "scope": "Actual GPU-less server+avatar-prep+nativeVP8 dependency build; Kokoro not included",
              "models_present": False, "gpu_tested": False, "published": False, "promotion_eligible": False,
              "disk_free_after": shutil.disk_usage(args.work).free}
    (reports / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
