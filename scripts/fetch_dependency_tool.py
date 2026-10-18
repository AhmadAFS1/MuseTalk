#!/usr/bin/env python3
"""Obtain a pinned uv executable from its verified official Linux wheel, no pip."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import time
import zipfile

from dependency_artifacts import require
from fetch_dependency_wheels import acquire_one, sha256_file

TOOL = {"name": "uv", "version": "0.13.1",
        "filename": "uv-0.13.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "size_bytes": 17279385,
        "sha256": "696dcf6d807d85da978f4db8472da81dccc1254b18b52cbd1367c4d923c421e0",
        "source": "https://files.pythonhosted.org/packages/14/0b/6602a613533602e7a083e908c4375f9b0e78a94ba2e21c29b21c41a0a611/uv-0.13.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"}


def fetch(root):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Pinned Linux tool only")
    root = Path(root)
    require(not root.exists() and not root.is_symlink(), "Tool directory already claimed")
    start = time.monotonic()
    root.mkdir(mode=0o700, parents=True)
    acquisition = acquire_one(TOOL, root)
    with zipfile.ZipFile(root / TOOL["filename"]) as archive:
        matches = [m for m in archive.infolist() if PurePosixPath(m.filename).name == "uv" and not m.is_dir()]
        require(len(matches) == 1, "Ambiguous uv executable")
        member = matches[0]
        require(not member.filename.startswith("/") and ".." not in PurePosixPath(member.filename).parts
                and not stat.S_ISLNK(member.external_attr >> 16) and 0 < member.file_size <= 100 * 1024**2,
                "Unsafe uv executable member")
        payload = archive.read(member)
    binary = root / "uv"
    with binary.open("xb") as out:
        out.write(payload)
    os.chmod(binary, 0o700)
    version = subprocess.check_output([str(binary), "--version"], text=True, timeout=10).strip()
    require(version.split()[:2] == ["uv", TOOL["version"]], "Downloaded tool version differs")
    result = {"state": "verified-tool", "version": version, "binary_sha256": sha256_file(binary),
              "wheel_sha256": TOOL["sha256"], "seconds": time.monotonic()-start, "acquisition": acquisition,
              "tool_bootstrap_must_be_charged_to_fresh_delivery": True}
    (root / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(fetch(p.parse_args().output)), flush=True)
