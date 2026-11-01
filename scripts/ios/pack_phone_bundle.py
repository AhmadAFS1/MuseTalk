#!/usr/bin/env python3
"""Pack an exported phone avatar bundle for distribution.

Input: the `avatar/` directory written by prepare_avatar_bundle.py (--phone-only or full).
Output: <out>/<avatar_id>.tar  (plain, uncompressed ustar: the phone unpacks it with HarnessCore's tar reader,
                                iOS has no zstd and frames/masks are already JPEG/PNG so compression buys nothing)
        <out>/<avatar_id>.json (sidecar: avatar_id, bytes, sha256, frames, fps, width, height, exporter_commit, created_at)

Only avatar.json, frames/, masks/ and latents_f16.bin go in; the harness-only extras (audio clip, reference
windows, check frames) are left out even when present. Entries are written in a fixed order with fixed metadata
so the same input gives the same bytes and sha256.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEP_FILES = ("avatar.json", "latents_f16.bin")
KEEP_DIRS = ("frames", "masks")


def exporter_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644 if path.is_file() else 0o755
    if path.is_file():
        with path.open("rb") as f:
            tar.addfile(info, f)
    else:
        tar.addfile(info)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("avatar_dir", type=Path, help="the avatar/ directory written by prepare_avatar_bundle.py")
    ap.add_argument("--avatar-id", default=None, help="defaults to avatar.json's avatar_id")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = args.avatar_dir
    meta = json.loads((src / "avatar.json").read_text())
    avatar_id = args.avatar_id or meta["avatar_id"]
    for f in KEEP_FILES:
        if not (src / f).is_file():
            print(f"missing {f} in {src}", file=sys.stderr)
            return 2
    for d in KEEP_DIRS:
        if not (src / d).is_dir():
            print(f"missing {d}/ in {src}", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    tar_path = args.out / f"{avatar_id}.tar"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for f in KEEP_FILES:
            add(tar, src / f, f)
        for d in KEEP_DIRS:
            add(tar, src / d, d)
            for p in sorted((src / d).iterdir()):
                if p.is_file():
                    add(tar, p, f"{d}/{p.name}")
    data = buf.getvalue()
    tar_path.write_bytes(data)
    sidecar = {
        "avatar_id": avatar_id,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "frames": int(meta["unique_frames"]),
        "cycle_length": int(meta["cycle_length"]),
        "fps": int(meta["fps"]),
        "width": int(meta["frame_width"]),
        "height": int(meta["frame_height"]),
        "format": "phone-avatar-bundle/1",
        "exporter_commit": exporter_commit(),
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out / f"{avatar_id}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"{tar_path} {len(data) / 1e6:.1f} MB sha256 {sidecar['sha256'][:12]}… frames {sidecar['frames']} fps {sidecar['fps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
