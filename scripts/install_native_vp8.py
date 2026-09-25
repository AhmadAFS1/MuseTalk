#!/usr/bin/env python3
"""Install the pinned VP8 encoder files alongside, without replacing aiortc."""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("native_vp8_manifest.json")
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / ".runtime" / "native_vp8"


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported native VP8 manifest version")
    return manifest


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in {"..", "."} for p in path.parts):
        raise RuntimeError(f"Unsafe native VP8 archive path: {value}")
    return path


def validate_install(directory, manifest=None):
    """Hash-check every executable and provenance file before native loading."""
    manifest = load_manifest() if manifest is None else manifest
    directory = Path(directory)
    for item in manifest["extracted"] + manifest.get("bundled_notices", []):
        path = directory / relative_path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Native VP8 file missing or wrong size: {path}")
        if digest(path.read_bytes()) != item["sha256"]:
            raise RuntimeError(f"Native VP8 hash mismatch: {path}")
    receipt_path = directory / "installation.json"
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Native VP8 installation receipt is missing or invalid: {receipt_path}") from exc
    if receipt.get("wheel_sha256") != manifest["wheel_sha256"]:
        raise RuntimeError("Native VP8 installation receipt names another wheel")
    return {"directory": str(directory.resolve()), "wheel_sha256": manifest["wheel_sha256"],
            "file_count": len(manifest["extracted"]) + len(manifest.get("bundled_notices", []))}


def install(directory, wheel_path=None, *, manifest=None):
    manifest = load_manifest() if manifest is None else manifest
    directory = Path(directory).expanduser().absolute()
    if directory.exists():
        return {**validate_install(directory, manifest), "reused": True}
    directory.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(manifest["wheel_size_bytes"])
    if wheel_path is not None:
        wheel_path = Path(wheel_path)
        if wheel_path.stat().st_size != expected_size:
            raise RuntimeError("Native VP8 wheel size differs from the pinned artifact")
        payload = wheel_path.read_bytes()
    else:
        with urllib.request.urlopen(manifest["wheel_url"], timeout=30) as response:
            payload = response.read(expected_size + 1)
    if len(payload) != expected_size or digest(payload) != manifest["wheel_sha256"]:
        raise RuntimeError("Native VP8 wheel failed the pinned size/SHA256 check")
    with tempfile.TemporaryDirectory(prefix=".native-vp8-install-", dir=directory.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "pinned.whl"
        archive_path.write_bytes(payload)
        staged = temporary / "payload"
        staged.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for item in manifest["extracted"]:
                relative = relative_path(item["path"])
                info = archive.getinfo(item["path"])
                if info.file_size != item["bytes"]:
                    raise RuntimeError(f"Native VP8 archive size mismatch: {relative}")
                data = archive.read(info)
                if digest(data) != item["sha256"]:
                    raise RuntimeError(f"Native VP8 archive hash mismatch: {relative}")
                destination = staged / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        for item in manifest.get("bundled_notices", []):
            source = MANIFEST_PATH.parent / relative_path(item["source"])
            data = source.read_bytes()
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise RuntimeError(f"Native VP8 bundled notice hash mismatch: {source}")
            destination = staged / relative_path(item["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        (staged / "installation.json").write_text(json.dumps({
            "schema_version": 1, "package": manifest["package"], "version": manifest["version"],
            "wheel_url": manifest["wheel_url"], "wheel_sha256": manifest["wheel_sha256"],
            "license": manifest["license"], "purpose": "VP8 encoder only; installed aiortc unchanged",
        }, indent=2) + "\n")
        validate_install(staged, manifest)
        if directory.exists():
            validate_install(directory, manifest)
        else:
            os.rename(staged, directory)
    return {**validate_install(directory, manifest), "reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--wheel", type=Path, help="Use the exact pinned wheel already downloaded")
    parser.add_argument("--verify", action="store_true", help="Verify an existing installation without downloads")
    args = parser.parse_args()
    result = validate_install(args.directory) if args.verify else install(args.directory, args.wheel)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
