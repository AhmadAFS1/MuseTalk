#!/usr/bin/env python3
"""Package, upload, restore, and verify MuseTalk TRT/INT8 artifacts.

The bundle intentionally contains runtime artifacts only, not the whole repo.
It is designed for the Vast on-start path: download one archive from S3,
extract it into the repo, verify checksums, then select the best TRT profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


BUNDLE_MANIFEST = ".musetalk_trt_artifact_manifest.json"
BUNDLE_CHECKSUMS = ".musetalk_trt_artifact_SHA256SUMS"

DEFAULT_REQUIRED_FILES = (
    "models/tensorrt_unet_static_bs8_20260529/unet_trt.ts",
    "models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json",
)
DEFAULT_REQUIRED_DIRS = (
    "calibration/vae_decoder",
    "models/tensorrt/stagewise_int8_onnx_qdq_cache",
)
DEFAULT_OPTIONAL_PATHS = (
    "models/tensorrt_unet_static_bs16_20260704/unet_trt.ts",
    "models/tensorrt_unet_static_bs16_20260704/unet_trt_meta.json",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _iter_files(root: Path, rel_path: str) -> list[Path]:
    path = root / rel_path
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def _validate_unet_meta(root: Path, rel_meta_path: str) -> None:
    meta_path = root / rel_meta_path
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing UNet TRT metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    validation = meta.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise RuntimeError(f"UNet TRT metadata is not validated: {meta_path}")


def _collect_entries(
    root: Path,
    *,
    required_files: list[str],
    required_dirs: list[str],
    optional_paths: list[str],
    strict: bool,
) -> list[dict[str, str | int]]:
    missing: list[str] = []
    entries: dict[str, dict[str, str | int]] = {}

    for rel_path in required_files:
        files = _iter_files(root, rel_path)
        if not files:
            missing.append(rel_path)
            continue
        for file_path in files:
            rel_file = _rel(file_path, root)
            entries[rel_file] = {
                "path": rel_file,
                "sha256": _sha256(file_path),
                "size": file_path.stat().st_size,
            }

    for rel_path in required_dirs:
        files = _iter_files(root, rel_path)
        if not files:
            missing.append(rel_path)
            continue
        for file_path in files:
            rel_file = _rel(file_path, root)
            entries[rel_file] = {
                "path": rel_file,
                "sha256": _sha256(file_path),
                "size": file_path.stat().st_size,
            }

    for rel_path in optional_paths:
        for file_path in _iter_files(root, rel_path):
            rel_file = _rel(file_path, root)
            entries[rel_file] = {
                "path": rel_file,
                "sha256": _sha256(file_path),
                "size": file_path.stat().st_size,
            }

    if missing and strict:
        raise FileNotFoundError(
            "Missing required TRT artifact paths:\n  - " + "\n  - ".join(missing)
        )

    return [entries[key] for key in sorted(entries)]


def _write_sidecars(root: Path, manifest: dict) -> None:
    manifest_path = root / BUNDLE_MANIFEST
    checksums_path = root / BUNDLE_CHECKSUMS
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = [
        f"{entry['sha256']}  {entry['path']}"
        for entry in sorted(manifest["files"], key=lambda item: item["path"])
    ]
    checksums_path.write_text("\n".join(lines) + "\n")


def create_bundle(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    required_files = _parse_csv(args.required_files) or list(DEFAULT_REQUIRED_FILES)
    required_dirs = _parse_csv(args.required_dirs) or list(DEFAULT_REQUIRED_DIRS)
    optional_paths = _parse_csv(args.optional_paths) or list(DEFAULT_OPTIONAL_PATHS)

    entries = _collect_entries(
        root,
        required_files=required_files,
        required_dirs=required_dirs,
        optional_paths=optional_paths,
        strict=args.strict,
    )
    if "models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json" in required_files:
        _validate_unet_meta(root, "models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json")

    manifest = {
        "schema": 1,
        "created_at": _utc_stamp(),
        "profile": args.profile,
        "repo_hint": root.name,
        "required_files": required_files,
        "required_dirs": required_dirs,
        "optional_paths": optional_paths,
        "files": entries,
    }
    _write_sidecars(root, manifest)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for rel_path in [BUNDLE_MANIFEST, BUNDLE_CHECKSUMS]:
            archive.add(root / rel_path, arcname=rel_path)
        for entry in entries:
            rel_path = str(entry["path"])
            archive.add(root / rel_path, arcname=rel_path)

    print(f"Wrote bundle: {output}")
    print(f"Files: {len(entries)}")
    print(f"SHA256: {_sha256(output)}")
    return 0


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client():
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "boto3/botocore are required for S3 artifact access. "
            "Run the TRT setup script first."
        ) from exc
    return boto3.client(
        "s3",
        region_name=(
            os.getenv("TRT_ARTIFACT_S3_REGION")
            or os.getenv("AVATAR_S3_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or None
        ),
        endpoint_url=os.getenv("TRT_ARTIFACT_S3_ENDPOINT_URL") or None,
        config=Config(
            connect_timeout=int(os.getenv("TRT_ARTIFACT_S3_CONNECT_TIMEOUT_SECONDS", "10")),
            read_timeout=int(os.getenv("TRT_ARTIFACT_S3_READ_TIMEOUT_SECONDS", "300")),
            retries={
                "max_attempts": int(os.getenv("TRT_ARTIFACT_S3_RETRY_ATTEMPTS", "3")),
                "mode": os.getenv("TRT_ARTIFACT_S3_RETRY_MODE", "standard"),
            },
        ),
    )


def upload_bundle(args: argparse.Namespace) -> int:
    bundle = args.bundle.resolve()
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle does not exist: {bundle}")
    bucket, key = _parse_s3_uri(args.s3_uri)
    extra_args = {
        "Metadata": {
            "source": "musetalk-trt-artifact-bundle",
            "sha256": _sha256(bundle),
        }
    }
    _s3_client().upload_file(str(bundle), bucket, key, ExtraArgs=extra_args)
    print(f"Uploaded {bundle} to s3://{bucket}/{key}")
    return 0


def _download_to_temp(uri: str) -> Path:
    parsed = urlparse(uri)
    suffix = ".tar.gz"
    fd, raw_path = tempfile.mkstemp(prefix="musetalk-trt-artifact-", suffix=suffix)
    os.close(fd)
    path = Path(raw_path)
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(uri)
        _s3_client().download_file(bucket, key, str(path))
    elif parsed.scheme in {"", "file"}:
        source = Path(parsed.path if parsed.scheme == "file" else uri).expanduser()
        shutil.copyfile(source, path)
    else:
        raise ValueError(f"Unsupported artifact URI scheme: {parsed.scheme}")
    return path


def restore_bundle(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    archive_path = _download_to_temp(args.uri)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Unsafe archive member path: {member.name}")
            archive.extractall(root)
    finally:
        archive_path.unlink(missing_ok=True)

    print(f"Restored TRT artifact bundle into {root}")
    return verify_bundle(args)


def verify_bundle(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    manifest_path = root / BUNDLE_MANIFEST
    if not manifest_path.exists():
        if args.strict:
            raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
        print(f"No artifact manifest found at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    for entry in manifest.get("files", []):
        rel_path = entry["path"]
        path = root / rel_path
        if not path.exists():
            failures.append(f"missing {rel_path}")
            continue
        digest = _sha256(path)
        if digest != entry["sha256"]:
            failures.append(f"checksum mismatch {rel_path}")

    if failures:
        message = "TRT artifact verification failed:\n  - " + "\n  - ".join(failures)
        if args.strict:
            raise RuntimeError(message)
        print(message)
        return 1

    print(f"Verified TRT artifact manifest: {manifest_path}")
    print(f"Files: {len(manifest.get('files', []))}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--strict", action="store_true", help="Fail on missing/invalid artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a tar.gz artifact bundle.")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--profile", default="vae-int8-unet-trt-split8")
    create.add_argument("--required-files", default=",".join(DEFAULT_REQUIRED_FILES))
    create.add_argument("--required-dirs", default=",".join(DEFAULT_REQUIRED_DIRS))
    create.add_argument("--optional-paths", default=",".join(DEFAULT_OPTIONAL_PATHS))

    upload = subparsers.add_parser("upload", help="Upload a bundle to S3.")
    upload.add_argument("--bundle", type=Path, required=True)
    upload.add_argument("--s3-uri", required=True)

    restore = subparsers.add_parser("restore", help="Restore a bundle from S3 or a local file.")
    restore.add_argument("--uri", required=True)

    subparsers.add_parser("verify", help="Verify the restored manifest/checksums.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            return create_bundle(args)
        if args.command == "upload":
            return upload_bundle(args)
        if args.command == "restore":
            return restore_bundle(args)
        if args.command == "verify":
            return verify_bundle(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
