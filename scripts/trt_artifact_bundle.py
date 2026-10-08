#!/usr/bin/env python3
"""Package, upload, restore, and verify MuseTalk TRT/INT8 artifacts.

The bundle intentionally contains runtime artifacts only, not the whole repo.
It is designed for the Vast on-start path: download one archive from S3,
extract it into the repo, verify checksums, then select the best TRT profile.

Two sidecar layouts:
  * default: the manifest/SHA256SUMS live in the repo root (the tracked RTX 3090
    split8 bundle of recipe legacy_int8; unchanged behaviour);
  * --sidecar-dir DIR: they live in DIR (e.g. .runtime/trt_artifacts/<bundle>),
    next to a restore stamp that binds them to one archive SHA256. Used by the
    pinned bundles in configs/trt_bundles/*.json (recipe r5); restore
    --skip-if-verified then skips the download on reboot, and adopt stamps files
    that are already present without downloading the payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


BUNDLE_MANIFEST = ".musetalk_trt_artifact_manifest.json"
BUNDLE_CHECKSUMS = ".musetalk_trt_artifact_SHA256SUMS"
BUNDLE_SIDECARS = (BUNDLE_MANIFEST, BUNDLE_CHECKSUMS)
# Written next to the sidecars (only with --sidecar-dir) after a restore verified: binds them to one archive.
RESTORE_STAMP = ".musetalk_trt_artifact_restored.json"

DEFAULT_REQUIRED_FILES = (
    "models/tensorrt_unet_static_bs8_20260529/unet_trt.ts",
    "models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json",
)
DEFAULT_REQUIRED_DIRS = (
    "calibration/vae_decoder",
    "models/tensorrt/stagewise_int8_onnx_qdq_cache",
)
DEFAULT_OPTIONAL_PATHS: tuple[str, ...] = ()


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


def _sidecar_dir(args: argparse.Namespace, root: Path) -> Path:
    """Where the manifest/SHA256SUMS live: the repo root (legacy default) or --sidecar-dir."""
    raw = getattr(args, "sidecar_dir", None)
    if raw is None:
        return root
    return (raw if raw.is_absolute() else root / raw).resolve()


def _require_free(path: Path, need: int, label: str) -> None:
    free = shutil.disk_usage(path).free
    if free < need:
        raise RuntimeError(f"Not enough disk for {label} at {path}: need {need} bytes, free {free}")


def _payload_growth(root: Path, members: list) -> int:
    """Bytes the payload adds: regular members minus same-path regular files it overwrites in place."""
    need = 0
    for member in members:
        if not member.isreg():
            continue
        try:
            st = os.lstat(root / member.name)
            existing = st.st_size if stat.S_ISREG(st.st_mode) else 0
        except FileNotFoundError:
            existing = 0
        need += max(0, member.size - existing)
    return need


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


def _add_entry(entries: dict, root: Path, file_path: Path, keep_symlinks: bool) -> None:
    if keep_symlinks and file_path.is_symlink():
        target = Path(os.path.realpath(file_path))
        _rel(target, root)  # ValueError when the link leaves the repo
        link_rel = file_path.relative_to(root).as_posix()
        entries[link_rel] = {
            "path": link_rel,
            "sha256": _sha256(target),
            "size": target.stat().st_size,
            "symlink": os.path.relpath(target, file_path.parent),
        }
        file_path = target
    rel_file = _rel(file_path, root)
    entries[rel_file] = {
        "path": rel_file,
        "sha256": _sha256(file_path),
        "size": file_path.stat().st_size,
    }


def _collect_entries(
    root: Path,
    *,
    required_files: list[str],
    required_dirs: list[str],
    optional_paths: list[str],
    strict: bool,
    keep_symlinks: bool = False,
) -> list[dict[str, str | int]]:
    missing: list[str] = []
    entries: dict[str, dict[str, str | int]] = {}

    for rel_path in required_files:
        files = _iter_files(root, rel_path)
        if not files:
            missing.append(rel_path)
            continue
        for file_path in files:
            _add_entry(entries, root, file_path, keep_symlinks)

    for rel_path in required_dirs:
        files = _iter_files(root, rel_path)
        if not files:
            missing.append(rel_path)
            continue
        for file_path in files:
            _add_entry(entries, root, file_path, keep_symlinks)

    for rel_path in optional_paths:
        for file_path in _iter_files(root, rel_path):
            _add_entry(entries, root, file_path, keep_symlinks)

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
    # argparse already supplies defaults when omitted. An explicit empty string
    # means none, not "restore the defaults" (serving-only bundles rely on this).
    required_files = _parse_csv(args.required_files)
    required_dirs = _parse_csv(args.required_dirs)
    optional_paths = _parse_csv(args.optional_paths)

    entries = _collect_entries(
        root,
        required_files=required_files,
        required_dirs=required_dirs,
        optional_paths=optional_paths,
        strict=args.strict,
        keep_symlinks=args.keep_symlinks,
    )
    if not entries:
        raise ValueError("No TRT artifact payload files selected")
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
    sidecar_dir = _sidecar_dir(args, root)
    if sidecar_dir != root:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    _write_sidecars(sidecar_dir, manifest)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=args.compresslevel) as archive:
        for rel_path in [BUNDLE_MANIFEST, BUNDLE_CHECKSUMS]:
            archive.add(sidecar_dir / rel_path, arcname=rel_path)
        for entry in entries:
            rel_path = str(entry["path"])
            if "symlink" in entry:  # only with --keep-symlinks: a relative link inside the repo
                info = tarfile.TarInfo(rel_path)
                info.type, info.linkname, info.mode = tarfile.SYMTYPE, str(entry["symlink"]), 0o777
                info.mtime = int(datetime.now(timezone.utc).timestamp())
                archive.addfile(info)
            else:
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


def _download_to_temp(uri: str, stage_dir: Path | None = None) -> tuple[Path, str | None]:
    parsed = urlparse(uri)
    suffix = ".tar.gz"
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix="musetalk-trt-artifact-", suffix=suffix,
        dir=str(stage_dir) if stage_dir is not None else None,
    )
    os.close(fd)
    path = Path(raw_path)
    try:
        return path, _fetch_archive(uri, parsed, path, stage_dir)
    except BaseException:
        path.unlink(missing_ok=True)  # never leave the placeholder behind
        raise


def _fetch_archive(uri: str, parsed, path: Path, stage_dir: Path | None) -> str | None:
    remote_sha256 = None
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(uri)
        client = _s3_client()
        head = client.head_object(Bucket=bucket, Key=key)
        remote_sha256 = head.get("Metadata", {}).get("sha256")
        if stage_dir is not None:
            _require_free(stage_dir, int(head.get("ContentLength", 0)), "the archive download")
        client.download_file(bucket, key, str(path))
    elif parsed.scheme in {"", "file"}:
        source = Path(parsed.path if parsed.scheme == "file" else uri).expanduser()
        if stage_dir is not None:
            _require_free(stage_dir, source.stat().st_size, "the archive copy")
        shutil.copyfile(source, path)
    else:
        raise ValueError(f"Unsupported artifact URI scheme: {parsed.scheme}")
    return remote_sha256


def _read_stamp(sidecar_dir: Path) -> dict:
    try:
        return json.loads((sidecar_dir / RESTORE_STAMP).read_text())
    except (OSError, ValueError):
        return {}


def restore_bundle(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    sidecar_dir = _sidecar_dir(args, root)
    expected_sha256 = (
        args.expected_sha256
        or os.getenv("MUSETALK_TRT_ARTIFACT_SHA256", "").strip()
        or None
    )
    if args.skip_if_verified:
        if sidecar_dir == root or not expected_sha256:
            raise ValueError("--skip-if-verified needs --sidecar-dir and an expected archive SHA256")
        stamp = _read_stamp(sidecar_dir)
        if str(stamp.get("archive_sha256", "")).lower() == expected_sha256.lower() and _verify_manifest(
            root, sidecar_dir / BUNDLE_MANIFEST, strict=False
        ) == 0:
            print(f"TRT artifact bundle {expected_sha256} already restored and verified; skipping download")
            return 0
    if sidecar_dir != root:
        (sidecar_dir / RESTORE_STAMP).unlink(missing_ok=True)
    stage_dir = args.stage_dir or (
        Path(os.environ["MUSETALK_TRT_ARTIFACT_STAGE_DIR"])
        if os.getenv("MUSETALK_TRT_ARTIFACT_STAGE_DIR", "").strip() else None
    )
    archive_path, remote_sha256 = _download_to_temp(args.uri, stage_dir)
    try:
        actual_sha256 = _sha256(archive_path)
        for label, digest in (
            ("expected", expected_sha256),
            ("S3 metadata", remote_sha256),
        ):
            if digest and actual_sha256.lower() != digest.lower():
                raise RuntimeError(
                    f"TRT artifact archive SHA256 mismatch ({label}): "
                    f"expected {digest}, got {actual_sha256}"
                )
        print(f"Verified TRT artifact archive SHA256: {actual_sha256}")

        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Unsafe archive member path: {member.name}")
            if sidecar_dir == root:
                archive.extractall(root)  # legacy: sidecars land in the repo root, exactly as before
            else:
                sidecars = [m for m in members if PurePosixPath(m.name).as_posix() in BUNDLE_SIDECARS]
                payload = [m for m in members if m not in sidecars]
                if sorted(PurePosixPath(m.name).as_posix() for m in sidecars) != sorted(BUNDLE_SIDECARS):
                    raise RuntimeError(f"Archive lacks {BUNDLE_SIDECARS} at its root")
                _require_free(root, _payload_growth(root, payload), "the bundle payload")
                archive.extractall(root, members=payload)
                sidecar_dir.mkdir(parents=True, exist_ok=True)
                for member in sidecars:  # written after the payload, never into the repo root
                    data = archive.extractfile(member).read()
                    (sidecar_dir / PurePosixPath(member.name).name).write_bytes(data)
    finally:
        archive_path.unlink(missing_ok=True)

    print(f"Restored TRT artifact bundle into {root}")
    rc = verify_bundle(args)
    if rc == 0 and sidecar_dir != root:
        _write_stamp(sidecar_dir, actual_sha256, args.uri, "restored", "downloaded archive")
    return rc


def _write_stamp(sidecar_dir: Path, archive_sha256: str, uri: str, mode: str, sha_source: str) -> None:
    stamp = {
        "archive_sha256": archive_sha256,
        "archive_sha256_source": sha_source,
        "mode": mode,
        "restored_at": _utc_stamp(),
        "uri": uri,
    }
    (sidecar_dir / RESTORE_STAMP).write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")


def _read_head_sidecars(uri: str) -> tuple[dict[str, bytes], str | None, str]:
    """(sidecars, archive sha256, where the sha came from) without downloading the payload: the
    sidecars are the archive's first members, so a streamed read stops after a few hundred KB."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(uri)
        client = _s3_client()
        archive_sha256 = client.head_object(Bucket=bucket, Key=key).get("Metadata", {}).get("sha256")
        stream = client.get_object(Bucket=bucket, Key=key)["Body"]
        sha_source = "S3 object metadata (written by upload after hashing the archive)"
    elif parsed.scheme in {"", "file"}:
        source = Path(parsed.path if parsed.scheme == "file" else uri).expanduser()
        archive_sha256 = _sha256(source)
        stream = source.open("rb")
        sha_source = "local archive"
    else:
        raise ValueError(f"Unsupported artifact URI scheme: {parsed.scheme}")
    sidecars: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name).as_posix()
                if name not in BUNDLE_SIDECARS or not member.isreg():
                    break
                sidecars[name] = archive.extractfile(member).read()
                if len(sidecars) == len(BUNDLE_SIDECARS):
                    break
    finally:
        stream.close()
    return sidecars, archive_sha256, sha_source


def adopt_bundle(args: argparse.Namespace) -> int:
    """Bind files already in the repo (built here, or restored earlier without a stamp) to a
    published bundle: read its sidecars from the archive head, verify every local file against
    them, then write the sidecars and the restore stamp. Nothing is downloaded or extracted."""
    root = args.repo_root.resolve()
    sidecar_dir = _sidecar_dir(args, root)
    if sidecar_dir == root:
        raise ValueError("adopt needs --sidecar-dir (the repo-root sidecars belong to the legacy bundle)")
    sidecars, archive_sha256, sha_source = _read_head_sidecars(args.uri)
    if not archive_sha256:
        raise RuntimeError(f"{args.uri} carries no sha256 (S3 metadata missing); use restore instead")
    if archive_sha256.lower() != args.expected_sha256.lower():
        raise RuntimeError(
            f"Archive SHA256 mismatch ({sha_source}): expected {args.expected_sha256}, got {archive_sha256}"
        )
    if sorted(sidecars) != sorted(BUNDLE_SIDECARS):
        raise RuntimeError(f"Archive does not start with {BUNDLE_SIDECARS}; use restore instead")
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / RESTORE_STAMP).unlink(missing_ok=True)
    for name, data in sidecars.items():
        (sidecar_dir / name).write_bytes(data)
    rc = _verify_manifest(root, sidecar_dir / BUNDLE_MANIFEST, args.strict)
    if rc == 0:
        _write_stamp(sidecar_dir, archive_sha256.lower(), args.uri, "adopted", sha_source)
        print(f"Adopted local files as TRT artifact bundle {archive_sha256.lower()}")
    return rc


def verify_bundle(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    return _verify_manifest(root, _sidecar_dir(args, root) / BUNDLE_MANIFEST, args.strict)


def _verify_manifest(root: Path, manifest_path: Path, strict: bool) -> int:
    if not manifest_path.exists():
        if strict:
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
        if strict:
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
    parser.add_argument(
        "--sidecar-dir", type=Path, default=None,
        help=(
            "Directory for the manifest/SHA256SUMS sidecars (relative to --repo-root). "
            "Default: the repo root, which holds the legacy RTX 3090 split8 bundle's sidecars."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a tar.gz artifact bundle.")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--profile", default="vae-int8-unet-trt-split8")
    create.add_argument("--required-files", default=",".join(DEFAULT_REQUIRED_FILES))
    create.add_argument("--required-dirs", default=",".join(DEFAULT_REQUIRED_DIRS))
    create.add_argument("--optional-paths", default=",".join(DEFAULT_OPTIONAL_PATHS),
                        help="Comma-separated optional payload paths; pass an empty string to include none")
    create.add_argument("--keep-symlinks", action="store_true",
                        help="Store in-repo symlinks as relative links (engine-set folders) instead of dropping them.")
    create.add_argument("--compresslevel", type=int, default=9, help="gzip level (default 9; TRT plans barely compress, 1 is ~as small).")

    upload = subparsers.add_parser("upload", help="Upload a bundle to S3.")
    upload.add_argument("--bundle", type=Path, required=True)
    upload.add_argument("--s3-uri", required=True)

    restore = subparsers.add_parser("restore", help="Restore a bundle from S3 or a local file.")
    restore.add_argument("--uri", required=True)
    restore.add_argument(
        "--expected-sha256",
        help=(
            "Expected archive SHA256. Defaults to "
            "MUSETALK_TRT_ARTIFACT_SHA256 when set."
        ),
    )
    restore.add_argument(
        "--stage-dir", type=Path, default=None,
        help=(
            "Where to stage the downloaded archive. Defaults to MUSETALK_TRT_ARTIFACT_STAGE_DIR, "
            "else the system temp dir."
        ),
    )
    restore.add_argument(
        "--skip-if-verified", action="store_true",
        help=(
            "With --sidecar-dir: skip the download when the stored sidecars belong to the expected "
            "archive and every file still verifies."
        ),
    )

    adopt = subparsers.add_parser(
        "adopt",
        help=(
            "With --sidecar-dir: verify files already in the repo against a published bundle's "
            "sidecars (read from the archive head) and stamp them as restored. No payload download."
        ),
    )
    adopt.add_argument("--uri", required=True)
    adopt.add_argument("--expected-sha256", required=True)

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
        if args.command == "adopt":
            return adopt_bundle(args)
        if args.command == "verify":
            return verify_bundle(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
