import os
import random
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional


_REQUIRED_AVATAR_FILES = (
    "avator_info.json",
    "coords.pkl",
    "mask_coords.pkl",
    "latents.pt",
)
_REQUIRED_AVATAR_GLOBS = (
    ("full_imgs", "*.png"),
    ("mask", "*.png"),
)
_NOT_FOUND_ERROR_CODES = {
    "404",
    "NoSuchKey",
    "NotFound",
}
_PERMANENT_ERROR_CODES = {
    "AccessDenied",
    "AllAccessDisabled",
    "AuthorizationHeaderMalformed",
    "InvalidAccessKeyId",
    "InvalidBucketName",
    "NoSuchBucket",
    "SignatureDoesNotMatch",
}
_RETRYABLE_ERROR_CODES = {
    "RequestTimeout",
    "RequestTimeoutException",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
    "500",
    "502",
    "503",
    "504",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return int(default)
    try:
        return int(raw_value)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return float(default)
    try:
        return float(raw_value)
    except ValueError:
        return float(default)


class AvatarS3Store:
    """Optional S3 backing store for prepared avatar directories."""

    def __init__(
        self,
        *,
        enabled: bool,
        bucket: str,
        prefix: str,
        version: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        client: Any = None,
        retry_attempts: int = 3,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 4.0,
        retry_mode: str = "standard",
        connect_timeout_seconds: int = 5,
        read_timeout_seconds: int = 120,
        log_fn: Callable[[str], None] = print,
    ):
        self.enabled = enabled
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.version = version
        self.region = region
        self.endpoint_url = endpoint_url
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_base_delay_seconds = max(0.0, float(retry_base_delay_seconds))
        self.retry_max_delay_seconds = max(0.0, float(retry_max_delay_seconds))
        self.retry_mode = retry_mode or "standard"
        self.connect_timeout_seconds = max(1, int(connect_timeout_seconds))
        self.read_timeout_seconds = max(1, int(read_timeout_seconds))
        self._client = client
        self._log_fn = log_fn
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "upload_attempts": 0,
            "upload_successes": 0,
            "upload_failures": 0,
            "upload_retries": 0,
            "upload_skipped_incomplete": 0,
            "download_attempts": 0,
            "download_successes": 0,
            "download_failures": 0,
            "download_misses": 0,
            "download_retries": 0,
            "restore_replacements": 0,
            "bytes_uploaded": 0,
            "bytes_downloaded": 0,
            "last_upload_seconds": None,
            "last_download_seconds": None,
            "last_upload_key": None,
            "last_download_key": None,
            "last_error": None,
        }

        if not self.enabled:
            return
        if not self.bucket:
            self._log("disabled because AVATAR_S3_BUCKET is empty")
            self.enabled = False
            return
        if self._client is None:
            self._client = self._build_client()
        self._log(
            "enabled "
            f"bucket={self.bucket} prefix={self.prefix or '(root)'} "
            f"version={self.version} retry_attempts={self.retry_attempts}"
        )

    @classmethod
    def from_env(cls, version: str) -> "AvatarS3Store":
        return cls(
            enabled=_env_flag("AVATAR_S3_ENABLED", False),
            bucket=os.getenv("AVATAR_S3_BUCKET", "").strip(),
            prefix=os.getenv("AVATAR_S3_PREFIX", "avatars"),
            version=version,
            region=(
                os.getenv("AVATAR_S3_REGION", "").strip()
                or os.getenv("AWS_REGION", "").strip()
                or None
            ),
            endpoint_url=os.getenv("AVATAR_S3_ENDPOINT_URL", "").strip() or None,
            retry_attempts=_env_int("AVATAR_S3_RETRY_ATTEMPTS", 3),
            retry_base_delay_seconds=_env_float("AVATAR_S3_RETRY_BASE_DELAY_SECONDS", 0.5),
            retry_max_delay_seconds=_env_float("AVATAR_S3_RETRY_MAX_DELAY_SECONDS", 4.0),
            retry_mode=os.getenv("AVATAR_S3_RETRY_MODE", "standard").strip() or "standard",
            connect_timeout_seconds=_env_int("AVATAR_S3_CONNECT_TIMEOUT_SECONDS", 5),
            read_timeout_seconds=_env_int("AVATAR_S3_READ_TIMEOUT_SECONDS", 120),
        )

    def _build_client(self):
        import boto3
        from botocore.config import Config

        config = Config(
            connect_timeout=self.connect_timeout_seconds,
            read_timeout=self.read_timeout_seconds,
            retries={
                "max_attempts": self.retry_attempts,
                "mode": self.retry_mode,
            },
        )
        return boto3.client(
            "s3",
            region_name=self.region or None,
            endpoint_url=self.endpoint_url or None,
            config=config,
        )

    def get_stats(self) -> dict:
        with self._metrics_lock:
            metrics = dict(self._metrics)
        return {
            "enabled": self.enabled,
            "bucket": self.bucket if self.enabled else None,
            "prefix": self.prefix,
            "version": self.version,
            "retry_attempts": self.retry_attempts,
            **metrics,
        }

    def _object_key(self, avatar_id: str) -> str:
        self._validate_avatar_id(avatar_id)
        path = f"{self.version}/{avatar_id}.tar.gz"
        if not self.prefix:
            return path
        return f"{self.prefix}/{path}"

    def upload_avatar_dir(self, avatar_id: str, avatar_dir: Path | str) -> bool:
        if not self.enabled or self._client is None:
            return False

        start_time = time.monotonic()
        archive_path = None
        object_key = None
        avatar_dir = Path(avatar_dir)
        try:
            object_key = self._object_key(avatar_id)
            self._validate_prepared_avatar_dir(avatar_dir)

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                archive_path = Path(tmp_file.name)
            file_count = self._create_archive(avatar_id, avatar_dir, archive_path)
            archive_size = archive_path.stat().st_size
            extra_args = {
                "Metadata": {
                    "avatar-id": avatar_id,
                    "musetalk-version": self.version,
                    "source": "musetalk-avatar-s3-store",
                }
            }

            self._metric_add("upload_attempts", 1)
            self._run_with_retries(
                "upload",
                avatar_id,
                object_key,
                lambda: self._client.upload_file(
                    str(archive_path),
                    self.bucket,
                    object_key,
                    ExtraArgs=extra_args,
                ),
            )
            elapsed = time.monotonic() - start_time
            self._record_success(
                "upload",
                object_key,
                archive_size,
                elapsed,
            )
            self._log(
                f"upload success avatar_id={avatar_id} "
                f"key=s3://{self.bucket}/{object_key} "
                f"files={file_count} bytes={archive_size} elapsed={elapsed:.2f}s"
            )
            return True
        except ValueError as exc:
            self._metric_add("upload_skipped_incomplete", 1)
            self._record_error(exc)
            self._log(f"upload skipped avatar_id={avatar_id} error={self._safe_error(exc)}")
            return False
        except Exception as exc:
            self._metric_add("upload_failures", 1)
            self._record_error(exc)
            key_text = f" key=s3://{self.bucket}/{object_key}" if object_key else ""
            self._log(f"upload failed avatar_id={avatar_id}{key_text} error={self._safe_error(exc)}")
            return False
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)

    def download_avatar_dir(self, avatar_id: str, avatars_root: Path | str) -> bool:
        if not self.enabled or self._client is None:
            return False

        start_time = time.monotonic()
        archive_path = None
        stage_parent = None
        object_key = None
        avatars_root = Path(avatars_root)
        target_dir = avatars_root / avatar_id
        try:
            object_key = self._object_key(avatar_id)
            avatars_root.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                archive_path = Path(tmp_file.name)

            self._metric_add("download_attempts", 1)
            try:
                self._run_with_retries(
                    "download",
                    avatar_id,
                    object_key,
                    lambda: self._client.download_file(
                        self.bucket,
                        object_key,
                        str(archive_path),
                    ),
                )
            except Exception as exc:
                if self._is_not_found_error(exc):
                    self._metric_add("download_misses", 1)
                    self._record_error(exc)
                    self._log(
                        f"restore miss avatar_id={avatar_id} "
                        f"key=s3://{self.bucket}/{object_key}"
                    )
                    return False
                raise

            archive_size = archive_path.stat().st_size
            stage_parent, staged_avatar_dir = self._extract_archive_to_staging(
                archive_path,
                avatars_root,
                avatar_id,
            )
            replaced = self._install_staged_avatar(staged_avatar_dir, target_dir)
            if replaced:
                self._metric_add("restore_replacements", 1)

            elapsed = time.monotonic() - start_time
            self._record_success(
                "download",
                object_key,
                archive_size,
                elapsed,
            )
            self._log(
                f"restore success avatar_id={avatar_id} "
                f"key=s3://{self.bucket}/{object_key} "
                f"bytes={archive_size} replaced={replaced} elapsed={elapsed:.2f}s"
            )
            return True
        except Exception as exc:
            self._metric_add("download_failures", 1)
            self._record_error(exc)
            key_text = f" key=s3://{self.bucket}/{object_key}" if object_key else ""
            self._log(f"restore failed avatar_id={avatar_id}{key_text} error={self._safe_error(exc)}")
            return False
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            if stage_parent is not None:
                shutil.rmtree(stage_parent, ignore_errors=True)

    def _create_archive(self, avatar_id: str, avatar_dir: Path, archive_path: Path) -> int:
        file_count = 0
        with tarfile.open(archive_path, "w:gz") as tar:
            root_info = tarfile.TarInfo(avatar_id)
            root_info.type = tarfile.DIRTYPE
            root_info.mode = 0o755
            tar.addfile(root_info)

            for path in sorted(avatar_dir.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Refusing to archive symlink: {path}")
                if not path.is_dir() and not path.is_file():
                    raise ValueError(f"Refusing to archive special file: {path}")

                relative_name = path.relative_to(avatar_dir).as_posix()
                archive_name = f"{avatar_id}/{relative_name}"
                info = tar.gettarinfo(str(path), arcname=archive_name)
                if info.isdir():
                    tar.addfile(info)
                elif info.isfile():
                    with path.open("rb") as source:
                        tar.addfile(info, source)
                    file_count += 1
                else:
                    raise ValueError(f"Refusing to archive unsupported member: {path}")
        return file_count

    def _extract_archive_to_staging(
        self,
        archive_path: Path,
        avatars_root: Path,
        avatar_id: str,
    ) -> tuple[Path, Path]:
        safe_prefix = self._safe_temp_prefix(avatar_id)
        stage_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{safe_prefix}.restore-",
                dir=str(avatars_root),
            )
        )
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar:
                    member_name = self._normalize_member_name(member.name)
                    parts = PurePosixPath(member_name).parts
                    if parts[0] != avatar_id:
                        raise ValueError(
                            "Archive root mismatch: "
                            f"expected {avatar_id!r}, got {parts[0]!r}"
                        )
                    if member.isdir():
                        destination = self._safe_member_destination(stage_parent, member_name)
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise ValueError(
                            f"Refusing to extract unsupported archive member: {member.name}"
                        )

                    destination = self._safe_member_destination(stage_parent, member_name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError(f"Archive member has no file body: {member.name}")
                    with source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    try:
                        destination.chmod(member.mode & 0o777)
                    except OSError:
                        pass

            staged_avatar_dir = stage_parent / avatar_id
            self._validate_prepared_avatar_dir(staged_avatar_dir)
            return stage_parent, staged_avatar_dir
        except Exception:
            shutil.rmtree(stage_parent, ignore_errors=True)
            raise

    def _install_staged_avatar(self, staged_avatar_dir: Path, target_dir: Path) -> bool:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        backup_dir = None
        replaced = target_dir.exists() or target_dir.is_symlink()
        try:
            if replaced:
                backup_dir = target_dir.parent / f".{target_dir.name}.backup-{uuid.uuid4().hex}"
                os.replace(target_dir, backup_dir)
            os.replace(staged_avatar_dir, target_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
                os.replace(backup_dir, target_dir)
            raise
        finally:
            if backup_dir is not None and (backup_dir.exists() or backup_dir.is_symlink()):
                if backup_dir.is_symlink() or backup_dir.is_file():
                    backup_dir.unlink(missing_ok=True)
                else:
                    shutil.rmtree(backup_dir, ignore_errors=True)
        return replaced

    def _run_with_retries(
        self,
        operation: str,
        avatar_id: str,
        object_key: str,
        func: Callable[[], Any],
    ) -> Any:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return func()
            except Exception as exc:
                if (
                    self._is_not_found_error(exc)
                    or attempt >= self.retry_attempts
                    or not self._should_retry(exc)
                ):
                    raise

                self._metric_add(f"{operation}_retries", 1)
                delay = self._retry_delay_seconds(attempt)
                self._log(
                    f"{operation} retry avatar_id={avatar_id} "
                    f"key=s3://{self.bucket}/{object_key} "
                    f"attempt={attempt + 1}/{self.retry_attempts} "
                    f"delay={delay:.2f}s error={self._safe_error(exc)}"
                )
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError(f"{operation} retry loop exhausted for {avatar_id}")

    def _retry_delay_seconds(self, attempt: int) -> float:
        if self.retry_base_delay_seconds <= 0:
            return 0.0
        delay = self.retry_base_delay_seconds * (2 ** max(0, attempt - 1))
        delay = min(delay, self.retry_max_delay_seconds)
        jitter = 0.75 + (random.random() * 0.5)
        return delay * jitter

    @staticmethod
    def _validate_avatar_id(avatar_id: str) -> None:
        path = PurePosixPath(avatar_id)
        if not avatar_id or path.is_absolute():
            raise ValueError(f"Invalid avatar_id for S3 persistence: {avatar_id!r}")
        if len(path.parts) != 1 or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"Invalid avatar_id for S3 persistence: {avatar_id!r}")

    @staticmethod
    def _validate_prepared_avatar_dir(avatar_dir: Path) -> None:
        missing = []
        if not avatar_dir.is_dir():
            missing.append(str(avatar_dir))
        for relative_path in _REQUIRED_AVATAR_FILES:
            if not (avatar_dir / relative_path).is_file():
                missing.append(relative_path)
        for relative_dir, pattern in _REQUIRED_AVATAR_GLOBS:
            if not any((avatar_dir / relative_dir).glob(pattern)):
                missing.append(f"{relative_dir}/{pattern}")
        if missing:
            raise ValueError(
                "avatar directory is incomplete; missing "
                + ", ".join(missing[:8])
            )

    @staticmethod
    def _normalize_member_name(member_name: str) -> str:
        path = PurePosixPath(member_name)
        if path.is_absolute():
            raise ValueError(f"Refusing to extract absolute path: {member_name}")
        parts = path.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Refusing to extract unsafe path: {member_name}")
        return path.as_posix()

    @staticmethod
    def _safe_member_destination(root: Path, member_name: str) -> Path:
        destination = (root / Path(*PurePosixPath(member_name).parts)).resolve()
        root_resolved = root.resolve()
        try:
            destination.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to extract path outside target directory: {member_name}"
            ) from exc
        return destination

    @staticmethod
    def _safe_temp_prefix(avatar_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in avatar_id)
        return cleaned[:48] or "avatar"

    @staticmethod
    def _exception_code(exc: Exception) -> Optional[str]:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return None
        error = response.get("Error")
        if not isinstance(error, dict):
            return None
        code = error.get("Code")
        if code is None:
            return None
        return str(code)

    def _is_not_found_error(self, exc: Exception) -> bool:
        return self._exception_code(exc) in _NOT_FOUND_ERROR_CODES

    def _should_retry(self, exc: Exception) -> bool:
        code = self._exception_code(exc)
        if code in _PERMANENT_ERROR_CODES or code in _NOT_FOUND_ERROR_CODES:
            return False
        if code in _RETRYABLE_ERROR_CODES:
            return True
        if code is not None and code.isdigit():
            return int(code) >= 500
        return True

    def _record_success(
        self,
        operation: str,
        object_key: str,
        byte_count: int,
        elapsed_seconds: float,
    ) -> None:
        with self._metrics_lock:
            self._metrics[f"{operation}_successes"] += 1
            self._metrics[f"bytes_{operation}ed"] += int(byte_count)
            self._metrics[f"last_{operation}_seconds"] = round(elapsed_seconds, 3)
            self._metrics[f"last_{operation}_key"] = object_key
            self._metrics["last_error"] = None

    def _record_error(self, exc: Exception) -> None:
        with self._metrics_lock:
            self._metrics["last_error"] = self._safe_error(exc)

    def _metric_add(self, key: str, amount: int) -> None:
        with self._metrics_lock:
            self._metrics[key] += amount

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = str(exc).replace("\n", " ").strip()
        if not text:
            text = exc.__class__.__name__
        return text[:500]

    def _log(self, message: str) -> None:
        self._log_fn(f"[avatar_s3] {message}")
