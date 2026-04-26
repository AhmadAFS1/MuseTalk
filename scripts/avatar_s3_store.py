import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


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
    ):
        self.enabled = enabled
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.version = version
        self.region = region
        self.endpoint_url = endpoint_url
        self._client = None
        self._warned_unavailable = False

        if not self.enabled:
            return
        if not self.bucket:
            print("⚠️  AVATAR_S3_ENABLED=1 but AVATAR_S3_BUCKET is empty; disabling S3 avatar store.")
            self.enabled = False
            return

        try:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=self.region or None,
                endpoint_url=self.endpoint_url or None,
            )
        except Exception as exc:
            self.enabled = False
            print(f"⚠️  Failed to initialize S3 client for avatar store: {exc}")

    @classmethod
    def from_env(cls, version: str) -> "AvatarS3Store":
        return cls(
            enabled=_env_flag("AVATAR_S3_ENABLED", False),
            bucket=os.getenv("AVATAR_S3_BUCKET", "").strip(),
            prefix=os.getenv("AVATAR_S3_PREFIX", "avatars"),
            version=version,
            region=os.getenv("AWS_REGION", "").strip() or None,
            endpoint_url=os.getenv("AVATAR_S3_ENDPOINT_URL", "").strip() or None,
        )

    def _object_key(self, avatar_id: str) -> str:
        return f"{self.prefix}/{self.version}/{avatar_id}.tar.gz"

    @staticmethod
    def _safe_extract(archive_path: Path, target_dir: Path) -> None:
        with tarfile.open(archive_path, "r:gz") as tar:
            target_dir_resolved = target_dir.resolve()
            for member in tar.getmembers():
                member_path = (target_dir / member.name).resolve()
                if not str(member_path).startswith(str(target_dir_resolved)):
                    raise ValueError(f"Refusing to extract path outside target directory: {member.name}")
            tar.extractall(target_dir)

    def upload_avatar_dir(self, avatar_id: str, avatar_dir: Path) -> bool:
        if not self.enabled or self._client is None:
            return False
        if not avatar_dir.exists():
            return False

        object_key = self._object_key(avatar_id)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            archive_path = Path(tmp_file.name)
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(avatar_dir, arcname=avatar_id)
            self._client.upload_file(str(archive_path), self.bucket, object_key)
            print(f"☁️  Uploaded avatar '{avatar_id}' assets to s3://{self.bucket}/{object_key}")
            return True
        except Exception as exc:
            print(f"⚠️  Failed to upload avatar '{avatar_id}' to S3: {exc}")
            return False
        finally:
            try:
                archive_path.unlink(missing_ok=True)
            except Exception:
                pass

    def download_avatar_dir(self, avatar_id: str, avatars_root: Path) -> bool:
        if not self.enabled or self._client is None:
            return False

        object_key = self._object_key(avatar_id)
        avatars_root.mkdir(parents=True, exist_ok=True)
        target_dir = avatars_root / avatar_id

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            archive_path = Path(tmp_file.name)
        try:
            self._client.download_file(self.bucket, object_key, str(archive_path))
            if target_dir.exists():
                import shutil

                shutil.rmtree(target_dir)
            self._safe_extract(archive_path, avatars_root)
            print(f"☁️  Restored avatar '{avatar_id}' from s3://{self.bucket}/{object_key}")
            return True
        except Exception as exc:
            if not self._warned_unavailable:
                print(f"ℹ️  Avatar '{avatar_id}' not restored from S3 ({exc})")
                self._warned_unavailable = True
            return False
        finally:
            try:
                archive_path.unlink(missing_ok=True)
            except Exception:
                pass
