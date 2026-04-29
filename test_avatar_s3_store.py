import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.avatar_s3_store import AvatarS3Store


class FakeS3Error(Exception):
    def __init__(self, code: str, message: str = "fake s3 error"):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


class FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.upload_failures = []
        self.download_failures = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        if self.upload_failures:
            raise self.upload_failures.pop(0)
        self.objects[(bucket, key)] = {
            "body": Path(filename).read_bytes(),
            "extra_args": ExtraArgs or {},
        }

    def download_file(self, bucket, key, filename):
        if self.download_failures:
            raise self.download_failures.pop(0)
        try:
            body = self.objects[(bucket, key)]["body"]
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey", "missing test object") from exc
        Path(filename).write_bytes(body)


def write_prepared_avatar(root: Path, avatar_id: str, marker: str = "new") -> Path:
    avatar_dir = root / avatar_id
    (avatar_dir / "full_imgs").mkdir(parents=True)
    (avatar_dir / "mask").mkdir()
    (avatar_dir / "vid_output").mkdir()
    (avatar_dir / "avator_info.json").write_text('{"avatar_id": "%s"}' % avatar_id)
    (avatar_dir / "coords.pkl").write_bytes(b"coords")
    (avatar_dir / "mask_coords.pkl").write_bytes(b"mask-coords")
    (avatar_dir / "latents.pt").write_bytes(b"latents")
    (avatar_dir / "full_imgs" / "00000000.png").write_bytes(b"png")
    (avatar_dir / "mask" / "00000000.png").write_bytes(b"png")
    (avatar_dir / "marker.txt").write_text(marker)
    return avatar_dir


def build_malicious_archive() -> bytes:
    payload = b"bad"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("avatar/../evil.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class AvatarS3StoreTest(unittest.TestCase):
    def make_store(self, client: FakeS3Client) -> AvatarS3Store:
        return AvatarS3Store(
            enabled=True,
            bucket="test-bucket",
            prefix="avatars",
            version="v15",
            client=client,
            retry_attempts=2,
            retry_base_delay_seconds=0,
            log_fn=lambda message: None,
        )

    def test_upload_and_atomic_restore_round_trip(self):
        client = FakeS3Client()
        store = self.make_store(client)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = write_prepared_avatar(root, "avatar", marker="from-source")

            self.assertTrue(store.upload_avatar_dir("avatar", str(source_dir)))
            self.assertIn(("test-bucket", "avatars/v15/avatar.tar.gz"), client.objects)

            old_target = root / "avatar"
            old_marker = old_target / "marker.txt"
            old_marker.write_text("old-local-copy")

            self.assertTrue(store.download_avatar_dir("avatar", root))
            self.assertEqual(old_marker.read_text(), "from-source")

            stats = store.get_stats()
            self.assertEqual(stats["upload_successes"], 1)
            self.assertEqual(stats["download_successes"], 1)
            self.assertEqual(stats["restore_replacements"], 1)
            self.assertGreater(stats["bytes_uploaded"], 0)
            self.assertGreater(stats["bytes_downloaded"], 0)

    def test_restore_rejects_unsafe_archive_without_touching_existing_avatar(self):
        client = FakeS3Client()
        store = self.make_store(client)
        client.objects[("test-bucket", "avatars/v15/avatar.tar.gz")] = {
            "body": build_malicious_archive(),
            "extra_args": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_prepared_avatar(root, "avatar", marker="keep-me")

            self.assertFalse(store.download_avatar_dir("avatar", root))
            self.assertEqual((root / "avatar" / "marker.txt").read_text(), "keep-me")
            self.assertFalse((root / "evil.txt").exists())

            stats = store.get_stats()
            self.assertEqual(stats["download_failures"], 1)
            self.assertEqual(stats["download_successes"], 0)

    def test_upload_retries_transient_s3_errors(self):
        client = FakeS3Client()
        client.upload_failures.append(FakeS3Error("SlowDown", "please retry"))
        store = self.make_store(client)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = write_prepared_avatar(root, "avatar")

            self.assertTrue(store.upload_avatar_dir("avatar", source_dir))

            stats = store.get_stats()
            self.assertEqual(stats["upload_attempts"], 1)
            self.assertEqual(stats["upload_retries"], 1)
            self.assertEqual(stats["upload_successes"], 1)


if __name__ == "__main__":
    unittest.main()
