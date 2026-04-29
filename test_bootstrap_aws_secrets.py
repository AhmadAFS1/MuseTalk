import contextlib
import io
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_aws_secrets import (
    _exports_from_payload,
    _verify_s3_buckets,
    _write_export_file,
)


class BootstrapAwsSecretsTest(unittest.TestCase):
    def test_exports_from_payload_adds_runtime_aliases_without_values_in_keys(self):
        with contextlib.redirect_stdout(io.StringIO()):
            exports = _exports_from_payload(
                {
                    "AWS_ACCESS_KEY_ID": "test-access-key",
                    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
                    "AWS_DEFAULT_REGION": "us-east-1",
                    "AVATAR_S3_BUCKET": "lingua-musetalk-s3-storage",
                    "LINGUA_WORKER_DEFAULT_CAPACITY": 1,
                    "invalid-name": "skip-me",
                    "OPTIONAL_NONE": None,
                }
            )

        self.assertEqual(exports["AWS_REGION"], "us-east-1")
        self.assertEqual(exports["AVATAR_S3_REGION"], "us-east-1")
        self.assertEqual(exports["AVATAR_S3_ENABLED"], "1")
        self.assertEqual(exports["LINGUA_WORKER_DEFAULT_CAPACITY"], "1")
        self.assertNotIn("invalid-name", exports)
        self.assertNotIn("OPTIONAL_NONE", exports)

    def test_write_export_file_is_0600_and_shell_quotes_values(self):
        exports = {
            "PLAIN": "value",
            "NEEDS_QUOTING": "value with spaces and ' quote",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret.env"
            _write_export_file(path, exports)

            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, stat.S_IRUSR | stat.S_IWUSR)

            content = path.read_text()
            self.assertIn("export PLAIN=value", content)
            self.assertIn("export NEEDS_QUOTING=", content)

            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"source {shlex.quote(str(path))}; printf %s \"$NEEDS_QUOTING\"",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout, exports["NEEDS_QUOTING"])

    def test_verify_s3_requires_runtime_secret_credentials(self):
        exports = {
            "AVATAR_S3_BUCKET": "lingua-musetalk-s3-storage",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        with self.assertRaisesRegex(RuntimeError, "AWS_ACCESS_KEY_ID"):
            _verify_s3_buckets(exports, "us-east-1")


if __name__ == "__main__":
    unittest.main()
