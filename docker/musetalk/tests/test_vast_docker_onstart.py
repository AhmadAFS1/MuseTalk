"""Dedicated Docker startup wrapper; synthetic files, no cloud/GPU access."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/vast_docker_onstart.sh"


class VastDockerOnstartTests(unittest.TestCase):
    def run_fixture(self, arguments=(), *, missing=None, symlink=None):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            runtime = base / "baked"
            names = ("release.json", "app/docker/musetalk/entrypoint.sh",
                     "app/docker/musetalk/supervise.py", "app/scripts/vast_onstart.sh",
                     "venv/bin/python")
            for name in names:
                path = runtime / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == missing:
                    continue
                if name == symlink:
                    target = base / "substitute"
                    target.write_text("synthetic")
                    path.symlink_to(target)
                else:
                    path.write_text('printf "CANONICAL:%s\\n" "$1"\n' if name.endswith("entrypoint.sh")
                                    else "synthetic")
                if name == "venv/bin/python":
                    path.chmod(0o700)
            # Only the test copy substitutes fixed paths. Production has no
            # injected launcher/runtime-root escape hatch.
            script = base / "test.sh"
            script.write_text(SCRIPT.read_text().replace("/opt/musetalk", str(runtime)))
            return subprocess.run(["/bin/bash", str(script), *arguments],
                                  env={"PATH": "/usr/bin:/bin", "AUTO_SETUP": "1"},
                                  text=True, capture_output=True)

    def test_default_executes_canonical_serve_after_entrypoint_marker(self):
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unix_seconds=", result.stdout)
        self.assertIn("action=serve", result.stdout)
        self.assertTrue(result.stdout.endswith("CANONICAL:serve\n"))
        self.assertNotIn("READY", result.stdout)

    def test_allowed_modes_preserved(self):
        for mode in ("serve", "onstart", "check"):
            with self.subTest(mode=mode):
                result = self.run_fixture((mode,))
                self.assertEqual(result.returncode, 0)
                self.assertTrue(result.stdout.endswith("CANONICAL:" + mode + "\n"))

    def test_unknown_or_extra_arguments_fail_without_echoing_input(self):
        for args in (("synthetic-sensitive-value",), ("serve", "synthetic-sensitive-value")):
            result = self.run_fixture(args)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("synthetic-sensitive-value", result.stdout + result.stderr)
            self.assertNotIn("CANONICAL", result.stdout)

    def test_missing_baked_file_never_repairs_or_launches(self):
        for name in ("release.json", "app/docker/musetalk/entrypoint.sh",
                     "app/docker/musetalk/supervise.py", "app/scripts/vast_onstart.sh",
                     "venv/bin/python"):
            with self.subTest(name=name):
                result = self.run_fixture(missing=name)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("CANONICAL", result.stdout)

    def test_symlinked_runtime_metadata_is_rejected(self):
        self.assertEqual(self.run_fixture(symlink="release.json").returncode, 1)

    def test_image_and_tracked_context_select_dedicated_script(self):
        import sys
        sys.path.insert(0, str(ROOT / "docker/musetalk"))
        import context
        self.assertTrue(context.allowed("scripts/vast_docker_onstart.sh"))
        dockerfile = (ROOT / "docker/musetalk/Dockerfile").read_text()
        self.assertIn('"/opt/musetalk/app/scripts/vast_3090_docker_boot.sh"', dockerfile)
        self.assertIn('CMD ["serve"]', dockerfile)
        executable = "\n".join(line for line in SCRIPT.read_text().splitlines()
                               if not line.lstrip().startswith("#"))
        for command in ("apt-get", "pip install", "git clone", "docker login", "curl ", "wget "):
            self.assertNotIn(command, executable)


if __name__ == "__main__":
    unittest.main()
