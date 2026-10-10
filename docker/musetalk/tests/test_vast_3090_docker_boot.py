"""Synthetic copy/paste bootstrap tests. No AWS, registry, Docker or GPU access."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/vast_3090_docker_boot.sh'


class BootTests(unittest.TestCase):
    def run_script(self, *, env=None, args=(), missing=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            runtime = base / 'baked'
            paths = ('release.json', 'app/scripts/vast_docker_onstart.sh', 'venv/bin/python')
            for name in paths:
                path = runtime / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if missing and name == 'release.json':
                    continue
                path.write_text('printf "MODE:%s SETUP:%s SECRET:%s\\n" "$1" "$AUTO_SETUP" "${MUSETALK_AWS_SECRET_ID:-unset}"\n'
                                if name.endswith('.sh') else 'synthetic')
                path.chmod(0o700)
            script = base / 'boot.sh'
            script.write_text(SCRIPT.read_text().replace('/opt/musetalk', str(runtime))
                              .replace('/workspace', str(base / 'state')))
            values = {'PATH': '/usr/bin:/bin', 'AWS_ACCESS_KEY_ID': 'synthetic-id',
                      'AWS_SECRET_ACCESS_KEY': 'synthetic-secret', 'AUTO_SETUP': '1',
                      'MUSETALK_AWS_SECRET_ID': 'stale-secret'}
            values.update(env or {})
            return subprocess.run(['/bin/bash', str(script), *args], env=values,
                                  text=True, capture_output=True)

    def test_injected_config_avoids_second_secret_fetch_and_install(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('MODE:onstart SETUP:0 SECRET:unset', result.stdout)
        self.assertNotIn('synthetic-secret', result.stdout + result.stderr)
        self.assertNotIn('synthetic-id', result.stdout + result.stderr)

    def test_missing_image_or_credentials_fails_without_repair(self):
        for values in ({'missing': True}, {'env': {'AWS_SECRET_ACCESS_KEY': ''}}):
            result = self.run_script(**values)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn('MODE:', result.stdout)

    def test_check_does_not_require_cloud_credentials(self):
        result = self.run_script(args=('check',), env={'AWS_ACCESS_KEY_ID': '', 'AWS_SECRET_ACCESS_KEY': ''})
        self.assertEqual(result.returncode, 0)
        self.assertIn('MODE:check', result.stdout)

    def test_serving_and_secret_reader_modes_are_explicit(self):
        result = self.run_script(args=('serve',), env={'MUSETALK_RUNTIME_CONFIG_SOURCE': 'secretsmanager'})
        self.assertEqual(result.returncode, 0)
        self.assertIn('MODE:serve SETUP:0 SECRET:stale-secret', result.stdout)
        result = self.run_script(env={'MUSETALK_RUNTIME_CONFIG_SOURCE': 'synthetic-sensitive-value'})
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('synthetic-sensitive-value', result.stdout + result.stderr)
        self.assertEqual(self.run_script(args=('unknown',)).returncode, 2)
        self.assertEqual(self.run_script(args=('serve', 'extra')).returncode, 2)

    def test_script_has_no_registry_or_install_commands(self):
        executable = '\n'.join(line for line in SCRIPT.read_text().splitlines()
                               if not line.lstrip().startswith('#'))
        for command in ('git clone', 'apt-get', 'pip install', 'docker login', 'docker pull', 'SETUP_CLEAN=1'):
            self.assertNotIn(command, executable)


if __name__ == '__main__':
    unittest.main()
