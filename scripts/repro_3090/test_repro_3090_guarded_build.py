"""CPU-only argument/exit contracts; fake lease never launches a GPU process."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class GuardedBuild(unittest.TestCase):
    def invoke(self, enabled, missing=False):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            env = {**os.environ, "MUSETALK_REPRO_OUT": str(out),
                   "MUSETALK_REPRO_GPU_WATCH": str(enabled),
                   "MUSETALK_REPRO_EXPLICIT_PROFILE": "1"}
            script = '''
source "$1/scripts/repro_400fps/lib.sh"
GUARD=fake_guard
fake_guard() { printf '%s\\0' "$@"; return 7; }
if [ "$2" = 1 ]; then REPO="$OUT/no-such-repo"; fi
guarded native_test 14 /actual/python 'script with spaces.py' --value 'two words'
'''
            run = subprocess.run(["bash", "-c", script, "test", str(ROOT), str(int(missing))],
                                 env=env, capture_output=True, text=True)
            log = out / "native_test.log"
            args = log.read_bytes().split(b"\0")[:-1] if log.exists() else []
            return run, [x.decode() for x in args], out

    def test_default_preserves_command_arguments_and_failure(self):
        run, args, _ = self.invoke(0)
        self.assertEqual(run.returncode, 7)
        self.assertEqual(args, ["--min-avail-gb", "14", "--label", "repro_native_test", "--",
                                "/actual/python", "script with spaces.py", "--value", "two words"])

    def test_opt_in_watchdog_inside_single_lease(self):
        run, args, out = self.invoke(1)
        self.assertEqual(run.returncode, 7)
        self.assertEqual(args[:6], ["--min-avail-gb", "14", "--label", "repro_native_test", "--", "python3"])
        self.assertEqual(args[6:10], [str(ROOT / "scripts/repro_3090/watch.py"), "--out",
                                      str(out / "native_test.gpu_watch.jsonl"), "--"])
        self.assertEqual(args[10:], ["/actual/python", "script with spaces.py", "--value", "two words"])

    def test_missing_watchdog_fails_before_lease(self):
        run, args, _ = self.invoke(1, missing=True)
        self.assertEqual(run.returncode, 1)
        self.assertIn("GPU watchdog missing", run.stdout)
        self.assertEqual(args, [])


if __name__ == "__main__":
    unittest.main()
