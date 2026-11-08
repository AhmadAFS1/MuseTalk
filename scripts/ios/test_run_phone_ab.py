"""Offline phone-run safety regressions. All devicectl calls use a local mock; no phone access."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).with_name("run_phone_ab.sh")
TEXT = SOURCE.read_text()
MARK = "echo \"== $(date '+%H:%M:%S') order $ORDER;"
# Split before the first device action; exercise production functions and the actual retry/final-validation loops.
START = TEXT.index(MARK)
PREFIX, BODY = TEXT[:START], TEXT[START:]
MOCK = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ["MOCK_STATE"])
if args[:4] == ["devicectl", "device", "info", "processes"]:
    if os.environ.get("MOCK_FAIL"):
        sys.exit(1)
    out = Path(args[args.index("--json-output") + 1])
    out.write_text(state.read_text())
elif args[:4] == ["devicectl", "device", "process", "terminate"]:
    pid = int(args[args.index("--pid") + 1])
    data = json.loads(state.read_text())
    with open(os.environ["MOCK_TERMINATIONS"], "a") as f:
        f.write(str(pid) + "\\n")
    data["result"]["runningProcesses"] = [r for r in data["result"]["runningProcesses"] if r["processIdentifier"] != pid]
    state.write_text(json.dumps(data))
else:
    sys.exit("unexpected mock command: " + repr(args))
'''


def row(exe, pid):
    return {"executable": "file:///" + exe, "processIdentifier": pid}


class PhoneSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.mock = self.bin / "xcrun"
        self.mock.write_text(MOCK)
        self.mock.chmod(0o755)
        self.state = self.root / "state.json"
        self.terms = self.root / "terminated.txt"
        self.env = {**os.environ, "PATH": str(self.bin) + ":" + os.environ["PATH"],
                    "MOCK_STATE": str(self.state), "MOCK_TERMINATIONS": str(self.terms)}
        self.set_rows([row("sbin/launchd", 1)])

    def set_rows(self, rows):
        self.state.write_text(json.dumps({"result": {"runningProcesses": rows}}))

    def run_shell(self, suffix, body=""):
        script = self.root / "test.zsh"
        script.write_text(PREFIX + suffix + body)
        return subprocess.run(["zsh", str(script), str(self.root / "out"), "D S S D",
                               "D=--unet modelD --infer-workers 2", "S=--unet modelS --infer-workers 2"],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def test_shared_phone_query_fails_closed(self):
        self.assertEqual(self.run_shell('\nprocess_state\n').stdout.strip(), "free")
        self.set_rows([row("sbin/launchd", 1), row("tmp/SoulXStudio.app/SoulXStudio", 12)])
        self.assertEqual(self.run_shell('\nprocess_state\n').stdout.strip(), "busy")
        self.env["MOCK_FAIL"] = "1"
        self.assertNotEqual(self.run_shell('\nprocess_state\n').returncode, 0)
        del self.env["MOCK_FAIL"]
        self.state.write_text('{"result": {"runningProcesses": []}}')
        self.assertNotEqual(self.run_shell('\nprocess_state\n').returncode, 0)

    def test_terminate_only_exact_harness_and_verify_gone(self):
        self.set_rows([row("sbin/launchd", 1), row("tmp/SoulXStudio.app/SoulXStudio", 12),
                       row("tmp/MuseTalkService", 13), row("tmp/MuseTalkBench.app/MuseTalkBench", 14)])
        result = self.run_shell('\nstop_musetalk\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.terms.read_text(), "14\n")
        self.assertEqual([r["processIdentifier"] for r in json.loads(self.state.read_text())["result"]["runningProcesses"]],
                         [1, 12, 13])

    def test_watchdog_stops_harness_when_soulx_appears(self):
        self.set_rows([row("sbin/launchd", 1), row("tmp/SoulXStudio.app/SoulXStudio", 12),
                       row("tmp/MuseTalkBench.app/MuseTalkBench", 14)])
        result = self.run_shell("\nlaunch() { command sleep 1; }\nsleep() { command sleep .01; }\nguarded \"$OUT/guard.log\" 10 --lipsync\n")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.terms.read_text(), "14\n")
        self.assertIn("SoulX app appeared", (self.root / "out/summary.txt").read_text())

    def workflow(self, fail=False, mismatch=False):
        sample = {"fps": 25, "frames": 444, "avatar": "heldout", "seconds": 17.76, "generated": 444,
                  "generated_fps": 25., "shown": 442, "stage_infer_ms": 75., "infer_workers": 2,
                  "skipped": 0, "dropped_late": 0, "stalls": 0}
        d = json.dumps({**sample, "unet": "modelD"})
        s = json.dumps({**sample, "unet": "wrong" if mismatch else "modelS"})
        mock = '''
wait_ready() { :; }
sleep() { :; }
guarded() {
  local log=$1
  if [[ "$*" == *--autorun* ]]; then
    print -r -- '[bench] 50.0 ms/frame mean' '[bench] 35.0 ms/frame mean' '[bench] thermal nominal' > "$log"
  elif [[ "$*" == *--unet\\ modelD* ]]; then
    print -r -- 'LIPSYNC_RESULT %s' > "$log"
  else
    print -r -- 'LIPSYNC_RESULT %s' > "$log"
  fi
}
''' % (d, s)
        if fail:
            mock = mock.replace("'LIPSYNC_RESULT " + d + "'", "'no result'")
        return self.run_shell(mock, BODY)

    def test_exhausted_retries_fail_and_do_not_claim_done(self):
        result = self.workflow(fail=True)
        self.assertEqual(result.returncode, 6, result.stderr)
        summary = (self.root / "out/summary.txt").read_text()
        self.assertIn("failed after 3 attempts", summary)
        self.assertNotIn("verified 4", summary)
        self.assertNotIn(" done", summary)

    def test_all_results_models_workers_and_order_verified(self):
        result = self.workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified 4 ordered results", (self.root / "out/summary.txt").read_text())
        result = self.workflow(mismatch=True)
        self.assertEqual(result.returncode, 6, result.stderr)


if __name__ == "__main__":
    unittest.main()
