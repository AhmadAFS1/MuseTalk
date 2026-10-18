"""Installer command/timer tests: stdlib + Bash only, never pip, venv, apt or CUDA.

Run with any host Python >= 3.8: python3 -B -m unittest test_install_package_timing -v
Only selected function definitions are loaded, not the installer entry point. The
venv interpreter is a shell function that records arguments and advances a fake
clock; no Python dependency installation or resolution is attempted.
"""

from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
INSTALLER = (ROOT / "scripts/install_musetalk.sh").read_text()
LOGGING = (ROOT / "scripts/lib/step_logging.sh").read_text()


def function(source, name):
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}", source)
    if not match:
        raise AssertionError(f"Cannot find function {name}")
    return match.group(0)


class PackageTimingTests(unittest.TestCase):
    def test_locked_readonly_check_cannot_omit_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); contract=root/'contract.json'; contract.write_text('{}')
            for mode,config,cache,expected in (
                ('locked-wheelhouse-uv','','/cache',2),
                ('locked-wheelhouse-uv',str(contract),'',2),
                ('locked-wheelhouse-uv',str(contract),'/cache',0),
                ('source-pip','','',0),
            ):
                body='set -eu\n'+ '\n'.join(f'{k}={shlex.quote(v)}' for k,v in {
                    'HOST_PY':'mock_python','STATE_HELPER':'/state_helper',
                    'DEPENDENCY_INSTALL_MODE':mode,'DEPENDENCY_CONTRACT':config,
                    'DEPENDENCY_CACHE':cache}.items())+'\n'
                body+='die() { exit "$1"; }\nmock_python() { printf "CALL"; printf "|%s" "$@"; }\n'
                body+=function(INSTALLER,'state_helper')+'\nstate_helper check --venv /venv\n'
                result=subprocess.run(['/bin/bash','-c',body],capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,expected,result.stderr)
                if expected:
                    self.assertNotIn('CALL',result.stdout)
                elif mode!='source-pip':
                    self.assertIn('|--dependency-contract|'+str(contract)+'|--dependency-cache|/cache',result.stdout)
                else:
                    self.assertNotIn('--dependency-contract',result.stdout)

    def test_locked_clean_allows_missing_only_and_does_not_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);existing=root/'venv';existing.mkdir();marker=existing/'keep';marker.write_text('keep')
            broken=root/'broken';broken.symlink_to(root/'absent')
            for path,expected in ((root/'new',0),(existing,1),(broken,1)):
                body='set -eu\nCLEAN=1\nVENV_PATH='+shlex.quote(str(path))+'\n'+function(INSTALLER,'locked_clean_is_safe')+'\nlocked_clean_is_safe\n'
                result=subprocess.run(['/bin/bash','-c',body],capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,expected,result.stderr)
            self.assertEqual(marker.read_text(),'keep')

    def shell(self, body, *, kokoro=1, legacy=0, prep=0, server_exit=0, kokoro_exit=0,
              wheelhouse=False):
        with tempfile.TemporaryDirectory(prefix="musetalk package timing ") as temp:
            fields = {
                "MODE": "plan", "REPO_ROOT": str(ROOT), "VENV_PATH": "/workspace/test venv",
                "VENV_EXISTS": 0, "VENV_TORCH_TAG": "", "TARGET_MATRIX": "cu121",
                "MATRIX_REASON": "mock", "GPU_DESC": "mock", "PYTHON_BIN": "unused",
                "CLEAN": 0, "CONSTRAINTS": "/requirements/constraints-cu121.txt",
                "PYTORCH_INDEX": "https://download.pytorch.org/whl/cu121",
                "NVIDIA_INDEX_URL": "https://pypi.nvidia.com", "KOKORO": kokoro,
                "LEGACY_INT8": legacy, "AVATAR_PREP": prep, "NATIVE_VP8": 0, "CHIN_TOOLS": 0,
                "CHIN_VENV": "unused", "NATIVE_VP8_DIR": "unused", "SKIP_WEIGHTS": 1,
                "SELFTEST": 0, "GPU_VISIBLE": 0, "SELFTEST_BUCKETS": 8, "STATE_FILE": "unused",
                "REQ_DIR": "/requirements with spaces", "WHEELHOUSE": temp if wheelhouse else "",
                "VENV_PY": "mock_python", "CLOCK_FILE": str(Path(temp) / "clock"),
                "STEP_LOG_CURRENT_PHASE": "Phase 3: Python packages", "LAST_STEP_STATUS": 0,
                "SERVER_EXIT": server_exit, "KOKORO_EXIT": kokoro_exit,
                "DEPENDENCY_INSTALL_MODE": "source-pip", "DEPENDENCY_CONTRACT": "/reviewed contract.json",
                "DEPENDENCY_CACHE": "/workspace/dependency cache", "SCRIPT_DIR": "/scripts",
            }
            source = "set -Eeuo pipefail\n" + "\n".join(
                f"{key}={shlex.quote(str(value))}" for key, value in fields.items()
            ) + "\n"
            source += r'''
flag_on() { [[ "$1" == "1" ]]; }
step_log_emit() { printf 'LOG|%s|%s\n' "$1" "$2"; }
log() { step_log_emit INFO "$*"; }
step_record_result() { printf 'RECORD|%s|%s|%s\n' "$1" "$2" "$3"; }
date() {
  if [[ "$1" == +%s ]]; then
    local now; read -r now < "$CLOCK_FILE"; printf '%s\n' "$now"
  else
    return 99
  fi
}
mock_python() {
  printf 'CALL'; printf '|%s' "$@"; printf '\n'
  if [[ " $* " == *kokoro.in* ]]; then
    printf '18\n' > "$CLOCK_FILE"; return "$KOKORO_EXIT"
  fi
  printf '11\n' > "$CLOCK_FILE"; return "$SERVER_EXIT"
}
avatar_prep_step() { printf 'PREP_CALL\n'; printf '21\n' > "$CLOCK_FILE"; }
printf '0\n' > "$CLOCK_FILE"
'''
            for name in ("on_err", "resolve_package_commands", "print_plan", "run_strict_step",
                         "pip_install_groups_step", "pip_install_kokoro_step", "phase_packages"):
                source += function(INSTALLER, name) + "\n"
            source += function(LOGGING, "step_format_duration") + "\n"
            source += body + "\n"
            # No inherited Python/pip/CUDA configuration can turn a mock into a real install.
            env = {"PATH": "/usr/bin:/bin", "LANG": "C"}
            return subprocess.run(["/bin/bash", "-c", source], capture_output=True, text=True,
                                  timeout=10, env=env)

    def commands(self, **kwargs):
        result = self.shell("resolve_package_commands\nprint_plan", **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    def test_server_and_kokoro_commands_are_separate(self):
        commands = self.commands()
        self.assertIn("server.in", commands["PIP_COMMAND"])
        self.assertNotIn("kokoro.in", commands["PIP_COMMAND"])
        self.assertIn("kokoro.in", commands["PIP_KOKORO_COMMAND"])
        self.assertNotIn("server.in", commands["PIP_KOKORO_COMMAND"])

    def test_locked_mode_uses_one_acquisition_wrapper_with_explicit_groups(self):
        result=self.shell('''
DEPENDENCY_INSTALL_MODE=locked-wheelhouse-uv
PYTHON_BIN=mock_python
phase_packages
''',prep=1)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        calls=[line for line in result.stdout.splitlines() if line.startswith('CALL|')]
        self.assertEqual(len(calls),1)
        self.assertIn('|/scripts/install_locked_dependencies.py|',calls[0])
        self.assertIn('|--group|full|',calls[0])
        self.assertIn('|--contract|/reviewed contract.json|',calls[0])
        self.assertNotIn('PREP_CALL',result.stdout)

    def test_both_commands_keep_constraints_and_indexes(self):
        for key, command in self.commands().items():
            if key not in ("PIP_COMMAND", "PIP_KOKORO_COMMAND"):
                continue
            args = shlex.split(command)
            self.assertIn("/requirements/constraints-cu121.txt", args)
            self.assertIn("https://download.pytorch.org/whl/cu121", args)
            self.assertIn("https://pypi.nvidia.com", args)
            self.assertNotIn("--no-deps", args)
            self.assertNotIn("--upgrade", args)

    def test_cu128_commands_share_the_selected_matrix(self):
        result = self.shell('''
TARGET_MATRIX=cu128
CONSTRAINTS=/requirements/constraints-cu128.txt
PYTORCH_INDEX=https://download.pytorch.org/whl/cu128
resolve_package_commands
print_plan
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            if line.startswith(("PIP_COMMAND=", "PIP_KOKORO_COMMAND=")):
                self.assertIn("constraints-cu128.txt", line)
                self.assertIn("/whl/cu128", line)
                self.assertNotIn("cu121", line)

    def test_disabled_kokoro_plan_has_no_install_command(self):
        commands = self.commands(kokoro=0)
        self.assertEqual(commands["PIP_KOKORO_COMMAND"], "disabled")
        self.assertNotIn("kokoro.in", commands["PIP_COMMAND"])

    def test_legacy_int8_stays_in_server_group(self):
        commands = self.commands(legacy=1)
        self.assertIn("legacy-int8.in", commands["PIP_COMMAND"])
        self.assertNotIn("legacy-int8.in", commands["PIP_KOKORO_COMMAND"])

    def test_wheelhouse_and_spaced_paths_survive_quoting(self):
        commands = self.commands(wheelhouse=True)
        for key in ("PIP_COMMAND", "PIP_KOKORO_COMMAND"):
            args = shlex.split(commands[key])
            link = args[args.index("--find-links") + 1]
            self.assertIn("musetalk package timing ", link)
            requirement = args[args.index("-r") + 1]
            self.assertTrue(requirement.startswith("/requirements with spaces/"), requirement)

    def test_success_records_independent_server_kokoro_and_prep_timers(self):
        result = self.shell("resolve_package_commands\nphase_packages\nprintf 'FINISHED\\n'", prep=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [line for line in result.stdout.splitlines() if line.startswith("RECORD|")]
        self.assertEqual(records, [
            "RECORD|Phase 3: Python packages :: Install server packages (server.in)|11|OK",
            "RECORD|Phase 3: Python packages :: Install Kokoro packages (kokoro.in)|7|OK",
            "RECORD|Phase 3: Python packages :: Install avatar-prep stack (mmcv/mmdet/mmpose)|3|OK",
        ])
        calls = [line for line in result.stdout.splitlines() if line.startswith("CALL|")]
        self.assertEqual(len(calls), 2)
        self.assertIn("server.in", calls[0])
        self.assertNotIn("kokoro.in", calls[0])
        self.assertIn("kokoro.in", calls[1])
        self.assertIn("FINISHED", result.stdout)

    def test_disabled_kokoro_has_no_call_or_timer(self):
        result = self.shell("resolve_package_commands\nphase_packages", kokoro=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Skipping Kokoro packages", result.stdout)
        self.assertNotIn("kokoro.in", result.stdout)
        self.assertEqual(result.stdout.count("RECORD|"), 1)

    def test_server_failure_stops_kokoro_and_prep(self):
        result = self.shell("resolve_package_commands\nphase_packages\nprintf 'FINISHED\\n'",
                            server_exit=23, prep=1)
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertIn("Install server packages (server.in)|11|FAIL", result.stdout)
        self.assertNotIn("kokoro.in", result.stdout)
        self.assertNotIn("PREP_CALL", result.stdout)
        self.assertNotIn("FINISHED", result.stdout)

    def test_kokoro_failure_stops_prep(self):
        result = self.shell("resolve_package_commands\nphase_packages\nprintf 'FINISHED\\n'",
                            kokoro_exit=24, prep=1)
        self.assertEqual(result.returncode, 24, result.stdout + result.stderr)
        self.assertIn("Install server packages (server.in)|11|OK", result.stdout)
        self.assertIn("Install Kokoro packages (kokoro.in)|7|FAIL", result.stdout)
        self.assertNotIn("PREP_CALL", result.stdout)
        self.assertNotIn("FINISHED", result.stdout)

    def test_unselected_prep_has_no_call_or_timer(self):
        result = self.shell("resolve_package_commands\nphase_packages")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("PREP_CALL", result.stdout)
        self.assertEqual(result.stdout.count("RECORD|"), 2)

    def test_legacy_label_identifies_its_share_of_core_timing(self):
        result = self.shell("resolve_package_commands\nphase_packages", legacy=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Install server packages (server.in + legacy-int8.in)|11|OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
