#!/usr/bin/env python3
"""Run the baked startup and supervise its API in a private Docker PID namespace."""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO = Path('/opt/character-runtime/app')
WORKER = REPO / 'character_factory/character_worker'
ROOT = Path('/workspace/character_worker')
API = WORKER / 'character_api.py'
SSH_ROOT = Path('/run/character-worker-ssh')
FAILURE_RETENTION_SECONDS = 90
TERM_GRACE_SECONDS = 8


def log(message):
    print(f'[character-entrypoint] {message}', flush=True)


def configure_ssh():
    key = os.environ.get('CHARACTER_WORKER_SSH_PUBLIC_KEY') or os.environ.get('SSH_PUBLIC_KEY') or ''
    keys = key.strip().splitlines()
    pattern = r'(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)) [A-Za-z0-9+/]+=*(?: [^\x00-\x1f\x7f]*)?'
    if not keys or len(keys) > 16 or len(key) > 32768 or not all(re.fullmatch(pattern, line) for line in keys):
        raise ValueError('diagnostic SSH requires valid public keys')
    SSH_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    Path('/run/sshd').mkdir(mode=0o755, parents=True, exist_ok=True)
    authorized = SSH_ROOT / 'authorized_keys'
    authorized.write_text('\n'.join(keys) + '\n')
    authorized.chmod(0o600)
    host_key = SSH_ROOT / 'host_ed25519_key'
    ssh_env = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C.UTF-8'}
    subprocess.run(['/usr/bin/ssh-keygen', '-l', '-f', str(authorized)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ssh_env, timeout=10)
    if not host_key.exists():
        subprocess.run(['/usr/bin/ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(host_key)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ssh_env, timeout=10)
    command = ['/usr/sbin/sshd', '-D', '-e', '-f', '/dev/null',
               '-o', f'HostKey={host_key}', '-o', f'AuthorizedKeysFile={authorized}',
               '-o', 'PermitRootLogin=prohibit-password', '-o', 'PasswordAuthentication=no',
               '-o', 'KbdInteractiveAuthentication=no', '-o', 'AuthenticationMethods=publickey',
               '-o', 'UsePAM=yes', '-o', 'DisableForwarding=yes', '-o', 'PermitTunnel=no',
               '-o', 'AllowUsers=root']
    return command, ssh_env


def api_pid():
    try:
        pid = int((ROOT / 'character_api.pid').read_text())
        if pid > 1 and os.fsencode(API) in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:2]:
            return pid
    except (OSError, ValueError):
        pass
    return None


class SetupInterrupted(Exception):
    pass


class Supervisor:
    def __init__(self):
        self.children = {}
        self.interrupted = None
        self.setting_up = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.interrupt)

    def interrupt(self, signum, _frame):
        self.interrupted = signum
        if self.setting_up:
            raise SetupInterrupted()

    def start(self, argv, env):
        pid = os.fork()
        if pid == 0:
            try:
                os.execve(argv[0], argv, env)
            except OSError:
                log(f'could not start {Path(argv[0]).name}')
                os._exit(127)
        self.children[pid] = None
        return pid

    def reap(self):
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return False
            if pid == 0:
                return True
            if pid in self.children:
                self.children[pid] = os.waitstatus_to_exitcode(status)

    def run(self, startup_env, ssh):
        ssh_pid = self.start(*ssh) if ssh else None
        startup = self.start(['/bin/bash', str(WORKER / 'vast_onstart_character.sh')], startup_env)
        startup_env.clear()
        observed_api = False
        while not self.interrupted:
            self.reap()
            current_api = api_pid()
            if ssh_pid and self.children[ssh_pid] is not None:
                log('diagnostic SSH exited')
                return 1
            boot = self.children[startup]
            if boot is not None and boot != 0:
                log(f'authoritative startup failed with status {boot}')
                # Preserve failed health and log access long enough for the controller to record them.
                deadline = time.monotonic() + FAILURE_RETENTION_SECONDS
                while (current_api or ssh_pid) and not self.interrupted and time.monotonic() < deadline:
                    self.reap()
                    current_api = api_pid()
                    if ssh_pid and self.children[ssh_pid] is not None:
                        ssh_pid = None
                    time.sleep(0.2)
                if self.interrupted:
                    return 128 + self.interrupted
                return boot if boot > 0 else 128 - boot
            if not current_api and (observed_api or boot == 0):
                log('worker API exited')
                return 1
            observed_api |= current_api is not None
            time.sleep(0.2)
        return 128 + self.interrupted

    def shutdown(self):
        if not self.reap():
            return
        # This program only runs as PID 1. Signal the private namespace, including
        # H3 subprocesses that deliberately create separate sessions.
        for sig, grace in ((self.interrupted or signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, 2)):
            try:
                os.kill(-1, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace
            while self.reap() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not self.reap():
                return


def main():
    if os.getpid() != 1 or len(sys.argv) != 1:
        log('the baked entrypoint requires a private Docker PID namespace and no arguments')
        return 64
    supervisor = Supervisor()
    try:
        supervisor.setting_up = True
        if supervisor.interrupted:
            return 128 + supervisor.interrupted
        pins = {'MUSETALK_REF': r'[a-f0-9]{40}',
                'CHARACTER_WORKER_RUNTIME_MANIFEST_SHA256': r'[a-f0-9]{64}',
                'CHARACTER_WORKER_IMAGE_DIGEST': r'ghcr\.io/ahmadafs1/musetalk-rtx3090@sha256:[a-f0-9]{64}'}
        if any(not re.fullmatch(pattern, os.environ.get(name, '')) for name, pattern in pins.items()):
            log('immutable deployment pins are missing or invalid')
            return 64
        try:
            actual = subprocess.check_output(['/usr/bin/git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                             text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
            if actual != os.environ['MUSETALK_REF']:
                log('baked source differs from the deployment pin')
                return 64
            ssh_enabled = os.environ.get('CHARACTER_WORKER_ENABLE_SSH', '0')
            if ssh_enabled not in ('0', '1'):
                raise ValueError('invalid diagnostic SSH mode')
            ssh = configure_ssh() if ssh_enabled == '1' else None
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            log(f'entrypoint setup failed ({type(error).__name__})')
            return 64
        startup_env = {**os.environ, 'REPO_ROOT': str(REPO), 'CHARACTER_WORKER_REPO': str(REPO),
                       'WORKSPACE': '/workspace', 'CHARACTER_WORKER_ROOT': str(ROOT),
                       'CHARACTER_WORKER_RUNTIME_MODE': 'prebuilt', 'CHARACTER_WORKER_REQUIRE_TLS': '1'}
        os.environ.pop('CHARACTER_WORKER_TLS_KEY_B64', None)
        supervisor.setting_up = False
        if supervisor.interrupted:
            return 128 + supervisor.interrupted
        return supervisor.run(startup_env, ssh)
    except SetupInterrupted:
        return 128 + supervisor.interrupted
    finally:
        supervisor.setting_up = False
        supervisor.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
