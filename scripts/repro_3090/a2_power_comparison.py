#!/usr/bin/env python3
"""One fixed A2 300->350->300 W experiment; default prints a plan only.

No arbitrary commands, altered recipe, privilege escalation, nested GPU lease,
or quality promotion. Run through the existing owned-target credential bridge.
The bridge timeout must exceed the remaining execution window by >=10 seconds:
min(900 s, expiry deadline minus 185 s minus launch time). Near expiry the
absolute deadline, rather than the 900 s cap, ends this experiment first.
Invoke this Python file as the bridge's DIRECT child (no bash/sh intermediary):
the TERM grace must wait for this process's restoration handler, not a shell.
SIGKILL/host loss cannot be recovered in-process: an absent restore receipt is
an operator incident, not evidence that the original power was restored.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import safe_capture

ROOT = Path('/workspace/MuseTalk')
RUN = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
OUTPUT = RUN / 'native'
PYTHON = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'
GPU = 'GPU-a39e62bc-2405-e19d-ba7f-2c51647a46b2'
HOST = '03f3421071ce'
DEADLINE = dt.datetime(2026, 10, 8, 23, 45, tzinfo=dt.timezone.utc).timestamp()
DESCRIPTOR = RUN / 'provisioning/a2_owned_target.json'
PROFILE = ROOT / 'scripts/repro_3090/profiles/native.env'
INPUTS = RUN / 'harnesses/tracking-a2-inputs-v1.json'
PAIR = OUTPUT / 'a2_tracking_pair_2202_tracking-parity/report.json'
ENGINE = ROOT / 'models/tensorrt_unet_stagewise_sm86_r5_v1'
TAESD = ROOT / 'models/taesd/trt_native_sm86_r5_v1'
TAESD_KEY = '1e967e6e715c9f1a8375'
PINNED = {
    PROFILE: '7e5b80827c4541c4ee982e4a878496028053a07439daf7bd48b82786d6dd8b8c',
    INPUTS: '4e63c1afa903b9fb52d8ce09fc4a123bd71e12d504af3c7133aa829ce7cc3abb',
    PAIR: '741a98179ffc2b3ff33e221581e5ab45f334749f9258f504c5ebde1daed56545',
    ENGINE / 'bs16/manifest.json': 'f66b46ca38d0e34af69ee5c01be52d93cc3d2f1ba3ac7b8a68ae0426c3658316',
    TAESD / f'taesd_trt_{TAESD_KEY}.json': '4dcc2174a8e47a1cb918751a312f946c74f47c6e8ab4f3c9b9917cb0c20a8673',
}
FIELDS = 'uuid,name,power.limit,power.min_limit,power.default_limit,power.max_limit,temperature.gpu,utilization.gpu,memory.used'


class Refused(Exception):
    def __init__(self, reason, safe_record=None):
        super().__init__(reason)
        self.safe_record = safe_record


def require(ok, reason):
    if not ok:
        raise Refused(reason)


def command(label):
    require(bool(re.fullmatch(r'a2_power350_[A-Za-z0-9_-]{1,48}', label)), 'unsafe_label')
    return [PYTHON, str(ROOT / 'scripts/repro_3090/runner.py'), 'aggregate',
            '--profile', str(PROFILE), '--engine-root', str(ENGINE), '--taesd-dir', str(TAESD),
            '--taesd-key', TAESD_KEY, '--input-manifest', str(INPUTS), '--out', str(OUTPUT),
            '--label', label, '--python', PYTHON, '--corpus', str(ROOT / 'calibration/unet_multi_avatar_20260928'),
            '--accepted-root', '/workspace/experiments/avatar_diversity_20260927', '--workspace', '/workspace',
            '--target', 'native', '--stages', 'T', '--loops', '24', '--thermal-warmup-s', '120',
            '--tracking-overlap', '--tracking-parity-report', str(PAIR)]


def validate_state(row, expected_power, initial=False):
    require(row.get('uuid') == GPU and row.get('name') == 'NVIDIA GeForce RTX 3090', 'wrong_gpu_identity')
    for field in FIELDS.split(',')[2:]:
        require(isinstance(row.get(field), (int, float)) and math.isfinite(row[field]), 'invalid_nvml_measurement')
    require(abs(row['power.limit'] - expected_power) < .1, 'unexpected_power_readback')
    require(row['temperature.gpu'] <= 87, 'thermal_limit')
    if initial:
        require(row['power.min_limit'] <= 300 and row['power.default_limit'] == 350 and row['power.max_limit'] == 350,
                'unsupported_power_range')
        require(row['utilization.gpu'] <= 5 and row['memory.used'] <= 600, 'gpu_not_idle')


def proc_snapshot():
    """Only PID/PPID/PGRP/start ticks; never command lines or environments."""
    result = {}
    for item in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = item.read_text()
            tail = text[text.rindex(')') + 2:].split()
            result[int(item.parent.name)] = (int(tail[1]), int(tail[2]), int(tail[19]), tail[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants(snapshot, parent):
    selected = set()
    frontier = {parent}
    while frontier:
        nxt = {pid for pid, row in snapshot.items() if row[0] in frontier} - selected
        selected.update(nxt)
        frontier = nxt
    return selected


class Backend:
    def __init__(self, label, directory):
        self.label, self.directory = label, directory
        self.process = None
        self.log = None
        self.stopped = False
        self.known = {}
        # Do not let ambient lease, interpreter or tuning overrides change the experiment.
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('BOX_GUARD_', 'PYTHON', 'REPRO_', 'MUSETALK_', 'HLS_', 'WEBRTC_')) and k not in ('LD_PRELOAD', 'LD_AUDIT')}
        self.env['PATH'] = '/workspace/.venvs/musetalk_trt_stagewise/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        # Preserve the bridge-approved native library/device selection, and its
        # no-user-site policy; this is a power-only comparison, not a loader test.
        self.env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')

    def nvml(self, *args):
        # Both power setting and readback use the same bounded, ordinary utility.
        try:
            output = safe_capture.capture(['nvidia-smi', *args], cwd=ROOT, env=self.env,
                                          stage='unspecified', timeout_s=1, terminate_grace_s=0,
                                          reap_grace_s=.2, output_limit_bytes=8192)
            return 0, output
        except safe_capture.CaptureFailure as exc:
            record = exc.record  # Fixed diagnostic vocabulary, never raw stderr.
            if (record.get('failure') == 'NONZERO_EXIT' and record.get('cleanup') == 'leader_reaped'
                    and isinstance(record.get('returncode'), int) and record['returncode'] != 0):
                return record['returncode'], ''
            raise Refused('nvml_capture_failed', record) from None

    def state(self):
        rc, out = self.nvml(f'--query-gpu={FIELDS}', '--format=csv,noheader,nounits')
        require(rc == 0, 'nvml_read_failed')
        rows = list(csv.reader(out.strip().splitlines()))
        require(len(rows) == 1 and len(rows[0]) == len(FIELDS.split(',')), 'visible_gpu_count_or_schema')
        values = [v.strip() for v in rows[0]]
        try:
            return dict(zip(FIELDS.split(','), values[:2] + [float(v) for v in values[2:]]))
        except ValueError:
            raise Refused('invalid_nvml_measurement') from None

    def set_power(self, watts):
        require(watts in (300, 350), 'unsupported_setpoint')
        rc, _ = self.nvml('-i', GPU, f'--power-limit={watts}')
        return rc

    def preflight(self):
        require(sys.platform == 'linux' and socket.gethostname() == HOST, 'wrong_host')
        data = json.loads(DESCRIPTOR.read_text())
        expected = {'instance_id': '54909897', 'label': 'musetalk-r5-3090-dev-20261008-a2',
                    'worker_alias': 'musetalk-3090-build-54909897', 'worker_hostname': HOST, 'gpu_uuid': GPU,
                    'ledger': '/home/ec2-user/.local/state/musetalk-r5-3090-dev-20261008-a2/startup-ledger.json'}
        require(all(data.get(k) == v for k, v in expected.items()), 'wrong_owned_descriptor')
        require(dt.datetime.fromisoformat(data['deadline_utc'].replace('Z', '+00:00')).timestamp() == DEADLINE,
                'wrong_expiry_deadline')
        require(ROOT.resolve() == Path(__file__).resolve().parents[2], 'wrong_checkout_root')
        for path, digest in PINNED.items():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'fixed_input_hash_mismatch')
        require(len(json.loads(INPUTS.read_text())['files']) == 878, 'wrong_input_count')
        require(not (OUTPUT / f'{self.label}_aggregate').exists(), 'benchmark_output_exists')
        # Same canonical check as other jobs, WITHOUT acquiring an enduring lease.
        # No raw output is logged because co-tenant command lines may be sensitive.
        affinity = sorted(os.sched_getaffinity(0))
        require(bool(affinity), 'empty_cpu_affinity')
        host_load = os.getloadavg()[0]
        r = subprocess.run(['bash', str(ROOT / 'scripts/box_guard.sh'), 'check', '--min-avail-gb', '12',
                            '--need-disk-gb', '3', '--max-load', str(len(affinity)), '--max-gpu-mem-mib', '600',
                            '--max-gpu-util', '5'], cwd=ROOT, env=self.env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        require(r.returncode == 0, 'lease_cotenant_or_resources_not_quiet')
        require(not Path('/workspace/.gpu_lease.pause').exists(), 'gpu_lease_paused')
        rc, apps = self.nvml('--query-compute-apps=pid', '--format=csv,noheader,nounits')
        require(rc == 0 and not apps.strip(), 'foreign_gpu_workload')
        result = self.state()
        result['host_load_gate'] = {'affinity_cpu_ids': affinity, 'affinity_cpu_count': len(affinity),
                                    'max_host_load': len(affinity), 'observed_load1': host_load,
                                    'cpu_idle_proven': False,
                                    'meaning': 'catastrophic host load guard; not effective CPU quota or idle proof'}
        return result

    def start(self):
        # Reparent orphaned canonical setsid children to this wrapper so an abort
        # cannot leave a detached GPU workload behind. No daemon escapes the bridge.
        require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'subreaper_unavailable')
        self.log = (self.directory / 'canonical-run.log').open('x')
        self.process = subprocess.Popen(command(self.label), cwd=ROOT, env=self.env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def poll(self):
        self.track()
        return self.process.poll()

    def track(self):
        snap = proc_snapshot()
        for pid in descendants(snap, os.getpid()):
            self.known[pid] = snap[pid]

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        # Stop forking first. A second snapshot captures children born in the
        # first snapshot race; subreaper catches orphaned setsid/watch children.
        for _ in range(4):
            self.track()
            snap = proc_snapshot()
            for pid, row in self.known.items():
                if snap.get(pid, (None, None, None))[2] == row[2]:
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
        snap = proc_snapshot()
        for pid, row in self.known.items():
            if snap.get(pid, (None, None, None))[2] != row[2]:
                continue
            try:
                if row[1] == pid:  # only a group led by our still-identical child
                    os.killpg(pid, signal.SIGKILL)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process:
            try:
                self.process.wait(timeout=.3)
            except subprocess.TimeoutExpired:
                raise Refused('owned_child_did_not_exit') from None
        # A dead runner does not prove its setsid GPU descendants also exited.
        # Reap adopted children and reject any still-live owned process.
        until = time.monotonic() + .3
        while True:
            try:
                while os.waitpid(-1, os.WNOHANG)[0] > 0:
                    pass
            except ChildProcessError:
                pass
            snap = proc_snapshot()
            live = [pid for pid, row in self.known.items()
                    if pid in snap and snap[pid][2] == row[2] and snap[pid][3] != 'Z']
            if not live:
                break
            if time.monotonic() >= until:
                raise Refused('owned_descendant_did_not_exit')
            time.sleep(.02)
        if self.log:
            self.log.close()


def experiment(backend, receipt, save, *, wall=time.time, mono=time.monotonic,
               sleep=time.sleep, interrupted=lambda: False):
    """Dependency injection keeps failure/restore tests entirely CPU-only."""
    start = mono()
    restore_required = False
    rc = 2
    receipt.update(status='INVALID', quality_status='REJECTED_UNCHANGED', release_ready=False,
                   sust_authorized=False, events=[], canonical_returncode=None)

    def event(kind, **values):
        receipt['events'].append({'event': kind, 'utc': dt.datetime.fromtimestamp(wall(), dt.timezone.utc).isoformat(),
                                  'elapsed_s': mono() - start, **values})
        save(receipt)

    def within_budget():
        require(not interrupted(), 'operator_signal')
        # Reserve five seconds INSIDE both boundaries for process cleanup and
        # the bounded restore setter + independent readback.
        require(mono() - start < 895 and wall() < DEADLINE - 185, 'runtime_or_expiry_limit')

    try:
        within_budget()
        before = backend.preflight()
        event('power_before', state=before)
        validate_state(before, 300, initial=True)
        within_budget()
        restore_required = True  # Set BEFORE calling a setter with ambiguous failures.
        event('power_set_attempt', requested_watts=350)
        setter_rc = backend.set_power(350)
        event('power_set_result', returncode=setter_rc)
        require(setter_rc == 0, 'power_set_refused')
        after = backend.state()
        event('power_set_readback', state=after)
        validate_state(after, 350)
        within_budget()
        backend.start()
        next_poll = mono()
        while True:
            within_budget()
            if mono() >= next_poll:
                measured = backend.state()
                event('monitor', state=measured)
                validate_state(measured, 350)
                next_poll += 2
            child_rc = backend.poll()
            if child_rc is not None:
                receipt['canonical_returncode'] = child_rc
                rc = child_rc if child_rc >= 0 else 128 - child_rc
                receipt['status'] = 'T_PASS_QUALITY_STILL_REJECTED' if rc == 0 else 'CANONICAL_NONZERO'
                break
            sleep(.1)
    except Exception as exc:
        # Only our enumerated reason is public; external exception text can hold data.
        receipt['failure'] = str(exc) if isinstance(exc, Refused) else type(exc).__name__
        if isinstance(exc, Refused) and exc.safe_record is not None:
            receipt['safe_capture_failure'] = exc.safe_record
        rc = 2
    finally:
        try:
            backend.stop()
        except Exception:
            receipt['cleanup_failure'] = 'owned_process_cleanup_failed'
            rc = 2
        if restore_required:
            # Do not allow receipt I/O failure to skip either the setter or readback.
            restore_rc, restored, problem = None, None, None
            restore_capture_failures = []
            try:
                restore_rc = backend.set_power(300)
            except Exception as exc:
                problem = str(exc) if isinstance(exc, Refused) else type(exc).__name__
                if isinstance(exc, Refused) and exc.safe_record is not None:
                    restore_capture_failures.append(exc.safe_record)
            try:
                restored = backend.state()
                # Thermal violation is retained in the original failure; restoring
                # only claims device identity and actual 300 W, not cooldown.
                require(restored.get('uuid') == GPU and restored.get('name') == 'NVIDIA GeForce RTX 3090', 'restore_wrong_identity')
                require(abs(restored['power.limit'] - 300) < .1, 'restore_readback_failed')
                require(restore_rc == 0, 'restore_setter_failed')
            except Exception as exc:
                problem = str(exc) if isinstance(exc, Refused) else type(exc).__name__
                if isinstance(exc, Refused) and exc.safe_record is not None:
                    restore_capture_failures.append(exc.safe_record)
            receipt['restore'] = {'requested_watts': 300, 'setter_returncode': restore_rc, 'state': restored,
                                  'verified': problem is None, 'failure': problem,
                                  'safe_capture_failures': restore_capture_failures}
            if problem:
                rc = 2
        receipt['returncode'] = rc
        if rc == 2:
            receipt['status'] = 'INVALID'
        save(receipt)
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true', help='read-only host/idle/input checks; never set power')
    args = parser.parse_args(argv)
    plan = {'schema': 'a2_power_comparison_v1', 'instance_id': '54909897', 'hostname': HOST, 'gpu_uuid': GPU,
            'deadline_utc': '2026-10-08T23:45:00Z', 'hard_runtime_seconds': 900, 'deadline_margin_seconds': 180,
            'monitor_period_seconds': 2, 'max_gpu_temperature_c': 87, 'original_power_watts': 300,
            'experimental_power_watts': 350, 'argv': command(args.label), 'pinned_sha256': {str(k): v for k, v in PINNED.items()},
            'bridge_direct_child_argv': [PYTHON, str(ROOT / 'scripts/repro_3090/a2_power_comparison.py'),
                                         '--execute', '--label', args.label],
            'bridge_no_shell_intermediary': True,
            'quality_status': 'REJECTED_UNCHANGED', 'release_ready': False,
            'lease_policy': 'preflight idle check; runner alone acquires canonical GPU leases; exclusive operator required',
            'fps_gain_assumption': 'none; no linear power/FPS inference'}
    directory = OUTPUT / f'{args.label}_power'
    if not args.execute:
        if args.preflight:
            require(time.time() < DEADLINE - 180, 'expired')
            plan['power_before'] = Backend(args.label, directory).preflight()
            validate_state(plan['power_before'], 300, initial=True)
        print(json.dumps(plan, indent=2))
        return 0
    # No files on the wrong host, even if --execute is accidentally used locally.
    require(sys.platform == 'linux' and socket.gethostname() == HOST, 'wrong_host')
    directory.mkdir(exist_ok=False)
    stop = {'signal': None}
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old[sig] = signal.signal(sig, lambda number, frame: stop.update(signal=number))
    def save(data):
        data['signal'] = stop['signal']
        with (directory / 'power-receipt.json').open('w') as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
    try:
        return experiment(Backend(args.label, directory), plan, save, interrupted=lambda: stop['signal'] is not None)
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'status': 'INVALID', 'failure': str(exc) if isinstance(exc, Refused) else type(exc).__name__,
                          'quality_status': 'REJECTED_UNCHANGED', 'release_ready': False}))
        sys.exit(2)
