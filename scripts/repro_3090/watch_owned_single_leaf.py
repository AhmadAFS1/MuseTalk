"""Default-off A3 single-CUDA-leaf ownership watch; canonical watch is unchanged.

Ownership is inferred from a live native own CUDA context plus a successful
complete NVML query containing exactly one process, never from a first/new PID.
No CUDA children, exec, persistent PID allowlist, MPS, MIG, or vGPU are supported.
NVML v3 ABI/semantics: NVIDIA/go-nvml pkg/nvml/nvml.h. MPS isolation semantics:
docs.nvidia.com/deploy/mps/latest/architecture.html#client-attach-detach.
"""
import argparse
import ctypes as C
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

HOST = '1e7c09cffcb3'
UUID = 'GPU-ea6411bc-775f-6685-f1a4-28b6b4011a3d'
DEADLINE = '2026-10-09T02:45:00Z'
ALLOCATION_BYTES = 2 << 20
QUERY_SECONDS = 3
FINISH_SECONDS = 10
TARGET_NAMES = {'build_owned_taesd_fp32_a3.py', 'taesd_fp32_candidate_child.py',
                'taesd_fp32_candidate_aggregate.py', 'run_owned_legacy_serial_render_a3.py',
                'assemble_owned_unet_tail_a3.py', 'build_owned_unet_fp16_down3_a3.py'}
CANONICAL_TARGETS = {
    'gate_taesd_trt.py': '3baf8976e4fb25a908809e68d6ac126c07ea525baed7475a6443263221e28e99',
    'chin_multistream_render.py': 'df4e290b33d752be82d6d2ab738bc5d3e21aa439ffd05f1c3c852a8af1fd4a29',
    'validate_unet_backend.py': '81b74eddf5aaff8348ac27cce67b92763e937e09309762d137062b0a23f1d7a0',
    'srccache_exact.py': '59d27983e85f7c438d655dc4bc6dfc0fa739ac38f8d22482372a48e8e8908895',
}
BOOTSTRAP = """import hashlib,sys,types,json
p,h,s=sys.argv[1:];b=open(p,'rb').read(1<<20)
assert hashlib.sha256(b).hexdigest()==h
m=types.ModuleType('_owned_single_leaf_checked');m.__file__=p
exec(compile(b,p,'exec'),m.__dict__);m.leaf(json.loads(s))
"""


class Rejected(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Rejected(reason)


def checked_source(path, digest):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe_source_path')
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'explicit_source_digest_required')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno()); body = handle.read(1 << 20)
        require(stat.S_ISREG(info.st_mode) and 0 < len(body) == info.st_size <= (1 << 20), 'source_size_or_type')
    require(hashlib.sha256(body).hexdigest() == digest, 'source_sha256_mismatch')
    return body


class Process(C.Structure):
    # nvmlProcessInfo_t/v2, used by GetComputeRunningProcesses_v3: 24 bytes.
    _fields_ = [('pid', C.c_uint), ('memory', C.c_ulonglong),
                ('gpu_instance', C.c_uint), ('compute_instance', C.c_uint)]


class Nvml:
    def __init__(self, uuid):
        require(C.sizeof(Process) == 24 and Process.memory.offset == 8, 'nvml_v3_abi_mismatch')
        self.lib = C.CDLL('libnvidia-ml.so.1'); self.handle = C.c_void_p()
        self.call('nvmlInit_v2', [])
        self.call('nvmlDeviceGetHandleByUUID', [C.c_char_p, C.POINTER(C.c_void_p)], uuid.encode(), C.byref(self.handle))
        actual = C.create_string_buffer(96)
        self.call('nvmlDeviceGetUUID', [C.c_void_p, C.c_char_p, C.c_uint], self.handle, actual, 96)
        require(actual.value.decode() == uuid, 'nvml_uuid_mismatch')
        version = C.create_string_buffer(96)
        self.call('nvmlSystemGetDriverVersion', [C.c_char_p, C.c_uint], version, 96)
        require(version.value == b'595.91.07', 'unexpected_nvml_driver')
        virtual = C.c_uint()
        self.call('nvmlDeviceGetVirtualizationMode', [C.c_void_p, C.POINTER(C.c_uint)], self.handle, C.byref(virtual))
        require(virtual.value == 0, 'virtualization_not_native_none')
        current, pending = C.c_uint(), C.c_uint()
        status = self.call('nvmlDeviceGetMigMode', [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(C.c_uint)],
                           self.handle, C.byref(current), C.byref(pending), allow_error=True)
        require(status == 3, 'mig_not_explicitly_unsupported')

    def call(self, name, types, *args, allow_error=False):
        fn = getattr(self.lib, name); fn.argtypes = types; fn.restype = C.c_int
        done, result = threading.Event(), []
        def invoke():
            try:
                result.append((True, fn(*args)))
            except BaseException as exc:
                result.append((False, exc))
            finally:
                done.set()
        threading.Thread(target=invoke, daemon=True).start()
        require(done.wait(QUERY_SECONDS), 'nvml_query_timeout')
        success, status = result[0]
        if not success:
            raise status
        require(allow_error or status == 0, 'nvml_query_not_success')
        return status

    def rows(self):
        count = C.c_uint(64); values = (Process * 64)()
        self.call('nvmlDeviceGetComputeRunningProcesses_v3',
                  [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(Process)], self.handle, C.byref(count), values)
        require(count.value <= 64, 'nvml_truncated_or_invalid_count')
        return [(int(v.pid), int(v.memory)) for v in values[:count.value]]


def sole_context_pid(rows):
    require(isinstance(rows, list) and len(rows) == 1, 'sole_live_compute_process_required')
    pid, memory = rows[0]
    require(type(pid) is int and pid > 0 and type(memory) is int
            and ALLOCATION_BYTES <= memory < ((1 << 64) - 1), 'invalid_context_pid_or_memory')
    return pid


def verify_bound(rows, pid):
    require(sole_context_pid(rows) == pid, 'foreign_or_changed_gpu_process')


def quiet_mps(path):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe_mps_path')
    info = path.stat()
    require(path.is_dir() and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
            and not list(path.iterdir()), 'mps_directory_not_private_empty')


def cuda_uuid():
    lib = C.CDLL('libcuda.so.1'); device = C.c_int(); uuid = (C.c_ubyte * 16)()
    get_device = lib.cuDeviceGet; get_device.argtypes = [C.POINTER(C.c_int), C.c_int]; get_device.restype = C.c_int
    require(get_device(C.byref(device), 0) == 0, 'cuda_device_query_failed')
    get_uuid = getattr(lib, 'cuDeviceGetUuid_v2', None)
    require(get_uuid is not None, 'cuda_uuid_v2_unavailable')
    get_uuid.argtypes = [C.c_void_p, C.c_int]; get_uuid.restype = C.c_int
    require(get_uuid(uuid, device.value) == 0, 'cuda_uuid_query_failed')
    value = bytes(uuid).hex()
    return 'GPU-' + '-'.join((value[:8], value[8:12], value[12:16], value[16:20], value[20:]))


def parent_death_signal(parent_pid):
    require(os.getppid() == parent_pid, 'watch_parent_changed')
    prctl = C.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [C.c_int, C.c_ulong, C.c_ulong, C.c_ulong, C.c_ulong]; prctl.restype = C.c_int
    require(prctl(1, signal.SIGTERM, 0, 0, 0) == 0 and os.getppid() == parent_pid,
            'parent_death_signal_or_parent_changed')


def send_receipt(fd, message):
    body = (json.dumps(message) + '\n').encode()
    require(len(body) < 4096 and os.write(fd, body) == len(body), 'ownership_receipt_write_failed')


def leaf(spec):
    parent_death_signal(spec['parent_pid'])
    quiet_mps(spec['mps'])
    body = checked_source(spec['target'], spec['target_sha256'])
    import torch  # import only; original CUDA initialization remains deferred
    original, owner = torch.cuda._lazy_init, os.getpid()
    retained, phase, init_lock = [], {'busy': False, 'bound': False}, threading.RLock()
    def initialize_locked(*args, **kwargs):
        require(os.getpid() == owner, 'cuda_fork_child_forbidden')
        if phase['bound'] or phase['busy']:
            return original(*args, **kwargs)
        phase['busy'] = True
        send_receipt(spec['report_fd'], {'phase': 'BEGIN_INIT', 'container_pid': owner})
        require(os.read(spec['ack_fd'], 1) == b'0', 'initial_quiet_ack_missing')
        result = original(*args, **kwargs)
        require(torch.cuda.device_count() == 1, 'one_visible_cuda_device_required')
        retained.append(torch.empty(ALLOCATION_BYTES, device='cuda:0', dtype=torch.uint8))
        torch.cuda.synchronize(0)
        actual = cuda_uuid(); require(actual == spec['uuid'], 'cuda_uuid_mismatch')
        quiet_mps(spec['mps'])
        pid = sole_context_pid(Nvml(spec['uuid']).rows())
        message = {'container_pid': owner, 'host_pid': pid, 'cuda_uuid': actual,
                   'allocation_bytes': ALLOCATION_BYTES}
        send_receipt(spec['report_fd'], message)
        require(os.read(spec['ack_fd'], 1) == b'1', 'ownership_ack_missing')
        phase['receipt'] = message
        phase['bound'] = True; phase['busy'] = False
        return result
    def initialize(*args, **kwargs):
        with init_lock:  # recursive same-thread allocation only; other threads await ACK
            return initialize_locked(*args, **kwargs)
    torch.cuda._lazy_init = initialize
    path = Path(spec['target']); sys.path.insert(0, str(path.parents[2]))
    sys.argv = [str(path), *spec['arguments']]
    namespace = {'__name__': '__main__', '__file__': str(path), '__package__': None, '__spec__': None,
                 '_owned_context_retained': retained}
    def target_finished(returncode):
        # Keep the actual allocation and context alive until the observer checks
        # this final receipt. Interpreter teardown may then remove the context.
        with init_lock:
            if phase['bound']:
                require(os.getpid() == owner and len(retained) == 1, 'finished_context_not_retained')
                send_receipt(spec['report_fd'], {**phase['receipt'], 'phase': 'TARGET_FINISHED',
                                               'returncode': returncode})
                require(os.read(spec['ack_fd'], 1) == b'2', 'target_finished_ack_missing')
    try:
        exec(compile(body, str(path), 'exec'), namespace)
    except SystemExit as exc:
        code = exc.code
        target_finished(0 if code is None else (int(code) & 255) if isinstance(code, int) else 1)
        raise  # preserve the checked target's original SystemExit and stderr
    else:
        target_finished(0)


def validate_receipt(message, child_pid, uuid):
    require(isinstance(message, dict) and set(message) == {'container_pid', 'host_pid', 'cuda_uuid', 'allocation_bytes'},
            'ownership_receipt_schema')
    require(type(message['container_pid']) is int and message['container_pid'] == child_pid
            and message['cuda_uuid'] == uuid and type(message['allocation_bytes']) is int
            and message['allocation_bytes'] == ALLOCATION_BYTES, 'ownership_receipt_identity')
    require(type(message['host_pid']) is int and message['host_pid'] > 0, 'ownership_receipt_pid')
    return message['host_pid']


def validate_finished(message, child_pid, uuid, host_pid):
    require(isinstance(message, dict) and set(message) == {
        'phase', 'returncode', 'container_pid', 'host_pid', 'cuda_uuid', 'allocation_bytes'},
        'target_finished_schema')
    require(message['phase'] == 'TARGET_FINISHED' and type(message['returncode']) is int
            and 0 <= message['returncode'] <= 255, 'target_finished_status')
    identity = {k: v for k, v in message.items() if k not in {'phase', 'returncode'}}
    require(validate_receipt(identity, child_pid, uuid) == host_pid, 'target_finished_host_pid_changed')
    return message['returncode']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--enable', action='store_true'); p.add_argument('--out', required=True)
    p.add_argument('--target', required=True); p.add_argument('--target-sha256', required=True)
    p.add_argument('--deadline-utc', required=True); p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    require(args.enable and args.arguments[:1] == ['--'], 'explicit_enable_and_separator_required')
    require(args.deadline_utc == DEADLINE, 'owned_allocation_deadline_required')
    require(socket.gethostname() == HOST, 'wrong_owned_hostname')
    holder = Path(os.environ.get('BOX_GUARD_LEASE_FILE', '/workspace/.gpu_lease') + '.holder')
    require(os.getsid(0) == os.getpgrp() == os.getpid() and holder.is_file(), 'canonical_isolated_guard_required')
    deadline = dt.datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    require(deadline.utcoffset() == dt.timedelta(0) and 120 < (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 86400,
            'deadline_utc_or_cleanup_margin')
    target = Path(args.target)
    require(target.name in TARGET_NAMES or target.name in CANONICAL_TARGETS, 'target_not_allowed')
    if target.name in CANONICAL_TARGETS:
        require(args.target_sha256 == CANONICAL_TARGETS[target.name], 'canonical_target_pin_required')
    checked_source(target, args.target_sha256)
    output = Path(args.out)
    require(output.is_absolute() and not any(x.is_symlink() for x in (output, *output.parents)), 'unsafe_watch_output')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as log, tempfile.TemporaryDirectory(prefix='owned-a3-mps-') as mps:
        child = None; pipes = []
        def emit(row):
            log.write(json.dumps({'utc': dt.datetime.now(dt.timezone.utc).isoformat(), **row}) + '\n'); log.flush()
        try:
            quiet_mps(mps); nvml = Nvml(UUID); require(not nvml.rows(), 'initial_foreign_gpu_process')
            report_r, report_w = os.pipe(); ack_r, ack_w = os.pipe()
            pipes = [report_r, report_w, ack_r, ack_w]
            spec = {'target': str(target), 'target_sha256': args.target_sha256, 'arguments': args.arguments[1:],
                    'uuid': UUID, 'mps': mps, 'report_fd': report_w, 'ack_fd': ack_r, 'parent_pid': os.getpid()}
            env = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_MPS_')}
            env.update(CUDA_VISIBLE_DEVICES=UUID, CUDA_MPS_PIPE_DIRECTORY=mps)
            source = Path(__file__).absolute(); digest = hashlib.sha256(source.read_bytes()).hexdigest()
            checked_source(source, digest)
            child = subprocess.Popen([sys.executable, '-B', '-c', BOOTSTRAP, str(source), digest, json.dumps(spec)],
                                     env=env, pass_fds=(report_w, ack_r))
            os.close(report_w); os.close(ack_r); pipes = [report_r, ack_w]
            data, bound, started, init_started = b'', None, time.monotonic(), None
            finished, finish_started = None, None
            emit({'status': 'ENROLLMENT_PENDING', 'container_pid': child.pid, 'target_sha256': args.target_sha256,
                  'retained_context_allocation_bytes': ALLOCATION_BYTES, 'allocation_outside_target_timing': True})
            while child.poll() is None:
                require(dt.datetime.now(dt.timezone.utc) < deadline - dt.timedelta(seconds=120), 'cleanup_deadline_reached')
                require(finish_started is None or time.monotonic() - finish_started < FINISH_SECONDS,
                        'target_finished_exit_timeout')
                if finished is not None:
                    # The target is authenticated complete; CUDA is intentionally
                    # being torn down. Only a bounded, matching process exit can
                    # complete this final phase, not another GPU ownership claim.
                    time.sleep(0.05)
                    continue
                quiet_mps(mps); rows = nvml.rows()
                message = None
                if finished is None and select.select([report_r], [], [], 0)[0]:
                    chunk = os.read(report_r, 4096); require(chunk, 'ownership_pipe_closed')
                    data += chunk; require(len(data) <= 4096, 'ownership_receipt_oversized')
                    if b'\n' in data:
                        require(data.endswith(b'\n') and data.count(b'\n') == 1, 'ownership_receipt_framing')
                        message = json.loads(data); data = b''
                if bound is not None:
                    if finished is None:
                        verify_bound(rows, bound)  # never waive empty rows before DONE
                        if message is not None:
                            finished = validate_finished(message, child.pid, UUID, bound)
                            verify_bound(nvml.rows(), bound)
                            finish_started = time.monotonic()
                            require(os.write(ack_w, b'2') == 1, 'target_finished_ack_write_failed')
                            emit({'status': 'TARGET_FINISHED_ACK', 'container_pid': child.pid,
                                  'host_pid': bound, 'gpu_uuid': UUID, 'target_returncode': finished})
                else:
                    require(not rows if init_started is None else len(rows) <= 1, 'foreign_gpu_during_enrollment')
                    require(time.monotonic() - started < 120, 'ownership_handshake_timeout')
                    require(init_started is None or time.monotonic() - init_started < 10, 'cuda_initialization_binding_timeout')
                    if message is not None:
                        if init_started is None:
                            require(message == {'phase': 'BEGIN_INIT', 'container_pid': child.pid}
                                    and type(message['container_pid']) is int, 'initialization_receipt_identity')
                            require(not nvml.rows(), 'foreign_gpu_before_cuda_initialization')
                            init_started = time.monotonic(); os.write(ack_w, b'0')
                            emit({'status': 'QUIET_INITIALIZATION_ACK', 'container_pid': child.pid})
                        else:
                            bound = validate_receipt(message, child.pid, UUID)
                            verify_bound(nvml.rows(), bound)
                            emit({'status': 'LIVE_CONTEXT_MEMBERSHIP_BOUND', 'container_pid': child.pid, 'host_pid': bound,
                                  'gpu_uuid': UUID, 'gpu_rows': rows})
                            os.write(ack_w, b'1')
                emit({'status': 'WATCH', 'gpu_rows': rows, 'bound_host_pid': bound}); time.sleep(0.25)
            require(bound is not None, 'leaf_exited_without_ownership_binding')
            require(finished is not None, 'leaf_exited_without_target_finished_ack')
            require(child.returncode == finished, 'target_finished_exit_code_mismatch')
            emit({'status': 'COMPLETE', 'returncode': child.returncode, 'bound_host_pid': bound})
            return child.returncode
        except BaseException as exc:
            emit({'status': 'INVALID', 'error_type': type(exc).__name__,
                  'reason': str(exc) if isinstance(exc, Rejected) else 'observer_or_leaf_failure'})
            if child is not None:
                os.killpg(os.getpgrp(), signal.SIGTERM)  # only this canonical isolated group
            return 2
        finally:
            for descriptor in pipes:
                os.close(descriptor)


if __name__ == '__main__':
    raise SystemExit(main())
