"""A3-only CPU driver identity probe; no CUDA context or GPU workload.

Exact NVIDIA 595.91.07 public ABI: allocate one own root client, query its
read-only RC-report control, free that client. An unchanged processId sentinel
is failure, not a guessed host PID. This is NOT a watchdog or acceptance gate.
"""
import ctypes as C
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import sys


U32, U16, U64 = C.c_uint32, C.c_uint16, C.c_uint64


class Alloc(C.Structure):
    _fields_ = [('root', U32), ('parent', U32), ('new', U32), ('class_id', U32),
                ('params', U64), ('params_size', U32), ('status', U32)]


class Control(C.Structure):
    _fields_ = [('client', U32), ('object', U32), ('command', U32), ('flags', U32),
                ('params', U64), ('params_size', U32), ('status', U32)]


class Free(C.Structure):
    _fields_ = [('root', U32), ('parent', U32), ('object', U32), ('status', U32)]


class Entry(C.Structure):
    _fields_ = [('tag', U32), ('value', U32), ('attribute', U32)]


class Report(C.Structure):
    _fields_ = [('request_index', U16), ('report_index', U16), ('gpu_tag', U32),
                ('report_time', U32), ('start_index', U16), ('end_index', U16),
                ('report_type', U16), ('flags', U32), ('report_count', U16),
                ('owner', U32), ('process_id', U32), ('entries', Entry * 200)]


def ioctl(fd, number, record):
    command = (3 << 30) | (C.sizeof(record) << 16) | (ord('F') << 8) | number
    buffer = bytearray(C.string_at(C.addressof(record), C.sizeof(record)))
    fcntl.ioctl(fd, command, buffer, True)
    C.memmove(C.addressof(record), bytes(buffer), len(buffer))


def main():
    assert sys.argv[1:] == ['--execute'], 'explicit_execution_required'
    assert socket.gethostname() == '1e7c09cffcb3', 'wrong_owned_hostname'
    assert (dt.datetime(2026, 10, 9, 2, 45, tzinfo=dt.timezone.utc)
            - dt.datetime.now(dt.timezone.utc)).total_seconds() > 600, 'cleanup_margin_required'
    assert '595.91.07' in Path('/proc/driver/nvidia/version').read_text(), 'wrong_driver_abi'
    assert C.sizeof(Alloc) == C.sizeof(Control) == 32 and C.sizeof(Free) == 16, 'wrong_ioctl_abi'
    assert C.sizeof(Report) == 2436 and Report.process_id.offset == 32, 'wrong_report_abi'
    row = {'schema': 'owned_a3_rm_pid_probe_v1', 'hostname': socket.gethostname(),
           'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'container_pid': os.getpid(),
           'driver': '595.91.07', 'command': 'NV0000_CTRL_CMD_NVD_GET_RCERR_RPT',
           'cuda_context_created': False, 'gpu_workload_started': False,
           'host_pid_mapping_verified': False, 'status': 'IN_PROGRESS'}
    fd, handle = None, None
    try:
        fd = os.open('/dev/nvidiactl', os.O_RDWR | os.O_CLOEXEC)
        alloc = Alloc(0, 0, 0, 0x41, 0, 0, 0xffffffff)
        ioctl(fd, 0x2b, alloc)
        row['allocation_status'] = alloc.status
        assert alloc.status == 0 and alloc.new != 0, 'own_client_allocation_failed'
        handle = alloc.new
        report = Report()
        report.owner = 0xffffffff
        report.process_id = 0xffffffff
        control = Control(handle, handle, 0x607, 0, C.addressof(report), C.sizeof(report), 0xffffffff)
        ioctl(fd, 0x2a, control)
        row['control_status'] = control.status
        row['process_id_sentinel_changed'] = report.process_id != 0xffffffff
        # Error copyout is not a documented positive identity result. Never
        # promote a value unless the complete read-only operation succeeded.
        if control.status == 0 and 0 < report.process_id < 0xffffffff:
            row['host_pid_observed'] = report.process_id
            row['status'] = 'HOST_PID_FIELD_OBSERVED_REQUIRES_INDEPENDENT_REVIEW'
        else:
            row['status'] = 'NO_TRUSTWORTHY_HOST_PID_MAPPING'
    except BaseException as exc:
        row['status'] = 'PROBE_FAILED_CLOSED'
        row['error_type'] = type(exc).__name__
    finally:
        if fd is not None and handle is not None:
            try:
                freed = Free(handle, 0, handle, 0xffffffff)
                ioctl(fd, 0x29, freed)
                row['own_client_free_status'] = freed.status
            except BaseException as exc:
                row['own_client_free_error_type'] = type(exc).__name__
        if fd is not None:
            os.close(fd)
            row['own_control_fd_closed'] = True
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
