"""Default-off allocation binding for the unchanged reviewed single-leaf watch.

An exact, operator-pinned non-secret descriptor binds the new instance, host,
GPU and deadline. This is not a rental authorization or an expiry-timer install.
The reviewed CUDA/NVML ownership protocol and canonical target pins are retained.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import types

WATCH_SHA = '2f03d2761e6319e27a58917099722bf279ce5d695e4682eaa5637de9d4db737f'
FIELDS = {'instance_id', 'label', 'worker_alias', 'worker_hostname', 'gpu_uuid', 'ledger', 'deadline_utc'}


def binding(path, digest, now=None):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe owned descriptor path')
    with path.open('rb') as handle:
        raw = handle.read(32769)
    if not re.fullmatch('[0-9a-f]{64}', digest) or not 0 < len(raw) <= 32768 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('owned descriptor hash mismatch')
    doc = json.loads(raw)
    if not isinstance(doc, dict) or set(doc) != FIELDS or not all(isinstance(v, str) for v in doc.values()):
        raise ValueError('owned descriptor schema mismatch')
    instance, label = doc['instance_id'], doc['label']
    if not re.fullmatch('[1-9][0-9]{6,11}', instance) or instance == '51074906':
        raise ValueError('protected or invalid instance')
    if not re.fullmatch('musetalk-r5-3090-dev-[A-Za-z0-9_.-]{4,80}', label):
        raise ValueError('experimental development label required')
    if doc['worker_alias'] != 'musetalk-3090-build-' + instance or doc['ledger'] != '/home/ec2-user/.local/state/' + label + '/startup-ledger.json':
        raise ValueError('owned alias or ledger mismatch')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]{0,63}', doc['worker_hostname']) or not re.fullmatch('GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}', doc['gpu_uuid']):
        raise ValueError('owned host or GPU identity mismatch')
    deadline = dt.datetime.fromisoformat(doc['deadline_utc'].replace('Z', '+00:00'))
    now = now or dt.datetime.now(dt.timezone.utc)
    if deadline.utcoffset() != dt.timedelta(0) or deadline.microsecond or not 120 < (deadline - now).total_seconds() <= 86400:
        raise ValueError('owned deadline expired or invalid')
    return doc, deadline


def reviewed_watch():
    path = Path(__file__).resolve().with_name('watch_owned_single_leaf.py')
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != WATCH_SHA:
        raise ValueError('reviewed watch changed')
    module = types.ModuleType('_checked_owned_single_leaf'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--enable', action='store_true')
    p.add_argument('--owned-target-json', required=True)
    p.add_argument('--owned-target-sha256', required=True)
    p.add_argument('watch_arguments', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if not a.enable or a.watch_arguments[:1] != ['--']:
        raise ValueError('explicit enable and watch separator required')
    doc, deadline = binding(a.owned_target_json, a.owned_target_sha256)
    if socket.gethostname() != doc['worker_hostname']:
        raise ValueError('wrong owned hostname')
    facts = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,compute_cap,driver_version',
                                     '--format=csv,noheader'], text=True, timeout=10).strip().split(',')
    if [v.strip() for v in facts] != ['NVIDIA GeForce RTX 3090', doc['gpu_uuid'], '8.6', '595.91.07']:
        raise ValueError('owned RTX3090 runtime identity mismatch')
    watch = reviewed_watch()
    watch.HOST, watch.UUID = doc['worker_hostname'], doc['gpu_uuid']
    watch.DEADLINE = deadline.strftime('%Y-%m-%dT%H:%M:%SZ')
    watch.TARGET_NAMES = watch.TARGET_NAMES | {'run_owned_unet_overlay.py'}
    # No CUDA/NVML logic, timeout, ownership check or canonical pin is changed.
    return watch.main(a.watch_arguments[1:])


if __name__ == '__main__':
    raise SystemExit(main())
