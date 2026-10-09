"""Explicit new-allocation binding for the exact prior CPU input audit.

Only host/UUID/instance bookkeeping and the deadline are adapted. No model,
source hash, metadata rule, download revision, or quality bound is altered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types

AUDIT_SHA = '630f101f6ffb6a6bda4ec7eba063b04041432ca28d60e77c6161d5c80a79b16f'
BINDING_SHA = 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9'


def checked(path, digest):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute nonsymlink source required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('reviewed source changed')
    return raw


def adapted(raw, document):
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA:
        raise ValueError('reviewed CPU audit changed')
    replacements = {
        "'9dad9291adec'": repr(document['worker_hostname']),
        "'GPU-050b4bc5-99f6-305e-8c7f-fd00ea44551f'": repr(document['gpu_uuid']),
        "'54957508'": repr(document['instance_id']),
        "'owned_a4_cpu_original_input_audit_v1'": "'owned_cpu_original_input_audit_v2'",
    }
    text = raw.decode()
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError('unique bookkeeping literal required')
        text = text.replace(old, new)
    return text.encode()


def load(raw, path):
    module = types.ModuleType('_checked_cpu_inputs'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--owned-target-json', required=True)
    p.add_argument('--owned-target-sha256', required=True)
    p.add_argument('--restore-syncnet', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if not a.execute:
        raise ValueError('explicit execution required')
    here = Path(__file__).resolve().parent
    path = here / 'watch_owned_single_leaf_target.py'
    binder = load(checked(path, BINDING_SHA), path)
    document, deadline = binder.binding(a.owned_target_json, a.owned_target_sha256)
    original = here / 'prepare_owned_a4_cpu_inputs.py'
    audit = load(adapted(checked(original, AUDIT_SHA), document), original)
    audit.DEADLINE = deadline
    args = ['--execute', '--out', a.out]
    if a.restore_syncnet: args.append('--restore-syncnet')
    return audit.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
