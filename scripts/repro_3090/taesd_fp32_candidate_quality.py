#!/usr/bin/env python3
"""Explicit, default-off routing of the unchanged 3090 quality runner.

Only its TAESD gate and six-avatar capture children select the hash-bound
candidate. The original child guard/watch, inputs, numerical tools, failure
codes and output schema remain unchanged. No GPU or cloud work occurs on import.
This does not establish quality parity or authorize release selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME_SHA = 'b4c358d3ebab73db127ea3c6ea69497cd85013e58bd31f3c573a6a9404d22835'
PINS = {
    'report': 'a102cb4ebdf7ed2f5e7828f3c061a2a43a6ed870a6a2e7fb72767f51b573ef8d',
    'safe_capture': 'd65738f27337641c2da919a572974f007a1445dfaa88c26d2eb1e6784393182a',
    'quality_envelope': 'a5a98185dad4db0d1b89c2d3e04b8832721d99461cca7381605b62b035697e55',
    'tracking_parity': '34a5155f159f95e12b527959f92a447e89e918368a092fe44958b3a9f8e90e3e',
    'runner': '0ede69c7a6f97aae57c38aec27e530b06beb1cc22175c89aa374a0fc66fb1f1d',
}
TARGETS = {
    'scripts/repro_400fps/gate_taesd_trt.py': ('taesd', 'gate'),
    'scripts/chin_multistream_render.py': ('quality_capture', 'render'),
}
RUNNER_OPTIONS = {'--profile', '--engine-root', '--taesd-key', '--taesd-dir', '--input-manifest',
                  '--out', '--label', '--python', '--corpus', '--accepted-root', '--workspace', '--target'}
# The true -c __main__ has no __file__/__spec__, so spawn imports only its
# importable CPU worker, not a second, unchecked copy of the launcher file.
# Both launcher and target execute checked bytes in separate __main__ namespaces.
CHILD_BOOTSTRAP = '''import hashlib, os, pathlib, stat, sys
p = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
if not p.is_absolute() or any(x.is_symlink() for x in (p, *p.parents)):
    raise SystemExit(2)
with os.fdopen(os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as f:
    info = os.fstat(f.fileno())
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 1048576:
        raise SystemExit(2)
    raw = f.read(1048577)
if len(raw) != info.st_size or hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit(2)
sys.argv = [str(p), *sys.argv[3:]]
scope = {"__name__": "__main__", "__file__": str(p), "__package__": None, "__spec__": None}
exec(compile(raw, str(p), "exec"), scope)
'''


def require(value, reason):
    if not value:
        raise ValueError(reason)


def runtime_api():
    file = HERE / 'taesd_fp32_candidate_runtime.py'
    require(not any(p.is_symlink() for p in (file, *file.parents)), 'runtime_symlink_forbidden')
    source = file.read_bytes()
    require(hashlib.sha256(source).hexdigest() == RUNTIME_SHA, 'runtime_helper_changed')
    module = types.ModuleType('_verified_candidate_quality_runtime')
    module.__file__ = str(file)
    exec(compile(source, str(file), 'exec'), module.__dict__)
    return module


def selection(api, manifest_path, manifest_sha256):
    raw = api.checked_bytes(Path(manifest_path), manifest_sha256, 2 << 20)
    meta = api.parse_json(raw)
    require(isinstance(meta, dict) and meta.get('schema') == api.SCHEMA,
            'candidate_manifest_schema_mismatch')
    require(meta.get('status') == 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED'
            and all(meta.get(k) is False for k in ('quality_accepted', 'full_gate_captured',
                'performance_measured', 'default_selection_changed', 'release_ready')),
            'candidate_manifest_claims_invalid')
    require(meta.get('fingerprint', {}).get('recipe') == api.RECIPE
            and meta['fingerprint'].get('precision') == 'fp16_with_final_conv_fp32'
            and meta.get('key') == api.key_for(meta['fingerprint']), 'candidate_identity_invalid')
    require(Path(manifest_path).name == f"taesd_trt_{meta['key']}.json", 'manifest_filename_mismatch')
    require(all(api.valid_digest(meta.get(k)) for k in ('decoder_plan_sha256', 'post_plan_sha256')),
            'candidate_plan_digest_invalid')
    return {k: meta[k] for k in ('key', 'decoder_plan_sha256', 'post_plan_sha256')}


def validate_receipt(api, path, manifest_sha256, identity, target, returncode):
    raw = api.regular_bytes(Path(path), 2 << 20)
    data = api.parse_json(raw)
    require(data.get('schema') == 'taesd_fp32_candidate_child_v1'
            and data.get('target') == target and data.get('manifest_sha256') == manifest_sha256,
            'candidate_child_receipt_binding_mismatch')
    require(data.get('candidate_key') == identity['key']
            and all(data.get(k) == identity[k] for k in ('decoder_plan_sha256', 'post_plan_sha256')),
            'candidate_child_loaded_wrong_artifact')
    require(type(data.get('candidate_calls')) is int and data['candidate_calls'] >= 1
            and data.get('candidate_attempts') == data['candidate_calls'], 'candidate_not_successfully_loaded')
    require(type(data.get('returncode')) is int and data['returncode'] == returncode
            and returncode in ((0, 1) if target == 'gate' else (0,))
            and data.get('status') == ('complete' if returncode == 0 else 'failed'),
            'candidate_child_exit_invalid')
    require(all(data.get(k) is False for k in ('quality_accepted', 'performance_accepted', 'release_ready')),
            'candidate_child_claims_invalid')
    return {'target': target, 'manifest_sha256': manifest_sha256, **identity,
            'receipt': str(path), 'receipt_sha256': api.sha(raw),
            'candidate_calls': data['candidate_calls'], 'returncode': returncode}


@contextmanager
def pinned_runner(api):
    """Execute checked bytes, not import caches; restore process state exactly."""
    saved_modules = {name: sys.modules.get(name) for name in PINS}
    saved_path, saved_argv, saved_env, saved_cwd = sys.path[:], sys.argv[:], os.environ.copy(), os.getcwd()
    try:
        for name, digest in PINS.items():
            source = api.checked_bytes(HERE / (name + '.py'), digest, 1 << 20)
            module = types.ModuleType(name)
            module.__file__ = str(HERE / (name + '.py'))
            sys.modules[name] = module
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
        yield sys.modules['runner']
    finally:
        for name, prior in saved_modules.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
        sys.path[:] = saved_path
        sys.argv[:] = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)
        os.chdir(saved_cwd)


@contextmanager
def route_children(runner, api, manifest, manifest_sha256, identity, launcher, launcher_sha256, records):
    require(api.valid_digest(launcher_sha256), 'explicit_child_launcher_digest_required')
    original = runner.child

    def routed(args, out, env, label, command, gb=12):
        strings = list(map(str, command))
        known = TARGETS.get(strings[1]) if len(strings) >= 2 else None
        if known is None:
            return original(args, out, env, label, command, gb)
        expected_label, target = known
        require(label == expected_label and strings[0] == args.python, 'unexpected_candidate_child_command')
        require(args.suite == 'quality' and args.taesd_key == identity['key']
                and Path(args.taesd_dir).absolute() == Path(manifest).parent,
                'runner_candidate_selection_mismatch')
        receipt = Path(out) / (label + '.candidate_child.json')
        require(not receipt.exists() and not receipt.is_symlink(), 'candidate_child_receipt_exists')
        output = Path(out) / 'taesd' if target == 'gate' else Path(out)
        wrapped = [args.python, '-B', '-c', CHILD_BOOTSTRAP, str(launcher), launcher_sha256,
                   '--enable', '--target', target,
                   '--manifest', str(manifest), '--manifest-sha256', manifest_sha256,
                   '--receipt', str(receipt), '--output-dir', str(output), '--', *strings[2:]]
        rc = original(args, out, env, label, wrapped, gb)
        try:
            records.append(validate_receipt(api, receipt, manifest_sha256, identity, target, rc))
        except (ValueError, OSError) as exc:
            # Let the unchanged runner write its normal INVALID report.
            raise runner.checks.Invalid(str(exc)) from None
        return rc

    runner.child = routed
    try:
        yield
    finally:
        runner.child = original


def execute(*, manifest, manifest_sha256, child_sha256, proof, runner_argv, enable=False):
    require(enable is True, 'candidate_quality_not_explicitly_enabled')
    api = runtime_api()
    proof = Path(proof)
    require(proof.is_absolute() and proof.parent.is_dir() and not proof.exists()
            and not any(p.is_symlink() for p in (proof, *proof.parents)), 'fresh_absolute_proof_required')
    manifest = Path(manifest)
    identity = selection(api, manifest, manifest_sha256)
    launcher = HERE / 'taesd_fp32_candidate_child.py'
    api.checked_bytes(launcher, child_sha256, 1 << 20)
    require(runner_argv and runner_argv[0] == 'quality', 'quality_suite_only')
    require(all(not x.startswith('--') or x.split('=', 1)[0] in RUNNER_OPTIONS for x in runner_argv[1:]),
            'unsupported_or_abbreviated_quality_runner_option')
    record = {'schema': 'taesd_fp32_candidate_quality_routing_v1', 'status': 'invalid',
              'manifest': str(manifest), 'manifest_sha256': manifest_sha256, **identity,
              'runner_sources': PINS, 'child_launcher_sha256': child_sha256, 'children': [],
              'quality_accepted': False, 'performance_accepted': False, 'release_ready': False,
              'original_gates_modified': False, 'input_manifest_modified': False}
    rc = 2
    try:
        with pinned_runner(api) as runner:
            sys.argv[:] = [str(HERE / 'runner.py'), *runner_argv]
            with route_children(runner, api, manifest, manifest_sha256, identity, launcher,
                                child_sha256, record['children']):
                rc = runner.main()
            require(rc in (0, 1, 2), 'unexpected_runner_exit')
            if rc in (0, 1):
                require([r['target'] for r in record['children']] == ['gate', 'render'],
                        'candidate_quality_child_coverage_missing')
                record['status'] = 'complete_original_gates_reported'
    except BaseException:
        rc = 2
        raise
    finally:
        record['returncode'] = rc
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        api._write_new(proof, api.json_bytes(record))
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enable', action='store_true')
    for name in ('manifest', 'manifest-sha256', 'child-sha256', 'proof'):
        p.add_argument('--' + name, required=True)
    p.add_argument('runner_args', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    require(args.runner_args[:1] == ['--'], 'explicit_runner_separator_required')
    return execute(manifest=args.manifest, manifest_sha256=args.manifest_sha256,
                   child_sha256=args.child_sha256, proof=args.proof,
                   runner_argv=args.runner_args[1:], enable=args.enable)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ValueError:
        print(json.dumps({'status': 'INVALID', 'reason': 'candidate_quality_contract_rejected'}))
        raise SystemExit(2) from None
