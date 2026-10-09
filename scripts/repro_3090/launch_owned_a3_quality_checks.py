"""Default-off CPU orchestration of unchanged canonical A3 numerical gates."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/workspace/MuseTalk')
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
WATCH_SHA = '2f03d2761e6319e27a58917099722bf279ce5d695e4682eaa5637de9d4db737f'
BOUND_WATCH_SHA = 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9'
WATCH565_SHA = '72d89c1917c15600560d64eec9c6a304bbe7f0ee4ec8864beee85eebe08b4606'
TARGETS = {'unet': ('scripts/validate_unet_backend.py', '81b74eddf5aaff8348ac27cce67b92763e937e09309762d137062b0a23f1d7a0'),
           'srccache': ('scripts/repro_400fps/srccache_exact.py', '59d27983e85f7c438d655dc4bc6dfc0fa739ac38f8d22482372a48e8e8908895'),
           'taesd': ('scripts/repro_400fps/gate_taesd_trt.py', '3baf8976e4fb25a908809e68d6ac126c07ea525baed7475a6443263221e28e99')}


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def owned_watch(here, document, digest, driver):
    if not document or not digest or driver not in ('595.91.07', '565.77'):
        raise ValueError('explicit owned descriptor and supported driver required')
    name, expected = ('watch_owned_single_leaf_565.py', WATCH565_SHA) if driver == '565.77' else (
        'watch_owned_single_leaf_target.py', BOUND_WATCH_SHA)
    path = here / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('selected owned monitor source changed')
    return [str(path), '--enable', '--owned-target-json', str(document),
            '--owned-target-sha256', digest, '--'], {name: expected}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--stage', choices=tuple(TARGETS), required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--engine-root', type=Path, required=True)
    p.add_argument('--engine-manifest-sha256', required=True)
    p.add_argument('--taesd-dir', type=Path, required=True)
    p.add_argument('--taesd-key', required=True)
    p.add_argument('--taesd-hardware', choices=('none', 'ampere_plus'), required=True)
    p.add_argument('--owned-target-json', type=Path)
    p.add_argument('--owned-target-sha256')
    p.add_argument('--owned-driver', choices=('595.91.07', '565.77'), default='595.91.07')
    p.add_argument('--successor-inputs', type=Path, default=BASE / 'harnesses/quality-inputs-a3-metadata-v1.json')
    p.add_argument('--successor-envelope', type=Path, default=BASE / 'quality/reference-envelope-a3-metadata-v1.json')
    p.add_argument('--lineage-receipt', type=Path, default=BASE / 'quality/a3-metadata-lineage-v1.json')
    p.add_argument('--lineage-receipt-sha256', default='16b252f9006c8888fabe82e026872e998ef4c2e958da8c02db876aa6e13a3bf4')
    a = p.parse_args(argv)
    if not a.execute:
        p.error('explicit owned execution required')
    if bool(a.owned_target_json) != bool(a.owned_target_sha256):
        p.error('exact owned descriptor and hash required together')
    if not re.fullmatch('[a-zA-Z0-9_-]{1,100}', a.label):
        p.error('safe label required')
    here = ROOT / 'scripts/repro_3090'; sys.path.insert(0, str(here))
    pins = {'runner.py': '0ede69c7a6f97aae57c38aec27e530b06beb1cc22175c89aa374a0fc66fb1f1d',
            'report.py': 'a102cb4ebdf7ed2f5e7828f3c061a2a43a6ed870a6a2e7fb72767f51b573ef8d',
            'preregister_metadata_quality_lineage.py': 'd5036b697c06efedcf95c9e81b99efd61129f0f07be50572f03296fabfc3e1ae',
            'watch_owned_single_leaf.py': WATCH_SHA}
    for name, digest in pins.items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != digest:
            raise ValueError('source pin mismatch: ' + name)
    deadline = '2026-10-09T02:45:00Z'
    watch_prefix = [str(here / 'watch_owned_single_leaf.py')]
    owned_target = None
    if a.owned_target_json:
        bound_path = here / 'watch_owned_single_leaf_target.py'
        if hashlib.sha256(bound_path.read_bytes()).hexdigest() != BOUND_WATCH_SHA:
            raise ValueError('bound watch source changed')
        import watch_owned_single_leaf_target as bound
        owned_target, end = bound.binding(a.owned_target_json, a.owned_target_sha256)
        if socket.gethostname() != owned_target['worker_hostname']:
            raise ValueError('wrong owned hostname')
        deadline = end.strftime('%Y-%m-%dT%H:%M:%SZ')
        watch_prefix, selected_pins = owned_watch(here, a.owned_target_json, a.owned_target_sha256, a.owned_driver)
        pins.update(selected_pins)
    elif a.owned_driver != '595.91.07':
        p.error('driver565 requires an explicit owned descriptor')
    elif socket.gethostname() != '1e7c09cffcb3':
        p.error('owned A3 identity required unless exact new descriptor supplied')
    import runner
    import report as checks
    import preregister_metadata_quality_lineage as lineage
    engine = runner.engine(a.engine_root)
    checks.require(engine['manifest_sha256'] == a.engine_manifest_sha256, 'engine manifest mismatch')
    meta_path = a.taesd_dir / ('taesd_trt_' + a.taesd_key + '.json')
    meta = checks.read(meta_path)
    checks.require(meta['key'] == a.taesd_key and meta['fingerprint']['batch'] == 8, 'TAESD identity mismatch')
    checks.require(meta['fingerprint'].get('hardware_compatibility_level', 'none') == a.taesd_hardware, 'TAESD hardware mismatch')
    for kind in ('decoder', 'post'):
        checks.require(checks.sha256(a.taesd_dir / meta[kind + '_plan']) == meta[kind + '_plan_sha256'], 'corrupt TAESD plan')
    verification = lineage.verify_preregistration(parent_inputs=BASE / 'harnesses/quality-inputs-v1.json',
        parent_envelope=BASE / 'quality/reference-envelope-v2.json', successor_inputs=a.successor_inputs,
        successor_envelope=a.successor_envelope, receipt_path=a.lineage_receipt,
        expected_receipt_sha256=a.lineage_receipt_sha256)
    os.umask(0o077); out = BASE / 'quality' / a.label; out.mkdir(mode=0o700, exist_ok=False)
    write(out / 'input_lineage_verified.json', {**verification, 'verified_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    values = runner.profile(here / 'profiles/native.env')
    values.update(MUSETALK_UNET_STAGEWISE_CACHE_DIR=str(a.engine_root), MUSETALK_UNET_STAGEWISE_PROBE_CHECK='1',
        MUSETALK_UNET_STAGEWISE_PROBE_TOL='0', MUSETALK_TAESD_TRT_DIR=str(a.taesd_dir), MUSETALK_TAESD_TRT_HW_COMPAT=a.taesd_hardware)
    if a.stage == 'taesd':
        values.update(MUSETALK_TAESD_COMPILE='1', MUSETALK_TAESD_COMPILE_MODE='max-autotune')
    profile = out / 'effective.env'
    with profile.open('x') as handle:
        handle.write(''.join(k + '=' + v + '\n' for k, v in sorted(values.items())))
    env, removed = runner.clean_environment(os.environ, values)
    env.update(BOX_GUARD_LEASE_FILE='/workspace/.gpu_lease', MUSETALK_REPRO_RUNTIME_ENV=str(profile),
        MUSETALK_REPRO_ACCEPTED='/workspace/experiments/avatar_diversity_20260927', MUSETALK_REPRO_WORKSPACE='/workspace', REPRO_GATE_OUT=str(out / 'taesd'))
    write(out / 'selection.json', {'engine': engine, 'taesd': meta, 'source_pins': pins, 'effective_profile': values,
        'discarded_ambient_keys': removed, 'owned_target': owned_target, 'quality_accepted': False, 'release_ready': False})
    target, target_sha = TARGETS[a.stage]
    calls = []
    corpus = ROOT / 'calibration/unet_multi_avatar_20260928'
    if a.stage == 'unet':
        for split in ('main', 'holdout'):
            calls.append((split, ['--backend', 'runtime', '--capture-dir', str(corpus / 'holdout' if split == 'holdout' else corpus),
                '--padded-batch-size', '8', '--group-captures', '2', '--limit', '0', '--warmup', '1', '--iters', '2',
                '--fail-mae', '0.01', '--fail-max-abs', '0.5', '--report-path', str(out / ('unet_' + split + '.json'))]))
    elif a.stage == 'srccache':
        calls.append(('srccache', ['--root', str(a.engine_root), '--baseline-root', str(out / 'intentionally_no_sm89_baseline'),
                                 '--corpus', str(corpus), '--out', str(out / 'srccache.json')]))
    else:
        calls.append(('taesd', ['--no-record', '--corpus', str(corpus)]))
    records = []
    for name, arguments in calls:
        command = ['/bin/bash', 'scripts/box_guard.sh', 'run', '--wait-min', '0', '--min-avail-gb', '8', '--label', a.label + '_' + name,
            '--', '/workspace/.venvs/musetalk_trt_stagewise/bin/python', *watch_prefix, '--enable',
            '--out', str(out / (name + '.owned_watch.jsonl')), '--target', str(ROOT / target), '--target-sha256', target_sha,
            '--deadline-utc', deadline, '--', *arguments]
        start = time.monotonic(); started = dt.datetime.now(dt.timezone.utc).isoformat()
        with (out / (name + '.log')).open('x') as handle:
            run = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'status': 'RUNNING', 'stage': name, 'guard_pid': run.pid, 'out': str(out)}), flush=True)
            timed_out = False
            try:
                rc = run.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(run.pid, signal.SIGTERM)
                try:
                    rc = run.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid, signal.SIGKILL); rc = run.wait(timeout=10)
        record = {'stage': name, 'returncode': rc, 'started_utc': started,
                  'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'wall_s': time.monotonic() - start, 'command': command,
                  'timed_out': timed_out}
        write(out / (name + '.child.json'), record); records.append(record)
        checks.require(not timed_out and rc in (0, 1), 'canonical gate invalid/crashed/timed out')
        if a.stage == 'unet':
            summary = checks.read(out / ('unet_' + name + '.json'))['summary']
            checks.require(summary['files'] == (176 if name == 'main' else 48), 'incomplete UNet corpus')
            checks.require(rc == (0 if summary['mae_max'] <= .01 and summary['max_abs_max'] <= .5 else 1), 'UNet exit differs from unchanged thresholds')
        elif a.stage == 'taesd':
            gate = checks.read(out / 'taesd/gate_taesd_trt.json')
            checks.require(gate['engine']['key'] == a.taesd_key and gate['gate']['bit_exact_checks'] == 'PASS', 'wrong decoder or hard exactness failure')
    write(out / 'children.json', {'children': records, 'quality_accepted': False, 'full698_assessed': False})
    print(json.dumps({'status': 'ORIGINAL_GATES_COMPLETED', 'out': str(out), 'children': records}), flush=True)
    return 1 if any(row['returncode'] for row in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
