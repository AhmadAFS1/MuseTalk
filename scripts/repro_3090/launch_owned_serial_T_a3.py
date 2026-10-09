"""CPU-only, default-off A3 T launcher; exact inputs, one guarded CUDA leaf."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/workspace/MuseTalk')
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
PINS = {
    'watch_owned_single_leaf.py': 'b25c3db443fbe08ee00f14f9164f3124759817f0a043f4deb06fec5114a1205a',
    'run_owned_legacy_serial_render_a3.py': '91a12437eb9be6c55905626218592c5680b7f8010d783ea1f16d3e23354c06dd',
    'runner.py': '0ede69c7a6f97aae57c38aec27e530b06beb1cc22175c89aa374a0fc66fb1f1d',
    'report.py': 'a102cb4ebdf7ed2f5e7828f3c061a2a43a6ed870a6a2e7fb72767f51b573ef8d',
    'preregister_metadata_quality_lineage.py': 'd5036b697c06efedcf95c9e81b99efd61129f0f07be50572f03296fabfc3e1ae',
}


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def capture(command):
    return subprocess.check_output(command, cwd=ROOT, text=True, timeout=20).strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--label', required=True)
    p.add_argument('--engine-root', type=Path, required=True)
    p.add_argument('--taesd-dir', type=Path, required=True)
    p.add_argument('--taesd-key', required=True)
    p.add_argument('--taesd-hardware', choices=('none', 'ampere_plus'), required=True)
    args = p.parse_args(argv)
    if not args.execute or socket.gethostname() != '1e7c09cffcb3':
        p.error('explicit owned A3 execution required')
    if not re.fullmatch('[a-zA-Z0-9_-]{1,100}', args.label) or not re.fullmatch('[0-9a-f]{20}', args.taesd_key):
        p.error('invalid label or TAESD key')
    here = ROOT / 'scripts/repro_3090'
    sys.path.insert(0, str(here))
    for name, digest in PINS.items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != digest:
            raise ValueError('checked source mismatch: ' + name)
    import runner
    import report as checks
    import preregister_metadata_quality_lineage as lineage
    out = BASE / 'native' / args.label
    os.umask(0o077); out.mkdir(mode=0o700, exist_ok=False)
    verification = lineage.verify_preregistration(
        parent_inputs=BASE / 'harnesses/quality-inputs-v1.json',
        parent_envelope=BASE / 'quality/reference-envelope-v2.json',
        successor_inputs=BASE / 'harnesses/quality-inputs-a3-metadata-v1.json',
        successor_envelope=BASE / 'quality/reference-envelope-a3-metadata-v1.json',
        receipt_path=BASE / 'quality/a3-metadata-lineage-v1.json',
        expected_receipt_sha256='16b252f9006c8888fabe82e026872e998ef4c2e958da8c02db876aa6e13a3bf4')
    verification['verified_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write(out / 'input_lineage_verified.json', verification)
    engine = runner.engine(args.engine_root)
    meta_path = args.taesd_dir / ('taesd_trt_' + args.taesd_key + '.json')
    meta = checks.read(meta_path)
    checks.require(meta['key'] == args.taesd_key, 'TAESD key mismatch')
    fp = meta['fingerprint']
    checks.require(hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:20] == args.taesd_key, 'TAESD fingerprint mismatch')
    checks.require(fp.get('hardware_compatibility_level', 'none') == args.taesd_hardware and fp['batch'] == 8 and fp['opt_level'] == 3, 'TAESD profile mismatch')
    for kind in ('decoder', 'post'):
        checks.require(checks.sha256(args.taesd_dir / meta[kind + '_plan']) == meta[kind + '_plan_sha256'], 'TAESD corrupt plan')
    smi_fields = 'name,uuid,compute_cap,memory.total,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu'
    rows = list(csv.reader(capture(['nvidia-smi', '--query-gpu=' + smi_fields, '--format=csv,noheader,nounits']).splitlines()))
    checks.require(len(rows) == 1, 'one physical GPU required')
    gpu = dict(zip(smi_fields.split(','), (x.strip() for x in rows[0])))
    checks.require(gpu['uuid'] == 'GPU-ea6411bc-775f-6685-f1a4-28b6b4011a3d' and gpu['name'] == 'NVIDIA GeForce RTX 3090'
                   and gpu['compute_cap'] == '8.6' and gpu['driver_version'] == '595.91.07', 'owned GPU identity changed')
    apps = capture(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    checks.require(not apps, 'GPU not isolated')
    values = runner.profile(here / 'profiles/native.env')
    values.update(MUSETALK_UNET_STAGEWISE_CACHE_DIR=str(args.engine_root), MUSETALK_UNET_STAGEWISE_PROBE_CHECK='1',
                  MUSETALK_UNET_STAGEWISE_PROBE_TOL='0', MUSETALK_TAESD_TRT_DIR=str(args.taesd_dir),
                  MUSETALK_TAESD_TRT_HW_COMPAT=args.taesd_hardware)
    profile = out / 'effective.env'
    with profile.open('x') as handle:
        handle.write(''.join(k + '=' + v + '\n' for k, v in sorted(values.items())))
    env, removed = runner.clean_environment(os.environ, values)
    env.update(BOX_GUARD_LEASE_FILE='/workspace/.gpu_lease', MUSETALK_REPRO_RUNTIME_ENV=str(profile),
               MUSETALK_REPRO_ACCEPTED='/workspace/experiments/avatar_diversity_20260927', MUSETALK_REPRO_WORKSPACE='/workspace')
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    cgroup = {name: (Path('/sys/fs/cgroup') / name).read_text().strip() for name in
              ('memory.max', 'memory.current', 'cpu.max', 'cpuset.cpus.effective') if (Path('/sys/fs/cgroup') / name).is_file()}
    environment = {'schema': 'owned_a3_serial_T_environment_v1', 'measurement_class': 'current_owned_host',
        'gpu': gpu, 'engines': [engine], 'taesd': {'key': args.taesd_key, 'manifest_sha256': checks.sha256(meta_path),
        'fingerprint': fp, 'decoder_plan_sha256': meta['decoder_plan_sha256'], 'post_plan_sha256': meta['post_plan_sha256']},
        'input_lineage': verification, 'source_pins': PINS, 'profile_sha256': checks.sha256(profile),
        'effective_profile': values, 'discarded_ambient_keys': removed, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'memory': memory, 'cgroup': cgroup, 'disk_free_bytes': shutil.disk_usage(ROOT).free,
        'shm': dict(zip(('total', 'used', 'free'), shutil.disk_usage('/dev/shm'))),
        'git_revision': capture(['git', 'rev-parse', 'HEAD']), 'git_status': capture(['git', 'status', '--porcelain']),
        'foreign_gpu_apps_at_preflight': apps, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(out / 'environment.json', environment)
    command = ['/bin/bash', 'scripts/box_guard.sh', 'run', '--wait-min', '0', '--min-avail-gb', '12', '--label', args.label,
        '--', '/workspace/.venvs/musetalk_trt_stagewise/bin/python', str(here / 'watch_owned_single_leaf.py'), '--enable',
        '--out', str(out / 'owned_watch.jsonl'), '--target', str(here / 'run_owned_legacy_serial_render_a3.py'),
        '--target-sha256', PINS['run_owned_legacy_serial_render_a3.py'], '--deadline-utc', '2026-10-09T02:45:00Z', '--',
        '--enable', '--stage', 'T', '--output-dir', str(out), '--label', args.label]
    start = time.monotonic(); started = dt.datetime.now(dt.timezone.utc).isoformat(); timed_out = False
    with (out / 'child.log').open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'status': 'RUNNING', 'outer_pid': os.getpid(), 'guard_pid': process.pid, 'out': str(out), 'started_utc': started}), flush=True)
        try:
            rc = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(process.pid, signal.SIGTERM)
            try:
                rc = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); rc = process.wait(timeout=10)
    child = {'started_utc': started, 'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'wall_s': time.monotonic() - start, 'returncode': rc, 'timed_out': timed_out, 'command': command}
    write(out / 'child.json', child)
    if rc:
        print(json.dumps(child), flush=True); return 2
    data = checks.read(out / (args.label + '.json'))
    checks.loaded_backend(data, args.engine_root, args.taesd_key, meta['decoder_plan_sha256'], engine['manifest'])
    result = checks.aggregate(data, 'T', 400)
    write(out / 'full_recipe_result.json', {'schema': 'owned_a3_full_recipe_T_v1', 'result': result,
        'quality_accepted': False, 'release_ready': False, 'runtime': data['versions'], 'environment': environment,
        'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
