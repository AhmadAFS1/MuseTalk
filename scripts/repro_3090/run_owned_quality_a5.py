"""CPU orchestration of the exact quality runner with owned GPU isolation.

Only execution/ownership and completed serial reporting are adapted. Original
quality calculations, workload, gates and failure propagation remain intact.
The original runner's CUDA runtime probe is itself guarded, not run unleased.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types

ROOT = Path('/workspace/MuseTalk')
HERE = ROOT / 'scripts/repro_3090'
PINS = {'runner.py':'0ede69c7a6f97aae57c38aec27e530b06beb1cc22175c89aa374a0fc66fb1f1d',
        'watch_owned_single_leaf_target.py':'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9',
        'watch_owned_single_leaf_565.py':'72d89c1917c15600560d64eec9c6a304bbe7f0ee4ec8864beee85eebe08b4606'}
TARGETS = {
    'validate_unet_backend.py':('scripts/validate_unet_backend.py','81b74eddf5aaff8348ac27cce67b92763e937e09309762d137062b0a23f1d7a0'),
    'srccache_exact.py':('scripts/repro_400fps/srccache_exact.py','59d27983e85f7c438d655dc4bc6dfc0fa739ac38f8d22482372a48e8e8908895'),
    'gate_taesd_trt.py':('scripts/repro_400fps/gate_taesd_trt.py','3baf8976e4fb25a908809e68d6ac126c07ea525baed7475a6443263221e28e99'),
    'run_owned_legacy_serial_render_a3.py':('scripts/repro_3090/run_owned_legacy_serial_render_a3.py','e23604396244ecdc68c3531a2c072b93b1ad784849fe991c7abcb7a1a072fd78'),
    'owned_runtime_probe.py':('scripts/repro_3090/owned_runtime_probe.py','67bc64fb3d3330459ddc42df040525f32652230f32dd584697d2220df19e76ae')}


def checked(path, digest):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('absolute nonsymlink source required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('checked source changed')
    return raw


def module(name):
    path = HERE / name
    raw = checked(path, PINS[name])
    m = types.ModuleType('_checked_' + path.stem); m.__file__ = str(path)
    sys.path.insert(0,str(HERE)); exec(compile(raw,str(path),'exec'),m.__dict__)
    return m


def quality_capture_arguments(out, label, descriptor, digest):
    return ['--enable','--stage','Q','--output-dir',str(out),'--label',label,
            '--owned-target-json',str(descriptor),'--owned-target-sha256',digest]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--owned-target-json',type=Path,required=True)
    p.add_argument('--owned-target-sha256',required=True)
    p.add_argument('runner_arguments',nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if not a.execute or a.runner_arguments[:2] != ['--','quality']:
        raise ValueError('explicit owned quality execution required')
    owned, deadline = module('watch_owned_single_leaf_target.py').binding(a.owned_target_json,a.owned_target_sha256)
    if socket.gethostname() != owned['worker_hostname'] or (deadline-dt.datetime.now(dt.timezone.utc)).total_seconds() <= 600:
        raise ValueError('owned host or cleanup margin changed')
    checked(HERE/'watch_owned_single_leaf_565.py',PINS['watch_owned_single_leaf_565.py'])
    runner = module('runner.py')
    arguments = a.runner_arguments[1:]
    out = Path(arguments[arguments.index('--out')+1]) / (arguments[arguments.index('--label')+1]+'_quality')
    if out.parent != ROOT/'docs/fps_comparisons/rtx3090_r5_20261008/quality' or out.exists():
        raise ValueError('fresh scoped quality output required')
    records = []
    def gpu_call(label,target,child_args,env,gb):
        if dt.datetime.now(dt.timezone.utc) >= deadline-dt.timedelta(seconds=120):
            raise ValueError('cleanup margin reached')
        relative,digest = TARGETS[target]
        checked(ROOT/relative,digest)
        command = ['/bin/bash','scripts/box_guard.sh','run','--wait-min','0','--min-avail-gb',str(gb),
            '--label','a5_quality_'+label,'--','/workspace/.venvs/musetalk_trt_stagewise/bin/python',
            str(HERE/'watch_owned_single_leaf_565.py'),'--enable','--owned-target-json',str(a.owned_target_json),
            '--owned-target-sha256',a.owned_target_sha256,'--','--enable','--out',str(out/(label+'.owned_watch.jsonl')),
            '--target',str(ROOT/relative),'--target-sha256',digest,'--deadline-utc',deadline.strftime('%Y-%m-%dT%H:%M:%SZ'),'--',*map(str,child_args)]
        clean = {**env,'BOX_GUARD_LEASE_FILE':'/workspace/.gpu_lease'}
        start = time.monotonic()
        with (out/(label+'.log')).open('x') as log:
            run = subprocess.run(command,cwd=ROOT,env=clean,stdout=log,stderr=subprocess.STDOUT,
                                 timeout=max(1,(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()-120))
        records.append({'label':label,'returncode':run.returncode,'wall_s':time.monotonic()-start,
                        'target_sha256':digest,'ownership_monitor_sha256':PINS['watch_owned_single_leaf_565.py']})
        return run.returncode
    old_capture = runner.capture
    def capture(command,env=None,*,stage='unspecified',timeout_s=30):
        if stage != 'runtime_import':
            return old_capture(command,env,stage=stage,timeout_s=timeout_s)
        rc = gpu_call('runtime_probe','owned_runtime_probe.py',['--enable'],env or os.environ,8)
        if rc:
            raise ValueError('guarded runtime probe failed')
        lines = [line[len('OWNED_RUNTIME '):] for line in (out/'runtime_probe.log').read_text().splitlines() if line.startswith('OWNED_RUNTIME ')]
        if len(lines) != 1:
            raise ValueError('runtime probe missing or ambiguous')
        json.loads(lines[0]); return lines[0]
    def child(args,output,env,label,command,gb=12):
        target = Path(command[1]).name
        if target == 'quality_ab_metrics.py':
            if command[2] != 'pair' or '--syncnet' in command:
                raise ValueError('only original CPU pair metrics permitted')
            start = time.monotonic()
            with (output/(label+'.log')).open('x') as log:
                run = subprocess.run(list(map(str,command)),cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':''},
                    stdout=log,stderr=subprocess.STDOUT,timeout=min(600,max(1,(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()-120)))
            records.append({'label':label,'returncode':run.returncode,'wall_s':time.monotonic()-start,'measurement_scope':'original CPU pair metrics; CUDA hidden'})
            return run.returncode
        if target == 'chin_multistream_render.py':
            target = 'run_owned_legacy_serial_render_a3.py'
            label_arg = str(command[command.index('--label')+1])
            child_args = quality_capture_arguments(output,label_arg,a.owned_target_json,a.owned_target_sha256)
        else:
            child_args = command[2:]
        return gpu_call(label,target,child_args,env,gb)
    runner.capture = capture; runner.child = child
    sys.argv = [str(HERE/'runner.py'),*arguments]
    try:
        return runner.main()
    finally:
        if out.is_dir():
            with (out/'owned_execution.json').open('x') as record:
                json.dump(dict(owned_target=owned,source_pins=PINS,targets=TARGETS,children=records,
                    canonical_quality_math_changed=False,canonical_renderer_changed=False,
                    all_native_artifact_claim=False,quality_accepted=False,release_ready=False),record,indent=2)


if __name__ == '__main__': raise SystemExit(main())
