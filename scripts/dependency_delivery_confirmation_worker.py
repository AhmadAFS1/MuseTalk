#!/usr/bin/env python3
"""One clean B6 confirmation after integration; only disposable test cleanup."""
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import benchmark_dependency_delivery as delivery
from dependency_artifacts import require
from fetch_dependency_wheels import sha256_file
from install_locked_dependencies import active_venv
from dependency_s3_range_probe import SHA,KEY

ROOT=Path('/workspace/.dependency-optimization-20261015-a1')


def execute(envelope,deadline):
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Matching Linux only')
    stop=dt.datetime.fromisoformat(deadline).timestamp()-300
    while time.time()+300<stop:
        p=ROOT/'canonical-integration-a1/receipt.json'
        if p.exists():
            require(json.loads(p.read_text())['state']=='passed','Canonical integration did not pass');break
        time.sleep(3)
    else:raise ValueError('No lease reserve for clean confirmation')
    out=ROOT/'confirmation-a1';require(not out.exists(),'Confirmation already claimed; inspect same job')
    out.mkdir(mode=0o700);result={'state':'running'}
    try:
        restore=json.loads((ROOT/'s3-full-a1/receipt.json').read_text())
        require(restore['state']=='restored-native-and-health-passed' and restore['original_venv_restored']
                and restore['original_API_healthy'],'No accepted/restored S3 experiment')
        paths=[ROOT/'s3-full-a1/single-get.tar.zst',ROOT/'s3-full-a1/range-get.tar.zst']
        before=shutil.disk_usage(ROOT).free
        for p in paths:
            require(p.is_file() and not p.is_symlink() and p.stat().st_uid==os.getuid()
                    and sha256_file(p)==SHA,'No verified owned disposable archive')
            p.unlink()
        trial=ROOT/'uv-nocompile-rep-1';owner=json.loads((trial/'owned-trial.json').read_text());receipt=json.loads((trial/'receipt.json').read_text())
        venv=trial/'venv'
        require(owner['uid']==os.getuid() and owner['trial']==trial.name
                and receipt['state']=='microbenchmark-passed' and receipt['compile_bytecode'] is False,
                'No accepted owned no-compile environment')
        require(venv.is_dir() and not venv.is_symlink() and venv.resolve().parent==trial
                and not active_venv(venv),'Unsafe or active disposable no-compile environment')
        shutil.rmtree(venv)
        result['retired']={'paths':[str(p) for p in paths]+[str(venv)],
            'reclaimed_bytes':shutil.disk_usage(ROOT).free-before,
            'recovery':'archives remain in immutable private S3; exact wheels/locks and all receipts retained',
            'remote_archive_key':KEY,'archive_sha256':SHA,
            'source_compiled_uv_optin_envs_preserved':True}
        (out/'retirement.json').write_text(json.dumps(result['retired'],indent=2)+'\n')
        args=SimpleNamespace(experiment_root=ROOT,trial='delivery-clean-confirmation-4-4',
            contract=ROOT/'tools-v9/dependency-contract.json',method='wheelhouse-uv',workers=4,
            group='full',repo_root=Path('/workspace/MuseTalk'),reference_wheels=ROOT/'wheelhouse-build-a2/wheels')
        result['delivery']=delivery.run(args,envelope)
        require(result['delivery']['state']=='delivery-trial-passed','Clean delivery confirmation failed')
        result['state']='passed'
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
            detail=str(error) if type(error) is ValueError else 'Inspect protected confirmation logs')
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
