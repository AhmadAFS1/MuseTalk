#!/usr/bin/env python3
"""Matched continuous delivery trials; signed tickets remain memory-only."""
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import benchmark_dependency_delivery as benchmark
from dependency_artifacts import require
from fetch_dependency_wheels import sha256_file


def execute(deadline, envelope, attempt="a1"):
    require(platform.system()=="Linux" and platform.machine()=="x86_64", "Matching Linux only")
    root=Path('/workspace/.dependency-optimization-20261015-a1')
    require(json.loads((root/'local-matrix-a1/receipt.json').read_text())['state']=='complete', 'Prior local matrix is not terminal/passed')
    require(attempt in {'a1','a2'}, 'Unreviewed delivery attempt')
    if attempt=='a2':
        previous=json.loads((root/'delivery-source-control-1/receipt.json').read_text())
        require(previous['state']=='failed' and previous.get('detail')=='Artifact HTTP failure 400', 'Different previous failure; investigate before recovery')
    control=root/('delivery-matrix-'+attempt)
    require(not control.exists(), 'Delivery matrix already claimed; inspect same job')
    control.mkdir(mode=0o700)
    stop=dt.datetime.fromisoformat(deadline).timestamp()-300
    events=[];trials=[]
    def event(stage,**fields):
        row={'stage':stage,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**fields};events.append(row)
        with (control/'events.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def retire(name):
        outer=root/name;inner=root/('inner-'+name)
        owner=json.loads((inner/'owned-trial.json').read_text());receipt=json.loads((inner/'receipt.json').read_text())
        require(owner['uid']==os.getuid() and owner['trial']==inner.name and receipt['state']=='microbenchmark-passed', 'No accepted owned inner trial')
        require(json.loads((outer/'receipt.json').read_text())['state']=='delivery-trial-passed', 'Delivery trial not accepted')
        paths=[inner/'venv',inner/'package-cache',outer/'wheels']
        before=shutil.disk_usage(root).free
        for p in paths:
            require(p.is_dir() and not p.is_symlink() and p.resolve().parent in {inner.resolve(),outer.resolve()}
                    and p.parent.parent==root, 'Unsafe disposable delivery child')
            shutil.rmtree(p)
        value={'paths':[str(p) for p in paths],'receipt_sha256':sha256_file(outer/'receipt.json'),
               'reclaimed_free_bytes':shutil.disk_usage(root).free-before,'recovery':'exact manifest/wheels preserved separately; trial logs and receipts retained'}
        (outer/'retirement.json').write_text(json.dumps(value,indent=2)+'\n')
        event('DELIVERY_TRIAL_RETIRED',trial=name,**value)
    try:
        cases=[('source-pip',1,'source-control-1'),('wheelhouse-pip',1,'locked-pip-serial-1'),
               ('wheelhouse-uv',2,'uv-prefetch-2-1'),('wheelhouse-uv',4,'uv-prefetch-4-1'),
               ('source-pip',1,'source-control-2'),('wheelhouse-uv',4,'uv-prefetch-4-2'),
               ('source-pip',1,'source-control-3'),('wheelhouse-uv',4,'uv-prefetch-4-3')]
        for method,workers,suffix in cases:
            require(time.time()+1000<stop,'Insufficient lease reserve for another full delivery trial')
            name='delivery-'+('a2-' if attempt=='a2' else '')+suffix
            event('DELIVERY_TRIAL_STARTED',trial=name,method=method,workers=workers)
            args=SimpleNamespace(experiment_root=root,trial=name,contract=root/('tools-v5' if attempt=='a2' else 'tools-v4')/'dependency-contract.json',
                method=method,workers=workers,group='full',repo_root=Path('/workspace/MuseTalk'),
                reference_wheels=root/'wheelhouse-build-a2/wheels')
            receipt=benchmark.run(args,envelope)
            require(receipt['state']=='delivery-trial-passed','Full delivery trial failed: '+name)
            value={k:receipt[k] for k in ('trial','method','workers','packages_ready_seconds','validated_ready_seconds')}
            value['acquisition_seconds']=receipt['acquisition']['seconds']
            trials.append(value)
            event('DELIVERY_TRIAL_PASSED',**value)
            retire(name)
        event('DELIVERY_MATRIX_COMPLETE',scope='dependency acquisition/installation/import/CUDA checks, not a complete Vast boot')
        result={'state':'complete','trials':trials,'events':events}
    except BaseException as error:
        event('DELIVERY_MATRIX_FAILED',error_type=type(error).__name__,detail=str(error) if type(error) is ValueError else 'Inspect protected trial receipts')
        result={'state':'failed','trials':trials,'events':events}
    (control/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
