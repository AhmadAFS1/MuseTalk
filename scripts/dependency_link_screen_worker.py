#!/usr/bin/env python3
"""Local copy-versus-hardlink screen; no remote acquisition or default switch."""
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import benchmark_dependency_install as bench
from dependency_artifacts import require
import fetch_dependency_wheels as fetch
import install_locked_dependencies as locked


def execute(root,deadline,attempt='a1'):
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Matching Linux only')
    require(root==Path('/workspace/.dependency-canary-20261016-a1') and root.is_dir()
            and root.resolve()==root and root.stat().st_uid==os.getuid(),'Owned canary root required')
    require(attempt in {'a1','a2'},'Unreviewed link screen attempt')
    if attempt=='a2':
        prior=json.loads((root/'link-screen-a1/receipt.json').read_text())
        require(prior['state']=='failed' and prior.get('detail')=='Explicit experiment root required',
                'Different prior failure; do not replay')
    out=root/('link-screen-'+attempt);require(not out.exists(),'Link screen already claimed')
    require(time.time()<dt.datetime.fromisoformat(deadline).timestamp()-420,'Insufficient link-screen lease')
    out.mkdir(mode=0o700);rows=[]
    # Reuse the benchmark's existing dedicated-root guard rather than weaken it.
    benchroot=Path('/workspace/.dependency-optimization-canary-links-20261016-a1')
    require(not benchroot.exists(),'Benchmark namespace already exists')
    artifact=Path('/workspace/.cache/musetalk-dependencies/725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194')
    contract=json.loads(Path('/workspace/MuseTalk/configs/dependencies/rtx3090-cp310-cu121-screening.json').read_text())
    result={'state':'running','scope':'one ordered local-wheel copy/hardlink pair; not fresh delivery or full boot','cases':rows}
    try:
        for mode in ('copy','hardlink'):
            require(time.time()<dt.datetime.fromisoformat(deadline).timestamp()-300,'Lease reserve before local link case')
            trial='link-'+mode+'-1'
            value=bench.run(SimpleNamespace(experiment_root=benchroot,trial=trial,manifest=artifact/'manifest.json',
                manifest_sha256=contract['manifest_sha256'],wheels=artifact/'wheels',group='full',installer='uv',
                compile_bytecode=True,link_mode=mode,python='python3.10',uv=artifact/'uv-tool/uv',
                uv_sha256=contract['uv_binary_sha256'],source_pip=False,repo_root=Path('/workspace/MuseTalk')))
            require(value['state']=='microbenchmark-passed' and value['package_consistency']['exit_status']==0,
                    'Link mode failed native/inventory/consistency checks')
            case=benchroot/trial;venv=case/'venv';cache=case/'package-cache'
            require(venv.stat().st_dev==cache.stat().st_dev,'Cross-filesystem link comparison')
            sample=venv/'lib/python3.10/site-packages/torch/__init__.py'
            sha=fetch.sha256_file(sample);links=sample.stat().st_nlink
            require(mode!='hardlink' or links>1,'Requested hardlink was not realized')
            cleaned=bench.command([str(artifact/'uv-tool/uv'),'cache','clean','--cache-dir',str(cache)],
                                  'supported-cache-clean',case,dict(os.environ))
            require(cleaned['exit_status']==0 and fetch.sha256_file(sample)==sha,'Cache clean changed installed file')
            native=bench.command([str(venv/'bin/python'),'-B','-c',bench.GPU_PROBE],
                                 'gpu-after-cache-clean',case,dict(os.environ))
            require(native['exit_status']==0,'Installed env failed after supported cache clean')
            row={'mode':mode,'packages_seconds':value['packages_ready_seconds'],
                 'validated_seconds':value['validated_ready_seconds'],'sample_nlink_before_clean':links,
                 'sample_nlink_after_clean':sample.stat().st_nlink,'installed_file_hash_unchanged':True,
                 'GPU_after_cache_clean':True,'same_filesystem':True}
            rows.append(row)
            (case/'link-evidence.json').write_text(json.dumps(row,indent=2)+'\n')
            owner=json.loads((case/'owned-trial.json').read_text())
            require(owner['uid']==os.getuid() and owner['trial']==trial and not locked.active_venv(venv),
                    'Cannot retire foreign or active test venv')
            for path in (venv,cache):
                require(path.parent==case and path.resolve()==path and not path.is_symlink(),'Unsafe trial retirement child')
                if path.exists():shutil.rmtree(path)
            print(json.dumps({'stage':'LINK_CASE_PASSED',**row}),flush=True)
        result.update(state='passed',validated_reduction_percent=100*(1-rows[1]['validated_seconds']/rows[0]['validated_seconds']),
                      limits=['one ordered screen per method; no median or fleet effect','remote acquisition excluded','copy default unchanged'])
    except BaseException as error:
        result.update(state='failed',error_type=type(error).__name__,
                      detail=str(error) if type(error) is ValueError else 'Protected diagnostics retained')
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result
