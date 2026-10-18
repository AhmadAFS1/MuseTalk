#!/usr/bin/env python3
"""Offline evidence consistency audit; never runs pip, CUDA or network calls.

This validates the captured evidence's coverage and calculations, not a new
native execution or production rollout. The Markdown audit owns conditional
branch decisions and disclosed limitations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from dependency_artifacts import require, validate_manifest
from summarize_dependency_optimization import summarize


def verify(folder):
    names=('results.json','canary-results.json','final-screen-results.json','wheelhouse-manifest-a2.json')
    data={};hashes={}
    for name in names:
        raw=(folder/name).read_bytes()
        require(not re.search(rb'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|X-Amz-(?:Credential|Signature|Security-Token)',
                              raw,re.I),'Credential pattern in evidence')
        data[name]=json.loads(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    a,b,c=(data[n] for n in names[:3]);manifest=validate_manifest(data[names[3]])
    contract=json.loads((folder.parent.parent/'configs/dependencies/rtx3090-cp310-cu121-screening.json').read_text())
    require(hashes[names[3]]==contract['manifest_sha256']
            and manifest['artifact_id']==contract['artifact_id'],'Independent manifest trust differs')
    require(len(manifest['artifacts'])==194,'Different full package closure')
    trials=a['trials_and_builds'];computed=summarize(a)
    controls=next(x for x in computed['clean_summary'] if x['method']=='source-pip')
    winner=next(x for x in computed['clean_summary'] if x['method']=='wheelhouse-uv' and x['workers']==4)
    require(controls['n']==3 and winner['n']==3,'Missing clean matched repetitions')
    for name in controls['trials']+winner['trials']:
        row=trials[name];inner=row['installation']
        require(row['state']=='delivery-trial-passed' and row['remote_acquisition_included']
                and not row['full_Vast_boot_included'] and row['group']=='full','Wrong comparison boundary')
        require(inner['state']=='microbenchmark-passed' and not inner['inventory_diff']
                and inner['package_consistency']['exit_status']==0,'Native inventory/consistency failed')
        require(all(s['exit_status']==0 for s in inner['steps'])
                and 'gpu-imports-and-execution' in [s['step'] for s in inner['steps']], 'Missing CUDA validation')
        require(row['acquisition']['cache_hits']==0,'Destination cache was not fresh')
    require(winner['median_validated_reduction_vs_source_percent']>=20,'Winner missed adoption target')
    videos=[]
    for key in ('application_matrix','cold_jit_application_matrix'):
        matrix=a[key];require(matrix['state']=='passed-output-and-timing-captured','Application matrix failed')
        for case in matrix['cases']:
            require(len(case['samples'])==4,'Missing repeated application samples')
            for sample in case['samples']:
                require({s['codec_type'] for s in sample['streams']}=={'audio','video'},'Missing decoded media track')
                videos.append(sample['sha256'])
    require(len(videos)==28 and len(set(videos))==1,'Controlled video equivalence differs')
    require(trials['followup-a1']['state']=='passed' and trials['canonical-integration-a1']['state']=='passed'
            and trials['warm-start-a1']['state']=='passed','Missing reuse/fault/canonical evidence')
    accepted=b['acceptance_a2'];require(accepted['state']=='passed' and len(accepted['cases'])==2,'Fresh canary failed')
    installed=accepted['native_install_evidence']['locked_install']
    require(installed['state']=='accepted' and not installed['inventory_diff'],'Fresh full install failed')
    require(len({x['video_sha256'] for x in accepted['cases']})==1,'Fresh repeated outputs differ')
    s3=trials['s3-full-a1'];require(s3['state']=='restored-native-and-health-passed' and s3['original_venv_restored']
                                and s3['original_API_healthy'],'First S3 rollback evidence missing')
    failed_s3=b['extra_test_receipts']['s3-full-a1']
    require(failed_s3['state']=='failed' and failed_s3['original_venv_restored']
            and not failed_s3['original_API_healthy'],'Second S3 failure was lost or relabeled')
    overlap=c['trials']['overlap'];require(overlap['state']=='passed' and len(overlap['cases'])==2,'Overlap incomplete')
    for case in overlap['cases']:
        require(case['wheels']['cache_hits']==0 and len(case['wheels']['artifacts'])==194
                and case['engine']['trusted_sha256_verified'],'Full overlap payload validation missing')
    links=c['trials']['link-screen'];require(links['state']=='passed' and len(links['cases'])==2,'Link screen incomplete')
    for case in links['cases']:
        require(case['same_filesystem'] and case['installed_file_hash_unchanged'] and case['GPU_after_cache_clean'],
                'Link/cache-clean validation missing')
        row=c['trials'][case['trial']]
        require(not row['inventory_diff'] and row['package_consistency']['exit_status']==0
                and all(s['exit_status']==0 for s in row['steps']),'Link native checks failed')
    require(links['cases'][1]['sample_nlink_before_clean']>1,'Hardlinks not realized')
    for document in (a,b,c):
        require(document['ledger']['state']=='cleaned' and document['cleanup']['provider_absence_verified'],
                'Owned cleanup missing')
    inventory=c['final_inventory']
    require(inventory is not None and not inventory['owned_remaining']
            and set(inventory['expiry_units'].values())=={'inactive'},'Final owned-resource audit missing')
    return {'state':'captured-evidence-consistent','input_sha256':hashes,
            'source_validated_median_seconds':controls['validated_ready_seconds']['median'],
            'winner_validated_median_seconds':winner['validated_ready_seconds']['median'],
            'winner_reduction_percent':winner['median_validated_reduction_vs_source_percent'],
            'controlled_identical_videos':len(videos),'second_host_packages_seconds':b['dependency_stage']['seconds'],
            'overlap_screen_reduction_percent':overlap['screening_reduction_percent'],
            'link_screen_reduction_percent':100*(1-links['cases'][1]['validated_seconds']/links['cases'][0]['validated_seconds']),
            'owned_resources_remaining':0,'conditional_branches':'See requirement coverage Markdown; no unrun alternative is labeled failed.',
            'scope':'offline captured-evidence audit, not a new native test or production readiness certification'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--output',type=Path);args=p.parse_args();value=verify(args.evidence_dir)
    if args.output:args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
