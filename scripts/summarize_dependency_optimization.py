#!/usr/bin/env python3
"""Recompute reported timings from sanitized receipts; no network or installs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(document):
    trials=document['trials_and_builds'];rows=[];groups={}
    for name,value in trials.items():
        if not name.startswith('delivery-') or value.get('state')!='delivery-trial-passed':continue
        row={key:value[key] for key in ('trial','method','workers','packages_ready_seconds','validated_ready_seconds')}
        row['acquisition_seconds']=value['acquisition']['seconds']
        row['scope']='continuous full-stack dependency delivery, not full Vast boot'
        row['observer_note']='12.6 MB evidence collection occurred after acquisition, during install' if name=='delivery-a2-uv-prefetch-4-3' else None
        rows.append(row);groups.setdefault((row['method'],row['workers']),[]).append(row)
    summary=[];clean_summary=[]
    for (method,workers),values in groups.items():
        result={'method':method,'workers':workers,'n':len(values),'trials':[v['trial'] for v in values]}
        for metric in ('packages_ready_seconds','validated_ready_seconds','acquisition_seconds'):
            nums=[v[metric] for v in values]
            result[metric]={'individual':nums,'median':statistics.median(nums),'min':min(nums),'max':max(nums)}
        summary.append(result)
        clean=[v for v in values if not v['observer_note']]
        if clean:
            nums=[v['validated_ready_seconds'] for v in clean]
            clean_summary.append({'method':method,'workers':workers,'n':len(clean),
                'trials':[v['trial'] for v in clean],'validated_ready_seconds':{
                    'individual':nums,'median':statistics.median(nums),'min':min(nums),'max':max(nums)}})
    control=next((r for r in summary if r['method']=='source-pip'),None)
    for value in summary:
        value['median_validated_reduction_vs_source_percent']=(100*(1-value['validated_ready_seconds']['median']/control['validated_ready_seconds']['median'])) if control else None
    for value in clean_summary:
        value['median_validated_reduction_vs_source_percent']=(100*(1-value['validated_ready_seconds']['median']/control['validated_ready_seconds']['median'])) if control else None
    return {'scope':'same-host measurement summary; no fleet p95 or rollout certification','goal_complete':False,
        'source_prepared_utc':document['prepared_utc'],'trials':rows,'summary':summary,'clean_summary':clean_summary,
        'coverage_note':'This first-host input alone cannot decide whole-goal completion. See the separate canary/final-screen datasets and requirement coverage audit.',
        'not_proven_by_this_input':['different-host confirmation','fresh-worker canary','final cleanup verification']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();raw=args.input.read_bytes();out=summarize(json.loads(raw));out['input_sha256']=hashlib.sha256(raw).hexdigest()
    rendered=json.dumps(out,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(rendered)
    print(json.dumps({'summary':out['summary'],'output':str(args.output) if args.output else None}))


if __name__=='__main__':main()
