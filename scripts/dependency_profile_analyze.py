#!/usr/bin/env python3
"""Summarize copied profiling receipts without importing pip or CUDA packages."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
from pathlib import Path
import re
import statistics


def union_seconds(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start>end:
            total += max(0,stop-start)
        else:
            total += max(0,stop-end)
        end=max(end or stop,stop)
    return total


def summarize(root):
    events=[]
    for path in root.glob('*pip-*.jsonl'):
        for line in path.read_text(errors='replace').splitlines():
            try:
                record=json.loads(line)
                if isinstance(record,dict):
                    events.append(record)
            except ValueError:
                pass
    runs=defaultdict(list)
    for record in events:
        runs[record.get('run_id')].append(record)
    result=[]
    for run_id, records in runs.items():
        records.sort(key=lambda r:r['mono'])
        starts={r['span_id']:r for r in records if 'span_id' in r and r['kind'].endswith('_start')}
        ends={r['span_id']:r for r in records if 'span_id' in r and r['kind'].endswith('_end')}
        pip_start=next((r for r in records if r['kind']=='pip_start'),{})
        pip_end=next((r for r in records if r['kind']=='pip_end'),{})
        finished=[r for r in records if r['kind'].endswith('_end') and r['kind']!='pip_end']
        active=[r for sid,r in starts.items() if sid not in ends]
        by_kind=defaultdict(list)
        for record in finished:
            by_kind[record['kind']].append(record)
        network=[]
        for sid,end in ends.items():
            if end['kind'] in ('request_end','body_end','download_end') and sid in starts:
                network.append((starts[sid]['mono'],end['mono']))
        host=defaultdict(lambda: {'request_count':0,'request_seconds':0.,'body_read_seconds':0.,
                                  'body_bytes':0,'uncached_body_bytes':0,'statuses':Counter()})
        for record in finished:
            if record['kind'] not in ('request_end','body_end'):
                continue
            item=host[record.get('host','unknown')]
            if record['kind']=='request_end':
                item['request_count']+=1
                item['request_seconds']+=record['seconds']
                item['statuses'][str(record.get('status'))]+=1
            else:
                item['body_bytes']+=record.get('bytes',0)
                item['uncached_body_bytes']+=record.get('bytes',0) if not record.get('from_cache') else 0
                item['body_read_seconds']+=record.get('read_seconds',0)
        def top(kind,n=15):
            keys=('package','host','file','seconds','read_seconds','bytes','status','from_cache','outcome','utc')
            return [{k:r[k] for k in keys if k in r} for r in sorted(by_kind.get(kind,[]),key=lambda r:r['seconds'],reverse=True)[:n]]
        def stats(kind):
            values=[r['seconds'] for r in by_kind.get(kind,[])]
            return {'count':len(values),'sum_seconds':sum(values),'median_seconds':statistics.median(values) if values else None,
                    'max_seconds':max(values) if values else None}
        packages=defaultdict(lambda:defaultdict(float))
        for record in finished:
            if record.get('package'):
                packages[record['package']][record['kind']]+=record['seconds']
        downloads=[{k:r[k] for k in ('host','file','seconds','outcome','utc') if k in r}
                   for r in by_kind.get('download_end',[])]
        result.append({'run_id':run_id,'group':pip_start.get('group'),'started_utc':pip_start.get('utc'),
            'pid':pip_start.get('pid'),'start_mono':pip_start.get('mono'),'end_mono':pip_end.get('mono'),
            'pip_version':pip_start.get('pip_version'),'status':'finished' if pip_end else 'running',
            'seconds':pip_end.get('seconds'),'exit_status':pip_end.get('exit_status'),
            'last_event_utc':records[-1]['utc'],'unavailable_hooks':[r for r in records if r['kind']=='hook' and not r['available']],
            'completed_span_stats':{kind:stats(kind) for kind in sorted(by_kind)},
            'observed_network_union_seconds':union_seconds(network),'network_by_host':dict(host),
            'slow_requests':top('request_end'),'slow_bodies':top('body_end'),
            'slow_downloads':top('download_end'),
            'slow_preparation':top('prepare_end'),'slow_wheel_install':top('wheel_install_end'),
            'slow_builds':top('wheel_build_end'),'slow_source_metadata':top('sdist_metadata_end'),
            'all_package_spans_seconds':dict(packages),'all_download_spans':downloads,
            'active_spans':[{k:r[k] for k in ('kind','package','host','file','utc','mono') if k in r} for r in active]})
    result.sort(key=lambda r:r['started_utc'] or '')
    bootstrap=next(iter(root.glob('*bootstrap.log')),None)
    steps=[]; tail=[]
    if bootstrap:
        lines=bootstrap.read_text(errors='replace').splitlines()
        for line in lines:
            if any(s in line for s in ('START:','DONE:','FAILED:','checkout_start','checkout_ready','[bootstrap] start','[bootstrap] failed','phase finished in','health check')):
                steps.append(line[:1800])
        tail=lines[-12:]
    samples=[]
    for path in root.glob('*resources.jsonl'):
        for line in path.read_text(errors='replace').splitlines():
            try:
                row=json.loads(line)
                if row.get('kind')=='resource_sample':
                    samples.append(row)
            except ValueError:
                pass
    resources={'samples':len(samples),'first_utc':samples[0]['utc'] if samples else None,
               'last_utc':samples[-1]['utc'] if samples else None}
    def net_counters(sample):
        rx=tx=0
        for line in sample.get('network','').splitlines():
            if ':' not in line:
                continue
            interface,rest=line.split(':',1)
            if interface.strip()=='lo':
                continue
            fields=rest.split()
            if len(fields)>=9:
                rx+=int(fields[0]);tx+=int(fields[8])
        return rx,tx
    for run in result:
        start,end=run['start_mono'],run['end_mono']
        selected=[s for s in samples if start is not None and s['mono']>=start and (end is None or s['mono']<=end)]
        own=[p for s in selected for p in s.get('processes',[]) if p['pid']==run['pid']]
        if selected:
            first,last=net_counters(selected[0]),net_counters(selected[-1])
            duration=selected[-1]['mono']-selected[0]['mono']
            run['resource_samples']={'count':len(selected),'interval_seconds':duration,
                'all_container_received_bytes':max(0,last[0]-first[0]),
                'all_container_sent_bytes':max(0,last[1]-first[1]),
                'all_container_mean_received_mbps':max(0,last[0]-first[0])*8/1e6/duration if duration else None,
                'main_pip_cpu_ticks_observed':(max(p['utime_ticks']+p['stime_ticks'] for p in own)-min(p['utime_ticks']+p['stime_ticks'] for p in own)) if own else None,
                'main_pip_peak_rss_pages_observed':max((p['rss_pages'] for p in own),default=None),
                'caveat':'Whole-container traffic includes observer SSH; CPU excludes child build processes. First/last samples are inside phase, up to ~2s from boundaries.'}
    diagnostics=[]
    for path in root.glob('*.debug.log'):
        matched=Counter()
        for line in path.read_text(errors='replace').splitlines():
            for name,pattern in [('retry',r'Retrying \('),('timeout',r'(?:ReadTimeout|ConnectTimeout|timed out)'),
                                 ('backtracking',r'(?:looking at multiple versions|backtracking)'),
                                 ('resolver_conflict',r'ResolutionImpossible')]:
                if re.search(pattern,line,re.I):matched[name]+=1
        diagnostics.append({'file':path.name,'bytes':path.stat().st_size,'signals':dict(matched)})
    return {'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pip_runs':result,
            'startup_steps':steps,'bootstrap_tail':tail,'resource_collection':resources,
            'debug_log_diagnostics':diagnostics,
            'interpretation':'Nested durations overlap. Network union includes HTTP request/body and pip downloader spans (including local writes), but excludes unhooked subprocess transfers. Cache counters describe response consumption, not wire-level billing.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root),indent=2))
