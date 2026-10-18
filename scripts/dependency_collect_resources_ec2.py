#!/usr/bin/env python3
"""Preserve a compressed read-only resource prefix after timed tests finish."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import dependency_optimization_ec2 as c


def main():
    b=c.library();b.require(os.getuid()==0,'Root-owned EC2 evidence only')
    payload='''import base64,gzip,hashlib,json,pathlib,os
root=pathlib.Path(ROOT);receipt=root/'warm-start-a1/receipt.json'
assert receipt.exists(),'Wait until warm startup is terminal before resource transfer'
p=root/'resources.jsonl';assert p.is_file() and not p.is_symlink() and p.stat().st_size<=512*1024**2
raw=p.read_bytes()
if not raw.endswith(b'\\n'):raw=raw.rsplit(b'\\n',1)[0]+b'\\n'
rows=[json.loads(line) for line in raw.splitlines()];samples=[r for r in rows if r.get('kind')=='resource_sample']
memory=[int(r['cgroup_memory.current']) for r in samples if str(r.get('cgroup_memory.current','')).strip().isdigit()]
rss=[sum(p['rss_pages'] for p in r.get('processes',[]))*os.sysconf('SC_PAGE_SIZE') for r in samples]
compressed=gzip.compress(raw,compresslevel=1,mtime=0)
result={'raw_bytes':len(raw),'compressed_bytes':len(compressed),'raw_sha256':hashlib.sha256(raw).hexdigest(),
 'compressed_sha256':hashlib.sha256(compressed).hexdigest(),'samples':len(samples),'sampling_seconds':2,
 'first_sample_utc':samples[0]['utc'] if samples else None,'last_sample_utc':samples[-1]['utc'] if samples else None,
 'sampled_max_cgroup_memory_bytes':max(memory) if memory else None,
 'sampled_max_selected_processes_total_RSS_bytes':max(rss) if rss else None,
 'limits':['sampling started after baseline installation began','sampled maxima are not instantaneous peaks','kernel CPU/network/disk counters can be host-wide','filesystem free-space peak was not continuously sampled'],
 'gzip_base64':base64.b64encode(compressed).decode()}
print(json.dumps(result))
'''.replace('ROOT',repr(c.WORKER_ROOT))
    value=json.loads(c.worker(b,payload));data=base64.b64decode(value.pop('gzip_base64'))
    b.require(hashlib.sha256(data).hexdigest()==value['compressed_sha256'],'Resource transfer hash failed')
    raw=gzip.decompress(data);b.require(hashlib.sha256(raw).hexdigest()==value['raw_sha256'],'Resource payload hash failed')
    import re
    b.require(not re.search(rb'(?:AKIA|ASIA)[A-Z0-9]{16}|X-Amz-(?:Credential|Signature|Security-Token)|gh[pousr]_[A-Za-z0-9]{20,}',raw,re.I),
              'Resource evidence credential pattern; retain privately')
    path=c.HERE/'dependency-resources.jsonl.gz';path.write_bytes(data);os.chmod(path,0o600)
    os.chown(path,c.HERE.stat().st_uid,c.HERE.stat().st_gid)
    b.save('resources-preservation.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
