#!/usr/bin/env python3
"""Choose a qualified creation-GPU profile and enforce its effective RAM limits."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time


def memory_facts(meminfo: Path = Path('/proc/meminfo'), cgroup: Path = Path('/sys/fs/cgroup')) -> dict:
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in meminfo.read_text().splitlines() if ':' in line}
    host = int(fields['MemTotal'].split()[0]) // 1024
    limit = None
    source = 'host-only; no cgroup limit file'
    for path in (cgroup / 'memory.max', cgroup / 'memory/memory.limit_in_bytes', cgroup / 'memory.limit_in_bytes'):
        if not path.exists():
            continue
        raw = path.read_text().strip()
        source = str(path)
        if raw == 'max':
            source += ' (unlimited)'
        else:
            limit = int(raw)
            if limit <= 0:
                raise ValueError('cgroup memory limit must be positive')
        break
    effective = min(host, limit // (1024 * 1024)) if limit is not None else host
    return {'host_ram_mib': host, 'cgroup_limit_bytes': limit, 'limit_source': source, 'effective_ram_mib': effective}


def select_profile(requirements: dict, gpu_name: str, capability: str, vram_mib: float,
                   memory: dict, allow_untested: bool = False) -> dict:
    cap, vram = float(capability), float(vram_mib)
    if not math.isfinite(cap) or not math.isfinite(vram):
        raise ValueError('GPU capability/VRAM must be finite')
    exact = next((p for p in requirements.get('qualified_gpu_profiles', []) if gpu_name in p['gpu_names']), None)
    if exact:
        if cap != float(exact['compute_capability']):
            raise ValueError(f'{gpu_name} requires compute {exact["compute_capability"]}, got {cap}')
        profile, status = exact, 'qualified'
    else:
        interval = requirements['compute_capability']
        tested = float(interval['min']) <= cap < float(interval['max_exclusive'])
        if not tested and not allow_untested:
            raise ValueError(f'compute capability {cap} is outside qualified profiles; use CHARACTER_WORKER_ALLOW_UNTESTED_GPU=1 only for a deliberate qualification')
        profile = {'id': 'existing-sm89-to-sm9x' if tested else 'experimental-unqualified',
                   'min_vram_mib': requirements['min_vram_mib'], 'min_ram_mib': requirements['min_ram_mib']}
        status = 'qualified' if tested else 'experimental'
    if vram < profile['min_vram_mib']:
        raise ValueError(f'{profile["id"]} needs total VRAM >= {profile["min_vram_mib"]} MiB, got {vram:g}')
    if memory['effective_ram_mib'] < profile['min_ram_mib']:
        raise ValueError(f'{profile["id"]} needs effective host/container RAM >= {profile["min_ram_mib"]} MiB, got {memory["effective_ram_mib"]}')
    return {'profile_id': profile['id'], 'qualification_status': status,
            'gpu_name': gpu_name, 'compute_capability': cap, 'total_vram_mib': int(vram),
            'minimum_vram_mib': profile['min_vram_mib'], 'minimum_ram_mib': profile['min_ram_mib'],
            'memory': memory, 'execution_policy': 'one active job; sequential H3 then MuseTalk asset preparation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--gpu-name', required=True)
    parser.add_argument('--compute-capability', required=True)
    parser.add_argument('--vram-mib', type=float, required=True)
    parser.add_argument('--meminfo', type=Path, default=Path('/proc/meminfo'))
    parser.add_argument('--cgroup', type=Path, default=Path('/sys/fs/cgroup'))
    parser.add_argument('--allow-untested', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        requirements = json.loads(args.lock.read_text())['host_requirements']
        record = select_profile(requirements, args.gpu_name, args.compute_capability, args.vram_mib,
                                memory_facts(args.meminfo, args.cgroup), args.allow_untested)
        record['checked_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        tmp = args.output.with_name(args.output.name + '.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(args.output)
        print(f'hardware profile {record["profile_id"]}: {record["qualification_status"]}; effective RAM {record["memory"]["effective_ram_mib"]} MiB, total VRAM {record["total_vram_mib"]} MiB')
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f'hardware preflight failed: {exc}\n')


if __name__ == '__main__':
    main()
