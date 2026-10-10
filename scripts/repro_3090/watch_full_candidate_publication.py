#!/usr/bin/env python3
"""Bounded, read-only CI wait and verified receipt collection; never rents a GPU."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from fetch_dependency_ci_evidence import read_api, require

STEPS = {
    'Validate non-secret pinned request and CPU contracts': 'CPU_CONTRACTS',
    "Verify streaming audit against this runner's Docker export format": 'STREAM_EXPORT_SMOKE',
    'Verify existing package is private before building': 'PRIVATE_PACKAGE_CHECK',
    'Assemble private pinned inputs, build and CPU-check without cloud credentials': 'FULL_BUILD_CPU_CHECK',
    'Audit every layer and privately publish a nonpromotable full candidate': 'LAYER_AUDIT_PRIVATE_PUBLISH',
    'Independent private exact-digest pull and offline CPU check': 'INDEPENDENT_PULL_CPU_CHECK',
    'Preserve small non-secret receipts only': 'ARTIFACT_UPLOAD',
    'Remove ephemeral registry login': 'REGISTRY_LOGOUT',
}


def snapshot(run, jobs, run_id, revision):
    require(run.get('id') == run_id and run.get('head_sha') == revision
            and run.get('name') == 'MuseTalk private full candidate', 'Unexpected CI identity')
    require(run.get('status') in {'queued', 'in_progress', 'completed'}, 'Unknown CI status')
    require(all(job.get('name') in {'candidate', 'verify-candidate'} for job in jobs), 'Unexpected CI job')
    phases = []
    for job in jobs:
        for step in job.get('steps', []):
            if step.get('status') == 'in_progress':
                phases.append(STEPS.get(step.get('name'), 'RUNNER_SETUP_OR_CHECKOUT'))
    return {'status': run['status'], 'conclusion': run.get('conclusion'), 'phases': sorted(set(phases))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--request-revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    require(args.run_id > 0 and re.fullmatch('[0-9a-f]{40}', args.request_revision)
            and 60 <= args.timeout_seconds <= 9000, 'Invalid bounded CI request')
    require(not args.out.exists() and not args.out.is_symlink(), 'Publication wait output must be new')
    args.out.mkdir(parents=True)
    journal = args.out / 'progress.md'
    started, previous = time.monotonic(), None

    def record(phase, **values):
        result = {'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'phase': phase,
                  'run_id': args.run_id, **values}
        with journal.open('a') as output:
            output.write('\n## ' + result['at_utc'] + ' — ' + phase + '\n\n```json\n'
                         + json.dumps(result, indent=2) + '\n```\n')
        print(json.dumps(result), flush=True)

    record('WAITING_FOR_PUBLICATION', request_revision=args.request_revision,
           source_revision='c06624da9d7cebd6aa8f3dd6ad4a0dc8306ec6d6', gpu_rental=False)
    try:
        while time.monotonic() - started < args.timeout_seconds:
            run = json.loads(read_api(f'actions/runs/{args.run_id}'))
            jobs = json.loads(read_api(f'actions/runs/{args.run_id}/jobs'))['jobs']
            state = snapshot(run, jobs, args.run_id, args.request_revision)
            if state != previous:
                record('CI_PROGRESS', **state)
                previous = state
            else:
                print(json.dumps({'phase': 'CI_HEARTBEAT', 'phases': state['phases'],
                                  'elapsed_minutes': round((time.monotonic() - started) / 60, 1)}), flush=True)
            if state['status'] == 'completed':
                require(state['conclusion'] == 'success', 'CI publication failed; no GPU rented')
                collector = Path(__file__).with_name('fetch_full_candidate_evidence.py')
                result = subprocess.run([sys.executable, '-B', str(collector), '--run-id', str(args.run_id),
                    '--request-revision', args.request_revision, '--out', str(args.out / 'evidence')],
                    capture_output=True, timeout=300)
                require(result.returncode == 0, 'Publication evidence rejected; details suppressed')
                proof = json.loads((args.out / 'evidence/verified-full-image.json').read_text())
                record('VERIFIED_PRIVATE_FULL_IMAGE', image=proof['image'],
                       compressed_layer_bytes=proof['compressed_layer_bytes'],
                       independent_pull=proof['independent_pull'], offline_cpu_check=proof['offline_cpu_check'],
                       gpu_tested=False, startup_measured=False, production_enabled=False)
                return
            time.sleep(45)
        raise TimeoutError('Bounded CI wait expired')
    except Exception as error:
        record('STOPPED_REQUIRES_INSPECTION', error_type=type(error).__name__,
               gpu_rental=False, publication_verified=False, details_suppressed=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
