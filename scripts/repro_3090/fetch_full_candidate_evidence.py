#!/usr/bin/env python3
"""Verify fixed full-image CI artifacts through remote gh without copying auth."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from fetch_dependency_ci_evidence import read_api, require

RUN = 38085895073
REQUEST_REV = '8ed5be05af7c9d8c7e5589cd3cb6aa4947f05281'
SOURCE_REV = 'c06624da9d7cebd6aa8f3dd6ad4a0dc8306ec6d6'
REPORT_NAMES = {'private-inputs.json', 'source-inventory.json', 'build-metadata.json',
                'image-inspect.json', 'image-history.jsonl', 'installed-dependencies.json',
                'cpu-check.log', 'build-result.json'}
LAYOUT = {**{'musetalk-private-candidate/reports/' + name: name for name in REPORT_NAMES},
          'ghcr-candidate-publication/result.json': 'publication-result.json',
          'ghcr-candidate-publication/layer-scan.json': 'layer-scan.json'}


def verified_zip(raw, artifact, layout):
    require(len(raw) == artifact['size_in_bytes'] <= 2 * 1024**2
            and artifact['digest'] == 'sha256:' + hashlib.sha256(raw).hexdigest(), 'Artifact transport mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(layout) and {item.filename for item in infos} == set(layout), 'Unexpected artifact files')
        require(all(not item.is_dir() and item.file_size <= 1024**2
                    and ((item.external_attr >> 16) & 0o170000) != 0o120000 for item in infos), 'Unsafe artifact entry')
        return {output: archive.read(name) for name, output in layout.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(), 'Evidence output must be new')
    run = json.loads(read_api(f'actions/runs/{RUN}'))
    require(run['id'] == RUN and run['head_sha'] == REQUEST_REV and run['status'] == 'completed'
            and run['conclusion'] == 'success', 'Successful fixed full-image run not proved')
    jobs = json.loads(read_api(f'actions/runs/{RUN}/jobs'))['jobs']
    require({job['name'] for job in jobs} == {'candidate', 'verify-candidate'}
            and all(job['conclusion'] == 'success' for job in jobs), 'Full build/independent CPU check not proved')
    artifacts = json.loads(read_api(f'actions/runs/{RUN}/artifacts'))['artifacts']
    files, receipts = {}, []
    for name, layout in (('musetalk-ghcr-full-candidate', LAYOUT),
                          ('musetalk-ghcr-independent-candidate', {'result.json': 'independent-pull.json'})):
        matches = [a for a in artifacts if a['name'] == name]
        require(len(matches) == 1 and not matches[0]['expired'], 'Unique unexpired artifact required')
        item = matches[0]
        require(item['workflow_run']['id'] == RUN, 'Artifact run identity mismatch')
        raw = read_api(f"actions/artifacts/{item['id']}/zip")
        files.update(verified_zip(raw, item, layout))
        receipts.append({key: item[key] for key in ('id', 'name', 'size_in_bytes', 'digest')})
    publication = json.loads(files['publication-result.json'])
    independent = json.loads(files['independent-pull.json'])
    build = json.loads(files['build-result.json'])
    require(publication['schema'] == 'musetalk_ghcr_candidate_publication_v1'
            and publication['source_revision'] == SOURCE_REV and publication['published'] is True
            and publication['serving_image'] is True and publication['promotion_eligible'] is False
            and publication['anonymous_pull'] == 'DENIED'
            and publication['package']['visibility'] == 'private', 'Publication identity/scope mismatch')
    image = publication['image']
    require(re.fullmatch(r'ghcr\.io/ahmadafs1/musetalk-rtx3090@sha256:[0-9a-f]{64}', image)
            and independent['image'] == image and independent['status'] == 'PASS'
            and independent['private_visibility'] == 'VERIFIED' and independent['anonymous_pull'] == 'DENIED'
            and build['source_revision'] == SOURCE_REV and build['cpu_build_check'] == 'PASS', 'Independent digest/build mismatch')
    proof = {'schema': 'musetalk_full_candidate_verification_v1', 'ci_url': run['html_url'], 'run_id': RUN,
             'request_revision': REQUEST_REV, 'source_revision': SOURCE_REV, 'workflow_conclusion': 'success',
             'image': image, 'compressed_layer_bytes': publication['compressed_layer_bytes'],
             'published': True, 'serving_image': True, 'independent_pull': 'PASS', 'offline_cpu_check': 'PASS',
             'private_visibility': 'VERIFIED', 'anonymous_pull': 'DENIED', 'promotion_eligible': False,
             'gpu_tested': False, 'startup_measured': False, 'artifacts': receipts,
             'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)} for name, data in files.items()}}
    args.out.mkdir(parents=True)
    for name, data in files.items():
        with (args.out / name).open('xb') as output:
            output.write(data)
    with (args.out / 'verified-full-image.json').open('x') as output:
        json.dump(proof, output, indent=2)
        output.write('\n')
    print(json.dumps({'status': 'VERIFIED_FULL_PRIVATE_IMAGE', 'image': image,
                      'compressed_layer_bytes': proof['compressed_layer_bytes'], 'gpu_tested': False}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'NOT_VERIFIED', 'error_type': type(exc).__name__}))
        raise SystemExit(2)
