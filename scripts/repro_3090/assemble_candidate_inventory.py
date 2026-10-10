#!/usr/bin/env python3
"""Inventory actual private-candidate inputs without granting publication approval.

No network, credentials, release.json, CI dispatch, extraction or acceptance
flags. Output is an engineering assembly record, not a deployable manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

RELEASE = Path('docs/fps_comparisons/rtx3090_r5_20261008/release')
DOSSIER = Path('docker/musetalk/notice_dossier_20261008')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
        size += len(block)
    return {'sha256': digest.hexdigest(), 'size_bytes': size}


def fingerprint(path):
    require(path.is_file() and not path.is_symlink(), 'Input must be a regular non-symlink file')
    with path.open('rb') as stream:
        return digest_stream(stream)


def native_inventory(path, expected):
    records = {}
    sidecars = {'.musetalk_trt_artifact_manifest.json', '.musetalk_trt_artifact_SHA256SUMS'}
    manifest = None
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            require(member.isreg() and name not in records and not name.startswith('/')
                    and '..' not in Path(name).parts, 'Unsafe/duplicate native member')
            require(name in sidecars or name.startswith('models/'), 'Unexpected native member')
            stream = archive.extractfile(member)
            if name == '.musetalk_trt_artifact_manifest.json':
                require(member.size <= 1024**2, 'Oversized native manifest')
                data = stream.read()
                manifest = json.loads(data)
                record = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
            else:
                record = digest_stream(stream)
            require(record['size_bytes'] == member.size, 'Truncated native member')
            records[name] = record
    require(manifest is not None and manifest.get('schema') == 1, 'Missing native manifest')
    declared = {item['path']: {'sha256': item['sha256'], 'size_bytes': item['size']}
                for item in manifest['files']}
    require(len(declared) == len(manifest['files']) == expected and set(records) == set(declared) | sidecars,
            'Native payload set differs from preserved manifest')
    require(all(records[name] == record for name, record in declared.items()), 'Native payload hash mismatch')
    return declared, records['.musetalk_trt_artifact_manifest.json']


def assemble(root):
    sys.path.insert(0, str(root / 'docker/musetalk'))
    import context
    import release
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    source = context.inventory(root)
    for name, expected in source.items():
        data = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=root)
        require({'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)} == expected,
                'Allowlisted runtime source differs from committed source')
    model_map_path = root / DOSSIER / 'public-model-file-license-map.json'
    model_map = json.loads(model_map_path.read_text())
    weights_receipt_path = root / RELEASE / 'docker_model_archive_receipt_0551.json'
    weights_receipt = json.loads(weights_receipt_path.read_text())
    weights = root / RELEASE / 'docker-model-stage-0515.oaOuVq/weights.tar.gz'
    weights_record = fingerprint(weights)
    require(weights_record == {'sha256': weights_receipt['archive_sha256'],
                               'size_bytes': weights_receipt['archive_bytes']}, 'Weight archive mismatch')
    native_receipt_path = root / RELEASE / 'native_v1_diagnostic_persistence.json'
    native_receipt = json.loads(native_receipt_path.read_text())
    native = root / RELEASE / native_receipt['archive']['filename']
    native_record = fingerprint(native)
    require(native_record == {'sha256': native_receipt['archive']['sha256'],
                              'size_bytes': native_receipt['archive']['bytes']}, 'Native archive mismatch')
    native_files, sidecar = native_inventory(native, native_receipt['archive']['payload_files'])
    notices = {}
    for item in model_map['model_files']:
        require(item['disposition'] == 'ELIGIBLE_WITH_LISTED_NOTICES', 'Unreviewed public model')
        for name, expected in item['retained_notices'].items():
            actual = fingerprint(root / DOSSIER / name)
            require(actual == {k: expected[k] for k in ('sha256', 'size_bytes')}, 'Model notice mismatch')
            notices[name] = actual
    private = json.loads((root / RELEASE / 'private_model_parallel_read_verify_1442.json').read_text())
    external = {name: entry for name, entry in private['external_model_files'].items()
                if name != 'models/syncnet/latentsync_syncnet.pt'}
    require(len(external) == 4 and all('version_id' not in entry['source'] for entry in external.values()),
            'Expected four ordinary-GetObject runtime model inputs')
    ci = root / RELEASE / 'dependency_ci_e234988'
    apt = json.loads((ci / 'apt-pins.json').read_text())
    require(release.REQUIRED_APT <= {item.split('=')[0] for item in apt}, 'Missing OS pins')
    return {
        'schema': 'musetalk_candidate_assembly_inventory_v1', 'status': 'ASSEMBLED_NOT_AUTHORIZED',
        'source_revision': revision, 'source_files': source, 'source_matches_commit': True,
        'worktree_clean': not bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)),
        'cuda_base': release.CUDA_DEVEL_BASE, 'cuda_runtime_base': release.CUDA_RUNTIME_BASE,
        'platform': 'linux/amd64', 'matrix': 'cu121', 'kokoro': False, 'avatar_prep': True,
        'bundle_name': release.NATIVE_V1_CANDIDATE, 'archives': {
            'weights.tar.gz': weights_record, 'native.tar.gz': native_record},
        'weight_files': {item['path']: {k: item[k] for k in ('sha256', 'size_bytes', 'license_basis')}
                         for item in model_map['model_files']},
        'weight_archive_members_rehashed': False,
        'weight_members_prior_restore_receipt': fingerprint(weights_receipt_path),
        'native_files': native_files, 'native_members_rehashed': True,
        'bundle_manifest': sidecar, 'notices': notices, 'external_model_files': external,
        'apt_packages': apt, 'dependency_inventory': fingerprint(ci / 'installed-dependencies.json'),
        'dependency_ci_provenance': fingerprint(ci / 'provenance.json'),
        'native_input_derivation_status': 'PRESERVED_BUILD_METADATA_WITHOUT_COMPLETE_INPUT_HASH_TRACE',
        'redistribution_reviewed': False, 'promotion_eligible': False, 'published': False,
        'gpu_tested': False, 'cold_start_measured': False,
        'remaining_review_findings': model_map['remaining_release_gates'],
        'limitation': 'Not release.json or build authorization. No public/private release clearance, '
                      'quality/FPS promotion, secure CI metadata upload or GPU acceptance is granted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(), 'Output must be new')
    result = assemble(args.root.resolve())
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'source_revision': result['source_revision'],
                      'weight_files': len(result['weight_files']), 'native_files': len(result['native_files']),
                      'notices': len(result['notices']), 'external_models': len(result['external_model_files']),
                      'publication_authorized': False}))


if __name__ == '__main__':
    main()
