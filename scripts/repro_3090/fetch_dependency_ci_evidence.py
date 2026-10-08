#!/usr/bin/env python3
"""Retrieve this fixed public CI artifact without copying remote auth or writing there."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import re

REPO = 'AhmadAFS1/MuseTalk'
RUN = 37794887252
REV = 'ff8b77bd5854fe008e98be3fe1a83a65ffb2e39c'
ARTIFACT = 11561525271
SHA = 'c3f997ad755ea1badc66c3ca13ab676b0d5c5a95a04a19fe077d91d5bf60bf09'
FIXED_RUNS = {
    'ff8b77b': (RUN, REV, ARTIFACT, SHA),
    'ead7e01': (37803474546, 'ead7e0116577a7b32da341af514c8371c187cc9c', 11564001703,
                '0ceddda09be865dd10e0bc7a77c380ee7e6184dfd1a3f62d0be787860f2cae23'),
    '7a78e4a': (37830969052, '7a78e4abf09ffcc640e2979d5b08887fa70bdd4f', 11575324075,
                'c8487e775511bb7f25b931cca8e0433d1512ac4f2f711c98b36149e8e28345f7'),
    'a57f2de': (37833533181, 'a57f2de33f7d9441734b1a0868c5d8987b14a88b', 11578438305,
                'b8f6ab30f029dba4e42dc331e0eb6282ecf0e9f65d7df6f66131d04ca9ed21f8'),
}
EXPECTED = {'source-inventory.json', 'base-identity.json', 'apt-pins.json', 'build-metadata.json',
            'image-inspect.json', 'image-history.jsonl', 'pip-freeze.txt', 'dpkg-packages.txt',
            'runtime-pruning.json', 'result.json'}
LAYOUTS = {
    '7a78e4a': {**{f'musetalk-dependency-preflight/reports/{name}': name for name in EXPECTED},
                'musetalk-cpu-contracts/installer.log': 'installer.log'},
}
LAYOUTS['a57f2de'] = dict(LAYOUTS['7a78e4a'])


def require(ok, code):
    if not ok:
        raise ValueError(code)


def read_api(endpoint):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '3-way-head-talk',
                             '/usr/bin/gh', 'api', f'repos/{REPO}/{endpoint}'], capture_output=True, timeout=45)
    require(result.returncode == 0, 'readonly_remote_api_failed')
    require(0 < len(result.stdout) <= 2 * 1024**2, 'unexpected_response_size')
    return result.stdout


def entries(raw, *, layout=None, digest=None):
    layout = layout if layout is not None else {name: name for name in EXPECTED}
    require(len(set(layout.values())) == len(layout)
            and all(re.fullmatch(r'[A-Za-z0-9_.-]+', name) and name not in ('.', '..') for name in layout.values()),
            'unsafe_output_layout')
    require(hashlib.sha256(raw).hexdigest() == (SHA if digest is None else digest), 'artifact_archive_sha_mismatch')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    require(len(archive.infolist()) == len(layout)
            and {i.filename for i in archive.infolist()} == set(layout), 'unexpected_artifact_file_set')
    require(all(i.file_size <= 512 * 1024 and not i.is_dir()
                and ((i.external_attr >> 16) & 0o170000) != 0o120000 for i in archive.infolist()),
            'unsafe_artifact_member')
    return {layout[name]: archive.read(name) for name in sorted(layout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--fixed-run', choices=tuple(FIXED_RUNS), default='ff8b77b')
    args = parser.parse_args()
    global RUN, REV, ARTIFACT, SHA
    RUN, REV, ARTIFACT, SHA = FIXED_RUNS[args.fixed_run]
    require(not args.out.exists() and not args.out.is_symlink(), 'output_must_be_new')
    run = json.loads(read_api(f'actions/runs/{RUN}'))
    require(run['id'] == RUN and run['head_sha'] == REV and run['status'] == 'completed'
            and run['conclusion'] == 'success', 'expected_successful_ci_not_proven')
    artifact = json.loads(read_api(f'actions/artifacts/{ARTIFACT}'))
    require(artifact['id'] == ARTIFACT and artifact['name'] == 'musetalk-dependency-preflight'
            and artifact['digest'] == 'sha256:' + SHA and not artifact['expired']
            and artifact['workflow_run']['id'] == RUN, 'artifact_identity_mismatch')
    raw = read_api(f'actions/artifacts/{ARTIFACT}/zip')
    require(len(raw) == artifact['size_in_bytes'], 'artifact_archive_size_mismatch')
    layout = LAYOUTS.get(args.fixed_run, {name: name for name in EXPECTED})
    files = entries(raw, layout=layout)
    result = json.loads(files['result.json'])
    require(result['source_revision'] == REV and result['status'] == 'PASS'
            and not result['published'] and not result['gpu_tested'], 'artifact_result_scope_mismatch')
    args.out.mkdir(parents=True)
    for name, data in files.items():
        with (args.out / name).open('xb') as stream:
            stream.write(data)
    provenance = {'schema': 'musetalk_dependency_ci_evidence_v1', 'ci_url': run['html_url'],
                  'run_id': RUN, 'source_revision': REV, 'conclusion': run['conclusion'],
                  'artifact_id': ARTIFACT, 'artifact_archive_bytes': len(raw),
                  'artifact_archive_sha256': SHA, 'github_declared_digest_matches': True,
                  'transport': 'Read-only remote gh API through existing SSH alias; no remote file/auth copy',
                  'files': [{'name': name, 'archive_path': next(k for k, v in layout.items() if v == name),
                             'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                            for name, data in files.items()],
                  'scope': 'Dependency-only CPU build/import and size evidence; no GPU/release/publication acceptance'}
    with (args.out / 'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'VERIFIED_CI_EVIDENCE', 'files': len(files), 'archive_sha256': SHA}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'INVALID', 'error_type': type(exc).__name__,
                          'reason': str(exc) if isinstance(exc, ValueError) else 'details_suppressed'}))
        raise SystemExit(2)
