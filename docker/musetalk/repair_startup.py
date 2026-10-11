#!/usr/bin/env python3
"""Narrow, private candidate overlay for the published SyncNet boot mismatch.

Not a general image updater: the base digest/source and sole permitted source
change are fixed. Preserve all model/native hashes, notices and acceptance
limitations. No cloud credentials or private build archives enter the build.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil

import release

BASE_REVISION = 'c06624da9d7cebd6aa8f3dd6ad4a0dc8306ec6d6'
BASE_IMAGE = ('ghcr.io/ahmadafs1/musetalk-rtx3090@sha256:'
              '388ebf73de4de996ed2fff9c1d88d669a288c32cf603ce2e41c69eef26d5e6fc')
PATCH_PATH = 'scripts/vast_onstart.sh'
OLD_BLOCK = '''  if avatar_prep_requested; then
    model_files+=(
      "models/syncnet/latentsync_syncnet.pt"
      "models/dwpose/dw-ll_ucoco_384.pth"
      "models/face_detection/s3fd.pth"
    )
  fi'''
NEW_BLOCK = '''  if avatar_prep_requested; then
    # Match AVATAR_PREP_MODEL_FILES in musetalk_install_state.py. SyncNet is
    # training/quality-only and must not block serving or avatar preparation.
    model_files+=(
      "models/dwpose/dw-ll_ucoco_384.pth"
      "models/face_detection/s3fd.pth"
    )
  fi'''


def fingerprint(path):
    return {'sha256': release.sha256(path), 'size_bytes': path.stat().st_size}


def repaired_manifest(base_root, source_root, manifest, revision):
    release.require(re.fullmatch('[0-9a-f]{40}', revision or ''), 'Full repair revision required')
    release.require(manifest['source_revision'] == BASE_REVISION
                    and manifest['status'] == 'candidate'
                    and manifest.get('image_visibility') == 'private'
                    and manifest.get('promotion_eligible') is False, 'Exact private candidate base required')
    release.verify_source(base_root, manifest)
    release.verify_evidence(base_root / 'release_evidence', manifest)
    old = (base_root / PATCH_PATH).read_text()
    release.require(old.count(OLD_BLOCK) == 1, 'Expected old validation block required')
    replacement = source_root / PATCH_PATH
    release.require(replacement.read_text() == old.replace(OLD_BLOCK, NEW_BLOCK),
                    'Only the reviewed SyncNet validation change is allowed')
    for name, entry in manifest['source_files'].items():
        if name != PATCH_PATH:
            release.check_file(source_root, name, entry)
    result = copy.deepcopy(manifest)
    result['source_revision'] = revision
    result['source_files'][PATCH_PATH] = fingerprint(replacement)
    result['startup_repair'] = {
        'base_image': BASE_IMAGE, 'base_source_revision': BASE_REVISION,
        'changed_source_files': [PATCH_PATH],
        'scope': 'Serving/avatar-preparation validation excludes training-only SyncNet; no model/native changes',
        'gpu_validation': 'NOT_RUN',
    }
    return result


def prepare(base_root, source_root, out, revision):
    release.require(not out.exists(), 'Fresh repair context required')
    manifest = release.load_manifest(base_root.parent / 'release.json')
    updated = repaired_manifest(base_root, source_root, manifest, revision)
    out.mkdir(parents=True)
    shutil.copyfile(source_root / PATCH_PATH, out / 'vast_onstart.sh')
    review_name = updated['packaging_review_file']
    review = json.loads((base_root / 'release_evidence' / review_name).read_text())
    review['source_revision'] = revision
    review['startup_repair_base_revision'] = BASE_REVISION
    target = out / 'release_evidence' / review_name
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(review, indent=2) + '\n')
    updated['evidence'][review_name] = fingerprint(target)
    (out / 'release.json').write_text(json.dumps(updated, indent=2) + '\n')
    release.load_manifest(out / 'release.json')
    (out / 'Dockerfile').write_text(
        'FROM ' + BASE_IMAGE + '\n'
        'COPY vast_onstart.sh /opt/musetalk/app/scripts/vast_onstart.sh\n'
        'COPY release.json /opt/musetalk/release.json\n'
        'COPY release_evidence/ /opt/musetalk/app/release_evidence/\n'
        'LABEL org.opencontainers.image.revision="' + revision + '"\n'
        'RUN CUDA_VISIBLE_DEVICES="" /opt/musetalk/venv/bin/python '
        'docker/musetalk/release.py cpu-check --root /opt/musetalk/app '
        '--manifest /opt/musetalk/release.json\n')
    # Reports are host-only and never copied into the image.
    (out / 'reports').mkdir()


def verify_evidence(root):
    manifest = release.load_manifest(root.parent / 'release.json')
    release.verify_source(root, manifest)
    release.verify_evidence(root / 'release_evidence', manifest)
    release.verify_model_contract(root, manifest)
    release.descriptor(root, manifest)
    release.require(manifest['startup_repair']['base_image'] == BASE_IMAGE,
                    'Repair lineage mismatch')
    print('Repaired immutable source/evidence/model/native contracts: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify-evidence'])
    parser.add_argument('--base-root', type=Path, default=Path('/opt/musetalk/app'))
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--revision')
    args = parser.parse_args()
    if args.mode == 'prepare':
        release.require(args.source_root and args.out, 'Source and output required')
        prepare(args.base_root, args.source_root, args.out, args.revision)
    else:
        verify_evidence(args.base_root)


if __name__ == '__main__':
    main()
