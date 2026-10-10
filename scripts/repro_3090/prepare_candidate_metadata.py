#!/usr/bin/env python3
"""Create pinned metadata for the approved private GHCR build, not a release pass.

Uses the preserved native candidate, exact installed pins and real failed quality/
FPS records. No credentials/URLs, model bytes, CI dispatch or rental in output.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

import assemble_candidate_inventory as assembly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    assembly.require(not out.exists() and not out.is_symlink(), 'Metadata output must be new')
    inventory = assembly.assemble(root)
    sys.path.insert(0, str(root / 'docker/musetalk'))
    import release
    source_revision = inventory['source_revision']
    bundle_sha = inventory['archives']['native.tar.gz']['sha256']
    model_map = json.loads((root / assembly.DOSSIER / 'public-model-file-license-map.json').read_text())
    weights = {item['path']: {**{key: item[key] for key in ('sha256', 'size_bytes')},
                              'license_id': item['license_basis'], 'public_redistribution': True}
               for item in model_map['model_files']}
    native = {name: {**entry, 'license_id': 'MIT-derived; retained source notices and provenance limitations',
                     'public_redistribution': False, 'private_delivery_authorized': True}
              for name, entry in inventory['native_files'].items()}
    quality_path = root / 'docs/fps_comparisons/rtx3090_r5_20261008/quality/native_v1_quality_decision.json'
    quality = json.loads(quality_path.read_text())
    assembly.require(quality['candidate'] == 'native_sm86_r5_v1' and quality['decision'] == 'rejected',
                     'Preserve the actual strict native quality verdict')
    quality['bundle_sha256'] = bundle_sha
    quality['packaging_binding'] = {'original_record': assembly.fingerprint(quality_path),
                                   'limitation': 'Historical strict verdict, not new image/GPU acceptance'}
    aggregate_path = root / 'docs/fps_comparisons/rtx3090_r5_20261008/native/a6_latent_ab_20261009/a6-latents-ab-summary.json'
    original_aggregate = json.loads(aggregate_path.read_text())
    unet_manifest = 'models/tensorrt_unet_stagewise_sm86_r5_v1/bs16/manifest.json'
    assembly.require(original_aggregate['native_manifest_sha256'] == native[unet_manifest]['sha256'],
                     'Historical aggregate native manifest does not match this archive')
    aggregate = {'gpu': 'NVIDIA GeForce RTX 3090', 'bundle_sha256': bundle_sha,
                 'status': original_aggregate['status'],
                 'limitation': 'Historical six-avatar sustained tests missed 400 FPS. Not a fresh image '
                               'measurement, live encoding/RTP result, or production/default acceptance.',
                 'original_record': assembly.fingerprint(aggregate_path),
                 'arms': {name: {key: arm[key] for key in ('status', 'windows', 'weighted_aggregate_fps')}
                          for name, arm in original_aggregate['arms'].items()}}
    findings = {'schema': 'musetalk_private_packaging_findings_v1', 'scope': 'private_deployment',
                'source_revision': source_revision, 'bundle_sha256': bundle_sha,
                'public_redistribution_reviewed': False, 'blanket_use_rights_clearance': False,
                'notice_policy': 'preserve_bundled_and_model_notices',
                'remaining_findings': inventory['remaining_review_findings'],
                'native_derivation': {'status': inventory['native_input_derivation_status'],
                    'intended_unet_input': weights['models/musetalkV15/unet.pth'],
                    'intended_taesd_input': weights['models/taesd/diffusion_pytorch_model.safetensors'],
                    'limitation': 'Preserved build metadata identifies source paths/toolchain; complete '
                                  'historical input-hash trace is not present. No fabricated provenance pass.'},
                'authorization_basis': 'Operator explicitly selected private GHCR and requested full image '
                                       'build plus a fresh RTX 3090 startup/app-boot test; no public distribution.',
                'notice_handling': 'Model notices copied by exact hash; pip distribution notices and OS '
                                   'copyright files retained. Final build records its actual dependency inventory.',
                'scope_limit': 'No public license clearance, customer image delivery or production promotion.'}
    out.mkdir(parents=True)
    notices = {}
    for name, expected in inventory['notices'].items():
        destination = 'licenses/' + Path(name).name
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / assembly.DOSSIER / name, target)
        assembly.require(assembly.fingerprint(target) == expected, 'Copied model notice mismatch')
        notices[destination] = expected
    # Preserve existing version-specific runtime references as supplemental
    # notices, not a replacement for actual wheel/base/dpkg notice bytes.
    for name in ('nvidia_container_license.txt', 'tensorrt_10_3_libs_license.txt',
                 'tensorrt_10_3_bindings_license.txt', 'tensorrt_10_3_oss_license.txt',
                 'tensorrt_10_3_oss_notice.txt', 'torch_tensorrt_2_5_license.txt',
                 'pyav_16_1_license.txt', 'imageio_ffmpeg_0_6_license.txt', 'pytorch_2_5_1_license.txt',
                 'mmcv_2_1_license.txt'):
        target = out / 'licenses' / name
        shutil.copyfile(root / assembly.DOSSIER / 'supplemental' / name, target)
        notices['licenses/' + name] = assembly.fingerprint(target)
    evidence = {}
    for name, value in (('quality.json', quality), ('aggregate.json', aggregate), ('packaging.json', findings)):
        target = out / 'evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + '\n')
        evidence['evidence/' + name] = assembly.fingerprint(target)
    manifest = {'schema': release.SCHEMA, 'status': 'candidate', 'promotion_eligible': False,
                'candidate_reason': 'Private startup/app boot validation of the preserved native 3090 '
                                    'configuration; original quality/FPS gates not passed. No production rollout.',
                'image_visibility': 'private', 'redistribution_reviewed': False,
                'redistribution_scope': 'baked_model_files_only', 'private_model_usage_rights': 'unresolved',
                **{key: inventory[key] for key in ('source_revision', 'source_files', 'cuda_base',
                   'cuda_runtime_base', 'platform', 'matrix', 'kokoro', 'avatar_prep', 'bundle_name',
                   'archives', 'apt_packages', 'external_model_files')},
                'vp8_encoder': 'native', 'model_files': {**weights, **native},
                'bundle_manifest_sha256': inventory['bundle_manifest']['sha256'],
                'notices': notices, 'evidence': evidence, 'quality_decision_file': 'evidence/quality.json',
                'aggregate_acceptance_file': 'evidence/aggregate.json',
                'packaging_review_file': 'evidence/packaging.json'}
    (out / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifest = release.load_manifest(out / 'release.json')
    release.verify_evidence(out, manifest)
    release.verify_source(root, manifest)
    release.verify_model_contract(root, manifest)
    release.descriptor(root, manifest)
    for name, entry in notices.items():
        release.scan_text(release.check_file(out, name, entry))
    archive_path = out.parent / (out.name + '.tar.gz')
    assembly.require(not archive_path.exists(), 'Metadata archive must be new')
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name in ['release.json', *sorted(notices), *sorted(evidence)]:
            archive.add(out / name, arcname=name, recursive=False)
    print(json.dumps({'status': 'PINNED_PRIVATE_CANDIDATE_METADATA', 'source_revision': source_revision,
                      'metadata': assembly.fingerprint(archive_path), 'model_files': len(manifest['model_files']),
                      'notices': len(notices), 'public_review_pass': False, 'promotion_eligible': False}))


if __name__ == '__main__':
    main()
