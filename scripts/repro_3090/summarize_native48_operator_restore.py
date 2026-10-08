"""Reverify the three fixed clean restores and emit private evidence metadata."""
import hashlib
import json
from pathlib import Path

from persist_native48_media_operator import require, sha_file, NAMES
import privacy_attestation as privacy

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release'
FOLDERS = ('native48_fresh_cpu_restore_1654_part1', 'native48_fresh_cpu_restore_1703_part2',
           'native48_fresh_cpu_restore_1710_part3')
OUT = RELEASE / 'native48_private_persistence_and_restore_1712.json'


def main():
    source = RELEASE / 'native48_operator_persistence_1645.json'
    data = json.loads(source.read_text())
    require(data['status'] == 'PASS_ALL3_PRIVATE_ARCHIVE_FRESH_GET_SHA' and data['release_ready'] is False,
            'all3 verified private archives required')
    rows = data['objects']
    require(len(rows) == 3 and {row['filename'] for row in rows} == NAMES, 'exact archive set required')
    pack = RELEASE / 'native_v1_production48_media_1616/preservation_receipt.json'
    require(sha_file(pack) == '57d446dfca7d1386100430cced8da1629103c9eeff18c424dcac3a3361f6e630',
            'fixed pack receipt SHA mismatch')
    expected = {row['filename']: row for row in json.loads(pack.read_text())['archives']}
    require(data['bucket'] == privacy.BUCKET and data['region'] == privacy.REGION
            and data['expected_owner'] == privacy.OWNER, 'fixed private destination mismatch')
    all_files, all_poses, restored = {}, set(), []
    for number, (row, folder) in enumerate(zip(rows, FOLDERS), 1):
        require(row['filename'] == f'native-v1-production48-review-part{number}of3.tar.gz'
                and row['fresh_get_verified'] is True, 'archive order/content proof mismatch')
        pin = expected[row['filename']]
        require((row['bytes'], row['sha256']) == (pin['bytes'], pin['sha256'])
                and isinstance(row.get('version_id'), str) and row['version_id'], 'archive pins/version missing')
        base = RELEASE / folder
        side = RELEASE / (folder + '_sidecars')
        manifest_path = side / '.musetalk_trt_artifact_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        stamp = json.loads((side / '.musetalk_trt_artifact_restored.json').read_text())
        require(stamp['archive_sha256'] == row['sha256'] and stamp['mode'] == 'restored', 'fresh restore SHA mismatch')
        files = manifest['files']
        require(len(files) == (101 if number == 1 else 100), 'payload count mismatch')
        poses = set()
        for entry in files:
            rel = Path(entry['path'])
            require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe payload path')
            path = base / rel
            require(path.is_relative_to(base) and not path.is_symlink() and path.is_file()
                    and path.stat().st_size == entry['size'] and sha_file(path) == entry['sha256'], 'restored payload SHA/size mismatch')
            if entry['path'] in all_files:
                require(all_files[entry['path']] == (entry['size'], entry['sha256']), 'repeated global input differs')
            all_files[entry['path']] = (entry['size'], entry['sha256'])
            if rel.name == 'review_with_audio.mkv':
                poses.add(rel.parent.name)
        require(len(poses) == 16 and not poses & all_poses, 'pose overlap or missing partition')
        all_poses |= poses
        restored.append({'filename': row['filename'], 'bytes': row['bytes'], 'sha256': row['sha256'],
                         'bucket': data['bucket'], 'key': row['key'], 'version_id': row['version_id'],
                         'fresh_get_verified': True, 'restore_stamp_at_utc': stamp['restored_at'],
                         'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                         'payload_files_reverified': len(files), 'pose_ids': sorted(poses),
                         'cpu_restore_folder': folder})
    require(len(all_poses) == 48 and len(all_files) == 293, 'all48 unique payload topology mismatch')
    result = {'schema': 'native48_private_evidence_persistence_v1', 'status': 'PASS_ALL3_FRESH_GET_AND_CLEAN_CPU_RESTORE',
              'operator_receipt_sha256': sha_file(source), 'total_archive_bytes': sum(row['bytes'] for row in rows),
              'objects': restored, 'unique_payload_files_reverified': len(all_files), 'unique_pose_count': len(all_poses),
              'visual_or_audio_subjective_acceptance': False, 'gpu_image_or_cold_boot_acceptance': False,
              'native_quality_decision': 'REJECTED_UNCHANGED', 'release_ready': False, 'cloud_mutations': False,
              'scope': 'Private archival/content-integrity evidence only; no active serving bundle or public media/image release'}
    existing = OUT.exists()
    if existing:
        require(not OUT.is_symlink() and json.loads(OUT.read_text()) == result,
                'existing descriptor differs; no overwrite permitted')
    else:
        with OUT.open('x') as out:
            json.dump(result, out, indent=2)
            out.write('\n')
    print(json.dumps({'status': result['status'], 'unique_payload_files': len(all_files),
                      'poses': len(all_poses), 'existing_descriptor_reverified_read_only': existing}))


if __name__ == '__main__':
    main()
