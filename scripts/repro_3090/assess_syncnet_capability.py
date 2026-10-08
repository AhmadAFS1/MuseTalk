"""Bind preserved core tensors to the complete diagnostic inventory, without parity claims."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
INV = BASE / 'startup/diagnostic_preparation_file_inventory_1851.json'
OUTPUT = BASE / 'startup/syncnet_dependency_assessment_1855.json'
assert not OUTPUT.exists()
inventory = json.loads(INV.read_text())
probe = json.loads((BASE / 'startup/missing_syncnet_capability_v2_1841.json').read_text())
receipt = json.loads((BASE / 'release/private_missing_syncnet_v2_1847.json').read_text())
assert probe['status'] == 'PASS_EXERCISED_API_CAPABILITIES_WITH_SYNCNET_ABSENT'
assert probe['checkpoint_restored'] and probe['api_stopped']
assert receipt['status'] == 'PASS_PRIVATE_CONDITIONAL_PUT_FRESH_GET_ALL_PAYLOAD_SHA'
core_names = {'latents.pt', 'coords.pkl', 'mask_coords.pkl', 'avator_info.json'}
rows = [r for r in inventory['files'] if r['path'] in core_names]
assert len(rows) == 8 and inventory['payload_files'] == 1930
for row in rows:
    path = BASE / 'startup/private_diagnostic_prepare_core_1851' / row['avatar'] / row['path']
    assert not path.is_symlink() and path.stat().st_size == row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
source_rows = [r for r in inventory['files'] if r['path'] == 'input_video.mp4']
assert len(source_rows) == 2 and all(r['sha256'] == '064183b0d8f80105159c687858566ad271fd768037773f8cfaacb3ad3452837e' for r in source_rows)
data = {'schema': 'syncnet_api_dependency_assessment_v1', 'status': 'EXERCISED_CAPABILITIES_PASS_INSTALLER_CHANGE_NOT_YET_IMPLEMENTED',
        'capabilities_observed_without_checkpoint': ['isolated canonical API startup', 'cached human-WAV speech serving', 'fresh canonical avatar preparation'],
        'checkpoint_bytes': 1488019828, 'checkpoint_restored': True, 'api_stopped': True,
        'core_files_bound_to_remote_inventory': rows, 'inventory_files': 1930,
        'inventory_sha256': hashlib.sha256(INV.read_bytes()).hexdigest(),
        'private_archive_sha256': receipt['sha256'], 'private_archive_bytes': receipt['bytes'],
        'core_payload_private_s3_fresh_get_verified': True, 'complete_diagnostic_png_payload_preserved': False,
        'partial_recursive_copy_cancelled_and_retained_locally': True,
        'quality_or_latent_parity_established': False, 'scored_startup_savings_seconds': None,
        'production_or_installer_changed': False, 'release_ready': False,
        'next_action': 'Separate training-only SyncNet weights from avatar-prep download/check contracts, retain explicit backward-compatible training/full behavior, test installer negative/positive cases and image builds before measuring a genuine fresh boot.',
        'limitations': ['TTS disabled; not every optional API feature was exercised.', 'Direct canonical launcher, not full on-start/installer or EC2 readiness.', 'New preparation is diagnostic only; no established avatar was replaced and latent randomness was not controlled.', 'Diagnostic PNG hashes are preserved, but their complete raw payload is not archived; the canonical source and production48 review evidence have separate preservation.']}
with OUTPUT.open('x') as handle:
    json.dump(data, handle, indent=2)
    handle.write('\n')
print(json.dumps({'status': data['status'], 'core_files': len(rows), 'release_ready': False}))
