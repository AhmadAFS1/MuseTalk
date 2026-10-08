"""Read-only complete hash inventory; preserve core tensors separately from diagnostic PNGs."""
import hashlib
import json
from pathlib import Path
import socket

ROOT = Path('/workspace/MuseTalk')
IDS = ('startup_trace_v2_black_man_1810', 'startup_missing_syncnet_black_man_v2_1841')
OUTPUT = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup/diagnostic_preparation_file_inventory_1851.json'
assert socket.gethostname() == 'a830e00ce20c'
assert not OUTPUT.exists() and not OUTPUT.is_symlink()
rows = []
for avatar in IDS:
    folder = ROOT / 'results/v15/avatars' / avatar
    for path in sorted(folder.rglob('*')):
        assert not path.is_symlink()
        if path.is_dir():
            continue
        assert path.is_file() and path.stat().st_size < 32 * 1024**2
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024**2), b''):
                digest.update(chunk)
        rows.append({'avatar': avatar, 'path': path.relative_to(folder).as_posix(),
                     'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
for avatar in IDS:
    for kind in ('full_imgs', 'mask'):
        assert sum(r['avatar'] == avatar and r['path'].startswith(kind + '/') and
                   r['path'].endswith('.png') for r in rows) == 480
data = {'schema': 'diagnostic_preparation_complete_file_inventory_v1', 'status': 'PASS_COMPLETE_HASH_INVENTORY',
        'release_ready': False, 'files': rows, 'payload_files': len(rows),
        'preservation_scope': 'Core latents/coords/mask_coords/avator_info transferred separately. All PNG hashes inventoried; complete diagnostic PNG payload recovery is NOT claimed. Canonical source video already preserved independently.'}
with OUTPUT.open('x') as handle:
    json.dump(data, handle, indent=2)
print(json.dumps({'status': data['status'], 'payload_files': len(rows)}))
