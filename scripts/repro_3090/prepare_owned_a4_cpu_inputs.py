"""A4-only CPU input restoration/audit; no CUDA imports or quality acceptance."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/workspace/MuseTalk')
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008'
INPUTS = BASE / 'harnesses/quality-inputs-v1.json'
PARENT_SHA = '6d3ab6ef31605c2231605e03042e82361a27112589ef6d7f6f8ff8f4b016eea5'
SYNCNET_SHA = '38fa63bad3ed2332f647c40a5dc616cb0e233db8579f698f62af4c41965c4da5'
DEADLINE = dt.datetime(2026, 10, 9, 5, 30, tzinfo=dt.timezone.utc)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--restore-syncnet', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if not a.execute or socket.gethostname() != '9dad9291adec':
        raise ValueError('explicit owned A4 CPU execution required')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    if uuid != 'GPU-050b4bc5-99f6-305e-8c7f-fd00ea44551f' or (DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 600:
        raise ValueError('owned identity or cleanup margin mismatch')
    if a.out.exists() or a.out.is_symlink() or a.out.parent.resolve() != BASE / 'provisioning':
        raise ValueError('fresh scoped receipt required')
    if digest(INPUTS) != PARENT_SHA:
        raise ValueError('original input manifest changed')
    document = json.loads(INPUTS.read_text())
    if len(document['files']) != 878:
        raise ValueError('original 878 inputs required')
    checkpoint = ROOT / 'models/syncnet/latentsync_syncnet.pt'
    if a.restore_syncnet:
        if checkpoint.exists() and digest(checkpoint) != SYNCNET_SHA:
            raise ValueError('refusing to overwrite changed SyncNet payload')
        from huggingface_hub import hf_hub_download
        hf_hub_download('ByteDance/LatentSync', 'latentsync_syncnet.pt',
                        revision='405eda8eab9f65c1a6e0c292a5dee5a08089e2ae',
                        local_dir=str(checkpoint.parent))
        if digest(checkpoint) != SYNCNET_SHA:
            raise ValueError('downloaded SyncNet payload differs')
    missing, changed, metadata = [], [], []
    exact = 0
    for row in document['files']:
        path = (INPUTS.parent / row['path']).resolve()
        if not path.is_file():
            missing.append(row['path'])
            continue
        observed = dict(path=row['path'], bytes=path.stat().st_size, sha256=digest(path))
        if observed == row:
            exact += 1
        elif row['path'].endswith('.metadata'):
            metadata.append(observed)
        else:
            changed.append(dict(expected=row, observed=observed))
    result = dict(schema='owned_a4_cpu_original_input_audit_v1', instance_id='54957508',
                  hostname=socket.gethostname(), gpu_uuid=uuid,
                  finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  status='NONMETADATA_EXACT' if not missing and not changed else 'INCOMPLETE',
                  parent_input_manifest_sha256=PARENT_SHA, input_paths=878, exact_paths=exact,
                  missing=missing, changed_nonmetadata=changed, changed_metadata=metadata,
                  syncnet_restored=a.restore_syncnet, original_manifest_modified=False,
                  gpu_imports=False, quality_accepted=False, release_ready=False)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('missing', 'changed_nonmetadata', 'changed_metadata')}
                     | dict(missing_count=len(missing), changed_nonmetadata_count=len(changed), changed_metadata_count=len(metadata))))
    return 0 if result['status'] == 'NONMETADATA_EXACT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
