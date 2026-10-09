"""CPU-only, explicit metadata-only successor; never alter original evidence.

Audit is the default. Preregistration writes three fresh artifacts, receipt last.
The unchanged quality_envelope loader/comparator can consume the new manifest
and envelope; verify_preregistration must authenticate their ancestry first.
No candidate results are accepted as inputs and no quality acceptance is issued.
Missing historical metadata remains UNKNOWN, not proof of global irrecoverability.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[2]
PARENT_INPUT_SHA = '6d3ab6ef31605c2231605e03042e82361a27112589ef6d7f6f8ff8f4b016eea5'
PARENT_ENVELOPE_SHA = '8c2c25961109171f5bd18f88d3bf9f0433b9cba31e48f19869cd75a25e7471e4'
HELPER_SHA = 'a5a98185dad4db0d1b89c2d3e04b8832721d99461cca7381605b62b035697e55'
WORKER_PATH = '../../../../scripts/chin_multistream/worker.py'
WORKER_SHA = '2e6e88fbe106ca964b2b5e31af44203ce5dff201eb2c37436726b65520cc447b'
METADATA = {
    'models/.cache/huggingface/download/auxiliary/s3fd-619a316812.pth.metadata': 'models/auxiliary/s3fd-619a316812.pth',
    **{f'models/.cache/huggingface/download/musetalkV15/{n}.metadata': f'models/musetalkV15/{n}'
       for n in ('musetalk.json', 'unet.pth')},
    **{f'models/{group}/.cache/huggingface/download/{n}.metadata': f'models/{group}/{n}'
       for group, names in {'dwpose': ('dw-ll_ucoco_384.pth',), 'sd-vae': ('config.json', 'diffusion_pytorch_model.bin'),
           'syncnet': ('latentsync_syncnet.pt',), 'taesd': ('config.json', 'diffusion_pytorch_model.safetensors'),
           'whisper': ('config.json', 'preprocessor_config.json', 'pytorch_model.bin')}.items() for n in names},
}
METADATA = {'../../../../' + k: '../../../../' + v for k, v in METADATA.items()}
LINEAGE_SCHEMA = 'preregistered_metadata_only_quality_lineage_v1'
DERIVED_STATUS = 'PREREGISTERED_METADATA_ONLY_SUCCESSOR_ENVELOPE'


class Rejected(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Rejected(reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def invalid(value):
        raise Rejected('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute_nonsymlink_path_required')
    return path.resolve()


def inspect_file(path, *, keep=False, git_blob=False):
    path = safe_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'regular_file_required')
        require(not keep or before.st_size <= (8 << 20), 'bounded_document_required')
        digest = hashlib.sha256()
        blob = hashlib.sha1(b'blob ' + str(before.st_size).encode() + b'\0') if git_blob else None
        chunks, size = [], 0
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            size += len(chunk); digest.update(chunk)
            if blob is not None: blob.update(chunk)
            if keep: chunks.append(chunk)
        after = os.fstat(stream.fileno())
    require(size == before.st_size and (before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'input_changed_during_read')
    return {'bytes': size, 'sha256': digest.hexdigest(),
            'git_blob_sha1': blob.hexdigest() if blob is not None else None, 'raw': b''.join(chunks) if keep else None}


def pinned_document(path, digest):
    result = inspect_file(path, keep=True)
    require(result['sha256'] == digest, 'immutable_parent_hash_mismatch')
    return parse(result['raw'])


def parents(parent_inputs, parent_envelope):
    inputs = pinned_document(parent_inputs, PARENT_INPUT_SHA)
    envelope = pinned_document(parent_envelope, PARENT_ENVELOPE_SHA)
    require(inputs['schema'] == 'repro_3090_inputs_v1' and len(inputs['files']) == 878, 'original_878_manifest_required')
    require(envelope['schema'] == 'r5_measured_quality_envelope_v1'
            and envelope['status'] == 'FROZEN_REFERENCE_ENVELOPE'
            and envelope['input_manifest_sha256'] == PARENT_INPUT_SHA, 'original_envelope_binding_required')
    require(envelope['helper_sha256'] == HELPER_SHA and
            inspect_file(ROOT / 'scripts/repro_3090/quality_envelope.py')['sha256'] == HELPER_SHA, 'original_helper_changed')
    require(len(envelope['avatars']) == 6 and len(envelope['global_bounds']) == 104
            and all(len(v['bounds']) == 99 for v in envelope['avatars'].values()), 'original_698_bounds_required')
    require(all(r['input_manifest_sha256'] == PARENT_INPUT_SHA for r in envelope['references']), 'original_reference_binding_changed')
    return inputs, envelope


def audit(inputs, base):
    """Every path is read; all metadata is validated, even byte-identical rows."""
    rows = inputs['files']
    originals = {row['path']: row for row in rows}
    require(len(rows) == len(originals) == 878, 'duplicate_or_missing_878_paths')
    require({p for p in originals if p.endswith('.metadata')} == set(METADATA), 'exact_12_metadata_allowlist_required')
    require(originals[WORKER_PATH]['sha256'] == WORKER_SHA, 'historical_worker_required')
    require(set(METADATA.values()) <= set(originals), 'metadata_payload_missing_from_manifest')
    observed, successor_rows, changes = {}, [], []
    for before in rows:
        require(set(before) == {'path', 'bytes', 'sha256'}, 'original_row_schema_changed')
        path = Path(base) / before['path']
        got = inspect_file(path, keep=before['path'] in METADATA, git_blob=before['path'] in METADATA.values())
        after = {'path': before['path'], 'bytes': got['bytes'], 'sha256': got['sha256']}
        if before['path'] not in METADATA:
            require(after == before, 'nonmetadata_content_changed:' + before['path'])
        elif after != before:
            changes.append({'before': before, 'after': after})
        observed[before['path']] = got
        successor_rows.append(after)
    metadata_proofs = []
    for name, payload_name in METADATA.items():
        raw, payload = observed[name]['raw'], observed[payload_name]
        require(len(raw) <= 4096, 'metadata_oversized')
        try:
            fields = raw.decode('utf-8').splitlines()
            require(len(fields) == 3 and re.fullmatch('[0-9a-f]{40}', fields[0])
                    and re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', fields[1]), 'invalid_metadata_syntax')
            stamp = float(fields[2])
            require(math.isfinite(stamp) and stamp > 0, 'invalid_metadata_timestamp')
        except (UnicodeError, ValueError, IndexError):
            raise Rejected('invalid_metadata_syntax_or_timestamp') from None
        expected = payload['sha256'] if len(fields[1]) == 64 else payload['git_blob_sha1']
        require(fields[1] == expected, 'metadata_etag_does_not_match_original_payload')
        metadata_proofs.append({'path': name, 'payload_path': payload_name,
            'payload_sha256': payload['sha256'], 'payload_bytes': payload['bytes'], 'commit_syntax_valid': True,
            'upstream_commit_independently_verified': False, 'etag': fields[1], 'etag_content_verified': True,
            'etag_algorithm': 'sha256' if len(fields[1]) == 64 else 'git_blob_sha1',
            'metadata_sha256': observed[name]['sha256'], 'timestamp_syntax_valid': True})
    successor = {**inputs, 'files': successor_rows}
    return successor, {'paths': 878, 'nonmetadata_exact': 866, 'metadata_validated': 12,
        'changed_metadata_count': len(changes), 'changes': changes, 'metadata_proofs': metadata_proofs,
        'worker_sha256': WORKER_SHA, 'original_metadata_recoverability': 'UNKNOWN'}


def build_envelope(parent, manifest_sha, at, audit_sha):
    require(dt.datetime.fromisoformat(at).utcoffset() == dt.timedelta(0)
            and dt.datetime.fromisoformat(at) > dt.datetime.fromisoformat(parent['frozen_utc']), 'preregistration_timestamp_invalid')
    derived = copy.deepcopy(parent)
    derived.update(status=DERIVED_STATUS, frozen_utc=at, input_manifest_sha256=manifest_sha,
        metadata_only_lineage={'schema': LINEAGE_SCHEMA, 'parent_envelope_sha256': PARENT_ENVELOPE_SHA,
            'parent_input_manifest_sha256': PARENT_INPUT_SHA, 'parent_frozen_utc': parent['frozen_utc'],
            'successor_input_manifest_sha256': manifest_sha, 'preregistered_utc': at,
            'audit_sha256': audit_sha, 'bounds_changed': False, 'original_reference_bindings_changed': False,
            'quality_accepted': False, 'release_ready': False})
    return derived


def prepare(parent_inputs, parent_envelope):
    inputs, envelope = parents(parent_inputs, parent_envelope)
    successor, audit_record = audit(inputs, safe_path(parent_inputs).parent)
    at = dt.datetime.now(dt.timezone.utc).isoformat()
    new_manifest_sha = sha(encoded(successor))
    derived = build_envelope(envelope, new_manifest_sha, at, sha(encoded(audit_record)))
    receipt = {'schema': LINEAGE_SCHEMA, 'status': 'PREREGISTERED_METADATA_EQUIVALENCE_ONLY',
        'parent_input_manifest_sha256': PARENT_INPUT_SHA, 'parent_envelope_sha256': PARENT_ENVELOPE_SHA,
        'parent_frozen_utc': envelope['frozen_utc'], 'preregistered_utc': at,
        'successor_input_manifest_sha256': new_manifest_sha, 'successor_envelope_sha256': sha(encoded(derived)),
        'helper_sha256': HELPER_SHA, 'audit': audit_record, 'metric_count': 698,
        'quality_accepted': False, 'performance_accepted': False, 'release_ready': False,
        'originals_modified': False, 'candidate_results_consumed': False}
    return successor, derived, receipt


def write_new(path, raw):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600), 'wb') as output:
        output.write(raw); output.flush(); os.fsync(output.fileno())


def preregister(*, parent_inputs, parent_envelope, out_inputs=None, out_envelope=None, receipt_path=None, execute=False):
    """Audit by default; never write metadata, model files, sources, or old reports."""
    successor, derived, receipt = prepare(parent_inputs, parent_envelope)
    if execute is not True:
        return {**receipt, 'status': 'AUDIT_ONLY_NO_PREREGISTRATION', 'outputs_written': False}
    outputs = [safe_path(p) for p in (out_inputs, out_envelope, receipt_path)]
    require(len(set(outputs)) == 3 and not any(p.exists() for p in outputs), 'fresh_distinct_outputs_required')
    require(outputs[0].parent == safe_path(parent_inputs).parent
            and outputs[1].parent == outputs[2].parent == safe_path(parent_envelope).parent, 'fresh_sibling_outputs_required')
    require(receipt['audit']['changed_metadata_count'] > 0, 'metadata_successor_not_needed')
    # Receipt is the final commit marker; partial output is not a valid lineage.
    for path, value in zip(outputs, (successor, derived, receipt)):
        write_new(path, encoded(value))
    return {**receipt, 'outputs_written': True, 'receipt_sha256': sha(encoded(receipt))}


def verify_preregistration(*, parent_inputs, parent_envelope, successor_inputs, successor_envelope,
                          receipt_path, expected_receipt_sha256):
    """Re-read every actual file and prove the permitted delta before comparison."""
    original, envelope = parents(parent_inputs, parent_envelope)
    receipt = pinned_document(receipt_path, expected_receipt_sha256)
    require(receipt['schema'] == LINEAGE_SCHEMA and receipt['status'] == 'PREREGISTERED_METADATA_EQUIVALENCE_ONLY'
            and receipt['parent_input_manifest_sha256'] == PARENT_INPUT_SHA
            and receipt['parent_envelope_sha256'] == PARENT_ENVELOPE_SHA, 'lineage_parent_mismatch')
    require(all(receipt[k] is False for k in ('quality_accepted', 'performance_accepted', 'release_ready',
                                            'originals_modified', 'candidate_results_consumed')), 'lineage_claim_invalid')
    require(safe_path(successor_inputs).parent == safe_path(parent_inputs).parent
            and safe_path(successor_envelope).parent == safe_path(receipt_path).parent == safe_path(parent_envelope).parent,
            'lineage_path_rebinding_forbidden')
    successor = pinned_document(successor_inputs, receipt['successor_input_manifest_sha256'])
    derived = pinned_document(successor_envelope, receipt['successor_envelope_sha256'])
    fresh, audit_record = audit(original, safe_path(parent_inputs).parent)
    require(encoded(successor) == encoded(fresh) and encoded(receipt['audit']) == encoded(audit_record), 'lineage_actual_inputs_changed')
    expected = build_envelope(envelope, receipt['successor_input_manifest_sha256'],
                              receipt['preregistered_utc'], sha(encoded(audit_record)))
    require(encoded(derived) == encoded(expected), 'envelope_changed_beyond_metadata_binding')
    require(receipt['metric_count'] == 698 and receipt['helper_sha256'] == HELPER_SHA
            and receipt['parent_frozen_utc'] == envelope['frozen_utc'], 'lineage_policy_changed')
    return {'status': 'VERIFIED_METADATA_EQUIVALENCE_ONLY', 'input_manifest_sha256': receipt['successor_input_manifest_sha256'],
            'envelope_sha256': receipt['successor_envelope_sha256'], 'preregistered_utc': receipt['preregistered_utc'],
            'quality_accepted': False, 'release_ready': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--parent-inputs', type=Path, required=True)
    parser.add_argument('--parent-envelope', type=Path, required=True)
    for name in ('out-inputs', 'out-envelope', 'receipt'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute:
        require(all((args.out_inputs, args.out_envelope, args.receipt)), 'explicit_output_paths_required')
    result = preregister(parent_inputs=args.parent_inputs, parent_envelope=args.parent_envelope,
        out_inputs=args.out_inputs, out_envelope=args.out_envelope, receipt_path=args.receipt, execute=args.execute)
    print(json.dumps({k: v for k, v in result.items() if k != 'audit'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Rejected, OSError, KeyError, TypeError):
        print(json.dumps({'status': 'INVALID', 'quality_accepted': False, 'release_ready': False}))
        raise SystemExit(2) from None
