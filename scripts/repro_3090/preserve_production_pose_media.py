#!/usr/bin/env python3
"""Index or losslessly preserve the fixed rejected native48 diagnostic.

Pack mode is CPU-only on owned A1 and refuses active live streams. It verifies
every muxed frame and human-audio sample before omitting duplicate silent video.
Three <5GB archives permit later conditional private PUT, not public release.
No cloud calls, GPU work, model deserialization, deletion, or input rewrite.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import wave

import warm_isolated_api as warm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import trt_artifact_bundle as bundle

REPORT_ROOT = 'docs/fps_comparisons/rtx3090_r5_20261008/avatars/native_v1_all48_cycle_v2_1523'
REPORT_SHA = '8d4fb71f1b56cf844c86091ca105771fe2d0a0f702ddaeb2a0654eb0f9ee35fe'
INVALID_REPORT = 'docs/fps_comparisons/rtx3090_r5_20261008/avatars/native_v1_all48_1500/report.json'
INVALID_SHA = '28fc6103829b3f6f26f0fc671cccc28a6c762c75deffedec4331426e73c9cd8d'
OUTPUT_PARENT = warm.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release'
VERIFIED_PROOF = OUTPUT_PARENT / 'native_v1_production48_media_1608/decoded_media_proof.json'
VERIFIED_PROOF_SHA = '4eae33de59483c6a626caf3512eaf96c6d58faac76cf26193f454cfb5d035036'


def require(ok, reason):
    warm.require(ok, reason)


def sha_file(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
        return value.hexdigest()


def load_report(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024**2, 'invalid report file')
    require(sha_file(path) == REPORT_SHA, 'fixed terminal report SHA mismatch')
    data = json.loads(path.read_text())
    rows = data.get('results') or []
    require(data.get('status') == 'PASS' and data.get('coverage_complete') is True
            and data.get('release_ready') is False and len(rows) == 48
            and len({r['avatar_id'] for r in rows}) == 48, 'incomplete terminal48 report')
    require(all(r.get('status') == 'PASS' and r.get('inputs_unchanged_after_render') is True
                and r.get('recipe_checks', {}).get('passes') is True for r in rows), 'pose checks incomplete')
    return data


def index(data):
    rows = data['results']
    return {'schema': 'production_pose_terminal_index_v1', 'candidate': 'native_sm86_r5_v1',
            'status': 'PASS_RENDER_COMPATIBILITY_ONLY', 'completed_poses': 48, 'characters': 16,
            'raw_report': {'sha256': REPORT_SHA, 'git_payload_included': False},
            'visual_review_status': data['visual_review_status'], 'release_ready': False,
            'native_numerical_rejection_waived': False, 'capacity_acceptance': False,
            'backends': data['backends'], 'recipe': data['recipe'], 'audio': data['audio'],
            'not_claimed': data['not_claimed'], 'poses': [
                {'avatar_id': r['avatar_id'], 'character': r['character'], 'pose': r['pose'],
                 'recipe_checks': r['recipe_checks'], 'frames_digest': r['clip']['frames_digest'],
                 'review_clip': r['review_clip'], 'contact': r['contact_samples']['file'],
                 'arrays': r['array_artifacts']} for r in rows]}


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(warm.ROOT), 'media escapes fixed root')
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'media path has symlink')
        if item == warm.ROOT:
            break
    require(path.is_file(), 'media file missing')
    return path


def verify_record(row):
    path = safe_path(row['path'])
    require(path.stat().st_size == row['bytes'] and sha_file(path) == row['sha256'], 'media size/SHA mismatch')
    return path


def decoded_frame_hashes(path, width, height):
    require((width, height) == (512, 896), 'original resolution changed')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
                             '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size, result = width * height * 3, []
    try:
        while True:
            raw = proc.stdout.read(size)
            if not raw:
                break
            require(len(raw) == size and len(result) < 240, 'truncated or extra muxed video frame')
            digest = hashlib.sha256(f'({height}, {width}, 3)|uint8|'.encode())
            digest.update(raw)
            result.append(digest.hexdigest())
        require(proc.wait(timeout=15) == 0, 'muxed video decode failed')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return result


def verify_audio(path, expected_pcm):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
                                   '-map', '0:a:0', '-c:a', 'pcm_s16le', '-f', 's16le', '-'],
                                  timeout=60, stderr=subprocess.DEVNULL)
    require(raw == expected_pcm, 'muxed human audio differs from original first10s')
    return hashlib.sha256(raw).hexdigest()


def write_new(path, data):
    with path.open('x') as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write('\n')


def validated_previous_proof(proof, data, pcm_sha):
    require(proof.get('status') == 'PASS_ALL48_MUXED_PIXEL_AND_HUMAN_PCM_INTEGRITY'
            and proof.get('report_sha256') == REPORT_SHA and proof.get('release_ready') is False
            and proof.get('cloud_mutations') is False, 'unproven previous decoded media')
    rows = proof.get('poses') or []
    require(len(rows) == 48 and len({r.get('avatar_id') for r in rows}) == 48, 'previous decoded coverage incomplete')
    actual = {row['avatar_id']: row for row in rows}
    for row in data['results']:
        result = actual.get(row['avatar_id']) or {}
        require(result.get('all240_frame_hashes_match') is True
                and result.get('frames_digest') == row['clip']['frames_digest']
                and result.get('original10s_pcm_sha256') == pcm_sha, 'previous frame/audio proof mismatch')
    return proof


def pack(data, out, reuse_decoded_proof=False):
    require(socket.gethostname() == 'a830e00ce20c', 'pack only on owned A1')
    require(dt.datetime.now(dt.timezone.utc) < warm.DEADLINE, 'A1 deadline passed')
    require(out.is_absolute() and out.parent.resolve() == OUTPUT_PARENT and not out.exists(), 'fresh fixed output directory required')
    require(warm.request('/webrtc/sessions/stats?view=lifetime')[1].get('active_streams') == 0, 'timed live work active')
    require(shutil.disk_usage(OUTPUT_PARENT).free > 16 * 1024**3, 'insufficient pack space')
    report_path = warm.ROOT / REPORT_ROOT / 'report.json'
    invalid_path = safe_path(warm.ROOT / INVALID_REPORT)
    require(sha_file(invalid_path) == INVALID_SHA, 'historical invalid report SHA mismatch')
    audio = verify_record(data['audio'])
    conditioning = verify_record(data['audio']['conditioning_file'])
    with wave.open(str(audio), 'rb') as source:
        require((source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2), 'source audio format changed')
        expected_pcm = source.readframes(160000)
    require(len(expected_pcm) == 320000, 'first10s source audio incomplete')
    out.mkdir()
    proof = {'schema': 'native48_lossless_media_preservation_v1', 'status': 'IN_PROGRESS',
             'report_sha256': REPORT_SHA, 'release_ready': False, 'cloud_mutations': False,
             'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'poses': [], 'archives': [],
             'omitted': 'Only duplicate silent frames.mkv files. Every included audio-bearing MKV must decode to all240 original frame hashes and the first10s original PCM; no resize, media reencode, or cache change.'}
    if reuse_decoded_proof:
        require(not VERIFIED_PROOF.is_symlink() and VERIFIED_PROOF.stat().st_size < 64 * 1024
                and sha_file(VERIFIED_PROOF) == VERIFIED_PROOF_SHA, 'fixed decoded-proof SHA mismatch')
        proof = validated_previous_proof(json.loads(VERIFIED_PROOF.read_text()), data,
                                         hashlib.sha256(expected_pcm).hexdigest())
        proof['decoded_proof_reused'] = {'path': str(VERIFIED_PROOF), 'sha256': VERIFIED_PROOF_SHA,
                                        'original_decoder_helper_sha256': '2122cf9aa7e54fb594866a0b4337cac32ab697c14e6fa39b3ea9312d704a1387',
                                        'current_media_size_and_hash_reverified_before_pack': True}
    selected_files = []
    for row in data['results']:
        review = verify_record(row['review_clip'])
        contact = verify_record(row['contact_samples']['file'])
        arrays = [verify_record(record) for record in row['array_artifacts']]
        pose = safe_path(review.parent / 'pose.json')
        require(json.loads(pose.read_text()) == row, 'per-pose receipt differs from terminal report')
        if not reuse_decoded_proof:
            frames = decoded_frame_hashes(review, row['clip']['width'], row['clip']['height'])
            require(len(frames) == 240 and frames == row['clip']['frame_sha256'], 'muxed frame hashes differ')
            pcm_sha = verify_audio(review, expected_pcm)
            proof['poses'].append({'avatar_id': row['avatar_id'], 'all240_frame_hashes_match': True,
                                   'frames_digest': row['clip']['frames_digest'], 'original10s_pcm_sha256': pcm_sha})
        selected_files.append([review, contact, *arrays, pose])
        print(json.dumps({'media_verified': len(selected_files), 'expected': 48}), flush=True)
    proof['status'] = 'PASS_ALL48_MUXED_PIXEL_AND_HUMAN_PCM_INTEGRITY'
    write_new(out / 'decoded_media_proof.json', proof)
    for part in range(3):
        files = [report_path, audio, conditioning, out / 'decoded_media_proof.json']
        if part == 0:
            files.append(invalid_path)
        files += [path for group in selected_files[part * 16:(part + 1) * 16] for path in group]
        archive = out / f'native-v1-production48-review-part{part + 1}of3.tar.gz'
        sidecar = out / f'part{part + 1}_sidecars'
        require(not archive.exists() and not sidecar.exists(), 'bundle output exists')
        bundle.create_bundle(SimpleNamespace(repo_root=warm.ROOT, required_files=','.join(str(p.relative_to(warm.ROOT)) for p in files),
                                            required_dirs='', optional_paths='', strict=True, keep_symlinks=False,
                                            profile='REJECTED-native-sm86-v1-production48-media', sidecar_dir=sidecar,
                                            output=archive, compresslevel=1))
        require(archive.stat().st_size < 5 * 1024**3, 'archive exceeds conditional single-PUT limit')
        proof['archives'].append({'filename': archive.name, 'bytes': archive.stat().st_size,
                                  'sha256': sha_file(archive), 'poses': 16})
    proof['finished_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_new(out / 'preservation_receipt.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('index', 'pack'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-decoded-proof', action='store_true',
                        help='Reuse only the fixed SHA-bound successful48 decode proof after rechecking every current media file hash')
    args = parser.parse_args()
    data = load_report(args.report)
    if args.mode == 'index':
        write_new(args.out, index(data))
    else:
        require(args.report == warm.ROOT / REPORT_ROOT / 'report.json', 'pack requires fixed worker report')
        pack(data, args.out, args.reuse_decoded_proof)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
