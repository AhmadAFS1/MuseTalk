"""CPU-only matched preprocessing evidence; never inference or quality approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/workspace/MuseTalk')
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/avatars'
NEW = BASE / 'a4_fixed_geometry_all6_v1'
OLD = Path('/workspace/experiments/avatar_diversity_20260927')
IDS = ('black_man_short_beard', 'black_woman', 'east_asian_man_goatee',
       'middle_eastern_man_full_beard', 'south_asian_woman', 'white_man_clean_shaven')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--comparison-sha256', required=True)
    p.add_argument('--bundle-manifest-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not a.execute or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('explicit execute and hidden CUDA devices required')
    if a.out != BASE / 'a4_preparation_preprocessing_review_0449' or a.out.exists():
        p.error('fresh fixed diagnostic output required')
    comparison = NEW / 'comparison.json'
    if sha(comparison) != a.comparison_sha256:
        raise ValueError('comparison binding changed')
    manifest_path = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/release/a4_latents_0440_sidecars/.musetalk_trt_artifact_manifest.json'
    if sha(manifest_path) != a.bundle_manifest_sha256:
        raise ValueError('new cache archive manifest changed')
    archive_files = {row['path']: row for row in json.loads(manifest_path.read_text())['files']}
    report = json.loads(comparison.read_text())
    if report['status'] != 'PASS_REPEAT_EXACT_DIAGNOSTIC_ONLY' or set(report['identities']) != set(IDS):
        raise ValueError('complete six-avatar repeat evidence required')
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    torch.set_num_threads(1); cv2.setNumThreads(1)
    a.out.mkdir(mode=0o700)
    evidence = {'schema': 'a4_cpu_preprocessing_review_assets_v1', 'scope': 'Source crops and jaw masks only, not generated decoder/composed video',
                'comparison_sha256': a.comparison_sha256, 'gpu_inference': False,
                'quality_accepted': False, 'visual_inspection': 'NOT_PERFORMED_BY_SCRIPT', 'identities': {}}
    for identity in IDS:
        old, new = OLD / identity, NEW / 'repeat1' / identity
        for folder in (old, new):
            if sha(folder / 'source.mp4') != report['validated_inputs'][identity + '/source.mp4']['sha256']:
                raise ValueError('source bytes changed')
        for name in ('cache.pt', 'masks.npz'):
            if sha(old / name) != report['validated_inputs'][identity + '/' + name]['sha256']:
                raise ValueError('original cache changed')
            path = new / name; pin = archive_files[path.relative_to(ROOT).as_posix()]
            if path.stat().st_size != pin['size'] or sha(path) != pin['sha256']:
                raise ValueError('new cache archive pin mismatch')
        # Both exact file hashes are independently checked before loading the
        # canonical cache pickle containing NumPy boxes as well as tensors.
        ca, cb = [torch.load(folder / 'cache.pt', map_location='cpu', weights_only=False) for folder in (old, new)]
        la, lb = [c['latents'].float().numpy() for c in (ca, cb)]
        per_frame = np.abs(la - lb).reshape(240, -1).max(axis=1)
        ba, bb = [np.asarray(c['boxes']) for c in (ca, cb)]
        cba, cbb = [np.asarray(c['cropboxes']) for c in (ca, cb)]
        box_changed = np.flatnonzero(np.any(ba != bb, axis=1) | np.any(cba != cbb, axis=1)).tolist()
        mask_metrics = report['identities'][identity]['historical_vs_repeat1']['masks']
        worst_mask = max(range(240), key=lambda i: mask_metrics[str(i)]['mean_abs'] if 'mean_abs' in mask_metrics[str(i)] else -1)
        indices = sorted(set([0, 119, 239, int(per_frame.argmax()), worst_mask, *box_changed]))
        frames = {}; cap = cv2.VideoCapture(str(old / 'source.mp4'))
        for i in range(240):
            ok, frame = cap.read()
            if not ok or frame.shape != (896, 512, 3):
                raise ValueError('canonical source framing changed')
            if i in indices:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        sheet = Image.new('RGB', (1280, len(indices) * 480), 'white'); draw = ImageDraw.Draw(sheet)
        with np.load(old / 'masks.npz', allow_pickle=False) as ma, np.load(new / 'masks.npz', allow_pickle=False) as mb:
            for row, i in enumerate(indices):
                frame = frames[i]; offset = row * 480
                sheet.paste(Image.fromarray(frame).resize((256, 448)), (0, offset + 24))
                draw.text((4, offset + 4), f'{identity} f{i}', fill='black')
                for arm, cache, masks, start in [('historical', ca, ma, 256), ('A4 repeat1', cb, mb, 768)]:
                    box = tuple(int(x) for x in cache['boxes'][i]); x, y, x1, y1 = box
                    crop = cv2.resize(frame[y:y1, x:x1], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    mask = masks[str(i)]; cx, cy, cx1, cy1 = map(int, cache['cropboxes'][i])
                    if mask.shape != (cy1 - cy, cx1 - cx):
                        raise ValueError('mask/cropbox geometry mismatch')
                    alpha = np.zeros((896, 512), dtype=np.uint8)
                    xa, ya, xb, yb = max(0, cx), max(0, cy), min(512, cx1), min(896, cy1)
                    alpha[ya:yb, xa:xb] = mask[ya - cy:yb - cy, xa - cx:xb - cx]
                    face_mask = cv2.resize(alpha[y:y1, x:x1], (256, 256), interpolation=cv2.INTER_NEAREST)
                    sheet.paste(Image.fromarray(crop), (start, offset + 24))
                    sheet.paste(Image.fromarray(face_mask).convert('RGB'), (start + 256, offset + 24))
                    draw.text((start + 4, offset + 4), arm + ' source crop', fill='black')
                    draw.text((start + 260, offset + 4), arm + ' jaw alpha', fill='black')
                    draw.text((start + 4, offset + 290), 'box=' + str(box), fill='black')
                    draw.text((start + 4, offset + 310), 'cropbox=' + str((cx, cy, cx1, cy1)), fill='black')
                draw.text((260, offset + 338), f'latent max abs={per_frame[i]:.8f}; mask mean abs={mask_metrics[str(i)].get("mean_abs")}', fill='black')
        path = a.out / (identity + '.png'); sheet.save(path)
        evidence['identities'][identity] = {'frames': indices, 'box_changed_frames': box_changed,
            'highest_latent_difference_frame': int(per_frame.argmax()), 'highest_mask_mean_difference_frame': worst_mask,
            'contact_sheet': str(path), 'contact_sha256': sha(path),
            'historical_cache_sha256': sha(old / 'cache.pt'), 'new_cache_sha256': sha(new / 'cache.pt')}
    with (a.out / 'assets.json').open('x') as stream:
        json.dump(evidence, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': 'PREPROCESSING_REVIEW_ASSETS_ONLY', 'identities': len(evidence['identities']), 'out': str(a.out)}))


if __name__ == '__main__':
    main()
