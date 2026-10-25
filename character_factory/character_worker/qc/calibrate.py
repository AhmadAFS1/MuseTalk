#!/usr/bin/env python3
"""Run video_qc over known-good clips and report every check's margin to its threshold.

Use it when tuning qc/profiles.json: a threshold that a reference clip already
violates would reject production output that a human accepted.

  python qc/calibrate.py --profile strict \
      --spec /path/video.mp4,talking,/path/anchor.png,240 \
      --references-dir /workspace/minimax-h3-workflows \
      --out qc/calibration/strict_<date>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_qc import EXPECTED_FRAMES, analyze_video, load_profiles, resolve_thresholds  # noqa: E402

PRESET_FRAMES = {'short-japanese': 73, 'short-latina': 73}


def specs_from_references(root: Path) -> list[dict]:
    presets = json.loads((root / 'workflows/presets.json').read_text())
    out = []
    for name, preset in presets.items():
        video = root / preset['reference_video']
        if not video.is_file():
            continue
        out.append(dict(video=video, pose=preset.get('pose', 'talking'), anchor=root / preset['image'],
                        frames=PRESET_FRAMES.get(name, EXPECTED_FRAMES.get(preset.get('pose', 'talking'))), label=name))
    for name in ('musetalk-japanese', 'musetalk-latina'):
        video = root / 'references' / name / 'video.mp4'
        if video.is_file():
            out.append(dict(video=video, pose='talking', anchor=None, frames=240, label=name))
    return out


def parse_spec(text: str) -> dict:
    parts = text.split(',')
    video, pose = Path(parts[0]), parts[1]
    anchor = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    frames = int(parts[3]) if len(parts) > 3 and parts[3] else EXPECTED_FRAMES.get(pose)
    return dict(video=video, pose=pose, anchor=anchor, frames=frames, label=video.parent.name + '/' + video.name)


def margin(check: dict) -> float | None:
    value, threshold = check.get('value'), check.get('threshold')
    if not isinstance(value, (int, float)) or not isinstance(threshold, (int, float)) or isinstance(value, bool):
        return None
    name = check['name']
    if name.endswith('.duration'):
        return None                           # two-sided tolerance check
    if any(key in name for key in ('min_', '.moving.', 'detection_fraction', 'rms_dbfs', 'first_frame_ssim', 'mid_frames')):
        return round(value - threshold, 4)       # higher is better
    return round(threshold - value, 4)           # lower is better


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--profile', default='strict')
    p.add_argument('--profiles', type=Path, default=Path(__file__).with_name('profiles.json'))
    p.add_argument('--spec', action='append', default=[], help='video,pose[,anchor[,frames]]')
    p.add_argument('--references-dir', type=Path)
    p.add_argument('--face-backend', default='auto')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    specs = [parse_spec(s) for s in args.spec]
    if args.references_dir:
        specs += specs_from_references(args.references_dir)
    if not specs:
        p.error('give --spec and/or --references-dir')
    profiles = load_profiles(args.profiles)
    rows, worst = [], {}
    for spec in specs:
        thresholds = resolve_thresholds(profiles, args.profile, spec['pose'])
        started = time.monotonic()
        report = analyze_video(spec['video'], spec['pose'], thresholds, spec['anchor'], spec['frames'],
                               args.face_backend, None, keep_series=False)
        row = dict(label=spec['label'], pose=spec['pose'], passed=report['pass'], hard=report['hard_failures'],
                   soft=report['warnings'], seconds=round(time.monotonic() - started, 1), backend=report.get('face_backend'),
                   motion=report.get('motion'), face=report.get('face'), mouth=report.get('mouth'),
                   anchor_ssim=report.get('anchor_ssim'), audio=report.get('audio'),
                   checks={c['name']: {'value': c['value'], 'threshold': c['threshold'], 'margin': margin(c), 'pass': c['pass']}
                           for c in report['checks']})
        rows.append(row)
        for name, c in row['checks'].items():
            if c['margin'] is not None:
                key = (spec['pose'], name)
                if key not in worst or c['margin'] < worst[key]['margin']:
                    worst[key] = dict(margin=c['margin'], value=c['value'], threshold=c['threshold'], clip=spec['label'])
        status = 'PASS' if report['pass'] else 'FAIL'
        print(f"{status:4} {spec['label']:28} {spec['pose']:8} hard={report['hard_failures']} soft={report['warnings']} "
              f"median_diff={report.get('motion', {}).get('median_frame_diff')} mouth={row['mouth']} {row['seconds']}s", flush=True)
    print('\nSmallest margins per (pose, check); negative means a reference clip violates the threshold:')
    for (pose, name), w in sorted(worst.items(), key=lambda kv: kv[1]['margin']):
        print(f"  {w['margin']:>9}  {pose:8} {name:36} value={w['value']} threshold={w['threshold']}  ({w['clip']})")
    result = dict(profile=args.profile, profiles_file=str(args.profiles), generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  clips=rows, smallest_margins={f'{pose}:{name}': w for (pose, name), w in worst.items()})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print('wrote', args.out)
    return 0 if all(r['passed'] for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
