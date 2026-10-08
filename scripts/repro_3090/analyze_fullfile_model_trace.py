"""Summarize observed model file access; absence is not global dependency proof."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/startup'
TRACE = BASE / 'native_api_fullfile_trace_v2_1802.log'


def model_files(source=None):
    result = []
    source = Path(source) if source is not None else ROOT / 'scripts/musetalk_install_state.py'
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {'SERVER_MODEL_FILES', 'AVATAR_PREP_MODEL_FILES', 'TRAINING_MODEL_FILES'} for t in node.targets):
            result += ast.literal_eval(node.value)
    assert len(result) == 14 and len(set(result)) == 14
    return result


def scan(lines, models, control_pid):
    rows = {name: {'path_matching_lines': 0, 'successful_completed_opens': 0,
                   'literal_basename_matching_lines': 0,
                   'control_pid_completed_opens': 0, 'other_pid_completed_opens': 0,
                   'matching_unfinished_lines': 0} for name in models}
    total = 0
    for line in lines:
        total += 1
        for name in models:
            if name.rsplit('/', 1)[-1] in line:
                rows[name]['literal_basename_matching_lines'] += 1
        match = re.match(r'\s*(\d+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\(', line)
        if not match:
            continue
        pid, syscall = int(match[1]), match[2]
        paths = re.findall(r'"([^"\n]+)"', line)
        for name in models:
            if not any(path == name or path.endswith('/' + name)
                       for path in paths):
                continue
            row = rows[name]
            row['path_matching_lines'] += 1
            if '<unfinished ...>' in line:
                row['matching_unfinished_lines'] += 1
            if syscall in {'open', 'openat', 'openat2'} and re.search(r'\)\s+=\s+[0-9]+(?:\s|$)', line):
                row['successful_completed_opens'] += 1
                field = 'control_pid_completed_opens' if pid == control_pid else 'other_pid_completed_opens'
                row[field] += 1
    return total, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    output = parser.parse_args().out
    assert output.parent.resolve() == BASE and not output.exists() and not output.is_symlink()
    control = json.loads(TRACE.with_suffix('.control.json').read_text())
    started = json.loads(TRACE.with_suffix('.start.json').read_text())
    speech = json.loads((BASE / 'fullfile_trace_v2_speech_probe_1808.json').read_text())
    prep = json.loads((BASE / 'fullfile_trace_v2_avatar_prepare_reconcile_1821.json').read_text())
    stopped = json.loads((BASE / 'fullfile_trace_v2_stop_1820.json').read_text())
    assert stopped['api_process_gone'] and stopped['api_port_closed']
    assert started['path_filters_used'] is False and control['read_bytes'] == 1
    with TRACE.open(errors='replace') as handle:
        count, rows = scan(handle, model_files(), control['pid'])
    assert rows['models/musetalkV15/unet.pth']['control_pid_completed_opens'] >= 1
    digest = hashlib.sha256()
    with TRACE.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    data = {'schema': 'observed_fullfile_model_access_v1', 'status': 'PASS_VALIDATED_OBSERVATION_SCOPE_ONLY',
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'trace_bytes': TRACE.stat().st_size,
            'trace_sha256': digest.hexdigest(), 'trace_lines': count, 'positive_control_pid': control['pid'],
            'positive_control_observed': True, 'all_child_file_syscalls_requested': True,
            'path_filters_used': False, 'models': rows, 'speech_probe_status': speech['status'],
            'fresh_prep_reconciliation_status': prep['status'],
            'scored_timing': False, 'release_ready': False, 'safe_asset_removal_established': False,
            'limitations': ['Only startup, one cached human-speech stream and one fresh canonical avatar preparation were exercised.',
                'All weights were present. Missing-weight boot was not tested. TTS was disabled in this diagnostic profile.',
                'Completed-open counts exclude unfinished/resumed pairs; path-matching line counts include unfinished attempts.',
                'Relative bare filenames cannot be resolved without per-thread cwd tracking; model paths shown use explicit paths.',
                'Literal basename counts include unresolved paths and are ambiguous for shared names such as config.json.',
                'Control PID ran the bootstrap before exec; its opens are not attributed automatically to the API.',
                'No per-operation boot duration or fresh-instance/EC2/TURN readiness claim.']}
    with output.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': data['status'], 'trace_bytes': data['trace_bytes'], 'trace_lines': count,
                      'syncnet': rows['models/syncnet/latentsync_syncnet.pt'], 'safe_asset_removal_established': False}))


if __name__ == '__main__':
    main()
