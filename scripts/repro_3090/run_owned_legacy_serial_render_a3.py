"""Default-off reporting adapter for the frozen A3 serial worker, not a renderer.

The checked renderer and historical worker bytes remain unchanged. Only two
diagnostic keys in an already-completed repdone message are added. Original
frame counts, timestamps, timing values, hashes and all processing are retained.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

ROOT = Path('/workspace/MuseTalk')
HOST = '1e7c09cffcb3'
DEADLINE = dt.datetime(2026, 10, 9, 2, 45, tzinfo=dt.timezone.utc)
RENDERER_SHA = 'df4e290b33d752be82d6d2ab738bc5d3e21aa439ffd05f1c3c852a8af1fd4a29'
WORKER_SHA = '2e6e88fbe106ca964b2b5e31af44203ce5dff201eb2c37436726b65520cc447b'
REPORT_FIELDS = {'tracking_overlap': False, 'timing_semantics': 'serial tracking then composition'}
BOUND_WATCH_SHA = 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9'


def owned_binding(path=None, digest=None):
    if bool(path) != bool(digest):
        raise ValueError('owned descriptor and exact hash required together')
    if path is None:
        return None, HOST, DEADLINE
    source = ROOT / 'scripts/repro_3090/watch_owned_single_leaf_target.py'
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != BOUND_WATCH_SHA:
        raise ValueError('checked allocation binding changed')
    sys.path.insert(0, str(source.parent))
    import watch_owned_single_leaf_target as bound
    target, deadline = bound.binding(path, digest)
    return target, target['worker_hostname'], deadline


def serial_reporting(messages):
    """Copy message containers only; preserve original payload values by identity."""
    if not isinstance(messages, dict):
        raise ValueError('repdone mapping required')
    result = {}
    for stream, message in messages.items():
        if (type(stream) is not int or not isinstance(message, tuple) or len(message) != 4
                or message[0] != 'repdone' or type(message[1]) is not int
                or message[1] != stream or type(message[2]) is not int
                or not isinstance(message[3], dict)):
            raise ValueError('historical repdone schema mismatch')
        stats = message[3]
        if any(key in stats for key in REPORT_FIELDS):
            raise ValueError('historical worker reporting fields unexpectedly present')
        result[stream] = (*message[:3], {**stats, **REPORT_FIELDS})
    return result


def adapt_collector(collector):
    original = collector.wait_for
    def wait_for(self, kind, streams, timeout=600.0):
        messages = original(self, kind, streams, timeout=timeout)
        return serial_reporting(messages) if kind == 'repdone' else messages
    collector.wait_for = wait_for


def renderer_args(stage, output, label):
    if stage not in ('T', 'SUST'):
        raise ValueError('only canonical T/SUST stages supported')
    return ['--mode', 'multi', '--backend', 'stagewise16_taesdtrt', '--identities', 'all',
            '--streams', '6', '--loops', '24' if stage == 'T' else '20', '--repeats', '2' if stage == 'T' else '5',
            '--align', 'stream8', '--pack', '16', '--decode-split', '8', '--depth', '2',
            '--ring-slots', '3', '--cv2-threads', '2', '--blas-threads', '1',
            '--min-timed-s', '60', '--thermal-warmup-s', '120',
            '--out-root', str(output), '--label', label]


def load_renderer(root):
    # This stdlib-only helper verifies actual checked bytes without editing them.
    # The watch deliberately uses an inert -c bootstrap, not this script as main;
    # its sys.path therefore does not automatically contain this helper directory.
    sys.path.insert(0, str(root / 'scripts/repro_3090'))
    from watch_owned_single_leaf import checked_source
    renderer = root / 'scripts/chin_multistream_render.py'
    worker = root / 'scripts/chin_multistream/worker.py'
    checked_source(worker, WORKER_SHA)
    body = checked_source(renderer, RENDERER_SHA)
    namespace = {'__name__': '_checked_legacy_renderer', '__file__': str(renderer),
                 '__package__': None, '__spec__': None}
    exec(compile(body, str(renderer), 'exec'), namespace)
    adapt_collector(namespace['Collector'])
    return namespace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--enable', action='store_true')
    parser.add_argument('--stage', choices=('T', 'SUST'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--owned-target-json', type=Path)
    parser.add_argument('--owned-target-sha256')
    args = parser.parse_args(argv)
    if not args.enable:
        parser.error('explicit --enable required')
    target, host, deadline = owned_binding(args.owned_target_json, args.owned_target_sha256)
    if socket.gethostname() != host or not Path(os.environ.get('BOX_GUARD_LEASE_FILE', '/workspace/.gpu_lease') + '.holder').is_file():
        raise ValueError('exact owned canonical guard required')
    if (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 600:
        raise ValueError('insufficient allocation cleanup margin')
    if not args.output_dir.is_absolute() or not re.fullmatch('[a-zA-Z0-9_-]{1,100}', args.label):
        raise ValueError('absolute output and safe label required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    argv = renderer_args(args.stage, args.output_dir, args.label)
    namespace = load_renderer(ROOT)
    receipt = {'schema': 'a3_legacy_serial_reporting_adapter_v1',
               'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'renderer_sha256': RENDERER_SHA, 'worker_sha256': WORKER_SHA,
               'added_reporting_fields': REPORT_FIELDS, 'rendering_changes': False,
               'original_timing_values_changed': False, 'renderer_arguments': argv,
               'owned_target': target, 'deadline_utc': deadline.isoformat(),
               'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    with (args.output_dir / 'reporting_adapter.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    namespace['main'](argv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
