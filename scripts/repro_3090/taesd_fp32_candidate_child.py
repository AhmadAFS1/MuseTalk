"""Explicit single-child candidate routing, never a default loader or startup hook.

Run only under the canonical caller's owned-worker guard/watch. This launcher
does not prove frozen-input identity, numerical acceptance, engine precision,
throughput, or release readiness. Importing it is stdlib-only and inert.

The target's verified bytes execute in their own __main__ namespace while the
real sys.modules['__main__'] remains this inert launcher. Thus spawn workers
re-import the launcher, not the render target; the target's worker.main is an
importable module function, never a function pickled from its __main__ namespace.
Only the exact VFD import is intercepted, inside a restoring context manager.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
VFD_NAME = 'scripts.vae_fast_decoder'
VFD_SHA = 'aea1eb059085c1799c6162d516d028d385be107c36da8a105fa1f838eb5e9882'
RUNTIME_SHA = 'e348ceb6cc5ca8b3255716da07ca88c04bf46b4a499f04a66f8a7bc540cfc31b'
TARGETS = {
    'gate': ('scripts/repro_400fps/gate_taesd_trt.py', '3baf8976e4fb25a908809e68d6ac126c07ea525baed7475a6443263221e28e99'),
    'render': ('scripts/chin_multistream_render.py', 'df4e290b33d752be82d6d2ab738bc5d3e21aa439ffd05f1c3c852a8af1fd4a29'),
}
STRICT_ENV = {'MUSETALK_VAE_BACKEND': 'taesd', 'MUSETALK_TAESD_BACKEND': 'trt',
              'MUSETALK_TAESD_TRT_STRICT': '1', 'MUSETALK_TAESD_TRT_BUILD': '0',
              'MUSETALK_TRT_FALLBACK': '0', 'MUSETALK_TAESD_TRT_FUSED_POST': '1'}


class RoutingRejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RoutingRejected(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute_nonsymlink_path_required')
    require('..' not in path.parts, 'parent_traversal_forbidden')
    return path


def checked(path, expected, limit=2 << 20):
    path = safe_path(path)
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'explicit_sha256_required')
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, 'invalid_file_type_or_size')
        raw = handle.read(limit + 1)
    require(len(raw) == info.st_size and digest(raw) == expected, 'pinned_bytes_mismatch')
    return raw


def verified_runtime():
    path = ROOT / 'scripts/repro_3090/taesd_fp32_candidate_runtime.py'
    raw = checked(path, RUNTIME_SHA)
    module = types.ModuleType('_candidate_child_verified_runtime')
    module.__file__, module.__package__ = str(path), ''
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


class StrictParser(argparse.ArgumentParser):
    def error(self, message):
        raise RoutingRejected('unsupported_or_incomplete_arguments')


def validate_target_args(target, argv, output_dir):
    """No arbitrary flags, abbreviations, duplicate options, or reduced workloads."""
    require(target in TARGETS, 'unknown_target')
    tokens = [token.split('=', 1)[0] for token in argv if token.startswith('--')]
    require(len(tokens) == len(set(tokens)), 'duplicate_target_option')
    ap = StrictParser(add_help=False, allow_abbrev=False)
    if target == 'gate':
        ap.add_argument('--no-record', action='store_true', required=True)
        ap.add_argument('--corpus', required=True)
        args = ap.parse_args(argv)
        corpus = safe_path(args.corpus)
        require(corpus.is_dir(), 'corpus_unavailable')
        main_files = sorted(corpus.glob('unet_io_*.pt'))
        holdout_files = sorted((corpus / 'holdout').glob('unet_io_*.pt'))
        require(len(main_files) == 352 and len(holdout_files) == 96, 'full_448_file_corpus_required')
        files = main_files + holdout_files
        require(all(safe_path(p).is_file() and p.stat().st_size > 0 for p in files), 'invalid_corpus_file')
        require(not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())), 'fresh_gate_output_required')
        return {'corpus_path': str(corpus), 'files_expected': 448, 'main_files_expected': 352,
                'holdout_files_expected': 96, 'frames_expected': 3584}
    for name, default in (('mode', 'multi'), ('backend', 'stagewise16_taesdtrt'), ('identities', 'all'), ('align', 'stream8')):
        ap.add_argument('--' + name, default=default)
    for name, default in (('streams', 6), ('loops', 1), ('repeats', 1), ('pack', 16), ('decode-split', 8),
                          ('depth', 2), ('ring-slots', 3), ('crf', 12), ('cv2-threads', 2), ('blas-threads', 1)):
        ap.add_argument('--' + name, type=int, default=default)
    for name in ('save-arrays', 'encode', 'compare-accepted'):
        ap.add_argument('--' + name, action='store_true', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args(argv)
    expected = {'mode': 'multi', 'backend': 'stagewise16_taesdtrt', 'identities': 'all', 'align': 'stream8',
                'streams': 6, 'loops': 1, 'repeats': 1, 'pack': 16, 'decode_split': 8, 'depth': 2,
                'ring_slots': 3, 'crf': 12, 'cv2_threads': 2, 'blas_threads': 1}
    require(all(getattr(args, k) == v for k, v in expected.items()), 'noncanonical_capture_arguments')
    # Require these explicitly instead of accepting potentially changing defaults.
    require({'--backend', '--streams', '--loops', '--repeats'} <= set(tokens), 'explicit_capture_workload_required')
    require(safe_path(args.out_root) == output_dir, 'capture_output_mismatch')
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,119}', args.label), 'invalid_capture_label')
    require(not (output_dir / args.label).exists() and not (output_dir / (args.label + '.json')).exists(), 'fresh_capture_output_required')
    return {'streams_expected': 6, 'frames_per_stream_expected': 240, 'label': args.label}


@contextmanager
def scoped_candidate(runtime, manifest, manifest_sha, expected, vfd_bytes, state):
    """Lazy exact-name import: no torch/VFD import before render's worker spawn."""
    require(VFD_NAME not in sys.modules and 'vae_fast_decoder' not in sys.modules, 'fresh_child_vfd_required')
    parent_before = sys.modules.get('scripts')
    missing = object()
    parent_attr = getattr(parent_before, 'vae_fast_decoder', missing)
    originals = []

    class CandidateImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == VFD_NAME:
                require(not originals, 'vfd_reload_forbidden')
                return importlib.util.spec_from_loader(fullname, self, origin=str(ROOT / 'scripts/vae_fast_decoder.py'))
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__file__ = str(ROOT / 'scripts/vae_fast_decoder.py')
            exec(compile(vfd_bytes, module.__file__, 'exec'), module.__dict__)
            original = module.load_taesd_trt_backend
            originals.append((module, original))

            def candidate(device, runtime_dtype=None, *, model=None):
                state['candidate_attempts'] += 1
                require(str(device) == 'cuda:0' and isinstance(device, module.torch.device), 'exact_cuda0_device_required')
                require(runtime_dtype is module.torch.float16 and model is not None, 'exact_fp16_and_reference_model_required')
                require(all(os.environ.get(k) == v for k, v in STRICT_ENV.items()), 'strict_no_fallback_environment_required')
                backend = runtime.load_candidate(manifest_path=manifest, expected_manifest_sha256=manifest_sha,
                                                 device=device, model=model, enable=True)
                require(getattr(backend, 'name', None) == 'taesd_trt', 'unexpected_backend_name')
                require(all(backend.meta.get(k) == expected[k] for k in
                            ('key', 'decoder_plan_sha256', 'post_plan_sha256')), 'returned_backend_identity_mismatch')
                require(Path(backend.paths['meta']) == manifest, 'returned_manifest_path_mismatch')
                state['candidate_calls'] += 1
                state['invocations'].append({'candidate_key': expected['key'], 'decoder_plan_sha256': expected['decoder_plan_sha256'],
                    'post_plan_sha256': expected['post_plan_sha256'], 'device': 'cuda:0', 'dtype': 'torch.float16',
                    'reference_model_supplied': True, 'manifest_sha256': manifest_sha})
                return backend

            module.load_taesd_trt_backend = candidate

    hook = CandidateImport()
    sys.meta_path.insert(0, hook)
    try:
        yield
    finally:
        sys.meta_path[:] = [item for item in sys.meta_path if item is not hook]
        for module, original in originals:
            module.load_taesd_trt_backend = original
        sys.modules.pop(VFD_NAME, None)
        parent = sys.modules.get('scripts')
        if parent is not None:
            if parent_attr is missing:
                parent.__dict__.pop('vae_fast_decoder', None)
            else:
                parent.vae_fast_decoder = parent_attr


@contextmanager
def process_context(target_path, argv, output_dir, target):
    old_argv, old_path, old_env, old_cwd = sys.argv, list(sys.path), dict(os.environ), os.getcwd()
    try:
        sys.argv = [str(target_path), *argv]
        sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
        os.chdir(ROOT)
        os.environ.update(STRICT_ENV)
        if target == 'gate':
            os.environ['REPRO_GATE_OUT'] = str(output_dir)
        yield
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        os.environ.clear()
        os.environ.update(old_env)
        os.chdir(old_cwd)


def execute_target(raw, path):
    # Deliberately do not replace sys.modules['__main__']: see spawn contract above.
    namespace = {'__name__': '__main__', '__file__': str(path), '__package__': None,
                 '__spec__': None, '__builtins__': __builtins__}
    exec(compile(raw, str(path), 'exec'), namespace)


def launch(*, target, target_args, manifest_path, manifest_sha256, receipt_path, output_dir, enable=False):
    require(enable is True, 'explicit_enable_required')
    require(target in TARGETS, 'unknown_target')
    manifest, receipt, out = (safe_path(p) for p in (manifest_path, receipt_path, output_dir))
    require(receipt.parent.is_dir() and not receipt.exists(), 'fresh_receipt_required')
    require(out.parent.is_dir(), 'existing_output_parent_required')
    workload = validate_target_args(target, target_args, out)
    runtime = verified_runtime()  # pinned stdlib-only API; no ONNX/CUDA yet
    meta = runtime.parse_json(checked(manifest, manifest_sha256))
    require(meta.get('schema') == runtime.SCHEMA and meta.get('status') == 'BUILD_AND_PROBE_ONLY_QUALITY_UNTESTED', 'candidate_manifest_required')
    require(re.fullmatch('[0-9a-f]{20}', meta.get('key', '')), 'invalid_candidate_key')
    require(all(runtime.valid_digest(meta.get(k)) for k in ('decoder_plan_sha256', 'post_plan_sha256')), 'invalid_candidate_plan_hash')
    require(manifest.name == f"taesd_trt_{meta['key']}.json", 'candidate_manifest_name_mismatch')
    target_path = ROOT / TARGETS[target][0]
    raw = checked(target_path, TARGETS[target][1])
    vfd_bytes = checked(ROOT / 'scripts/vae_fast_decoder.py', VFD_SHA)
    state = {'schema': 'taesd_fp32_candidate_child_v1', 'status': 'invalid', 'returncode': 2,
        'target_returncode': None, 'target': target, 'manifest_sha256': manifest_sha256,
        'candidate_key': meta['key'], 'decoder_plan_sha256': meta['decoder_plan_sha256'],
        'post_plan_sha256': meta['post_plan_sha256'], 'candidate_calls': 0, 'candidate_attempts': 0,
        'invocations': [], 'workload': workload, 'target_sha256': TARGETS[target][1],
        'canonical_vfd_sha256': VFD_SHA, 'runtime_helper_sha256': RUNTIME_SHA,
        'quality_accepted': False, 'performance_accepted': False, 'release_ready': False,
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w') as handle:
        # An interrupted/killed run leaves INVALID, never an empty/optimistic PASS.
        json.dump(state, handle, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
        error = None
        try:
            with process_context(target_path, target_args, out, target), \
                 scoped_candidate(runtime, manifest, manifest_sha256, meta, vfd_bytes, state):
                try:
                    execute_target(raw, target_path)
                    state['target_returncode'] = 0
                except SystemExit as exc:
                    state['target_returncode'] = exc.code if type(exc.code) is int else (0 if exc.code is None else 1)
            require(state['candidate_calls'] >= 1, 'candidate_never_invoked')
            require(state['candidate_calls'] == state['candidate_attempts'], 'candidate_invocation_failed')
            state['returncode'] = state['target_returncode']
            state['status'] = 'complete' if state['returncode'] == 0 else 'failed'
        except BaseException as exc:
            error = exc
            state['status'] = 'invalid' if isinstance(exc, RoutingRejected) or not state['candidate_calls'] else 'failed'
            state['returncode'] = 2 if state['status'] == 'invalid' else (130 if isinstance(exc, KeyboardInterrupt) else 1)
            state['error_type'] = type(exc).__name__  # no exception text, model data, or environment values
        finally:
            state['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            handle.seek(0); json.dump(state, handle, sort_keys=True, indent=2); handle.write('\n')
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        if error is not None:
            if state['status'] == 'invalid' and not isinstance(error, RoutingRejected):
                raise RoutingRejected('candidate_child_invalid') from None
            raise error
    return state['returncode']


def main(argv=None):
    ap = StrictParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--enable', action='store_true')
    ap.add_argument('--target', choices=sorted(TARGETS), required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('target_args', nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    require(args.target_args and args.target_args[0] == '--', 'explicit_target_argument_separator_required')
    return launch(target=args.target, target_args=args.target_args[1:], manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256, receipt_path=args.receipt, output_dir=args.output_dir, enable=args.enable)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RoutingRejected:
        print('candidate child INVALID: explicit routing contract rejected', file=sys.stderr)
        raise SystemExit(2)
