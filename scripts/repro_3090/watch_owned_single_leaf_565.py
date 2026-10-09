"""Default-off, explicit driver-565.77 binding of the reviewed ownership watch.

The historical monitor remains byte-identical. Exactly its driver comparison
is adapted after authenticating the entire source. Both parent and CUDA leaf
apply this same adaptation; every NVML/context/fork/deadline check is retained.
This is a new harness version, not permission to reuse an old scored report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import types

WATCH_SHA = '2f03d2761e6319e27a58917099722bf279ce5d695e4682eaa5637de9d4db737f'
BINDING_SHA = 'e4e429e0dd2a42cbc1b5791a26eb8361a2c0dcd5fa6c5af320c7bcf8ffa0d4f9'
BUILDER_SHA = '1f0c0c54f2d94dc7358a7b79f4e0176dfdb747b78140552fad35c34d32917d49'
OPT3_CHILD_SHA = '33dfc108cbde9652782598e260a775ff5bdcf4eb7b244cf2b9aa861da0d0f6d8'
DRIVER = '565.77'
OLD_COMPARISON = b"require(version.value == b'595.91.07', 'unexpected_nvml_driver')"
NEW_COMPARISON = b"require(version.value == b'565.77', 'unexpected_nvml_driver')"


def source(path, expected):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute nonsymlink source required')
    raw = path.read_bytes()
    if not 0 < len(raw) <= 1 << 20 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('reviewed source changed')
    return raw


def adapted_source(raw):
    if hashlib.sha256(raw).hexdigest() != WATCH_SHA or raw.count(OLD_COMPARISON) != 1:
        raise ValueError('reviewed watch or unique driver comparison changed')
    return raw.replace(OLD_COMPARISON, NEW_COMPARISON)


def module_at(name, raw, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def reviewed_binding():
    path = Path(__file__).resolve().with_name('watch_owned_single_leaf_target.py')
    return module_at('_checked_original_binding', source(path, BINDING_SHA), path)


def reviewed_watch():
    path = Path(__file__).resolve().with_name('watch_owned_single_leaf.py')
    raw = adapted_source(source(path, WATCH_SHA))
    # Parent's source/child bootstrap must authenticate THIS wrapper, otherwise
    # the leaf would silently revert to the historical driver's comparison.
    return module_at('_checked_565_watch', raw, Path(__file__).absolute())


def configure(watch, document):
    watch.HOST = document['worker_hostname']
    watch.UUID = document['gpu_uuid']
    watch.DEADLINE = document['deadline_utc'].replace('+00:00', 'Z')
    # New builder admission remains exact-source-pinned, not an arbitrary name.
    watch.CANONICAL_TARGETS = {
        **watch.CANONICAL_TARGETS, 'build_unet_stagewise.py': BUILDER_SHA,
        'build_owned_unet_opt3.py': OPT3_CHILD_SHA}


def identity(document):
    if socket.gethostname() != document['worker_hostname']:
        raise ValueError('wrong owned hostname')
    fields = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,uuid,compute_cap,driver_version',
         '--format=csv,noheader'], text=True, timeout=10).strip().split(',')
    if [v.strip() for v in fields] != [
            'NVIDIA GeForce RTX 3090', document['gpu_uuid'], '8.6', DRIVER]:
        raise ValueError('owned GPU or explicitly bound driver mismatch')


def leaf_with_binding(spec, document):
    # The document is embedded by the authenticated parent bootstrap, not
    # recovered from a mutable environment or a first-seen NVML process.
    identity(document)
    if spec['uuid'] != document['gpu_uuid']:
        raise ValueError('leaf UUID differs from parent binding')
    watch = reviewed_watch()
    configure(watch, document)
    return watch.leaf(spec)


def bootstrap(document):
    encoded = repr(json.dumps(document, sort_keys=True, separators=(',', ':')))
    return (
        'import hashlib,sys,types,json\n'
        'p,h,s=sys.argv[1:]\n'
        'with open(p,"rb") as f: b=f.read(1<<20)\n'
        'assert hashlib.sha256(b).hexdigest()==h\n'
        'm=types.ModuleType("_owned_565_checked");m.__file__=p\n'
        'exec(compile(b,p,"exec"),m.__dict__)\n'
        'm.leaf_with_binding(json.loads(s),json.loads(' + encoded + '))\n'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--enable', action='store_true')
    parser.add_argument('--owned-target-json', required=True)
    parser.add_argument('--owned-target-sha256', required=True)
    parser.add_argument('watch_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.enable or args.watch_arguments[:1] != ['--']:
        raise ValueError('explicit enable and separator required')
    document, deadline = reviewed_binding().binding(
        args.owned_target_json, args.owned_target_sha256)
    identity(document)
    watch = reviewed_watch()
    configure(watch, document)
    watch.DEADLINE = deadline.strftime('%Y-%m-%dT%H:%M:%SZ')
    watch.BOOTSTRAP = bootstrap(document)
    return watch.main(args.watch_arguments[1:])


if __name__ == '__main__':
    raise SystemExit(main())
