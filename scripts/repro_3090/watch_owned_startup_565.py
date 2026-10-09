"""Exact-source startup-only extension of the strict owned 565 monitor.

The complete original monitor and allocation binder remain unchanged. Only
one new SHA-pinned CUDA leaf is admitted; no PID/context/deadline relaxation.
"""
from pathlib import Path
import types

BASE_SHA = '72d89c1917c15600560d64eec9c6a304bbe7f0ee4ec8864beee85eebe08b4606'
CHILD_SHA = '8c9d6274afbe8aaeb8356edd4006ef6c8cf7835d2c7c6f5816383ebeda78678c'


def base():
    import hashlib
    path = Path(__file__).resolve().with_name('watch_owned_single_leaf_565.py')
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('reviewed complete 565 monitor changed')
    m = types.ModuleType('_checked_startup_565'); m.__file__ = str(Path(__file__).absolute())
    exec(compile(raw,str(path),'exec'),m.__dict__)
    original = m.configure
    def configure(watch,document):
        original(watch,document)
        watch.CANONICAL_TARGETS = {**watch.CANONICAL_TARGETS,'startup_model_probe.py':CHILD_SHA}
    m.configure = configure
    return m


def leaf_with_binding(spec,document):
    return base().leaf_with_binding(spec,document)


def main(argv=None):
    return base().main(argv)


if __name__=='__main__': raise SystemExit(main())
