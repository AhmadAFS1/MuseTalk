#!/usr/bin/env python3
"""Observational Linux startup profiler. No installs unless invoked as `pip ...`.

The pip mode executes the real pip entrypoint with its original arguments,
adding only a private debug log and (where supported) install report. Internal
hooks are best-effort and their coverage is recorded. Never log credentials,
complete URLs, command-line environments, request headers or response bodies.
"""
import datetime as dt
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import threading
import time
import uuid
from urllib.parse import unquote, urlsplit

ROOT = Path(os.environ.get('MUSETALK_DEPENDENCY_PROFILE_DIR', '/workspace/.dependency-profile-20261015'))
LOCK = threading.RLock()
OUT = None
RUN = None


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_url(value):
    """Allowlisted host + bounded filename only; discard auth/path/query/fragment."""
    try:
        u = urlsplit(str(value))
        name = unquote(u.path.rstrip('/').rsplit('/', 1)[-1])
        if not re.fullmatch(r'[A-Za-z0-9_.+~-]{1,240}', name):
            name = '[path-redacted]'
        host = u.hostname or ''
        if not re.fullmatch(r'[A-Za-z0-9.-]{1,180}', host):
            host = '[host-redacted]'
        return {'host': host, 'file': name}
    except Exception:
        return {'host': '[unavailable]', 'file': '[unavailable]'}


def emit(kind, **fields):
    # Observability errors must not replace pip's outcome.
    try:
        row = {'kind': kind, 'run_id': RUN, 'utc': utc(), 'mono': time.monotonic(),
               'pid': os.getpid(), **fields}
        line = json.dumps(row, allow_nan=False) + '\n'
        with LOCK:
            with OUT.open('a', encoding='utf-8') as f:
                f.write(line)
    except Exception:
        pass


def label_req(value):
    name = getattr(value, 'name', None)
    req = getattr(value, 'req', None)
    if not name and req is not None:
        name = getattr(req, 'name', None)
    return str(name)[:180] if name else None


def span_wrapper(original, kind, identify=None, generator=False):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        sid = uuid.uuid4().hex
        try:
            fields = identify(args, kwargs) if identify else {}
        except Exception:
            fields = {}
        started = time.monotonic()
        emit(kind + '_start', span_id=sid, **fields)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            emit(kind + '_end', span_id=sid, seconds=time.monotonic()-started,
                 outcome=type(error).__name__, **fields)
            raise
        if generator:
            def consume():
                outcome = 'ok'
                try:
                    yield from result
                except BaseException as error:
                    outcome = type(error).__name__
                    raise
                finally:
                    emit(kind + '_end', span_id=sid, seconds=time.monotonic()-started,
                         outcome=outcome, **fields)
            return consume()
        if kind=='request':
            fields.update(status=getattr(result,'status_code',None),
                          from_cache=bool(getattr(result,'from_cache',False)))
        emit(kind + '_end', span_id=sid, seconds=time.monotonic()-started,
             outcome='ok', **fields)
        return result
    return wrapped


def patch(module, target, kind, identify=None, generator=False):
    try:
        obj = importlib.import_module(module)
        parts = target.split('.')
        for name in parts[:-1]:
            obj = getattr(obj, name)
        name = parts[-1]
        original = getattr(obj, name)
        if not callable(original):
            raise TypeError('noncallable')
        setattr(obj, name, span_wrapper(original, kind, identify, generator))
        emit('hook', target=module+'.'+target, available=True)
    except Exception as error:
        emit('hook', target=module+'.'+target, available=False, error_type=type(error).__name__)


def hooks():
    patch('socket', 'getaddrinfo', 'dns', lambda a,k: {'host': str(a[0])[:180]})
    patch('pip._internal.network.session', 'PipSession.request', 'request',
          lambda a,k: safe_url(k.get('url', a[2] if len(a)>2 else '')))
    # Response.iter_content accounts for body consumption separately from headers.
    try:
        from pip._vendor.requests.models import Response
        original = Response.iter_content
        @functools.wraps(original)
        def iter_content(response, *args, **kwargs):
            sid, started, count, reading = uuid.uuid4().hex, time.monotonic(), 0, 0.0
            info = safe_url(response.url)
            info.update(status=response.status_code, from_cache=bool(getattr(response, 'from_cache', False)))
            emit('body_start', span_id=sid, **info)
            outcome = 'ok'
            try:
                iterator = iter(original(response, *args, **kwargs))
                while True:
                    before = time.monotonic()
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        reading += time.monotonic()-before
                        break
                    reading += time.monotonic()-before
                    count += len(chunk)
                    yield chunk
            except BaseException as error:
                outcome = type(error).__name__
                raise
            finally:
                emit('body_end', span_id=sid, seconds=time.monotonic()-started,
                     read_seconds=reading, bytes=count, outcome=outcome, **info)
        Response.iter_content = iter_content
        emit('hook', target='requests.Response.iter_content', available=True)
    except Exception as error:
        emit('hook', target='requests.Response.iter_content', available=False, error_type=type(error).__name__)
    req = lambda a,k: {'package': label_req(a[1] if len(a)>1 else k.get('req'))}
    patch('pip._internal.resolution.resolvelib.resolver', 'Resolver.resolve', 'resolve')
    patch('pip._internal.operations.prepare', 'RequirementPreparer.prepare_linked_requirement', 'prepare', req)
    patch('pip._internal.operations.prepare', 'RequirementPreparer.prepare_linked_requirements_more', 'prepare_more')
    patch('pip._internal.network.download', 'Downloader.__call__', 'download',
          lambda a,k: safe_url(getattr(a[1], 'url', '')))
    patch('pip._internal.network.download', 'BatchDownloader.__call__', 'batch_download', generator=True)
    patch('pip._internal.operations.install.wheel', 'install_wheel', 'wheel_install',
          lambda a,k: {'package': str(k.get('name', a[0] if a else ''))[:180]})
    # req_install imports this function by value on some pip versions.
    patch('pip._internal.req.req_install', 'install_wheel', 'wheel_install',
          lambda a,k: {'package': str(k.get('name', a[0] if a else ''))[:180]})
    patch('pip._internal.req.req_install', 'InstallRequirement.install', 'requirement_install',
          lambda a,k: {'package': label_req(a[0])})
    patch('pip._internal.wheel_builder', '_build_one', 'wheel_build',
          lambda a,k: {'package': label_req(a[0] if a else k.get('req'))})
    patch('pip._internal.distributions.sdist', 'SourceDistribution.prepare_distribution_metadata', 'sdist_metadata',
          lambda a,k: {'package': label_req(getattr(a[0], 'req', None))})


def pip_main(args):
    global OUT, RUN
    os.umask(0o077)
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    RUN = uuid.uuid4().hex
    OUT = ROOT / ('pip-'+RUN+'.jsonl')
    group = 'other'
    joined = ' '.join(args)
    for needle, name in [('server.in','server'),('kokoro.in','kokoro'),('mmcv','mmcv'),
                         ('mmengine','avatar-prep'),('chumpy','chumpy'),('pip==','bootstrap')]:
        if needle in joined:
            group = name
            break
    import pip
    emit('pip_start', group=group, pip_version=pip.__version__, executable=sys.executable,
         arguments_sha256=hashlib.sha256(json.dumps(args).encode()).hexdigest(),
         caches_preserved=True, indexes_and_pins_unchanged=True)
    hooks()
    argv = ['pip', '--log', str(ROOT/('pip-'+RUN+'.debug.log')), *args]
    version = tuple(int(x) for x in re.findall(r'\d+', pip.__version__)[:2])
    if 'install' in args and version >= (22,2) and '--report' not in args:
        argv.extend(['--report', str(ROOT/('pip-'+RUN+'.report.json'))])
    sys.argv = argv
    started, status = time.monotonic(), 0
    try:
        runpy.run_module('pip', run_name='__main__')
    except SystemExit as error:
        status = error.code or 0
        raise
    except BaseException as error:
        status = type(error).__name__
        raise
    finally:
        emit('pip_end', group=group, seconds=time.monotonic()-started, exit_status=status)


def observe(deadline):
    """Container/system counters only. No process argv or environments."""
    global OUT, RUN
    os.umask(0o077)
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    OUT, RUN = ROOT/'resources.jsonl', 'observer'
    emit('observer_start', deadline_epoch=deadline, sampling_seconds=2)
    while time.time() < deadline:
        fields = {}
        for name, path in [('cpu','/proc/stat'), ('memory','/proc/meminfo'),
                           ('network','/proc/net/dev'), ('disk','/proc/diskstats'),
                           ('load','/proc/loadavg'), ('pressure_io','/proc/pressure/io')]:
            try:
                fields[name] = Path(path).read_text()[:80000]
            except OSError:
                pass
        processes = []
        for d in Path('/proc').iterdir():
            if not d.name.isdigit():
                continue
            try:
                stat = (d/'stat').read_text()
                comm = stat[stat.index('(')+1:stat.rindex(')')]
                if not any(s in comm.lower() for s in ('python','pip','gcc','cc1','ninja','git','apt','dpkg','curl','wget','cmake','ld')):
                    continue
                tail = stat[stat.rindex(')')+2:].split()
                io = (d/'io').read_text()
                processes.append({'pid':int(d.name),'name':comm,'ppid':int(tail[1]),
                                  'utime_ticks':int(tail[11]),'stime_ticks':int(tail[12]),
                                  'rss_pages':int(tail[21]),'io':io})
            except (OSError, ValueError, IndexError):
                continue
        fields['processes'] = processes
        for name in ('cpu.stat','memory.current','memory.max','cpu.max','io.stat'):
            try:
                fields['cgroup_'+name] = (Path('/sys/fs/cgroup')/name).read_text()[:20000]
            except OSError:
                pass
        emit('resource_sample', **fields)
        time.sleep(2)
    emit('observer_end')


def stream():
    for line in sys.stdin:
        print('[profile '+utc()+'] '+line, end='', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'pip':
        pip_main(sys.argv[2:])
    elif mode == 'observe':
        observe(float(sys.argv[2]))
    elif mode == 'stream':
        stream()
    else:
        raise SystemExit('Unknown profile mode')
