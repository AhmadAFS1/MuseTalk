#!/usr/bin/env python3
"""Opt-in seeding of locked artifacts from a base image's site-packages.

A reviewed contract may pin a seed index (path, sha256, bytes): for selected
wheels of the locked manifest, the wheel's own RECORD rows (path, sha256, size)
as built by scripts/build_dependency_seed_index.py from the pinned wheels.

When a seed site is configured, every RECORD file of each candidate artifact is
read from that directory, hashed and written to a private staging area inside
the new venv in one pass, so the bytes that are later published are exactly the
bytes that were verified. An artifact is seeded only if ALL of its files match
in size and sha256 and none is a symlink or sits under a symlinked or escaping
directory; any problem rejects that whole artifact (its staging is removed) and
it is downloaded as before. Wheels with .data/ directories are never seeded.

Seeded artifacts are excluded from acquisition, then published into the venv's
site-packages (rename from staging) after the venv exists and before the locked
installs. The dist-info gets the wheel RECORD rows plus an INSTALLER file
('musetalk-seed'); console/gui entry points get launchers in venv/bin. The
unchanged cumulative hash locks then see the exact-version distributions as
already installed (pip and uv both leave them alone and need no wheel).

Bytecode in the seed site is NOT covered by any wheel RECORD, so copying it is
a separate opt-in: by default the installer's own --compile-bytecode compiles
the verified sources.
"""
import base64
import configparser
from concurrent.futures import ThreadPoolExecutor
import csv
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import stat
import sys
import time

from dependency_artifacts import require

SCHEMA = 1
KIND = "musetalk-dependency-seed-index"
INSTALLER = "musetalk-seed"
STAGING = ".musetalk-seed-staging"
CHUNK = 1024 * 1024
MAX_INDEX_JSON = 256 * 1024**2
MAX_PYC = 64 * 1024**2
HASH = re.compile(r"sha256=[A-Za-z0-9_-]{43}")
SCRIPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DOTTED = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
ENTRY_POINT = re.compile(r"\s*(" + DOTTED + r")\s*:\s*(" + DOTTED + r")\s*(?:\[[^\]]*\])?\s*")
ENTRY_KEYS = {"name", "version", "filename", "sha256", "size_bytes", "dist_info", "has_data_dir",
              "files", "installed_bytes", "record"}
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class SeedMismatch(Exception):
    """A seed-site file cannot be proven identical to the pinned wheel file."""

    def __init__(self, reason, path=None):
        super().__init__(reason)
        self.reason = reason
        self.path = path


def record_hash(digest):
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def safe_record_path(path):
    """Relative POSIX path with no traversal, empty, NUL or backslash components."""
    require(isinstance(path, str) and path and "\x00" not in path and "\\" not in path
            and not path.startswith("/"), "Unsafe seed record path")
    parts = path.split("/")
    require(all(p not in ("", ".", "..") for p in parts), "Unsafe seed record path")
    return path


def is_data_path(path):
    return path.split("/", 1)[0].endswith(".data")


def encode_index(document):
    """Deterministic bytes: canonical JSON, gzip without name or timestamp."""
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return gzip.compress(raw, compresslevel=9, mtime=0)


def validate_index(index, manifest):
    require(isinstance(index, dict) and index.get("schema") == SCHEMA and index.get("kind") == KIND,
            "Unknown seed index format")
    require(set(index) == {"schema", "kind", "artifact_id", "artifacts"}, "Unexpected seed index fields")
    require(index["artifact_id"] == manifest["artifact_id"], "Seed index belongs to another artifact")
    rows = {a["name"]: a for a in manifest["artifacts"]}
    entries = index["artifacts"]
    require(isinstance(entries, list) and entries, "Empty seed index")
    names = [e.get("name") if isinstance(e, dict) else None for e in entries]
    require(names == sorted(set(names), key=str) and None not in names, "Seed index must be sorted and unique")
    for e in entries:
        require(set(e) == ENTRY_KEYS, "Unexpected seed entry fields")
        a = rows.get(e["name"])
        require(a is not None and all(e[k] == a[k] for k in ("version", "filename", "sha256", "size_bytes")),
                "Seed entry differs from the locked manifest: " + str(e["name"]))
        dist_info = e["dist_info"]
        require(isinstance(dist_info, str) and dist_info.endswith(".dist-info") and "/" not in dist_info
                and safe_record_path(dist_info), "Unsafe seed dist-info name")
        record = e["record"]
        require(isinstance(record, list) and record, "Empty seed record")
        seen = set()
        for row in record:
            require(isinstance(row, list) and len(row) == 3, "Malformed seed record row")
            path, digest, size = row
            safe_record_path(path)
            require(isinstance(digest, str) and HASH.fullmatch(digest) is not None, "Seed record needs sha256")
            require(type(size) is int and size >= 0, "Seed record needs a size")
            require(path not in seen, "Duplicate seed record path")
            seen.add(path)
            top = path.split("/", 1)[0]
            require(top == dist_info or not top.endswith(".dist-info"), "Seed record writes foreign metadata")
        require(dist_info + "/METADATA" in seen, "Seed record lacks METADATA")
        require(not seen & {dist_info + "/RECORD", dist_info + "/INSTALLER"}, "Seed record lists installer files")
        require(e["has_data_dir"] is any(is_data_path(p) for p in seen), "Seed .data flag differs from record")
        require(e["files"] == len(record) and e["installed_bytes"] == sum(r[2] for r in record),
                "Seed record totals differ")
    return index


def load_index(contract, manifest, repo):
    """The contract pins the index bytes; any mismatch fails closed."""
    pin = contract["seed_index"]
    repo = Path(repo)
    path = repo / pin["path"]
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, "Seed index missing or unsafe")
    require(path.stat().st_size == pin["bytes"], "Seed index size differs from contract")
    raw = path.read_bytes()
    require(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
            "Seed index digest differs from contract")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
        data = source.read(MAX_INDEX_JSON + 1)
    require(len(data) <= MAX_INDEX_JSON, "Seed index too large")
    return validate_index(json.loads(data), manifest)


class SeedSite:
    """Read-only access below one resolved directory; symlinks are never followed."""

    def __init__(self, path):
        self.root = os.path.realpath(path)
        self._dirs = {"": True}

    def _dir_ok(self, parent):
        ok = self._dirs.get(parent)
        if ok is None:
            expected = os.path.join(self.root, parent)
            ok = os.path.realpath(expected) == expected and os.path.isdir(expected)
            self._dirs[parent] = ok
        return ok

    def open(self, rel):
        if not self._dir_ok(posixpath.dirname(rel)):
            raise SeedMismatch("symlinked-escaping-or-missing-directory", rel)
        try:
            # O_NONBLOCK: a FIFO planted in the site must be rejected, not block the boot.
            fd = os.open(os.path.join(self.root, rel), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | O_CLOEXEC)
        except FileNotFoundError:
            raise SeedMismatch("missing", rel) from None
        except OSError as error:
            raise SeedMismatch("symlink" if error.errno == errno.ELOOP else "unreadable", rel) from None
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise SeedMismatch("not-a-regular-file", rel)
        return fd, info


def write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def copy_verified(site, rel, dest, size=None, digest=None, limit=None):
    """Hash exactly the bytes written to dest; dest is removed on any mismatch."""
    fd, info = site.open(rel)
    created = False
    try:
        if size is not None and info.st_size != size:
            raise SeedMismatch("size", rel)
        if limit is not None and info.st_size > limit:
            raise SeedMismatch("too-large", rel)
        out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | O_CLOEXEC,
                      0o755 if info.st_mode & 0o111 else 0o644)
        created = True
        value = hashlib.sha256()
        total = 0
        bound = size if size is not None else limit
        try:
            while True:
                try:
                    chunk = os.read(fd, CHUNK)
                except OSError:
                    raise SeedMismatch("unreadable", rel) from None
                if not chunk:
                    break
                total += len(chunk)
                if bound is not None and total > bound:
                    raise SeedMismatch("size", rel)
                value.update(chunk)
                write_all(out, chunk)
        finally:
            os.close(out)
        if size is not None and total != size:
            raise SeedMismatch("size", rel)
        actual = record_hash(value.digest())
        if digest is not None and actual != digest:
            raise SeedMismatch("hash", rel)
        os.utime(dest, ns=(info.st_atime_ns, info.st_mtime_ns))  # Keeps timestamp pycs valid.
        return actual, total
    except BaseException:
        if created:
            os.unlink(dest)
        raise
    finally:
        os.close(fd)


def entry_points(root, entry):
    rel = entry["dist_info"] + "/entry_points.txt"
    if rel not in {row[0] for row in entry["record"]}:
        return []
    parser = configparser.ConfigParser(delimiters=("=",), interpolation=None, default_section="\x00none")
    parser.optionxform = str
    try:
        parser.read_string((Path(root) / rel).read_text("utf-8"))
    except (configparser.Error, UnicodeDecodeError):
        raise SeedMismatch("entry-points", rel) from None
    scripts = []
    for section in ("console_scripts", "gui_scripts"):
        if parser.has_section(section):
            for name, value in parser[section].items():
                match = ENTRY_POINT.fullmatch(value)
                if not SCRIPT_NAME.fullmatch(name) or match is None:
                    raise SeedMismatch("entry-points", rel)
                scripts.append([name, match.group(1), match.group(2)])
    if len({s[0] for s in scripts}) != len(scripts):
        raise SeedMismatch("entry-points", rel)
    return sorted(scripts)


def stage_artifact(entry, site, staging, bytecode_tag=None):
    started = time.monotonic()
    base = {"name": entry["name"], "version": entry["version"], "files": entry["files"],
            "installed_bytes": entry["installed_bytes"], "wheel_bytes_avoided": 0}
    if entry["has_data_dir"]:
        return dict(base, state="rejected", reason="wheel-data-dir", path=None, files_verified=0,
                    seconds=time.monotonic() - started)
    root = staging / entry["name"]
    os.mkdir(root, 0o700)
    made = set()
    verified = 0
    try:
        for rel, digest, size in entry["record"]:
            dest = os.path.join(root, rel)
            parent = os.path.dirname(dest)
            if parent not in made:
                os.makedirs(parent, 0o755, exist_ok=True)
                made.add(parent)
            copy_verified(site, rel, dest, size, digest)
            verified += 1
        scripts = entry_points(root, entry)
        bytecode = []
        if bytecode_tag:
            # Opt-in only: pycs are outside every wheel RECORD, i.e. unverified code.
            listed = {row[0] for row in entry["record"]}
            for rel, _, _ in entry["record"]:
                if not rel.endswith(".py"):
                    continue
                folder, name = posixpath.split(rel)
                pyc = posixpath.join(folder, "__pycache__", name[:-3] + "." + bytecode_tag + ".pyc")
                if pyc in listed:
                    continue
                dest = os.path.join(root, pyc)
                try:
                    os.makedirs(os.path.dirname(dest), 0o755, exist_ok=True)
                    digest, size = copy_verified(site, pyc, dest, limit=MAX_PYC)
                except SeedMismatch:
                    continue
                bytecode.append([pyc, digest, size])
    except SeedMismatch as mismatch:
        shutil.rmtree(root)
        return dict(base, state="rejected", reason=mismatch.reason, path=mismatch.path,
                    files_verified=verified, seconds=time.monotonic() - started)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return dict(base, state="staged", reason=None, path=None, files_verified=verified,
                bytecode_files=len(bytecode), seconds=time.monotonic() - started,
                _root=root, _scripts=scripts, _bytecode=bytecode)


def stage(index, rows, site, venv, workers=4, bytecode=False, index_sha256=None):
    """Verify and stage every candidate of the group. Never raises for seed-site
    content problems: those artifacts are simply not seeded."""
    started = time.monotonic()
    receipt = {"state": "none-seeded", "reason": None, "site": str(site), "index_sha256": index_sha256,
               "bytecode": bool(bytecode), "candidates": 0, "seeded": [], "rejected": [],
               "bytes_avoided": 0, "files_verified": 0, "installed_bytes": 0, "artifacts": []}
    plan = {"receipt": receipt, "staged": [], "staging": None}
    real = os.path.realpath(site)
    target = os.path.realpath(venv)
    if not os.path.isdir(real):
        receipt.update(state="skipped", reason="seed-site-unavailable")
        return plan
    if real == target or real.startswith(target + os.sep):
        receipt.update(state="skipped", reason="seed-site-inside-target-venv")
        return plan
    wanted = {a["name"]: a for a in rows}
    candidates = sorted((e for e in index["artifacts"] if e["name"] in wanted),
                        key=lambda e: (-e["installed_bytes"], e["name"]))
    receipt["candidates"] = len(candidates)
    if not candidates:
        receipt.update(state="skipped", reason="no-candidates-in-group")
        return plan
    staging = Path(venv) / STAGING
    os.mkdir(staging, 0o700)
    plan["staging"] = staging
    reader = SeedSite(real)
    tag = sys.implementation.cache_tag if bytecode else None
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(candidates))),
                            thread_name_prefix="seed-verify") as pool:
        results = list(pool.map(lambda e: stage_artifact(e, reader, staging, tag), candidates))
    entries = {e["name"]: e for e in candidates}
    claimed = {}
    for result in sorted(results, key=lambda r: r["name"]):
        entry = entries[result["name"]]
        if result["state"] == "staged":
            # Two seeded wheels may share identical files (nvidia/__init__.py), never differing ones.
            conflict = next((r[0] for r in entry["record"] if claimed.get(r[0], r[1]) != r[1]), None)
            if conflict is not None:
                shutil.rmtree(result.pop("_root"))
                result.update(state="rejected", reason="path-conflict", path=conflict)
            else:
                claimed.update((r[0], r[1]) for r in entry["record"])
                result["wheel_bytes_avoided"] = entry["size_bytes"]
                plan["staged"].append(dict(entry, root=result.pop("_root"), scripts=result.pop("_scripts"),
                                           bytecode=result.pop("_bytecode")))
        for key in ("_root", "_scripts", "_bytecode"):
            result.pop(key, None)
        receipt["artifacts"].append(result)
    seeded = [r for r in receipt["artifacts"] if r["state"] == "staged"]
    for r in seeded:
        r["state"] = "seeded"
    receipt.update(state="seeded" if seeded else "none-seeded",
                   reason=None if seeded else "all-candidates-rejected",
                   seeded=[r["name"] for r in seeded],
                   rejected=[r["name"] for r in receipt["artifacts"] if r["state"] == "rejected"],
                   bytes_avoided=sum(r["wheel_bytes_avoided"] for r in seeded),
                   files_verified=sum(r["files_verified"] for r in seeded),
                   installed_bytes=sum(r["installed_bytes"] for r in seeded),
                   stage_seconds=time.monotonic() - started)
    if not seeded:
        shutil.rmtree(staging)
        plan["staging"] = None
    return plan


def without(manifest, names):
    """Acquisition view: identical manifest minus seeded artifacts."""
    names = set(names)
    return dict(manifest, artifacts=[a for a in manifest["artifacts"] if a["name"] not in names])


def site_packages(venv):
    found = sorted((Path(venv) / "lib").glob("python3.*/site-packages"))
    require(len(found) == 1, "Venv site-packages not found")
    site = found[0]
    require(site.is_dir() and not site.is_symlink() and site.resolve() == site, "Unsafe venv site-packages")
    return site


def launcher(python, module, attribute):
    """pip's console-script template; /bin/sh trampoline for long or spaced paths."""
    if len(python) > 125 or any(c.isspace() for c in python):
        head = "#!/bin/sh\n'''exec' " + shlex.quote(python) + ' "$0" "$@"\n' + "' '''\n"
    else:
        head = "#!" + python + "\n"
    return (head + "# -*- coding: utf-8 -*-\nimport re\nimport sys\nfrom " + module + " import "
            + attribute.split(".")[0] + "\nif __name__ == '__main__':\n"
            "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
            "    sys.exit(" + attribute + "())\n")


def write_new(path, data, mode):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | O_CLOEXEC, mode)
    try:
        write_all(fd, data)
    finally:
        os.close(fd)
    return record_hash(hashlib.sha256(data).digest()), len(data)


def publish(plan, venv, site=None):
    """Move staged, verified files into the venv. Collisions with anything the
    venv already holds fail closed (identical files shared by two seeded wheels
    are the only exception)."""
    started = time.monotonic()
    venv = Path(venv)
    site = Path(site) if site is not None else site_packages(venv)
    bindir = venv / "bin"
    python = str(bindir / "python")
    claimed = {}
    published = scripts_written = bytecode_files = 0
    for entry in plan["staged"]:
        root = entry["root"]
        for rel, digest, _ in entry["record"] + entry["bytecode"]:
            source = os.path.join(root, rel)
            target = site / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(target):
                require(claimed.get(rel) == digest, "Seeded file collides with an existing venv file: " + rel)
                os.unlink(source)
                continue
            os.rename(source, target)
            claimed[rel] = digest
            published += 1
        bytecode_files += len(entry["bytecode"])
        rows = [list(r) for r in entry["record"]] + [list(r) for r in entry["bytecode"]]
        dist_info = site / entry["dist_info"]
        rows.append([entry["dist_info"] + "/INSTALLER",
                     *write_new(dist_info / "INSTALLER", (INSTALLER + "\n").encode(), 0o644)])
        for name, module, attribute in entry["scripts"]:
            path = bindir / name
            require(not os.path.lexists(path), "Seeded entry point collides with an existing venv script: " + name)
            rows.append([os.path.relpath(path, site),
                         *write_new(path, launcher(python, module, attribute).encode(), 0o755)])
            scripts_written += 1
        rows.append([entry["dist_info"] + "/RECORD", "", ""])
        text = io.StringIO()
        csv.writer(text, lineterminator="\n").writerows(rows)
        write_new(dist_info / "RECORD", text.getvalue().encode(), 0o644)
    if plan.get("staging") is not None:
        shutil.rmtree(plan["staging"])
    return {"publish_seconds": time.monotonic() - started, "files_published": published,
            "scripts_written": scripts_written, "bytecode_files": bytecode_files}
