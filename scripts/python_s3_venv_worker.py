#!/usr/bin/env python3
"""Bounded, test-worker-only dependency archive/restore experiment (not installer integration).

Use system Python on the owned Linux RTX 3090. Config arrives over SSH stdin,
never in command-line arguments. No AWS credentials are part of the archive.
"""
import hashlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LABEL = "musetalk-python-s3-venv-20261011-a3"
ROOT = Path("/workspace") / LABEL
REPO = Path("/workspace/MuseTalk")
VENV = Path("/workspace/.venvs/musetalk_trt_stagewise")
BACKUP = VENV.with_name(VENV.name + ".s3-rollback-20261011-a3")
ARCHIVE = ROOT / "built-venv.tar.zst"
MANIFEST = ROOT / "manifest.json"
PREFIX = "/python-venv/20261011/" + LABEL + "/"
DEADLINE = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def lease_remaining():
    return (DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds() if DEADLINE else float("inf")


def save(name, value):
    p = ROOT / name
    require(p.parent == ROOT, "Evidence must stay in the experiment directory")
    p.write_text(json.dumps(value, indent=2) + "\n")
    os.chmod(p, 0o600)


def emit(value):
    try:
        print(json.dumps(value), flush=True)
    except BrokenPipeError:
        # Preserve durable evidence when the SSH transport disappears. The
        # operation remains bounded by the original lease and command timeouts.
        sys.stdout = open(os.devnull, 'w')


def event(stage, **fields):
    value = {"stage": stage, "worker_monotonic": time.monotonic(), **fields}
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(value) + "\n")
    emit(value)


def command(args, *, timeout=120, env=None, input=None, log=None):
    require(lease_remaining() > 20, "Worker lease nearly expired")
    timeout = min(timeout, lease_remaining() - 10)
    started = time.monotonic()
    event("COMMAND_STARTED", executable=Path(args[0]).name, log=log)
    try:
        r = subprocess.run(args, cwd=REPO, env=env, input=input,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        event("COMMAND_TIMEOUT", executable=Path(args[0]).name, log=log,
              seconds=round(time.monotonic()-started, 3))
        raise
    if log:
        # Tool logs must never contain signed requests or credential values.
        (ROOT / log).write_bytes(r.stdout + r.stderr)
    event("COMMAND_COMPLETED", executable=Path(args[0]).name, log=log,
          returncode=r.returncode, seconds=round(time.monotonic()-started, 3))
    require(r.returncode == 0, "Command failed: " + Path(args[0]).name)
    return r.stdout


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def package_facts():
    code = '''import importlib.metadata as m,json,sys
p={d.metadata['Name'].lower():d.version for d in m.distributions()}
editable=[]
for d in m.distributions():
 s=d.read_text('direct_url.json')
 if s and json.loads(s).get('dir_info',{}).get('editable'):editable.append(d.metadata['Name'])
print(json.dumps({'packages':p,'python':sys.version,'base_prefix':sys.base_prefix,
 'base_executable':sys._base_executable,'editable_packages':editable}))'''
    return json.loads(command([str(VENV / "bin/python"), "-B", "-c", code]))


def validate_native(env=None):
    start = time.monotonic()
    command(["bash", "scripts/install_musetalk.sh", "--check", "--check-imports",
             "--with-kokoro", "--with-avatar-prep", "--with-native-vp8",
             "--matrix", "cu121", "--report", str(ROOT / "install-check.json")],
            timeout=120, log="install-check.log", env=env)
    command([str(VENV / "bin/python"), "-B", "scripts/musetalk_selftest.py",
             "--repo-root", str(REPO), "--out", str(ROOT / "gpu-selftest.json"),
             "--buckets", "8", "--skip-compile", "--iters", "5"],
            timeout=180, log="gpu-selftest.log", env=env)
    result = json.loads((ROOT / "gpu-selftest.json").read_text())
    require(result.get("cuda_ok") is True and result.get("trt_import_ok") is True,
            "Restored CUDA/TensorRT validation failed")
    event("VALIDATION_PASS", seconds=round(time.monotonic()-start, 3),
          cuda_ok=True, trt_import_ok=True, pip_install_executed=False,
          note="check-only installer; selftest does not torch.compile or build TensorRT plans")


def check_signed_url(url):
    p = urllib.parse.urlsplit(url)
    hosts = {"lingua-musetalk-s3-storage.s3.amazonaws.com",
             "lingua-musetalk-s3-storage.s3.us-east-1.amazonaws.com"}
    require(p.scheme == "https" and p.hostname in hosts and p.port in (None, 443)
            and not p.username and not p.password and p.path.startswith(PREFIX)
            and not p.fragment and not any(part in {".", ".."} for part in
                                          urllib.parse.unquote(p.path).split("/")),
            "Unexpected signed S3 destination")


def restore_disk_budget(free, payload, archive):
    needed = payload + archive + 2 * 1024**3
    require(free >= needed, "Insufficient free disk for fresh S3 download plus restored venv")
    return needed


def audit_member(member, base_executable):
    path = PurePosixPath(member.name)
    require(not path.is_absolute() and path.parts and path.parts[0] == VENV.name
            and ".." not in path.parts, "Unsafe archive member path")
    require(member.isdir() or member.isfile() or member.issym() or member.islnk(),
            "Unsupported archive member type")
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute():
            internal = Path(str(target)).resolve().is_relative_to(VENV)
            interpreter = (path.parent == PurePosixPath(VENV.name) / "bin"
                           and Path(str(target)).resolve() == Path(base_executable).resolve())
            require(member.issym() and (internal or interpreter), "Unsafe external archive link")
        else:
            candidate = (path.parent / target) if member.issym() else target
            normalized = []
            for part in candidate.parts:
                if part == "..":
                    require(len(normalized) > 1, "Archive link escapes environment")
                    normalized.pop()
                elif part != ".":
                    normalized.append(part)
            require(normalized and normalized[0] == VENV.name, "Archive link escapes environment")
    return path


def audit_archive(path, base_executable):
    before = time.monotonic()
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    paths, links = set(), set()
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                require(lease_remaining() > 20, "Archive audit exceeded worker lease")
                p = audit_member(member, base_executable)
                require(p not in paths, "Duplicate archive member")
                paths.add(p)
                if member.issym():
                    links.add(p)
        require(proc.wait(timeout=10) == 0, "Archive decompression failed")
        require(not any(parent in links for p in paths for parent in p.parents),
                "Archive member is nested beneath a symlink")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()
    event("ARCHIVE_SAFETY_PASS", members=len(paths), seconds=round(time.monotonic()-before, 3))


def ensure_api_stopped():
    require(not Path("/workspace/logs/musetalk/api_server_8000.pid").exists(), "API PID file remains")
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = p.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not any(a.endswith(b"/api_server.py") or a == b"api_server.py" for a in args),
                "API process remains after canonical stop")


def build(finalize=False):
    if finalize:
        require(ROOT.is_dir() and not ROOT.is_symlink() and ARCHIVE.is_file()
                and not ARCHIVE.is_symlink() and not MANIFEST.exists(), "No safe unfinished archive to finalize")
        records = [json.loads(l) for l in (ROOT / 'events.jsonl').read_text().splitlines()]
        completed = [r for r in records if r.get('stage')=='COMMAND_COMPLETED'
                     and r.get('log')=='archive-build.log' and r.get('returncode')==0]
        require(len(completed)==1, "Exactly one successful tar completion required")
        compression = completed[0]['seconds']
        ensure_api_stopped()
    else:
        require(not ROOT.exists(), "Build already attempted; never overwrite evidence")
        ROOT.mkdir(mode=0o700)
    facts = package_facts()
    require(platform.system() == "Linux" and platform.machine() == "x86_64"
            and facts["python"].startswith("3.10.") and not facts["editable_packages"],
            "Environment is not a compatible clean Linux cp310 venv")
    validate_native()
    # Stop the API before creating a reproducible snapshot. No inference jobs are active.
    runtime = worker_runtime_env()
    runtime.update(DRAIN_TIMEOUT_SECONDS="20", STARTUP_TIMEOUT_SECONDS="180")
    command(["bash", "scripts/vast_server_ctl.sh", "stop"], env=runtime,
            timeout=60, log="baseline-stop.log")
    ensure_api_stopped()
    if not subprocess.run(["which", "zstd"], capture_output=True).returncode == 0:
        command(["apt-get", "install", "-y", "zstd"], timeout=90, log="zstd-install.log")
    payload = 0
    links = []
    forbidden = {".env", ".aws", ".musetalk-runtime.env", "credentials", "id_rsa", "id_ed25519"}
    allowed_base = Path(facts["base_executable"]).resolve()
    for p in VENV.rglob("*"):
        require(p.name not in forbidden, "Secret-like filename found in venv")
        if p.is_symlink():
            resolved = p.resolve()
            require(resolved.is_relative_to(VENV) or resolved == allowed_base,
                    "Unsupported external venv symlink")
            links.append({"path": str(p.relative_to(VENV)), "target": os.readlink(p)})
        elif p.is_file():
            if p.suffix == ".pth":
                for line in p.read_text().splitlines():
                    entry = line.strip()
                    if entry and not entry.startswith(("#", "import ", "import\t")):
                        target = (p.parent / entry).resolve()
                        require(target.is_relative_to(VENV), "External .pth dependency found")
            payload += p.stat().st_size
    pins = {str(p.relative_to(REPO)): digest(p) for p in (REPO / "requirements").glob("*") if p.is_file()}
    contract = {"venv_path": str(VENV), "platform": "linux-x86_64", "matrix": "cu121",
                "facts": facts, "requirements_sha256": pins,
                "groups": {"server": True, "kokoro": True, "avatar_prep": True,
                           "legacy_int8": False, "chin_tools": False},
                "pyvenv_cfg": (VENV / "pyvenv.cfg").read_text(), "symlinks": links,
                "distro": Path("/etc/os-release").read_text(), "libc": platform.libc_ver()}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if not finalize:
        require(shutil.disk_usage(ROOT).free >= payload + 2 * 1024**3,
                "Insufficient free disk for a conservative archive-build budget")
        before = time.monotonic()
        command(["tar", "-C", str(VENV.parent), "-I", "zstd -3 -T0", "-cf", str(ARCHIVE), VENV.name],
                timeout=180, log="archive-build.log")
        compression = time.monotonic() - before
    audit_archive(ARCHIVE, allowed_base)
    before = time.monotonic()
    sha = digest(ARCHIVE)
    m = {"schema": "musetalk_python_venv_experiment_v1", "label": LABEL,
         "contract": contract, "contract_sha256": fingerprint,
         "archive_bytes": ARCHIVE.stat().st_size, "payload_file_bytes": payload,
         "archive_sha256": sha, "compression_seconds": round(compression, 3),
         "finalized_after_ssh_disconnect": finalize,
         "hash_seconds": round(time.monotonic()-before, 3),
         "source_revision": command(["git", "rev-parse", "HEAD"]).decode().strip(),
         "installer_sha256": digest(REPO / "scripts/install_musetalk.sh"),
         "external_assets_included": False, "base_image": "vastai/pytorch:cuda-12.1-auto",
         "base_image_digest": "provider image_uuid recorded separately; immutable digest not supplied"}
    save("manifest.json", m)
    event("ARCHIVE_BUILT", archive_bytes=m["archive_bytes"], payload_file_bytes=payload,
          compression_seconds=m["compression_seconds"], archive_sha256=sha,
          contract_sha256=fingerprint, package_count=len(facts["packages"]))
    emit({"result": "build", "manifest": m})


def verify_archive():
    m = json.loads(MANIFEST.read_text())
    require(m['label']==LABEL and ARCHIVE.stat().st_size==m['archive_bytes']
            and digest(ARCHIVE)==m['archive_sha256'], "Completed archive bytes differ from manifest")
    require(package_facts()==m['contract']['facts'], "Source packages changed after the archive build")
    event('COMPLETED_ARCHIVE_REVERIFIED', archive_bytes=m['archive_bytes'])
    emit({'result':'verify_archive','manifest':m})


def upload_part(c):
    require(ARCHIVE.is_file(), "Build archive missing")
    check_signed_url(c["url"])
    start = time.monotonic()
    with ARCHIVE.open("rb") as f:
        f.seek(c["offset"])
        data = f.read(c["bytes"])
    require(len(data) == c["bytes"] and len(data) <= 128 * 1024**2, "Bad multipart range")
    request = urllib.request.Request(c["url"], data=data, method="PUT")
    with urllib.request.urlopen(request, timeout=120) as r:
        require(r.status == 200, "Part upload failed")
        etag = r.headers["ETag"]
    receipt={"result": "upload_part", "part": c["part"], "etag": etag,
             "bytes": len(data), "seconds": round(time.monotonic()-start, 3)}
    save('uploaded-part-'+str(c['part'])+'.json',receipt)
    emit(receipt)
    return receipt


def upload_batch(c):
    require(1 <= len(c['parts']) <= 1000, 'Invalid batch size')
    def transfer(part):
        try:
            require(lease_remaining()>140,'Insufficient part lease')
            return upload_part(part)
        except Exception as error:
            receipt={'part':part['part'],'failed':True,'error_type':type(error).__name__}
            reason=getattr(error,'reason',None)
            receipt['reason_type']=type(reason).__name__ if reason is not None else None
            code=getattr(error,'code',None)
            receipt['http_status']=code if type(code) is int else None
            number=getattr(reason,'errno',None)
            receipt['reason_errno']=number if type(number) is int else None
            save('failed-part-'+str(part['part'])+'.json',receipt)
            return receipt
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(transfer,c['parts']))
    receipt={'result':'upload_batch','parts':results,'seconds':round(time.monotonic()-started,3)}
    save('upload-batch-result.json',receipt)
    emit(receipt)


def worker_runtime_env():
    # Copy only explicitly needed worker fields; never return/log PID-1 credentials.
    env = dict(os.environ)
    allowed = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
               "AWS_REGION", "AWS_DEFAULT_REGION", "AVATAR_S3_BUCKET", "AVATAR_S3_REGION",
               "AVATAR_S3_PREFIX", "AVATAR_S3_ENABLED", "TRT_ARTIFACT_S3_BUCKET", "TRT_ARTIFACT_S3_REGION",
               # Preserve public launch mappings so relay wrappers need not use
               # the tr|awk early-exit PID-1 fallback (SIGPIPE under pipefail).
               "VAST_TCP_PORT_1455", "VAST_UDP_PORT_3478", "PUBLIC_IPADDR"}
    for field in Path("/proc/1/environ").read_bytes().split(b"\0"):
        k, sep, v = field.partition(b"=")
        if sep and k.decode(errors="replace") in allowed:
            env[k.decode()] = v.decode()
    for k in list(env):
        if k.startswith("LINGUA_") or k in {"MUSETALK_AWS_SECRET_ID", "MUSETALK_AWS_SECRET_REGION"}:
            env.pop(k)
    env.update(REPO_ROOT=str(REPO), VENV_PATH=str(VENV), PROFILE="throughput_record", PORT="8000")
    return env


def probe():
    # Read-only launch check, run before spending time on package installation.
    print(json.dumps({"result": "probe", "python": sys.version.split()[0],
                      "gpu": "NVIDIA GeForce RTX 3090", "platform": platform.platform(),
                      "venv_exists": VENV.is_dir(), "writes_or_installs": False}), flush=True)


def diagnose():
    # Only inspect our helper's check logs, never credentials or whole environment.
    secrets = [v for k, v in worker_runtime_env().items()
               if any(word in k for word in ("SECRET", "TOKEN", "ACCESS_KEY", "PASSWORD")) and len(v) >= 8]
    def redact(text):
        import re
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r'https?://[^\s<>"\x27]+', '[URL REDACTED]', text)
        return re.sub(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]+\b', '[REDACTED]', text)
    files = {}
    for name in ("install-check.json", "install-check.log", "gpu-selftest.json", "gpu-selftest.log", "events.jsonl", "archive-build.log"):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size-16000))
                text = f.read().decode('utf-8', 'replace')
            files[name] = {"bytes": path.stat().st_size, "sanitized_tail": redact(text)}
    print(json.dumps({"result": "diagnose", "root_exists": ROOT.exists(),
                      "archive_exists": ARCHIVE.is_file(), "manifest_exists": MANIFEST.is_file(),
                      "files": files}), flush=True)


def restore(c):
    manifest = json.loads(MANIFEST.read_text())
    require(not BACKUP.exists() and VENV.is_dir() and not VENV.is_symlink(), "Unsafe rollback paths")
    require(c["sha256"] == manifest["archive_sha256"] and c["bytes"] == manifest["archive_bytes"],
            "Controller/archive manifest mismatch")
    check_signed_url(c["url"])
    free = shutil.disk_usage(ROOT).free
    needed = restore_disk_budget(free, manifest["payload_file_bytes"], c["bytes"])
    event("RESTORE_DISK_BUDGET_PASS", free_bytes=free, needed_bytes=needed)
    downloaded = ROOT / "downloaded-from-s3.tar.zst"
    stage = ROOT / "restore-stage"
    require(not downloaded.exists() and not stage.exists(), "Restore already attempted")
    runtime = worker_runtime_env()
    runtime.update(DRAIN_TIMEOUT_SECONDS="20", STARTUP_TIMEOUT_SECONDS="180")
    command(["bash", "scripts/vast_server_ctl.sh", "stop"], env=runtime,
            timeout=60, log="restore-stop.log")
    ensure_api_stopped()
    VENV.rename(BACKUP)
    event("VENV_REMOVED_FROM_WORKING_PATH", recoverable_backup=str(BACKUP), final_path_absent=not VENV.exists())
    started = time.monotonic()
    try:
        before = time.monotonic()
        with urllib.request.urlopen(c["url"], timeout=120) as r, downloaded.open("xb") as f:
            require(r.status == 200, "S3 GET failed")
            while True:
                require(lease_remaining() > 20, "S3 download exceeded worker lease")
                block = r.read(8 * 1024**2)
                if not block:
                    break
                f.write(block)
        download_seconds = time.monotonic() - before
        require(downloaded.stat().st_size == c["bytes"], "S3 GET length mismatch")
        event("S3_DOWNLOAD_COMPLETE", bytes=c["bytes"], seconds=round(download_seconds, 3),
              decimal_MB_per_second=round(c["bytes"]/download_seconds/10**6, 3), requests=1,
              local_archive_reused=False)
        before = time.monotonic()
        require(digest(downloaded) == c["sha256"], "Downloaded archive checksum mismatch")
        event("S3_CHECKSUM_PASS", seconds=round(time.monotonic()-before, 3))
        previous=[json.loads(l) for l in (ROOT/'events.jsonl').read_text().splitlines()]
        audited=any(r.get('stage')=='ARCHIVE_SAFETY_PASS' for r in previous)
        built=any(r.get('stage')=='ARCHIVE_BUILT' and r.get('archive_sha256')==c['sha256'] for r in previous)
        if audited and built:
            event('ARCHIVE_SAFETY_REUSED_FOR_IDENTICAL_BYTES', sha256=c['sha256'],
                  note='Fresh GET checksum matches this same worker\u2019s previously audited archive')
        else:
            audit_archive(downloaded, Path(manifest["contract"]["facts"]["base_executable"]).resolve())
        before = time.monotonic()
        stage.mkdir(mode=0o700)
        command(["tar", "--no-same-owner", "-I", "zstd", "-xf", str(downloaded), "-C", str(stage)],
                timeout=180, log="archive-extract.log")
        require(set(p.name for p in stage.iterdir()) == {VENV.name}, "Archive has unexpected top-level paths")
        (stage / VENV.name).rename(VENV)
        event("EXTRACTION_PUBLISHED", seconds=round(time.monotonic()-before, 3), path=str(VENV))
        before = time.monotonic()
        require(package_facts() == manifest["contract"]["facts"], "Restored packages/base interpreter differ")
        event("PACKAGE_MANIFEST_MATCH", seconds=round(time.monotonic()-before, 3))
        restore_seconds = time.monotonic() - started
        validate_native()
        before = time.monotonic()
        command(["bash", "scripts/vast_server_ctl.sh", "start"], env=runtime,
                timeout=240, log="restored-server-start.log")
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            health = json.load(r)
        require(health.get("ok") is True and health.get("service") == "musetalk", "API health failed")
        receipt = {"restore_to_packages_seconds": round(restore_seconds, 3),
                   "restart_to_health_seconds": round(time.monotonic()-before, 3),
                   "removed_venv_to_health_seconds": round(time.monotonic()-started, 3),
                   "pip_install_executed": False, "health": health,
                   "scope": "same-host dependency replacement; code/system/models/engines retained"}
        save("restore-result.json", receipt)
        event("RESTORE_ACCEPTED", **receipt)
    except Exception:
        # Keep failed restored files/evidence; recover original environment if possible.
        if VENV.exists():
            VENV.rename(ROOT / "failed-restored-venv")
        if BACKUP.exists():
            BACKUP.rename(VENV)
            event("ROLLBACK_RESTORED", api_restart_attempted=False)
        raise


def main(c):
    global DEADLINE
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux only; never install on Mac")
    require(os.getuid() == 0 and c.get("label") == LABEL, "Owned-worker experiment required")
    DEADLINE = dt.datetime.fromisoformat(c["deadline_utc"])
    require(DEADLINE.tzinfo is not None and 20 < lease_remaining() <= 7200, "Valid bounded lease required")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    require(gpu == "NVIDIA GeForce RTX 3090", "Exact single RTX 3090 required")
    os.umask(0o077)
    modes = {"probe": probe, "diagnose": diagnose, "build": build,
             "finalize": lambda: build(finalize=True), "verify_archive": verify_archive,
             "upload_part": lambda: upload_part(c), "upload_batch": lambda: upload_batch(c),
             "restore": lambda: restore(c)}
    require(c["mode"] in modes, "Unsupported mode")
    if c['mode'] in {'build','finalize','restore'}:
        with open('/workspace/.'+LABEL+'.operation.lock','a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('An owned-worker operation is still in progress')
            modes[c['mode']]()
    else:
        modes[c["mode"]]()


if __name__ == "__main__":
    try:
        main(json.loads(sys.stdin.readline()))
    except Exception as error:
        # Never emit exception URLs, request bodies, environment or credentials.
        print(json.dumps({"worker_operation_failed": True, "error_type": type(error).__name__,
                          "safe_detail": str(error) if type(error) is ValueError else "Details suppressed"}), flush=True)
        raise SystemExit(2)
