#!/usr/bin/env python3
"""GPU-keyed engine store for MuseTalk: list / build / validate / adopt / restore / publish / ensure.

Manages three engine kinds (see scripts/musetalk_engine_keys.py for keys, layout and usability):

  unet_ts         torch_tensorrt static bs8 UNet .ts         built by scripts/tensorrt_export.py
  unet_stagewise  11 per-block FP16 TensorRT UNet plans      built by scripts/build_unet_stagewise.py
  taesd_trt       TAESD decoder + fused post TensorRT plans  built by scripts/vae_fast_decoder.py build

The CLI itself is stdlib-only and never imports torch/tensorrt: `list`, `key`, `check-corpus`,
`clean`, `publish` and the download half of `restore` work without CUDA. GPU work (build, validate)
always runs in a subprocess under the venv python, with its output appended to a build log
(<repo>/logs/<kind>_engine_*.log, or MUSETALK_ENGINE_LOG_DIR / $WORKSPACE/logs/musetalk).

Every entry a command creates is assembled in <store>/.<key>.partial-* and renamed into
<store>/<key>/bs<N>/ only when complete. An entry is usable only after validation on this host:
  unet_ts         validate_unet_backend.py --backend trt against calibration/unet_portable_bs8
                  (gate mae_max <= 0.01, max_abs_max <= 0.5); optional --cudagraphs manual|runtime
                  re-validates through the live loader with MUSETALK_TRT_UNET_CUDAGRAPHS set
  unet_stagewise  validate_unet_backend.py --backend runtime with MUSETALK_UNET_BACKEND=trt_stagewise
                  (the loader also checks every plan sha256 and the exact probe output hash)
  taesd_trt       vae_fast_decoder.py verify (plan sha256s + exact probe output hashes + fused post
                  bit-exact vs the repo post), MUSETALK_TAESD_TRT_BUILD=0 so nothing is rebuilt

Existing engines are never modified: `adopt` creates a store entry of SYMLINKS to them (plus a
copied unet_trt_meta.json for the .ts) and validates it. `publish` uploads only when explicitly
invoked (or build/ensure with --publish / MUSETALK_UNET_ENGINE_PUBLISH=1).

Exit codes (all commands):
  0 success / an engine is usable       1 operation failed (build, validation gate, unexpected error)
  2 usage error, or `ensure --require` found no usable engine
  3 nothing usable / preflight refused / nothing to restore (the server runs without this engine)
  4 refused: adopt device mismatch, publish of an unvalidated engine, restore checksum/key mismatch

Human-readable progress goes to stderr; the machine-readable result is one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import os
import pickletools
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # both `python scripts/unet_engine_store.py` and `from scripts import unet_engine_store`
    from scripts import musetalk_engine_keys as ek
except ImportError:  # pragma: no cover - scripts/ itself on sys.path
    import musetalk_engine_keys as ek  # type: ignore

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NONE, EXIT_REFUSED = 0, 1, 2, 3, 4

KINDS = ek.KINDS
UNET_TS, STAGEWISE, TAESD = ek.KIND_UNET_TS, ek.KIND_UNET_STAGEWISE, ek.KIND_TAESD_TRT

DEFAULT_CORPUS_REL = "calibration/unet_portable_bs8"
CORPUS_SCHEMA = "musetalk_unet_portable_corpus_v1"
REQUIRED_CAPTURE_KEYS = ("kind", "unet_io_batch", "latent_batch", "audio_feature_batch", "pred_latents")
DEFAULT_FAIL_MAE = 0.01
DEFAULT_FAIL_MAX_ABS = 0.5
DEFAULT_LIMIT = 16

REMOTE_ENV = {
    UNET_TS: "MUSETALK_UNET_ENGINE_REMOTE",
    STAGEWISE: "MUSETALK_UNET_STAGEWISE_ENGINE_REMOTE",
    TAESD: "MUSETALK_TAESD_TRT_ENGINE_REMOTE",
}
REMOTE_SUBDIR = {UNET_TS: "unet-engines", STAGEWISE: "unet-stagewise-engines", TAESD: "taesd-trt-engines"}
ARCHIVE_NAME = {UNET_TS: "unet_trt.tar", STAGEWISE: "engine.tar", TAESD: "engine.tar"}

# (VRAM GB, MemAvailable GB, disk GB, timeout min) per kind for `build`; env names override.
BUILD_LIMITS = {
    UNET_TS: {"vram": ("MUSETALK_UNET_BUILD_MIN_VRAM_GB", 8.0),
              "mem": ("MUSETALK_UNET_BUILD_MIN_MEM_AVAILABLE_GB", 14.0),
              "disk": ("MUSETALK_UNET_BUILD_MIN_DISK_GB", 5.0), "timeout_min": 30.0},
    STAGEWISE: {"vram": ("MUSETALK_UNET_STAGEWISE_BUILD_MIN_VRAM_GB", 8.0),
                "mem": ("MUSETALK_UNET_STAGEWISE_BUILD_MIN_MEM_AVAILABLE_GB", 12.0),
                "disk": ("MUSETALK_UNET_STAGEWISE_BUILD_MIN_DISK_GB", 4.0), "timeout_min": 45.0},
    TAESD: {"vram": ("MUSETALK_TAESD_TRT_BUILD_MIN_VRAM_GB", 4.0),
            "mem": ("MUSETALK_TAESD_TRT_BUILD_MIN_MEM_AVAILABLE_GB", 4.0),
            "disk": ("MUSETALK_TAESD_TRT_BUILD_MIN_DISK_GB", 1.0), "timeout_min": 10.0},
}
VALIDATE_LIMITS = {
    UNET_TS: {"mem": ("MUSETALK_UNET_VALIDATE_MIN_MEM_AVAILABLE_GB", 10.0), "timeout_min": 15.0},
    STAGEWISE: {"mem": ("MUSETALK_UNET_STAGEWISE_VALIDATE_MIN_MEM_AVAILABLE_GB", 4.0), "timeout_min": 10.0},
    TAESD: {"mem": ("MUSETALK_TAESD_TRT_VALIDATE_MIN_MEM_AVAILABLE_GB", 3.0), "timeout_min": 5.0},
}
# Child processes get a clean view of the runtime knobs: the store sets exactly what each tool needs.
SCRUB_PREFIXES = ("MUSETALK_UNET_", "MUSETALK_TRT_", "MUSETALK_TAESD_", "MUSETALK_VAE_")
SCRUB_EXACT = ("MUSETALK_FREE_EAGER_UNET", "MUSETALK_COMPILE", "MUSETALK_UNET_CALIBRATION_CAPTURE")
TAESD_RECIPE_KEEP = ("MUSETALK_TAESD_TRT_OPT_LEVEL", "MUSETALK_TAESD_TRT_STRONGLY_TYPED",
                     "MUSETALK_TAESD_LOCAL_DIR", "MUSETALK_TAESD_MODEL")
CUDAGRAPH_MODES = ("manual", "runtime")


class StoreError(RuntimeError):
    def __init__(self, message: str, code: int = EXIT_FAIL, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}


# --------------------------------------------------------------------------- logging / env
def log(message: str) -> None:
    print(f"[engine_store {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log(f"WARNING: ignoring non-numeric {name}={raw!r}; using {default}")
        return default


def env_on(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def publish_requested(flag: bool) -> bool:
    return bool(flag) or env_on("MUSETALK_UNET_ENGINE_PUBLISH") or env_on("MUSETALK_ENGINE_PUBLISH")


def child_env(extra: Dict[str, str], keep: Tuple[str, ...] = ()) -> Dict[str, str]:
    env = {}
    for name, value in os.environ.items():
        if name in keep or not (name.startswith(SCRUB_PREFIXES) or name in SCRUB_EXACT):
            env[name] = value
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def rel_to_repo(path: Any, repo_root: Path) -> str:
    try:
        return str(Path(os.path.abspath(str(path))).relative_to(os.path.abspath(str(repo_root))))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo_root = Path(os.path.abspath(str(args.repo_root)))
        self.store_arg = getattr(args, "store", None) or None
        self.venv_python, self.venv_dir = resolve_venv(args)
        self.raw_facts = ek.detect_engine_facts(self.venv_dir)
        self.facts = ek.normalize_facts(self.raw_facts)

    def store(self, kind: str) -> Path:
        return ek.store_root(kind, self.repo_root, self.store_arg)

    def key(self, kind: str) -> Tuple[Optional[str], Optional[str]]:
        return ek.engine_key_or_none(kind, self.facts)

    def require_key(self, kind: str) -> str:
        key, error = self.key(kind)
        if key is None:
            raise StoreError(f"no engine key for {kind} on this host: {error}", EXIT_NONE)
        return key

    def script(self, name: str) -> Path:
        return self.repo_root / "scripts" / name

    def host_summary(self) -> Dict[str, Any]:
        nf = self.facts
        return {
            "gpu_name": nf.get("gpu_name"), "compute_capability": nf.get("compute_capability"),
            "gpu_memory_total_mib": nf.get("gpu_memory_total_mib"),
            "tensorrt_version": nf.get("tensorrt_version"),
            "torch_tensorrt_version": nf.get("torch_tensorrt_version"),
            "torch_version": nf.get("torch_version"), "venv_python": str(self.venv_python),
            "facts_source": self.raw_facts.get("facts_source", "detected"),
            "gpu_query_error": self.raw_facts.get("gpu_query_error"),
            "keys": {kind: self.key(kind)[0] for kind in KINDS},
        }


def resolve_venv(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    """venv python for subprocesses + the venv dir whose dist-info gives the TRT versions."""
    if getattr(args, "venv_python", None):
        python = Path(args.venv_python)
        venv = Path(args.venv) if getattr(args, "venv", None) else python.parent.parent
        return python, venv
    if getattr(args, "venv", None):
        venv = Path(args.venv)
        return venv / "bin" / "python", venv
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return Path(sys.executable), Path(sys.prefix)
    default = Path(os.environ.get("WORKSPACE", "/workspace")) / ".venvs" / "musetalk_trt_stagewise"
    if (default / "bin" / "python").exists():
        return default / "bin" / "python", default
    return Path(sys.executable), None


# --------------------------------------------------------------------------- host resources
def measure_mem_available_mb() -> Optional[float]:
    """min(/proc/meminfo MemAvailable, cgroup v2 max - current + reclaimable file/slab), like box_guard."""
    meminfo = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    meminfo = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    cgroup = None
    try:
        raw_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        if raw_max != "max":
            reclaim = 0
            for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
                name, _, value = line.partition(" ")
                if name in ("active_file", "inactive_file", "slab_reclaimable"):
                    reclaim += int(value)
            cgroup = (int(raw_max) - current + reclaim) / (1024.0 * 1024.0)
    except (OSError, ValueError):
        cgroup = None
    values = [v for v in (meminfo, cgroup) if v is not None]
    return min(values) if values else None


def host_resources(ctx: Ctx, disk_path: Path) -> Dict[str, Any]:
    ram = ctx.raw_facts.get("ram") if isinstance(ctx.raw_facts.get("ram"), dict) else {}
    mem_mb = None
    mem_source = "measured"
    for field in ("effective_available_mb", "mem_available_mb"):
        if ram.get(field) is not None:
            mem_mb, mem_source = float(ram[field]), "facts"
            break
    if mem_mb is None:
        mem_mb = measure_mem_available_mb()
    if ctx.raw_facts.get("disk_free_gb") is not None:
        disk_gb, disk_source = float(ctx.raw_facts["disk_free_gb"]), "facts"
    else:
        probe = Path(disk_path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        disk_gb, disk_source = shutil.disk_usage(str(probe)).free / 1e9, "measured"
    total = ctx.facts.get("gpu_memory_total_mib")
    used = ctx.facts.get("gpu_memory_used_mib")
    return {
        "mem_available_gb": round(mem_mb / 1024.0, 2) if mem_mb is not None else None,
        "mem_source": mem_source, "disk_free_gb": round(disk_gb, 2), "disk_source": disk_source,
        "vram_total_gb": round(total / 1024.0, 2) if total else None,
        "vram_free_gb": round((total - used) / 1024.0, 2) if total and used is not None else None,
    }


def preflight(ctx: Ctx, kind: str, phase: str, disk_path: Path) -> Dict[str, Any]:
    """Resource/tool checks before a GPU subprocess. phase: build | validate."""
    problems: List[str] = []
    warnings: List[str] = []
    if not ctx.facts.get("has_gpu"):
        problems.append("no visible NVIDIA GPU")
    if not Path(ctx.venv_python).exists():
        problems.append(f"venv python not found: {ctx.venv_python}")
    res = host_resources(ctx, disk_path)
    if phase == "build":
        limits = BUILD_LIMITS[kind]
        min_vram = env_float(*limits["vram"])
        min_mem = env_float(*limits["mem"])
        min_disk = env_float(*limits["disk"])
        if res["vram_total_gb"] is not None and res["vram_total_gb"] < min_vram:
            problems.append(f"GPU VRAM {res['vram_total_gb']} GB < {min_vram} GB ({limits['vram'][0]})")
        if res["disk_free_gb"] < min_disk:
            problems.append(f"disk free {res['disk_free_gb']} GB < {min_disk} GB ({limits['disk'][0]})")
        if res["vram_free_gb"] is not None and res["vram_free_gb"] < 4.0:
            warnings.append(f"only {res['vram_free_gb']} GB VRAM free: another GPU process may make the build OOM")
    else:
        limits = VALIDATE_LIMITS[kind]
        min_mem = env_float(*limits["mem"])
    if res["mem_available_gb"] is None:
        warnings.append("MemAvailable unknown; RAM check skipped")
    elif res["mem_available_gb"] < min_mem:
        problems.append(f"MemAvailable {res['mem_available_gb']} GB < {min_mem} GB ({limits['mem'][0]})")
    tools = {UNET_TS: ("tensorrt_export.py" if phase == "build" else "validate_unet_backend.py"),
             STAGEWISE: ("build_unet_stagewise.py" if phase == "build" else "validate_unet_backend.py"),
             TAESD: "vae_fast_decoder.py"}
    tool = ctx.script(tools[kind])
    if not tool.exists():
        problems.append(f"missing {rel_to_repo(tool, ctx.repo_root)}")
    if phase == "build" and kind in (UNET_TS, STAGEWISE):
        for weight in ("models/musetalkV15/unet.pth", "models/musetalkV15/musetalk.json"):
            if not (ctx.repo_root / weight).exists():
                problems.append(f"missing UNet weights {weight} (run download_weights.sh)")
    if kind == TAESD and not (ctx.repo_root / "models/taesd/config.json").exists() \
            and not os.environ.get("MUSETALK_TAESD_LOCAL_DIR"):
        warnings.append("models/taesd missing: the TAESD tool will try the Hugging Face download")
    return {"ok": not problems, "phase": phase, "problems": problems, "warnings": warnings, "resources": res}


# --------------------------------------------------------------------------- corpus
def corpus_path(ctx: Ctx, override: Optional[str] = None) -> Path:
    raw = override or os.environ.get("MUSETALK_UNET_VALIDATION_CORPUS", "").strip() or DEFAULT_CORPUS_REL
    path = Path(raw)
    return path if path.is_absolute() else ctx.repo_root / path


def capture_problems(path: Path) -> List[str]:
    """Check a torch.save'd unet_io capture WITHOUT torch: zip archive + pickle keys."""
    try:
        with zipfile.ZipFile(str(path)) as archive:
            pickles = [n for n in archive.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
            if not pickles:
                return [f"{path.name}: no data.pkl (not a torch.save zip archive)"]
            data = archive.read(pickles[0])
    except (OSError, zipfile.BadZipFile) as exc:
        return [f"{path.name}: unreadable capture ({type(exc).__name__}: {exc})"]
    strings = set()
    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(data)):
            if isinstance(arg, str) and "UNICODE" in opcode.name:
                strings.add(arg)
    except Exception as exc:  # malformed pickle
        return [f"{path.name}: malformed pickle ({type(exc).__name__}: {exc})"]
    missing = [key for key in REQUIRED_CAPTURE_KEYS if key not in strings]
    return [f"{path.name}: missing capture keys {missing}"] if missing else []


def check_corpus(path: Path, deep: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"dir": str(path), "ok": False, "files": 0, "manifest_sha256": None,
                           "problems": [], "warnings": []}
    if not path.is_dir():
        out["problems"].append(f"validation corpus not found: {path}")
        return out
    manifest_path = path / "manifest.json"
    captures = sorted(path.glob("unet_io_*_bs8_*.pt"))
    if manifest_path.exists():
        out["manifest_sha256"] = ek.sha256_file(manifest_path)
        try:
            manifest = ek.read_json(manifest_path) or {}
        except ValueError as exc:
            out["problems"].append(str(exc))
            return out
        if manifest.get("schema") != CORPUS_SCHEMA:
            out["warnings"].append(f"unexpected corpus schema {manifest.get('schema')!r}")
        listed = manifest.get("files") or []
        for row in listed:
            file_path = path / str(row.get("file"))
            if not file_path.exists():
                out["problems"].append(f"missing capture {row.get('file')}")
                continue
            if row.get("bytes") is not None and file_path.stat().st_size != int(row["bytes"]):
                out["problems"].append(f"{row.get('file')}: size {file_path.stat().st_size} != {row['bytes']}")
                continue
            if deep and row.get("sha256") and ek.sha256_file(file_path) != row["sha256"]:
                out["problems"].append(f"{row.get('file')}: sha256 mismatch (corrupted copy)")
        unlisted = sorted({p.name for p in captures} - {str(r.get("file")) for r in listed})
        if unlisted:
            out["warnings"].append(f"captures not in manifest (still validated): {unlisted[:4]}")
    else:
        out["warnings"].append("no manifest.json: capture checksums not verified")
    if not captures:
        out["problems"].append(f"no unet_io_*_bs8_*.pt captures in {path}")
    if deep:
        for capture in captures:
            out["problems"].extend(capture_problems(capture))
    out["files"] = len(captures)
    out["ok"] = not out["problems"]
    return out


# --------------------------------------------------------------------------- locking / partials
@contextlib.contextmanager
def store_lock(store: Path, key: str, wait_s: Optional[float] = None) -> Iterator[None]:
    store.mkdir(parents=True, exist_ok=True)
    wait_s = env_float("MUSETALK_ENGINE_STORE_LOCK_WAIT_S", 3600.0) if wait_s is None else wait_s
    lock_path = store / f".{key}.lock"
    with open(lock_path, "a+") as handle:
        deadline = time.time() + wait_s
        announced = False
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    raise StoreError(f"store entry {key} is locked by another process ({lock_path})", EXIT_FAIL)
                if not announced:
                    log(f"waiting for {lock_path} (another build/validate holds it)")
                    announced = True
                time.sleep(2.0)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_PARTIAL_RE = re.compile(r"^\.(?P<key>.+)\.partial-bs(?P<batch>\d+)-(?:(?P<resume>resume)|(?P<pid>\d+)-(?P<stamp>\d+))$")
_OLD_RE = re.compile(r"^\.(?P<key>.+)\.old-bs(?P<batch>\d+)-(?P<pid>\d+)-(?P<stamp>\d+)$")
_PUBLISH_TMP_RE = re.compile(r"^\.publish-(?P<pid>\d+)-(?P<stamp>\d+)$")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def new_partial(store: Path, key: str, batch: int, resume: bool = False) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    if resume:
        root = store / f".{key}.partial-bs{batch}-resume"
    else:
        root = store / f".{key}.partial-bs{batch}-{os.getpid()}-{time.time_ns()}"
    (root / f"bs{batch}").mkdir(parents=True, exist_ok=True)
    return root


def list_partials(store: Path) -> List[Dict[str, Any]]:
    out = []
    if not store.is_dir():
        return out
    for child in sorted(store.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        for label, regex in (("partial", _PARTIAL_RE), ("old", _OLD_RE), ("publish_tmp", _PUBLISH_TMP_RE)):
            match = regex.match(child.name)
            if match:
                break
        else:
            continue
        info = match.groupdict()
        pid = int(info["pid"]) if info.get("pid") else None
        out.append({
            "path": str(child), "key": info.get("key"), "batch": int(info["batch"]) if info.get("batch") else None,
            "kind": label, "resume": bool(info.get("resume")), "pid": pid,
            "alive": pid_alive(pid) if pid else None,
            "age_s": round(time.time() - child.stat().st_mtime, 1),
        })
    return out


def safe_rmtree(path: Path, store: Path) -> None:
    """rmtree only inside the store and never through a symlink (symlinked engines stay untouched)."""
    path = Path(path)
    if path.is_symlink():
        path.unlink()
        return
    real = os.path.realpath(str(path))
    real_store = os.path.realpath(str(store))
    if not (real + os.sep).startswith(real_store + os.sep) or real == real_store:
        raise StoreError(f"refusing to delete {path}: not inside the store {store}")
    shutil.rmtree(real)


def clean_partials(store: Path, key: Optional[str] = None, include_resume: bool = False) -> List[str]:
    removed = []
    for item in list_partials(store):
        if key and item["key"] not in (key, None):
            continue
        if item["resume"] and not include_resume:
            continue
        if item["pid"] and item["alive"] and item["pid"] != os.getpid():
            continue
        if item["pid"] == os.getpid():
            continue
        safe_rmtree(Path(item["path"]), store)
        removed.append(item["path"])
    if removed:
        log(f"removed {len(removed)} stale partial/old dir(s) under {store}")
    return removed


def promote(partial_root: Path, batch: int, store: Path, key: str) -> Path:
    """Atomically move <partial_root>/bs<N> to <store>/<key>/bs<N> (an old entry is renamed away first)."""
    source = partial_root / f"bs{batch}"
    final = store / key / f"bs{batch}"
    final.parent.mkdir(parents=True, exist_ok=True)
    old = None
    if final.exists() or final.is_symlink():
        old = store / f".{key}.old-bs{batch}-{os.getpid()}-{time.time_ns()}"
        os.rename(str(final), str(old))
    os.rename(str(source), str(final))
    if old is not None:
        safe_rmtree(old, store)
    try:
        partial_root.rmdir()
    except OSError:
        if partial_root.exists():
            safe_rmtree(partial_root, store)
    log(f"published entry {final}")
    return final


def discard_partial(partial_root: Path, store: Path) -> None:
    if partial_root.exists():
        safe_rmtree(partial_root, store)


# --------------------------------------------------------------------------- subprocess runner
def log_dir(ctx: Ctx) -> Path:
    candidates = []
    if os.environ.get("MUSETALK_ENGINE_LOG_DIR"):
        candidates.append(Path(os.environ["MUSETALK_ENGINE_LOG_DIR"]))
    candidates.append(ctx.repo_root / "logs")
    candidates.append(Path(os.environ.get("WORKSPACE", "/workspace")) / "logs" / "musetalk")
    candidates.append(ctx.store(UNET_TS) / ".logs")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(str(candidate), os.W_OK):
                return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def log_path_for(ctx: Ctx, kind: str, key: str, batch: int, phase: str) -> Path:
    if kind == UNET_TS and phase == "build":
        name = f"unet_engine_build_{key}.log"
    else:
        name = f"{kind}_engine_{phase}_{key}_bs{batch}.log"
    return log_dir(ctx) / name


def run_logged(cmd: List[str], env: Dict[str, str], cwd: Path, log_path: Path, timeout_s: float,
               label: str) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, "ab") as handle:
        offset = handle.tell()
        header = f"\n=== {ek.utc_now()} {label}: {' '.join(cmd)} (cwd {cwd})\n"
        handle.write(header.encode())
        handle.flush()
        log(f"{label}: running {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))} "
            f"(timeout {timeout_s / 60:.0f} min, log {log_path})")
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(cwd), env=env,
                                start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            log(f"{label}: timeout after {timeout_s / 60:.1f} min; terminating process group {proc.pid}")
            _kill_group(proc)
            rc = proc.returncode if proc.returncode is not None else -9
        except BaseException:
            _kill_group(proc)
            raise
    seconds = round(time.time() - started, 1)
    output = _read_from(log_path, offset)
    log(f"{label}: rc={rc} in {seconds}s")
    return {"rc": rc, "seconds": seconds, "timed_out": timed_out, "log": str(log_path),
            "output": output, "tail": output[-3000:]}


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, grace in ((signal.SIGTERM, 30.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _read_from(path: Path, offset: int, limit: int = 4 << 20) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


def last_json_object(text: str, required: Tuple[str, ...] = ()) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    positions = [m.start() for m in re.finditer(r"(?m)^\{", text)]
    for pos in reversed(positions):
        try:
            obj, _end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in required):
            return obj
    return None


# --------------------------------------------------------------------------- validation
def _validation_base(key: str, method: str, **fields: Any) -> Dict[str, Any]:
    record = ek.empty_validation()
    record.update({"engine_key": key, "method": method, "validated_utc": ek.utc_now()})
    record.update(fields)
    return record


def _unet_group(batch: int) -> int:
    return batch // 8 if batch % 8 == 0 and batch >= 8 else 1


def run_validation(ctx: Ctx, kind: str, entry: Path, fingerprint: Dict[str, Any], mode: Optional[str] = None,
                   corpus_override: Optional[str] = None, timeout_min: Optional[float] = None) -> Dict[str, Any]:
    """Validate one entry directory on this GPU; returns a validation record (does not touch the fingerprint)."""
    key = str(fingerprint.get("engine_key"))
    batch = int(fingerprint.get("batch") or ek.DEFAULT_BATCH[kind])
    attempts = int(((fingerprint.get("validation") or {}).get("attempts") or 0)) + 1
    pre = preflight(ctx, kind, "validate", entry)
    for warning in pre["warnings"]:
        log(f"WARNING: {warning}")
    method = {UNET_TS: "validate_unet_backend.py --backend " + ("runtime" if mode else "trt"),
              STAGEWISE: "validate_unet_backend.py --backend runtime (MUSETALK_UNET_BACKEND=trt_stagewise)",
              TAESD: "vae_fast_decoder.py verify"}[kind]
    if mode:
        method += f" MUSETALK_TRT_UNET_CUDAGRAPHS={mode}"
    if not pre["ok"]:
        log("validation not run: " + "; ".join(pre["problems"]))
        return _validation_base(key, method, status="not_run", reason="; ".join(pre["problems"]),
                                preflight=pre, attempts=attempts - 1)
    timeout_s = 60.0 * (timeout_min or VALIDATE_LIMITS[kind]["timeout_min"])
    log_path = log_path_for(ctx, kind, key, batch, "validate")
    report_name = ek.VALIDATION_FILE if not mode else f"validation_cudagraphs_{mode}.json"
    report_tmp = entry / f".{report_name}.tmp-{os.getpid()}"
    if kind in (UNET_TS, STAGEWISE):
        corpus = corpus_path(ctx, corpus_override)
        corpus_check = check_corpus(corpus)
        if not corpus_check["ok"]:
            log("validation not run: corpus problems: " + "; ".join(corpus_check["problems"][:3]))
            return _validation_base(key, method, status="not_run", attempts=attempts - 1,
                                    reason="validation corpus unusable: " + "; ".join(corpus_check["problems"][:3]))
        fail_mae = env_float("MUSETALK_UNET_VALIDATE_FAIL_MAE", DEFAULT_FAIL_MAE)
        fail_max_abs = env_float("MUSETALK_UNET_VALIDATE_FAIL_MAX_ABS", DEFAULT_FAIL_MAX_ABS)
        cmd = [str(ctx.venv_python), str(ctx.script("validate_unet_backend.py")),
               "--capture-dir", str(corpus), "--padded-batch-size", "8", "--limit", str(DEFAULT_LIMIT),
               "--fail-mae", str(fail_mae), "--fail-max-abs", str(fail_max_abs),
               "--warmup", "2", "--iters", "5", "--report-path", str(report_tmp)]
        group = _unet_group(batch)
        if group > 1:
            cmd += ["--group-captures", str(group)]
        extra: Dict[str, str] = {"MUSETALK_TRT_FALLBACK": "0"}
        engine = entry / ek.UNET_TS_ENGINE_FILE
        if kind == UNET_TS and not mode:
            cmd += ["--backend", "trt", "--trt-path", str(engine)]
        elif kind == UNET_TS:
            cmd += ["--backend", "runtime"]
            extra.update({"MUSETALK_UNET_BACKEND": "trt", "MUSETALK_TRT_UNET_ENABLED": "1",
                          "MUSETALK_TRT_UNET_PATHS": f"{batch}:{engine}", "MUSETALK_TRT_UNET_CUDAGRAPHS": mode})
        else:
            cmd += ["--backend", "runtime"]
            extra.update({"MUSETALK_UNET_BACKEND": "trt_stagewise",
                          "MUSETALK_UNET_STAGEWISE_CACHE_DIR": str(entry.parent),
                          "MUSETALK_UNET_STAGEWISE_BATCH": str(batch),
                          "MUSETALK_UNET_STAGEWISE_PROBE_CHECK": "1",
                          "MUSETALK_UNET_STAGEWISE_VERIFY_SHA": "1",
                          "MUSETALK_UNET_STAGEWISE_CUDAGRAPH": "1"})
        run = run_logged(cmd, child_env(extra), ctx.repo_root, log_path, timeout_s, f"validate {kind} {key} bs{batch}")
        report = None
        try:
            report = ek.read_json(report_tmp) if report_tmp.exists() else None
        except ValueError:
            report = None
        summary = (report or {}).get("summary") or {}
        record = _validation_base(
            key, method, rc=run["rc"], seconds=run["seconds"], log=run["log"], attempts=attempts,
            mae_max=summary.get("mae_max"), max_abs_max=summary.get("max_abs_max"),
            files=summary.get("files"), capture_dir=rel_to_repo(corpus, ctx.repo_root),
            corpus_manifest_sha256=corpus_check["manifest_sha256"], group_captures=group,
            gate={"fail_mae": fail_mae, "fail_max_abs": fail_max_abs}, report=report_name)
        if report is None:
            record.update(status="error", error=f"validator wrote no report (rc={run['rc']}, "
                                                f"timed_out={run['timed_out']}); tail: {run['tail'][-600:]}")
        else:
            gate_ok = (summary.get("mae_max") is not None and summary.get("max_abs_max") is not None
                       and float(summary["mae_max"]) <= fail_mae and float(summary["max_abs_max"]) <= fail_max_abs
                       and int(summary.get("files") or 0) > 0)
            if run["rc"] == 0 and gate_ok:
                record.update(passed=True, status="passed")
            elif not gate_ok:
                record.update(status="failed", error=f"gate failed: mae_max={summary.get('mae_max')} "
                                                     f"max_abs_max={summary.get('max_abs_max')} files={summary.get('files')}")
            else:
                record.update(status="error", error=f"validator rc={run['rc']}")
            report["store"] = {"engine_key": key, "entry": str(entry), "validated_utc": record["validated_utc"],
                               "method": method}
            ek.write_json_atomic(entry / report_name, report)
        with contextlib.suppress(OSError):
            report_tmp.unlink()
        return record
    # taesd_trt
    runtime_fp = fingerprint.get("runtime_fingerprint") or {}
    extra = {"MUSETALK_TAESD_BACKEND": "trt", "MUSETALK_TAESD_TRT_DIR": str(entry),
             "MUSETALK_TAESD_TRT_BATCH": str(batch), "MUSETALK_TAESD_TRT_BUILD": "0",
             "MUSETALK_TAESD_TRT_STRICT": "1"}
    if "opt_level" in runtime_fp:
        extra["MUSETALK_TAESD_TRT_OPT_LEVEL"] = str(int(runtime_fp["opt_level"]))
    if "strongly_typed" in runtime_fp:
        extra["MUSETALK_TAESD_TRT_STRONGLY_TYPED"] = "1" if runtime_fp["strongly_typed"] else "0"
    cmd = [str(ctx.venv_python), str(ctx.script("vae_fast_decoder.py")), "verify", "--batch", str(batch)]
    run = run_logged(cmd, child_env(extra, keep=("MUSETALK_TAESD_LOCAL_DIR", "MUSETALK_TAESD_MODEL")),
                     ctx.repo_root, log_path, timeout_s, f"validate {kind} {key} bs{batch}")
    result = last_json_object(run["output"], ("key", "probe"))
    record = _validation_base(key, method, rc=run["rc"], seconds=run["seconds"], log=run["log"],
                              attempts=attempts, report=report_name)
    expected_key = fingerprint.get("runtime_key")
    if run["rc"] == 0 and result and (not expected_key or result.get("key") == expected_key):
        meta = _read_taesd_meta(entry, fingerprint)
        record.update(passed=True, status="passed", runtime_key=result.get("key"), probe=result.get("probe"),
                      gate_verdict=((meta or {}).get("gate") or {}).get("verdict"))
    elif run["rc"] == 0 and result:
        record.update(status="failed", error=f"verify loaded runtime key {result.get('key')} != {expected_key}")
    else:
        record.update(status="failed" if run["rc"] not in (None, -9) and not run["timed_out"] else "error",
                      error=f"vae_fast_decoder verify rc={run['rc']}; tail: {run['tail'][-600:]}")
    ek.write_json_atomic(entry / report_name, {"store": {"engine_key": key, "entry": str(entry)},
                                               "record": record, "verify_output": result})
    return record


def _read_taesd_meta(entry: Path, fingerprint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = fingerprint.get("engine_file")
    if not name:
        return None
    try:
        return ek.read_json(entry / str(name))
    except ValueError:
        return None


def apply_validation(ctx: Ctx, kind: str, entry: Path, fingerprint: Dict[str, Any], record: Dict[str, Any],
                     mode: Optional[str] = None) -> Dict[str, Any]:
    """Merge a validation record into the fingerprint (+ the entry's own unet_trt_meta.json copy)."""
    fingerprint = dict(fingerprint)
    previous = dict(fingerprint.get("validation") or {})
    if mode:
        modes = dict(previous.get("modes") or {})
        modes[f"cudagraphs_{mode}"] = record
        previous["modes"] = modes
        fingerprint["validation"] = previous
    else:
        if record.get("status") == "not_run" and previous.get("passed") is True:
            # a validation that could not start (e.g. low RAM) never demotes an already-validated entry
            previous["last_attempt"] = {k: record.get(k) for k in ("status", "reason", "validated_utc")}
            fingerprint["validation"] = previous
        else:
            merged = dict(record)
            if previous.get("modes"):
                merged["modes"] = previous["modes"]
            fingerprint["validation"] = merged
    ek.write_fingerprint(entry, fingerprint)
    if kind == UNET_TS and not mode:
        _sync_ts_meta_validation(entry, fingerprint["validation"])
    return fingerprint


def _sync_ts_meta_validation(entry: Path, validation: Dict[str, Any]) -> None:
    meta_path = entry / ek.UNET_TS_META_FILE
    if meta_path.is_symlink():  # never write through a link into an original directory
        content = meta_path.read_bytes()
        meta_path.unlink()
        meta_path.write_bytes(content)
    try:
        meta = ek.read_json(meta_path) or {}
    except ValueError:
        meta = {}
    status = validation.get("status")
    meta["validation"] = {
        "passed": True if validation.get("passed") else (False if status == "failed" else None),
        "status": status, "source": "unet_engine_store", "engine_key": validation.get("engine_key"),
        "capture_dir": validation.get("capture_dir"), "mae_max": validation.get("mae_max"),
        "max_abs_max": validation.get("max_abs_max"), "max_mae": (validation.get("gate") or {}).get("fail_mae"),
        "max_abs": (validation.get("gate") or {}).get("fail_max_abs"), "padded_batch_size": 8,
        "report_path": ek.VALIDATION_FILE, "validated_utc": validation.get("validated_utc"),
    }
    ek.write_json_atomic(meta_path, meta)


# --------------------------------------------------------------------------- adopt sources
class Source:
    """Files an adopted/restored entry will contain: links {entry name: target}, copies {name: bytes}."""

    def __init__(self, kind: str, batch: int, origin: str) -> None:
        self.kind = kind
        self.batch = batch
        self.origin = origin
        self.links: Dict[str, str] = {}
        self.copies: Dict[str, bytes] = {}
        self.problems: List[str] = []
        self.notes: List[str] = []
        self.fields: Dict[str, Any] = {}


def _synth_ts_meta(batch: int) -> Dict[str, Any]:
    return {"type": "unet", "batch_range": [batch, batch], "opt_batch": batch, "latent_shape": [8, 32, 32],
            "encoder_hidden_states_shape": [50, 384], "dtype": "float16", "save_format": "torchscript",
            "example_source": "synthesized by unet_engine_store adopt (original had no meta)"}


def source_unet_ts(ctx: Ctx, ts_path: Path, batch: int, allow_same_cc: bool, meta_path: Optional[Path] = None) -> Source:
    src = Source(UNET_TS, batch, str(ts_path))
    if not ts_path.exists():
        src.problems.append(f"engine not found: {ts_path}")
        return src
    real = os.path.realpath(str(ts_path))
    size = os.stat(real).st_size
    scan = ek.scan_embedded_devices(ts_path, cache_dir=ctx.store(UNET_TS) / ".scan_cache")
    log(f"embedded device scan of {ts_path}: {[d['device_string'] for d in scan['devices']]} "
        f"({'cached' if scan.get('cached') else str(scan.get('seconds')) + 's'})")
    problems = ek.embedded_device_problems(scan, ctx.facts)
    if allow_same_cc:
        problems = [p for p in problems if "was built on" not in p]
    src.problems.extend(problems)
    candidates = [meta_path] if meta_path else [ts_path.with_name(ek.UNET_TS_META_FILE),
                                                 Path(real).with_name(ek.UNET_TS_META_FILE)]
    meta = None
    for candidate in candidates:
        if candidate and candidate.exists():
            try:
                meta = ek.read_json(candidate)
                src.notes.append(f"unet_trt_meta.json copied from {candidate}")
                break
            except ValueError as exc:
                src.notes.append(f"ignored malformed meta {candidate}: {exc}")
    if not isinstance(meta, dict):
        meta = _synth_ts_meta(batch)
        src.notes.append("unet_trt_meta.json synthesized (static batch range)")
    batch_range = meta.get("batch_range")
    if isinstance(batch_range, list) and len(batch_range) == 2:
        if not (int(batch_range[0]) <= batch <= int(batch_range[1])):
            src.problems.append(f"meta batch_range {batch_range} does not cover bs{batch}")
    original_validation = meta.get("validation")
    if isinstance(original_validation, dict) and original_validation.get("passed") is False:
        src.notes.append("original meta says validation.passed=false; the store re-validates anyway")
    meta = dict(meta)
    meta["validation_original"] = original_validation
    meta["validation"] = {"passed": None, "status": "pending", "source": "unet_engine_store"}
    src.links[ek.UNET_TS_ENGINE_FILE] = real
    src.copies[ek.UNET_TS_META_FILE] = (json.dumps(meta, indent=2) + "\n").encode()
    src.fields = {"engine_bytes": size, "engine_sha256": None,
                  "embedded_device": scan["devices"][0]["device_string"] if scan.get("devices") else None,
                  "embedded_devices": [d["device_string"] for d in scan.get("devices") or []],
                  "original_path": str(ts_path), "original_realpath": real,
                  "save_format": meta.get("save_format"), "batch_range": batch_range}
    src.notes.append("tensorrt/torch_tensorrt versions are the venv's: a torch_tensorrt .ts does not record them; "
                     "the embedded device string and on-GPU validation are the compatibility evidence")
    return src


def source_stagewise(ctx: Ctx, bs_dir: Path, batch: Optional[int], allow_same_cc: bool) -> Source:
    manifest_path = bs_dir / ek.STAGEWISE_MANIFEST_FILE
    try:
        manifest = ek.read_json(manifest_path)
    except ValueError as exc:
        manifest = None
        src = Source(STAGEWISE, batch or 0, str(bs_dir))
        src.problems.append(str(exc))
        return src
    src = Source(STAGEWISE, int(batch or (manifest or {}).get("batch") or 0), str(bs_dir))
    if not isinstance(manifest, dict):
        src.problems.append(f"no stagewise manifest at {manifest_path}")
        return src
    facts = ek.stagewise_manifest_facts(manifest)
    if not facts["schema_ok"]:
        src.problems.append(f"unexpected manifest schema {facts['schema']!r}")
    if not facts["complete"]:
        src.problems.append("engine set is incomplete (manifest complete=false); finish the build first")
    if facts["missing_blocks"]:
        src.problems.append(f"manifest lacks blocks {facts['missing_blocks']}")
    if batch and int(manifest.get("batch") or 0) != int(batch):
        src.problems.append(f"manifest batch {manifest.get('batch')} != requested bs{batch}")
    src.problems.extend(ek.host_mismatch(ctx.facts, facts["gpu_name"], facts["compute_capability"],
                                         facts["tensorrt_version"], allow_same_cc=allow_same_cc))
    files: Dict[str, int] = {}
    blocks = manifest.get("blocks") or {}
    for name in ek.STAGEWISE_BLOCKS:
        engine_file = (blocks.get(name) or {}).get("engine_file")
        if not engine_file:
            continue
        path = bs_dir / engine_file
        if not path.exists():
            src.problems.append(f"missing plan {path}")
            continue
        src.links[engine_file] = os.path.realpath(str(path))
        files[engine_file] = path.stat().st_size
    probe_file = (manifest.get("probe") or {}).get("output_file") or "probe_output.pt"
    if (bs_dir / probe_file).exists():
        src.links[probe_file] = os.path.realpath(str(bs_dir / probe_file))
        files[probe_file] = (bs_dir / probe_file).stat().st_size
    else:
        src.problems.append(f"missing {probe_file} (the runtime probe check needs it)")
    src.links[ek.STAGEWISE_MANIFEST_FILE] = os.path.realpath(str(manifest_path))
    files[ek.STAGEWISE_MANIFEST_FILE] = manifest_path.stat().st_size
    if facts["torch_version"] and ctx.facts.get("torch_version") and facts["torch_version"] != ctx.facts["torch_version"]:
        src.notes.append(f"plans exported under torch {facts['torch_version']}, venv has {ctx.facts['torch_version']}")
    src.fields = {"manifest_sha256": ek.sha256_file(manifest_path), "files": files,
                  "engine_bytes": sum(v for k, v in files.items() if k.endswith(".plan")),
                  "probe_output_sha256": facts["probe_output_sha256"],
                  "stagewise_torch_version": facts["torch_version"],
                  "stagewise_gpu": facts["gpu_name"],
                  "original_dir": str(bs_dir), "original_realpath": os.path.realpath(str(bs_dir))}
    return src


def source_taesd(ctx: Ctx, meta_path: Path, batch: Optional[int], allow_same_cc: bool) -> Source:
    try:
        meta = ek.read_json(meta_path)
    except ValueError as exc:
        src = Source(TAESD, batch or 0, str(meta_path))
        src.problems.append(str(exc))
        return src
    src = Source(TAESD, int(batch or ((meta or {}).get("fingerprint") or {}).get("batch") or 0), str(meta_path))
    if not isinstance(meta, dict):
        src.problems.append(f"no TAESD TRT meta at {meta_path}")
        return src
    facts = ek.taesd_meta_facts(meta)
    if not facts["schema_ok"]:
        src.problems.append(f"unexpected TAESD meta schema {facts['schema']!r}")
    if batch and int(facts["batch"] or 0) != int(batch):
        src.problems.append(f"meta batch {facts['batch']} != requested bs{batch}")
    src.problems.extend(ek.host_mismatch(ctx.facts, facts["gpu_name"], facts["compute_capability"],
                                         facts["tensorrt_version"], allow_same_cc=allow_same_cc))
    files: Dict[str, int] = {}
    directory = meta_path.parent
    for field in ("decoder_plan", "post_plan"):
        name = meta.get(field)
        if not name or not (directory / name).exists():
            src.problems.append(f"missing {field} {name!r} next to {meta_path.name}")
            continue
        src.links[name] = os.path.realpath(str(directory / name))
        files[name] = (directory / name).stat().st_size
    src.links[meta_path.name] = os.path.realpath(str(meta_path))
    files[meta_path.name] = meta_path.stat().st_size
    src.fields = {"engine_file": meta_path.name, "runtime_key": meta.get("key"),
                  "runtime_fingerprint": meta.get("fingerprint"), "files": files,
                  "engine_bytes": sum(v for k, v in files.items() if k.endswith(".plan")),
                  "decoder_plan_sha256": meta.get("decoder_plan_sha256"),
                  "post_plan_sha256": meta.get("post_plan_sha256"), "gate_verdict": facts["gate_verdict"],
                  "original_path": str(meta_path)}
    return src


def source_from_entry(ctx: Ctx, kind: str, entry: Path, allow_same_cc: bool) -> Source:
    """Another store entry (same cc/TRT, different GPU name) as an adopt source."""
    fingerprint = ek.read_fingerprint(entry) or {}
    batch = int(fingerprint.get("batch") or 0)
    if kind == UNET_TS:
        return source_unet_ts(ctx, entry / ek.UNET_TS_ENGINE_FILE, batch, allow_same_cc,
                              meta_path=entry / ek.UNET_TS_META_FILE)
    if kind == STAGEWISE:
        return source_stagewise(ctx, entry, batch, allow_same_cc)
    return source_taesd(ctx, entry / str(fingerprint.get("engine_file")), batch, allow_same_cc)


def materialize(partial_root: Path, source: Source) -> Path:
    target = partial_root / f"bs{source.batch}"
    target.mkdir(parents=True, exist_ok=True)
    for name, link_target in sorted(source.links.items()):
        os.symlink(link_target, str(target / name))
    for name, data in sorted(source.copies.items()):
        (target / name).write_bytes(data)
    return target


# --------------------------------------------------------------------------- adopt
def adopt(ctx: Ctx, kind: str, source: Source, no_validate: bool = False, force: bool = False,
          corpus: Optional[str] = None, timeout_min: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
    key = ctx.require_key(kind)
    result: Dict[str, Any] = {"kind": kind, "key": key, "batch": source.batch, "source": source.origin,
                              "problems": source.problems, "notes": source.notes}
    if source.problems:
        log(f"refusing to adopt {source.origin}: " + "; ".join(source.problems))
        result["status"] = "refused"
        return EXIT_REFUSED, result
    if not source.batch:
        result.update(status="refused", problems=["engine batch unknown"])
        return EXIT_REFUSED, result
    store = ctx.store(kind)
    final = store / key / f"bs{source.batch}"
    with store_lock(store, key):
        existing = None
        try:
            existing = ek.read_fingerprint(final) if final.exists() else None
        except ValueError:
            existing = None
        same_origin = bool(existing) and _same_origin(existing, source)
        if existing and not force:
            if not same_origin:
                result.update(status="refused", problems=[f"{final} already holds another engine "
                                                          f"(source {existing.get('source')}); use --force to replace"])
                return EXIT_REFUSED, result
            verdict = ek.usable({"dir": str(final), "fingerprint": existing}, ctx.facts)
            if verdict or no_validate:
                result.update(status="already_adopted", entry=str(final), usable=verdict.ok, reasons=verdict.reasons)
                return (EXIT_OK if verdict or no_validate else EXIT_NONE), result
            log(f"{final} already adopted but not usable ({verdict.reason}); re-validating")
            record = run_validation(ctx, kind, final, existing, corpus_override=corpus, timeout_min=timeout_min)
            fingerprint = apply_validation(ctx, kind, final, existing, record)
            return _adopt_result(result, final, fingerprint, record, ctx)
        clean_partials(store, key)
        partial = new_partial(store, key, source.batch)
        try:
            staged = materialize(partial, source)
            fingerprint = ek.make_fingerprint(kind, ctx.facts, source.batch, "adopted", **source.fields)
            fingerprint["notes"] = list(source.notes)
            fingerprint["adopted_utc"] = ek.utc_now()
            ek.write_fingerprint(staged, fingerprint)
            if no_validate:
                record = _validation_base(key, "none", status="not_run", reason="adopt --no-validate")
                fingerprint = apply_validation(ctx, kind, staged, fingerprint, record)
            else:
                record = run_validation(ctx, kind, staged, fingerprint, corpus_override=corpus, timeout_min=timeout_min)
                if record["status"] == "failed":
                    discard_partial(partial, store)
                    result.update(status="validation_failed", validation=record)
                    log(f"adopt of {source.origin} FAILED validation: {record.get('error')}")
                    return EXIT_FAIL, result
                fingerprint = apply_validation(ctx, kind, staged, fingerprint, record)
            final = promote(partial, source.batch, store, key)
        except BaseException:
            discard_partial(partial, store)
            raise
    return _adopt_result(result, final, fingerprint, record, ctx)


def _adopt_result(result: Dict[str, Any], final: Path, fingerprint: Dict[str, Any], record: Dict[str, Any],
                  ctx: Ctx) -> Tuple[int, Dict[str, Any]]:
    verdict = ek.usable({"dir": str(final), "fingerprint": fingerprint}, ctx.facts)
    result.update(status="adopted", entry=str(final), validation=record, usable=verdict.ok, reasons=verdict.reasons)
    if verdict:
        log(f"adopted and validated: {final}")
        return EXIT_OK, result
    status = record.get("status")
    if status == "not_run":
        log(f"adopted {final} WITHOUT validation ({record.get('reason')}); run `validate` on a GPU to use it")
        return EXIT_NONE, result
    return EXIT_FAIL, result


def _same_origin(fingerprint: Dict[str, Any], source: Source) -> bool:
    for field in ("original_realpath",):
        if fingerprint.get(field) and fingerprint.get(field) == source.fields.get(field):
            return True
    if source.kind == TAESD:
        return bool(fingerprint.get("runtime_key")) and fingerprint.get("runtime_key") == source.fields.get("runtime_key")
    return False


# --------------------------------------------------------------------------- build
def build(ctx: Ctx, kind: str, batch: int, force: bool = False, timeout_min: Optional[float] = None,
          max_minutes: Optional[float] = None, corpus: Optional[str] = None,
          publish: bool = False, remote: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    key = ctx.require_key(kind)
    store = ctx.store(kind)
    final = store / key / f"bs{batch}"
    result: Dict[str, Any] = {"kind": kind, "key": key, "batch": batch, "entry": str(final)}
    if kind == UNET_TS and batch != 8:
        raise StoreError("unet_ts builds are static bs8 only (the portable corpus and the runtime path map are bs8)",
                         EXIT_USAGE)
    existing = _described(ctx, kind, final)
    if existing and existing["usable"] and not force:
        log(f"{final} is already usable; nothing to build (use --force to rebuild)")
        result.update(status="already_usable", usable=True)
        return EXIT_OK, result
    pre = preflight(ctx, kind, "build", store)
    result["preflight"] = pre
    for warning in pre["warnings"]:
        log(f"WARNING: {warning}")
    corpus_dir = corpus_path(ctx, corpus)
    if kind in (UNET_TS, STAGEWISE):
        corpus_check = check_corpus(corpus_dir)
        if not corpus_check["ok"]:
            pre["problems"].extend(corpus_check["problems"][:3])
            pre["ok"] = False
    if not pre["ok"]:
        log("build preflight refused: " + "; ".join(pre["problems"]))
        result["status"] = "preflight_refused"
        return EXIT_NONE, result
    timeout_s = 60.0 * (timeout_min or BUILD_LIMITS[kind]["timeout_min"])
    log_path = log_path_for(ctx, kind, key, batch, "build")
    with store_lock(store, key):
        clean_partials(store, key)
        partial = new_partial(store, key, batch, resume=(kind == STAGEWISE))
        staged = partial / f"bs{batch}"
        keep_partial = False
        try:
            if kind == UNET_TS:
                fingerprint, record = _build_unet_ts(ctx, key, batch, staged, corpus_dir, log_path, timeout_s)
            elif kind == STAGEWISE:
                fingerprint, record, keep_partial = _build_stagewise(ctx, key, batch, partial, staged, store,
                                                                     log_path, timeout_s, max_minutes, corpus)
            else:
                fingerprint, record = _build_taesd(ctx, key, batch, staged, log_path, timeout_s)
            if not record.get("passed"):
                result.update(status="failed", validation=record)
                log(f"build of {kind} {key} bs{batch} FAILED: {record.get('error') or record.get('status')}")
                if not keep_partial:
                    discard_partial(partial, store)
                return EXIT_FAIL, result
            ek.write_fingerprint(staged, fingerprint)
            if kind == UNET_TS:
                _sync_ts_meta_validation(staged, fingerprint["validation"])
            final = promote(partial, batch, store, key)
        except StoreError as exc:
            if not keep_partial:
                discard_partial(partial, store)
            result.update(status="failed", error=str(exc))
            return exc.code, result
        except BaseException:
            if kind != STAGEWISE:
                discard_partial(partial, store)
            raise
    result.update(status="built", usable=True, validation=record, log=str(log_path))
    if publish_requested(publish):
        rc, published = publish_entry(ctx, kind, final, remote)
        result["publish"] = published
        if rc != EXIT_OK:
            log("WARNING: build succeeded but publish failed (the local engine is usable)")
    return EXIT_OK, result


def _build_unet_ts(ctx: Ctx, key: str, batch: int, staged: Path, corpus_dir: Path, log_path: Path,
                   timeout_s: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = staged / ek.VALIDATION_FILE
    fail_mae = env_float("MUSETALK_UNET_VALIDATE_FAIL_MAE", DEFAULT_FAIL_MAE)
    fail_max_abs = env_float("MUSETALK_UNET_VALIDATE_FAIL_MAX_ABS", DEFAULT_FAIL_MAX_ABS)
    cmd = [str(ctx.venv_python), str(ctx.script("tensorrt_export.py")), "--components", "unet",
           "--batch-sizes", str(batch), "--output-dir", str(staged), "--precision", "fp16",
           "--save-format", "exported_program", "--workspace-gb", "2", "--min-block-size", "1",
           "--unet-capture-dir", str(corpus_dir), "--validate-unet-capture-dir", str(corpus_dir),
           "--validate-unet-limit", str(DEFAULT_LIMIT), "--validate-unet-padded-batch-size", str(batch),
           "--validate-unet-max-mae", str(fail_mae), "--validate-unet-max-abs", str(fail_max_abs),
           "--validate-unet-report-path", str(report), "--require-valid-unet",
           "--warmup", "2", "--iters", "5"]
    run = run_logged(cmd, child_env({}), ctx.repo_root, log_path, timeout_s, f"build unet_ts {key}")
    engine = staged / ek.UNET_TS_ENGINE_FILE
    corpus_check = check_corpus(corpus_dir, deep=False)
    record = _validation_base(key, "tensorrt_export.py --require-valid-unet (validate_unet_backend --backend trt)",
                              rc=run["rc"], seconds=run["seconds"], log=run["log"], attempts=1,
                              capture_dir=rel_to_repo(corpus_dir, ctx.repo_root),
                              corpus_manifest_sha256=corpus_check["manifest_sha256"],
                              gate={"fail_mae": fail_mae, "fail_max_abs": fail_max_abs},
                              report=ek.VALIDATION_FILE, group_captures=1)
    try:
        payload = ek.read_json(report) if report.exists() else None
    except ValueError:
        payload = None
    summary = (payload or {}).get("summary") or {}
    record.update(mae_max=summary.get("mae_max"), max_abs_max=summary.get("max_abs_max"), files=summary.get("files"))
    if run["rc"] != 0 or not engine.exists() or not payload or not payload.get("passed"):
        record.update(status="failed" if payload and payload.get("passed") is False else "error",
                      error=f"tensorrt_export rc={run['rc']} engine={'present' if engine.exists() else 'missing'} "
                            f"report={'passed' if payload and payload.get('passed') else 'missing/failed'}; "
                            f"tail: {run['tail'][-600:]}")
        return {}, record
    record.update(passed=True, status="passed")
    log("hashing the new engine and scanning its embedded device string")
    scan = ek.scan_embedded_devices(engine, cache_dir=None)
    problems = ek.embedded_device_problems(scan, ctx.facts)
    if problems and scan.get("devices"):
        record.update(passed=False, status="failed", error="built engine targets another device: " + "; ".join(problems))
        return {}, record
    meta = ek.read_json(staged / ek.UNET_TS_META_FILE) or {}
    fingerprint = ek.make_fingerprint(
        UNET_TS, ctx.facts, batch, "built", engine_bytes=engine.stat().st_size,
        engine_sha256=ek.sha256_file(engine),
        embedded_device=scan["devices"][0]["device_string"] if scan.get("devices") else None,
        embedded_devices=[d["device_string"] for d in scan.get("devices") or []],
        save_format=meta.get("save_format"), batch_range=meta.get("batch_range"),
        build={"seconds": run["seconds"], "log": run["log"], "tool": "scripts/tensorrt_export.py"})
    if not scan.get("devices"):
        fingerprint["notes"].append("no embedded device string found in the built engine")
    fingerprint["validation"] = record
    return fingerprint, record


def _build_stagewise(ctx: Ctx, key: str, batch: int, partial: Path, staged: Path, store: Path, log_path: Path,
                     timeout_s: float, max_minutes: Optional[float], corpus: Optional[str]
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    key_dir = store / key
    key_dir.mkdir(parents=True, exist_ok=True)
    opt_level = os.environ.get("MUSETALK_UNET_STAGEWISE_BUILD_OPT_LEVEL", "5").strip() or "5"
    cmd = [str(ctx.venv_python), str(ctx.script("build_unet_stagewise.py")), "--batch", str(batch),
           "--root", str(partial), "--opt-level", opt_level, "--timing-cache", str(key_dir / "timing_cache.bin"),
           "--report", str(partial / "build_report.json")]
    if max_minutes:
        cmd += ["--max-minutes", str(max_minutes)]
    run = run_logged(cmd, child_env({}), ctx.repo_root, log_path, timeout_s, f"build unet_stagewise {key} bs{batch}")
    try:
        manifest = ek.read_json(staged / ek.STAGEWISE_MANIFEST_FILE) or {}
    except ValueError:
        manifest = {}
    if run["rc"] != 0 or not manifest.get("complete"):
        resumable = bool(manifest.get("blocks")) and not manifest.get("complete")
        record = _validation_base(key, "build_unet_stagewise.py", status="error", rc=run["rc"],
                                  error=(f"build rc={run['rc']} complete={manifest.get('complete')} "
                                         f"missing={manifest.get('missing_blocks')}; "
                                         + ("partial kept for resume; " if resumable else "")
                                         + f"tail: {run['tail'][-600:]}"))
        return {}, record, resumable
    source = source_stagewise(ctx, staged, batch, allow_same_cc=False)
    if source.problems:
        return {}, _validation_base(key, "build_unet_stagewise.py", status="failed",
                                    error="built set failed store checks: " + "; ".join(source.problems)), False
    fields = dict(source.fields)
    for name in ("original_dir", "original_realpath"):
        fields.pop(name, None)
    fields["build"] = {"seconds": run["seconds"], "log": run["log"], "tool": "scripts/build_unet_stagewise.py",
                       "opt_level": int(opt_level)}
    fingerprint = ek.make_fingerprint(STAGEWISE, ctx.facts, batch, "built", **fields)
    record = run_validation(ctx, STAGEWISE, staged, fingerprint, corpus_override=corpus)
    fingerprint["validation"] = record
    return fingerprint, record, record.get("status") == "not_run"


def _build_taesd(ctx: Ctx, key: str, batch: int, staged: Path, log_path: Path, timeout_s: float
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extra = {"MUSETALK_TAESD_BACKEND": "trt", "MUSETALK_TAESD_TRT_DIR": str(staged),
             "MUSETALK_TAESD_TRT_BATCH": str(batch), "MUSETALK_TAESD_TRT_BUILD": "1"}
    cmd = [str(ctx.venv_python), str(ctx.script("vae_fast_decoder.py")), "build", "--batch", str(batch)]
    run = run_logged(cmd, child_env(extra, keep=TAESD_RECIPE_KEEP), ctx.repo_root, log_path, timeout_s,
                     f"build taesd_trt {key} bs{batch}")
    metas = sorted(staged.glob("taesd_trt_*.json"))
    verify = last_json_object(run["output"], ("key", "probe"))
    method = "vae_fast_decoder.py build (+ its load/verify)"
    if run["rc"] != 0 or len(metas) != 1 or not verify:
        return {}, _validation_base(key, method, status="error", rc=run["rc"],
                                    error=f"build rc={run['rc']} metas={len(metas)}; tail: {run['tail'][-600:]}")
    source = source_taesd(ctx, metas[0], batch, allow_same_cc=False)
    if source.problems:
        return {}, _validation_base(key, method, status="failed",
                                    error="built engine failed store checks: " + "; ".join(source.problems))
    meta = ek.read_json(metas[0]) or {}
    for field, sha_field in (("decoder_plan", "decoder_plan_sha256"), ("post_plan", "post_plan_sha256")):
        if ek.sha256_file(staged / meta[field]) != meta.get(sha_field):
            return {}, _validation_base(key, method, status="failed", error=f"{field} sha256 differs from its meta")
    fields = dict(source.fields)
    fields.pop("original_path", None)
    fields["build"] = {"seconds": run["seconds"], "log": run["log"], "tool": "scripts/vae_fast_decoder.py build"}
    fingerprint = ek.make_fingerprint(TAESD, ctx.facts, batch, "built", **fields)
    record = _validation_base(key, method, passed=verify.get("key") == meta.get("key"),
                              status="passed" if verify.get("key") == meta.get("key") else "failed",
                              rc=run["rc"], seconds=run["seconds"], log=run["log"], attempts=1,
                              runtime_key=verify.get("key"), probe=verify.get("probe"),
                              gate_verdict=(meta.get("gate") or {}).get("verdict"), report=ek.VALIDATION_FILE)
    ek.write_json_atomic(staged / ek.VALIDATION_FILE, {"store": {"engine_key": key}, "record": record,
                                                       "verify_output": verify})
    fingerprint["validation"] = record
    return fingerprint, record


# --------------------------------------------------------------------------- remote store
def default_remote(kind: str) -> Optional[str]:
    explicit = os.environ.get(REMOTE_ENV[kind], "").strip()
    if explicit:
        return explicit
    base = os.environ.get("MUSETALK_ENGINE_REMOTE_BASE", "").strip()
    if base:
        return base.rstrip("/") + "/" + REMOTE_SUBDIR[kind]
    bucket = os.environ.get("TRT_ARTIFACT_S3_BUCKET", "").strip()
    if bucket:
        return f"s3://{bucket}/trt-artifacts/{REMOTE_SUBDIR[kind]}"
    return None


class Remote:
    """file:///path or s3://bucket/prefix. boto3 is imported only for s3."""

    def __init__(self, uri: str) -> None:
        self.uri = uri.rstrip("/")
        parsed = urlparse(self.uri)
        if parsed.scheme == "file" or (not parsed.scheme and self.uri.startswith("/")):
            self.scheme = "file"
            self.root = Path(parsed.path if parsed.scheme else self.uri)
        elif parsed.scheme == "s3" and parsed.netloc:
            self.scheme = "s3"
            self.bucket = parsed.netloc
            self.prefix = parsed.path.strip("/")
            self._client = None
        else:
            raise StoreError(f"unsupported remote {uri!r} (expected s3://bucket/prefix or file:///path)", EXIT_USAGE)

    def _s3(self):
        if self._client is None:
            try:
                import boto3  # noqa: WPS433 (lazy on purpose)
                from botocore.config import Config
            except ModuleNotFoundError as exc:
                raise StoreError("boto3/botocore are required for s3:// engine remotes", EXIT_FAIL) from exc
            self._client = boto3.client(
                "s3",
                region_name=(os.environ.get("TRT_ARTIFACT_S3_REGION") or os.environ.get("AVATAR_S3_REGION")
                             or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None),
                endpoint_url=os.environ.get("TRT_ARTIFACT_S3_ENDPOINT_URL") or None,
                config=Config(connect_timeout=10, read_timeout=300, retries={"max_attempts": 3, "mode": "standard"}),
            )
        return self._client

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def describe(self, rel: str) -> str:
        return f"{self.uri}/{rel}"

    def read_json(self, rel: str) -> Optional[Dict[str, Any]]:
        if self.scheme == "file":
            try:
                return ek.read_json(self.root / rel)
            except ValueError as exc:
                raise StoreError(f"remote fingerprint unreadable: {exc}", EXIT_REFUSED) from exc
        try:
            body = self._s3().get_object(Bucket=self.bucket, Key=self._key(rel))["Body"].read()
        except Exception as exc:  # botocore ClientError without importing botocore eagerly
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise StoreError(f"cannot read {self.describe(rel)}: {type(exc).__name__}: {exc}", EXIT_FAIL) from exc
        try:
            return json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"remote fingerprint unreadable: {exc}", EXIT_REFUSED) from exc

    def download(self, rel: str, dest: Path) -> None:
        if self.scheme == "file":
            shutil.copyfile(str(self.root / rel), str(dest))
            return
        self._s3().download_file(self.bucket, self._key(rel), str(dest))

    def upload(self, src: Path, rel: str) -> None:
        if self.scheme == "file":
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            shutil.copyfile(str(src), str(tmp))
            os.replace(str(tmp), str(target))
            return
        self._s3().upload_file(str(src), self.bucket, self._key(rel))


def remote_rel(key: str, batch: int, name: str) -> str:
    return f"{key}/bs{batch}/{name}"


def archive_members(kind: str, entry: Path, fingerprint: Dict[str, Any]) -> List[str]:
    if kind == UNET_TS:
        return [ek.UNET_TS_ENGINE_FILE, ek.UNET_TS_META_FILE]
    return sorted((fingerprint.get("files") or {}).keys())


def make_archive(kind: str, entry: Path, fingerprint: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Uncompressed tar of the engine files (symlinks dereferenced; regular files only)."""
    members = archive_members(kind, entry, fingerprint)
    with tarfile.open(str(dest), "w", dereference=True) as tar:
        for name in members:
            path = entry / name
            if not path.exists():
                raise StoreError(f"cannot publish: {path} missing", EXIT_FAIL)
            info = tar.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(str(path), "rb") as handle:
                tar.addfile(info, handle)
    return {"file": ARCHIVE_NAME[kind], "sha256": ek.sha256_file(dest), "bytes": dest.stat().st_size,
            "members": members}


def safe_extract(tar_path: Path, dest: Path, allowed: List[str]) -> None:
    with tarfile.open(str(tar_path), "r") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        for member in members:
            if not member.isfile():
                raise StoreError(f"archive member {member.name!r} is not a regular file", EXIT_REFUSED)
            if member.name not in allowed or "/" in member.name or member.name.startswith("."):
                raise StoreError(f"unexpected archive member {member.name!r}", EXIT_REFUSED)
        missing = sorted(set(allowed) - set(names))
        if missing:
            raise StoreError(f"archive lacks {missing}", EXIT_REFUSED)
        for member in members:
            source = tar.extractfile(member)
            if source is None:
                raise StoreError(f"cannot read archive member {member.name}", EXIT_REFUSED)
            with open(str(dest / member.name), "wb") as handle:
                shutil.copyfileobj(source, handle, 16 << 20)


def publish_entry(ctx: Ctx, kind: str, entry: Path, remote_uri: Optional[str], force: bool = False
                  ) -> Tuple[int, Dict[str, Any]]:
    fingerprint = ek.read_fingerprint(entry)
    result: Dict[str, Any] = {"kind": kind, "entry": str(entry)}
    if not fingerprint:
        log(f"publish: no store entry at {entry}")
        result.update(status="not_found", reason="no store entry (fingerprint.json) to publish")
        return EXIT_NONE, result
    verdict = ek.usable({"dir": str(entry), "fingerprint": fingerprint}, ctx.facts)
    if not verdict:
        log(f"refusing to publish {entry}: not validated on this host ({verdict.reason})")
        result.update(status="refused", reason=verdict.reason)
        return EXIT_REFUSED, result
    uri = remote_uri or default_remote(kind)
    if not uri:
        result.update(status="no_remote", reason=f"set {REMOTE_ENV[kind]} or TRT_ARTIFACT_S3_BUCKET, or pass --remote")
        log("publish: no remote configured")
        return EXIT_NONE, result
    remote = Remote(uri)
    key, batch = str(fingerprint["engine_key"]), int(fingerprint["batch"])
    store = ctx.store(kind)
    if kind == UNET_TS and not fingerprint.get("engine_sha256"):
        log("hashing the engine for the remote fingerprint (adopted engines have none)")
        fingerprint["engine_sha256"] = ek.sha256_file(entry / ek.UNET_TS_ENGINE_FILE)
        ek.write_fingerprint(entry, fingerprint)
    existing = remote.read_json(remote_rel(key, batch, ek.FINGERPRINT_FILE))
    identity = _engine_identity(kind, fingerprint)
    if existing and not force:
        if _engine_identity(kind, existing) == identity:
            log(f"{remote.describe(remote_rel(key, batch, ''))} already holds this engine")
            result.update(status="already_published", remote=remote.describe(remote_rel(key, batch, "")))
            return EXIT_OK, result
        result.update(status="refused", reason="remote holds a different engine for this key; use --force")
        return EXIT_REFUSED, result
    need_gb = (sum(os.path.getsize(str(entry / m)) for m in archive_members(kind, entry, fingerprint)) / 1e9) + 0.5
    free_gb = host_resources(ctx, store)["disk_free_gb"]
    if free_gb < need_gb:
        result.update(status="failed", reason=f"need {need_gb:.1f} GB free for the archive, have {free_gb}")
        return EXIT_FAIL, result
    tmp_dir = store / f".publish-{os.getpid()}-{time.time_ns()}"
    tmp_dir.mkdir(parents=True)
    try:
        archive = tmp_dir / ARCHIVE_NAME[kind]
        started = time.time()
        info = make_archive(kind, entry, fingerprint, archive)
        remote_fp = dict(fingerprint)
        remote_fp["archive"] = info
        remote_fp["published_utc"] = ek.utc_now()
        remote_fp["published_from"] = {"gpu_name": ctx.facts.get("gpu_name"), "source": fingerprint.get("source")}
        fp_file = tmp_dir / ek.FINGERPRINT_FILE
        fp_file.write_text(json.dumps(remote_fp, indent=1) + "\n")
        log(f"uploading {info['bytes'] / 1e9:.2f} GB to {remote.describe(remote_rel(key, batch, ''))}")
        remote.upload(archive, remote_rel(key, batch, ARCHIVE_NAME[kind]))
        remote.upload(fp_file, remote_rel(key, batch, ek.FINGERPRINT_FILE))  # last: its presence means complete
        result.update(status="published", remote=remote.describe(remote_rel(key, batch, "")),
                      archive=info, seconds=round(time.time() - started, 1))
    finally:
        safe_rmtree(tmp_dir, store)
    return EXIT_OK, result


def _engine_identity(kind: str, fingerprint: Dict[str, Any]) -> Any:
    if kind == UNET_TS:
        return fingerprint.get("engine_sha256")
    if kind == STAGEWISE:
        return fingerprint.get("probe_output_sha256"), fingerprint.get("manifest_sha256")
    return fingerprint.get("decoder_plan_sha256"), fingerprint.get("post_plan_sha256")


def restore(ctx: Ctx, kind: str, batch: int, remote_uri: Optional[str], no_validate: bool = False,
            force: bool = False, corpus: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    key = ctx.require_key(kind)
    store = ctx.store(kind)
    final = store / key / f"bs{batch}"
    result: Dict[str, Any] = {"kind": kind, "key": key, "batch": batch, "entry": str(final)}
    existing = _described(ctx, kind, final)
    if existing and existing["usable"] and not force:
        result.update(status="already_usable", usable=True)
        return EXIT_OK, result
    uri = remote_uri or default_remote(kind)
    if not uri:
        result.update(status="no_remote")
        log("restore: no remote configured")
        return EXIT_NONE, result
    remote = Remote(uri)
    result["remote"] = remote.describe(remote_rel(key, batch, ""))
    remote_fp = remote.read_json(remote_rel(key, batch, ek.FINGERPRINT_FILE))
    if not remote_fp:
        log(f"restore: {result['remote']} has no engine for this key")
        result["status"] = "not_found"
        return EXIT_NONE, result
    problems = ek.fingerprint_problems(remote_fp, kind)
    if remote_fp.get("engine_key") != key:
        problems.append(f"remote fingerprint key {remote_fp.get('engine_key')} != {key}")
    if int(remote_fp.get("batch") or 0) != batch:
        problems.append(f"remote batch {remote_fp.get('batch')} != bs{batch}")
    archive_info = remote_fp.get("archive") or {}
    if not archive_info.get("sha256") or not archive_info.get("bytes"):
        problems.append("remote fingerprint has no archive sha256/bytes")
    if problems:
        result.update(status="refused", problems=problems)
        return EXIT_REFUSED, result
    need_gb = 2.0 * float(archive_info["bytes"]) / 1e9 + 0.5
    free_gb = host_resources(ctx, store)["disk_free_gb"]
    if free_gb < need_gb:
        result.update(status="preflight_refused", problems=[f"need {need_gb:.1f} GB free, have {free_gb}"])
        return EXIT_NONE, result
    with store_lock(store, key):
        clean_partials(store, key)
        partial = new_partial(store, key, batch)
        staged = partial / f"bs{batch}"
        try:
            archive = partial / ARCHIVE_NAME[kind]
            started = time.time()
            log(f"downloading {float(archive_info['bytes']) / 1e9:.2f} GB from {result['remote']}")
            remote.download(remote_rel(key, batch, ARCHIVE_NAME[kind]), archive)
            digest = ek.sha256_file(archive)
            if digest != archive_info["sha256"] or archive.stat().st_size != int(archive_info["bytes"]):
                raise StoreError(f"archive checksum mismatch ({digest[:12]} != {str(archive_info['sha256'])[:12]})",
                                 EXIT_REFUSED)
            safe_extract(archive, staged, list(archive_info.get("members") or archive_members(kind, staged, remote_fp)))
            archive.unlink()
            _verify_restored(ctx, kind, staged, remote_fp)
            fingerprint = dict(remote_fp)
            for field in ("archive", "published_utc", "published_from"):
                fingerprint.pop(field, None)
            fingerprint.update(source="restored", restored_from=result["remote"], restored_utc=ek.utc_now(),
                               remote_validation=remote_fp.get("validation"),
                               validation=ek.empty_validation(note="restored; must be validated on this host"))
            for field in ("original_path", "original_realpath", "original_dir"):
                fingerprint.pop(field, None)
            fingerprint["notes"] = list(fingerprint.get("notes") or []) + [
                f"restored from {result['remote']} (published by {(remote_fp.get('published_from') or {}).get('gpu_name')})"]
            ek.write_fingerprint(staged, fingerprint)
            result["download_seconds"] = round(time.time() - started, 1)
            if no_validate:
                record = _validation_base(key, "none", status="not_run", reason="restore --no-validate")
            else:
                record = run_validation(ctx, kind, staged, fingerprint, corpus_override=corpus)
                if record["status"] == "failed":
                    discard_partial(partial, store)
                    result.update(status="validation_failed", validation=record)
                    return EXIT_FAIL, result
            fingerprint = apply_validation(ctx, kind, staged, fingerprint, record)
            final = promote(partial, batch, store, key)
        except StoreError as exc:
            discard_partial(partial, store)
            result.update(status="refused" if exc.code == EXIT_REFUSED else "failed", error=str(exc))
            log(f"restore failed: {exc}")
            return exc.code, result
        except BaseException:
            discard_partial(partial, store)
            raise
    verdict = ek.usable({"dir": str(final), "fingerprint": fingerprint}, ctx.facts)
    result.update(status="restored", usable=verdict.ok, reasons=verdict.reasons, validation=record)
    return (EXIT_OK if verdict else (EXIT_NONE if record.get("status") == "not_run" else EXIT_FAIL)), result


def _verify_restored(ctx: Ctx, kind: str, staged: Path, fingerprint: Dict[str, Any]) -> None:
    """Integrity + host compatibility of extracted files before anything is registered."""
    if kind == UNET_TS:
        engine = staged / ek.UNET_TS_ENGINE_FILE
        if fingerprint.get("engine_sha256") and ek.sha256_file(engine) != fingerprint["engine_sha256"]:
            raise StoreError("restored engine sha256 differs from the fingerprint", EXIT_REFUSED)
        scan = ek.scan_embedded_devices(engine, cache_dir=None)
        problems = ek.embedded_device_problems(scan, ctx.facts)
        if problems and scan.get("devices"):
            raise StoreError("restored engine targets another GPU: " + "; ".join(problems), EXIT_REFUSED)
        return
    files = fingerprint.get("files") or {}
    for name, size in files.items():
        if (staged / name).stat().st_size != int(size):
            raise StoreError(f"restored {name} size differs from the fingerprint", EXIT_REFUSED)
    if kind == STAGEWISE:
        if ek.sha256_file(staged / ek.STAGEWISE_MANIFEST_FILE) != fingerprint.get("manifest_sha256"):
            raise StoreError("restored manifest.json sha256 differs from the fingerprint", EXIT_REFUSED)
        manifest = ek.read_json(staged / ek.STAGEWISE_MANIFEST_FILE) or {}
        for name, block in (manifest.get("blocks") or {}).items():
            sha = block.get("engine_sha256")
            if sha and ek.sha256_file(staged / block["engine_file"]) != sha:
                raise StoreError(f"restored plan {block['engine_file']} sha256 mismatch", EXIT_REFUSED)
        facts = ek.stagewise_manifest_facts(manifest)
        problems = ek.host_mismatch(ctx.facts, facts["gpu_name"], facts["compute_capability"], facts["tensorrt_version"])
        if problems:
            raise StoreError("restored plans do not match this host: " + "; ".join(problems), EXIT_REFUSED)
        return
    for name, sha_field in (("decoder", "decoder_plan_sha256"), ("post", "post_plan_sha256")):
        sha = fingerprint.get(sha_field)
        matches = [n for n in files if n.endswith(f".{name}.plan") or n.endswith(f".{name}_bgr_u8.plan")]
        for plan in matches:
            if sha and ek.sha256_file(staged / plan) != sha:
                raise StoreError(f"restored {plan} sha256 mismatch", EXIT_REFUSED)


# --------------------------------------------------------------------------- ensure
def _described(ctx: Ctx, kind: str, entry: Path, allow_same_cc: bool = False) -> Optional[Dict[str, Any]]:
    if not entry.is_dir():
        return None
    for item in ek.list_entries(kind, ctx.repo_root, ctx.store_arg):
        if os.path.abspath(item["dir"]) == os.path.abspath(str(entry)):
            return ek.describe_entry(item, ctx.facts, allow_same_cc=allow_same_cc)
    return None


def ensure(ctx: Ctx, kind: str, batch: int, provision: str, require: bool = False, publish: bool = False,
           remote: Optional[str] = None, corpus: Optional[str] = None, allow_same_cc: bool = False
           ) -> Tuple[int, Dict[str, Any]]:
    started = time.time()
    steps: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"kind": kind, "batch": batch, "provision": provision, "steps": steps}

    def step(name: str, status: str, t0: float, **detail: Any) -> None:
        entry = {"step": name, "status": status, "seconds": round(time.time() - t0, 1)}
        entry.update(detail)
        steps.append(entry)
        log(f"ensure {kind} step={name} status={status} ({entry['seconds']}s)"
            + (f": {detail.get('reason')}" if detail.get("reason") else ""))

    def finish(code: int) -> Tuple[int, Dict[str, Any]]:
        found = ek.find_engine(kind, ctx.facts, ctx.repo_root, batch=batch, store=ctx.store_arg)
        result["entry"] = found
        result["result"] = "usable" if found else "none"
        result["seconds"] = round(time.time() - started, 1)
        if found:
            return EXIT_OK, result
        if require:
            log(f"ensure {kind}: no usable engine and --require was given")
            return EXIT_USAGE, result
        return (code if code not in (EXIT_OK,) else EXIT_NONE), result

    t0 = time.time()
    key, key_error = ctx.key(kind)
    result["key"] = key
    if key is None:
        step("check", "none", t0, reason=f"no engine key on this host: {key_error}")
        return finish(EXIT_NONE)
    if ek.find_engine(kind, ctx.facts, ctx.repo_root, batch=batch, store=ctx.store_arg):
        step("check", "usable", t0)
        return finish(EXIT_OK)
    step("check", "none", t0)
    if provision == "off":
        step("provision", "skipped", time.time(), reason="--provision off")
        return finish(EXIT_NONE)
    store = ctx.store(kind)
    final = store / key / f"bs{batch}"
    max_attempts = int(env_float("MUSETALK_ENGINE_VALIDATE_MAX_ATTEMPTS", 3))

    # 0. an entry for this key exists but was never (successfully) validated here
    t0 = time.time()
    try:
        existing = ek.read_fingerprint(final) if (final / ek.FINGERPRINT_FILE).exists() else None
    except ValueError as exc:
        log(f"WARNING: unreadable fingerprint in {final} ({exc}); it will be replaced")
        existing = {"validation": {"status": "failed"}, "_broken": True}
    if final.exists() and existing is None:
        existing = {"validation": {"status": "failed"}, "_broken": True}
    stale = False
    if existing and not existing.get("_broken") and (existing.get("validation") or {}).get("passed"):
        verdict = ek.usable({"dir": str(final), "fingerprint": existing}, ctx.facts)
        stale = not verdict
        if stale:
            step("revalidate", "stale", t0, reason=f"validated entry no longer matches its files ({verdict.reason}); "
                                                  "re-adopting / rebuilding")
            t0 = time.time()
    if existing and not existing.get("_broken") and not stale:
        validation = existing.get("validation") or {}
        status = validation.get("status") or "not_run"
        attempts = int(validation.get("attempts") or 0)
        if status in ("not_run", "error") and attempts < max_attempts:
            with store_lock(store, key):
                record = run_validation(ctx, kind, final, existing, corpus_override=corpus)
                apply_validation(ctx, kind, final, existing, record)
            step("revalidate", record["status"], t0, reason=record.get("error") or record.get("reason"))
            if record.get("passed"):
                return finish(EXIT_OK)
        else:
            step("revalidate", "skipped", t0, reason=f"entry status {status} after {attempts} attempt(s); "
                                                   "rebuild with `build --force` or re-adopt with --force")

    # 1. adopt engines that already exist on disk
    if provision in ("auto", "adopt"):
        t0 = time.time()
        adopted = False
        for candidate in _adopt_candidates(ctx, kind, batch, allow_same_cc):
            if existing and not existing.get("_broken") and not stale and _same_origin(existing, candidate):
                status = (existing.get("validation") or {}).get("status")
                reason = ("this engine already FAILED validation here; `adopt --force` to retry" if status == "failed"
                          else f"already registered as {final} (status {status}); see the revalidate step")
                step("adopt", "skipped", t0, source=candidate.origin, reason=reason)
                t0 = time.time()
                continue
            try:
                rc, detail = adopt(ctx, kind, candidate, corpus=corpus, force=bool(existing))
            except StoreError as exc:
                rc, detail = exc.code, {"status": "failed", "problems": [str(exc)]}
            step("adopt", detail.get("status", str(rc)), t0, source=candidate.origin,
                 reason="; ".join(detail.get("problems") or []) or (detail.get("validation") or {}).get("error"))
            t0 = time.time()
            if rc == EXIT_OK:
                adopted = True
                break
        if adopted:
            return finish(EXIT_OK)
        if not steps or steps[-1]["step"] != "adopt":
            step("adopt", "skipped", t0, reason="no compatible engine on disk")

    # 2. restore from the remote store
    if provision in ("auto", "restore"):
        t0 = time.time()
        uri = remote or default_remote(kind)
        if not uri:
            step("restore", "skipped", t0, reason=f"no remote ({REMOTE_ENV[kind]} / TRT_ARTIFACT_S3_BUCKET unset)")
        else:
            try:
                rc, detail = restore(ctx, kind, batch, uri, corpus=corpus, force=bool(existing))
            except StoreError as exc:
                rc, detail = exc.code, {"status": "failed", "error": str(exc)}
            step("restore", detail.get("status", str(rc)), t0, reason=detail.get("error"))
            if rc == EXIT_OK:
                return finish(EXIT_OK)

    # 3. build when resources allow
    if provision in ("auto", "build"):
        t0 = time.time()
        if provision == "auto" and not env_on("MUSETALK_ENGINE_AUTO_BUILD", True):
            step("build", "skipped", t0, reason="MUSETALK_ENGINE_AUTO_BUILD=0")
        elif kind == UNET_TS and batch != 8:
            step("build", "skipped", t0, reason="unet_ts builds are bs8 only")
        else:
            try:
                rc, detail = build(ctx, kind, batch, force=bool(existing), corpus=corpus, publish=publish,
                                   remote=remote)
            except StoreError as exc:
                rc, detail = exc.code, {"status": "failed", "error": str(exc)}
            reason = detail.get("error") or "; ".join((detail.get("preflight") or {}).get("problems") or []) \
                or (detail.get("validation") or {}).get("error")
            step("build", detail.get("status", str(rc)), t0, reason=reason or None)
            if rc == EXIT_OK:
                return finish(EXIT_OK)
            if rc == EXIT_FAIL:
                return finish(EXIT_NONE)
    return finish(EXIT_NONE)


def _adopt_candidates(ctx: Ctx, kind: str, batch: int, allow_same_cc: bool) -> List[Source]:
    """Compatible legacy engines first, then same-cc store entries (only with allow_same_cc)."""
    out: List[Source] = []
    for item in ek.legacy_candidates(kind, ctx.repo_root, ctx.store_arg):
        if item.get("error"):
            continue
        if kind == UNET_TS:
            source = source_unet_ts(ctx, Path(item["path"]), batch, allow_same_cc=False)
        elif kind == STAGEWISE:
            if int(item.get("batch") or 0) != batch:
                continue
            source = source_stagewise(ctx, Path(item["path"]), batch, allow_same_cc=False)
        else:
            if int(item.get("batch") or 0) != batch:
                continue
            source = source_taesd(ctx, Path(item["path"]), batch, allow_same_cc=False)
        if source.problems:
            log(f"skip adopt candidate {item['path']}: " + "; ".join(source.problems))
            continue
        out.append(source)
    if allow_same_cc:
        key = ctx.require_key(kind)
        for entry in ek.list_entries(kind, ctx.repo_root, ctx.store_arg):
            if entry["key"] == key or entry["batch"] != batch or not ek.same_cc_key(entry["key"], key):
                continue
            fp = entry.get("fingerprint") or {}
            if not (fp.get("validation") or {}).get("passed"):
                continue
            source = source_from_entry(ctx, kind, Path(entry["dir"]), allow_same_cc=True)
            if not source.problems:
                source.notes.append(f"same-cc adoption from store entry {entry['key']}")
                out.append(source)
    return out


# --------------------------------------------------------------------------- CLI commands
def emit(obj: Dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(obj, indent=1, default=str)
    print(text)
    out = getattr(args, "json_out", None)
    if out:
        ek.write_json_atomic(Path(out), obj)


def kinds_for(args: argparse.Namespace) -> List[str]:
    return list(KINDS) if getattr(args, "kind", None) in (None, "all") else [args.kind]


def cmd_key(ctx: Ctx, args: argparse.Namespace) -> int:
    kind = args.kind if args.kind != "all" else UNET_TS
    key, error = ctx.key(kind)
    emit({"kind": kind, "key": key, "error": error, "host": ctx.host_summary()}, args)
    if key is None:
        log(f"no {kind} key: {error}")
    return EXIT_OK if key else EXIT_NONE


def cmd_list(ctx: Ctx, args: argparse.Namespace) -> int:
    out: Dict[str, Any] = {"host": ctx.host_summary(), "kinds": {}}
    for kind in kinds_for(args):
        key, error = ctx.key(kind)
        entries = [ek.describe_entry(e, ctx.facts, allow_same_cc=args.allow_same_cc)
                   for e in ek.list_entries(kind, ctx.repo_root, ctx.store_arg)]
        legacy = ek.legacy_candidates(kind, ctx.repo_root, ctx.store_arg)
        for item in legacy:
            if kind == UNET_TS:
                cache_dir = ctx.store(UNET_TS) / ".scan_cache"
                if args.scan:
                    scan = ek.scan_embedded_devices(item["path"], cache_dir=cache_dir)
                else:
                    scan = _cached_scan(item["path"], cache_dir)
                if scan:
                    item["embedded_devices"] = [d["device_string"] for d in scan.get("devices") or []]
                    item["compatible_problems"] = ek.embedded_device_problems(scan, ctx.facts)
                else:
                    item["embedded_devices"] = None
                    item["compatible_problems"] = ["not scanned yet (list --scan, or adopt/ensure scans it)"]
            elif not item.get("error"):
                item["compatible_problems"] = ek.host_mismatch(ctx.facts, item.get("gpu_name"),
                                                               item.get("compute_capability"),
                                                               item.get("tensorrt_version"))
        best = ek.find_engine(kind, ctx.facts, ctx.repo_root, store=ctx.store_arg)
        out["kinds"][kind] = {
            "store": str(ctx.store(kind)), "key": key, "key_error": error, "entries": entries,
            "best": best["dir"] if best else None, "legacy_candidates": legacy,
            "partials": list_partials(ctx.store(kind)),
        }
    emit(out, args)
    return EXIT_OK


def _cached_scan(path: str, cache_dir: Path) -> Optional[Dict[str, Any]]:
    real = os.path.realpath(path)
    cache_file = ek._scan_cache_path(cache_dir, real)  # noqa: SLF001 (same package)
    try:
        cached = ek.read_json(cache_file)
    except ValueError:
        return None
    if not isinstance(cached, dict):
        return None
    stat = os.stat(real)
    if cached.get("size") != stat.st_size or cached.get("mtime_ns") != stat.st_mtime_ns:
        return None
    return cached


def cmd_check_corpus(ctx: Ctx, args: argparse.Namespace) -> int:
    result = check_corpus(corpus_path(ctx, args.corpus), deep=True)
    emit(result, args)
    if result["ok"]:
        log(f"corpus OK: {result['files']} captures in {result['dir']}")
    else:
        log("corpus problems: " + "; ".join(result["problems"][:5]))
    return EXIT_OK if result["ok"] else EXIT_FAIL


def _entry_for_args(ctx: Ctx, args: argparse.Namespace) -> Tuple[str, Path]:
    if args.engine_dir:
        entry = Path(os.path.abspath(args.engine_dir))
        if entry.name == ek.UNET_TS_ENGINE_FILE:
            entry = entry.parent
        fingerprint = ek.read_fingerprint(entry)
        if not fingerprint:
            raise StoreError(f"{entry} is not a store entry (no fingerprint.json); register it with `adopt` first",
                             EXIT_USAGE)
        kind = ek.SCHEMA_TO_KIND.get(fingerprint.get("schema"))
        if not kind:
            raise StoreError(f"{entry}: unknown fingerprint schema {fingerprint.get('schema')!r}", EXIT_USAGE)
        if args.kind not in (None, "all") and args.kind != kind:
            raise StoreError(f"{entry} holds a {kind} engine, not {args.kind}", EXIT_USAGE)
        return kind, entry
    kind = args.kind if args.kind not in (None, "all") else UNET_TS
    key = ctx.require_key(kind)
    batch = ek.preferred_batch(kind, args.batch)
    return kind, ctx.store(kind) / key / f"bs{batch}"


def cmd_validate(ctx: Ctx, args: argparse.Namespace) -> int:
    kind, entry = _entry_for_args(ctx, args)
    fingerprint = ek.read_fingerprint(entry)
    if not fingerprint:
        raise StoreError(f"no store entry at {entry}", EXIT_NONE)
    if args.cudagraphs and kind != UNET_TS:
        raise StoreError("--cudagraphs applies to unet_ts only", EXIT_USAGE)
    store = entry.parent.parent
    with store_lock(store, str(fingerprint["engine_key"])):
        record = run_validation(ctx, kind, entry, fingerprint, mode=args.cudagraphs, corpus_override=args.corpus,
                                timeout_min=args.timeout_min)
        fingerprint = apply_validation(ctx, kind, entry, fingerprint, record, mode=args.cudagraphs)
    verdict = ek.usable({"dir": str(entry), "fingerprint": fingerprint}, ctx.facts)
    emit({"kind": kind, "entry": str(entry), "mode": args.cudagraphs or "plain", "validation": record,
          "usable": verdict.ok, "reasons": verdict.reasons}, args)
    status = record.get("status")
    log(f"validate {kind} {entry}: {status.upper()}"
        + (f" mae_max={record.get('mae_max')} max_abs_max={record.get('max_abs_max')}" if kind != TAESD else ""))
    return {"passed": EXIT_OK, "not_run": EXIT_NONE}.get(status, EXIT_FAIL)


def cmd_adopt(ctx: Ctx, args: argparse.Namespace) -> int:
    kind = args.kind if args.kind not in (None, "all") else None
    path = Path(os.path.abspath(args.ts or args.dir or args.meta or ""))
    if args.ts:
        kind = kind or UNET_TS
    elif args.meta:
        kind = kind or TAESD
    elif args.dir:
        if kind is None:
            if (path / ek.STAGEWISE_MANIFEST_FILE).exists() or list(path.glob("bs*/manifest.json")):
                kind = STAGEWISE
            elif list(path.glob("taesd_trt_*.json")):
                kind = TAESD
            elif (path / ek.UNET_TS_ENGINE_FILE).exists():
                kind = UNET_TS
    else:
        raise StoreError("adopt needs --ts PATH (unet_ts), --dir DIR (stagewise bs dir / TAESD dir) or --meta FILE",
                         EXIT_USAGE)
    if kind is None:
        raise StoreError(f"cannot tell which engine kind {path} holds; pass --kind", EXIT_USAGE)
    if kind == UNET_TS:
        ts = path / ek.UNET_TS_ENGINE_FILE if path.is_dir() else path
        source = source_unet_ts(ctx, ts, args.batch or 8, allow_same_cc=args.allow_same_cc)
    elif kind == STAGEWISE:
        bs_dir = path
        if not (path / ek.STAGEWISE_MANIFEST_FILE).exists():
            batch = ek.preferred_batch(STAGEWISE, args.batch)
            bs_dir = path / f"bs{batch}"
        source = source_stagewise(ctx, bs_dir, args.batch, allow_same_cc=args.allow_same_cc)
    else:
        meta = path
        if path.is_dir():
            metas = sorted(path.glob("taesd_trt_*.json"))
            wanted = ek.preferred_batch(TAESD, args.batch)
            metas = [m for m in metas if int(((ek.read_json(m) or {}).get("fingerprint") or {}).get("batch") or 0) == wanted]
            if len(metas) != 1:
                raise StoreError(f"expected exactly one bs{wanted} taesd_trt_*.json in {path}, found {len(metas)}; "
                                 "pass --meta FILE", EXIT_USAGE)
            meta = metas[0]
        source = source_taesd(ctx, meta, args.batch, allow_same_cc=args.allow_same_cc)
    rc, result = adopt(ctx, kind, source, no_validate=args.no_validate, force=args.force, corpus=args.corpus,
                       timeout_min=args.timeout_min)
    emit(result, args)
    return rc


def cmd_build(ctx: Ctx, args: argparse.Namespace) -> int:
    kind = args.kind if args.kind not in (None, "all") else UNET_TS
    rc, result = build(ctx, kind, ek.preferred_batch(kind, args.batch), force=args.force,
                       timeout_min=args.timeout_min, max_minutes=args.max_minutes, corpus=args.corpus,
                       publish=args.publish, remote=args.remote)
    emit(result, args)
    return rc


def cmd_restore(ctx: Ctx, args: argparse.Namespace) -> int:
    kind = args.kind if args.kind not in (None, "all") else UNET_TS
    rc, result = restore(ctx, kind, ek.preferred_batch(kind, args.batch), args.remote,
                         no_validate=args.no_validate, force=args.force, corpus=args.corpus)
    emit(result, args)
    return rc


def cmd_publish(ctx: Ctx, args: argparse.Namespace) -> int:
    kind, entry = _entry_for_args(ctx, args)
    rc, result = publish_entry(ctx, kind, entry, args.remote, force=args.force)
    emit(result, args)
    return rc


def cmd_ensure(ctx: Ctx, args: argparse.Namespace) -> int:
    kind = args.kind if args.kind not in (None, "all") else UNET_TS
    provision = args.provision or os.environ.get("MUSETALK_UNET_ENGINE_PROVISION", "").strip() or "auto"
    if provision not in ("auto", "adopt", "restore", "build", "off"):
        raise StoreError(f"invalid --provision {provision!r}", EXIT_USAGE)
    rc, result = ensure(ctx, kind, ek.preferred_batch(kind, args.batch), provision, require=args.require,
                        publish=args.publish, remote=args.remote, corpus=args.corpus,
                        allow_same_cc=args.allow_same_cc)
    emit(result, args)
    log(f"ensure {kind}: {result['result']} ({result['seconds']}s)"
        + (f" -> {result['entry']['dir']}" if result.get("entry") else ""))
    return rc


def cmd_clean(ctx: Ctx, args: argparse.Namespace) -> int:
    removed: Dict[str, List[str]] = {}
    for kind in kinds_for(args):
        removed[kind] = clean_partials(ctx.store(kind), include_resume=args.resume)
    emit({"removed": removed}, args)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", default=str(ROOT), help="MuseTalk repo root (default: this checkout)")
    common.add_argument("--store", default=None, help="store root for --kind (default per kind, see module doc)")
    common.add_argument("--kind", choices=list(KINDS) + ["all"], default=None,
                        help="engine kind (default unet_ts; list/clean default all)")
    common.add_argument("--batch", type=int, default=None, help="engine batch (default: kind's runtime default)")
    common.add_argument("--venv", default=None, help="venv whose python runs builds/validation (dist-info versions)")
    common.add_argument("--venv-python", default=None, help="explicit python for subprocesses")
    common.add_argument("--corpus", default=None, help=f"UNet validation corpus (default {DEFAULT_CORPUS_REL})")
    common.add_argument("--json-out", default=None, help="also write the JSON result to this file")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("key", parents=[common], help="print the engine key for this host")
    p = sub.add_parser("list", parents=[common], help="store entries + usability, legacy candidates, partials")
    p.add_argument("--scan", action="store_true", help="scan unscanned legacy .ts files for their device string")
    p.add_argument("--allow-same-cc", action="store_true")
    sub.add_parser("check-corpus", parents=[common], help="verify the portable UNet validation corpus (CPU only)")
    p = sub.add_parser("build", parents=[common], help="build + validate + publish atomically into the store")
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout-min", type=float, default=None)
    p.add_argument("--max-minutes", type=float, default=None, help="unet_stagewise: stop before a new block after M min")
    p.add_argument("--publish", action="store_true", help="publish to the remote after a successful build")
    p.add_argument("--remote", default=None)
    p = sub.add_parser("validate", parents=[common], help="validate a store entry on this GPU")
    p.add_argument("--engine-dir", default=None, help="store entry dir (default: this host's entry for --kind)")
    p.add_argument("--cudagraphs", choices=CUDAGRAPH_MODES, default=None,
                   help="unet_ts: validate through the live loader with MUSETALK_TRT_UNET_CUDAGRAPHS=<mode>")
    p.add_argument("--timeout-min", type=float, default=None)
    p = sub.add_parser("adopt", parents=[common], help="register an existing engine (symlinks; never modified)")
    p.add_argument("--ts", default=None, help="legacy torch_tensorrt unet_trt.ts")
    p.add_argument("--dir", default=None, help="stagewise bs<N> dir (or its root) / TAESD TRT dir")
    p.add_argument("--meta", default=None, help="TAESD taesd_trt_<k>.json")
    p.add_argument("--no-validate", action="store_true", help="register only (CPU); validate later on a GPU")
    p.add_argument("--force", action="store_true", help="replace an existing entry for this key")
    p.add_argument("--allow-same-cc", action="store_true", help="accept an engine from another GPU of the same cc")
    p.add_argument("--timeout-min", type=float, default=None)
    p = sub.add_parser("restore", parents=[common], help="download + verify + validate an engine for this key")
    p.add_argument("--remote", default=None, help="s3://bucket/prefix or file:///path (default from env)")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("publish", parents=[common], help="upload a validated entry (explicit only)")
    p.add_argument("--remote", default=None)
    p.add_argument("--engine-dir", default=None)
    p.add_argument("--force", action="store_true", help="overwrite a different remote engine for this key")
    p = sub.add_parser("ensure", parents=[common], help="make an engine usable: adopt -> restore -> build")
    p.add_argument("--provision", choices=["auto", "adopt", "restore", "build", "off"], default=None,
                   help="default $MUSETALK_UNET_ENGINE_PROVISION or auto")
    p.add_argument("--require", action="store_true", help="exit 2 (not 3) when no engine is usable at the end")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--remote", default=None)
    p.add_argument("--allow-same-cc", action="store_true")
    p = sub.add_parser("clean", parents=[common], help="remove stale partial/old dirs (dead writers)")
    p.add_argument("--resume", action="store_true", help="also remove resumable stagewise partials")
    return parser


COMMANDS = {
    "key": cmd_key, "list": cmd_list, "check-corpus": cmd_check_corpus, "build": cmd_build,
    "validate": cmd_validate, "adopt": cmd_adopt, "restore": cmd_restore, "publish": cmd_publish,
    "ensure": cmd_ensure, "clean": cmd_clean,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctx = Ctx(args)
        return COMMANDS[args.command](ctx, args)
    except StoreError as exc:
        log(f"ERROR: {exc}")
        print(json.dumps({"status": "error", "error": str(exc), **exc.detail}, indent=1, default=str))
        return exc.code
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except Exception as exc:  # unexpected: never a traceback-only failure for the boot log
        log(f"ERROR: unexpected {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=1))
        return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
