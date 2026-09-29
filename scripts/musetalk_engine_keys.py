#!/usr/bin/env python3
"""Engine keys, fingerprints and usability rules for MuseTalk's GPU-bound engines.

Stdlib only (python3 >= 3.8). Never imports torch/tensorrt: every fact comes from
nvidia-smi, the venv's ``site-packages/*.dist-info`` directory names, or an injected
facts JSON (``MUSETALK_HOST_FACTS_JSON``), so the resolver (scripts/musetalk_host_profile.py)
and the engine store (scripts/unet_engine_store.py) can decide without touching CUDA.

Three engine kinds, each keyed by GPU + TensorRT (+ torch_tensorrt for the .ts):

  kind            what                                         default store root
  --------------  -------------------------------------------  -----------------------------------
  unet_ts         torch_tensorrt static bs8 ``unet_trt.ts``     models/tensorrt_unet
  unet_stagewise  11 per-block FP16 TensorRT plans + manifest   models/tensorrt_unet_stagewise
                  (scripts/unet_stagewise_trt.py, built by
                  scripts/build_unet_stagewise.py)
  taesd_trt       TAESD decoder + fused post TensorRT plans     models/taesd/trt
                  (scripts/vae_fast_decoder.py TaesdTrtBackend)

Store roots are overridable per kind: MUSETALK_UNET_ENGINE_STORE, MUSETALK_UNET_STAGEWISE_ENGINE_STORE,
MUSETALK_TAESD_TRT_ENGINE_STORE (relative paths are resolved against the repo root).

Store layout (identical for every kind)::

  <store>/<engine_key>/bs<N>/fingerprint.json     store record (schema per kind, see FINGERPRINT_SCHEMAS)
  <store>/<engine_key>/bs<N>/validation.json      full report of the last validation run
  <store>/<engine_key>/bs<N>/<engine files>       unet_ts:        unet_trt.ts (+ unet_trt_meta.json)
                                                  unet_stagewise: manifest.json, <block>.plan x11, probe_output.pt
                                                  taesd_trt:      taesd_trt_<k>.json/.decoder.plan/.post_bgr_u8.plan
  <store>/.<engine_key>.partial-*                 entries under construction (renamed atomically)

engine_key:
  unet_ts:        sm<major><minor>-<gpu slug>-trt<tensorrt>-tt<torch_tensorrt>
                  e.g. sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0
  unet_stagewise: sm<major><minor>-<gpu slug>-trt<tensorrt>        (raw TensorRT plans)
  taesd_trt:      sm<major><minor>-<gpu slug>-trt<tensorrt>        (raw TensorRT plans)

An engine is USABLE only if its fingerprint says validation.passed is true, the validation was
recorded for the same engine key, the host key matches (exact; or same cc/TRT/torch_tensorrt with
a different GPU name when the caller passes allow_same_cc=True), and its engine files exist with
the recorded sizes (symlinks allowed).

Facts accepted by engine_key()/usable()/find_engine() (any of these shapes):
  * the resolver's ``detect`` JSON: {"gpus": [{index,name,compute_capability,memory_total_mib,...}],
    optional "selected_gpu" (dict or index) / "selected_gpu_index" / "gpu" (dict),
    "venv": {"torch","tensorrt","torch_tensorrt",...}}; with only "gpus", the first
    CUDA_VISIBLE_DEVICES entry (else gpus[0]) is selected;
  * a flat dict: {gpu_name, compute_capability, tensorrt_version, torch_tensorrt_version, torch_version};
  * the output of normalize_facts() itself.

CLI (debug helper; the real CLIs are musetalk_host_profile.py and unet_engine_store.py):
  musetalk_engine_keys.py key --kind unet_ts [--venv V]
  musetalk_engine_keys.py scan PATH [--cache-dir D]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

KIND_UNET_TS = "unet_ts"
KIND_UNET_STAGEWISE = "unet_stagewise"
KIND_TAESD_TRT = "taesd_trt"
KINDS = (KIND_UNET_TS, KIND_UNET_STAGEWISE, KIND_TAESD_TRT)

FINGERPRINT_SCHEMAS = {
    KIND_UNET_TS: "musetalk_unet_engine_v1",
    KIND_UNET_STAGEWISE: "musetalk_unet_stagewise_engine_v1",
    KIND_TAESD_TRT: "musetalk_taesd_trt_engine_v1",
}
SCHEMA_TO_KIND = {schema: kind for kind, schema in FINGERPRINT_SCHEMAS.items()}
FINGERPRINT_FILE = "fingerprint.json"
VALIDATION_FILE = "validation.json"

DEFAULT_STORE_REL = {
    KIND_UNET_TS: "models/tensorrt_unet",
    KIND_UNET_STAGEWISE: "models/tensorrt_unet_stagewise",
    KIND_TAESD_TRT: "models/taesd/trt",
}
STORE_ENV = {
    KIND_UNET_TS: "MUSETALK_UNET_ENGINE_STORE",
    KIND_UNET_STAGEWISE: "MUSETALK_UNET_STAGEWISE_ENGINE_STORE",
    KIND_TAESD_TRT: "MUSETALK_TAESD_TRT_ENGINE_STORE",
}
DEFAULT_BATCH = {KIND_UNET_TS: 8, KIND_UNET_STAGEWISE: 16, KIND_TAESD_TRT: 8}
# Runtime env var that selects the engine batch (the resolver/launcher may set it).
BATCH_ENV = {
    KIND_UNET_TS: "",
    KIND_UNET_STAGEWISE: "MUSETALK_UNET_STAGEWISE_BATCH",
    KIND_TAESD_TRT: "MUSETALK_TAESD_TRT_BATCH",
}

# Engines that existed before the store (adopt candidates; never modified).
LEGACY_UNET_TS_PATHS = (
    "models/tensorrt_unet_sm89_bs8_local/unet_trt.ts",
    "models/tensorrt_unet_static_bs8_20260529/unet_trt.ts",
)
LEGACY_UNET_TS_GLOBS = ("models/tensorrt_unet_*/unet_trt.ts", "models/*/tensorrt_unet_*/unet_trt.ts")
LEGACY_STAGEWISE_GLOBS = ("models/tensorrt_unet_stagewise_*/bs*/manifest.json",)
LEGACY_TAESD_GLOBS = ("models/taesd/trt/taesd_trt_*.json",)

UNET_TS_ENGINE_FILE = "unet_trt.ts"
UNET_TS_META_FILE = "unet_trt_meta.json"
STAGEWISE_MANIFEST_FILE = "manifest.json"
STAGEWISE_MANIFEST_SCHEMA = "musetalk_unet_stagewise_trt_v1"
STAGEWISE_BLOCKS = ("head", "down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "tail")
TAESD_META_SCHEMA = "taesd_trt_engine_v1"

# torch_tensorrt embeds "device%major%minor%type%name" (e.g. 0%8%9%0%NVIDIA GeForce RTX 4070 SUPER).
# The name is restricted to printable ASCII without '%' so trailing binary never leaks into it.
DEVICE_RE = re.compile(rb"(\d{1,3})%(\d{1,3})%(\d{1,3})%(\d)%(NVIDIA[\x20-\x24\x26-\x7e]{1,80})")
SCAN_CHUNK_BYTES = 64 << 20
SCAN_OVERLAP_BYTES = 4096
SCAN_CACHE_SCHEMA = "musetalk_engine_device_scan_v1"

VALIDATION_STATUSES = ("passed", "failed", "error", "not_run")


class EngineKeyError(ValueError):
    """The host facts are missing a field the engine key needs (e.g. no GPU, no TensorRT)."""


# --------------------------------------------------------------------------- small helpers
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slug(name: Any) -> str:
    """Lowercase, every run of non-alphanumerics -> '-', stripped: 'NVIDIA GeForce RTX 4070 SUPER'
    -> 'nvidia-geforce-rtx-4070-super'."""
    text = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower())
    return text.strip("-")


def public_version(value: Any) -> Optional[str]:
    """'2.5.1+cu121' -> '2.5.1'; '' / None -> None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.split("+", 1)[0].strip() or None


def version_tuple(value: Any, width: int = 3) -> Optional[Tuple[int, ...]]:
    text = public_version(value)
    if text is None:
        return None
    parts = []
    for token in text.split("."):
        match = re.match(r"\d+", token)
        if not match:
            break
        parts.append(int(match.group(0)))
    if not parts:
        return None
    parts = (parts + [0] * width)[:width]
    return tuple(parts)


def same_version(a: Any, b: Any, width: int = 3) -> bool:
    ta, tb = version_tuple(a, width), version_tuple(b, width)
    return ta is not None and ta == tb


def parse_cc(value: Any) -> Optional[Tuple[int, int]]:
    """'8.9' | 8.9 | [8, 9] | (8, 9) | 'sm89' | 'sm_89' | '89' | 'sm120' -> (major, minor)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(value, float):
        value = repr(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(?:sm_?|compute_?)?(\d{2,3})", text)
    if match:
        digits = match.group(1)
        return int(digits[:-1]), int(digits[-1])
    return None


def cc_string(cc: Optional[Tuple[int, int]]) -> Optional[str]:
    return f"{cc[0]}.{cc[1]}" if cc else None


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    """tmp file in the same directory + fsync + rename (never writes through a symlink)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; the server may run as another user
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(obj, indent=1, sort_keys=False, default=str))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Optional[Any]:
    """None if missing; raises ValueError on malformed JSON."""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


# --------------------------------------------------------------------------- facts
def _gpu_from_any(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name") or raw.get("gpu_name")
    cc = parse_cc(raw.get("compute_capability", raw.get("compute_cap", raw.get("cc"))))
    if not name and cc is None:
        return None

    def _num(*keys):
        for key in keys:
            value = raw.get(key)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    return {
        "index": raw.get("index"),
        "uuid": raw.get("uuid"),
        "name": str(name).strip() if name else None,
        "compute_capability": cc_string(cc),
        "memory_total_mib": _num("memory_total_mib", "memory.total", "memory_total"),
        "memory_used_mib": _num("memory_used_mib", "memory.used", "memory_used"),
    }


def _visible_first(environ: Optional[Dict[str, str]] = None) -> Tuple[bool, Optional[str]]:
    """(restricted, first token) from CUDA_VISIBLE_DEVICES. restricted=False when unset."""
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" not in env:
        return False, None
    raw = env.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens or tokens[0] in {"-1", "none", "NoDevFiles"}:
        return True, None
    return True, tokens[0]


def _select_gpu(gpus: List[Dict[str, Any]], environ: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    parsed = [gpu for gpu in (_gpu_from_any(g) for g in gpus) if gpu]
    if not parsed:
        return None
    restricted, first = _visible_first(environ)
    if not restricted:
        return parsed[0]
    if first is None:
        return None
    for gpu in parsed:
        if first.isdigit() and str(gpu.get("index")) == first:
            return gpu
        if gpu.get("uuid") and str(gpu["uuid"]) == first:
            return gpu
    if first.isdigit() and int(first) < len(parsed) and all(g.get("index") is None for g in parsed):
        return parsed[int(first)]
    return None


def normalize_facts(facts: Optional[Dict[str, Any]], environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Flatten any accepted facts shape (see module doc) into one dict.

    Keys: has_gpu, gpu_name, gpu_slug, compute_capability ('8.9'), cc (tuple), gpu_index,
    gpu_memory_total_mib, gpu_memory_used_mib, tensorrt_version, torch_tensorrt_version,
    torch_version, torch_cuda_tag, ram (dict, passthrough), disk_free_gb (passthrough).
    """
    facts = dict(facts or {})
    if facts.get("_normalized"):
        return facts
    gpu = None
    # An explicit "gpu" key is the caller's selection (the resolver sets it to None when no GPU is
    # visible, e.g. CUDA_VISIBLE_DEVICES=''), so never re-select from "gpus" in that case.
    explicit = "gpu" in facts
    if isinstance(facts.get("gpu"), dict):
        gpu = _gpu_from_any(facts["gpu"])
    selected = None if explicit else facts.get("selected_gpu", facts.get("selected_gpu_index"))
    gpus = None if explicit else (facts.get("gpus") if isinstance(facts.get("gpus"), list) else None)
    if gpu is None and isinstance(selected, dict):
        gpu = _gpu_from_any(selected)
    if gpu is None and isinstance(selected, int) and not isinstance(selected, bool) and gpus:
        for candidate in gpus:
            if isinstance(candidate, dict) and candidate.get("index") == selected:
                gpu = _gpu_from_any(candidate)
                break
        if gpu is None and 0 <= selected < len(gpus):
            gpu = _gpu_from_any(gpus[selected])
    if gpu is None and gpus is not None and selected is None:
        gpu = _select_gpu(gpus, environ)
    if gpu is None and gpus is None and not explicit and (facts.get("gpu_name") or facts.get("compute_capability")):
        gpu = _gpu_from_any({
            "name": facts.get("gpu_name"),
            "compute_capability": facts.get("compute_capability"),
            "memory_total_mib": facts.get("gpu_memory_total_mib"),
            "memory_used_mib": facts.get("gpu_memory_used_mib"),
            "index": facts.get("gpu_index"),
        })

    venv = facts.get("venv") if isinstance(facts.get("venv"), dict) else {}

    def _pick(*values):
        for value in values:
            if value not in (None, ""):
                return value
        return None

    torch_version = _pick(facts.get("torch_version"), venv.get("torch"), venv.get("torch_version"))
    out: Dict[str, Any] = {
        "_normalized": True,
        "has_gpu": bool(gpu and gpu.get("name") and gpu.get("compute_capability")),
        "gpu_name": gpu.get("name") if gpu else None,
        "gpu_slug": slug(gpu.get("name")) if gpu and gpu.get("name") else None,
        "compute_capability": gpu.get("compute_capability") if gpu else None,
        "cc": parse_cc(gpu.get("compute_capability")) if gpu else None,
        "gpu_index": gpu.get("index") if gpu else None,
        "gpu_memory_total_mib": gpu.get("memory_total_mib") if gpu else None,
        "gpu_memory_used_mib": gpu.get("memory_used_mib") if gpu else None,
        "tensorrt_version": public_version(_pick(facts.get("tensorrt_version"), venv.get("tensorrt"),
                                                 venv.get("tensorrt_version"))),
        "torch_tensorrt_version": public_version(_pick(facts.get("torch_tensorrt_version"),
                                                       venv.get("torch_tensorrt"),
                                                       venv.get("torch_tensorrt_version"))),
        "torch_version": str(torch_version) if torch_version else None,
        "torch_cuda_tag": _pick(facts.get("torch_cuda_tag"), venv.get("torch_cuda_tag"),
                                _cuda_tag(torch_version)),
    }
    for passthrough in ("ram", "disk_free_gb", "machine", "cpu"):
        if passthrough in facts:
            out[passthrough] = facts[passthrough]
    return out


def _cuda_tag(torch_version: Any) -> Optional[str]:
    if not torch_version or "+" not in str(torch_version):
        return None
    return str(torch_version).split("+", 1)[1] or None


def engine_key(kind: str, facts: Dict[str, Any]) -> str:
    """Engine key for this kind on the host described by facts; raises EngineKeyError when incomplete."""
    if kind not in KINDS:
        raise EngineKeyError(f"unknown engine kind {kind!r} (expected one of {', '.join(KINDS)})")
    nf = normalize_facts(facts)
    if not nf["has_gpu"]:
        raise EngineKeyError("no visible NVIDIA GPU in the host facts")
    if not nf["tensorrt_version"]:
        raise EngineKeyError("TensorRT is not installed in the venv (no tensorrt*.dist-info)")
    major, minor = nf["cc"]
    key = f"sm{major}{minor}-{nf['gpu_slug']}-trt{nf['tensorrt_version']}"
    if kind == KIND_UNET_TS:
        if not nf["torch_tensorrt_version"]:
            raise EngineKeyError("torch_tensorrt is not installed in the venv (no torch_tensorrt*.dist-info)")
        key += f"-tt{nf['torch_tensorrt_version']}"
    return key


def engine_key_or_none(kind: str, facts: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    try:
        return engine_key(kind, facts), None
    except EngineKeyError as exc:
        return None, str(exc)


_KEY_RE = re.compile(r"^sm(\d{2,3})-(.+?)-trt([0-9][0-9A-Za-z.]*)(?:-tt([0-9][0-9A-Za-z.]*))?$")


def parse_engine_key(key: str) -> Optional[Dict[str, Any]]:
    """'sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0' -> {cc:(8,9), gpu_slug, tensorrt, torch_tensorrt}."""
    match = _KEY_RE.match(str(key or ""))
    if not match:
        return None
    digits = match.group(1)
    return {
        "cc": (int(digits[:-1]), int(digits[-1])),
        "gpu_slug": match.group(2),
        "tensorrt": match.group(3),
        "torch_tensorrt": match.group(4),
    }


def same_cc_key(a: str, b: str) -> bool:
    """True when two keys differ only in the GPU name (same cc, TensorRT and torch_tensorrt)."""
    pa, pb = parse_engine_key(a), parse_engine_key(b)
    if not pa or not pb:
        return False
    return (pa["cc"] == pb["cc"] and pa["tensorrt"] == pb["tensorrt"]
            and pa["torch_tensorrt"] == pb["torch_tensorrt"])


# --------------------------------------------------------------------------- detection (no torch)
def _site_packages_dirs(venv: Path) -> List[Path]:
    dirs: List[Path] = []
    for lib in ("lib", "lib64"):
        base = venv / lib
        if not base.is_dir():
            continue
        for child in sorted(base.glob("python3*")):
            sp = child / "site-packages"
            if sp.is_dir() and sp not in dirs:
                dirs.append(sp)
    return dirs


def _dist_versions(site_dirs: Iterable[Path]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for sp in site_dirs:
        try:
            names = os.listdir(sp)
        except OSError:
            continue
        for entry in names:
            if not entry.endswith(".dist-info"):
                continue
            stem = entry[: -len(".dist-info")]
            if "-" not in stem:
                continue
            name, version = stem.rsplit("-", 1)
            norm = re.sub(r"[-_.]+", "_", name).lower()
            versions.setdefault(norm, version)
    return versions


def venv_versions(venv: Optional[Path]) -> Dict[str, Any]:
    """Versions read from ``<venv>/lib*/python3*/site-packages/*.dist-info`` names (no imports)."""
    out: Dict[str, Any] = {"path": str(venv) if venv else None, "site_packages": []}
    if not venv:
        return out
    venv = Path(venv)
    site_dirs = _site_packages_dirs(venv)
    out["site_packages"] = [str(p) for p in site_dirs]
    if site_dirs:
        match = re.match(r"python(\d+\.\d+)", site_dirs[0].parent.name)
        out["python_version"] = match.group(1) if match else None
    versions = _dist_versions(site_dirs)
    out["torch"] = versions.get("torch")
    out["torch_cuda_tag"] = _cuda_tag(versions.get("torch"))
    trt = None
    for candidate in ("tensorrt", "tensorrt_cu12", "tensorrt_cu13", "tensorrt_cu11"):
        if versions.get(candidate):
            trt = versions[candidate]
            out["tensorrt_dist"] = candidate
            break
    out["tensorrt"] = trt
    out["torch_tensorrt"] = versions.get("torch_tensorrt")
    out["triton"] = versions.get("triton")
    for extra in ("aiortc", "av", "cffi", "diffusers", "boto3"):
        out[extra] = versions.get(extra)
    return out


def query_nvidia_smi(environ: Optional[Dict[str, str]] = None, timeout_s: float = 15.0) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    env = os.environ if environ is None else environ
    binary = env.get("MUSETALK_NVIDIA_SMI", "").strip() or "nvidia-smi"
    fields = "index,uuid,name,compute_cap,memory.total,memory.used,driver_version"
    try:
        proc = subprocess.run(
            [binary, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False,
            universal_newlines=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"nvidia-smi unavailable: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return [], f"nvidia-smi exit {proc.returncode}: {proc.stderr.strip()[:200]}"
    gpus: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        index, uuid, name, cc, total, used, driver = parts[:7]

        def _f(value):
            try:
                return float(value)
            except ValueError:
                return None

        gpus.append({
            "index": int(index) if index.isdigit() else index,
            "uuid": uuid,
            "name": name,
            "compute_capability": cc,
            "memory_total_mib": _f(total),
            "memory_used_mib": _f(used),
            "driver_version": driver,
        })
    return gpus, None


def detect_engine_facts(venv: Optional[Path] = None, environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Raw facts for engine decisions: GPUs (nvidia-smi) + venv versions (dist-info).

    MUSETALK_HOST_FACTS_JSON=<path> injects facts (tests / offline decisions); fields it provides
    win, missing ones are detected (a present "gpus" key, even empty, disables nvidia-smi)."""
    env = os.environ if environ is None else environ
    facts: Dict[str, Any] = {}
    injected_path = env.get("MUSETALK_HOST_FACTS_JSON", "").strip()
    if injected_path:
        loaded = read_json(Path(injected_path))
        if not isinstance(loaded, dict):
            raise ValueError(f"MUSETALK_HOST_FACTS_JSON={injected_path} does not hold a JSON object")
        facts.update(loaded)
        facts["facts_source"] = f"injected:{injected_path}"
    has_gpu_info = any(k in facts for k in ("gpus", "gpu", "selected_gpu", "gpu_name"))
    if not has_gpu_info:
        gpus, error = query_nvidia_smi(env)
        facts["gpus"] = gpus
        if error:
            facts["gpu_query_error"] = error
    has_venv_info = isinstance(facts.get("venv"), dict) or facts.get("tensorrt_version")
    if not has_venv_info and venv:
        facts["venv"] = venv_versions(Path(venv))
    return facts


# --------------------------------------------------------------------------- store paths
def store_root(kind: str, repo_root: Path, store: Optional[Any] = None,
               environ: Optional[Dict[str, str]] = None) -> Path:
    env = os.environ if environ is None else environ
    raw = str(store) if store else env.get(STORE_ENV[kind], "").strip()
    path = Path(raw) if raw else Path(DEFAULT_STORE_REL[kind])
    if not path.is_absolute():
        path = Path(repo_root) / path
    return Path(os.path.abspath(str(path)))


def entry_dir(kind: str, repo_root: Path, key: str, batch: int, store: Optional[Any] = None) -> Path:
    return store_root(kind, repo_root, store) / key / f"bs{int(batch)}"


def preferred_batch(kind: str, batch: Optional[int] = None, environ: Optional[Dict[str, str]] = None) -> int:
    if batch:
        return int(batch)
    env = os.environ if environ is None else environ
    name = BATCH_ENV.get(kind) or ""
    raw = env.get(name, "").strip() if name else ""
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_BATCH[kind]


# --------------------------------------------------------------------------- fingerprints
def read_fingerprint(path: Any) -> Optional[Dict[str, Any]]:
    """Read <dir>/fingerprint.json (or the given file). None when missing, ValueError when invalid."""
    path = Path(path)
    if path.is_dir() or path.name != FINGERPRINT_FILE:
        path = path / FINGERPRINT_FILE
    data = read_json(path)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: fingerprint is not a JSON object")
    return data


def write_fingerprint(entry_path: Any, fingerprint: Dict[str, Any]) -> Path:
    problems = fingerprint_problems(fingerprint)
    if problems:
        raise ValueError("refusing to write an invalid fingerprint: " + "; ".join(problems))
    path = Path(entry_path) / FINGERPRINT_FILE
    write_json_atomic(path, fingerprint)
    return path


def empty_validation(status: str = "not_run", note: Optional[str] = None) -> Dict[str, Any]:
    out = {"passed": False, "status": status, "engine_key": None, "validated_utc": None,
           "mae_max": None, "max_abs_max": None, "capture_dir": None, "files": None}
    if note:
        out["note"] = note
    return out


def make_fingerprint(kind: str, facts: Dict[str, Any], batch: int, source: str, **fields: Any) -> Dict[str, Any]:
    """Base record for a store entry; kind-specific fields go in **fields."""
    if source not in {"built", "adopted", "restored"}:
        raise ValueError(f"invalid fingerprint source {source!r}")
    nf = normalize_facts(facts)
    fingerprint: Dict[str, Any] = {
        "schema": FINGERPRINT_SCHEMAS[kind],
        "kind": kind,
        "engine_key": engine_key(kind, nf),
        "gpu_name": nf["gpu_name"],
        "compute_capability": nf["compute_capability"],
        "tensorrt_version": nf["tensorrt_version"],
        "torch_version": nf["torch_version"],
        "batch": int(batch),
        "source": source,
        "created_utc": utc_now(),
        "notes": [],
        "validation": empty_validation(),
    }
    if kind == KIND_UNET_TS:
        fingerprint["torch_tensorrt_version"] = nf["torch_tensorrt_version"]
        fingerprint.update({"engine_file": UNET_TS_ENGINE_FILE, "engine_bytes": None,
                            "engine_sha256": None, "embedded_device": None})
    elif kind == KIND_UNET_STAGEWISE:
        fingerprint.update({"engine_file": STAGEWISE_MANIFEST_FILE, "manifest_sha256": None,
                            "files": {}, "engine_bytes": None})
    else:
        fingerprint.update({"engine_file": None, "runtime_key": None, "runtime_fingerprint": None,
                            "files": {}, "engine_bytes": None})
    fingerprint.update(fields)
    return fingerprint


def fingerprint_problems(fingerprint: Any, kind: Optional[str] = None) -> List[str]:
    problems: List[str] = []
    if not isinstance(fingerprint, dict):
        return ["fingerprint is not an object"]
    schema = fingerprint.get("schema")
    fp_kind = SCHEMA_TO_KIND.get(schema)
    if fp_kind is None:
        problems.append(f"unknown fingerprint schema {schema!r}")
    elif kind and fp_kind != kind:
        problems.append(f"fingerprint schema {schema!r} is for kind {fp_kind}, not {kind}")
    for field in ("engine_key", "gpu_name", "compute_capability", "tensorrt_version", "batch", "source"):
        if fingerprint.get(field) in (None, ""):
            problems.append(f"missing {field}")
    if fp_kind == KIND_UNET_TS and not fingerprint.get("torch_tensorrt_version"):
        problems.append("missing torch_tensorrt_version")
    if fp_kind and fingerprint.get("engine_key") and not parse_engine_key(str(fingerprint["engine_key"])):
        problems.append(f"unparseable engine_key {fingerprint.get('engine_key')!r}")
    if not isinstance(fingerprint.get("validation"), dict):
        problems.append("missing validation object")
    return problems


class Usability:
    """Result of usable(): truthy when usable; unpacks as (ok, reason); .match is exact|same_cc|None."""

    __slots__ = ("ok", "match", "reasons", "host_key", "entry_key")

    def __init__(self, ok: bool, match: Optional[str], reasons: List[str],
                 host_key: Optional[str] = None, entry_key: Optional[str] = None) -> None:
        self.ok = bool(ok)
        self.match = match
        self.reasons = list(reasons)
        self.host_key = host_key
        self.entry_key = entry_key

    def __bool__(self) -> bool:
        return self.ok

    @property
    def reason(self) -> str:
        if self.reasons:
            return "; ".join(self.reasons)
        return "usable" if self.match == "exact" else f"usable ({self.match})"

    def __iter__(self) -> Iterator[Any]:
        yield self.ok
        yield self.reason

    def __getitem__(self, index: int) -> Any:
        return (self.ok, self.reason)[index]

    def __len__(self) -> int:
        return 2

    def to_dict(self) -> Dict[str, Any]:
        return {"usable": self.ok, "match": self.match, "reasons": self.reasons,
                "host_key": self.host_key, "entry_key": self.entry_key}

    def __repr__(self) -> str:
        return f"Usability(ok={self.ok}, match={self.match!r}, reasons={self.reasons!r})"


def _entry_parts(entry: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Path], List[str]]:
    """Accept an entry dict (from list_entries), a fingerprint dict, or an entry directory path."""
    problems: List[str] = []
    if isinstance(entry, (str, Path)):
        directory = Path(entry)
        try:
            return read_fingerprint(directory), directory, problems
        except ValueError as exc:
            return None, directory, [str(exc)]
    if isinstance(entry, dict):
        if "schema" in entry and "engine_key" in entry:
            directory = entry.get("_dir")
            return entry, Path(directory) if directory else None, problems
        directory = entry.get("dir")
        fingerprint = entry.get("fingerprint")
        if fingerprint is None and directory:
            try:
                fingerprint = read_fingerprint(Path(directory))
            except ValueError as exc:
                problems.append(str(exc))
        return fingerprint, Path(directory) if directory else None, problems
    return None, None, ["unsupported entry type"]


def engine_file_problems(kind: str, fingerprint: Dict[str, Any], directory: Optional[Path]) -> List[str]:
    """Cheap existence/size checks of the engine files (stat + a 16 KB manifest hash; no engine reads)."""
    if directory is None:
        return ["entry directory unknown"]
    problems: List[str] = []
    if kind == KIND_UNET_TS:
        engine = directory / str(fingerprint.get("engine_file") or UNET_TS_ENGINE_FILE)
        if not engine.exists():
            problems.append(f"engine file missing: {engine}" + (" (dangling symlink)" if engine.is_symlink() else ""))
        else:
            recorded = fingerprint.get("engine_bytes")
            if recorded is not None and engine.stat().st_size != int(recorded):
                problems.append(f"engine size changed: {engine.stat().st_size} != recorded {recorded}")
        if not (directory / UNET_TS_META_FILE).exists():
            problems.append(f"missing {UNET_TS_META_FILE} next to the engine (runtime needs batch_range)")
        return problems
    files = fingerprint.get("files") or {}
    if not files:
        problems.append("fingerprint lists no engine files")
    # The TAESD meta JSON is rewritten in place by the quality-gate script (it records gate.verdict),
    # so its size is not an identity; the plan hashes inside it are (checked below).
    mutable = {str(fingerprint.get("engine_file"))} if kind == KIND_TAESD_TRT else set()
    for name, size in sorted(files.items()):
        path = directory / name
        if not path.exists():
            problems.append(f"engine file missing: {path}" + (" (dangling symlink)" if path.is_symlink() else ""))
            continue
        if size is not None and name not in mutable and path.stat().st_size != int(size):
            problems.append(f"{name} size changed: {path.stat().st_size} != recorded {size}")
    if kind == KIND_UNET_STAGEWISE:
        manifest = directory / STAGEWISE_MANIFEST_FILE
        recorded = fingerprint.get("manifest_sha256")
        if manifest.exists() and recorded:
            if sha256_file(manifest) != recorded:
                problems.append("manifest.json changed since validation (rebuilt?); re-adopt/validate")
        elif not manifest.exists():
            problems.append(f"missing {manifest}")
    if kind == KIND_TAESD_TRT:
        meta_name = fingerprint.get("engine_file")
        if not meta_name or not (directory / str(meta_name)).exists():
            problems.append(f"TAESD TRT meta file missing: {meta_name}")
        else:
            try:
                meta = read_json(directory / str(meta_name)) or {}
            except ValueError as exc:
                meta = {}
                problems.append(str(exc))
            for field in ("decoder_plan_sha256", "post_plan_sha256"):
                recorded = fingerprint.get(field)
                if recorded and meta.get(field) != recorded:
                    problems.append(f"TAESD TRT {field} changed since validation (engine rebuilt); re-validate")
    return problems


def quality_gate(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Quality verdict for the engine's lever, separate from usability (integrity on this GPU).

    unet_ts / unet_stagewise: the store validation IS the accuracy gate (mae/max_abs vs eager captures).
    taesd_trt: gate.verdict written into the engine meta by the G-TAESD gate script (<= 3 LSB max,
    <= 0.2 LSB mean vs compiled TAESD); None when the gate never ran. Re-read on every call because the
    gate script updates the meta after adoption."""
    fingerprint = entry.get("fingerprint") or {}
    kind = entry.get("kind") or SCHEMA_TO_KIND.get(fingerprint.get("schema"))
    validation = fingerprint.get("validation") or {}
    if kind in (KIND_UNET_TS, KIND_UNET_STAGEWISE):
        verdict = "PASS" if validation.get("passed") else ("FAIL" if validation.get("status") == "failed" else None)
        return {"verdict": verdict, "source": "store validation", "mae_max": validation.get("mae_max"),
                "max_abs_max": validation.get("max_abs_max")}
    meta = None
    if entry.get("dir") and fingerprint.get("engine_file"):
        try:
            meta = read_json(Path(entry["dir"]) / str(fingerprint["engine_file"]))
        except ValueError:
            meta = None
    gate = (meta or {}).get("gate") or {}
    return {"verdict": gate.get("verdict"), "source": "engine meta gate (G-TAESD)", "report": gate.get("report"),
            "full_max_lsb": gate.get("G_TAESD_full_max"), "full_mean_lsb": gate.get("G_TAESD_full_mean")}


def usable(entry: Any, facts: Dict[str, Any], allow_same_cc: bool = False,
           check_files: bool = True) -> Usability:
    """Is this store entry usable on the host described by facts?

    entry: entry dict from list_entries(), a fingerprint dict (optionally with "_dir"), or an entry dir.
    Exact key match is required unless allow_same_cc=True, which also accepts an entry whose key
    differs only in the GPU name (match="same_cc"; the resolver should warn)."""
    fingerprint, directory, problems = _entry_parts(entry)
    if fingerprint is None:
        return Usability(False, None, problems or ["no fingerprint.json (not a store entry)"])
    kind = SCHEMA_TO_KIND.get(fingerprint.get("schema"))
    problems.extend(fingerprint_problems(fingerprint))
    entry_key = fingerprint.get("engine_key")
    host_key, key_error = (None, None)
    if kind:
        host_key, key_error = engine_key_or_none(kind, facts)
    if problems:
        return Usability(False, None, problems, host_key, entry_key)
    reasons: List[str] = []
    match: Optional[str] = None
    if key_error:
        reasons.append(f"host facts incomplete: {key_error}")
    elif entry_key == host_key:
        match = "exact"
    elif allow_same_cc and same_cc_key(str(entry_key), str(host_key)):
        match = "same_cc"
    else:
        reasons.append(f"engine key {entry_key} does not match host key {host_key}")
    validation = fingerprint.get("validation") or {}
    if validation.get("passed") is not True:
        status = validation.get("status") or "not_run"
        reasons.append(f"not validated (validation status: {status})")
    elif validation.get("engine_key") != entry_key:
        reasons.append(f"validation was recorded for key {validation.get('engine_key')}, not {entry_key}")
    if check_files and kind:
        reasons.extend(engine_file_problems(kind, fingerprint, directory))
    return Usability(not reasons and match is not None, match if not reasons else None, reasons, host_key, entry_key)


def entry_env(entry: Dict[str, Any]) -> Dict[str, str]:
    """Runtime env that points the server at this entry (suggestion for the resolver)."""
    kind = entry.get("kind")
    directory = Path(entry["dir"])
    batch = int(entry.get("batch") or DEFAULT_BATCH.get(kind, 8))
    if kind == KIND_UNET_TS:
        return {"MUSETALK_TRT_UNET_PATHS": f"{batch}:{directory / UNET_TS_ENGINE_FILE}"}
    if kind == KIND_UNET_STAGEWISE:
        return {"MUSETALK_UNET_STAGEWISE_CACHE_DIR": str(directory.parent),
                "MUSETALK_UNET_STAGEWISE_BATCH": str(batch)}
    if kind == KIND_TAESD_TRT:
        env = {"MUSETALK_TAESD_TRT_DIR": str(directory), "MUSETALK_TAESD_TRT_BATCH": str(batch),
               "MUSETALK_TAESD_TRT_BUILD": "0"}
        runtime_fp = (entry.get("fingerprint") or {}).get("runtime_fingerprint") or {}
        if "opt_level" in runtime_fp:
            env["MUSETALK_TAESD_TRT_OPT_LEVEL"] = str(int(runtime_fp["opt_level"]))
        if "strongly_typed" in runtime_fp:
            env["MUSETALK_TAESD_TRT_STRONGLY_TYPED"] = "1" if runtime_fp["strongly_typed"] else "0"
        return env
    return {}


def list_entries(kind: str, repo_root: Path, store: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Every <store>/<key>/bs<N>/ directory (dot-dirs and partials skipped), with its fingerprint."""
    root = store_root(kind, repo_root, store)
    entries: List[Dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for key_dir in sorted(root.iterdir()):
        if key_dir.name.startswith(".") or not key_dir.is_dir() or not parse_engine_key(key_dir.name):
            continue
        for bs_dir in sorted(key_dir.iterdir()):
            match = re.fullmatch(r"bs(\d+)", bs_dir.name)
            if not match or not bs_dir.is_dir():
                continue
            entry: Dict[str, Any] = {
                "kind": kind, "key": key_dir.name, "batch": int(match.group(1)),
                "dir": str(bs_dir), "layout": "store", "fingerprint_path": str(bs_dir / FINGERPRINT_FILE),
                "fingerprint": None, "error": None,
            }
            try:
                entry["fingerprint"] = read_fingerprint(bs_dir)
            except ValueError as exc:
                entry["error"] = str(exc)
            fp = entry["fingerprint"] or {}
            if fp and fp.get("engine_key") != key_dir.name:
                entry["error"] = f"fingerprint engine_key {fp.get('engine_key')} != directory {key_dir.name}"
            if fp and int(fp.get("batch") or 0) != entry["batch"]:
                entry["error"] = f"fingerprint batch {fp.get('batch')} != directory bs{entry['batch']}"
            entry["source"] = fp.get("source")
            entry["validation"] = fp.get("validation")
            if kind == KIND_UNET_TS:
                entry["engine_path"] = str(bs_dir / UNET_TS_ENGINE_FILE)
            else:
                entry["engine_path"] = str(bs_dir)
            entries.append(entry)
    return entries


def describe_entry(entry: Dict[str, Any], facts: Dict[str, Any], allow_same_cc: bool = False) -> Dict[str, Any]:
    """entry + usability fields + env suggestion (what `list` / `find-engine` print)."""
    out = dict(entry)
    if entry.get("error"):
        verdict = Usability(False, None, [entry["error"]])
    else:
        verdict = usable(entry, facts, allow_same_cc=allow_same_cc)
    out.update({"usable": verdict.ok, "match": verdict.match, "reasons": verdict.reasons})
    out["env"] = entry_env(entry) if verdict.ok else {}
    out["quality_gate"] = quality_gate(entry)
    return out


def find_engine(kind: str, facts: Dict[str, Any], repo_root: Path, batch: Optional[int] = None,
                store: Optional[Any] = None, allow_same_cc: bool = False) -> Optional[Dict[str, Any]]:
    """Best usable entry for this host: exact key first, then same_cc (when allowed); the requested
    batch (or the kind's runtime batch env / default) first, then any other batch."""
    want = preferred_batch(kind, batch)
    candidates = []
    for entry in list_entries(kind, repo_root, store):
        described = describe_entry(entry, facts, allow_same_cc=allow_same_cc)
        if described["usable"]:
            candidates.append(described)
    if not candidates:
        return None
    if batch:
        candidates = [c for c in candidates if int(c["batch"]) == int(batch)]
        if not candidates:
            return None

    def rank(entry: Dict[str, Any]) -> Tuple[int, int, int]:
        return (0 if entry["match"] == "exact" else 1, 0 if int(entry["batch"]) == want else 1, -int(entry["batch"]))

    return sorted(candidates, key=rank)[0]


# --------------------------------------------------------------------------- legacy engines
def _dedupe_real(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        try:
            real = os.path.realpath(str(path))
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        out.append(Path(path))
    return out


def _under(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(str(path))).relative_to(os.path.realpath(str(root)))
        return True
    except ValueError:
        return False


def legacy_candidates(kind: str, repo_root: Path, store: Optional[Any] = None,
                      environ: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Engines outside the store that `adopt` can register (never modified by the store).

    unet_ts:        known .ts paths + globs + MUSETALK_UNET_ADOPT_PATHS (colon list); device facts come
                    from the scan cache when present (see scan_embedded_devices).
    unet_stagewise: <root>/bs<N>/manifest.json written by build_unet_stagewise.py
                    (models/tensorrt_unet_stagewise_*/ and MUSETALK_UNET_STAGEWISE_CACHE_DIR).
    taesd_trt:      flat taesd_trt_<k>.json metas (models/taesd/trt/ and MUSETALK_TAESD_TRT_DIR)."""
    env = os.environ if environ is None else environ
    repo_root = Path(repo_root)
    root = store_root(kind, repo_root, store, env)
    out: List[Dict[str, Any]] = []
    if kind == KIND_UNET_TS:
        paths: List[Path] = [repo_root / rel for rel in LEGACY_UNET_TS_PATHS]
        for pattern in LEGACY_UNET_TS_GLOBS:
            paths.extend(sorted(repo_root.glob(pattern)))
        for raw in env.get("MUSETALK_UNET_ADOPT_PATHS", "").split(":"):
            if raw.strip():
                path = Path(raw.strip())
                paths.append(path if path.is_absolute() else repo_root / path)
        for path in _dedupe_real(p for p in paths if p.exists()):
            if _under(path, root):
                continue
            meta_path = path.with_name(UNET_TS_META_FILE)
            meta = None
            try:
                meta = read_json(meta_path)
            except ValueError:
                meta = None
            batch_range = (meta or {}).get("batch_range") if isinstance(meta, dict) else None
            out.append({
                "kind": kind, "layout": "legacy", "path": str(path),
                "realpath": os.path.realpath(str(path)), "bytes": path.stat().st_size,
                "meta_path": str(meta_path) if meta_path.exists() else None,
                "batch_range": batch_range,
                "meta_validation_passed": ((meta or {}).get("validation") or {}).get("passed") if isinstance(meta, dict) else None,
            })
        return out
    if kind == KIND_UNET_STAGEWISE:
        manifests: List[Path] = []
        for pattern in LEGACY_STAGEWISE_GLOBS:
            manifests.extend(sorted(repo_root.glob(pattern)))
        cache_dir = env.get("MUSETALK_UNET_STAGEWISE_CACHE_DIR", "").strip()
        if cache_dir:
            base = Path(cache_dir) if Path(cache_dir).is_absolute() else repo_root / cache_dir
            manifests.extend(sorted(base.glob("bs*/manifest.json")))
        for manifest_path in _dedupe_real(p for p in manifests if p.exists()):
            if _under(manifest_path, root):
                continue
            item: Dict[str, Any] = {"kind": kind, "layout": "legacy", "path": str(manifest_path.parent)}
            try:
                manifest = read_json(manifest_path) or {}
            except ValueError as exc:
                item["error"] = str(exc)
                out.append(item)
                continue
            item.update(stagewise_manifest_facts(manifest))
            out.append(item)
        return out
    metas: List[Path] = []
    for pattern in LEGACY_TAESD_GLOBS:
        metas.extend(sorted(repo_root.glob(pattern)))
    trt_dir = env.get("MUSETALK_TAESD_TRT_DIR", "").strip()
    if trt_dir:
        base = Path(trt_dir) if Path(trt_dir).is_absolute() else repo_root / trt_dir
        metas.extend(sorted(base.glob("taesd_trt_*.json")))
    for meta_path in _dedupe_real(p for p in metas if p.exists()):
        if _under(meta_path.parent, root) and os.path.realpath(str(meta_path.parent)) != os.path.realpath(str(root)):
            continue  # inside a store entry (flat files directly in the store root are legacy)
        item = {"kind": kind, "layout": "legacy", "path": str(meta_path)}
        try:
            meta = read_json(meta_path) or {}
        except ValueError as exc:
            item["error"] = str(exc)
            out.append(item)
            continue
        item.update(taesd_meta_facts(meta))
        out.append(item)
    return out


def stagewise_manifest_facts(manifest: Dict[str, Any]) -> Dict[str, Any]:
    cc = parse_cc(manifest.get("compute_capability"))
    blocks = manifest.get("blocks") or {}
    return {
        "schema": manifest.get("schema"),
        "schema_ok": manifest.get("schema") == STAGEWISE_MANIFEST_SCHEMA,
        "complete": bool(manifest.get("complete")),
        "batch": manifest.get("batch"),
        "gpu_name": manifest.get("gpu"),
        "compute_capability": cc_string(cc),
        "tensorrt_version": manifest.get("tensorrt_version"),
        "torch_version": manifest.get("torch_version"),
        "blocks": sorted(blocks),
        "missing_blocks": [b for b in STAGEWISE_BLOCKS if b not in blocks],
        "probe_output_sha256": (manifest.get("probe") or {}).get("output_sha256"),
    }


def taesd_meta_facts(meta: Dict[str, Any]) -> Dict[str, Any]:
    fp = meta.get("fingerprint") or {}
    return {
        "schema": meta.get("schema"),
        "schema_ok": meta.get("schema") == TAESD_META_SCHEMA,
        "runtime_key": meta.get("key"),
        "batch": fp.get("batch"),
        "gpu_name": fp.get("gpu"),
        "compute_capability": cc_string(parse_cc(fp.get("compute_capability"))),
        "tensorrt_version": fp.get("tensorrt"),
        "opt_level": fp.get("opt_level"),
        "strongly_typed": fp.get("strongly_typed"),
        "decoder_plan": meta.get("decoder_plan"),
        "post_plan": meta.get("post_plan"),
        "gate_verdict": (meta.get("gate") or {}).get("verdict"),
    }


def host_mismatch(facts: Dict[str, Any], gpu_name: Any, compute_capability: Any, tensorrt_version: Any = None,
                  allow_same_cc: bool = False) -> List[str]:
    """Reasons an artifact recorded for (gpu_name, cc, trt) cannot run on this host ([] = compatible)."""
    nf = normalize_facts(facts)
    reasons: List[str] = []
    if not nf["has_gpu"]:
        return ["no visible NVIDIA GPU"]
    cc = parse_cc(compute_capability)
    if cc != nf["cc"]:
        reasons.append(f"built for sm{cc_string(cc)} but this GPU is sm{nf['compute_capability']}")
    if slug(gpu_name) != nf["gpu_slug"] and not allow_same_cc:
        reasons.append(f"built on {gpu_name!r} but this GPU is {nf['gpu_name']!r}")
    if tensorrt_version is not None and not same_version(tensorrt_version, nf["tensorrt_version"], 4):
        reasons.append(f"built with TensorRT {tensorrt_version} but the venv has {nf['tensorrt_version']}")
    return reasons


# --------------------------------------------------------------------------- embedded device scan
def _scan_cache_path(cache_dir: Path, realpath: str) -> Path:
    return Path(cache_dir) / (hashlib.sha256(realpath.encode()).hexdigest()[:24] + ".json")


_DEVICE_FIELDS_RE = re.compile(rb"(\d{1,3})%(\d{1,3})%(\d{1,3})%(\d)%(NVIDIA[^%]{1,80})")


def _framed_device(buffer: bytes, match: "re.Match", needle_pos: int) -> Optional[Dict[str, Any]]:
    """Exact device string when it is a pickled str (BINUNICODE 'X'+u32 len or SHORT_BINUNICODE 0x8c+u8).

    torch_tensorrt 2.5 stores it as X<len>'0%8%9%0%NVIDIA GeForce RTX 4070 SUPER' followed by a pickle
    opcode, so the length prefix is the only reliable end marker. Up to 3 leading 'digits' of the regex
    match may really be length bytes, hence the small search."""
    for delta in range(0, 4):
        start = match.start() + delta
        if start >= needle_pos:
            break
        length = None
        if start >= 5 and buffer[start - 5] == 0x58:  # 'X' BINUNICODE
            length = int.from_bytes(buffer[start - 4:start], "little")
        elif start >= 2 and buffer[start - 2] == 0x8C:  # SHORT_BINUNICODE
            length = buffer[start - 1]
        if length is None or not (needle_pos - start + 8 <= length <= 200):
            continue
        raw = bytes(buffer[start:start + length])
        fields = _DEVICE_FIELDS_RE.match(raw)
        if fields is None:
            continue
        return {
            "device_string": raw.decode("ascii", "replace"),
            "device_index": int(fields.group(1)), "major": int(fields.group(2)), "minor": int(fields.group(3)),
            "device_type": int(fields.group(4)), "name": fields.group(5).decode("ascii", "replace").strip(),
            "framed": True, "start": start,
        }
    return None


def _unframed_device(match: "re.Match") -> Dict[str, Any]:
    return {
        "device_string": match.group(0).decode("ascii", "replace").rstrip(),
        "device_index": int(match.group(1)), "major": int(match.group(2)), "minor": int(match.group(3)),
        "device_type": int(match.group(4)), "name": match.group(5).decode("ascii", "replace").rstrip(),
        "framed": False, "start": match.start(),
    }


def _device_name_matches(device: Dict[str, Any], host_slug: str) -> bool:
    device_slug = slug(device.get("name"))
    if device_slug == host_slug:
        return True
    # Unframed strings may carry one or two trailing non-'%' bytes (e.g. a pickle opcode letter).
    return (not device.get("framed")) and device_slug.startswith(host_slug) and len(device_slug) - len(host_slug) <= 2


def scan_embedded_devices(path: Any, cache_dir: Optional[Any] = None, chunk_bytes: int = SCAN_CHUNK_BYTES,
                          overlap: int = SCAN_OVERLAP_BYTES, max_devices: int = 32,
                          use_cache: bool = True) -> Dict[str, Any]:
    """Stream a torch_tensorrt .ts and collect every distinct embedded device string.

    Reads chunk_bytes at a time (64 MiB default) with an `overlap`-byte tail carried between chunks so
    a string split across a chunk boundary is still found exactly once. The result is cached in
    <cache_dir>/<sha(realpath)>.json keyed by (size, mtime_ns); the original file is never written."""
    path = Path(path)
    realpath = os.path.realpath(str(path))
    stat = os.stat(realpath)
    cache_file = _scan_cache_path(Path(cache_dir), realpath) if cache_dir else None
    if cache_file and use_cache:
        try:
            cached = read_json(cache_file)
        except ValueError:
            cached = None
        if (isinstance(cached, dict) and cached.get("schema") == SCAN_CACHE_SCHEMA
                and cached.get("realpath") == realpath and cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns):
            cached["cached"] = True
            return cached
    if overlap < 256:
        raise ValueError("overlap must be >= 256 bytes (longest device string is ~100 bytes)")
    started = time.time()
    devices: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}
    matches = 0
    needle = b"%NVIDIA"
    prefix = 16  # >= the longest "ddd%ddd%ddd%d" prefix (14 bytes) in front of "%NVIDIA"
    tail = b""
    tail_abs = 0  # absolute file offset of tail[0]
    skip = 0  # needles before this buffer offset were handled in the previous buffer
    with open(realpath, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            final = not chunk
            buffer = tail + chunk
            buffer_abs = tail_abs
            # Only needles well inside the buffer are handled here, so the name after them is complete;
            # the rest is carried (with `prefix` bytes in front) into the next buffer.
            limit = len(buffer) if final else max(0, len(buffer) - overlap)
            pos = buffer.find(needle, skip)
            while pos != -1 and pos < limit:
                match = DEVICE_RE.search(buffer, max(0, pos - prefix), pos + len(needle) + 81)
                if match is not None and match.start(5) == pos + 1:
                    matches += 1
                    parsed = _framed_device(buffer, match, pos) or _unframed_device(match)
                    ident = (parsed["major"], parsed["minor"], parsed["device_type"], parsed["name"])
                    if ident not in devices and len(devices) < max_devices:
                        parsed["first_offset"] = buffer_abs + parsed.pop("start")
                        devices[ident] = parsed
                pos = buffer.find(needle, pos + 1)
            if final:
                break
            new_start = max(0, limit - prefix)
            tail = buffer[new_start:]
            tail_abs = buffer_abs + new_start
            skip = limit - new_start
    result = {
        "schema": SCAN_CACHE_SCHEMA, "path": str(path), "realpath": realpath, "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "scanned_utc": utc_now(), "seconds": round(time.time() - started, 3),
        "matches": matches, "devices": list(devices.values()), "cached": False,
    }
    if cache_file:
        try:
            write_json_atomic(cache_file, result)
        except OSError:
            pass
    return result


def embedded_device_problems(scan: Dict[str, Any], facts: Dict[str, Any]) -> List[str]:
    """[] when every embedded device string matches this GPU's cc and name."""
    nf = normalize_facts(facts)
    if not nf["has_gpu"]:
        return ["no visible NVIDIA GPU"]
    devices = scan.get("devices") or []
    if not devices:
        return ["no embedded device string found (not a torch_tensorrt engine?)"]
    problems = []
    for device in devices:
        if (device["major"], device["minor"]) != nf["cc"]:
            problems.append(
                f"engine targets sm{device['major']}.{device['minor']} ({device['name']}) "
                f"but this GPU is sm{nf['compute_capability']} ({nf['gpu_name']})")
        elif not _device_name_matches(device, nf["gpu_slug"]):
            problems.append(f"engine was built on {device['name']!r}, this GPU is {nf['gpu_name']!r}")
    return problems


# --------------------------------------------------------------------------- CLI (debug)
def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MuseTalk engine key helper (stdlib only)")
    sub = parser.add_subparsers(dest="command", required=True)
    key_p = sub.add_parser("key", help="print the engine key for this host")
    key_p.add_argument("--kind", choices=KINDS, default=KIND_UNET_TS)
    key_p.add_argument("--venv", default=os.environ.get("VIRTUAL_ENV", ""))
    scan_p = sub.add_parser("scan", help="scan a .ts for embedded device strings")
    scan_p.add_argument("path")
    scan_p.add_argument("--cache-dir", default="")
    args = parser.parse_args(argv)
    if args.command == "key":
        facts = detect_engine_facts(Path(args.venv) if args.venv else None)
        key, error = engine_key_or_none(args.kind, facts)
        if key is None:
            print(f"error: {error}", file=sys.stderr)
            return 3
        print(key)
        return 0
    print(json.dumps(scan_embedded_devices(args.path, cache_dir=args.cache_dir or None), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
