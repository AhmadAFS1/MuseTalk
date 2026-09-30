#!/usr/bin/env python3
"""MuseTalk host profile: detect host facts and resolve the serving recipe.

Component A of docs/startup_rework_20260928/STARTUP_CONTRACT.md. This file is
stdlib-only (python3 >= 3.8) and NEVER imports torch, tensorrt or aiortc: the
venv is inspected through its site-packages/*.dist-info directory names.

Subcommands
  detect       [--repo-root R] [--venv V]           host facts JSON on stdout
  resolve      --recipe fast|fast300|r5|legacy_int8  writes .runtime/musetalk_resolved.{env,json}
               [--repo-root R] [--venv V] [--out ENVFILE] [--report JSONFILE]
  verify-log   --log L [--offset B] --expect-vae X [--expect-unet Y] [--timeout S]
  engine-key   --kind unet_ts|unet_stagewise|taesd_trt
  find-engine  --kind K [--batch N]                  JSON of the best usable engine, exit 3 if none

Layering (highest wins): 1 caller env > 2 overrides files (MUSETALK_ENV_OVERRIDES_FILE,
colon list, default <repo>/.runtime/musetalk_overrides.env, first file wins) >
3 recipe levers (configs/recipes/<recipe>.env, fast300 and r5 only, each gated) >
4 resolver-computed values > 5 code defaults. The resolver reads layers 1 and 2
itself (only-if-unset, exactly like the launcher), so its dependents (bucket
coupling, warmups) are computed from what the server will really see.

scripts/run_musetalk_server.sh loads the overrides before calling resolve and passes
MUSETALK_ENV_CALLER_KEYS / MUSETALK_ENV_OVERRIDE_KEYS (comma lists) so the report
attributes each value to the right layer; without them the source is inferred.

Recipes
  fast        (DEFAULT) compiled TAESD + the validated torch_tensorrt bs8 .ts UNet for this
              engine key, else eager (reason recorded). MUSETALK_UNET_MODE=auto|trt|eager.
  fast300     fast + the levers of configs/recipes/fast300.env (MUSETALK_RECIPE_FILE overrides).
              "# @lever <group> requires=<p>,..." groups are atomic: every prerequisite
              (engine:unet_stagewise, engine:taesd_trt [store-usable AND G-TAESD PASS],
              engine:unet_ts, unet:trt_any, vp8:native_preflight, h264:native_fallback,
              gpu:nvenc, code:<path>) must hold and every line must pass its own gate, else the
              whole group is dropped with the reason (resolution re-runs until stable).
              Ungrouped lines are single levers gated by the LEVERS table (engine levers need a
              validated engine, pass-through levers need code that reads them).
  r5          fast + the levers of configs/recipes/r5.env, in the same format. Its engine group
              requires bundle:<name>: the pinned S3 engine bundle configs/trt_bundles/<name>.json,
              for this exact engine key (GPU model + TensorRT), restored or adopted with a stamp
              for the pinned archive SHA256 by scripts/trt_artifact_bundle.py, every file present.
              The resolver then points the stagewise UNet and TAESD TRT at the bundle's engines
              (they are not engine-store entries). On any other host the group is dropped with
              its reason and the host serves the fast engines plus r5's serving levers.
  legacy_int8 emits only MUSETALK_RECIPE=legacy_int8 (the launcher execs the old chain).
Engines come ONLY from scripts/musetalk_engine_keys.py (the engine store's module).
Pass-through serving levers are never emitted unless a recipe line enables them; the
report's "levers" section shows every lever's effective value and source layer.

verify-log expectations: --expect-vae taesd (either TAESD backend) | taesd_compiled |
taesd_trt | pytorch | trt_stagewise (old INT8 SD-VAE) | any | auto; --expect-unet trt
(.ts: tensorrt_unet[_multi]) | trt_stagewise | tensorrt_any | eager | any | auto.
'auto' reads "expect" from --resolved (or <--repo-root>/.runtime/musetalk_resolved.json).

Exit codes: resolve 0 ok / 2 hard error; verify-log 0 match / 1 mismatch /
3 not found in time; find-engine 3 when none; 2 usage errors.

Test hooks: MUSETALK_HOST_FACTS_JSON=<path> replaces any top-level facts
section it contains (gpus, cpu, ram, disk_free_gb, machine, venv, ...);
MUSETALK_NVIDIA_SMI overrides the nvidia-smi binary.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent

FACTS_SCHEMA = "musetalk_host_facts_v1"
REPORT_SCHEMA = "musetalk_resolved_v1"
SELFTEST_SCHEMA = "musetalk_gpu_selftest_v1"
RECIPES = ("fast", "fast300", "r5", "legacy_int8")
RECIPE_FILE_RECIPES = ("fast300", "r5")  # recipes that read configs/recipes/<recipe>.env
TRT_BUNDLE_DIR = Path("configs") / "trt_bundles"  # pinned S3 engine bundles (bundle:<name> prerequisite)
TRT_BUNDLE_STAMP = ".musetalk_trt_artifact_restored.json"  # written by scripts/trt_artifact_bundle.py
TRT_BUNDLE_MANIFEST = ".musetalk_trt_artifact_manifest.json"
ENGINE_KINDS = ("unet_ts", "unet_stagewise", "taesd_trt")
TS_ENGINE_BATCH = 8  # MultiTrtUnetBackend only serves multiples of the .ts engine batch

# Old launcher / measured constants (see contract "Facts established").
TS_VS_EAGER_UNET_RATIO = 24.22 / 37.0  # RTX 4070 SUPER, bs8: TRT .ts 24.22 ms vs eager ~37 ms
POWER_CAP_RATIO = 0.85

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def log(msg: str) -> None:
    print(f"[musetalk_host_profile] {msg}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_float(value, default=None):
    try:
        text = str(value).strip()
        if text.startswith("[") or text.upper() in ("N/A", ""):
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _mib(value):
    number = _to_float(value)
    return int(round(number)) if number is not None else None


def parse_bool(value):
    """True/False for recognised boolean spellings, None otherwise."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def parse_buckets(raw: str):
    """'8,16' -> [8, 16] (sorted, unique, positive). Raises ValueError on junk."""
    out = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)  # ValueError propagates
        if value <= 0:
            raise ValueError(f"batch size must be positive: {token}")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("empty bucket list")
    return sorted(out)


def shell_single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_text(path, default=None):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def _read_json(path, default=None):
    text = _read_text(path)
    if text is None:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


# --------------------------------------------------------------------------- env files
_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _unquote_env_value(raw: str) -> str:
    """Value part of KEY=VALUE: 'single' (with '\\'' escapes), "double", or bare (trailing # comment dropped)."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw[0] == "'":
        # Concatenation of '...' chunks and \' escapes, as shell_single_quote writes them.
        out, i = [], 0
        while i < len(raw):
            ch = raw[i]
            if ch == "'":
                end = raw.find("'", i + 1)
                if end < 0:
                    out.append(raw[i + 1:])
                    break
                out.append(raw[i + 1:end])
                i = end + 1
            elif ch == "\\" and i + 1 < len(raw):
                out.append(raw[i + 1])
                i += 2
            elif ch.isspace() or ch == "#":
                break
            else:
                out.append(ch)
                i += 1
        return "".join(out)
    if raw[0] == '"':
        end = raw.rfind('"')
        body = raw[1:end] if end > 0 else raw[1:]
        return re.sub(r'\\(["\\$`])', r"\1", body)
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def parse_env_file(path):
    """[(key, value, lineno)] from KEY=VALUE / export KEY=VALUE lines; blanks/comments ignored."""
    text = _read_text(path)
    if text is None:
        return []
    entries = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        entries.append((match.group(1), _unquote_env_value(match.group(2)), lineno))
    return entries


_LEVER_GROUP_RE = re.compile(r"^\s*#\s*@lever\s+(\S+)(?:\s+requires=(\S*))?\s*$")


def parse_recipe_file(path):
    """(groups, entries) of a recipe file (configs/recipes/<name>.env; parsed, never sourced).

    "# @lever <name> requires=<p1>,<p2>" starts a group; the KEY=VALUE lines that follow, up to the
    next blank line, belong to it and are enabled or dropped together. "#KEY=VALUE" is a lever that is
    switched off (a comment). Lines outside any group are single-lever implicit groups.
    groups: [{"name", "requires": [...], "line", "keys": [...]}]; entries: [(key, value, lineno, group)]."""
    text = _read_text(path) or ""
    groups, entries, current, seen = [], [], None, set()
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        match = _LEVER_GROUP_RE.match(line)
        if match:
            name = match.group(1)
            if name in seen:
                name = f"{name}@{lineno}"
            seen.add(name)
            current = {"name": name, "requires": [r.strip() for r in (match.group(2) or "").split(",") if r.strip()],
                       "line": lineno, "keys": []}
            groups.append(current)
            continue
        if not stripped:
            current = None
            continue
        if stripped.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key = match.group(1)
        entries.append((key, _unquote_env_value(match.group(2)), lineno, current["name"] if current else None))
        if current is not None:
            current["keys"].append(key)
    return groups, entries


def overrides_files(repo_root: Path, environ) -> list:
    raw = environ.get("MUSETALK_ENV_OVERRIDES_FILE")
    if raw is None:
        return [repo_root / ".runtime" / "musetalk_overrides.env"]
    return [Path(p) for p in raw.split(":") if p.strip()]


# --------------------------------------------------------------------------- detect: GPU
_SMI_FIELDS = ["index", "name", "compute_cap", "memory.total", "memory.used", "power.limit",
               "power.default_limit", "driver_version", "uuid"]


def _run_smi(binary: str, fields):
    cmd = [binary, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
                          universal_newlines=True)
    return proc.returncode, proc.stdout, proc.stderr


def query_gpus(binary: str = None):
    """(gpus, info) from nvidia-smi. Never raises; info carries the error text."""
    binary = binary or os.environ.get("MUSETALK_NVIDIA_SMI") or "nvidia-smi"
    info = {"binary": binary, "ok": False, "error": None}
    resolved = shutil.which(binary) if os.sep not in binary else (binary if os.path.exists(binary) else None)
    if not resolved:
        info["error"] = f"{binary} not found"
        return [], info
    fields = list(_SMI_FIELDS)
    try:
        rc, out, err = _run_smi(resolved, fields)
        if rc != 0 and "compute_cap" in (err + out):
            fields.remove("compute_cap")  # drivers < 510 do not know the field
            rc, out, err = _run_smi(resolved, fields)
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return [], info
    if rc != 0:
        info["error"] = (err or out).strip()[:500] or f"exit {rc}"
        return [], info
    gpus = []
    for row in csv.reader(io.StringIO(out), skipinitialspace=True):
        if not row or not "".join(row).strip():
            continue
        rec = dict(zip(fields, [c.strip() for c in row]))
        cc = rec.get("compute_cap")
        gpus.append({
            "index": _to_int(rec.get("index")),
            "name": rec.get("name"),
            "compute_capability": cc if cc and re.match(r"^\d+\.\d+$", cc) else None,
            "memory_total_mib": _mib(rec.get("memory.total")),
            "memory_used_mib": _mib(rec.get("memory.used")),
            "power_limit_w": _to_float(rec.get("power.limit")),
            "power_default_limit_w": _to_float(rec.get("power.default_limit")),
            "driver_version": rec.get("driver_version"),
            "uuid": rec.get("uuid"),
        })
    info["ok"] = True
    return gpus, info


def select_gpu(gpus, cuda_visible_devices):
    """(gpu or None, reason). First CUDA_VISIBLE_DEVICES entry (index or GPU-/MIG- uuid prefix)."""
    if not gpus:
        return None, "no GPU reported by nvidia-smi"
    if cuda_visible_devices is None:
        return gpus[0], "first GPU (CUDA_VISIBLE_DEVICES unset)"
    tokens = [t.strip() for t in str(cuda_visible_devices).split(",") if t.strip()]
    if not tokens or tokens[0] in ("-1", "none", "NoDevFiles"):
        return None, f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} hides every GPU"
    first = tokens[0]
    if re.match(r"^\d+$", first):
        for gpu in gpus:
            if gpu.get("index") == int(first):
                return gpu, f"CUDA_VISIBLE_DEVICES first entry {first}"
        return None, f"CUDA_VISIBLE_DEVICES first entry {first} is not an nvidia-smi index"
    for gpu in gpus:
        uuid = gpu.get("uuid") or ""
        if uuid and (uuid.startswith(first) or first.startswith(uuid)):
            return gpu, f"CUDA_VISIBLE_DEVICES first entry {first} (uuid)"
    return None, f"CUDA_VISIBLE_DEVICES first entry {first!r} matches no GPU uuid"


def cc_tuple(cc):
    if not cc:
        return None
    match = re.match(r"^(\d+)\.(\d+)$", str(cc).strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


# --------------------------------------------------------------------------- detect: CPU / RAM / disk
def cgroup_cpu_quota(cgroup_root="/sys/fs/cgroup"):
    root = Path(cgroup_root)
    text = _read_text(root / "cpu.max")
    if text is not None:  # cgroup v2: "<quota|max> <period>"
        parts = text.split()
        if len(parts) >= 2 and parts[0] != "max":
            quota, period = _to_float(parts[0]), _to_float(parts[1])
            if quota and period and quota > 0 and period > 0:
                return round(quota / period, 3)
        return None
    for sub in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):  # cgroup v1
        quota = _to_float(_read_text(root / sub / "cpu.cfs_quota_us"))
        period = _to_float(_read_text(root / sub / "cpu.cfs_period_us"))
        if quota is not None and period:
            return round(quota / period, 3) if quota > 0 else None
    return None


def detect_cpu(cgroup_root="/sys/fs/cgroup"):
    nproc = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = nproc
    quota = cgroup_cpu_quota(cgroup_root)
    candidates = [nproc, affinity]
    if quota:
        candidates.append(max(1, int(math.ceil(quota))))
    return {"nproc": nproc, "affinity": affinity, "cgroup_quota_cpus": quota,
            "effective": max(1, min(candidates))}


def _meminfo(path="/proc/meminfo"):
    values = {}
    for line in (_read_text(path) or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            values[parts[0][:-1]] = _to_int(parts[1], 0)  # kB
    return values


def _memory_stat(path):
    stats = {}
    for line in (_read_text(path) or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            stats[parts[0]] = _to_int(parts[1], 0)
    return stats


def cgroup_memory(cgroup_root="/sys/fs/cgroup"):
    """(limit_bytes or None, available_estimate_bytes or None), like box_guard.sh."""
    root = Path(cgroup_root)
    raw_max = _read_text(root / "memory.max")
    if raw_max is not None:  # v2
        raw_max = raw_max.strip()
        if raw_max == "max" or not raw_max:
            return None, None
        limit = _to_int(raw_max)
        current = _to_int(_read_text(root / "memory.current"))
        if limit is None or current is None:
            return limit, None
        stat = _memory_stat(root / "memory.stat")
        reclaim = stat.get("active_file", 0) + stat.get("inactive_file", 0) + stat.get("slab_reclaimable", 0)
        return limit, max(0, limit - current + reclaim)
    v1 = root / "memory"
    limit = _to_int(_read_text(v1 / "memory.limit_in_bytes"))
    if limit is None or limit >= (1 << 60):
        return None, None
    usage = _to_int(_read_text(v1 / "memory.usage_in_bytes"))
    if usage is None:
        return limit, None
    stat = _memory_stat(v1 / "memory.stat")
    reclaim = stat.get("total_active_file", 0) + stat.get("total_inactive_file", 0)
    return limit, max(0, limit - usage + reclaim)


def detect_ram(meminfo_path="/proc/meminfo", cgroup_root="/sys/fs/cgroup"):
    info = _meminfo(meminfo_path)
    total_mb = info.get("MemTotal", 0) // 1024
    avail_mb = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
    limit, cg_avail = cgroup_memory(cgroup_root)
    limit_mb = limit // (1024 * 1024) if limit else None
    cg_avail_mb = cg_avail // (1024 * 1024) if cg_avail is not None else None
    return {
        "mem_total_mb": total_mb,
        "mem_available_mb": avail_mb,
        "cgroup_limit_mb": limit_mb,
        "cgroup_available_mb": cg_avail_mb,
        "effective_total_mb": min(total_mb, limit_mb) if limit_mb else total_mb,
        "effective_available_mb": min(avail_mb, cg_avail_mb) if cg_avail_mb is not None else avail_mb,
    }


def disk_free_gb(path) -> float:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(str(probe)).free / 1024 ** 3, 1)
    except OSError:
        return None


# --------------------------------------------------------------------------- detect: venv (no imports)
def _norm_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


_EGG_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.]+?)-(?P<ver>[^-]+?)(?:-py\d[^-]*)?\.egg-info$")


def _dist_version_from_metadata(dist_dir: Path):
    for fname in ("METADATA", "PKG-INFO"):
        text = _read_text(dist_dir / fname)
        if text:
            for line in text.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
                if not line.strip():
                    break
    return None


def site_packages_dirs(venv: Path):
    out = []
    for lib in ("lib", "lib64"):
        base = venv / lib
        if base.is_dir():
            for child in sorted(base.iterdir()):
                sp = child / "site-packages"
                if child.name.startswith("python") and sp.is_dir() and sp not in out:
                    out.append(sp)
    return out


def installed_packages(site_packages: Path) -> dict:
    """normalized name -> version, from *.dist-info / *.egg-info names (METADATA as fallback)."""
    packages = {}
    try:
        names = sorted(os.listdir(str(site_packages)))
    except OSError:
        return packages
    for entry in names:
        name = version = None
        if entry.endswith(".dist-info"):
            stem = entry[: -len(".dist-info")]
            if "-" in stem:
                name, version = stem.split("-", 1)
            else:
                name, version = stem, _dist_version_from_metadata(site_packages / entry)
        elif entry.endswith(".egg-info"):
            match = _EGG_RE.match(entry)
            if match:
                name, version = match.group("name"), match.group("ver")
        if name and version:
            packages.setdefault(_norm_pkg(name), version)
    return packages


def _pyvenv_version(venv: Path):
    text = _read_text(venv / "pyvenv.cfg") or ""
    for key in ("version_info", "version"):
        match = re.search(rf"^\s*{key}\s*=\s*(\d+\.\d+(?:\.\d+)?)", text, re.M)
        if match:
            return match.group(1)
    return None


def _cuda_tag(torch_version, packages):
    if torch_version and "+" in torch_version:
        local = torch_version.split("+", 1)[1]
        if re.match(r"^(cu\d+|cpu|rocm[\d.]+)$", local):
            return local
    for name, ver in packages.items():
        if re.match(r"^nvidia_cuda_runtime_cu\d+$", name) and ver:
            parts = ver.split(".")
            if len(parts) >= 2:
                return f"cu{parts[0]}{parts[1]}"
    return None


_TRT_DIST_ORDER = ("tensorrt_cu13_bindings", "tensorrt_cu12_bindings", "tensorrt_bindings",
                   "tensorrt_cu13", "tensorrt_cu12", "tensorrt")
_REPORTED_PACKAGES = ("torchvision", "torchaudio", "diffusers", "transformers", "numpy", "nvidia_modelopt",
                      "kokoro", "mediapipe", "opencv_python", "fastapi", "uvicorn", "boto3", "onnx")


def detect_venv(venv) -> dict:
    venv = Path(venv) if venv else None
    facts = {"path": str(venv) if venv else None, "exists": bool(venv and venv.is_dir()),
             "python": None, "python_version": None, "site_packages": None,
             "torch": None, "torch_cuda_tag": None, "tensorrt": None, "torch_tensorrt": None,
             "triton": None, "aiortc": None, "av": None, "cffi": None, "packages": {}}
    if not facts["exists"]:
        return facts
    for cand in ("bin/python", "bin/python3"):
        if (venv / cand).exists():
            facts["python"] = str(venv / cand)
            break
    sps = site_packages_dirs(venv)
    version = _pyvenv_version(venv)
    chosen = None
    for sp in sps:
        if version and sp.parent.name == "python" + ".".join(version.split(".")[:2]):
            chosen = sp
            break
    chosen = chosen or (sps[0] if sps else None)
    if not version and chosen is not None:
        match = re.match(r"^python(\d+\.\d+)$", chosen.parent.name)
        version = match.group(1) if match else None
    facts["python_version"] = version
    if chosen is None:
        return facts
    facts["site_packages"] = str(chosen)
    packages = installed_packages(chosen)
    facts["torch"] = packages.get("torch")
    facts["torch_cuda_tag"] = _cuda_tag(facts["torch"], packages)
    facts["tensorrt"] = next((packages[n] for n in _TRT_DIST_ORDER if packages.get(n)), None)
    for key in ("torch_tensorrt", "triton", "aiortc", "av", "cffi"):
        facts[key] = packages.get(key)
    facts["packages"] = {k: packages[k] for k in _REPORTED_PACKAGES if k in packages}
    return facts


def _cuda_tag_number(tag):
    match = re.match(r"^cu(\d+)$", tag or "")
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- detect: entry point
def _load_injected_facts():
    path = os.environ.get("MUSETALK_HOST_FACTS_JSON", "").strip()
    if not path:
        return None, None
    data = _read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"MUSETALK_HOST_FACTS_JSON={path} is not a JSON object")
    return data, path


def engine_facts_from(facts: dict) -> dict:
    gpu = facts.get("gpu") or {}
    venv = facts.get("venv") or {}
    return {"gpu_name": gpu.get("name"), "compute_capability": gpu.get("compute_capability"),
            "tensorrt_version": venv.get("tensorrt"), "torch_tensorrt_version": venv.get("torch_tensorrt"),
            "torch_version": venv.get("torch"), "driver_version": gpu.get("driver_version")}


def detect(repo_root, venv, environ=None) -> dict:
    environ = os.environ if environ is None else environ
    repo_root = Path(repo_root).resolve()
    injected, injected_path = _load_injected_facts()
    injected = injected or {}
    facts = {"schema": FACTS_SCHEMA, "created_utc": utc_now(), "repo_root": str(repo_root),
             "injected_from": injected_path, "injected_sections": sorted(injected)}
    if "gpus" in injected:
        facts["gpus"] = injected["gpus"]
        facts["nvidia_smi"] = {"binary": None, "ok": True, "error": None, "injected": True}
    else:
        gpus, info = query_gpus()
        facts["gpus"], facts["nvidia_smi"] = gpus, info
    cvd = injected.get("cuda_visible_devices", environ.get("CUDA_VISIBLE_DEVICES"))
    facts["cuda_visible_devices"] = cvd
    gpu, reason = select_gpu(facts["gpus"], cvd)
    facts["gpu"] = gpu
    facts["selected_gpu_index"] = gpu.get("index") if gpu else None
    facts["gpu_selection"] = reason
    facts["cpu"] = dict(detect_cpu(), **injected["cpu"]) if "cpu" in injected else detect_cpu()
    facts["ram"] = dict(detect_ram(), **injected["ram"]) if "ram" in injected else detect_ram()
    facts["disk_free_gb"] = injected["disk_free_gb"] if "disk_free_gb" in injected else disk_free_gb(repo_root)
    facts["machine"] = injected.get("machine", platform.machine())
    facts["os"] = injected.get("os", platform.system())
    venv_facts = detect_venv(venv)
    if isinstance(injected.get("venv"), dict):
        venv_facts.update(injected["venv"])
    facts["venv"] = venv_facts
    for key, value in injected.items():  # any other injected section wins as-is
        if key not in facts or key in ("gpu", "selected_gpu_index"):
            facts[key] = value
    facts["engine_facts"] = engine_facts_from(facts)
    return facts


# --------------------------------------------------------------------------- engine store adapter
def slug(name) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", str(name or "").lower())).strip("-")


def _strip_local(version):
    return str(version).split("+", 1)[0] if version else version


def builtin_engine_key(kind: str, facts: dict):
    """Fallback for musetalk_engine_keys.engine_key: sm<cc>-<gpu slug>-trt<v>[-tt<v>]."""
    ef = facts.get("engine_facts") or engine_facts_from(facts)
    cc = cc_tuple(ef.get("compute_capability"))
    if not cc or not ef.get("gpu_name") or not ef.get("tensorrt_version"):
        return None
    key = f"sm{cc[0]}{cc[1]}-{slug(ef['gpu_name'])}-trt{_strip_local(ef['tensorrt_version'])}"
    if kind == "unet_ts":
        if not ef.get("torch_tensorrt_version"):
            return None
        key += f"-tt{_strip_local(ef['torch_tensorrt_version'])}"
    return key


_KIND_STORE_DEFAULT = {"unet_ts": "models/tensorrt_unet", "unet_stagewise": "models/tensorrt_unet_stagewise",
                       "taesd_trt": "models/taesd/trt"}
_KIND_STORE_ENV = {"unet_ts": "MUSETALK_UNET_ENGINE_STORE",
                   "unet_stagewise": "MUSETALK_UNET_STAGEWISE_ENGINE_STORE",
                   "taesd_trt": "MUSETALK_TAESD_TRT_ENGINE_STORE"}
LEGACY_STAGEWISE_ROOT = "models/tensorrt_unet_stagewise_sm89"  # unet_stagewise_trt.default_cache_root()


class EngineStore:
    """Engine lookup through scripts/musetalk_engine_keys.py (owned by the engine-store component:
    engine_key_or_none, store_root, list_entries, describe_entry, entry_env, legacy_candidates).

    That module is the single source of truth for usability. If it is missing or broken, no engine
    is usable (auto mode serves eager, trt mode is a hard error) and the report says why."""

    def __init__(self, repo_root: Path, env_get=None):
        self.repo_root = Path(repo_root)
        self.env_get = env_get or (lambda key: os.environ.get(key))
        self.module = None
        self.module_error = None
        self.key_errors = {}
        path = self.repo_root / "scripts" / "musetalk_engine_keys.py"
        if not path.exists():
            path = SCRIPT_DIR / "musetalk_engine_keys.py"
        if not path.exists():
            self.module_error = f"{path} not found"
            return
        try:
            spec = importlib.util.spec_from_file_location("musetalk_engine_keys", str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("musetalk_engine_keys", module)
            spec.loader.exec_module(module)
            for name in ("engine_key_or_none", "store_root", "list_entries", "describe_entry", "entry_env"):
                if not callable(getattr(module, name, None)):
                    raise AttributeError(f"musetalk_engine_keys.{name} missing")
            self.module = module
        except Exception as exc:  # a broken sibling module must not break the resolver
            self.module_error = f"{type(exc).__name__}: {exc}"

    @property
    def backend(self) -> str:
        return "musetalk_engine_keys" if self.module is not None else f"unavailable ({self.module_error})"

    def engine_key(self, kind, facts):
        if self.module is None:
            return builtin_engine_key(kind, facts)
        try:
            key, err = self.module.engine_key_or_none(kind, facts)
        except Exception as exc:
            key, err = None, f"{type(exc).__name__}: {exc}"
        if err:
            self.key_errors[kind] = err
        return key

    def store_root(self, kind) -> Path:
        raw = (self.env_get(_KIND_STORE_ENV[kind]) or "").strip() or None
        if self.module is not None:
            try:
                return Path(self.module.store_root(kind, self.repo_root, store=raw))
            except Exception:
                pass
        root = Path(raw or _KIND_STORE_DEFAULT[kind])
        return root if root.is_absolute() else (self.repo_root / root)

    def entries(self, kind, facts, allow_same_cc=False):
        """Every store entry of this kind, described for this host (usable, match, reasons, env)."""
        if self.module is None:
            return []
        out = []
        try:
            for raw in self.module.list_entries(kind, self.repo_root, store=self.store_root(kind)):
                entry = self.module.describe_entry(raw, facts, allow_same_cc=allow_same_cc)
                entry["root"] = str(Path(entry["dir"]).parent)
                entry["reason"] = "; ".join(entry.get("reasons") or []) or None
                out.append(entry)
        except Exception as exc:
            self.module_error = f"list/describe {kind}: {type(exc).__name__}: {exc}"
            return []
        return out

    def find(self, kind, facts, batch=None, allow_same_cc=False, prefer_batch=None):
        """(best usable entry or None, all entries). Exact key before same_cc, then prefer_batch, then larger."""
        entries = self.entries(kind, facts, allow_same_cc=allow_same_cc)
        usable = [e for e in entries if e.get("usable") and (batch is None or _to_int(e.get("batch")) == batch)]
        usable.sort(key=lambda e: (0 if e.get("match") == "exact" else 1,
                                   0 if prefer_batch and _to_int(e.get("batch")) == prefer_batch else 1,
                                   -(_to_int(e.get("batch")) or 0)))
        return (usable[0] if usable else None), entries

    def entry_env(self, entry) -> dict:
        if self.module is None or not entry:
            return {}
        try:
            return dict(self.module.entry_env(entry))
        except Exception:
            return {}

    def legacy(self, kind):
        if self.module is None or not callable(getattr(self.module, "legacy_candidates", None)):
            return []
        try:
            return list(self.module.legacy_candidates(kind, self.repo_root, store=self.store_root(kind)))
        except Exception:
            return []


def compact_entry(entry):
    out = {k: entry.get(k) for k in ("key", "batch", "dir", "layout", "source", "usable", "match", "reason")}
    out["quality_verdict"] = (entry.get("quality_gate") or {}).get("verdict")
    return out


def quality_verdict(entry):
    """PASS | FAIL | None: the lever's quality gate as the engine store reports it (for unet_* this is the
    store validation; for taesd_trt it is G-TAESD recorded in the engine meta, separate from usability)."""
    return (entry or {}).get("quality_gate", {}).get("verdict") if entry else None


def engine_path_of(entry):
    if not entry:
        return None
    if entry.get("engine_path"):
        return entry["engine_path"]
    fp = entry.get("fingerprint") or {}
    return str(Path(entry["dir"]) / (fp.get("engine_file") or "unet_trt.ts")) if entry.get("dir") else None


# --------------------------------------------------------------------------- lever registry
# kind: bool | int | choice | str ; category: engine | encoder | serving | memory | timing | debug
def _lever(category, kind="str", choices=None, minimum=None, maximum=None, default="", raises=False, note=""):
    return {"category": category, "kind": kind, "choices": choices, "min": minimum, "max": maximum,
            "default": default, "raises": raises, "note": note}


LEVERS = {
    # engines (fast300 gates live in Resolver._gate_*)
    "MUSETALK_UNET_BACKEND": _lever("engine", "choice", ["trt_stagewise", "tensorrt_stagewise"], default="(eager)",
                                    note="recipe files may only select trt_stagewise; trt/eager come from MUSETALK_UNET_MODE"),
    "MUSETALK_UNET_STAGEWISE_BATCH": _lever("engine", "int", minimum=1, default="16"),
    "MUSETALK_UNET_STAGEWISE_CACHE_DIR": _lever("engine", default="models/tensorrt_unet_stagewise_sm89"),
    "MUSETALK_UNET_STAGEWISE_CUDAGRAPH": _lever("engine", "bool", default="1"),
    "MUSETALK_UNET_STAGEWISE_PROBE_CHECK": _lever("engine", "bool", default="1"),
    "MUSETALK_UNET_STAGEWISE_PROBE_TOL": _lever("engine", default="0"),
    "MUSETALK_UNET_STAGEWISE_VERIFY_SHA": _lever("engine", "bool", default="1"),
    "MUSETALK_TAESD_BACKEND": _lever("engine", "choice", ["compiled", "trt", "tensorrt"], default="compiled"),
    "MUSETALK_TAESD_TRT_DIR": _lever("engine", default="models/taesd/trt"),
    "MUSETALK_TAESD_TRT_BATCH": _lever("engine", "int", minimum=1, default="8"),
    "MUSETALK_TAESD_TRT_BUILD": _lever("engine", "bool", default="1"),
    "MUSETALK_TAESD_TRT_STRICT": _lever("engine", "bool", default="0"),
    "MUSETALK_TAESD_TRT_FUSED_POST": _lever("engine", "bool", default="1"),
    "MUSETALK_TAESD_TRT_OPT_LEVEL": _lever("engine", "int", minimum=0, maximum=5, default="3"),
    "MUSETALK_TAESD_TRT_STRONGLY_TYPED": _lever("engine", "bool", default="0"),
    "MUSETALK_TRT_UNET_CUDAGRAPHS": _lever("engine", "choice",
                                           ["0", "off", "false", "no", "none", "1", "on", "true", "yes", "manual",
                                            "runtime"], default="0", raises=True),
    "MUSETALK_FREE_EAGER_UNET": _lever("memory", "bool", default="0"),
    # encoders
    "WEBRTC_VP8_ENCODER": _lever("encoder", "choice", ["pyav", "native"], default="pyav", raises=True),
    "WEBRTC_NATIVE_VP8_THREADS": _lever("encoder", "int", minimum=1, maximum=16, raises=True),
    "WEBRTC_NATIVE_VP8_MAX_BITRATE_BPS": _lever("encoder", "int", minimum=1),
    "WEBRTC_NATIVE_VP8_BITRATE_FLOOR_BPS": _lever("encoder", "int", minimum=0),
    "WEBRTC_H264_IMPL": _lever("encoder", "choice", ["aiortc", "x264tuned", "nvenc"], default="aiortc", raises=True),
    "WEBRTC_H264_X264_PRESET": _lever("encoder", default="veryfast"),
    "WEBRTC_H264_X264_THREADS": _lever("encoder", "int", minimum=0, maximum=64, default="1"),
    "WEBRTC_H264_NVENC_PRESET": _lever("encoder", default="p2"),
    "WEBRTC_H264_NVENC_TUNE": _lever("encoder", default="ll"),
    "WEBRTC_NVENC_MAX_SESSIONS": _lever("encoder", "int", minimum=0, maximum=64, default="12"),
    # serving
    "WEBRTC_NONBLOCKING_HANDOFF": _lever("serving", "bool", default="0"),
    "WEBRTC_HANDOFF_MAX_PENDING_FRAMES": _lever("serving", "int", minimum=0, default="64"),
    "WEBRTC_HANDOFF_CONVERT_THREADS": _lever("serving", "int", minimum=0, default="0"),
    "WEBRTC_YUV_IN_COMPOSE": _lever("serving", "bool", default="0"),
    "WEBRTC_IDLE_FRAME_CACHE": _lever("serving", "bool", default="0"),
    "WEBRTC_IDLE_FRAME_CACHE_MAX_MB": _lever("serving", "int", minimum=0, default="1024"),
    "WEBRTC_IDLE_FRAME_CACHE_DECODE_THREADS": _lever("serving", "int", minimum=0, default="4"),
    "WEBRTC_IDLE_FRAME_CACHE_WORKERS": _lever("serving", "int", minimum=1, default="2"),
    "WEBRTC_LIFETIME_COUNTERS": _lever("serving", "bool", default="0"),
    "WEBRTC_LIFETIME_SEND_RING": _lever("serving", "int", minimum=1, default="256"),
    "WEBRTC_GROUP_MAX_COUNT": _lever("serving", "int", minimum=1, default="12"),
    "MUSETALK_DISABLE_LOCAL_TTS": _lever("serving", "bool", default="0"),
    "MUSETALK_THREAD_CAPS": _lever("serving", "bool", default="0"),
    "MUSETALK_IDLE_DECODE_THREADS": _lever("serving", "int", minimum=1, default="1"),
    "MUSETALK_MOTION_DECODE_THREADS": _lever("serving", "int", minimum=1),
    "MUSETALK_TORCH_INTRAOP_THREADS": _lever("serving", "int", minimum=0, default="4"),
    "MUSETALK_CV2_THREADS": _lever("serving", "int", minimum=0),
    "MUSETALK_FFMPEG_EXECUTOR_WORKERS": _lever("serving", "int", minimum=1, default="4"),
    "HLS_GPU_PIPELINE_DEPTH": _lever("serving", "int", minimum=1, maximum=4, default="1",
                                     note="clamped to 1..4 by hls_gpu_scheduler"),
    "HLS_GPU_OUTPUT_RING": _lever("serving", "int", minimum=0, default="0"),
    "HLS_GPU_BLOCKING_WAIT": _lever("serving", "bool", default="0"),
    "HLS_SCHEDULER_POLICY": _lever("serving", "choice", ["roundrobin", "edf"], default="roundrobin", raises=True),
    "HLS_SKIP_GPU_FOR_RAW": _lever("serving", "bool", default="0"),
    "HLS_SKIP_CROSSFADE_COPY": _lever("serving", "bool", default="0"),
    "MUSETALK_WHISPER_STREAM": _lever("serving", "bool", default="0"),
    # timing
    "HLS_GPU_EVENT_TIMING": _lever("timing", "bool", default="0"),
    "HLS_GPU_STAGE_SYNC_TIMING": _lever("timing", "bool", default="1"),
    "MUSETALK_VAE_DECODE_TIMING_SYNC": _lever("timing", "bool", default="1"),
    # avatar memory layout
    "MUSETALK_AVATAR_MASK_CHANNELS": _lever("memory", "choice", ["1", "3"], default="3", raises=True),
    "MUSETALK_AVATAR_MASK_STORE": _lever("memory", "choice", ["decoded", "png"], default="decoded", raises=True),
    "MUSETALK_AVATAR_FRAME_STORE": _lever("memory", "choice", ["decoded", "png"], default="decoded", raises=True),
    "MUSETALK_AVATAR_DECODED_LRU_FRAMES": _lever("memory", "int", minimum=0, default="24", raises=True),
    "MUSETALK_AVATAR_PNG_READAHEAD": _lever("memory", "int", minimum=0, default="8", raises=True),
    "MUSETALK_AVATAR_PNG_DECODE_WORKERS": _lever("memory", "int", minimum=0, default="4", raises=True),
    "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA": _lever("memory", "choice", ["0", "1"], default="1", raises=True),
    "MUSETALK_AVATAR_DEDUP_CYCLE": _lever("memory", "bool", default="1"),
    "MALLOC_ARENA_MAX": _lever("memory", "int", minimum=1, default="(glibc: 8 x cores)",
                               note="read by glibc malloc, not by the Python code (implemented=False is expected)"),
    # smoke-test only
    "WEBRTC_HANDOFF_VERIFY": _lever("debug", "bool", default="0", note="smoke-test only"),
    "WEBRTC_PREENCODE_SHA_DIR": _lever("debug", note="smoke-test only"),
}

# Knobs a recipe file may not set: they define the fast contract itself.
PROTECTED_RECIPE_KNOBS = {"MUSETALK_RECIPE", "MUSETALK_VAE_BACKEND", "MUSETALK_TRT_FALLBACK", "MUSETALK_TRT_ENABLED",
                          "MUSETALK_TRT_UNET_ENABLED", "MUSETALK_TRT_UNET_PATHS", "MUSETALK_TRT_UNET_PATH",
                          "MUSETALK_UNET_CALIBRATION_CAPTURE", "MUSETALK_VAE_CALIBRATION_CAPTURE", "MUSETALK_COMPILE",
                          "MUSETALK_RECIPE_FILE"}
BUCKET_KNOBS = {"HLS_SCHEDULER_FIXED_BATCH_SIZES", "MUSETALK_TAESD_WARMUP_BATCHES",
                "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"}
# Knobs the resolver computes itself (or reads as inputs). A recipe-file line for one of them enters
# the recipe layer directly; every other recipe line is a lever that must pass its gate.
MANAGED_KNOBS = BUCKET_KNOBS | {
    "HLS_SCHEDULER_MAX_BATCH", "HLS_SCHEDULER_STARTUP_SLICE_SIZE", "MUSETALK_TAESD_COMPILE", "MUSETALK_WARM_RUNTIME",
    "HLS_PREP_WORKERS", "HLS_COMPOSE_WORKERS", "HLS_ENCODE_WORKERS", "MUSETALK_AVATAR_LOAD_WORKERS",
    "HLS_MAX_PENDING_JOBS", "MUSETALK_WHISPER_SEGMENT_BATCH_SIZE", "AVATAR_CACHE_MAX_MEMORY_MB",
    "AVATAR_CACHE_TTL_SECONDS", "GPU_TOTAL_MEMORY_GB", "MUSETALK_BLEND_FIXED_POINT", "MUSETALK_BLEND_SHRINK_MASK_BBOX",
    "WEBRTC_BATCH_FRAME_CALLBACK", "WEBRTC_SYNC_MODE", "WEBRTC_AUDIO_SYNC_STRATEGY", "WEBRTC_VIDEO_PREBUFFER_SECONDS",
    "WEBRTC_ADAPTIVE_FPS", "WEBRTC_TRIM_EDGE_SILENCE", "WEBRTC_POSE_CROSSFADE_FRAMES",
    "WEBRTC_POSE_FORCED_CROSSFADE_FRAMES", "WEBRTC_POSE_MAX_SEMANTIC_DRIFT_SECONDS", "HLS_CHUNK_VIDEO_ENCODER",
    "HLS_CHUNK_ENCODER_PRESET", "HLS_CHUNK_ENCODER_CRF", "HLS_CHUNK_PREPARE_AUDIO_SIDECAR", "PROFILE",
    "MUSETALK_UNET_MODE", "MUSETALK_TRT_UNET_MIN_VRAM_GB", "MUSETALK_TRT_UNET_MIN_MEM_AVAILABLE_GB",
}

# In-flight flag registries (read with ast, never imported/edited): (file, dict variable, has defaults).
FLAG_REGISTRIES = (
    ("scripts/webrtc_media_flags.py", "FLAGS", True),
    ("scripts/hls_gpu_scheduler.py", "SCHEDULER_PIPELINE_FLAGS", True),
    ("scripts/vae_fast_decoder.py", "_TRT_FLAG_DOC", False),
)

# Files that mention knob names without reading them (excluded from the "implemented" scan).
_SCAN_EXCLUDE_RE = re.compile(r"^(test_|bench|benchmark_|validate_|build_|replay_|video_ab|experiment_|select_|"
                              r"install_|musetalk_host_profile|musetalk_engine_keys|unet_engine_store|"
                              r"musetalk_selftest|profile_|inspect_|measure_|audit_|review_)")


def parse_flag_registry(path, variable):
    """{name: value} of a module-level dict literal, parsed with ast (the module is never executed, so a
    torch import in an in-flight module cannot leak into the resolver). {} when absent/unparseable."""
    import ast

    text = _read_text(path)
    if not text:
        return {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if any(isinstance(t, ast.Name) and t.id == variable for t in targets):
            try:
                data = ast.literal_eval(value)
            except ValueError:
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def load_flag_registries(repo_root: Path) -> dict:
    """{name: {"default": str|None, "registry": file}} from every in-flight flag registry."""
    out = {}
    for rel, variable, has_defaults in FLAG_REGISTRIES:
        for name, value in parse_flag_registry(Path(repo_root) / rel, variable).items():
            default = None
            if has_defaults and isinstance(value, (tuple, list)) and value:
                default = str(value[0])
            out.setdefault(str(name), {"default": default, "registry": rel})
    return out


class KnobScanner:
    """Which env names does the serving code actually read? (quoted-string scan, cached)."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._names = None

    def _files(self):
        root = self.repo_root
        files = [root / "api_server.py"]
        scripts = root / "scripts"
        if scripts.is_dir():
            files += [p for p in sorted(scripts.glob("*.py")) if not _SCAN_EXCLUDE_RE.match(p.name)]
        pkg = root / "musetalk"
        if pkg.is_dir():
            files += sorted(pkg.rglob("*.py"))
        return [f for f in files if f.is_file()]

    def names(self):
        if self._names is None:
            found = set()
            pattern = re.compile(r"[\"']([A-Z][A-Z0-9_]{2,})[\"']")
            for path in self._files():
                found.update(pattern.findall(_read_text(path, "")))
            self._names = found
        return self._names

    def reads(self, name) -> bool:
        return name in self.names()


def validate_lever_value(name, value):
    """None if acceptable, else an error string."""
    spec = LEVERS.get(name)
    if spec is None:
        return None
    text = str(value).strip()
    if text == "":
        return None  # every reader treats an empty value as its code default
    kind = spec["kind"]
    if kind == "bool":
        return None if parse_bool(text) is not None else f"{name}={value!r} is not a boolean (0/1)"
    if kind == "int":
        number = _to_int(text)
        if number is None:
            return f"{name}={value!r} is not an integer"
        if spec["min"] is not None and number < spec["min"]:
            return f"{name}={value!r} is below {spec['min']}"
        if spec["max"] is not None and number > spec["max"]:
            return f"{name}={value!r} is above {spec['max']}"
        return None
    if kind == "choice":
        return None if text.lower() in spec["choices"] else f"{name}={value!r} not in {spec['choices']}"
    return None


# --------------------------------------------------------------------------- layered view
class EnvView:
    """caller env > overrides files > recipe levers > resolver values."""

    def __init__(self, environ, repo_root: Path):
        self.caller = dict(environ)
        self.override_files = overrides_files(repo_root, self.caller)
        self.override_files_used = []
        self.overrides = {}  # key -> (value, file)
        for path in self.override_files:
            entries = parse_env_file(path)
            if Path(path).is_file():
                self.override_files_used.append(str(path))
            for key, value, _lineno in entries:
                if key not in self.overrides:  # first file wins (only-if-unset loading)
                    self.overrides[key] = (value, str(path))
        # Hints from scripts/run_musetalk_server.sh, which loads the overrides before calling us:
        # comma lists of keys whose current value came from the caller / an overrides file.
        over_keys = self.caller.get("MUSETALK_ENV_OVERRIDE_KEYS")
        caller_keys = self.caller.get("MUSETALK_ENV_CALLER_KEYS")
        self.override_hint = None if over_keys is None else {k.strip() for k in over_keys.split(",") if k.strip()}
        self.caller_hint = None if caller_keys is None else {k.strip() for k in caller_keys.split(",") if k.strip()}
        self.recipe = {}  # key -> (value, recipe name)
        self.resolved = {}  # key -> value

    def upper(self, key):
        """(value, source) from the caller or overrides layers, or (None, None)."""
        if key in self.caller:
            value = self.caller[key]
            over = self.overrides.get(key)
            where = over[1] if over is not None else "overrides file"
            if self.override_hint is not None and key in self.override_hint:
                return value, f"overrides:{where}"
            if self.caller_hint is not None and key in self.caller_hint:
                return value, "caller"
            if self.override_hint is None and over is not None and over[0] == value:
                return value, f"overrides:{where}"  # no launcher hint: infer from the value
            return value, "caller"
        if key in self.overrides:
            value, path = self.overrides[key]
            return value, f"overrides:{path}"
        return None, None

    def get(self, key, default=None):
        value, _src = self.lookup(key)
        return default if value is None else value

    def lookup(self, key):
        value, src = self.upper(key)
        if src is not None:
            return value, src
        if key in self.recipe:
            return self.recipe[key][0], f"recipe:{self.recipe[key][1]}"
        if key in self.resolved:
            return self.resolved[key], "resolver"
        return None, None

    def is_upper(self, key) -> bool:
        return self.upper(key)[1] is not None


# --------------------------------------------------------------------------- resolver
STATE_PREREQS = {"engine:unet_ts", "unet:trt_any"}  # depend on the resolved UNet; checked after a pass
EXPENSIVE_PREREQS = {"vp8:native_preflight", "gpu:nvenc"}  # subprocesses; only when the cheap ones pass


class Resolver:
    """One resolution pass. resolve_recipe() re-runs passes until every @lever group is either fully
    enabled or excluded (groups are atomic), sharing the expensive caches between passes."""

    def __init__(self, repo_root, venv, recipe, facts, environ=None, excluded_groups=None, shared=None):
        self.repo_root = Path(repo_root).resolve()
        self.venv = Path(venv) if venv else None
        self.environ = os.environ if environ is None else environ
        self.view = EnvView(self.environ, self.repo_root)
        self.recipe = recipe
        self.facts = facts
        self.shared = shared if shared is not None else {}
        self.shared.setdefault("prereq", {})
        self.store = self.shared.setdefault("store", EngineStore(self.repo_root, self.view.get))
        self.scanner = self.shared.setdefault("scanner", KnobScanner(self.repo_root))
        self.excluded_groups = dict(excluded_groups or {})
        self.recipe_groups = []
        self.passes = 1
        self.decisions = []
        self.emitted = {}
        self.warnings = []
        self.errors = []
        self.notes = []
        self.recipe_levers = []
        self.engines = {}
        self.unet_entry = None  # store entry of the resolved .ts UNet (None for caller-supplied paths)
        self.expect = {"vae": "taesd", "unet": "any"}
        self.unet = {"backend": None, "engine": None}
        self.recipe_file = None

    # -- helpers ---------------------------------------------------------------------------
    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def error(self, msg):
        if msg not in self.errors:
            self.errors.append(msg)

    def emit(self, knob, value, reason):
        """Record a resolver value; the effective value (caller/overrides/recipe win) is what gets emitted."""
        value = str(value)
        self.view.resolved[knob] = value
        effective, source = self.view.lookup(knob)
        if source.startswith("recipe:"):
            reason = f"{source} lever ({reason})" if effective == value else \
                f"{source} sets {effective!r} (resolver default {value!r}: {reason})"
        elif source != "resolver":
            if effective != value:
                reason = f"{source} set {effective!r} (resolver would pick {value!r}: {reason})"
            else:
                reason = f"{source} set it (same as resolver: {reason})"
            if effective == "":
                self.warn(f"{knob} is exported EMPTY by {source}: it blocks the resolved value and the server "
                          f"falls back to its code default")
        self.emitted[knob] = effective
        self.decisions.append({"knob": knob, "value": effective, "source": source, "reason": reason})
        return effective

    def float_knob(self, name, default):
        value = _to_float(self.view.get(name))
        return default if value is None else value

    # -- entry point -----------------------------------------------------------------------
    def run(self):
        if self.recipe not in RECIPES:
            self.error(f"unknown recipe {self.recipe!r} (expected one of {', '.join(RECIPES)})")
            return self
        # The recipe actually resolved is the truth for this knob (the launcher dispatches on it).
        self.emitted["MUSETALK_RECIPE"] = self.recipe
        self.decisions.append({"knob": "MUSETALK_RECIPE", "value": self.recipe, "source": "resolver",
                               "reason": "selected recipe (--recipe, else MUSETALK_RECIPE, else overrides, else fast)"})
        if self.recipe == "legacy_int8":
            self.expect = {"vae": "any", "unet": "any"}
            self.notes.append("legacy_int8: the launcher execs scripts/run_trt_stagewise_server.sh (old chain)")
            self._levers_report()
            return self
        self._gpu_checks()
        self._core_recipe()
        if self.recipe in RECIPE_FILE_RECIPES:
            self._load_recipe_levers()
        self._resolve_unet()
        self._resolve_buckets()
        self._resolve_vae_levers()
        self._cpu_ram()
        self._blend_webrtc()
        self._encoders()
        self._chin_tools()
        self._validate_upper_levers()
        self._apply_remaining_recipe_levers()
        self._levers_report()
        self._estimate()
        return self

    # -- GPU ------------------------------------------------------------------------------
    def _gpu_checks(self):
        gpu = self.facts.get("gpu")
        venv = self.facts.get("venv") or {}
        allow_no_gpu = parse_bool(self.view.get("MUSETALK_ALLOW_NO_GPU")) is True
        if not venv.get("exists"):
            self.warn(f"venv not found at {venv.get('path')}: package facts unknown")
        if gpu is None:
            msg = f"no usable NVIDIA GPU ({self.facts.get('gpu_selection')}; nvidia-smi: " \
                  f"{(self.facts.get('nvidia_smi') or {}).get('error') or 'ok'})"
            if allow_no_gpu:
                self.warn(msg + "; continuing because MUSETALK_ALLOW_NO_GPU=1")
            else:
                self.error(msg + " (set MUSETALK_ALLOW_NO_GPU=1 to resolve anyway)")
            return
        cc = cc_tuple(gpu.get("compute_capability"))
        tag_num = _cuda_tag_number(venv.get("torch_cuda_tag"))
        if cc is None:
            self.warn(f"compute capability of {gpu.get('name')} unknown (old driver?): TRT engines disabled")
        elif cc >= (10, 0) and venv.get("torch") and (tag_num is None or tag_num < 128):
            self.error(f"{gpu.get('name')} is compute capability {gpu.get('compute_capability')} (Blackwell) but the "
                       f"venv torch is {venv.get('torch')} ({venv.get('torch_cuda_tag')}); it has no SASS for "
                       f"sm{cc[0]}{cc[1]}. Reinstall: scripts/install_musetalk.sh --matrix cu128 --clean")
        elif cc < (5, 0):
            self.warn(f"compute capability {gpu.get('compute_capability')} is below sm50: unsupported by torch")
        limit, default = gpu.get("power_limit_w"), gpu.get("power_default_limit_w")
        if limit and default and limit < POWER_CAP_RATIO * default:
            self.warn(f"GPU power-capped: expect lower fps (power.limit {limit:.0f} W < "
                      f"{POWER_CAP_RATIO:.0%} of default {default:.0f} W)")
        vram_mib = gpu.get("memory_total_mib") or 0
        if vram_mib and vram_mib < 8000:
            self.warn(f"VRAM {vram_mib / 1024:.1f} GB < 8 GB: TRT UNet disabled in auto mode, expect OOM risk")
        ram = self.facts.get("ram") or {}
        if ram.get("effective_total_mb") and ram["effective_total_mb"] < 16 * 1024:
            self.warn(f"host RAM {ram['effective_total_mb'] / 1024:.1f} GB < 16 GB total")

    def _vram_gb(self):
        gpu = self.facts.get("gpu") or {}
        mib = gpu.get("memory_total_mib")
        return (mib / 1024.0) if mib else None

    def _vram_ok(self, min_gb):
        """Nominal GB (1000 MiB) so an '8 GB' card reporting 8188 MiB passes an 8 GB threshold."""
        gpu = self.facts.get("gpu") or {}
        mib = gpu.get("memory_total_mib")
        return bool(mib) and mib >= min_gb * 1000

    def _selftest(self):
        # Same override the installer uses for --selftest-out (install_musetalk.sh MUSETALK_GPU_SELFTEST_FILE).
        raw = (self.view.get("MUSETALK_GPU_SELFTEST_FILE") or "").strip()
        path = Path(raw) if raw else self.repo_root / ".runtime" / "gpu_selftest.json"
        if not path.is_absolute():
            path = self.repo_root / path
        data = _read_json(path)
        if not isinstance(data, dict):
            return None, None
        gpu = self.facts.get("gpu") or {}
        venv = self.facts.get("venv") or {}
        st_gpu = data.get("gpu") or {}
        if data.get("schema") not in (None, SELFTEST_SCHEMA):
            return None, f"{path}: unexpected schema {data.get('schema')!r}"
        if st_gpu.get("name") != gpu.get("name") or cc_tuple(st_gpu.get("compute_capability")) != \
                cc_tuple(gpu.get("compute_capability")):
            return None, f"{path} is for {st_gpu.get('name')} sm{st_gpu.get('compute_capability')}, ignored"
        if data.get("torch_version") != venv.get("torch"):
            return None, f"{path} was made with torch {data.get('torch_version')} (venv {venv.get('torch')}), ignored"
        return data, None

    # -- core fast recipe --------------------------------------------------------------------
    def _core_recipe(self):
        vae = self.emit("MUSETALK_VAE_BACKEND", "taesd", "fast recipe decoder (compiled TAESD)")
        if vae.strip().lower() not in ("taesd", "tiny", "tiny_vae"):
            self.warn(f"MUSETALK_VAE_BACKEND={vae!r} overrides the fast recipe's TAESD decoder")
        selftest, why = self._selftest()
        if why:
            self.notes.append(why)
        compile_value, reason = "1", "compiled TAESD (torch.compile per warmed batch)"
        if selftest and (selftest.get("taesd") or {}).get("compile_ok") is False:
            compile_value = "0"
            reason = f"gpu_selftest.json: TAESD compile failed on this GPU ({(selftest.get('taesd') or {}).get('error')})"
            self.warn("TAESD torch.compile failed in the GPU self-test: serving eager TAESD (slower)")
        self.emit("MUSETALK_TAESD_COMPILE", compile_value, reason)
        self.emit("MUSETALK_TRT_FALLBACK", "0", "no silent fallback (one flag covers UNet/TRT-VAE/TAESD fallbacks)")
        self.emit("MUSETALK_TRT_ENABLED", "0", "no legacy single-engine TRT VAE")
        self.emit("MUSETALK_COMPILE", "0", "torch.compile UNet failed its gate; TRT/TAESD paths skip it anyway")
        self.emit("MUSETALK_WARM_RUNTIME", "1", "warm audio + Whisper path at startup")
        self.emit("MUSETALK_UNET_CALIBRATION_CAPTURE", "0", "no capture in serving")
        self.emit("MUSETALK_VAE_CALIBRATION_CAPTURE", "0", "no capture in serving")

    # -- recipe file ---------------------------------------------------------------------------
    def _recipe_file_path(self):
        raw = self.view.get("MUSETALK_RECIPE_FILE")
        if raw:
            path = Path(raw)
            return path if path.is_absolute() else self.repo_root / path
        return self.repo_root / "configs" / "recipes" / f"{self.recipe}.env"

    def _load_recipe_levers(self):
        path = self._recipe_file_path()
        self.recipe_file = str(path)
        self._pending_recipe = []
        if not path.is_file():
            self.warn(f"recipe file {path} missing: {self.recipe} == fast")
            return
        groups, entries = parse_recipe_file(path)
        self.recipe_groups = [dict(g, status=None, reason=None, prereqs={}) for g in groups]
        if not entries:
            self.notes.append(f"{path}: no lever enabled (all commented out): {self.recipe} == fast")
        for group in self.recipe_groups:  # static prerequisites, once per group (cached across passes)
            if group["name"] not in self.excluded_groups and group["keys"]:
                ok, why = self._check_prereqs(group, entries, state=False)
                if not ok:
                    self.excluded_groups[group["name"]] = why
        for key, value, lineno, gname in entries:
            item = {"name": key, "requested": value, "line": lineno, "group": gname, "status": None, "reason": None}
            self.recipe_levers.append(item)
            if gname and gname in self.excluded_groups:
                self._drop(item, f"group {gname} dropped: {self.excluded_groups[gname]}")
                continue
            if key in PROTECTED_RECIPE_KNOBS:
                self._drop(item, "core fast-recipe knob; a recipe file may not change it (use the overrides file)")
                continue
            if self.view.is_upper(key):
                value_up, src = self.view.upper(key)
                item["status"], item["reason"] = "overridden", f"{src} sets {value_up!r} (caller/overrides win)"
                continue
            bad = validate_lever_value(key, value)
            if key in BUCKET_KNOBS and not bad:
                try:
                    parse_buckets(value)
                except ValueError as exc:
                    bad = f"{key}={value!r} is not a batch list ({exc})"
            if bad:
                self._drop(item, bad)
                continue
            spec = LEVERS.get(key)
            if spec and spec["category"] == "debug":
                self._drop(item, "smoke-test-only flag; never part of a serving recipe")
                continue
            if key in MANAGED_KNOBS:
                # A resolver-managed knob (buckets, workers, thresholds...): it enters the recipe layer now
                # so every dependent (bucket coupling, UNet gates) is computed from it.
                self._enable(item, "recipe value for a resolver-managed knob (dependents computed from it)")
                continue
            self._pending_recipe.append(item)

    # -- @lever prerequisites ----------------------------------------------------------------------
    def _group_value(self, entries, group, key):
        for k, v, _lineno, gname in entries:
            if gname == group["name"] and k == key:
                return v
        return None

    def _check_prereqs(self, group, entries, state):
        """(ok, reason). state=False: static prerequisites (cheap first, subprocesses last);
        state=True: the ones that depend on the resolved UNet."""
        wanted = [p for p in group["requires"] if (p in STATE_PREREQS) == state]
        wanted.sort(key=lambda p: 1 if p in EXPENSIVE_PREREQS else 0)
        for prereq in wanted:
            ok, why = self._prereq(prereq, group, entries)
            group["prereqs"][prereq] = {"ok": ok, "detail": why}
            if not ok:
                return False, f"prerequisite {prereq} failed: {why}"
        return True, None

    def _prereq(self, prereq, group, entries):
        kind, _, arg = prereq.partition(":")
        gpu = self.facts.get("gpu")
        if prereq == "engine:unet_stagewise":
            if gpu is None:
                return False, "no GPU visible"
            batch = _to_int(self._group_value(entries, group, "MUSETALK_UNET_STAGEWISE_BATCH"))
            entry, _ = self.store.find("unet_stagewise", self.facts, batch=batch)
            if entry is None:
                legacy = self._matching_legacy_stagewise(batch)
                return False, (f"no unet_stagewise engine validated for key "
                               f"{self.store.engine_key('unet_stagewise', self.facts)}"
                               + (f" at bs{batch}" if batch else "")
                               + (f" (legacy set {[c['path'] for c in legacy]} matches: scripts/unet_engine_store.py "
                                  "adopt --kind unet_stagewise, then validate)" if legacy else ""))
            return True, f"{entry.get('key')} bs{entry.get('batch')}"
        if prereq == "engine:taesd_trt":
            if gpu is None:
                return False, "no GPU visible"
            batch = _to_int(self._group_value(entries, group, "MUSETALK_TAESD_TRT_BATCH"))
            entry, _ = self.store.find("taesd_trt", self.facts, batch=batch, prefer_batch=8)
            if entry is None:
                return False, (f"no taesd_trt engine validated for key {self.store.engine_key('taesd_trt', self.facts)}"
                               " (scripts/unet_engine_store.py ensure --kind taesd_trt)")
            verdict = quality_verdict(entry)
            if verdict != "PASS":
                return False, (f"taesd_trt engine {entry.get('dir')} is store-usable but its G-TAESD quality gate is "
                               f"{verdict or 'not recorded'} (needs PASS in the engine meta)")
            return True, f"{entry.get('key')} bs{entry.get('batch')} G-TAESD PASS"
        if prereq == "engine:unet_ts":
            ok = self.unet.get("backend") == "trt"
            if not ok:
                return False, f"resolved UNet is {self.unet.get('backend')} (needs the validated .ts)"
            # A CUDA-graph lever also needs the store's record for that mode (unet_engine_store.py
            # validate --kind unet_ts --cudagraphs <mode> writes validation.modes.cudagraphs_<mode>).
            mode = str(self._group_value(entries, group, "MUSETALK_TRT_UNET_CUDAGRAPHS") or "").strip().lower()
            if mode in ("manual", "runtime"):
                entry = self.unet_entry
                if entry is None:
                    return False, (f"MUSETALK_TRT_UNET_CUDAGRAPHS={mode} needs a store-validated .ts; the resolved "
                                   "engine came from caller MUSETALK_TRT_UNET_PATHS (no validation record)")
                validation = (entry.get("fingerprint") or {}).get("validation") or {}
                record = (validation.get("modes") or {}).get(f"cudagraphs_{mode}") or {}
                if record.get("passed") is not True:
                    return False, (f".ts engine {entry.get('key')} has no passed cudagraphs_{mode} validation "
                                   f"(status {record.get('status') or 'not_run'}; run scripts/unet_engine_store.py "
                                   f"validate --kind unet_ts --cudagraphs {mode})")
                return True, f"resolved UNet is trt; cudagraphs_{mode} validated for {entry.get('key')}"
            return True, "resolved UNet is trt"
        if prereq == "unet:trt_any":
            ok = self.unet.get("backend") in ("trt", "trt_stagewise")
            return ok, f"resolved UNet is {self.unet.get('backend')}"
        if prereq == "h264:native_fallback":
            return self.native_vp8_keeps_h264()
        if prereq == "vp8:native_preflight":
            return self.native_vp8_preflight()
        if prereq == "gpu:nvenc":
            return self.pyav_has_encoder("h264_nvenc") if gpu is not None else (False, "no GPU visible")
        if kind == "bundle":
            return self._cached(prereq, lambda: self._bundle_check(arg))
        if kind == "code":
            path = self.repo_root / arg
            first = group["keys"][0] if group["keys"] else None
            text = _read_text(path)
            if text is None:
                return False, f"{arg} missing"
            if first and first not in text:
                return False, f"{arg} does not mention {first} (lever not implemented in this checkout)"
            return True, f"{arg} mentions {first}"
        return False, f"unknown prerequisite {prereq!r}"

    def _bundle_check(self, name):
        """(ok, why) for bundle:<name>: configs/trt_bundles/<name>.json is for this exact engine key,
        its stamp binds the stored sidecars to the pinned archive SHA256, and every file of the
        bundle manifest is present with its recorded size (stat only; the restore hashed them)."""
        desc_path = self.repo_root / TRT_BUNDLE_DIR / f"{name}.json"
        try:
            desc = json.loads(desc_path.read_text())
        except (OSError, ValueError) as exc:
            return False, f"{desc_path} unreadable ({type(exc).__name__})"
        if self.facts.get("gpu") is None:
            return False, "no GPU visible"
        want = (desc.get("host") or {}).get("engine_key")
        have = self.store.engine_key("unet_stagewise", self.facts)
        if not want or have != want:
            return False, (f"bundle is for {want}; this host is {have} (raw TensorRT plans load only on the "
                           "exact GPU model + TensorRT they were built on)")
        sidecar_dir = self.repo_root / str(desc.get("sidecar_dir") or f".runtime/trt_artifacts/{name}")
        try:
            stamp = json.loads((sidecar_dir / TRT_BUNDLE_STAMP).read_text())
        except (OSError, ValueError):
            stamp = {}
        sha = str(desc.get("sha256") or "").lower()
        if not sha or str(stamp.get("archive_sha256") or "").lower() != sha:
            return False, (f"not restored: {sidecar_dir / TRT_BUNDLE_STAMP} does not bind archive {sha[:12]} "
                           "(scripts/vast_onstart.sh restores it for recipe r5; by hand: "
                           "scripts/trt_artifact_bundle.py --sidecar-dir ... restore|adopt)")
        try:
            files = json.loads((sidecar_dir / TRT_BUNDLE_MANIFEST).read_text()).get("files") or []
        except (OSError, ValueError):
            files = []
        if not files:
            return False, f"{sidecar_dir / TRT_BUNDLE_MANIFEST} missing or empty"
        bad = []
        for entry in files:
            try:
                size = (self.repo_root / entry["path"]).stat().st_size
            except (OSError, KeyError, TypeError):
                bad.append(f"missing {entry.get('path') if isinstance(entry, dict) else entry}")
                continue
            if size != entry.get("size"):
                bad.append(f"size changed {entry['path']}")
        if bad:
            return False, f"{len(bad)} bundle file(s) missing or changed since the restore: {bad[:3]}"
        self.shared.setdefault("bundles", {})[name] = desc
        return True, (f"{name} {stamp.get('mode') or 'restored'} {stamp.get('restored_at')} for {want}; "
                      f"{len(files)} files present")

    def _group_bundle(self, item):
        """Descriptor of the pinned bundle that the item's @lever group requires (and that passed)."""
        if not item or not item.get("group"):
            return None
        for group in self.recipe_groups:
            if group["name"] == item["group"]:
                for prereq in group["requires"]:
                    if prereq.startswith("bundle:"):
                        return self.shared.get("bundles", {}).get(prereq.partition(":")[2])
        return None

    def _cached(self, key, fn):
        cache = self.shared["prereq"]
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    def _venv_python(self):
        py = (self.facts.get("venv") or {}).get("python")
        return py if py and os.access(py, os.X_OK) else None

    def native_vp8_preflight(self):
        """(ok, why): static prerequisites, then the contract's CPU preflight in the venv (CUDA hidden)."""
        def run():
            ok, problems = self.native_vp8_static_check()
            if not ok:
                return False, "; ".join(problems)
            py = self._venv_python()
            if not py:
                return False, "venv python not executable"
            env = {k: v for k, v in self.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
                   or k.startswith("WEBRTC_NATIVE_VP8_")}
            env.update({"WEBRTC_VP8_ENCODER": "native", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                        "WEBRTC_NATIVE_VP8_DIR": self.view.get("WEBRTC_NATIVE_VP8_DIR")
                        or str(self.repo_root / ".runtime" / "native_vp8")})
            code = "from scripts import webrtc_native_vp8 as n; n.configure_vp8_encoder('preflight')"
            try:
                proc = subprocess.run([py, "-B", "-c", code], cwd=str(self.repo_root), env=env, timeout=60,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"preflight could not run: {type(exc).__name__}: {exc}"
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
                return False, f"preflight failed: {tail[0][:300]}"
            return True, "preflight passed"
        return self._cached("vp8:native_preflight", run)

    def pyav_has_encoder(self, codec):
        """(ok, why): PyAV in the venv lists the encoder (no GPU is opened; the runtime falls back to
        x264tuned when NVENC cannot open or the session cap is reached)."""
        def run():
            py = self._venv_python()
            if not py:
                return False, "venv python not executable"
            code = "import sys, av; av.codec.Codec(sys.argv[1], 'w')"
            env = {k: v for k, v in self.environ.items() if k in ("PATH", "HOME", "LANG", "TMPDIR")}
            env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
            try:
                proc = subprocess.run([py, "-B", "-c", code, codec], env=env, timeout=60, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, universal_newlines=True)
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"check could not run: {type(exc).__name__}: {exc}"
            return (proc.returncode == 0,
                    f"PyAV {'has' if proc.returncode == 0 else 'lacks'} the {codec} encoder")
        return self._cached(f"pyav:{codec}", run)

    def group_violations(self):
        """{group: reason} for groups (not yet excluded) that are partially dropped or whose
        state prerequisites fail after this pass."""
        out = {}
        for group in self.recipe_groups:
            name = group["name"]
            if name in self.excluded_groups or not group["keys"]:
                continue
            items = [i for i in self.recipe_levers if i.get("group") == name]
            dropped = [i for i in items if i["status"] == "dropped"]
            if dropped:
                out[name] = "; ".join(f"{i['name']} dropped ({i['reason']})" for i in dropped)
                continue
            ok, why = self._check_prereqs(group, [(i["name"], i["requested"], i["line"], name) for i in items],
                                          state=True)
            if not ok:
                out[name] = why
        return out

    def finalize_groups(self):
        for group in self.recipe_groups:
            items = [i for i in self.recipe_levers if i.get("group") == group["name"]]
            if not group["keys"]:
                group["status"], group["reason"] = "off", "every line is commented out"
            elif group["name"] in self.excluded_groups:
                group["status"], group["reason"] = "dropped", self.excluded_groups[group["name"]]
            elif items and all(i["status"] == "overridden" for i in items):
                group["status"], group["reason"] = "overridden", "caller/overrides set every key of the group"
            else:
                group["status"], group["reason"] = "enabled", "all prerequisites hold"

    def _drop(self, item, reason):
        item["status"], item["reason"] = "dropped", reason
        self.view.recipe.pop(item["name"], None)

    def _enable(self, item, reason):
        item["status"], item["reason"] = "enabled", reason
        self.view.recipe[item["name"]] = (item["requested"], self.recipe)

    def _pending(self, name):
        for item in getattr(self, "_pending_recipe", []):
            if item["name"] == name and item["status"] is None:
                return item
        return None

    def _apply_remaining_recipe_levers(self):
        """Serving/timing/memory levers from the recipe file: enabled when code reads them."""
        for item in getattr(self, "_pending_recipe", []):
            if item["status"] is not None:
                continue
            name = item["name"]
            if item.get("group"):
                # An explicit @lever group states its own prerequisites (code:<path> etc.).
                self._enable(item, f"@lever group {item['group']}")
            elif not self.scanner.reads(name):
                self._drop(item, f"no serving code in this tree reads {name} (lever not implemented yet)")
                continue
            else:
                self._enable(item, "pass-through lever; code reads it" + (
                    "" if name in LEVERS else " (no prerequisite gate defined for this name)"))
            self.emitted[name] = item["requested"]
            self.decisions.append({"knob": name, "value": item["requested"], "source": f"recipe:{self.recipe}",
                                   "reason": item["reason"]})

    # -- UNet -----------------------------------------------------------------------------------
    def _unet_mode(self):
        mode = (self.view.get("MUSETALK_UNET_MODE") or "auto").strip().lower()
        if mode not in ("auto", "trt", "eager"):
            self.error(f"MUSETALK_UNET_MODE={mode!r} must be auto, trt or eager")
            mode = "auto"
        return mode

    def _resolve_unet(self):
        mode = self._unet_mode()
        backend_up, src = self.view.upper("MUSETALK_UNET_BACKEND")
        enabled_up = parse_bool(self.view.upper("MUSETALK_TRT_UNET_ENABLED")[0])
        stagewise_item = self._pending("MUSETALK_UNET_BACKEND")
        if backend_up is not None and backend_up.strip() != "":
            b = backend_up.strip().lower()
            if b in ("trt_stagewise", "tensorrt_stagewise"):
                self._drop_stagewise_levers(f"{src} already selects MUSETALK_UNET_BACKEND={backend_up}")
                return self._unet_stagewise(explicit=True, source=src)
            if stagewise_item:
                self._drop(stagewise_item, f"{src} sets MUSETALK_UNET_BACKEND={backend_up!r}")
            if b in ("trt", "tensorrt"):
                return self._unet_ts(mode="trt", why=f"{src} set MUSETALK_UNET_BACKEND={backend_up}")
            if b in ("eager", "pytorch", "torch"):
                return self._unet_eager(f"{src} set MUSETALK_UNET_BACKEND={backend_up}")
            self.warn(f"MUSETALK_UNET_BACKEND={backend_up!r} ({src}) is not a known backend; trt_runtime treats it "
                      "as PyTorch unless MUSETALK_TRT_UNET_ENABLED=1")
            self.expect["unet"] = "any"
            self.unet["backend"] = backend_up
            return None
        if enabled_up is True and mode != "eager":
            if stagewise_item:
                self._drop(stagewise_item, "caller set MUSETALK_TRT_UNET_ENABLED=1 (.ts engine)")
            return self._unet_ts(mode="trt", why="caller set MUSETALK_TRT_UNET_ENABLED=1")
        if stagewise_item:
            if mode != "auto":
                self._drop(stagewise_item, f"MUSETALK_UNET_MODE={mode} pins the UNet ({'.ts TRT' if mode == 'trt' else 'eager'})")
            elif self._unet_stagewise(explicit=False, source=f"recipe:{self.recipe}", item=stagewise_item):
                return None
        if mode == "eager":
            return self._unet_eager("MUSETALK_UNET_MODE=eager")
        return self._unet_ts(mode=mode, why=f"MUSETALK_UNET_MODE={mode}")

    def _drop_stagewise_levers(self, reason):
        for name in ("MUSETALK_UNET_BACKEND", "MUSETALK_UNET_STAGEWISE_BATCH"):
            item = self._pending(name)
            if item:
                self._drop(item, reason)

    def _buckets_hint(self):
        raw = self.view.get("HLS_SCHEDULER_FIXED_BATCH_SIZES")
        try:
            return parse_buckets(raw) if raw else None
        except ValueError:
            return None

    def _unet_eager(self, reason):
        self.emit("MUSETALK_UNET_BACKEND", "eager", reason)
        enabled = self.emit("MUSETALK_TRT_UNET_ENABLED", "0", reason)
        if parse_bool(enabled) is True:
            self.warn("MUSETALK_TRT_UNET_ENABLED=1 from the caller/overrides still requests the TRT UNet "
                      "(trt_runtime._trt_unet_requested) although the resolved UNet is eager")
        self.unet.update({"backend": "eager", "engine": None, "reason": reason})
        self.expect["unet"] = "eager"
        self._drop_ts_only_levers("UNet runs eager PyTorch")
        return None

    def _drop_ts_only_levers(self, reason):
        item = self._pending("MUSETALK_TRT_UNET_CUDAGRAPHS")
        if item:
            self._drop(item, f"needs the torch_tensorrt .ts UNet; {reason}")

    def _parse_ts_paths(self, raw):
        out = {}
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"entry {token!r} is not <batch>:<path>")
            batch, path = token.split(":", 1)
            out[int(batch)] = path
        return out

    def _unet_ts(self, mode, why):
        """TRT .ts UNet in auto|trt mode; falls back to eager (auto) or errors (trt)."""
        hard = mode == "trt"
        min_vram = self.float_knob("MUSETALK_TRT_UNET_MIN_VRAM_GB", 8.0)
        min_mem = self.float_knob("MUSETALK_TRT_UNET_MIN_MEM_AVAILABLE_GB", 10.0)
        gpu = self.facts.get("gpu")
        buckets = self._buckets_hint()
        reasons = []
        if gpu is None:
            if hard:
                self.error("MUSETALK_UNET_MODE/BACKEND asks for the TRT UNet but no GPU is visible")
            return self._unet_eager("no GPU visible")
        paths_up, paths_src = self.view.upper("MUSETALK_TRT_UNET_PATHS")
        entry, key = None, self.store.engine_key("unet_ts", self.facts)
        self.engines["unet_ts"] = {"key": key, "store": str(self.store.store_root("unet_ts")),
                                   "backend": self.store.backend}
        if paths_up:
            try:
                path_map = self._parse_ts_paths(paths_up)
            except ValueError as exc:
                self.error(f"MUSETALK_TRT_UNET_PATHS ({paths_src}) invalid: {exc}")
                return None
            missing = [p for p in path_map.values() if not (Path(p) if Path(p).is_absolute()
                                                               else self.repo_root / p).exists()]
            if missing:  # the server would fail to start with MUSETALK_TRT_FALLBACK=0, in any mode
                self.error(f"MUSETALK_TRT_UNET_PATHS ({paths_src}) names missing engines: {missing}")
                return None
            self.warn(f"MUSETALK_TRT_UNET_PATHS from {paths_src} is not checked against the engine store's "
                      "validation records")
            engine_batches = sorted(path_map)
            engine_desc = {"source": paths_src, "paths": path_map}
        else:
            # Contract: exact key, else same cc + TRT + torch_tensorrt from another GPU model (warned).
            entry, all_entries = self.store.find("unet_ts", self.facts, batch=TS_ENGINE_BATCH, allow_same_cc=True)
            self.engines["unet_ts"]["candidates"] = [compact_entry(e) for e in all_entries]
            if entry is None:
                hint = [c.get("path") for c in self.store.legacy("unet_ts")]
                self.engines["unet_ts"]["legacy_candidates"] = hint
                msg = (f"no usable unet_ts engine for key {key} in {self.store.store_root('unet_ts')}"
                       + (f" [{self.store.key_errors['unet_ts']}]" if self.store.key_errors.get("unet_ts") else "")
                       + (f" (unregistered legacy engine(s) {hint}: run scripts/unet_engine_store.py ensure "
                          f"--kind unet_ts --provision adopt)" if hint else
                          " (run scripts/unet_engine_store.py ensure --kind unet_ts --provision auto)"))
                if hard:
                    self.error(msg)
                    return None
                return self._unet_eager(f"auto: {msg}")
            if entry.get("match") == "same_cc":
                self.warn(f"TRT UNet engine {entry.get('dir')} is a same-sm engine from another GPU model "
                          f"({entry.get('reason')}); validated there, not on this exact card")
            engine_batches = [TS_ENGINE_BATCH]
            engine_desc = {"source": "store", "dir": entry.get("dir"), "key": entry.get("key"),
                           "match": entry.get("match"), "path": engine_path_of(entry)}
        if not self._vram_ok(min_vram):
            reasons.append(f"VRAM {self._vram_gb() or 0:.1f} GB < MUSETALK_TRT_UNET_MIN_VRAM_GB={min_vram:g}")
        avail_mb = (self.facts.get("ram") or {}).get("effective_available_mb") or 0
        low_ram = avail_mb < min_mem * 1024
        if low_ram:
            msg = (f"MemAvailable {avail_mb / 1024:.1f} GB < MUSETALK_TRT_UNET_MIN_MEM_AVAILABLE_GB={min_mem:g} "
                   "(the .ts load peaks ~9.5 GB host RSS)")
            if hard:
                self.warn(msg)
            else:
                reasons.append(msg)
        if buckets:
            bad = [b for b in buckets if not any(b % eb == 0 for eb in engine_batches)]
            if bad:
                msg = (f"HLS_SCHEDULER_FIXED_BATCH_SIZES={','.join(map(str, buckets))} has buckets {bad} that are not "
                       f"multiples of the TRT UNet engine batch {engine_batches}")
                if hard:
                    self.error(msg)
                    return None
                reasons.append(msg)
            max_up = _to_int(self.view.get("HLS_SCHEDULER_MAX_BATCH"))
            if max_up and max_up > max(buckets):
                msg = (f"HLS_SCHEDULER_MAX_BATCH={max_up} > largest bucket {max(buckets)}: batches above the largest "
                       "bucket are not padded, which the static TRT UNet cannot serve")
                if hard:
                    self.error(msg)
                    return None
                reasons.append(msg)
        if reasons and not hard:
            return self._unet_eager("auto: " + "; ".join(reasons))
        for r in reasons:
            self.warn(r + " (MUSETALK_UNET_MODE=trt keeps TRT)")
        self.emit("MUSETALK_UNET_BACKEND", "trt", why)
        self.emit("MUSETALK_TRT_UNET_ENABLED", "1", why)
        if not paths_up:
            engine_path = str(Path(engine_path_of(entry)).absolute())
            meta = Path(engine_path).with_name("unet_trt_meta.json")
            if not meta.exists():
                self.warn(f"{meta} missing next to the engine; trt_runtime reads it for batch_range/dtype")
            self.emit("MUSETALK_TRT_UNET_PATHS", f"{TS_ENGINE_BATCH}:{engine_path}",
                      f"validated {entry.get('match')} engine {entry.get('key')}")
        else:
            self.emit("MUSETALK_TRT_UNET_PATHS", paths_up, "caller-supplied engine paths")
        self.unet.update({"backend": "trt", "engine": engine_desc, "reason": why})
        self.unet_entry = entry if not paths_up else None
        self.expect["unet"] = "trt"
        return None

    def _stagewise_batch_request(self):
        raw, src = self.view.upper("MUSETALK_UNET_STAGEWISE_BATCH")
        if raw is None:
            item = self._pending("MUSETALK_UNET_STAGEWISE_BATCH")
            raw = item["requested"] if item else None
            src = f"recipe:{self.recipe}" if item else None
        return (_to_int(raw), src) if raw is not None else (None, None)

    def _unet_stagewise(self, explicit, source, item=None):
        """Stagewise FP16 UNet. explicit=True: caller chose it (respected, warn/err on gaps);
        explicit=False: fast300 lever (enabled only if a validated engine exists), or a lever of a
        group that requires a pinned bundle (the bundle's engine set, see _unet_stagewise_bundle)."""
        bundle = None if explicit else self._group_bundle(item)
        if bundle is not None:
            return self._unet_stagewise_bundle(bundle, source, item)
        gpu = self.facts.get("gpu")
        key = self.store.engine_key("unet_stagewise", self.facts)
        self.engines["unet_stagewise"] = {"key": key, "store": str(self.store.store_root("unet_stagewise")),
                                          "legacy_root": str(self.repo_root / LEGACY_STAGEWISE_ROOT),
                                          "backend": self.store.backend}
        want_batch, batch_src = self._stagewise_batch_request()
        batch_item = self._pending("MUSETALK_UNET_STAGEWISE_BATCH")
        # Raw TensorRT plans: exact key only (no same-cc reuse across GPU models).
        entries = self.store.entries("unet_stagewise", self.facts) if gpu else []
        self.engines["unet_stagewise"]["candidates"] = [compact_entry(e) for e in entries]
        usable = [e for e in entries if e.get("usable")]
        cache_up, cache_src = self.view.upper("MUSETALK_UNET_STAGEWISE_CACHE_DIR")
        if cache_up:
            cache_root = Path(cache_up) if Path(cache_up).is_absolute() else self.repo_root / cache_up
            usable = [e for e in usable if e.get("root") and
                      os.path.realpath(str(e["root"])) == os.path.realpath(str(cache_root))]
        if want_batch:
            chosen_pool = [e for e in usable if _to_int(e.get("batch")) == want_batch]
        else:
            buckets = self._buckets_hint()
            chosen_pool = [e for e in usable if buckets and _to_int(e.get("batch")) == max(buckets)] or usable
        chosen_pool.sort(key=lambda e: (0 if e.get("match") == "exact" else 1, -(_to_int(e.get("batch")) or 0)))
        entry = chosen_pool[0] if chosen_pool else None
        adopt_hint = ""
        if entry is None and gpu is not None:
            legacy = self._matching_legacy_stagewise(want_batch)
            self.engines["unet_stagewise"]["legacy_candidates"] = legacy
            if legacy:
                adopt_hint = (f"; legacy set(s) {[c['path'] for c in legacy]} match this GPU: adopt + validate with "
                              "scripts/unet_engine_store.py adopt --kind unet_stagewise")
        min_vram = self.float_knob("MUSETALK_TRT_UNET_MIN_VRAM_GB", 8.0)
        problems = []
        if gpu is None:
            problems.append("no GPU visible")
        if entry is None:
            problems.append(f"no validated unet_stagewise engine for key {key}"
                            + (f" at batch {want_batch}" if want_batch else "")
                            + (f" under {cache_up} ({cache_src})" if cache_up else "")
                            + (adopt_hint or " (build: scripts/unet_engine_store.py build --kind unet_stagewise "
                                             "--batch N)"))
        if gpu is not None and not self._vram_ok(min_vram):
            problems.append(f"VRAM {self._vram_gb() or 0:.1f} GB < {min_vram:g} GB")
        if entry is not None and entry.get("match") == "same_cc":
            self.warn(f"stagewise engine {entry.get('dir')} comes from another GPU model with the same sm")
        if not explicit:
            if problems:
                self._drop(item, "; ".join(problems))
                if batch_item:
                    self._drop(batch_item, "stagewise UNet lever dropped")
                return False
            self._enable(item, f"validated stagewise engine {entry.get('key')} bs{entry.get('batch')} ({entry.get('match')})")
            if batch_item:
                if want_batch == _to_int(entry.get("batch")):
                    self._enable(batch_item, "matches the validated engine batch")
                else:
                    self._drop(batch_item, f"engine batch is {entry.get('batch')}")
            self.emit("MUSETALK_UNET_BACKEND", "trt_stagewise", item["reason"])
        else:
            if problems:
                batch = want_batch or 16
                probe_root = Path(cache_up) if cache_up else self.repo_root / LEGACY_STAGEWISE_ROOT
                if not probe_root.is_absolute():
                    probe_root = self.repo_root / probe_root
                manifest = probe_root / f"bs{batch}" / "manifest.json"
                if manifest.exists():
                    self.warn(f"{source} selects trt_stagewise; {'; '.join(problems)}. {manifest} exists but has "
                              "no validation record for this key (server will load it unvalidated)")
                else:
                    self.error(f"{source} selects MUSETALK_UNET_BACKEND=trt_stagewise but {'; '.join(problems)} and "
                               f"{manifest} does not exist (the server would fail: MUSETALK_TRT_FALLBACK=0)")
            self.emit("MUSETALK_UNET_BACKEND", "trt_stagewise", f"{source} choice respected")
        self.emit("MUSETALK_TRT_UNET_ENABLED", "0", "stagewise UNet does not load the .ts engine")
        engine_batch = want_batch
        if entry is not None:
            engine_batch = _to_int(entry.get("batch"))
            env = self.store.entry_env(entry)
            root = env.get("MUSETALK_UNET_STAGEWISE_CACHE_DIR") or str(Path(entry["dir"]).parent)
            self.emit("MUSETALK_UNET_STAGEWISE_CACHE_DIR", str(Path(root).absolute()),
                      f"validated engine set {entry.get('key')} bs{engine_batch}")
            self.emit("MUSETALK_UNET_STAGEWISE_BATCH", str(engine_batch), "engine batch of the validated set")
        elif engine_batch:
            self.emit("MUSETALK_UNET_STAGEWISE_BATCH", str(engine_batch), f"{batch_src} request")
        self.unet.update({"backend": "trt_stagewise", "engine_batch": engine_batch or 16,
                          "engine": ({"dir": entry.get("dir"), "key": entry.get("key"), "match": entry.get("match"),
                                      "layout": entry.get("layout")} if entry else None),
                          "reason": source})
        self.expect["unet"] = "trt_stagewise"
        self._drop_ts_only_levers("UNet runs the stagewise engines")
        return True

    def _unet_stagewise_bundle(self, bundle, source, item):
        """Stagewise UNet from a pinned bundle (bundle:<name> held): its engine set is not an engine-store
        entry (srccache INT8 blocks), so the resolver points the server at the bundle directory."""
        name = bundle.get("name")
        spec = (bundle.get("engines") or {}).get("unet_stagewise") or {}
        batch = _to_int(spec.get("batch"))
        cache_rel = str(spec.get("cache_dir") or "")
        root = self.repo_root / cache_rel
        key = self.store.engine_key("unet_stagewise", self.facts)
        self.engines["unet_stagewise"] = {"key": key, "bundle": name, "dir": str(root), "batch": batch}
        batch_item = self._pending("MUSETALK_UNET_STAGEWISE_BATCH")
        want_batch, _ = self._stagewise_batch_request()
        cache_up, cache_src = self.view.upper("MUSETALK_UNET_STAGEWISE_CACHE_DIR")
        problems = []
        if not cache_rel or not batch:
            problems.append(f"bundle {name} names no unet_stagewise cache_dir/batch")
        elif not (root / f"bs{batch}" / "manifest.json").is_file():
            problems.append(f"{root}/bs{batch}/manifest.json missing")
        if want_batch and batch and want_batch != batch:
            problems.append(f"batch {want_batch} requested but the bundle engines are bs{batch}")
        if cache_up:
            up_root = Path(cache_up) if Path(cache_up).is_absolute() else self.repo_root / cache_up
            if os.path.realpath(str(up_root)) != os.path.realpath(str(root)):
                problems.append(f"{cache_src} pins MUSETALK_UNET_STAGEWISE_CACHE_DIR={cache_up}, not the bundle set")
        if problems:
            self._drop(item, "; ".join(problems))
            if batch_item:
                self._drop(batch_item, "stagewise UNet lever dropped")
            return False
        reason = f"pinned bundle {name} ({str(bundle.get('sha256'))[:12]}) restored for {key}"
        self._enable(item, reason)
        if batch_item:
            self._enable(batch_item, "matches the bundle engine batch")
        self.emit("MUSETALK_UNET_BACKEND", "trt_stagewise", reason)
        self.emit("MUSETALK_TRT_UNET_ENABLED", "0", "stagewise UNet does not load the .ts engine")
        self.emit("MUSETALK_UNET_STAGEWISE_CACHE_DIR", str(root), reason)
        self.emit("MUSETALK_UNET_STAGEWISE_BATCH", str(batch), "engine batch of the bundle set")
        self.unet.update({"backend": "trt_stagewise", "engine_batch": batch,
                          "engine": {"dir": str(root / f"bs{batch}"), "key": key, "match": f"bundle:{name}",
                                     "layout": "bundle"},
                          "reason": source})
        self.expect["unet"] = "trt_stagewise"
        self._drop_ts_only_levers("UNet runs the stagewise engines")
        return True

    def _matching_legacy_stagewise(self, want_batch):
        """Legacy build_unet_stagewise.py sets (outside the store) whose manifest matches this host."""
        out = []
        mismatch = getattr(self.store.module, "host_mismatch", None)
        for cand in self.store.legacy("unet_stagewise"):
            if cand.get("error") or not cand.get("complete") or not cand.get("schema_ok", True):
                continue
            if want_batch and _to_int(cand.get("batch")) != want_batch:
                continue
            problems = mismatch(self.facts, cand.get("gpu_name"), cand.get("compute_capability"),
                                cand.get("tensorrt_version")) if callable(mismatch) else ["host_mismatch unavailable"]
            if not problems:
                out.append({"path": cand.get("path"), "batch": cand.get("batch")})
        return out

    # -- buckets ----------------------------------------------------------------------------------
    def _resolve_buckets(self):
        backend = self.unet.get("backend")
        default = str(self.unet.get("engine_batch") or 16) if backend == "trt_stagewise" else str(TS_ENGINE_BATCH)
        raw = self.emit("HLS_SCHEDULER_FIXED_BATCH_SIZES", default,
                        "engine batch" if backend in ("trt", "trt_stagewise") else "matches warmups (no lazy compile)")
        try:
            buckets = parse_buckets(raw)
        except ValueError as exc:
            self.error(f"HLS_SCHEDULER_FIXED_BATCH_SIZES={raw!r} is invalid: {exc}")
            return
        csv_buckets = ",".join(str(b) for b in buckets)
        taesd = self.emit("MUSETALK_TAESD_WARMUP_BATCHES", csv_buckets, "= buckets (compiled TAESD is dynamic=False)")
        stage = self.emit("MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES", csv_buckets,
                          "= buckets (api_server snaps WebRTC batch sizes to this list)")
        for name, value in (("MUSETALK_TAESD_WARMUP_BATCHES", taesd), ("MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES", stage)):
            try:
                if parse_buckets(value) != buckets:
                    self.warn(f"{name}={value} differs from HLS_SCHEDULER_FIXED_BATCH_SIZES={csv_buckets}: "
                              "unwarmed buckets compile lazily on the scheduler thread / WebRTC sizes snap wrongly")
            except ValueError:
                self.error(f"{name}={value!r} is not a batch list")
        max_batch = self.emit("HLS_SCHEDULER_MAX_BATCH", str(max(buckets)), "= max(buckets)")
        max_n = _to_int(max_batch)
        if max_n is None or max_n <= 0:
            self.error(f"HLS_SCHEDULER_MAX_BATCH={max_batch!r} is invalid")
            max_n = max(buckets)
        elif max_n > max(buckets) and backend != "trt":
            self.warn(f"HLS_SCHEDULER_MAX_BATCH={max_n} > largest bucket {max(buckets)}: unpadded batch shapes")
        self.emit("HLS_SCHEDULER_STARTUP_SLICE_SIZE", str(min(8, max_n)), "= min(8, max batch)")
        if backend == "trt_stagewise":
            eb = self.unet.get("engine_batch") or 16
            waste = [b for b in buckets if b % eb != 0]
            if waste:
                self.warn(f"stagewise engine batch {eb} vs buckets {csv_buckets}: buckets {waste} are padded/split "
                          "onto the engine batch (wasted rows)")
        self.buckets = buckets

    # -- VAE levers (fast300) ---------------------------------------------------------------------
    def _resolve_vae_levers(self):
        taesd_up, src = self.view.upper("MUSETALK_TAESD_BACKEND")
        item = self._pending("MUSETALK_TAESD_BACKEND")
        trt_items = [self._pending(n) for n in LEVERS if n.startswith("MUSETALK_TAESD_TRT_")]
        trt_items = [i for i in trt_items if i]
        vae = (self.view.get("MUSETALK_VAE_BACKEND") or "").strip().lower()
        if vae not in ("taesd", "tiny", "tiny_vae"):
            self.expect["vae"] = {"": "pytorch", "trt_stagewise": "trt_stagewise",
                                  "tensorrt_stagewise": "trt_stagewise"}.get(vae, "any")
            for i in ([item] if item else []) + trt_items:
                self._drop(i, f"MUSETALK_VAE_BACKEND={vae!r} is not TAESD")
            return
        if taesd_up is not None:
            kind = taesd_up.strip().lower()
            if item:
                self._drop(item, f"{src} sets MUSETALK_TAESD_BACKEND={taesd_up!r}")
            if kind in ("trt", "tensorrt"):
                self.expect["vae"] = "taesd_trt"
                entry = self._taesd_trt_entry()
                if entry is None:
                    self.warn(f"{src} selects MUSETALK_TAESD_BACKEND=trt without a validated taesd_trt engine for "
                              "this key: the server builds/loads it unvalidated or falls back to compiled TAESD "
                              "(verify-log will flag the fallback)")
                elif quality_verdict(entry) != "PASS":
                    self.warn(f"{src} selects MUSETALK_TAESD_BACKEND=trt; the engine's G-TAESD quality gate is "
                              f"{quality_verdict(entry) or 'not recorded'} (not PASS)")
                self._emit_taesd_trt(entry, f"{src} choice respected", trt_items)
            else:
                self.expect["vae"] = "taesd_compiled"
                for i in trt_items:
                    self._drop(i, "TAESD TRT not selected")
            return
        if item is None:
            self.expect["vae"] = "taesd_compiled"
            for i in trt_items:
                self._drop(i, "MUSETALK_TAESD_BACKEND=trt is not enabled")
            return
        if item["requested"].strip().lower() not in ("trt", "tensorrt"):
            self._enable(item, "compiled TAESD (fast default)")
            self.emitted["MUSETALK_TAESD_BACKEND"] = item["requested"]
            self.decisions.append({"knob": "MUSETALK_TAESD_BACKEND", "value": item["requested"],
                                   "source": f"recipe:{self.recipe}", "reason": item["reason"]})
            self.expect["vae"] = "taesd_compiled"
            for i in trt_items:
                self._drop(i, "TAESD TRT not selected")
            return
        bundle = self._group_bundle(item)
        if bundle is not None:
            return self._taesd_trt_bundle(bundle, item, trt_items)
        entry = self._taesd_trt_entry()
        verdict = quality_verdict(entry)
        if self.facts.get("gpu") is None or entry is None or verdict != "PASS":
            reason = "no GPU visible" if self.facts.get("gpu") is None else (
                f"no validated taesd_trt engine for key {self.engines['taesd_trt'].get('key')} batch "
                f"{self._taesd_trt_batch() or 8} (build: scripts/unet_engine_store.py build --kind taesd_trt; gate: "
                "G-TAESD)" if entry is None else
                f"taesd_trt engine {entry.get('dir')} is store-usable but its G-TAESD quality gate is "
                f"{verdict or 'not recorded'} (needs PASS)")
            self._drop(item, reason)
            for i in trt_items:
                self._drop(i, "TAESD TRT lever dropped")
            self.expect["vae"] = "taesd_compiled"
            return
        self._enable(item, f"validated taesd_trt engine {entry.get('key')} bs{entry.get('batch')}")
        self.expect["vae"] = "taesd_trt"
        self.emitted["MUSETALK_TAESD_BACKEND"] = "trt"
        self.decisions.append({"knob": "MUSETALK_TAESD_BACKEND", "value": "trt", "source": f"recipe:{self.recipe}",
                               "reason": item["reason"]})
        self._emit_taesd_trt(entry, item["reason"], trt_items)

    def _taesd_trt_bundle(self, bundle, item, trt_items):
        """TAESD TRT from a pinned bundle (bundle:<name> held): the bundle's plans in their own directory.
        The engine store's G-TAESD gate does not apply; the bundle was accepted as a whole (see its recipe)."""
        name = bundle.get("name")
        spec = (bundle.get("engines") or {}).get("taesd_trt") or {}
        engine_dir = self.repo_root / str(spec.get("dir") or "")
        runtime_key = str(spec.get("key") or "")
        batch = _to_int(spec.get("batch"))
        self.engines["taesd_trt"] = {"key": self.store.engine_key("taesd_trt", self.facts), "bundle": name,
                                     "dir": str(engine_dir), "runtime_key": runtime_key, "batch": batch}
        problems = []
        if not runtime_key or not batch or not spec.get("dir"):
            problems.append(f"bundle {name} names no taesd_trt dir/key/batch")
        elif not (engine_dir / f"taesd_trt_{runtime_key}.json").is_file():
            problems.append(f"{engine_dir}/taesd_trt_{runtime_key}.json missing")
        batch_up = self._taesd_trt_batch()
        if batch_up and batch and batch_up != batch:
            problems.append(f"MUSETALK_TAESD_TRT_BATCH={batch_up} but the bundle engine is bs{batch}")
        dir_up, dir_src = self.view.upper("MUSETALK_TAESD_TRT_DIR")
        if dir_up:
            up_dir = Path(dir_up) if Path(dir_up).is_absolute() else self.repo_root / dir_up
            if os.path.realpath(str(up_dir)) != os.path.realpath(str(engine_dir)):
                problems.append(f"{dir_src} pins MUSETALK_TAESD_TRT_DIR={dir_up}, not the bundle engine")
        if problems:
            self._drop(item, "; ".join(problems))
            for i in trt_items:
                self._drop(i, "TAESD TRT lever dropped")
            self.expect["vae"] = "taesd_compiled"
            return
        reason = f"pinned bundle {name} TAESD TRT engine {runtime_key} bs{batch}"
        self._enable(item, reason)
        self.expect["vae"] = "taesd_trt"
        self.emitted["MUSETALK_TAESD_BACKEND"] = "trt"
        self.decisions.append({"knob": "MUSETALK_TAESD_BACKEND", "value": "trt", "source": f"recipe:{self.recipe}",
                               "reason": reason})
        for i in trt_items:
            name_i = i["name"]
            if name_i == "MUSETALK_TAESD_TRT_DIR":
                self._drop(i, "the resolver points MUSETALK_TAESD_TRT_DIR at the bundle engine")
            elif name_i == "MUSETALK_TAESD_TRT_BUILD" and parse_bool(i["requested"]) is not False:
                self._drop(i, "the bundle engine is pre-built; serve-time builds are disabled (use 0)")
            elif name_i == "MUSETALK_TAESD_TRT_BATCH" and _to_int(i["requested"]) != batch:
                self._drop(i, f"the bundle engine is bs{batch} (part of its runtime key)")
            else:
                self._enable(i, "TAESD TRT enabled")
                self.emitted[name_i] = i["requested"]
                self.decisions.append({"knob": name_i, "value": i["requested"], "source": f"recipe:{self.recipe}",
                                       "reason": i["reason"]})
        self.emit("MUSETALK_TAESD_TRT_DIR", str(engine_dir), reason)
        self.emit("MUSETALK_TAESD_TRT_BATCH", str(batch), reason)
        if "MUSETALK_TAESD_TRT_BUILD" not in self.emitted:
            self.emit("MUSETALK_TAESD_TRT_BUILD", "0", "bundle engine pre-built; never build inside the server")

    def _taesd_trt_batch(self):
        """Explicitly requested TAESD TRT engine batch (caller/overrides or recipe line), else None."""
        raw = self.view.upper("MUSETALK_TAESD_TRT_BATCH")[0]
        if raw is None:
            pending = self._pending("MUSETALK_TAESD_TRT_BATCH")
            raw = pending["requested"] if pending else None
        return _to_int(raw) if raw not in (None, "") else None

    def _taesd_trt_entry(self):
        batch = self._taesd_trt_batch()
        key = self.store.engine_key("taesd_trt", self.facts)
        # Raw TensorRT plans whose runtime key embeds the GPU name: exact key only.
        entry, entries = (self.store.find("taesd_trt", self.facts, batch=batch, prefer_batch=8)
                          if self.facts.get("gpu") else (None, []))
        self.engines["taesd_trt"] = {"key": key, "batch": batch or 8, "store": str(self.store.store_root("taesd_trt")),
                                     "backend": self.store.backend, "candidates": [compact_entry(e) for e in entries]}
        if entry is None:
            self.engines["taesd_trt"]["legacy_candidates"] = [c.get("path") for c in self.store.legacy("taesd_trt")]
        return entry

    def _emit_taesd_trt(self, entry, reason, trt_items):
        env = self.store.entry_env(entry) if entry is not None else {}
        for i in trt_items:
            name = i["name"]
            if name == "MUSETALK_TAESD_TRT_DIR":
                self._drop(i, "the resolver points MUSETALK_TAESD_TRT_DIR at the validated engine")
            elif name == "MUSETALK_TAESD_TRT_BUILD" and parse_bool(i["requested"]) is not False:
                self._drop(i, "serve-time builds are disabled; the engine store builds at install (use 0)")
            elif name in env and str(env[name]) != str(i["requested"]).strip():
                self._drop(i, f"the validated engine was built with {name}={env[name]} (part of its runtime key)")
            else:
                self._enable(i, "TAESD TRT enabled")
                self.emitted[name] = i["requested"]
                self.decisions.append({"knob": name, "value": i["requested"], "source": f"recipe:{self.recipe}",
                                       "reason": i["reason"]})
        for name, value in env.items():
            effective = self.emit(name, value, f"validated taesd_trt engine {entry.get('key')} ({reason})")
            if name in ("MUSETALK_TAESD_TRT_BATCH", "MUSETALK_TAESD_TRT_OPT_LEVEL", "MUSETALK_TAESD_TRT_STRONGLY_TYPED") \
                    and str(effective) != str(value):
                self.warn(f"{name}={effective} differs from the validated engine ({value}): the runtime key will not "
                          "match and the server falls back to compiled TAESD")
        if entry is not None and "MUSETALK_TAESD_TRT_BUILD" not in env:
            self.emit("MUSETALK_TAESD_TRT_BUILD", "0", "engine pre-built and validated; never build inside the server")

    # -- CPU / RAM ---------------------------------------------------------------------------------
    def _cpu_ram(self):
        cpu = self.facts.get("cpu") or {}
        eff = int(cpu.get("effective") or 1)
        workers = 8 if eff >= 12 else max(2, eff // 2)
        why = f"effective CPUs {eff} (nproc {cpu.get('nproc')}, affinity {cpu.get('affinity')}, " \
              f"cgroup quota {cpu.get('cgroup_quota_cpus')})"
        for knob in ("HLS_PREP_WORKERS", "HLS_COMPOSE_WORKERS", "HLS_ENCODE_WORKERS"):
            self.emit(knob, str(workers), why)
        self.emit("MUSETALK_AVATAR_LOAD_WORKERS", str(min(8, eff)), "min(8, effective CPUs)")
        self.emit("HLS_MAX_PENDING_JOBS", "24", "admission cap above the 15-20 stream target")
        self.emit("MUSETALK_WHISPER_SEGMENT_BATCH_SIZE", "4", "old launcher value")
        ram = self.facts.get("ram") or {}
        total = int(ram.get("effective_total_mb") or 0)
        cache = max(2048, min(16384, int(0.2 * total)))
        self.emit("AVATAR_CACHE_MAX_MEMORY_MB", str(cache), f"clamp(0.2 x host RAM {total} MB, 2048, 16384)")
        self.emit("AVATAR_CACHE_TTL_SECONDS", "3600", "old launcher value")
        gpu = self.facts.get("gpu") or {}
        if gpu.get("memory_total_mib"):
            self.emit("GPU_TOTAL_MEMORY_GB", f"{gpu['memory_total_mib'] / 1024:.1f}",
                      "nvidia-smi VRAM (avoids the silent 24 GB fallback)")

    # -- blend / WebRTC / HLS -------------------------------------------------------------------------
    def _blend_webrtc(self):
        why = "old launcher value"
        for knob, value in (("MUSETALK_BLEND_FIXED_POINT", "1"), ("MUSETALK_BLEND_SHRINK_MASK_BBOX", "1"),
                            ("WEBRTC_BATCH_FRAME_CALLBACK", "1"), ("WEBRTC_SYNC_MODE", "strict_fifo"),
                            ("WEBRTC_AUDIO_SYNC_STRATEGY", "timestamp_locked"),
                            ("WEBRTC_VIDEO_PREBUFFER_SECONDS", "2.0"), ("WEBRTC_ADAPTIVE_FPS", "0"),
                            ("WEBRTC_TRIM_EDGE_SILENCE", "0"), ("WEBRTC_POSE_CROSSFADE_FRAMES", "2"),
                            ("WEBRTC_POSE_FORCED_CROSSFADE_FRAMES", "4"),
                            ("WEBRTC_POSE_MAX_SEMANTIC_DRIFT_SECONDS", "0.75"),
                            ("HLS_CHUNK_VIDEO_ENCODER", "libx264"), ("HLS_CHUNK_ENCODER_PRESET", "ultrafast"),
                            ("HLS_CHUNK_ENCODER_CRF", "28"), ("HLS_CHUNK_PREPARE_AUDIO_SIDECAR", "1"),
                            ("PROFILE", "throughput_record"), ("PYTHONFAULTHANDLER", "1"),
                            ("PYTHONUNBUFFERED", "1")):
            self.emit(knob, value, why)

    # -- encoders --------------------------------------------------------------------------------
    def native_vp8_static_check(self):
        """(ok, problems) without importing aiortc: platform, venv pins, installed files."""
        problems = []
        venv = self.facts.get("venv") or {}
        if self.facts.get("machine") != "x86_64":
            problems.append(f"machine {self.facts.get('machine')} (needs x86_64)")
        pyv = venv.get("python_version") or ""
        if not pyv.startswith("3.10"):
            problems.append(f"venv python {pyv or 'unknown'} (needs CPython 3.10)")
        for pkg, want in (("aiortc", "1.14.0"), ("av", "16.1.0"), ("cffi", "2.1.1")):
            if venv.get(pkg) != want:
                problems.append(f"{pkg} {venv.get(pkg)} (needs {want})")
        raw_dir = self.view.get("WEBRTC_NATIVE_VP8_DIR") or str(self.repo_root / ".runtime" / "native_vp8")
        directory = Path(raw_dir)
        manifest = _read_json(self.repo_root / "scripts" / "native_vp8_manifest.json") or {}
        if not (directory / "installation.json").is_file():
            problems.append(f"{directory}/installation.json missing (run scripts/install_native_vp8.py)")
        else:
            for item in (manifest.get("extracted") or []) + (manifest.get("bundled_notices") or []):
                path = directory / str(item.get("path", ""))
                if not path.is_file() or (item.get("bytes") and path.stat().st_size != item["bytes"]):
                    problems.append(f"native VP8 file missing or wrong size: {path}")
                    break
        return not problems, problems

    def native_vp8_keeps_h264(self):
        """(ok, why). Conservative: needs an explicit H.264-fallback marker in webrtc_native_vp8.py."""
        path = self.repo_root / "scripts" / "webrtc_native_vp8.py"
        text = _read_text(path, "")
        marker = re.search(r"^NATIVE_VP8_H264_FALLBACK\s*=\s*True\b", text, re.M)
        reject = None
        for lineno, line in enumerate(text.splitlines(), 1):
            if "H264-only offers are unsupported" in line:
                reject = lineno
                break
        if marker and reject is None:
            return True, "webrtc_native_vp8.py declares NATIVE_VP8_H264_FALLBACK = True"
        where = f"scripts/webrtc_native_vp8.py:{reject}" if reject else "scripts/webrtc_native_vp8.py"
        return False, (f"native VP8 rejects H264-only offers ({where} validate_native_offer -> HTTP 400 in "
                       "api_server.py webrtc offer handler); H.264-only clients would get no video. Not enabled "
                       "until the module declares NATIVE_VP8_H264_FALLBACK = True")

    def _encoders(self):
        self.emit("WEBRTC_NATIVE_VP8_DIR", str(self.repo_root / ".runtime" / "native_vp8"),
                  "installer provisions native VP8 here")
        item = self._pending("WEBRTC_VP8_ENCODER")
        vp8_up, src = self.view.upper("WEBRTC_VP8_ENCODER")
        static_ok, problems = self.native_vp8_static_check()
        h264_ok, h264_why = self.native_vp8_keeps_h264()
        self.native_vp8 = {"static_ok": static_ok, "problems": problems, "h264_path_ok": h264_ok,
                           "h264_reason": h264_why}
        if item and vp8_up is None:
            want = item["requested"].strip().lower()
            if want == "native":
                if not static_ok:
                    self._drop(item, "native VP8 preflight prerequisites fail: " + "; ".join(problems))
                elif not h264_ok:
                    self._drop(item, h264_why)
                else:
                    ok, why = self.native_vp8_preflight()
                    if ok:
                        self._enable(item, "native VP8 preflight passed; H.264 clients keep an H.264 path")
                    else:
                        self._drop(item, why)
            else:
                self._enable(item, "pyav VP8")
        vp8 = self.emit("WEBRTC_VP8_ENCODER", "pyav",
                        "default (native rejects H264-only offers)" if self.recipe == "fast" else
                        "pyav unless the fast300 native lever passes its gates")
        if vp8.strip().lower() == "native":
            if not static_ok:
                self.warn("WEBRTC_VP8_ENCODER=native but the native VP8 prerequisites fail: " + "; ".join(problems)
                          + " (the launcher preflight will refuse unless MUSETALK_VP8_FALLBACK=1)")
            if not h264_ok:
                self.warn("WEBRTC_VP8_ENCODER=native: H264-only WebRTC offers get HTTP 400")
        else:
            thr = self._pending("WEBRTC_NATIVE_VP8_THREADS")
            if thr:
                self._drop(thr, "native VP8 not active")
        h264 = self.view.get("WEBRTC_H264_IMPL")
        item = self._pending("WEBRTC_H264_IMPL")
        if h264 is None and item:
            if not (self.repo_root / "scripts" / "webrtc_h264_override.py").exists():
                self._drop(item, "scripts/webrtc_h264_override.py missing")
            else:
                note = ("NVENC: process-wide cap WEBRTC_NVENC_MAX_SESSIONS (GeForce driver limit ~12), x264tuned "
                        "beyond it or if NVENC cannot open" if item["requested"].lower() == "nvenc" else
                        "H.264 encoder override")
                self._enable(item, note)
                self.emitted["WEBRTC_H264_IMPL"] = item["requested"]
                self.decisions.append({"knob": "WEBRTC_H264_IMPL", "value": item["requested"],
                                       "source": f"recipe:{self.recipe}", "reason": note})

    # -- chin tools ---------------------------------------------------------------------------------
    def _chin_tools(self):
        workspace = Path(self.view.get("WORKSPACE") or "/workspace")
        venv_raw = self.view.get("MUSETALK_CHIN_TOOLS_VENV") or str(workspace / ".venvs" / "musetalk_chin_tools")
        venv = Path(venv_raw)
        py = venv / "bin" / "python"
        up, src = self.view.upper("MUSETALK_CHIN_TRACKER_PYTHON")
        self.chin = {"venv": str(venv), "python": str(py), "present": False}
        if up is not None:
            if "SoulX-FlashHead" in up:
                self.warn(f"MUSETALK_CHIN_TRACKER_PYTHON ({src}) borrows another project's venv ({up}); install the "
                          "dedicated one: scripts/install_musetalk.sh --with-chin-tools")
            if not Path(up).exists():
                self.warn(f"MUSETALK_CHIN_TRACKER_PYTHON={up} ({src}) does not exist")
            self.emit("MUSETALK_CHIN_TRACKER_PYTHON", str(py), "dedicated chin-tools venv")
            return
        sps = site_packages_dirs(venv) if venv.is_dir() else []
        has_mp = any(installed_packages(sp).get("mediapipe") for sp in sps)
        if py.exists() and has_mp:
            self.chin["present"] = True
            self.emit("MUSETALK_CHIN_TRACKER_PYTHON", str(py), "dedicated chin-tools venv (mediapipe present)")
        else:
            self.notes.append(f"chin-tools venv not installed at {venv} (optional: scripts/install_musetalk.sh "
                              "--with-chin-tools); MUSETALK_CHIN_TRACKER_PYTHON not set")

    # -- validate caller/overrides levers ---------------------------------------------------------------
    def _validate_upper_levers(self):
        backend = self.unet.get("backend")
        for name, spec in LEVERS.items():
            value, src = self.view.upper(name)
            if value is None:
                continue
            if name == "MUSETALK_UNET_BACKEND":
                continue  # handled in _resolve_unet
            bad = validate_lever_value(name, value)
            if not bad:
                continue
            fatal = spec["raises"]
            if name == "MUSETALK_TRT_UNET_CUDAGRAPHS" and backend != "trt":
                fatal = False
            if name == "WEBRTC_NATIVE_VP8_THREADS" and (self.view.get("WEBRTC_VP8_ENCODER") or "").lower() != "native":
                fatal = False
            (self.error if fatal else self.warn)(f"{bad} ({src})" + (" - the server raises on it" if fatal else ""))
        free, src = self.view.upper("MUSETALK_FREE_EAGER_UNET")
        item = self._pending("MUSETALK_FREE_EAGER_UNET")
        capture = parse_bool(self.view.get("MUSETALK_UNET_CALIBRATION_CAPTURE")) is True
        if item and free is None:
            if backend not in ("trt", "trt_stagewise"):
                self._drop(item, f"needs a TensorRT UNet backend (UNet: {backend})")
            elif capture:
                self._drop(item, "MUSETALK_UNET_CALIBRATION_CAPTURE=1 needs the eager UNet")
        elif parse_bool(free) is True and capture:
            self.warn("MUSETALK_FREE_EAGER_UNET=1 with MUSETALK_UNET_CALIBRATION_CAPTURE=1 breaks capture")
        cg = self._pending("MUSETALK_TRT_UNET_CUDAGRAPHS")
        if cg and backend != "trt":
            self._drop(cg, f"needs the torch_tensorrt .ts UNet (UNet: {backend})")
        for name in [n for n in LEVERS if n.startswith("MUSETALK_UNET_STAGEWISE_")]:
            pend = self._pending(name)
            if pend and backend != "trt_stagewise":
                self._drop(pend, "stagewise UNet not enabled")

    # -- levers report -----------------------------------------------------------------------------
    def _levers_report(self):
        """Every known lever: effective value + source layer. Pass-through levers are never emitted by
        the resolver itself; they show here so the operator sees what the server will run."""
        registries = load_flag_registries(self.repo_root)
        names = list(LEVERS) + [n for n in sorted(registries) if n not in LEVERS]
        names += [i["name"] for i in self.recipe_levers if i["name"] not in names]
        report = []
        for name in names:
            reg = registries.get(name) or {}
            spec = LEVERS.get(name) or {
                "category": "engine" if str(reg.get("registry", "")).endswith("vae_fast_decoder.py") else "serving",
                "default": "", "note": ""}
            effective, src = self.view.lookup(name)
            if name in self.emitted:
                effective = self.emitted[name]
                src = src or "resolver"
            default = reg.get("default") if reg.get("default") is not None else spec.get("default", "")
            report.append({"name": name, "category": spec["category"],
                           "effective": effective if effective is not None else default,
                           "source": src or "code default (unset)", "code_default": default,
                           "implemented": self.scanner.reads(name), "registry": reg.get("registry"),
                           "note": spec.get("note") or None})
        self.levers = report

    # -- estimate --------------------------------------------------------------------------------------
    def _estimate(self):
        selftest, why = self._selftest()
        if not selftest:
            self.estimate = {"available": False, "line": "estimate: n/a (no matching .runtime/gpu_selftest.json"
                                                        + (f": {why}" if why else "") + ")"}
            return
        taesd = selftest.get("taesd") or {}
        unet = selftest.get("unet_eager") or {}
        compiled = self.emitted.get("MUSETALK_TAESD_COMPILE") == "1" and taesd.get("compile_ok") is not False
        taesd_ms = _to_float(taesd.get("ms_bs8") if compiled else taesd.get("eager_ms_bs8"))
        taesd_note = "compiled" if compiled else "eager"
        if self.expect.get("vae") == "taesd_trt":
            trt = selftest.get("taesd_trt") or {}
            trt_ms = _to_float(trt.get("ms_bs8")) if trt.get("ok") else None  # same decode() call as taesd.ms_bs8
            if trt_ms:
                taesd_ms, taesd_note = trt_ms, "TAESD TRT, measured"
            else:
                taesd_note += " (TAESD TRT not timed by the self-test: upper bound)"
        eager_ms = _to_float(unet.get("ms_bs8")) if unet.get("ok") else None
        backend = self.unet.get("backend")
        unet_ms, unet_note = None, None
        if eager_ms:
            if backend == "eager":
                unet_ms, unet_note = eager_ms, "eager, measured"
            elif backend == "trt":
                unet_ms, unet_note = eager_ms * TS_VS_EAGER_UNET_RATIO, "TRT .ts, PROJECTED from eager x 0.655 (4070S ratio)"
            elif backend == "trt_stagewise":
                unet_ms = eager_ms * TS_VS_EAGER_UNET_RATIO * (2.40 / 3.03)
                unet_note = "stagewise, PROJECTED (4070S .ts ratio x plan 2.40/3.03 ms/frame)"
        if not (taesd_ms and unet_ms):
            self.estimate = {"available": False, "line": "estimate: n/a (gpu_selftest.json lacks UNet or TAESD timings)"}
            return
        fps = 8000.0 / (unet_ms + taesd_ms)
        self.estimate = {"available": True, "gpu_path_fps": round(fps, 1), "unet_ms_bs8": round(unet_ms, 2),
                         "taesd_ms_bs8": round(taesd_ms, 2), "unet_note": unet_note, "taesd_note": taesd_note,
                         "line": (f"estimate: GPU-path ~{fps:.0f} fps (UNet {unet_ms:.1f} ms/bs8 [{unet_note}] + "
                                  f"TAESD {taesd_ms:.1f} ms/bs8 [{taesd_note}]; excludes compose/encode/WebRTC)")}

    # -- outputs -----------------------------------------------------------------------------------------
    def summary_lines(self):
        gpu = self.facts.get("gpu") or {}
        lines = [f"recipe={self.recipe} gpu={gpu.get('name')} sm{gpu.get('compute_capability')} "
                 f"vram={gpu.get('memory_total_mib')}MiB venv torch={(self.facts.get('venv') or {}).get('torch')}"]
        if self.recipe == "legacy_int8":
            lines.append("legacy_int8: old chain (run_trt_stagewise_server.sh)")
            return lines
        e = self.emitted
        unet = self.unet.get("backend")
        engine = self.unet.get("engine") or {}
        lines.append(f"VAE={e.get('MUSETALK_VAE_BACKEND')} taesd_backend={e.get('MUSETALK_TAESD_BACKEND', 'compiled')} "
                     f"compile={e.get('MUSETALK_TAESD_COMPILE')} | UNet={unet} "
                     f"{engine.get('path') or engine.get('dir') or ''} ({self.unet.get('reason')})")
        lines.append(f"buckets={e.get('HLS_SCHEDULER_FIXED_BATCH_SIZES')} max={e.get('HLS_SCHEDULER_MAX_BATCH')} "
                     f"workers prep/compose/encode={e.get('HLS_PREP_WORKERS')} cache={e.get('AVATAR_CACHE_MAX_MEMORY_MB')}MB "
                     f"vp8={e.get('WEBRTC_VP8_ENCODER')} h264={e.get('WEBRTC_H264_IMPL', 'aiortc')}")
        if self.recipe_groups:
            on = [g["name"] for g in self.recipe_groups if g["status"] == "enabled"]
            off = [g["name"] for g in self.recipe_groups if g["status"] == "off"]
            lines.append(f"{self.recipe} groups enabled: {', '.join(on) or 'none'}; switched off in the file: {len(off)}")
        if self.recipe_levers:
            enabled = [i["name"] for i in self.recipe_levers if i["status"] == "enabled"]
            dropped = [f"{i['name']} ({i['reason']})" for i in self.recipe_levers if i["status"] == "dropped"]
            lines.append(f"{self.recipe} levers enabled: {', '.join(enabled) or 'none'}")
            if dropped:
                lines.append(f"{self.recipe} levers dropped: {'; '.join(dropped)}")
        lines.append(f"expect: vae={self.expect['vae']} unet={self.expect['unet']}")
        if getattr(self, "estimate", None):
            lines.append(self.estimate["line"])
        return lines

    def report(self):
        return {
            "schema": REPORT_SCHEMA, "created_utc": utc_now(), "recipe": self.recipe,
            "repo_root": str(self.repo_root), "venv": str(self.venv) if self.venv else None,
            "ok": not self.errors, "errors": self.errors, "warnings": self.warnings, "notes": self.notes,
            "expect": self.expect, "unet": self.unet, "engines": self.engines,
            "engine_store_backend": self.store.backend, "engine_store_module_error": self.store.module_error,
            "layers": {"overrides_files": [str(p) for p in self.view.override_files],
                       "overrides_files_used": self.view.override_files_used,
                       "recipe_file": self.recipe_file,
                       "order": "caller > overrides > recipe levers > resolver > code default"},
            "recipe_groups": self.recipe_groups, "resolve_passes": self.passes,
            "recipe_levers": self.recipe_levers, "levers": getattr(self, "levers", []),
            "native_vp8": getattr(self, "native_vp8", None), "chin_tools": getattr(self, "chin", None),
            "estimate": getattr(self, "estimate", None), "summary": self.summary_lines(),
            "decisions": self.decisions, "emitted": self.emitted, "facts": self.facts,
        }

    def env_text(self):
        gpu = self.facts.get("gpu") or {}
        head = [f"# Generated by scripts/musetalk_host_profile.py resolve at {utc_now()}",
                f"# recipe={self.recipe} gpu={gpu.get('name')!s} sm{gpu.get('compute_capability')} "
                f"venv={self.venv}",
                "# Regenerated on EVERY launch - do not edit. Operator levers go in .runtime/musetalk_overrides.env.",
                "# Loading rule: export KEY only if it is unset (caller > overrides > this file).",
                "# Values are single-quoted; a literal ' is written as '\\''."]
        if self.errors:
            head.append("# RESOLUTION FAILED - no knobs emitted:")
            head += [f"#   {e}" for e in self.errors]
            return "\n".join(head) + "\n"
        body = [f"{k}={shell_single_quote(v)}" for k, v in self.emitted.items()]
        return "\n".join(head + body) + "\n"


def resolve_recipe(repo_root, venv, recipe, facts, environ=None, max_passes=32):
    """Resolve, re-running until every @lever group is atomic (fully enabled or excluded)."""
    excluded, shared = {}, {}
    resolver = None
    for attempt in range(1, max_passes + 1):
        resolver = Resolver(repo_root, venv, recipe, facts, environ, excluded_groups=excluded, shared=shared).run()
        resolver.passes = attempt
        violations = resolver.group_violations() if not resolver.errors else {}
        if not violations:
            break
        excluded = dict(resolver.excluded_groups, **violations)
    resolver.finalize_groups()
    return resolver


# --------------------------------------------------------------------------- verify-log
VAE_ACTIVE_RE = re.compile(r"VAE decode backend active: (\S+)")
VAE_PYTORCH_RE = re.compile(r"VAE decode backend: PyTorch")
UNET_ACTIVE_RE = re.compile(r"UNet backend active: (\S+)")
UNET_PYTORCH_RE = re.compile(r"UNet backend: PyTorch")
EXTRA_PATTERNS = {
    "taesd_trt_fallback": re.compile(r"TAESD TRT unavailable \((.*)\); using compiled TAESD"),
    "taesd_trt_loaded": re.compile(r"TAESD TRT backend: (key=\S+ .*)"),
    "eager_unet_released": re.compile(r"Eager UNet released \(MUSETALK_FREE_EAGER_UNET=1\)"),
    "vp8_native": re.compile(r"VP8 encoder=native (.*)"),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
}
# Names printed by scripts/avatar_manager_parallel.py:191/193 (VAE) and :336/:342 (UNet):
#   VAE: taesd (TaesdVaeDecodeBackend), taesd_trt (TaesdTrtBackend), tensorrt_stagewise[_int8_mixed],
#        tensorrt, tensorrt_hybrid; "VAE decode backend: PyTorch" -> pytorch
#   UNet: tensorrt_unet (TrtUnetBackend), tensorrt_unet_multi (MultiTrtUnetBackend, MUSETALK_TRT_UNET_PATHS),
#         tensorrt_unet_stagewise (StagewiseTrtUnetBackend); "UNet backend: PyTorch" -> pytorch (eager)
VAE_EXPECT = {"taesd": {"taesd", "taesd_trt"}, "taesd_compiled": {"taesd"}, "taesd_trt": {"taesd_trt"},
              "pytorch": {"pytorch"}, "trt_stagewise": {"tensorrt_stagewise", "tensorrt_stagewise_int8_mixed"},
              "any": None}
UNET_EXPECT = {"trt": {"tensorrt_unet", "tensorrt_unet_multi"}, "trt_stagewise": {"tensorrt_unet_stagewise"},
               "tensorrt_any": {"tensorrt_unet", "tensorrt_unet_multi", "tensorrt_unet_stagewise"},
               "eager": {"pytorch"}, "any": None}


def scan_log_text(text):
    found = {"vae": None, "unet": None, "extras": {}}
    for line in text.splitlines():
        if found["vae"] is None:
            m = VAE_ACTIVE_RE.search(line)
            if m:
                found["vae"] = m.group(1)
            elif VAE_PYTORCH_RE.search(line):
                found["vae"] = "pytorch"
        if found["unet"] is None:
            m = UNET_ACTIVE_RE.search(line)
            if m:
                found["unet"] = m.group(1)
            elif UNET_PYTORCH_RE.search(line):
                found["unet"] = "pytorch"
        for name, pattern in EXTRA_PATTERNS.items():
            if name not in found["extras"]:
                m = pattern.search(line)
                if m:
                    found["extras"][name] = m.group(1) if m.groups() else True
    return found


def verify_log(log_path, offset=0, expect_vae="taesd", expect_unet="any", timeout=30.0, poll=1.0):
    """(exit_code, result dict)."""
    if expect_vae not in VAE_EXPECT:
        return 2, {"status": "usage", "error": f"--expect-vae must be one of {sorted(VAE_EXPECT)}"}
    if expect_unet not in UNET_EXPECT:
        return 2, {"status": "usage", "error": f"--expect-unet must be one of {sorted(UNET_EXPECT)}"}
    deadline = time.time() + max(0.0, float(timeout))
    notes = []
    found = {"vae": None, "unet": None, "extras": {}}
    while True:
        try:
            size = os.path.getsize(log_path)
            start = int(offset or 0)
            if start > size:
                notes.append(f"offset {start} > log size {size} (rotated/truncated?): scanning from 0")
                start = 0
            with open(log_path, "rb") as handle:
                handle.seek(start)
                data = handle.read()
            found = scan_log_text(data.decode("utf-8", errors="replace"))
        except OSError as exc:
            notes.append(f"cannot read {log_path}: {exc}")
        result = {"log": str(log_path), "offset": offset, "expected": {"vae": expect_vae, "unet": expect_unet},
                  "found": {"vae": found["vae"], "unet": found["unet"]}, "extras": found["extras"],
                  "notes": sorted(set(notes))}
        mismatches = []
        vae_ok_set, unet_ok_set = VAE_EXPECT[expect_vae], UNET_EXPECT[expect_unet]
        if found["vae"] is not None and vae_ok_set is not None and found["vae"] not in vae_ok_set:
            mismatches.append(f"VAE decode backend {found['vae']!r} (expected {expect_vae}: {sorted(vae_ok_set)})")
        if found["unet"] is not None and unet_ok_set is not None and found["unet"] not in unet_ok_set:
            mismatches.append(f"UNet backend {found['unet']!r} (expected {expect_unet}: {sorted(unet_ok_set)})")
        if mismatches:
            result.update(status="mismatch", mismatches=mismatches)
            return 1, result
        vae_done = found["vae"] is not None or vae_ok_set is None
        unet_done = found["unet"] is not None or unet_ok_set is None
        if vae_done and unet_done:
            result.update(status="match")
            return 0, result
        if time.time() >= deadline:
            result.update(status="not_found", missing=[k for k, done in (("vae", vae_done), ("unet", unet_done))
                                                       if not done])
            return 3, result
        time.sleep(max(0.05, float(poll)))


# --------------------------------------------------------------------------- CLI
def _default_venv(environ):
    workspace = environ.get("WORKSPACE") or "/workspace"
    return environ.get("VENV_PATH") or str(Path(workspace) / ".venvs" / "musetalk_trt_stagewise")


def cmd_detect(args):
    facts = detect(args.repo_root, args.venv)
    print(json.dumps(facts, indent=1, sort_keys=False))
    return 0


def cmd_resolve(args):
    repo_root = Path(args.repo_root).resolve()
    recipe = args.recipe or os.environ.get("MUSETALK_RECIPE")
    if not recipe:
        for path in overrides_files(repo_root, os.environ):
            for key, value, _ in parse_env_file(path):
                if key == "MUSETALK_RECIPE":
                    recipe = value
                    break
            if recipe:
                break
    recipe = (recipe or "fast").strip()
    if args.recipe and os.environ.get("MUSETALK_RECIPE") and os.environ["MUSETALK_RECIPE"] != args.recipe:
        log(f"warning: --recipe {args.recipe} differs from MUSETALK_RECIPE={os.environ['MUSETALK_RECIPE']}; "
            "using --recipe")
    facts = detect(repo_root, args.venv)
    resolver = resolve_recipe(repo_root, args.venv, recipe, facts)
    out = Path(args.out) if args.out else repo_root / ".runtime" / "musetalk_resolved.env"
    report_path = Path(args.report) if args.report else repo_root / ".runtime" / "musetalk_resolved.json"
    atomic_write_text(out, resolver.env_text())
    atomic_write_text(report_path, json.dumps(resolver.report(), indent=1) + "\n")
    for line in resolver.summary_lines():
        log(line)
    for w in resolver.warnings:
        log(f"WARNING: {w}")
    for e in resolver.errors:
        log(f"ERROR: {e}")
    log(f"wrote {out} and {report_path}")
    return 2 if resolver.errors else 0


def _expect_from_resolved(path, which):
    data = _read_json(path) or {}
    return (data.get("expect") or {}).get(which)


def cmd_verify_log(args):
    resolved = args.resolved
    if resolved is None and args.repo_root:
        candidate = Path(args.repo_root) / ".runtime" / "musetalk_resolved.json"
        resolved = str(candidate) if candidate.exists() else None
    expect_vae, expect_unet = args.expect_vae, args.expect_unet
    for which in ("vae", "unet"):
        value = expect_vae if which == "vae" else expect_unet
        if value == "auto":
            got = _expect_from_resolved(resolved, which) if resolved else None
            if not got:
                print(json.dumps({"status": "usage", "error": f"--expect-{which} auto needs --resolved with an "
                                                              "'expect' section"}))
                return 2
            if which == "vae":
                expect_vae = got
            else:
                expect_unet = got
    rc, result = verify_log(args.log, args.offset, expect_vae, expect_unet, args.timeout, args.poll)
    print(json.dumps(result, indent=1))
    status = result.get("status")
    if rc == 0:
        log(f"verify-log MATCH vae={result['found']['vae']} unet={result['found']['unet']}")
    elif rc == 1:
        log("verify-log MISMATCH: " + "; ".join(result.get("mismatches", [])))
    elif rc == 3:
        log(f"verify-log NOT FOUND within {args.timeout:g}s: missing {result.get('missing')} "
            f"(found vae={result['found']['vae']} unet={result['found']['unet']})")
    else:
        log(f"verify-log {status}: {result.get('error')}")
    if result.get("extras", {}).get("taesd_trt_fallback"):
        log(f"WARNING: TAESD TRT fell back to compiled TAESD: {result['extras']['taesd_trt_fallback']}")
    return rc


def _cli_store(args):
    repo_root = Path(args.repo_root).resolve()
    view = EnvView(os.environ, repo_root)  # store-root overrides may live in the overrides file
    return EngineStore(repo_root, view.get)


def cmd_engine_key(args):
    facts = detect(args.repo_root, args.venv)
    store = _cli_store(args)
    key = store.engine_key(args.kind, facts)
    if not key:
        log(f"cannot compute a {args.kind} engine key: {store.key_errors.get(args.kind) or store.module_error} "
            f"({json.dumps(facts.get('engine_facts'))})")
        return 2
    print(key)
    return 0


def cmd_find_engine(args):
    facts = detect(args.repo_root, args.venv)
    store = _cli_store(args)
    batch = args.batch if args.batch else (TS_ENGINE_BATCH if args.kind == "unet_ts" else None)
    # Same policy as resolve: same-cc reuse only for the .ts UNet (warned); raw plans need the exact key.
    entry, entries = store.find(args.kind, facts, batch=batch, allow_same_cc=(args.kind == "unet_ts"),
                                prefer_batch=8 if args.kind == "taesd_trt" else None)
    payload = {"kind": args.kind, "key": store.engine_key(args.kind, facts), "key_error": store.key_errors.get(args.kind),
               "store": str(store.store_root(args.kind)), "backend": store.backend, "batch": batch,
               "engine": entry, "env": store.entry_env(entry),
               "engine_path": engine_path_of(entry) if args.kind == "unet_ts" else (entry or {}).get("dir"),
               "candidates": [compact_entry(e) for e in entries],
               "legacy_candidates": store.legacy(args.kind) if entry is None else []}
    print(json.dumps(payload, indent=1, default=str))
    return 0 if entry else 3


def build_parser():
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")

    def common(p):
        p.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
        p.add_argument("--venv", default=_default_venv(env))

    p = sub.add_parser("detect", help="print host facts JSON")
    common(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("resolve", help="write the resolved env + report")
    common(p)
    p.add_argument("--out", default=None, help="env file (default <repo>/.runtime/musetalk_resolved.env)")
    p.add_argument("--report", default=None, help="JSON report (default <repo>/.runtime/musetalk_resolved.json)")
    p.add_argument("--recipe", default=None, choices=RECIPES,
                   help="default: MUSETALK_RECIPE, then the overrides files, then fast")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("verify-log", help="check the server log for the expected backends")
    p.add_argument("--log", required=True)
    p.add_argument("--offset", type=int, default=0, help="byte offset recorded before the server was spawned")
    p.add_argument("--expect-vae", default="taesd", choices=sorted(VAE_EXPECT) + ["auto"])
    p.add_argument("--expect-unet", default="any", choices=sorted(UNET_EXPECT) + ["auto"])
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--poll", type=float, default=1.0)
    p.add_argument("--resolved", default=None, help="musetalk_resolved.json (for 'auto' expectations)")
    p.add_argument("--repo-root", default=None, help="finds .runtime/musetalk_resolved.json for 'auto'")
    p.set_defaults(func=cmd_verify_log)

    for name, func, helptext in (("engine-key", cmd_engine_key, "print the engine key for this GPU + venv"),
                                 ("find-engine", cmd_find_engine, "JSON of the best usable engine (exit 3: none)")):
        p = sub.add_parser(name, help=helptext)
        common(p)
        p.add_argument("--kind", default="unet_ts", choices=ENGINE_KINDS)
        if name == "find-engine":
            p.add_argument("--batch", type=int, default=None)
        p.set_defaults(func=func)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
