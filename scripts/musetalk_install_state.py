#!/usr/bin/env python3
"""Installer state helper for scripts/install_musetalk.sh (stdlib only, Python >= 3.8).

Never imports torch, tensorrt or any package from the target venv. Everything about
the venv is read from its files (pyvenv.cfg, site-packages/*.dist-info) or from tiny
`<venv>/bin/python -c` subprocesses that only call importlib.util.find_spec (which
locates a top-level package without executing it).

Subcommands
  detect-matrix  [--matrix auto|cu121|cu128]            -> JSON {matrix, reason, gpu}
  venv-facts     --venv V                               -> JSON facts about a venv
  check          --repo-root R --venv V [...]           -> exit 0 ok, 10 clean install needed,
                                                           11 repairable in place
  stamp          --repo-root R --venv V --matrix M ...  -> writes .runtime/install_state.json

Test hooks (same names the resolver uses):
  MUSETALK_HOST_FACTS_JSON  path to a JSON file with {"gpus": [{"name", "compute_capability"}...]}
  MUSETALK_NVIDIA_SMI       nvidia-smi binary to run instead of `nvidia-smi`
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # --check is read-only: never drop __pycache__ into the repo

STATE_SCHEMA = "musetalk_install_state_v1"
CHECK_SCHEMA = "musetalk_install_check_v1"
EXIT_OK = 0
EXIT_CLEAN = 10
EXIT_REPAIR = 11
MATRICES = ("cu121", "cu128")
DEFAULT_CHIN_PINS_FILE = "requirements/chin-tools.in"

# Group name -> requirements file (relative to the repo root).
GROUP_FILES = {
    "server": "requirements/server.in",
    "kokoro": "requirements/kokoro.in",
    "legacy_int8": "requirements/legacy-int8.in",
    "avatar_prep": "requirements/avatar-prep.in",
}
# Top-level modules whose presence (find_spec, not import) proves a group is importable.
GROUP_MODULES = {
    "server": ["torch", "torchvision", "torchaudio", "torch_tensorrt", "tensorrt", "onnx", "diffusers",
               "transformers", "cv2", "numpy", "aiortc", "av", "cffi", "fastapi", "uvicorn", "boto3",
               "librosa", "soundfile", "numba", "imageio", "ffmpeg", "omegaconf", "multipart"],
    "kokoro": ["kokoro", "misaki", "spacy", "en_core_web_sm", "espeakng_loader"],
    "legacy_int8": ["modelopt", "onnx"],
    "avatar_prep": ["mmcv", "mmdet", "mmengine", "mmpose"],
}
# Model files the server needs (relative to the repo root). TAESD is the fast-recipe decoder.
SERVER_MODEL_FILES = [
    "models/musetalkV15/musetalk.json",
    "models/musetalkV15/unet.pth",
    "models/sd-vae/config.json",
    "models/sd-vae/diffusion_pytorch_model.bin",
    "models/whisper/config.json",
    "models/whisper/pytorch_model.bin",
    "models/whisper/preprocessor_config.json",
    "models/face-parse-bisent/79999_iter.pth",
    "models/face-parse-bisent/resnet18-5c106cde.pth",
    "models/taesd/config.json",
    "models/taesd/diffusion_pytorch_model.safetensors",
]
AVATAR_PREP_MODEL_FILES = [
    "models/dwpose/dw-ll_ucoco_384.pth",
    "models/syncnet/latentsync_syncnet.pt",
    "models/face_detection/s3fd.pth",
]
KEY_VERSIONS = ["torch", "torchvision", "torchaudio", "torch-tensorrt", "tensorrt-cu12", "tensorrt", "triton",
                "onnx", "numpy", "diffusers", "transformers", "aiortc", "av", "cffi", "kokoro", "spacy",
                "nvidia-modelopt", "mmcv", "setuptools", "pip"]
NATIVE_VP8_PACKAGES = {"aiortc": "1.14.0", "av": "16.1.0", "cffi": "2.1.1"}


# ----------------------------------------------------------------------------- helpers
def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def parse_requirement_names(path: Path):
    """Top-level requirement names (+ pinned URL versions) from a requirements .in file."""
    out = []
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        url_version = None
        if " @ " in line:
            name, url = [part.strip() for part in line.split(" @ ", 1)]
            match = re.search(r"/[A-Za-z0-9_.]+?-(\d[^-/]*)-py\d", url)
            url_version = match.group(1) if match else None
        else:
            name = re.split(r"[\s;<>=!~\[(]", line, maxsplit=1)[0]
        name = name.split("[", 1)[0]
        if name:
            out.append({"name": norm(name), "raw": line, "url_version": url_version})
    return out


def parse_constraints(path: Path) -> dict:
    """{normalized name: (operator, version)} for '==' / '===' lines."""
    pins = {}
    if not path.is_file():
        return pins
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or " @ " in line:
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(===|==)\s*([^\s;]+)", line)
        if match:
            pins[norm(match.group(1))] = (match.group(2), match.group(3))
    return pins


def version_matches(installed: str, op: str, pinned: str) -> bool:
    if op == "===":
        return installed == pinned
    if installed == pinned:
        return True
    # PEP 440: '==X' without a local label matches any local build of X.
    return "+" not in pinned and installed.split("+", 1)[0] == pinned


# ----------------------------------------------------------------------------- GPU / matrix
def gpu_facts() -> dict:
    """{'gpus': [...], 'selected': {...}|None, 'source': str, 'error': str|None} without torch."""
    facts_path = os.environ.get("MUSETALK_HOST_FACTS_JSON", "").strip()
    gpus = []
    source = "nvidia-smi"
    error = None
    if facts_path:
        source = f"MUSETALK_HOST_FACTS_JSON={facts_path}"
        try:
            data = json.loads(Path(facts_path).read_text())
            for index, gpu in enumerate(data.get("gpus") or []):
                gpus.append({"index": int(gpu.get("index", index)), "name": str(gpu.get("name", "")),
                             "compute_capability": str(gpu.get("compute_capability", ""))})
        except (OSError, ValueError) as exc:
            error = f"cannot read host facts: {exc}"
    else:
        smi = os.environ.get("MUSETALK_NVIDIA_SMI", "").strip() or shutil.which("nvidia-smi") or ""
        if not smi:
            error = "nvidia-smi not found"
        else:
            try:
                proc = subprocess.run([smi, "--query-gpu=index,name,compute_cap", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=20)
                if proc.returncode != 0:
                    error = f"nvidia-smi exit {proc.returncode}: {proc.stderr.strip()[:200]}"
                for line in proc.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 3 and parts[0].isdigit():
                        gpus.append({"index": int(parts[0]), "name": parts[1], "compute_capability": parts[2]})
            except (OSError, subprocess.SubprocessError) as exc:
                error = f"nvidia-smi failed: {exc}"
    selected = None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1", "none", "NoDevFiles"):
        gpus_visible = []
    elif visible and visible.split(",")[0].strip().isdigit():
        wanted = int(visible.split(",")[0].strip())
        gpus_visible = [gpu for gpu in gpus if gpu["index"] == wanted] or gpus[:1]
    else:
        gpus_visible = gpus
    if gpus_visible:
        selected = gpus_visible[0]
    return {"gpus": gpus, "selected": selected, "source": source, "error": error}


def cc_tuple(value: str):
    match = re.match(r"^\s*(\d+)\.(\d+)", value or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def detect_matrix(requested: str = "auto") -> dict:
    facts = gpu_facts()
    selected = facts["selected"]
    if requested in MATRICES:
        return {"matrix": requested, "reason": "requested explicitly", "gpu": selected, "gpu_source": facts["source"],
                "gpu_error": facts["error"], "gpu_visible": selected is not None}
    if selected is None:
        return {"matrix": "cu121", "reason": "no GPU visible (default cu121)", "gpu": None,
                "gpu_source": facts["source"], "gpu_error": facts["error"], "gpu_visible": False}
    cc = cc_tuple(selected.get("compute_capability", ""))
    if cc is None:
        return {"matrix": "cu121", "reason": f"compute capability unknown for {selected.get('name')} (default cu121)",
                "gpu": selected, "gpu_source": facts["source"], "gpu_error": facts["error"], "gpu_visible": True}
    if cc >= (10, 0):
        return {"matrix": "cu128", "reason": f"compute capability {cc[0]}.{cc[1]} >= 10.0 (Blackwell) needs cu128",
                "gpu": selected, "gpu_source": facts["source"], "gpu_error": facts["error"], "gpu_visible": True}
    return {"matrix": "cu121", "reason": f"compute capability {cc[0]}.{cc[1]} < 10.0",
            "gpu": selected, "gpu_source": facts["source"], "gpu_error": facts["error"], "gpu_visible": True}


# ----------------------------------------------------------------------------- venv facts
def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def read_pyvenv_cfg(venv: Path) -> dict:
    cfg = {}
    try:
        for line in (venv / "pyvenv.cfg").read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                cfg[key.strip().lower()] = value.strip()
    except OSError:
        pass
    return cfg


def site_packages(venv: Path):
    candidates = sorted((venv / "lib").glob("python3.*/site-packages")) if (venv / "lib").is_dir() else []
    return candidates[0] if candidates else None


def installed_dists(sp) -> dict:
    """{normalized name: {'name', 'version', 'dist_info'}} from *.dist-info directory names."""
    dists = {}
    if sp is None or not Path(sp).is_dir():
        return dists
    for entry in Path(sp).iterdir():
        if not entry.name.endswith(".dist-info"):
            continue
        stem = entry.name[: -len(".dist-info")]
        match = re.match(r"^(.+?)-(\d[^-]*)$", stem)
        if not match:
            continue
        name, version = match.group(1), match.group(2)
        meta = entry / "METADATA"
        try:
            for line in meta.read_text(errors="replace").splitlines()[:40]:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip() or name
                elif line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip() or version
        except OSError:
            pass
        dists[norm(name)] = {"name": name, "version": version, "dist_info": entry.name}
    return dists


def torch_cuda_tag(dists: dict):
    torch = dists.get("torch")
    if not torch:
        return None
    match = re.search(r"\+(cu\d+|cpu|rocm[\d.]+)$", torch["version"])
    return match.group(1) if match else "pypi-default"


def run_venv_python(venv: Path, code: str, args=(), timeout: int = 60):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    try:
        proc = subprocess.run([str(venv_python(venv)), "-B", "-c", code, *args], capture_output=True, text=True,
                              timeout=timeout, cwd="/", env=env)
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


def venv_facts(venv: Path) -> dict:
    venv = Path(venv)
    py = venv_python(venv)
    cfg = read_pyvenv_cfg(venv)
    sp = site_packages(venv)
    dists = installed_dists(sp)
    facts = {
        "venv": str(venv),
        "exists": venv.exists(),
        "python": str(py),
        "python_executable": py.exists() and os.access(py, os.X_OK),
        "pyvenv_version": cfg.get("version_info") or cfg.get("version"),
        "python_version": None,
        "machine": None,
        "site_packages": str(sp) if sp else None,
        "dist_count": len(dists),
        "torch_cuda_tag": torch_cuda_tag(dists),
        "versions": {name: dists[name]["version"] for name in KEY_VERSIONS if name in dists},
    }
    if facts["python_executable"]:
        rc, out, err = run_venv_python(venv, "import sys,platform;print(sys.version.split()[0]);print(platform.machine())",
                                       timeout=30)
        if rc == 0 and out.strip():
            lines = out.strip().splitlines()
            facts["python_version"] = lines[0]
            facts["machine"] = lines[1] if len(lines) > 1 else None
        else:
            facts["python_error"] = (err or out).strip()[-300:]
    return facts


def find_specs(venv: Path, modules):
    code = ("import importlib.util,json,sys\n"
            "out={}\n"
            "for m in sys.argv[1:]:\n"
            "    try:\n"
            "        out[m]=importlib.util.find_spec(m) is not None\n"
            "    except Exception as exc:\n"
            "        out[m]=False\n"
            "print(json.dumps(out))\n")
    rc, out, err = run_venv_python(venv, code, modules, timeout=60)
    if rc != 0:
        return None, (err or out).strip()[-300:]
    try:
        return json.loads(out.strip().splitlines()[-1]), None
    except (ValueError, IndexError):
        return None, out.strip()[-300:]


# ----------------------------------------------------------------------------- check
class Report:
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.clean = []
        self.repair = []

    def ok(self, name, detail=""):
        self.checks.append({"check": name, "status": "ok", "detail": detail})

    def warn(self, name, detail):
        self.checks.append({"check": name, "status": "warn", "detail": detail})
        self.warnings.append(f"{name}: {detail}")

    def need_clean(self, name, detail):
        self.checks.append({"check": name, "status": "clean", "detail": detail})
        self.clean.append(f"{name}: {detail}")

    def need_repair(self, name, detail):
        self.checks.append({"check": name, "status": "repair", "detail": detail})
        self.repair.append(f"{name}: {detail}")

    @property
    def exit_code(self):
        if self.clean:
            return EXIT_CLEAN
        if self.repair:
            return EXIT_REPAIR
        return EXIT_OK


def load_stamp(path: Path):
    try:
        data = json.loads(path.read_text())
        return data if data.get("schema") == STATE_SCHEMA else None
    except (OSError, ValueError):
        return None


def same_path(a, b) -> bool:
    try:
        return os.path.realpath(str(a)) == os.path.realpath(str(b))
    except OSError:
        return str(a) == str(b)


def resolve_group(explicit: str, stamp_groups: dict, key: str, default: bool) -> tuple:
    """(enabled, source) for a group: explicit flag > stamp > default."""
    if explicit in ("1", "0"):
        return explicit == "1", "flag"
    if stamp_groups and key in stamp_groups:
        value = stamp_groups[key]
        if isinstance(value, dict):
            value = value.get("enabled", False)
        return bool(value), "stamp"
    return default, "default"


def native_vp8_supported(facts: dict) -> bool:
    return (platform.system() == "Linux" and (facts.get("machine") or platform.machine()) == "x86_64"
            and str(facts.get("python_version") or "").startswith("3.10."))


def validate_native_vp8(repo_root: Path, directory: Path):
    """Hash-verify the native VP8 install with the repo's own stdlib validator (no downloads)."""
    module_path = repo_root / "scripts" / "install_native_vp8.py"
    if not module_path.is_file():
        return False, f"{module_path} missing"
    try:
        spec = importlib.util.spec_from_file_location("_musetalk_install_native_vp8", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.validate_install(directory)
        return True, f"{result.get('file_count')} files verified in {result.get('directory')}"
    except Exception as exc:  # validator raises RuntimeError with a precise message
        return False, f"{type(exc).__name__}: {exc}"


def hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def check_chin_venv(chin_venv: Path, pins_file: Path, report: Report) -> dict:
    info = {"venv": str(chin_venv), "python": str(venv_python(chin_venv))}
    py = venv_python(chin_venv)
    if not (py.exists() and os.access(py, os.X_OK)):
        report.need_repair("chin_tools", f"venv python missing: {py} (install with --with-chin-tools)")
        return info
    dists = installed_dists(site_packages(chin_venv))
    pins = {}
    for req in parse_requirement_names(pins_file):
        match = re.search(r"===?\s*([^\s;]+)", req["raw"])
        if match:
            pins[req["name"]] = match.group(1)
    bad = []
    for name, version in pins.items():
        have = dists.get(name, {}).get("version")
        if have != version:
            bad.append(f"{name} {have or 'missing'} != {version}")
    info["versions"] = {name: dists.get(name, {}).get("version") for name in pins}
    if bad:
        report.need_repair("chin_tools", "; ".join(bad))
    else:
        report.ok("chin_tools", f"{chin_venv} ({', '.join(f'{k} {v}' for k, v in info['versions'].items())})")
    return info


def cmd_check(args) -> int:
    repo = Path(args.repo_root).resolve()
    venv = Path(args.venv)
    report = Report()
    result = {"schema": CHECK_SCHEMA, "created_utc": utc_now(), "repo_root": str(repo), "venv": str(venv)}

    stamp_path = Path(args.state_file) if args.state_file else repo / ".runtime" / "install_state.json"
    stamp = load_stamp(stamp_path)
    if stamp is not None and not same_path(stamp.get("venv", ""), venv):
        report.warn("stamp", f"{stamp_path} describes another venv ({stamp.get('venv')}); ignoring it")
        stamp = None
    stamp_groups = (stamp or {}).get("groups") or {}

    # 1. venv python + interpreter version.
    facts = venv_facts(venv)
    result["venv_facts"] = facts
    if not facts["exists"]:
        report.need_clean("venv", f"{venv} does not exist")
    elif not facts["python_executable"]:
        report.need_clean("venv", f"{facts['python']} missing or not executable")
    elif not facts.get("python_version"):
        report.need_clean("venv", f"venv python does not run: {facts.get('python_error', 'no output')}")
    elif not facts["python_version"].startswith("3.10."):
        report.need_clean("venv", f"venv python is {facts['python_version']}; the pinned stack needs CPython 3.10")
    else:
        report.ok("venv", f"{venv} (CPython {facts['python_version']}, {facts['dist_count']} dists)")

    # 2. matrix.
    matrix_info = detect_matrix(args.matrix)
    tag = facts.get("torch_cuda_tag")
    expected = matrix_info["matrix"]
    if args.matrix == "auto" and not matrix_info["gpu_visible"]:
        # A GPU-less check (image build, CUDA hidden) must never demand a clean reinstall.
        expected = (stamp or {}).get("matrix") or (tag if tag in MATRICES else "cu121")
        report.warn("matrix", f"no GPU visible ({matrix_info.get('gpu_error') or 'none'}); "
                              f"matrix not verified against hardware, assuming {expected}")
        matrix_info = dict(matrix_info, matrix=expected, reason="no GPU visible: kept the installed matrix")
    result["matrix"] = matrix_info
    if facts["exists"] and facts["python_executable"]:
        if tag is None:
            report.need_repair("matrix", "torch is not installed in the venv")
        elif tag != expected:
            report.need_clean("matrix", f"venv has torch {facts['versions'].get('torch')} ({tag}) but "
                                        f"{matrix_info['reason']} -> needs {expected}")
        else:
            report.ok("matrix", f"{expected}: torch {facts['versions'].get('torch')} ({matrix_info['reason']})")
    if stamp is not None and stamp.get("matrix") and stamp.get("matrix") != expected:
        report.need_clean("stamp_matrix", f"install stamp says {stamp.get('matrix')}, expected {expected}")

    # 3. stamp.
    constraints = repo / "requirements" / f"constraints-{expected}.txt"
    if stamp is None:
        if not report.clean:
            report.warn("stamp", f"no install stamp at {stamp_path}: validating a pre-existing venv by its "
                                 "packages and files (OK if everything below passes)")
    else:
        now_sha = sha256_file(constraints)
        if stamp.get("constraints_sha256") and now_sha and stamp.get("constraints_sha256") != now_sha:
            report.warn("stamp", f"{constraints.name} changed since the install on {stamp.get('created_utc')}; "
                                 "installed versions are compared with the current pins below")
        else:
            report.ok("stamp", f"{stamp_path} ({stamp.get('created_utc')})")

    # 4. groups.
    groups = {"server": (True, "always")}
    groups["kokoro"] = resolve_group(args.kokoro, stamp_groups, "kokoro", True)
    groups["legacy_int8"] = resolve_group(args.legacy_int8, stamp_groups, "legacy_int8", False)
    groups["avatar_prep"] = resolve_group(args.avatar_prep, stamp_groups, "avatar_prep", False)
    groups["chin_tools"] = resolve_group(args.chin_tools, stamp_groups, "chin_tools", False)
    native_mode = args.native_vp8
    if native_mode in ("1", "0"):
        groups["native_vp8"] = (native_mode == "1", "flag")
    elif stamp_groups and "native_vp8" in stamp_groups:
        groups["native_vp8"] = resolve_group("", stamp_groups, "native_vp8", False)
    else:
        groups["native_vp8"] = (native_vp8_supported(facts), "auto")
    result["groups"] = {name: {"enabled": enabled, "source": source} for name, (enabled, source) in groups.items()}

    # 5. packages + pins (only meaningful when the interpreter works).
    if facts.get("python_version"):
        sp_dists = installed_dists(Path(facts["site_packages"])) if facts.get("site_packages") else {}
        pins = parse_constraints(constraints)
        if not pins:
            report.need_repair("constraints", f"{constraints} missing or empty")
        missing = []
        url_mismatch = []
        for group, file_rel in GROUP_FILES.items():
            if not groups.get(group, (False, ""))[0]:
                continue
            for req in parse_requirement_names(repo / file_rel):
                have = sp_dists.get(req["name"])
                if have is None:
                    missing.append(f"{req['name']} ({group})")
                elif req["url_version"] and have["version"] != req["url_version"]:
                    url_mismatch.append(f"{req['name']} {have['version']} != {req['url_version']}")
        mismatched = []
        for name, (op, version) in sorted(pins.items()):
            have = sp_dists.get(name)
            if have is not None and not version_matches(have["version"], op, version):
                mismatched.append(f"{name} {have['version']} != {op}{version}")
        if missing:
            report.need_repair("packages", "missing: " + ", ".join(missing[:40]))
        else:
            report.ok("packages", "all top-level requirements of the selected groups are installed")
        if mismatched or url_mismatch:
            report.need_repair("pins", f"{len(mismatched) + len(url_mismatch)} installed versions differ from "
                                       f"{constraints.name}: " + "; ".join((mismatched + url_mismatch)[:25]))
        elif pins:
            report.ok("pins", f"every installed package matches {constraints.name} ({len(pins)} pins)")
        modules = []
        for group, names in GROUP_MODULES.items():
            if groups.get(group, (False, ""))[0]:
                modules.extend(name for name in names if name not in modules)
        specs, error = find_specs(venv, modules)
        if specs is None:
            report.need_repair("modules", f"cannot probe modules: {error}")
        else:
            absent = [name for name, present in specs.items() if not present]
            if absent:
                report.need_repair("modules", "not importable (find_spec): " + ", ".join(absent))
            else:
                report.ok("modules", f"{len(specs)} top-level modules resolvable (find_spec, not imported)")
        result["module_specs"] = specs

    # 6. model files.
    model_files = list(SERVER_MODEL_FILES)
    if groups["avatar_prep"][0]:
        model_files += AVATAR_PREP_MODEL_FILES
    missing_files = [rel for rel in model_files
                     if not ((repo / rel).is_file() and (repo / rel).stat().st_size > 0)]
    if missing_files:
        detail = "missing: " + ", ".join(missing_files)
        if args.skip_weights:
            report.warn("weights", detail + " (--skip-weights)")
        else:
            report.need_repair("weights", detail + " (run download_weights.sh via the installer)")
    else:
        report.ok("weights", f"{len(model_files)} model files present (incl. models/taesd)")
    if groups["kokoro"][0]:
        kokoro_dir = hf_hub_cache() / "models--hexgrad--Kokoro-82M" / "snapshots"
        voices = list(kokoro_dir.glob("*/voices/*.pt")) if kokoro_dir.is_dir() else []
        if list(kokoro_dir.glob("*/kokoro-v1_0.pth")) if kokoro_dir.is_dir() else []:
            report.ok("kokoro_cache", f"Kokoro-82M cached ({len(voices)} voices) in {kokoro_dir.parent}")
        else:
            report.warn("kokoro_cache", f"Kokoro-82M not pre-cached in {hf_hub_cache()} (downloads on first TTS use)")

    # 7. native VP8.
    native_dir = Path(args.native_vp8_dir) if args.native_vp8_dir else repo / ".runtime" / "native_vp8"
    enabled, source = groups["native_vp8"]
    if enabled:
        ok, detail = validate_native_vp8(repo, native_dir)
        result["native_vp8"] = {"directory": str(native_dir), "verified": ok, "detail": detail}
        if ok:
            report.ok("native_vp8", detail)
        elif source == "auto":
            report.warn("native_vp8", f"not provisioned ({detail}); optional, WEBRTC_VP8_ENCODER stays pyav")
        else:
            report.need_repair("native_vp8", detail)

    # 8. chin tools.
    if groups["chin_tools"][0]:
        stamp_chin = stamp_groups.get("chin_tools") if isinstance(stamp_groups.get("chin_tools"), dict) else {}
        chin_venv = Path(args.chin_venv or stamp_chin.get("venv") or default_chin_venv(repo))
        result["chin_tools"] = check_chin_venv(chin_venv, repo / DEFAULT_CHIN_PINS_FILE, report)

    result.update({"checks": report.checks, "warnings": report.warnings, "needs_clean": report.clean,
                   "needs_repair": report.repair, "exit_code": report.exit_code,
                   "verdict": {EXIT_OK: "ok", EXIT_CLEAN: "clean_install_required",
                               EXIT_REPAIR: "repairable_in_place"}[report.exit_code]})
    emit(result, args)
    for entry in report.checks:
        mark = {"ok": "ok  ", "warn": "WARN", "clean": "FAIL", "repair": "FAIL"}[entry["status"]]
        print(f"[install-check] {mark} {entry['check']}: {entry['detail']}", file=sys.stderr)
    print(f"[install-check] verdict: {result['verdict']} (exit {report.exit_code})", file=sys.stderr)
    return report.exit_code


def default_chin_venv(repo: Path) -> str:
    workspace = os.environ.get("WORKSPACE") or (str(repo.parent) if repo.parent != repo else "/workspace")
    return os.path.join(workspace, ".venvs", "musetalk_chin_tools")


def emit(payload: dict, args) -> None:
    if getattr(args, "report", None):
        write_json_atomic(Path(args.report), payload)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))


# ----------------------------------------------------------------------------- stamp
def load_json_file(path):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def cmd_stamp(args) -> int:
    repo = Path(args.repo_root).resolve()
    venv = Path(args.venv)
    facts = venv_facts(venv)
    constraints = repo / "requirements" / f"constraints-{args.matrix}.txt"
    groups = json.loads(args.groups_json) if args.groups_json else {}
    native = load_json_file(args.native_vp8_json)
    if native is not None:
        groups["native_vp8"] = dict(native, enabled=bool(native.get("enabled", True)))
    chin = None
    if groups.get("chin_tools"):
        chin_venv = Path(args.chin_venv or default_chin_venv(repo))
        chin_dists = installed_dists(site_packages(chin_venv))
        chin = {"enabled": True, "venv": str(chin_venv), "python": str(venv_python(chin_venv)),
                "versions": {name: chin_dists.get(name, {}).get("version")
                             for name in ("mediapipe", "numpy", "opencv-contrib-python", "protobuf")}}
        groups["chin_tools"] = chin
    smoke = load_json_file(args.smoke_json)
    selftest = load_json_file(args.selftest_json)
    in_files = {group: sha256_file(repo / rel) for group, rel in GROUP_FILES.items()}
    in_files["chin_tools"] = sha256_file(repo / DEFAULT_CHIN_PINS_FILE)
    exports = {}
    if chin:
        # Consumed by the resolver (only-if-unset) so live chin work never borrows another project's venv.
        exports["MUSETALK_CHIN_TRACKER_PYTHON"] = chin["python"]
    if native and native.get("verified"):
        exports["WEBRTC_NATIVE_VP8_DIR"] = native.get("directory")
    payload = {
        "schema": STATE_SCHEMA,
        "created_utc": utc_now(),
        "installer": "scripts/install_musetalk.sh",
        "repo_root": str(repo),
        "matrix": args.matrix,
        "python": {"bin": args.python_bin, "version": facts.get("python_version"), "machine": facts.get("machine")},
        "venv": str(venv),
        "constraints_file": str(constraints.relative_to(repo)) if constraints.is_file() else str(constraints),
        "constraints_sha256": sha256_file(constraints),
        "server_in_sha256": in_files["server"],
        "group_in_sha256": in_files,
        "groups": groups,
        "versions": dict(facts.get("versions") or {}, torch_cuda_tag=facts.get("torch_cuda_tag")),
        "encoders": (smoke or {}).get("encoders"),
        "torch_arch_flags": (smoke or {}).get("torch_arch_flags"),
        "import_smoke": {"ok": (smoke or {}).get("ok"), "failed": (smoke or {}).get("failed"),
                         "deferred": (smoke or {}).get("deferred")} if smoke else None,
        "gpu_selftest": {"path": args.selftest_path, "cuda_ok": (selftest or {}).get("cuda_ok"),
                         "taesd_compile_ok": ((selftest or {}).get("taesd") or {}).get("compile_ok")}
        if selftest else None,
        "exports": exports,
    }
    write_json_atomic(Path(args.out), payload)
    print(json.dumps({"stamp": str(args.out), "matrix": args.matrix, "exports": exports}))
    return 0


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect-matrix", help="print the install matrix for the visible GPU")
    p.add_argument("--matrix", default="auto", choices=("auto",) + MATRICES)
    p.add_argument("--field", default="", help="print only this field (e.g. matrix)")
    p.add_argument("--shell", action="store_true", help="print shlex-quoted DETECTED_* assignments for bash eval")

    p = sub.add_parser("venv-facts", help="print JSON facts about a venv (no imports)")
    p.add_argument("--venv", required=True)
    p.add_argument("--field", default="")

    p = sub.add_parser("check", help="read-only install check (exit 0 / 10 clean / 11 repair)")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--venv", required=True)
    p.add_argument("--matrix", default="auto", choices=("auto",) + MATRICES)
    for flag in ("kokoro", "native-vp8", "legacy-int8", "avatar-prep", "chin-tools"):
        p.add_argument(f"--{flag}", default="", choices=("", "0", "1"), help="1/0 = requested on/off; empty = stamp/default")
    p.add_argument("--native-vp8-dir", default="")
    p.add_argument("--chin-venv", default="")
    p.add_argument("--state-file", default="")
    p.add_argument("--skip-weights", action="store_true")
    p.add_argument("--report", default="", help="also write the JSON report here")
    p.add_argument("--json", action="store_true", help="print the JSON report to stdout")

    p = sub.add_parser("stamp", help="write .runtime/install_state.json")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--venv", required=True)
    p.add_argument("--matrix", required=True, choices=MATRICES)
    p.add_argument("--python-bin", default="python3.10")
    p.add_argument("--groups-json", default="{}")
    p.add_argument("--native-vp8-json", default="")
    p.add_argument("--chin-venv", default="")
    p.add_argument("--smoke-json", default="")
    p.add_argument("--selftest-json", default="")
    p.add_argument("--selftest-path", default="")
    p.add_argument("--out", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect-matrix":
        info = detect_matrix(args.matrix)
        if args.shell:
            gpu = info.get("gpu")
            desc = f"{gpu['name']} (cc {gpu['compute_capability']})" if gpu else "none"
            for key, value in (("DETECTED_MATRIX", info["matrix"]), ("DETECTED_REASON", info["reason"]),
                               ("DETECTED_GPU_VISIBLE", "1" if info["gpu_visible"] else "0"),
                               ("DETECTED_GPU_DESC", desc), ("DETECTED_GPU_ERROR", info.get("gpu_error") or "")):
                print(f"{key}={shlex.quote(str(value))}")
            return 0
        print(info.get(args.field, "") if args.field else json.dumps(info))
        return 0
    if args.command == "venv-facts":
        facts = venv_facts(Path(args.venv))
        value = facts.get(args.field) if args.field else facts
        print(value if isinstance(value, str) else ("" if value is None else json.dumps(value)))
        return 0
    if args.command == "check":
        return cmd_check(args)
    if args.command == "stamp":
        return cmd_stamp(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
