#!/usr/bin/env python3
"""Linux-only installer microbenchmarks; not a fresh-host/application acceptance claim.

Run in a new owned trial directory. Local wheel acquisition is EXCLUDED and
reported as such. Full-delivery trials must start an outer timer before fetching
artifacts/tools and attach this receipt rather than relabel this microbenchmark.
"""
import argparse
import csv
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import time
import zipfile

from dependency_artifacts import GROUPS, normalized, offline_lock, require, selected
from fetch_dependency_wheels import read_trusted, sha256_file

IMPORTS = {"server": ["torch", "torchvision", "torchaudio", "tensorrt", "onnx", "numpy",
                         "diffusers", "transformers", "cv2", "aiortc", "av", "cffi", "fastapi", "boto3",
                         "librosa", "soundfile", "numba", "imageio", "ffmpeg", "omegaconf"],
           "kokoro": ["kokoro", "misaki", "spacy", "en_core_web_sm", "espeakng_loader"],
           "avatar-prep": ["mmengine", "mmcv", "mmcv._ext", "mmdet", "mmpose"]}

# Torch-TensorRT 2.5.0 initializes CUDA at import. The canonical installer defers
# it from the CUDA-hidden smoke too. Test it explicitly WITH a visible GPU;
# neither omit the check nor classify a hidden-device import as binary failure.
GPU_PROBE = """import json,torch,torch_tensorrt,tensorrt
assert torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1'
assert torch_tensorrt.__version__=='2.5.0' and tensorrt.__version__=='10.3.0'
assert torch.cuda.get_device_capability()==(8,6)
value=torch.arange(8,device='cuda',dtype=torch.float32).square().sum().item()
assert value==140.0
torch.cuda.synchronize()
print(json.dumps({'cuda_ok':True,'torch':torch.__version__,'cuda':torch.version.cuda,'torch_tensorrt':torch_tensorrt.__version__,'tensorrt':tensorrt.__version__,'probe_value':value}))
"""


def source_commands(py, repo, group):
    common = ["-c", str(repo/"requirements/constraints-cu121.txt"), "--extra-index-url", "https://download.pytorch.org/whl/cu121",
              "--extra-index-url", "https://pypi.nvidia.com"]
    pip = [str(py), "-m", "pip", "install", "--disable-pip-version-check"]
    if group == "server":return [("source-server", pip+["-r", str(repo/"requirements/server.in"), *common])]
    if group == "server+kokoro":return [("source-kokoro", pip+["-r", str(repo/"requirements/kokoro.in"), *common])]
    return [("source-avatar-prerequisites", pip+["setuptools<81", "ninja", "psutil", *common]),
            ("source-mmcv", pip+["--no-deps", str(repo/"third_party_wheels/mmcv/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl")]),
            ("source-chumpy", pip+["--no-build-isolation", "chumpy", "-c", str(repo/"requirements/constraints-cu121.txt")]),
            ("source-avatar", pip+["mmengine", "mmdet", "mmpose", *common])]


def verify_hashless_installed(a, reference_wheels, venv):
    require(a["name"] in {"jinja2", "markupsafe"}, "Unreviewed hashless source report")
    wheel = Path(reference_wheels)/a["filename"]
    require(sha256_file(wheel) == a["sha256"], "Reference wheel hash differs")
    site = Path(venv)/"lib/python3.10/site-packages"
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        records = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
        require(len(records) == 1, "Reference wheel has ambiguous RECORD")
        for name, h, size in csv.reader(io.StringIO(archive.read(records[0]).decode())):
            if not h:continue
            require(h.startswith("sha256="), "Unreviewed RECORD hash algorithm")
            target = site/name
            require(target.resolve().is_relative_to(site.resolve()) and target.is_file(), "Installed reference file absent/unsafe")
            raw = target.read_bytes()
            observed = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            require(observed == h[7:] and len(raw) == int(size), "Installed file differs from trusted reference wheel")
            checked += 1
    require(checked > 0, "Empty reference RECORD")
    return checked


def verify_source_reports(m, reports, reference_wheels, venv, group="full"):
    expected = {a["name"]: a for a in m["artifacts"]}
    verified, files, seen = 0, {}, set()
    for path in reports:
        for row in json.loads(path.read_text())["install"]:
            name = normalized(row["metadata"]["name"])
            require(name in expected and row["metadata"]["version"] == expected[name]["version"], "Source selected a different package/version")
            a = expected[name]
            seen.add(name)
            observed = row.get("download_info", {}).get("archive_info", {}).get("hashes", {}).get("sha256")
            if observed:
                sha = a["build_provenance"]["input_sha256"] if a["source"].startswith("build:") else a["sha256"]
                require(observed == sha, "Source selected a different artifact hash: "+name)
                verified += 1
            else:files[name] = verify_hashless_installed(a, reference_wheels, venv)
    wanted = {a["name"] for a in selected(m, group) if a["introduced_by"] != "bootstrap"}
    require(seen == wanted, "Source reports do not cover the complete selected closure")
    return {"source_report_artifact_hashes_verified": verified, "hashless_report_installed_files_verified": files}


def trial_path(root, name):
    root = Path(root)
    require(root.is_absolute() and root.parent == Path("/workspace")
            and re.fullmatch(r"\.dependency-optimization-[a-z0-9-]+", root.name), "Explicit experiment root required")
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) is not None, "Invalid trial name")
    require(not root.is_symlink(), "Experiment symlink rejected")
    root.mkdir(mode=0o700, exist_ok=True)
    require(root.stat().st_uid == os.getuid(), "Foreign experiment directory")
    target = root / name
    require(not target.exists() and not target.is_symlink(), "Trial already claimed; inspect existing receipt")
    return target


def command(command, name, root, env, timeout=900):
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (root / (name + ".log")).open("xb") as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            status = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            status = 124
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = {"step": name, "seconds": time.monotonic() - start, "exit_status": status,
           "child_cpu_seconds": (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
           "child_output_blocks": after.ru_oublock - before.ru_oublock,
           "child_maxrss_kib_cumulative_high_water": after.ru_maxrss, "log": name + ".log"}
    with (root / "events.jsonl").open("a") as out:
        out.write(json.dumps(row) + "\n")
    return row


def install_command(installer, py, uv, cache, wheels, lock, compile_bytecode, link_mode):
    common = ["--no-index", "--find-links", str(wheels), "--require-hashes", "--only-binary=:all:", "-r", str(lock)]
    if installer == "pip":
        return [str(py), "-m", "pip", "install", "--disable-pip-version-check", "--cache-dir", str(cache),
                *( ["--no-compile"] if not compile_bytecode else []), *common]
    require(installer == "uv" and uv, "Pinned uv required")
    return [str(uv), "--no-config", "pip", "install", "--python", str(py), "--offline", "--cache-dir", str(cache),
            "--link-mode", link_mode, *( ["--compile-bytecode"] if compile_bytecode else []), *common]


def run(args):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux only: no macOS install tests")
    start = time.monotonic()
    root = trial_path(args.experiment_root, args.trial)
    require(shutil.disk_usage(root.parent).free >= 20 * 1024**3, "Insufficient trial disk")
    root.mkdir(mode=0o700)
    (root / "owned-trial.json").write_text(json.dumps({"uid": os.getuid(), "trial": args.trial, "created_epoch": time.time()}))
    result = {"trial": args.trial, "installer": args.installer, "group": args.group,
              "compile_bytecode": args.compile_bytecode, "link_mode": args.link_mode if args.installer == "uv" else None,
              "cache_state": {"venv": "fresh", "installer_cache": "empty", "wheel_source": "local-verified",
                              "OS_page_cache": "unknown; not dropped", "CDN": "not used"},
              "remote_acquisition_included": False, "application_acceptance": "NOT_TESTED_BY_THIS_MICROBENCHMARK",
              "state": "running", "steps": []}
    result["timer_start_monotonic"] = start
    source_mode = getattr(args, "source_pip", False)
    result["source_pip"] = source_mode
    def step(cmd, label, env, required=True):
        receipt = command(cmd, label, root, env)
        result["steps"].append(receipt)
        require(not required or receipt["exit_status"] == 0, "Trial step failed: " + label)
        return receipt
    try:
        m = read_trusted(args.manifest, args.manifest_sha256)
        result["artifact_id"] = m["artifact_id"]
        wheels = args.wheels.resolve(strict=True)
        for a in selected(m, "bootstrap" if source_mode else args.group):
            path = wheels / a["filename"]
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == a["size_bytes"]
                    and sha256_file(path) == a["sha256"], "Wheel preflight failed: " + a["name"])
        result["verification_seconds"] = time.monotonic() - start
        facts = json.loads(subprocess.check_output([args.python, "-c", "import json,platform;print(json.dumps({'python':platform.python_version(),'glibc':platform.libc_ver()[1]}))"], text=True))
        require(facts["python"].startswith("3.10."), "Wrong Python interpreter")
        result["platform"] = {**facts, "os": "Linux", "machine": "x86_64"}
        if args.installer == "uv":
            require(args.uv and args.uv_sha256 and sha256_file(args.uv) == args.uv_sha256, "Untrusted uv binary")
            result["uv_version"] = subprocess.check_output([str(args.uv), "--version"], text=True, timeout=10).strip()
            result["uv_binary_sha256"] = args.uv_sha256
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "PIP_", "UV_", "PYTHON", "VIRTUAL_ENV"))}
        env.update({"PIP_CONFIG_FILE": "/dev/null", "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PYTHONNOUSERSITE": "1", "UV_PYTHON_DOWNLOADS": "never", "UV_NO_PROGRESS": "1"})
        if source_mode:
            require(args.installer == "pip" and args.compile_bytecode, "Original-source control uses pip with normal compiled bytecode")
            repo = Path(args.repo_root).resolve(strict=True)
            for name, sha in m["requirement_inputs"].items():
                require(sha256_file(repo/"requirements"/name) == sha, "Source requirements fingerprint changed")
            env.pop("PIP_NO_INDEX")
        venv = root / "venv"
        cache = root / "package-cache"
        cache.mkdir()
        env["PIP_CACHE_DIR"] = str(cache)
        step([args.python, "-m", "venv", str(venv)], "venv", env)
        py = venv / "bin/python"
        bootstrap = root / "bootstrap.lock"
        bootstrap.write_text(offline_lock(m, "bootstrap"))
        step(install_command("pip", py, None, cache, wheels, bootstrap, True, "copy"), "bootstrap", env)
        order = ["server", "server+kokoro", "full"]
        reports = []
        for group in order[:order.index(args.group) + 1]:
            if source_mode:
                for name, cmd in source_commands(py, repo, group):
                    report = root/(name+".report.json")
                    reports.append(report)
                    step(cmd+["--report", str(report)], name, env)
            else:
                lock = root / (group + ".lock")
                lock.write_text(offline_lock(m, group))
                step(install_command(args.installer, py, args.uv, cache, wheels, lock, args.compile_bytecode, args.link_mode),
                     "install-" + group, env)
        inventory = subprocess.check_output([str(py), "-c", "import importlib.metadata as m,json,re;print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions()}))"], text=True, env=env)
        expected = {a["name"]: a["version"] for a in selected(m, args.group)}
        actual = json.loads(inventory)
        result["inventory_diff"] = {k: {"expected": expected.get(k), "actual": actual.get(k)}
                                    for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}
        require(not result["inventory_diff"], "Installed inventory differs")
        if source_mode:
            result["source_artifact_validation"] = verify_source_reports(m, reports, args.reference_wheels, venv, args.group)
        result["package_consistency"] = step([str(py), "-m", "pip", "check"], "pip-check", env, required=False)
        result["packages_ready_seconds"] = time.monotonic() - start
        modules = [module for g in GROUPS[args.group] for module in IMPORTS.get(g, [])]
        code = "import importlib,json,time;out=[]\nfor name in " + repr(modules) + ":\n start=time.monotonic();importlib.import_module(name);out.append({'module':name,'seconds':time.monotonic()-start})\nprint(json.dumps(out))"
        cpu_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
        step([str(py), "-c", code], "first-imports", cpu_env)
        step([str(py), "-c", code], "second-process-imports", cpu_env)
        step([str(py), "-c", GPU_PROBE], "gpu-imports-and-execution", env)
        result["validated_ready_seconds"] = time.monotonic() - start
        result["state"] = "microbenchmark-passed"
    except BaseException as error:
        result.update(state="failed", error_type=type(error).__name__,
                      detail=str(error) if type(error) is ValueError else "Inspect protected per-step log")
    finally:
        result["outer_seconds"] = time.monotonic() - start
        result["free_disk_bytes_after"] = shutil.disk_usage(root).free
        (root / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--wheels", type=Path, required=True)
    p.add_argument("--group", choices=("server", "server+kokoro", "full"), default="full")
    p.add_argument("--installer", choices=("pip", "uv"), required=True)
    p.add_argument("--compile-bytecode", action="store_true")
    p.add_argument("--link-mode", choices=("copy", "hardlink", "clone"), default="copy")
    p.add_argument("--python", default="python3.10")
    p.add_argument("--uv", type=Path)
    p.add_argument("--uv-sha256")
    p.add_argument("--source-pip", action="store_true")
    p.add_argument("--repo-root", type=Path, default=Path("/workspace/MuseTalk"))
    p.add_argument("--reference-wheels", type=Path)
    args = p.parse_args()
    result = run(args)
    print(json.dumps({"result": result}), flush=True)
    return 0 if result["state"] == "microbenchmark-passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
