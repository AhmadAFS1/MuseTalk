#!/usr/bin/env python3
"""Build only the four small sdists once, using verified inputs on matching Linux.

Never run this on macOS. Every output directory is a new experiment-owned claim;
an interrupted build is evidence to inspect, not permission to overwrite it.
"""
import argparse
import copy
from email.parser import BytesParser
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import zipfile

from dependency_artifacts import digest, identity, normalized, offline_lock, require, validate_manifest
from fetch_dependency_wheels import read_trusted, sha256_file


def checked_wheel(path, expected_name, expected_version):
    require(path.is_file() and not path.is_symlink() and path.suffix == ".whl", "Not a regular wheel")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(all(not n.startswith("/") and ".." not in Path(n).parts for n in names), "Unsafe wheel member")
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "Ambiguous wheel metadata")
        info = BytesParser().parsebytes(archive.read(metadata[0]))
    require(normalized(info["Name"]) == expected_name and info["Version"] == expected_version, "Wheel name/version differs")
    parts = path.name[:-4].rsplit("-", 3)
    require(len(parts) == 4, "Invalid wheel tags")
    return {"python": parts[1], "abi": parts[2], "platform": parts[3]}


def run_step(command, name, root, environment, timeout=300):
    start = time.monotonic()
    log = root / (name + ".log")
    with log.open("xb") as out:
        result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=environment, timeout=timeout)
    receipt = {"name": name, "seconds": time.monotonic() - start, "exit_status": result.returncode,
               "log": log.name}
    with (root / "steps.jsonl").open("a") as out:
        out.write(json.dumps(receipt) + "\n")
    require(result.returncode == 0, "Linux build step failed: " + name)
    return receipt


def build(manifest, source_cache, output, python):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Matching Linux only; no macOS package tests")
    source_cache = Path(source_cache).resolve(strict=True)
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Build output already claimed")
    require(shutil.disk_usage(output.parent).free >= 25 * 1024**3, "Insufficient Linux build disk")
    facts = json.loads(subprocess.check_output([python, "-c", "import json,platform,sys;print(json.dumps({'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'glibc':platform.libc_ver()[1]}))"], text=True))
    require(facts["python"].startswith("3.10.") and facts["system"] == "Linux" and facts["machine"] == "x86_64", "Wrong builder interpreter")
    output.mkdir(mode=0o700)
    wheels = output / "wheels"
    wheels.mkdir()
    m = copy.deepcopy(manifest)
    start = time.monotonic()
    # Hash every source once before any executable build code is run.
    for a in m["artifacts"]:
        path = source_cache / a["filename"]
        require(path.is_file() and not path.is_symlink() and sha256_file(path) == a["sha256"], "Input absent or hash mismatch: " + a["name"])
        a["size_bytes"] = path.stat().st_size
        if a["format"] == "wheel":
            checked_wheel(path, a["name"], a["version"])
            os.link(path, wheels / path.name)  # Same owned filesystem; never alter original bytes.
    m["artifact_id"] = digest(identity(m))
    validate_manifest(m)
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "PIP_", "UV_", "PYTHON"))}
    environment.update({"PIP_CONFIG_FILE": "/dev/null", "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                        "PIP_CACHE_DIR": str(output / "build-cache"), "PYTHONNOUSERSITE": "1",
                        # TensorRT 10.3's upstream setup.py otherwise invokes pip
                        # inside bdist_wheel. Its supported switch declares the
                        # same pinned libs/bindings as normal dependencies instead.
                        "NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP": "1",
                        "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": "",
                        "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "no_proxy": ""})
    builder = output / "builder"
    run_step([python, "-m", "venv", str(builder)], "create-builder", output, environment)
    py = str(builder / "bin/python")
    bootstrap = output / "bootstrap.lock"
    bootstrap.write_text(offline_lock(m, "bootstrap"))
    run_step([py, "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "--require-hashes",
              "--only-binary=:all:", "-r", str(bootstrap)], "bootstrap-builder", output, environment)
    numpy = next(a for a in m["artifacts"] if a["name"] == "numpy")
    # Chumpy's known setup.py imports NumPy/pip; match the existing controlled build.
    run_step([py, "-m", "pip", "install", "--no-index", "--no-deps", str(wheels / numpy["filename"])],
             "build-prerequisite-numpy", output, environment)
    require({a["name"] for a in m["artifacts"] if a["format"] != "wheel"} ==
            {"antlr4-python3-runtime", "tensorrt-cu12", "docopt", "chumpy"}, "Unreviewed source-build inventory")
    for a in m["artifacts"]:
        if a["format"] == "wheel":
            continue
        original = {k: a[k] for k in ("filename", "sha256", "source")}
        destination = output / ("built-" + a["name"])
        destination.mkdir()
        receipt = run_step([py, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
                            "--wheel-dir", str(destination), str(source_cache / a["filename"])],
                           "build-" + a["name"], output, environment)
        results = list(destination.glob("*.whl"))
        require(len(results) == 1, "Build did not produce exactly one wheel")
        result = results[0]
        a["wheel_tags"] = checked_wheel(result, a["name"], a["version"])
        a.update(filename=result.name, sha256=sha256_file(result), size_bytes=result.stat().st_size,
                 format="wheel", source="build:" + original["sha256"], hash_provenance="verified-local-Linux-build")
        a["build_provenance"] = {**facts, "input_sha256": original["sha256"], "input_filename": original["filename"],
                                 "input_source": original["source"], "pip": "26.2.1", "setuptools": "60.2.0",
                                 "wheel": "0.48.0", "numpy": "1.23.5", "no_build_isolation": True,
                                 "build_seconds": receipt["seconds"]}
        os.link(result, wheels / result.name)
    m["artifact_id"] = digest(identity(m))
    m["status"] = "complete-local-wheels; Linux closure/install/native validation pending"
    validate_manifest(m)
    (output / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    for group in ("bootstrap", "server", "server+kokoro", "full"):
        (output / (group + ".lock")).write_text(offline_lock(m, group))
    receipt = {"state": "wheels-built", "seconds": time.monotonic() - start, "artifact_id": m["artifact_id"],
               "manifest_sha256": sha256_file(output / "manifest.json"), "packages": len(m["artifacts"]),
               "native_compatibility_proven": False, "network_isolation": "no-index+local-inputs+closed-proxy; not a kernel network namespace"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", default="python3.10")
    a = p.parse_args()
    print(json.dumps(build(read_trusted(a.manifest, a.manifest_sha256), a.source_cache, a.output, a.python)), flush=True)


if __name__ == "__main__":
    main()
