#!/usr/bin/env python3
"""Fail-closed, GPU-less release assembly and immutable-runtime verification.

This is NOT a native engine builder or a quality/performance benchmark. A real
validated release manifest must be supplied; no sample is deployable by default.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

SCHEMA = "musetalk_docker_release_v1"
MANIFEST_NAME = ".musetalk_trt_artifact_manifest.json"
CHECKSUM_NAME = ".musetalk_trt_artifact_SHA256SUMS"
STAMP_NAME = ".musetalk_trt_artifact_restored.json"
SHA = re.compile(r"[0-9a-f]{64}\Z")
APT = re.compile(r"[a-z0-9][a-z0-9+.-]*(?::amd64)?=[A-Za-z0-9.+:~_-]+\Z")
# Exact linux/amd64 CUDA12.1/cuDNN8 Ubuntu22.04 pair exercised by the
# dependency-only CI experiment. An arbitrary pinned digest is not a proven pair.
CUDA_DEVEL_BASE = "nvidia/cuda@sha256:cc55d151af1e8e083f3210af753a5cfbcbc5455421531eb0459887026bb4699f"
CUDA_RUNTIME_BASE = "nvidia/cuda@sha256:810756cab1c28ce693499a5c2ebb66f6d10a61d026998c8606bad449643a4c49"
REQUIRED_APT = {"python3", "python3.10", "python3.10-venv", "python3.10-dev", "ffmpeg", "coturn",
                "curl", "git", "build-essential", "ca-certificates", "libgl1", "libglib2.0-0",
                "libsm6", "libxext6", "libxrender1", "tini", "util-linux"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
)
EXTERNAL_MODEL_ALLOWLIST = {
    "models/syncnet/latentsync_syncnet.pt",
    "models/face-parse-bisent/79999_iter.pth",
    "models/face-parse-bisent/resnet18-5c106cde.pth",
    "models/auxiliary/s3fd-619a316812.pth",
    "models/face_detection/s3fd.pth",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(value):
    require(isinstance(value, str) and bool(value), "Empty/non-string release path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == value,
            "Unsafe/noncanonical release path")
    return value


def checked_path(root, value):
    path = Path(root) / relative(value)
    require(path.resolve().is_relative_to(Path(root).resolve()), "Release path escapes root")
    require(not path.is_symlink(), "Release symlinks must be materialized before packaging")
    return path


def check_file(root, name, entry):
    path = checked_path(root, name)
    require(SHA.fullmatch(str(entry.get("sha256", ""))), f"Missing SHA-256: {name}")
    require(path.is_file(), f"Release file missing: {name}")
    require(path.stat().st_size == entry.get("size_bytes"), f"Release size mismatch: {name}")
    require(sha256(path) == entry["sha256"], f"Release SHA-256 mismatch: {name}")
    return path


def scan_text(path):
    require(path.stat().st_size < 10 * 1024 * 1024, "Unexpected large source/notice file")
    data = path.read_bytes()
    require(not any(pattern.search(data) for pattern in SECRET_PATTERNS),
            "Possible credential in release input (value suppressed)")


def external_models(m):
    external = m.get("external_model_files", {})
    require(isinstance(external, dict) and set(external) <= EXTERNAL_MODEL_ALLOWLIST,
            "External model path is outside the private runtime allowlist")
    require(not set(external).intersection(m["model_files"]), "A model cannot be both public and private")
    if external:
        require(m.get("redistribution_scope") == "baked_model_files_only",
                "Private model manifest must limit redistribution approval to baked files")
        require(m.get("private_model_usage_rights") in {"unresolved", "reviewed"},
                "Private fetch is not a license clearance; record usage-rights status")
    for name, entry in external.items():
        require(isinstance(entry, dict) and SHA.fullmatch(str(entry.get("sha256", "")))
                and type(entry.get("size_bytes")) is int and 0 < entry["size_bytes"] <= 4 * 1024**3,
                "Private model needs a real SHA-256 and bounded positive size")
        require(entry.get("public_redistribution") is False and entry.get("private_delivery_authorized") is True,
                "Private model must be explicitly excluded from publication and authorized for runtime delivery")
        source = entry.get("source", {})
        require(isinstance(source, dict) and set(source) <= {"type", "bucket", "key", "region", "expected_owner", "version_id"},
                "Private model source contains unsupported fields/URL")
        require(source.get("type") == "s3" and re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", source.get("bucket", "")),
                "Private model requires a named S3 bucket, not a URL/access point")
        require(re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", source.get("region", ""))
                and re.fullmatch(r"\d{12}", source.get("expected_owner", "")), "S3 region and expected account owner required")
        key = relative(source.get("key", ""))
        require(re.fullmatch(r"[A-Za-z0-9._/-]+", key) and key.endswith("/sha256/" + entry["sha256"] + "/" + Path(name).name),
                "Private S3 key must be content-addressed and end in the exact model filename")
        if "version_id" in source:
            require(isinstance(source["version_id"], str) and 0 < len(source["version_id"]) <= 1024,
                    "Invalid S3 version ID")
    return external


def selected_runtime_base(m):
    if "cuda_runtime_base" not in m:
        # Existing manifests keep the development base, without inheriting
        # the dependency-install layer or duplicating payloads in a new layer.
        return m["cuda_base"]
    require(m["cuda_base"] == CUDA_DEVEL_BASE and m["cuda_runtime_base"] == CUDA_RUNTIME_BASE,
            "Runtime base is not the measured matching CUDA/cuDNN pair")
    return m["cuda_runtime_base"]


def verify_build_bases(m, build_base, runtime_base):
    require(build_base == m["cuda_base"] and runtime_base == selected_runtime_base(m),
            "Build/runtime base differs from release manifest")


def load_manifest(path):
    scan_text(Path(path))
    m = json.loads(Path(path).read_text())
    require(m.get("schema") == SCHEMA, "Release schema mismatch")
    require(m.get("status") in {"validated", "candidate"}, "Release status must be validated or candidate")
    if m["status"] == "candidate":
        require(m.get("promotion_eligible") is False and bool(m.get("candidate_reason")),
                "Candidate must be explicitly nonpromotable with a recorded reason")
    require(re.fullmatch(r"[0-9a-f]{40}", m.get("source_revision", "")), "Full source commit required")
    require(re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", m.get("cuda_base", "")), "Pinned CUDA base required")
    selected_runtime_base(m)
    require(m.get("platform") == "linux/amd64", "Release must target linux/amd64")
    require(m.get("matrix") == "cu121", "Only validated cu121 is supported")
    require(m.get("bundle_name") == "rtx3090-r5-srcg50-int8", "Native 3090 descriptor required")
    require(m.get("redistribution_reviewed") is True, "Public redistribution review required")
    require(m.get("avatar_prep") is True, "This full API image requires avatar-prep dependencies")
    require(type(m.get("kokoro")) is bool, "Declare local TTS capability explicitly")
    require(m.get("vp8_encoder") in {"native", "pyav"}, "Declare validated VP8 encoder")
    for collection in ("source_files", "model_files", "notices", "evidence"):
        require(isinstance(m.get(collection), dict) and bool(m[collection]), f"Missing {collection}")
        for name in m[collection]:
            relative(name)
    require(set(m.get("archives", {})) == {"weights.tar.gz", "native.tar.gz"}, "Exactly two archives required")
    packages = m.get("apt_packages", [])
    require(isinstance(packages, list) and all(isinstance(p, str) and APT.fullmatch(p) for p in packages),
            "Exact apt name=version pins required")
    require(REQUIRED_APT <= {p.split("=", 1)[0].split(":", 1)[0] for p in packages}, "Required OS packages missing")
    require(len(packages) == len(set(p.split("=", 1)[0] for p in packages)), "Duplicate apt package")
    require(SHA.fullmatch(m.get("bundle_manifest_sha256", "")), "Bundle manifest digest required")
    for name, entry in m["model_files"].items():
        require(name.startswith("models/") and len(PurePosixPath(name).parts) >= 3, "Only model assets may be baked")
        require(entry.get("public_redistribution") is True and entry.get("license_id"),
                f"Missing model redistribution/license decision: {name}")
    external_models(m)
    require(all(n.startswith("licenses/") for n in m["notices"]), "Notices must be under licenses/")
    if m["kokoro"]:
        kokoro = [n for n in m["model_files"] if n.startswith("models/hf-cache/hub/models--hexgrad--Kokoro-82M/")]
        require(any("/snapshots/" in n and n.endswith(".pth") for n in kokoro)
                and any("/voices/" in n and n.endswith(".pt") for n in kokoro)
                and any(n.endswith("/refs/main") for n in kokoro), "Kokoro needs its pinned, materialized HF cache")
    return m


def descriptor(root, m):
    path = Path(root) / "configs/trt_bundles" / (m["bundle_name"] + ".json")
    d = json.loads(path.read_text())
    require(d.get("name") == m["bundle_name"], "Descriptor name mismatch")
    require(d.get("sha256") == m["archives"]["native.tar.gz"]["sha256"], "Descriptor/archive mismatch")
    require(d.get("size_bytes") == m["archives"]["native.tar.gz"]["size_bytes"], "Descriptor/archive size mismatch")
    require(d.get("host", {}).get("engine_key") == "sm86-nvidia-geforce-rtx-3090-trt10.3.0", "Native sm86 key required")
    require(d["host"].get("compute_capability") == "8.6", "Native sm86 capability required")
    sidecar = relative(d.get("sidecar_dir", ""))
    require(sidecar.startswith(".runtime/trt_artifacts/"), "Unexpected sidecar location")
    require(d.get("engines", {}).get("unet_stagewise", {}).get("batch") == 16, "UNet bs16 required")
    require(d.get("engines", {}).get("taesd_trt", {}).get("batch") == 8, "TAESD bs8 required")
    recipe = (Path(root) / "configs/recipes/r5.env").read_text()
    require(m["bundle_name"] in recipe, "r5 recipe does not select native 3090 bundle")
    return d


def verify_evidence(assets, m):
    for name, entry in m["evidence"].items():
        check_file(assets, name, entry)
        scan_text(Path(assets) / name)
    quality_name = m.get("quality_decision_file", "")
    aggregate_name = m.get("aggregate_acceptance_file", "")
    require(quality_name in m["evidence"] and aggregate_name in m["evidence"], "Acceptance evidence required")
    q = json.loads((Path(assets) / quality_name).read_text())
    candidate = m["status"] == "candidate"
    require(q.get("bundle_sha256") == m["archives"]["native.tar.gz"]["sha256"], "Quality bundle mismatch")
    require(q.get("decision") in ({"accepted", "rejected", "incomplete"} if candidate else {"accepted"}),
            "Quality decision must accept this exact native bundle for a release")
    require("strict_original_gates" in q and (candidate or q.get("quality_parity_with_reference") == "PASS"),
            "Separate parity and strict-original verdicts required")
    require(candidate or q.get("visual_inspection_file") in m["evidence"], "Hashed visual inspection record required")
    a = json.loads((Path(assets) / aggregate_name).read_text())
    require(a.get("bundle_sha256") == m["archives"]["native.tar.gz"]["sha256"], "Aggregate bundle mismatch")
    require(a.get("gpu") == "NVIDIA GeForce RTX 3090", "Aggregate must be measured on RTX 3090")
    if candidate:
        # Missing/incomplete measurements remain explicit. Never fabricate a release pass.
        require(a.get("status") in {"PASS", "FAIL", "INVALID", "NOT_RUN"}, "Candidate aggregate status required")
        require(bool(a.get("limitation")), "Candidate aggregate limitation required")
        return
    for group, count in (("T", 2), ("SUST", 5)):
        windows = a.get(group, [])
        require(isinstance(windows, list) and len(windows) >= count, f"Missing aggregate {group} windows")
        for w in windows:
            seconds, frames = w.get("elapsed_seconds", 0), w.get("completed_valid_frames", 0)
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 60,
                    "Invalid aggregate shared duration")
            require(type(frames) is int and frames > 0 and frames / seconds >= 400,
                    "Native aggregate window below 400 FPS")
            require(w.get("status") == "PASS" and w.get("raw_report") in m["evidence"], "Raw window evidence required")


def verify_source(root, m):
    for name, entry in m["source_files"].items():
        check_file(root, name, entry)
        scan_text(Path(root) / name)


def verify_model_contract(root, m):
    # Keep the canonical installer as the source of truth without importing its
    # module or duplicating its lists. Full-API assets must be declared even when
    # CPU build checks defer the narrowly allowlisted private files to runtime.
    tree = ast.parse((Path(root) / "scripts/musetalk_install_state.py").read_text())
    required = set()
    lists = {"SERVER_MODEL_FILES", "AVATAR_PREP_MODEL_FILES"}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in lists:
                    required.update(ast.literal_eval(node.value))
                    found.add(target.id)
    require(found == lists, "Canonical model contract unavailable")
    require(required <= set(m["model_files"]) | set(external_models(m)),
            "Release omits a canonical full-API model declaration")


def extract(archive, root, allowed):
    """All members must be independently listed; reject links/devices/traversal/duplicates."""
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar.getmembers():
            name = relative(member.name)
            if member.isdir():
                require(any(n.startswith(name + "/") for n in allowed), "Unlisted archive directory")
                continue
            require(member.isreg() and name in allowed and name not in seen, "Unsafe/unlisted archive member")
            seen.add(name)
            destination = checked_path(root, name)
            require(not destination.exists(), "Archive overwrites an existing file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            destination.chmod(0o644)
        require(seen == set(allowed), "Archive missing expected files")


def stage(root, m, assets, revision, base, channel="validated", runtime_base=None):
    require(m["status"] == channel, "Build channel must explicitly match manifest status")
    require(m["source_revision"] == revision and m["cuda_base"] == base, "Build identity mismatch")
    verify_build_bases(m, base, selected_runtime_base(m) if runtime_base is None else runtime_base)
    verify_source(root, m)
    verify_model_contract(root, m)
    actual_source = {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()}
    require(actual_source - {".dockerignore"} == set(m["source_files"]), "Build context contains unlisted or missing source files")
    require(not any((Path(root) / name).exists() for name in external_models(m)),
            "Private runtime model must not be present in public build source")
    verify_evidence(assets, m)
    d = descriptor(root, m)
    for name, entry in m["archives"].items():
        check_file(assets, name, entry)
    native_names = set()
    with tarfile.open(Path(assets) / "native.tar.gz", "r:gz") as tar:
        bm = tar.extractfile(MANIFEST_NAME).read()
        require(hashlib.sha256(bm).hexdigest() == m["bundle_manifest_sha256"], "Bundle sidecar mismatch")
        files = json.loads(bm).get("files", [])
        require(files, "Native bundle manifest empty")
        for e in files:
            name = relative(e["path"])
            require(name in m["model_files"] and name not in native_names, "Unlisted/duplicate native asset")
            require(e["sha256"] == m["model_files"][name]["sha256"] and e["size"] == m["model_files"][name]["size_bytes"],
                    "Native and release asset manifests differ")
            native_names.add(name)
    extract(Path(assets) / "native.tar.gz", root, native_names | {MANIFEST_NAME, CHECKSUM_NAME})
    sidecar = checked_path(root, d["sidecar_dir"])
    sidecar.mkdir(parents=True, exist_ok=True)
    for name in (MANIFEST_NAME, CHECKSUM_NAME):
        (Path(root) / name).rename(sidecar / name)
    extract(Path(assets) / "weights.tar.gz", root, set(m["model_files"]) - native_names)
    for name, entry in m["model_files"].items():
        check_file(root, name, entry)
    for name, entry in m["notices"].items():
        source = check_file(assets, name, entry)
        target = checked_path(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Copy only hashed evidence, not the external release directory wholesale.
    for name in m["evidence"]:
        target = checked_path(Path(root) / "release_evidence", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(assets) / name, target)


def install_args(m):
    return ["--matrix", "cu121", "--venv", "/opt/musetalk/venv", "--skip-apt", "--no-selftest",
            "--with-avatar-prep", "--with-native-vp8", "--with-kokoro" if m["kokoro"] else "--without-kokoro"]


def cpu_check(root, m):
    verify_source(root, m)
    verify_model_contract(root, m)
    descriptor(root, m)
    for name, entry in {**m["model_files"], **m["notices"]}.items():
        check_file(root, name, entry)
    # No private payload enters CI/image layers. Verify all baked hashes above;
    # the complete canonical model-presence check runs after private runtime fetch.
    external = external_models(m)
    require(not any((Path(root) / name).exists() for name in external), "Private runtime model was baked into image")
    subprocess.run(["bash", str(Path(root) / "scripts/install_musetalk.sh"), "--check", "--check-imports",
                    *install_args(m), *(["--skip-weights"] if external else [])], check=True, cwd=root)
    # Importing mmcv._ext alone also succeeds for a CPU-only wheel. This image
    # promises CUDA avatar prep, so verify the extension's compiled CUDA version.
    subprocess.run(["/opt/musetalk/venv/bin/python", "-c",
                    "from mmcv.ops import get_compiling_cuda_version; "
                    "v=str(get_compiling_cuda_version()); "
                    "assert v.startswith('12.1'), 'mmcv extension lacks validated CUDA 12.1 support: '+v; "
                    "print('mmcv compiling CUDA version:',v)"], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=root)


def runtime(root, m, venv):
    verify_source(root, m)
    d = descriptor(root, m)
    for name, entry in {**m["model_files"], **external_models(m)}.items():
        check_file(root, name, entry)
    sidecar = checked_path(root, d["sidecar_dir"])
    require(sha256(sidecar / MANIFEST_NAME) == m["bundle_manifest_sha256"], "Baked bundle sidecar changed")
    subprocess.run([str(Path(venv) / "bin/python"), "-B", str(Path(root) / "scripts/musetalk_host_profile.py"),
                    "bundle-check", "--bundle", m["bundle_name"], "--host-only", "--repo-root", str(root),
                    "--venv", str(venv)], check=True)
    spec = importlib.util.spec_from_file_location("bundle", Path(root) / "scripts/trt_artifact_bundle.py")
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
    # Canonical hash verification. This stamp proves artifact integrity, never GPU warmup/quality.
    bundle._verify_manifest(Path(root), sidecar / MANIFEST_NAME, strict=True)
    bundle._write_stamp(sidecar, d["sha256"], "image:sha256:" + d["sha256"],
                        "image-verified", "validated release manifest and canonical file hashes")


def fetch_private_model(entry, target):
    """Bounded parallel read with pinned content; runtime-injected credentials only.

    Ordinary GetObject is used unless a manifest explicitly requests a version.
    A content-addressed key and full SHA are still mandatory. HEAD checks before
    and after prevent a changing current version from passing this transfer.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    source = entry["source"]
    key, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    require(key and secret, "Private models require approved runtime AWS credentials after secret bootstrap")
    client = boto3.client("s3", region_name=source["region"],
                          aws_access_key_id=key, aws_secret_access_key=secret,
                          aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                          config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=30,
                                        retries={"max_attempts": 2, "mode": "standard"},
                                        ignore_configured_endpoint_urls=True))
    arguments = {"Bucket": source["bucket"], "Key": source["key"], "ExpectedBucketOwner": source["expected_owner"]}
    if source.get("version_id"):
        arguments["VersionId"] = source["version_id"]
    try:
        before = client.head_object(**arguments)
        require(before.get("ContentLength") == entry["size_bytes"], "Private S3 model size mismatch")
        require(before.get("Metadata", {}).get("sha256") == entry["sha256"], "Private S3 model metadata mismatch")
        version = before.get("VersionId")
        require(isinstance(version, str) and 0 < len(version) <= 1024, "Private S3 model version unavailable")
        if source.get("version_id"):
            require(version == source["version_id"], "Private S3 model version mismatch")
        extra = {k: v for k, v in arguments.items() if k not in ("Bucket", "Key")}
        client.download_file(arguments["Bucket"], arguments["Key"], str(target), ExtraArgs=extra,
                             Config=TransferConfig(max_concurrency=4, multipart_chunksize=8 * 1024**2,
                                                   num_download_attempts=1))
        require(target.stat().st_size == entry["size_bytes"] and sha256(target) == entry["sha256"],
                "Private S3 model downloaded content mismatch")
        after = client.head_object(**arguments)
        require(after.get("VersionId") == version and after.get("ContentLength") == entry["size_bytes"]
                and after.get("Metadata", {}).get("sha256") == entry["sha256"],
                "Private S3 model changed during download")
        return {"method": "parallel_download_file_sha256", "max_concurrency": 4,
                "multipart_chunksize": 8 * 1024**2, "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"], "observed_version_id": version,
                "version_id_requested": bool(source.get("version_id")), "head_version_unchanged": True}
    except Exception as exc:
        # Never print SDK requests, headers, secret values or signed query strings.
        raise ValueError("Private S3 model fetch failed (" + type(exc).__name__ + ")") from None


def runtime_models(root, m, cache, fetch=fetch_private_model):
    """Verify every cache hit, fetch only missing pinned files, never overwrite."""
    start = time.monotonic()
    external = external_models(m)
    report = {"phase": "private_model_restore", "models": len(external), "downloaded_bytes": 0,
              "cache_hits": 0, "already_present": 0, "usage_rights_status": m.get("private_model_usage_rights", "not_applicable")}
    if external:
        require(cache.is_absolute() and not cache.is_symlink(), "Private cache must be an absolute non-symlink path")
        cache.mkdir(parents=True, exist_ok=True)
        cache.chmod(0o700)
    for name, entry in external.items():
        target = checked_path(root, name)
        if target.exists():
            check_file(root, name, entry)
            report["already_present"] += 1
            continue
        cache_name = entry["sha256"]
        cached = checked_path(cache, cache_name)
        if cached.exists():
            check_file(cache, cache_name, entry)
            report["cache_hits"] += 1
        else:
            fd, raw = tempfile.mkstemp(prefix="download-", dir=cache)
            os.close(fd)
            temporary = Path(raw)
            try:
                fetch(entry, temporary)
                check_file(cache, temporary.name, entry)
                # Hard-link publication is atomic and refuses an unexpected existing file.
                os.link(temporary, cached)
                report["downloaded_bytes"] += entry["size_bytes"]
            finally:
                temporary.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=".private-model-", dir=target.parent)
        os.close(fd)
        temporary = Path(raw)
        try:
            shutil.copyfile(cached, temporary)
            check_file(target.parent, temporary.name, entry)
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        check_file(root, name, entry)
    report["elapsed_seconds"] = time.monotonic() - start
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def taesd_policy(root, m, d):
    spec = d["engines"]["taesd_trt"]
    name = relative(spec["dir"]) + "/taesd_trt_" + str(spec["key"]) + ".json"
    require(name in m["model_files"], "TAESD metadata is not pinned by release manifest")
    meta = json.loads(check_file(root, name, m["model_files"][name]).read_text())
    fp = meta.get("fingerprint", {})
    require(meta.get("key") == spec["key"] == hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:20],
            "TAESD fingerprint/key mismatch")
    require(fp.get("gpu") == "NVIDIA GeForce RTX 3090" and fp.get("compute_capability") == "8.6"
            and fp.get("batch") == 8 and fp.get("tensorrt") == "10.3.0"
            and fp.get("hardware_compatibility_level", "none") == "none", "TAESD metadata is not native sm86 bs8")
    require(type(fp.get("opt_level")) is int and 0 <= fp["opt_level"] <= 5
            and type(fp.get("strongly_typed")) is bool, "TAESD build flags missing")
    return {"MUSETALK_TAESD_TRT_OPT_LEVEL": str(fp["opt_level"]),
            "MUSETALK_TAESD_TRT_STRONGLY_TYPED": "1" if fp["strongly_typed"] else "0"}


def policy(m, d):
    settings = {
        "AUTO_SETUP": "0", "SETUP_CLEAN": "0", "SETUP_SELFTEST": "0", "SETUP_MATRIX": "cu121", "SETUP_SKIP_WEIGHTS": "0",
        "SETUP_FULL_STACK": "1", "SETUP_KOKORO": "1" if m["kokoro"] else "0", "SETUP_NATIVE_VP8": "1",
        "MUSETALK_RECIPE": "r5", "MUSETALK_VERIFY_RECIPE": "strict", "MUSETALK_R5_BUNDLE_RESTORE": "baked",
        "MUSETALK_UNET_ENGINE_PROVISION": "off", "MUSETALK_TAESD_TRT_PROVISION": "off",
        "MUSETALK_UNET_STAGEWISE_PROVISION": "off", "MUSETALK_TAESD_TRT_BUILD": "0",
        "MUSETALK_TAESD_TRT_STRICT": "1", "MUSETALK_UNET_STAGEWISE_VERIFY_SHA": "1",
        "MUSETALK_UNET_STAGEWISE_PROBE_CHECK": "1", "MUSETALK_UNET_STAGEWISE_PROBE_TOL": "0",
        "MUSETALK_TRT_FALLBACK": "0", "MUSETALK_VP8_FALLBACK": "0",
        "WEBRTC_VP8_ENCODER": m["vp8_encoder"], "LINGUA_WORKER_CALLBACK_REQUIRED": "1",
        "LINGUA_CONTROL_PLANE_ENABLED": "1", "LINGUA_CONTROL_PLANE_ENV_FILE": "/dev/null",
        "TURN_ENV_FORCE": "1",
        "MUSETALK_SECRETS_STRICT": "true", "MUSETALK_ENV_OVERRIDES_FILE": "",
        "MUSETALK_VAE_BACKEND": "taesd", "MUSETALK_TAESD_BACKEND": "trt",
        "MUSETALK_TAESD_TRT_HW_COMPAT": "none", "MUSETALK_UNET_MODE": "auto",
        "MUSETALK_UNET_BACKEND": "trt_stagewise", "MUSETALK_UNET_STAGEWISE_BATCH": "16",
        "HLS_SCHEDULER_FIXED_BATCH_SIZES": "16", "MUSETALK_TAESD_TRT_BATCH": "8",
        "MUSETALK_DISABLE_LOCAL_TTS": "0" if m["kokoro"] else "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    settings["MUSETALK_UNET_STAGEWISE_CACHE_DIR"] = relative(d["engines"]["unet_stagewise"]["cache_dir"])
    settings["MUSETALK_TAESD_TRT_DIR"] = relative(d["engines"]["taesd_trt"]["dir"])
    if m["status"] == "candidate":
        settings["LINGUA_WORKER_CALLBACK_REQUIRED"] = "0"
        settings["LINGUA_CONTROL_PLANE_ENABLED"] = "0"
    return settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["stage", "runtime-base", "apt", "install", "cpu-check", "runtime", "runtime-models", "policy", "revision"])
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("/opt/musetalk/app"))
    p.add_argument("--assets", type=Path)
    p.add_argument("--revision")
    p.add_argument("--base")
    p.add_argument("--runtime-base")
    p.add_argument("--bootstrap-python")
    p.add_argument("--channel", choices=["validated", "candidate"], default="validated")
    p.add_argument("--venv", default="/opt/musetalk/venv")
    p.add_argument("--cache", type=Path)
    a = p.parse_args()
    m = load_manifest(a.manifest)
    if a.command == "stage":
        require("python3=" + str(a.bootstrap_python) in m["apt_packages"], "Bootstrap Python pin mismatch")
        stage(a.root, m, a.assets, a.revision, a.base, a.channel, a.runtime_base)
    elif a.command == "runtime-base":
        require(a.base == selected_runtime_base(m), "Final-stage runtime base differs from release manifest")
    elif a.command == "apt":
        subprocess.run(["apt-get", "update", "-y"], check=True)
        subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", *m["apt_packages"]],
                       check=True, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    elif a.command == "install":
        subprocess.run(["bash", str(a.root / "scripts/install_musetalk.sh"), *install_args(m), "--skip-weights"],
                       check=True, cwd=a.root,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FORCE_CUDA": "1",
                            "MMCV_WITH_OPS": "1", "TORCH_CUDA_ARCH_LIST": "8.6", "MAX_JOBS": "2"})
    elif a.command == "cpu-check":
        cpu_check(a.root, m)
    elif a.command == "runtime":
        runtime(a.root, m, a.venv)
    elif a.command == "runtime-models":
        require(a.cache is not None, "Private model cache path required")
        runtime_models(a.root, m, a.cache)
    elif a.command == "policy":
        if m["status"] == "candidate":
            require(os.environ.get("MUSETALK_CANDIDATE_STANDALONE") == "1",
                    "NONPROMOTABLE image requires MUSETALK_CANDIDATE_STANDALONE=1; production registration forbidden")
            print("unset LINGUA_CONTROL_PLANE_BASE_URL LINGUA_WORKER_REGISTER_URL LINGUA_WORKER_HEARTBEAT_URL LINGUA_WORKER_TOKEN")
        d = descriptor(a.root, m)
        for key, value in {**policy(m, d), **taesd_policy(a.root, m, d)}.items():
            print("export " + key + "=" + shlex.quote(value))
    elif a.command == "revision":
        print(m["source_revision"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        print("Release validation failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
