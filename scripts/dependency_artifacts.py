#!/usr/bin/env python3
"""Capture validated pip evidence; never resolve or install packages on macOS.

An acquisition manifest can contain sdists. It is NOT an offline wheel lock until
every selected artifact has a verified wheel and the closure passes Linux tests.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import urllib.request

SCHEMA = 1
SOURCE = "5cc706e90e50e93da1310628c025a84199cd8042"
BOOTSTRAP = {"pip": "26.2.1", "setuptools": "60.2.0", "wheel": "0.48.0", "packaging": "24.2"}
REQUIRED = {**BOOTSTRAP, "torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121",
            "torchaudio": "2.5.1+cu121", "torch-tensorrt": "2.5.0",
            "tensorrt-cu12": "10.3.0", "tensorrt-cu12-bindings": "10.3.0",
            "tensorrt-cu12-libs": "10.3.0", "numpy": "1.23.5", "onnx": "1.17.0",
            "aiortc": "1.14.0", "av": "16.1.0", "cffi": "2.1.1"}
GROUPS = {"bootstrap": ("bootstrap",), "server": ("bootstrap", "server"),
          "server+kokoro": ("bootstrap", "server", "kokoro"),
          "full": ("bootstrap", "server", "kokoro", "avatar-prep")}
ORIGINS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org",
           "pypi.nvidia.com", "github.com"}
INPUTS = ("server.in", "kokoro.in", "avatar-prep.in", "constraints-cu121.txt")
MMCV = "third_party_wheels/mmcv/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def normalized(name):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name), "Invalid distribution name")
    return re.sub(r"[-_.]+", "-", name).lower()


def safe_source(url, name):
    u = urllib.parse.urlsplit(url)
    require(not u.username and not u.password and not u.query, "Credentialed or temporary source URL rejected")
    require(not u.fragment or re.fullmatch(r"sha256=[a-f0-9]{64}", u.fragment), "Unexpected URL fragment")
    filename = PurePosixPath(urllib.parse.unquote(u.path)).name
    require(filename and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]*", filename), "Unsafe artifact filename")
    if u.scheme == "file":
        require(name == "mmcv" and u.path.endswith("/" + MMCV) and not u.netloc, "Unreviewed local artifact")
        return "repo:" + MMCV, filename
    require(u.scheme == "https" and u.hostname in ORIGINS and u.port in (None, 443), "Unreviewed source origin")
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, "", "")), filename


def artifact(row, group):
    m = row["metadata"]
    name = normalized(m["name"])
    version = m["version"]
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version) is not None, "Unsafe version")
    info = row["download_info"]
    sha = info.get("archive_info", {}).get("hashes", {}).get("sha256")
    require(isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha), "Missing source SHA256")
    source, filename = safe_source(info["url"], name)
    is_wheel = filename.endswith(".whl")
    require(is_wheel or filename.endswith((".tar.gz", ".zip", ".tar.bz2")), "Unknown package format")
    result = {"name": name, "version": version, "filename": filename, "sha256": sha,
              "source": source, "size_bytes": None, "format": "wheel" if is_wheel else "sdist",
              "introduced_by": group, "requested": bool(row.get("requested")),
              "requested_extras": sorted(row.get("requested_extras", [])),
              "requires_dist": m.get("requires_dist", []), "requires_python": m.get("requires_python"),
              "build_provenance": None, "hash_provenance": row.get("hash_provenance", "pip-install-report")}
    if is_wheel:
        parts = filename[:-4].rsplit("-", 3)
        require(len(parts) == 4, "Malformed wheel filename")
        result["wheel_tags"] = dict(zip(("python", "abi", "platform"), parts[1:]))
    else:
        result["wheel_tags"] = None
    return result


def complete_missing_hash(row):
    """Two historical cached reports omit hashes; cross-check small originals.

    This verifies today's recorded-source bytes against the official PyPI index,
    not the missing historical payload. Fresh native trials remain mandatory.
    """
    if row.get("download_info", {}).get("archive_info", {}).get("hashes", {}).get("sha256"):
        return row
    name = normalized(row["metadata"]["name"])
    require(name in {"jinja2", "markupsafe"}, "Unreviewed hashless artifact")
    source, filename = safe_source(row["download_info"]["url"], name)
    require(urllib.parse.urlsplit(source).hostname == "download.pytorch.org", "Unreviewed hashless source")
    with urllib.request.urlopen("https://pypi.org/pypi/" + name + "/" + row["metadata"]["version"] + "/json", timeout=20) as response:
        data = json.loads(response.read(5 * 1024**2))
    matches = [r for r in data["urls"] if r["filename"].lower() == filename.lower() and not r.get("yanked")]
    require(len(matches) == 1 and 0 < matches[0]["size"] < 8 * 1024**2, "No small official-index hash match")
    with urllib.request.urlopen(source, timeout=20) as response:
        payload = response.read(8 * 1024**2 + 1)
    sha = hashlib.sha256(payload).hexdigest()
    require(len(payload) == matches[0]["size"] and sha == matches[0]["digests"]["sha256"], "Recorded source differs from official-index bytes")
    result = copy.deepcopy(row)
    result["download_info"]["archive_info"] = {"hashes": {"sha256": sha}}
    result["hash_provenance"] = "recorded-source-redownload-crosschecked-against-official-PyPI; historical-payload-hash-unavailable"
    return result


def bootstrap_reports():
    """Small official-index metadata reads only, no pip or package download."""
    rows = []
    for name, version in BOOTSTRAP.items():
        request = urllib.request.Request("https://pypi.org/pypi/" + name + "/" + version + "/json",
                                         headers={"User-Agent": "MuseTalk-artifact-capture/1"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read(5 * 1024**2))
        wheels = [r for r in data["urls"] if r["filename"].endswith("-py3-none-any.whl") and not r.get("yanked")]
        require(len(wheels) == 1, "Ambiguous bootstrap wheel")
        w = wheels[0]
        row = {"metadata": {"name": name, "version": version,
                             "requires_dist": data["info"].get("requires_dist") or [],
                             "requires_python": data["info"].get("requires_python")},
               "download_info": {"url": w["url"], "archive_info": {"hashes": w["digests"]}}, "requested": True}
        a = artifact(row, "bootstrap")
        a["size_bytes"] = w["size"]
        a["hash_provenance"] = "official-PyPI-index; bootstrap-has-no-historical-install-report"
        rows.append(a)
    return rows


def group_for_report(path):
    labels = set()
    event_path = path.with_name(path.name.replace(".report.json", ".jsonl"))
    for line in event_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("kind") == "pip_start":
            labels.add(event.get("group"))
    require(len(labels) == 1, "Report has no unambiguous group evidence")
    label = labels.pop()
    require(label in {"server", "kokoro", "avatar-prep", "mmcv", "chumpy", "other"}, "Unknown group evidence")
    return label


def identity(manifest):
    # Source revision is provenance, NOT cache invalidation for unrelated code.
    return {"schema": manifest["schema"], "contract": manifest["contract"],
            "requirement_inputs": manifest["requirement_inputs"],
            "artifacts": [{k: a[k] for k in ("name", "version", "filename", "sha256", "source",
                            "format", "introduced_by", "requires_dist", "requires_python",
                            "requested_extras", "build_provenance")} for a in manifest["artifacts"]]}


def validate_manifest(m):
    require(m["schema"] == SCHEMA and m["artifact_id"] == digest(identity(m)), "Manifest identity mismatch")
    require(m["contract"] == {"os": "Linux", "machine": "x86_64", "implementation": "cpython",
                             "python": "3.10", "matrix": "cu121", "validated_glibc": "2.35"}, "Wrong platform contract")
    inventory = {a["name"]: a["version"] for a in m["artifacts"]}
    require(len(inventory) == len(m["artifacts"]), "Duplicate distribution")
    require(all(inventory.get(n) == v for n, v in REQUIRED.items()), "GPU/bootstrap contract mismatch")
    for a in m["artifacts"]:
        normalized(a["name"])
        require(re.fullmatch(r"[a-f0-9]{64}", a["sha256"]) is not None, "Invalid artifact hash")
        require(a["introduced_by"] in GROUPS["full"], "Invalid artifact group")
        if a["source"].startswith("build:"):
            proof = a["build_provenance"]
            require(a["format"] == "wheel" and isinstance(proof, dict)
                    and a["source"] == "build:" + proof["input_sha256"]
                    and re.fullmatch(r"[a-f0-9]{64}", proof["input_sha256"])
                    and proof.get("python", "").startswith("3.10.")
                    and proof.get("system") == "Linux" and proof.get("machine") == "x86_64",
                    "Missing matching Linux wheel build provenance")
        elif a["source"].startswith("repo:"):
            require(a["name"] == "mmcv" and a["source"] == "repo:" + MMCV, "Wrong local source")
        else:
            source, filename = safe_source(a["source"], a["name"])
            require(source == a["source"] and filename == a["filename"], "Source filename mismatch")
        require(a["size_bytes"] is None or type(a["size_bytes"]) is int and a["size_bytes"] > 0, "Invalid artifact size")
    return m


def capture(evidence, requirements, bootstrap):
    rows = list(bootstrap)
    provenance = []
    reports = sorted(Path(evidence).glob("pip-*.report.json"))
    require(len(reports) == 6, "Expected the complete six-report baseline")
    for path in reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        env = report["environment"]
        require(env["sys_platform"] == "linux" and env["platform_machine"] == "x86_64"
                and env["implementation_name"] == "cpython" and env["python_version"] == "3.10", "Wrong evidence platform")
        raw_group = group_for_report(path)
        group = raw_group if raw_group in {"server", "kokoro"} else "avatar-prep"
        provenance.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "group": group})
        additions = [artifact(complete_missing_hash(r), group) for r in report["install"]]
        if raw_group == "other":
            require({a["name"] for a in additions} == {"ninja"}, "Unreviewed other-group package")
        rows.extend(additions)
    require(len({r["name"] for r in rows}) == len(rows), "Reports contain replacement packages; inspect manually")
    manifest = {"schema": SCHEMA,
                "contract": {"os": "Linux", "machine": "x86_64", "implementation": "cpython",
                             "python": "3.10", "matrix": "cu121", "validated_glibc": "2.35"},
                "requirement_inputs": {n: hashlib.sha256((Path(requirements)/n).read_bytes()).hexdigest() for n in INPUTS},
                "source_revision_provenance": SOURCE, "reports": provenance,
                "status": "acquisition-candidate; Linux wheel builds/closure/native validation pending",
                "artifacts": sorted(rows, key=lambda a: a["name"])}
    manifest["artifact_id"] = digest(identity(manifest))
    return validate_manifest(manifest)


def selected(m, group):
    return [a for a in m["artifacts"] if a["introduced_by"] in GROUPS[group]]


def offline_lock(m, group):
    validate_manifest(m)
    artifacts = selected(m, group)
    require(all(a["format"] == "wheel" and a["size_bytes"] for a in artifacts), "Offline lock blocked: unbuilt/unmeasured artifacts")
    return "# Complete verified local wheels; no URL requirements.\n" + "\n".join(
        a["name"] + "===" + a["version"] + " --hash=sha256:" + a["sha256"] for a in artifacts) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["capture", "inspect", "lock"])
    p.add_argument("--evidence", type=Path)
    p.add_argument("--requirements", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--group", choices=GROUPS, default="full")
    args = p.parse_args()
    if args.mode == "capture":
        require(args.evidence and args.requirements and args.output, "Capture paths required")
        m = capture(args.evidence, args.requirements, bootstrap_reports())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        require(not args.output.exists(), "Refusing to overwrite a captured manifest")
        args.output.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    else:
        require(args.manifest is not None, "Manifest required")
        m = validate_manifest(json.loads(args.manifest.read_text()))
        if args.mode == "lock":
            print(offline_lock(m, args.group), end="")
            return
    print(json.dumps({"artifact_id": m["artifact_id"], "groups": {g: len(selected(m, g)) for g in GROUPS},
                      "source_archives": [a["name"] for a in m["artifacts"] if a["format"] != "wheel"],
                      "status": m["status"]}))


if __name__ == "__main__":
    main()
