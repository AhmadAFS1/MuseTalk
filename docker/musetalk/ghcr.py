#!/usr/bin/env python3
"""Private GHCR bootstrap/publication. Never grants serving acceptance.

Authentication stays in the runner environment; no token is sent to Docker build
or persisted in reports. A model-bearing image requires the existing release
checks, separately from this registry transport implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.error
import urllib.request

import release

REPOSITORY = "AhmadAFS1/MuseTalk"
IMAGE = "ghcr.io/ahmadafs1/musetalk-rtx3090"
PACKAGE_API = "users/AhmadAFS1/packages/container/musetalk-rtx3090"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SECRET_RULES = ("private-key-marker", "aws-access-key-id", "github-token", "github-fine-grained-token")
_stage = "request-validation"
_diagnostic_work = None


class AuditFinding(ValueError):
    """Fixed-rule diagnostic; never contains matched bytes or exception text."""
    def __init__(self, rule, *, path=None, layer=None, file_sha256=None, file_size=None):
        release.require(rule in {"unsafe-path", "credential-path", *SECRET_RULES}, "Unknown audit rule")
        self.detail = {"rule": rule}
        for key, value in (("path", path), ("layer", layer)):
            if value is not None:
                self.detail[key + "_sha256"] = hashlib.sha256(str(value).encode()).hexdigest()
                # Ordinary installed paths are useful; unsafe/arbitrary values
                # are fingerprint-only, never reflected verbatim into CI logs.
                if (len(str(value)) <= 240 and re.fullmatch(r"[A-Za-z0-9_./+-]+", str(value))
                        and not any(p.search(str(value).encode()) for p in release.SECRET_PATTERNS)):
                    self.detail[key] = str(value)
        if file_sha256 is not None:
            release.require(release.SHA.fullmatch(file_sha256), "Audit file hash invalid")
            self.detail["file_sha256"] = file_sha256
        if file_size is not None:
            self.detail["file_size_bytes"] = int(file_size)
        super().__init__("Image audit rejected (matched content suppressed)")


def stage(name):
    global _stage
    release.require(re.fullmatch(r"[a-z][a-z-]{0,63}", name), "Invalid fixed stage")
    _stage = name
    print("GHCR stage: " + name, flush=True)


def failure_record(exc):
    # Exception messages, subprocess output and URLs are deliberately excluded.
    known_types = {"ValueError", "KeyError", "TypeError", "OSError", "FileNotFoundError", "JSONDecodeError",
                   "ReadError", "StreamError", "CompressionError", "HeaderError", "URLError", "HTTPError",
                   "RegistryOperationError", "AuditFinding"}
    kind = type(exc).__name__
    result = {"schema": "musetalk_ghcr_failure_v1", "status": "FAIL", "stage": _stage,
              "exception_type": kind if kind in known_types else "OtherError",
              "details_suppressed": True, "publication_verified": False}
    if isinstance(exc, AuditFinding):
        result["finding"] = exc.detail
    if isinstance(exc, RegistryOperationError):
        # This class already contains fixed operation names and numeric status.
        result["operation"] = str(exc)
    return result


def report_failure(exc, work=None):
    result = failure_record(exc)
    if work is not None:
        try:
            work.mkdir(parents=True, exist_ok=True)
            with (work / "failure.json").open("x") as output:
                output.write(json.dumps(result, indent=2) + "\n")
        except OSError:
            pass  # Never overwrite a receipt or obscure the original failure.
    print(json.dumps(result, sort_keys=True), file=__import__("sys").stderr)


class RegistryOperationError(RuntimeError):
    """Diagnostic containing only a fixed operation name and numeric status."""


def command(argv, *, payload=None):
    # On error, do not reproduce subprocess output: authentication/request details
    # must not leak through diagnostics. Commands contain no credential arguments.
    result = subprocess.run(argv, input=payload, text=True, capture_output=True)
    if result.returncode:
        operations = {("docker", "login"): "registry login", ("docker", "buildx"): "image build/inspect",
                      ("docker", "tag"): "local tagging", ("docker", "push"): "registry push",
                      ("docker", "pull"): "registry pull", ("gh", "api"): "package API",
                      ("docker", "save"): "layer export", ("docker", "image"): "local image inspect",
                      ("docker", "history"): "local history inspect"}
        operation = operations.get(tuple(argv[:2]), "registry subprocess")
        status = re.search(r"\(HTTP (\d{3})\)", result.stderr)
        suffix = " HTTP " + status[1] if status else ""
        raise RegistryOperationError(f"{operation} failed (exit {result.returncode}{suffix}; output suppressed)")
    return result.stdout


def login():
    stage("registry-login")
    token = os.environ.get("GH_TOKEN", "")
    actor = os.environ.get("GITHUB_ACTOR", "")
    release.require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY,
                    "Publication must run in the intended repository")
    release.require(token and re.fullmatch(r"[A-Za-z0-9-]+", actor), "Runner authentication missing")
    command(["docker", "login", "ghcr.io", "--username", actor, "--password-stdin"], payload=token + "\n")


def private_package():
    stage("package-visibility")
    package = json.loads(command(["gh", "api", PACKAGE_API]))
    release.require(package.get("name") == "musetalk-rtx3090" and package.get("visibility") == "private",
                    "GHCR package is absent or not private; refusing publication")
    return {"name": package["name"], "visibility": package["visibility"]}


def anonymous_denied(digest, opener=urllib.request.urlopen):
    stage("anonymous-denial")
    release.require(DIGEST.fullmatch(digest), "Actual manifest digest required")
    token_url = "https://ghcr.io/token?service=ghcr.io&scope=repository:ahmadafs1/musetalk-rtx3090:pull"
    try:
        with opener(token_url, timeout=30) as response:
            body = json.load(response)
        token = body.get("token", "")
        release.require(bool(token), "Anonymous registry response has no token")
        request = urllib.request.Request(IMAGE.replace("ghcr.io/", "https://ghcr.io/v2/") + "/manifests/" + digest,
                                         headers={"Authorization": "Bearer " + token,
                                                  "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"})
        with opener(request, timeout=30):
            pass
    except urllib.error.HTTPError as exc:
        release.require(exc.code in {401, 403}, "Anonymous access check failed for an inconclusive HTTP reason")
        return True
    # A network error is not proof of privacy; other exceptions propagate safely.
    raise ValueError("Anonymous manifest download succeeded; package is not private")


def pushed_digest(output):
    found = re.findall(r"\bdigest: (sha256:[0-9a-f]{64})\b", output)
    release.require(len(found) == 1, "Push did not identify one actual manifest digest")
    return found[0]


def push(image, tag):
    release.require(re.fullmatch(r"(?:bootstrap|dependency|candidate)-[0-9a-f]{40}", tag), "Unsafe/nonpromotable tag")
    destination = IMAGE + ":" + tag
    stage("image-tag")
    command(["docker", "tag", image, destination])
    stage("registry-push")
    digest = pushed_digest(command(["docker", "push", destination]))
    return IMAGE + "@" + digest, digest


def bootstrap(work, revision):
    release.require(not work.exists(), "Bootstrap work directory must be new")
    work.mkdir(parents=True)
    # Nothing from the repository, model payload, or environment enters this image.
    (work / "Dockerfile").write_text(
        'FROM scratch\n'
        'LABEL org.opencontainers.image.source="https://github.com/' + REPOSITORY + '" '
        'org.opencontainers.image.title="PRIVATE registry bootstrap; NOT A SERVING IMAGE" '
        'io.musetalk.release-channel="bootstrap-no-runtime"\n')
    local = "musetalk-registry-bootstrap:" + revision
    print("GHCR stage: build non-serving placeholder", flush=True)
    command(["docker", "buildx", "build", "--platform", "linux/amd64", "--load", "--tag", local, str(work)])
    # A brand-new package defaults private. Only this non-sensitive placeholder
    # may be sent before API verification; never a dependency/model-bearing image.
    print("GHCR stage: push non-serving placeholder", flush=True)
    reference, digest = push(local, "bootstrap-" + revision)
    print("GHCR stage: verify private package visibility", flush=True)
    privacy = private_package()
    print("GHCR stage: verify anonymous denial", flush=True)
    anonymous_denied(digest)
    return {"schema": "musetalk_ghcr_bootstrap_v1", "image": reference, "digest": digest,
            "source_revision": revision, "package": privacy, "anonymous_pull": "DENIED",
            "published": True, "serving_image": False, "promotion_eligible": False}


def scan_layer(stream, name):
    stage("audit-layer")
    files = 0
    with tarfile.open(fileobj=stream, mode="r|") as layer:
        for member in layer:
            if member.isdir() and member.name in {".", "./"}:
                continue  # A root directory marker is not a payload path.
            path = member.name.removeprefix("./")
            try:
                release.relative(path)
            except (ValueError, TypeError):
                raise AuditFinding("unsafe-path", path=path, layer=name) from None
            forbidden = ("root/.ssh/", "root/.aws/", "root/.config/gh/", "root/.docker/",
                         "opt/musetalk/app/.git/", "opt/musetalk/app/.env")
            if any(path.startswith(prefix) for prefix in forbidden):
                raise AuditFinding("credential-path", path=path, layer=name)
            if not member.isreg():
                continue
            files += 1
            tail = b""
            checksum = hashlib.sha256()
            finding = None
            with layer.extractfile(member) as content:
                for block in iter(lambda: content.read(1024 * 1024), b""):
                    checksum.update(block)
                    data = tail + block
                    if finding is None:
                        finding = next((rule for rule, p in zip(SECRET_RULES, release.SECRET_PATTERNS)
                                        if p.search(data)), None)
                    tail = data[-256:]
            if finding is not None:
                raise AuditFinding(finding, path=path, layer=name, file_sha256=checksum.hexdigest(),
                                   file_size=member.size)
    return {"layer": name, "regular_files_scanned": files}


def audit(image, work):
    release.require(not work.exists(), "Layer audit output must be new")
    work.mkdir(parents=True)
    stage("audit-image-inspect")
    inspect = json.loads(command(["docker", "image", "inspect", image]))[0]
    release.require(inspect.get("Architecture") == "amd64" and inspect.get("Os") == "linux", "Wrong image platform")
    config = inspect.get("Config", {})
    stage("audit-image-config")
    metadata = json.dumps(config).encode()
    release.require(not any(p.search(metadata) for p in release.SECRET_PATTERNS), "Possible image-config credential")
    stage("audit-image-history")
    history = command(["docker", "history", "--no-trunc", "--format", "{{json .}}", image]).encode()
    release.require(not any(p.search(history) for p in release.SECRET_PATTERNS), "Possible image-history credential")
    archive = work / "image-layer-audit.tar"
    stage("audit-layer-export")
    command(["docker", "save", "--output", str(archive), image])
    rows = []
    stage("audit-archive-layout")
    with tarfile.open(archive) as exported:
        manifests = json.load(exported.extractfile("manifest.json"))
        release.require(len(manifests) == 1, "Audit needs exactly one image")
        for name in manifests[0]["Layers"]:
            release.relative(name)
            rows.append(scan_layer(exported.extractfile(name), name))
    # Delete only the exact newly created, successfully audited task-local export.
    archive.unlink()
    report = {"schema": "musetalk_ghcr_layer_scan_v1", "status": "PASS", "image_config_id": inspect["Id"],
              "uncompressed_image_bytes": inspect["Size"], "layers": rows,
              "limitation": "Heuristic credential/path scan of every exported layer; not license, GPU, quality or serving acceptance"}
    (work / "layer-scan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def publish_dependency(image, work, revision):
    privacy = private_package()  # Before transmitting any dependency payload.
    stage("dependency-identity")
    inspect = json.loads(command(["docker", "image", "inspect", image]))[0]
    labels = inspect.get("Config", {}).get("Labels", {})
    release.require(labels.get("org.opencontainers.image.revision") == revision
                    and labels.get("io.musetalk.release-channel") == "dependency-diagnostic-no-models",
                    "Only an explicitly nonpromotable dependency diagnostic is supported")
    scan = audit(image, work)
    reference, digest = push(image, "dependency-" + revision)
    private_package()  # Detect visibility changes during the build/push window.
    anonymous_denied(digest)
    stage("published-manifest-identity")
    manifest = json.loads(command(["docker", "buildx", "imagetools", "inspect", "--raw", reference]))
    release.require(bool(manifest.get("layers")) and manifest.get("config", {}).get("digest") == inspect["Id"],
                    "Registry manifest differs from audited local image")
    result = {"schema": "musetalk_ghcr_dependency_publication_v1", "image": reference, "digest": digest,
              "source_revision": revision, "package": privacy, "anonymous_pull": "DENIED",
              "compressed_layer_bytes": sum(layer["size"] for layer in manifest["layers"]),
              "compressed_config_bytes": manifest["config"]["size"], "layer_scan": scan,
              "published": True, "serving_image": False, "promotion_eligible": False,
              "limitation": "No model weights, native bundle, Kokoro or serving release; NOT ready for Vast production"}
    return result


def verify(reference):
    release.require(reference.startswith(IMAGE + "@") and DIGEST.fullmatch(reference.split("@", 1)[1]),
                    "Exact intended image digest required")
    private_package()
    anonymous_denied(reference.split("@", 1)[1])
    stage("registry-pull")
    command(["docker", "pull", "--platform", "linux/amd64", reference])
    inspect = json.loads(command(["docker", "image", "inspect", reference]))[0]
    release.require(inspect.get("Os") == "linux" and inspect.get("Architecture") == "amd64", "Wrong pulled platform")
    return {"schema": "musetalk_ghcr_independent_pull_v1", "image": reference,
            "image_config_id": inspect["Id"], "private_visibility": "VERIFIED", "anonymous_pull": "DENIED",
            "status": "PASS", "gpu_tested": False, "serving_accepted": False}


def publish_candidate(image, work, revision, manifest_path, reports):
    """Publish only an already CPU-checked, explicitly nonpromotable full image."""
    stage("candidate-manifest")
    manifest = release.load_manifest(manifest_path)
    release.require(manifest["source_revision"] == revision and manifest["status"] == "candidate"
                    and manifest.get("promotion_eligible") is False,
                    "Only a reviewed nonpromotable candidate manifest is supported")
    receipt = json.loads((reports / "build-result.json").read_text())
    release.require(receipt.get("schema") == "musetalk_docker_ci_build_v1"
                    and receipt.get("image") == image and receipt.get("source_revision") == revision
                    and receipt.get("channel") == "candidate" and receipt.get("cpu_build_check") == "PASS"
                    and receipt.get("promotion_eligible") is False,
                    "Matching full candidate CPU-build receipt required")
    privacy = private_package()  # Before any model-bearing transmission.
    stage("candidate-image-identity")
    inspect = json.loads(command(["docker", "image", "inspect", image]))[0]
    labels = inspect.get("Config", {}).get("Labels", {})
    release.require(labels.get("org.opencontainers.image.revision") == revision
                    and labels.get("io.musetalk.release-channel") == "candidate"
                    and labels.get("io.musetalk.cuda-runtime-base") == release.selected_runtime_base(manifest),
                    "Candidate image identity differs from reviewed build")
    baked = json.loads(command(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/cat",
                               image, "/opt/musetalk/release.json"]))
    release.require(baked == manifest, "Baked release manifest differs from reviewed inputs")
    scan = audit(image, work)
    reference, digest = push(image, "candidate-" + revision)
    private_package()
    anonymous_denied(digest)
    remote = json.loads(command(["docker", "buildx", "imagetools", "inspect", "--raw", reference]))
    release.require(bool(remote.get("layers")) and remote.get("config", {}).get("digest") == inspect["Id"],
                    "Registry candidate differs from audited local image")
    return {"schema": "musetalk_ghcr_candidate_publication_v1", "image": reference, "digest": digest,
            "source_revision": revision, "release_manifest_sha256": release.sha256(manifest_path),
            "package": privacy, "anonymous_pull": "DENIED", "layer_scan": scan,
            "compressed_layer_bytes": sum(layer["size"] for layer in remote["layers"]),
            "published": True, "serving_image": True, "promotion_eligible": False,
            "gpu_tested": False, "production_ready": False,
            "limitation": "Nonpromotable full candidate; fresh GPU, quality, external calls and cold-host timing still required"}


def main():
    global _diagnostic_work
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bootstrap", "dependency", "candidate", "verify"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image")
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--build-reports", type=Path)
    args = parser.parse_args()
    release.require(not args.work.exists(), "Diagnostic/publication work must be new")
    _diagnostic_work = args.work
    release.require(re.fullmatch(r"[0-9a-f]{40}", args.revision), "Full reviewed commit required")
    login()
    if args.action == "bootstrap":
        result = bootstrap(args.work, args.revision)
        output = args.work / "result.json"
    elif args.action == "dependency":
        release.require(bool(args.image), "Audited local image required")
        result = publish_dependency(args.image, args.work, args.revision)
        output = args.work / "result.json"
    elif args.action == "candidate":
        release.require(bool(args.image) and args.release_manifest and args.build_reports,
                        "Full candidate image, reviewed manifest and CPU receipts required")
        result = publish_candidate(args.image, args.work, args.revision, args.release_manifest, args.build_reports)
        output = args.work / "result.json"
    else:
        release.require(bool(args.image) and not args.work.exists(), "Fresh independent-pull output required")
        args.work.mkdir(parents=True)
        result = verify(args.image)
        output = args.work / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("image=" + result["image"] + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report_failure(exc, _diagnostic_work)
        raise SystemExit(1)
