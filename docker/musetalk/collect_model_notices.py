#!/usr/bin/env python3
"""Acquire pinned public notices/reference hashes, or bind a later staged model tree.

Neither command grants publication approval. collect downloads only text/JSON,
never model weights; bind reads only the explicitly inventoried target files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import urlparse

import release

FILES = {
    "musetalk_v15": {"musetalkV15/musetalk.json": "models/musetalkV15/musetalk.json",
                     "musetalkV15/unet.pth": "models/musetalkV15/unet.pth"},
    "sd_vae_ft_mse": {name: "models/sd-vae/" + name for name in ("config.json", "diffusion_pytorch_model.bin")},
    "whisper_tiny_hf_conversion": {name: "models/whisper/" + name for name in ("config.json", "pytorch_model.bin", "preprocessor_config.json")},
    "taesd": {name: "models/taesd/" + name for name in ("config.json", "diffusion_pytorch_model.safetensors")},
    "dwpose_l_384": {"dw-ll_ucoco_384.pth": "models/dwpose/dw-ll_ucoco_384.pth"},
}
APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
APACHE_SHA = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def fetch(url):
    parsed = urlparse(url)
    release.require(parsed.scheme == "https" and parsed.hostname in {
        "huggingface.co", "raw.githubusercontent.com", "www.apache.org"}, "Unapproved notice/reference host")
    release.require(url == APACHE_URL or re.search(r"/[0-9a-f]{40}(?:/|\?)", url), "Pinned source revision required")
    data = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                                    "--max-time", "30", "--max-filesize", "1048576",
                                    "--proto", "=https", "--proto-redir", "=https", url])
    release.require(len(data) <= 1024**2, "Text/reference payload too large")
    return data


def collect(review_path, output):
    review = json.loads(review_path.read_text())
    models = [m for m in review["models"] if m["id"] in {*FILES, "kokoro_optional"}]
    release.require(len(models) == len(FILES) + 1, "Expected reviewed permissive-model inventory absent")
    output.mkdir(parents=True, exist_ok=False)
    licenses = output / "licenses"
    licenses.mkdir()
    sources = {name: review["sources"][name] for m in models for name in m["sources"]}
    sources["apache_2_0_license_text"] = {"url": APACHE_URL, "sha256": APACHE_SHA}
    def notice(item):
        name, source = item
        data = fetch(source["url"])
        digest = hashlib.sha256(data).hexdigest()
        release.require(digest == source["sha256"], "Pinned notice changed: " + name)
        path = licenses / (name + ".txt")
        path.write_bytes(data)
        release.scan_text(path)
        return "licenses/" + path.name, {"sha256": digest, "size_bytes": len(data), "source_url": source["url"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        notices = dict(pool.map(notice, sorted(sources.items())))
    references = []
    for model in models:
        repository, revision = model["source_repository"], model["observed_revision"]
        api_url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
        metadata = json.loads(fetch(api_url))
        release.require(metadata.get("sha") == revision, "HF metadata revision differs")
        siblings = {s["rfilename"]: s for s in metadata["siblings"]}
        paths = FILES.get(model["id"])
        if paths is None:
            prefix = f"models/hf-cache/hub/models--hexgrad--Kokoro-82M/snapshots/{revision}/"
            paths = {name: prefix + name for name in ["config.json", "kokoro-v1_0.pth", *model["default_voice_files"]]}
        for upstream, target in paths.items():
            sibling = siblings[upstream]
            if sibling.get("lfs"):
                digest, size = sibling["lfs"]["sha256"], sibling["lfs"]["size"]
            else:
                release.require(upstream.endswith(".json"), "Never fetch non-JSON model payloads")
                data = fetch(f"https://huggingface.co/{repository}/resolve/{revision}/{upstream}")
                digest, size = hashlib.sha256(data).hexdigest(), len(data)
            expected = model.get("upstream_weight_reference", {})
            if expected.get("file") == upstream:
                release.require((digest, size) == (expected["sha256"], expected["size_bytes"]), "Frozen weight reference differs")
            if upstream in model.get("upstream_voice_references", {}):
                release.require(digest == model["upstream_voice_references"][upstream], "Frozen voice reference differs")
            references.append({"model_id": model["id"], "target_path": target, "repository": repository,
                               "revision": revision, "upstream_path": upstream, "sha256": digest, "size_bytes": size,
                               "byte_binding": "TARGET_NOT_READ", "public_redistribution_approved": False})
    result = {"schema": "musetalk_notice_reference_inventory_v1", "review_sha256": release.sha256(review_path),
              "publication_approved": False, "actual_model_bytes_verified": False,
              "scope": "Pinned notices/model cards plus upstream reference hashes only; no weight download or GPU worker IO",
              "limitations": ["SD-VAE full copyright attribution remains unresolved; MuseTalk MIT notice does not license SD-VAE.",
                              "Whisper original MIT and HF Apache-2.0 conversion notices remain distinct.",
                              "TAESD FP16/native plans require derivative provenance; mismatch must not be relabeled as upstream match.",
                              "Kokoro is optional; actual selected capability, voice/data attribution and transitive package notices remain separate.",
                              "No HOLD model is approved, fetched, or included. CUDA/TRT/OS/Python distribution review remains separate."],
              "notices": notices, "models": references}
    (output / "notice-reference-inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "notices": len(notices), "model_references": len(references),
                      "publication_approved": False, "actual_model_bytes_verified": False}))


def bind(index, root, output):
    data = json.loads(index.read_text())
    release.require(data.get("schema") == "musetalk_notice_reference_inventory_v1", "Wrong reference inventory schema")
    release.require(not output.exists(), "Refusing binding report overwrite")
    records = []
    for reference in data["models"]:
        path = release.checked_path(root, reference["target_path"])
        actual = {"sha256": release.sha256(path), "size_bytes": path.stat().st_size} if path.is_file() else None
        status = "MISSING" if actual is None else "MATCH" if all(actual[k] == reference[k] for k in actual) else "DIFFERENT_REQUIRES_PROVENANCE"
        records.append({**reference, "byte_binding": status, "actual": actual})
    result = {"schema": "musetalk_model_byte_binding_v1", "reference_inventory_sha256": release.sha256(index),
              "publication_approved": False, "models": records,
              "limitation": "Byte identity is not license signoff; differences, missing optional models and derivatives need explicit disposition"}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "counts": {status: sum(r["byte_binding"] == status for r in records)
                       for status in ("MATCH", "MISSING", "DIFFERENT_REQUIRES_PROVENANCE")}, "publication_approved": False}))


def bind_captured(index, captured, captured_sha256, captured_base, output):
    """Compare two small frozen inventories, performing no GPU worker/model IO."""
    release.require(release.sha256(captured) == captured_sha256, "Captured model inventory checksum differs")
    release.relative(captured_base)
    data = json.loads(index.read_text())
    snapshot = json.loads(captured.read_text())
    release.require(data.get("schema") == "musetalk_notice_reference_inventory_v1"
                    and snapshot.get("schema") == "repro_3090_inputs_v1", "Wrong inventory schema")
    actual_files = {}
    for item in snapshot["files"]:
        name = posixpath.normpath(captured_base + "/" + item["path"])
        if name.startswith("models/"):
            release.relative(name)
            release.require(name not in actual_files, "Duplicate captured model path")
            actual_files[name] = {"sha256": item["sha256"], "size_bytes": item["bytes"]}
    records = []
    for reference in data["models"]:
        actual = actual_files.get(reference["target_path"])
        status = "NOT_IN_CAPTURE" if actual is None else "MATCH" if all(actual[k] == reference[k] for k in actual) else "DIFFERENT_REQUIRES_PROVENANCE"
        records.append({**reference, "byte_binding": status, "actual": actual})
    result = {"schema": "musetalk_model_byte_binding_v1", "reference_inventory_sha256": release.sha256(index),
              "captured_inventory_sha256": captured_sha256, "captured_inventory_base": captured_base,
              "publication_approved": False, "models": records,
              "limitation": "Matches bind captured target bytes, not license signoff. NOT_IN_CAPTURE is not a claim that a model is absent on the worker."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "counts": {status: sum(r["byte_binding"] == status for r in records)
                       for status in ("MATCH", "NOT_IN_CAPTURE", "DIFFERENT_REQUIRES_PROVENANCE")}, "publication_approved": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--review", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    bind_parser = commands.add_parser("bind")
    bind_parser.add_argument("--index", type=Path, required=True)
    bind_parser.add_argument("--root", type=Path, required=True)
    bind_parser.add_argument("--output", type=Path, required=True)
    captured_parser = commands.add_parser("bind-captured")
    captured_parser.add_argument("--index", type=Path, required=True)
    captured_parser.add_argument("--captured", type=Path, required=True)
    captured_parser.add_argument("--captured-sha256", required=True)
    captured_parser.add_argument("--captured-base", required=True)
    captured_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.review, args.output)
    elif args.command == "bind":
        bind(args.index, args.root, args.output)
    else:
        bind_captured(args.index, args.captured, args.captured_sha256, args.captured_base, args.output)
