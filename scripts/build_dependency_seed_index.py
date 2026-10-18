#!/usr/bin/env python3
"""Build the pinned seed index for base-image seeding from the pinned wheels.

Every candidate wheel must match the trusted manifest (sha256 and size). Its own
RECORD is then checked against the archive: every listed file must exist with
the recorded sha256 and size, and the archive may contain nothing unlisted. The
index stores, per artifact, name/version/filename/wheel sha256/size, the
.dist-info directory and the RECORD rows exactly as listed (minus RECORD itself)
in deterministic gzipped JSON. Wheels with .data/ entries are flagged
(has_data_dir) and the installer never seeds them.

No network: run it where the pinned wheels already are, e.g. a worker's locked
installer cache after a normal (unseeded) install:

  python3 -B scripts/build_dependency_seed_index.py \\
    --manifest /workspace/.cache/musetalk-dependencies/<artifact_id>/manifest.json \\
    --wheels /workspace/.cache/musetalk-dependencies/<artifact_id>/wheels \\
    --contract configs/dependencies/rtx3090-cp310-cu121-screening.json --write-contract

The output defaults to configs/dependencies/seed-index-<artifact_id>.json.gz;
--write-contract pins it as the contract's seed_index {path, sha256, bytes}.
"""
import argparse
import csv
import email.parser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

from dependency_artifacts import normalized, require
import dependency_seed as seed
import fetch_dependency_wheels as fetch

REPO = Path(__file__).resolve().parent.parent
# Measured hash-identical in vastai/pytorch:cuda-12.1-auto /venv/main (2026-10-18).
# Not reusable there: nvidia-nvjitlink-cu12 (12.1.105 vs 12.9.86), numpy, tensorrt*.
DEFAULT_NAMES = ("nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-nvrtc-cu12",
                 "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
                 "nvidia-cusolver-cu12", "nvidia-cusparse-cu12", "nvidia-nccl-cu12", "nvidia-nvtx-cu12",
                 "torch", "torchaudio", "torchvision", "triton")
MAX_RECORD = 64 * 1024**2


def index_path(artifact_id):
    return "configs/dependencies/seed-index-" + artifact_id + ".json.gz"


def wheel_entry(row, wheel):
    """One index entry from a wheel whose bytes match the manifest row."""
    wheel = Path(wheel)
    require(wheel.is_file() and not wheel.is_symlink(), "Wheel missing: " + row["filename"])
    require(wheel.stat().st_size == row["size_bytes"] and fetch.sha256_file(wheel) == row["sha256"],
            "Wheel differs from the locked manifest: " + row["filename"])
    with zipfile.ZipFile(wheel) as archive:
        infos = [i for i in archive.infolist() if not i.filename.endswith("/")]
        names = [i.filename for i in infos]
        require(len(set(names)) == len(names), "Duplicate wheel members")
        tops = {n.split("/", 1)[0] for n in names if n.split("/", 1)[0].endswith(".dist-info")}
        require(len(tops) == 1, "Wheel needs exactly one .dist-info directory")
        dist_info = tops.pop()
        record_name = dist_info + "/RECORD"
        require(record_name in names, "Wheel RECORD missing")
        require(archive.getinfo(record_name).file_size <= MAX_RECORD, "Wheel RECORD too large")
        text = archive.read(record_name).decode("utf-8")
        record = []
        for fields in csv.reader(io.StringIO(text)):
            if not fields:
                continue
            require(len(fields) == 3, "Malformed RECORD row")
            path, digest, size = fields
            if path == record_name:
                require(not digest, "RECORD must not hash itself")
                continue
            require(digest.startswith("sha256=") and size.isdigit(), "Unhashed or unsized RECORD row: " + path)
            record.append([seed.safe_record_path(path), digest, int(size)])
        listed = [r[0] for r in record]
        require(len(set(listed)) == len(listed), "Duplicate RECORD path")
        require(set(listed) == set(names) - {record_name}, "Wheel members and RECORD differ")
        by_name = {i.filename: i for i in infos}
        for path, digest, size in record:
            info = by_name[path]
            require(info.file_size == size, "RECORD size differs from archive: " + path)
            value = hashlib.sha256()
            with archive.open(info) as source:
                while chunk := source.read(fetch.CHUNK):
                    value.update(chunk)
            require(seed.record_hash(value.digest()) == digest, "RECORD hash differs from archive: " + path)
        metadata = email.parser.HeaderParser().parsestr(archive.read(dist_info + "/METADATA").decode("utf-8"))
        require(normalized(metadata["Name"]) == row["name"] and metadata["Version"] == row["version"],
                "Wheel METADATA differs from the locked manifest: " + row["filename"])
    return {"name": row["name"], "version": row["version"], "filename": row["filename"],
            "sha256": row["sha256"], "size_bytes": row["size_bytes"], "dist_info": dist_info,
            "has_data_dir": any(seed.is_data_path(p) for p in listed),
            "files": len(record), "installed_bytes": sum(r[2] for r in record), "record": record}


def build(manifest, wheels, names=DEFAULT_NAMES):
    rows = {a["name"]: a for a in manifest["artifacts"]}
    require(names and len(set(names)) == len(names), "Seed candidates must be unique")
    missing = sorted(set(names) - set(rows))
    require(not missing, "Seed candidates absent from manifest: " + ", ".join(missing))
    entries = [wheel_entry(rows[n], Path(wheels) / rows[n]["filename"]) for n in sorted(names)]
    document = {"schema": seed.SCHEMA, "kind": seed.KIND, "artifact_id": manifest["artifact_id"],
                "artifacts": entries}
    return seed.validate_index(document, manifest)


def write(document, output):
    data = seed.encode_index(document)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(output)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", help="Expected manifest digest (default: the contract's)")
    p.add_argument("--wheels", type=Path, required=True, help="Directory holding the pinned wheels")
    p.add_argument("--contract", type=Path, help="Reviewed contract the index belongs to")
    p.add_argument("--write-contract", action="store_true", help="Pin the result as the contract's seed_index")
    p.add_argument("--name", action="append", help="Seed candidate (repeatable; default: measured set)")
    p.add_argument("--repo-root", type=Path, default=REPO)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    contract = json.loads(args.contract.read_text()) if args.contract else None
    expected = args.manifest_sha256 or (contract or {}).get("manifest_sha256")
    require(expected and re.fullmatch("[a-f0-9]{64}", expected), "Manifest digest must be pinned")
    manifest = fetch.read_trusted(args.manifest, expected)
    if contract:
        require(contract.get("artifact_id") == manifest["artifact_id"], "Contract belongs to another artifact")
    require(not args.write_contract or contract, "--write-contract needs --contract")
    relative = index_path(manifest["artifact_id"])
    output = args.output or args.repo_root / relative
    document = build(manifest, args.wheels, tuple(args.name or DEFAULT_NAMES))
    pin = write(document, output)
    field = {"path": relative, **pin}
    if args.write_contract:
        require(Path(output).resolve() == (args.repo_root / relative).resolve(),
                "Contract pins only configs/dependencies/seed-index-<artifact_id>.json.gz")
        contract["seed_index"] = field
        args.contract.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"seed_index": field, "output": str(output),
                      "artifacts": [{k: e[k] for k in ("name", "version", "files", "installed_bytes",
                                                       "size_bytes", "has_data_dir")}
                                    for e in document["artifacts"]]}, indent=2))


if __name__ == "__main__":
    main()
