#!/usr/bin/env python3
"""Derive a locked manifest whose tensorrt-cu12-libs is a runtime-only repack.

The pinned tensorrt_cu12_libs-10.3.0 wheel is 2,037,510,258 bytes, of which
~1.92 GB compressed are libnvinfer_builder_resource.so plus a Windows copy
(libnvinfer_builder_resource_win.so). Serving workers only deserialize the
prebuilt r5 engines: on 2026-10-18 a fresh RTX 3090 served byte-identical
acceptance videos with both libraries removed. Workers that build engines must
keep the original wheel (use the original manifest).

Run on matching Linux (Python 3.10) with the original wheel at hand:
  python3.10 scripts/derive_runtime_trt_manifest.py --manifest M --manifest-sha256 S \
      --wheel tensorrt_cu12_libs-10.3.0-...whl --output DIR
The repacked wheel keeps name, version and filename; its dist-info carries a
MUSETALK_RUNTIME_ONLY marker and a rewritten RECORD.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_artifacts import digest, identity, offline_lock, require, validate_manifest  # noqa: E402
from fetch_dependency_wheels import read_trusted, sha256_file  # noqa: E402

NAME = "tensorrt-cu12-libs"
DROP = re.compile(r"^tensorrt_libs/libnvinfer_builder_resource(_win)?\.so(\.[0-9.]+)?$")


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def repack(source, destination, original_sha256):
    """Copy every member except the builder resources, in original order and with
    original timestamps/compression, then a marker and a rewritten RECORD."""
    require(sha256_file(source) == original_sha256, "Original TensorRT libs wheel hash mismatch")
    destination = Path(destination)
    require(not destination.exists(), "Repack output already exists")
    removed = []
    with zipfile.ZipFile(source) as src:
        names = src.namelist()
        require(all(not n.startswith("/") and ".." not in Path(n).parts for n in names), "Unsafe wheel member")
        records = [n for n in names if n.endswith(".dist-info/RECORD")]
        require(len(records) == 1, "Ambiguous wheel RECORD")
        record_name = records[0]
        dist_info = record_name.rsplit("/", 1)[0]
        marker_name = dist_info + "/MUSETALK_RUNTIME_ONLY"
        rows = []
        with zipfile.ZipFile(destination, "x") as out:
            for info in src.infolist():
                if info.filename == record_name:
                    continue
                if DROP.match(info.filename):
                    removed.append({"path": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size})
                    continue
                data = src.read(info)
                out.writestr(info, data)
                rows.append((info.filename, record_hash(data), str(len(data))))
            require(len(removed) == 2, "Expected exactly the two builder-resource libraries")
            marker = json.dumps({"original_sha256": original_sha256, "removed": removed,
                                 "reason": "serving workers deserialize prebuilt engines only"},
                                indent=2, sort_keys=True).encode() + b"\n"
            template = src.getinfo(record_name)
            minfo = zipfile.ZipInfo(marker_name, date_time=template.date_time)
            minfo.compress_type, minfo.external_attr = template.compress_type, template.external_attr
            out.writestr(minfo, marker)
            rows.append((marker_name, record_hash(marker), str(len(marker))))
            rows.append((record_name, "", ""))
            record = "".join(",".join(r) + "\n" for r in rows).encode()
            rinfo = zipfile.ZipInfo(record_name, date_time=template.date_time)
            rinfo.compress_type, rinfo.external_attr = template.compress_type, template.external_attr
            out.writestr(rinfo, record)
    return removed


def derive(manifest, wheel, output, facts):
    output = Path(output)
    require(not output.exists(), "Output already claimed")
    m = copy.deepcopy(manifest)
    rows = [a for a in m["artifacts"] if a["name"] == NAME]
    require(len(rows) == 1 and not rows[0]["source"].startswith("build:"), "Expected the upstream TensorRT libs row")
    row = rows[0]
    wheel = Path(wheel)
    require(wheel.name == row["filename"], "Wheel filename differs from the manifest")
    output.mkdir(mode=0o700)
    built = output / row["filename"]
    removed = repack(wheel, built, row["sha256"])
    original = {k: row[k] for k in ("filename", "sha256", "source", "size_bytes")}
    row.update(sha256=sha256_file(built), size_bytes=built.stat().st_size, source="build:" + original["sha256"],
               hash_provenance="verified-local-Linux-runtime-repack")
    row["build_provenance"] = {**facts, "input_sha256": original["sha256"], "input_filename": original["filename"],
                               "input_source": original["source"], "input_bytes": original["size_bytes"],
                               "transform": "runtime-only: removed TensorRT builder-resource libraries",
                               "removed": [r["path"] for r in removed]}
    m["artifact_id"] = digest(identity(m))
    m["status"] = "derived: runtime-only tensorrt-cu12-libs; Linux install/native validation pending"
    validate_manifest(m)
    (output / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    for group in ("bootstrap", "server", "server+kokoro", "full"):
        (output / (group + ".lock")).write_text(offline_lock(m, group))
    receipt = {"state": "derived", "artifact_id": m["artifact_id"], "manifest_sha256": sha256_file(output / "manifest.json"),
               "manifest_bytes": (output / "manifest.json").stat().st_size, "trt_wheel_sha256": row["sha256"],
               "trt_wheel_bytes": row["size_bytes"], "original_trt_wheel_bytes": original["size_bytes"],
               "removed": removed, "build_rows": sum(a["source"].startswith("build:") for a in m["artifacts"])}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--wheel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "x86_64"
            and platform.python_version().startswith("3.10."), "Matching Linux Python 3.10 only")
    facts = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
             "glibc": platform.libc_ver()[1]}
    print(json.dumps(derive(read_trusted(a.manifest, a.manifest_sha256), a.wheel, a.output, facts)), flush=True)


if __name__ == "__main__":
    main()
