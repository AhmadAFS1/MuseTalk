"""Exact scheduler A/B evidence, not native numerical/visual release acceptance."""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import re
import struct
import zipfile

import quality_envelope
import report as checks

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ordered_tracking_same_engine_pair_v1"
ARRAYS = {
    "generated_landmarks.npy": ("<f4", (240, 478, 2), "<f"),
    "chin_delta.npy": ("<f8", (240, 161), "<d"),
}


def _array_hashes(path):
    """Stream exact little-endian, finite canonical NPY payloads; never unpickle."""
    results = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        checks.require(len(infos) == len(ARRAYS) and {x.filename for x in infos} == set(ARRAYS), "unexpected landmark/delta NPZ members")
        for name, (dtype, shape, fmt) in ARRAYS.items():
            info = archive.getinfo(name)
            expected = math.prod(shape) * struct.calcsize(fmt)
            checks.require(not info.is_dir() and ((info.external_attr >> 16) & 0o170000) != 0o120000
                           and expected <= info.file_size <= expected + 4108, "unsafe or wrong-size array member")
            with archive.open(name) as stream:
                checks.require(stream.read(6) == b"\x93NUMPY", "invalid array NPY magic")
                version = stream.read(2)
                checks.require(version in (b"\x01\x00", b"\x02\x00"), "unsupported array NPY version")
                length = int.from_bytes(stream.read(2 if version[0] == 1 else 4), "little")
                checks.require(0 < length <= 4096, "invalid array NPY header length")
                header = ast.literal_eval(stream.read(length).decode("ascii"))
                checks.require(header == {"descr": dtype, "fortran_order": False, "shape": shape}, "canonical array shape/dtype changed")
                digest, count = hashlib.sha256(), 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    count += len(chunk)
                    checks.require(count <= expected and len(chunk) % struct.calcsize(fmt) == 0, "array payload size mismatch")
                    checks.require(all(math.isfinite(x[0]) for x in struct.iter_unpack(fmt, chunk)), "nonfinite landmark/delta value")
                    digest.update(chunk)
                checks.require(count == expected, "truncated array payload")
                results[name] = {"dtype": dtype, "shape": list(shape), "bytes": count, "sha256": digest.hexdigest(), "finite": True}
    return results


def array_hashes(path):
    try:
        return _array_hashes(path)
    except checks.Invalid:
        raise
    except (zipfile.BadZipFile, SyntaxError, UnicodeError, struct.error, ValueError) as exc:
        raise checks.Invalid("malformed landmark/delta capture") from exc


def identity(environment):
    engine = environment["engines"][0]
    return {"gpu_uuid": environment["gpu"]["uuid"], "actual_gpu": environment["actual_gpu"],
            "compute_capability": environment["compute_capability"], "runtime": environment["runtime"],
            "input_manifest_sha256": environment["input_manifest_sha256"], "input_count": environment["input_count"],
            "profile_sha256": environment["profile_sha256"], "effective_profile": environment["effective_profile"],
            "engine_root": engine["root"], "manifest_sha256": engine["manifest_sha256"], "plan_sha256": engine["plan_sha256"],
            "taesd": environment["taesd"]}


def current_harness_hashes():
    files = [ROOT / "scripts/chin_multistream_render.py", *sorted((ROOT / "scripts/chin_multistream").glob("*.py"))]
    return {p.name: checks.sha256(p) for p in files}


def capture(path, environment, selected):
    path = Path(path).resolve()
    data = checks.read(path)
    pairs = quality_envelope.validate_capture(data)
    checks.require(data["args"].get("tracking_overlap", False) is selected, "capture tracking mode mismatch")
    label = data["label"]
    checks.require(isinstance(label, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", label) and label not in (".", ".."), "unsafe capture label")
    root = path.parent / label
    checks.require(not root.is_symlink() and root.resolve().parent == path.parent, "capture directory escapes report parent")
    engine = environment["engines"][0]
    checks.loaded_backend(data, engine["root"], environment["taesd"]["key"], environment["taesd"]["decoder_plan_sha256"], engine["manifest"])
    checks.require(data["harness_code_sha256"] == current_harness_hashes(), "capture harness source changed")
    integrity = data["code_integrity"]
    checks.require(set(integrity["files"]) == {"backend.py", "tracker_worker.py", "chin.py", "render_stage.py"}, "canonical source coverage missing")
    checks.require(all(re.fullmatch(r"[A-Za-z0-9_]+\.py", name) for name in integrity["files"]), "unsafe canonical source name")
    checks.require(all(checks.sha256(ROOT / "character_factory/h3_avatar_workflow" / name) == digest
                       for name, digest in integrity["files"].items())
                   and checks.sha256(ROOT / "musetalk/utils/blending.py") == integrity["blending_py"], "canonical source changed after capture")
    checks.require(len(data["repeats"]) == 1 and data["repeats"][0]["frames"] == 1440, "capture completed frame coverage mismatch")
    workers = data["repeats"][0]["per_worker"]
    checks.require(len(workers) == 6, "capture missing workers")
    rows = {}
    for index, ident in enumerate(checks.IDENTITIES):
        worker = workers[str(index)]
        checks.require(worker["identity"] == ident and worker["frames"] == 240
                       and worker.get("tracking_overlap", False) is selected, "capture worker mode/identity/frames mismatch")
        clips = worker["clips"]
        checks.require(len(clips) == 1 and clips[0]["loop"] == 0, "capture worker clip coverage mismatch")
        checks.require([clips[0]["raw_refined_sha256"], clips[0]["generated_faces_sha256"]] == pairs[ident][0], "capture worker and summary hashes disagree")
        prefix = root / f"stream{index:02d}_{ident}"
        faces, arrays = Path(str(prefix) + "_faces.npz"), Path(str(prefix) + "_arrays.npz")
        checks.require(all(p.is_file() and not p.is_symlink() for p in (faces, arrays)), "missing or unsafe raw capture file")
        try:
            face_sha = quality_envelope.face_array_hash(faces)
        except (zipfile.BadZipFile, SyntaxError, UnicodeError) as exc:
            raise checks.Invalid("malformed face capture") from exc
        checks.require(face_sha == pairs[ident][0][1], "saved faces differ from reported completed capture")
        rows[ident] = {"generated_faces_sha256": face_sha, "raw_refined_sha256": pairs[ident][0][0],
                       "arrays": array_hashes(arrays), "faces_file_sha256": checks.sha256(faces), "arrays_file_sha256": checks.sha256(arrays)}
    return data, {"path": str(path), "report_sha256": checks.sha256(path), "avatars": rows}


def compare(serial_path, overlap_path, environment):
    checks.require(Path(serial_path).resolve() != Path(overlap_path).resolve(), "paired captures must be distinct")
    serial, a = capture(serial_path, environment, False)
    overlap, b = capture(overlap_path, environment, True)
    ignored = {"tracking_overlap", "label", "out_root"}
    checks.require({k: v for k, v in serial["args"].items() if k not in ignored}
                   == {k: v for k, v in overlap["args"].items() if k not in ignored}, "paired workload settings changed")
    for key in ("code_integrity", "harness_code_sha256", "backends", "versions", "env"):
        checks.require(serial[key] == overlap[key], f"paired {key} identity changed")
    matches = {}
    for ident in checks.IDENTITIES:
        left, right = a["avatars"][ident], b["avatars"][ident]
        matches[ident] = {k: left[k] == right[k] for k in ("generated_faces_sha256", "raw_refined_sha256")}
        matches[ident].update({name: left["arrays"][name] == right["arrays"][name] for name in ARRAYS})
    return {"schema": SCHEMA, "status": "PASS" if all(all(row.values()) for row in matches.values()) else "FAIL",
            "scope": "exact_same_engine_scheduler_pair_only_not_release_quality_or_throughput",
            "identity": identity(environment), "harness_code_sha256": serial["harness_code_sha256"],
            "parity_tool_sha256": checks.sha256(__file__), "serial": a, "overlap": b, "matches": matches,
            "frames_per_identity": 240, "identities": list(checks.IDENTITIES), "release_quality_accepted": False}


def require_passed(path, environment):
    parent = checks.read(path)
    checks.require(parent.get("schema") == "repro_3090_v1" and parent.get("suite") == "tracking-parity"
                   and parent.get("status") == "PASS" and len(parent.get("results", [])) == 1, "successful tracking-parity suite required")
    row = parent["results"][0]
    checks.require(row.get("schema") == SCHEMA and row.get("status") == "PASS"
                   and row["identity"] == identity(environment), "tracking-parity identity differs from this GPU/input/engine/profile")
    # Re-read the actual captures/NPZs; an edited receipt cannot assert parity.
    actual = compare(row["serial"]["path"], row["overlap"]["path"], environment)
    checks.require(actual == row and actual["status"] == "PASS", "tracking-parity evidence changed or fails")
    return {"path": str(Path(path).resolve()), "sha256": checks.sha256(path), "verified": True}
