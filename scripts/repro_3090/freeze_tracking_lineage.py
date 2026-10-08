#!/usr/bin/env python3
"""Explicit successor input lineage, never a rewrite of original quality evidence.

All 878 paths remain mandatory. Only the exact reviewed scheduler-worker revision
and valid Hugging Face cache metadata for byte-identical model payloads may differ.
This authorizes a same-engine scheduler pair, not native release acceptance.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA = "6d3ab6ef31605c2231605e03042e82361a27112589ef6d7f6f8ff8f4b016eea5"
WORKER = "../../../../scripts/chin_multistream/worker.py"
WORKER_BEFORE = "2e6e88fbe106ca964b2b5e31af44203ce5dff201eb2c37436726b65520cc447b"
WORKER_AFTER = "935fcd94323bdd3605b819db36816932fc18cab249184b0e2647e42cba4707db"
METADATA = {
    "models/.cache/huggingface/download/auxiliary/s3fd-619a316812.pth.metadata": "models/auxiliary/s3fd-619a316812.pth",
    **{f"models/.cache/huggingface/download/musetalkV15/{n}.metadata": f"models/musetalkV15/{n}"
       for n in ("musetalk.json", "unet.pth")},
    **{f"models/{group}/.cache/huggingface/download/{n}.metadata": f"models/{group}/{n}"
       for group, names in {
           "dwpose": ("dw-ll_ucoco_384.pth",),
           "sd-vae": ("config.json", "diffusion_pytorch_model.bin"),
           "syncnet": ("latentsync_syncnet.pt",),
           "taesd": ("config.json", "diffusion_pytorch_model.safetensors"),
           "whisper": ("config.json", "preprocessor_config.json", "pytorch_model.bin"),
       }.items() for n in names},
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def derive(parent, base, repo):
    rows, changes = [], []
    originals = {r["path"]: r for r in parent["files"]}
    if len(originals) != len(parent["files"]):
        raise ValueError("duplicate frozen path")
    for before in parent["files"]:
        path = (base / before["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"missing required frozen input: {before['path']}")
        after = {"path": before["path"], "bytes": path.stat().st_size, "sha256": digest(path)}
        if after != before:
            relative = before["path"].removeprefix("../../../../")
            if before["path"] == WORKER and before["sha256"] == WORKER_BEFORE and after["sha256"] == WORKER_AFTER:
                reason = "exact reviewed default-off ordered tracking worker; canonical math source unchanged"
            elif relative in METADATA:
                payload = repo / METADATA[relative]
                payload_row = originals.get("../../../../" + METADATA[relative])
                if not payload_row or digest(payload) != payload_row["sha256"] or payload.stat().st_size != payload_row["bytes"]:
                    raise ValueError("metadata cannot excuse changed model payload")
                fields = path.read_text().splitlines()
                if len(fields) != 3 or not re.fullmatch(r"[a-f0-9]{40}", fields[0]) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", fields[1]):
                    raise ValueError("invalid Hugging Face metadata")
                timestamp = float(fields[2])
                if not math.isfinite(timestamp) or timestamp <= 0:
                    raise ValueError("invalid download timestamp")
                if len(fields[1]) == 64:
                    expected_etag = payload_row["sha256"]
                else:
                    raw = payload.read_bytes()
                    expected_etag = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                if fields[1] != expected_etag:
                    raise ValueError("metadata etag does not identify frozen payload")
                reason = "fresh non-runtime download metadata; associated payload equals original frozen bytes"
            else:
                raise ValueError(f"unapproved input change: {before['path']}")
            changes.append({"before": before, "after": after, "reason": reason})
        rows.append(after)
    return {**parent, "files": rows}, changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "out", "receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    a = p.parse_args()
    parent_path, out, receipt = (x.resolve() for x in (a.parent, a.out, a.receipt))
    if out.exists() or receipt.exists() or out.parent != parent_path.parent or out == receipt:
        p.error("new same-directory manifest and distinct new receipt required")
    raw = parent_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        p.error("not the original immutable reference-input manifest")
    parent = json.loads(raw)
    if len(parent["files"]) != 878:
        p.error("original 878-file coverage required")
    successor, changes = derive(parent, parent_path.parent, a.repo_root.resolve())
    body = json.dumps(successor, indent=2) + "\n"
    evidence = {"schema": "explicit_tracking_successor_lineage_v1", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "parent_manifest_sha256": PARENT_SHA, "parent_manifest_modified": False,
                "parent_manifest_exact_match": False, "successor_manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "files": 878, "paths_removed": [], "runtime_payloads_match_original": True,
                "changes": changes, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.repo_root, text=True).strip(),
                "scope": "new same-engine serial/overlap scheduler comparison lineage only",
                "original_quality_bounds_modified": False, "native_release_accepted": False}
    with out.open("x") as stream:
        stream.write(body)
    with receipt.open("x") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "changes"}))


if __name__ == "__main__":
    main()
