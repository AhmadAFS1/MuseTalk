#!/usr/bin/env python3
"""Pack the phone's Core ML models for download: one plain tar per compiled model plus a manifest.

Input: a HarnessBundle directory (scripts/ios/prepare_harness_bundle.py), from which only the models the lipsync
path needs are taken: the batch-1 chunked UNet, the batch-1 TAESD, the Whisper log-mel and encoder, and a
meta.json trimmed to those entries (HarnessCore's `Harness` reads it unchanged).

Output: <out>/<name>.mlmodelc.tar for each model (ustar, uncompressed: the weights are already int8/fp16 blobs)
        <out>/meta.json
        <out>/manifest.json  {version, format, export_commit, created_at, meta: "meta.json",
                              files: [{name, file, bytes, sha256, kind}]}
The phone downloads the manifest, then each tar whose sha256 it does not already have, unpacks them next to
meta.json, and loads from that folder exactly as the harness loads from HarnessBundle.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def export_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def tar_dir(root: Path, arc_root: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for p in sorted(root.rglob("*")):
            info = tar.gettarinfo(str(p), arcname=f"{arc_root}/{p.relative_to(root).as_posix()}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644 if p.is_file() else 0o755
            if p.is_file():
                with p.open("rb") as f:
                    tar.addfile(info, f)
            else:
                tar.addfile(info)
    return buf.getvalue()


def select_models(models: list, unet: str) -> list:
    """Keep exactly the declared plain UNet or its chunked equivalent, plus its runtime helpers."""
    keep = []
    for model in models:
        kind = model.get('kind')
        if kind == 'unet_chunk' and model.get('chunk_of') == unet:
            keep.append(model)
        elif kind == 'unet' and model.get('name') in (unet, f'{unet}_chunked'):
            keep.append(model)
        elif kind == 'taesd' and model.get('batch') == 1:
            keep.append(model)
        elif kind in ('whisper_logmel', 'whisper_encoder'):
            keep.append(model)
    selected = [m for m in keep if m.get('kind') == 'unet']
    if len(selected) != 1:
        raise ValueError(f'expected exactly one selected UNet, found {len(selected)} for {unet}')
    if selected[0].get('name') == f'{unet}_chunked' and not any(m.get('kind') == 'unet_chunk' for m in keep):
        raise ValueError('chunked UNet has no chunk models')
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=REPO_ROOT / "ios" / "MuseTalkBench" / "HarnessBundle")
    ap.add_argument("--unet", default="unet_anec0_int8w_b1", help="chunk parent of the UNet to ship")
    ap.add_argument("--version", default="v1")
    ap.add_argument('--render-fps', type=int, choices=range(1, 61))
    ap.add_argument('--quality-profile')
    ap.add_argument('--checkpoint-sha256')
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((args.bundle / "meta.json").read_text())
    models = meta["models"]
    keep = select_models(models, args.unet)
    files = sorted({m["file"] for m in keep if m.get("file")})
    if not files:
        print("no models matched; check --unet and meta.json", file=sys.stderr)
        return 2
    # HarnessCore's `Harness.init` loads the three gate capture tensors eagerly (~9 MB); ship them so a downloaded
    # pack loads exactly like HarnessBundle and the gate check stays available on the phone.
    captures = [meta["unet"][k]["file"] for k in ("latents", "audio", "ref") if meta.get("unet", {}).get(k, {}).get("file")]
    files += [f for f in captures if f not in files]

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in files:
        src = args.bundle / f
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            return 2
        data = tar_dir(src, f) if src.is_dir() else None
        if data is None:
            data = src.read_bytes()
            tar_name = f
        else:
            tar_name = f + ".tar"
        (args.out / tar_name).write_bytes(data)
        kinds = sorted({m["kind"] for m in keep if m.get("file") == f})
        entries.append({"name": f, "file": tar_name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "kind": kinds[0] if kinds else ""})
        print(f"  {tar_name} {len(data) / 1e6:.1f} MB")

    # HarnessCore's BundleMeta needs `gate` and `unet` (the gate capture descriptions) to decode; the capture
    # files themselves are not shipped and are only read by the gate check, which the lipsync path never runs.
    trimmed = dict(meta)
    trimmed["models"] = keep
    for k in ("whisper", "mac_reference"):
        trimmed.pop(k, None)
    (args.out / "meta.json").write_text(json.dumps(trimmed, indent=2) + "\n")
    manifest = {
        "version": args.version,
        "format": "phone-model-pack/1",
        "unet": args.unet,
        "export_commit": export_commit(),
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta": "meta.json",
        "files": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }
    for name, value in (('render_fps', args.render_fps), ('quality_profile', args.quality_profile),
                         ('checkpoint_sha256', args.checkpoint_sha256)):
        if value is not None:
            manifest[name] = value
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {len(entries)} files, {manifest['total_bytes'] / 1e6:.0f} MB, export {manifest['export_commit'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
