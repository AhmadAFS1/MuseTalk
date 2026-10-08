#!/usr/bin/env python3
"""Read-only S3 cache audit into a NEW local root; no GPU or render-quality claim.

Downloads all 48 published caches only with --execute-download. Restoration uses
AvatarS3Store, including its canonical archive safety checks. No existing avatar
directory is reused, repaired, or replaced. Requires boto3, Pillow, and torch.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.avatar_s3_store import AvatarS3Store


POSES = {"idle", "talking", "smiling"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
SCHEMA = "rtx3090_avatar_content_audit_v1"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def publication_items(manifest, expected_characters=16):
    """Validate coverage and exact canonical S3 mapping before any download."""
    characters = manifest.get("characters")
    if not isinstance(characters, list) or len(characters) != expected_characters:
        raise ValueError("manifest_character_coverage")
    seen_characters, seen_avatars, items = set(), set(), []
    for character in characters:
        identifier = character.get("id", "")
        if not IDENTIFIER.fullmatch(identifier) or identifier in seen_characters:
            raise ValueError("manifest_character_identity")
        seen_characters.add(identifier)
        if set(character.get("poses", {})) != POSES:
            raise ValueError("manifest_pose_coverage")
        for pose in sorted(POSES):
            cache = character["poses"][pose]
            avatar_id = cache.get("avatar_id", "")
            if not IDENTIFIER.fullmatch(avatar_id) or avatar_id in seen_avatars:
                raise ValueError("manifest_avatar_identity")
            seen_avatars.add(avatar_id)
            parsed = urlsplit(cache.get("s3_uri", ""))
            parts = parsed.path.lstrip("/").split("/")
            if (parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment
                    or parsed.username or len(parts) < 2 or any(p in {"", ".", ".."} for p in parts)
                    or parts[-1] != avatar_id + ".tar.gz"):
                raise ValueError("manifest_s3_mapping")
            if not SHA256.fullmatch(cache.get("source_video_sha256", "")):
                raise ValueError("manifest_source_video_hash")
            if type(cache.get("bytes")) is not int or cache["bytes"] <= 0:
                raise ValueError("manifest_archive_size")
            metadata = cache.get("metadata", {})
            if metadata.get("avatar-id") != avatar_id or metadata.get("musetalk-version") != parts[-2]:
                raise ValueError("manifest_canonical_metadata")
            if cache.get("archive_sha256") and not SHA256.fullmatch(cache["archive_sha256"]):
                raise ValueError("manifest_archive_hash")
            items.append({
                "character": identifier, "pose": pose, "avatar_id": avatar_id,
                "s3_uri": cache["s3_uri"], "bucket": parsed.netloc,
                "key": parsed.path.lstrip("/"), "prefix": "/".join(parts[:-2]),
                "version": parts[-2], "expected_archive_bytes": cache["bytes"],
                "expected_archive_sha256": cache.get("archive_sha256"),
                "expected_source_video_sha256": cache["source_video_sha256"],
                "recorded_etag": cache.get("etag"),
            })
    return items


class HashingDownloadClient:
    """Observe the canonical downloader's tempfile before canonical cleanup."""
    def __init__(self, client, item):
        self.client, self.item, self.archive = client, item, None

    def download_file(self, bucket, key, filename):
        if (bucket, key) != (self.item["bucket"], self.item["key"]):
            raise ValueError("unexpected_download_target")
        self.client.download_file(bucket, key, filename)
        self.archive = {"bytes": Path(filename).stat().st_size, "sha256": digest(filename)}
        self.archive["size_matches_publication"] = self.archive["bytes"] == self.item["expected_archive_bytes"]
        expected = self.item["expected_archive_sha256"]
        self.archive["sha256_matches_publication"] = self.archive["sha256"] == expected if expected else None
        self.archive["historical_archive_sha256_available"] = bool(expected)
        if not self.archive["size_matches_publication"] or self.archive["sha256_matches_publication"] is False:
            raise ValueError("archive_integrity_mismatch")


def inspect_images(paths, image_module):
    """Decode every PNG on CPU, and fingerprint filename/content inventory."""
    shapes, modes, inventory = set(), set(), hashlib.sha256()
    for path in paths:
        with image_module.open(path) as img:
            if img.format != "PNG":
                raise ValueError("non_png_frame")
            img.verify()
        # verify() checks structure; load() forces actual pixel decompression.
        with image_module.open(path) as img:
            img.load()
            if img.width <= 0 or img.height <= 0:
                raise ValueError("invalid_image_dimensions")
            shapes.add((img.width, img.height))
            modes.add(img.mode)
        inventory.update((path.name + "\0" + digest(path) + "\n").encode())
    if not paths:
        raise ValueError("missing_images")
    return {"count": len(paths), "dimensions_width_height": sorted(shapes),
            "modes": sorted(modes), "all_pngs_decoded": True,
            "filename_content_inventory_sha256": inventory.hexdigest()}


def inspect_latents(path, torch_module, expected_count):
    """Never fall back to unsafe pickle or use an accelerator device."""
    value = torch_module.load(str(path), map_location="cpu", weights_only=True)
    is_tensor = torch_module.is_tensor
    if is_tensor(value):
        storage = "tensor"
        shape = tuple(value.shape)
        if len(shape) not in {4, 5} or not shape or shape[0] == 0:
            raise ValueError("latent_cycle_shape")
        tensors = (value,)
        count = shape[0]
        sample_shape = shape[1:]
    elif isinstance(value, (list, tuple)) and value and all(is_tensor(t) for t in value):
        storage, tensors, count = type(value).__name__, value, len(value)
        sample_shape = tuple(value[0].shape)
        shape = (count,) + sample_shape
        if any(tuple(t.shape) != sample_shape for t in value):
            raise ValueError("inconsistent_latent_shapes")
    else:
        raise ValueError("unsupported_latent_container")
    if sample_shape not in {(8, 32, 32), (1, 8, 32, 32)}:
        raise ValueError("unexpected_musetalk_latent_shape")
    if count != expected_count:
        raise ValueError("latent_frame_count_mismatch")
    dtypes = set()
    for tensor in tensors:
        if tensor.device.type != "cpu":
            raise ValueError("latent_not_on_cpu")
        dtype = str(tensor.dtype)
        if dtype not in {"torch.float16", "torch.float32", "torch.bfloat16"}:
            raise ValueError("unexpected_latent_dtype")
        dtypes.add(dtype)
        # Bound temporary finite-check allocation for large consolidated cycles.
        for chunk in tensor.split(32, dim=0):
            if not bool(torch_module.isfinite(chunk).all().item()):
                raise ValueError("nonfinite_latents")
    if len(dtypes) != 1:
        raise ValueError("inconsistent_latent_dtypes")
    return {"sha256": digest(path), "storage": storage, "shape": list(shape),
            "cycle_count": count, "dtypes": sorted(dtypes), "all_finite": True,
            "load_map_location": "cpu", "load_weights_only": True,
            "gpu_used": False, "shape_interpretation": "cycle, optional singleton batch, 8 channels, 32x32"}


def inspect_restored(item, avatar_dir, torch_module, image_module):
    info_path = avatar_dir / "avator_info.json"
    info = json.loads(info_path.read_text())
    if not isinstance(info, dict) or info.get("avatar_id") != item["avatar_id"]:
        raise ValueError("avatar_metadata_identity")
    if info.get("version") != item["version"]:
        raise ValueError("avatar_metadata_version")
    video = avatar_dir / "input_video.mp4"
    video_sha = digest(video)
    if video_sha != item["expected_source_video_sha256"]:
        raise ValueError("source_video_hash_mismatch")
    frames, masks = sorted((avatar_dir / "full_imgs").glob("*.png")), sorted((avatar_dir / "mask").glob("*.png"))
    if [p.name for p in frames] != [p.name for p in masks]:
        raise ValueError("frame_mask_inventory_mismatch")
    frame_info, mask_info = inspect_images(frames, image_module), inspect_images(masks, image_module)
    latent_info = inspect_latents(avatar_dir / "latents.pt", torch_module, len(frames))
    selected = {k: info[k] for k in ("avatar_id", "version", "bbox_shift", "fixed_face_height",
                "fixed_face_height_anchor_px", "video_layout") if k in info}
    provenance_keys = ("encoder_checkpoint_sha256", "vae_checkpoint_sha256", "preprocess_revision",
                       "preprocess_config_sha256", "source_video_sha256")
    provenance = {key: {"status": "RECORDED_NOT_INDEPENDENTLY_VERIFIED", "value": info[key]}
                  if key in info else {"status": "UNAVAILABLE_IN_CACHE_METADATA"} for key in provenance_keys}
    samples = []
    for idx in sorted({0, len(frames) // 2, len(frames) - 1}):
        samples.append({"cycle_index": idx, "frame": str(frames[idx]), "mask": str(masks[idx]),
                        "frame_sha256": digest(frames[idx]), "mask_sha256": digest(masks[idx])})
    return {
        "metadata": {"sha256": digest(info_path), "keys": sorted(info), "selected_fields": selected},
        "source_video": {"path": str(video), "bytes": video.stat().st_size, "sha256": video_sha,
                         "matches_publication": True, "decoded_or_visually_reviewed": False},
        "frames": frame_info, "masks": mask_info, "latents": latent_info, "provenance": provenance,
        "coordinate_pickles": {name: {"sha256": digest(avatar_dir / name), "bytes": (avatar_dir / name).stat().st_size,
                                    "loaded": False, "reason": "No unrestricted pickle deserialization; coordinate semantics not audited"}
                               for name in ("coords.pkl", "mask_coords.pkl")},
        "visual_contact_samples": samples, "visual_review_status": "NOT_PERFORMED",
        "render_compatibility_status": "NOT_TESTED", "gpu_performance_status": "NOT_TESTED",
    }


def audit_item(item, avatars_root, client, torch_module, image_module, region=None):
    start = time.monotonic()
    result = {**item, "started_utc": utc_now(), "status": "INVALID"}
    observed = HashingDownloadClient(client, item)
    try:
        target = avatars_root / item["avatar_id"]
        if target.exists() or target.is_symlink():
            raise ValueError("refuse_existing_avatar_target")
        store = AvatarS3Store(enabled=True, bucket=item["bucket"], prefix=item["prefix"],
                              version=item["version"], region=region, client=observed,
                              retry_attempts=1, log_fn=lambda message: None)
        if not store.download_avatar_dir(item["avatar_id"], avatars_root):
            # Deliberately do not expose canonical exception strings/credential context.
            result["reason"] = "canonical_restore_failed"
        else:
            result["restored_path"] = str(target)
            result["contents"] = inspect_restored(item, target, torch_module, image_module)
            result["status"] = "PASS"
    except ValueError as exc:
        # Only our allowlisted short codes, never arbitrary upstream exception text.
        code = str(exc)
        result["reason"] = code if re.fullmatch(r"[a-z_]{1,80}", code) else "content_validation_failed"
        result["status"] = "FAIL"
    except Exception as exc:
        result["reason"] = type(exc).__name__
    result["archive"] = observed.archive
    result["elapsed_local_monotonic_seconds"] = time.monotonic() - start
    result["finished_utc"] = utc_now()
    return result


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".avatar-audit-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_audit(manifest_path, output, audit_root, *, execute_download=False,
              profile=None, region=None, minimum_free_gib=2.0, cpu_threads=2):
    raw = manifest_path.read_bytes()
    items = publication_items(json.loads(raw))
    if audit_root.exists() or audit_root.is_symlink():
        raise ValueError("audit_root_must_not_exist")
    if output.exists() or output.is_symlink():
        raise ValueError("output_must_not_exist")
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("invalid_minimum_free_space")
    report = {
        "schema": SCHEMA, "status": "PLAN_ONLY", "started_utc": utc_now(),
        "manifest": str(manifest_path.resolve()), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_root": str(audit_root.resolve()), "expected_characters": 16, "expected_pose_caches": 48,
        "expected_archive_download_bytes": sum(i["expected_archive_bytes"] for i in items),
        "credential_reference": profile or "boto3 default credential chain", "gpu_used": False,
        "scope": "Canonical restore, fresh archive hash/size, published source-video hash, PNG decode/inventory, safe CPU latent shape/dtype/finiteness",
        "not_claimed": ["historical archive SHA match without published hash", "coordinate-pickle semantics",
                        "encoder/preprocessing compatibility without provenance", "visual quality", "render compatibility", "GPU performance"],
        "expanded_disk_requirement_known": False, "objects": [],
    }
    if not execute_download:
        write_report(output, report)
        return report
    import boto3
    import torch
    from PIL import Image
    torch.set_num_threads(cpu_threads)
    from botocore.config import Config
    client = boto3.Session(profile_name=profile, region_name=region).client(
        "s3", config=Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3, "mode": "standard"}))
    # mkdir(exist_ok=False) is essential: canonical store may replace targets.
    audit_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    avatars_root = audit_root / "avatars"
    avatars_root.mkdir(mode=0o700)
    report["status"] = "RUNNING"
    report["runtime"] = {"torch": torch.__version__, "pillow": Image.__version__, "cpu_threads": cpu_threads}
    write_report(output, report)
    for item in items:
        # Conservative per-object headroom; expansion ratio cannot be proven from HEAD.
        free = shutil.disk_usage(audit_root).free
        if free < item["expected_archive_bytes"] * 3 + minimum_free_gib * 1024 ** 3:
            report["objects"].append({**item, "status": "INVALID", "reason": "insufficient_disk_headroom", "free_bytes": free})
            break
        result = audit_item(item, avatars_root, client, torch, Image, region)
        report["objects"].append(result)
        write_report(output, report)
        print(json.dumps({"completed": len(report["objects"]), "of": 48, "avatar_id": item["avatar_id"], "status": result["status"]}), flush=True)
    counts = {status: sum(r["status"] == status for r in report["objects"]) for status in ("PASS", "FAIL", "INVALID")}
    report["counts"] = counts
    report["coverage_complete"] = len(report["objects"]) == 48
    report["status"] = "INVALID" if counts["INVALID"] or not report["coverage_complete"] else "FAIL" if counts["FAIL"] else "PASS"
    report["finished_utc"] = utc_now()
    write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--execute-download", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    try:
        report = run_audit(args.manifest, args.out, args.audit_root, execute_download=args.execute_download,
                           profile=args.profile, region=args.region, minimum_free_gib=args.minimum_free_gib,
                           cpu_threads=args.cpu_threads)
        print(json.dumps({"status": report["status"], "report": str(args.out),
                          "expected_download_bytes": report["expected_archive_download_bytes"]}))
        return 0 if report["status"] in {"PLAN_ONLY", "PASS"} else 2
    except Exception as exc:
        print(json.dumps({"status": "INVALID", "reason": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
