#!/usr/bin/env python3
"""Separate production-cache render audit; never a six-fixture or visual PASS.

Default is a CPU-only plan. --execute-render runs a guarded GPU child after the
operator has frozen the benchmark/quality candidate. No download, preparation,
cache repair, resize, model build, server launch, or avatar-directory write occurs.

All 16 characters x 3 poses are selected by default. --avatar-id is an explicitly
incomplete diagnostic. Use a new --out for every attempt. GPU integration of this
adapter remains unverified until an actual run is inspected.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import report as checks
import runner
from scripts.audit_3090_avatar_contents import publication_items

N, FPS = 240, 24
CONTACT_INDICES = (0, 48, 96, 144, 192, 239)
CRITICAL = (
    "character_factory/h3_avatar_workflow/chin.py",
    "character_factory/h3_avatar_workflow/tracker_worker.py",
    "musetalk/utils/blending.py", "musetalk/utils/audio_processor.py",
    "musetalk/models/unet.py", "scripts/chin_multistream/gpu.py",
    "scripts/video_ab_chin_render.py", "scripts/unet_stagewise_trt.py",
    "scripts/vae_fast_decoder.py",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_items(publication, selected=()):
    items = publication_items(publication)
    ids = {row["avatar_id"] for row in items}
    checks.require(len(selected) == len(set(selected)), "duplicate selected avatar")
    checks.require(set(selected) <= ids, "selected avatar absent from publication")
    return [row for row in items if not selected or row["avatar_id"] in selected]


def independent_output(out, avatars_root):
    out, avatars_root = Path(out).resolve(), Path(avatars_root).resolve()
    checks.require(not out.is_relative_to(avatars_root) and not avatars_root.is_relative_to(out),
                   "output and avatar roots must be disjoint")
    checks.require(not out.exists(), "output exists; use a new output directory")


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": checks.sha256(path)}


def make_plan(publication_path, avatars_root, audio, audio_sha256, speech_source, selected=()):
    checks.require(re.fullmatch(r"[a-f0-9]{64}", audio_sha256) is not None, "invalid expected audio SHA256")
    checks.require(bool(speech_source.strip()), "known real-speech provenance description required")
    checks.require(checks.sha256(audio) == audio_sha256, "audio differs from supplied real-speech reference")
    publication = checks.read(publication_path)
    items = select_items(publication, selected)
    for item in items:
        item["cache_path"] = str((Path(avatars_root) / item["avatar_id"]).resolve())
        checks.require(Path(item["cache_path"]).parent == Path(avatars_root).resolve(), "cache escaped avatar root")
        item["cache_present"] = Path(item["cache_path"]).is_dir()
    return {
        "schema": "rtx3090_production_pose_render_v1", "status": "PLAN_ONLY",
        "scope": "offline production-cache render compatibility only",
        "expected_characters": 16, "expected_poses": 48, "selected_poses": len(items),
        "coverage_complete": False, "full_48_render_compatibility": "NOT_EVALUATED",
        "diagnostic_subset": len(items) != 48, "visual_review_status": "NOT_PERFORMED",
        "release_ready": False, "gpu_used": False,
        "publication": file_record(publication_path), "avatars_root": str(Path(avatars_root).resolve()),
        "audio": {**file_record(audio), "declared_real_speech_source": speech_source,
                  "speech_content_verified_automatically": False, "start_seconds": 0, "duration_seconds": N / FPS},
        "recipe": {"frames": N, "fps": FPS, "native_frame_shape_hwc": [896, 512, 3],
                   "generated_face_shape_hwc": [256, 256, 3], "chin_strength": 1.0,
                   "composition": "full saved face box and mask; no mouth-only or extra cheek/jaw attenuation",
                   "chin": "frozen chin.prepare_refined + chin.prepare + chin.corrected_refined",
                   "temporal_filter": "existing symmetric .25/.5/.25 with edge replication and [0,.18] clamp",
                   "cache_reencoding": False, "source_cycle_start": 0, "resize_or_crop_output": False,
                   "cache_encoder_provenance": "not inferred from latent shape; separate audit required"},
        "not_claimed": ["visual approval", "quality equivalence to the six canonical fixtures",
                        "live API chin-path parity", "live pose transitions", "capacity or FPS acceptance",
                        "multilingual speech coverage", "restored cache encoder provenance"],
        "contact_sample_frames": list(CONTACT_INDICES), "items": items, "results": [],
    }


class _IntegerDType:
    """Non-executing stand-in for the only NumPy dtype used by coordinate caches."""
    def __init__(self, code, *unused):
        checks.require(code in ("i4", "i8", "u4", "u8"), "unsupported coordinate scalar dtype")
        self.code, self.order = code, "little" if sys.byteorder == "little" else "big"

    def __setstate__(self, state):
        checks.require(isinstance(state, tuple) and len(state) == 8 and state[0] == 3
                       and state[1] in ("<", ">", "=") and state[2:] == (None, None, None, -1, -1, 0),
                       "unsupported coordinate dtype state")
        if state[1] != "=":
            self.order = "little" if state[1] == "<" else "big"


def _integer_scalar(dtype, raw):
    checks.require(type(dtype) is _IntegerDType and type(raw) is bytes
                   and len(raw) == int(dtype.code[1]), "invalid coordinate scalar")
    return int.from_bytes(raw, byteorder=dtype.order, signed=dtype.code[0] == "i")


class _CoordinatesOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("numpy", "dtype"):
            return _IntegerDType
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return _integer_scalar
        raise checks.Invalid("coordinate pickle requested forbidden global")


def coordinates(path, count):
    data = Path(path).read_bytes()
    checks.require(len(data) <= 4 * 1024 * 1024, "coordinate pickle unexpectedly large")
    stream = io.BytesIO(data)
    value = _CoordinatesOnly(stream).load()
    checks.require(not stream.read(), "trailing coordinate pickle data")
    checks.require(type(value) in (list, tuple) and len(value) == count, "coordinate cycle count mismatch")
    result = []
    for row in value:
        checks.require(type(row) in (list, tuple) and len(row) == 4
                       and all(type(v) is int for v in row), "invalid coordinate row")
        result.append(tuple(row))
    return result


def validate_geometry(box, cropbox, frame_shape, mask_shape):
    checks.require(tuple(frame_shape) == (896, 512, 3), "unsupported native resolution; resizing is forbidden")
    x0, y0, x1, y1 = box
    cx0, cy0, cx1, cy1 = cropbox
    checks.require(0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 896, "invalid face box")
    # Canonical PIL-expanded mask crops can legitimately extend outside the
    # image. The existing blend plan clips them; never alter/recompute the box.
    checks.require(cx0 <= x0 < x1 <= cx1 and cy0 <= y0 < y1 <= cy1
                   and 0 < cx1 - cx0 <= 2048 and 0 < cy1 - cy0 <= 2048,
                   "invalid saved mask crop box")
    checks.require(tuple(mask_shape) == (cy1 - cy0, cx1 - cx0), "mask does not match saved crop box")


def cache_files(item):
    base = Path(item["cache_path"])
    checks.require(base.is_dir(), "published cache not restored")
    frames = sorted((base / "full_imgs").glob("*.png"))
    masks = sorted((base / "mask").glob("*.png"))
    checks.require(len(frames) >= N and [p.name for p in frames] == [p.name for p in masks],
                   "cache frame/mask coverage mismatch or fewer than 240 frames")
    checks.require([p.name for p in frames] == [f"{i:08d}.png" for i in range(len(frames))],
                   "noncanonical frame indices")
    fixed = [base / name for name in ("avator_info.json", "latents.pt", "coords.pkl", "mask_coords.pkl", "input_video.mp4")]
    paths = fixed + frames[:N] + masks[:N]
    checks.require(all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(base.resolve()) for p in paths),
                   "missing or symlinked cache input")
    info = checks.read(fixed[0])
    checks.require(info.get("avatar_id") == item["avatar_id"] and info.get("version") == item["version"],
                   "cache metadata identity mismatch")
    checks.require(info.get("fixed_face_height") is False, "fixed-face-height cache not approved by this audit")
    checks.require(checks.sha256(base / "input_video.mp4") == item["expected_source_video_sha256"],
                   "source video differs from publication")
    return base, frames, masks, [file_record(p) for p in paths]


def load_cache(item, torch, np, cv2):
    base, frame_paths, mask_paths, inventory = cache_files(item)
    count = len(frame_paths)
    boxes, cropboxes = coordinates(base / "coords.pkl", count), coordinates(base / "mask_coords.pkl", count)
    frames, masks = [], {}
    for i in range(N):
        frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_paths[i]), cv2.IMREAD_UNCHANGED)
        checks.require(frame is not None and mask is not None and mask.dtype == np.uint8, "PNG decode failed")
        if mask.ndim == 3:
            checks.require(mask.shape[2] == 3 and np.array_equal(mask[:, :, 0], mask[:, :, 1])
                           and np.array_equal(mask[:, :, 0], mask[:, :, 2]), "non-grayscale saved mask")
            mask = mask[:, :, 0]
        validate_geometry(boxes[i], cropboxes[i], frame.shape, mask.shape)
        frames.append(frame)
        masks[str(i)] = mask
    value = torch.load(base / "latents.pt", map_location="cpu", weights_only=True)
    if isinstance(value, (list, tuple)):
        checks.require(bool(value) and all(torch.is_tensor(v) for v in value), "invalid latent list")
        checks.require(all(tuple(v.shape) in ((1, 8, 32, 32), (8, 32, 32)) for v in value), "invalid latent sample shape")
        value = torch.stack([v.reshape(8, 32, 32) for v in value])
    checks.require(torch.is_tensor(value) and tuple(value.shape) in ((count, 8, 32, 32), (count, 1, 8, 32, 32)),
                   "latent cycle shape mismatch")
    checks.require(value.dtype in (torch.float16, torch.float32, torch.bfloat16), "invalid latent dtype")
    latent = value[:N].reshape(N, 8, 32, 32).contiguous()
    checks.require(bool(torch.isfinite(latent).all()), "nonfinite selected latents")
    d = {"cache": {"latents": latent, "boxes": boxes[:N], "cropboxes": cropboxes[:N]}, "masks": masks}
    return frames, d, inventory, {"cycle_count": count, "selected_indices": list(range(N)),
                                 "latent_dtype": str(value.dtype), "latent_finiteness_scope": "selected 240 frames only",
                                 "native_frame_shape_hwc": list(frames[0].shape)}


def verify_frozen_inputs(manifest_path, paths):
    manifest_path = Path(manifest_path).resolve()
    entries = checks.read(manifest_path)["files"]
    lookup = {}
    for row in entries:
        path = (manifest_path.parent / row["path"]).resolve()
        checks.require(path not in lookup, "duplicate frozen input")
        lookup[path] = row["sha256"]
    for path in paths:
        path = Path(path).resolve()
        checks.require(path in lookup and checks.sha256(path) == lookup[path], "unfrozen or changed runtime/audio/recipe input")
    return {"manifest": file_record(manifest_path), "verified_files": [file_record(p) for p in paths]}


def runtime_policy(args):
    values = runner.profile(args.profile)
    values.update(MUSETALK_UNET_BACKEND="trt_stagewise", MUSETALK_UNET_STAGEWISE_BATCH="16",
                  MUSETALK_UNET_STAGEWISE_CACHE_DIR=str(Path(args.engine_root).resolve()),
                  MUSETALK_UNET_STAGEWISE_VERIFY_SHA="1", MUSETALK_UNET_STAGEWISE_PROBE_CHECK="1",
                  MUSETALK_UNET_STAGEWISE_PROBE_TOL="0", MUSETALK_TRT_FALLBACK="0",
                  MUSETALK_VAE_BACKEND="taesd", MUSETALK_TAESD_BACKEND="trt",
                  MUSETALK_TAESD_TRT_BUILD="0", MUSETALK_TAESD_TRT_STRICT="1", MUSETALK_TAESD_TRT_BATCH="8",
                  MUSETALK_TAESD_TRT_DIR=str(Path(args.taesd_dir).resolve()), MUSETALK_TAESD_WARMUP_BATCHES="8",
                  MUSETALK_SOURCE_MOUTH_BLEND="0", MUSETALK_SIDE_JAW_BLEND="0", MUSETALK_CHEEK_ONLY_BLEND="0",
                  MUSETALK_FIXED_FACE_HEIGHT="0", MUSETALK_BLEND_FIXED_POINT="1", MUSETALK_BLEND_SHRINK_MASK_BBOX="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return values


def frozen_runtime(args):
    weights = sorted(p for p in Path(args.whisper_root).rglob("*") if p.is_file())
    checks.require(bool(weights), "missing local Whisper model")
    return verify_frozen_inputs(args.frozen_inputs, [ROOT / p for p in CRITICAL] + weights + [args.audio])


def runtime_identity(args, torch):
    import tensorrt
    import torch_tensorrt
    checks.require(torch.cuda.device_count() == 1, "exactly one CUDA device required")
    name = torch.cuda.get_device_name(0)
    cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
    checks.gpu_identity(name, cc, False)
    checks.require(torch.__version__ == "2.5.1+cu121" and tensorrt.__version__.startswith("10.3.")
                   and torch_tensorrt.__version__.startswith("2.5."), "runtime is not pinned r5 matrix")
    engine = runner.engine(args.engine_root)
    m = engine["manifest"]
    compat = m.get("hardware_compatibility_level") or "none"
    checks.require(compat in ("none", "ampere_plus"), "unsupported UNet compatibility")
    if compat == "none":
        checks.require(m.get("gpu") == name and str(m.get("compute_capability")).replace(".", "").replace(",", "").replace(" ", "").strip("[]()") == "86",
                       "native UNet target mismatch")
    meta_path = Path(args.taesd_dir) / f"taesd_trt_{args.taesd_key}.json"
    meta = checks.read(meta_path)
    fp = meta["fingerprint"]
    checks.require(meta.get("key") == args.taesd_key and hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:20] == args.taesd_key,
                   "TAESD fingerprint/key mismatch")
    checks.require(fp["batch"] == 8 and fp["tensorrt"] == tensorrt.__version__, "TAESD runtime/batch mismatch")
    checks.require(fp.get("hardware_compatibility_level", "none") == os.environ.get("MUSETALK_TAESD_TRT_HW_COMPAT", "none")
                   and fp["opt_level"] == int(os.environ.get("MUSETALK_TAESD_TRT_OPT_LEVEL", "3")), "TAESD profile mismatch")
    if fp.get("hardware_compatibility_level", "none") == "none":
        checks.require(fp["gpu"] == name and fp["compute_capability"] == cc, "TAESD native target mismatch")
    for kind in ("decoder", "post"):
        checks.require(checks.sha256(Path(args.taesd_dir) / meta[kind + "_plan"]) == meta[kind + "_plan_sha256"], "corrupt TAESD plan")
    return {"gpu": name, "compute_capability": cc, "torch": torch.__version__, "tensorrt": tensorrt.__version__,
            "torch_tensorrt": torch_tensorrt.__version__, "engine": engine, "taesd": meta,
            "taesd_metadata": file_record(meta_path)}


def speech_conditioning(args, torch):
    from transformers import WhisperModel
    from musetalk.models.unet import PositionalEncoding
    from musetalk.utils.audio_processor import AudioProcessor
    processor = AudioProcessor(str(args.whisper_root))
    whisper = WhisperModel.from_pretrained(str(args.whisper_root), local_files_only=True).half().cuda().eval()
    pe = PositionalEncoding(d_model=384).cuda().half().eval()
    features, samples = processor.get_audio_feature(str(args.audio))
    checks.require(samples >= 16000 * N / FPS, "real speech is shorter than ten seconds")
    with torch.inference_mode():
        chunks = processor.get_whisper_chunk(features, "cuda", torch.float16, whisper, samples, fps=FPS,
                                             audio_padding_length_left=2, audio_padding_length_right=2)
        chunks = chunks[:N] if torch.is_tensor(chunks) else torch.stack(chunks[:N])
        audio = pe(chunks.cuda().half()).cpu().contiguous()
    checks.require(tuple(audio.shape) == (N, 50, 384) and bool(torch.isfinite(audio).all()), "invalid audio conditioning")
    del whisper, pe, processor, features, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return audio


def validate_source_timeline(stream, cycle_count):
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    source_count = int(stream["nb_frames"])
    checks.require(denominator > 0 and numerator == FPS * denominator
                   and (stream["width"], stream["height"]) == (512, 896) and source_count > 0,
                   "source timeline/resolution differs from frozen 24fps recipe")
    # APIAvatar._process_frames persists forward + reverse source cycles.
    # A short smiling source can still have >=240 approved saved cache frames.
    # Never duplicate/resize new frames here: use the existing cycle unchanged.
    checks.require(cycle_count == 2 * source_count and cycle_count >= N,
                   "saved cache is not the canonical source forward/reverse cycle")
    return {**stream, "cache_cycle_frames": cycle_count, "cache_cycle_mapping": "forward_plus_reverse",
            "selected_cache_frames": N, "new_frames_generated_or_resampled": False}


def source_timeline(item, cycle_count):
    source = Path(item["cache_path"]) / "input_video.mp4"
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(source)],
                                  text=True, stderr=subprocess.PIPE)
    streams = json.loads(raw)["streams"]
    checks.require(len(streams) == 1, "source video stream missing")
    return validate_source_timeline(streams[0], cycle_count)


def render_pose(args, item, audio, unet, decoder, identity, video, chin, torch, np, cv2):
    from scripts.chin_multistream.gpu import GpuIssuer, describe_backends
    dest = Path(args.out) / item["avatar_id"]
    dest.mkdir()
    row = {"avatar_id": item["avatar_id"], "character": item["character"], "pose": item["pose"],
           "status": "INVALID", "visual_review_status": "NOT_PERFORMED", "release_ready": False}
    tracker = None
    try:
        frames, d, inventory, cache_meta = load_cache(item, torch, np, cv2)
        row.update(cache=cache_meta, input_inventory=inventory,
                   source_timeline=source_timeline(item, cache_meta["cycle_count"]))
        d["cache"]["audio"] = audio
        tracker = video.Tracker(ROOT, Path(args.tracker_python), dest / "tracker.log")
        tracker.reset()
        d["p"] = np.asarray([tracker.track(f, f, (0, 0, 512, 896))[0] for f in frames], np.float32)
        checks.require(d["p"].shape == (N, 478, 2) and bool(np.isfinite(d["p"]).all()), "invalid source landmarks")
        d["g"] = np.zeros_like(d["p"])
        chin.prepare_refined(d, frames)
        with torch.inference_mode():
            issuer = GpuIssuer(unet, decoder, .18215, {"pose": d["cache"]["latents"].cuda()},
                               {"pose": audio.cuda()}, pack=16, decode_split=8, depth=1)
            faces = []
            tracker.reset()
            for base in range(0, N, 16):
                job = issuer.submit([("pose", base, None), ("pose", base + 8, None)])
                batch = issuer.wait(job).copy()
                issuer.release(job)
                checks.require(batch.shape == (16, 256, 256, 3), "generated face shape mismatch")
                for j, face in enumerate(batch):
                    d["g"][base + j] = tracker.track(frames[base + j], face, d["cache"]["boxes"][base + j])[0]
                    faces.append(face)
            del issuer
        checks.require(bool(np.isfinite(d["g"]).all()), "nonfinite generated landmarks")
        # Reuse the frozen full-sequence implementation of the exact same three-tap
        # lookahead filter. This audit is offline and makes no scheduling/FPS claim.
        chin.prepare(d)
        outputs = [chin.corrected_refined(f, d, i, faces[i]) for i, f in enumerate(frames)]
        checks.require(all(o.shape == f.shape and o.dtype == np.uint8 for o, f in zip(outputs, frames)), "output resolution/dtype changed")
        row["backends"] = describe_backends(unet, decoder)
        checks.loaded_backend(row, args.engine_root, args.taesd_key, identity["taesd"]["decoder_plan_sha256"], identity["engine"]["manifest"])
        row["recipe_checks"] = video.recipe_checks(chin, frames, d, faces, outputs)
        row["clip"] = video.store_clip(dest, outputs, FPS)
        # Mux the same real speech without modifying the lossless video samples.
        preview = dest / "review_with_audio.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-n", "-i", str(dest / "frames.mkv"),
                        "-i", str(args.audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "flac", "-t", str(N / FPS), str(preview)], check=True)
        row["review_clip"] = file_record(preview)
        np.save(dest / "source_landmarks.npy", d["p"])
        np.save(dest / "generated_landmarks.npy", d["g"])
        np.save(dest / "chin_delta.npy", d["chin_delta"])
        contacts, contact_rows = [], []
        for i in CONTACT_INDICES:
            standard = chin.standard(frames[i], d, i, faces[i])
            contacts.append(np.concatenate([frames[i], standard, outputs[i]], axis=1))
            contact_rows.append({"frame": i, "time_seconds": i / FPS, "source_cycle_index": i,
                                 "source_png": str(Path(item["cache_path"]) / "full_imgs" / f"{i:08d}.png"),
                                 "face_box_xyxy": list(map(int, d["cache"]["boxes"][i])),
                                 "source_pixels_sha256": video.sha_array(frames[i]),
                                 "standard_pixels_sha256": video.sha_array(standard),
                                 "refined_pixels_sha256": video.sha_array(outputs[i])})
        contact_path = dest / "contact_source_standard_refined.png"
        checks.require(cv2.imwrite(str(contact_path), np.concatenate(contacts, axis=0)), "contact PNG write failed")
        row["contact_samples"] = {"file": file_record(contact_path), "columns": ["source", "standard", "refined_chin_100_percent"],
                                   "tile_shape_hwc": [896, 512, 3], "downscaled": False, "rows": contact_rows}
        row["array_artifacts"] = [file_record(dest / name) for name in ("source_landmarks.npy", "generated_landmarks.npy", "chin_delta.npy")]
        checks.require(all(checks.sha256(entry["path"]) == entry["sha256"] for entry in inventory), "cache changed during audit")
        row["inputs_unchanged_after_render"] = True
        row["status"] = "PASS" if row["recipe_checks"]["passes"] else "FAIL"
    except Exception as exc:
        # Do not persist upstream exception text, environment values, or credentials.
        row["error_type"] = type(exc).__name__
        if isinstance(exc, checks.Invalid):
            row["reason"] = str(exc)
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception as exc:
                row.update(status="INVALID", tracker_cleanup_error=type(exc).__name__)
        write_json(dest / "pose.json", row)
        gc.collect()
        torch.cuda.empty_cache()
    return row


def summarize(rows, selected_count):
    complete = selected_count > 0 and len(rows) == selected_count and len({r["avatar_id"] for r in rows}) == selected_count
    status = "INVALID" if not complete or any(r["status"] not in ("PASS", "FAIL") for r in rows) else "FAIL" if any(r["status"] == "FAIL" for r in rows) else "PASS"
    full = complete and selected_count == 48
    return {"status": status, "coverage_complete": full,
            "full_48_render_compatibility": status if full else "INCOMPLETE",
            "visual_review_status": "NOT_PERFORMED", "release_ready": False}


def worker(args):
    out = Path(args.out)
    data = checks.read(out / "report.json")
    checks.require(data["status"] == "PLAN_ONLY", "worker refuses stale/non-plan output")
    # The public launcher creates this child under the existing guard/watch process.
    cmdline = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    checks.require(str(HERE / "watch.py").encode() in cmdline, "worker requires GPU-isolation watchdog")
    data["status"] = "RUNNING"
    write_json(out / "report.json", data)
    try:
        checks.require(checks.sha256(args.publication) == data["publication"]["sha256"], "publication changed after plan")
        checks.require(checks.sha256(args.audio) == data["audio"]["sha256"], "audio changed after plan")
        data["frozen_inputs"] = frozen_runtime(args)
        import torch
        import numpy as np
        import cv2
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        os.chdir(ROOT)
        data["gpu_used"] = True
        identity = runtime_identity(args, torch)
        data["runtime"] = identity
        # Import helpers only after explicit environment sanitation. Do not call
        # their historical setup/env-file loader or six-identity input loader.
        from scripts import video_ab_chin_render as video
        from scripts.chin_multistream.gpu import describe_backends
        chin, _ = video.load_chin(ROOT)
        audio = speech_conditioning(args, torch)
        data["audio"]["conditioning_sha256"] = video.sha_array(audio.numpy())
        torch.save(audio, out / "audio_conditioning.pt")
        data["audio"]["conditioning_file"] = file_record(out / "audio_conditioning.pt")
        unet, decoder, _, _ = video.setup_backends(ROOT)
        data["backends"] = describe_backends(unet, decoder)
        checks.loaded_backend(data, args.engine_root, args.taesd_key, identity["taesd"]["decoder_plan_sha256"], identity["engine"]["manifest"])
        write_json(out / "report.json", data)
        for item in data["items"]:
            row = render_pose(args, item, audio, unet, decoder, identity, video, chin, torch, np, cv2)
            data["results"].append(row)
            write_json(out / "report.json", data)
            print(json.dumps({"avatar_id": row["avatar_id"], "status": row["status"], "completed": len(data["results"])}), flush=True)
        data.update(summarize(data["results"], data["selected_poses"]))
        frozen_runtime(args)
        checks.require(runner.engine(args.engine_root)["manifest_sha256"] == identity["engine"]["manifest_sha256"], "engines changed during audit")
        checks.require(checks.sha256(identity["taesd_metadata"]["path"]) == identity["taesd_metadata"]["sha256"], "TAESD metadata changed during audit")
    except Exception as exc:
        data.update(status="INVALID", full_48_render_compatibility="INVALID", error_type=type(exc).__name__)
        if isinstance(exc, checks.Invalid):
            data["reason"] = str(exc)
    write_json(out / "report.json", data)
    return {"PASS": 0, "FAIL": 1}.get(data["status"], 2)


def normalize_paths(args):
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # A venv interpreter is commonly a symlink to /usr/bin/python.
            # Resolving it changes sys.prefix and silently discards its packages.
            setattr(args, key, value.absolute() if key == "tracker_python" else value.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("publication", "avatars-root", "audio", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--speech-source", required=True, help="known real-speech source/provenance; not an automatic speech classifier")
    parser.add_argument("--avatar-id", action="append", default=[], help="diagnostic subset only; never full-48 acceptance")
    parser.add_argument("--execute-render", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    for flag in ("profile", "engine-root", "taesd-dir", "tracker-python", "whisper-root", "frozen-inputs"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--taesd-key")
    args = parser.parse_args(argv)
    normalize_paths(args)
    if args._worker:
        return worker(args)
    independent_output(args.out, args.avatars_root)
    data = make_plan(args.publication, args.avatars_root, args.audio, args.audio_sha256, args.speech_source, args.avatar_id)
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", data)
    if not args.execute_render:
        print(json.dumps({"status": "PLAN_ONLY", "selected_poses": data["selected_poses"], "gpu_used": False}))
        return 0
    try:
        checks.require(all(getattr(args, key) for key in ("profile", "engine_root", "taesd_dir", "tracker_python", "whisper_root", "frozen_inputs", "taesd_key")),
                       "execution requires explicit frozen inputs/profile/engine/TAESD/Whisper/tracker paths")
        checks.require(re.fullmatch(r"[0-9a-f]{20}", args.taesd_key) is not None, "invalid TAESD key")
        checks.require(all(i["cache_present"] for i in data["items"]), "selected published caches are not all restored")
        frozen_runtime(args)
        values = runtime_policy(args)
        env, removed = runner.clean_environment(os.environ, values)
        data["effective_profile"] = values
        data["discarded_ambient_keys"] = removed
        data["adapter"] = file_record(__file__)
        write_json(args.out / "report.json", data)
        args.label = "production_pose_audit"
        command = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
        for key, value in vars(args).items():
            if key in ("_worker", "execute_render", "label"):
                continue
            if key == "avatar_id":
                for avatar_id in value:
                    command += ["--avatar-id", avatar_id]
            else:
                command += ["--" + key.replace("_", "-"), str(value)]
        code = runner.child(args, args.out.resolve(), env, "production_pose_render", command, gb=8)
        data = checks.read(args.out / "report.json")
        if code not in (0, 1) or data["status"] not in ("PASS", "FAIL"):
            data.update(status="INVALID", full_48_render_compatibility="INVALID", child_returncode=code)
            write_json(args.out / "report.json", data)
            return 2
        return code
    except Exception as exc:
        data.update(status="INVALID", error_type=type(exc).__name__)
        if isinstance(exc, checks.Invalid):
            data["reason"] = str(exc)
        write_json(args.out / "report.json", data)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
