#!/usr/bin/env python3
"""Prepare 30 genuinely new static-portrait caches using production MuseTalk code.

This is a static identity generalization benchmark: the source head/body do not
move. Production DWPose/S3FD, SD-VAE and jaw masks are used unchanged. One source
PNG becomes the production two-frame forward/reverse cache; the reviewer cycles
it for 250 output frames. No server, UNet, S3, or registration is started.

Run from a fresh owned CUDA worker with the full avatar-prep environment:
  python scripts/ios/prepare_new_identity_review.py --characters characters.json \
    --portrait-root portraits --audio data/audio/ai-assistant.mpga data/audio/eng.wav \
    --checkpoint round2_bk_tiny_best.pt --checkpoint-sha256 <64hex> --out /isolated/out

--dry-run only checks small metadata/inputs and hashes files; it imports no Torch
or production model code. --resume accepts identical inputs/configuration only;
--rebuild-incomplete explicitly permits rebuilding an incomplete/corrupt cache
inside this benchmark's own output tree.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
COUNT, FPS, FRAMES = 30, 25, 250
CHUNK = 1 << 20
PROTOCOL = "new_static_avatars_r5_v1"
NORMALIZATION = {"width": 512, "height": 896, "method": "EXIF transpose, RGB composite, aspect-preserving Lanczos resize, centered neutral padding",
                 "padding_rgb": [128, 128, 128], "crop": False}
PREP_SETTINGS = {"version": "v15", "extra_margin": 10, "parsing_mode": "jaw", "bbox_shift": 0,
                 "left_cheek_width": 90, "right_cheek_width": 90, "vae_dtype": "float16",
                 "unet_loaded": False, "source_frames": 1, "cycle_frames": 2,
                 "cuda_matmul_tf32": True, "cudnn_tf32": True, "cudnn_benchmark": True}
PREP_ENV = {"MUSETALK_FIXED_FACE_HEIGHT": "0", "MUSETALK_SOURCE_MOUTH_BLEND": "0", "MUSETALK_SIDE_JAW_BLEND": "0"}
CODE_FILES = ("scripts/api_avatar.py", "musetalk/models/vae.py", "musetalk/utils/preprocessing.py",
              "musetalk/utils/blending.py", "musetalk/utils/mmlab_compat.py",
              "musetalk/utils/face_parsing/__init__.py", "musetalk/utils/face_parsing/model.py",
              "musetalk/utils/face_detection/api.py", "musetalk/utils/face_detection/detection/sfd/sfd_detector.py",
              "musetalk/utils/face_parsing/resnet.py",
              "musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py")
MODEL_FILES = ("models/sd-vae/config.json", "models/dwpose/dw-ll_ucoco_384.pth",
               "models/face_detection/s3fd.pth", "models/face-parse-bisent/79999_iter.pth",
               "models/face-parse-bisent/resnet18-5c106cde.pth")
SAFE_ID = re.compile(r"[a-z][a-z0-9_]{2,63}")


class PrepError(Exception):
    pass


def require(ok, message):
    if not ok:
        raise PrepError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def freeze(source, dest, expected):
    require(not dest.is_symlink(), "frozen input cannot be a symlink")
    if dest.exists():
        require(dest.is_file() and sha(dest) == expected, "frozen input differs; use a new output directory")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=dest.name + ".", suffix=".part", dir=dest.parent, delete=False) as dst:
        temp = Path(dst.name)
        try:
            with source.open("rb") as src:
                shutil.copyfileobj(src, dst, CHUNK)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        require(sha(temp) == expected, "input changed while freezing")
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def blocked_id(aid):
    identity = re.split(r"_(?:idle|smiling|talking)_", aid, maxsplit=1)[0]
    # Entire identities, including all three poses, are held out of this new set.
    old = {"en_01_maya_abe451d3", "en_02_claire_a833ee1f", "en_03_marcus_19eb1ea8", "en_04_owen_4fd8c9da",
           "es_01_valeria_aed74c50", "es_02_lucia_e554c30b", "es_03_mateo_884600e1", "es_04_diego_532cfc1f",
           "ur_01_ayesha_1d9f8dcf", "ur_02_zara_2fe6a913", "ur_03_hamza_0c24ed7f", "ur_04_bilal_b9b0df23",
           "zh_01_mei_ca138def", "zh_02_lan_09242cdb", "zh_03_wei_44dc1943", "zh_04_jun_7e6a6f0b",
           "japanese_realtime", "latina_guided_20260925", "chin_h3_japanese_new", "chin_h3_latina_new",
           "chinese_bob_pink_bedroom", "indian_realtime", "div_black_woman", "div_east_asian_man_goatee",
           "div_middle_eastern_man_full_beard", "div_white_man_clean_shaven", "div_south_asian_woman",
           "div_black_man_short_beard", "codex_smoke"}
    return any(identity == name or identity.startswith(name + "_") for name in old)


def inspect_inputs(args):
    require(re.fullmatch(r"[a-fA-F0-9]{64}", args.checkpoint_sha256 or "") is not None, "checkpoint SHA256 must be 64 hex characters")
    require(args.seed >= 0, "seed must be nonnegative")
    root = args.portrait_root.resolve(strict=True)
    require(root.is_dir(), "portrait root is not a directory")
    characters_bytes = args.characters.read_bytes()
    config = json.loads(characters_bytes)
    characters = config.get("characters") if isinstance(config, dict) else config
    require(isinstance(characters, list) and len(characters) == COUNT, "characters JSON must contain exactly 30 characters")
    ids, hashes, rows = set(), set(), []
    for character in characters:
        require(isinstance(character, dict), "each character must be an object")
        aid = character.get("id")
        require(isinstance(aid, str) and SAFE_ID.fullmatch(aid) and not blocked_id(aid), "character ID is invalid or belongs to an old identity")
        require(aid not in ids, "character IDs must be unique")
        ids.add(aid)
        rel = character.get("portrait", f"{aid}.png")
        require(isinstance(rel, str), "portrait must be a filename relative to portrait root")
        source = (root / rel).resolve(strict=True)
        require(source.is_relative_to(root) and source.is_file(), "portrait must be a file inside portrait root")
        require(source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}, "unsupported portrait image type")
        digest = sha(source)
        if "portrait_sha256" in character:
            accepted = character["portrait_sha256"]
            require(isinstance(accepted, str) and re.fullmatch(r"[a-f0-9]{64}", accepted) is not None,
                    "declared portrait_sha256 must be 64 lowercase hex characters")
            require(digest == accepted, "portrait differs from its accepted generation SHA256")
        require(digest not in hashes, "30 distinct portrait byte hashes are required")
        hashes.add(digest)
        rows.append({"id": aid, "source": str(source), "sha256": digest, "bytes": source.stat().st_size,
                     "suffix": source.suffix.lower(), "metadata": character})
    audio, audio_hashes, audio_names = [], set(), set()
    for value in args.audio:
        path = value.resolve(strict=True)
        require(path.is_file() and re.fullmatch(r"[A-Za-z0-9_.-]+", path.name), "audio filename is invalid")
        require(path.stem not in audio_names, "audio stems must be distinct")
        audio_names.add(path.stem)
        digest = sha(path)
        require(digest not in audio_hashes, "audio clips must be distinct")
        audio_hashes.add(digest)
        audio.append({"source": str(path), "name": path.name, "sha256": digest, "bytes": path.stat().st_size})
    require(len(audio) == 2, "exactly two audio clips are required")
    checkpoint = args.checkpoint.resolve(strict=True)
    require(checkpoint.is_file() and sha(checkpoint) == args.checkpoint_sha256.lower(), "checkpoint differs from expected SHA256")
    code, models, missing_models = {}, {}, []
    for rel in CODE_FILES:
        path = REPO / rel
        require(path.is_file(), "production prep code is missing")
        code[rel] = sha(path)
    code["scripts/ios/prepare_new_identity_review.py"] = sha(__file__)
    model_paths = list(MODEL_FILES)
    vae_weights = [rel for rel in ("models/sd-vae/diffusion_pytorch_model.safetensors", "models/sd-vae/diffusion_pytorch_model.bin") if (REPO / rel).is_file()]
    if not vae_weights:
        missing_models.append("models/sd-vae/diffusion_pytorch_model.{bin,safetensors}")
    model_paths += vae_weights
    for rel in model_paths:
        if not (REPO / rel).is_file():
            missing_models.append(rel)
        else:
            models[rel] = sha(REPO / rel)
    require(args.dry_run or not missing_models, "production prep weights are missing; use the full CUDA avatar-prep environment")
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "mmpose", "mmcv", "mmengine", "mmdet", "pillow", "opencv-python", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    runtime = {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "packages": packages}
    payload = {"schema_version": 1, "protocol": PROTOCOL, "avatar_kind": "static_portrait", "fps": FPS, "frames": FRAMES,
               "characters_json_sha256": hashlib.sha256(characters_bytes).hexdigest(), "seed": args.seed, "normalization": NORMALIZATION,
               "prep": PREP_SETTINGS, "prep_environment": PREP_ENV, "runtime": runtime,
               "production_code_sha256": code, "model_sha256": models,
               "characters": rows, "audio": audio, "checkpoint": str(checkpoint), "checkpoint_sha256": args.checkpoint_sha256.lower()}
    return payload, missing_models


def normalize(source, destination):
    from PIL import Image, ImageOps
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        original_size = list(image.size)
        if image.mode == "RGBA" or "transparency" in image.info:
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, tuple(NORMALIZATION["padding_rgb"]))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = image.convert("RGB")
        fitted = ImageOps.contain(rgb, (512, 896), method=Image.Resampling.LANCZOS)
        pad = [(512 - fitted.width) // 2, (896 - fitted.height) // 2]
        canvas = Image.new("RGB", (512, 896), tuple(NORMALIZATION["padding_rgb"]))
        canvas.paste(fitted, tuple(pad))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".part")
        canvas.save(temp, format="PNG")
        os.replace(temp, destination)
    return {"original_dimensions": original_size, "resized_dimensions": list(fitted.size), "padding_left_top": pad,
            "normalized_dimensions": [512, 896], "normalized_sha256": sha(destination), "normalization": NORMALIZATION}


def reviewed_cache_files(cache):
    require(cache.is_dir() and not cache.is_symlink(), "avatar cache is not a regular directory")
    files = {}
    for rel in ("coords.pkl", "mask_coords.pkl", "latents.pt"):
        path = cache / rel
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, "avatar cache is incomplete")
        files[rel] = sha(path)
    image_groups = []
    for folder in ("full_imgs", "mask"):
        require((cache / folder).is_dir() and not (cache / folder).is_symlink(), "avatar image directory is incomplete")
        paths = sorted((cache / folder).glob("*.[jpJP][pnPN]*[gG]"))
        require(len(paths) == 2, "production one-frame prep must produce a two-frame static cycle")
        group = []
        for path in paths:
            require(path.is_file() and not path.is_symlink(), "avatar reviewed images cannot be links")
            digest = sha(path)
            files[path.relative_to(cache).as_posix()] = digest
            group.append(digest)
        require(len(set(group)) == 1, "static source cycle images or masks differ")
        image_groups.append(len(paths))
    require(image_groups[0] == image_groups[1], "frame/mask cycle counts differ")
    return files


@contextlib.contextmanager
def cwd(path):
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


@contextlib.contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_production_preparer():
    # Everything heavy is lazy, so --help and --dry-run do not import Torch.
    sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "musetalk/utils")]
    import torch
    from types import SimpleNamespace
    from scripts.api_avatar import APIAvatar
    from musetalk.models.vae import VAE
    from musetalk.utils.face_parsing import FaceParsing
    require(torch.cuda.is_available(), "production preparation requires the approved owned CUDA worker")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    vae = VAE(model_path=str(REPO / "models/sd-vae"), use_float16=True)
    vae.vae.eval().requires_grad_(False)
    parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    # Preload lazy production models while relative code/weight paths resolve
    # inside the clone. Later constructor calls reuse this module unchanged.
    from musetalk.utils import preprocessing  # noqa: F401
    settings = SimpleNamespace(version="v15", extra_margin=10, parsing_mode="jaw")

    def prepare(aid, source_dir, cache, seed):
        runtime = cache.parents[3]
        require(cache == runtime / "results/v15/avatars" / aid, "unexpected isolated production cache path")
        runtime.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with cwd(runtime), torch.inference_mode():
            # Call the original decorated constructor without subclassing or
            # monkey-patching any production code. Only CWD is isolated.
            avatar = APIAvatar(avatar_id=aid, video_path=str(source_dir), bbox_shift=0,
                               batch_size=1, vae=vae, unet=None, pe=None, fp=parser, args=settings,
                               preparation=True, force_recreate=False)
            require(Path(avatar.avatar_path).resolve() == cache, "production prep escaped isolated cache path")
        del avatar
    return prepare


def execute(args, payload, missing_models, preparer_factory=load_production_preparer):
    out = args.out.expanduser().resolve()
    # A benchmark cannot be placed in production caches or overwrite portrait inputs.
    require(out != REPO and not (REPO / "results").resolve().is_relative_to(out)
            and not out.is_relative_to((REPO / "results").resolve()), "output must be isolated from production results")
    sources = ([Path(row["source"]) for row in payload["characters"]]
               + [Path(row["source"]) for row in payload["audio"]]
               + [Path(payload["checkpoint"]), args.characters.resolve()])
    require(not any(source.is_relative_to(out) for source in sources), "source inputs must live outside the output directory")
    request_sha = signature(payload)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "characters": COUNT, "audio": 2, "protocol": PROTOCOL,
                          "request_sha256": request_sha, "missing_local_model_files": missing_models,
                          "production_models_loaded": False, "static_pose_only": True}, sort_keys=True))
        return
    require(not args.rebuild_incomplete or args.resume, "--rebuild-incomplete also requires --resume")
    request_file = out / "prep_inputs.json"
    if out.exists() and any(out.iterdir()):
        require(args.resume and request_file.is_file(), "nonempty output requires --resume and matching prep_inputs.json")
        saved = json.loads(request_file.read_text())
        require(saved.get("request_sha256") == request_sha and saved.get("inputs") == payload,
                "resume input/configuration hashes differ; use a new output directory")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(request_file, {"request_sha256": request_sha, "inputs": payload})
    for path in out.rglob("*"):
        require(not path.is_symlink(), "benchmark output tree cannot contain links")
    freeze(args.characters.resolve(), out / "characters.original.json", payload["characters_json_sha256"])
    audio_rows = []
    for row in payload["audio"]:
        target = out / "audio" / row["name"]
        freeze(Path(row["source"]), target, row["sha256"])
        audio_rows.append({"file": str(target), "sha256": row["sha256"]})
    avatar_rows, latent_hashes, frame_hashes, prepare = [], set(), set(), None
    with cwd(REPO), environment(PREP_ENV):
        for index, row in enumerate(payload["characters"]):
            aid = row["id"]
            original = out / "portraits_original" / (aid + row["suffix"])
            freeze(Path(row["source"]), original, row["sha256"])
            source_dir = out / "normalized" / aid
            normalized = source_dir / "00000000.png"
            norm_record = out / "normalized" / aid / "normalization.json"
            norm, normalized_ready = None, False
            if normalized.is_file() and norm_record.is_file():
                try:
                    norm = json.loads(norm_record.read_text())
                    normalized_ready = (norm.get("source_sha256") == row["sha256"]
                                        and norm.get("request_sha256") == request_sha
                                        and norm.get("normalized_sha256") == sha(normalized))
                except (ValueError, OSError, AttributeError):
                    normalized_ready = False
            if not normalized_ready:
                require(not (normalized.exists() or norm_record.exists()) or args.rebuild_incomplete,
                        "incomplete/corrupt normalized portrait; explicitly resume with --rebuild-incomplete")
                norm = normalize(original, normalized)
                norm.update(source_sha256=row["sha256"], request_sha256=request_sha)
                write_json(norm_record, norm)
            cache = out / "runtime/results/v15/avatars" / aid
            marker = cache / "benchmark_cache.json"
            ready, cache_files = False, None
            if marker.is_file():
                saved = json.loads(marker.read_text())
                require(saved.get("request_sha256") == request_sha, "cache configuration differs")
                try:
                    cache_files = reviewed_cache_files(cache)
                    ready = saved.get("cache_files") == cache_files and saved.get("normalized_sha256") == norm["normalized_sha256"]
                except PrepError:
                    ready = False
            if cache.exists() and not ready:
                require(args.rebuild_incomplete, "incomplete/corrupt cache; explicitly resume with --rebuild-incomplete")
                require(cache.parent == out / "runtime/results/v15/avatars" and cache.name == aid and not cache.is_symlink(), "unsafe rebuild target")
                shutil.rmtree(cache)
            if not ready:
                if prepare is None:
                    prepare = preparer_factory()
                prepare(aid, source_dir, cache, args.seed + index)
                cache_files = reviewed_cache_files(cache)
                write_json(marker, {"request_sha256": request_sha, "id": aid, "seed": args.seed + index,
                                    "normalized_sha256": norm["normalized_sha256"], "cache_files": cache_files})
            require(cache_files["latents.pt"] not in latent_hashes, "latent cache hashes must be unique across the 30 identities")
            latent_hashes.add(cache_files["latents.pt"])
            first_frame = next(digest for path, digest in sorted(cache_files.items()) if path.startswith("full_imgs/"))
            require(first_frame not in frame_hashes, "first-frame image hashes must be unique across the 30 identities")
            frame_hashes.add(first_frame)
            avatar_rows.append({"id": aid, "portrait": str(original), "portrait_sha256": row["sha256"],
                                "cache_dir": str(cache), "cache_files": cache_files,
                                "normalized_portrait": str(normalized), "normalization": norm,
                                "metadata": row["metadata"], "seed": args.seed + index})
            print(f"prepared {index + 1}/{COUNT}: {aid}" + (" (verified resume)" if ready else ""), flush=True)
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "avatar_kind": "static_portrait", "fps": FPS, "frames": FRAMES,
                "checkpoint_sha256": payload["checkpoint_sha256"], "checkpoint_file": payload["checkpoint"],
                "avatars": avatar_rows, "audio": audio_rows, "request_sha256": request_sha,
                "normalization": NORMALIZATION, "production_prep": PREP_SETTINGS,
                "limitation": "New static identity generalization only. Source head/body are frozen; this does not validate natural head motion, new H3 videos, or personalities."}
    write_json(out / "new_avatar_manifest.json", manifest)
    print(json.dumps({"status": "complete", "characters": COUNT, "audio": 2,
                      "manifest": str(out / "new_avatar_manifest.json"), "request_sha256": request_sha,
                      "checkpoint_sha256": payload["checkpoint_sha256"], "static_pose_only": True}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characters", type=Path, required=True)
    parser.add_argument("--portrait-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs=2, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        payload, missing = inspect_inputs(args)
        execute(args, payload, missing)
    except (PrepError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"preparation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
