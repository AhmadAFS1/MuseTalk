#!/usr/bin/env python3
"""Generate idle, talking, and smiling LTX 2.3 videos from one portrait.

This is the standalone, roster-free entry point for the three-pose Lingua avatar set.
It uses the accepted V6 idle, V14 talking-motion, and V8 smile prompts from
config/prompt_packs/pose_prompt_pack_v1.json. The same input image is conditioned at
both ends of every LTX render. Delivery files are silent H.264 videos whose decoded
first and last frames are verified to be identical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any


FACTORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = FACTORY_ROOT / "config/generation_defaults.json"
POSE_SPEC_PATH = FACTORY_ROOT / "config/pose_spec.json"
PROMPT_PACK_PATH = FACTORY_ROOT / "config/prompt_packs/pose_prompt_pack_v1.json"

POSES = {
    "idle": "idle_active_listening",
    "talking": "speaking_direct_v14_subtle",
    "smiling": "light_smile",
}

PRONOUNS = {
    # "The person" keeps the documented singular verbs grammatical while remaining neutral.
    "they": {"subject": "the person", "possessive": "their"},
    "she": {"subject": "she", "possessive": "her"},
    "he": {"subject": "he", "possessive": "his"},
}

END_GUIDE_NODE = "standalone_end_guide"
FIRST_PASS_CONDITION_NODES = ("5658:5352", "5658:5355")
SECOND_PASS_CONDITION_NODES = ("5667:5456", "5667:5451")
CONCAT_NODE = "5658:4528"


class RenderError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def capitalise_sentences(text: str) -> str:
    output: list[str] = []
    capitalise = True
    for character in text:
        output.append(character.upper() if capitalise and character.isalpha() else character)
        if character.isalpha():
            capitalise = False
        elif character in ".!?":
            capitalise = True
    return "".join(output)


def fill_standalone_prompt(text: str, pronoun: str) -> str:
    profile = PRONOUNS[pronoun]
    replacements = {
        "{pronoun_subject}": profile["subject"],
        "{pronoun_possessive}": profile["possessive"],
        "{subject}": "person shown in the reference image",
        "{wardrobe}": "the same clothing shown in the reference image",
        "{hair}": "the same hair shown in the reference image",
        "{setting}": "the setting shown in the reference image",
        "{lighting}": "the lighting shown in the reference image",
    }
    for token, value in replacements.items():
        text = text.replace(token, value)
    if re.search(r"\{[^{}]+\}", text):
        raise RenderError(f"Unresolved prompt placeholder in: {text}")
    return capitalise_sentences(text)


def render_specs(pose_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {item["render_key"]: item for item in pose_spec["renders"]}
    missing = set(POSES.values()) - set(specs)
    if missing:
        raise RenderError(f"Pose spec is missing documented renders: {', '.join(sorted(missing))}")
    return specs


def validate_node(workflow: dict[str, Any], node_id: str, class_type: str | None = None) -> None:
    if node_id not in workflow:
        raise RenderError(f"Workflow is missing required node {node_id}")
    if class_type and workflow[node_id].get("class_type") != class_type:
        actual = workflow[node_id].get("class_type")
        raise RenderError(f"Workflow node {node_id} is {actual!r}; expected {class_type!r}")


def add_last_frame_guide(workflow: dict[str, Any], nodes: dict[str, str], strength: float) -> None:
    first_guide = nodes["add_guide"]
    validate_node(workflow, first_guide, "LTXVAddGuide")
    validate_node(workflow, nodes["preprocess_image"], "LTXVPreprocess")

    workflow[first_guide]["inputs"]["frame_idx"] = 0
    workflow[first_guide]["inputs"]["strength"] = strength
    first_inputs = workflow[first_guide]["inputs"]
    workflow[END_GUIDE_NODE] = {
        "class_type": "LTXVAddGuide",
        "_meta": {"title": "Standalone matching last-frame guide"},
        "inputs": {
            "positive": [first_guide, 0],
            "negative": [first_guide, 1],
            "vae": copy.deepcopy(first_inputs["vae"]),
            "latent": [first_guide, 2],
            "image": copy.deepcopy(first_inputs["image"]),
            "frame_idx": -1,
            "strength": strength,
        },
    }

    for node_id in (*FIRST_PASS_CONDITION_NODES, *SECOND_PASS_CONDITION_NODES):
        validate_node(workflow, node_id)
        workflow[node_id]["inputs"]["positive"] = [END_GUIDE_NODE, 0]
        workflow[node_id]["inputs"]["negative"] = [END_GUIDE_NODE, 1]
    validate_node(workflow, CONCAT_NODE, "LTXVConcatAVLatent")
    workflow[CONCAT_NODE]["inputs"]["video_latent"] = [END_GUIDE_NODE, 2]


def build_workflow(
    *,
    base_workflow: dict[str, Any],
    defaults: dict[str, Any],
    prompt_pack: dict[str, Any],
    render: dict[str, Any],
    pose_name: str,
    pronoun: str,
    input_name: str,
    output_prefix: str,
    image_crc: int,
    pose_index: int,
    attempt: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = copy.deepcopy(base_workflow)
    nodes = defaults["node_ids"]
    sampling = defaults["sampling"]
    render_key = render["render_key"]
    pose_prompts = prompt_pack["renders"][render_key]

    required_nodes = {
        "load_image": "LoadImage",
        "duration_seconds_int": "INTConstant",
        "fps_float": "PrimitiveFloat",
        "prompt_relay_encode": "PromptRelayEncode",
        "negative_text_encode": "CLIPTextEncode",
        "video_combine": "VHS_VideoCombine",
    }
    for name, class_type in required_nodes.items():
        validate_node(workflow, nodes[name], class_type)

    global_template = prompt_pack["global_prompt_template"].replace(
        "The exact same person from the reference image, a {subject} wearing {wardrobe},",
        "The exact same person and clothing from the reference image,",
    )
    global_prompt = fill_standalone_prompt(global_template, pronoun)
    local_prompts = " | ".join(
        fill_standalone_prompt(segment, pronoun) for segment in pose_prompts["local_prompts"]
    )

    seed_offset = (image_crc % 100000) * 8 + pose_index + attempt * 1000
    stage1_seed = defaults["seeds"]["base_stage1"] + seed_offset
    stage2_seed = defaults["seeds"]["base_stage2"] + seed_offset

    workflow[nodes["load_image"]]["inputs"]["image"] = input_name
    workflow[nodes["duration_seconds_int"]]["inputs"]["value"] = render["seconds"]
    workflow[nodes["prompt_relay_encode"]]["inputs"]["global_prompt"] = global_prompt
    workflow[nodes["prompt_relay_encode"]]["inputs"]["local_prompts"] = local_prompts
    workflow[nodes["prompt_relay_encode"]]["inputs"]["segment_lengths"] = pose_prompts["segment_lengths"]
    workflow[nodes["negative_text_encode"]]["inputs"]["text"] = prompt_pack["negative_prompt"]
    workflow[nodes["preprocess_image"]]["inputs"]["img_compression"] = sampling["img_compression"]
    workflow[nodes["resize_image"]]["inputs"]["resize_type.width"] = sampling["resize_width"]
    workflow[nodes["attention_tuner"]]["inputs"]["audio_to_video_scale"] = sampling[
        "audio_to_video_scale"
    ]
    workflow[nodes["stage1_noise_seed"]]["inputs"]["noise_seed"] = stage1_seed
    workflow[nodes["stage2_noise_seed"]]["inputs"]["noise_seed"] = stage2_seed
    workflow[nodes["video_combine"]]["inputs"]["filename_prefix"] = output_prefix
    add_last_frame_guide(workflow, nodes, float(sampling["guide_strength"]))

    fps = int(workflow[nodes["fps_float"]]["inputs"]["value"])
    width = int(workflow[nodes["empty_latent_video"]]["inputs"]["width"])
    height = int(workflow[nodes["empty_latent_video"]]["inputs"]["height"])
    computed_frames = 1 + 8 * round((render["seconds"] * fps - 1) / 8)
    if computed_frames != render["frame_count"]:
        raise RenderError(
            f"{pose_name}: graph computes {computed_frames} frames, but the documented pose "
            f"requires {render['frame_count']}"
        )

    metadata = {
        "pose": pose_name,
        "render_key": render_key,
        "lineage": render["lineage"],
        "seconds_requested": render["seconds"],
        "frame_count": render["frame_count"],
        "duration_seconds": render["duration_seconds"],
        "fps": fps,
        "width": width,
        "height": height,
        "seed_pair": [stage1_seed, stage2_seed],
        "attempt": attempt,
        "global_prompt": global_prompt,
        "local_prompts": pose_prompts["local_prompts"],
        "local_prompt_resolved": local_prompts,
        "segment_lengths": pose_prompts["segment_lengths"],
        "negative_prompt": prompt_pack["negative_prompt"],
        "first_last_native_conditioning": True,
        "guide_frame_indices": [0, -1],
        "guide_strength": float(sampling["guide_strength"]),
    }
    return workflow, metadata


def find_video_output(history_record: dict[str, Any], comfy_output_dir: Path) -> Path:
    for output in history_record.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for item in output.get(key, []):
                filename = item.get("filename", "")
                if filename.lower().endswith(".mp4"):
                    path = comfy_output_dir / item.get("subfolder", "") / filename
                    if path.is_file():
                        return path
    raise RenderError("ComfyUI completed successfully but returned no readable MP4")


def submit_and_wait(
    *, base_url: str, workflow: dict[str, Any], timeout_seconds: int, poll_seconds: float
) -> tuple[str, dict[str, Any]]:
    submitted = request_json(f"{base_url}/prompt", {"prompt": workflow}, timeout=60)
    if submitted.get("node_errors"):
        raise RenderError(f"ComfyUI rejected the graph:\n{json.dumps(submitted['node_errors'], indent=2)}")
    prompt_id = submitted.get("prompt_id")
    if not prompt_id:
        raise RenderError(f"ComfyUI returned no prompt_id: {submitted}")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        history = request_json(f"{base_url}/history/{prompt_id}", timeout=60)
        record = history.get(prompt_id)
        if record is None:
            time.sleep(poll_seconds)
            continue
        status = record.get("status", {})
        if status.get("status_str") != "success":
            raise RenderError(f"ComfyUI failed prompt {prompt_id}: {json.dumps(status, indent=2)}")
        return prompt_id, record
    raise RenderError(f"Timed out after {timeout_seconds}s waiting for ComfyUI prompt {prompt_id}")


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def probe_video(path: Path) -> dict[str, Any]:
    completed = run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
        ],
        capture=True,
    )
    return json.loads(completed.stdout)


def verify_exact_endpoints(path: Path, width: int, height: int, last_index: int) -> str:
    completed = run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"select=eq(n\\,0)+eq(n\\,{last_index})",
            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture=True,
    )
    frame_bytes = width * height * 3
    if len(completed.stdout) != frame_bytes * 2:
        raise RenderError(
            f"Endpoint verification expected {frame_bytes * 2} bytes, got {len(completed.stdout)}"
        )
    first = completed.stdout[:frame_bytes]
    last = completed.stdout[frame_bytes:]
    if first != last:
        raise RenderError("Delivery video's decoded first and last frames are not identical")
    return hashlib.sha256(first).hexdigest()


def package_delivery(
    source: Path,
    destination: Path,
    expected_frames: int,
    fps: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{destination.stem}-", dir=destination.parent) as temporary:
        frame_dir = Path(temporary)
        run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0",
                "-vsync", "0", str(frame_dir / "frame_%05d.png"),
            ]
        )
        frames = sorted(frame_dir.glob("frame_*.png"))
        if len(frames) != expected_frames:
            raise RenderError(
                f"Expected {expected_frames} decoded frames from {source}, got {len(frames)}"
            )
        shutil.copy2(frames[0], frames[-1])

        temporary_video = frame_dir / "delivery.mp4"
        run(
            [
                "ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                "-i", str(frame_dir / "frame_%05d.png"), "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-qp", "18", "-x264-params",
                "keyint=1:min-keyint=1:scenecut=0:bframes=0", str(temporary_video),
            ]
        )
        probe = probe_video(temporary_video)
        video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
        audio_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
        if len(video_streams) != 1 or audio_streams:
            raise RenderError("Packaged delivery must contain exactly one video stream and no audio")
        stream = video_streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        if (width, height) != (expected_width, expected_height):
            raise RenderError(
                f"Packaged delivery is {width}x{height}; expected {expected_width}x{expected_height}"
            )
        actual_frames = int(stream.get("nb_frames", expected_frames))
        if actual_frames != expected_frames:
            raise RenderError(f"Packaged delivery has {actual_frames} frames; expected {expected_frames}")
        actual_fps = Fraction(stream["r_frame_rate"])
        if actual_fps != fps:
            raise RenderError(f"Packaged delivery is {actual_fps} fps; expected {fps}")
        if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
            raise RenderError(
                f"Packaged delivery is {stream.get('codec_name')}/{stream.get('pix_fmt')}; "
                "expected h264/yuv420p"
            )
        endpoint_hash = verify_exact_endpoints(temporary_video, width, height, expected_frames - 1)
        os.replace(temporary_video, destination)

    return {
        "file": str(destination),
        "sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": expected_frames,
        "duration_seconds": round(expected_frames / fps, 6),
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "audio_streams": 0,
        "decoded_first_last_exact": True,
        "decoded_endpoint_rgb_sha256": endpoint_hash,
        "last_frame_replaced_with_first": True,
        "fade_or_crossfade_used": False,
    }


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    return stem or "avatar"


def parse_args() -> argparse.Namespace:
    defaults = load_json(DEFAULTS_PATH)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Portrait image used for all three poses.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for videos and manifest.json.")
    parser.add_argument(
        "--poses", nargs="+", choices=tuple(POSES), default=list(POSES),
        help="Subset to render (default: idle talking smiling).",
    )
    parser.add_argument(
        "--pronouns", choices=tuple(PRONOUNS), default="they",
        help="Pronouns used in the documented motion prompts (default: they).",
    )
    parser.add_argument("--attempt", type=int, default=0, help="Reroll number; each attempt shifts seeds by 1000.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing requested pose videos.")
    parser.add_argument("--keep-native", action="store_true", help="Also retain each raw ComfyUI MP4.")
    parser.add_argument("--dry-run", action="store_true", help="Write patched workflow JSON without contacting ComfyUI.")
    parser.add_argument("--base-url", default=defaults["comfyui"]["base_url"])
    parser.add_argument("--workflow", type=Path, default=Path(defaults["comfyui"]["workflow_api_json"]))
    parser.add_argument("--comfy-input-dir", type=Path, default=Path(defaults["comfyui"]["input_dir"]))
    parser.add_argument("--comfy-output-dir", type=Path, default=Path(defaults["comfyui"]["output_dir"]))
    parser.add_argument("--poll-seconds", type=float, default=float(defaults["comfyui"]["poll_seconds"]))
    parser.add_argument("--timeout-seconds", type=int, default=int(defaults["comfyui"]["job_timeout_seconds"]))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = args.image.resolve()
    output_dir = args.output_dir.resolve()
    if not image.is_file():
        raise RenderError(f"Input image does not exist: {image}")
    if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise RenderError("Input image must be PNG, JPEG, or WebP")
    if args.attempt < 0:
        raise RenderError("--attempt must be zero or greater")
    if not args.workflow.is_file():
        raise RenderError(f"LTX workflow does not exist: {args.workflow}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RenderError("ffmpeg and ffprobe are required")

    defaults = load_json(DEFAULTS_PATH)
    pose_spec = load_json(POSE_SPEC_PATH)
    prompt_pack = load_json(PROMPT_PACK_PATH)
    specs = render_specs(pose_spec)
    base_workflow = load_json(args.workflow)
    image_hash = sha256_file(image)
    image_crc = zlib.crc32(image.read_bytes()) & 0xFFFFFFFF
    job_id = f"{safe_stem(image)}-{image_hash[:10]}"
    input_relative = Path("character_factory") / "standalone" / f"{job_id}{image.suffix.lower()}"

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir = output_dir / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for pose_name in args.poses:
        render_key = POSES[pose_name]
        render = specs[render_key]
        output_prefix = f"character_factory/standalone/{job_id}_{pose_name}"
        workflow, metadata = build_workflow(
            base_workflow=base_workflow,
            defaults=defaults,
            prompt_pack=prompt_pack,
            render=render,
            pose_name=pose_name,
            pronoun=args.pronouns,
            input_name=input_relative.as_posix(),
            output_prefix=output_prefix,
            image_crc=image_crc,
            pose_index=list(POSES).index(pose_name),
            attempt=args.attempt,
        )
        write_json(workflow_dir / f"{pose_name}.json", workflow)
        selected.append((pose_name, render_key, workflow, metadata))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator": str(Path(__file__).resolve()),
        "source_image": str(image),
        "source_image_sha256": image_hash,
        "prompt_pack": prompt_pack["pack_id"],
        "pronouns": args.pronouns,
        "attempt": args.attempt,
        "delivery_policy": {
            "native_same_image_conditioning_at_frames": [0, -1],
            "decoded_exact_endpoints": True,
            "last_frame_replaced_with_first": True,
            "fade_or_crossfade": False,
            "audio_streams": 0,
        },
        "poses": {},
    }

    if args.dry_run:
        for pose_name, _, _, metadata in selected:
            manifest["poses"][pose_name] = {"status": "dry_run", **metadata}
            print(
                f"[dry-run] {pose_name}: {metadata['frame_count']} frames, "
                f"seeds={metadata['seed_pair'][0]}/{metadata['seed_pair'][1]}"
            )
        write_json(output_dir / "manifest.json", manifest)
        print(f"Wrote {len(selected)} workflows to {workflow_dir}")
        return 0

    pending = [item for item in selected if args.force or not (output_dir / f"{item[0]}.mp4").exists()]
    if pending:
        try:
            request_json(f"{args.base_url}/system_stats", timeout=15)
        except (urllib.error.URLError, OSError) as error:
            raise RenderError(
                f"LTX ComfyUI is not reachable at {args.base_url}: {error}. Start it with "
                "/workspace/LTX-2.3/scripts/start-q4-comfyui.sh"
            ) from error
        input_destination = args.comfy_input_dir / input_relative
        input_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, input_destination)

    for pose_name, _, workflow, metadata in selected:
        destination = output_dir / f"{pose_name}.mp4"
        if destination.exists() and not args.force:
            manifest["poses"][pose_name] = {
                "status": "skipped_existing",
                **metadata,
                "delivery": {"file": str(destination), "sha256": sha256_file(destination)},
            }
            print(f"SKIP   {pose_name}: {destination}")
            continue

        print(f"QUEUE  {pose_name}: {metadata['frame_count']} frames", flush=True)
        started = time.monotonic()
        prompt_id, history = submit_and_wait(
            base_url=args.base_url,
            workflow=workflow,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        native = find_video_output(history, args.comfy_output_dir)
        if args.keep_native:
            shutil.copy2(native, output_dir / f"{pose_name}.native.mp4")
        delivery = package_delivery(
            native,
            destination,
            metadata["frame_count"],
            metadata["fps"],
            metadata["width"],
            metadata["height"],
        )
        elapsed = round(time.monotonic() - started, 2)
        manifest["poses"][pose_name] = {
            "status": "completed",
            **metadata,
            "prompt_id": prompt_id,
            "source_comfy_output": str(native),
            "elapsed_seconds": elapsed,
            "delivery": delivery,
        }
        write_json(output_dir / "manifest.json", manifest)
        print(f"SAVED  {pose_name}: {destination} ({elapsed / 60:.1f} min)", flush=True)

    write_json(output_dir / "manifest.json", manifest)
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RenderError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}")
