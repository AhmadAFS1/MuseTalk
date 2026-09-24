#!/usr/bin/env python3
"""Render native LTX 2.3 idle/talking/smile clips for one portrait.

This is the reusable, roster-free generator for one portrait. It deliberately
uses the compact native Q4 graph. The historical default prompt pack is an
Indian-lineage composite; --prompt-pack selects a versioned, tested pack such
as the Japanese three-pose selection.
It does not use SoulX, Segmind, NAG, or Prompt Relay.

All three renders use the same portrait at native LTX guide indices 0 and -1.
The default centered crop fills the frame with real portrait pixels; --guide-fit
edge_pad reproduces the older edge-replicated guide for historical comparisons.
The default pack produces 241-frame MP4s at 24 fps; an alternate pack may set
frame_count per pose. The final decoded frame is replaced
with the first before all-intra H.264 encoding so the endpoints are pixel exact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FACTORY_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PACK_PATH = FACTORY_ROOT / "config/prompt_packs/native_indian_three_pose_v1.json"
ACCEPTED_GRAPH_PATH = Path(
    "/workspace/experiments/ltx23_native_flf_talking_20260922/graphs/generation.json"
)
COMFY_ROOT = Path("/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/ComfyUI")
COMFY_PYTHON = Path(
    "/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python"
)
GPU_LOCK = Path("/workspace/SoulX-FlashHead/.gpu-owner.lock")
WIDTH = 512
HEIGHT = 832
FPS = 24
FRAMES = 241


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


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._") or "avatar"


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


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def probe_dimensions(path: Path) -> tuple[int, int]:
    completed = run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        capture=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def prepare_guide(source: Path, destination: Path, fit: str = "center_crop") -> dict[str, Any]:
    """Fit a portrait to 512x832 without stretching or inventing border pixels."""
    source_width, source_height = probe_dimensions(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fit == "center_crop":
        if source_width * HEIGHT <= source_height * WIDTH:
            filter_graph = f"scale={WIDTH}:-2:flags=lanczos,crop={WIDTH}:{HEIGHT}:0:(ih-{HEIGHT})/2"
        else:
            filter_graph = f"scale=-2:{HEIGHT}:flags=lanczos,crop={WIDTH}:{HEIGHT}:(iw-{WIDTH})/2:0"
        run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(source),
                "-vf", filter_graph, "-frames:v", "1", str(destination),
            ]
        )
        policy = "center_crop_no_side_padding"
    elif fit == "edge_pad":
        policy = prepare_legacy_edge_padded_guide(source, destination, source_width, source_height)
    else:
        raise RenderError(f"Unknown guide fit: {fit}")
    if probe_dimensions(destination) != (WIDTH, HEIGHT):
        raise RenderError(f"Prepared guide is not {WIDTH}x{HEIGHT}: {destination}")
    return {
        "source_dimensions": [source_width, source_height],
        "guide_dimensions": [WIDTH, HEIGHT],
        "requested_fit": fit,
        "policy": policy,
        "sha256": sha256_file(destination),
    }


def prepare_legacy_edge_padded_guide(
    source: Path, destination: Path, source_width: int, source_height: int,
) -> str:
    """Preserve the earlier guide geometry for replaying historical renders."""
    scaled_width = max(2, 2 * round((source_width * HEIGHT / source_height) / 2))
    if scaled_width <= WIDTH:
        left = (WIDTH - scaled_width) // 2
        right = WIDTH - scaled_width - left
        if left and right:
            graph = (
                f"[0:v]scale={scaled_width}:{HEIGHT}[main];"
                f"[main]split=3[a][b][c];"
                f"[a]crop=1:{HEIGHT}:0:0,scale={left}:{HEIGHT}:flags=neighbor[l];"
                f"[c]crop=1:{HEIGHT}:{scaled_width - 1}:0,scale={right}:{HEIGHT}:flags=neighbor[r];"
                f"[l][b][r]hstack=inputs=3[out]"
            )
            run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-filter_complex", graph, "-map", "[out]", "-frames:v", "1", str(destination),
                ]
            )
            policy = "height_fit_edge_replicated_side_padding"
        else:
            run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-vf", f"scale={WIDTH}:{HEIGHT}", "-frames:v", "1", str(destination),
                ]
            )
            policy = "exact_resize"
    else:
        run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(source),
                "-vf", f"scale={WIDTH}:-2,crop={WIDTH}:{HEIGHT}:0:(ih-{HEIGHT})/2",
                "-frames:v", "1", str(destination),
            ]
        )
        policy = "width_fit_center_crop"
    return policy


def build_generation_graph(
    base: dict[str, Any], profile: dict[str, Any], input_name: str, prefix: str,
    frame_count: int,
) -> dict[str, Any]:
    graph = json.loads(json.dumps(base))
    graph["pos"]["inputs"]["text"] = profile["positive_prompt"]
    graph["neg"]["inputs"]["text"] = profile["negative_prompt"]
    graph["noise"]["inputs"]["noise_seed"] = int(profile["seed"])
    graph["image"]["inputs"]["image"] = input_name
    graph["empty"]["inputs"].update({"width": WIDTH, "height": HEIGHT, "length": frame_count})
    graph["audio"]["inputs"].update({"frames_number": frame_count, "frame_rate": FPS})
    graph["portrait"]["inputs"].update({"frame_idx": 0, "strength": 1.0})
    graph["end_guide"]["inputs"].update({"frame_idx": -1, "strength": 1.0})
    graph["concat"]["inputs"]["video_latent"] = ["end_guide", 2]
    graph["guider"]["inputs"]["positive"] = ["end_guide", 0]
    graph["guider"]["inputs"]["negative"] = ["end_guide", 1]
    graph["crop"]["inputs"]["positive"] = ["end_guide", 0]
    graph["crop"]["inputs"]["negative"] = ["end_guide", 1]
    graph["save"]["inputs"]["filename_prefix"] = prefix
    return graph


def build_decode_graph(latent_name: str, prefix: str, base: dict[str, Any]) -> dict[str, Any]:
    return {
        "vae": base["vae"],
        "load": {"class_type": "LoadLatent", "inputs": {"latent": latent_name}},
        "decode": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["load", 0],
                "vae": ["vae", 0],
                "tile_size": 256,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 16,
            },
        },
        "save_images": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": prefix},
        },
    }


def start_server(port: int, log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    command = [
        str(COMFY_PYTHON), str(COMFY_ROOT / "main.py"),
        "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
        "--lowvram", "--reserve-vram", "3", "--cache-none", "--disable-pinned-memory",
        "--disable-async-offload", "--preview-method", "none", "--use-pytorch-cross-attention",
    ]
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    process = subprocess.Popen(command, cwd=COMFY_ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if process.poll() is not None:
            log.close()
            raise RenderError(f"ComfyUI exited during startup with code {process.returncode}; see {log_path}")
        try:
            request_json(f"{base_url}/system_stats", timeout=5)
            return process, log
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    process.terminate()
    log.close()
    raise RenderError(f"ComfyUI startup timed out; see {log_path}")


def stop_server(process: subprocess.Popen[bytes] | None, log: Any) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if log is not None:
        log.close()


def submit_and_wait(
    *, base_url: str, graph: dict[str, Any], label: str, run_dir: Path,
    process: subprocess.Popen[bytes], timeout_seconds: int,
) -> tuple[str, dict[str, Any], float]:
    write_json(run_dir / f"{label}-graph.json", graph)
    submitted = request_json(
        f"{base_url}/prompt",
        {"prompt": graph, "client_id": "character-factory-native-indian-lineage"},
    )
    write_json(run_dir / f"{label}-submit.json", submitted)
    if submitted.get("node_errors"):
        raise RenderError(json.dumps(submitted["node_errors"], indent=2))
    prompt_id = submitted["prompt_id"]
    print(f"QUEUED {label}: {prompt_id}", flush=True)
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RenderError(f"ComfyUI exited during {label} with code {process.returncode}")
        history = request_json(f"{base_url}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            write_json(run_dir / f"{label}-history.json", record)
            if record.get("status", {}).get("status_str") != "success":
                raise RenderError(json.dumps(record.get("status", {}), indent=2))
            elapsed = time.monotonic() - started
            print(f"DONE   {label}: {elapsed / 60:.1f} min", flush=True)
            return prompt_id, record, elapsed
        time.sleep(5)
    raise RenderError(f"Timed out waiting for {label}")


def history_item(record: dict[str, Any], node: str, key: str) -> dict[str, Any]:
    items = record.get("outputs", {}).get(node, {}).get(key, [])
    if not items:
        raise RenderError(f"Missing {key} output from node {node}")
    return items[0]


def package_frames(frame_paths: list[Path], native: Path, delivery: Path, frame_count: int) -> dict[str, Any]:
    if len(frame_paths) != frame_count:
        raise RenderError(f"Expected {frame_count} decoded frames, got {len(frame_paths)}")
    delivery.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{delivery.stem}-", dir=delivery.parent) as temporary:
        frame_dir = Path(temporary)
        for index, source in enumerate(frame_paths):
            shutil.copy2(source, frame_dir / f"frame_{index:05d}.png")
        run(
            [
                "ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
                "-i", str(frame_dir / "frame_%05d.png"), "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-qp", "18", "-x264-params",
                "keyint=1:min-keyint=1:scenecut=0:bframes=0", str(native),
            ]
        )
        shutil.copy2(frame_dir / "frame_00000.png", frame_dir / f"frame_{frame_count - 1:05d}.png")
        run(
            [
                "ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
                "-i", str(frame_dir / "frame_%05d.png"), "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-qp", "18", "-x264-params",
                "keyint=1:min-keyint=1:scenecut=0:bframes=0", str(delivery),
            ]
        )
    raw = run(
        ["ffmpeg", "-v", "error", "-i", str(delivery), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture=True,
    ).stdout
    frame_bytes = WIDTH * HEIGHT * 3
    if len(raw) != frame_count * frame_bytes:
        raise RenderError(f"Delivery decode returned {len(raw)} bytes")
    first, last = raw[:frame_bytes], raw[-frame_bytes:]
    if first != last:
        raise RenderError("Delivery decoded first and last frames are not identical")
    return {
        "file": str(delivery),
        "sha256": sha256_file(delivery),
        "native_file": str(native),
        "native_sha256": sha256_file(native),
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / FPS, 6),
        "decoded_first_last_exact": True,
        "decoded_endpoint_rgb_sha256": hashlib.sha256(first).hexdigest(),
        "last_frame_replaced_with_first": True,
        "fade_or_crossfade_used": False,
        "audio_streams": 0,
    }


def create_contact_sheet(video: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-vf", "fps=1,scale=256:416,tile=5x2", "-frames:v", "1", str(destination),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--poses", nargs="+", choices=("idle", "talking", "smiling"),
        default=["idle", "talking", "smiling"],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-pack", type=Path, default=PROMPT_PACK_PATH)
    parser.add_argument(
        "--guide-fit", choices=("center_crop", "edge_pad"), default="center_crop",
        help="Crop to fill the frame, or reproduce the earlier edge-padded guide.",
    )
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = args.image.resolve()
    output_dir = args.output_dir.resolve()
    if not image.is_file():
        raise RenderError(f"Input image does not exist: {image}")
    prompt_pack_path = args.prompt_pack.resolve()
    for required in (prompt_pack_path, ACCEPTED_GRAPH_PATH, COMFY_PYTHON, COMFY_ROOT / "main.py"):
        if not required.exists():
            raise RenderError(f"Required accepted-lineage asset is missing: {required}")
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RenderError(f"{program} is required")

    prompt_pack = load_json(prompt_pack_path)
    base_graph = load_json(ACCEPTED_GRAPH_PATH)
    selected = {pose: prompt_pack["poses"][pose] for pose in args.poses}
    image_hash = sha256_file(image)
    job_id = f"{safe_stem(image)}-{image_hash[:10]}-{args.guide_fit}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "run"
    graph_dir = output_dir / "graphs"
    review_dir = output_dir / "review"
    native_dir = output_dir / "native"
    for directory in (run_dir, graph_dir, review_dir, native_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_relative = Path("character_factory_native") / f"{job_id}.png"
    input_destination = COMFY_ROOT / "input" / input_relative
    guide = prepare_guide(image, input_destination, args.guide_fit)
    shutil.copy2(input_destination, output_dir / "guide-512x832.png")

    graphs: dict[str, dict[str, Any]] = {}
    for pose, profile in selected.items():
        frame_count = int(profile.get("frame_count", FRAMES))
        if frame_count < 9 or (frame_count - 1) % 8:
            raise RenderError(f"{pose}: frame_count must equal 1 + a multiple of 8")
        graph = build_generation_graph(
            base_graph,
            profile,
            input_relative.as_posix(),
            f"character_factory_native/{job_id}_{pose}",
            frame_count,
        )
        graphs[pose] = graph
        write_json(graph_dir / f"{pose}-generation.json", graph)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "generator": str(Path(__file__).resolve()),
        "source_image": str(image),
        "source_image_sha256": image_hash,
        "prepared_guide": guide,
        "prompt_pack": prompt_pack["pack_id"],
        "prompt_pack_path": str(prompt_pack_path),
        "prompt_pack_sha256": sha256_file(prompt_pack_path),
        "lineage": prompt_pack["lineage"],
        "accepted_graph": str(ACCEPTED_GRAPH_PATH),
        "workflow": {
            "model": base_graph["model"]["inputs"]["unet_name"],
            "text_encoder": base_graph["text"]["inputs"]["clip_name1"],
            "resolution": [WIDTH, HEIGHT],
            "fps": FPS,
            "frames_by_pose": {pose: int(profile.get("frame_count", FRAMES)) for pose, profile in selected.items()},
            "same_portrait_guide_at_frame_indices": [0, -1],
            "guide_strength": 1.0,
            "sampler": "euler",
            "sigmas": base_graph["sigmas"]["inputs"]["sigmas"],
            "cfg": 1.0,
            "prompt_relay_used": False,
            "nag_used": False,
            "soulx_used": False,
            "segmind_used": False,
        },
        "delivery_policy": {
            "decoded_first_last_exact": True,
            "last_frame_replaced_with_first": True,
            "fade_or_crossfade": False,
            "audio_streams": 0,
        },
        "poses": {},
    }
    if args.dry_run:
        for pose, profile in selected.items():
            manifest["poses"][pose] = {"status": "dry_run", **profile}
        write_json(output_dir / "manifest.json", manifest)
        print(f"Dry run wrote {len(selected)} accepted-lineage graphs to {graph_dir}")
        return 0

    pending = [pose for pose in selected if args.force or not (output_dir / f"{pose}.mp4").exists()]
    if not pending:
        print("All requested outputs already exist; use --force to regenerate.")
        return 0

    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = GPU_LOCK.open("a")
    process: subprocess.Popen[bytes] | None = None
    log: Any = None
    latent_inputs: dict[str, str] = {}
    generation_evidence: dict[str, dict[str, Any]] = {}
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print("GPU lock acquired; starting accepted native LTX generation worker.", flush=True)
        process, log = start_server(args.port, run_dir / "generation-server.log")
        for pose in pending:
            prompt_id, record, elapsed = submit_and_wait(
                base_url=base_url,
                graph=graphs[pose],
                label=f"{pose}-generation",
                run_dir=run_dir,
                process=process,
                timeout_seconds=args.timeout_seconds,
            )
            item = history_item(record, "save", "latents")
            source = COMFY_ROOT / "output" / item.get("subfolder", "") / item["filename"]
            retained = native_dir / f"{pose}.latent"
            shutil.copy2(source, retained)
            latent_name = f"character_factory_native/{job_id}_{pose}.latent"
            latent_destination = COMFY_ROOT / "input" / latent_name
            latent_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, latent_destination)
            latent_inputs[pose] = latent_name
            generation_evidence[pose] = {
                "generation_prompt_id": prompt_id,
                "generation_elapsed_seconds": round(elapsed, 2),
                "latent_file": str(retained),
                "latent_sha256": sha256_file(retained),
            }
        stop_server(process, log)
        process = None
        log = None

        print("Generation complete; restarting a clean worker for the accepted 64/16 tiled decode.", flush=True)
        process, log = start_server(args.port, run_dir / "decode-server.log")
        for pose in pending:
            decode_graph = build_decode_graph(
                latent_inputs[pose],
                f"character_factory_native/{job_id}_{pose}_frames/frame",
                base_graph,
            )
            write_json(graph_dir / f"{pose}-decode.json", decode_graph)
            prompt_id, record, elapsed = submit_and_wait(
                base_url=base_url,
                graph=decode_graph,
                label=f"{pose}-decode",
                run_dir=run_dir,
                process=process,
                timeout_seconds=args.timeout_seconds,
            )
            images = record.get("outputs", {}).get("save_images", {}).get("images", [])
            frame_paths = [
                COMFY_ROOT / "output" / item.get("subfolder", "") / item["filename"]
                for item in images
            ]
            native_video = native_dir / f"{pose}.mp4"
            delivery_video = output_dir / f"{pose}.mp4"
            delivery = package_frames(frame_paths, native_video, delivery_video, int(selected[pose].get("frame_count", FRAMES)))
            create_contact_sheet(delivery_video, review_dir / f"{pose}-contact.jpg")
            profile = selected[pose]
            manifest["poses"][pose] = {
                "status": "completed",
                "seed": profile["seed"],
                "positive_prompt": profile["positive_prompt"],
                "negative_prompt": profile["negative_prompt"],
                "prompt_source": profile["prompt_source"],
                **generation_evidence[pose],
                "decode_prompt_id": prompt_id,
                "decode_elapsed_seconds": round(elapsed, 2),
                "delivery": delivery,
                "contact_sheet": str(review_dir / f"{pose}-contact.jpg"),
            }
            write_json(output_dir / "manifest.json", manifest)
            print(f"SAVED  {pose}: {delivery_video}", flush=True)
        stop_server(process, log)
        process = None
        log = None
    finally:
        stop_server(process, log)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    write_json(output_dir / "manifest.json", manifest)
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RenderError, OSError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
