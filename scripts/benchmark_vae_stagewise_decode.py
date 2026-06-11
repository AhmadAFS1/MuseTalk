#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trt_runtime import load_vae_trt_decoder  # noqa: E402
from scripts.validate_vae_backend import (  # noqa: E402
    compare_vae_backend_outputs,
    load_calibration_latents,
    load_reference_vae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one MuseTalk stagewise VAE decode backend config."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--calibration-dir", default="./calibration/vae_decoder")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--int8-stages",
        required=True,
        help="Comma-separated MUSETALK_TRT_STAGEWISE_INT8_STAGES value.",
    )
    parser.add_argument(
        "--split-up-blocks",
        default="",
        help="Optional comma-separated MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS value.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def set_backend_env(args: argparse.Namespace) -> None:
    os.environ["MUSETALK_TRT_ENABLED"] = "1"
    os.environ["MUSETALK_VAE_BACKEND"] = "trt_stagewise"
    os.environ["MUSETALK_TRT_FALLBACK"] = "0"
    os.environ["MUSETALK_TRT_STAGEWISE_PRECISION"] = "int8_mixed"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_FRONTEND"] = "onnx_qdq"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_STAGES"] = args.int8_stages
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES"] = "1"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR"] = str(
        Path(args.calibration_dir).resolve()
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_BATCHES"] = "8"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO"] = "minmax"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR"] = str(
        Path(args.cache_dir).resolve()
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE"] = "1"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE"] = "0"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE"] = "1"
    os.environ["MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"] = str(max(1, args.batch_size))
    if args.split_up_blocks.strip():
        os.environ["MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS"] = args.split_up_blocks
    else:
        os.environ.pop("MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS", None)


def measure_decode(backend, latents: torch.Tensor, scaling_factor: float, output_dtype: torch.dtype, warmup_iters: int, iters: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(max(0, warmup_iters)):
            backend.decode(latents=latents, scaling_factor=scaling_factor, output_dtype=output_dtype)
        if latents.is_cuda:
            torch.cuda.synchronize(latents.device)

        started_at = time.perf_counter()
        for _ in range(max(1, iters)):
            backend.decode(latents=latents, scaling_factor=scaling_factor, output_dtype=output_dtype)
        if latents.is_cuda:
            torch.cuda.synchronize(latents.device)
        elapsed_s = time.perf_counter() - started_at
    avg_s = elapsed_s / max(1, iters)
    fps = float(latents.shape[0]) / avg_s if avg_s > 0 else 0.0
    return avg_s, fps


def main() -> int:
    args = parse_args()
    set_backend_env(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda:0")

    vae = load_reference_vae(device=device)
    batch_size = max(1, args.batch_size)
    latents = load_calibration_latents(args.calibration_dir, batch_size).to(
        device=device,
        dtype=torch.float16,
    )

    backend = load_vae_trt_decoder(
        device=device,
        scaling_factor=vae.scaling_factor,
        vae_module=vae.vae,
    )
    if backend is None:
        raise RuntimeError("Stagewise backend did not activate.")

    backend.warmup([batch_size])
    compare_dir = Path(args.output_json).with_suffix("")
    comparison = compare_vae_backend_outputs(
        vae=vae,
        backend=backend,
        batch_latents=latents,
        backend_name=args.label,
        output_dir=compare_dir,
    )
    avg_s, fps = measure_decode(
        backend=backend,
        latents=latents,
        scaling_factor=vae.scaling_factor,
        output_dtype=vae.runtime_dtype,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    report = {
        "label": args.label,
        "batch_size": batch_size,
        "iters": max(1, args.iters),
        "warmup_iters": max(0, args.warmup_iters),
        "int8_stages": [stage.strip() for stage in args.int8_stages.split(",") if stage.strip()],
        "split_up_blocks": [
            token.strip() for token in args.split_up_blocks.split(",") if token.strip()
        ],
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "calibration_dir": str(Path(args.calibration_dir).resolve()),
        "avg_decode_s": avg_s,
        "decode_fps": fps,
        "comparison": comparison,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
