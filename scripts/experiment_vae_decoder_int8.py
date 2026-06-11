#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trt_runtime import load_vae_trt_decoder
from scripts.validate_vae_backend import (
    compare_vae_backend_outputs,
    load_calibration_latents,
    load_reference_vae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one isolated mixed INT8/FP16 VAE decoder stagewise backend "
            "from captured pred_latents and compare it against PyTorch FP16."
        )
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Stage to compile with INT8. May be repeated or comma-separated. "
            "Default: decoder_up_block_1."
        ),
    )
    parser.add_argument(
        "--split-up-block",
        action="append",
        default=[],
        help=(
            "Decoder up-block index to split into resnet/upsampler substages "
            "before compiling. May be repeated or comma-separated. Example: 3."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument(
        "--calibration-dir",
        default="./calibration/vae_decoder",
        help="Directory containing captured VAE pred_latents .pt files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./models/tensorrt/stagewise_int8_calibration_cache",
    )
    parser.add_argument(
        "--output-dir",
        default="./tmp/vae_decoder_int8_experiment",
    )
    parser.add_argument(
        "--algo",
        choices=["minmax", "entropy2", "entropy", "legacy"],
        default="minmax",
    )
    parser.add_argument(
        "--enabled-precisions",
        default="int8",
        help="Comma-separated precisions for selected INT8 stages. Default: int8.",
    )
    parser.add_argument(
        "--frontend",
        choices=["onnx_qdq", "torchscript_ptq"],
        default="onnx_qdq",
        help="INT8 compiler frontend. Default: onnx_qdq.",
    )
    parser.add_argument("--workspace-gb", type=float, default=1.0)
    parser.add_argument("--min-block-size", type=int, default=1)
    parser.add_argument(
        "--calibration-format",
        choices=["tensor", "list"],
        default="tensor",
        help="Use list only to diagnose Torch-TensorRT single-input calibrator wiring.",
    )
    parser.add_argument(
        "--torch-executed-op",
        action="append",
        default=[],
        help=(
            "Aten op to force onto PyTorch in the INT8 TorchScript compile. "
            "May be repeated or comma-separated. Default runtime behavior keeps "
            "group_norm on PyTorch."
        ),
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--require-full-compilation", action="store_true")
    parser.add_argument(
        "--allow-empty-cache",
        action="store_true",
        help="Do not fail if TensorRT returns without writing a PTQ cache.",
    )
    return parser.parse_args()


def parse_stages(raw_values: list[str]) -> list[str]:
    stages: list[str] = []
    seen: set[str] = set()
    for raw in raw_values or ["decoder_up_block_1"]:
        for token in raw.split(","):
            stage = token.strip()
            if stage and stage not in seen:
                stages.append(stage)
                seen.add(stage)
    return stages or ["decoder_up_block_1"]


def parse_split_up_blocks(raw_values: list[str]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for raw in raw_values or []:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            index = int(token)
            if index < 0 or index > 3:
                raise RuntimeError(
                    f"Unsupported --split-up-block {index}; expected 0,1,2,3."
                )
            if index not in seen:
                indices.append(index)
                seen.add(index)
    return indices


def set_experiment_env(args: argparse.Namespace, stages: list[str]) -> None:
    os.environ["MUSETALK_TRT_ENABLED"] = "1"
    os.environ["MUSETALK_VAE_BACKEND"] = "trt_stagewise"
    os.environ["MUSETALK_TRT_FALLBACK"] = "0"
    os.environ["MUSETALK_TRT_STAGEWISE_PRECISION"] = "int8_mixed"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_STAGES"] = ",".join(stages)
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES"] = "1"
    os.environ["MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"] = str(max(1, args.batch_size))
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR"] = str(
        Path(args.calibration_dir).resolve()
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_BATCHES"] = str(
        max(1, args.calibration_batches)
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO"] = args.algo
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR"] = str(Path(args.cache_dir).resolve())
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE"] = "1" if args.use_cache else "0"
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_FRONTEND"] = args.frontend
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS"] = args.enabled_precisions
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE"] = str(
        max(1, args.min_block_size)
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION"] = (
        "1" if args.require_full_compilation else "0"
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE"] = (
        "0" if args.allow_empty_cache else "1"
    )
    os.environ["MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT"] = args.calibration_format
    os.environ["MUSETALK_TRT_STAGEWISE_WORKSPACE_GB"] = str(max(0.25, args.workspace_gb))
    split_up_blocks = parse_split_up_blocks(args.split_up_block)
    if split_up_blocks:
        os.environ["MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS"] = ",".join(
            str(index) for index in split_up_blocks
        )
    if args.torch_executed_op:
        ops: list[str] = []
        for raw in args.torch_executed_op:
            ops.extend(token.strip() for token in raw.split(",") if token.strip())
        os.environ["MUSETALK_TRT_STAGEWISE_INT8_TORCH_EXECUTED_OPS"] = ",".join(ops)


def experiment_metadata(args: argparse.Namespace, stages: list[str], batch_size: int) -> dict:
    return {
        "batch_size": batch_size,
        "int8_stages": stages,
        "calibration_dir": str(Path(args.calibration_dir).resolve()),
        "calibration_batches": max(1, args.calibration_batches),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "algo": args.algo,
        "frontend": args.frontend,
        "enabled_precisions": args.enabled_precisions,
        "int8_min_block_size": max(1, args.min_block_size),
        "require_full_compilation": bool(args.require_full_compilation),
        "calibration_format": args.calibration_format,
        "allow_empty_cache": bool(args.allow_empty_cache),
        "allow_unsafe_stages": os.getenv(
            "MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES",
            "0",
        ),
        "split_up_blocks": parse_split_up_blocks(args.split_up_block),
        "torch_executed_ops": os.getenv(
            "MUSETALK_TRT_STAGEWISE_INT8_TORCH_EXECUTED_OPS",
            "group_norm",
        ),
    }


def write_failure_report(
    output_dir: Path,
    args: argparse.Namespace,
    stages: list[str],
    batch_size: int,
    exc: BaseException,
) -> None:
    report = {
        "status": "failed",
        **experiment_metadata(args, stages, batch_size),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), file=sys.stderr)


def main() -> int:
    args = parse_args()
    stages = parse_stages(args.stage)
    set_experiment_env(args, stages)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_reference_vae(device=device)
    batch_size = max(1, args.batch_size)
    batch_latents = load_calibration_latents(args.calibration_dir, batch_size).to(
        device=device,
        dtype=torch.float16,
    )

    try:
        backend = load_vae_trt_decoder(
            device=device,
            scaling_factor=vae.scaling_factor,
            vae_module=vae.vae,
        )
        if backend is None:
            raise RuntimeError("Stagewise INT8 backend did not activate.")

        report = compare_vae_backend_outputs(
            vae=vae,
            backend=backend,
            batch_latents=batch_latents,
            backend_name="trt_stagewise_int8",
            output_dir=output_dir,
            extra_report_fields={
                "status": "passed",
                "backend_name": getattr(backend, "name", type(backend).__name__),
                **experiment_metadata(args, stages, batch_size),
            },
        )
    except Exception as exc:
        write_failure_report(output_dir, args, stages, batch_size, exc)
        return 1

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
