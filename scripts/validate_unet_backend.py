import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.utils import load_all_model
from scripts.trt_runtime import _load_serialized_trt_module


logger = logging.getLogger("validate_unet_backend")


def parse_precision(raw: str) -> torch.dtype:
    value = (raw or "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {raw!r}")


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _capture_batch_size(path: Path) -> int | None:
    """
    Return the scheduler padded batch size for a capture.

    Static TensorRT UNet experiments need to validate only captures whose
    padded batch matches the exact engine shape. Prefer the filename because it
    is cheap, and fall back to the payload for older or renamed captures.
    """
    match = None
    for part in path.stem.split("_"):
        if part.startswith("bs") and part[2:].isdigit():
            match = int(part[2:])
    if match is not None:
        return match

    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if payload.get("kind") != "unet_io_batch":
        return None
    try:
        return int(payload.get("padded_batch", payload["latent_batch"].shape[0]))
    except Exception:
        return None


def discover_captures(capture_dir: Path, limit: int, padded_batch_size: int = 0) -> list[Path]:
    paths = sorted(capture_dir.glob("unet_io_*.pt"))
    if padded_batch_size > 0:
        paths = [
            path
            for path in paths
            if _capture_batch_size(path) == int(padded_batch_size)
        ]
    if limit > 0:
        paths = paths[:limit]
    return paths


def load_pytorch_unet(args, device: torch.device, precision: torch.dtype):
    logger.info("Loading PyTorch UNet reference model on %s", device)
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    del vae, pe
    unet.model = unet.model.to(device=device, dtype=precision).eval()
    unet.model.requires_grad_(False)
    return unet.model


def load_backend(args, device: torch.device, precision: torch.dtype):
    if args.backend == "pytorch":
        model = load_pytorch_unet(args, device=device, precision=precision)

        def run(latent_batch, audio_feature_batch, timesteps):
            return model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample

        return run

    engine_path = Path(args.trt_path)
    logger.info("Loading TensorRT UNet candidate from %s", engine_path)
    module = _load_serialized_trt_module(engine_path=engine_path, device=device)

    def run(latent_batch, audio_feature_batch, timesteps):
        del timesteps
        output = module(latent_batch, audio_feature_batch)
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
            return output.sample
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise RuntimeError(f"Unexpected UNet backend output type: {type(output).__name__}")

    return run


def tensor_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (candidate.float() - reference.float()).abs().reshape(-1)
    return {
        "mae": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "p95_abs": float(torch.quantile(diff, 0.95).item()),
        "max_abs": float(diff.max().item()),
    }


def benchmark_run(run_backend, latent_batch, audio_feature_batch, timesteps, warmup: int, iters: int, device):
    if iters <= 0:
        return None
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = run_backend(latent_batch, audio_feature_batch, timesteps)
        synchronize(device)
        started_at = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = run_backend(latent_batch, audio_feature_batch, timesteps)
        synchronize(device)
    latency_ms = (time.perf_counter() - started_at) * 1000.0 / max(1, iters)
    return {
        "latency_ms": float(latency_ms),
        "frames_per_sec": float(latent_batch.shape[0] / (latency_ms / 1000.0)),
    }


def validate_capture(path: Path, run_backend, args, device: torch.device, precision: torch.dtype) -> dict:
    payload = torch.load(path, map_location="cpu")
    if payload.get("kind") != "unet_io_batch":
        raise RuntimeError(f"{path} is not a UNet capture file")

    latent_batch = payload["latent_batch"].to(device=device, dtype=precision)
    audio_feature_batch = payload["audio_feature_batch"].to(device=device, dtype=precision)
    timesteps = payload.get("timesteps", torch.tensor([0], dtype=torch.long)).to(device=device)
    reference = payload["pred_latents"].to(device=device, dtype=precision)

    actual_batch = int(payload.get("actual_batch", latent_batch.shape[0]))
    if args.actual_only:
        latent_batch = latent_batch[:actual_batch]
        audio_feature_batch = audio_feature_batch[:actual_batch]
        reference = reference[:actual_batch]

    with torch.inference_mode():
        candidate = run_backend(latent_batch, audio_feature_batch, timesteps)
        synchronize(device)

    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"{path} output shape mismatch: candidate={tuple(candidate.shape)} "
            f"reference={tuple(reference.shape)}"
        )

    stats = tensor_stats(candidate, reference)
    bench = benchmark_run(
        run_backend=run_backend,
        latent_batch=latent_batch,
        audio_feature_batch=audio_feature_batch,
        timesteps=timesteps,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return {
        "path": str(path),
        "sequence": int(payload.get("sequence", -1)),
        "actual_batch": actual_batch,
        "padded_batch": int(payload.get("padded_batch", latent_batch.shape[0])),
        "compared_batch": int(latent_batch.shape[0]),
        "items": payload.get("items", []),
        **stats,
        "benchmark": bench,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ["mae", "rmse", "p95_abs", "max_abs"]
    summary = {"files": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_max"] = max(values)
    latency_values = [
        float(row["benchmark"]["latency_ms"])
        for row in rows
        if row.get("benchmark") is not None
    ]
    fps_values = [
        float(row["benchmark"]["frames_per_sec"])
        for row in rows
        if row.get("benchmark") is not None
    ]
    if latency_values:
        summary["latency_ms_mean"] = sum(latency_values) / len(latency_values)
        summary["latency_ms_max"] = max(latency_values)
    if fps_values:
        summary["frames_per_sec_mean"] = sum(fps_values) / len(fps_values)
        summary["frames_per_sec_min"] = min(fps_values)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a UNet backend against real scheduler capture files."
    )
    parser.add_argument(
        "--capture-dir",
        default="./calibration/unet",
        help="Directory containing unet_io_*.pt files captured by the scheduler.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Maximum capture files to validate.")
    parser.add_argument(
        "--padded-batch-size",
        type=int,
        default=0,
        help=(
            "Only validate captures with this scheduler padded batch size. "
            "Use this for exact static TensorRT engines such as batch 8 or 16."
        ),
    )
    parser.add_argument("--device", default="auto", help="Device to run on, usually auto or cuda:0.")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Runtime precision for the backend validation pass.",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "trt"],
        default="pytorch",
        help="Candidate backend to compare against saved scheduler references.",
    )
    parser.add_argument(
        "--trt-path",
        default="./models/tensorrt/unet_trt.ts",
        help="Serialized TensorRT UNet path when --backend=trt.",
    )
    parser.add_argument(
        "--unet-model-path",
        default="models/musetalkV15/unet.pth",
        help="Path to the MuseTalk UNet weights for --backend=pytorch.",
    )
    parser.add_argument(
        "--unet-config",
        default="models/musetalkV15/musetalk.json",
        help="Path to the MuseTalk UNet config JSON for --backend=pytorch.",
    )
    parser.add_argument("--vae-type", default="sd-vae", help="VAE directory under models/.")
    parser.add_argument(
        "--actual-only",
        action="store_true",
        help="Compare only real rows and ignore padded batch rows.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per capture.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations per capture.")
    parser.add_argument("--fail-mae", type=float, default=0.0, help="Fail if max per-file MAE exceeds this.")
    parser.add_argument(
        "--fail-max-abs",
        type=float,
        default=0.0,
        help="Fail if max per-file max_abs exceeds this.",
    )
    parser.add_argument("--report-path", default="", help="Optional JSON report output path.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    precision = parse_precision(args.precision)
    capture_dir = Path(args.capture_dir)
    paths = discover_captures(capture_dir, args.limit, args.padded_batch_size)
    if not paths:
        suffix = (
            f" with padded_batch_size={args.padded_batch_size}"
            if args.padded_batch_size > 0
            else ""
        )
        logger.error("No UNet capture files found in %s%s", capture_dir, suffix)
        return 1

    run_backend = load_backend(args, device=device, precision=precision)
    rows = []
    for path in paths:
        row = validate_capture(path, run_backend, args, device=device, precision=precision)
        rows.append(row)
        bench = row.get("benchmark") or {}
        logger.info(
            "%s batch=%s mae=%.6g rmse=%.6g p95=%.6g max=%.6g latency_ms=%s fps=%s",
            path.name,
            row["compared_batch"],
            row["mae"],
            row["rmse"],
            row["p95_abs"],
            row["max_abs"],
            f"{bench.get('latency_ms'):.2f}" if bench else "n/a",
            f"{bench.get('frames_per_sec'):.1f}" if bench else "n/a",
        )

    summary = summarize(rows)
    report = {
        "backend": args.backend,
        "precision": args.precision,
        "capture_dir": str(capture_dir),
        "padded_batch_size": int(args.padded_batch_size),
        "actual_only": bool(args.actual_only),
        "summary": summary,
        "files": rows,
    }
    print(json.dumps(report["summary"], indent=2))

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("Wrote report to %s", report_path)

    failed = False
    if args.fail_mae > 0 and summary.get("mae_max", 0.0) > args.fail_mae:
        logger.error("MAE gate failed: %.6g > %.6g", summary["mae_max"], args.fail_mae)
        failed = True
    if args.fail_max_abs > 0 and summary.get("max_abs_max", 0.0) > args.fail_max_abs:
        logger.error("max_abs gate failed: %.6g > %.6g", summary["max_abs_max"], args.fail_max_abs)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
