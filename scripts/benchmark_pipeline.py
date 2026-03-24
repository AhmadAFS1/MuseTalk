import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Local modification: this differs from the original MuseTalk code.
# Benchmark entrypoint now applies the local CPU tuning hooks used in this repo.
from runtime_cpu_tuning import apply_cpu_tuning_early, apply_cpu_tuning_runtime

apply_cpu_tuning_early("scripts.benchmark_pipeline")

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.utils import load_all_model
from scripts.trt_runtime import load_vae_trt_decoder

# Local modification: this differs from the original MuseTalk code.
# Runtime CPU tuning is re-applied after imports so benchmarking matches server behavior.
apply_cpu_tuning_runtime("scripts.benchmark_pipeline")


logger = logging.getLogger("benchmark_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class BenchmarkInterrupted(RuntimeError):
    pass


_INTERRUPTED = False


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:
        return str(signum)


def _handle_interrupt(signum, _frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    raise KeyboardInterrupt(f"Received {_signal_name(signum)}")


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)


def check_interrupted() -> None:
    if _INTERRUPTED:
        raise BenchmarkInterrupted("Benchmark interrupted by user request")


def env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value)


def parse_batch_sizes(raw: str) -> List[int]:
    values = []
    seen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = max(1, int(token))
        except ValueError:
            continue
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def compile_modes_for(label: str) -> List[str]:
    raw_modes = (
        os.getenv(f"MUSETALK_COMPILE_{label.upper()}_MODES")
        or os.getenv("MUSETALK_COMPILE_MODES")
        or os.getenv("MUSETALK_COMPILE_MODE")
        or "reduce-overhead,max-autotune"
    )
    modes = []
    seen = set()
    for token in raw_modes.split(","):
        mode = token.strip().lower()
        if not mode:
            continue
        if mode in {"default", "none"}:
            mode = "default"
        if mode in seen:
            continue
        modes.append(mode)
        seen.add(mode)
    return modes or ["reduce-overhead"]


def maybe_compile_module(label: str, module):
    if not env_enabled("MUSETALK_COMPILE", "0"):
        return module, None
    if not env_enabled(f"MUSETALK_COMPILE_{label.upper()}", "1"):
        return module, None
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available; leaving %s eager", label)
        return module, None

    for mode in compile_modes_for(label):
        try:
            kwargs = {"fullgraph": False}
            if mode != "default":
                kwargs["mode"] = mode
            compiled = torch.compile(module, **kwargs)
            return compiled, mode
        except Exception as exc:
            logger.warning(
                "%s compile failed for mode=%s: %s: %s",
                label,
                mode,
                type(exc).__name__,
                exc,
            )
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return module, None


def tensor_to_bgr_uint8(image: torch.Tensor):
    array = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    array = (array * 255).round().astype("uint8")
    return array[..., ::-1]


def measure_ms(fn, iters: int, device: torch.device, runtime_context) -> Tuple[float, object]:
    last = None
    synchronize(device)
    start = time.perf_counter()
    with runtime_context():
        for _ in range(iters):
            check_interrupted()
            last = fn()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
    return elapsed_ms, last


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def default_unet_model_path() -> str:
    return os.path.join("models", "musetalkV15", "unet.pth")


def default_unet_config_path() -> str:
    return os.path.join("models", "musetalkV15", "musetalk.json")


def benchmark_pipeline(args) -> dict:
    check_interrupted()
    device = resolve_device(args.device)
    configure_runtime(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Loading MuseTalk models on %s", device)
    try:
        vae, unet, pe = load_all_model(
            unet_model_path=args.unet_model_path,
            vae_type=args.vae_type,
            unet_config=args.unet_config,
            device=device,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory while loading benchmark models. "
            "Stop the API server or other GPU-heavy processes before running "
            "scripts/benchmark_pipeline.py, or retry with --device cpu for a non-GPU smoke test."
        ) from exc

    pe = pe.half().to(device).eval()
    pe.requires_grad_(False)
    vae.vae = vae.vae.half().to(device).eval()
    vae.vae.requires_grad_(False)
    vae.runtime_dtype = vae.vae.dtype
    vae_backend = load_vae_trt_decoder(
        device=device,
        scaling_factor=vae.scaling_factor,
        vae_module=vae.vae,
    )
    vae.set_decode_backend(vae_backend)
    unet.model = unet.model.half().to(device).eval()
    unet.model.requires_grad_(False)

    compiled_unet_mode = None
    compiled_vae_mode = None
    unet.model, compiled_unet_mode = maybe_compile_module("UNET", unet.model)
    if not vae.has_decode_backend():
        vae.vae, compiled_vae_mode = maybe_compile_module("VAE", vae.vae)
    runtime_context = torch.no_grad if (compiled_unet_mode or compiled_vae_mode) else torch.inference_mode

    timesteps = torch.tensor([0], device=device)
    batch_sizes = args.batch_sizes
    unet_in_channels = int(
        getattr(
            getattr(unet.model, "config", None),
            "in_channels",
            getattr(getattr(unet.model, "conv_in", None), "in_channels", 8),
        )
    )

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compile_enabled": env_enabled("MUSETALK_COMPILE", "0"),
        "compiled_unet_mode": compiled_unet_mode,
        "compiled_vae_mode": compiled_vae_mode,
        "vae_decode_backend": vae.get_decode_backend_name(),
        "unet_dtype": str(unet.model.dtype),
        "vae_dtype": str(vae.vae.dtype),
        "batch_sizes": batch_sizes,
        "warmup": args.warmup,
        "iters": args.iters,
        "unet_model_path": args.unet_model_path,
        "unet_config": args.unet_config,
        "vae_type": args.vae_type,
    }

    logger.info("UNet dtype: %s", metadata["unet_dtype"])
    logger.info("VAE dtype: %s", metadata["vae_dtype"])
    logger.info("VAE decode backend: %s", metadata["vae_decode_backend"])
    logger.info("UNet compile mode: %s", compiled_unet_mode or "eager")
    if vae.has_decode_backend():
        logger.info("VAE compile mode: skipped (%s backend active)", vae.get_decode_backend_name())
    else:
        logger.info("VAE compile mode: %s", compiled_vae_mode or "eager")

    results = []

    for bs in batch_sizes:
        check_interrupted()
        logger.info("Benchmarking batch size %d", bs)
        raw_audio = torch.randn(bs, 50, 384, device=device, dtype=unet.model.dtype)
        latent_batch = torch.randn(
            bs,
            unet_in_channels,
            32,
            32,
            device=device,
            dtype=unet.model.dtype,
        )

        with runtime_context():
            conditioned = pe(raw_audio)
            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=conditioned,
            ).sample.to(device=device, dtype=vae.vae.dtype)
            decoded = vae.decode_latents_tensor(pred_latents)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(args.warmup):
            check_interrupted()
            with runtime_context():
                conditioned = pe(raw_audio)
                pred_latents = unet.model(
                    latent_batch,
                    timesteps,
                    encoder_hidden_states=conditioned,
                ).sample.to(device=device, dtype=vae.vae.dtype)
                decoded = vae.decode_latents_tensor(pred_latents)
                _ = tensor_to_bgr_uint8(decoded)
                _ = vae.decode_latents(pred_latents)
            synchronize(device)

        pe_ms, conditioned = measure_ms(lambda: pe(raw_audio), args.iters, device, runtime_context)
        unet_ms, pred_latents = measure_ms(
            lambda: unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=conditioned,
            ).sample.to(device=device, dtype=vae.vae.dtype),
            args.iters,
            device,
            runtime_context,
        )
        vae_decode_ms, decoded = measure_ms(
            lambda: vae.decode_latents_tensor(pred_latents),
            args.iters,
            device,
            runtime_context,
        )
        transfer_ms, _ = measure_ms(
            lambda: tensor_to_bgr_uint8(decoded),
            args.iters,
            device,
            runtime_context,
        )
        vae_full_ms, _ = measure_ms(
            lambda: vae.decode_latents(pred_latents),
            args.iters,
            device,
            runtime_context,
        )

        pipeline_total_ms = pe_ms + unet_ms + vae_full_ms
        decomposed_total_ms = pe_ms + unet_ms + vae_decode_ms + transfer_ms
        throughput_fps = bs / (pipeline_total_ms / 1000.0)

        result = {
            "batch_size": bs,
            "pe_ms": round(pe_ms, 2),
            "unet_ms": round(unet_ms, 2),
            "vae_decode_ms": round(vae_decode_ms, 2),
            "transfer_ms": round(transfer_ms, 2),
            "vae_full_path_ms": round(vae_full_ms, 2),
            "pipeline_total_ms": round(pipeline_total_ms, 2),
            "decomposed_total_ms": round(decomposed_total_ms, 2),
            "frames_per_sec": round(throughput_fps, 1),
        }

        if device.type == "cuda":
            result["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
            result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 1)

        results.append(result)

        logger.info(
            "BS=%2d | PE=%6.2fms | UNet=%7.2fms | VAE-decode=%7.2fms | "
            "Transfer=%6.2fms | VAE-full=%7.2fms | Total=%7.2fms | Throughput=%6.1f fps",
            bs,
            pe_ms,
            unet_ms,
            vae_decode_ms,
            transfer_ms,
            vae_full_ms,
            pipeline_total_ms,
            throughput_fps,
        )

    summary = {
        "metadata": metadata,
        "results": results,
        "best": max(results, key=lambda item: item["frames_per_sec"]) if results else None,
    }

    if summary["best"] is not None:
        best = summary["best"]
        logger.info(
            "Best throughput: %.1f fps at batch size %d",
            best["frames_per_sec"],
            best["batch_size"],
        )
        logger.info(
            "Max sustainable fps/stream at 8 concurrent: %.1f fps",
            best["frames_per_sec"] / 8.0,
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the MuseTalk model path (PE + UNet + VAE + transfer).",
    )
    parser.add_argument(
        "--batch-sizes",
        default="4,8,16,24,32,40,48",
        help="Comma-separated batch sizes to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations per batch size.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Measured iterations per batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use, for example auto, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--unet-model-path",
        default=default_unet_model_path(),
        help="Path to the MuseTalk UNet weights.",
    )
    parser.add_argument(
        "--unet-config",
        default=default_unet_config_path(),
        help="Path to the MuseTalk UNet config JSON.",
    )
    parser.add_argument(
        "--vae-type",
        default="sd-vae",
        help="VAE directory under models/, for example sd-vae or sd-vae-ft-mse.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write benchmark results as JSON.",
    )
    return parser


def main() -> int:
    install_signal_handlers()
    parser = build_parser()
    args = parser.parse_args()
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)
    if not args.batch_sizes:
        parser.error("No valid batch sizes were provided.")

    try:
        summary = benchmark_pipeline(args)
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by Ctrl+C; stopping immediately")
        return 130
    except BenchmarkInterrupted:
        logger.warning("Benchmark interrupted; stopping immediately")
        return 130

    logger.info("")
    logger.info("=== Summary Table ===")
    logger.info(
        "%-5s %-8s %-10s %-12s %-11s %-13s %-12s",
        "BS",
        "PE(ms)",
        "UNet(ms)",
        "VAE GPU(ms)",
        "Xfer(ms)",
        "VAE Full(ms)",
        "FPS",
    )
    logger.info("-" * 84)
    for row in summary["results"]:
        logger.info(
            "%-5d %-8.2f %-10.2f %-12.2f %-11.2f %-13.2f %-12.1f",
            row["batch_size"],
            row["pe_ms"],
            row["unet_ms"],
            row["vae_decode_ms"],
            row["transfer_ms"],
            row["vae_full_path_ms"],
            row["frames_per_sec"],
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote JSON results to %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
