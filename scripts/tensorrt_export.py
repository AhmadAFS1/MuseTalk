import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.utils import load_all_model


logger = logging.getLogger("tensorrt_export")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


LATENT_H = 32
LATENT_W = 32
UNET_LATENT_C = 8
VAE_LATENT_C = 4
AUDIO_SEQ_LEN = 50
AUDIO_DIM = 384
DECODED_C = 3
DECODED_H = 256
DECODED_W = 256


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
    values.sort()
    return values


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_torch_tensorrt():
    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError(
            "torch_tensorrt is not installed. Install it before exporting MuseTalk TensorRT engines."
        ) from exc
    return torch_tensorrt


class UNetTRTWrapper(torch.nn.Module):
    """
    Added code: TensorRT export wrapper for MuseTalk's UNet path.

    The live HLS scheduler already applies positional encoding before UNet, so
    the export signature matches the actual runtime path:
      latent: (B, 8, 32, 32)
      encoder_hidden_states: (B, 50, 384)
    """

    def __init__(self, unet_module: torch.nn.Module, device: torch.device):
        super().__init__()
        self.unet = unet_module
        self.register_buffer("timesteps", torch.tensor([0], device=device, dtype=torch.long))

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            latent,
            self.timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample


class VAEDecodeTRTWrapper(torch.nn.Module):
    """
    Added code: TensorRT export wrapper for MuseTalk's VAE decoder.

    The scaling factor and output normalization are baked into the graph so the
    exported engine matches the current PyTorch `decode_latents_tensor()` path.
    """

    def __init__(self, vae_module: torch.nn.Module, scaling_factor: float):
        super().__init__()
        # Added code: export only the narrow decoder path instead of the full
        # AutoencoderKL.decode(...) wrapper. This avoids extra diffusers helper
        # methods that are difficult to script on the current TRT stack.
        self.post_quant_conv = vae_module.post_quant_conv
        self.decoder = vae_module.decoder
        self.register_buffer(
            "scaling_factor_tensor",
            torch.tensor(float(scaling_factor), dtype=torch.float32),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        scaled = latent / self.scaling_factor_tensor.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        decoded_latent = self.post_quant_conv(scaled)
        image = self.decoder(decoded_latent)
        return (image / 2 + 0.5).clamp(0, 1)


def trace_torchscript_module(
    module: torch.nn.Module,
    trace_inputs: list[torch.Tensor],
):
    with torch.no_grad():
        traced = torch.jit.trace(
            module,
            tuple(trace_inputs),
            check_trace=False,
            strict=False,
        )
        try:
            traced = torch.jit.freeze(traced.eval())
        except Exception:
            traced = traced.eval()
    return traced


def compile_trt_module(
    module: torch.nn.Module,
    inputs: list,
    trace_inputs: list[torch.Tensor],
    save_path: Path,
    workspace_gb: float,
    min_block_size: int,
) -> Path:
    torch_tensorrt = require_torch_tensorrt()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Compiling %s", save_path.name)

    # Added code: Torch-TensorRT IR compatibility fallback.
    # The local 1.4.x stack works best through the older TorchScript-specific
    # API. Prefer that first, then fall back to dynamo only if needed.
    preferred_ir = os.getenv("MUSETALK_TRT_IR", "").strip().lower()
    if preferred_ir:
        ir_candidates = [preferred_ir]
    else:
        ir_candidates = ["torchscript", "dynamo_compile"]

    last_exc = None
    compiled = None
    selected_ir = None
    for ir in ir_candidates:
        try:
            started_at = time.perf_counter()
            if ir in {"torchscript", "ts", "default"} and hasattr(torch_tensorrt, "ts"):
                traced = trace_torchscript_module(module, trace_inputs)
                compiled = torch_tensorrt.ts.compile(
                    traced,
                    inputs=inputs,
                    enabled_precisions={torch.float16},
                    workspace_size=int(workspace_gb * (1 << 30)),
                    min_block_size=min_block_size,
                    truncate_long_and_double=True,
                    require_full_compilation=True,
                )
            else:
                compile_kwargs = {
                    "ir": ir,
                    "inputs": inputs,
                    "enabled_precisions": {torch.float16},
                    "workspace_size": int(workspace_gb * (1 << 30)),
                    "min_block_size": min_block_size,
                }

                if ir == "dynamo_compile":
                    compile_kwargs["pass_through_build_failures"] = False

                compiled = torch_tensorrt.compile(
                    module,
                    **compile_kwargs,
                )
            elapsed_s = time.perf_counter() - started_at
            selected_ir = ir
            logger.info("Compilation finished in %.1fs using ir=%s", elapsed_s, ir)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "TensorRT compile failed for ir=%s: %s: %s",
                ir,
                type(exc).__name__,
                exc,
            )

    if compiled is None:
        raise RuntimeError(
            f"TensorRT compilation failed for all IR modes: {ir_candidates}"
        ) from last_exc

    if isinstance(compiled, (torch.jit.ScriptModule, torch.jit.ScriptFunction)):
        torch.jit.save(compiled, str(save_path))
    elif hasattr(torch_tensorrt, "save"):
        torch_tensorrt.save(compiled, str(save_path), output_format="torchscript")
    else:
        raise RuntimeError(
            "Compiled TensorRT module is not a TorchScript module, and this "
            "torch_tensorrt version does not provide torch_tensorrt.save()."
        )
    size_mb = save_path.stat().st_size / 1e6
    logger.info("Saved %s (%.1f MB) with ir=%s", save_path, size_mb, selected_ir)
    return save_path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def export_unet(
    unet_module: torch.nn.Module,
    output_dir: Path,
    batch_sizes: List[int],
    device: torch.device,
    workspace_gb: float,
    min_block_size: int,
) -> Path:
    torch_tensorrt = require_torch_tensorrt()

    wrapper = UNetTRTWrapper(unet_module, device=device).eval().to(device).half()
    min_bs, max_bs = min(batch_sizes), max(batch_sizes)
    opt_bs = batch_sizes[len(batch_sizes) // 2]

    inputs = [
        torch_tensorrt.Input(
            min_shape=(min_bs, UNET_LATENT_C, LATENT_H, LATENT_W),
            opt_shape=(opt_bs, UNET_LATENT_C, LATENT_H, LATENT_W),
            max_shape=(max_bs, UNET_LATENT_C, LATENT_H, LATENT_W),
            dtype=torch.float16,
        ),
        torch_tensorrt.Input(
            min_shape=(min_bs, AUDIO_SEQ_LEN, AUDIO_DIM),
            opt_shape=(opt_bs, AUDIO_SEQ_LEN, AUDIO_DIM),
            max_shape=(max_bs, AUDIO_SEQ_LEN, AUDIO_DIM),
            dtype=torch.float16,
        ),
    ]

    engine_path = output_dir / "unet_trt.ts"
    compile_trt_module(
        module=wrapper,
        inputs=inputs,
        trace_inputs=[
            torch.randn(
                opt_bs,
                UNET_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=torch.float16,
            ),
            torch.randn(
                opt_bs,
                AUDIO_SEQ_LEN,
                AUDIO_DIM,
                device=device,
                dtype=torch.float16,
            ),
        ],
        save_path=engine_path,
        workspace_gb=workspace_gb,
        min_block_size=min_block_size,
    )

    meta = {
        "type": "unet",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [UNET_LATENT_C, LATENT_H, LATENT_W],
        "encoder_hidden_states_shape": [AUDIO_SEQ_LEN, AUDIO_DIM],
        "dtype": "float16",
    }
    write_json(output_dir / "unet_trt_meta.json", meta)
    return engine_path


def export_vae(
    vae_module: torch.nn.Module,
    scaling_factor: float,
    output_dir: Path,
    batch_sizes: List[int],
    device: torch.device,
    workspace_gb: float,
    min_block_size: int,
) -> Path:
    torch_tensorrt = require_torch_tensorrt()

    wrapper = VAEDecodeTRTWrapper(vae_module, scaling_factor=scaling_factor).eval().to(device).half()
    min_bs, max_bs = min(batch_sizes), max(batch_sizes)
    opt_bs = batch_sizes[len(batch_sizes) // 2]

    inputs = [
        torch_tensorrt.Input(
            min_shape=(min_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            opt_shape=(opt_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            max_shape=(max_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            dtype=torch.float16,
        ),
    ]

    engine_path = output_dir / "vae_decoder_trt.ts"
    compile_trt_module(
        module=wrapper,
        inputs=inputs,
        trace_inputs=[
            torch.randn(
                opt_bs,
                VAE_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=torch.float16,
            ),
        ],
        save_path=engine_path,
        workspace_gb=workspace_gb,
        min_block_size=min_block_size,
    )

    meta = {
        "type": "vae_decoder",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [VAE_LATENT_C, LATENT_H, LATENT_W],
        "output_shape": [DECODED_C, DECODED_H, DECODED_W],
        "dtype": "float16",
        "scaling_factor": float(scaling_factor),
        "expects_raw_latents": True,
        "output_range": [0.0, 1.0],
    }
    write_json(output_dir / "vae_decoder_trt_meta.json", meta)
    return engine_path


def benchmark_engine(
    engine_path: Path,
    make_inputs,
    batch_sizes: Iterable[int],
    device: torch.device,
    warmup: int,
    iters: int,
    label: str,
) -> List[dict]:
    model = torch.jit.load(str(engine_path), map_location=device).eval()
    results = []

    for bs in batch_sizes:
        inputs = make_inputs(bs)

        for _ in range(warmup):
            with torch.no_grad():
                model(*inputs)
        synchronize(device)

        started_at = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                model(*inputs)
        synchronize(device)

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0 / max(1, iters)
        fps = bs / (elapsed_ms / 1000.0)
        row = {
            "batch_size": bs,
            "latency_ms": round(elapsed_ms, 2),
            "frames_per_sec": round(fps, 1),
        }
        results.append(row)
        logger.info("%s BS=%2d latency=%.2f ms throughput=%.1f fps", label, bs, elapsed_ms, fps)

    return results


def default_unet_model_path() -> str:
    return os.path.join("models", "musetalkV15", "unet.pth")


def default_unet_config_path() -> str:
    return os.path.join("models", "musetalkV15", "musetalk.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MuseTalk components to TensorRT.")
    parser.add_argument(
        "--components",
        choices=["vae", "unet", "all"],
        default="vae",
        help="Which components to export. Default is vae because VAE is the current top priority.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="4,8,16,24,32,40,48",
        help="Comma-separated batch sizes for TensorRT dynamic shape ranges.",
    )
    parser.add_argument(
        "--output-dir",
        default="./models/tensorrt",
        help="Directory to store exported TensorRT engines and metadata.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=1.0,
        help="TensorRT workspace size in GB.",
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=3,
        help="Torch-TensorRT min_block_size.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the exported engine(s) after export.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for the optional benchmark.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed iterations for the optional benchmark.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use, usually auto or cuda:0.",
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
        help="VAE directory under models/.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)
    if not args.batch_sizes:
        parser.error("No valid batch sizes were provided.")

    device = resolve_device(args.device)
    if device.type != "cuda":
        logger.error("TensorRT export requires CUDA. Resolved device was %s.", device)
        return 1

    try:
        require_torch_tensorrt()
    except RuntimeError as exc:
        logger.error("%s", exc)
        logger.error("Install torch-tensorrt before running this export script.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MuseTalk models on %s", device)
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    del pe

    unet.model = unet.model.half().to(device).eval()
    unet.model.requires_grad_(False)
    vae.vae = vae.vae.half().to(device).eval()
    vae.vae.requires_grad_(False)
    vae.runtime_dtype = vae.vae.dtype

    exported_paths = {}

    if args.components in {"vae", "all"}:
        exported_paths["vae"] = export_vae(
            vae_module=vae.vae,
            scaling_factor=vae.scaling_factor,
            output_dir=output_dir,
            batch_sizes=args.batch_sizes,
            device=device,
            workspace_gb=args.workspace_gb,
            min_block_size=args.min_block_size,
        )

    if args.components in {"unet", "all"}:
        exported_paths["unet"] = export_unet(
            unet_module=unet.model,
            output_dir=output_dir,
            batch_sizes=args.batch_sizes,
            device=device,
            workspace_gb=args.workspace_gb,
            min_block_size=args.min_block_size,
        )

    if args.benchmark:
        logger.info("")
        logger.info("========== TensorRT Benchmark ==========")

        if "vae" in exported_paths:
            vae_results = benchmark_engine(
                engine_path=exported_paths["vae"],
                make_inputs=lambda bs: [
                    torch.randn(bs, VAE_LATENT_C, LATENT_H, LATENT_W, device=device, dtype=torch.float16),
                ],
                batch_sizes=args.batch_sizes,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                label="VAE-TRT",
            )
            best = max(vae_results, key=lambda row: row["frames_per_sec"])
            logger.info(
                "Best VAE TRT throughput: %.1f fps at batch size %d",
                best["frames_per_sec"],
                best["batch_size"],
            )

        if "unet" in exported_paths:
            unet_results = benchmark_engine(
                engine_path=exported_paths["unet"],
                make_inputs=lambda bs: [
                    torch.randn(bs, UNET_LATENT_C, LATENT_H, LATENT_W, device=device, dtype=torch.float16),
                    torch.randn(bs, AUDIO_SEQ_LEN, AUDIO_DIM, device=device, dtype=torch.float16),
                ],
                batch_sizes=args.batch_sizes,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                label="UNet-TRT",
            )
            best = max(unet_results, key=lambda row: row["frames_per_sec"])
            logger.info(
                "Best UNet TRT throughput: %.1f fps at batch size %d",
                best["frames_per_sec"],
                best["batch_size"],
            )

    logger.info("TensorRT export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
