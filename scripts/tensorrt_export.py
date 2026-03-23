import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, List

import torch
from torch.export import ExportedProgram


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.utils import load_all_model
from scripts.validate_vae_backend import compare_vae_backend_outputs, load_cached_latents


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


def parse_precision(raw: str) -> torch.dtype:
    value = (raw or "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {raw!r}")


def resolve_torch_executed_ops(raw_values: List[str]):
    """
    Added code: allow correctness-first Torch-TensorRT experiments that keep
    selected ops on the PyTorch side while still compiling the surrounding
    graph to TRT.
    """
    resolved = set()
    for raw in raw_values or []:
        name = (raw or "").strip().lower()
        if not name:
            continue
        if name == "native_group_norm":
            resolved.add(torch.ops.aten.native_group_norm.default)
            continue
        if name == "group_norm" and hasattr(torch.ops.aten, "group_norm"):
            resolved.add(torch.ops.aten.group_norm.default)
            continue
        if name == "scaled_dot_product_attention" and hasattr(
            torch.ops.aten, "scaled_dot_product_attention"
        ):
            resolved.add(torch.ops.aten.scaled_dot_product_attention.default)
            continue
        raise ValueError(
            f"Unsupported --torch-executed-op value: {raw!r}. "
            "Expected one of native_group_norm, group_norm, scaled_dot_product_attention."
        )
    return resolved


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _sanitize_for_pickle(value):
    """
    Added code: Torch-TensorRT 2.5.0 can stash OpOverload / PyCapsule-backed
    objects inside runtime metadata when `torch_executed_ops` is used. Those
    objects are not pickleable, so mixed-graph experiments crash before we can
    inspect their outputs. Sanitize metadata recursively into pickle-safe
    primitives/strings.
    """
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            _sanitize_for_pickle(key): _sanitize_for_pickle(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_pickle(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_pickle(item) for item in value)
    if isinstance(value, set):
        return sorted(_sanitize_for_pickle(item) for item in value)
    try:
        pickle.dumps(value)
        return value
    except Exception:
        pass
    module_name = getattr(value, "__module__", "")
    qual_name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
    if module_name or qual_name:
        return ".".join(part for part in [module_name, qual_name] if part)
    return repr(value)


def _patch_torch_tensorrt_metadata_encoder() -> None:
    """
    Added code: patch Torch-TensorRT's metadata encoder once per process so
    correctness experiments using `torch_executed_ops` do not fail while
    pickling runtime metadata.
    """
    try:
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )
    except Exception:
        return

    current = getattr(TorchTensorRTModule, "encode_metadata", None)
    if getattr(current, "_musetalk_sanitized", False):
        return

    def encode_metadata(self, metadata):
        import base64
        import copy

        sanitized = _sanitize_for_pickle(copy.deepcopy(metadata))
        dumped_metadata = pickle.dumps(sanitized)
        encoded_metadata = base64.b64encode(dumped_metadata).decode("utf-8")
        return encoded_metadata

    encode_metadata._musetalk_sanitized = True
    TorchTensorRTModule.encode_metadata = encode_metadata


def require_torch_tensorrt():
    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError(
            "torch_tensorrt is not installed. Install it before exporting MuseTalk TensorRT engines."
        ) from exc
    _patch_torch_tensorrt_metadata_encoder()
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


class VAEPreTRTWrapper(torch.nn.Module):
    """
    Added code: TRT-safe prefix for a hybrid VAE decode experiment.

    This stage keeps the parts that matched PyTorch exactly in stage inspection:
    - latent scaling
    - post_quant_conv
    - decoder.conv_in
    """

    def __init__(self, vae_module: torch.nn.Module, scaling_factor: float):
        super().__init__()
        self.post_quant_conv = vae_module.post_quant_conv
        self.conv_in = vae_module.decoder.conv_in
        self.register_buffer(
            "scaling_factor_tensor",
            torch.tensor(float(scaling_factor), dtype=torch.float32),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        scaled = latent / self.scaling_factor_tensor.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        sample = self.post_quant_conv(scaled)
        return self.conv_in(sample)


class VAETailTRTWrapper(torch.nn.Module):
    """
    Added code: TRT-safe suffix for a hybrid VAE decode experiment.

    The current stage-localization data says the first broken region is the
    decoder mid-block. This wrapper starts *after* that region and keeps the
    rest of the decoder in TRT so we can test a PyTorch-island mid-block.
    """

    def __init__(self, vae_module: torch.nn.Module):
        super().__init__()
        decoder = vae_module.decoder
        self.up_blocks = decoder.up_blocks
        self.conv_norm_out = decoder.conv_norm_out
        self.conv_act = decoder.conv_act
        self.conv_out = decoder.conv_out
        self.output_dtype = next(iter(decoder.up_blocks.parameters())).dtype

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(self.output_dtype)
        for up_block in self.up_blocks:
            sample = up_block(sample, None)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return (sample / 2 + 0.5).clamp(0, 1)


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


def normalize_trt_ir(raw_ir: str) -> str:
    """
    Added code: normalize frontend labels across Torch-TensorRT releases.

    Older experiment notes and helper code used `dynamo_compile`, but current
    Torch-TensorRT releases expect the public frontend name `dynamo`.
    """
    value = raw_ir.strip().lower()
    if value == "dynamo_compile":
        return "dynamo"
    return value


def compile_trt_module(
    module: torch.nn.Module,
    inputs: list,
    trace_inputs: list[torch.Tensor],
    save_inputs: list[torch.Tensor],
    save_path: Path,
    workspace_gb: float,
    min_block_size: int,
    save_format: str,
    torch_executed_ops=None,
) -> Path:
    torch_tensorrt = require_torch_tensorrt()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Compiling %s", save_path.name)

    # Added code: Torch-TensorRT IR compatibility fallback.
    # The local 1.4.x stack works best through the older TorchScript-specific
    # API. Prefer that first, then fall back to dynamo only if needed.
    preferred_ir = normalize_trt_ir(os.getenv("MUSETALK_TRT_IR", ""))
    if preferred_ir:
        ir_candidates = [preferred_ir]
    else:
        ir_candidates = ["dynamo", "torchscript"] if torch_executed_ops else ["torchscript", "dynamo"]

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
                if torch_executed_ops:
                    compile_kwargs["torch_executed_ops"] = torch_executed_ops

                if ir == "dynamo":
                    # Added code: keep the default dynamo fallback strict so we
                    # do not silently accept partial TRT conversion in the
                    # benchmark/export path. When correctness experiments
                    # explicitly mark selected ops for PyTorch fallback, allow
                    # a mixed graph instead of rejecting it up front.
                    compile_kwargs["pass_through_build_failures"] = False
                    compile_kwargs["require_full_compilation"] = not bool(
                        torch_executed_ops
                    )

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

    selected_save_format = save_format.strip().lower() if save_format else "auto"
    if selected_save_format not in {"auto", "torchscript", "exported_program"}:
        raise RuntimeError(
            f"Unsupported TensorRT save format: {save_format!r}. "
            "Expected auto, torchscript, or exported_program."
        )

    if isinstance(compiled, (torch.jit.ScriptModule, torch.jit.ScriptFunction)):
        torch.jit.save(compiled, str(save_path))
        selected_save_format = "torchscript"
    elif hasattr(torch_tensorrt, "save"):
        # Added code: dynamo compilation on newer Torch-TensorRT releases
        # returns a torch.fx.GraphModule. Saving that as TorchScript requires
        # example tensor inputs so torch_tensorrt.save(...) can retrace it.
        # Added code: the current gray-mask regression appears downstream of
        # the old GraphModule -> TorchScript retrace/save/load path. Prefer the
        # native exported_program save format for FX GraphModules unless the
        # caller explicitly requests otherwise.
        if selected_save_format == "auto":
            selected_save_format = "exported_program"
        save_kwargs = {"output_format": selected_save_format}
        if selected_save_format == "exported_program":
            # Added code: the default non-retrace ExportedProgram save path
            # produced unloadable symbolic-shape artifacts on this TRT stack.
            # Prefer the retrace export path unless the caller explicitly turns
            # it off.
            save_kwargs["retrace"] = env_flag("MUSETALK_TRT_SAVE_RETRACE", "1")
        # Added code: use separate save-time example tensors so the exported
        # program path can avoid creating an oversized execution context at the
        # opt batch just to serialize the module.
        chosen_save_inputs = save_inputs or trace_inputs
        if chosen_save_inputs:
            save_kwargs["inputs"] = [tensor.detach() for tensor in chosen_save_inputs]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch_tensorrt.save(compiled, str(save_path), **save_kwargs)
    else:
        raise RuntimeError(
            "Compiled TensorRT module is not a TorchScript module, and this "
            "torch_tensorrt version does not provide torch_tensorrt.save()."
        )
    size_mb = save_path.stat().st_size / 1e6
    logger.info(
        "Saved %s (%.1f MB) with ir=%s save_format=%s",
        save_path,
        size_mb,
        selected_ir,
        selected_save_format,
    )
    return save_path, selected_save_format


def load_serialized_trt_module(engine_path: Path):
    """
    Added code: use Torch-TensorRT's own loader so exported_program artifacts
    do not need to masquerade as TorchScript for benchmarking or runtime use.
    """
    torch_tensorrt = require_torch_tensorrt()
    loaded = torch_tensorrt.load(str(engine_path))
    if isinstance(loaded, ExportedProgram):
        loaded = loaded.module()
    return loaded.eval()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def update_json(path: Path, payload: dict) -> None:
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except Exception:
            current = {}
    current.update(payload)
    write_json(path, current)


def export_unet(
    unet_module: torch.nn.Module,
    output_dir: Path,
    batch_sizes: List[int],
    device: torch.device,
    workspace_gb: float,
    min_block_size: int,
    save_format: str,
    precision: torch.dtype,
    torch_executed_ops=None,
) -> Path:
    torch_tensorrt = require_torch_tensorrt()

    wrapper = UNetTRTWrapper(unet_module, device=device).eval().to(device=device, dtype=precision)
    min_bs, max_bs = min(batch_sizes), max(batch_sizes)
    opt_bs = batch_sizes[len(batch_sizes) // 2]

    inputs = [
        torch_tensorrt.Input(
            min_shape=(min_bs, UNET_LATENT_C, LATENT_H, LATENT_W),
            opt_shape=(opt_bs, UNET_LATENT_C, LATENT_H, LATENT_W),
            max_shape=(max_bs, UNET_LATENT_C, LATENT_H, LATENT_W),
            dtype=precision,
        ),
        torch_tensorrt.Input(
            min_shape=(min_bs, AUDIO_SEQ_LEN, AUDIO_DIM),
            opt_shape=(opt_bs, AUDIO_SEQ_LEN, AUDIO_DIM),
            max_shape=(max_bs, AUDIO_SEQ_LEN, AUDIO_DIM),
            dtype=precision,
        ),
    ]

    engine_path = output_dir / "unet_trt.ts"
    engine_path, resolved_save_format = compile_trt_module(
        module=wrapper,
        inputs=inputs,
        trace_inputs=[
            torch.randn(
                opt_bs,
                UNET_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
            torch.randn(
                opt_bs,
                AUDIO_SEQ_LEN,
                AUDIO_DIM,
                device=device,
                dtype=precision,
            ),
        ],
        save_inputs=[
            torch.randn(
                min_bs,
                UNET_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
            torch.randn(
                min_bs,
                AUDIO_SEQ_LEN,
                AUDIO_DIM,
                device=device,
                dtype=precision,
            ),
        ],
        save_path=engine_path,
        workspace_gb=workspace_gb,
        min_block_size=min_block_size,
        save_format=save_format,
        torch_executed_ops=torch_executed_ops,
    )

    meta = {
        "type": "unet",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [UNET_LATENT_C, LATENT_H, LATENT_W],
        "encoder_hidden_states_shape": [AUDIO_SEQ_LEN, AUDIO_DIM],
        "dtype": str(precision).replace("torch.", ""),
        "save_format": resolved_save_format,
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
    save_format: str,
    precision: torch.dtype,
    torch_executed_ops=None,
) -> Path:
    torch_tensorrt = require_torch_tensorrt()

    wrapper = VAEDecodeTRTWrapper(vae_module, scaling_factor=scaling_factor).eval().to(
        device=device,
        dtype=precision,
    )
    min_bs, max_bs = min(batch_sizes), max(batch_sizes)
    opt_bs = batch_sizes[len(batch_sizes) // 2]

    inputs = [
        torch_tensorrt.Input(
            min_shape=(min_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            opt_shape=(opt_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            max_shape=(max_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            dtype=precision,
        ),
    ]

    engine_path = output_dir / "vae_decoder_trt.ts"
    engine_path, resolved_save_format = compile_trt_module(
        module=wrapper,
        inputs=inputs,
        trace_inputs=[
            torch.randn(
                opt_bs,
                VAE_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
        ],
        save_inputs=[
            torch.randn(
                min_bs,
                VAE_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
        ],
        save_path=engine_path,
        workspace_gb=workspace_gb,
        min_block_size=min_block_size,
        save_format=save_format,
        torch_executed_ops=torch_executed_ops,
    )

    meta = {
        "type": "vae_decoder",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [VAE_LATENT_C, LATENT_H, LATENT_W],
        "output_shape": [DECODED_C, DECODED_H, DECODED_W],
        "dtype": str(precision).replace("torch.", ""),
        "scaling_factor": float(scaling_factor),
        "expects_raw_latents": True,
        "output_range": [0.0, 1.0],
        "save_format": resolved_save_format,
    }
    write_json(output_dir / "vae_decoder_trt_meta.json", meta)
    return engine_path


def export_vae_hybrid(
    vae_module: torch.nn.Module,
    scaling_factor: float,
    output_dir: Path,
    batch_sizes: List[int],
    device: torch.device,
    workspace_gb: float,
    min_block_size: int,
    save_format: str,
    precision: torch.dtype,
    torch_executed_ops=None,
) -> dict:
    """
    Added code: export a hybrid VAE backend made of:
    - TRT prefix (safe)
    - PyTorch decoder mid-block (correctness island)
    - TRT tail (safe candidate)
    """
    torch_tensorrt = require_torch_tensorrt()

    min_bs, max_bs = min(batch_sizes), max(batch_sizes)
    opt_bs = batch_sizes[len(batch_sizes) // 2]
    pre_inputs = [
        torch_tensorrt.Input(
            min_shape=(min_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            opt_shape=(opt_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            max_shape=(max_bs, VAE_LATENT_C, LATENT_H, LATENT_W),
            dtype=precision,
        ),
    ]
    tail_inputs = [
        torch_tensorrt.Input(
            min_shape=(min_bs, 512, LATENT_H, LATENT_W),
            opt_shape=(opt_bs, 512, LATENT_H, LATENT_W),
            max_shape=(max_bs, 512, LATENT_H, LATENT_W),
            dtype=precision,
        ),
    ]

    pre_wrapper = VAEPreTRTWrapper(vae_module, scaling_factor=scaling_factor).eval().to(
        device=device,
        dtype=precision,
    )
    tail_wrapper = VAETailTRTWrapper(vae_module).eval().to(
        device=device,
        dtype=precision,
    )

    pre_path = output_dir / "vae_pre_trt.ts"
    tail_path = output_dir / "vae_tail_trt.ts"

    pre_path, pre_save_format = compile_trt_module(
        module=pre_wrapper,
        inputs=pre_inputs,
        trace_inputs=[
            torch.randn(
                opt_bs,
                VAE_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
        ],
        save_inputs=[
            torch.randn(
                min_bs,
                VAE_LATENT_C,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
        ],
        save_path=pre_path,
        workspace_gb=workspace_gb,
        min_block_size=min_block_size,
        save_format=save_format,
        torch_executed_ops=torch_executed_ops,
    )

    tail_path, tail_save_format = compile_trt_module(
        module=tail_wrapper,
        inputs=tail_inputs,
        trace_inputs=[
            torch.randn(
                opt_bs,
                512,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
        ],
        save_inputs=[
            torch.randn(
                min_bs,
                512,
                LATENT_H,
                LATENT_W,
                device=device,
                dtype=precision,
            ),
        ],
        save_path=tail_path,
        workspace_gb=workspace_gb,
        min_block_size=min_block_size,
        save_format=save_format,
        torch_executed_ops=torch_executed_ops,
    )

    meta = {
        "type": "vae_decoder_hybrid",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [VAE_LATENT_C, LATENT_H, LATENT_W],
        "mid_block_shape": [512, LATENT_H, LATENT_W],
        "output_shape": [DECODED_C, DECODED_H, DECODED_W],
        "dtype": str(precision).replace("torch.", ""),
        "scaling_factor": float(scaling_factor),
        "expects_raw_latents": True,
        "output_range": [0.0, 1.0],
        "hybrid": {
            "pre_path": pre_path.name,
            "pre_save_format": pre_save_format,
            "mid_block_backend": "pytorch",
            "tail_path": tail_path.name,
            "tail_save_format": tail_save_format,
        },
    }
    write_json(output_dir / "vae_decoder_trt_meta.json", meta)
    return {
        "pre": pre_path,
        "tail": tail_path,
        "meta": output_dir / "vae_decoder_trt_meta.json",
    }


def validate_exported_vae(
    vae,
    engine_path: Path,
    meta_path: Path,
    avatar_id: str,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    max_mae: float,
    precision: torch.dtype,
) -> dict:
    """
    Added code: compare the saved TRT artifact against the live PyTorch VAE
    path before treating an export as trustworthy for avatar output.
    """
    cached_latents = load_cached_latents(avatar_id)
    batch_latents = cached_latents[:batch_size].to(device=device, dtype=precision)
    backend_module = load_serialized_trt_module(engine_path).to(device)

    report = compare_vae_backend_outputs(
        vae=vae,
        backend=backend_module,
        batch_latents=batch_latents,
        backend_name="trt",
        output_dir=output_dir,
        extra_report_fields={
            "avatar_id": avatar_id,
            "batch_size": int(batch_size),
            "engine_path": str(engine_path.resolve()),
        },
    )
    report["passed"] = bool(report["mae"] <= max_mae)
    report["max_mae"] = float(max_mae)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    update_json(
        meta_path,
        {
            "validation": {
                "avatar_id": avatar_id,
                "batch_size": int(batch_size),
                "mae": report["mae"],
                "max_abs": report["max_abs"],
                "max_mae": float(max_mae),
                "passed": report["passed"],
                "report_path": str(report_path.resolve()),
            }
        },
    )
    return report


def benchmark_engine(
    engine_path: Path,
    make_inputs,
    batch_sizes: Iterable[int],
    device: torch.device,
    warmup: int,
    iters: int,
    label: str,
) -> List[dict]:
    model = load_serialized_trt_module(engine_path)
    model = model.to(device)
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
        choices=["vae", "vae_hybrid", "unet", "all"],
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
        "--save-format",
        choices=["auto", "exported_program", "torchscript"],
        default="auto",
        help=(
            "Serialization format for compiled TRT modules. "
            "Default is auto, which prefers exported_program for dynamo/FX outputs."
        ),
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
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision for TRT export. Use fp32 as a correctness experiment if fp16 output looks unstable.",
    )
    parser.add_argument(
        "--validate-avatar-id",
        default="test_avatar",
        help=(
            "Prepared avatar id under results/v15/avatars/<avatar_id> used for "
            "post-export VAE correctness validation. Use an empty string to skip."
        ),
    )
    parser.add_argument(
        "--validate-batch-size",
        type=int,
        default=4,
        help="Batch size for post-export VAE correctness validation.",
    )
    parser.add_argument(
        "--validate-max-mae",
        type=float,
        default=0.05,
        help="Maximum acceptable MAE between PyTorch VAE and TRT VAE validation output.",
    )
    parser.add_argument(
        "--require-valid-vae",
        action="store_true",
        help="Fail the export if post-export VAE validation exceeds --validate-max-mae.",
    )
    parser.add_argument(
        "--torch-executed-op",
        action="append",
        default=[],
        help=(
            "Keep selected ops on the PyTorch side during TRT compilation. "
            "Useful for correctness experiments. Supported: native_group_norm, "
            "group_norm, scaled_dot_product_attention."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)
    if not args.batch_sizes:
        parser.error("No valid batch sizes were provided.")
    precision = parse_precision(args.precision)
    torch_executed_ops = resolve_torch_executed_ops(args.torch_executed_op)

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

    unet.model = unet.model.to(device=device, dtype=precision).eval()
    unet.model.requires_grad_(False)
    vae.vae = vae.vae.to(device=device, dtype=precision).eval()
    vae.vae.requires_grad_(False)
    vae.runtime_dtype = precision

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
            save_format=args.save_format,
            precision=precision,
            torch_executed_ops=torch_executed_ops,
        )
        validate_avatar_id = args.validate_avatar_id.strip()
        if validate_avatar_id:
            validation_dir = output_dir / "validation"
            try:
                validation_report = validate_exported_vae(
                    vae=vae,
                    engine_path=exported_paths["vae"],
                    meta_path=output_dir / "vae_decoder_trt_meta.json",
                    avatar_id=validate_avatar_id,
                    batch_size=max(1, int(args.validate_batch_size)),
                    device=device,
                    output_dir=validation_dir,
                    max_mae=float(args.validate_max_mae),
                    precision=precision,
                )
                logger.info(
                    "VAE validation: passed=%s mae=%.6f max_abs=%.6f report=%s",
                    validation_report["passed"],
                    validation_report["mae"],
                    validation_report["max_abs"],
                    validation_dir / "report.json",
                )
                if args.require_valid_vae and not validation_report["passed"]:
                    raise RuntimeError(
                        "Post-export VAE validation failed: "
                        f"mae={validation_report['mae']:.6f} > max_mae={args.validate_max_mae:.6f}"
                    )
            except Exception as exc:
                update_json(
                    output_dir / "vae_decoder_trt_meta.json",
                    {
                        "validation": {
                            "avatar_id": validate_avatar_id,
                            "batch_size": max(1, int(args.validate_batch_size)),
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    },
                )
                if args.require_valid_vae:
                    raise
                logger.warning("VAE validation failed: %s: %s", type(exc).__name__, exc)

    if args.components == "vae_hybrid":
        exported_paths["vae_hybrid"] = export_vae_hybrid(
            vae_module=vae.vae,
            scaling_factor=vae.scaling_factor,
            output_dir=output_dir,
            batch_sizes=args.batch_sizes,
            device=device,
            workspace_gb=args.workspace_gb,
            min_block_size=args.min_block_size,
            save_format=args.save_format,
            precision=precision,
            torch_executed_ops=torch_executed_ops,
        )
        validate_avatar_id = args.validate_avatar_id.strip()
        if validate_avatar_id:
            validation_dir = output_dir / "validation"
            try:
                from scripts.trt_runtime import load_vae_trt_decoder

                backend = load_vae_trt_decoder(
                    device=device,
                    scaling_factor=vae.scaling_factor,
                    vae_module=vae.vae,
                    trt_dir=output_dir,
                    force=True,
                )
                if backend is None:
                    raise RuntimeError("Hybrid TRT VAE backend did not activate after export.")
                cached_latents = load_cached_latents(validate_avatar_id)
                batch_latents = cached_latents[: max(1, int(args.validate_batch_size))].to(
                    device=device,
                    dtype=precision,
                )
                validation_report = compare_vae_backend_outputs(
                    vae=vae,
                    backend=backend,
                    batch_latents=batch_latents,
                    backend_name="trt_hybrid",
                    output_dir=validation_dir,
                    extra_report_fields={
                        "avatar_id": validate_avatar_id,
                        "batch_size": max(1, int(args.validate_batch_size)),
                        "engine_path": str(output_dir.resolve()),
                    },
                )
                validation_report["passed"] = bool(
                    validation_report["mae"] <= float(args.validate_max_mae)
                )
                validation_report["max_mae"] = float(args.validate_max_mae)
                report_path = validation_dir / "report.json"
                report_path.write_text(json.dumps(validation_report, indent=2))
                update_json(
                    output_dir / "vae_decoder_trt_meta.json",
                    {
                        "validation": {
                            "avatar_id": validate_avatar_id,
                            "batch_size": max(1, int(args.validate_batch_size)),
                            "mae": validation_report["mae"],
                            "max_abs": validation_report["max_abs"],
                            "max_mae": float(args.validate_max_mae),
                            "passed": validation_report["passed"],
                            "report_path": str(report_path.resolve()),
                        }
                    },
                )
                logger.info(
                    "Hybrid VAE validation: passed=%s mae=%.6f max_abs=%.6f report=%s",
                    validation_report["passed"],
                    validation_report["mae"],
                    validation_report["max_abs"],
                    report_path,
                )
                if args.require_valid_vae and not validation_report["passed"]:
                    raise RuntimeError(
                        "Post-export hybrid VAE validation failed: "
                        f"mae={validation_report['mae']:.6f} > max_mae={args.validate_max_mae:.6f}"
                    )
            except Exception as exc:
                update_json(
                    output_dir / "vae_decoder_trt_meta.json",
                    {
                        "validation": {
                            "avatar_id": validate_avatar_id,
                            "batch_size": max(1, int(args.validate_batch_size)),
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    },
                )
                if args.require_valid_vae:
                    raise
                logger.warning(
                    "Hybrid VAE validation failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

    if args.components in {"unet", "all"}:
        exported_paths["unet"] = export_unet(
            unet_module=unet.model,
            output_dir=output_dir,
            batch_sizes=args.batch_sizes,
            device=device,
            workspace_gb=args.workspace_gb,
            min_block_size=args.min_block_size,
            save_format=args.save_format,
            precision=precision,
            torch_executed_ops=torch_executed_ops,
        )

    if args.benchmark:
        logger.info("")
        logger.info("========== TensorRT Benchmark ==========")

        if "vae" in exported_paths:
            vae_results = benchmark_engine(
                engine_path=exported_paths["vae"],
                make_inputs=lambda bs: [
                    torch.randn(bs, VAE_LATENT_C, LATENT_H, LATENT_W, device=device, dtype=precision),
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
                    torch.randn(bs, UNET_LATENT_C, LATENT_H, LATENT_W, device=device, dtype=precision),
                    torch.randn(bs, AUDIO_SEQ_LEN, AUDIO_DIM, device=device, dtype=precision),
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
