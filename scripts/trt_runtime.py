import copy
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import torch
from torch.export import ExportedProgram


logger = logging.getLogger("trt_runtime")


def _ensure_torch_tensorrt_registered():
    """
    Added code: import torch_tensorrt before loading serialized TRT-backed
    TorchScript modules so the custom `torch.classes.tensorrt.Engine` type is
    registered in the current process.
    """
    try:
        import torch_tensorrt
    except Exception as exc:
        raise RuntimeError(
            "TensorRT VAE runtime requested, but torch_tensorrt could not be imported."
        ) from exc

    try:
        getattr(torch.classes.tensorrt, "Engine")
    except Exception as exc:
        raise RuntimeError(
            "torch_tensorrt imported, but torch.classes.tensorrt.Engine is not registered."
        ) from exc

    _patch_torch_tensorrt_metadata_encoder()
    return torch_tensorrt


def _sanitize_for_pickle(value):
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
        import copy as _copy

        sanitized = _sanitize_for_pickle(_copy.deepcopy(metadata))
        dumped_metadata = pickle.dumps(sanitized)
        encoded_metadata = base64.b64encode(dumped_metadata).decode("utf-8")
        return encoded_metadata

    encode_metadata._musetalk_sanitized = True
    TorchTensorRTModule.encode_metadata = encode_metadata


def _load_serialized_trt_module(engine_path: Path, device: torch.device):
    """
    Added code: load serialized TRT artifacts through Torch-TensorRT itself so
    both TorchScript and exported_program save formats are supported.
    """
    torch_tensorrt = _ensure_torch_tensorrt_registered()
    loaded = torch_tensorrt.load(str(engine_path))
    if isinstance(loaded, ExportedProgram):
        loaded = loaded.module()
    return loaded.to(device).eval()


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _requested_vae_backend() -> str:
    return os.getenv("MUSETALK_VAE_BACKEND", "").strip().lower()


def _trt_vae_requested() -> bool:
    requested_backend = _requested_vae_backend()
    if requested_backend in {"trt", "tensorrt", "trt_stagewise", "tensorrt_stagewise"}:
        return True
    return _env_enabled("MUSETALK_TRT_ENABLED", "0") or _env_enabled(
        "MUSETALK_TRT_VAE_ENABLED", "0"
    )


def _stagewise_backend_requested() -> bool:
    return _requested_vae_backend() in {"trt_stagewise", "tensorrt_stagewise"}


def _allow_fallback() -> bool:
    return _env_enabled("MUSETALK_TRT_FALLBACK", "1")


def _require_validated_artifact() -> bool:
    """
    Added code: allow callers to require a passed post-export validation report
    before activating a serialized TRT VAE artifact.
    """
    return _env_enabled("MUSETALK_TRT_REQUIRE_VALIDATION", "0")


def _trt_dir() -> Path:
    return Path(os.getenv("MUSETALK_TRT_DIR", "./models/tensorrt"))


def _trt_vae_path() -> Path:
    return Path(os.getenv("MUSETALK_TRT_VAE_PATH", str(_trt_dir() / "vae_decoder_trt.ts")))


def _trt_vae_meta_path(engine_path: Path) -> Path:
    default_meta = engine_path.with_name("vae_decoder_trt_meta.json")
    return Path(os.getenv("MUSETALK_TRT_VAE_META_PATH", str(default_meta)))


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [token.strip() for token in raw.split(",") if token.strip()]


def _stagewise_torch_stage_names() -> set[str]:
    return set(_parse_csv_env("MUSETALK_TRT_STAGEWISE_TORCH_STAGES", ""))


def _stagewise_workspace_gb() -> float:
    try:
        return max(0.25, float(os.getenv("MUSETALK_TRT_STAGEWISE_WORKSPACE_GB", "1.0")))
    except Exception:
        return 1.0


def _stagewise_min_block_size() -> int:
    try:
        return max(1, int(os.getenv("MUSETALK_TRT_STAGEWISE_MIN_BLOCK_SIZE", "3")))
    except Exception:
        return 3


def _stagewise_enabled_precisions(runtime_dtype: torch.dtype) -> set[torch.dtype]:
    return {runtime_dtype}


def _stagewise_torch_executed_ops() -> set:
    resolved = set()
    for raw in _parse_csv_env(
        "MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS", "native_group_norm"
    ):
        lowered = raw.lower()
        if lowered == "native_group_norm":
            resolved.add(torch.ops.aten.native_group_norm.default)
            continue
        if lowered == "group_norm" and hasattr(torch.ops.aten, "group_norm"):
            resolved.add(torch.ops.aten.group_norm.default)
            continue
        if lowered == "scaled_dot_product_attention" and hasattr(
            torch.ops.aten, "scaled_dot_product_attention"
        ):
            resolved.add(torch.ops.aten.scaled_dot_product_attention.default)
            continue
        raise RuntimeError(
            f"Unsupported MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS entry: {raw!r}"
        )
    return resolved


class _StagewisePreDecode(torch.nn.Module):
    """
    Added code: exact-match prefix observed during stage inspection.
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
        sample = latent / self.scaling_factor_tensor.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        sample = self.post_quant_conv(sample)
        sample = self.conv_in(sample)
        return sample.contiguous()


class _StagewiseMidBlock(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.mid_block = decoder.mid_block
        self.output_dtype = next(iter(decoder.up_blocks.parameters())).dtype

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.mid_block(sample.contiguous(), None)
        return sample.to(self.output_dtype).contiguous()


class _StagewiseUpBlock(torch.nn.Module):
    def __init__(self, up_block: torch.nn.Module):
        super().__init__()
        self.up_block = up_block

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.up_block(sample.contiguous(), None).contiguous()


class _StagewisePostprocess(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.conv_norm_out = decoder.conv_norm_out
        self.conv_act = decoder.conv_act
        self.conv_out = decoder.conv_out

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_norm_out(sample.contiguous())
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return ((sample / 2 + 0.5).clamp(0, 1)).contiguous()


class StagewiseTrtVaeDecodeBackend:
    """
    Added code: compile the VAE decoder as exact-batch stage modules instead of
    one monolithic TRT engine. This keeps the live path close to the stagewise
    correctness results and lets us keep fragile ops such as
    `native_group_norm` on the PyTorch side inside each compiled stage.
    """

    name = "tensorrt_stagewise"

    def __init__(
        self,
        vae_module: torch.nn.Module,
        device: torch.device,
        scaling_factor: float,
        runtime_dtype: torch.dtype = torch.float16,
        workspace_gb: float = 1.0,
        min_block_size: int = 3,
        torch_executed_ops: Optional[set] = None,
        torch_stage_names: Optional[set[str]] = None,
    ):
        self.device = device
        self.scaling_factor = float(scaling_factor)
        self.runtime_dtype = runtime_dtype
        self.workspace_gb = float(workspace_gb)
        self.min_block_size = int(min_block_size)
        self.torch_executed_ops = set(torch_executed_ops or set())
        self.torch_stage_names = set(torch_stage_names or set())
        decoder = vae_module.decoder
        self.stage_modules = {
            "decoder_pre": _StagewisePreDecode(vae_module, scaling_factor),
            "decoder_mid_block": _StagewiseMidBlock(decoder),
            "decoder_up_block_0": _StagewiseUpBlock(decoder.up_blocks[0]),
            "decoder_up_block_1": _StagewiseUpBlock(decoder.up_blocks[1]),
            "decoder_up_block_2": _StagewiseUpBlock(decoder.up_blocks[2]),
            "decoder_up_block_3": _StagewiseUpBlock(decoder.up_blocks[3]),
            "decoder_postprocess": _StagewisePostprocess(decoder),
        }
        for module in self.stage_modules.values():
            module.to(device=device, dtype=runtime_dtype).eval()
            module.requires_grad_(False)
        self.stage_order = [
            "decoder_pre",
            "decoder_mid_block",
            "decoder_up_block_0",
            "decoder_up_block_1",
            "decoder_up_block_2",
            "decoder_up_block_3",
            "decoder_postprocess",
        ]
        self.compiled_by_batch: dict[int, dict[str, object]] = {}

    @classmethod
    def load(
        cls,
        vae_module: torch.nn.Module,
        device: torch.device,
        scaling_factor: float,
    ) -> "StagewiseTrtVaeDecodeBackend":
        if vae_module is None:
            raise RuntimeError(
                "Stagewise TRT VAE backend requires the live VAE module."
            )
        runtime_dtype = getattr(vae_module, "dtype", torch.float16)
        if runtime_dtype not in {torch.float16, torch.float32}:
            runtime_dtype = torch.float16
        return cls(
            vae_module=vae_module,
            device=device,
            scaling_factor=scaling_factor,
            runtime_dtype=runtime_dtype,
            workspace_gb=_stagewise_workspace_gb(),
            min_block_size=_stagewise_min_block_size(),
            torch_executed_ops=_stagewise_torch_executed_ops(),
            torch_stage_names=_stagewise_torch_stage_names(),
        )

    def _compile_stage(self, stage_name: str, sample_input: torch.Tensor):
        if stage_name in self.torch_stage_names:
            return self.stage_modules[stage_name]

        torch_tensorrt = _ensure_torch_tensorrt_registered()
        example = sample_input.detach().to(
            device=self.device,
            dtype=self.runtime_dtype,
        ).contiguous()
        shape = tuple(int(dim) for dim in example.shape)
        inputs = [
            torch_tensorrt.Input(
                min_shape=shape,
                opt_shape=shape,
                max_shape=shape,
                dtype=self.runtime_dtype,
            )
        ]
        compile_kwargs = {
            "ir": "dynamo",
            "inputs": inputs,
            "enabled_precisions": _stagewise_enabled_precisions(self.runtime_dtype),
            "workspace_size": int(self.workspace_gb * (1 << 30)),
            "min_block_size": self.min_block_size,
            "pass_through_build_failures": False,
            "require_full_compilation": not bool(self.torch_executed_ops),
        }
        if self.torch_executed_ops:
            compile_kwargs["torch_executed_ops"] = self.torch_executed_ops
        module = self.stage_modules[stage_name]
        logger.info(
            "Compiling stagewise TRT VAE stage %s for batch=%s",
            stage_name,
            shape[0],
        )
        return torch_tensorrt.compile(module, **compile_kwargs).eval()

    def _ensure_batch(self, batch_size: int) -> None:
        if batch_size in self.compiled_by_batch:
            return

        compiled = {}
        sample = torch.randn(
            batch_size,
            4,
            32,
            32,
            device=self.device,
            dtype=self.runtime_dtype,
        )
        with torch.no_grad():
            current = sample
            for stage_name in self.stage_order:
                compiled_stage = self._compile_stage(stage_name, current)
                compiled[stage_name] = compiled_stage
                current = compiled_stage(current.contiguous())
                if isinstance(current, (tuple, list)):
                    current = current[0]
                current = current.contiguous()
        self.compiled_by_batch[batch_size] = compiled
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def warmup(self) -> None:
        self._ensure_batch(4)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        scaling_factor: float,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if abs(float(scaling_factor) - self.scaling_factor) > 1e-6:
            logger.warning(
                "Stagewise TRT VAE scaling factor mismatch: runtime=%s backend=%s",
                scaling_factor,
                self.scaling_factor,
            )

        batch_size = int(latents.shape[0])
        self._ensure_batch(batch_size)
        compiled = self.compiled_by_batch[batch_size]
        current = latents.to(device=self.device, dtype=self.runtime_dtype).contiguous()
        for stage_name in self.stage_order:
            current = compiled[stage_name](current.contiguous())
            if isinstance(current, (tuple, list)):
                current = current[0]
            current = current.contiguous()
        if output_dtype is not None and current.dtype != output_dtype:
            current = current.to(dtype=output_dtype)
        return current


class TrtVaeDecodeBackend:
    """
    Wrapper around a TorchScript TensorRT VAE decoder.

    This backend expects an exported VAE decoder module that accepts MuseTalk's
    raw predicted latents with shape (B, 4, 32, 32) and returns decoded frames
    with shape (B, 3, 256, 256) in the 0..1 range.
    """

    name = "tensorrt"

    def __init__(
        self,
        module,
        device: torch.device,
        expects_raw_latents: bool = True,
        batch_range: Optional[tuple[int, int]] = None,
        opt_batch: Optional[int] = None,
    ):
        self.module = module
        self.device = device
        self.expects_raw_latents = expects_raw_latents
        self.runtime_dtype = self._infer_runtime_dtype(module)
        self.batch_range = batch_range
        self.opt_batch = opt_batch

    @staticmethod
    def _infer_runtime_dtype(module) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return torch.float16
        except Exception:
            return torch.float16

    @staticmethod
    def _coerce_output(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            tensor = output[0]
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise RuntimeError(f"Unexpected TensorRT VAE output type: {type(output).__name__}")

    @classmethod
    def load(
        cls,
        engine_path: Path,
        meta_path: Path,
        device: torch.device,
    ) -> "TrtVaeDecodeBackend":
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        logger.info("Loading TensorRT VAE decoder from %s", engine_path)
        module = _load_serialized_trt_module(engine_path=engine_path, device=device)

        expects_raw_latents = bool(meta.get("expects_raw_latents", True))
        batch_range = None
        raw_batch_range = meta.get("batch_range")
        if isinstance(raw_batch_range, list) and len(raw_batch_range) == 2:
            try:
                batch_range = (int(raw_batch_range[0]), int(raw_batch_range[1]))
            except Exception:
                batch_range = None
        opt_batch = meta.get("opt_batch")
        try:
            opt_batch = int(opt_batch) if opt_batch is not None else None
        except Exception:
            opt_batch = None

        validation = meta.get("validation") or {}
        if validation:
            passed = validation.get("passed")
            report_path = validation.get("report_path")
            if passed is False:
                message = (
                    "TensorRT VAE artifact is marked invalid by post-export validation"
                    f" (report={report_path}, meta={meta_path})."
                )
                if _require_validated_artifact():
                    raise RuntimeError(message)
                logger.warning("%s", message)
        elif _require_validated_artifact():
            raise RuntimeError(
                "TensorRT VAE artifact is missing post-export validation metadata: "
                f"{meta_path}"
            )

        backend = cls(
            module=module,
            device=device,
            expects_raw_latents=expects_raw_latents,
            batch_range=batch_range,
            opt_batch=opt_batch,
        )
        backend.warmup()
        return backend

    @torch.no_grad()
    def warmup(self) -> None:
        warmup_batch = self.opt_batch or 1
        if self.batch_range is not None:
            warmup_batch = max(self.batch_range[0], min(warmup_batch, self.batch_range[1]))
        dummy_latents = torch.randn(
            warmup_batch,
            4,
            32,
            32,
            device=self.device,
            dtype=self.runtime_dtype,
        )
        output = self.module(dummy_latents)
        output = self._coerce_output(output)
        if output.dim() != 4 or output.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected TensorRT VAE warmup output shape: {tuple(output.shape)}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        scaling_factor: float,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        batch_size = int(latents.shape[0])
        if self.batch_range is not None:
            min_batch, max_batch = self.batch_range
            if not (min_batch <= batch_size <= max_batch):
                raise RuntimeError(
                    "TensorRT VAE engine batch support mismatch: "
                    f"got batch={batch_size}, supported range=[{min_batch}, {max_batch}]."
                )

        if self.expects_raw_latents:
            model_input = latents
        else:
            model_input = (1 / scaling_factor) * latents

        model_input = model_input.to(device=self.device, dtype=self.runtime_dtype)
        output = self.module(model_input)
        output = self._coerce_output(output)
        if output_dtype is not None and output.dtype != output_dtype:
            output = output.to(dtype=output_dtype)
        return output


class HybridTrtVaeDecodeBackend:
    """
    Added code: hybrid VAE backend that keeps the decoder mid-block on the
    PyTorch side while using TRT for the safe prefix and suffix stages.
    """

    name = "tensorrt_hybrid"

    def __init__(
        self,
        pre_module,
        mid_block: torch.nn.Module,
        tail_module,
        device: torch.device,
        runtime_dtype: torch.dtype = torch.float16,
        batch_range: Optional[tuple[int, int]] = None,
        opt_batch: Optional[int] = None,
    ):
        self.pre_module = pre_module
        self.mid_block = mid_block
        self.tail_module = tail_module
        self.device = device
        self.runtime_dtype = runtime_dtype
        self.batch_range = batch_range
        self.opt_batch = opt_batch

    @classmethod
    def load(
        cls,
        base_dir: Path,
        meta: dict,
        vae_module: torch.nn.Module,
        device: torch.device,
    ) -> "HybridTrtVaeDecodeBackend":
        hybrid = meta.get("hybrid") or {}
        pre_path = _resolve_path(base_dir, hybrid.get("pre_path", "vae_pre_trt.ts"))
        tail_path = _resolve_path(base_dir, hybrid.get("tail_path", "vae_tail_trt.ts"))
        if not pre_path.exists():
            raise FileNotFoundError(f"Hybrid TRT pre stage not found: {pre_path}")
        if not tail_path.exists():
            raise FileNotFoundError(f"Hybrid TRT tail stage not found: {tail_path}")
        if vae_module is None:
            raise RuntimeError(
                "Hybrid TRT VAE backend requires the live VAE module so the decoder mid-block can run in PyTorch."
            )

        pre_module = _load_serialized_trt_module(pre_path, device=device)
        tail_module = _load_serialized_trt_module(tail_path, device=device)

        # Added code: keep the mid-block as a local PyTorch island so we can
        # bypass the first known-bad TRT stage without mutating the caller's
        # main VAE module.
        mid_block = copy.deepcopy(vae_module.decoder.mid_block).to(
            device=device,
            dtype=getattr(vae_module, "dtype", torch.float16),
        ).eval()
        mid_block.requires_grad_(False)

        batch_range = None
        raw_batch_range = meta.get("batch_range")
        if isinstance(raw_batch_range, list) and len(raw_batch_range) == 2:
            try:
                batch_range = (int(raw_batch_range[0]), int(raw_batch_range[1]))
            except Exception:
                batch_range = None

        opt_batch = meta.get("opt_batch")
        try:
            opt_batch = int(opt_batch) if opt_batch is not None else None
        except Exception:
            opt_batch = None

        return cls(
            pre_module=pre_module,
            mid_block=mid_block,
            tail_module=tail_module,
            device=device,
            runtime_dtype=next(mid_block.parameters()).dtype,
            batch_range=batch_range,
            opt_batch=opt_batch,
        )

    @torch.no_grad()
    def warmup(self) -> None:
        warmup_batch = self.opt_batch or 1
        if self.batch_range is not None:
            warmup_batch = max(self.batch_range[0], min(warmup_batch, self.batch_range[1]))
        dummy_latents = torch.randn(
            warmup_batch,
            4,
            32,
            32,
            device=self.device,
            dtype=self.runtime_dtype,
        )
        output = self.decode(
            latents=dummy_latents,
            scaling_factor=1.0,
            output_dtype=self.runtime_dtype,
        )
        if output.dim() != 4 or output.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected hybrid TRT VAE warmup output shape: {tuple(output.shape)}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        scaling_factor: float,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        del scaling_factor  # scaling is already embedded in the TRT prefix stage

        batch_size = int(latents.shape[0])
        if self.batch_range is not None:
            min_batch, max_batch = self.batch_range
            if not (min_batch <= batch_size <= max_batch):
                raise RuntimeError(
                    "Hybrid TRT VAE engine batch support mismatch: "
                    f"got batch={batch_size}, supported range=[{min_batch}, {max_batch}]."
                )

        sample = self.pre_module(latents.to(device=self.device, dtype=self.runtime_dtype))
        if isinstance(sample, (tuple, list)):
            sample = sample[0]
        sample = self.mid_block(sample.to(dtype=self.runtime_dtype), None)
        output = self.tail_module(sample.to(dtype=self.runtime_dtype))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output_dtype is not None and output.dtype != output_dtype:
            output = output.to(dtype=output_dtype)
        return output


def load_vae_trt_decoder(
    device: Optional[torch.device] = None,
    scaling_factor: Optional[float] = None,
    vae_module: Optional[torch.nn.Module] = None,
    trt_dir: Optional[Path] = None,
    force: bool = False,
):
    """
    Load the TensorRT VAE decoder backend when explicitly requested.

    Returns `None` when the backend is disabled or unavailable, so the caller
    can continue using the standard PyTorch VAE path.
    """
    if not force and not _trt_vae_requested():
        return None

    resolved_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if resolved_device.type != "cuda":
        logger.warning("TensorRT VAE requested, but CUDA is unavailable; using PyTorch VAE")
        return None

    if trt_dir is not None:
        os.environ["MUSETALK_TRT_DIR"] = str(Path(trt_dir).resolve())

    if _stagewise_backend_requested():
        try:
            backend = StagewiseTrtVaeDecodeBackend.load(
                vae_module=vae_module,
                device=resolved_device,
                scaling_factor=float(scaling_factor or 1.0),
            )
            backend.warmup()
            logger.info("TensorRT VAE backend is active")
            return backend
        except Exception as exc:
            if _allow_fallback():
                logger.warning(
                    "Failed to activate stagewise TensorRT VAE backend (%s: %s); falling back to PyTorch VAE",
                    type(exc).__name__,
                    exc,
                )
                return None
            raise

    engine_path = _trt_vae_path()
    meta_path = _trt_vae_meta_path(engine_path)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    # Added code: hybrid stage exports do not necessarily emit a single
    # `vae_decoder_trt.ts` file. Allow activation to proceed when the metadata
    # says this is a hybrid artifact directory.
    if not engine_path.exists() and meta.get("type") != "vae_decoder_hybrid":
        message = f"TensorRT VAE engine not found: {engine_path}"
        if _allow_fallback():
            logger.warning("%s; falling back to PyTorch VAE", message)
            return None
        raise FileNotFoundError(message)

    try:
        if meta.get("type") == "vae_decoder_hybrid":
            backend = HybridTrtVaeDecodeBackend.load(
                base_dir=engine_path.parent,
                meta=meta,
                vae_module=vae_module,
                device=resolved_device,
            )
            validation = meta.get("validation") or {}
            if validation:
                passed = validation.get("passed")
                report_path = validation.get("report_path")
                if passed is False:
                    message = (
                        "Hybrid TensorRT VAE artifact is marked invalid by post-export validation"
                        f" (report={report_path}, meta={meta_path})."
                    )
                    if _require_validated_artifact():
                        raise RuntimeError(message)
                    logger.warning("%s", message)
            elif _require_validated_artifact():
                raise RuntimeError(
                    "Hybrid TensorRT VAE artifact is missing post-export validation metadata: "
                    f"{meta_path}"
                )
            backend.warmup()
        else:
            backend = TrtVaeDecodeBackend.load(
                engine_path=engine_path,
                meta_path=meta_path,
                device=resolved_device,
            )
        logger.info("TensorRT VAE backend is active")
        return backend
    except Exception as exc:
        if _allow_fallback():
            logger.warning(
                "Failed to activate TensorRT VAE backend (%s: %s); falling back to PyTorch VAE",
                type(exc).__name__,
                exc,
            )
            return None
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    backend = load_vae_trt_decoder()
    if backend is None:
        logger.info("TensorRT VAE backend is not active")
    else:
        logger.info("TensorRT VAE backend loaded successfully")
