import copy
import json
import logging
import os
import pickle
import time
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


def _requested_unet_backend() -> str:
    return os.getenv("MUSETALK_UNET_BACKEND", "").strip().lower()


def _trt_vae_requested() -> bool:
    requested_backend = _requested_vae_backend()
    if requested_backend in {"trt", "tensorrt", "trt_stagewise", "tensorrt_stagewise"}:
        return True
    return _env_enabled("MUSETALK_TRT_ENABLED", "0") or _env_enabled(
        "MUSETALK_TRT_VAE_ENABLED", "0"
    )


def _trt_unet_requested() -> bool:
    requested_backend = _requested_unet_backend()
    if requested_backend in {"trt", "tensorrt"}:
        return True
    return _env_enabled("MUSETALK_TRT_UNET_ENABLED", "0")


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


def _trt_unet_path() -> Path:
    return Path(os.getenv("MUSETALK_TRT_UNET_PATH", str(_trt_dir() / "unet_trt.ts")))


def _trt_unet_meta_path(engine_path: Path) -> Path:
    default_meta = engine_path.with_name("unet_trt_meta.json")
    return Path(os.getenv("MUSETALK_TRT_UNET_META_PATH", str(default_meta)))


def _trt_unet_path_map() -> dict[int, Path]:
    """
    Parse optional exact-batch UNet TensorRT artifacts.

    Format:
      MUSETALK_TRT_UNET_PATHS=8:models/.../unet_trt.ts,16:models/.../unet_trt.ts
    """
    raw = os.getenv("MUSETALK_TRT_UNET_PATHS", "").strip()
    if not raw:
        return {}

    paths: dict[int, Path] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise RuntimeError(
                "Invalid MUSETALK_TRT_UNET_PATHS entry "
                f"{token!r}; expected '<batch>:<path>'."
            )
        batch_raw, path_raw = token.split(":", 1)
        batch_size = int(batch_raw.strip())
        if batch_size <= 0:
            raise RuntimeError(
                "Invalid MUSETALK_TRT_UNET_PATHS batch size: "
                f"{batch_raw!r}."
            )
        paths[batch_size] = Path(path_raw.strip())
    return paths


def _allow_unvalidated_unet_trt() -> bool:
    return _env_enabled("MUSETALK_TRT_UNET_ALLOW_UNVALIDATED", "0")


def _parse_torch_dtype(raw_value: object, default: torch.dtype = torch.float16) -> torch.dtype:
    value = str(raw_value or "").replace("torch.", "").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    return default


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_positive_int_list(raw: str) -> list[int]:
    parsed: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        parsed.append(value)
    return parsed


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _stagewise_torch_stage_names() -> set[str]:
    return set(_parse_csv_env("MUSETALK_TRT_STAGEWISE_TORCH_STAGES", ""))


def _stagewise_precision_policy() -> str:
    value = os.getenv("MUSETALK_TRT_STAGEWISE_PRECISION", "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return "fp16"
    if value in {"int8", "int8_mixed", "mixed_int8"}:
        return "int8_mixed"
    raise RuntimeError(
        "Unsupported MUSETALK_TRT_STAGEWISE_PRECISION value: "
        f"{value!r}. Expected fp16 or int8_mixed."
    )


def _stagewise_int8_stage_names() -> set[str]:
    return set(_parse_csv_env("MUSETALK_TRT_STAGEWISE_INT8_STAGES", ""))


def _stagewise_int8_calibration_dir() -> Optional[Path]:
    raw = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR", "").strip()
    if not raw:
        raw = os.getenv("MUSETALK_VAE_CALIBRATION_DIR", "").strip()
    return Path(raw) if raw else None


def _stagewise_int8_calibration_batches() -> int:
    try:
        return max(1, int(os.getenv("MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_BATCHES", "8")))
    except Exception:
        return 8


def _stagewise_int8_calibration_algo() -> str:
    value = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO", "minmax")
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "minmax": "minmax",
        "min_max": "minmax",
        "entropy": "entropy",
        "entropy2": "entropy2",
        "entropy_2": "entropy2",
        "legacy": "legacy",
    }
    if value not in aliases:
        raise RuntimeError(
            "Unsupported MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO value: "
            f"{value!r}. Expected minmax, entropy2, entropy, or legacy."
        )
    return aliases[value]


def _stagewise_int8_cache_dir() -> Path:
    raw = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR", "").strip()
    if raw:
        return Path(raw)
    return _trt_dir() / "stagewise_int8_calibration_cache"


def _stagewise_int8_min_block_size() -> int:
    try:
        return max(1, int(os.getenv("MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE", "1")))
    except Exception:
        return 1


def _stagewise_int8_require_full_compilation() -> bool:
    return _env_enabled("MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION", "0")


def _stagewise_int8_require_calibration_cache() -> bool:
    return _env_enabled("MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE", "1")


def _stagewise_int8_frontend() -> str:
    value = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_FRONTEND", "onnx_qdq")
    value = value.strip().lower().replace("-", "_")
    if value in {"onnx_qdq", "torchscript_ptq"}:
        return value
    if value in {"ptq", "ts_ptq", "torchscript"}:
        return "torchscript_ptq"
    raise RuntimeError(
        "Unsupported MUSETALK_TRT_STAGEWISE_INT8_FRONTEND value: "
        f"{value!r}. Expected onnx_qdq or torchscript_ptq."
    )


def _stagewise_int8_calibration_format() -> str:
    value = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT", "tensor")
    value = value.strip().lower()
    if value in {"tensor", "list"}:
        return value
    raise RuntimeError(
        "Unsupported MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT value: "
        f"{value!r}. Expected tensor or list."
    )


def _stagewise_int8_enabled_precisions(runtime_dtype: torch.dtype) -> set[torch.dtype]:
    raw = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS", "int8")
    resolved: set[torch.dtype] = set()
    for token in raw.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value in {"runtime", "vae"}:
            resolved.add(runtime_dtype)
            continue
        if value in {"fp16", "float16", "half"}:
            resolved.add(torch.float16)
            continue
        if value in {"fp32", "float32", "float"}:
            resolved.add(torch.float32)
            continue
        if value in {"int8", "i8"}:
            resolved.add(torch.int8)
            continue
        raise RuntimeError(
            "Unsupported MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS token: "
            f"{token!r}. Expected int8, fp16, fp32, or runtime."
        )
    return resolved or {torch.int8}


def _stagewise_int8_ts_torch_executed_ops() -> list[str]:
    raw = os.getenv("MUSETALK_TRT_STAGEWISE_INT8_TORCH_EXECUTED_OPS", "group_norm")
    mapping = {
        "group_norm": "aten::group_norm",
        "native_group_norm": "aten::native_group_norm",
        "silu": "aten::silu",
        "upsample_nearest2d": "aten::upsample_nearest2d",
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue
        lowered = value.lower()
        op_name = value if value.startswith("aten::") else mapping.get(lowered)
        if not op_name:
            raise RuntimeError(
                "Unsupported MUSETALK_TRT_STAGEWISE_INT8_TORCH_EXECUTED_OPS token: "
                f"{token!r}. Expected group_norm, native_group_norm, silu, "
                "upsample_nearest2d, or an aten:: operator name."
            )
        if op_name not in seen:
            resolved.append(op_name)
            seen.add(op_name)
    return resolved


_STAGEWISE_VAE_STAGE_NAMES = (
    "decoder_pre",
    "decoder_mid_block",
    "decoder_up_block_0",
    "decoder_up_block_1",
    "decoder_up_block_2",
    "decoder_up_block_3",
    "decoder_postprocess",
)


def _stagewise_int8_live_blocked_stage_names() -> set[str]:
    raw = os.getenv(
        "MUSETALK_TRT_STAGEWISE_INT8_LIVE_BLOCKED_STAGES",
        ",".join(_STAGEWISE_VAE_STAGE_NAMES),
    )
    if raw.strip().lower() in {"", "0", "false", "no", "off", "none"}:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


class _StageCalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[torch.Tensor], as_input_list: bool = False):
        self.samples = [sample.detach().contiguous() for sample in samples]
        self.as_input_list = bool(as_input_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.as_input_list:
            return [sample]
        return sample


def _cache_file_has_bytes(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _trt_dtype_to_torch(dtype) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    if hasattr(trt.DataType, "BOOL") and dtype == trt.DataType.BOOL:
        return torch.bool
    raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dtype}")


class _TensorRtOnnxStage(torch.nn.Module):
    def __init__(
        self,
        engine_bytes: bytes,
        device: torch.device,
        logger_severity: Optional[int] = None,
    ):
        super().__init__()
        import tensorrt as trt

        severity = trt.Logger.WARNING if logger_severity is None else logger_severity
        self.trt_logger = trt.Logger(severity)
        self.runtime = trt.Runtime(self.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT ONNX/QDQ stage engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        self.device = device
        self.input_name = None
        self.output_name = None
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("TensorRT ONNX/QDQ stage must have one input and one output.")

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.contiguous()
        output_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(self.output_name))
        output_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.output_name))
        output = torch.empty(
            output_shape,
            device=self.device,
            dtype=output_dtype,
        )
        if not self.context.set_tensor_address(self.input_name, sample.data_ptr()):
            raise RuntimeError(f"Failed to bind TensorRT input tensor {self.input_name}.")
        if not self.context.set_tensor_address(self.output_name, output.data_ptr()):
            raise RuntimeError(f"Failed to bind TensorRT output tensor {self.output_name}.")
        stream = torch.cuda.current_stream(device=self.device).cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT ONNX/QDQ stage execution failed.")
        return output


def _stagewise_warmup_batches() -> list[int]:
    # Local modification: this differs from the original MuseTalk code.
    # Warm the exact live batch buckets at startup so the first real request
    # does not pay the compile penalty for newly enabled stagewise shapes.
    explicit = os.getenv("MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES", "").strip()
    if explicit:
        parsed = _parse_positive_int_list(explicit)
        if parsed:
            return parsed

    fixed_sizes = os.getenv("HLS_SCHEDULER_FIXED_BATCH_SIZES", "").strip()
    if fixed_sizes:
        parsed = _parse_positive_int_list(fixed_sizes)
        if parsed:
            return parsed

    max_batch = os.getenv("HLS_SCHEDULER_MAX_BATCH", "").strip()
    if max_batch:
        try:
            return [max(1, int(max_batch))]
        except Exception:
            pass

    return [4]


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
        precision_policy: str = "fp16",
        int8_stage_names: Optional[set[str]] = None,
        int8_calibration_dir: Optional[Path] = None,
        int8_calibration_batches: int = 8,
        int8_calibration_algo: str = "minmax",
        int8_cache_dir: Optional[Path] = None,
        int8_min_block_size: int = 1,
        int8_require_full_compilation: bool = False,
        int8_require_calibration_cache: bool = True,
        int8_frontend: str = "onnx_qdq",
        int8_calibration_format: str = "tensor",
        int8_enabled_precisions: Optional[set[torch.dtype]] = None,
        int8_ts_torch_executed_ops: Optional[list[str]] = None,
    ):
        self.device = device
        self.scaling_factor = float(scaling_factor)
        self.runtime_dtype = runtime_dtype
        self.workspace_gb = float(workspace_gb)
        self.min_block_size = int(min_block_size)
        self.torch_executed_ops = set(torch_executed_ops or set())
        self.torch_stage_names = set(torch_stage_names or set())
        self.precision_policy = precision_policy
        self.int8_stage_names = set(int8_stage_names or set())
        self.int8_frontend = int8_frontend
        if self.precision_policy == "int8_mixed" and not self.int8_stage_names:
            raise RuntimeError(
                "MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed requires an explicit "
                "MUSETALK_TRT_STAGEWISE_INT8_STAGES list. There is no safe default "
                "stage set yet."
            )
        if (
            self.precision_policy == "int8_mixed"
            and self.int8_frontend == "torchscript_ptq"
            and not _env_enabled(
                "MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES",
                "0",
            )
        ):
            blocked_stages = _stagewise_int8_live_blocked_stage_names()
            selected_blocked_stages = sorted(self.int8_stage_names & blocked_stages)
            if selected_blocked_stages:
                raise RuntimeError(
                    "VAE stagewise INT8 is blocked for live serving on this "
                    "Torch-TensorRT/TensorRT stack. Isolated PTQ experiments on "
                    f"{selected_blocked_stages} either crashed TensorRT "
                    "calibration or produced unusable decoded images. Keep the "
                    "API on MUSETALK_TRT_STAGEWISE_PRECISION=fp16, or run "
                    "scripts/experiment_vae_decoder_int8.py for offline "
                    "diagnostics. To deliberately bypass this live-serving guard, "
                    "set MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES=1."
                )
        self.int8_calibration_dir = int8_calibration_dir
        self.int8_calibration_batches = max(1, int(int8_calibration_batches))
        self.int8_calibration_algo = int8_calibration_algo
        self.int8_cache_dir = int8_cache_dir or _stagewise_int8_cache_dir()
        self.int8_min_block_size = max(1, int(int8_min_block_size))
        self.int8_require_full_compilation = bool(int8_require_full_compilation)
        self.int8_require_calibration_cache = bool(int8_require_calibration_cache)
        self.int8_calibration_format = int8_calibration_format
        self.int8_enabled_precisions = set(int8_enabled_precisions or {torch.int8})
        self.int8_ts_torch_executed_ops = list(int8_ts_torch_executed_ops or [])
        self.name = (
            "tensorrt_stagewise_int8_mixed"
            if self.precision_policy == "int8_mixed"
            else "tensorrt_stagewise"
        )
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
        self.stage_order = list(_STAGEWISE_VAE_STAGE_NAMES)
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
            precision_policy=_stagewise_precision_policy(),
            int8_stage_names=_stagewise_int8_stage_names(),
            int8_calibration_dir=_stagewise_int8_calibration_dir(),
            int8_calibration_batches=_stagewise_int8_calibration_batches(),
            int8_calibration_algo=_stagewise_int8_calibration_algo(),
            int8_cache_dir=_stagewise_int8_cache_dir(),
            int8_min_block_size=_stagewise_int8_min_block_size(),
            int8_require_full_compilation=_stagewise_int8_require_full_compilation(),
            int8_require_calibration_cache=_stagewise_int8_require_calibration_cache(),
            int8_frontend=_stagewise_int8_frontend(),
            int8_calibration_format=_stagewise_int8_calibration_format(),
            int8_enabled_precisions=_stagewise_int8_enabled_precisions(runtime_dtype),
            int8_ts_torch_executed_ops=_stagewise_int8_ts_torch_executed_ops(),
        )

    def _is_int8_stage(self, stage_name: str) -> bool:
        return (
            self.precision_policy == "int8_mixed"
            and stage_name in self.int8_stage_names
            and stage_name not in self.torch_stage_names
        )

    def _stage_enabled_precisions(self, stage_name: str) -> set[torch.dtype]:
        if self._is_int8_stage(stage_name):
            return set(self.int8_enabled_precisions)
        return _stagewise_enabled_precisions(self.runtime_dtype)

    def _load_int8_calibration_latents(self, batch_size: int) -> list[torch.Tensor]:
        if self.precision_policy != "int8_mixed":
            return []
        if self.int8_calibration_dir is None:
            raise RuntimeError(
                "MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed requires "
                "MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR or "
                "MUSETALK_VAE_CALIBRATION_DIR."
            )
        if not self.int8_calibration_dir.exists():
            raise FileNotFoundError(
                f"Stagewise INT8 calibration directory not found: {self.int8_calibration_dir}"
            )

        batches: list[torch.Tensor] = []
        for path in sorted(self.int8_calibration_dir.rglob("*.pt")):
            payload = _torch_load_cpu(path)
            if isinstance(payload, dict):
                latents = payload.get("pred_latents")
                if latents is None:
                    latents = payload.get("latents")
            else:
                latents = payload
            if not isinstance(latents, torch.Tensor):
                continue
            if latents.dim() != 4 or latents.shape[1] != 4:
                continue
            if latents.shape[0] < batch_size:
                continue
            batches.append(
                latents[:batch_size]
                .to(device=self.device, dtype=self.runtime_dtype)
                .contiguous()
            )
            if len(batches) >= self.int8_calibration_batches:
                break

        if not batches:
            raise RuntimeError(
                "No usable VAE INT8 calibration batches found under "
                f"{self.int8_calibration_dir} for batch={batch_size}."
            )
        return batches

    @staticmethod
    def _coerce_stage_output(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            tensor = output[0]
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise RuntimeError(f"Unexpected stagewise VAE stage output: {type(output).__name__}")

    def _int8_calibration_algo_type(self):
        from torch_tensorrt.ts.ptq import CalibrationAlgo

        if self.int8_calibration_algo == "minmax":
            return CalibrationAlgo.MINMAX_CALIBRATION
        if self.int8_calibration_algo == "entropy2":
            return CalibrationAlgo.ENTROPY_CALIBRATION_2
        if self.int8_calibration_algo == "entropy":
            return CalibrationAlgo.ENTROPY_CALIBRATION
        if self.int8_calibration_algo == "legacy":
            return CalibrationAlgo.LEGACY_CALIBRATION
        raise RuntimeError(
            f"Unsupported stagewise INT8 calibration algorithm: {self.int8_calibration_algo}"
        )

    def _make_int8_calibrator(
        self,
        stage_name: str,
        calibration_inputs: list[torch.Tensor],
    ):
        if not calibration_inputs:
            raise RuntimeError(
                f"Stage {stage_name} was selected for INT8, but no calibration inputs are available."
            )
        from torch.utils.data import DataLoader
        from torch_tensorrt.ts.ptq import DataLoaderCalibrator

        self.int8_cache_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(calibration_inputs[0].shape[0])
        cache_file = self.int8_cache_dir / (
            f"{stage_name}_bs{batch_size}_{self.int8_calibration_algo}.cache"
        )
        use_cache = _env_enabled("MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE", "1") and _cache_file_has_bytes(cache_file)
        dataset = _StageCalibrationDataset(
            [
                sample.to(device=self.device, dtype=self.runtime_dtype).contiguous()
                for sample in calibration_inputs
            ],
            as_input_list=self.int8_calibration_format == "list",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda samples: samples[0],
        )
        first = calibration_inputs[0].detach()
        logger.info(
            (
                "Preparing stagewise INT8 calibrator for %s with %d batches "
                "algo=%s cache=%s use_cache=%s input_format=%s first_shape=%s "
                "first_range=[%.5f, %.5f]"
            ),
            stage_name,
            len(calibration_inputs),
            self.int8_calibration_algo,
            cache_file,
            use_cache,
            self.int8_calibration_format,
            tuple(int(dim) for dim in first.shape),
            float(first.float().min().item()),
            float(first.float().max().item()),
        )
        return (
            DataLoaderCalibrator(
                dataloader,
                algo_type=self._int8_calibration_algo_type(),
                cache_file=str(cache_file),
                use_cache=use_cache,
                device=self.device,
            ),
            cache_file,
            use_cache,
        )

    def _compile_stage_int8_ptq(
        self,
        torch_tensorrt,
        stage_name: str,
        module: torch.nn.Module,
        example: torch.Tensor,
        shape: tuple[int, ...],
        inputs: list,
        calibration_inputs: list[torch.Tensor],
    ):
        calibrator, cache_file, use_cache = self._make_int8_calibrator(
            stage_name=stage_name,
            calibration_inputs=calibration_inputs,
        )
        traced = torch.jit.trace(
            module,
            example,
            strict=False,
            check_trace=False,
        ).eval()
        logger.info(
            (
                "Compiling stagewise TRT VAE stage %s for batch=%s precision=int8_ptq "
                "enabled_precisions=%s min_block_size=%s require_full_compilation=%s "
                "torch_executed_ops=%s"
            ),
            stage_name,
            shape[0],
            sorted(str(dtype) for dtype in self._stage_enabled_precisions(stage_name)),
            self.int8_min_block_size,
            self.int8_require_full_compilation,
            self.int8_ts_torch_executed_ops,
        )
        compile_kwargs = {
            "ir": "ts",
            "inputs": inputs,
            "enabled_precisions": self._stage_enabled_precisions(stage_name),
            "workspace_size": int(self.workspace_gb * (1 << 30)),
            "min_block_size": self.int8_min_block_size,
            "require_full_compilation": self.int8_require_full_compilation,
            "calibrator": calibrator,
            "truncate_long_and_double": True,
        }
        if self.int8_ts_torch_executed_ops and not self.int8_require_full_compilation:
            compile_kwargs["torch_executed_ops"] = self.int8_ts_torch_executed_ops
        compiled = torch_tensorrt.compile(
            traced,
            **compile_kwargs,
        ).eval()
        if (
            self.int8_require_calibration_cache
            and not use_cache
            and not _cache_file_has_bytes(cache_file)
        ):
            raise RuntimeError(
                "Stagewise INT8 calibration finished without writing a non-empty "
                f"cache for stage={stage_name} batch={shape[0]} cache={cache_file}. "
                "This usually means TensorRT did not calibrate any INT8 layers for "
                "the selected stage. Try a different stage, "
                "MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS=int8, "
                "MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE=1, or set "
                "MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION=1 for a "
                "stricter diagnostic build."
            )
        return compiled

    def _compile_stage_int8_onnx_qdq(
        self,
        stage_name: str,
        module: torch.nn.Module,
        example: torch.Tensor,
        shape: tuple[int, ...],
        calibration_inputs: list[torch.Tensor],
    ):
        if not calibration_inputs:
            raise RuntimeError(
                f"Stage {stage_name} was selected for INT8, but no calibration inputs are available."
            )
        try:
            import modelopt.torch.quantization as mtq
            import tensorrt as trt
        except Exception as exc:
            raise RuntimeError(
                "MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq requires "
                "nvidia-modelopt, onnx, and TensorRT in the active venv. Run "
                "scripts/setup_trt_stagewise_server_env.sh --install-modelopt "
                "or install nvidia-modelopt==0.23.2 and onnx<1.18."
            ) from exc

        self.int8_cache_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(shape[0])
        artifact_stem = f"{stage_name}_bs{batch_size}_{self.int8_calibration_algo}_onnx_qdq"
        onnx_path = self.int8_cache_dir / f"{artifact_stem}.onnx"
        engine_path = self.int8_cache_dir / f"{artifact_stem}.plan"
        use_cache = _env_enabled("MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE", "1")

        if use_cache and _cache_file_has_bytes(engine_path):
            logger.info(
                "Loading cached TensorRT ONNX/QDQ INT8 stage %s batch=%s from %s",
                stage_name,
                batch_size,
                engine_path,
            )
            return _TensorRtOnnxStage(engine_path.read_bytes(), self.device).eval()

        quantized_module = copy.deepcopy(module).to(
            device=self.device,
            dtype=self.runtime_dtype,
        ).eval()
        quantized_module.requires_grad_(False)
        calibration_samples = [
            sample.to(device=self.device, dtype=self.runtime_dtype).contiguous()
            for sample in calibration_inputs
        ]

        def forward_loop(model):
            with torch.no_grad():
                for calibration_sample in calibration_samples:
                    model(calibration_sample.contiguous())

        logger.info(
            (
                "Quantizing stagewise VAE stage %s with ModelOpt ONNX/QDQ "
                "batch=%s calibration_batches=%s"
            ),
            stage_name,
            batch_size,
            len(calibration_samples),
        )
        quantized_module = mtq.quantize(
            quantized_module,
            mtq.INT8_DEFAULT_CFG,
            forward_loop,
        ).eval()

        logger.info("Exporting ONNX/QDQ INT8 stage %s to %s", stage_name, onnx_path)
        with torch.no_grad():
            torch.onnx.export(
                quantized_module,
                example,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                do_constant_folding=True,
            )

        logger.info("Building TensorRT ONNX/QDQ INT8 stage %s to %s", stage_name, engine_path)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(
                f"Failed to parse ONNX/QDQ INT8 stage {stage_name}: {errors}"
            )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(self.workspace_gb * (1 << 30)),
        )
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT failed to build ONNX/QDQ INT8 stage {stage_name}.")
        engine_bytes = bytes(serialized_engine)
        engine_path.write_bytes(engine_bytes)
        return _TensorRtOnnxStage(engine_bytes, self.device).eval()

    def _compile_stage(
        self,
        stage_name: str,
        sample_input: torch.Tensor,
        calibration_inputs: Optional[list[torch.Tensor]] = None,
    ):
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
            "enabled_precisions": self._stage_enabled_precisions(stage_name),
            "workspace_size": int(self.workspace_gb * (1 << 30)),
            "min_block_size": self.min_block_size,
            "pass_through_build_failures": False,
            "require_full_compilation": not bool(self.torch_executed_ops),
            "truncate_long_and_double": True,
        }
        if self.torch_executed_ops:
            compile_kwargs["torch_executed_ops"] = self.torch_executed_ops
        module = self.stage_modules[stage_name]
        if self._is_int8_stage(stage_name):
            if self.int8_frontend == "onnx_qdq":
                return self._compile_stage_int8_onnx_qdq(
                    stage_name=stage_name,
                    module=module,
                    example=example,
                    shape=shape,
                    calibration_inputs=calibration_inputs or [],
                )
            return self._compile_stage_int8_ptq(
                torch_tensorrt=torch_tensorrt,
                stage_name=stage_name,
                module=module,
                example=example,
                shape=shape,
                inputs=inputs,
                calibration_inputs=calibration_inputs or [],
            )
        logger.info(
            "Compiling stagewise TRT VAE stage %s for batch=%s precision=%s",
            stage_name,
            shape[0],
            "int8_mixed" if self._is_int8_stage(stage_name) else "fp16",
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
        calibration_current = self._load_int8_calibration_latents(batch_size)
        with torch.no_grad():
            current = sample
            for stage_name in self.stage_order:
                compiled_stage = self._compile_stage(
                    stage_name,
                    current,
                    calibration_inputs=calibration_current
                    if self._is_int8_stage(stage_name)
                    else None,
                )
                compiled[stage_name] = compiled_stage
                current = self._coerce_stage_output(compiled_stage(current.contiguous()))
                current = current.to(dtype=self.runtime_dtype).contiguous()
                if calibration_current:
                    next_calibration = []
                    for calibration_sample in calibration_current:
                        calibration_output = self._coerce_stage_output(
                            compiled_stage(calibration_sample.contiguous())
                        )
                        next_calibration.append(
                            calibration_output.to(dtype=self.runtime_dtype).contiguous()
                        )
                    calibration_current = next_calibration
        self.compiled_by_batch[batch_size] = compiled
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def warmup(self, batch_sizes: Optional[list[int]] = None) -> None:
        # Local modification: this differs from the original MuseTalk code.
        # Startup warmup now follows the configured live bucket sizes and logs
        # visible progress so batch compiles do not look like a silent freeze.
        resolved_batches = batch_sizes or _stagewise_warmup_batches()
        resolved_batches = [max(1, int(batch)) for batch in resolved_batches]
        if not resolved_batches:
            resolved_batches = [4]

        warmup_started_at = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        warmup_finished_at = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None

        precision_detail = self.precision_policy
        if self.precision_policy == "int8_mixed":
            precision_detail += f" stages={sorted(self.int8_stage_names)}"
        print(f"🔥 Stagewise TRT warmup batches: {resolved_batches} precision={precision_detail}")

        total_wall_started_at = time.time()
        for batch_size in resolved_batches:
            compile_started_at = time.time()
            already_ready = batch_size in self.compiled_by_batch
            if already_ready:
                print(f"   ♻️  Stagewise TRT batch={batch_size} already warm")
                continue

            print(f"   🔥 Warming stagewise TRT batch={batch_size}...")
            if warmup_started_at is not None and warmup_finished_at is not None:
                warmup_started_at.record()
            self._ensure_batch(batch_size)
            if warmup_started_at is not None and warmup_finished_at is not None:
                warmup_finished_at.record()
                torch.cuda.synchronize(self.device)
                elapsed_s = warmup_started_at.elapsed_time(warmup_finished_at) / 1000.0
            else:
                elapsed_s = time.time() - compile_started_at
            print(f"   ✅ Stagewise TRT batch={batch_size} ready in {elapsed_s:.2f}s")

        print(
            "✅ Stagewise TRT warmup complete "
            f"(batches={resolved_batches}, total={time.time() - total_wall_started_at:.2f}s)"
        )

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
            current = self._coerce_stage_output(compiled[stage_name](current.contiguous()))
            current = current.to(dtype=self.runtime_dtype).contiguous()
        if output_dtype is not None and current.dtype != output_dtype:
            current = current.to(dtype=output_dtype)
        return current


class _UNetBackendOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class TrtUnetBackend(torch.nn.Module):
    """
    Opt-in runtime wrapper for a serialized TensorRT MuseTalk UNet.

    The exported UNet graph accepts the same real scheduler inputs captured by
    `MUSETALK_UNET_CALIBRATION_CAPTURE`, except timesteps are baked into the
    export wrapper. The public call signature still matches diffusers-style
    UNet usage so existing scheduler code can keep reading `.sample`.
    """

    name = "tensorrt_unet"

    def __init__(
        self,
        module,
        device: torch.device,
        runtime_dtype: torch.dtype = torch.float16,
        batch_range: Optional[tuple[int, int]] = None,
        opt_batch: Optional[int] = None,
    ):
        super().__init__()
        self.module = module
        self.device = device
        self.runtime_dtype = runtime_dtype
        self.batch_range = batch_range
        self.opt_batch = opt_batch
        self.dtype = runtime_dtype

    @staticmethod
    def _coerce_output(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
            return output.sample
        if isinstance(output, (tuple, list)) and output:
            tensor = output[0]
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise RuntimeError(f"Unexpected TensorRT UNet output type: {type(output).__name__}")

    @classmethod
    def load(
        cls,
        engine_path: Path,
        meta_path: Path,
        device: torch.device,
    ) -> "TrtUnetBackend":
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            validation = meta.get("validation")
            if (
                isinstance(validation, dict)
                and validation.get("passed") is False
                and not _allow_unvalidated_unet_trt()
            ):
                raise RuntimeError(
                    "TensorRT UNet artifact failed post-export validation: "
                    f"{meta_path}"
                )

        logger.info("Loading TensorRT UNet from %s", engine_path)
        module = _load_serialized_trt_module(engine_path=engine_path, device=device)

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

        backend = cls(
            module=module,
            device=device,
            runtime_dtype=_parse_torch_dtype(meta.get("dtype"), torch.float16),
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
            8,
            32,
            32,
            device=self.device,
            dtype=self.runtime_dtype,
        )
        dummy_audio = torch.randn(
            warmup_batch,
            50,
            384,
            device=self.device,
            dtype=self.runtime_dtype,
        )
        output = self.module(dummy_latents, dummy_audio)
        output = self._coerce_output(output)
        if output.dim() != 4 or output.shape[1:] != (4, 32, 32):
            raise RuntimeError(
                f"Unexpected TensorRT UNet warmup output shape: {tuple(output.shape)}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def forward(
        self,
        latent: torch.Tensor,
        timesteps=None,
        *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> _UNetBackendOutput:
        del timesteps
        if encoder_hidden_states is None:
            raise RuntimeError("TensorRT UNet backend requires encoder_hidden_states")

        batch_size = int(latent.shape[0])
        if self.batch_range is not None:
            min_batch, max_batch = self.batch_range
            if not (min_batch <= batch_size <= max_batch):
                raise RuntimeError(
                    "TensorRT UNet engine batch support mismatch: "
                    f"got batch={batch_size}, supported range=[{min_batch}, {max_batch}]."
                )

        model_latent = latent.to(device=self.device, dtype=self.runtime_dtype)
        model_audio = encoder_hidden_states.to(device=self.device, dtype=self.runtime_dtype)
        output = self.module(model_latent, model_audio)
        return _UNetBackendOutput(self._coerce_output(output))


class MultiTrtUnetBackend(torch.nn.Module):
    """
    Route padded scheduler batches across validated exact-batch TensorRT UNets.

    This lets the live server use a known-good batch-8 engine for both batch-8
    work and batch-16 work split into two exact batch-8 calls, without enabling
    an artifact that failed capture validation.
    """

    name = "tensorrt_unet_multi"

    def __init__(
        self,
        backends_by_batch: dict[int, TrtUnetBackend],
        device: torch.device,
    ):
        super().__init__()
        if not backends_by_batch:
            raise RuntimeError("MultiTrtUnetBackend requires at least one engine")
        self.backends_by_batch = torch.nn.ModuleDict(
            {str(batch): backend for batch, backend in sorted(backends_by_batch.items())}
        )
        self.device = device
        self.dtype = next(iter(backends_by_batch.values())).dtype

    def _backend_for_batch(self, batch_size: int) -> tuple[TrtUnetBackend, int]:
        exact_key = str(batch_size)
        if exact_key in self.backends_by_batch:
            return self.backends_by_batch[exact_key], batch_size

        for chunk_size in sorted((int(key) for key in self.backends_by_batch), reverse=True):
            if chunk_size < batch_size and batch_size % chunk_size == 0:
                return self.backends_by_batch[str(chunk_size)], chunk_size

        supported = ", ".join(sorted(self.backends_by_batch.keys(), key=int))
        raise RuntimeError(
            "TensorRT UNet engine batch support mismatch: "
            f"got batch={batch_size}, supported exact/splittable batches={supported}."
        )

    @torch.no_grad()
    def forward(
        self,
        latent: torch.Tensor,
        timesteps=None,
        *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> _UNetBackendOutput:
        if encoder_hidden_states is None:
            raise RuntimeError("TensorRT UNet backend requires encoder_hidden_states")

        batch_size = int(latent.shape[0])
        backend, chunk_size = self._backend_for_batch(batch_size)
        if chunk_size == batch_size:
            return backend(
                latent,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            )

        outputs = []
        for start in range(0, batch_size, chunk_size):
            end = start + chunk_size
            chunk_output = backend(
                latent[start:end],
                timesteps,
                encoder_hidden_states=encoder_hidden_states[start:end],
            )
            outputs.append(chunk_output.sample)
        return _UNetBackendOutput(torch.cat(outputs, dim=0))


def load_unet_trt_backend(
    device: Optional[torch.device] = None,
    trt_dir: Optional[Path] = None,
    force: bool = False,
):
    """
    Load a TensorRT UNet backend when explicitly requested.

    The returned module is call-compatible with the current scheduler's
    `unet.model(...).sample` path. It returns `None` when disabled or, with
    fallback enabled, when the artifact cannot be loaded. With fallback disabled,
    activation failures are raised.
    """
    if not force and not _trt_unet_requested():
        return None

    resolved_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if resolved_device.type != "cuda":
        message = "TensorRT UNet requested, but CUDA is unavailable"
        if _allow_fallback():
            logger.warning("%s; using PyTorch UNet", message)
            return None
        raise RuntimeError(message)

    if trt_dir is not None:
        os.environ["MUSETALK_TRT_DIR"] = str(Path(trt_dir).resolve())

    try:
        path_map = _trt_unet_path_map()
        if path_map:
            backends_by_batch: dict[int, TrtUnetBackend] = {}
            for batch_size, engine_path in sorted(path_map.items()):
                meta_path = engine_path.with_name("unet_trt_meta.json")
                if not engine_path.exists():
                    raise FileNotFoundError(
                        f"TensorRT UNet engine not found for batch {batch_size}: "
                        f"{engine_path}"
                    )
                backends_by_batch[batch_size] = TrtUnetBackend.load(
                    engine_path=engine_path,
                    meta_path=meta_path,
                    device=resolved_device,
                )
            backend = MultiTrtUnetBackend(
                backends_by_batch=backends_by_batch,
                device=resolved_device,
            )
        else:
            engine_path = _trt_unet_path()
            meta_path = _trt_unet_meta_path(engine_path)
            if not engine_path.exists():
                raise FileNotFoundError(f"TensorRT UNet engine not found: {engine_path}")
            backend = TrtUnetBackend.load(
                engine_path=engine_path,
                meta_path=meta_path,
                device=resolved_device,
            )
        logger.info("TensorRT UNet backend is active")
        return backend
    except Exception as exc:
        if _allow_fallback():
            logger.warning(
                "Failed to activate TensorRT UNet backend (%s: %s); falling back to PyTorch UNet",
                type(exc).__name__,
                exc,
            )
            return None
        raise


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

    Returns `None` when the backend is disabled or, with fallback enabled,
    unavailable. With fallback disabled, activation failures are raised so the
    caller does not silently run a non-TRT path.
    """
    if not force and not _trt_vae_requested():
        return None

    resolved_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if resolved_device.type != "cuda":
        message = "TensorRT VAE requested, but CUDA is unavailable"
        if _allow_fallback():
            logger.warning("%s; using PyTorch VAE", message)
            return None
        raise RuntimeError(message)

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
