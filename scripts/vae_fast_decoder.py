"""
Added code: a distilled tiny VAE decoder backend for the MuseTalk hot path.

The SD-VAE decoder is ~76% of the GPU batch on this pipeline (see
`docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md`). Years of
TensorRT/INT8 work took it from 142.5 ms -> 82.9 ms at batch 8. Swapping the
decoder for a distilled one (TAESD, 1.2M params vs 49.5M) takes the same work to
~12 ms, because it removes the high-resolution `up_block_2/3` convolutions
entirely rather than quantising them.

TAESD decodes the same SD 1.x latent space that `sd-vae-ft-mse` uses, so it is a
drop-in for MuseTalk's post-UNet latents. It implements the same backend
contract as `StagewiseTrtVaeDecodeBackend`: `.name`, `.warmup()`, and
`.decode(latents, scaling_factor, output_dtype) -> [0,1] NCHW float tensor`.

Enable with `MUSETALK_VAE_BACKEND=taesd`. Rollback is removing that one variable.

Added code (plan item 2.1, docs/musetalk_4070s_300fps_plan_2026-09-27.md): a
TensorRT FP16 build of the same vendored TAESD decoder, `TaesdTrtBackend`,
selected with `MUSETALK_TAESD_BACKEND=trt` (default `compiled` = the
torch.compile backend above, unchanged). It decodes the FULL 256x256 face (no
row crop: the chin tracker reads the whole generated face), runs static bs8
sub-batches (a partial tail is zero-padded), and offers two outputs:
  - `.decode(...)`        -> [0,1] fp16 NCHW, the same contract as above;
  - `.decode_bgr_u8(...)` -> uint8 BGR NHWC on the GPU, i.e. the repo's fast
    postprocess (musetalk/models/vae.py MUSETALK_VAE_FAST_POSTPROCESS) run as
    one TensorRT kernel on the decoder engine's own fp16 output. It is
    bit-identical to that postprocess for every finite fp16 input.
The two ~3 MB plans are persisted under models/taesd/trt/ keyed by a
fingerprint (TensorRT version, GPU, ONNX hash, batch, build recipe), built on
first use when missing, and verified at load with a fixed probe batch whose
output hashes are stored next to them. Build offline with
  python scripts/vae_fast_decoder.py build
Flags: MUSETALK_TAESD_BACKEND, MUSETALK_TAESD_TRT_{DIR,BATCH,BUILD,STRICT,
FUSED_POST,OPT_LEVEL,STRONGLY_TYPED} (see `_TRT_FLAG_DOC`).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "madebyollin/taesd"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int_list(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return sorted(out)


def taesd_backend_requested() -> bool:
    return os.getenv("MUSETALK_VAE_BACKEND", "").strip().lower() in {"taesd", "tiny", "tiny_vae"}


class TaesdVaeDecodeBackend:
    """Distilled tiny-VAE decoder with the stagewise backend's interface."""

    name = "taesd"

    def __init__(
        self,
        model,
        device: torch.device,
        runtime_dtype: torch.dtype,
        compile_enabled: bool,
        compile_mode: str,
    ) -> None:
        self.model = model
        self.device = device
        self.runtime_dtype = runtime_dtype
        self.compile_enabled = compile_enabled
        self.compile_mode = compile_mode
        self._compiled_by_batch: dict[int, object] = {}
        self._decode_count = 0
        self._decode_total_s = 0.0
        self.timing_enabled = _env_bool("MUSETALK_TAESD_TIMING", False)
        self.timing_interval = 50

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        device: torch.device,
        runtime_dtype: torch.dtype = torch.float16,
    ) -> "TaesdVaeDecodeBackend":
        from diffusers import AutoencoderTiny

        model_id = os.getenv("MUSETALK_TAESD_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        # Prefer the vendored copy so a cold start needs no network.
        local_dir = os.getenv("MUSETALK_TAESD_LOCAL_DIR", "").strip()
        if not local_dir:
            vendored = Path(__file__).resolve().parent.parent / "models" / "taesd"
            if (vendored / "config.json").exists():
                local_dir = str(vendored)
        source = local_dir if local_dir and Path(local_dir).exists() else model_id

        started_at = time.time()
        model = AutoencoderTiny.from_pretrained(source, torch_dtype=runtime_dtype)
        model = model.to(device).eval()
        model.requires_grad_(False)

        n_params = sum(p.numel() for p in model.decoder.parameters())
        logger.info(
            "Loaded TAESD decoder from %s in %.1fs (%.2fM decoder params, dtype=%s)",
            source,
            time.time() - started_at,
            n_params / 1e6,
            runtime_dtype,
        )

        backend = cls(
            model=model,
            device=device,
            runtime_dtype=runtime_dtype,
            compile_enabled=_env_bool("MUSETALK_TAESD_COMPILE", True),
            compile_mode=os.getenv("MUSETALK_TAESD_COMPILE_MODE", "max-autotune").strip()
            or "max-autotune",
        )
        return backend

    # --------------------------------------------------------------- compile
    def _raw_decode(self, latents: torch.Tensor) -> torch.Tensor:
        # TAESD consumes the SCALED latent (what the UNet emits) directly and
        # returns [-1, 1]; it carries scaling_factor=1.0 and does not want the
        # 1/scaling_factor division the AutoencoderKL path applies. Getting this
        # wrong silently costs ~16x in reconstruction error, so it is asserted
        # by scripts/validate_vae_backend.py rather than left to convention.
        image = self.model.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)

    def _callable_for_batch(self, batch_size: int):
        if not self.compile_enabled:
            return self._raw_decode
        fn = self._compiled_by_batch.get(batch_size)
        if fn is None:
            try:
                fn = torch.compile(self._raw_decode, mode=self.compile_mode, dynamic=False)
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning(
                    "torch.compile unavailable for TAESD (%s: %s); using eager",
                    type(exc).__name__,
                    exc,
                )
                self.compile_enabled = False
                return self._raw_decode
            self._compiled_by_batch[batch_size] = fn
        return fn

    # ---------------------------------------------------------------- warmup
    def warmup(self, batch_sizes: Optional[list[int]] = None) -> None:
        sizes = batch_sizes or _env_int_list("MUSETALK_TAESD_WARMUP_BATCHES", "8")
        if not sizes:
            return
        started_at = time.time()
        with torch.inference_mode():
            for batch_size in sizes:
                probe = torch.zeros(
                    (batch_size, 4, 32, 32), device=self.device, dtype=self.runtime_dtype
                )
                fn = self._callable_for_batch(batch_size)
                for _ in range(3):
                    fn(probe)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        logger.info(
            "TAESD warmup complete (batches=%s, compile=%s, total=%.2fs)",
            sizes,
            self.compile_enabled,
            time.time() - started_at,
        )

    # ---------------------------------------------------------------- decode
    def decode(
        self,
        latents: torch.Tensor,
        scaling_factor: float,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # scaling_factor is accepted for interface parity. TAESD does not use
        # the AutoencoderKL scaling convention; see _raw_decode.
        del scaling_factor

        started_at = time.perf_counter() if self.timing_enabled else 0.0
        current = latents.to(device=self.device, dtype=self.runtime_dtype).contiguous()
        fn = self._callable_for_batch(int(current.shape[0]))
        image = fn(current)
        if output_dtype is not None and image.dtype != output_dtype:
            image = image.to(dtype=output_dtype)

        if self.timing_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            self._decode_count += 1
            self._decode_total_s += time.perf_counter() - started_at
            if self._decode_count % self.timing_interval == 0:
                print(
                    f"TAESD decode timing calls={self._decode_count} "
                    f"avg={self._decode_total_s / self._decode_count:.4f}s",
                    flush=True,
                )
        return image


# ============================================================================
# Added code: TensorRT TAESD backend (plan item 2.1).
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRT_DIR = REPO_ROOT / "models" / "taesd" / "trt"

# Bump when the exported graph, the build recipe or the post kernel changes; it
# is part of the engine fingerprint, so a bump forces a fresh engine key.
TAESD_TRT_RECIPE = "taesd_decoder_fp16_full_height_v1"
TAESD_TRT_POST_RECIPE = "bgr_u8_nhwc_post_fp32_rne_v1"
TAESD_TRT_META_SCHEMA = "taesd_trt_engine_v1"
_PROBE_SEED = 20260928
_LATENT_CHANNELS = 4
_LATENT_HW = 32
_IMAGE_HW = 256

_TRT_FLAG_DOC = {
    "MUSETALK_TAESD_BACKEND": "compiled (default: torch.compile TAESD, today's behaviour) | trt",
    "MUSETALK_TAESD_TRT_DIR": "engine directory (default models/taesd/trt)",
    "MUSETALK_TAESD_TRT_BATCH": "static engine batch (default 8); larger inputs run as sub-batches",
    "MUSETALK_TAESD_TRT_BUILD": "1 (default): build the engine on first use when missing; 0: refuse",
    "MUSETALK_TAESD_TRT_STRICT": "0 (default): on any TRT load/verify failure fall back to compiled "
                                 "TAESD; 1: raise instead",
    "MUSETALK_TAESD_TRT_FUSED_POST": "1 (default): VAE.decode_latents uses decode_bgr_u8 (bit-identical "
                                     "fused post); 0: engine fp16 output + the repo torch post",
    "MUSETALK_TAESD_TRT_OPT_LEVEL": "builder optimization level (default 3); part of the fingerprint",
    "MUSETALK_TAESD_TRT_STRONGLY_TYPED": "0 (default): FP16 builder flag on the fp16 ONNX; 1: strongly "
                                         "typed network; part of the fingerprint",
    "MUSETALK_TAESD_TRT_HW_COMPAT": "none (default): engines for this GPU model (its name is in the key); "
                                    "ampere_plus: one pair of engines for every Ampere-or-newer GPU (TensorRT "
                                    "hardware compatibility; the key omits the GPU); part of the fingerprint",
}

# Hardware-compatible engines (MUSETALK_TAESD_TRT_HW_COMPAT=ampere_plus) run on any GPU of the level. The
# build GPU keeps the exact probe hashes; another GPU model is checked against the recorded fp16 probe
# image with this relative-L2 bound (a broken engine differs by O(1)).
TAESD_HW_COMPAT_MIN_CAPABILITY = {"ampere_plus": (8, 0)}
TAESD_CROSS_GPU_REL_L2_MAX = 0.01


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default


def taesd_trt_hw_compat(value: Optional[str] = None) -> str:
    """'none' (default) or 'ampere_plus' from MUSETALK_TAESD_TRT_HW_COMPAT (or the given value)."""
    raw = os.getenv("MUSETALK_TAESD_TRT_HW_COMPAT", "") if value is None else value
    raw = raw.strip().lower().replace("-", "_").replace("+", "_plus")
    if raw in ("", "0", "none", "off", "false", "no"):
        return "none"
    if raw in ("ampere_plus", "ampere"):
        return "ampere_plus"
    raise ValueError(f"MUSETALK_TAESD_TRT_HW_COMPAT={raw!r} must be none or ampere_plus")


def _check_hw_compat_device(hw_compat: str, device: torch.device) -> None:
    if hw_compat == "none":
        return
    cap = tuple(torch.cuda.get_device_capability(device))
    minimum = TAESD_HW_COMPAT_MIN_CAPABILITY[hw_compat]
    if cap < minimum:
        raise TaesdTrtVerificationError(
            f"TAESD TRT {hw_compat} engines need sm{minimum[0]}{minimum[1]} or newer; device is sm{cap[0]}{cap[1]}"
        )


def taesd_backend_kind() -> str:
    """'compiled' (default, today's behaviour) or 'trt' from MUSETALK_TAESD_BACKEND."""
    raw = os.getenv("MUSETALK_TAESD_BACKEND", "").strip().lower()
    if raw in ("", "compiled", "torch", "torch_compile", "default"):
        return "compiled"
    if raw in ("trt", "tensorrt"):
        return "trt"
    logger.warning(
        "Unknown MUSETALK_TAESD_BACKEND=%r (expected 'compiled' or 'trt'); using compiled TAESD", raw
    )
    return "compiled"


def _sha256_bytes(data) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def repo_fast_postprocess_gpu(image: torch.Tensor) -> torch.Tensor:
    """The GPU half of musetalk/models/vae.py MUSETALK_VAE_FAST_POSTPROCESS, verbatim.

    [0,1] NCHW -> uint8 BGR NHWC, still on the device. Used as the reference the
    fused TensorRT post is checked against (at load and by the gate script).
    """
    return (
        image.detach()
        .float()
        .mul(255)
        .round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .flip(1)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def taesd_probe_latents(batch: int) -> torch.Tensor:
    """Fixed CPU-generated probe batch (fp16, [batch,4,32,32]) used to verify engines."""
    generator = torch.Generator(device="cpu").manual_seed(_PROBE_SEED)
    probe = torch.randn(
        (batch, _LATENT_CHANNELS, _LATENT_HW, _LATENT_HW), generator=generator, dtype=torch.float32
    )
    return probe.to(torch.float16).contiguous()


class _TaesdDecodeGraph(torch.nn.Module):
    """Exactly TaesdVaeDecodeBackend._raw_decode, as a module for ONNX export."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        image = self.model.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)


def export_taesd_decoder_onnx(model, batch: int, device: torch.device) -> bytes:
    """Export the fp16 TAESD decode graph (static [batch,4,32,32] -> [batch,3,256,256])."""
    import warnings

    dtype = next(model.parameters()).dtype
    example = torch.zeros(
        (batch, _LATENT_CHANNELS, _LATENT_HW, _LATENT_HW), device=device, dtype=dtype
    )
    buffer = io.BytesIO()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _TaesdDecodeGraph(model).eval(),
            (example,),
            buffer,
            input_names=["latents"],
            output_names=["image"],
            opset_version=17,
            do_constant_folding=True,
        )
    return buffer.getvalue()


_TRT_LOGGER = None


def _trt_logger():
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        import tensorrt as trt

        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return _TRT_LOGGER


def taesd_trt_fingerprint(
    onnx_sha256: str, batch: int, device: torch.device, opt_level: int, strongly_typed: bool,
    hw_compat: str = "none",
) -> dict:
    import tensorrt as trt

    major, minor = torch.cuda.get_device_capability(device)
    fingerprint = {
        "recipe": TAESD_TRT_RECIPE,
        "post_recipe": TAESD_TRT_POST_RECIPE,
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{major}.{minor}",
        "onnx_sha256": onnx_sha256,
        "batch": int(batch),
        "latent_shape": [int(batch), _LATENT_CHANNELS, _LATENT_HW, _LATENT_HW],
        "precision": "fp16",
        "opt_level": int(opt_level),
        "strongly_typed": bool(strongly_typed),
    }
    if hw_compat != "none":
        # one engine for every GPU of the level: the key names the level, not this GPU model
        del fingerprint["gpu"], fingerprint["compute_capability"]
        fingerprint["hardware_compatibility_level"] = hw_compat
    return fingerprint


def _fingerprint_key(fingerprint: dict) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]


def taesd_trt_paths(engine_dir: Path, key: str) -> dict:
    return {
        "decoder": engine_dir / f"taesd_trt_{key}.decoder.plan",
        "post": engine_dir / f"taesd_trt_{key}.post_bgr_u8.plan",
        "meta": engine_dir / f"taesd_trt_{key}.json",
        "timing_cache": engine_dir / "taesd_trt_timing.cache",
        "probe_ref": engine_dir / f"taesd_trt_{key}.probe_fp16.pt",
        "lock": engine_dir / ".build.lock",
    }


def build_taesd_decoder_plan(
    onnx_bytes: bytes,
    opt_level: int = 3,
    strongly_typed: bool = False,
    timing_cache_path: Optional[Path] = None,
    workspace_bytes: int = 1 << 30,
    hw_compat: str = "none",
) -> bytes:
    """ONNX (fp16 TAESD decode graph) -> serialized FP16 TensorRT plan."""
    import tensorrt as trt

    log = _trt_logger()
    builder = trt.Builder(log)
    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, log)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse the TAESD ONNX: {errors}")
    config = builder.create_builder_config()
    if not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = int(opt_level)
    if hw_compat == "ampere_plus":
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    cache_blob = b""
    if timing_cache_path is not None and Path(timing_cache_path).exists():
        cache_blob = Path(timing_cache_path).read_bytes()
    timing_cache = config.create_timing_cache(cache_blob)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the TAESD decoder engine")
    if timing_cache_path is not None:
        _atomic_write(Path(timing_cache_path), bytes(config.get_timing_cache().serialize()))
    return bytes(plan)


def build_bgr_u8_post_plan(batch: int, height: int = _IMAGE_HW, width: int = _IMAGE_HW,
                           hw_compat: str = "none") -> bytes:
    """[batch,3,H,W] fp16 [0,1] -> [batch,H,W,3] uint8 BGR, as ONE TensorRT kernel.

    Mirrors the repo fast postprocess op for op, in a STRONGLY TYPED network so
    TensorRT cannot lower any of it to fp16: cast fp16->fp32, x255 (fp32),
    round-half-to-even (torch.round == TRT kROUND), clamp [0,255], cast to
    uint8 (exact: the value is already an integer in range), reverse the
    channel axis (RGB->BGR) and transpose to NHWC.
    """
    import numpy as np
    import tensorrt as trt

    log = _trt_logger()
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    image = network.add_input("image", trt.DataType.HALF, (batch, 3, height, width))

    keep_alive = []  # TensorRT reads constant weights at build time; keep them referenced

    def constant(value: float):
        weights = np.array([value], dtype=np.float32)
        keep_alive.append(weights)
        layer = network.add_constant((1, 1, 1, 1), weights)
        return layer.get_output(0)

    x = network.add_cast(image, trt.DataType.FLOAT).get_output(0)
    x = network.add_elementwise(x, constant(255.0), trt.ElementWiseOperation.PROD).get_output(0)
    x = network.add_unary(x, trt.UnaryOperation.ROUND).get_output(0)
    x = network.add_elementwise(x, constant(0.0), trt.ElementWiseOperation.MAX).get_output(0)
    x = network.add_elementwise(x, constant(255.0), trt.ElementWiseOperation.MIN).get_output(0)
    # Channel reverse + NHWC transpose are pure data movement, so doing them on the (already
    # integer-valued) fp32 tensor before the final uint8 cast is byte-identical to the repo post
    # (cast, then flip/permute). TensorRT 10.3 rejects Slice on UInt8 inputs.
    x = network.add_slice(
        x, start=(0, 2, 0, 0), shape=(batch, 3, height, width), stride=(1, -1, 1, 1)
    ).get_output(0)
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = trt.Permutation([0, 2, 3, 1])
    out = network.add_cast(shuffle.get_output(0), trt.DataType.UINT8).get_output(0)
    out.name = "bgr_u8"
    network.mark_output(out)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if hw_compat == "ampere_plus":
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    plan = builder.build_serialized_network(network, config)
    del keep_alive
    if plan is None:
        raise RuntimeError("TensorRT failed to build the TAESD uint8 BGR post engine")
    return bytes(plan)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class _TrtEngine:
    """One deserialized static-shape engine with a single input and a single output."""

    _DTYPES = None

    def __init__(self, plan: bytes, label: str) -> None:
        import tensorrt as trt

        if _TrtEngine._DTYPES is None:
            _TrtEngine._DTYPES = {
                trt.DataType.HALF: torch.float16,
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.UINT8: torch.uint8,
            }
        self.label = label
        self.runtime = trt.Runtime(_trt_logger())
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize the {label} engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRT could not create an execution context for {label}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"{label} engine must have 1 input and 1 output, has {names}")
        self.input_name, self.output_name = inputs[0], outputs[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_dtype = self._DTYPES[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = self._DTYPES[self.engine.get_tensor_dtype(self.output_name)]
        if any(d < 0 for d in self.input_shape + self.output_shape):
            raise RuntimeError(f"{label} engine must be static-shape, got {self.input_shape}")

    def run(self, input_ptr: int, output_ptr: int, stream_handle: int) -> None:
        self.context.set_tensor_address(self.input_name, input_ptr)
        self.context.set_tensor_address(self.output_name, output_ptr)
        if not self.context.execute_async_v3(stream_handle):
            raise RuntimeError(f"TensorRT execute_async_v3 failed for {self.label}")


class TaesdTrtVerificationError(RuntimeError):
    """The persisted engine does not reproduce its recorded probe output."""


class TaesdTrtBackend:
    """TensorRT FP16 TAESD decoder with TaesdVaeDecodeBackend's public interface.

    `.decode()` returns [0,1] fp16 NCHW (what `_raw_decode` returns);
    `.decode_bgr_u8()` returns the repo fast postprocess of that same fp16
    output (uint8 BGR NHWC, on the GPU) from one fused TensorRT kernel.
    Inputs of any batch size run as static `batch`-sized sub-batches; a partial
    tail is zero-padded (TAESD has no cross-sample ops, so padding cannot change
    the real rows; the gate script checks this bit for bit:
    docs/fps_comparisons/4070s_300fps_impl_20260928/taesd_trt/gate_taesd_trt.py).
    """

    name = "taesd_trt"
    compile_enabled = False
    compile_mode = "tensorrt"

    def __init__(
        self,
        decoder_engine: _TrtEngine,
        post_engine: Optional[_TrtEngine],
        device: torch.device,
        runtime_dtype: torch.dtype,
        batch: int,
        meta: dict,
        paths: dict,
        model=None,
    ) -> None:
        self.decoder_engine = decoder_engine
        self.post_engine = post_engine
        self.device = device
        self.runtime_dtype = runtime_dtype
        self.batch = int(batch)
        self.meta = meta
        self.paths = paths
        self.model = model
        expected_in = (self.batch, _LATENT_CHANNELS, _LATENT_HW, _LATENT_HW)
        expected_out = (self.batch, 3, _IMAGE_HW, _IMAGE_HW)
        if decoder_engine.input_shape != expected_in or decoder_engine.output_shape != expected_out:
            raise RuntimeError(
                f"TAESD TRT decoder shapes {decoder_engine.input_shape}->{decoder_engine.output_shape}, "
                f"expected {expected_in}->{expected_out}"
            )
        if decoder_engine.input_dtype != torch.float16 or decoder_engine.output_dtype != torch.float16:
            raise RuntimeError("TAESD TRT decoder engine must have fp16 I/O")
        if post_engine is not None and (
            post_engine.input_shape != expected_out
            or post_engine.output_shape != (self.batch, _IMAGE_HW, _IMAGE_HW, 3)
            or post_engine.output_dtype != torch.uint8
        ):
            raise RuntimeError("TAESD TRT post engine has unexpected I/O")
        self.fused_post_enabled = post_engine is not None and _env_bool(
            "MUSETALK_TAESD_TRT_FUSED_POST", True
        )
        self._lock = threading.Lock()
        self._last_stream = None
        self._last_event = torch.cuda.Event()
        self._decode_count = 0
        self._decode_total_s = 0.0
        self.timing_enabled = _env_bool("MUSETALK_TAESD_TIMING", False)
        self.timing_interval = 50

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        device: torch.device,
        runtime_dtype: torch.dtype = torch.float16,
        model=None,
    ) -> "TaesdTrtBackend":
        return load_taesd_trt_backend(device=device, runtime_dtype=runtime_dtype, model=model)

    # --------------------------------------------------------------- helpers
    def _prepare(self, latents: torch.Tensor) -> torch.Tensor:
        current = latents.to(device=self.device, dtype=torch.float16).contiguous()
        if current.dim() != 4 or tuple(current.shape[1:]) != (_LATENT_CHANNELS, _LATENT_HW, _LATENT_HW):
            raise ValueError(f"TAESD TRT expects [N,4,32,32] latents, got {tuple(current.shape)}")
        return current

    def _begin(self) -> torch.cuda.Stream:
        stream = torch.cuda.current_stream(self.device)
        # One execution context must never run on two streams at once: when the
        # caller switches streams, order the new stream after the last enqueue.
        if self._last_stream is not None and stream != self._last_stream:
            stream.wait_event(self._last_event)
        return stream

    def _end(self, stream: torch.cuda.Stream) -> None:
        self._last_event.record(stream)
        self._last_stream = stream

    def _chunks(self, latents: torch.Tensor):
        """Yield (start, take, fp16 [batch,4,32,32] input) per static sub-batch."""
        n = int(latents.shape[0])
        b = self.batch
        for start in range(0, n, b):
            take = min(b, n - start)
            if take == b:
                yield start, take, latents[start:start + b]
            else:
                padded = torch.zeros(
                    (b, _LATENT_CHANNELS, _LATENT_HW, _LATENT_HW), device=self.device, dtype=torch.float16
                )
                padded[:take].copy_(latents[start:])
                yield start, take, padded

    def _timed(self, started_at: float) -> None:
        if not self.timing_enabled:
            return
        torch.cuda.synchronize(self.device)
        self._decode_count += 1
        self._decode_total_s += time.perf_counter() - started_at
        if self._decode_count % self.timing_interval == 0:
            print(
                f"TAESD TRT decode timing calls={self._decode_count} "
                f"avg={self._decode_total_s / self._decode_count:.4f}s",
                flush=True,
            )

    # ---------------------------------------------------------------- decode
    def decode(
        self,
        latents: torch.Tensor,
        scaling_factor: float,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """[N,4,32,32] scaled latents -> [0,1] fp16 NCHW (TaesdVaeDecodeBackend contract)."""
        del scaling_factor  # TAESD consumes the scaled latent directly; see _raw_decode.
        started_at = time.perf_counter() if self.timing_enabled else 0.0
        z = self._prepare(latents)
        n = int(z.shape[0])
        out = torch.empty((n, 3, _IMAGE_HW, _IMAGE_HW), device=self.device, dtype=torch.float16)
        with self._lock:
            stream = self._begin()
            handle = stream.cuda_stream
            scratch = None
            for start, take, chunk in self._chunks(z):
                if take == self.batch:
                    self.decoder_engine.run(chunk.data_ptr(), out[start:start + take].data_ptr(), handle)
                else:
                    if scratch is None:
                        scratch = torch.empty(
                            (self.batch, 3, _IMAGE_HW, _IMAGE_HW), device=self.device, dtype=torch.float16
                        )
                    self.decoder_engine.run(chunk.data_ptr(), scratch.data_ptr(), handle)
                    out[start:start + take].copy_(scratch[:take])
            self._end(stream)
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(dtype=output_dtype)
        self._timed(started_at)
        return out

    def decode_bgr_u8(self, latents: torch.Tensor) -> torch.Tensor:
        """[N,4,32,32] -> uint8 BGR NHWC [N,256,256,3] on the GPU.

        Equals repo_fast_postprocess_gpu(self.decode(latents)) bit for bit: the
        same decoder engine writes fp16, then one fused TensorRT post kernel
        converts it.
        """
        if self.post_engine is None:
            return repo_fast_postprocess_gpu(self.decode(latents, 1.0))
        started_at = time.perf_counter() if self.timing_enabled else 0.0
        z = self._prepare(latents)
        n = int(z.shape[0])
        out = torch.empty((n, _IMAGE_HW, _IMAGE_HW, 3), device=self.device, dtype=torch.uint8)
        image = torch.empty((self.batch, 3, _IMAGE_HW, _IMAGE_HW), device=self.device, dtype=torch.float16)
        with self._lock:
            stream = self._begin()
            handle = stream.cuda_stream
            for start, take, chunk in self._chunks(z):
                self.decoder_engine.run(chunk.data_ptr(), image.data_ptr(), handle)
                if take == self.batch:
                    self.post_engine.run(image.data_ptr(), out[start:start + take].data_ptr(), handle)
                else:
                    tail = torch.empty(
                        (self.batch, _IMAGE_HW, _IMAGE_HW, 3), device=self.device, dtype=torch.uint8
                    )
                    self.post_engine.run(image.data_ptr(), tail.data_ptr(), handle)
                    out[start:start + take].copy_(tail[:take])
            self._end(stream)
        self._timed(started_at)
        return out

    # ---------------------------------------------------------------- warmup
    def warmup(self, batch_sizes: Optional[list[int]] = None) -> None:
        """Runs the already-built engines; nothing is compiled or built here."""
        sizes = batch_sizes or _env_int_list("MUSETALK_TAESD_WARMUP_BATCHES", "8")
        if not sizes:
            return
        started_at = time.time()
        with torch.inference_mode():
            for batch_size in sizes:
                probe = torch.zeros(
                    (batch_size, _LATENT_CHANNELS, _LATENT_HW, _LATENT_HW),
                    device=self.device,
                    dtype=torch.float16,
                )
                for _ in range(3):
                    self.decode(probe, 1.0)
                    if self.fused_post_enabled:
                        self.decode_bgr_u8(probe)
        torch.cuda.synchronize(self.device)
        logger.info(
            "TAESD TRT warmup complete (batches=%s, engine batch=%d, fused_post=%s, total=%.2fs)",
            sizes,
            self.batch,
            self.fused_post_enabled,
            time.time() - started_at,
        )

    # ----------------------------------------------------------------- probe
    def probe_hashes(self, return_image: bool = False):
        """Decode the fixed probe batch; return fp16 / fused-u8 hashes and a post check
        (and the fp16 image when return_image)."""
        probe = taesd_probe_latents(self.batch).to(self.device)
        with torch.inference_mode():
            image = self.decode(probe, 1.0)
            reference_u8 = repo_fast_postprocess_gpu(image)
            fused_u8 = self.decode_bgr_u8(probe) if self.post_engine is not None else reference_u8
            torch.cuda.synchronize(self.device)
        mismatched = int((fused_u8 != reference_u8).sum().item())
        hashes = {
            "probe_seed": _PROBE_SEED,
            "probe_shape": list(probe.shape),
            "fp16_sha256": _sha256_tensor(image),
            "u8_repo_post_sha256": _sha256_tensor(reference_u8),
            "u8_fused_sha256": _sha256_tensor(fused_u8),
            "fused_vs_repo_post_mismatched_bytes": mismatched,
        }
        return (hashes, image) if return_image else hashes


def _load_taesd_model(device: torch.device, runtime_dtype: torch.dtype):
    """The vendored TAESD in the same way TaesdVaeDecodeBackend.load resolves it."""
    return TaesdVaeDecodeBackend.load(device=device, runtime_dtype=runtime_dtype).model


def build_taesd_trt_engines(
    model,
    device: torch.device,
    batch: int = 8,
    engine_dir: Optional[Path] = None,
    opt_level: Optional[int] = None,
    strongly_typed: Optional[bool] = None,
    force: bool = False,
    onnx_bytes: Optional[bytes] = None,
    hw_compat: Optional[str] = None,
) -> dict:
    """Build (or reuse) the persisted decoder + post plans; return their meta record."""
    import fcntl

    engine_dir = Path(engine_dir or os.getenv("MUSETALK_TAESD_TRT_DIR", "").strip() or DEFAULT_TRT_DIR)
    opt_level = _env_int("MUSETALK_TAESD_TRT_OPT_LEVEL", 3) if opt_level is None else int(opt_level)
    if strongly_typed is None:
        strongly_typed = _env_bool("MUSETALK_TAESD_TRT_STRONGLY_TYPED", False)
    hw_compat = taesd_trt_hw_compat(hw_compat)
    _check_hw_compat_device(hw_compat, device)
    onnx_bytes = onnx_bytes or export_taesd_decoder_onnx(model, batch, device)
    fingerprint = taesd_trt_fingerprint(_sha256_bytes(onnx_bytes), batch, device, opt_level, strongly_typed,
                                        hw_compat)
    key = _fingerprint_key(fingerprint)
    paths = taesd_trt_paths(engine_dir, key)
    engine_dir.mkdir(parents=True, exist_ok=True)
    with open(paths["lock"], "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not force and paths["meta"].exists() and paths["decoder"].exists() and paths["post"].exists():
                return json.loads(paths["meta"].read_text())
            started_at = time.time()
            # a hardware-compatible build keeps its own timing cache (tactics timed under the same restriction)
            timing_cache_path = (paths["timing_cache"] if hw_compat == "none"
                                 else paths["timing_cache"].with_name(f"taesd_trt_timing.{hw_compat}.cache"))
            decoder_plan = build_taesd_decoder_plan(
                onnx_bytes, opt_level=opt_level, strongly_typed=strongly_typed,
                timing_cache_path=timing_cache_path, hw_compat=hw_compat,
            )
            decoder_s = time.time() - started_at
            post_started_at = time.time()
            post_plan = build_bgr_u8_post_plan(batch, hw_compat=hw_compat)
            post_s = time.time() - post_started_at
            decoder_engine = _TrtEngine(decoder_plan, "taesd decoder")
            post_engine = _TrtEngine(post_plan, "taesd bgr_u8 post")
            probe_backend = TaesdTrtBackend(
                decoder_engine, post_engine, device, torch.float16, batch, meta={}, paths=paths
            )
            probe, probe_image = probe_backend.probe_hashes(return_image=True)
            if probe["fused_vs_repo_post_mismatched_bytes"] != 0:
                raise TaesdTrtVerificationError(
                    f"fused post differs from the repo post on the probe batch: {probe}"
                )
            if hw_compat != "none":
                # reference for GPU models other than this one (see load_taesd_trt_backend)
                buffer = io.BytesIO()
                torch.save(probe_image.cpu(), buffer)
                _atomic_write(paths["probe_ref"], buffer.getvalue())
                probe["reference_file"] = paths["probe_ref"].name
                probe["cross_gpu_rel_l2_max"] = TAESD_CROSS_GPU_REL_L2_MAX
            meta = {
                "schema": TAESD_TRT_META_SCHEMA,
                "key": key,
                "fingerprint": fingerprint,
                "decoder_plan": paths["decoder"].name,
                "decoder_plan_sha256": _sha256_bytes(decoder_plan),
                "decoder_plan_bytes": len(decoder_plan),
                "post_plan": paths["post"].name,
                "post_plan_sha256": _sha256_bytes(post_plan),
                "post_plan_bytes": len(post_plan),
                "probe": probe,
                "build": {
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "decoder_build_s": round(decoder_s, 2),
                    "post_build_s": round(post_s, 2),
                    "torch": torch.__version__,
                    "pid": os.getpid(),
                    "timing_cache": timing_cache_path.name,
                    "gpu": torch.cuda.get_device_name(device),
                    "compute_capability": ".".join(str(v) for v in torch.cuda.get_device_capability(device)),
                    "hardware_compatibility_level": hw_compat,
                },
            }
            _atomic_write(paths["decoder"], decoder_plan)
            _atomic_write(paths["post"], post_plan)
            _atomic_write(paths["meta"], json.dumps(meta, indent=1).encode())
            logger.info(
                "Built TAESD TRT engines key=%s (decoder %.1fs, post %.1fs) in %s",
                key, decoder_s, post_s, engine_dir,
            )
            del probe_backend, decoder_engine, post_engine
            return meta
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_taesd_trt_backend(
    device: torch.device,
    runtime_dtype: torch.dtype = torch.float16,
    model=None,
) -> TaesdTrtBackend:
    """Load (building on first use when allowed) and verify the persisted TRT TAESD."""
    if device.type != "cuda":
        raise RuntimeError("TAESD TRT needs CUDA")
    if runtime_dtype != torch.float16:
        raise RuntimeError(
            f"TAESD TRT is an fp16 engine; the caller runs the VAE in {runtime_dtype}"
        )
    started_at = time.time()
    batch = _env_int("MUSETALK_TAESD_TRT_BATCH", 8)
    if batch <= 0:
        raise ValueError(f"MUSETALK_TAESD_TRT_BATCH must be positive, got {batch}")
    engine_dir = Path(os.getenv("MUSETALK_TAESD_TRT_DIR", "").strip() or DEFAULT_TRT_DIR)
    opt_level = _env_int("MUSETALK_TAESD_TRT_OPT_LEVEL", 3)
    strongly_typed = _env_bool("MUSETALK_TAESD_TRT_STRONGLY_TYPED", False)
    hw_compat = taesd_trt_hw_compat()
    _check_hw_compat_device(hw_compat, device)
    if model is None:
        model = _load_taesd_model(device, torch.float16)
    onnx_bytes = export_taesd_decoder_onnx(model, batch, device)
    fingerprint = taesd_trt_fingerprint(_sha256_bytes(onnx_bytes), batch, device, opt_level, strongly_typed,
                                        hw_compat)
    key = _fingerprint_key(fingerprint)
    paths = taesd_trt_paths(engine_dir, key)
    built_now = False
    if not (paths["meta"].exists() and paths["decoder"].exists() and paths["post"].exists()):
        if not _env_bool("MUSETALK_TAESD_TRT_BUILD", True):
            raise FileNotFoundError(
                f"TAESD TRT engine {paths['meta'].name} is missing in {engine_dir} and "
                "MUSETALK_TAESD_TRT_BUILD=0 (build it with: python scripts/vae_fast_decoder.py build)"
            )
        logger.warning("TAESD TRT engine key=%s missing; building it now (one-off, ~15-60 s)", key)
        build_taesd_trt_engines(
            model, device, batch=batch, engine_dir=engine_dir, opt_level=opt_level,
            strongly_typed=strongly_typed, onnx_bytes=onnx_bytes, hw_compat=hw_compat,
        )
        built_now = True
    meta = json.loads(paths["meta"].read_text())
    if meta.get("fingerprint") != fingerprint:
        raise TaesdTrtVerificationError(
            f"TAESD TRT meta fingerprint mismatch in {paths['meta']}: {meta.get('fingerprint')} != {fingerprint}"
        )
    decoder_plan = paths["decoder"].read_bytes()
    post_plan = paths["post"].read_bytes()
    for label, plan, recorded in (
        ("decoder", decoder_plan, meta.get("decoder_plan_sha256")),
        ("post", post_plan, meta.get("post_plan_sha256")),
    ):
        if _sha256_bytes(plan) != recorded:
            raise TaesdTrtVerificationError(f"TAESD TRT {label} plan hash does not match {paths['meta']}")
    backend = TaesdTrtBackend(
        _TrtEngine(decoder_plan, "taesd decoder"),
        _TrtEngine(post_plan, "taesd bgr_u8 post"),
        device,
        torch.float16,
        batch,
        meta=meta,
        paths=paths,
        model=model,
    )
    probe, probe_image = backend.probe_hashes(return_image=True)
    recorded = meta.get("probe") or {}
    mismatched = [f for f in ("fp16_sha256", "u8_fused_sha256") if probe[f] != recorded.get(f)]
    probe_status = "exact"
    if mismatched:
        # A hardware-compatible engine on a GPU model other than its build GPU may round differently: accept the
        # recorded relative-L2 bound on the fp16 probe image there. The build GPU itself stays bit-exact.
        device_name = torch.cuda.get_device_name(device)
        build_gpu = (meta.get("build") or {}).get("gpu")
        bound = recorded.get("cross_gpu_rel_l2_max")
        ref_path = paths["probe_ref"]
        if hw_compat == "none" or bound is None or not ref_path.exists() or device_name == build_gpu:
            field = mismatched[0]
            raise TaesdTrtVerificationError(
                f"TAESD TRT probe {field} mismatch (engine {key}): {probe[field]} != {recorded.get(field)}"
            )
        ref = torch.load(ref_path, map_location="cpu", weights_only=True).float()
        rel = float((probe_image.float().cpu() - ref).norm() / ref.norm())
        if rel > float(bound):
            raise TaesdTrtVerificationError(
                f"TAESD TRT probe on {device_name} (engine {key} built on {build_gpu}): rel_l2 {rel:.3g} > {bound}"
            )
        probe_status = f"cross_gpu:rel_l2={rel:.3g}"
        logger.warning("TAESD TRT probe on %s (engine built on %s): rel_l2 %.3g <= %s", device_name, build_gpu,
                       rel, bound)
    if probe["fused_vs_repo_post_mismatched_bytes"] != 0:
        raise TaesdTrtVerificationError(f"TAESD TRT fused post differs from the repo post: {probe}")
    gate = meta.get("gate") or {}
    logger.info(
        "Loaded TAESD TRT key=%s batch=%d built_now=%s fused_post=%s gate=%s in %.1fs",
        key, batch, built_now, backend.fused_post_enabled, gate.get("verdict", "not recorded"),
        time.time() - started_at,
    )
    print(
        f"TAESD TRT backend: key={key} batch={batch} built_now={built_now} "
        f"fused_post={backend.fused_post_enabled} probe={probe_status} hw_compat={hw_compat} "
        f"gate={gate.get('verdict', 'not recorded')}",
        flush=True,
    )
    return backend


def load_taesd_decoder(
    device: Optional[torch.device] = None,
    runtime_dtype: torch.dtype = torch.float16,
    force: bool = False,
):
    """Return a TAESD backend when requested, else None.

    Honours MUSETALK_TRT_FALLBACK: with fallback disabled a load failure raises
    so the server does not silently serve a slower decoder than the operator
    asked for.

    Added code: MUSETALK_TAESD_BACKEND=trt returns a TaesdTrtBackend. If it
    cannot be loaded or verified, MUSETALK_TAESD_TRT_STRICT=0 (default) falls
    back to the compiled TAESD backend (today's decoder) with a warning, and
    MUSETALK_TAESD_TRT_STRICT=1 raises. Unset/`compiled` is unchanged.
    """
    if not force and not taesd_backend_requested():
        return None

    resolved_device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    allow_fallback = _env_bool("MUSETALK_TRT_FALLBACK", True)
    try:
        if taesd_backend_kind() == "trt":
            compiled = TaesdVaeDecodeBackend.load(device=resolved_device, runtime_dtype=runtime_dtype)
            try:
                backend = load_taesd_trt_backend(
                    device=resolved_device, runtime_dtype=runtime_dtype, model=compiled.model
                )
            except Exception as exc:
                if _env_bool("MUSETALK_TAESD_TRT_STRICT", False):
                    raise
                logger.warning(
                    "TAESD TRT backend unavailable (%s: %s); using compiled TAESD",
                    type(exc).__name__,
                    exc,
                )
                print(f"⚠️  TAESD TRT unavailable ({type(exc).__name__}: {exc}); using compiled TAESD", flush=True)
                backend = compiled
            else:
                del compiled
        else:
            backend = TaesdVaeDecodeBackend.load(
                device=resolved_device, runtime_dtype=runtime_dtype
            )
        backend.warmup()
        return backend
    except Exception as exc:
        if allow_fallback:
            logger.warning(
                "Failed to activate TAESD VAE backend (%s: %s); falling back to PyTorch VAE",
                type(exc).__name__,
                exc,
            )
            return None
        raise


def _main(argv: Optional[list[str]] = None) -> int:
    """CLI: build or verify the persisted TRT TAESD engine (run under box_guard.sh)."""
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT TAESD engine tool")
    parser.add_argument("command", choices=["build", "verify", "flags"])
    parser.add_argument("--batch", type=int, default=None, help="engine batch (default MUSETALK_TAESD_TRT_BATCH or 8)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the engine exists")
    args = parser.parse_args(argv)
    if args.command == "flags":
        print(json.dumps(_TRT_FLAG_DOC, indent=1))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.batch is not None:
        os.environ["MUSETALK_TAESD_TRT_BATCH"] = str(args.batch)
    device = torch.device("cuda:0")
    model = _load_taesd_model(device, torch.float16)
    if args.command == "build":
        meta = build_taesd_trt_engines(
            model, device, batch=_env_int("MUSETALK_TAESD_TRT_BATCH", 8), force=args.force
        )
        print(json.dumps({k: meta[k] for k in ("key", "fingerprint", "decoder_plan_bytes", "post_plan_bytes",
                                               "probe", "build")}, indent=1))
    backend = load_taesd_trt_backend(device, torch.float16, model=model)
    print(json.dumps({"key": backend.meta.get("key"), "probe": backend.probe_hashes()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
