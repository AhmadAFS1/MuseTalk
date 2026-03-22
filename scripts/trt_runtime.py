import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch


logger = logging.getLogger("trt_runtime")


def _ensure_torch_tensorrt_registered():
    """
    Added code: import torch_tensorrt before loading serialized TRT-backed
    TorchScript modules so the custom `torch.classes.tensorrt.Engine` type is
    registered in the current process.
    """
    try:
        import torch_tensorrt  # noqa: F401
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


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _requested_vae_backend() -> str:
    return os.getenv("MUSETALK_VAE_BACKEND", "").strip().lower()


def _trt_vae_requested() -> bool:
    requested_backend = _requested_vae_backend()
    if requested_backend in {"trt", "tensorrt"}:
        return True
    return _env_enabled("MUSETALK_TRT_ENABLED", "0") or _env_enabled(
        "MUSETALK_TRT_VAE_ENABLED", "0"
    )


def _allow_fallback() -> bool:
    return _env_enabled("MUSETALK_TRT_FALLBACK", "1")


def _trt_dir() -> Path:
    return Path(os.getenv("MUSETALK_TRT_DIR", "./models/tensorrt"))


def _trt_vae_path() -> Path:
    return Path(os.getenv("MUSETALK_TRT_VAE_PATH", str(_trt_dir() / "vae_decoder_trt.ts")))


def _trt_vae_meta_path(engine_path: Path) -> Path:
    default_meta = engine_path.with_name("vae_decoder_trt_meta.json")
    return Path(os.getenv("MUSETALK_TRT_VAE_META_PATH", str(default_meta)))


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
        _ensure_torch_tensorrt_registered()
        module = torch.jit.load(str(engine_path), map_location=device).eval()

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


def load_vae_trt_decoder(
    device: Optional[torch.device] = None,
    scaling_factor: Optional[float] = None,
):
    """
    Load the TensorRT VAE decoder backend when explicitly requested.

    Returns `None` when the backend is disabled or unavailable, so the caller
    can continue using the standard PyTorch VAE path.
    """
    del scaling_factor  # reserved for future metadata validation

    if not _trt_vae_requested():
        return None

    resolved_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if resolved_device.type != "cuda":
        logger.warning("TensorRT VAE requested, but CUDA is unavailable; using PyTorch VAE")
        return None

    engine_path = _trt_vae_path()
    meta_path = _trt_vae_meta_path(engine_path)

    if not engine_path.exists():
        message = f"TensorRT VAE engine not found: {engine_path}"
        if _allow_fallback():
            logger.warning("%s; falling back to PyTorch VAE", message)
            return None
        raise FileNotFoundError(message)

    try:
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
