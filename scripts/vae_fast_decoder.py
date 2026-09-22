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
"""

from __future__ import annotations

import logging
import os
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


def load_taesd_decoder(
    device: Optional[torch.device] = None,
    runtime_dtype: torch.dtype = torch.float16,
    force: bool = False,
) -> Optional[TaesdVaeDecodeBackend]:
    """Return a TAESD backend when requested, else None.

    Honours MUSETALK_TRT_FALLBACK: with fallback disabled a load failure raises
    so the server does not silently serve a slower decoder than the operator
    asked for.
    """
    if not force and not taesd_backend_requested():
        return None

    resolved_device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    allow_fallback = _env_bool("MUSETALK_TRT_FALLBACK", True)
    try:
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
