# MuseTalk Model-Level Optimization Plan

## Status

- **FP16**: ✅ Already active — no further gains here
- **torch.compile reduce-overhead**: ✅ Already active
- **Smaller model/GPU-path experiments**: ✅ Tested and exhausted enough to justify escalation
  - GPU-resident conditioning: tested, failed, reverted
  - vectorized audio prompts: tested, failed, reverted
  - GPU-resident latent cycles: tested, failed, reverted
  - explicit SDPA attention tuning: tested, failed, reverted
- **Target**: 8 concurrent HLS streams at 12fps (96 frames/s) on a single 24GB GPU
- **Current throughput**: ~96 frames/s at 100% GPU utilization (zero headroom)
- **Goal**: Reduce per-frame GPU time by 2–3× to create headroom

This document covers two implementation paths:
1. **TensorRT compilation** (highest impact, 2–3× expected speedup)
2. **ONNX Runtime with CUDA EP** (medium impact, 1.5–2× expected speedup)

Both are preceded by a mandatory benchmarking step to establish precise baselines.

This document is now the active next branch because the smaller in-architecture PyTorch-path refactors did not materially move the familiar `~2.0 avg / ~3.1 max` throttle band.

---

## Table of Contents

- [Phase 0: Baseline Benchmark](#phase-0-baseline-benchmark)
- [Phase 1: TensorRT Integration](#phase-1-tensorrt-integration)
  - [1A: Installation](#1a-installation)
  - [1B: Export Script](#1b-export-script)
  - [1C: Runtime Integration](#1c-runtime-integration)
  - [1D: Server Configuration](#1d-server-configuration)
  - [1E: Expected Results](#1e-expected-results)
- [Phase 2: ONNX Runtime Integration](#phase-2-onnx-runtime-integration)
  - [2A: Installation](#2a-installation)
  - [2B: Export Script](#2b-export-script)
  - [2C: Runtime Integration](#2c-runtime-integration)
  - [2D: Server Configuration](#2d-server-configuration)
  - [2E: Expected Results](#2e-expected-results)
- [Phase 3: Validation](#phase-3-validation)
- [Decision Matrix](#decision-matrix)
- [File Inventory](#file-inventory)

---

## Phase 0: Baseline Benchmark

**Run this first before any optimization.** It measures exact per-component timing
so you know where time is spent and can verify improvements.

### Script

```python
# filepath: /content/MuseTalk/scripts/benchmark_pipeline.py
"""
Measure exact per-component latency for each batch size.
This tells you WHERE time is spent so you know what to optimize.

Usage:
    python -m scripts.benchmark_pipeline
"""
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("benchmark")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def benchmark_pipeline():
    from musetalk.utils.utils import load_all_model

    audio_processor, vae, unet, pe = load_all_model("v15")
    unet = unet.eval().cuda()
    vae = vae.eval().cuda()

    # Report dtypes
    unet_dtype = next(unet.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype
    logger.info("UNet dtype: %s", unet_dtype)
    logger.info("VAE  dtype: %s", vae_dtype)
    logger.info(
        "UNet params: %s",
        f"{sum(p.numel() for p in unet.parameters()):,}",
    )
    logger.info(
        "VAE  params: %s",
        f"{sum(p.numel() for p in vae.parameters()):,}",
    )

    batch_sizes = [4, 8, 16, 24, 32, 40, 48]
    warmup = 20
    iters = 50
    results = []

    dtype = torch.float16 if unet_dtype == torch.float16 else torch.float32

    for bs in batch_sizes:
        # Inputs
        latent = torch.randn(bs, 8, 32, 32, device="cuda", dtype=dtype)
        whisper = torch.randn(bs, 1, 384, device="cuda", dtype=dtype)
        pe_input = torch.randn(bs, 1, 64, device="cuda", dtype=dtype)
        vae_input = torch.randn(bs, 4, 32, 32, device="cuda", dtype=dtype)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = unet(latent, whisper, pe_input)
                _ = vae.decode(1 / 0.18215 * vae_input).sample
        torch.cuda.synchronize()

        # Benchmark UNet
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                unet_out = unet(latent, whisper, pe_input)
        torch.cuda.synchronize()
        unet_ms = (time.perf_counter() - t0) / iters * 1000

        # Benchmark VAE decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                vae_out = vae.decode(1 / 0.18215 * vae_input).sample
        torch.cuda.synchronize()
        vae_ms = (time.perf_counter() - t0) / iters * 1000

        # Benchmark GPU→CPU transfer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = vae_out.cpu()
        torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - t0) / iters * 1000

        total_ms = unet_ms + vae_ms + transfer_ms
        fps = bs / (total_ms / 1000)

        result = {
            "batch_size": bs,
            "unet_ms": round(unet_ms, 2),
            "vae_ms": round(vae_ms, 2),
            "transfer_ms": round(transfer_ms, 2),
            "total_ms": round(total_ms, 2),
            "frames_per_sec": round(fps, 1),
            "unet_pct": round(unet_ms / total_ms * 100, 1),
            "vae_pct": round(vae_ms / total_ms * 100, 1),
        }
        results.append(result)

        logger.info(
            "BS=%2d | UNet: %7.2fms (%4.1f%%) | VAE: %7.2fms (%4.1f%%) | "
            "Transfer: %5.2fms | Total: %7.2fms | Throughput: %6.1f fps",
            bs,
            unet_ms,
            result["unet_pct"],
            vae_ms,
            result["vae_pct"],
            transfer_ms,
            total_ms,
            fps,
        )

    # Summary table
    logger.info("\n=== Summary ===")
    logger.info(
        "%-5s  %-10s  %-10s  %-10s  %-10s  %-12s",
        "BS", "UNet(ms)", "VAE(ms)", "Xfer(ms)", "Total(ms)", "Throughput",
    )
    logger.info("-" * 70)
    for r in results:
        logger.info(
            "%-5d  %-10.2f  %-10.2f  %-10.2f  %-10.2f  %-12.1f",
            r["batch_size"],
            r["unet_ms"],
            r["vae_ms"],
            r["transfer_ms"],
            r["total_ms"],
            r["frames_per_sec"],
        )

    # Feasibility check
    logger.info("\n=== Can we hit 96 fps (8 streams × 12fps)? ===")
    for r in results:
        status = "✅ YES" if r["frames_per_sec"] >= 96 else "❌ NO"
        headroom = r["frames_per_sec"] - 96
        logger.info(
            "BS=%2d: %6.1f fps → %s (headroom: %+.1f fps)",
            r["batch_size"],
            r["frames_per_sec"],
            status,
            headroom,
        )

    best = max(results, key=lambda r: r["frames_per_sec"])
    max_fps_per_stream = best["frames_per_sec"] / 8
    logger.info(
        "\nBest throughput: %.1f fps at BS=%d",
        best["frames_per_sec"],
        best["batch_size"],
    )
    logger.info(
        "Max sustainable fps/stream at 8 concurrent: %.1f fps",
        max_fps_per_stream,
    )
    logger.info(
        "Recommendation: --musetalk-fps %d for 8 concurrent streams",
        int(max_fps_per_stream),
    )


if __name__ == "__main__":
    benchmark_pipeline()
```

### How to run

```bash
cd /content/MuseTalk
python -m scripts.benchmark_pipeline
```

Save the output. You will compare against it after each optimization.

---

## Phase 1: TensorRT Integration

### 1A: Installation

```bash
# Check your CUDA version first
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Install torch-tensorrt matching your CUDA version
# For CUDA 12.1:
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu124

# Verify installation
python -c "import torch_tensorrt; print('torch_tensorrt version:', torch_tensorrt.__version__)"
```

### 1B: Export Script

```python
# filepath: /content/MuseTalk/scripts/tensorrt_export.py
"""
Export MuseTalk UNet and VAE to TensorRT engines with dynamic batch sizes.

Usage:
    python -m scripts.tensorrt_export
    python -m scripts.tensorrt_export --benchmark
    python -m scripts.tensorrt_export --batch-sizes 4,8,16,24,32,40,48 --benchmark
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("tensorrt_export")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Default model geometry ────────────────────────────────────────────────
LATENT_H = 32
LATENT_W = 32
UNET_LATENT_C = 8       # masked face (4) + reference (4)
VAE_LATENT_C = 4
WHISPER_DIM = 384
PE_DIM = 64


# ── UNet wrapper ──────────────────────────────────────────────────────────
class UNetWrapper(torch.nn.Module):
    """
    Thin wrapper that accepts a flat (latent, whisper, pe) signature so
    torch_tensorrt can trace it cleanly.  Adjust the forward() call if
    your musetalk UNet uses keyword arguments or a different order.
    """

    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        latent: torch.Tensor,
        whisper_features: torch.Tensor,
        positional_encoding: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(latent, whisper_features, positional_encoding)


# ── VAE decode wrapper ────────────────────────────────────────────────────
class VAEDecodeWrapper(torch.nn.Module):
    """
    Wraps the VAE so the scaling factor is baked into the graph.
    """

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = (1.0 / 0.18215) * latent
        return self.vae.decode(latent).sample


# ── Export helpers ─────────────────────────────────────────────────────────

def _compile_trt(
    module: torch.nn.Module,
    inputs: list,
    save_path: Path,
    workspace_gb: float = 1.0,
    min_block_size: int = 3,
) -> Path:
    """Compile *module* with Torch-TensorRT and persist."""
    import torch_tensorrt

    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Compiling %s …", save_path.stem)
    t0 = time.perf_counter()
    trt_module = torch_tensorrt.compile(
        module,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float16},
        truncate_long_and_double=True,
        workspace_size=int(workspace_gb * (1 << 30)),
        min_block_size=min_block_size,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Compilation finished in %.1fs", elapsed)

    torch_tensorrt.save(trt_module, str(save_path), output_format="torchscript")
    logger.info("Saved → %s (%.1f MB)", save_path, save_path.stat().st_size / 1e6)
    return save_path


def export_unet(
    unet: torch.nn.Module,
    output_dir: Path,
    batch_sizes: list[int],
) -> Path:
    """Export UNet to TensorRT with dynamic batch dimension."""
    import torch_tensorrt

    wrapper = UNetWrapper(unet).eval().cuda().half()
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
            min_shape=(min_bs, 1, WHISPER_DIM),
            opt_shape=(opt_bs, 1, WHISPER_DIM),
            max_shape=(max_bs, 1, WHISPER_DIM),
            dtype=torch.float16,
        ),
        torch_tensorrt.Input(
            min_shape=(min_bs, 1, PE_DIM),
            opt_shape=(opt_bs, 1, PE_DIM),
            max_shape=(max_bs, 1, PE_DIM),
            dtype=torch.float16,
        ),
    ]

    logger.info(
        "UNet TRT config: batch [%d, %d] opt=%d, latent=(%d,%d,%d), whisper=%d, pe=%d",
        min_bs, max_bs, opt_bs, UNET_LATENT_C, LATENT_H, LATENT_W, WHISPER_DIM, PE_DIM,
    )

    engine_path = output_dir / "unet_trt.ts"
    _compile_trt(wrapper, inputs, engine_path)

    # Persist metadata for the loader
    meta = {
        "type": "unet",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [UNET_LATENT_C, LATENT_H, LATENT_W],
        "whisper_dim": WHISPER_DIM,
        "pe_dim": PE_DIM,
        "dtype": "float16",
    }
    (output_dir / "unet_trt_meta.json").write_text(json.dumps(meta, indent=2))
    return engine_path


def export_vae(
    vae: torch.nn.Module,
    output_dir: Path,
    batch_sizes: list[int],
) -> Path:
    """Export VAE decoder to TensorRT with dynamic batch dimension."""
    import torch_tensorrt

    wrapper = VAEDecodeWrapper(vae).eval().cuda().half()
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

    logger.info(
        "VAE TRT config: batch [%d, %d] opt=%d",
        min_bs, max_bs, opt_bs,
    )

    engine_path = output_dir / "vae_decoder_trt.ts"
    _compile_trt(wrapper, inputs, engine_path)

    meta = {
        "type": "vae_decoder",
        "batch_range": [min_bs, max_bs],
        "opt_batch": opt_bs,
        "latent_shape": [VAE_LATENT_C, LATENT_H, LATENT_W],
        "dtype": "float16",
    }
    (output_dir / "vae_decoder_trt_meta.json").write_text(json.dumps(meta, indent=2))
    return engine_path


# ── Benchmark ─────────────────────────────────────────────────────────────

def benchmark_trt_engine(
    engine_path: Path,
    make_inputs,                    # callable(bs) → list[Tensor]
    batch_sizes: list[int],
    warmup: int = 20,
    iters: int = 100,
    label: str = "model",
) -> list[dict]:
    """Load a saved TRT engine and measure latency per batch size."""
    model = torch.jit.load(str(engine_path)).cuda().eval()
    results = []

    for bs in batch_sizes:
        inputs = make_inputs(bs)

        # warmup
        for _ in range(warmup):
            with torch.no_grad():
                model(*inputs)
        torch.cuda.synchronize()

        # timed loop
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                model(*inputs)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
        fps = bs / (ms / 1000)
        results.append({"batch_size": bs, "ms": round(ms, 2), "fps": round(fps, 1)})
        logger.info("%s  BS=%2d  %.2f ms  %.1f fps", label, bs, ms, fps)

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export MuseTalk models to TensorRT")
    parser.add_argument(
        "--batch-sizes",
        default="4,8,16,24,32,40,48",
        help="Comma-separated batch sizes for dynamic shape range",
    )
    parser.add_argument("--output-dir", default="./models/tensorrt")
    parser.add_argument("--version", default="v15", choices=["v1", "v15"])
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark after export")
    args = parser.parse_args()

    batch_sizes = sorted(int(x) for x in args.batch_sizes.split(","))
    output_dir = Path(args.output_dir)

    # ── Load models ───────────────────────────────────────────────────
    from musetalk.utils.utils import load_all_model

    audio_processor, vae, unet, pe = load_all_model(args.version)
    logger.info(
        "Models loaded  UNet dtype=%s  VAE dtype=%s",
        next(unet.parameters()).dtype,
        next(vae.parameters()).dtype,
    )

    # ── Export ────────────────────────────────────────────────────────
    unet_path = export_unet(unet, output_dir, batch_sizes)
    vae_path = export_vae(vae, output_dir, batch_sizes)

    # ── Optional benchmark ────────────────────────────────────────────
    if args.benchmark:
        logger.info("\n========== TensorRT Benchmark ==========")

        def make_unet_inputs(bs):
            return [
                torch.randn(bs, UNET_LATENT_C, LATENT_H, LATENT_W, device="cuda", dtype=torch.float16),
                torch.randn(bs, 1, WHISPER_DIM, device="cuda", dtype=torch.float16),
                torch.randn(bs, 1, PE_DIM, device="cuda", dtype=torch.float16),
            ]

        def make_vae_inputs(bs):
            return [
                torch.randn(bs, VAE_LATENT_C, LATENT_H, LATENT_W, device="cuda", dtype=torch.float16),
            ]

        unet_results = benchmark_trt_engine(unet_path, make_unet_inputs, batch_sizes, label="UNet-TRT")
        vae_results = benchmark_trt_engine(vae_path, make_vae_inputs, batch_sizes, label="VAE-TRT ")

        logger.info("\n=== Combined Throughput ===")
        logger.info("%-5s  %-12s  %-12s  %-12s  %-14s", "BS", "UNet(ms)", "VAE(ms)", "Total(ms)", "Throughput")
        logger.info("-" * 65)
        for u, v in zip(unet_results, vae_results):
            bs = u["batch_size"]
            total = u["ms"] + v["ms"]
            fps = bs / (total / 1000)
            logger.info("%-5d  %-12.2f  %-12.2f  %-12.2f  %-14.1f", bs, u["ms"], v["ms"], total, fps)

        best_fps = max(
            u["batch_size"] / ((u["ms"] + v["ms"]) / 1000)
            for u, v in zip(unet_results, vae_results)
        )
        logger.info("\nPeak throughput: %.1f fps", best_fps)
        logger.info("Max fps/stream at 8 concurrent: %.1f", best_fps / 8)


if __name__ == "__main__":
    main()
```

### 1C: Runtime Integration

This module provides a drop-in replacement inference function that the scheduler
calls instead of the normal PyTorch forward pass. It loads TRT engines at startup
and falls back to the original PyTorch models if engines are not found.

```python
# filepath: /content/MuseTalk/scripts/trt_runtime.py
"""
TensorRT runtime wrapper for MuseTalk UNet + VAE.

Provides load_trt_models() and TRTInferenceEngine that can be used
as a drop-in replacement for the PyTorch forward path in the HLS
GPU scheduler.

Environment variables:
    MUSETALK_TRT_DIR          Path to TRT engine directory (default: ./models/tensorrt)
    MUSETALK_TRT_ENABLED      Set to "1" to enable (default: "0")
    MUSETALK_TRT_FALLBACK     Set to "1" to fall back to PyTorch on load failure (default: "1")
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger("trt_runtime")

_TRT_DIR = Path(os.getenv("MUSETALK_TRT_DIR", "./models/tensorrt"))
_TRT_ENABLED = os.getenv("MUSETALK_TRT_ENABLED", "0") == "1"
_TRT_FALLBACK = os.getenv("MUSETALK_TRT_FALLBACK", "1") == "1"


class TRTInferenceEngine:
    """
    Wraps TRT-compiled UNet and VAE decoder for batched inference.

    Usage:
        engine = TRTInferenceEngine.load(trt_dir)
        if engine is not None:
            frames = engine.generate_batch(latent, whisper, pe)
    """

    def __init__(
        self,
        unet_trt: torch.jit.ScriptModule,
        vae_trt: torch.jit.ScriptModule,
        unet_meta: dict,
        vae_meta: dict,
    ):
        self.unet = unet_trt
        self.vae = vae_trt
        self.unet_meta = unet_meta
        self.vae_meta = vae_meta
        self.max_batch = unet_meta.get("batch_range", [1, 48])[1]

        # Pre-allocate pinned CPU staging buffer for GPU→CPU transfer
        # Will be lazily resized on first use
        self._pinned_buffer: Optional[torch.Tensor] = None

    @classmethod
    def load(cls, trt_dir: Optional[Path] = None) -> Optional["TRTInferenceEngine"]:
        """
        Load TRT engines from disk. Returns None on failure if fallback is enabled.
        """
        trt_dir = trt_dir or _TRT_DIR

        unet_path = trt_dir / "unet_trt.ts"
        vae_path = trt_dir / "vae_decoder_trt.ts"
        unet_meta_path = trt_dir / "unet_trt_meta.json"
        vae_meta_path = trt_dir / "vae_decoder_trt_meta.json"

        for p in [unet_path, vae_path, unet_meta_path, vae_meta_path]:
            if not p.exists():
                msg = f"TRT engine file not found: {p}"
                if _TRT_FALLBACK:
                    logger.warning("%s — falling back to PyTorch", msg)
                    return None
                raise FileNotFoundError(msg)

        try:
            logger.info("Loading TRT UNet from %s", unet_path)
            unet_trt = torch.jit.load(str(unet_path)).cuda().eval()

            logger.info("Loading TRT VAE from %s", vae_path)
            vae_trt = torch.jit.load(str(vae_path)).cuda().eval()

            unet_meta = json.loads(unet_meta_path.read_text())
            vae_meta = json.loads(vae_meta_path.read_text())

            engine = cls(unet_trt, vae_trt, unet_meta, vae_meta)

            # Warmup with a small batch to trigger TRT context allocation
            logger.info("Warming up TRT engines …")
            dummy_latent = torch.randn(4, 8, 32, 32, device="cuda", dtype=torch.float16)
            dummy_whisper = torch.randn(4, 1, 384, device="cuda", dtype=torch.float16)
            dummy_pe = torch.randn(4, 1, 64, device="cuda", dtype=torch.float16)
            with torch.no_grad():
                _ = engine.unet(dummy_latent, dummy_whisper, dummy_pe)
                dummy_vae_in = torch.randn(4, 4, 32, 32, device="cuda", dtype=torch.float16)
                _ = engine.vae(dummy_vae_in)
            torch.cuda.synchronize()
            logger.info("TRT engines loaded and warmed up successfully")

            return engine

        except Exception:
            if _TRT_FALLBACK:
                logger.exception("Failed to load TRT engines — falling back to PyTorch")
                return None
            raise

    @torch.no_grad()
    def generate_batch(
        self,
        latent: torch.Tensor,
        whisper_features: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run UNet + VAE decode on a batch.

        Args:
            latent:           (B, 8, 32, 32) float16 GPU tensor
            whisper_features: (B, 1, 384)    float16 GPU tensor
            pe:               (B, 1, 64)     float16 GPU tensor

        Returns:
            frames: (B, 3, 256, 256) float16 GPU tensor (decoded images)
        """
        # UNet forward
        noise_pred = self.unet(latent, whisper_features, pe)

        # VAE decode (scaling factor is baked into VAEDecodeWrapper)
        frames = self.vae(noise_pred)

        return frames

    def transfer_to_cpu(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """
        Non-blocking GPU→CPU transfer using pinned memory.
        """
        shape = gpu_tensor.shape
        if self._pinned_buffer is None or self._pinned_buffer.shape != shape:
            self._pinned_buffer = torch.empty(
                shape, dtype=gpu_tensor.dtype, device="cpu"
            ).pin_memory()
        self._pinned_buffer.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self._pinned_buffer


def load_trt_models() -> Optional[TRTInferenceEngine]:
    """
    Top-level loader called from ParallelAvatarManager or the scheduler.
    Respects the MUSETALK_TRT_ENABLED environment variable.
    """
    if not _TRT_ENABLED:
        logger.info("TensorRT disabled (MUSETALK_TRT_ENABLED != 1)")
        return None
    return TRTInferenceEngine.load()
```

### Integration point in the scheduler

The HLS GPU scheduler's `_run_generation_batch` method needs a small change
to use TRT when available. The pattern:

```python
# filepath: /content/MuseTalk/scripts/hls_gpu_scheduler.py (conceptual diff)
# In __init__:
from scripts.trt_runtime import load_trt_models

class HLSGPUStreamScheduler:
    def __init__(self, ...):
        # ...existing code...
        self.trt_engine = load_trt_models()
        if self.trt_engine:
            logger.info("TRT engines active — max batch %d", self.trt_engine.max_batch)

    def _run_generation_batch(self, selected) -> None:
        # ...existing batch assembly code that produces latent, whisper, pe tensors...

        if self.trt_engine is not None:
            # TRT path — single fused call
            frames_gpu = self.trt_engine.generate_batch(latent, whisper_features, pe)
            frames_cpu = self.trt_engine.transfer_to_cpu(frames_gpu)
        else:
            # Original PyTorch path
            # ...existing unet + vae code...
            pass

        # ...existing compose dispatch code...
```

### 1D: Server Configuration

```bash
# Add to your server start command after exporting TRT engines:

# Export engines (one-time, ~5-15 minutes)
python -m scripts.tensorrt_export --benchmark

# Enable TRT at runtime
export MUSETALK_TRT_ENABLED=1
export MUSETALK_TRT_DIR=./models/tensorrt
export MUSETALK_TRT_FALLBACK=1

# The rest of your existing config stays the same
export MUSETALK_COMPILE=0              # ← disable torch.compile, TRT replaces it
export MUSETALK_COMPILE_UNET=0
export MUSETALK_COMPILE_VAE=0

export HLS_SCHEDULER_MAX_BATCH=48
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,24,32,40,48
# ...rest of existing config...

python api_server.py --host 0.0.0.0 --port 8000
```

### 1E: Expected Results

Based on similar UNet architectures at this scale (SD-1.5 class, ~860M params):

| Batch Size | torch.compile FP16 (est.) | TensorRT FP16 (est.) | Speedup |
|------------|---------------------------|----------------------|---------|
| 8          | ~25ms                     | ~10ms                | 2.5×    |
| 16         | ~45ms                     | ~18ms                | 2.5×    |
| 32         | ~85ms                     | ~33ms                | 2.6×    |
| 48         | ~125ms                    | ~48ms                | 2.6×    |

At 2.5× speedup, your effective throughput becomes ~240 fps at BS=48.
8 streams × 12fps = 96 fps needed → **GPU utilization drops to ~40%**.
This means avg_segment_interval should drop well below 2.0s and the
max_segment_interval spike should disappear entirely.

---

## Phase 2: ONNX Runtime Integration

This is the fallback path if TensorRT compilation fails (e.g., unsupported
ops in the UNet) or if you want a simpler deployment with fewer dependencies.

### 2A: Installation

```bash
# Remove CPU-only onnxruntime if present
pip uninstall onnxruntime -y 2>/dev/null

# Install GPU-accelerated version
pip install onnxruntime-gpu==1.19.0

# Verify CUDA EP is available
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available providers:', providers)
assert 'CUDAExecutionProvider' in providers, 'CUDA EP not found!'
print('✅ CUDA EP available')
"
```

### 2B: Export Script

```python
# filepath: /content/MuseTalk/scripts/onnx_export.py
"""
Export MuseTalk UNet and VAE to ONNX, then run with ORT CUDA EP.

Usage:
    python -m scripts.onnx_export
    python -m scripts.onnx_export --benchmark
    python -m scripts.onnx_export --optimize   # applies ORT graph optimizations
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("onnx_export")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LATENT_H = 32
LATENT_W = 32


# ── Wrappers ──────────────────────────────────────────────────────────────

class UNetONNXWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, whisper_features, positional_encoding):
        return self.unet(latent, whisper_features, positional_encoding)


class VAEDecodeONNXWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        latent = (1.0 / 0.18215) * latent
        return self.vae.decode(latent).sample


# ── Export ─────────────────────────────────────────────────────────────────

def export_unet_onnx(unet: torch.nn.Module, save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = UNetONNXWrapper(unet).eval().cuda().half()

    dummy_latent = torch.randn(1, 8, LATENT_H, LATENT_W, device="cuda", dtype=torch.float16)
    dummy_whisper = torch.randn(1, 1, 384, device="cuda", dtype=torch.float16)
    dummy_pe = torch.randn(1, 1, 64, device="cuda", dtype=torch.float16)

    logger.info("Exporting UNet to ONNX → %s", save_path)
    torch.onnx.export(
        wrapper,
        (dummy_latent, dummy_whisper, dummy_pe),
        str(save_path),
        input_names=["latent", "whisper_features", "positional_encoding"],
        output_names=["noise_pred"],
        dynamic_axes={
            "latent": {0: "batch"},
            "whisper_features": {0: "batch"},
            "positional_encoding": {0: "batch"},
            "noise_pred": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = save_path.stat().st_size / 1e6
    logger.info("UNet ONNX saved: %.1f MB", size_mb)
    return save_path


def export_vae_onnx(vae: torch.nn.Module, save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = VAEDecodeONNXWrapper(vae).eval().cuda().half()

    dummy = torch.randn(1, 4, LATENT_H, LATENT_W, device="cuda", dtype=torch.float16)

    logger.info("Exporting VAE decoder to ONNX → %s", save_path)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(save_path),
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch"},
            "image": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = save_path.stat().st_size / 1e6
    logger.info("VAE ONNX saved: %.1f MB", size_mb)
    return save_path


def optimize_onnx(input_path: Path, output_path: Path) -> Path:
    """Apply ORT offline graph optimizations (constant folding, fusion)."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.optimized_model_filepath = str(output_path)

    # Creating the session with optimized_model_filepath writes the optimized model
    _ = ort.InferenceSession(
        str(input_path),
        sess_opts,
        providers=["CUDAExecutionProvider"],
    )
    logger.info("Optimized ONNX saved → %s", output_path)
    return output_path


# ── ORT Session Factory ───────────────────────────────────────────────────

def create_ort_session(onnx_path: str, enable_cuda_graph: bool = False):
    """Create an ORT inference session with CUDA EP."""
    import onnxruntime as ort

    cuda_options = {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    if enable_cuda_graph:
        cuda_options["enable_cuda_graph"] = True

    providers = [("CUDAExecutionProvider", cuda_options)]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_opts.intra_op_num_threads = 4

    session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
    active = session.get_providers()
    logger.info("ORT session: %s  providers=%s", onnx_path, active)
    if "CUDAExecutionProvider" not in active:
        logger.warning("⚠️  CUDA EP not active — inference will be slow!")
    return session


# ── Benchmark ─────────────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def benchmark_ort_sessions(
    unet_session,
    vae_session,
    batch_sizes: list[int],
    warmup: int = 20,
    iters: int = 100,
) -> list[dict]:
    """Benchmark ORT UNet + VAE across batch sizes."""
    import onnxruntime as ort

    results = []

    for bs in batch_sizes:
        # Build numpy inputs (ORT default path)
        unet_feeds = {
            "latent": np.random.randn(bs, 8, LATENT_H, LATENT_W).astype(np.float16),
            "whisper_features": np.random.randn(bs, 1, 384).astype(np.float16),
            "positional_encoding": np.random.randn(bs, 1, 64).astype(np.float16),
        }
        vae_feeds = {
            "latent": np.random.randn(bs, 4, LATENT_H, LATENT_W).astype(np.float16),
        }

        # Warmup
        for _ in range(warmup):
            unet_session.run(None, unet_feeds)
            vae_session.run(None, vae_feeds)

        # Benchmark UNet
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            unet_session.run(None, unet_feeds)
        torch.cuda.synchronize()
        unet_ms = (time.perf_counter() - t0) / iters * 1000

        # Benchmark VAE
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            vae_session.run(None, vae_feeds)
        torch.cuda.synchronize()
        vae_ms = (time.perf_counter() - t0) / iters * 1000

        total = unet_ms + vae_ms
        fps = bs / (total / 1000)
        results.append({
            "batch_size": bs,
            "unet_ms": round(unet_ms, 2),
            "vae_ms": round(vae_ms, 2),
            "total_ms": round(total, 2),
            "fps": round(fps, 1),
        })
        logger.info(
            "BS=%2d | UNet: %7.2fms | VAE: %7.2fms | Total: %7.2fms | %.1f fps",
            bs, unet_ms, vae_ms, total, fps,
        )

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export MuseTalk to ONNX + ORT")
    parser.add_argument("--output-dir", default="./models/onnx")
    parser.add_argument("--version", default="v15", choices=["v1", "v15"])
    parser.add_argument("--optimize", action="store_true", help="Apply ORT graph optimizations")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch-sizes", default="4,8,16,24,32,40,48")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    batch_sizes = sorted(int(x) for x in args.batch_sizes.split(","))

    from musetalk.utils.utils import load_all_model

    audio_processor, vae, unet, pe = load_all_model(args.version)
    logger.info(
        "Models loaded  UNet=%s  VAE=%s",
        next(unet.parameters()).dtype,
        next(vae.parameters()).dtype,
    )

    # Export
    unet_onnx = export_unet_onnx(unet, output_dir / "unet.onnx")
    vae_onnx = export_vae_onnx(vae, output_dir / "vae_decoder.onnx")

    # Optional optimization pass
    if args.optimize:
        unet_onnx = optimize_onnx(unet_onnx, output_dir / "unet_opt.onnx")
        vae_onnx = optimize_onnx(vae_onnx, output_dir / "vae_decoder_opt.onnx")

    # Benchmark
    if args.benchmark:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("pip install onnxruntime-gpu")
            return

        unet_sess = create_ort_session(str(unet_onnx))
        vae_sess = create_ort_session(str(vae_onnx))

        logger.info("\n========== ORT Benchmark ==========")
        results = benchmark_ort_sessions(unet_sess, vae_sess, batch_sizes)

        logger.info("\n=== Summary ===")
        logger.info("%-5s  %-10s  %-10s  %-10s  %-12s", "BS", "UNet", "VAE", "Total", "Throughput")
        logger.info("-" * 55)
        for r in results:
            logger.info(
                "%-5d  %-10.2f  %-10.2f  %-10.2f  %-12.1f",
                r["batch_size"], r["unet_ms"], r["vae_ms"], r["total_ms"], r["fps"],
            )

        best = max(results, key=lambda r: r["fps"])
        logger.info("\nPeak: %.1f fps at BS=%d", best["fps"], best["batch_size"])
        logger.info("Max fps/stream at 8 concurrent: %.1f", best["fps"] / 8)


if __name__ == "__main__":
    main()
```

### 2C: Runtime Integration

```python
# filepath: /content/MuseTalk/scripts/ort_runtime.py
"""
ONNX Runtime wrapper for MuseTalk UNet + VAE.

Environment variables:
    MUSETALK_ORT_DIR          Path to ONNX model directory (default: ./models/onnx)
    MUSETALK_ORT_ENABLED      Set to "1" to enable (default: "0")
    MUSETALK_ORT_FALLBACK     Set to "1" to fall back to PyTorch on failure (default: "1")
    MUSETALK_ORT_CUDA_GRAPH   Set to "1" to enable CUDA graph capture in ORT (default: "0")
"""
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("ort_runtime")

_ORT_DIR = Path(os.getenv("MUSETALK_ORT_DIR", "./models/onnx"))
_ORT_ENABLED = os.getenv("MUSETALK_ORT_ENABLED", "0") == "1"
_ORT_FALLBACK = os.getenv("MUSETALK_ORT_FALLBACK", "1") == "1"
_ORT_CUDA_GRAPH = os.getenv("MUSETALK_ORT_CUDA_GRAPH", "0") == "1"


class ORTInferenceEngine:
    """
    ONNX Runtime inference engine for UNet + VAE.

    Usage:
        engine = ORTInferenceEngine.load()
        if engine is not None:
            frames = engine.generate_batch(latent, whisper, pe)
    """

    def __init__(self, unet_session, vae_session):
        self.unet_session = unet_session
        self.vae_session = vae_session
        self._pinned_buffer: Optional[torch.Tensor] = None

    @classmethod
    def load(cls, ort_dir: Optional[Path] = None) -> Optional["ORTInferenceEngine"]:
        ort_dir = ort_dir or _ORT_DIR

        # Prefer optimized models if they exist
        unet_path = ort_dir / "unet_opt.onnx"
        if not unet_path.exists():
            unet_path = ort_dir / "unet.onnx"

        vae_path = ort_dir / "vae_decoder_opt.onnx"
        if not vae_path.exists():
            vae_path = ort_dir / "vae_decoder.onnx"

        for p in [unet_path, vae_path]:
            if not p.exists():
                msg = f"ONNX model not found: {p}"
                if _ORT_FALLBACK:
                    logger.warning("%s — falling back to PyTorch", msg)
                    return None
                raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort

            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            if _ORT_CUDA_GRAPH:
                cuda_options["enable_cuda_graph"] = True

            providers = [("CUDAExecutionProvider", cuda_options)]

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_opts.intra_op_num_threads = 4

            logger.info("Loading ORT UNet: %s", unet_path)
            unet_sess = ort.InferenceSession(str(unet_path), sess_opts, providers=providers)

            logger.info("Loading ORT VAE: %s", vae_path)
            vae_sess = ort.InferenceSession(str(vae_path), sess_opts, providers=providers)

            engine = cls(unet_sess, vae_sess)

            # Warmup
            logger.info("Warming up ORT sessions …")
            dummy = engine.generate_batch(
                torch.randn(4, 8, 32, 32, device="cuda", dtype=torch.float16),
                torch.randn(4, 1, 384, device="cuda", dtype=torch.float16),
                torch.randn(4, 1, 64, device="cuda", dtype=torch.float16),
            )
            logger.info("ORT engines loaded and warmed up (output shape %s)", dummy.shape)
            return engine

        except Exception:
            if _ORT_FALLBACK:
                logger.exception("Failed to load ORT engines — falling back to PyTorch")
                return None
            raise

    def generate_batch(
        self,
        latent: torch.Tensor,
        whisper_features: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run UNet + VAE via ORT.

        Accepts GPU tensors, returns GPU tensor (B, 3, 256, 256) float16.
        """
        # ORT with numpy path (simplest, works reliably)
        # For higher perf, use IOBinding to avoid GPU→CPU→GPU round-trip.
        unet_feeds = {
            "latent": latent.cpu().numpy(),
            "whisper_features": whisper_features.cpu().numpy(),
            "positional_encoding": pe.cpu().numpy(),
        }
        unet_out = self.unet_session.run(None, unet_feeds)[0]

        vae_feeds = {"latent": unet_out}
        vae_out = self.vae_session.run(None, vae_feeds)[0]

        # Convert back to GPU tensor
        return torch.from_numpy(vae_out).cuda()

    def generate_batch_iobinding(
        self,
        latent: torch.Tensor,
        whisper_features: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero-copy GPU inference using ORT IOBinding.
        Avoids the GPU→CPU→GPU round-trip of the numpy path.

        NOTE: Requires onnxruntime-gpu with CUDA EP properly configured.
        Falls back to numpy path on failure.
        """
        try:
            import onnxruntime as ort

            bs = latent.shape[0]

            # ── UNet IOBinding ────────────────────────────────────────
            unet_binding = self.unet_session.io_binding()

            def _bind_input(binding, name, tensor):
                binding.bind_input(
                    name=name,
                    device_type="cuda",
                    device_id=0,
                    element_type=np.float16,
                    shape=tuple(tensor.shape),
                    buffer_ptr=tensor.data_ptr(),
                )

            _bind_input(unet_binding, "latent", latent)
            _bind_input(unet_binding, "whisper_features", whisper_features)
            _bind_input(unet_binding, "positional_encoding", pe)

            # Allocate output on GPU
            unet_out_shape = (bs, 4, 32, 32)
            unet_output = torch.empty(unet_out_shape, dtype=torch.float16, device="cuda")
            unet_binding.bind_output(
                name="noise_pred",
                device_type="cuda",
                device_id=0,
                element_type=np.float16,
                shape=unet_out_shape,
                buffer_ptr=unet_output.data_ptr(),
            )

            self.unet_session.run_with_iobinding(unet_binding)

            # ── VAE IOBinding ─────────────────────────────────────────
            vae_binding = self.vae_session.io_binding()
            _bind_input(vae_binding, "latent", unet_output)

            vae_out_shape = (bs, 3, 256, 256)
            vae_output = torch.empty(vae_out_shape, dtype=torch.float16, device="cuda")
            vae_binding.bind_output(
                name="image",
                device_type="cuda",
                device_id=0,
                element_type=np.float16,
                shape=vae_out_shape,
                buffer_ptr=vae_output.data_ptr(),
            )

            self.vae_session.run_with_iobinding(vae_binding)

            return vae_output

        except Exception:
            logger.warning("IOBinding failed, falling back to numpy path", exc_info=True)
            return self.generate_batch(latent, whisper_features, pe)

    def transfer_to_cpu(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        shape = gpu_tensor.shape
        if self._pinned_buffer is None or self._pinned_buffer.shape != shape:
            self._pinned_buffer = torch.empty(
                shape, dtype=gpu_tensor.dtype, device="cpu",
            ).pin_memory()
        self._pinned_buffer.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self._pinned_buffer


def load_ort_models() -> Optional[ORTInferenceEngine]:
    """
    Top-level loader. Respects MUSETALK_ORT_ENABLED env var.
    """
    if not _ORT_ENABLED:
        logger.info("ORT disabled (MUSETALK_ORT_ENABLED != 1)")
        return None
    return ORTInferenceEngine.load()
```

### Integration point in the scheduler

Same pattern as TRT but with ORT:

```python
# filepath: /content/MuseTalk/scripts/hls_gpu_scheduler.py (conceptual diff)
from scripts.trt_runtime import load_trt_models
from scripts.ort_runtime import load_ort_models

class HLSGPUStreamScheduler:
    def __init__(self, ...):
        # ...existing code...
        # Try TRT first, then ORT, then fall back to PyTorch
        self.trt_engine = load_trt_models()
        self.ort_engine = None
        if self.trt_engine is None:
            self.ort_engine = load_ort_models()

        if self.trt_engine:
            logger.info("Using TensorRT inference backend")
        elif self.ort_engine:
            logger.info("Using ONNX Runtime inference backend")
        else:
            logger.info("Using PyTorch inference backend")

    def _run_generation_batch(self, selected) -> None:
        # ...existing batch assembly that produces latent, whisper, pe...

        if self.trt_engine is not None:
            frames_gpu = self.trt_engine.generate_batch(latent, whisper_features, pe)
            frames_cpu = self.trt_engine.transfer_to_cpu(frames_gpu)
        elif self.ort_engine is not None:
            frames_gpu = self.ort_engine.generate_batch_iobinding(latent, whisper_features, pe)
            frames_cpu = self.ort_engine.transfer_to_cpu(frames_gpu)
        else:
            # ...existing PyTorch unet + vae code...
            pass

        # ...existing compose dispatch...
```

### 2D: Server Configuration

```bash
# Export ONNX models (one-time, ~1-2 minutes)
python -m scripts.onnx_export --optimize --benchmark

# Enable ORT at runtime
export MUSETALK_ORT_ENABLED=1
export MUSETALK_ORT_DIR=./models/onnx
export MUSETALK_ORT_FALLBACK=1
export MUSETALK_ORT_CUDA_GRAPH=0          # try "1" if shapes are fixed

# Disable torch.compile (ORT replaces it)
export MUSETALK_COMPILE=0
export MUSETALK_COMPILE_UNET=0
export MUSETALK_COMPILE_VAE=0

# Rest of config unchanged
export HLS_SCHEDULER_MAX_BATCH=48
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,24,32,40,48
# ...etc...

python api_server.py --host 0.0.0.0 --port 8000
```

### 2E: Expected Results

| Batch Size | torch.compile FP16 (est.) | ORT CUDA EP FP16 (est.) | Speedup |
|------------|---------------------------|-------------------------|---------|
| 8          | ~25ms                     | ~16ms                   | 1.6×    |
| 16         | ~45ms                     | ~28ms                   | 1.6×    |
| 32         | ~85ms                     | ~50ms                   | 1.7×    |
| 48         | ~125ms                    | ~72ms                   | 1.7×    |

At 1.7× speedup, peak throughput becomes ~163 fps at BS=48.
8 × 12fps = 96 needed → **GPU drops to ~59% utilization**.
Smaller gains than TRT but more portable and simpler to debug.

---

## Phase 3: Validation

After implementing either backend, re-run the baseline benchmark and the
load test to measure actual improvement.

### Step 1: Re-run pipeline benchmark

```bash
# Compare against Phase 0 baseline
python -m scripts.benchmark_pipeline
```

### Step 2: Load test with full config

```bash
# TensorRT backend
export MUSETALK_TRT_ENABLED=1
export MUSETALK_COMPILE=0
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --batch-size 4 \
  --playback-fps 24 \
  --musetalk-fps 12 \
  --hold-seconds 120

# OR ONNX Runtime backend
export MUSETALK_ORT_ENABLED=1
export MUSETALK_COMPILE=0
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --batch-size 4 \
  --playback-fps 24 \
  --musetalk-fps 12 \
  --hold-seconds 120
```

### Step 3: Verify expected improvements

| Metric                   | Current (PyTorch) | TRT Target  | ORT Target  |
|--------------------------|-------------------|-------------|-------------|
| avg_segment_interval     | 2.065s            | <1.5s       | <1.7s       |
| max_segment_interval     | 3.148s            | <2.2s       | <2.5s       |
| GPU utilization          | 100%              | <50%        | <65%        |
| Wall time (8 streams)    | 39.5s             | <25s        | <30s        |

---

## Decision Matrix

| Criteria               | TensorRT               | ONNX Runtime             |
|------------------------|------------------------|--------------------------|
| Expected speedup       | 2–3×                   | 1.5–2×                   |
| Installation effort    | Medium (torch-tensorrt)| Low (pip install)        |
| Export time            | 5–15 minutes           | 1–2 minutes              |
| Dynamic batch support  | Yes (min/opt/max range)| Yes (dynamic axes)       |
| Debugging difficulty   | Hard (opaque engine)   | Medium (ONNX graph)      |
| Risk of op unsupported | Medium                 | Low                      |
| Production maturity    | High (NVIDIA supported)| High (Microsoft backed)  |
| Fallback to PyTorch    | Built-in               | Built-in                 |

**Recommendation**: Try TensorRT first. If export fails due to unsupported ops,
fall back to ONNX Runtime. Both paths have automatic PyTorch fallback so the
server never fails to start.

---

## File Inventory

After implementing both paths, the new files are:

```
scripts/
├── benchmark_pipeline.py    # Phase 0: baseline measurement
├── tensorrt_export.py       # Phase 1B: TRT export + benchmark
├── trt_runtime.py           # Phase 1C: TRT runtime wrapper
├── onnx_export.py           # Phase 2B: ONNX export + benchmark
├── ort_runtime.py           # Phase 2C: ORT runtime wrapper
└── hls_gpu_scheduler.py     # Modified: backend selection in _run_generation_batch

models/
├── tensorrt/
│   ├── unet_trt.ts
│   ├── unet_trt_meta.json
│   ├── vae_decoder_trt.ts
│   └── vae_decoder_trt_meta.json
└── onnx/
    ├── unet.onnx
    ├── unet_opt.onnx
    ├── vae_decoder.onnx
    └── vae_decoder_opt.onnx
```

## Updated start_params.md

After validation, update the reference config:

```bash
# TensorRT backend (preferred)
export MUSETALK_TRT_ENABLED=1
export MUSETALK_TRT_DIR=./models/tensorrt
export MUSETALK_TRT_FALLBACK=1
export MUSETALK_COMPILE=0
export MUSETALK_COMPILE_UNET=0
export MUSETALK_COMPILE_VAE=0

# OR ONNX Runtime backend (fallback)
# export MUSETALK_ORT_ENABLED=1
# export MUSETALK_ORT_DIR=./models/onnx
# export MUSETALK_ORT_FALLBACK=1

# Everything else unchanged
export HLS_SCHEDULER_MAX_BATCH=48
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,24,32,40,48
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5
export HLS_STARTUP_CHUNK_COUNT=1
export HLS_PREP_WORKERS=8
export HLS_COMPOSE_WORKERS=8
export HLS_ENCODE_WORKERS=8
export HLS_MAX_PENDING_JOBS=24
export HLS_CHUNK_VIDEO_ENCODER=h264_nvenc
export HLS_PERSISTENT_SEGMENTER=0

python api_server.py --host 0.0.0.0 --port 8000
