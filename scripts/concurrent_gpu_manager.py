import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager


logger = logging.getLogger("gpu_memory_manager")


def _env_float(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s=%r", name, value)
    return default


def _bytes_from_gb(value_gb: float) -> int:
    return int(float(value_gb) * 1024**3)


def detect_total_gpu_memory_gb(gpu_id: int = 0, fallback_gb: float = 24.0) -> tuple[float, str]:
    """Return total GPU VRAM in GB plus the source used for the estimate."""
    explicit = _env_float("GPU_TOTAL_MEMORY_GB", "MUSETALK_GPU_TOTAL_MEMORY_GB")
    if explicit and explicit > 0:
        return explicit, os.getenv("GPU_MEMORY_DETECTION_SOURCE", "env")

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(int(gpu_id))
            return props.total_memory / 1024**3, "torch"
    except Exception as exc:
        logger.debug("torch GPU memory detection failed: %s", exc)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            first_value = result.stdout.strip().splitlines()[0].strip()
            memory_mb = float(first_value)
            if memory_mb > 0:
                return memory_mb / 1024.0, "nvidia-smi"
    except Exception as exc:
        logger.debug("nvidia-smi GPU memory detection failed: %s", exc)

    return float(fallback_gb), "fallback"


def default_reserved_memory_gb(total_memory_gb: float) -> float:
    """Reserve model/runtime headroom using the old 24GB behavior as the anchor."""
    explicit = _env_float("GPU_RESERVED_MEMORY_GB", "MUSETALK_GPU_RESERVED_MEMORY_GB")
    if explicit is not None:
        return max(0.0, explicit)

    total = max(1.0, float(total_memory_gb))
    # Old production behavior was 6GB reserved on 24GB cards. Keep that ratio,
    # but clamp it so smaller cards stay usable and larger cards keep headroom.
    return max(4.0, min(10.0, round(total * 0.25, 1)))


def _parse_batch_memory_gb(raw: str) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        batch_raw, gb_raw = token.split(":", 1)
        try:
            batch_size = int(batch_raw.strip())
            gb = float(gb_raw.strip())
        except ValueError:
            logger.warning("Ignoring invalid GPU_MEMORY_BATCH_GB token %r", token)
            continue
        if batch_size > 0 and gb > 0:
            parsed[batch_size] = gb
    return parsed


def default_memory_per_batch_gb() -> dict[int, float]:
    estimates = {
        1: 2.0,
        2: 3.5,
        4: 6.0,
        8: 10.0,
        16: 14.0,
        32: 22.0,
        48: 28.0,
        64: 34.0,
    }
    override = os.getenv("GPU_MEMORY_BATCH_GB", "").strip()
    if override:
        estimates.update(_parse_batch_memory_gb(override))
    return estimates


def recommended_scheduler_batch_config(total_memory_gb: float, profile: str = "throughput_record") -> dict:
    """
    Choose batch buckets from the actual VRAM class.

    The 24GB path preserves the known RTX 3090-compatible 8/16 TRT warmup.
    Larger cards include batch 4 for low-load WebRTC latency and add wider
    buckets so the scheduler can spend the extra VRAM on aggregate throughput.
    """
    profile = (profile or "throughput_record").strip().lower()
    total = float(total_memory_gb or 24.0)

    if profile == "baseline":
        max_batch = 4
        buckets = [4]
    elif total < 20:
        max_batch = 8
        buckets = [4, 8]
    elif total < 30:
        max_batch = 16
        buckets = [8, 16]
    elif total < 45:
        max_batch = 32
        buckets = [4, 8, 16, 32]
    else:
        max_batch = 48
        buckets = [4, 8, 16, 32, 48]

    return {
        "max_combined_batch_size": max_batch,
        "fixed_batch_sizes": buckets,
        "warmup_batches": buckets,
        "startup_slice_size": min(4, max_batch),
    }


class GPUMemoryManager:
    """
    Manages GPU memory budget to allow multiple concurrent inferences.
    """
    
    def __init__(
        self,
        total_memory_gb=None,
        reserved_gb=None,
        max_live_generations=1,
        gpu_id=0,
    ):
        """
        Args:
            total_memory_gb: Total GPU VRAM. None means auto-detect.
            reserved_gb: Reserved for models/overhead. None means auto-size.
        """
        if total_memory_gb is None:
            total_memory_gb, memory_source = detect_total_gpu_memory_gb(gpu_id=gpu_id)
        else:
            total_memory_gb = float(total_memory_gb)
            memory_source = "constructor"
        if reserved_gb is None:
            reserved_gb = default_reserved_memory_gb(total_memory_gb)
        else:
            reserved_gb = float(reserved_gb)

        self.gpu_id = int(gpu_id)
        self.total_memory_gb = float(total_memory_gb)
        self.reserved_memory_gb = max(0.0, float(reserved_gb))
        self.memory_source = memory_source
        self.total_memory = _bytes_from_gb(self.total_memory_gb)
        self.reserved_memory = min(
            _bytes_from_gb(self.reserved_memory_gb),
            max(0, self.total_memory - _bytes_from_gb(1.0)),
        )
        self.available_memory = self.total_memory - self.reserved_memory
        
        self.current_usage = 0
        self.lock = threading.Lock()
        self.memory_condition = threading.Condition(self.lock)

        self.max_live_generations = max(1, int(max_live_generations))
        self.compute_semaphore = threading.Semaphore(self.max_live_generations)
        self.compute_lock = threading.Lock()
        self.compute_slots_in_use = 0
        self.compute_holders = {}
        self.log_allocation_events = os.getenv(
            "GPU_MEMORY_LOG_ALLOCATIONS", ""
        ).lower() in {"1", "true", "yes", "on"}
        self.wait_log_threshold_s = float(
            os.getenv("GPU_MEMORY_WAIT_LOG_THRESHOLD_SECONDS", "1.0")
        )
        
        # Track per-batch memory usage (empirical lease estimates).
        self.memory_per_batch_gb = default_memory_per_batch_gb()
        self.memory_per_batch = {
            batch_size: _bytes_from_gb(gb)
            for batch_size, gb in self.memory_per_batch_gb.items()
        }

        logger.info(
            "GPU memory budget initialized: total=%.1fGB reserved=%.1fGB "
            "available=%.1fGB source=%s max_live_generations=%s",
            self.total_memory / 1024**3,
            self.reserved_memory / 1024**3,
            self.available_memory / 1024**3,
            self.memory_source,
            self.max_live_generations,
        )
    
    @contextmanager
    def allocate(self, batch_size):
        """
        Context manager to allocate GPU memory for a request.
        Blocks if not enough memory available.
        """
        required = self.memory_per_batch.get(batch_size, _bytes_from_gb(batch_size * 2.0))
        wait_start = time.monotonic()
        wait_logged = False
        
        with self.memory_condition:
            while self.current_usage + required > self.available_memory:
                if (
                    not wait_logged
                    and time.monotonic() - wait_start >= self.wait_log_threshold_s
                ):
                    wait_logged = True
                    logger.info(
                        "Waiting for GPU memory lease: need %.1fGB, in_use %.1fGB, free %.1fGB",
                        required / 1024**3,
                        self.current_usage / 1024**3,
                        (self.available_memory - self.current_usage) / 1024**3,
                    )
                self.memory_condition.wait(timeout=0.1)
            self.current_usage += required
            if self.log_allocation_events:
                logger.info(
                    "Allocated %.1fGB lease (total: %.1fGB)",
                    required / 1024**3,
                    self.current_usage / 1024**3,
                )
            elif wait_logged:
                logger.info(
                    "Acquired GPU memory lease after %.2fs: %.1fGB (total: %.1fGB)",
                    time.monotonic() - wait_start,
                    required / 1024**3,
                    self.current_usage / 1024**3,
                )
        
        try:
            yield
        finally:
            # Release memory
            with self.memory_condition:
                self.current_usage = max(0, self.current_usage - required)
                if self.log_allocation_events:
                    logger.info(
                        "Released %.1fGB lease (total: %.1fGB)",
                        required / 1024**3,
                        self.current_usage / 1024**3,
                    )
                self.memory_condition.notify_all()

    def try_acquire_compute_slot(self, request_id, stream_type="live_stream"):
        """Acquire a live-generation slot without blocking. Returns False if full."""
        acquired = self.compute_semaphore.acquire(blocking=False)
        if not acquired:
            return False

        with self.compute_lock:
            self.compute_slots_in_use += 1
            self.compute_holders[request_id] = {
                "stream_type": stream_type,
                "acquired_at": time.time(),
            }

        print(
            f"🎛️  Acquired compute slot for {request_id} "
            f"({self.compute_slots_in_use}/{self.max_live_generations} in use)"
        )
        return True

    def release_compute_slot(self, request_id):
        """Release a live-generation slot held by a request."""
        with self.compute_lock:
            holder = self.compute_holders.pop(request_id, None)
            if holder is None:
                return False
            self.compute_slots_in_use = max(0, self.compute_slots_in_use - 1)

        self.compute_semaphore.release()
        print(
            f"🎛️  Released compute slot for {request_id} "
            f"({self.compute_slots_in_use}/{self.max_live_generations} in use)"
        )
        return True
    
    def get_stats(self):
        """Get current memory statistics"""
        with self.lock:
            memory_stats = {
                'total_gb': self.total_memory / 1024**3,
                'reserved_gb': self.reserved_memory / 1024**3,
                'available_gb': self.available_memory / 1024**3,
                'current_usage_gb': self.current_usage / 1024**3,
                'free_gb': (self.available_memory - self.current_usage) / 1024**3,
                'source': self.memory_source,
                'batch_memory_gb': dict(sorted(self.memory_per_batch_gb.items())),
                'recommended_scheduler': self.recommended_scheduler_config(
                    os.getenv("PROFILE", "throughput_record")
                ),
            }
        with self.compute_lock:
            compute_stats = {
                'max_live_generations': self.max_live_generations,
                'slots_in_use': self.compute_slots_in_use,
                'slots_free': max(0, self.max_live_generations - self.compute_slots_in_use),
                'holders': dict(self.compute_holders),
            }

        memory_stats['compute'] = compute_stats
        return memory_stats

    def recommended_scheduler_config(self, profile: str = "throughput_record") -> dict:
        return recommended_scheduler_batch_config(self.total_memory_gb, profile=profile)
