import logging
import os
import threading
import time
from contextlib import contextmanager


logger = logging.getLogger("gpu_memory_manager")

class GPUMemoryManager:
    """
    Manages GPU memory budget to allow multiple concurrent inferences.
    """
    
    def __init__(self, total_memory_gb=24, reserved_gb=4, max_live_generations=1):
        """
        Args:
            total_memory_gb: Total GPU VRAM (RTX 4090 = 24GB)
            reserved_gb: Reserved for models/overhead
        """
        self.total_memory = total_memory_gb * 1024**3  # Convert to bytes
        self.reserved_memory = reserved_gb * 1024**3
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
        
        # Track per-batch memory usage (empirical)
        self.memory_per_batch = {
            1: 2.0 * 1024**3,   # ~2GB for batch_size=1
            2: 3.5 * 1024**3,   # ~3.5GB for batch_size=2
            4: 6.0 * 1024**3,   # ~6GB for batch_size=4
            8: 10.0 * 1024**3,  # ~10GB for batch_size=8
        }
    
    @contextmanager
    def allocate(self, batch_size):
        """
        Context manager to allocate GPU memory for a request.
        Blocks if not enough memory available.
        """
        required = self.memory_per_batch.get(batch_size, batch_size * 2 * 1024**3)
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
                'available_gb': self.available_memory / 1024**3,
                'current_usage_gb': self.current_usage / 1024**3,
                'free_gb': (self.available_memory - self.current_usage) / 1024**3
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
