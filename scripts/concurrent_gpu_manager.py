import torch
import threading
from contextlib import contextmanager

class GPUMemoryManager:
    """
    Manages GPU memory budget to allow multiple concurrent inferences.
    """
    
    def __init__(self, total_memory_gb=24, reserved_gb=4):
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
        
        # Wait for memory to be available
        while True:
            with self.lock:
                if self.current_usage + required <= self.available_memory:
                    self.current_usage += required
                    print(f"🔒 Allocated {required/1024**3:.1f}GB (total: {self.current_usage/1024**3:.1f}GB)")
                    break
            # Sleep briefly if not enough memory
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            import time
            time.sleep(0.1)
        
        try:
            yield
        finally:
            # Release memory
            with self.lock:
                self.current_usage -= required
                print(f"🔓 Released {required/1024**3:.1f}GB (total: {self.current_usage/1024**3:.1f}GB)")
    
    def get_stats(self):
        """Get current memory statistics"""
        with self.lock:
            return {
                'total_gb': self.total_memory / 1024**3,
                'available_gb': self.available_memory / 1024**3,
                'current_usage_gb': self.current_usage / 1024**3,
                'free_gb': (self.available_memory - self.current_usage) / 1024**3
            }