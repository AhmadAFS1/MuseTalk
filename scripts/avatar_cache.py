import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class CachedAvatar:
    """Represents an avatar loaded in memory"""
    avatar_id: str
    avatar_instance: any  # The actual Avatar object
    last_access_time: float
    memory_usage_mb: float
    load_count: int = 0


class AvatarCache:
    """
    Smart cache for avatars with:
    - TTL-based eviction (unload after X seconds of inactivity)
    - LRU eviction (unload least recently used)
    - Max memory limit
    - Automatic cleanup thread
    """
    
    def __init__(self, 
                 max_cached_avatars=5,
                 ttl_seconds=300,  # 5 minutes default
                 max_memory_mb=8000,  # 8GB max for avatars
                 cleanup_interval=60):  # Check every minute
        """
        Args:
            max_cached_avatars: Maximum number of avatars in memory
            ttl_seconds: Time-to-live (unload after this many seconds of inactivity)
            max_memory_mb: Maximum memory for avatar cache
            cleanup_interval: How often to run cleanup (seconds)
        """
        self.max_cached_avatars = max_cached_avatars
        self.ttl_seconds = ttl_seconds
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval
        
        # OrderedDict for LRU behavior
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
        # Cleanup thread
        self.cleanup_thread = None
        self.running = False
        
        # Stats
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'loads': 0
        }
    
    def start_cleanup(self):
        """Start background cleanup thread"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        print(f"🧹 Avatar cache cleanup started (TTL={self.ttl_seconds}s)")
    
    def stop_cleanup(self):
        """Stop cleanup thread"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        print("🛑 Avatar cache cleanup stopped")
    
    def _cleanup_loop(self):
        """Background thread that periodically evicts stale avatars"""
        while self.running:
            time.sleep(self.cleanup_interval)
            self._evict_stale()
    
    def _evict_stale(self):
        """Evict avatars that haven't been used for TTL seconds"""
        now = time.time()
        to_evict = []
        
        with self.lock:
            for avatar_id, cached in self.cache.items():
                age = now - cached.last_access_time
                if age > self.ttl_seconds:
                    to_evict.append(avatar_id)
        
        # Evict outside lock to avoid deadlock
        for avatar_id in to_evict:
            self.evict(avatar_id, reason="TTL expired")
    
    def get(self, avatar_id):
        """
        Get avatar from cache (or None if not cached).
        Updates last_access_time.
        """
        with self.lock:
            if avatar_id in self.cache:
                # Move to end (most recently used)
                cached = self.cache.pop(avatar_id)
                cached.last_access_time = time.time()
                cached.load_count += 1
                self.cache[avatar_id] = cached
                
                self.stats['hits'] += 1
                print(f"✅ Cache HIT: {avatar_id} (loads: {cached.load_count})")
                return cached.avatar_instance
            else:
                self.stats['misses'] += 1
                print(f"❌ Cache MISS: {avatar_id}")
                return None
    
    def put(self, avatar_id, avatar_instance, memory_usage_mb=500):
        """
        Add avatar to cache.
        May evict LRU avatar if cache is full.
        """
        with self.lock:
            # Check if we need to make room
            current_memory = sum(c.memory_usage_mb for c in self.cache.values())
            
            # Evict LRU if over limits
            while (len(self.cache) >= self.max_cached_avatars or 
                   current_memory + memory_usage_mb > self.max_memory_mb):
                if not self.cache:
                    break
                # Pop oldest (first item in OrderedDict)
                lru_id, lru_cached = self.cache.popitem(last=False)
                current_memory -= lru_cached.memory_usage_mb
                self._cleanup_avatar(lru_cached.avatar_instance)
                self.stats['evictions'] += 1
                print(f"🗑️  Evicted LRU avatar: {lru_id} (freed {lru_cached.memory_usage_mb:.1f}MB)")
            
            # Add new avatar
            cached = CachedAvatar(
                avatar_id=avatar_id,
                avatar_instance=avatar_instance,
                last_access_time=time.time(),
                memory_usage_mb=memory_usage_mb,
                load_count=1
            )
            self.cache[avatar_id] = cached
            self.stats['loads'] += 1
            
            print(f"📦 Cached avatar: {avatar_id} ({memory_usage_mb:.1f}MB, total: {len(self.cache)})")
    
    def evict(self, avatar_id, reason="Manual eviction"):
        """Manually evict an avatar from cache"""
        with self.lock:
            if avatar_id in self.cache:
                cached = self.cache.pop(avatar_id)
                self._cleanup_avatar(cached.avatar_instance)
                self.stats['evictions'] += 1
                print(f"🗑️  Evicted avatar: {avatar_id} - {reason}")
                return True
            return False
    
    def _cleanup_avatar(self, avatar_instance):
        """Clean up avatar resources"""
        # Clear any GPU tensors
        if hasattr(avatar_instance, 'input_latent_list_cycle'):
            del avatar_instance.input_latent_list_cycle
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear_all(self):
        """Clear entire cache"""
        with self.lock:
            for cached in self.cache.values():
                self._cleanup_avatar(cached.avatar_instance)
            self.cache.clear()
        print("🧹 Cache cleared")
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            total_memory = sum(c.memory_usage_mb for c in self.cache.values())
            hit_rate = (self.stats['hits'] / 
                       (self.stats['hits'] + self.stats['misses']) 
                       if self.stats['hits'] + self.stats['misses'] > 0 else 0)
            
            return {
                'cached_avatars': len(self.cache),
                'total_memory_mb': total_memory,
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': f"{hit_rate:.2%}",
                **self.stats,
                'avatars': [
                    {
                        'id': avatar_id,
                        'memory_mb': cached.memory_usage_mb,
                        'load_count': cached.load_count,
                        'age_seconds': time.time() - cached.last_access_time
                    }
                    for avatar_id, cached in self.cache.items()
                ]
            }