import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from scripts.realtime_inference import Avatar
from scripts.concurrent_gpu_manager import GPUMemoryManager
from scripts.avatar_cache import AvatarCache  # ✅ Import the cache


class ParallelAvatarManager:
    """
    Supports concurrent inference on single GPU with:
    - Memory budgeting for parallel requests
    - Smart avatar caching with TTL-based eviction
    - Thread-safe operations
    """
    
    def __init__(self, args, max_concurrent_inferences=3):
        """
        Args:
            max_concurrent_inferences: How many inferences can run simultaneously
                                      (limited by GPU VRAM)
        """
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # GPU memory manager
        self.gpu_memory = GPUMemoryManager(total_memory_gb=24, reserved_gb=6)
        
        # ✅ Smart avatar cache (replaces simple dict)
        self.avatar_cache = AvatarCache(
            max_cached_avatars=5,      # Max 5 avatars in VRAM
            ttl_seconds=300,           # Unload after 5 min inactivity
            max_memory_mb=6000,        # Max 6GB for avatars
            cleanup_interval=60        # Check every minute
        )
        
        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_inferences)
        
        # Request tracking
        self.active_requests = {}
        self.request_lock = threading.Lock()
        
        # Load models ONCE (shared across all requests)
        self._init_models()
        
        # Model access lock (critical for thread-safety)
        self.model_lock = threading.Lock()
        
        # ✅ Start cache cleanup thread
        self.avatar_cache.start_cleanup()
    
    def _init_models(self):
        """Load models once"""
        print("🔧 Loading models...")
        
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.args.unet_model_path,
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config,
            device=self.device
        )
        
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        
        self.audio_processor = AudioProcessor(feature_extractor_path=self.args.whisper_dir)
        weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(self.args.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        if self.args.version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=self.args.left_cheek_width,
                right_cheek_width=self.args.right_cheek_width
            )
        else:
            self.fp = FaceParsing()
        
        self.timesteps = torch.tensor([0], device=self.device)
        
        print("✅ Models loaded!")
    
    def _get_or_load_avatar(self, avatar_id, batch_size):
        """
        ✅ Get avatar from cache, or load from disk if not cached.
        This replaces the old _load_avatar method.
        """
        # Try cache first
        avatar = self.avatar_cache.get(avatar_id)
        
        if avatar is not None:
            return avatar  # Cache hit!
        
        # Cache miss - check if exists on disk
        if not self._avatar_exists(avatar_id):
            raise ValueError(f"Avatar {avatar_id} not found. Run preparation first.")
        
        # Load from disk
        print(f"📂 Loading avatar {avatar_id} from disk...")
        
        # Inject globals for Avatar class
        global vae, unet, pe, fp, args
        vae = self.vae
        unet = self.unet
        pe = self.pe
        fp = self.fp
        args = self.args
        
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path="",  # Will load from saved materials
            bbox_shift=0,
            batch_size=batch_size,
            preparation=False  # ✅ NO preparation, just load existing
        )
        
        # Estimate memory usage (empirical - adjust based on your testing)
        memory_usage_mb = 500
        
        # ✅ Add to cache (handles LRU eviction automatically)
        self.avatar_cache.put(avatar_id, avatar, memory_usage_mb)
        
        return avatar
    
    def prepare_avatar(self, avatar_id, video_path, bbox_shift=0, batch_size=20, force_recreate=False):
        """
        ✅ NEW: Prepare avatar (one-time operation).
        Saves materials to disk but does NOT load into VRAM.
        """
        exists = self._avatar_exists(avatar_id)
        
        if exists and not force_recreate:
            print(f"✅ Avatar {avatar_id} already prepared")
            return
        
        print(f"🔨 Preparing avatar {avatar_id}...")
        
        # Inject globals
        global vae, unet, pe, fp, args
        vae = self.vae
        unet = self.unet
        pe = self.pe
        fp = self.fp
        args = self.args
        
        # Create avatar with preparation=True
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift if self.args.version == "v1" else 0,
            batch_size=batch_size,
            preparation=True  # ✅ Run full preparation
        )
        
        print(f"✅ Avatar {avatar_id} prepared and saved to disk")
        
        # Note: We do NOT add to cache here
        # It will be loaded on-demand during first inference
    
    def _inference_worker(self, request_id, avatar_id, audio_path, batch_size, output_name, fps):
        """
        Worker function that runs in thread pool.
        Uses GPU memory budgeting to allow parallel execution.
        """
        print(f"🎬 [{request_id}] Starting inference for {avatar_id}")
        
        try:
            # Allocate GPU memory budget
            with self.gpu_memory.allocate(batch_size):
                
                # ✅ Get avatar (from cache or load from disk)
                avatar = self._get_or_load_avatar(avatar_id, batch_size)
                
                # Inject globals (protected by model_lock during inference)
                global vae, unet, pe, fp, args, audio_processor, whisper, timesteps
                
                # CRITICAL: Lock model access during inference
                # This ensures PyTorch operations don't conflict
                with self.model_lock:
                    vae = self.vae
                    unet = self.unet
                    pe = self.pe
                    fp = self.fp
                    args = self.args
                    audio_processor = self.audio_processor
                    whisper = self.whisper
                    timesteps = self.timesteps
                    
                    # Run inference
                    avatar.inference(
                        audio_path=audio_path,
                        out_vid_name=output_name or f"{avatar_id}_{uuid.uuid4().hex[:8]}",
                        fps=fps,
                        skip_save_images=False
                    )
                
                # Determine output path
                if self.args.version == "v15":
                    output_path = f"./results/{self.args.version}/avatars/{avatar_id}/vid_output/{output_name}.mp4"
                else:
                    output_path = f"./results/avatars/{avatar_id}/vid_output/{output_name}.mp4"
                
                print(f"✅ [{request_id}] Complete: {output_path}")
                
                return {'success': True, 'output_path': output_path}
        
        except Exception as e:
            print(f"❌ [{request_id}] Failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def generate_async(self, avatar_id, audio_path, batch_size=2, output_name=None, fps=25, callback=None):
        """
        Submit inference request (non-blocking).
        Returns immediately with request_id.
        """
        request_id = f"gen_{uuid.uuid4().hex[:8]}"
        
        # Submit to thread pool
        future = self.executor.submit(
            self._inference_worker,
            request_id, avatar_id, audio_path, batch_size, output_name, fps
        )
        
        # Track request
        with self.request_lock:
            self.active_requests[request_id] = {
                'avatar_id': avatar_id,
                'status': 'processing',
                'future': future
            }
        
        # Optional callback when complete
        if callback:
            future.add_done_callback(lambda f: callback(request_id, f.result()))
        
        print(f"📥 [{request_id}] Queued for {avatar_id}")
        return request_id
    
    def _avatar_exists(self, avatar_id):
        """Check if avatar prepared on disk"""
        if self.args.version == "v15":
            path = f"./results/{self.args.version}/avatars/{avatar_id}/latents.pt"
        else:
            path = f"./results/avatars/{avatar_id}/latents.pt"
        return os.path.exists(path)
    
    def evict_avatar(self, avatar_id):
        """
        ✅ NEW: Manually evict an avatar from cache (frees VRAM).
        Avatar still exists on disk.
        """
        return self.avatar_cache.evict(avatar_id, reason="Manual eviction")
    
    def get_request_status(self, request_id):
        """Check request status"""
        with self.request_lock:
            if request_id not in self.active_requests:
                return {'status': 'unknown'}
            
            req = self.active_requests[request_id]
            future = req['future']
            
            if future.done():
                try:
                    result = future.result()
                    return {
                        'status': 'completed',
                        'result': result
                    }
                except Exception as e:
                    return {
                        'status': 'failed',
                        'error': str(e)
                    }
            else:
                return {'status': 'processing'}
    
    def get_stats(self):
        """✅ UPDATED: Get comprehensive statistics"""
        gpu_stats = self.gpu_memory.get_stats()
        cache_stats = self.avatar_cache.get_stats()
        
        with self.request_lock:
            active_count = len([r for r in self.active_requests.values() 
                               if not r['future'].done()])
        
        return {
            'gpu': gpu_stats,
            'cache': cache_stats,  # ✅ Now includes cache info
            'active_requests': active_count,
            'total_requests': len(self.active_requests)
        }
    
    def shutdown(self):
        """✅ NEW: Graceful shutdown"""
        print("🛑 Shutting down...")
        self.avatar_cache.stop_cleanup()
        self.avatar_cache.clear_all()
        self.executor.shutdown(wait=True)
        print("✅ Shutdown complete")