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
from scripts.api_avatar import APIAvatar  # ✅ Use new API-friendly class
from scripts.concurrent_gpu_manager import GPUMemoryManager
from scripts.avatar_cache import AvatarCache


class ParallelAvatarManager:
    """
    Supports concurrent inference on single GPU with:
    - Memory budgeting for parallel requests
    - Smart avatar caching with TTL-based eviction
    - Thread-safe operations
    """
    
    def __init__(self, args, max_concurrent_inferences=3):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # GPU memory manager
        self.gpu_memory = GPUMemoryManager(total_memory_gb=24, reserved_gb=6)
        
        # Smart avatar cache
        self.avatar_cache = AvatarCache(
            max_cached_avatars=5,
            ttl_seconds=300,
            max_memory_mb=6000,
            cleanup_interval=60
        )
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_inferences)
        
        # Request tracking
        self.active_requests = {}
        self.request_lock = threading.Lock()
        
        # Load models ONCE
        self._init_models()
        
        # Model access lock
        self.model_lock = threading.Lock()
        
        # Start cache cleanup
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
        """Get avatar from cache or load from disk"""
        avatar = self.avatar_cache.get(avatar_id)
        
        if avatar is not None:
            return avatar
        
        if not self._avatar_exists(avatar_id):
            raise ValueError(f"Avatar {avatar_id} not found. Run preparation first.")
        
        print(f"📂 Loading avatar {avatar_id} from disk...")
        
        # ✅ Create APIAvatar with explicit parameters (no globals!)
        avatar = APIAvatar(
            avatar_id=avatar_id,
            video_path="",  # Not used when preparation=False
            bbox_shift=0,
            batch_size=batch_size,
            vae=self.vae,
            unet=self.unet,
            pe=self.pe,
            fp=self.fp,
            args=self.args,
            preparation=False,  # Load existing materials only
            force_recreate=False
        )
        
        memory_usage_mb = 500
        self.avatar_cache.put(avatar_id, avatar, memory_usage_mb)
        
        return avatar
    
    def prepare_avatar(self, avatar_id, video_path, bbox_shift=0, batch_size=20, force_recreate=False):
        """Prepare avatar (one-time operation)"""
        exists = self._avatar_exists(avatar_id)
        
        if exists and not force_recreate:
            print(f"✅ Avatar {avatar_id} already prepared")
            return
        
        print(f"🔨 Preparing avatar {avatar_id}...")
        
        # ✅ Create APIAvatar with preparation=True
        avatar = APIAvatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift if self.args.version == "v1" else 0,
            batch_size=batch_size,
            vae=self.vae,
            unet=self.unet,
            pe=self.pe,
            fp=self.fp,
            args=self.args,
            preparation=True,  # Prepare materials
            force_recreate=force_recreate
        )
        
        print(f"✅ Avatar {avatar_id} prepared and saved to disk")
    
    def _inference_worker(self, request_id, avatar_id, audio_path, batch_size, output_name, fps):
        """Worker function for inference"""
        print(f"🎬 [{request_id}] Starting inference for {avatar_id}")
        
        try:
            with self.gpu_memory.allocate(batch_size):
                avatar = self._get_or_load_avatar(avatar_id, batch_size)
                
                # ✅ Pass models explicitly (no globals!)
                with self.model_lock:
                    avatar.inference(
                        audio_path=audio_path,
                        audio_processor=self.audio_processor,
                        whisper=self.whisper,
                        timesteps=self.timesteps,
                        device=self.device,
                        out_vid_name=output_name or f"{avatar_id}_{uuid.uuid4().hex[:8]}",
                        fps=fps,
                        skip_save_images=False
                    )
                
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
        """Submit inference request (non-blocking)"""
        request_id = f"gen_{uuid.uuid4().hex[:8]}"
        
        future = self.executor.submit(
            self._inference_worker,
            request_id, avatar_id, audio_path, batch_size, output_name, fps
        )
        
        with self.request_lock:
            self.active_requests[request_id] = {
                'avatar_id': avatar_id,
                'status': 'processing',
                'future': future
            }
        
        if callback:
            future.add_done_callback(lambda f: callback(request_id, f.result()))
        
        print(f"📥 [{request_id}] Queued for {avatar_id}")
        return request_id
    
    def _avatar_exists(self, avatar_id):
        """Check if avatar exists on disk"""
        if self.args.version == "v15":
            path = f"./results/{self.args.version}/avatars/{avatar_id}/latents.pt"
        else:
            path = f"./results/avatars/{avatar_id}/latents.pt"
        return os.path.exists(path)
    
    def evict_avatar(self, avatar_id):
        """Manually evict avatar from cache"""
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
                    return {'status': 'completed', 'result': result}
                except Exception as e:
                    return {'status': 'failed', 'error': str(e)}
            else:
                return {'status': 'processing'}
    
    def get_stats(self):
        """Get comprehensive statistics"""
        gpu_stats = self.gpu_memory.get_stats()
        cache_stats = self.avatar_cache.get_stats()
        
        with self.request_lock:
            active_count = len([r for r in self.active_requests.values() if not r['future'].done()])
        
        return {
            'gpu': gpu_stats,
            'cache': cache_stats,
            'active_requests': active_count,
            'total_requests': len(self.active_requests)
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        print("🛑 Shutting down...")
        self.avatar_cache.stop_cleanup()
        self.avatar_cache.clear_all()
        self.executor.shutdown(wait=True)
        print("✅ Shutdown complete")