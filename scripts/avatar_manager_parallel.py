import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
import uuid
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from scripts.api_avatar import APIAvatar  # ✅ Use new API-friendly class
from scripts.concurrent_gpu_manager import GPUMemoryManager
from scripts.avatar_cache import AvatarCache
from scripts.avatar_s3_store import AvatarS3Store
from scripts.trt_runtime import load_vae_trt_decoder


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
        self.models_compiled = False
        self.unet_compiled = False
        self.vae_compiled = False
        self.unet_dtype = torch.float16
        self.vae_dtype = torch.float16
        self.eager_unet_model = None
        self.eager_vae_model = None
        
        max_live_generations = max(1, int(os.getenv("LIVE_MAX_CONCURRENT_GENERATIONS", "1")))

        # GPU memory manager
        self.gpu_memory = GPUMemoryManager(
            total_memory_gb=24,
            reserved_gb=6,
            max_live_generations=max_live_generations,
        )
        
        # Smart avatar cache
        self.avatar_cache = AvatarCache(
            max_cached_avatars=self._env_int("AVATAR_CACHE_MAX_AVATARS", 0),
            ttl_seconds=self._env_int("AVATAR_CACHE_TTL_SECONDS", 3600),
            max_memory_mb=self._env_float("AVATAR_CACHE_MAX_MEMORY_MB", 6000),
            cleanup_interval=60
        )
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_inferences)
        
        # Request tracking
        self.active_requests = {}
        self.request_lock = threading.Lock()
        self.avatar_load_locks = {}
        self.avatar_load_locks_lock = threading.Lock()
        self.avatar_restore_attempted = set()
        self.avatar_restore_lock = threading.Lock()
        self.avatar_s3_store = AvatarS3Store.from_env(version=self.args.version)
        
        # Load models ONCE
        self._init_models()
        
        # Model access lock
        self.model_lock = threading.Lock()
        
        # Start cache cleanup
        self.avatar_cache.start_cleanup()

    @staticmethod
    def _env_int(name, default):
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _env_float(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _parse_batch_size_list(raw):
        values = []
        seen = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = max(1, int(token))
            except ValueError:
                continue
            if value in seen:
                continue
            values.append(value)
            seen.add(value)
        values.sort()
        return values

    @classmethod
    def _default_compile_shape_batches(cls):
        override = os.getenv("HLS_SCHEDULER_FIXED_BATCH_SIZES", "").strip()
        if override:
            parsed = cls._parse_batch_size_list(override)
            if parsed:
                return parsed

        values = [4, 8, 16, 32]
        try:
            max_batch = max(32, int(os.getenv("HLS_SCHEDULER_MAX_BATCH", "32")))
        except (TypeError, ValueError):
            max_batch = 32

        if max_batch > 32:
            rounded_upper = ((max_batch + 15) // 16) * 16
            next_size = 48
            while next_size <= rounded_upper:
                values.append(next_size)
                next_size += 16
        return values
    
    def _init_models(self):
        """Load models once"""
        print("🔧 Loading models...")
        
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.args.unet_model_path,
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config,
            device=self.device
        )

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        self.pe = self.pe.half().to(self.device).eval()
        self.pe.requires_grad_(False)
        self.vae.vae = self.vae.vae.half().to(self.device).eval()
        self.vae.vae.requires_grad_(False)
        self.unet.model = self.unet.model.half().to(self.device).eval()
        self.unet.model.requires_grad_(False)
        self.eager_unet_model = self.unet.model
        self.eager_vae_model = self.vae.vae
        self.unet_dtype = self.unet.model.dtype
        self.vae_dtype = self.vae.vae.dtype
        self.unet_in_channels = int(
            getattr(
                getattr(self.unet.model, "config", None),
                "in_channels",
                getattr(getattr(self.unet.model, "conv_in", None), "in_channels", 8),
            )
        )
        self.unet.model_dtype = self.unet_dtype
        self.vae.runtime_dtype = self.vae_dtype
        self.vae_backend = load_vae_trt_decoder(
            device=self.device,
            scaling_factor=self.vae.scaling_factor,
            vae_module=self.vae.vae,
        )
        self.vae.set_decode_backend(self.vae_backend)
        self.vae_decode_backend_name = self.vae.get_decode_backend_name()
        if self.vae.has_decode_backend():
            print(f"✅ VAE decode backend active: {self.vae_decode_backend_name}")
        else:
            print("ℹ️  VAE decode backend: PyTorch")
        
        self.audio_processor = AudioProcessor(feature_extractor_path=self.args.whisper_dir)
        weight_dtype = self.unet_dtype
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
        
        self.compile_models()
        self._warm_runtime_paths()

    @staticmethod
    def _env_enabled(name, default="1"):
        value = os.getenv(name, default).strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _compile_mode_candidates(self, module_name):
        raw_modes = (
            os.getenv(f"MUSETALK_COMPILE_{module_name.upper()}_MODES")
            or os.getenv("MUSETALK_COMPILE_MODES")
            or os.getenv("MUSETALK_COMPILE_MODE")
            or "reduce-overhead,max-autotune"
        )
        candidates = []
        seen = set()
        for token in raw_modes.split(","):
            mode = token.strip()
            if not mode:
                continue
            normalized = mode.lower()
            if normalized in {"default", "none"}:
                normalized = "default"
            if normalized in seen:
                continue
            candidates.append(normalized)
            seen.add(normalized)
        if not candidates:
            candidates.append("reduce-overhead")
        return candidates

    def _compile_warmup_batches(self):
        raw_batches = os.getenv("MUSETALK_COMPILE_WARMUP_BATCHES")
        if raw_batches is None:
            return self._default_compile_shape_batches()

        batches = self._parse_batch_size_list(raw_batches)
        return batches or self._default_compile_shape_batches()

    def _log_compile_failure(self, label, mode, exc):
        mode_label = "default" if mode == "default" else mode
        print(
            f"   ⚠️ {label} compile failed ({mode_label}): "
            f"{type(exc).__name__}: {exc!r}"
        )
        if self._env_enabled("MUSETALK_COMPILE_TRACEBACK", "0"):
            traceback.print_exc()

    def _compile_module_with_fallback(self, label, module, fullgraph=False):
        for mode in self._compile_mode_candidates(label):
            try:
                kwargs = {"fullgraph": fullgraph}
                if mode != "default":
                    kwargs["mode"] = mode
                compiled = torch.compile(module, **kwargs)
                print(f"   ✅ {label} compiled ({'default' if mode == 'default' else mode})")
                return compiled, mode
            except Exception as exc:
                self._log_compile_failure(label, mode, exc)
                if hasattr(torch, "_dynamo"):
                    torch._dynamo.reset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return module, None

    def _warm_compiled_unet(self, batches):
        print("🔧 Warming compiled UNet...")
        for bs in batches:
            try:
                with torch.no_grad():
                    dw = torch.randn(bs, 50, 384, device=self.device, dtype=self.unet_dtype)
                    dl = torch.randn(
                        bs,
                        self.unet_in_channels,
                        32,
                        32,
                        device=self.device,
                        dtype=self.unet_dtype,
                    )
                    af = self.pe(dw)
                    _ = self.unet.model(
                        dl,
                        self.timesteps,
                        encoder_hidden_states=af,
                    ).sample
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print(f"   ✅ UNet warmup bs={bs}")
            except Exception as exc:
                print(f"   ⚠️ UNet warmup bs={bs}: {type(exc).__name__}: {exc!r}")
                if self._env_enabled("MUSETALK_COMPILE_TRACEBACK", "0"):
                    traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
        return True

    def _warm_compiled_vae(self, batches):
        print("🔧 Warming compiled VAE...")
        for bs in batches:
            try:
                with torch.no_grad():
                    latents = torch.randn(bs, 4, 32, 32, device=self.device, dtype=self.vae_dtype)
                    _ = self.vae.decode_latents_tensor(latents)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print(f"   ✅ VAE warmup bs={bs}")
            except Exception as exc:
                print(f"   ⚠️ VAE warmup bs={bs}: {type(exc).__name__}: {exc!r}")
                if self._env_enabled("MUSETALK_COMPILE_TRACEBACK", "0"):
                    traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
        return True

    def compile_models(self):
        """Compile UNet and VAE with per-module fallback instead of all-or-nothing restore."""
        if not self._env_enabled("MUSETALK_COMPILE", "0"):
            print("ℹ️  torch.compile disabled (set MUSETALK_COMPILE=1 to enable)")
            return

        if not hasattr(torch, 'compile'):
            print("⚠️  torch.compile requires PyTorch >= 2.0")
            return

        if self.unet_compiled or self.vae_compiled:
            print("ℹ️  torch.compile already applied")
            return

        print("🔧 Compiling UNet + VAE with per-module fallback (first run will be slow)...")

        eager_unet_model = self.eager_unet_model or self.unet.model
        eager_vae_model = self.eager_vae_model or self.vae.vae
        warmup_batches = self._compile_warmup_batches()
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

        if self._env_enabled("MUSETALK_COMPILE_UNET", "1"):
            compiled_unet, unet_mode = self._compile_module_with_fallback(
                label="UNet",
                module=eager_unet_model,
                fullgraph=False,
            )
            if unet_mode is not None:
                self.unet.model = compiled_unet
                self.unet_compiled = self._warm_compiled_unet(warmup_batches)
                if not self.unet_compiled:
                    print("⚠️ UNet compile warmup failed; restoring eager UNet")
                    self.unet.model = eager_unet_model
            else:
                self.unet.model = eager_unet_model
        else:
            print("ℹ️  UNet compile disabled (set MUSETALK_COMPILE_UNET=1 to enable)")

        if self.vae.has_decode_backend():
            print(
                "ℹ️  VAE compile skipped "
                f"({self.vae.get_decode_backend_name()} decode backend active)"
            )
            self.vae_compiled = False
        elif self._env_enabled("MUSETALK_COMPILE_VAE", "1"):
            compiled_vae, vae_mode = self._compile_module_with_fallback(
                label="VAE",
                module=eager_vae_model,
                fullgraph=False,
            )
            if vae_mode is not None:
                self.vae.vae = compiled_vae
                self.vae_compiled = self._warm_compiled_vae(warmup_batches)
                if not self.vae_compiled:
                    print("⚠️ VAE compile warmup failed; restoring eager VAE")
                    self.vae.vae = eager_vae_model
            else:
                self.vae.vae = eager_vae_model
        else:
            print("ℹ️  VAE compile disabled (set MUSETALK_COMPILE_VAE=1 to enable)")

        self.models_compiled = self.unet_compiled or self.vae_compiled

        if self.models_compiled:
            compiled_parts = []
            if self.unet_compiled:
                compiled_parts.append("UNet")
            if self.vae_compiled:
                compiled_parts.append("VAE")
            print(f"✅ Model compilation complete ({', '.join(compiled_parts)})")
        else:
            print("⚠️ No models compiled successfully; continuing in eager mode")

    def _warm_runtime_paths(self):
        """Warm Whisper and PE paths so the first live HLS request pays less cold-start cost."""
        if os.getenv("MUSETALK_WARM_RUNTIME", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            print("ℹ️  runtime warmup disabled (set MUSETALK_WARM_RUNTIME=1 to enable)")
            return

        try:
            print("🔧 Warming audio + Whisper runtime path...")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            input_features = self.audio_processor.feature_extractor(
                dummy_audio,
                return_tensors="pt",
                sampling_rate=16000,
            ).input_features.to(device=self.device, dtype=self.unet_dtype)
            with torch.no_grad():
                _ = self.whisper.encoder(input_features, output_hidden_states=True).hidden_states
                dummy_audio_batch = torch.randn(1, 50, 384, device=self.device, dtype=self.unet_dtype)
                _ = self.pe(dummy_audio_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("✅ Audio + Whisper warmup complete")
        except Exception as exc:
            print(f"⚠️  Audio + Whisper warmup failed: {exc}")
    
    def _get_or_load_avatar(self, avatar_id, batch_size):
        """Get avatar from cache or load from disk"""
        avatar = self.avatar_cache.get(avatar_id)
        
        if avatar is not None:
            # ✅ FIX: Update batch_size even for cached avatars
            if avatar.batch_size != batch_size:
                print(f"🔄 Updating batch_size for {avatar_id}: {avatar.batch_size} → {batch_size}")
                avatar.batch_size = batch_size
            return avatar
        
        if not self._avatar_exists(avatar_id):
            raise ValueError(f"Avatar {avatar_id} not found. Run preparation first.")

        with self._get_avatar_load_lock(avatar_id):
            # Another request may have loaded the avatar while we were waiting.
            avatar = self.avatar_cache.get(avatar_id)
            if avatar is not None:
                if avatar.batch_size != batch_size:
                    print(f"🔄 Updating batch_size for {avatar_id}: {avatar.batch_size} → {batch_size}")
                    avatar.batch_size = batch_size
                return avatar

            print(f"📂 Loading avatar {avatar_id} from disk...")
            
            # Create APIAvatar with explicit parameters
            avatar = APIAvatar(
                avatar_id=avatar_id,
                video_path="",
                bbox_shift=0,
                batch_size=batch_size,  # Use requested batch_size
                vae=self.vae,
                unet=self.unet,
                pe=self.pe,
                fp=self.fp,
                args=self.args,
                preparation=False,
                force_recreate=False
            )
            
            memory_usage_mb = self._estimate_avatar_cache_memory_mb(avatar)
            self.avatar_cache.put(avatar_id, avatar, memory_usage_mb)
            
            return avatar

    def _estimate_avatar_cache_memory_mb(self, avatar):
        if hasattr(avatar, "estimate_memory_usage_mb"):
            try:
                estimated_mb = float(avatar.estimate_memory_usage_mb())
                if estimated_mb > 0:
                    return estimated_mb
            except Exception as exc:
                print(f"⚠️  Failed to estimate avatar cache footprint for {getattr(avatar, 'avatar_id', 'unknown')}: {exc}")
        return 500.0

    def _get_avatar_load_lock(self, avatar_id):
        """Return a per-avatar lock so cold loads only happen once."""
        with self.avatar_load_locks_lock:
            lock = self.avatar_load_locks.get(avatar_id)
            if lock is None:
                lock = threading.Lock()
                self.avatar_load_locks[avatar_id] = lock
            return lock
    
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

        avatar_dir = self._avatar_dir(avatar_id)
        self.avatar_s3_store.upload_avatar_dir(avatar_id, avatar_dir)
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
        
        # ✅ PRE-FLIGHT CHECK: Verify avatar exists before queuing
        if not self._avatar_exists(avatar_id):
            raise ValueError(
                f"Avatar '{avatar_id}' not found. "
                f"Please prepare it first using POST /avatars/prepare"
            )
        
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
        path = self._avatar_latents_path(avatar_id)
        if os.path.exists(path):
            return True
        if self._restore_avatar_from_s3_if_needed(avatar_id):
            return os.path.exists(path)
        return False

    def _avatars_root(self):
        if self.args.version == "v15":
            return f"./results/{self.args.version}/avatars"
        return "./results/avatars"

    def _avatar_dir(self, avatar_id):
        return os.path.join(self._avatars_root(), avatar_id)

    def _avatar_latents_path(self, avatar_id):
        return os.path.join(self._avatar_dir(avatar_id), "latents.pt")

    def _restore_avatar_from_s3_if_needed(self, avatar_id):
        if not self.avatar_s3_store.enabled:
            return False

        with self.avatar_restore_lock:
            if avatar_id in self.avatar_restore_attempted:
                return False
            self.avatar_restore_attempted.add(avatar_id)

        avatar_lock = self._get_avatar_load_lock(avatar_id)
        with avatar_lock:
            if os.path.exists(self._avatar_latents_path(avatar_id)):
                return True
            return self.avatar_s3_store.download_avatar_dir(
                avatar_id=avatar_id,
                avatars_root=self._avatars_root_path(),
            )

    def _avatars_root_path(self):
        from pathlib import Path

        return Path(self._avatars_root())
    
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
