"""
API-friendly Avatar implementation for MuseTalk.
Does NOT modify original realtime_inference.py - creates new class from scratch.
"""

import os
import torch
import glob
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import threading
import queue
import time
import json
import sys
from pathlib import Path
import subprocess

from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import datagen


def _build_ffmpeg_chunk_cmd(
    *,
    width: int,
    height: int,
    fps: int,
    audio_path: str,
    start_time: float,
    duration: float,
    output_path: str,
    encoder: str,
):
    output_suffix = Path(output_path).suffix.lower()
    use_mpegts = output_suffix == ".ts"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-v", "error",
        "-nostats",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", audio_path,
    ]

    if encoder == "h264_nvenc":
        video_opts = [
            "-c:v", "h264_nvenc",
            "-preset", os.getenv("HLS_CHUNK_ENCODER_PRESET", "p1"),
            "-tune", os.getenv("HLS_CHUNK_ENCODER_TUNE", "ull"),
            "-rc", "constqp",
            "-qp", os.getenv("HLS_CHUNK_ENCODER_QP", "28"),
            "-pix_fmt", "yuv420p",
        ]
    else:
        video_opts = [
            "-c:v", "libx264",
            "-preset", os.getenv("HLS_CHUNK_ENCODER_PRESET", "ultrafast"),
            "-tune", "zerolatency",
            "-crf", os.getenv("HLS_CHUNK_ENCODER_CRF", "28"),
            "-pix_fmt", "yuv420p",
        ]

    if use_mpegts:
        gop = max(1, int(round(fps * duration)))
        ffmpeg_cmd += video_opts + [
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "48000",
            "-f", "mpegts",
            output_path,
        ]
    else:
        ffmpeg_cmd += video_opts + [
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof+faststart",
            "-frag_duration", str(int(duration * 1000000)),
            "-f", "mp4",
            output_path,
        ]

    return ffmpeg_cmd


@torch.no_grad()
class APIAvatar:
    """
    API-friendly avatar class - no user input prompts, proper exception handling.
    Inspired by realtime_inference.py but designed for server/API usage.
    """
    
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, 
                 vae, unet, pe, fp, args, preparation=True, force_recreate=False):
        """
        Args:
            avatar_id: Unique identifier for this avatar
            video_path: Path to source video or image directory
            bbox_shift: Bounding box shift value
            batch_size: Batch size for inference
            vae, unet, pe, fp: Model components (passed in, not global)
            args: Configuration namespace
            preparation: If True, prepare materials from video
            force_recreate: If True, recreate even if exists
        """
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.idx = 0
        
        # Store model references (no globals!)
        self.vae = vae
        self.unet = unet
        self.pe = pe
        self.fp = fp
        self.args = args
        self.unet_dtype = getattr(unet, "model_dtype", getattr(getattr(unet, "model", None), "dtype", torch.float16))
        self.vae_dtype = getattr(vae, "runtime_dtype", getattr(getattr(vae, "vae", None), "dtype", torch.float16))
        
        # Setup paths based on version
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"
        
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self._idle_frame_cache: list[np.ndarray] | None = None
        
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        self._cpu_pe_cache = {}
        
        # Initialize avatar
        self.init(preparation, force_recreate)

    def _finalize_latent_cycle(self) -> None:
        """Normalize latent storage to a contiguous tensor for fast indexing."""
        latents = self.input_latent_list_cycle
        if isinstance(latents, torch.Tensor):
            tensor = latents
        else:
            tensor = torch.stack(latents, dim=0)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.contiguous()
        if tensor.device.type == "cpu" and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        self.input_latent_list_cycle = tensor
        self.input_latent_cycle_tensor = self.input_latent_list_cycle
        if self.input_latent_cycle_tensor.dim() == 5 and self.input_latent_cycle_tensor.shape[1] == 1:
            batch_tensor = self.input_latent_cycle_tensor.squeeze(1).contiguous()
        else:
            batch_tensor = self.input_latent_cycle_tensor
        if batch_tensor.device.type == "cpu" and torch.cuda.is_available() and not batch_tensor.is_pinned():
            batch_tensor = batch_tensor.pin_memory()
        self.input_latent_cycle_batch_tensor = batch_tensor

    @staticmethod
    def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
        if not isinstance(tensor, torch.Tensor):
            return 0
        try:
            return int(tensor.untyped_storage().nbytes())
        except Exception:
            return int(tensor.element_size() * tensor.numel())

    @staticmethod
    def _numpy_sequence_nbytes(values) -> int:
        if not values:
            return 0
        total = 0
        for value in values:
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
        return total

    def estimate_memory_usage_bytes(self) -> int:
        """
        Estimate the RAM footprint of the loaded avatar materials.

        The cache uses this to budget avatars by the actual prepared materials
        in memory instead of assuming every avatar costs the same amount.
        """
        total = 0
        seen_tensor_storages = set()

        for attr in ("input_latent_cycle_tensor", "input_latent_cycle_batch_tensor"):
            tensor = getattr(self, attr, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            try:
                storage_ptr = tensor.untyped_storage().data_ptr()
            except Exception:
                storage_ptr = tensor.data_ptr()
            if storage_ptr in seen_tensor_storages:
                continue
            seen_tensor_storages.add(storage_ptr)
            total += self._tensor_storage_nbytes(tensor)

        total += self._numpy_sequence_nbytes(getattr(self, "frame_list_cycle", None))
        total += self._numpy_sequence_nbytes(getattr(self, "mask_list_cycle", None))
        total += self._numpy_sequence_nbytes(getattr(self, "_idle_frame_cache", None))

        coord_list = getattr(self, "coord_list_cycle", None) or []
        mask_coord_list = getattr(self, "mask_coords_list_cycle", None) or []
        total += len(coord_list) * 4 * 8
        total += len(mask_coord_list) * 4 * 8

        return int(total)

    def estimate_memory_usage_mb(self) -> float:
        return self.estimate_memory_usage_bytes() / (1024 * 1024)

    def apply_positional_encoding_cpu(self, audio_prompts: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding on CPU and cache the constant PE slice.

        The PE stage is just an additive buffer, so there is no need to pay for
        it repeatedly on the live GPU loop.
        """
        if not isinstance(audio_prompts, torch.Tensor) or audio_prompts.dim() != 3:
            return audio_prompts

        pe_buffer = getattr(self.pe, "pe", None)
        if pe_buffer is None:
            return audio_prompts

        cache_key = (audio_prompts.shape[1], str(audio_prompts.dtype))
        pe_slice = self._cpu_pe_cache.get(cache_key)
        if pe_slice is None:
            pe_slice = pe_buffer[:, :audio_prompts.shape[1], :].detach().to(
                device="cpu",
                dtype=audio_prompts.dtype,
            ).contiguous()
            self._cpu_pe_cache[cache_key] = pe_slice

        return (audio_prompts + pe_slice).contiguous()
    
    def init(self, preparation, force_recreate):
        """Initialize avatar - prepare or load existing materials"""
        
        if preparation:
            # PREPARATION MODE: Create avatar materials from video
            if os.path.exists(self.avatar_path):
                if force_recreate:
                    print(f"♻️  Re-creating avatar {self.avatar_id} (force_recreate=True)")
                    shutil.rmtree(self.avatar_path)
                    self._create_avatar()
                else:
                    print(f"✅ Avatar {self.avatar_id} already exists, loading existing materials")
                    self._load_existing_materials()
            else:
                print(f"🆕 Creating new avatar {self.avatar_id}")
                self._create_avatar()
        else:
            # INFERENCE MODE: Load existing materials only
            if not os.path.exists(self.avatar_path):
                raise ValueError(
                    f"Avatar {self.avatar_id} does not exist. "
                    f"Create it first with preparation=True"
                )
            
            # Check for bbox_shift mismatch
            if os.path.exists(self.avatar_info_path):
                with open(self.avatar_info_path, "r") as f:
                    saved_info = json.load(f)
                
                if saved_info.get('bbox_shift') != self.bbox_shift:
                    print(
                        f"⚠️  Warning: bbox_shift mismatch for {self.avatar_id}\n"
                        f"   Saved: {saved_info.get('bbox_shift')}, Requested: {self.bbox_shift}\n"
                        f"   Avatar may not work correctly. Consider re-preparing."
                    )
            
            self._load_existing_materials()
    
    def _create_avatar(self):
        """Create avatar materials from video"""
        print(f"🔨 Preparing avatar materials for {self.avatar_id}...")
        
        # Create directories
        os.makedirs(self.avatar_path, exist_ok=True)
        os.makedirs(self.full_imgs_path, exist_ok=True)
        os.makedirs(self.video_out_path, exist_ok=True)
        os.makedirs(self.mask_out_path, exist_ok=True)
        
        # ✅ NEW: Save input video for playback
        input_video_path = f"{self.avatar_path}/input_video.mp4"
        if os.path.isfile(self.video_path):
            # Copy video file
            shutil.copy2(self.video_path, input_video_path)
            print(f"📹 Saved input video: {input_video_path}")
        elif os.path.isdir(self.video_path):
            # Convert image directory to video (for consistency)
            print(f"📹 Converting image directory to video...")
            self._convert_images_to_video(self.video_path, input_video_path, fps=25)
        
        # Save avatar info
        self.avatar_info['input_video_path'] = input_video_path  # ✅ Track input video
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
        
        # Extract frames from video
        self._extract_frames()
        
        # Process frames
        self._process_frames()
        
        print(f"✅ Avatar {self.avatar_id} preparation complete")

    def _convert_images_to_video(self, image_dir, output_path, fps=25):
        """Convert image directory to video file"""
        import subprocess
        
        # Get first image to determine resolution
        images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        
        first_image = cv2.imread(os.path.join(image_dir, images[0]))
        height, width = first_image.shape[:2]
        
        # Use FFmpeg to create video from images
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', f"{image_dir}/*.png",
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Created video from images: {output_path}")

    def _extract_frames(self):
        """Extract frames from video or copy from directory"""
        if os.path.isfile(self.video_path):
            print(f"📹 Extracting frames from video: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            count = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(f"{self.full_imgs_path}/{count:08d}.png", frame)
                    count += 1
                else:
                    break
            cap.release()
            print(f"   Extracted {count} frames")
        elif os.path.isdir(self.video_path):
            print(f"📂 Copying frames from directory: {self.video_path}")
            files = sorted([f for f in os.listdir(self.video_path) if f.endswith('.png')])
            for filename in files:
                shutil.copyfile(
                    f"{self.video_path}/{filename}",
                    f"{self.full_imgs_path}/{filename}"
                )
            print(f"   Copied {len(files)} frames")
        else:
            raise ValueError(f"Invalid video_path: {self.video_path}")
    
    def _process_frames(self):
        """Process frames: detect faces, encode latents, create masks"""
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        
        print("🔍 Detecting faces and landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        
        print("🧠 Encoding latents...")
        input_latent_list = []
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Apply extra margin for v15
            if self.args.version == "v15":
                y2 = y2 + self.args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            
            # Crop and resize
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(
                crop_frame, (256, 256), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Encode with VAE
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        
        # Create cyclic lists (forward + backward)
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        print("🎭 Creating masks...")
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        
        for i, frame in enumerate(tqdm(self.frame_list_cycle, desc="Processing masks")):
            # Save frame
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            
            # Create mask
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = self.args.parsing_mode if self.args.version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], 
                fp=self.fp, mode=mode
            )
            
            # Save mask
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)
        
        # Save processed data
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        
        self._finalize_latent_cycle()
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
    
    def _load_existing_materials(self):
        """Load pre-processed avatar materials from disk"""
        print(f"📂 Loading avatar materials for {self.avatar_id}...")
        
        # Load latents
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        self._finalize_latent_cycle()
        
        # Load coordinates
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        
        # Load frames
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        # Load mask coordinates
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        
        # Load masks
        input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.png')))
        self.mask_list_cycle = read_imgs(input_mask_list)
        
        print(f"✅ Loaded {len(self.frame_list_cycle)} frames")
    
    def _process_result_frames(self, res_frame_queue, video_len, skip_save_images):
        """Background thread to process and blend result frames"""
        while True:
            if self.idx >= video_len - 1:
                break
            
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            try:
                combine_frame = self.compose_frame(res_frame, self.idx)
            except Exception:
                continue
            
            # Save frame if needed
            if not skip_save_images:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            
            self.idx += 1
    
    @torch.no_grad()
    def inference(self, audio_path, audio_processor, whisper, timesteps, device, 
                  out_vid_name=None, fps=25, skip_save_images=False):
        """
        Run inference to generate talking head video.
        
        Args:
            audio_path: Path to audio file
            audio_processor: Audio processor instance
            whisper: Whisper model instance
            timesteps: Timesteps tensor
            device: Torch device
            out_vid_name: Output video name (without extension)
            fps: Frames per second
            skip_save_images: Skip saving intermediate frames
        """
        os.makedirs(f"{self.avatar_path}/tmp", exist_ok=True)
        print(f"🎙️  Processing audio: {audio_path}")
        
        # Extract audio features
        start_time = time.time()
        weight_dtype = self.unet_dtype
        
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, weight_dtype=weight_dtype
        )
        
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        ).detach().cpu().contiguous()
        conditioning_chunks = self.apply_positional_encoding_cpu(whisper_chunks)
        
        print(f"   Audio processing took {(time.time() - start_time) * 1000:.0f}ms")
        
        # Setup frame processing
        video_num = len(conditioning_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        
        # Start background thread for frame blending
        process_thread = threading.Thread(
            target=self._process_result_frames,
            args=(res_frame_queue, video_num, skip_save_images)
        )
        process_thread.start()
        
        # Generate frames batch by batch
        gen = datagen(conditioning_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()
        
        for whisper_batch, latent_batch in tqdm(
            gen, 
            total=int(np.ceil(float(video_num) / self.batch_size)),
            desc="Generating frames"
        ):
            # Positional encoding was precomputed on CPU; only copy the final
            # conditioning states needed by the UNet batch.
            audio_feature_batch = whisper_batch.to(
                device=device,
                dtype=self.unet_dtype,
                non_blocking=True,
            )
            latent_batch = latent_batch.to(device=device, dtype=self.unet_dtype)
            
            # Run UNet
            pred_latents = self.unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            # Decode latents
            pred_latents = pred_latents.to(device=device, dtype=self.vae_dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            # Queue result frames
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        
        # Wait for frame processing to complete
        process_thread.join()
        
        inference_time = time.time() - start_time
        print(f"⚡ Generated {video_num} frames in {inference_time:.1f}s ({video_num/inference_time:.1f} FPS)")
        
        # Create final video
        if out_vid_name and not skip_save_images:
            self._create_video(out_vid_name, audio_path, fps)
        
        # Cleanup temp files
        if os.path.exists(f"{self.avatar_path}/tmp"):
            shutil.rmtree(f"{self.avatar_path}/tmp")
    
    def _create_video(self, out_vid_name, audio_path, fps):
        """Create final video from frames and audio"""
        print(f"🎬 Creating video: {out_vid_name}.mp4")
        
        # Frames to video
        cmd_img2video = (
            f"ffmpeg -y -v warning -r {fps} -f image2 "
            f"-i {self.avatar_path}/tmp/%08d.png "
            f"-vcodec libx264 -vf format=yuv420p -crf 18 "
            f"{self.avatar_path}/temp.mp4"
        )
        os.system(cmd_img2video)
        
        # Combine with audio
        output_vid = os.path.join(self.video_out_path, f"{out_vid_name}.mp4")
        cmd_combine_audio = (
            f"ffmpeg -y -v warning "
            f"-i {audio_path} -i {self.avatar_path}/temp.mp4 "
            f"{output_vid}"
        )
        os.system(cmd_combine_audio)
        
        # Cleanup
        os.remove(f"{self.avatar_path}/temp.mp4")
        
        print(f"✅ Video saved: {output_vid}")
        return output_vid

    def _get_idle_frames(self, max_frames: int = 24) -> list:
        """
        Load and cache idle video frames for smooth transitions.
        Returns up to max_frames frames (BGR) or an empty list on failure.
        """
        if self._idle_frame_cache is not None:
            return self._idle_frame_cache

        frames: list[np.ndarray] = []
        try:
            video_path = Path(self.video_path)
            if not video_path.exists() or not video_path.is_file():
                return []

            cap = cv2.VideoCapture(str(video_path))
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        except Exception as exc:
            print(f"⚠️ Failed to load idle frames: {exc}")
            frames = []

        self._idle_frame_cache = frames
        return frames

    def compose_frame(self, res_frame, cycle_index: int):
        """Blend a decoded face frame back into the avatar frame cycle."""
        bbox = self.coord_list_cycle[cycle_index % len(self.coord_list_cycle)]
        ori_frame = self.frame_list_cycle[cycle_index % len(self.frame_list_cycle)].copy()
        x1, y1, x2, y2 = bbox

        if res_frame.dtype != np.uint8:
            res_frame = res_frame.astype(np.uint8)
        res_frame_resized = cv2.resize(res_frame, (x2 - x1, y2 - y1))
        mask = self.mask_list_cycle[cycle_index % len(self.mask_list_cycle)]
        mask_crop_box = self.mask_coords_list_cycle[cycle_index % len(self.mask_coords_list_cycle)]
        return get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)

    @torch.no_grad()
    def inference_streaming(
        self,
        audio_path,
        audio_processor,
        whisper,
        timesteps,
        device,
        fps=25,
        chunk_duration_seconds=2,
        chunk_output_dir=None,
        frame_callback=None,
        emit_chunks=True,
        chunk_ext=".mp4",
        start_offset_seconds: float = 0.0,
        cancel_event=None,
        scratch_dir=None,
    ):
        """
        Stream video chunks as they're generated.
        
        Args:
            chunk_output_dir: Custom directory for chunks (enables multi-user isolation)
            start_offset_seconds: Optional offset into the avatar frame cycle (seconds)
        """
        
        # ═══════════════════════════════════════════════════════════
        # TIMING: Request received
        # ═══════════════════════════════════════════════════════════
        request_start = time.time()
        print(f"\n{'='*60}")
        print(f"🎬 STARTING STREAMING GENERATION")
        print(f"⏰ Request received at: {time.strftime('%H:%M:%S.{}'.format(int((request_start % 1) * 1000)))}")
        print(f"{'='*60}")
        
        # ✅ Use custom directory if provided, else use avatar-specific default
        if chunk_output_dir:
            chunk_dir = Path(chunk_output_dir)
            print(f"📁 Custom output directory: {chunk_dir}")
        else:
            chunk_dir = Path(self.avatar_path) / "chunks"
            print(f"📁 Default output directory: {chunk_dir}")
        if emit_chunks:
            chunk_dir.mkdir(parents=True, exist_ok=True)
        
        if scratch_dir:
            tmp_dir = str(Path(scratch_dir))
        else:
            tmp_dir = f"{self.avatar_path}/tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"📁 Scratch directory: {tmp_dir}")

        tmp_cleaned = False

        def cleanup_tmp_dir():
            nonlocal tmp_cleaned
            if tmp_cleaned:
                return
            tmp_cleaned = True
            print(f"🧹 Cleaning up scratch directory: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)

        def cancel_requested(stage: str) -> bool:
            if cancel_event is None or not cancel_event.is_set():
                return False
            print(f"⚠️  Streaming generation cancelled during {stage}")
            cleanup_tmp_dir()
            return True
        
        setup_elapsed = time.time() - request_start
        print(f"⏱️  Setup complete ({setup_elapsed:.3f}s)")

        if cancel_requested("setup"):
            return
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: AUDIO PROCESSING
        # ═══════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(f"🎙️  PHASE 1: Audio Processing")
        print(f"{'─'*60}")
        print(f"📄 Audio file: {audio_path}")
        
        audio_start = time.time()
        weight_dtype = self.unet_dtype

        if cancel_requested("audio feature extraction"):
            return
        
        print(f"⚙️  Extracting audio features (dtype: {weight_dtype})...")
        feature_extract_start = time.time()
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, weight_dtype=weight_dtype
        )
        feature_extract_elapsed = time.time() - feature_extract_start
        print(f"✓ Audio features extracted (librosa_length: {librosa_length}, took {feature_extract_elapsed:.3f}s)")

        if cancel_requested("whisper chunk creation"):
            return
        
        print(f"⚙️  Creating whisper chunks (fps: {fps})...")
        whisper_chunk_start = time.time()
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper,
            librosa_length, fps=fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        ).detach().cpu().contiguous()
        conditioning_chunks = self.apply_positional_encoding_cpu(whisper_chunks)
        whisper_chunk_elapsed = time.time() - whisper_chunk_start
        print(f"✓ Whisper chunks created ({whisper_chunk_elapsed:.3f}s)")
        
        audio_elapsed = time.time() - audio_start
        video_num = len(conditioning_chunks)
        frames_per_chunk = int(chunk_duration_seconds * fps)
        total_chunks = int(np.ceil(video_num / frames_per_chunk))
        
        print(f"✓ Audio processing complete ({audio_elapsed:.2f}s)")
        print(f"📊 Total frames: {video_num}")
        print(f"📊 Frames per chunk: {frames_per_chunk}")
        print(f"📊 Expected chunks: {total_chunks}")
        print(f"📊 Chunk duration: {chunk_duration_seconds}s")
        print(f"📊 Total duration: {video_num/fps:.2f}s")

        # Preload idle frames for automatic tail crossfade (no API changes)
        idle_frames = self._get_idle_frames(
            max_frames=max(8, min(24, int(fps * 0.8)))
        )
        crossfade_tail_frames = 0
        if emit_chunks and idle_frames:
            # Aim for ~0.3s fade, clamped to available frames and chunk size
            crossfade_tail_frames = max(4, int(fps * 0.15))
            crossfade_tail_frames = min(
                crossfade_tail_frames,
                len(idle_frames),
                frames_per_chunk - 1 if frames_per_chunk > 1 else crossfade_tail_frames
            )
            print(f"🎚️  Crossfade enabled: {crossfade_tail_frames} frames into idle video")
        else:
            print("🎚️  Crossfade disabled (no idle frames available)")
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 2: FRAME GENERATION
        # ═══════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(f"🎨 PHASE 2: Frame Generation & Streaming")
        print(f"{'─'*60}")
        
        offset_seconds = 0.0
        if start_offset_seconds:
            try:
                offset_seconds = float(start_offset_seconds)
            except (TypeError, ValueError):
                offset_seconds = 0.0
        if not (offset_seconds == offset_seconds) or offset_seconds in (float("inf"), float("-inf")):
            offset_seconds = 0.0
        if offset_seconds < 0:
            offset_seconds = 0.0

        start_offset_frames = int(round(offset_seconds * fps)) if offset_seconds > 0 else 0
        if start_offset_frames:
            print(f"INFO: start_offset_frames={start_offset_frames} ({offset_seconds:.2f}s)")

        gen = datagen(
            conditioning_chunks,
            self.input_latent_list_cycle,
            self.batch_size,
            delay_frame=start_offset_frames,
        )
        frame_buffer = []
        chunk_index = 0
        frame_idx = 0
        
        generation_start = time.time()
        total_batches = int(np.ceil(video_num / self.batch_size))
        
        print(f"⚙️  Starting generation loop (batch_size: {self.batch_size}, total_batches: {total_batches})")
        print(f"⏱️  Time to first frame generation: {generation_start - request_start:.3f}s")
        
        for batch_index, (whisper_batch, latent_batch) in enumerate(gen, start=1):
            if cancel_requested("generation loop"):
                return

            if batch_index == 1 or batch_index % 8 == 0 or batch_index == total_batches:
                print(f"    🔁 Batch {batch_index}/{total_batches} in progress")

            # Generate batch (no per-batch logging)
            audio_feature_batch = whisper_batch.to(
                device=device,
                dtype=self.unet_dtype,
                non_blocking=True,
            )
            
            latent_batch = latent_batch.to(device=device, dtype=self.unet_dtype)
            pred_latents = self.unet.model(
                latent_batch, timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            pred_latents = pred_latents.to(device=device, dtype=self.vae_dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            # Process each frame
            for res_frame in recon:
                if cancel_requested("frame processing"):
                    return

                cycle_index = frame_idx + start_offset_frames
                combine_frame = self.compose_frame(res_frame, cycle_index)
                frame_buffer.append(combine_frame)
                frame_idx += 1

                if frame_callback is not None:
                    try:
                        frame_callback(combine_frame, frame_idx, video_num)
                    except Exception as callback_err:
                        print(f"⚠️ frame_callback error: {callback_err}")
                
                # ═══════════════════════════════════════════════════════════
                # PHASE 3: CHUNK CREATION (when buffer is full)
                # ═══════════════════════════════════════════════════════════
                if len(frame_buffer) >= frames_per_chunk or frame_idx >= video_num:
                    print(f"\n  {'─'*56}")
                    print(f"  📦 CREATING CHUNK {chunk_index + 1}/{total_chunks}")
                    print(f"  {'─'*56}")
                    print(f"    📊 Buffer size: {len(frame_buffer)} frames")
                    print(f"    📊 Progress: {frame_idx}/{video_num} frames ({frame_idx/video_num*100:.1f}%)")
                    
                    # Track time to first chunk
                    if chunk_index == 0:
                        time_to_first_chunk = time.time() - request_start
                        print(f"    ⏱️  Time to first chunk: {time_to_first_chunk:.3f}s")
                    
                    if emit_chunks:
                        if cancel_requested("chunk creation"):
                            return

                        chunk_start = time.time()
                        is_final_chunk = frame_idx >= video_num
                        if is_final_chunk and crossfade_tail_frames > 0:
                            chunk_path = self._create_crossfade_chunk(
                                frames=frame_buffer,
                                idle_frames=idle_frames,
                                fade_frames=crossfade_tail_frames,
                                chunk_index=chunk_index,
                                audio_path=audio_path,
                                fps=fps,
                                start_frame=chunk_index * frames_per_chunk,
                                total_frames=video_num,
                                output_path=str(chunk_dir / f"chunk_{chunk_index:04d}{chunk_ext}")
                            )
                        else:
                            chunk_path = self._create_chunk(
                                frames=frame_buffer, 
                                chunk_index=chunk_index, 
                                audio_path=audio_path,
                                fps=fps,
                                start_frame=chunk_index * frames_per_chunk,
                                total_frames=video_num,
                                output_path=str(chunk_dir / f"chunk_{chunk_index:04d}{chunk_ext}")
                            )
                        chunk_elapsed = time.time() - chunk_start
                        
                        chunk_info = {
                            'chunk_path': chunk_path,
                            'chunk_index': chunk_index,
                            'total_chunks': total_chunks,
                            'duration_seconds': len(frame_buffer) / fps,
                            'creation_time': chunk_elapsed
                        }
                        
                        print(f"    ✅ Chunk created: {chunk_path}")
                        print(f"    ⏱️  Creation time: {chunk_elapsed:.2f}s")
                        print(f"    🎬 Duration: {chunk_info['duration_seconds']:.2f}s")
                        
                        yield chunk_info
                    else:
                        print(f"    🚀 Streaming {len(frame_buffer)} frames to live track (no chunk file)")
                    
                    frame_buffer = []
                    chunk_index += 1
        
        # ═══════════════════════════════════════════════════════════
        # FINAL SUMMARY
        # ═══════════════════════════════════════════════════════════
        total_elapsed = time.time() - generation_start
        request_total_elapsed = time.time() - request_start
        
        print(f"\n{'='*60}")
        print(f"✅ STREAMING GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Total frames generated: {frame_idx}")
        print(f"📊 Total chunks created: {chunk_index}")
        print(f"⏱️  Total generation time: {total_elapsed:.2f}s")
        print(f"⏱️  Total request time: {request_total_elapsed:.2f}s")
        print(f"⚡ Average FPS: {frame_idx/total_elapsed:.2f}")
        print(f"📁 Output directory: {chunk_dir}")
        print(f"{'='*60}\n")
        
        cleanup_tmp_dir()

    def _create_crossfade_chunk(
        self,
        frames: list,
        idle_frames: list,
        fade_frames: int,
        chunk_index: int,
        audio_path: str,
        fps: int,
        start_frame: int,
        total_frames: int,
        output_path: str,
    ):
        """Blend the tail of talking frames into idle frames for a seamless handoff."""
        if not frames or not idle_frames or fade_frames <= 0:
            return self._create_chunk(
                frames=frames,
                chunk_index=chunk_index,
                audio_path=audio_path,
                fps=fps,
                start_frame=start_frame,
                total_frames=total_frames,
                output_path=output_path,
            )

        fade_frames = min(fade_frames, len(frames), len(idle_frames))
        keep_count = max(0, len(frames) - fade_frames)
        final_frames: list[np.ndarray] = []

        if keep_count > 0:
            final_frames.extend(frames[:keep_count])

        for i in range(fade_frames):
            alpha = (i + 1) / (fade_frames + 1)
            eased_alpha = 1 - (1 - alpha) ** 2  # ease-out
            talk_frame = frames[keep_count + i] if (keep_count + i) < len(frames) else frames[-1]
            idle_frame = idle_frames[i % len(idle_frames)]

            if talk_frame.shape != idle_frame.shape:
                idle_frame = cv2.resize(idle_frame, (talk_frame.shape[1], talk_frame.shape[0]))

            blended = cv2.addWeighted(
                talk_frame.astype(np.float32), 1.0 - eased_alpha,
                idle_frame.astype(np.float32), eased_alpha,
                0
            ).astype(np.uint8)
            final_frames.append(blended)

        print(f"      🔀 Crossfade: {keep_count} talk frames + {fade_frames} blended frames")

        return self._create_chunk(
            frames=final_frames,
            chunk_index=chunk_index,
            audio_path=audio_path,
            fps=fps,
            start_frame=start_frame,
            total_frames=total_frames,
            output_path=output_path,
        )

    def _create_chunk(self, frames, chunk_index, audio_path, fps, start_frame, total_frames, output_path):
        """Create an encoded chunk (fMP4 for MSE or TS for HLS)."""
        chunk_start_time = time.time()

        start_time = start_frame / fps
        duration = len(frames) / fps

        output_suffix = Path(output_path).suffix.lower()
        use_mpegts = output_suffix == ".ts"

        if use_mpegts:
            print(f"      🔨 Creating TS segment {chunk_index} ({len(frames)} frames)...")
        else:
            print(f"      🔨 Creating fMP4 fragment {chunk_index} ({len(frames)} frames)...")
        
        height, width = frames[0].shape[:2]
        preferred_encoder = os.getenv("HLS_CHUNK_VIDEO_ENCODER", "h264_nvenc").strip() or "h264_nvenc"
        encoders_to_try = [preferred_encoder]
        if preferred_encoder != "libx264":
            encoders_to_try.append("libx264")

        last_error = None
        for encoder in encoders_to_try:
            ffmpeg_cmd = _build_ffmpeg_chunk_cmd(
                width=width,
                height=height,
                fps=fps,
                audio_path=audio_path,
                start_time=start_time,
                duration=duration,
                output_path=output_path,
                encoder=encoder,
            )
            proc = None
            try:
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                for frame in frames:
                    proc.stdin.write(frame.tobytes())

                proc.stdin.close()

                try:
                    returncode = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise RuntimeError(f"FFmpeg chunk encode timed out after 120s using {encoder}")

                if returncode != 0:
                    stderr_tail = ""
                    if proc.stderr is not None:
                        try:
                            stderr_tail = proc.stderr.read().decode("utf-8", errors="ignore")[-400:]
                        except Exception:
                            stderr_tail = ""
                    detail = f": {stderr_tail}" if stderr_tail else ""
                    raise RuntimeError(
                        f"FFmpeg failed with exit code {returncode} using {encoder}{detail}"
                    )
                if not Path(output_path).exists():
                    raise RuntimeError(f"Output file not created using {encoder}")

                file_size = Path(output_path).stat().st_size
                if file_size < 1024:
                    raise RuntimeError(f"Output file too small using {encoder}: {file_size} bytes")

                elapsed = time.time() - chunk_start_time
                print(f"      ✅ Segment created with {encoder} ({file_size/1024:.1f}KB, {elapsed:.2f}s)")
                return output_path
            except Exception as exc:
                last_error = exc
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                except OSError:
                    pass
                try:
                    Path(output_path).unlink(missing_ok=True)
                except OSError:
                    pass
                if encoder != encoders_to_try[-1]:
                    print(f"      ⚠️ Encoder {encoder} failed, retrying with fallback: {exc}")
                    continue
                print(f"      ❌ Chunk creation failed: {exc}")
                raise

        raise last_error or RuntimeError("Chunk creation failed without a reported error")
