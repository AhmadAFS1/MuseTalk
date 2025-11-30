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

from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import datagen


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
        
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        
        # Initialize avatar
        self.init(preparation, force_recreate)
    
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
        
        # Save avatar info
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
        
        # Extract frames from video
        self._extract_frames()
        
        # Process frames
        self._process_frames()
        
        print(f"✅ Avatar {self.avatar_id} preparation complete")
    
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
        
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
    
    def _load_existing_materials(self):
        """Load pre-processed avatar materials from disk"""
        print(f"📂 Loading avatar materials for {self.avatar_id}...")
        
        # Load latents
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        
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
            
            # Get corresponding original frame and bbox
            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = self.frame_list_cycle[self.idx % len(self.frame_list_cycle)].copy()
            x1, y1, x2, y2 = bbox
            
            # Resize result frame to bbox size
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            
            # Blend with mask
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            
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
        weight_dtype = self.unet.model.dtype
        
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
        )
        
        print(f"   Audio processing took {(time.time() - start_time) * 1000:.0f}ms")
        
        # Setup frame processing
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        
        # Start background thread for frame blending
        process_thread = threading.Thread(
            target=self._process_result_frames,
            args=(res_frame_queue, video_num, skip_save_images)
        )
        process_thread.start()
        
        # Generate frames batch by batch
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()
        
        for whisper_batch, latent_batch in tqdm(
            gen, 
            total=int(np.ceil(float(video_num) / self.batch_size)),
            desc="Generating frames"
        ):
            # Encode audio features
            audio_feature_batch = self.pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=self.unet.model.dtype)
            
            # Run UNet
            pred_latents = self.unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            # Decode latents
            pred_latents = pred_latents.to(device=device, dtype=self.vae.vae.dtype)
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

    @torch.no_grad()
    def inference_streaming(self, audio_path, audio_processor, whisper, timesteps, device, 
                            out_vid_name=None, fps=25, chunk_duration_seconds=2):
        """
        Stream video chunks as they're generated.
        
        Args:
            chunk_duration_seconds: Duration of each chunk (e.g., 2 seconds)
        
        Yields:
            dict with 'chunk_path', 'chunk_index', 'total_chunks'
        """
        os.makedirs(f"{self.avatar_path}/tmp", exist_ok=True)
        os.makedirs(f"{self.avatar_path}/chunks", exist_ok=True)
        
        # Process audio
        weight_dtype = self.unet.model.dtype
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, weight_dtype=weight_dtype
        )
        
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper,
            librosa_length, fps=fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        )
        
        video_num = len(whisper_chunks)
        frames_per_chunk = int(chunk_duration_seconds * fps)
        total_chunks = int(np.ceil(video_num / frames_per_chunk))
        
        # Generate frames and create chunks
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        frame_buffer = []
        chunk_index = 0
        frame_idx = 0
        
        for whisper_batch, latent_batch in gen:
            # Generate batch
            audio_feature_batch = self.pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=self.unet.model.dtype)
            
            pred_latents = self.unet.model(
                latent_batch, timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            pred_latents = pred_latents.to(device=device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            # Process each frame
            for res_frame in recon:
                # Blend frame
                bbox = self.coord_list_cycle[frame_idx % len(self.coord_list_cycle)]
                ori_frame = self.frame_list_cycle[frame_idx % len(self.frame_list_cycle)].copy()
                x1, y1, x2, y2 = bbox
                
                res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                mask = self.mask_list_cycle[frame_idx % len(self.mask_list_cycle)]
                mask_crop_box = self.mask_coords_list_cycle[frame_idx % len(self.mask_coords_list_cycle)]
                
                combine_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
                frame_buffer.append(combine_frame)
                frame_idx += 1
                
                # When buffer reaches chunk size, create video chunk
                if len(frame_buffer) >= frames_per_chunk or frame_idx >= video_num:
                    chunk_path = self._create_chunk(
                        frame_buffer, chunk_index, audio_path, fps,
                        start_frame=chunk_index * frames_per_chunk,
                        total_frames=video_num
                    )
                    
                    yield {
                        'chunk_path': chunk_path,
                        'chunk_index': chunk_index,
                        'total_chunks': total_chunks,
                        'duration_seconds': len(frame_buffer) / fps
                    }
                    
                    frame_buffer = []
                    chunk_index += 1
        
        # Cleanup
        shutil.rmtree(f"{self.avatar_path}/tmp", ignore_errors=True)

    def _create_chunk(self, frames, chunk_index, audio_path, fps, start_frame, total_frames):
        """Create a video chunk from frames"""
        chunk_dir = f"{self.avatar_path}/chunks/chunk_{chunk_index:04d}"
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save frames
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{chunk_dir}/{i:08d}.png", frame)
        
        # Create video from frames
        chunk_video = f"{chunk_dir}/video.mp4"
        cmd = (
            f"ffmpeg -y -v quiet -r {fps} -f image2 "
            f"-i {chunk_dir}/%08d.png "
            f"-vcodec libx264 -vf format=yuv420p -crf 23 "
            f"{chunk_video}"
        )
        os.system(cmd)
        
        # Extract audio segment
        start_time = start_frame / fps
        duration = len(frames) / fps
        
        chunk_with_audio = f"{self.avatar_path}/chunks/chunk_{chunk_index:04d}.mp4"
        cmd_audio = (
            f"ffmpeg -y -v quiet "
            f"-ss {start_time} -t {duration} -i {audio_path} "
            f"-i {chunk_video} "
            f"-c:v copy -c:a aac "
            f"{chunk_with_audio}"
        )
        os.system(cmd_audio)
        
        # Cleanup temp files
        shutil.rmtree(chunk_dir)
        
        return chunk_with_audio