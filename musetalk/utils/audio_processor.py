import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path="openai/whisper-tiny/"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        if not os.path.exists(wav_path):
            return None
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        # Split audio into 30s segments
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        features = []
        try:
            # Local modification: this differs from the original MuseTalk code.
            # We batch segment feature extraction to reduce per-segment Python overhead.
            # Batch feature extraction so long audios do not pay Python overhead
            # and extractor setup cost once per 30s segment.
            audio_features = self.feature_extractor(
                segments,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).input_features
            if weight_dtype is not None:
                audio_features = audio_features.to(dtype=weight_dtype)
            features = list(audio_features.split(1, dim=0))
        except Exception:
            # Local modification: this differs from the original MuseTalk code.
            # Keep the original per-segment behavior as a compatibility fallback.
            # Keep the older per-segment path as a compatibility fallback for
            # any environment where batched extraction behaves differently.
            features = []
            for segment in segments:
                audio_feature = self.feature_extractor(
                    segment,
                    return_tensors="pt",
                    sampling_rate=sampling_rate
                ).input_features
                if weight_dtype is not None:
                    audio_feature = audio_feature.to(dtype=weight_dtype)
                features.append(audio_feature)

        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        whisper_feature, num_frames = self.encode_whisper_feature(
            whisper_input_features=whisper_input_features,
            device=device,
            weight_dtype=weight_dtype,
            whisper=whisper,
            librosa_length=librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )
        return self.build_audio_prompts(
            whisper_feature=whisper_feature,
            num_frames=num_frames,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )

    def encode_whisper_feature(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature_parts = []
        encode_batch_size = max(1, int(os.getenv("MUSETALK_WHISPER_SEGMENT_BATCH_SIZE", "4")))

        # Local modification: this differs from the original MuseTalk code.
        # Multiple 30s mel chunks are encoded in small batches to improve throughput.
        # Process multiple 30s mel input features in small encoder batches so
        # longer audio clips do less host-side scheduling and keep the GPU fed.
        for start in range(0, len(whisper_input_features), encode_batch_size):
            feature_batch = whisper_input_features[start:start + encode_batch_size]
            batched_input = torch.cat(feature_batch, dim=0).to(device=device, dtype=weight_dtype)
            audio_feats = whisper.encoder(batched_input, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2).contiguous()
            if audio_feats.shape[0] > 1:
                audio_feats = audio_feats.reshape(
                    1,
                    audio_feats.shape[0] * audio_feats.shape[1],
                    audio_feats.shape[2],
                    audio_feats.shape[3],
                )
            whisper_feature_parts.append(audio_feats)

        whisper_feature = torch.cat(whisper_feature_parts, dim=1)
        # Trim the last segment to remove padding
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:,:actual_length,...]

        # Calculate padding amount
        padding_nums = math.ceil(whisper_idx_multiplier)
        # Add padding at start and end
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            whisper_feature,
            # Add extra padding to prevent out of bounds
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])
        ], 1)

        return whisper_feature.detach().cpu().contiguous(), num_frames

    def build_audio_prompts(
        self,
        whisper_feature,
        num_frames,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        start_frame=0,
        end_frame=None,
    ):
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        fps = int(fps)
        audio_fps = 50
        whisper_idx_multiplier = audio_fps / fps

        if end_frame is None:
            end_frame = num_frames
        start_frame = max(0, int(start_frame))
        end_frame = min(int(end_frame), int(num_frames))
        if end_frame <= start_frame:
            return torch.empty(
                (0, audio_feature_length_per_frame * whisper_feature.shape[2], whisper_feature.shape[3]),
                dtype=whisper_feature.dtype,
            )

        # Local modification: this differs from the original MuseTalk code.
        # Prompt windows are gathered with tensor indexing instead of a Python frame loop.
        feature_source = whisper_feature[0].contiguous()
        frame_indices = torch.arange(
            start_frame,
            end_frame,
            device=feature_source.device,
            dtype=torch.long,
        )
        audio_indices = torch.div(frame_indices * audio_fps, fps, rounding_mode='floor')
        window_offsets = torch.arange(
            audio_feature_length_per_frame,
            device=feature_source.device,
            dtype=torch.long,
        )
        gather_indices = (audio_indices[:, None] + window_offsets[None, :]).reshape(-1)
        audio_prompts = feature_source.index_select(0, gather_indices)
        audio_prompts = audio_prompts.reshape(
            end_frame - start_frame,
            audio_feature_length_per_frame,
            feature_source.shape[1],
            feature_source.shape[2],
        )
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        return audio_prompts.contiguous()

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    wav_path = "./2.wav"
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature.shape)
    print("librosa_feature_length:", librosa_feature_length)
