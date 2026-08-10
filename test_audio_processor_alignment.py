import unittest

import torch

from musetalk.utils.audio_processor import (
    AudioProcessor,
    WHISPER_FEATURE_STEPS_PER_MODEL_FRAME,
    pad_whisper_feature_for_musetalk,
)


class AudioProcessorAlignmentTest(unittest.TestCase):
    def test_model_context_padding_is_fixed_at_two_whisper_steps(self):
        raw = torch.arange(1, 31, dtype=torch.float32).reshape(1, 30, 1, 1)

        padded = pad_whisper_feature_for_musetalk(raw)

        self.assertEqual(WHISPER_FEATURE_STEPS_PER_MODEL_FRAME, 2)
        self.assertEqual(torch.count_nonzero(padded[:, :4]).item(), 0)
        self.assertEqual(padded[0, 4, 0, 0].item(), 1)

    def test_15fps_prompt_window_is_centered_on_current_audio_time(self):
        processor = AudioProcessor.__new__(AudioProcessor)
        raw = torch.arange(1, 101, dtype=torch.float32).reshape(1, 100, 1, 1)
        padded = pad_whisper_feature_for_musetalk(raw)

        prompts = processor.build_audio_prompts(
            whisper_feature=padded,
            num_frames=15,
            fps=15,
            start_frame=6,
            end_frame=7,
        )

        # Video frame 6 is sampled at Whisper index floor(6 * 50 / 15) = 20.
        # Four fixed left-context steps map the ten-value window to raw feature
        # indices 16..25 (stored here as values 17..26). The old fps-scaled
        # padding incorrectly selected indices 12..21, an 80 ms content lag.
        self.assertEqual(
            prompts[0, :, 0].tolist(),
            [float(value) for value in range(17, 27)],
        )

    def test_25fps_prompt_mapping_remains_unchanged(self):
        processor = AudioProcessor.__new__(AudioProcessor)
        raw = torch.arange(1, 101, dtype=torch.float32).reshape(1, 100, 1, 1)
        padded = pad_whisper_feature_for_musetalk(raw)

        prompts = processor.build_audio_prompts(
            whisper_feature=padded,
            num_frames=25,
            fps=25,
            start_frame=6,
            end_frame=7,
        )

        self.assertEqual(
            prompts[0, :, 0].tolist(),
            [float(value) for value in range(9, 19)],
        )


if __name__ == "__main__":
    unittest.main()
