import unittest
from unittest.mock import patch

import numpy as np

from scripts.webrtc_audio_timeline import (
    ANALYSIS_SAMPLE_RATE,
    prepare_webrtc_audio_timeline,
)


class WebRTCAudioTimelineTests(unittest.TestCase):
    @patch("scripts.webrtc_audio_timeline.detect_speech_bounds")
    @patch("scripts.webrtc_audio_timeline._decode_mono_pcm")
    def test_disabled_trimming_preserves_silent_audio_without_activity_detection(
        self,
        decode_pcm,
        detect_speech,
    ):
        decode_pcm.return_value = np.zeros(ANALYSIS_SAMPLE_RATE * 2, dtype=np.int16)

        timeline = prepare_webrtc_audio_timeline("silent.wav", enabled=False)

        detect_speech.assert_not_called()
        self.assertFalse(timeline.normalized)
        self.assertEqual("silent.wav", timeline.media_path)
        self.assertEqual(2.0, timeline.original_duration_seconds)
        self.assertEqual(2.0, timeline.media_duration_seconds)
        self.assertEqual(0.0, timeline.trim_start_seconds)
        self.assertEqual(2.0, timeline.trim_end_seconds)


if __name__ == "__main__":
    unittest.main()
