"""Conservative whole-file digital silence detection for WebRTC uploads.

No RMS/VAD threshold, channel mixing, resampling, or sample quantization is used.
One nonzero decoded sample in any channel keeps the normal speech compositor.
"""
from __future__ import annotations

import numpy as np


def is_exact_silence(audio_path, *, cancel_event=None):
    """True only after decoding the nonempty sole audio stream entirely as zero.

    This examines the original uploaded media, before timeline normalization.
    A lossy container is judged by its decoded samples, not its encoded bytes.
    Unknown/failed decoding and cancellation retain the ordinary audio path;
    its existing validation/cancellation handling remains authoritative.
    """
    if cancel_event is not None and cancel_event.is_set():
        return False
    try:
        import av
        seen_samples = False
        with av.open(str(audio_path)) as container:
            # A container with several audio streams can select a different
            # default stream in ffmpeg. Do not classify an ambiguous upload.
            if len(container.streams.audio) != 1:
                return False
            for frame in container.decode(audio=0):
                if cancel_event is not None and cancel_event.is_set():
                    return False
                samples = frame.to_ndarray()
                if not samples.size:
                    continue
                if samples.dtype.kind not in 'ifu':
                    return False
                # Unsigned PCM8 has its digital zero at128. Other unsigned
                # formats are not accepted as evidence of silence.
                if samples.dtype.kind == 'u':
                    if samples.dtype != np.uint8:
                        return False
                    zero = 128
                else:
                    zero = 0
                if not np.all(samples == zero):
                    return False
                seen_samples = True
        return seen_samples and not (cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False
