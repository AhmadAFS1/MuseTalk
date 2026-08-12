"""Normalize WebRTC TTS audio to the audible speech timeline.

MuseTalk generates one video frame schedule for the complete audio container.
TTS providers sometimes append seconds of digital or near-digital silence. If
that container duration is used unchanged, the generated mouth remains live
long after the speaker has audibly finished. This module removes only leading
and trailing silence, with small configurable padding, before the same file is
given to both MuseTalk and the WebRTC audio track.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


ANALYSIS_SAMPLE_RATE = 16_000


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class WebRTCAudioTimeline:
    source_path: str
    media_path: str
    original_duration_seconds: float
    media_duration_seconds: float
    speech_start_seconds: float
    speech_end_seconds: float
    trim_start_seconds: float
    trim_end_seconds: float
    leading_silence_removed_seconds: float
    trailing_silence_removed_seconds: float
    threshold_db: float
    normalized: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, float):
                result[key] = round(value, 6)
        return result


def _decode_mono_pcm(
    audio_path: Path,
    *,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode WebRTC audio: {error}")
    return np.frombuffer(result.stdout, dtype="<i2").copy()


def detect_speech_bounds(
    audio_path: str | Path,
    *,
    threshold_db: float = -45.0,
    window_seconds: float = 0.01,
    minimum_activity_seconds: float = 0.02,
) -> tuple[float, float, float]:
    """Return ``(first_activity, last_activity, duration)`` in seconds."""

    path = Path(audio_path)
    samples = _decode_mono_pcm(path)
    if samples.size == 0:
        raise RuntimeError(f"WebRTC audio contains no decoded samples: {path}")

    duration_seconds = samples.size / float(ANALYSIS_SAMPLE_RATE)
    window_samples = max(1, int(round(window_seconds * ANALYSIS_SAMPLE_RATE)))
    window_count = int(math.ceil(samples.size / float(window_samples)))
    padded = np.pad(
        samples.astype(np.float64),
        (0, window_count * window_samples - samples.size),
    )
    windows = padded.reshape(window_count, window_samples)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    threshold = 32768.0 * math.pow(10.0, float(threshold_db) / 20.0)
    active = rms >= threshold

    minimum_windows = max(
        1,
        int(math.ceil(minimum_activity_seconds / float(window_seconds))),
    )
    if minimum_windows > 1:
        # Keep only contiguous activity runs long enough to be speech.  A
        # convolution with ``> 0`` would instead *dilate* isolated clicks and
        # can turn one noisy sample near EOF into an untrimmed silent tail.
        filtered = np.zeros_like(active)
        padded_active = np.pad(active.astype(np.int8), (1, 1))
        edges = np.diff(padded_active)
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1)
        for run_start, run_end in zip(run_starts, run_ends):
            if int(run_end - run_start) >= minimum_windows:
                filtered[run_start:run_end] = True
        active = filtered

    indices = np.flatnonzero(active)
    if indices.size == 0:
        raise ValueError(
            f"WebRTC audio has no sustained activity above {threshold_db:.1f} dB: {path}"
        )

    speech_start = float(
        indices[0] * window_samples / float(ANALYSIS_SAMPLE_RATE)
    )
    speech_end = float(min(
        duration_seconds,
        (indices[-1] + 1) * window_samples / float(ANALYSIS_SAMPLE_RATE),
    ))
    return speech_start, speech_end, duration_seconds


def prepare_webrtc_audio_timeline(
    audio_path: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    enabled: Optional[bool] = None,
    threshold_db: Optional[float] = None,
    leading_padding_seconds: Optional[float] = None,
    trailing_padding_seconds: Optional[float] = None,
    minimum_trim_seconds: Optional[float] = None,
) -> WebRTCAudioTimeline:
    """Create the single timestamp source used by inference and audio playout."""

    source = Path(audio_path)
    trim_enabled = (
        _env_bool("WEBRTC_TRIM_EDGE_SILENCE", True)
        if enabled is None
        else bool(enabled)
    )
    threshold = (
        _env_float("WEBRTC_AUDIO_ACTIVITY_THRESHOLD_DB", -45.0)
        if threshold_db is None
        else float(threshold_db)
    )
    leading_padding = max(
        0.0,
        (
            _env_float("WEBRTC_AUDIO_LEADING_PADDING_SECONDS", 0.08)
            if leading_padding_seconds is None
            else float(leading_padding_seconds)
        ),
    )
    trailing_padding = max(
        0.0,
        (
            _env_float("WEBRTC_AUDIO_TRAILING_PADDING_SECONDS", 0.04)
            if trailing_padding_seconds is None
            else float(trailing_padding_seconds)
        ),
    )
    minimum_trim = max(
        0.0,
        (
            _env_float("WEBRTC_AUDIO_MINIMUM_TRIM_SECONDS", 0.25)
            if minimum_trim_seconds is None
            else float(minimum_trim_seconds)
        ),
    )

    if not trim_enabled:
        samples = _decode_mono_pcm(source)
        if samples.size == 0:
            raise RuntimeError(f"WebRTC audio contains no decoded samples: {source}")
        original_duration = samples.size / float(ANALYSIS_SAMPLE_RATE)
        return WebRTCAudioTimeline(
            source_path=str(source),
            media_path=str(source),
            original_duration_seconds=original_duration,
            media_duration_seconds=original_duration,
            speech_start_seconds=0.0,
            speech_end_seconds=original_duration,
            trim_start_seconds=0.0,
            trim_end_seconds=original_duration,
            leading_silence_removed_seconds=0.0,
            trailing_silence_removed_seconds=0.0,
            threshold_db=threshold,
            normalized=False,
        )

    speech_start, speech_end, original_duration = detect_speech_bounds(
        source,
        threshold_db=threshold,
    )
    proposed_start = max(0.0, speech_start - leading_padding)
    proposed_end = min(original_duration, speech_end + trailing_padding)
    leading_removed = proposed_start
    trailing_removed = max(0.0, original_duration - proposed_end)

    trim_start = proposed_start if leading_removed >= minimum_trim else 0.0
    trim_end = proposed_end if trailing_removed >= minimum_trim else original_duration
    should_normalize = bool(
        trim_enabled
        and (trim_start > 0.0 or trim_end < original_duration)
        and trim_end > trim_start
    )
    if not should_normalize:
        return WebRTCAudioTimeline(
            source_path=str(source),
            media_path=str(source),
            original_duration_seconds=original_duration,
            media_duration_seconds=original_duration,
            speech_start_seconds=speech_start,
            speech_end_seconds=speech_end,
            trim_start_seconds=0.0,
            trim_end_seconds=original_duration,
            leading_silence_removed_seconds=0.0,
            trailing_silence_removed_seconds=0.0,
            threshold_db=threshold,
            normalized=False,
        )

    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_timeline.wav")
    )
    if destination.resolve() == source.resolve():
        raise ValueError("WebRTC audio timeline output must differ from its source")
    filter_value = (
        f"atrim=start={trim_start:.6f}:end={trim_end:.6f},"
        "asetpts=PTS-STARTPTS"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-af",
        filter_value,
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not normalize WebRTC audio timeline: {error}")

    media_duration = trim_end - trim_start
    return WebRTCAudioTimeline(
        source_path=str(source),
        media_path=str(destination),
        original_duration_seconds=original_duration,
        media_duration_seconds=media_duration,
        speech_start_seconds=speech_start,
        speech_end_seconds=speech_end,
        trim_start_seconds=trim_start,
        trim_end_seconds=trim_end,
        leading_silence_removed_seconds=trim_start,
        trailing_silence_removed_seconds=(original_duration - trim_end),
        threshold_db=threshold,
        normalized=True,
    )
