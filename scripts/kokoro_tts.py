"""Lazy, process-local Kokoro TTS service for WebRTC test audio.

Kokoro is intentionally loaded on first use so ordinary MuseTalk workers do
not pay model startup time or reserve extra memory unless the test endpoint is
used. Calls are serialized because one cached pipeline/model is shared by the
FastAPI process.
"""

from __future__ import annotations

import importlib.util
import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


KOKORO_SAMPLE_RATE = 24_000
KOKORO_LANGUAGE_CODES = frozenset({"a", "b"})


@dataclass(frozen=True)
class KokoroSynthesis:
    wav_bytes: bytes
    sample_rate: int
    audio_seconds: float
    synthesis_seconds: float
    real_time_factor: float
    cold_start: bool
    voice: str
    language_code: str
    device: str


class KokoroTTSService:
    """Cache official ``kokoro.KPipeline`` instances by language and device."""

    def __init__(self) -> None:
        self._pipelines: dict[tuple[str, str], Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("kokoro") is not None

    @staticmethod
    def default_device() -> str:
        # Keep TTS off the latency-critical MuseTalk CUDA stream by default.
        # Operators can explicitly opt into CUDA after measuring headroom.
        return str(os.getenv("KOKORO_TTS_DEVICE", "cpu") or "cpu").strip()

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = [
                {"language_code": language_code, "device": device}
                for language_code, device in self._pipelines
            ]
        return {
            "available": self.available(),
            "model": "hexgrad/Kokoro-82M",
            "sample_rate": KOKORO_SAMPLE_RATE,
            "default_device": self.default_device(),
            "supported_language_codes": sorted(KOKORO_LANGUAGE_CODES),
            "loaded_pipelines": loaded,
        }

    @staticmethod
    def _validate(
        text: str,
        voice: str,
        language_code: str,
        speed: float,
    ) -> tuple[str, str, str, float]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("text is required")
        if len(normalized_text) > 2_000:
            raise ValueError("text must be 2000 characters or fewer")

        normalized_voice = str(voice or "af_heart").strip().lower()
        if not normalized_voice or len(normalized_voice) > 120:
            raise ValueError("voice is invalid")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_,:."
            for character in normalized_voice
        ):
            raise ValueError("voice contains unsupported characters")

        normalized_language = str(language_code or "a").strip().lower()
        if normalized_language not in KOKORO_LANGUAGE_CODES:
            raise ValueError("language_code must be 'a' (US) or 'b' (UK)")

        try:
            normalized_speed = float(speed)
        except (TypeError, ValueError) as exc:
            raise ValueError("speed must be a number") from exc
        if (
            normalized_speed != normalized_speed
            or normalized_speed in (float("inf"), float("-inf"))
            or not 0.5 <= normalized_speed <= 2.0
        ):
            raise ValueError("speed must be between 0.5 and 2.0")

        return (
            normalized_text,
            normalized_voice,
            normalized_language,
            normalized_speed,
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        language_code: str = "a",
        speed: float = 1.0,
        device: str | None = None,
    ) -> KokoroSynthesis:
        if not self.available():
            raise RuntimeError(
                "Kokoro is not installed; install kokoro==0.9.4 in the server environment"
            )

        (
            normalized_text,
            normalized_voice,
            normalized_language,
            normalized_speed,
        ) = self._validate(text, voice, language_code, speed)
        resolved_device = str(device or self.default_device()).strip() or "cpu"
        key = (normalized_language, resolved_device)

        # Model initialization, voice downloads, and forward passes all stay
        # inside one lock. This prevents duplicate cold loads and avoids using
        # the same PyTorch model concurrently from multiple request threads.
        with self._lock:
            started_at = time.monotonic()
            cold_start = key not in self._pipelines
            if cold_start:
                from kokoro import KPipeline

                self._pipelines[key] = KPipeline(
                    lang_code=normalized_language,
                    device=resolved_device,
                )

            pipeline = self._pipelines[key]
            chunks = []
            for result in pipeline(
                normalized_text,
                voice=normalized_voice,
                speed=normalized_speed,
            ):
                audio = getattr(result, "audio", None)
                if audio is None and isinstance(result, (tuple, list)) and result:
                    audio = result[-1]
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach()
                if hasattr(audio, "cpu"):
                    audio = audio.cpu()
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                chunks.append(audio)

            if not chunks:
                raise RuntimeError("Kokoro produced no audio")

            import numpy as np
            import soundfile as sf

            samples = np.concatenate(
                [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks]
            )
            output = io.BytesIO()
            sf.write(output, samples, KOKORO_SAMPLE_RATE, format="WAV", subtype="PCM_16")
            synthesis_seconds = time.monotonic() - started_at

        audio_seconds = len(samples) / float(KOKORO_SAMPLE_RATE)
        real_time_factor = (
            synthesis_seconds / audio_seconds if audio_seconds > 0 else 0.0
        )
        return KokoroSynthesis(
            wav_bytes=output.getvalue(),
            sample_rate=KOKORO_SAMPLE_RATE,
            audio_seconds=audio_seconds,
            synthesis_seconds=synthesis_seconds,
            real_time_factor=real_time_factor,
            cold_start=cold_start,
            voice=normalized_voice,
            language_code=normalized_language,
            device=resolved_device,
        )


kokoro_tts_service = KokoroTTSService()
