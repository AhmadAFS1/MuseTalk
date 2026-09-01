#!/usr/bin/env python3
"""Prepare and smoke-test the six-pose WebRTC protocol on a MuseTalk worker.

This script makes no TTS, animation, or other provider requests.  It uses
existing MP4 assets and an existing audio file supplied with ``--audio-file``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import mimetypes
import sys
import time
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

try:
    import aiohttp
except ModuleNotFoundError:  # Allows manifest/unit checks outside the GPU venv.
    aiohttp = None  # type: ignore[assignment]

try:
    from aiortc.contrib.media import MediaRecorder, MediaRelay
except ModuleNotFoundError:  # Allows manifest/unit checks outside the GPU venv.
    MediaRecorder = None  # type: ignore[assignment,misc]
    MediaRelay = None  # type: ignore[assignment,misc]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if aiohttp is not None:
    from load_test_webrtc import (  # noqa: E402
        AIORTC_IMPORT_ERROR,
        RTCConfiguration,
        RTCPeerConnection,
        SessionMetrics,
        build_rtc_configuration_from_payload,
        delete_webrtc_session,
        exchange_offer,
        is_live_ready,
        is_stream_complete,
        wait_for_peer_connection,
    )
else:  # pragma: no cover - exercised only in dependency-light tooling.
    AIORTC_IMPORT_ERROR = ModuleNotFoundError("aiohttp is not installed")
    RTCConfiguration = None
    RTCPeerConnection = None
    SessionMetrics = None
    build_rtc_configuration_from_payload = None
    delete_webrtc_session = None
    exchange_offer = None
    is_live_ready = None
    is_stream_complete = None
    wait_for_peer_connection = None


POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "nod_agree",
    "empathetic_head_tilt",
    "light_smile",
)
POSE_ID_SET = frozenset(POSE_IDS)
MVP_FOUR_POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "light_smile",
)
# Eulerian circuit over the complete directed four-pose graph. Every ordered
# non-self transition appears exactly once while starting/ending at neutral.
MVP_FOUR_ORDERED_TRANSITION_CIRCUIT = (
    "neutral_resting",
    "active_listening",
    "neutral_resting",
    "speaking_direct",
    "neutral_resting",
    "light_smile",
    "active_listening",
    "speaking_direct",
    "active_listening",
    "light_smile",
    "speaking_direct",
    "light_smile",
    "neutral_resting",
)
REACTION_POSE = {
    "none": None,
    "acknowledge": "nod_agree",
    "warmth": "light_smile",
    "empathy": "empathetic_head_tilt",
}
DEFAULT_MANIFEST = (
    ROOT
    / "configs"
    / "pose_test"
    / "sample_ai_human_ltx23_facetime_closeup_production_v1.json"
)
DEFAULT_ASSET_DIR = (
    ROOT
    / "assets"
    / "ltx23_pose_banks"
    / "sample_ai_human_facetime_closeup_production_v1"
    / "certified"
)
DEFAULT_AUDIO_FILE = ROOT / "data" / "audio" / "eng.wav"


class SmokeTestError(RuntimeError):
    pass


class SharedRecordingClock:
    """One receiver-side wall clock for every track written to the MP4."""

    def __init__(self) -> None:
        self.started_at: float | None = None

    def start(self) -> float:
        if self.started_at is None:
            self.started_at = time.monotonic()
        return self.started_at

    def elapsed(self) -> float:
        started_at = self.started_at
        if started_at is None:
            started_at = self.start()
        return max(0.0, time.monotonic() - started_at)


class WallClockAudioTrack:
    """Anchor audio once, then preserve receiver-visible RTP media spacing."""

    kind = "audio"

    def __init__(self, source: Any, clock: SharedRecordingClock) -> None:
        self.source = source
        self.clock = clock
        self._source_origin_seconds: float | None = None
        self._recording_origin_pts: int | None = None
        self._last_pts = -1
        self._last_source_seconds: float | None = None
        self._last_source_duration_seconds: float | None = None
        self._source_timestamp_anomalies = 0
        self._source_timestamp_missing = 0
        self._max_source_timestamp_error_seconds = 0.0
        self._source_timestamp_anomaly_details: list[dict[str, Any]] = []
        self._frames = 0

    async def recv(self) -> Any:
        frame = await self.source.recv()
        sample_rate = int(frame.sample_rate or 48000)
        samples = int(frame.samples or 0)
        source_seconds = (
            float(frame.pts * frame.time_base)
            if frame.pts is not None and frame.time_base is not None
            else None
        )
        if source_seconds is None:
            self._source_timestamp_missing += 1
        if self._recording_origin_pts is None:
            self._recording_origin_pts = int(round(self.clock.elapsed() * sample_rate))
            self._source_origin_seconds = source_seconds

        if (
            source_seconds is not None
            and self._last_source_seconds is not None
            and self._last_source_duration_seconds is not None
        ):
            source_step = source_seconds - self._last_source_seconds
            timestamp_error = abs(
                source_step - self._last_source_duration_seconds
            )
            if timestamp_error > max(2.0 / sample_rate, 1e-6):
                self._source_timestamp_anomalies += 1
                self._max_source_timestamp_error_seconds = max(
                    self._max_source_timestamp_error_seconds,
                    timestamp_error,
                )
                self._source_timestamp_anomaly_details.append(
                    {
                        "frame_index": self._frames,
                        "previous_source_seconds": self._last_source_seconds,
                        "current_source_seconds": source_seconds,
                        "source_step_seconds": source_step,
                        "expected_step_seconds": (
                            self._last_source_duration_seconds
                        ),
                        "timestamp_error_seconds": timestamp_error,
                        "excess_seconds": (
                            source_step - self._last_source_duration_seconds
                        ),
                        "sample_rate": sample_rate,
                    }
                )

        if source_seconds is not None and self._source_origin_seconds is not None:
            source_delta = max(0.0, source_seconds - self._source_origin_seconds)
            pts = self._recording_origin_pts + int(round(source_delta * sample_rate))
        else:
            pts = (
                self._recording_origin_pts
                if self._last_pts < 0
                else self._last_pts + max(1, samples)
            )

        pts = max(self._last_pts + 1, pts)
        self._last_pts = pts
        self._last_source_seconds = source_seconds
        self._last_source_duration_seconds = (
            samples / float(sample_rate) if samples > 0 else None
        )
        self._frames += 1
        frame.pts = pts
        frame.time_base = Fraction(1, sample_rate)
        return frame

    def get_stats(self) -> dict[str, Any]:
        return {
            "frames": self._frames,
            "source_timestamp_anomalies": self._source_timestamp_anomalies,
            "source_timestamp_missing": self._source_timestamp_missing,
            "max_source_timestamp_error_seconds": (
                self._max_source_timestamp_error_seconds
            ),
            "source_timestamp_anomaly_details": [
                dict(detail)
                for detail in self._source_timestamp_anomaly_details
            ],
            "timestamp_policy": "preserve_receiver_rtp_deltas",
        }


class WallClockVideoTrack:
    """Anchor video once while auditing every receiver-visible RTP delta."""

    kind = "video"

    def __init__(
        self,
        source: Any,
        clock: SharedRecordingClock,
        *,
        nominal_fps: float = 30.0,
    ) -> None:
        self.source = source
        self.clock = clock
        self._nominal_fps = float(nominal_fps)
        if self._nominal_fps <= 0:
            raise ValueError("nominal_fps must be positive")
        self._nominal_frame_seconds = 1.0 / self._nominal_fps
        self._fallback_frame_ticks = max(
            1,
            int(round(90_000.0 / self._nominal_fps)),
        )
        self._source_origin_seconds: float | None = None
        self._recording_origin_pts: int | None = None
        self._last_pts = -1
        self._last_source_seconds: float | None = None
        self._source_timestamp_anomalies = 0
        self._source_timestamp_missing = 0
        self._max_source_timestamp_error_seconds = 0.0
        self._source_timestamp_anomaly_details: list[dict[str, Any]] = []
        self._frames = 0

    async def recv(self) -> Any:
        frame = await self.source.recv()
        source_pts = frame.pts
        source_time_base = frame.time_base
        source_seconds = (
            float(source_pts * source_time_base)
            if source_pts is not None and source_time_base is not None
            else None
        )
        if source_seconds is None:
            self._source_timestamp_missing += 1
        if self._recording_origin_pts is None:
            self._recording_origin_pts = int(round(self.clock.elapsed() * 90_000))
            self._source_origin_seconds = source_seconds

        if source_seconds is not None and self._last_source_seconds is not None:
            source_step = source_seconds - self._last_source_seconds
            timestamp_error = abs(source_step - self._nominal_frame_seconds)
            source_tick_seconds = (
                abs(float(source_time_base))
                if source_time_base is not None
                else 0.0
            )
            timestamp_tolerance = max(2.0 * source_tick_seconds, 1e-6)
            if timestamp_error > timestamp_tolerance:
                step_intervals = source_step * self._nominal_fps
                nearest_intervals = int(round(step_intervals))
                integral_step = abs(step_intervals - nearest_intervals) <= (
                    timestamp_tolerance * self._nominal_fps
                )
                detail = {
                    "frame_index": self._frames,
                    "previous_source_seconds": self._last_source_seconds,
                    "current_source_seconds": source_seconds,
                    "source_step_seconds": source_step,
                    "expected_step_seconds": self._nominal_frame_seconds,
                    "timestamp_error_seconds": timestamp_error,
                    "step_frame_intervals": (
                        nearest_intervals if integral_step else None
                    ),
                    "excess_frame_intervals": (
                        nearest_intervals - 1 if integral_step else None
                    ),
                    "source_time_base_seconds": source_tick_seconds,
                }
                self._source_timestamp_anomaly_details.append(detail)
                self._source_timestamp_anomalies += 1
                self._max_source_timestamp_error_seconds = max(
                    self._max_source_timestamp_error_seconds,
                    timestamp_error,
                )

        # Anchor the first received frame to the shared recording epoch, then
        # preserve the sender's RTP spacing. MediaRelay can deliver several
        # queued frames in one event-loop turn; stamping every one from its
        # arrival wall time would collapse them onto the same encoder tick.
        if source_seconds is not None and self._source_origin_seconds is not None:
            source_delta = max(0.0, source_seconds - self._source_origin_seconds)
            pts = (
                self._recording_origin_pts
                + int(round(source_delta * 90_000))
            )
        else:
            pts = (
                self._recording_origin_pts
                if self._last_pts < 0
                else self._last_pts + self._fallback_frame_ticks
            )

        pts = max(self._last_pts + 1, pts)
        self._last_pts = pts
        self._last_source_seconds = source_seconds
        self._frames += 1
        frame.pts = pts
        frame.time_base = Fraction(1, 90_000)
        return frame

    def get_stats(self) -> dict[str, Any]:
        return {
            "frames": self._frames,
            "nominal_fps": self._nominal_fps,
            "nominal_frame_duration_seconds": self._nominal_frame_seconds,
            "source_timestamp_anomalies": self._source_timestamp_anomalies,
            "source_timestamp_missing": self._source_timestamp_missing,
            "max_source_timestamp_error_seconds": (
                self._max_source_timestamp_error_seconds
            ),
            "source_timestamp_anomaly_details": [
                dict(detail)
                for detail in self._source_timestamp_anomaly_details
            ],
            "timestamp_policy": "preserve_receiver_rtp_deltas",
        }


def _numeric_stat(
    mappings: list[dict[str, Any]],
    *keys: str,
) -> float | None:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    return None


def _case_timestamp_mappings(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every server stats scope that can carry shared RTP telemetry."""

    track_stats = case.get("final_track_stats") or {}
    if not isinstance(track_stats, dict):
        return []
    video_stats = track_stats.get("video") or {}
    audio_transport = track_stats.get("audio_transport") or {}
    mappings: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, dict) and value not in mappings:
            mappings.append(value)

    add(track_stats.get("sync_clock"))
    add(video_stats)
    add(video_stats.get("sync_clock") if isinstance(video_stats, dict) else None)
    add(audio_transport)
    if isinstance(audio_transport, dict):
        for source_key in ("last_source", "current_source"):
            source_stats = audio_transport.get(source_key)
            add(source_stats)
            if isinstance(source_stats, dict):
                add(source_stats.get("sync_clock"))
    return mappings


def _timestamp_anomaly_details(
    stats: dict[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    details = stats.get("source_timestamp_anomaly_details") or []
    if not isinstance(details, list) or not all(
        isinstance(detail, dict) for detail in details
    ):
        raise SmokeTestError(f"Receiver {label} timestamp details are invalid: {stats}")
    anomaly_count = int(stats.get("source_timestamp_anomalies") or 0)
    if anomaly_count != len(details):
        raise SmokeTestError(
            f"Receiver {label} timestamp anomaly count/detail mismatch: {stats}"
        )
    if int(stats.get("source_timestamp_missing") or 0):
        raise SmokeTestError(
            f"Receiver {label} frames were missing RTP timestamps: {stats}"
        )
    return [dict(detail) for detail in details]


def validate_recording_timestamp_proof(
    recording_track_stats: dict[str, Any],
    speaking_cases: list[dict[str, Any]],
    *,
    playback_fps: float,
) -> dict[str, Any]:
    """Validate receiver RTP cadence and the one declared turn-start rebase.

    The proof deliberately uses receiver source PTS plus server telemetry. It
    does not infer speaking/idle boundaries from pixels or waveform content.
    """

    if not recording_track_stats:
        return {
            "validated": False,
            "reason": "recording_disabled",
        }

    nominal_fps = float(playback_fps)
    if nominal_fps <= 0:
        raise SmokeTestError("playback_fps must be positive for RTP validation")
    nominal_frame_seconds = 1.0 / nominal_fps
    video_tolerance_seconds = max(2.0 / 90_000.0, 1e-6)

    video_stats = recording_track_stats.get("video") or {}
    audio_stats = recording_track_stats.get("audio") or {}
    if not isinstance(video_stats, dict) or not video_stats:
        raise SmokeTestError("Proof recording did not expose video RTP stats.")
    if not isinstance(audio_stats, dict) or not audio_stats:
        raise SmokeTestError("Proof recording did not expose audio RTP stats.")

    recorded_fps = _numeric_stat([video_stats], "nominal_fps")
    if recorded_fps is None or abs(recorded_fps - nominal_fps) > 1e-9:
        raise SmokeTestError(
            "Receiver video timestamp proof used the wrong nominal FPS: "
            f"expected={nominal_fps:g}, stats={video_stats}"
        )

    video_anomalies = _timestamp_anomaly_details(video_stats, label="video")
    audio_anomalies = _timestamp_anomaly_details(audio_stats, label="audio")
    declared_video_corrections: list[dict[str, Any]] = []
    declared_audio_corrections: list[dict[str, Any]] = []
    first_live_av_checks: list[dict[str, Any]] = []

    for case_index, case in enumerate(speaking_cases):
        mappings = _case_timestamp_mappings(case)
        track_stats = case.get("final_track_stats") or {}
        server_video_stats = (
            track_stats.get("video") or {}
            if isinstance(track_stats, dict)
            else {}
        )
        first_live_video_seconds = _numeric_stat(
            mappings,
            "first_live_video_rtp_seconds",
        )
        first_tts_seconds = _numeric_stat(
            mappings,
            "first_tts_transport_pts_seconds",
        )

        correction_frames_value = None
        if isinstance(server_video_stats, dict):
            for key in (
                "last_live_rtp_phase_correction_frames",
                "live_rtp_phase_correction_frames",
            ):
                if server_video_stats.get(key) is not None:
                    correction_frames_value = server_video_stats.get(key)
                    break
        video_correction_seconds = _numeric_stat(
            mappings,
            "video_rtp_phase_correction_seconds",
        )
        if correction_frames_value is None:
            correction_frames = (
                int(round(video_correction_seconds * nominal_fps))
                if video_correction_seconds is not None
                else 0
            )
        else:
            try:
                correction_frames = int(correction_frames_value)
            except (TypeError, ValueError) as exc:
                raise SmokeTestError(
                    "Server video phase-correction telemetry is invalid: "
                    f"{server_video_stats}"
                ) from exc
        if correction_frames < 0:
            raise SmokeTestError(
                "Server declared a negative video RTP phase correction: "
                f"{server_video_stats}"
            )
        if video_correction_seconds is not None and abs(
            video_correction_seconds
            - correction_frames * nominal_frame_seconds
        ) > video_tolerance_seconds:
            raise SmokeTestError(
                "Server video phase-correction seconds/frames disagree: "
                f"case={case_index}, frames={correction_frames}, "
                f"seconds={video_correction_seconds}"
            )
        if correction_frames:
            if first_live_video_seconds is None:
                raise SmokeTestError(
                    "Server declared a video RTP phase correction without the "
                    f"first-live RTP timestamp: case={case_index}"
                )
            declared_video_corrections.append(
                {
                    "case_index": case_index,
                    "first_live_video_rtp_seconds": first_live_video_seconds,
                    "correction_frames": correction_frames,
                    "correction_seconds": (
                        correction_frames * nominal_frame_seconds
                    ),
                }
            )

        audio_correction_seconds = _numeric_stat(
            mappings,
            "audio_rtp_phase_correction_seconds",
        ) or 0.0
        if audio_correction_seconds < 0:
            raise SmokeTestError(
                "Server declared a negative audio RTP phase correction: "
                f"case={case_index}, seconds={audio_correction_seconds}"
            )
        if audio_correction_seconds > 1e-6:
            if first_tts_seconds is None:
                raise SmokeTestError(
                    "Server declared an audio RTP phase correction without the "
                    f"actual first-TTS RTP timestamp: case={case_index}"
                )
            declared_audio_corrections.append(
                {
                    "case_index": case_index,
                    "first_tts_transport_pts_seconds": first_tts_seconds,
                    "correction_seconds": audio_correction_seconds,
                }
            )

        # Older workers do not expose actual first-TTS PTS. As soon as that
        # telemetry appears, it becomes a hard receiver-facing A/V assertion.
        if first_tts_seconds is not None:
            if first_live_video_seconds is None:
                raise SmokeTestError(
                    "Actual first-TTS RTP telemetry appeared without first-live "
                    f"video RTP telemetry: case={case_index}"
                )
            av_delta_seconds = abs(
                first_tts_seconds - first_live_video_seconds
            )
            if av_delta_seconds > nominal_frame_seconds + video_tolerance_seconds:
                raise SmokeTestError(
                    "First TTS/video RTP alignment exceeded one video frame: "
                    f"case={case_index}, audio={first_tts_seconds:.6f}, "
                    f"video={first_live_video_seconds:.6f}, "
                    f"delta={av_delta_seconds:.6f}, "
                    f"limit={nominal_frame_seconds:.6f}"
                )
            first_live_av_checks.append(
                {
                    "case_index": case_index,
                    "first_tts_transport_pts_seconds": first_tts_seconds,
                    "first_live_video_rtp_seconds": first_live_video_seconds,
                    "absolute_delta_seconds": av_delta_seconds,
                    "max_delta_seconds": nominal_frame_seconds,
                    "aligned": True,
                }
            )

    if len(video_anomalies) != len(declared_video_corrections):
        raise SmokeTestError(
            "Receiver video RTP timestamps contained an undeclared gap or "
            "missed a declared first-live phase correction: "
            f"anomalies={video_anomalies}, "
            f"declared={declared_video_corrections}"
        )
    unmatched_video = list(video_anomalies)
    for declaration in declared_video_corrections:
        expected_pts = declaration["first_live_video_rtp_seconds"]
        match = min(
            unmatched_video,
            key=lambda detail: abs(
                float(detail.get("current_source_seconds", float("inf")))
                - expected_pts
            ),
        )
        current_pts = _numeric_stat([match], "current_source_seconds")
        if current_pts is None or abs(current_pts - expected_pts) > video_tolerance_seconds:
            raise SmokeTestError(
                "Receiver video RTP gap was not located at the declared first-live "
                f"frame: anomaly={match}, declared={declaration}"
            )
        expected_excess_frames = declaration["correction_frames"]
        try:
            actual_excess_frames = int(match.get("excess_frame_intervals"))
        except (TypeError, ValueError) as exc:
            raise SmokeTestError(
                "Receiver video RTP gap was not an integral 30fps phase correction: "
                f"{match}"
            ) from exc
        if actual_excess_frames != expected_excess_frames:
            raise SmokeTestError(
                "Receiver video RTP gap size did not match server telemetry: "
                f"anomaly={match}, declared={declaration}"
            )
        unmatched_video.remove(match)

    if len(audio_anomalies) != len(declared_audio_corrections):
        raise SmokeTestError(
            "Receiver audio RTP timestamps contained an undeclared gap or "
            "missed a declared first-TTS phase correction: "
            f"anomalies={audio_anomalies}, "
            f"declared={declared_audio_corrections}"
        )
    unmatched_audio = list(audio_anomalies)
    for declaration in declared_audio_corrections:
        expected_pts = declaration["first_tts_transport_pts_seconds"]
        match = min(
            unmatched_audio,
            key=lambda detail: abs(
                float(detail.get("current_source_seconds", float("inf")))
                - expected_pts
            ),
        )
        current_pts = _numeric_stat([match], "current_source_seconds")
        sample_rate = int(match.get("sample_rate") or 48_000)
        audio_tolerance_seconds = max(2.0 / sample_rate, 1e-6)
        audio_packet_seconds = _numeric_stat(
            [match],
            "expected_step_seconds",
        ) or (960.0 / sample_rate)
        receiver_offset_seconds = (
            current_pts - expected_pts
            if current_pts is not None
            else float("inf")
        )
        # aiortc's receiver jitter buffer can begin the decoded stream on the
        # packet immediately after a forward RTP rebase.  Accept that single
        # normal 20 ms packet offset, but no larger receiver-visible skip.  The
        # correction-size assertion below still has to match server telemetry.
        if (
            current_pts is None
            or receiver_offset_seconds < -audio_tolerance_seconds
            or receiver_offset_seconds
            > audio_packet_seconds + audio_tolerance_seconds
        ):
            raise SmokeTestError(
                "Receiver audio RTP gap was not located at the declared "
                "first-TTS packet or the immediately following packet: "
                f"anomaly={match}, declared={declaration}"
            )
        excess_seconds = _numeric_stat([match], "excess_seconds")
        if excess_seconds is None or abs(
            excess_seconds - declaration["correction_seconds"]
        ) > audio_tolerance_seconds:
            raise SmokeTestError(
                "Receiver audio RTP gap size did not match server telemetry: "
                f"anomaly={match}, declared={declaration}"
            )
        declaration["receiver_first_observed_tts_pts_seconds"] = current_pts
        declaration["receiver_first_packet_offset_seconds"] = (
            receiver_offset_seconds
        )
        unmatched_audio.remove(match)

    return {
        "validated": True,
        "nominal_video_fps": nominal_fps,
        "nominal_video_frame_seconds": nominal_frame_seconds,
        "video_source_timestamp_anomalies": len(video_anomalies),
        "declared_video_phase_corrections": declared_video_corrections,
        "audio_source_timestamp_anomalies": len(audio_anomalies),
        "declared_audio_phase_corrections": declared_audio_corrections,
        "first_live_av_rtp_checks": first_live_av_checks,
        "actual_first_tts_rtp_available": bool(first_live_av_checks),
    }


def load_pose_set(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SmokeTestError(f"Pose manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SmokeTestError(f"Pose manifest is invalid JSON: {path}: {exc}") from exc

    if not isinstance(value, dict) or value.get("version") != 1:
        raise SmokeTestError("Pose manifest must be a version 1 object.")
    if value.get("default_pose_id") != "neutral_resting":
        raise SmokeTestError("Pose manifest default_pose_id must be neutral_resting.")
    if value.get("switch_mode") != "next_boundary":
        raise SmokeTestError("Pose manifest switch_mode must be next_boundary.")
    poses = value.get("poses")
    if not isinstance(poses, dict) or set(poses) != POSE_ID_SET:
        raise SmokeTestError(
            "Pose manifest must contain exactly: " + ", ".join(POSE_IDS)
        )
    for pose_id in POSE_IDS:
        entry = poses[pose_id]
        if not isinstance(entry, dict):
            raise SmokeTestError(f"Pose entry {pose_id} must be an object.")
        if not str(entry.get("avatar_id") or "").strip():
            raise SmokeTestError(f"Pose entry {pose_id} requires avatar_id.")
        asset_file = str(entry.get("asset_file") or f"{pose_id}.mp4").strip()
        if Path(asset_file).name != asset_file:
            raise SmokeTestError(
                f"Pose entry {pose_id}.asset_file must be a plain filename."
            )
        variants = entry.get("variants")
        if variants is not None:
            if pose_id != "speaking_direct" or not isinstance(variants, list):
                raise SmokeTestError(
                    "Only speaking_direct may define a variants array."
                )
            for variant in variants:
                if not isinstance(variant, dict):
                    raise SmokeTestError(
                        "speaking_direct variants must be objects."
                    )
                if not str(variant.get("variant_id") or "").strip():
                    raise SmokeTestError(
                        "speaking_direct variants require variant_id."
                    )
                if not str(variant.get("avatar_id") or "").strip():
                    raise SmokeTestError(
                        "speaking_direct variants require avatar_id."
                    )
                variant_asset = str(variant.get("asset_file") or "").strip()
                if not variant_asset or Path(variant_asset).name != variant_asset:
                    raise SmokeTestError(
                        "speaking_direct variants require a plain asset_file filename."
                    )
    return value


def worker_pose_manifest(pose_set: dict[str, Any]) -> dict[str, Any]:
    """Strip local paths and draft metadata from the worker session payload."""

    keep_fields = (
        "avatar_id",
        "role",
        "duration_seconds",
        "cycle_seconds",
        "fps",
        "frame_count",
    )
    manifest = {
        "version": 1,
        "pose_set_id": str(pose_set.get("pose_set_id") or ""),
        "default_pose_id": "neutral_resting",
        "switch_mode": "next_boundary",
        "poses": {
            pose_id: {
                key: pose_set["poses"][pose_id][key]
                for key in keep_fields
                if pose_set["poses"][pose_id].get(key) is not None
            }
            for pose_id in POSE_IDS
        },
    }
    for pose_id in POSE_IDS:
        entry = pose_set["poses"][pose_id]
        if entry.get("variants") is not None:
            manifest["poses"][pose_id]["variants"] = [
                {
                    "variant_id": str(variant["variant_id"]),
                    "avatar_id": str(variant["avatar_id"]),
                }
                for variant in entry["variants"]
            ]
            manifest["poses"][pose_id]["variant_policy"] = str(
                entry.get("variant_policy") or ""
            )
    return manifest


def pose_asset_entries(pose_set: dict[str, Any]) -> list[dict[str, str | None]]:
    """Return every unique physical cache source required by a pose set."""

    results: list[dict[str, str | None]] = []
    seen_avatar_ids: set[str] = set()
    for pose_id in POSE_IDS:
        entry = pose_set["poses"][pose_id]
        variants = entry.get("variants") or []
        if variants:
            for variant in variants:
                avatar_id = str(variant["avatar_id"])
                if avatar_id in seen_avatar_ids:
                    continue
                seen_avatar_ids.add(avatar_id)
                results.append(
                    {
                        "pose_id": pose_id,
                        "variant_id": str(variant["variant_id"]),
                        "avatar_id": avatar_id,
                        "asset_file": str(variant["asset_file"]),
                    }
                )
            continue
        avatar_id = str(entry["avatar_id"])
        if avatar_id in seen_avatar_ids:
            continue
        seen_avatar_ids.add(avatar_id)
        results.append(
            {
                "pose_id": pose_id,
                "variant_id": None,
                "avatar_id": avatar_id,
                "asset_file": str(entry.get("asset_file") or f"{pose_id}.mp4"),
            }
        )
    return results


def load_pose_plan_argument(raw_value: str | None) -> dict[str, Any] | None:
    """Load an optional inline v2 pose plan or an ``@file.json`` reference."""

    value = str(raw_value or "").strip()
    if not value:
        return None
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            value = path.read_text()
        except OSError as exc:
            raise SmokeTestError(f"Could not read pose-plan JSON: {path}: {exc}") from exc
    try:
        pose_plan = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SmokeTestError(f"pose-plan input is invalid JSON: {exc}") from exc
    if not isinstance(pose_plan, dict):
        raise SmokeTestError("pose-plan input must be a JSON object.")
    return pose_plan


def validate_compiled_pose_plan(
    requested_plan: dict[str, Any],
    compiled_plan: dict[str, Any],
    rendered_trace: list[dict[str, Any]],
    *,
    max_semantic_drift_seconds: float = 0.75,
) -> dict[str, Any]:
    """Assert v2 rendering and semantic cue timing against the audio clock."""

    if compiled_plan.get("status") != "compiled":
        raise SmokeTestError(
            "The v2 pose plan never reached compiled status: "
            f"{json.dumps(compiled_plan, sort_keys=True)[:1600]}"
        )

    requested_segments = [
        {
            "at_permille": int(segment["at_permille"]),
            "pose_id": str(segment["pose_id"]).strip().lower(),
        }
        for segment in requested_plan.get("segments") or []
    ]
    reported_requested = compiled_plan.get("requested_segments")
    if reported_requested != requested_segments:
        raise SmokeTestError(
            "Compiled pose-plan requested_segments do not match the submitted plan: "
            f"submitted={requested_segments}, reported={reported_requested}"
        )

    total_frames = int(compiled_plan.get("total_generation_frames") or 0)
    generation_fps = float(compiled_plan.get("generation_fps") or 0.0)
    effective_segments = list(compiled_plan.get("segments") or [])
    skipped_segments = list(compiled_plan.get("skipped_segments") or [])
    if total_frames <= 0 or generation_fps <= 0 or not effective_segments:
        raise SmokeTestError(
            "Compiled pose-plan telemetry is missing its frame clock or segments: "
            f"{json.dumps(compiled_plan, sort_keys=True)[:1600]}"
        )
    drift_limit_seconds = max(0.0, float(max_semantic_drift_seconds))
    drift_limit_frames = max(
        0,
        int(drift_limit_seconds * generation_fps),
    )

    accounted_requests: list[tuple[int, str]] = []
    alignment: list[dict[str, Any]] = []
    max_abs_semantic_drift_frames = 0
    cursor = 0
    for index, segment in enumerate(effective_segments):
        pose_id = str(segment.get("pose_id") or "")
        requested_at = int(segment.get("requested_at_permille"))
        requested_start = int(segment.get("requested_start_generation_frame"))
        effective_start = int(segment.get("effective_start_generation_frame"))
        effective_end = int(segment.get("effective_end_generation_frame"))
        snap_delay = int(segment.get("boundary_snap_delay_frames"))
        if effective_start != cursor or effective_end <= effective_start:
            raise SmokeTestError(
                "Compiled pose-plan segments are not positive and contiguous: "
                f"index={index}, cursor={cursor}, segment={segment}"
            )
        if effective_end > total_frames:
            raise SmokeTestError(
                "Compiled pose-plan segment exceeds the generated audio frame clock: "
                f"total={total_frames}, segment={segment}"
            )
        if snap_delay != effective_start - requested_start:
            raise SmokeTestError(
                "Compiled pose-plan boundary-snap telemetry is internally inconsistent: "
                f"segment={segment}"
            )
        semantic_drift = int(
            segment.get("semantic_drift_frames", snap_delay)
        )
        if semantic_drift != effective_start - requested_start:
            raise SmokeTestError(
                "Compiled pose-plan semantic-drift telemetry is internally "
                f"inconsistent: segment={segment}"
            )
        abs_semantic_drift = abs(semantic_drift)
        max_abs_semantic_drift_frames = max(
            max_abs_semantic_drift_frames,
            abs_semantic_drift,
        )
        if abs_semantic_drift > drift_limit_frames:
            raise SmokeTestError(
                "Pose cue exceeded the semantic timing limit: "
                f"pose={pose_id}, requested_frame={requested_start}, "
                f"effective_frame={effective_start}, "
                f"drift={semantic_drift / generation_fps:.3f}s, "
                f"limit={drift_limit_seconds:.3f}s"
            )
        switch_strategy = str(
            segment.get("switch_strategy") or "legacy_boundary_snap"
        )
        crossfade_frames = int(segment.get("crossfade_frames") or 0)
        if (
            switch_strategy == "requested_time_crossfade"
            and index > 0
            and crossfade_frames <= 0
        ):
            raise SmokeTestError(
                "A requested-time pose switch is missing its transition "
                f"crossfade: segment={segment}"
            )
        accounted_requests.append((requested_at, pose_id))
        alignment.append(
            {
                "pose_id": pose_id,
                "requested_at_permille": requested_at,
                "requested_start_generation_frame": requested_start,
                "effective_start_generation_frame": effective_start,
                "effective_end_generation_frame": effective_end,
                "boundary_snap_delay_frames": snap_delay,
                "boundary_snap_delay_seconds": round(
                    snap_delay / generation_fps,
                    3,
                ),
                "semantic_drift_frames": semantic_drift,
                "semantic_drift_seconds": round(
                    semantic_drift / generation_fps,
                    3,
                ),
                "switch_strategy": switch_strategy,
                "crossfade_frames": crossfade_frames,
            }
        )
        cursor = effective_end
    if cursor != total_frames:
        raise SmokeTestError(
            "Compiled pose-plan segments do not cover the complete generated frame clock: "
            f"covered={cursor}, total={total_frames}"
        )

    for skipped in skipped_segments:
        if str(skipped.get("pose_id") or "") == "speaking_direct":
            raise SmokeTestError(
                "The terminal speaking_direct cue was skipped: "
                f"{skipped}"
            )
        accounted_requests.append(
            (
                int(skipped.get("at_permille")),
                str(skipped.get("pose_id") or ""),
            )
        )
    expected_requests = [
        (segment["at_permille"], segment["pose_id"])
        for segment in requested_segments
    ]
    if sorted(accounted_requests) != sorted(expected_requests):
        raise SmokeTestError(
            "Compiled pose-plan telemetry did not account for every requested segment: "
            f"expected={expected_requests}, accounted={accounted_requests}"
        )

    expected_trace = [
        {
            "pose_id": str(segment["pose_id"]),
            "start_frame_index": int(segment["effective_start_generation_frame"]),
            "end_frame_index": int(segment["effective_end_generation_frame"]) - 1,
            "frame_count": (
                int(segment["effective_end_generation_frame"])
                - int(segment["effective_start_generation_frame"])
            ),
        }
        for segment in effective_segments
    ]
    normalized_trace = [
        {
            "pose_id": str(segment.get("pose_id") or ""),
            "start_frame_index": int(segment.get("start_frame_index")),
            "end_frame_index": int(segment.get("end_frame_index")),
            "frame_count": int(segment.get("frame_count")),
        }
        for segment in rendered_trace
    ]
    if normalized_trace != expected_trace:
        raise SmokeTestError(
            "Rendered pose trace does not match the compiled v2 timeline: "
            f"compiled={expected_trace}, rendered={normalized_trace}"
        )

    return {
        "validated": True,
        "requested_segment_count": len(requested_segments),
        "effective_segment_count": len(effective_segments),
        "skipped_segment_count": len(skipped_segments),
        "total_generation_frames": total_frames,
        "generation_fps": generation_fps,
        "max_semantic_drift_seconds": drift_limit_seconds,
        "max_abs_semantic_drift_frames": (
            max_abs_semantic_drift_frames
        ),
        "max_abs_semantic_drift_seconds": round(
            max_abs_semantic_drift_frames / generation_fps,
            3,
        ),
        "requested_vs_effective": alignment,
        "skipped_segments": skipped_segments,
        "rendered_trace_matches_compiled": True,
    }


async def response_body(response: aiohttp.ClientResponse) -> Any:
    text = await response.text()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def require_json(
    response: aiohttp.ClientResponse,
    *,
    action: str,
    accepted: tuple[int, ...] = (200,),
) -> dict[str, Any]:
    body = await response_body(response)
    if response.status not in accepted:
        raise SmokeTestError(
            f"{action} failed: HTTP {response.status}: "
            f"{json.dumps(body)[:800] if not isinstance(body, str) else body[:800]}"
        )
    if not isinstance(body, dict):
        raise SmokeTestError(f"{action} returned a non-JSON response: {str(body)[:500]}")
    return body


async def avatar_status(
    http: aiohttp.ClientSession,
    base_url: str,
    avatar_id: str,
) -> dict[str, Any]:
    async with http.get(f"{base_url}/avatars/{avatar_id}/cache/status") as response:
        return await require_json(response, action=f"status {avatar_id}")


async def prepare_avatar(
    http: aiohttp.ClientSession,
    *,
    base_url: str,
    avatar_id: str,
    video_path: Path,
    batch_size: int,
    force_recreate: bool,
) -> dict[str, Any]:
    if not video_path.is_file() or video_path.stat().st_size < 1024:
        raise SmokeTestError(f"Missing or invalid pose video: {video_path}")
    content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    params = {
        "avatar_id": avatar_id,
        "batch_size": batch_size,
        "bbox_shift": 0,
        "force_recreate": str(force_recreate).lower(),
    }
    with video_path.open("rb") as video:
        form = aiohttp.FormData()
        form.add_field(
            "video_file",
            video,
            filename=video_path.name,
            content_type=content_type,
        )
        async with http.post(
            f"{base_url}/avatars/prepare?{urlencode(params)}",
            data=form,
        ) as response:
            return await require_json(
                response,
                action=f"prepare {avatar_id}",
            )


async def warm_avatar(
    http: aiohttp.ClientSession,
    *,
    base_url: str,
    avatar_id: str,
    batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    params = {
        "batch_size": batch_size,
        "wait": "true",
        "timeout_seconds": timeout_seconds,
    }
    async with http.post(
        f"{base_url}/avatars/{avatar_id}/cache/warm?{urlencode(params)}"
    ) as response:
        body = await require_json(
            response,
            action=f"warm {avatar_id}",
            accepted=(200, 202),
        )
    if body.get("status") != "ready":
        raise SmokeTestError(
            f"Avatar {avatar_id} did not become ready: {json.dumps(body)[:800]}"
        )
    return body


async def evict_avatar_cache(
    http: aiohttp.ClientSession,
    base_url: str,
    avatar_id: str,
) -> dict[str, Any]:
    async with http.delete(f"{base_url}/avatars/{avatar_id}") as response:
        return await require_json(
            response,
            action=f"evict rebuilt cache {avatar_id}",
        )


async def ensure_six_avatars(
    http: aiohttp.ClientSession,
    *,
    base_url: str,
    pose_set: dict[str, Any],
    asset_dir: Path,
    prepare_missing: bool,
    force_recreate: bool,
    batch_size: int,
    warm_timeout: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for physical_entry in pose_asset_entries(pose_set):
        pose_id = str(physical_entry["pose_id"])
        variant_id = physical_entry["variant_id"]
        avatar_id = str(physical_entry["avatar_id"])
        label = f"{pose_id}/{variant_id}" if variant_id else pose_id
        status = await avatar_status(http, base_url, avatar_id)
        should_prepare = force_recreate or (
            not status.get("disk_prepared")
            and status.get("status") in {"missing", "failed"}
        )
        if should_prepare:
            if not prepare_missing and not force_recreate:
                raise SmokeTestError(
                    f"{avatar_id} is missing. Re-run with --prepare-missing."
                )
            asset_name = str(physical_entry["asset_file"])
            video_path = asset_dir / asset_name
            print(f"[prepare] {label}: {video_path}", flush=True)
            await prepare_avatar(
                http,
                base_url=base_url,
                avatar_id=avatar_id,
                video_path=video_path,
                batch_size=batch_size,
                force_recreate=force_recreate,
            )
            if force_recreate:
                # /avatars/prepare replaces disk materials but deliberately
                # does not invalidate an already loaded APIAvatar.
                await evict_avatar_cache(http, base_url, avatar_id)
            status = await avatar_status(http, base_url, avatar_id)

        if status.get("status") != "ready":
            print(f"[warm] {label}: {avatar_id}", flush=True)
            status = await warm_avatar(
                http,
                base_url=base_url,
                avatar_id=avatar_id,
                batch_size=batch_size,
                timeout_seconds=warm_timeout,
            )
        print(
            f"[ready] {label}: {avatar_id} "
            f"cached={status.get('cached')} disk={status.get('disk_prepared')}",
            flush=True,
        )
        results.append(
            {
                "pose_id": pose_id,
                "variant_id": variant_id,
                "avatar_id": avatar_id,
                "status": status.get("status"),
                "cached": bool(status.get("cached")),
                "disk_prepared": bool(status.get("disk_prepared")),
            }
        )
    # Recheck after the final warm so an LRU/memory eviction cannot masquerade
    # as a fully preloaded pose set.
    for result in results:
        status = await avatar_status(http, base_url, result["avatar_id"])
        result["status"] = status.get("status")
        result["cached"] = bool(status.get("cached"))
        result["disk_prepared"] = bool(status.get("disk_prepared"))
        if status.get("status") != "ready":
            raise SmokeTestError(
                "All six poses could not remain preloaded. "
                f"{result['avatar_id']} became {status.get('status')}; "
                "increase AVATAR_CACHE_MAX_MEMORY_MB / AVATAR_CACHE_MAX_AVATARS "
                "or reduce the test assets."
            )
    return results


async def consume_track(
    track,
    *,
    stop_event: asyncio.Event,
    counters: dict[str, int],
) -> None:
    try:
        while not stop_event.is_set():
            await track.recv()
            key = f"{getattr(track, 'kind', 'unknown')}_frames"
            counters[key] = counters.get(key, 0) + 1
    except asyncio.CancelledError:
        raise
    except Exception:
        if not stop_event.is_set():
            raise


async def post_event(
    http: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
    *,
    event: str,
    turn_id: str,
    seq: int,
    reaction_intent: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "turn_id": turn_id,
        "seq": seq,
    }
    if reaction_intent is not None:
        payload["reaction_intent"] = reaction_intent
    async with http.post(
        f"{base_url}/webrtc/sessions/{session_id}/events",
        json=payload,
    ) as response:
        body = await require_json(response, action=f"event {event}")
    print(f"[event {seq}] {event}: {json.dumps(body, sort_keys=True)}", flush=True)
    return body


async def queue_pose(
    http: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
    pose_id: str,
    *,
    replace_pending: bool = True,
) -> dict[str, Any]:
    async with http.post(
        f"{base_url}/webrtc/sessions/{session_id}/pose",
        json={
            "pose_id": pose_id,
            "effective": "next_boundary",
            "replace_pending": replace_pending,
        },
    ) as response:
        body = await require_json(response, action=f"queue pose {pose_id}")
    print(f"[pose] {pose_id}: {json.dumps(body, sort_keys=True)}", flush=True)
    return body


async def run_webrtc_smoke(
    http: aiohttp.ClientSession,
    *,
    base_url: str,
    pose_set: dict[str, Any],
    audio_file: Path,
    reaction_intent: str,
    speaking_case_count: int,
    pose_plan: dict[str, Any] | None,
    musetalk_fps: int,
    playback_fps: int,
    batch_size: int,
    completion_timeout: int,
    pose_recovery_timeout: int,
    event_delay: float,
    record_output: Path | None,
    showcase_six_poses: bool,
    showcase_mvp_four_exhaustive: bool,
    showcase_initial_neutral_seconds: float,
    showcase_timeout: int,
    max_pose_semantic_drift_ms: int,
    record_postroll_seconds: float,
) -> dict[str, Any]:
    if AIORTC_IMPORT_ERROR is not None or RTCPeerConnection is None:
        raise SmokeTestError(
            "aiortc is required for the end-to-end smoke test: "
            f"{AIORTC_IMPORT_ERROR}"
        )
    if record_output is not None and (
        MediaRecorder is None or MediaRelay is None
    ):
        raise SmokeTestError(
            "aiortc MediaRecorder and MediaRelay are required with --record-output."
        )
    if not audio_file.is_file() or audio_file.stat().st_size < 128:
        raise SmokeTestError(f"Sample audio file is missing or invalid: {audio_file}")

    manifest = worker_pose_manifest(pose_set)
    neutral_avatar_id = manifest["poses"]["neutral_resting"]["avatar_id"]
    params = {
        "avatar_id": neutral_avatar_id,
        "user_id": f"pose_smoke_{int(time.time())}",
        "fps": musetalk_fps,
        "playback_fps": playback_fps,
        "batch_size": batch_size,
        "chunk_duration": 2,
        "pose_switch_mode": "next_boundary",
        "pose_set": json.dumps(manifest, separators=(",", ":")),
    }
    async with http.post(
        f"{base_url}/webrtc/sessions/create?{urlencode(params)}"
    ) as response:
        created = await require_json(response, action="create pose WebRTC session")
    session_id = str(created.get("session_id") or "")
    if not session_id:
        raise SmokeTestError("Create response did not contain session_id.")

    stop_event = asyncio.Event()
    consumers: list[asyncio.Task] = []
    counters: dict[str, int] = {}
    pc = None
    recorder = None
    recorder_started = False
    relay = MediaRelay() if record_output is not None else None
    recording_clock = SharedRecordingClock()
    recording_track_wrappers: dict[str, Any] = {}
    recording_started_at: float | None = None
    turn_id_prefix = f"pose_smoke_{int(time.time() * 1000)}"
    final_status: dict[str, Any] = {}
    rendered_pose_trace: list[dict[str, Any]] = []
    speaking_cases: list[dict[str, Any]] = []
    showcase_pose_trace: list[dict[str, Any]] = []
    recording_timeline: list[dict[str, Any]] = []
    max_server_video_frames = 0
    max_server_audio_frames = 0
    metrics = SessionMetrics(session_id=session_id)
    started_at = time.monotonic()
    try:
        if record_output is not None:
            record_output = record_output.expanduser().resolve()
            record_output.parent.mkdir(parents=True, exist_ok=True)
            recorder = MediaRecorder(str(record_output))
        configuration: RTCConfiguration | None = build_rtc_configuration_from_payload(
            created.get("ice_servers") or []
        )
        pc = RTCPeerConnection(configuration=configuration)
        connected_event = asyncio.Event()
        failed_event = asyncio.Event()

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            if pc.connectionState == "connected":
                connected_event.set()
            elif pc.connectionState in {"failed", "closed"}:
                failed_event.set()

        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            if pc.iceConnectionState in {"connected", "completed"}:
                connected_event.set()
            elif pc.iceConnectionState in {"failed", "closed"}:
                failed_event.set()

        @pc.on("track")
        def on_track(track):
            consumer_track = relay.subscribe(track) if relay is not None else track
            consumers.append(
                asyncio.create_task(
                    consume_track(
                        consumer_track,
                        stop_event=stop_event,
                        counters=counters,
                    )
                )
            )
            if recorder is not None and relay is not None:
                recorder_track = relay.subscribe(track)
                if track.kind == "audio":
                    recorder_track = WallClockAudioTrack(
                        recorder_track,
                        recording_clock,
                    )
                elif track.kind == "video":
                    recorder_track = WallClockVideoTrack(
                        recorder_track,
                        recording_clock,
                        nominal_fps=playback_fps,
                    )
                recording_track_wrappers[track.kind] = recorder_track
                recorder.addTrack(recorder_track)

        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        await exchange_offer(
            http=http,
            base_url=base_url,
            session_id=session_id,
            pc=pc,
            metrics=metrics,
            ice_gather_timeout_s=10,
        )
        if recorder is not None:
            recording_started_at = recording_clock.start()
            await recorder.start()
            recorder_started = True
            print(f"[recording] {record_output}", flush=True)
        connected = await wait_for_peer_connection(
            pc=pc,
            connected_event=connected_event,
            failed_event=failed_event,
            timeout_s=60,
        )
        if not connected:
            raise SmokeTestError(
                f"WebRTC connection failed: pc={pc.connectionState} "
                f"ice={pc.iceConnectionState}"
            )
        print(f"[connected] session_id={session_id}", flush=True)
        showcase_clock_started_at = recording_started_at or time.monotonic()
        showcase_pose_trace.append(
            {
                "pose_id": "neutral_resting",
                "at_seconds": round(
                    time.monotonic() - showcase_clock_started_at,
                    3,
                ),
            }
        )
        recording_timeline.append(
            {
                "event": "non_speaking_pose",
                **showcase_pose_trace[-1],
            }
        )
        if showcase_initial_neutral_seconds > 0:
            print(
                "[showcase] holding initial neutral_resting for "
                f"{showcase_initial_neutral_seconds:.3f}s",
                flush=True,
            )
            await asyncio.sleep(showcase_initial_neutral_seconds)

        if showcase_mvp_four_exhaustive:
            # The first circuit node is already active when recording begins.
            # Queue the remaining nodes to exercise all 12 ordered transitions.
            cycle_pose_ids = list(MVP_FOUR_ORDERED_TRANSITION_CIRCUIT[1:])
            showcase_name = "four-pose exhaustive"
        else:
            # Prove that all six boundary switches can coexist in the server
            # queue. Neutral is last so the cycle returns to its starting state.
            cycle_pose_ids = [
                *(pose_id for pose_id in POSE_IDS if pose_id != "neutral_resting"),
                "neutral_resting",
            ]
            showcase_name = "six-pose"
        queued_pose_status = None
        for index, pose_id in enumerate(cycle_pose_ids):
            queued = await queue_pose(
                http,
                base_url,
                session_id,
                pose_id,
                replace_pending=index == 0,
            )
            queued_pose_status = queued.get("pose_status") or queued
        queued_cycle_state = []
        if isinstance(queued_pose_status, dict):
            current_cycle_pose = queued_pose_status.get("current_pose_id")
            if current_cycle_pose in cycle_pose_ids:
                queued_cycle_state.append(current_cycle_pose)
            queued_cycle_state.extend(
                queued_pose_status.get("queued_pose_ids") or []
            )
        queue_matches = queued_cycle_state == cycle_pose_ids
        if (
            not queue_matches
            and isinstance(queued_pose_status, dict)
            and queued_pose_status.get("current_pose_id") == "neutral_resting"
            and list(queued_pose_status.get("queued_pose_ids") or [])
            == cycle_pose_ids
        ):
            queue_matches = True
        if not queue_matches:
            raise SmokeTestError(
                f"The explicit pose queue did not retain the {showcase_name} "
                f"switches: {json.dumps(queued_pose_status)[:1200]}"
            )

        if showcase_six_poses or showcase_mvp_four_exhaustive:
            print(
                f"[showcase] waiting for the complete {showcase_name} cycle",
                flush=True,
            )
            expected_showcase_trace = ["neutral_resting", *cycle_pose_ids]
            showcase_deadline = time.monotonic() + showcase_timeout
            cycle_completed = False
            while time.monotonic() < showcase_deadline:
                await asyncio.sleep(0.2)
                async with http.get(
                    f"{base_url}/webrtc/sessions/{session_id}/status"
                ) as response:
                    status = await require_json(
                        response,
                        action=f"poll {showcase_name} showcase",
                    )
                current_pose_status = (
                    status.get("pose_protocol")
                    or status.get("pose_status")
                    or {}
                )
                current_pose_id = current_pose_status.get("current_pose_id")
                if (
                    current_pose_id in POSE_ID_SET
                    and current_pose_id
                    != showcase_pose_trace[-1]["pose_id"]
                ):
                    activation = {
                        "pose_id": current_pose_id,
                        "at_seconds": round(
                            time.monotonic() - showcase_clock_started_at,
                            3,
                        ),
                    }
                    showcase_pose_trace.append(activation)
                    recording_timeline.append(
                        {
                            "event": "non_speaking_pose",
                            **activation,
                        }
                    )
                    print(
                        f"[showcase] {current_pose_id} at "
                        f"{activation['at_seconds']:.3f}s",
                        flush=True,
                    )
                if (
                    len(showcase_pose_trace) > 1
                    and current_pose_id == "neutral_resting"
                    and not current_pose_status.get("queued_pose_ids")
                ):
                    cycle_completed = True
                    break
            observed_showcase_trace = [
                entry["pose_id"] for entry in showcase_pose_trace
            ]
            if not cycle_completed:
                raise SmokeTestError(
                    f"The {showcase_name} showcase did not complete in "
                    f"{showcase_timeout}s: {json.dumps(current_pose_status)[:1200]}"
                )
            if observed_showcase_trace != expected_showcase_trace:
                raise SmokeTestError(
                    f"The recorded {showcase_name} activation order was incomplete: "
                    f"expected {expected_showcase_trace}, "
                    f"observed {observed_showcase_trace}"
                )
            print("[showcase] complete", flush=True)

        def observe_server_status(
            status: dict[str, Any],
            case_trace: list[dict[str, Any]],
            case_max_frames: dict[str, int],
        ) -> None:
            nonlocal max_server_video_frames
            nonlocal max_server_audio_frames
            track_stats = status.get("track_stats") or {}
            video_stats = track_stats.get("video") or {}
            audio_stats = track_stats.get("audio") or {}
            video_frames = int(video_stats.get("frames_played") or 0)
            audio_frames = int(audio_stats.get("frames_sent") or 0)
            max_server_video_frames = max(
                max_server_video_frames,
                video_frames,
            )
            max_server_audio_frames = max(
                max_server_audio_frames,
                audio_frames,
            )
            case_max_frames["video"] = max(
                case_max_frames["video"],
                video_frames,
            )
            case_max_frames["audio"] = max(
                case_max_frames["audio"],
                audio_frames,
            )
            protocol_status = (
                status.get("pose_protocol")
                or status.get("pose_status")
                or {}
            )
            trace = protocol_status.get("rendered_pose_trace") or []
            if len(trace) >= len(case_trace):
                case_trace[:] = [dict(entry) for entry in trace]

        def pose_status(status: dict[str, Any]) -> dict[str, Any]:
            nested = status.get("pose_protocol") or status.get("pose_status")
            return nested if isinstance(nested, dict) else status

        async def run_speaking_case(
            case_reaction_intent: str,
            *,
            case_index: int,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal final_status
            turn_id = f"{turn_id_prefix}_{case_index + 1}"
            first_seq = case_index * 5 + 1

            # Exercise the deterministic event API. The first event replaces
            # any manual idle queue with the conversation state.
            await asyncio.sleep(event_delay)
            await post_event(
                http,
                base_url,
                session_id,
                event="user_speech_started",
                turn_id=turn_id,
                seq=first_seq,
            )
            await asyncio.sleep(event_delay)
            await post_event(
                http,
                base_url,
                session_id,
                event="user_speech_ended",
                turn_id=turn_id,
                seq=first_seq + 1,
            )
            await post_event(
                http,
                base_url,
                session_id,
                event="assistant_thinking",
                turn_id=turn_id,
                seq=first_seq + 2,
            )
            await asyncio.sleep(event_delay)
            await post_event(
                http,
                base_url,
                session_id,
                event="assistant_reaction_ready",
                turn_id=turn_id,
                seq=first_seq + 3,
                reaction_intent=case_reaction_intent,
            )

            reaction_pose = REACTION_POSE[case_reaction_intent]
            pose_sequence = [
                *(reaction_pose for _ in range(1) if reaction_pose),
                "speaking_direct",
                "neutral_resting",
            ]
            content_type = (
                mimetypes.guess_type(audio_file.name)[0]
                or "application/octet-stream"
            )
            pre_stream_counters = dict(counters)
            with audio_file.open("rb") as audio:
                form = aiohttp.FormData()
                form.add_field(
                    "audio_file",
                    audio,
                    filename=audio_file.name,
                    content_type=content_type,
                )
                form.add_field("reaction_intent", case_reaction_intent)
                form.add_field("pose_id", "speaking_direct")
                form.add_field(
                    "pose_sequence",
                    json.dumps(pose_sequence, separators=(",", ":")),
                )
                if pose_plan is not None:
                    form.add_field(
                        "pose_plan",
                        json.dumps(pose_plan, separators=(",", ":")),
                    )
                form.add_field("turn_id", turn_id)
                form.add_field("seq", str(first_seq + 4))
                form.add_field("effective", "next_boundary")
                form.add_field("mouth_mode", "lip_sync")
                form.add_field("audio_start", "immediate")
                async with http.post(
                    f"{base_url}/webrtc/sessions/{session_id}/stream",
                    data=form,
                ) as response:
                    accepted = await require_json(
                        response,
                        action=f"stream {case_reaction_intent} sample TTS",
                    )
            print(
                f"[stream {case_reaction_intent}] accepted: "
                f"{json.dumps(accepted, sort_keys=True)}",
                flush=True,
            )
            submitted_at = round(
                time.monotonic() - showcase_clock_started_at,
                3,
            )
            recording_timeline.append(
                {
                    "event": "speaking_stream_submitted",
                    "reaction_intent": case_reaction_intent,
                    "case_index": case_index,
                    "at_seconds": submitted_at,
                }
            )

            case_trace: list[dict[str, Any]] = []
            case_max_frames = {"video": 0, "audio": 0}
            compiled_pose_plan: dict[str, Any] = {}
            live_ready = False
            playout_released_at: float | None = None
            deadline = time.monotonic() + completion_timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                async with http.get(
                    f"{base_url}/webrtc/sessions/{session_id}/status"
                ) as response:
                    final_status = await require_json(
                        response,
                        action=f"poll {case_reaction_intent} TTS status",
                    )
                observe_server_status(
                    final_status,
                    case_trace,
                    case_max_frames,
                )
                if pose_plan is not None:
                    current_compiled_plan = (
                        pose_status(final_status).get("compiled_pose_plan")
                    )
                    if isinstance(current_compiled_plan, dict):
                        compiled_pose_plan = dict(current_compiled_plan)
                if not live_ready and is_live_ready(final_status):
                    live_ready = True
                    playout_released_at = round(
                        time.monotonic() - showcase_clock_started_at,
                        3,
                    )
                    print(
                        f"[status {case_reaction_intent}] live playout released",
                        flush=True,
                    )
                    recording_timeline.append(
                        {
                            "event": "speaking_playout_released",
                            "reaction_intent": case_reaction_intent,
                            "case_index": case_index,
                            "at_seconds": playout_released_at,
                        }
                    )
                if live_ready and is_stream_complete(final_status):
                    break
            if not live_ready:
                raise SmokeTestError(
                    f"{case_reaction_intent} stream never reached live-ready "
                    f"state: {json.dumps(final_status)[:1200]}"
                )
            if not is_stream_complete(final_status):
                raise SmokeTestError(
                    f"{case_reaction_intent} stream did not complete in "
                    f"{completion_timeout}s: {json.dumps(final_status)[:1200]}"
                )

            # Audio completion queues neutral at a clip boundary. Keep
            # consuming WebRTC until recovery actually activates.
            recovery_deadline = time.monotonic() + pose_recovery_timeout
            while time.monotonic() < recovery_deadline:
                current_pose_status = pose_status(final_status)
                if (
                    current_pose_status.get("current_pose_id")
                    == "neutral_resting"
                    and not current_pose_status.get("queued_pose_ids")
                ):
                    break
                await asyncio.sleep(0.5)
                async with http.get(
                    f"{base_url}/webrtc/sessions/{session_id}/status"
                ) as response:
                    final_status = await require_json(
                        response,
                        action="poll neutral pose recovery",
                    )
                observe_server_status(
                    final_status,
                    case_trace,
                    case_max_frames,
                )
                if pose_plan is not None:
                    current_compiled_plan = (
                        pose_status(final_status).get("compiled_pose_plan")
                    )
                    if isinstance(current_compiled_plan, dict):
                        compiled_pose_plan = dict(current_compiled_plan)
            current_pose_status = pose_status(final_status)
            if (
                current_pose_status.get("current_pose_id")
                != "neutral_resting"
                or current_pose_status.get("queued_pose_ids")
            ):
                raise SmokeTestError(
                    f"{case_reaction_intent} stream completed but the session "
                    "did not recover to neutral_resting in "
                    f"{pose_recovery_timeout}s: "
                    f"{json.dumps(current_pose_status)[:1200]}"
                )

            # Status can report the atomic neutral handoff before the returned
            # idle frames have crossed the receiver jitter buffer. Keep the
            # recorder open until a complete post-roll frame count has crossed
            # the receiver, so the proof MP4 visibly contains the completed
            # speaking-to-idle transition.
            if record_output is not None and record_postroll_seconds > 0:
                first_postroll_frame = counters.get("video_frames", 0)
                required_postroll_frames = max(
                    1,
                    int(record_postroll_seconds * playback_fps + 0.999999),
                )
                postroll_target = first_postroll_frame + required_postroll_frames
                postroll_deadline = (
                    time.monotonic() + record_postroll_seconds + 2.0
                )
                while (
                    counters.get("video_frames", 0) < postroll_target
                    and time.monotonic() < postroll_deadline
                ):
                    await asyncio.sleep(0.02)
                observed_postroll_frames = (
                    counters.get("video_frames", 0) - first_postroll_frame
                )
                if observed_postroll_frames < required_postroll_frames:
                    raise SmokeTestError(
                        "Neutral idle post-roll received only "
                        f"{observed_postroll_frames}/{required_postroll_frames} "
                        "required video frames."
                    )
                recording_timeline.append(
                    {
                        "event": "neutral_idle_postroll_complete",
                        "reaction_intent": case_reaction_intent,
                        "case_index": case_index,
                        "duration_seconds": record_postroll_seconds,
                        "video_frames": observed_postroll_frames,
                        "at_seconds": round(
                            time.monotonic() - showcase_clock_started_at,
                            3,
                        ),
                    }
                )

            post_stream_video_frames = (
                counters.get("video_frames", 0)
                - pre_stream_counters.get("video_frames", 0)
            )
            post_stream_audio_frames = (
                counters.get("audio_frames", 0)
                - pre_stream_counters.get("audio_frames", 0)
            )
            if post_stream_video_frames <= 0:
                raise SmokeTestError(
                    f"{case_reaction_intent} TTS received no WebRTC video frames."
                )
            if post_stream_audio_frames <= 0:
                raise SmokeTestError(
                    f"{case_reaction_intent} TTS received no WebRTC audio frames."
                )
            if case_max_frames["video"] <= 0:
                raise SmokeTestError(
                    f"{case_reaction_intent} TTS played no generated video frames."
                )
            if case_max_frames["audio"] <= 0:
                raise SmokeTestError(
                    f"{case_reaction_intent} TTS emitted no synchronized audio frames."
                )

            pose_plan_validation = None
            if pose_plan is not None:
                if accepted.get("pose_metadata_supported") is not True:
                    raise SmokeTestError(
                        "Worker accepted the stream without confirming "
                        "pose_metadata_supported=true."
                    )
                if accepted.get("pose_plan_supported") is not True:
                    raise SmokeTestError(
                        "Worker accepted the stream without confirming "
                        "pose_plan_supported=true."
                    )
                accepted_plan = accepted.get("pose_plan")
                if (
                    not isinstance(accepted_plan, dict)
                    or accepted_plan.get("accepted") is not True
                ):
                    raise SmokeTestError(
                        "Worker did not accept the submitted v2 pose plan: "
                        f"{json.dumps(accepted, sort_keys=True)[:1600]}"
                    )
                pose_plan_validation = validate_compiled_pose_plan(
                    pose_plan,
                    compiled_pose_plan,
                    case_trace,
                    max_semantic_drift_seconds=(
                        max_pose_semantic_drift_ms / 1000.0
                    ),
                )
            else:
                observed_rendered_poses = [
                    entry.get("pose_id")
                    for entry in case_trace
                    if entry.get("pose_id")
                ]
                expected_rendered_poses = [
                    *(reaction_pose for _ in range(1) if reaction_pose),
                    "speaking_direct",
                    "neutral_resting",
                ]
                missing_rendered_poses = [
                    pose_id
                    for pose_id in expected_rendered_poses
                    if pose_id not in observed_rendered_poses
                ]
                if missing_rendered_poses:
                    raise SmokeTestError(
                        f"The {case_reaction_intent} generated pose trace is "
                        f"incomplete; missing {missing_rendered_poses}: {case_trace}"
                    )

            selected_variant_render_keys = dict(
                (
                    accepted.get("pose_sequence")
                    or {}
                ).get("selected_pose_variant_render_keys")
                or {}
            )
            case_result = {
                "case_index": case_index,
                "turn_id": turn_id,
                "reaction_intent": case_reaction_intent,
                "generation_avatar_id": (
                    (accepted.get("pose_sequence") or {}).get(
                        "generation_avatar_id"
                    )
                ),
                "selected_pose_variant_render_keys": (
                    selected_variant_render_keys
                ),
                "pose_sequence": pose_sequence,
                "pose_plan": pose_plan,
                "compiled_pose_plan": compiled_pose_plan or None,
                "pose_plan_validation": pose_plan_validation,
                "submitted_at_seconds": submitted_at,
                "playout_released_at_seconds": playout_released_at,
                "rendered_pose_trace": case_trace,
                "video_frames_received": post_stream_video_frames,
                "audio_frames_received": post_stream_audio_frames,
                "server_generated_video_frames_played": case_max_frames["video"],
                "server_sample_audio_frames_sent": case_max_frames["audio"],
                # Preserve the server's receiver-visible media-horizon proof in
                # the capture log.  This makes it possible to audit that idle
                # was not restored until the final synchronized audio sample
                # had a corresponding live-video frame.
                "final_track_stats": final_status.get("track_stats") or {},
            }
            return case_result, current_pose_status

        speaking_reaction_intents = [reaction_intent] * speaking_case_count
        if showcase_mvp_four_exhaustive:
            # Empathy and warmth together cover the merged listener/empathy
            # pose and the smile pose while the MuseTalk mouth is live.
            for required_intent in ("empathy", "warmth"):
                if required_intent not in speaking_reaction_intents:
                    speaking_reaction_intents.append(required_intent)

        current_pose_status: dict[str, Any] = {}
        for case_index, case_reaction_intent in enumerate(
            speaking_reaction_intents
        ):
            case_result, current_pose_status = await run_speaking_case(
                case_reaction_intent,
                case_index=case_index,
            )
            speaking_cases.append(case_result)

        variant_rotation_validation = None
        direct_variants = (
            manifest["poses"]["speaking_direct"].get("variants") or []
        )
        if speaking_case_count > 1 and direct_variants:
            expected_variant_render_keys = [
                "speaking_direct__variant__"
                + str(direct_variants[index % len(direct_variants)]["variant_id"])
                for index in range(len(speaking_cases))
            ]
            observed_variant_render_keys = [
                str(
                    case["selected_pose_variant_render_keys"].get(
                        "speaking_direct"
                    )
                    or ""
                )
                for case in speaking_cases
            ]
            if observed_variant_render_keys != expected_variant_render_keys:
                raise SmokeTestError(
                    "Direct-speaking variant rotation did not match the "
                    "configured deterministic order: "
                    f"expected={expected_variant_render_keys}, "
                    f"observed={observed_variant_render_keys}"
                )
            for case, render_key in zip(
                speaking_cases,
                observed_variant_render_keys,
            ):
                if not any(
                    entry.get("pose_id") == "speaking_direct"
                    and entry.get("render_key") == render_key
                    for entry in case["rendered_pose_trace"]
                ):
                    raise SmokeTestError(
                        "Rendered direct-speaking trace omitted its selected "
                        f"physical variant {render_key}: "
                        f"{case['rendered_pose_trace']}"
                    )
            variant_rotation_validation = {
                "validated": True,
                "expected_render_keys": expected_variant_render_keys,
                "observed_render_keys": observed_variant_render_keys,
            }

        rendered_pose_trace = list(speaking_cases[0]["rendered_pose_trace"])
        pose_sequence = list(speaking_cases[0]["pose_sequence"])
        post_stream_video_frames = sum(
            int(case["video_frames_received"]) for case in speaking_cases
        )
        post_stream_audio_frames = sum(
            int(case["audio_frames_received"]) for case in speaking_cases
        )

        consumer_errors = []
        receiver_tracks_ended = 0
        for task in consumers:
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is None:
                    continue
                # aiortc raises MediaStreamError when a remote receiver track
                # ends. Once server completion, rendered-plan validation, and
                # neutral recovery have all passed above, that is a normal
                # terminal signal rather than an inference or transport error.
                if error.__class__.__name__ == "MediaStreamError":
                    receiver_tracks_ended += 1
                    continue
                consumer_errors.append(repr(error))
        if consumer_errors:
            raise SmokeTestError(
                f"WebRTC receiver task failed: {consumer_errors}"
            )

        recording_track_stats = {
            kind: wrapper.get_stats()
            for kind, wrapper in recording_track_wrappers.items()
            if hasattr(wrapper, "get_stats")
        }
        recording_timestamp_validation = validate_recording_timestamp_proof(
            recording_track_stats,
            speaking_cases,
            playback_fps=playback_fps,
        )

        result = {
            "ok": True,
            "session_id": session_id,
            "pose_set_id": manifest["pose_set_id"],
            "reaction_intent": reaction_intent,
            "pose_sequence": pose_sequence,
            "pose_plan": pose_plan,
            "compiled_pose_plans": [
                case["compiled_pose_plan"]
                for case in speaking_cases
                if case.get("compiled_pose_plan")
            ],
            "pose_plan_validations": [
                case["pose_plan_validation"]
                for case in speaking_cases
                if case.get("pose_plan_validation")
            ],
            "video_frames_received": counters.get("video_frames", 0),
            "audio_frames_received": counters.get("audio_frames", 0),
            "post_stream_video_frames_received": post_stream_video_frames,
            "post_stream_audio_frames_received": post_stream_audio_frames,
            "receiver_tracks_ended": receiver_tracks_ended,
            "server_generated_video_frames_played": max_server_video_frames,
            "server_sample_audio_frames_sent": max_server_audio_frames,
            "rendered_pose_trace": rendered_pose_trace,
            "speaking_cases": speaking_cases,
            "variant_rotation_validation": variant_rotation_validation,
            "speaking_pose_ids": sorted(
                {
                    str(entry.get("pose_id"))
                    for case in speaking_cases
                    for entry in case["rendered_pose_trace"]
                    if entry.get("pose_id")
                }
            ),
            "recording_timeline": recording_timeline,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "final_pose_id": current_pose_status.get("current_pose_id"),
            "final_stream_active": bool(final_status.get("active_stream")),
            "showcase_pose_trace": showcase_pose_trace,
            "non_speaking_ordered_transition_count": (
                12 if showcase_mvp_four_exhaustive else None
            ),
            "recording": str(record_output) if record_output is not None else None,
            "recording_track_stats": recording_track_stats,
            "recording_timestamp_validation": recording_timestamp_validation,
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    finally:
        if recorder is not None and recorder_started:
            with suppress(Exception):
                await recorder.stop()
        stop_event.set()
        for task in consumers:
            task.cancel()
        for task in consumers:
            with suppress(asyncio.CancelledError, Exception):
                await task
        if pc is not None:
            with suppress(Exception):
                await pc.close()
        with suppress(Exception):
            await delete_webrtc_session(http, base_url, session_id)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    pose_set = load_pose_set(args.manifest)
    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=30,
        sock_connect=30,
        sock_read=max(300, args.completion_timeout + 60),
    )
    async with aiohttp.ClientSession(timeout=timeout) as http:
        async with http.get(f"{base_url}/capabilities") as response:
            capabilities = await require_json(response, action="worker capabilities")
        supported = bool(
            isinstance(capabilities.get("features"), dict)
            and capabilities["features"].get("pose_sets_v1") is True
        )
        if not supported:
            raise SmokeTestError(
                "Worker does not advertise features.pose_sets_v1=true."
            )
        has_variants = any(
            bool(entry.get("variants"))
            for entry in pose_set["poses"].values()
        )
        if (
            has_variants
            and capabilities["features"].get("pose_variants_v1") is not True
        ):
            raise SmokeTestError(
                "Worker does not advertise features.pose_variants_v1=true."
            )
        if (
            args.pose_plan is not None
            and capabilities["features"].get("pose_plans_v2") is not True
        ):
            raise SmokeTestError(
                "Worker does not advertise features.pose_plans_v2=true."
            )

        avatars = await ensure_six_avatars(
            http,
            base_url=base_url,
            pose_set=pose_set,
            asset_dir=args.asset_dir,
            prepare_missing=args.prepare_missing,
            force_recreate=args.force_recreate,
            batch_size=args.batch_size,
            warm_timeout=args.warm_timeout,
        )
        if args.prepare_only:
            result = {
                "ok": True,
                "mode": "prepare_only",
                "pose_set_id": pose_set["pose_set_id"],
                "avatars": avatars,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return result
        smoke = await run_webrtc_smoke(
            http,
            base_url=base_url,
            pose_set=pose_set,
            audio_file=args.audio_file,
            reaction_intent=args.reaction_intent,
            speaking_case_count=args.speaking_case_count,
            pose_plan=args.pose_plan,
            musetalk_fps=args.musetalk_fps,
            playback_fps=args.playback_fps,
            batch_size=args.batch_size,
            completion_timeout=args.completion_timeout,
            pose_recovery_timeout=args.pose_recovery_timeout,
            event_delay=args.event_delay,
            record_output=args.record_output,
            showcase_six_poses=args.showcase_six_poses,
            showcase_mvp_four_exhaustive=args.showcase_mvp_four_exhaustive,
            showcase_initial_neutral_seconds=(
                args.showcase_initial_neutral_seconds
            ),
            showcase_timeout=args.showcase_timeout,
            max_pose_semantic_drift_ms=args.max_pose_semantic_drift_ms,
            record_postroll_seconds=args.record_postroll_seconds,
        )
        for avatar in avatars:
            status = await avatar_status(
                http,
                base_url,
                avatar["avatar_id"],
            )
            avatar["status_after_stream"] = status.get("status")
            avatar["cached_after_stream"] = bool(status.get("cached"))
            if status.get("status") != "ready":
                raise SmokeTestError(
                    "A pose cache was evicted or stripped during the stream: "
                    f"{avatar['avatar_id']} became {status.get('status')}."
                )
        smoke["avatars"] = avatars
        return smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a six-pose avatar bank and run one direct "
            "MuseTalk WebRTC conversation using an existing audio file."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--audio-file", type=Path, default=DEFAULT_AUDIO_FILE)
    parser.add_argument(
        "--prepare-missing",
        action="store_true",
        help="Upload and prepare pose MP4s only when the stable avatar ID is missing.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Rebuild every pose cache from the MP4s (slow and destructive to existing caches).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/warm all six caches and stop before opening a WebRTC session.",
    )
    parser.add_argument(
        "--reaction-intent",
        choices=tuple(REACTION_POSE),
        default="warmth",
    )
    parser.add_argument(
        "--speaking-case-count",
        type=int,
        default=1,
        help=(
            "Run this many assistant turns in one WebRTC session. Values above "
            "one validate deterministic direct-speaking variant rotation."
        ),
    )
    parser.add_argument(
        "--pose-plan-json",
        help=(
            "Optional v2 pose-plan JSON object. Prefix a path with @ to load "
            "the JSON from a file."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help=(
            "Legacy shorthand used for both MuseTalk generation and playback "
            "unless the corresponding split FPS option is provided."
        ),
    )
    parser.add_argument(
        "--musetalk-fps",
        type=int,
        help="MuseTalk lip-sync generation FPS (for example, 15).",
    )
    parser.add_argument(
        "--playback-fps",
        type=int,
        help="WebRTC output playback FPS (for example, 30).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warm-timeout", type=int, default=900)
    parser.add_argument("--completion-timeout", type=int, default=300)
    parser.add_argument("--pose-recovery-timeout", type=int, default=15)
    parser.add_argument("--event-delay", type=float, default=0.75)
    parser.add_argument(
        "--max-pose-semantic-drift-ms",
        type=int,
        default=750,
        help=(
            "Fail a v2 pose-plan smoke test when any rendered pose cue is "
            "more than this many milliseconds from its requested audio time."
        ),
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        help="Record the received WebRTC audio and video tracks to this MP4.",
    )
    parser.add_argument(
        "--record-postroll-seconds",
        type=float,
        default=0.75,
        help=(
            "Keep a proof recording open this long after neutral recovery so "
            "the speaking-to-idle handoff is visible (default: 0.75)."
        ),
    )
    parser.add_argument(
        "--showcase-six-poses",
        action="store_true",
        help=(
            "Wait for all six queued idle poses to play completely before "
            "starting the test TTS stream."
        ),
    )
    parser.add_argument(
        "--showcase-mvp-four-exhaustive",
        action="store_true",
        help=(
            "Record all 12 ordered non-speaking transitions among neutral, "
            "active-listening/empathetic, speaking-direct, and light-smile."
        ),
    )
    parser.add_argument(
        "--showcase-timeout",
        type=int,
        default=90,
        help="Maximum seconds to wait for the complete six-pose idle cycle.",
    )
    parser.add_argument(
        "--showcase-initial-neutral-seconds",
        type=float,
        default=0.0,
        help=(
            "Hold neutral_resting for this many seconds after recording starts "
            "and before the showcase pose queue is submitted."
        ),
    )
    args = parser.parse_args()
    try:
        args.pose_plan = load_pose_plan_argument(args.pose_plan_json)
    except SmokeTestError as exc:
        parser.error(str(exc))
    if args.musetalk_fps is None:
        args.musetalk_fps = args.fps
    if args.playback_fps is None:
        args.playback_fps = args.fps
    if args.force_recreate:
        args.prepare_missing = True
    if args.showcase_six_poses and args.showcase_mvp_four_exhaustive:
        parser.error(
            "--showcase-six-poses and --showcase-mvp-four-exhaustive "
            "cannot be used together."
        )
    if (
        args.musetalk_fps < 1
        or args.playback_fps < 1
        or args.batch_size < 1
        or args.warm_timeout < 1
        or args.completion_timeout < 1
        or args.pose_recovery_timeout < 1
        or args.showcase_timeout < 1
        or args.max_pose_semantic_drift_ms < 0
        or args.record_postroll_seconds < 0
        or args.showcase_initial_neutral_seconds < 0
        or not 1 <= args.speaking_case_count <= 4
    ):
        parser.error(
            "FPS, batch size, and timeout values must be positive; semantic "
            "drift must be non-negative; speaking-case-count must be 1-4."
        )
    return args


def main() -> int:
    args = parse_args()
    print(
        "NOTICE: this smoke test provides technical media evidence; final "
        "motion approval remains a human visual review.",
        file=sys.stderr,
    )
    if aiohttp is None:
        print(
            "POSE WEBRTC SMOKE FAILED: aiohttp is not installed in this Python "
            "environment. Run inside the MuseTalk server virtualenv.",
            file=sys.stderr,
        )
        return 1
    try:
        asyncio.run(run(args))
    except (SmokeTestError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        print(f"POSE WEBRTC SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
