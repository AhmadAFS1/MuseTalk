#!/usr/bin/env python3
"""Prepare and smoke-test the six-pose WebRTC protocol on a MuseTalk worker.

This script makes no TTS, animation, or other provider requests.  It uses
existing MP4 assets and an existing audio file supplied with ``--audio-file``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
REACTION_POSE = {
    "none": None,
    "acknowledge": "nod_agree",
    "warmth": "light_smile",
    "empathy": "empathetic_head_tilt",
}
DEFAULT_MANIFEST = (
    ROOT / "configs" / "pose_test" / "indian_tutor_essential_six_v1.json"
)
DEFAULT_ASSET_DIR = (
    ROOT / "data" / "video" / "segmind_indian_essential_six_v1"
)
DEFAULT_AUDIO_FILE = ROOT / "data" / "audio" / "eng.wav"


class SmokeTestError(RuntimeError):
    pass


class WallClockAudioTrack:
    """Preserve real-time gaps when a WebRTC sender replaces its audio source."""

    kind = "audio"

    def __init__(self, source: Any) -> None:
        self.source = source
        self._started_at: float | None = None
        self._next_pts = 0

    async def recv(self) -> Any:
        frame = await self.source.recv()
        now = time.monotonic()
        sample_rate = int(frame.sample_rate or 48000)
        samples = int(frame.samples or 0)
        if self._started_at is None:
            self._started_at = now
        wall_clock_pts = int(round((now - self._started_at) * sample_rate))
        # Normal packets stay sample-contiguous. If the sender pauses while
        # preparing TTS, advance to wall time so the recording retains silence
        # instead of collapsing the gap or overlapping reset RTP timestamps.
        if wall_clock_pts > self._next_pts + max(samples * 2, sample_rate // 10):
            self._next_pts = wall_clock_pts
        frame.pts = self._next_pts
        frame.time_base = Fraction(1, sample_rate)
        self._next_pts += samples
        return frame


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
    return {
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
    for pose_id in POSE_IDS:
        entry = pose_set["poses"][pose_id]
        avatar_id = str(entry["avatar_id"])
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
            asset_name = str(entry.get("asset_file") or f"{pose_id}.mp4")
            video_path = asset_dir / asset_name
            print(f"[prepare] {pose_id}: {video_path}", flush=True)
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
            print(f"[warm] {pose_id}: {avatar_id}", flush=True)
            status = await warm_avatar(
                http,
                base_url=base_url,
                avatar_id=avatar_id,
                batch_size=batch_size,
                timeout_seconds=warm_timeout,
            )
        print(
            f"[ready] {pose_id}: {avatar_id} "
            f"cached={status.get('cached')} disk={status.get('disk_prepared')}",
            flush=True,
        )
        results.append(
            {
                "pose_id": pose_id,
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
    fps: int,
    batch_size: int,
    completion_timeout: int,
    pose_recovery_timeout: int,
    event_delay: float,
    record_output: Path | None,
    showcase_six_poses: bool,
    showcase_timeout: int,
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
        "fps": fps,
        "playback_fps": fps,
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
    recording_started_at: float | None = None
    turn_id = f"pose_smoke_{int(time.time() * 1000)}_1"
    final_status: dict[str, Any] = {}
    rendered_pose_trace: list[dict[str, Any]] = []
    showcase_pose_trace: list[dict[str, Any]] = []
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
                    recorder_track = WallClockAudioTrack(recorder_track)
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
            await recorder.start()
            recorder_started = True
            recording_started_at = time.monotonic()
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

        # Prove that all six boundary switches can coexist in the server queue.
        # Neutral is last so the visual cycle returns to its starting state.
        cycle_pose_ids = [
            *(pose_id for pose_id in POSE_IDS if pose_id != "neutral_resting"),
            "neutral_resting",
        ]
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
        if queued_cycle_state != cycle_pose_ids:
            raise SmokeTestError(
                "The explicit pose queue did not retain all six boundary "
                f"switches: {json.dumps(queued_pose_status)[:1200]}"
            )

        if showcase_six_poses:
            print("[showcase] waiting for the complete six-pose cycle", flush=True)
            expected_showcase_trace = [
                "neutral_resting",
                *cycle_pose_ids,
            ]
            showcase_deadline = time.monotonic() + showcase_timeout
            cycle_completed = False
            while time.monotonic() < showcase_deadline:
                await asyncio.sleep(0.2)
                async with http.get(
                    f"{base_url}/webrtc/sessions/{session_id}/status"
                ) as response:
                    status = await require_json(
                        response,
                        action="poll six-pose showcase",
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
                    "The six-pose showcase did not complete in "
                    f"{showcase_timeout}s: {json.dumps(current_pose_status)[:1200]}"
                )
            if observed_showcase_trace != expected_showcase_trace:
                raise SmokeTestError(
                    "The recorded six-pose activation order was incomplete: "
                    f"expected {expected_showcase_trace}, "
                    f"observed {observed_showcase_trace}"
                )
            print("[showcase] complete", flush=True)

        # Exercise the deterministic event API. The first event deliberately
        # replaces the manual cycle queue with the conversation state.
        await asyncio.sleep(event_delay)
        await post_event(
            http,
            base_url,
            session_id,
            event="user_speech_started",
            turn_id=turn_id,
            seq=1,
        )
        await asyncio.sleep(event_delay)
        await post_event(
            http,
            base_url,
            session_id,
            event="user_speech_ended",
            turn_id=turn_id,
            seq=2,
        )
        await post_event(
            http,
            base_url,
            session_id,
            event="assistant_thinking",
            turn_id=turn_id,
            seq=3,
        )
        await asyncio.sleep(event_delay)
        await post_event(
            http,
            base_url,
            session_id,
            event="assistant_reaction_ready",
            turn_id=turn_id,
            seq=4,
            reaction_intent=reaction_intent,
        )

        reaction_pose = REACTION_POSE[reaction_intent]
        pose_sequence = [
            *(reaction_pose for _ in range(1) if reaction_pose),
            "speaking_direct",
            "neutral_resting",
        ]
        content_type = (
            mimetypes.guess_type(audio_file.name)[0] or "application/octet-stream"
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
            form.add_field("reaction_intent", reaction_intent)
            form.add_field("pose_id", "speaking_direct")
            form.add_field(
                "pose_sequence",
                json.dumps(pose_sequence, separators=(",", ":")),
            )
            form.add_field("turn_id", turn_id)
            form.add_field("seq", "5")
            form.add_field("effective", "next_boundary")
            form.add_field("mouth_mode", "lip_sync")
            form.add_field("audio_start", "immediate")
            async with http.post(
                f"{base_url}/webrtc/sessions/{session_id}/stream",
                data=form,
            ) as response:
                accepted = await require_json(response, action="stream sample TTS")
        print(f"[stream] accepted: {json.dumps(accepted, sort_keys=True)}", flush=True)

        def observe_server_status(status: dict[str, Any]) -> None:
            nonlocal max_server_video_frames
            nonlocal max_server_audio_frames
            nonlocal rendered_pose_trace
            track_stats = status.get("track_stats") or {}
            video_stats = track_stats.get("video") or {}
            audio_stats = track_stats.get("audio") or {}
            max_server_video_frames = max(
                max_server_video_frames,
                int(video_stats.get("frames_played") or 0),
            )
            max_server_audio_frames = max(
                max_server_audio_frames,
                int(audio_stats.get("frames_sent") or 0),
            )
            protocol_status = (
                status.get("pose_protocol")
                or status.get("pose_status")
                or {}
            )
            trace = protocol_status.get("rendered_pose_trace") or []
            if len(trace) >= len(rendered_pose_trace):
                rendered_pose_trace = [dict(entry) for entry in trace]

        live_ready = False
        deadline = time.monotonic() + completion_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            async with http.get(
                f"{base_url}/webrtc/sessions/{session_id}/status"
            ) as response:
                final_status = await require_json(response, action="poll status")
            observe_server_status(final_status)
            if not live_ready and is_live_ready(final_status):
                live_ready = True
                print("[status] live playout released", flush=True)
            if live_ready and is_stream_complete(final_status):
                break
        if not live_ready:
            raise SmokeTestError(
                f"Stream never reached live-ready state: {json.dumps(final_status)[:1200]}"
            )
        if not is_stream_complete(final_status):
            raise SmokeTestError(
                f"Stream did not complete in {completion_timeout}s: "
                f"{json.dumps(final_status)[:1200]}"
            )

        def pose_status(status: dict[str, Any]) -> dict[str, Any]:
            nested = status.get("pose_protocol") or status.get("pose_status")
            return nested if isinstance(nested, dict) else status

        # Audio completion queues neutral at a clip boundary. Keep consuming
        # WebRTC long enough to prove that recovery actually activates.
        recovery_deadline = time.monotonic() + pose_recovery_timeout
        while time.monotonic() < recovery_deadline:
            current_pose_status = pose_status(final_status)
            if (
                current_pose_status.get("current_pose_id") == "neutral_resting"
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
            observe_server_status(final_status)
        current_pose_status = pose_status(final_status)
        if (
            current_pose_status.get("current_pose_id") != "neutral_resting"
            or current_pose_status.get("queued_pose_ids")
        ):
            raise SmokeTestError(
                "Stream completed but the session did not recover to "
                f"neutral_resting in {pose_recovery_timeout}s: "
                f"{json.dumps(current_pose_status)[:1200]}"
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
                "WebRTC received no video frames after the TTS stream started."
            )
        if post_stream_audio_frames <= 0:
            raise SmokeTestError(
                "WebRTC received no audio frames after the TTS stream started."
            )
        if max_server_video_frames <= 0:
            raise SmokeTestError(
                "The server never reported playing a generated video frame."
            )
        if max_server_audio_frames <= 0:
            raise SmokeTestError(
                "The synchronized sample-audio track never emitted a frame."
            )

        observed_rendered_poses = [
            entry.get("pose_id")
            for entry in rendered_pose_trace
            if entry.get("pose_id")
        ]
        expected_rendered_poses = [
            *(reaction_pose for _ in range(1) if reaction_pose),
            "speaking_direct",
        ]
        missing_rendered_poses = [
            pose_id
            for pose_id in expected_rendered_poses
            if pose_id not in observed_rendered_poses
        ]
        if missing_rendered_poses:
            raise SmokeTestError(
                "The generated pose trace is incomplete; missing "
                f"{missing_rendered_poses}: {rendered_pose_trace}"
            )

        consumer_errors = []
        for task in consumers:
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is not None:
                    consumer_errors.append(repr(error))
        if consumer_errors:
            raise SmokeTestError(
                f"WebRTC receiver task failed: {consumer_errors}"
            )

        result = {
            "ok": True,
            "session_id": session_id,
            "pose_set_id": manifest["pose_set_id"],
            "reaction_intent": reaction_intent,
            "pose_sequence": pose_sequence,
            "video_frames_received": counters.get("video_frames", 0),
            "audio_frames_received": counters.get("audio_frames", 0),
            "post_stream_video_frames_received": post_stream_video_frames,
            "post_stream_audio_frames_received": post_stream_audio_frames,
            "server_generated_video_frames_played": max_server_video_frames,
            "server_sample_audio_frames_sent": max_server_audio_frames,
            "rendered_pose_trace": rendered_pose_trace,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "final_pose_id": current_pose_status.get("current_pose_id"),
            "final_stream_active": bool(final_status.get("active_stream")),
            "showcase_pose_trace": showcase_pose_trace,
            "recording": str(record_output) if record_output is not None else None,
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
            fps=args.fps,
            batch_size=args.batch_size,
            completion_timeout=args.completion_timeout,
            pose_recovery_timeout=args.pose_recovery_timeout,
            event_delay=args.event_delay,
            record_output=args.record_output,
            showcase_six_poses=args.showcase_six_poses,
            showcase_timeout=args.showcase_timeout,
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
            "Prepare the six Indian tutor pose avatars and run one direct "
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
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warm-timeout", type=int, default=900)
    parser.add_argument("--completion-timeout", type=int, default=300)
    parser.add_argument("--pose-recovery-timeout", type=int, default=15)
    parser.add_argument("--event-delay", type=float, default=0.75)
    parser.add_argument(
        "--record-output",
        type=Path,
        help="Record the received WebRTC audio and video tracks to this MP4.",
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
        "--showcase-timeout",
        type=int,
        default=90,
        help="Maximum seconds to wait for the complete six-pose idle cycle.",
    )
    args = parser.parse_args()
    if args.force_recreate:
        args.prepare_missing = True
    if (
        args.fps < 1
        or args.batch_size < 1
        or args.warm_timeout < 1
        or args.completion_timeout < 1
        or args.pose_recovery_timeout < 1
        or args.showcase_timeout < 1
    ):
        parser.error("FPS, batch size, and timeout values must be positive.")
    return args


def main() -> int:
    args = parse_args()
    print(
        "WARNING: this manifest is draft/test-only and switch_safe=false. "
        "The smoke test is for visual evaluation, not production approval.",
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
