import asyncio
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCRtpSender

from scripts.pose_protocol import (
    POSE_IDS,
    POSE_ID_SET,
    POSE_SWITCH_MODE,
    REACTION_POSE,
    normalize_pose_plan,
    normalize_pose_sequence,
    normalize_pose_set,
    normalize_session_event,
)
from scripts.webrtc_pose_router import LivePoseVideoRouter
from scripts.webrtc_tracks import SwitchableVideoStreamTrack, SilenceAudioStreamTrack, VideoSyncClock


def build_rtc_configuration(
    stun_urls: Optional[List[str]] = None,
    turn_urls: Optional[List[str]] = None,
    turn_user: Optional[str] = None,
    turn_pass: Optional[str] = None,
) -> RTCConfiguration:
    ice_servers: List[RTCIceServer] = []

    if stun_urls:
        ice_servers.append(RTCIceServer(urls=stun_urls))
    if turn_urls:
        ice_servers.append(
            RTCIceServer(urls=turn_urls, username=turn_user, credential=turn_pass)
        )

    return RTCConfiguration(iceServers=ice_servers)


@dataclass
class WebRTCSession:
    session_id: str
    avatar_id: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    fps: int = 10
    playback_fps: int = 10
    batch_size: int = 2
    chunk_duration: int = 2
    pc: Optional[RTCPeerConnection] = None
    idle_track: Optional[SwitchableVideoStreamTrack] = None
    idle_sender: Optional[RTCRtpSender] = None
    audio_sender: Optional[RTCRtpSender] = None
    silence_audio_track: Optional[SilenceAudioStreamTrack] = None
    audio_player: Optional[object] = None  # MediaPlayer instance, kept generic to avoid import here
    active_stream: Optional[str] = None
    # ``active_stream`` is part of the public session status and is also
    # touched by the shared HLS/WebRTC scheduler.  Keep a private reservation
    # owner as the authoritative per-session turn guard so a scheduler failure
    # cannot briefly clear ``active_stream`` and admit an overlapping request.
    stream_owner: Optional[str] = field(default=None, repr=False)
    stream_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    ice_servers: List[dict] = field(default_factory=list)
    ice_transport_policy: str = "all"
    sync_clock: Optional[VideoSyncClock] = None
    live_timing: Optional[dict] = None
    webrtc_live_reveal_delay_seconds: float = 0.0
    idle_pose_id: str = "default"
    idle_video_path: Optional[str] = None
    pose_protocol_enabled: bool = False
    pose_set: Dict[str, Any] = field(default_factory=dict)
    pose_switch_mode: str = "immediate"
    pose_video_paths: Dict[str, str] = field(default_factory=dict)
    pose_queue: List[str] = field(default_factory=list)
    current_pose_id: str = "default"
    last_pose_seq: int = -1
    last_pose_event: Optional[str] = None
    active_turn_id: Optional[str] = None
    user_speaking: bool = False
    assistant_active: bool = False
    reaction_turn_ids: set[str] = field(default_factory=set)
    generation_avatar_id: Optional[str] = None
    pose_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    live_pose_id: str = "default"
    live_pose_router: Optional[LivePoseVideoRouter] = None
    prepared_pose_avatar_ids: Dict[str, str] = field(default_factory=dict)
    rendered_pose_id: Optional[str] = None
    rendered_pose_frame_count: int = 0
    rendered_pose_trace: List[Dict[str, Any]] = field(default_factory=list)
    active_pose_plan: Dict[str, Any] = field(default_factory=dict)
    compiled_pose_plan: Optional[Dict[str, Any]] = None
    rendered_pose_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        return (time.time() - self.last_activity) > ttl_seconds

    def touch(self) -> None:
        self.last_activity = time.time()

    def avatar_id_for_pose(self, pose_id: str) -> Optional[str]:
        """Resolve the stable MuseTalk cache ID assigned to a protocol pose."""
        poses = self.pose_set.get("poses") if self.pose_protocol_enabled else None
        entry = poses.get(pose_id) if isinstance(poses, Mapping) else None
        if isinstance(entry, Mapping):
            avatar_id = str(entry.get("avatar_id") or "").strip()
            if avatar_id:
                return avatar_id
        return self.avatar_id if not self.pose_protocol_enabled else None

    def video_path_for_pose(self, pose_id: str) -> Optional[str]:
        path = str(self.pose_video_paths.get(pose_id) or "").strip()
        if path:
            return path
        if pose_id == self.current_pose_id:
            return self.idle_video_path
        return None

    def sync_pose_track_state(self) -> None:
        track = self.idle_track
        if track is None or not hasattr(track, "get_pose_status"):
            return
        status = track.get_pose_status()
        self.current_pose_id = str(
            status.get("current_pose_id") or self.current_pose_id
        )
        self.idle_pose_id = self.current_pose_id
        self.idle_video_path = str(
            status.get("current_idle_video_path") or self.idle_video_path or ""
        ) or None
        self.pose_queue = list(status.get("pending_pose_ids") or [])

    def reset_rendered_pose_trace(self) -> None:
        with self.rendered_pose_lock:
            self.rendered_pose_id = None
            self.rendered_pose_frame_count = 0
            self.rendered_pose_trace = []

    def record_rendered_pose_batch(
        self,
        pose_ids,
        start_frame_index: int,
    ) -> None:
        normalized_pose_ids = [
            str(pose_id or "").strip().lower()
            for pose_id in pose_ids
        ]
        if not normalized_pose_ids:
            return
        with self.rendered_pose_lock:
            for offset, pose_id in enumerate(normalized_pose_ids):
                if not pose_id:
                    continue
                frame_index = int(start_frame_index) + offset
                if (
                    self.rendered_pose_trace
                    and self.rendered_pose_trace[-1]["pose_id"] == pose_id
                    and self.rendered_pose_trace[-1]["end_frame_index"] + 1
                    == frame_index
                ):
                    self.rendered_pose_trace[-1]["end_frame_index"] = frame_index
                    self.rendered_pose_trace[-1]["frame_count"] += 1
                else:
                    self.rendered_pose_trace.append(
                        {
                            "pose_id": pose_id,
                            "start_frame_index": frame_index,
                            "end_frame_index": frame_index,
                            "frame_count": 1,
                        }
                    )
                self.rendered_pose_id = pose_id
                self.rendered_pose_frame_count += 1

    def rendered_pose_status(self) -> dict:
        with self.rendered_pose_lock:
            return {
                "rendered_pose_id": self.rendered_pose_id,
                "rendered_pose_frame_count": self.rendered_pose_frame_count,
                "rendered_pose_trace": [
                    dict(entry) for entry in self.rendered_pose_trace
                ],
            }

    def pose_status(self) -> dict:
        self.sync_pose_track_state()
        rendered_status = self.rendered_pose_status()
        missing_video_paths = [
            pose_id for pose_id in POSE_IDS
            if self.pose_protocol_enabled and not self.video_path_for_pose(pose_id)
        ]
        return {
            "version": 1 if self.pose_protocol_enabled else None,
            "supported": self.pose_protocol_enabled,
            "pose_plan_version": 2 if self.pose_protocol_enabled else None,
            "pose_set_id": self.pose_set.get("pose_set_id") if self.pose_set else None,
            "switch_mode": self.pose_switch_mode,
            "default_pose_id": (
                self.pose_set.get("default_pose_id")
                if self.pose_protocol_enabled
                else None
            ),
            "current_pose_id": self.current_pose_id,
            "queued_pose_ids": list(self.pose_queue),
            "last_seq": self.last_pose_seq,
            "last_event": self.last_pose_event,
            "active_turn_id": self.active_turn_id,
            "user_speaking": self.user_speaking,
            "assistant_active": self.assistant_active,
            "generation_avatar_id": self.generation_avatar_id or self.avatar_id,
            "live_pose_id": self.live_pose_id,
            "missing_video_paths": missing_video_paths,
            "active_pose_plan": (
                dict(self.active_pose_plan)
                if self.active_pose_plan
                else None
            ),
            "compiled_pose_plan": (
                dict(self.compiled_pose_plan)
                if self.compiled_pose_plan
                else None
            ),
            **rendered_status,
        }


class WebRTCSessionManager:
    def __init__(
        self,
        session_ttl_seconds: int = 3600,
        rtc_config: Optional[RTCConfiguration] = None,
        ice_servers: Optional[List[dict]] = None,
        ice_transport_policy: str = "all",
    ):
        self.sessions: Dict[str, WebRTCSession] = {}
        self.session_ttl = session_ttl_seconds
        self.lock = asyncio.Lock()
        self.cleanup_task = None
        self.rtc_config = rtc_config
        self.ice_servers = ice_servers or []
        self.ice_transport_policy = ice_transport_policy
        self.deleting_sessions: set[str] = set()

    @staticmethod
    def _delete_detach_grace_seconds() -> float:
        try:
            return max(0.0, float(os.getenv("WEBRTC_DELETE_DETACH_GRACE_SECONDS", "0.2")))
        except ValueError:
            return 0.2

    @staticmethod
    def _safe_replace_sender_track(
        sender: Optional[RTCRtpSender],
        replacement,
        label: str,
        session_id: str,
    ) -> None:
        if sender is None:
            return
        try:
            sender.replaceTrack(replacement)
            replacement_label = "None" if replacement is None else replacement.__class__.__name__
            print(
                f"🧊 WebRTC delete detach {label} sender session_id={session_id} "
                f"replacement={replacement_label}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"⚠️ WebRTC delete could not detach {label} sender "
                f"session_id={session_id}: {exc}",
                flush=True,
            )

    @staticmethod
    def _safe_stop_track(track, label: str, session_id: str, stopped_ids: set[int]) -> None:
        if track is None:
            return
        track_id = id(track)
        if track_id in stopped_ids:
            return
        stopped_ids.add(track_id)
        if getattr(track, "readyState", None) == "ended":
            return
        try:
            print(f"🧊 WebRTC delete stop {label} track session_id={session_id}", flush=True)
            track.stop()
        except Exception as exc:
            print(
                f"⚠️ WebRTC delete could not stop {label} track "
                f"session_id={session_id}: {exc}",
                flush=True,
            )

    def start_cleanup(self) -> None:
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            async with self.lock:
                expired = [
                    sid for sid, session in self.sessions.items()
                    if session.is_expired(self.session_ttl)
                ]
            for sid in expired:
                await self.delete_session(sid)

    async def create_session(
        self,
        avatar_id: str,
        idle_video_path: str,
        user_id: Optional[str] = None,
        fps: int = 10,
        playback_fps: Optional[int] = None,
        batch_size: int = 2,
        chunk_duration: int = 2,
        idle_pose_id: str = "default",
        pose_set: Optional[dict] = None,
        pose_switch_mode: str = "immediate",
        pose_video_paths: Optional[Mapping[str, str]] = None,
        prepared_pose_avatar_ids: Optional[Mapping[str, str]] = None,
        live_pose_id: Optional[str] = None,
    ) -> WebRTCSession:
        if playback_fps is None:
            playback_fps = fps

        normalized_pose_set: Dict[str, Any] = {}
        normalized_pose_paths: Dict[str, str] = {}
        normalized_switch_mode = str(pose_switch_mode or "immediate").strip().lower()
        if pose_set:
            raw_poses = pose_set.get("poses") if isinstance(pose_set, Mapping) else None
            if isinstance(raw_poses, Mapping):
                for pose_id, raw_entry in raw_poses.items():
                    if not isinstance(raw_entry, Mapping):
                        continue
                    raw_path = str(
                        raw_entry.get("video_path")
                        or raw_entry.get("idle_video_path")
                        or ""
                    ).strip()
                    if raw_path:
                        normalized_pose_paths[str(pose_id)] = raw_path
            normalized_pose_set = normalize_pose_set(pose_set)
            if normalized_switch_mode != POSE_SWITCH_MODE:
                raise ValueError(
                    f"pose_switch_mode must be {POSE_SWITCH_MODE} when pose_set is provided"
                )
        elif normalized_switch_mode not in ("immediate", POSE_SWITCH_MODE):
            raise ValueError(
                f"pose_switch_mode must be immediate or {POSE_SWITCH_MODE}"
            )

        if pose_video_paths:
            normalized_pose_paths.update(
                {
                    str(pose_id): str(path)
                    for pose_id, path in pose_video_paths.items()
                    if str(path or "").strip()
                }
            )
        normalized_pose_paths.setdefault("default", str(idle_video_path))

        pose_protocol_enabled = bool(normalized_pose_set)
        initial_pose_id = (
            str(normalized_pose_set.get("default_pose_id") or "neutral_resting")
            if pose_protocol_enabled
            else str(idle_pose_id or "default")
        )
        if pose_protocol_enabled:
            normalized_pose_paths.setdefault(initial_pose_id, str(idle_video_path))
        generation_avatar_id = avatar_id
        if pose_protocol_enabled:
            generation_avatar_id = str(
                normalized_pose_set["poses"]["speaking_direct"]["avatar_id"]
            )
        idle_source_fps = float(fps)
        if pose_protocol_enabled:
            try:
                idle_source_fps = float(
                    normalized_pose_set["poses"][initial_pose_id].get("fps")
                    or fps
                )
            except (TypeError, ValueError):
                idle_source_fps = float(fps)
        resolved_prepared_pose_avatar_ids = {
            str(pose_id).strip().lower(): str(prepared_avatar_id).strip()
            for pose_id, prepared_avatar_id in (
                prepared_pose_avatar_ids or {}
            ).items()
            if str(pose_id).strip() and str(prepared_avatar_id).strip()
        }
        if pose_protocol_enabled:
            for pose_id, entry in normalized_pose_set["poses"].items():
                resolved_prepared_pose_avatar_ids.setdefault(
                    pose_id,
                    str(entry["avatar_id"]),
                )
        resolved_prepared_pose_avatar_ids.setdefault("default", avatar_id)
        initial_live_pose_id = str(
            live_pose_id or (
                initial_pose_id if pose_protocol_enabled else "default"
            )
        ).strip().lower()
        if initial_live_pose_id not in normalized_pose_paths:
            raise ValueError(
                f"Live pose video path is unavailable: {initial_live_pose_id}"
            )

        session_id = secrets.token_urlsafe(16)
        pc = RTCPeerConnection(self.rtc_config)
        sync_clock = VideoSyncClock(fps)
        idle_track = SwitchableVideoStreamTrack(
            idle_video_path,
            source_fps=float(fps),
            output_fps=playback_fps,
            sync_clock=sync_clock,
            idle_pose_id=initial_pose_id,
            idle_source_fps=idle_source_fps,
        )
        silence_audio = SilenceAudioStreamTrack(sync_clock=sync_clock)
        live_pose_router = LivePoseVideoRouter(
            normalized_pose_paths,
            prepared_pose_id="default",
            prepared_pose_ids=set(resolved_prepared_pose_avatar_ids),
            initial_pose_id=initial_live_pose_id,
        )

        session = WebRTCSession(
            session_id=session_id,
            avatar_id=avatar_id,
            user_id=user_id,
            fps=fps,
            playback_fps=playback_fps,
            batch_size=batch_size,
            chunk_duration=chunk_duration,
            pc=pc,
            idle_track=idle_track,
            idle_sender=None,
            active_stream=None,
            silence_audio_track=silence_audio,
            ice_servers=self.ice_servers,
            ice_transport_policy=self.ice_transport_policy,
            sync_clock=sync_clock,
            idle_pose_id=initial_pose_id,
            idle_video_path=idle_video_path,
            pose_protocol_enabled=pose_protocol_enabled,
            pose_set=normalized_pose_set,
            pose_switch_mode=(
                POSE_SWITCH_MODE if pose_protocol_enabled else normalized_switch_mode
            ),
            pose_video_paths=normalized_pose_paths,
            current_pose_id=initial_pose_id,
            generation_avatar_id=generation_avatar_id,
            live_pose_id=initial_live_pose_id,
            live_pose_router=live_pose_router,
            prepared_pose_avatar_ids=resolved_prepared_pose_avatar_ids,
        )

        def _sync_active_pose(pose_id: str, video_path: str) -> None:
            session.current_pose_id = pose_id
            session.idle_pose_id = pose_id
            session.idle_video_path = video_path
            if hasattr(idle_track, "get_pose_status"):
                session.pose_queue = list(
                    idle_track.get_pose_status().get("pending_pose_ids") or []
                )

        idle_track.set_idle_pose_change_callback(_sync_active_pose)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            print(f"🧊 WebRTC[{session_id}] connectionState={state}")
            if state == "closed":
                await self.delete_session(session_id)

        async with self.lock:
            self.sessions[session_id] = session
            total_sessions = len(self.sessions)

        print(
            f"🧊 WebRTC session created session_id={session_id} avatar_id={avatar_id} "
            f"user_id={user_id} idle_pose_id={initial_pose_id} "
            f"pose_protocol_enabled={pose_protocol_enabled} "
            f"generation_avatar_id={generation_avatar_id} "
            f"live_pose_id={initial_live_pose_id} "
            f"fps={fps} playback_fps={playback_fps} "
            f"batch_size={batch_size} chunk_duration={chunk_duration} "
            f"total_sessions={total_sessions}",
            flush=True,
        )
        return session

    async def get_session(self, session_id: str) -> Optional[WebRTCSession]:
        async with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.touch()
            return session

    async def reserve_stream(
        self,
        session: WebRTCSession,
        request_id: str,
    ) -> tuple[bool, Optional[str]]:
        """Atomically reserve one WebRTC stream turn for a session.

        The reservation is taken before pose/audio staging begins.  The
        private owner survives scheduler code that may clear the public
        ``active_stream`` field before its completion callback runs.
        """

        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("request_id is required")
        async with session.stream_lock:
            existing = session.stream_owner or session.active_stream
            if existing is not None and existing != normalized_request_id:
                return False, existing
            session.stream_owner = normalized_request_id
            session.active_stream = normalized_request_id
            session.touch()
            return True, None

    async def finish_reserved_stream(
        self,
        session: WebRTCSession,
        request_id: str,
        *,
        turn_id: Optional[str] = None,
        recover_pose: bool = False,
    ) -> dict:
        """Finish only the request that still owns the session reservation.

        Pose recovery happens while the stream reservation is held, preventing
        a following request from staging its pose and then being overwritten
        by a late completion callback from the previous turn.
        """

        normalized_request_id = str(request_id or "").strip()
        normalized_turn_id = str(turn_id or "").strip() or None
        async with session.stream_lock:
            if session.stream_owner != normalized_request_id:
                return {
                    "released": False,
                    "reason": "stream_owner_mismatch",
                    "request_id": normalized_request_id,
                    "active_stream": session.active_stream,
                    "stream_owner": session.stream_owner,
                }

            pose_result = None
            pose_recovery_skipped = False
            try:
                if recover_pose:
                    owns_pose_turn = (
                        normalized_turn_id is None
                        or session.active_turn_id == normalized_turn_id
                    )
                    if owns_pose_turn:
                        pose_result = await self.finish_assistant_turn(
                            session,
                            turn_id=normalized_turn_id,
                        )
                    else:
                        pose_recovery_skipped = True
            finally:
                # Never clear a newer public request if external scheduler code
                # changed it after this request was reserved.
                if session.active_stream in (None, normalized_request_id):
                    session.active_stream = None
                session.stream_owner = None

            return {
                "released": True,
                "request_id": normalized_request_id,
                "pose_recovery": pose_result,
                "pose_recovery_skipped": pose_recovery_skipped,
            }

    async def _resolve_runtime_session(self, session_or_id) -> Optional[WebRTCSession]:
        if isinstance(session_or_id, WebRTCSession):
            session_or_id.touch()
            return session_or_id
        return await self.get_session(str(session_or_id or ""))

    @staticmethod
    def _stale_pose_sequence_result(
        session: WebRTCSession,
        seq: int,
    ) -> dict:
        return {
            "accepted": False,
            "reason": "stale_seq",
            "seq": seq,
            "last_seq": session.last_pose_seq,
            "pose_status": session.pose_status(),
        }

    async def _queue_pose_locked(
        self,
        session: WebRTCSession,
        pose_id: str,
        *,
        reason: str,
        replace_pending: bool = False,
    ) -> dict:
        pose_id = str(pose_id or "").strip().lower()
        if pose_id not in POSE_ID_SET:
            raise ValueError(
                "pose_id must be one of: " + ", ".join(POSE_IDS)
            )
        if not session.pose_protocol_enabled:
            return {
                "accepted": False,
                "queued": False,
                "reason": "pose_protocol_disabled",
                "pose_id": pose_id,
                "pose_status": session.pose_status(),
            }
        if session.idle_track is None:
            return {
                "accepted": False,
                "queued": False,
                "reason": "video_track_unavailable",
                "pose_id": pose_id,
                "pose_status": session.pose_status(),
            }

        video_path = session.video_path_for_pose(pose_id)
        if not video_path:
            return {
                "accepted": False,
                "queued": False,
                "reason": "pose_video_path_unavailable",
                "pose_id": pose_id,
                "pose_status": session.pose_status(),
            }

        result = await session.idle_track.switch_idle_video(
            video_path,
            transition_seconds=0.0,
            pose_id=pose_id,
            effective=session.pose_switch_mode,
            reason=reason,
            replace_pending=replace_pending,
        )
        session.sync_pose_track_state()
        return {
            "accepted": True,
            "pose_id": pose_id,
            "reason": reason,
            **result,
            "pose_status": session.pose_status(),
        }

    async def queue_pose(
        self,
        session,
        pose_id: str,
        reason: str = "manual",
        *,
        replace_pending: bool = False,
    ) -> dict:
        """Queue a protocol pose without changing legacy idle-switch behavior."""
        resolved = await self._resolve_runtime_session(session)
        if resolved is None:
            return {
                "accepted": False,
                "queued": False,
                "reason": "session_not_found",
            }
        async with resolved.pose_lock:
            return await self._queue_pose_locked(
                resolved,
                pose_id,
                reason=reason,
                replace_pending=replace_pending,
            )

    async def handle_pose_event(
        self,
        session,
        event: Mapping[str, Any],
    ) -> dict:
        """Apply one ordered Lingua session event to the body-pose queue."""
        resolved = await self._resolve_runtime_session(session)
        if resolved is None:
            return {"accepted": False, "reason": "session_not_found"}
        normalized = normalize_session_event(event)
        seq = int(normalized["seq"])

        async with resolved.pose_lock:
            if not resolved.pose_protocol_enabled:
                return {
                    "accepted": False,
                    "reason": "pose_protocol_disabled",
                    "pose_status": resolved.pose_status(),
                }
            if seq <= resolved.last_pose_seq:
                return self._stale_pose_sequence_result(resolved, seq)

            event_name = normalized["event"]
            turn_id = str(normalized.get("turn_id") or "").strip()
            switch_result: Optional[dict] = None
            deduped = False

            if turn_id:
                resolved.active_turn_id = turn_id

            if event_name == "user_speech_started":
                resolved.user_speaking = True
                if not resolved.assistant_active:
                    switch_result = await self._queue_pose_locked(
                        resolved,
                        "active_listening",
                        reason=event_name,
                        replace_pending=True,
                    )
            elif event_name in ("user_speech_ended", "assistant_thinking"):
                if event_name == "user_speech_ended":
                    resolved.user_speaking = False
                if not resolved.assistant_active:
                    switch_result = await self._queue_pose_locked(
                        resolved,
                        "active_listening",
                        reason=event_name,
                    )
            elif event_name == "assistant_reaction_ready":
                resolved.assistant_active = True
                reaction_key = turn_id or f"seq:{seq}"
                if reaction_key in resolved.reaction_turn_ids:
                    deduped = True
                else:
                    resolved.reaction_turn_ids.add(reaction_key)
                    reaction_pose = REACTION_POSE.get(
                        normalized.get("reaction_intent", "none")
                    )
                    if reaction_pose:
                        switch_result = await self._queue_pose_locked(
                            resolved,
                            reaction_pose,
                            reason=event_name,
                            replace_pending=True,
                        )
            elif event_name == "assistant_turn_aborted":
                resolved.assistant_active = False
                resolved.user_speaking = False
                resolved.active_pose_plan = {}
                resolved.compiled_pose_plan = None
                switch_result = await self._queue_pose_locked(
                    resolved,
                    "neutral_resting",
                    reason=event_name,
                    replace_pending=True,
                )

            resolved.last_pose_seq = seq
            resolved.last_pose_event = event_name
            return {
                "accepted": True,
                "event": event_name,
                "seq": seq,
                "turn_id": turn_id or None,
                "deduped": deduped,
                "switch": switch_result,
                "pose_status": resolved.pose_status(),
            }

    async def queue_pose_sequence(
        self,
        session,
        pose_sequence,
        *,
        seq: int,
        turn_id: str,
        reaction_intent: str = "none",
    ) -> dict:
        """Queue the validated reaction → speaking → neutral assistant order."""
        resolved = await self._resolve_runtime_session(session)
        if resolved is None:
            return {"accepted": False, "reason": "session_not_found"}
        sequence = normalize_pose_sequence(
            pose_sequence,
            reaction_intent=reaction_intent,
        )
        seq = int(seq)

        async with resolved.pose_lock:
            if not resolved.pose_protocol_enabled:
                return {
                    "accepted": False,
                    "reason": "pose_protocol_disabled",
                    "pose_status": resolved.pose_status(),
                }
            if seq <= resolved.last_pose_seq:
                return self._stale_pose_sequence_result(resolved, seq)

            resolved.assistant_active = True
            resolved.active_pose_plan = {}
            resolved.compiled_pose_plan = None
            resolved.active_turn_id = str(turn_id or "").strip() or resolved.active_turn_id
            resolved.reset_rendered_pose_trace()
            reaction_key = str(turn_id or "").strip() or f"seq:{seq}"
            results = []
            if resolved.live_pose_router is not None:
                runtime_sequence = list(sequence)
                reaction_pose = (
                    runtime_sequence[0]
                    if runtime_sequence
                    and runtime_sequence[0] in REACTION_POSE.values()
                    else None
                )
                resolved.sync_pose_track_state()
                reaction_already_active = bool(
                    reaction_pose and resolved.current_pose_id == reaction_pose
                )
                if (
                    resolved.idle_track is not None
                    and hasattr(resolved.idle_track, "clear_pending_idle_switches")
                ):
                    resolved.idle_track.clear_pending_idle_switches()
                    resolved.sync_pose_track_state()
                if reaction_already_active:
                    runtime_sequence = runtime_sequence[1:]
                if (
                    reaction_pose
                    and reaction_key not in resolved.reaction_turn_ids
                ):
                    resolved.reaction_turn_ids.add(reaction_key)
                results = resolved.live_pose_router.queue_pose_sequence(
                    runtime_sequence,
                    resolved.fps,
                    hold_last_pose=True,
                )
                resolved.live_pose_id = runtime_sequence[0]
            else:
                for index, pose_id in enumerate(sequence):
                    if pose_id in REACTION_POSE.values():
                        if reaction_key in resolved.reaction_turn_ids:
                            continue
                        resolved.reaction_turn_ids.add(reaction_key)
                    results.append(
                        await self._queue_pose_locked(
                            resolved,
                            pose_id,
                            reason="assistant_pose_sequence",
                            replace_pending=index == 0,
                        )
                    )

            resolved.last_pose_seq = seq
            resolved.last_pose_event = "assistant_stream"
            return {
                "accepted": True,
                "seq": seq,
                "turn_id": resolved.active_turn_id,
                "pose_sequence": sequence,
                "switches": results,
                "generation_avatar_id": (
                    resolved.generation_avatar_id or resolved.avatar_id
                ),
                "pose_status": resolved.pose_status(),
            }

    async def stage_pose_plan(
        self,
        session,
        pose_plan,
        *,
        seq: int,
        turn_id: str,
    ) -> dict:
        """Accept one v2 semantic plan before audio generation starts."""

        resolved = await self._resolve_runtime_session(session)
        if resolved is None:
            return {"accepted": False, "reason": "session_not_found"}
        normalized_plan = normalize_pose_plan(pose_plan)
        if not normalized_plan:
            raise ValueError("pose_plan is required")
        seq = int(seq)

        async with resolved.pose_lock:
            if not resolved.pose_protocol_enabled:
                return {
                    "accepted": False,
                    "reason": "pose_protocol_disabled",
                    "pose_status": resolved.pose_status(),
                }
            if seq <= resolved.last_pose_seq:
                return self._stale_pose_sequence_result(resolved, seq)

            if (
                resolved.idle_track is not None
                and hasattr(resolved.idle_track, "clear_pending_idle_switches")
            ):
                resolved.idle_track.clear_pending_idle_switches()
                resolved.sync_pose_track_state()

            resolved.assistant_active = True
            resolved.active_turn_id = (
                str(turn_id or "").strip()
                or resolved.active_turn_id
            )
            resolved.active_pose_plan = normalized_plan
            resolved.compiled_pose_plan = None
            resolved.reset_rendered_pose_trace()
            resolved.last_pose_seq = seq
            resolved.last_pose_event = "assistant_pose_plan"
            first_pose_id = normalized_plan["segments"][0]["pose_id"]
            resolved.live_pose_id = first_pose_id
            return {
                "accepted": True,
                "seq": seq,
                "turn_id": resolved.active_turn_id,
                "pose_plan": normalized_plan,
                "generation_avatar_id": (
                    resolved.generation_avatar_id or resolved.avatar_id
                ),
                "pose_status": resolved.pose_status(),
            }

    async def finish_assistant_turn(
        self,
        session,
        *,
        turn_id: Optional[str] = None,
    ) -> dict:
        """Recover to neutral after audio playback completes."""
        resolved = await self._resolve_runtime_session(session)
        if resolved is None:
            return {"accepted": False, "reason": "session_not_found"}
        async with resolved.pose_lock:
            resolved.assistant_active = False
            resolved.active_pose_plan = {}
            if turn_id and resolved.active_turn_id == turn_id:
                resolved.active_turn_id = None
            if resolved.live_pose_router is not None:
                try:
                    resolved.live_pose_router.switch_pose("neutral_resting")
                    resolved.live_pose_id = "neutral_resting"
                except (KeyError, ValueError):
                    pass
            return await self._queue_pose_locked(
                resolved,
                "neutral_resting",
                reason="assistant_audio_complete",
                replace_pending=True,
            )

    async def delete_session(self, session_id: str) -> bool:
        async with self.lock:
            if session_id in self.deleting_sessions:
                print(f"🧊 WebRTC delete already in progress session_id={session_id}", flush=True)
                return False
            self.deleting_sessions.add(session_id)
            session = self.sessions.pop(session_id, None)
            remaining_sessions = len(self.sessions)

        if session is None:
            async with self.lock:
                self.deleting_sessions.discard(session_id)
            print(f"🧊 WebRTC delete skipped missing session_id={session_id}", flush=True)
            return False

        print(
            f"🧊 WebRTC delete start session_id={session_id} active_stream={session.active_stream} "
            f"remaining_sessions={remaining_sessions}",
            flush=True,
        )
        idle_track = session.idle_track
        audio_sender_track = session.audio_sender.track if session.audio_sender else None
        silence_audio_track = session.silence_audio_track
        audio_player = session.audio_player
        pc = session.pc
        async with session.stream_lock:
            session.active_stream = None
            session.stream_owner = None
        stopped_track_ids: set[int] = set()

        try:
            if session.sync_clock is not None:
                session.sync_clock.close()

            # Do not close PyAV-backed tracks while aiortc sender tasks may still
            # be inside recv()/encode. Detach first, let in-flight callbacks
            # unwind, then close the peer connection and release local tracks.
            self._safe_replace_sender_track(session.idle_sender, None, "video", session_id)
            self._safe_replace_sender_track(session.audio_sender, None, "audio", session_id)
            detach_grace = self._delete_detach_grace_seconds()
            if detach_grace > 0:
                await asyncio.sleep(detach_grace)

            if pc is not None:
                print(
                    f"🧊 WebRTC delete close peer connection session_id={session_id} "
                    f"state={pc.connectionState}",
                    flush=True,
                )
                try:
                    await asyncio.wait_for(pc.close(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(
                        f"⚠️ WebRTC delete peer connection close timed out "
                        f"session_id={session_id}",
                        flush=True,
                    )

            self._safe_stop_track(audio_player, "active audio", session_id, stopped_track_ids)
            self._safe_stop_track(audio_sender_track, "audio sender", session_id, stopped_track_ids)
            self._safe_stop_track(silence_audio_track, "silence audio", session_id, stopped_track_ids)
            self._safe_stop_track(idle_track, "video", session_id, stopped_track_ids)
            if session.live_pose_router is not None:
                session.live_pose_router.close()

            session.idle_track = None
            session.idle_sender = None
            session.audio_sender = None
            session.silence_audio_track = None
            session.audio_player = None
            session.sync_clock = None
            session.live_pose_router = None
            session.pc = None
            print(f"🧊 WebRTC delete done session_id={session_id}", flush=True)
            return True
        finally:
            async with self.lock:
                self.deleting_sessions.discard(session_id)

    def get_live_sessions(self) -> list[dict]:
        """Return only WebRTC sessions that are actively streaming"""
        now = time.time()
        return [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "avatar_id": s.avatar_id,
                "generation_avatar_id": s.generation_avatar_id or s.avatar_id,
                "active_stream": s.active_stream or s.stream_owner,
                "age_seconds": now - s.created_at,
                "idle_seconds": now - s.last_activity,
                "fps": s.fps,
                "playback_fps": s.playback_fps,
                "chunk_duration": s.chunk_duration,
                "batch_size": s.batch_size,
                "player_url": f"/webrtc/player/{s.session_id}",
                "ice_transport_policy": s.ice_transport_policy,
                "pose_protocol": s.pose_status(),
            }
            for s in self.sessions.values()
            if s.active_stream is not None or s.stream_owner is not None
        ]
