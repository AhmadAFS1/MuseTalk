import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCRtpSender

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
    ice_servers: List[dict] = field(default_factory=list)
    ice_transport_policy: str = "all"
    sync_clock: Optional[VideoSyncClock] = None

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        return (time.time() - self.last_activity) > ttl_seconds

    def touch(self) -> None:
        self.last_activity = time.time()


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
    ) -> WebRTCSession:
        if playback_fps is None:
            playback_fps = fps
        session_id = secrets.token_urlsafe(16)
        pc = RTCPeerConnection(self.rtc_config)
        sync_clock = VideoSyncClock(fps)
        idle_track = SwitchableVideoStreamTrack(
            idle_video_path,
            source_fps=fps,
            output_fps=playback_fps,
            sync_clock=sync_clock,
        )
        silence_audio = SilenceAudioStreamTrack()

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
        )

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
            f"user_id={user_id} fps={fps} playback_fps={playback_fps} "
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
        try:
            if session.idle_track is not None:
                print(f"🧊 WebRTC delete stop video track session_id={session_id}", flush=True)
                session.idle_track.stop()
            if session.audio_sender and session.audio_sender.track:
                print(f"🧊 WebRTC delete stop audio sender track session_id={session_id}", flush=True)
                session.audio_sender.track.stop()
            if session.silence_audio_track:
                print(f"🧊 WebRTC delete stop silence track session_id={session_id}", flush=True)
                session.silence_audio_track.stop()
            if session.audio_player and hasattr(session.audio_player, "stop"):
                print(f"🧊 WebRTC delete stop audio player session_id={session_id}", flush=True)
                session.audio_player.stop()
            if session.pc is not None:
                print(
                    f"🧊 WebRTC delete close peer connection session_id={session_id} "
                    f"state={session.pc.connectionState}",
                    flush=True,
                )
                await session.pc.close()
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
                "active_stream": s.active_stream,
                "age_seconds": now - s.created_at,
                "idle_seconds": now - s.last_activity,
                "fps": s.fps,
                "playback_fps": s.playback_fps,
                "chunk_duration": s.chunk_duration,
                "batch_size": s.batch_size,
                "player_url": f"/webrtc/player/{s.session_id}",
                "ice_transport_policy": s.ice_transport_policy,
            }
            for s in self.sessions.values()
            if s.active_stream is not None
        ]
