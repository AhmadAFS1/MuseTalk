import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCRtpSender

from scripts.webrtc_tracks import SwitchableVideoStreamTrack, SilenceAudioStreamTrack


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
    ):
        self.sessions: Dict[str, WebRTCSession] = {}
        self.session_ttl = session_ttl_seconds
        self.lock = asyncio.Lock()
        self.cleanup_task = None
        self.rtc_config = rtc_config
        self.ice_servers = ice_servers or []

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
        idle_track = SwitchableVideoStreamTrack(
            idle_video_path,
            source_fps=fps,
            output_fps=playback_fps,
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
        )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            print(f"🧊 WebRTC[{session_id}] connectionState={state}")
            if state == "closed":
                await self.delete_session(session_id)

        async with self.lock:
            self.sessions[session_id] = session

        return session

    async def get_session(self, session_id: str) -> Optional[WebRTCSession]:
        async with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.touch()
            return session

    async def delete_session(self, session_id: str) -> bool:
        async with self.lock:
            session = self.sessions.pop(session_id, None)

        if session is None:
            return False

        if session.idle_track is not None:
            session.idle_track.stop()
        if session.audio_sender and session.audio_sender.track:
            session.audio_sender.track.stop()
        if session.silence_audio_track:
            session.silence_audio_track.stop()
        if session.audio_player and hasattr(session.audio_player, "stop"):
            session.audio_player.stop()
        if session.pc is not None:
            await session.pc.close()

        return True
