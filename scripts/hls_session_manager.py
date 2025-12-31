import asyncio
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class HlsSession:
    session_id: str
    avatar_id: str
    idle_video_path: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    output_dir: Path = field(default_factory=Path)
    manifest_path: Path = field(default_factory=Path)
    segment_dir: Path = field(default_factory=Path)
    batch_size: int = 2
    playback_fps: Optional[int] = None
    musetalk_fps: Optional[int] = None
    segment_duration: float = 2.0
    part_duration: Optional[float] = None
    status: str = "idle"

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        return (time.time() - self.last_activity) > ttl_seconds

    def touch(self) -> None:
        self.last_activity = time.time()


def _generate_idle_hls(
    video_path: Path,
    output_dir: Path,
    segment_duration: float,
    playback_fps: Optional[int] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "index.m3u8"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
    ]

    if playback_fps and playback_fps > 0:
        ffmpeg_cmd += ["-r", str(playback_fps)]

    gop_opts = []
    if playback_fps and playback_fps > 0:
        gop = max(1, int(round(playback_fps * segment_duration)))
        gop_opts = [
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
        ]

    ffmpeg_cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        *gop_opts,
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "48000",
        "-f", "hls",
        "-hls_time", str(segment_duration),
        "-hls_playlist_type", "vod",
        "-hls_flags", "independent_segments",
        "-hls_segment_type", "fmp4",
        "-hls_segment_filename", str(segments_dir / "seg_%06d.m4s"),
        str(manifest_path),
    ]

    result = subprocess.run(
        ffmpeg_cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr[-1000:]
        raise RuntimeError(f"ffmpeg HLS generation failed: {stderr}")

    if not manifest_path.exists():
        raise RuntimeError("HLS manifest not created")

    return manifest_path


class HlsSessionManager:
    def __init__(self, session_ttl_seconds: int = 3600, base_dir: str = "results/hls"):
        self.sessions: Dict[str, HlsSession] = {}
        self.session_ttl = session_ttl_seconds
        self.lock = asyncio.Lock()
        self.cleanup_task = None
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

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
        batch_size: int = 2,
        playback_fps: Optional[int] = None,
        musetalk_fps: Optional[int] = None,
        segment_duration: float = 2.0,
        part_duration: Optional[float] = None,
    ) -> HlsSession:
        session_id = secrets.token_urlsafe(16)
        output_dir = self.base_dir / session_id
        segment_dir = output_dir / "segments"
        manifest_path = output_dir / "index.m3u8"

        session = HlsSession(
            session_id=session_id,
            avatar_id=avatar_id,
            idle_video_path=idle_video_path,
            user_id=user_id,
            output_dir=output_dir,
            manifest_path=manifest_path,
            segment_dir=segment_dir,
            batch_size=batch_size,
            playback_fps=playback_fps,
            musetalk_fps=musetalk_fps,
            segment_duration=segment_duration,
            part_duration=part_duration,
        )

        async with self.lock:
            self.sessions[session_id] = session

        await asyncio.to_thread(
            _generate_idle_hls,
            Path(idle_video_path),
            output_dir,
            segment_duration,
            playback_fps,
        )

        print(f"✅ Created HLS session: {session_id} (avatar: {avatar_id}, user: {user_id})")
        return session

    async def get_session(self, session_id: str) -> Optional[HlsSession]:
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

        if session.output_dir.exists():
            shutil.rmtree(session.output_dir, ignore_errors=True)
        print(f"🗑️  Deleted HLS session: {session_id} (user: {session.user_id})")
        return True

    def get_stats(self) -> dict:
        return {
            "total_sessions": len(self.sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "avatar_id": s.avatar_id,
                    "age_seconds": time.time() - s.created_at,
                    "idle_seconds": time.time() - s.last_activity,
                    "batch_size": s.batch_size,
                    "playback_fps": s.playback_fps,
                    "musetalk_fps": s.musetalk_fps,
                    "segment_duration": s.segment_duration,
                    "part_duration": s.part_duration,
                    "status": s.status,
                }
                for s in self.sessions.values()
            ],
        }
