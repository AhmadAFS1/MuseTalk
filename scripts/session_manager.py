import asyncio
import secrets
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class UserSession:
    """Represents a single user's streaming session"""
    session_id: str
    avatar_id: str
    user_id: Optional[str] = None  # Optional user identifier from your app
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Streaming state
    active_stream: Optional[str] = None  # request_id if streaming
    chunk_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    # Session config
    batch_size: int = 2
    fps: int = 15
    chunk_duration: int = 2
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if session has expired"""
        return (time.time() - self.last_activity) > ttl_seconds
    
    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()


class SessionManager:
    """
    Manages user sessions for concurrent multi-user streaming.
    Each session is isolated with its own chunk queue and state.
    """
    
    def __init__(self, session_ttl_seconds: int = 3600):
        self.sessions: Dict[str, UserSession] = {}
        self.session_ttl = session_ttl_seconds
        self.lock = asyncio.Lock()
        self.cleanup_task = None
    
    def start_cleanup(self):
        """Start background cleanup of expired sessions"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            async with self.lock:
                expired = [
                    sid for sid, session in self.sessions.items()
                    if session.is_expired(self.session_ttl)
                ]
                
                for sid in expired:
                    session = self.sessions.pop(sid)
                    print(f"🗑️  Expired session: {sid} (user: {session.user_id})")
    
    async def create_session(
        self,
        avatar_id: str,
        user_id: Optional[str] = None,
        batch_size: int = 2,
        fps: int = 15,
        chunk_duration: int = 2
    ) -> UserSession:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(16)
        
        session = UserSession(
            session_id=session_id,
            avatar_id=avatar_id,
            user_id=user_id,
            batch_size=batch_size,
            fps=fps,
            chunk_duration=chunk_duration
        )
        
        async with self.lock:
            self.sessions[session_id] = session
        
        print(f"✅ Created session: {session_id} (avatar: {avatar_id}, user: {user_id})")
        return session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID and update activity"""
        async with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        async with self.lock:
            if session_id in self.sessions:
                session = self.sessions.pop(session_id)
                print(f"🗑️  Deleted session: {session_id} (user: {session.user_id})")
                return True
            return False
    
    async def get_user_sessions(self, user_id: str) -> list[UserSession]:
        """Get all sessions for a user"""
        async with self.lock:
            return [
                session for session in self.sessions.values()
                if session.user_id == user_id
            ]
    
    def get_stats(self) -> dict:
        """Get session statistics"""
        return {
            'total_sessions': len(self.sessions),
            'active_streams': sum(
                1 for s in self.sessions.values() 
                if s.active_stream is not None
            ),
            'sessions': [
                {
                    'session_id': s.session_id,
                    'user_id': s.user_id,
                    'avatar_id': s.avatar_id,
                    'active_stream': s.active_stream,
                    'age_seconds': time.time() - s.created_at,
                    'idle_seconds': time.time() - s.last_activity
                }
                for s in self.sessions.values()
            ]
        }

    def get_live_sessions(self) -> list[dict]:
        """Return only sessions that are actively streaming"""
        now = time.time()
        return [
            {
                'session_id': s.session_id,
                'user_id': s.user_id,
                'avatar_id': s.avatar_id,
                'active_stream': s.active_stream,
                'age_seconds': now - s.created_at,
                'idle_seconds': now - s.last_activity,
                'batch_size': s.batch_size,
                'fps': s.fps,
                'chunk_duration': s.chunk_duration,
            }
            for s in self.sessions.values()
            if s.active_stream is not None
        ]
