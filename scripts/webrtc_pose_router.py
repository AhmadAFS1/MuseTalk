"""Low-memory MP4 background routing for live WebRTC composition."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class LivePoseSnapshot:
    pose_id: str
    video_path: Optional[str]
    version: int
    origin_generation_frame: int
    generation_fps: float

    @property
    def uses_prepared_background(self) -> bool:
        return self.video_path is None


class _LoopingVideoDecoder:
    """Decode requested MP4 frames without retaining the full clip in RAM."""

    def __init__(self, video_path: str):
        import cv2

        self._cv2 = cv2
        self.video_path = str(video_path)
        self._lock = threading.Lock()
        self._capture = cv2.VideoCapture(self.video_path)
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Could not open live pose video: {self.video_path}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.fps = 25.0
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._next_frame_index = 0
        self.read_count = 0
        self.seek_count = 0
        self.failure_count = 0

    def _seek(self, frame_index: int) -> None:
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        self._next_frame_index = int(frame_index)
        self.seek_count += 1

    def _read_at_locked(self, frame_index: int):
        target = max(0, int(frame_index))
        if self.frame_count > 0:
            target %= self.frame_count

        if target < self._next_frame_index or target - self._next_frame_index > 120:
            self._seek(target)

        frame = None
        while self._next_frame_index <= target:
            ok, candidate = self._capture.read()
            if not ok:
                self._seek(0)
                ok, candidate = self._capture.read()
                if not ok:
                    self.failure_count += 1
                    return None
                frame = candidate
                self._next_frame_index = 1
                break
            frame = candidate
            self._next_frame_index += 1

        self.read_count += 1
        return frame

    def read_frames(self, frame_indices: list[int]) -> list:
        with self._lock:
            return [self._read_at_locked(frame_index) for frame_index in frame_indices]

    def close(self) -> None:
        with self._lock:
            self._capture.release()

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "video_path": self.video_path,
                "source_fps": self.fps,
                "source_frame_count": self.frame_count,
                "frames_returned": self.read_count,
                "seeks": self.seek_count,
                "failures": self.failure_count,
            }


class LivePoseVideoRouter:
    """Select an MP4 background independently from prepared MuseTalk latents."""

    def __init__(
        self,
        pose_video_paths: Dict[str, str],
        *,
        prepared_pose_id: str = "default",
        initial_pose_id: str = "default",
        decoder_factory: Callable[[str], object] = _LoopingVideoDecoder,
    ):
        self._lock = threading.RLock()
        self._pose_video_paths: Dict[str, str] = {}
        self._decoders: Dict[str, object] = {}
        self._decoder_factory = decoder_factory
        self.prepared_pose_id = str(prepared_pose_id or "default")
        self.active_pose_id = self.prepared_pose_id
        self._version = 0
        self._origin_generation_frame = 0
        self._origin_pending = True
        self._last_generation_frame = 0
        self._read_failures = 0
        self._closed = False

        for pose_id, video_path in pose_video_paths.items():
            self.register_pose(pose_id, video_path)
        self.switch_pose(initial_pose_id)

    def register_pose(self, pose_id: str, video_path: str) -> None:
        normalized_pose_id = str(pose_id or "").strip().lower()
        path = Path(video_path)
        if not normalized_pose_id:
            raise ValueError("pose_id is required")
        if not path.exists() or not path.is_file():
            raise ValueError(f"Live pose video does not exist: {path}")
        with self._lock:
            if self._closed:
                raise ValueError("Live pose router is closed")
            previous_path = self._pose_video_paths.get(normalized_pose_id)
            self._pose_video_paths[normalized_pose_id] = str(path)
            if previous_path and previous_path != str(path):
                decoder = self._decoders.pop(normalized_pose_id, None)
                if decoder is not None:
                    decoder.close()

    def switch_pose(self, pose_id: str) -> dict:
        normalized_pose_id = str(pose_id or self.prepared_pose_id).strip().lower()
        with self._lock:
            if self._closed:
                raise ValueError("Live pose router is closed")
            if normalized_pose_id not in self._pose_video_paths:
                raise KeyError(normalized_pose_id)
            changed = normalized_pose_id != self.active_pose_id
            if changed:
                self.active_pose_id = normalized_pose_id
                self._version += 1
                self._origin_pending = True
            return {
                "changed": changed,
                "live_pose_id": self.active_pose_id,
                "live_pose_version": self._version,
                "uses_prepared_background": self.active_pose_id == self.prepared_pose_id,
            }

    def snapshot(self, generation_frame_index: int, generation_fps: float) -> LivePoseSnapshot:
        safe_generation_frame = max(0, int(generation_frame_index))
        safe_generation_fps = max(0.001, float(generation_fps or 0.0))
        with self._lock:
            if self._closed:
                return LivePoseSnapshot(
                    pose_id=self.prepared_pose_id,
                    video_path=None,
                    version=self._version,
                    origin_generation_frame=safe_generation_frame,
                    generation_fps=safe_generation_fps,
                )
            self._last_generation_frame = max(self._last_generation_frame, safe_generation_frame)
            if self._origin_pending:
                self._origin_generation_frame = safe_generation_frame
                self._origin_pending = False
            video_path = None
            if self.active_pose_id != self.prepared_pose_id:
                video_path = self._pose_video_paths[self.active_pose_id]
            return LivePoseSnapshot(
                pose_id=self.active_pose_id,
                video_path=video_path,
                version=self._version,
                origin_generation_frame=self._origin_generation_frame,
                generation_fps=safe_generation_fps,
            )

    def read_background_frames(
        self,
        snapshot: LivePoseSnapshot,
        start_generation_frame: int,
        frame_count: int,
    ) -> list:
        if frame_count <= 0 or snapshot.uses_prepared_background:
            return [None] * max(0, int(frame_count))

        with self._lock:
            if self._closed:
                return [None] * int(frame_count)
            decoder = self._decoders.get(snapshot.pose_id)
            if decoder is None:
                decoder = self._decoder_factory(snapshot.video_path)
                self._decoders[snapshot.pose_id] = decoder
            source_fps = max(0.001, float(getattr(decoder, "fps", 25.0)))

        source_indices = []
        for offset in range(int(frame_count)):
            generation_index = int(start_generation_frame) + offset
            relative_frame = max(0, generation_index - snapshot.origin_generation_frame)
            source_index = int(relative_frame * source_fps / snapshot.generation_fps)
            source_indices.append(source_index)

        try:
            return decoder.read_frames(source_indices)
        except Exception:
            with self._lock:
                self._read_failures += 1
            return [None] * int(frame_count)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            decoders = list(self._decoders.values())
            self._decoders.clear()
        for decoder in decoders:
            decoder.close()

    def get_stats(self) -> dict:
        with self._lock:
            decoder_stats = {
                pose_id: decoder.get_stats()
                for pose_id, decoder in self._decoders.items()
                if hasattr(decoder, "get_stats")
            }
            return {
                "active_pose_id": self.active_pose_id,
                "prepared_pose_id": self.prepared_pose_id,
                "available_pose_ids": sorted(self._pose_video_paths),
                "version": self._version,
                "origin_generation_frame": self._origin_generation_frame,
                "last_generation_frame": self._last_generation_frame,
                "read_failures": self._read_failures,
                "decoders": decoder_stats,
            }
