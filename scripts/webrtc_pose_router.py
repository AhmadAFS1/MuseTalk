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
    is_queued: bool = False
    source_frame_offset: int = 0

    @property
    def uses_prepared_background(self) -> bool:
        return self.video_path is None


@dataclass(frozen=True)
class LivePoseQueueSegment:
    pose_id: str
    start_generation_frame: int
    end_generation_frame: int
    source_frame_offset: int = 0
    requested_at_permille: int = 0
    requested_start_generation_frame: int = 0


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
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
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
    """Select a pose cache/background independently from the WebRTC sender."""

    def __init__(
        self,
        pose_video_paths: Dict[str, str],
        *,
        prepared_pose_id: str = "default",
        prepared_pose_ids: Optional[set[str]] = None,
        initial_pose_id: str = "default",
        decoder_factory: Callable[[str], object] = _LoopingVideoDecoder,
    ):
        self._lock = threading.RLock()
        self._pose_video_paths: Dict[str, str] = {}
        self._decoders: Dict[str, object] = {}
        self._decoder_factory = decoder_factory
        self.prepared_pose_id = str(prepared_pose_id or "default")
        self._prepared_pose_ids = {self.prepared_pose_id}
        self._prepared_pose_ids.update(
            str(pose_id).strip().lower()
            for pose_id in (prepared_pose_ids or set())
            if str(pose_id).strip()
        )
        self.active_pose_id = self.prepared_pose_id
        self._version = 0
        self._origin_generation_frame = 0
        self._origin_pending = True
        self._last_generation_frame = 0
        self._read_failures = 0
        self._closed = False
        self._pose_queue: list[LivePoseQueueSegment] = []
        self._queue_hold_last_pose = True
        self._queue_mode = "sequence"
        self._pose_plan_request: Optional[dict] = None
        self._compiled_pose_plan: Optional[dict] = None

        for pose_id, video_path in pose_video_paths.items():
            self.register_pose(pose_id, video_path)
        self.switch_pose(initial_pose_id)

    def register_pose(self, pose_id: str, video_path: str) -> None:
        normalized_pose_id = str(pose_id or "").strip().lower()
        path = Path(video_path)
        if not normalized_pose_id:
            raise ValueError("pose_id is required")
        if not path.is_file():
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

    def set_prepared_pose_ids(self, pose_ids: set[str]) -> None:
        with self._lock:
            self._prepared_pose_ids = {self.prepared_pose_id}
            self._prepared_pose_ids.update(
                str(pose_id).strip().lower()
                for pose_id in pose_ids
                if str(pose_id).strip()
            )

    def switch_pose(self, pose_id: str) -> dict:
        normalized_pose_id = str(pose_id or self.prepared_pose_id).strip().lower()
        with self._lock:
            if self._closed:
                raise ValueError("Live pose router is closed")
            if normalized_pose_id not in self._pose_video_paths:
                raise KeyError(normalized_pose_id)
            changed = normalized_pose_id != self.active_pose_id
            queue_cleared = bool(self._pose_queue)
            if changed:
                self.active_pose_id = normalized_pose_id
                self._version += 1
                self._origin_pending = True
            if queue_cleared:
                self._pose_queue = []
                self._pose_plan_request = None
                self._compiled_pose_plan = None
                self._queue_mode = "sequence"
            return {
                "changed": changed,
                "live_pose_id": self.active_pose_id,
                "live_pose_version": self._version,
                "uses_prepared_background": (
                    self.active_pose_id in self._prepared_pose_ids
                ),
                "queue_cleared": queue_cleared,
            }

    def queue_pose_sequence(
        self,
        pose_ids: list[str],
        generation_fps: float,
        *,
        hold_last_pose: bool = True,
    ) -> list[dict]:
        """Route complete pose clips in deterministic generation-frame order."""

        safe_generation_fps = max(0.001, float(generation_fps or 0.0))
        normalized = [str(pose_id or "").strip().lower() for pose_id in pose_ids]
        if not normalized or any(not pose_id for pose_id in normalized):
            raise ValueError("At least one pose_id is required")

        with self._lock:
            if self._closed:
                raise ValueError("Live pose router is closed")
            for pose_id in normalized:
                if pose_id not in self._pose_video_paths:
                    raise KeyError(pose_id)

            segments = []
            start_frame = 0
            for pose_id in normalized:
                decoder = self._get_or_create_decoder_locked(pose_id)
                source_fps = max(0.001, float(getattr(decoder, "fps", 25.0)))
                source_frame_count = max(1, int(getattr(decoder, "frame_count", 1)))
                duration_frames = max(
                    1,
                    int(round(source_frame_count / source_fps * safe_generation_fps)),
                )
                end_frame = start_frame + duration_frames
                segments.append(
                    LivePoseQueueSegment(pose_id, start_frame, end_frame)
                )
                start_frame = end_frame

            self._pose_queue = segments
            self._queue_hold_last_pose = bool(hold_last_pose)
            self._queue_mode = "sequence"
            self._pose_plan_request = None
            self._compiled_pose_plan = None
            self.active_pose_id = normalized[0]
            self._origin_pending = False
            self._origin_generation_frame = 0
            self._version += 1
            return [
                {
                    "pose_id": segment.pose_id,
                    "start_generation_frame": segment.start_generation_frame,
                    "end_generation_frame": segment.end_generation_frame,
                    "duration_frames": (
                        segment.end_generation_frame
                        - segment.start_generation_frame
                    ),
                    "duration_seconds": round(
                        (
                            segment.end_generation_frame
                            - segment.start_generation_frame
                        )
                        / safe_generation_fps,
                        3,
                    ),
                }
                for segment in segments
            ]

    def queue_pose_plan(
        self,
        pose_plan: dict,
        total_generation_frames: int,
        generation_fps: float,
        *,
        hold_last_pose: bool = True,
    ) -> dict:
        """Stage a semantic audio-progress plan for boundary-safe compilation."""

        safe_total_frames = max(1, int(total_generation_frames))
        safe_generation_fps = max(0.001, float(generation_fps or 0.0))
        raw_segments = list(pose_plan.get("segments") or [])
        if not raw_segments:
            raise ValueError("pose_plan requires at least one segment")

        normalized = [
            {
                "at_permille": int(segment["at_permille"]),
                "pose_id": str(segment["pose_id"]).strip().lower(),
            }
            for segment in raw_segments
        ]
        with self._lock:
            if self._closed:
                raise ValueError("Live pose router is closed")
            for segment in normalized:
                if segment["pose_id"] not in self._pose_video_paths:
                    raise KeyError(segment["pose_id"])

            first = normalized[0]
            self._pose_plan_request = {
                "version": int(pose_plan.get("version") or 2),
                "clock": str(pose_plan.get("clock") or "audio_progress"),
                "segments": normalized,
                "on_complete": str(
                    pose_plan.get("on_complete") or "neutral_resting"
                ),
                "switch_mode": str(
                    pose_plan.get("switch_mode") or "next_boundary"
                ),
                "total_generation_frames": safe_total_frames,
                "generation_fps": safe_generation_fps,
            }
            self._compiled_pose_plan = {
                "status": "pending_phase_alignment",
                "requested_segments": [dict(segment) for segment in normalized],
                "segments": [],
                "skipped_segments": [],
                "total_generation_frames": safe_total_frames,
                "generation_fps": safe_generation_fps,
            }
            self._pose_queue = [
                LivePoseQueueSegment(
                    pose_id=first["pose_id"],
                    start_generation_frame=0,
                    end_generation_frame=safe_total_frames,
                    requested_at_permille=0,
                    requested_start_generation_frame=0,
                )
            ]
            self._queue_hold_last_pose = bool(hold_last_pose)
            self._queue_mode = "pose_plan"
            self.active_pose_id = first["pose_id"]
            self._origin_pending = False
            self._origin_generation_frame = 0
            self._version += 1
            return {
                **self._compiled_pose_plan,
                "requested_segments": [
                    dict(segment) for segment in normalized
                ],
            }

    def _pose_duration_frames_locked(
        self,
        pose_id: str,
        generation_fps: float,
    ) -> tuple[int, int]:
        decoder = self._get_or_create_decoder_locked(pose_id)
        source_fps = max(0.001, float(getattr(decoder, "fps", 25.0)))
        source_frame_count = max(
            1,
            int(getattr(decoder, "frame_count", 1)),
        )
        duration_frames = max(
            1,
            int(round(
                source_frame_count / source_fps * generation_fps
            )),
        )
        return duration_frames, source_frame_count

    def _compile_pose_plan_locked(
        self,
        source_frame_offset: int,
        generation_fps: float,
    ) -> dict:
        request = self._pose_plan_request
        if not request:
            return {
                "status": "missing",
                "segments": [],
                "skipped_segments": [],
            }

        safe_generation_fps = max(0.001, float(generation_fps or 0.0))
        total_frames = max(1, int(request["total_generation_frames"]))
        requested = [dict(segment) for segment in request["segments"]]
        minimum_terminal_frames = max(
            1,
            int(round(safe_generation_fps * 0.75)),
        )
        first_pose_id = requested[0]["pose_id"]
        _, first_source_frame_count = self._pose_duration_frames_locked(
            first_pose_id,
            safe_generation_fps,
        )
        normalized_source_offset = (
            max(0, int(source_frame_offset)) % first_source_frame_count
        )

        current_pose_id = first_pose_id
        current_start = 0
        current_source_offset = normalized_source_offset
        current_requested_anchor = 0
        compiled: list[LivePoseQueueSegment] = []
        skipped: list[dict] = []

        for request_index, next_request in enumerate(requested[1:], start=1):
            next_pose_id = next_request["pose_id"]
            desired_start = int(round(
                total_frames
                * int(next_request["at_permille"])
                / 1000.0
            ))
            if next_pose_id == current_pose_id:
                skipped.append(
                    {
                        **next_request,
                        "reason": "adjacent_duplicate_after_snapping",
                    }
                )
                continue

            current_duration, current_source_frame_count = (
                self._pose_duration_frames_locked(
                    current_pose_id,
                    safe_generation_fps,
                )
            )
            if current_start == 0 and current_source_offset:
                decoder = self._get_or_create_decoder_locked(current_pose_id)
                source_fps = max(
                    0.001,
                    float(getattr(decoder, "fps", 25.0)),
                )
                remaining_source_frames = max(
                    1,
                    current_source_frame_count - current_source_offset,
                )
                boundary = current_start + max(
                    1,
                    int(round(
                        remaining_source_frames
                        / source_fps
                        * safe_generation_fps
                    )),
                )
            else:
                boundary = current_start + current_duration
            while boundary < desired_start:
                boundary += current_duration

            # Expressive inserts must be able to play a complete certified clip
            # and leave a small direct tail when another semantic segment follows.
            later_request = (
                requested[request_index + 1]
                if request_index + 1 < len(requested)
                else None
            )
            if later_request is not None:
                next_duration, _ = self._pose_duration_frames_locked(
                    next_pose_id,
                    safe_generation_fps,
                )
                if (
                    boundary
                    + next_duration
                    + minimum_terminal_frames
                    > total_frames
                ):
                    skipped.append(
                        {
                            **next_request,
                            "reason": "insufficient_audio_for_complete_clip",
                            "requested_start_generation_frame": desired_start,
                            "earliest_safe_start_generation_frame": boundary,
                        }
                    )
                    continue

            if boundary >= total_frames - minimum_terminal_frames:
                skipped.append(
                    {
                        **next_request,
                        "reason": "no_safe_boundary_before_audio_end",
                        "requested_start_generation_frame": desired_start,
                        "earliest_safe_start_generation_frame": boundary,
                    }
                )
                continue

            compiled.append(
                LivePoseQueueSegment(
                    pose_id=current_pose_id,
                    start_generation_frame=current_start,
                    end_generation_frame=boundary,
                    source_frame_offset=current_source_offset,
                    requested_at_permille=current_requested_anchor,
                    requested_start_generation_frame=int(round(
                        total_frames * current_requested_anchor / 1000.0
                    )),
                )
            )
            current_pose_id = next_pose_id
            current_start = boundary
            current_source_offset = 0
            current_requested_anchor = int(next_request["at_permille"])

        compiled.append(
            LivePoseQueueSegment(
                pose_id=current_pose_id,
                start_generation_frame=current_start,
                end_generation_frame=total_frames,
                source_frame_offset=current_source_offset,
                requested_at_permille=current_requested_anchor,
                requested_start_generation_frame=int(round(
                    total_frames * current_requested_anchor / 1000.0
                )),
            )
        )
        self._pose_queue = compiled
        self.active_pose_id = compiled[0].pose_id
        self._origin_generation_frame = 0
        self._origin_pending = False
        self._version += 1

        compiled_segments = [
            {
                "pose_id": segment.pose_id,
                "requested_at_permille": segment.requested_at_permille,
                "requested_start_generation_frame": (
                    segment.requested_start_generation_frame
                ),
                "effective_start_generation_frame": (
                    segment.start_generation_frame
                ),
                "effective_end_generation_frame": (
                    segment.end_generation_frame
                ),
                "effective_start_seconds": round(
                    segment.start_generation_frame / safe_generation_fps,
                    3,
                ),
                "effective_end_seconds": round(
                    segment.end_generation_frame / safe_generation_fps,
                    3,
                ),
                "boundary_snap_delay_frames": (
                    segment.start_generation_frame
                    - segment.requested_start_generation_frame
                ),
                "source_frame_offset": segment.source_frame_offset,
            }
            for segment in compiled
        ]
        self._compiled_pose_plan = {
            "status": "compiled",
            "requested_segments": requested,
            "segments": compiled_segments,
            "skipped_segments": skipped,
            "total_generation_frames": total_frames,
            "generation_fps": safe_generation_fps,
            "source_frame_offset": normalized_source_offset,
        }
        return {
            **self._compiled_pose_plan,
            "requested_segments": [
                dict(segment)
                for segment in self._compiled_pose_plan["requested_segments"]
            ],
            "segments": [
                dict(segment)
                for segment in self._compiled_pose_plan["segments"]
            ],
            "skipped_segments": [
                dict(segment)
                for segment in self._compiled_pose_plan["skipped_segments"]
            ],
        }

    def get_compiled_pose_plan(self) -> Optional[dict]:
        with self._lock:
            if self._compiled_pose_plan is None:
                return None
            return {
                **self._compiled_pose_plan,
                "requested_segments": [
                    dict(segment)
                    for segment in self._compiled_pose_plan.get(
                        "requested_segments",
                        [],
                    )
                ],
                "segments": [
                    dict(segment)
                    for segment in self._compiled_pose_plan.get("segments", [])
                ],
                "skipped_segments": [
                    dict(segment)
                    for segment in self._compiled_pose_plan.get(
                        "skipped_segments",
                        [],
                    )
                ],
            }

    def align_first_queued_pose(
        self,
        source_frame_offset: int,
        generation_fps: float,
    ) -> list[dict]:
        """Start the first queued pose at the frozen idle phase.

        The first segment is shortened to its next source boundary, while every
        later segment remains a complete clip. This preserves boundary-aligned
        transitions after the initial idle-to-live handoff.
        """

        safe_generation_fps = max(0.001, float(generation_fps or 0.0))
        with self._lock:
            if self._closed:
                raise ValueError("Live pose router is closed")
            if not self._pose_queue:
                return []
            if self._queue_mode == "pose_plan":
                return self._compile_pose_plan_locked(
                    source_frame_offset,
                    safe_generation_fps,
                )["segments"]

            first = self._pose_queue[0]
            decoder = self._get_or_create_decoder_locked(first.pose_id)
            source_fps = max(0.001, float(getattr(decoder, "fps", 25.0)))
            source_frame_count = max(
                1,
                int(getattr(decoder, "frame_count", 1)),
            )
            normalized_offset = max(0, int(source_frame_offset)) % source_frame_count
            remaining_source_frames = max(1, source_frame_count - normalized_offset)
            first_duration = max(
                1,
                int(round(
                    remaining_source_frames
                    / source_fps
                    * safe_generation_fps
                )),
            )

            aligned = [
                LivePoseQueueSegment(
                    pose_id=first.pose_id,
                    start_generation_frame=0,
                    end_generation_frame=first_duration,
                    source_frame_offset=normalized_offset,
                )
            ]
            next_start = first_duration
            for segment in self._pose_queue[1:]:
                duration = max(
                    1,
                    segment.end_generation_frame
                    - segment.start_generation_frame,
                )
                aligned.append(
                    LivePoseQueueSegment(
                        pose_id=segment.pose_id,
                        start_generation_frame=next_start,
                        end_generation_frame=next_start + duration,
                        source_frame_offset=0,
                    )
                )
                next_start += duration

            self._pose_queue = aligned
            self._origin_generation_frame = 0
            self._origin_pending = False
            self._version += 1
            return [
                {
                    "pose_id": segment.pose_id,
                    "start_generation_frame": segment.start_generation_frame,
                    "end_generation_frame": segment.end_generation_frame,
                    "source_frame_offset": segment.source_frame_offset,
                }
                for segment in aligned
            ]

    def _get_or_create_decoder_locked(self, pose_id: str):
        decoder = self._decoders.get(pose_id)
        if decoder is None:
            decoder = self._decoder_factory(self._pose_video_paths[pose_id])
            self._decoders[pose_id] = decoder
        return decoder

    def snapshots_for_range(
        self,
        start_generation_frame: int,
        frame_count: int,
        generation_fps: float,
    ) -> list[LivePoseSnapshot]:
        return [
            self.snapshot(start_generation_frame + offset, generation_fps)
            for offset in range(max(0, int(frame_count)))
        ]

    def snapshot(
        self,
        generation_frame_index: int,
        generation_fps: float,
    ) -> LivePoseSnapshot:
        safe_frame = max(0, int(generation_frame_index))
        safe_fps = max(0.001, float(generation_fps or 0.0))
        with self._lock:
            if self._closed:
                return LivePoseSnapshot(
                    self.prepared_pose_id,
                    None,
                    self._version,
                    safe_frame,
                    safe_fps,
                )
            self._last_generation_frame = max(self._last_generation_frame, safe_frame)
            planned = next(
                (
                    segment
                    for segment in self._pose_queue
                    if (
                        segment.start_generation_frame
                        <= safe_frame
                        < segment.end_generation_frame
                    )
                ),
                None,
            )
            if planned is None and self._pose_queue and self._queue_hold_last_pose:
                planned = self._pose_queue[-1]
            if planned is not None:
                video_path = (
                    None
                    if planned.pose_id in self._prepared_pose_ids
                    else self._pose_video_paths[planned.pose_id]
                )
                return LivePoseSnapshot(
                    planned.pose_id,
                    video_path,
                    self._version,
                    planned.start_generation_frame,
                    safe_fps,
                    True,
                    planned.source_frame_offset,
                )
            if self._origin_pending:
                self._origin_generation_frame = safe_frame
                self._origin_pending = False
            video_path = (
                None
                if self.active_pose_id in self._prepared_pose_ids
                else self._pose_video_paths[self.active_pose_id]
            )
            return LivePoseSnapshot(
                self.active_pose_id,
                video_path,
                self._version,
                self._origin_generation_frame,
                safe_fps,
            )

    def source_frame_index(
        self,
        snapshot: LivePoseSnapshot,
        generation_frame_index: int,
    ) -> int:
        with self._lock:
            decoder = self._get_or_create_decoder_locked(snapshot.pose_id)
            source_fps = max(0.001, float(getattr(decoder, "fps", 25.0)))
        relative_frame = max(
            0,
            int(generation_frame_index) - snapshot.origin_generation_frame,
        )
        return snapshot.source_frame_offset + int(
            relative_frame * source_fps / snapshot.generation_fps
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
            decoder = self._get_or_create_decoder_locked(snapshot.pose_id)
        source_indices = [
            self.source_frame_index(snapshot, int(start_generation_frame) + offset)
            for offset in range(int(frame_count))
        ]
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
            return {
                "active_pose_id": self.active_pose_id,
                "prepared_pose_id": self.prepared_pose_id,
                "prepared_pose_ids": sorted(self._prepared_pose_ids),
                "available_pose_ids": sorted(self._pose_video_paths),
                "version": self._version,
                "origin_generation_frame": self._origin_generation_frame,
                "last_generation_frame": self._last_generation_frame,
                "read_failures": self._read_failures,
                "queue_mode": self._queue_mode,
                "queue": [
                    {
                        "pose_id": segment.pose_id,
                        "start_generation_frame": segment.start_generation_frame,
                        "end_generation_frame": segment.end_generation_frame,
                        "source_frame_offset": segment.source_frame_offset,
                        "requested_at_permille": segment.requested_at_permille,
                        "requested_start_generation_frame": (
                            segment.requested_start_generation_frame
                        ),
                    }
                    for segment in self._pose_queue
                ],
                "queue_hold_last_pose": self._queue_hold_last_pose,
                "compiled_pose_plan": self.get_compiled_pose_plan(),
                "decoders": {
                    pose_id: decoder.get_stats()
                    for pose_id, decoder in self._decoders.items()
                    if hasattr(decoder, "get_stats")
                },
            }
