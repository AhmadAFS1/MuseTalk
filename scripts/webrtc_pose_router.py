"""Low-memory MP4 background routing for live WebRTC composition."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_MAX_SEMANTIC_DRIFT_SECONDS = max(
    0.0,
    _env_float("WEBRTC_POSE_MAX_SEMANTIC_DRIFT_SECONDS", 0.75),
)
DEFAULT_FORCED_SWITCH_CROSSFADE_FRAMES = max(
    1,
    _env_int("WEBRTC_POSE_FORCED_CROSSFADE_FRAMES", 4),
)


@dataclass(frozen=True)
class LivePoseSnapshot:
    pose_id: str
    video_path: Optional[str]
    version: int
    origin_generation_frame: int
    generation_fps: float
    is_queued: bool = False
    source_frame_offset: int = 0
    switch_strategy: str = "continuous"
    crossfade_frames: int = 0
    render_key: Optional[str] = None

    @property
    def effective_render_key(self) -> str:
        return self.render_key or self.pose_id

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
    switch_strategy: str = "initial"
    crossfade_frames: int = 0
    render_key: Optional[str] = None


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
        pose_variant_render_keys: Optional[Dict[str, list[str]]] = None,
        initial_pose_id: str = "default",
        decoder_factory: Callable[[str], object] = _LoopingVideoDecoder,
        max_semantic_drift_seconds: float = DEFAULT_MAX_SEMANTIC_DRIFT_SECONDS,
        forced_switch_crossfade_frames: int = (
            DEFAULT_FORCED_SWITCH_CROSSFADE_FRAMES
        ),
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
        self._pose_variant_render_keys = {
            str(pose_id).strip().lower(): [
                str(render_key).strip().lower()
                for render_key in render_keys
                if str(render_key).strip()
            ]
            for pose_id, render_keys in (pose_variant_render_keys or {}).items()
            if str(pose_id).strip()
        }
        self._selected_variant_render_keys: Dict[str, str] = {
            pose_id: render_keys[0]
            for pose_id, render_keys in self._pose_variant_render_keys.items()
            if render_keys
        }
        self._variant_context_key = ""
        self._variant_context_assignments: Dict[str, Dict[str, str]] = {}
        self._variant_rotation_positions: Dict[str, int] = {
            pose_id: -1
            for pose_id in self._pose_variant_render_keys
        }
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
        self.max_semantic_drift_seconds = max(
            0.0,
            float(max_semantic_drift_seconds),
        )
        self.forced_switch_crossfade_frames = max(
            1,
            int(forced_switch_crossfade_frames),
        )

        for pose_id, video_path in pose_video_paths.items():
            self.register_pose(pose_id, video_path)
        self.switch_pose(initial_pose_id)

    def set_variant_context(self, context_key: str) -> dict[str, str]:
        """Select physical variants deterministically for one assistant turn."""

        normalized_context = str(context_key or "").strip()
        with self._lock:
            self._variant_context_key = normalized_context
            if normalized_context in self._variant_context_assignments:
                selected = dict(
                    self._variant_context_assignments[normalized_context]
                )
                self._selected_variant_render_keys = selected
                return selected
            selected: Dict[str, str] = {}
            for pose_id, render_keys in self._pose_variant_render_keys.items():
                if not render_keys:
                    continue
                if normalized_context:
                    index = (
                        self._variant_rotation_positions.get(pose_id, -1) + 1
                    ) % len(render_keys)
                    self._variant_rotation_positions[pose_id] = index
                else:
                    index = 0
                selected[pose_id] = render_keys[index]
            self._selected_variant_render_keys = selected
            if normalized_context:
                self._variant_context_assignments[normalized_context] = dict(
                    selected
                )
            return dict(selected)

    def _render_key_for_pose_locked(self, pose_id: str) -> str:
        return self._selected_variant_render_keys.get(pose_id, pose_id)

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
                    self._render_key_for_pose_locked(self.active_pose_id)
                    in self._prepared_pose_ids
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
                render_key = self._render_key_for_pose_locked(pose_id)
                decoder = self._get_or_create_decoder_locked(render_key)
                source_fps = max(0.001, float(getattr(decoder, "fps", 25.0)))
                source_frame_count = max(1, int(getattr(decoder, "frame_count", 1)))
                duration_frames = max(
                    1,
                    int(round(source_frame_count / source_fps * safe_generation_fps)),
                )
                end_frame = start_frame + duration_frames
                segments.append(
                    LivePoseQueueSegment(
                        pose_id,
                        start_frame,
                        end_frame,
                        render_key=render_key,
                    )
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
                    **(
                        {"render_key": segment.render_key}
                        if segment.render_key
                        and segment.render_key != segment.pose_id
                        else {}
                    ),
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
        """Stage a semantic audio-progress plan for bounded-drift compilation."""

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
                "max_semantic_drift_seconds": (
                    self.max_semantic_drift_seconds
                ),
                "forced_switch_crossfade_frames": (
                    self.forced_switch_crossfade_frames
                ),
            }
            self._compiled_pose_plan = {
                "status": "pending_phase_alignment",
                "requested_segments": [dict(segment) for segment in normalized],
                "segments": [],
                "skipped_segments": [],
                "total_generation_frames": safe_total_frames,
                "generation_fps": safe_generation_fps,
                "switch_policy": "bounded_semantic",
                "max_semantic_drift_seconds": (
                    self.max_semantic_drift_seconds
                ),
                "forced_switch_crossfade_frames": (
                    self.forced_switch_crossfade_frames
                ),
            }
            self._pose_queue = [
                LivePoseQueueSegment(
                    pose_id=first["pose_id"],
                    start_generation_frame=0,
                    end_generation_frame=safe_total_frames,
                    requested_at_permille=0,
                    requested_start_generation_frame=0,
                    render_key=self._render_key_for_pose_locked(first["pose_id"]),
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
        render_key: str,
        generation_fps: float,
    ) -> tuple[int, int]:
        decoder = self._get_or_create_decoder_locked(render_key)
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

    def _nearest_safe_boundary_locked(
        self,
        render_key: str,
        *,
        current_start: int,
        source_frame_offset: int,
        desired_start: int,
        generation_fps: float,
    ) -> int:
        """Return the closest complete-loop boundary after the segment starts."""

        duration_frames, source_frame_count = (
            self._pose_duration_frames_locked(
                render_key,
                generation_fps,
            )
        )
        normalized_source_offset = (
            max(0, int(source_frame_offset)) % source_frame_count
        )
        if normalized_source_offset:
            decoder = self._get_or_create_decoder_locked(render_key)
            source_fps = max(
                0.001,
                float(getattr(decoder, "fps", 25.0)),
            )
            remaining_source_frames = max(
                1,
                source_frame_count - normalized_source_offset,
            )
            first_boundary = current_start + max(
                1,
                int(round(
                    remaining_source_frames
                    / source_fps
                    * generation_fps
                )),
            )
        else:
            first_boundary = current_start + duration_frames

        if desired_start <= first_boundary:
            return first_boundary

        complete_cycles = max(
            0,
            (desired_start - first_boundary) // duration_frames,
        )
        previous_boundary = (
            first_boundary + complete_cycles * duration_frames
        )
        candidates = [previous_boundary]
        if previous_boundary < desired_start:
            candidates.append(previous_boundary + duration_frames)
        return min(
            candidates,
            key=lambda boundary: (
                abs(boundary - desired_start),
                1 if boundary > desired_start else 0,
            ),
        )

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
        max_semantic_drift_seconds = max(
            0.0,
            float(request["max_semantic_drift_seconds"]),
        )
        max_semantic_drift_frames = max(
            0,
            int(math.floor(
                max_semantic_drift_seconds * safe_generation_fps + 1e-9
            )),
        )
        forced_switch_crossfade_frames = max(
            1,
            int(request["forced_switch_crossfade_frames"]),
        )
        minimum_terminal_frames = max(
            1,
            int(round(safe_generation_fps * 0.75)),
        )
        first_pose_id = requested[0]["pose_id"]
        current_render_key = self._render_key_for_pose_locked(first_pose_id)
        _, first_source_frame_count = self._pose_duration_frames_locked(
            current_render_key,
            safe_generation_fps,
        )
        normalized_source_offset = (
            max(0, int(source_frame_offset)) % first_source_frame_count
        )

        current_pose_id = first_pose_id
        current_start = 0
        current_source_offset = normalized_source_offset
        current_requested_anchor = 0
        current_switch_strategy = "initial_phase_aligned"
        current_crossfade_frames = 0
        compiled: list[LivePoseQueueSegment] = []
        skipped: list[dict] = []

        for next_request in requested[1:]:
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

            if desired_start <= current_start:
                skipped.append(
                    {
                        **next_request,
                        "reason": "semantic_window_collapsed",
                        "requested_start_generation_frame": desired_start,
                        "current_effective_start_generation_frame": (
                            current_start
                        ),
                    }
                )
                continue

            if desired_start >= total_frames - minimum_terminal_frames:
                skipped.append(
                    {
                        **next_request,
                        "reason": "no_semantic_window_before_audio_end",
                        "requested_start_generation_frame": desired_start,
                        "latest_allowed_start_generation_frame": (
                            total_frames - minimum_terminal_frames - 1
                        ),
                    }
                )
                continue

            safe_boundary = self._nearest_safe_boundary_locked(
                current_render_key,
                current_start=current_start,
                source_frame_offset=current_source_offset,
                desired_start=desired_start,
                generation_fps=safe_generation_fps,
            )
            safe_boundary_drift = safe_boundary - desired_start
            if (
                safe_boundary < total_frames - minimum_terminal_frames
                and abs(safe_boundary_drift) <= max_semantic_drift_frames
            ):
                switch_frame = safe_boundary
                switch_strategy = "nearest_safe_boundary"
                crossfade_frames = 0
            else:
                # Semantic correctness wins when a 6-12 second source loop has
                # no certified boundary near the spoken cue. The incoming clip
                # begins at its canonical first frame and receives a slightly
                # longer, zero-duration crossfade in the compose stage.
                switch_frame = desired_start
                switch_strategy = "requested_time_crossfade"
                crossfade_frames = forced_switch_crossfade_frames

            compiled.append(
                LivePoseQueueSegment(
                    pose_id=current_pose_id,
                    start_generation_frame=current_start,
                    end_generation_frame=switch_frame,
                    source_frame_offset=current_source_offset,
                    requested_at_permille=current_requested_anchor,
                    requested_start_generation_frame=int(round(
                        total_frames * current_requested_anchor / 1000.0
                    )),
                    switch_strategy=current_switch_strategy,
                    crossfade_frames=current_crossfade_frames,
                    render_key=current_render_key,
                )
            )
            current_pose_id = next_pose_id
            current_render_key = self._render_key_for_pose_locked(next_pose_id)
            current_start = switch_frame
            current_source_offset = 0
            current_requested_anchor = int(next_request["at_permille"])
            current_switch_strategy = switch_strategy
            current_crossfade_frames = crossfade_frames

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
                switch_strategy=current_switch_strategy,
                crossfade_frames=current_crossfade_frames,
                render_key=current_render_key,
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
                "render_key": segment.render_key or segment.pose_id,
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
                "semantic_drift_frames": (
                    segment.start_generation_frame
                    - segment.requested_start_generation_frame
                ),
                "semantic_drift_seconds": round(
                    (
                        segment.start_generation_frame
                        - segment.requested_start_generation_frame
                    )
                    / safe_generation_fps,
                    3,
                ),
                "switch_strategy": segment.switch_strategy,
                "crossfade_frames": segment.crossfade_frames,
                "source_frame_offset": segment.source_frame_offset,
            }
            for segment in compiled
        ]
        max_abs_semantic_drift_frames = max(
            (
                abs(int(segment["semantic_drift_frames"]))
                for segment in compiled_segments
            ),
            default=0,
        )
        semantic_timing_valid = (
            max_abs_semantic_drift_frames <= max_semantic_drift_frames
        )
        if not semantic_timing_valid:
            raise RuntimeError(
                "Compiled pose plan exceeded its semantic drift limit: "
                f"observed={max_abs_semantic_drift_frames} frames, "
                f"limit={max_semantic_drift_frames} frames"
            )
        self._compiled_pose_plan = {
            "status": "compiled",
            "switch_policy": "bounded_semantic",
            "requested_segments": requested,
            "segments": compiled_segments,
            "skipped_segments": skipped,
            "total_generation_frames": total_frames,
            "generation_fps": safe_generation_fps,
            "source_frame_offset": normalized_source_offset,
            "max_semantic_drift_seconds": max_semantic_drift_seconds,
            "max_semantic_drift_frames": max_semantic_drift_frames,
            "max_abs_semantic_drift_frames": (
                max_abs_semantic_drift_frames
            ),
            "max_abs_semantic_drift_seconds": round(
                max_abs_semantic_drift_frames / safe_generation_fps,
                3,
            ),
            "semantic_timing_valid": semantic_timing_valid,
            "forced_switch_crossfade_frames": (
                forced_switch_crossfade_frames
            ),
            "requested_time_crossfade_count": sum(
                1
                for segment in compiled_segments
                if segment["switch_strategy"] == "requested_time_crossfade"
            ),
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
            first_render_key = first.render_key or first.pose_id
            decoder = self._get_or_create_decoder_locked(first_render_key)
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
                    render_key=first_render_key,
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
                        render_key=segment.render_key or segment.pose_id,
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
                    **(
                        {"render_key": segment.render_key}
                        if segment.render_key
                        and segment.render_key != segment.pose_id
                        else {}
                    ),
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
                render_key = planned.render_key or planned.pose_id
                video_path = (
                    None
                    if render_key in self._prepared_pose_ids
                    else self._pose_video_paths[render_key]
                )
                return LivePoseSnapshot(
                    planned.pose_id,
                    video_path,
                    self._version,
                    planned.start_generation_frame,
                    safe_fps,
                    True,
                    planned.source_frame_offset,
                    planned.switch_strategy,
                    planned.crossfade_frames,
                    render_key,
                )
            if self._origin_pending:
                self._origin_generation_frame = safe_frame
                self._origin_pending = False
            render_key = self._render_key_for_pose_locked(self.active_pose_id)
            video_path = (
                None
                if render_key in self._prepared_pose_ids
                else self._pose_video_paths[render_key]
            )
            return LivePoseSnapshot(
                self.active_pose_id,
                video_path,
                self._version,
                self._origin_generation_frame,
                safe_fps,
                render_key=render_key,
            )

    def source_frame_index(
        self,
        snapshot: LivePoseSnapshot,
        generation_frame_index: int,
    ) -> int:
        with self._lock:
            decoder = self._get_or_create_decoder_locked(
                snapshot.effective_render_key
            )
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
            decoder = self._get_or_create_decoder_locked(
                snapshot.effective_render_key
            )
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
                        "render_key": segment.render_key or segment.pose_id,
                        "start_generation_frame": segment.start_generation_frame,
                        "end_generation_frame": segment.end_generation_frame,
                        "source_frame_offset": segment.source_frame_offset,
                        "requested_at_permille": segment.requested_at_permille,
                        "requested_start_generation_frame": (
                            segment.requested_start_generation_frame
                        ),
                        "switch_strategy": segment.switch_strategy,
                        "crossfade_frames": segment.crossfade_frames,
                    }
                    for segment in self._pose_queue
                ],
                "queue_hold_last_pose": self._queue_hold_last_pose,
                "variant_context_key": self._variant_context_key,
                "selected_variant_render_keys": dict(
                    self._selected_variant_render_keys
                ),
                "variant_context_assignments": {
                    context: dict(assignments)
                    for context, assignments in self._variant_context_assignments.items()
                },
                "compiled_pose_plan": self.get_compiled_pose_plan(),
                "decoders": {
                    pose_id: decoder.get_stats()
                    for pose_id, decoder in self._decoders.items()
                    if hasattr(decoder, "get_stats")
                },
            }
