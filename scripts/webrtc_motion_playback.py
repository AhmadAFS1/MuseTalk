"""Optional motion recovery for the persistent WebRTC video track."""
import asyncio
import math
import time
from collections import deque

import av

from scripts.motion_transitions import IDLE, flow_blend


class MotionPlaybackMixin:
    def configure_motion_bank(self, bank):
        self.motion_bank = bank
        self._motion_idle_path = self._current_idle_video_path
        self._motion_token = 0
        self._motion_task = None
        self._motion_settled = asyncio.Event()
        self._motion_settled.set()
        self._motion_building = False
        self._emitted_motion = None
        self._popped_motion = None
        self._last_live_motion = None
        self._motion_trace = deque(maxlen=3600)
        self._motion_returns = deque(maxlen=30)
        self._motion_return_record = None

    def motion_status(self):
        if not getattr(self, "motion_bank", None):
            return None
        return {"enabled": True, "settled": self._motion_settled.is_set(),
                "building": self._motion_building,
                "last_emitted": self._emitted_motion,
                "returns": list(self._motion_returns),
                "trace": list(self._motion_trace)}

    async def wait_for_motion_settled(self):
        if getattr(self, "motion_bank", None):
            await asyncio.wait_for(self._motion_settled.wait(), timeout=3)

    def _close_motion(self):
        if getattr(self, "motion_bank", None):
            self._motion_token += 1
            self._motion_settled.set()

    def _note_motion_output(self, metadata):
        if not getattr(self, "motion_bank", None) or metadata is None:
            return
        self._emitted_motion = dict(metadata)
        self._emitted_motion["output_frame"] = self._rtp_frame_index
        self._motion_trace.append(self._emitted_motion)

    def _begin_motion_return(self):
        bank = getattr(self, "motion_bank", None)
        if bank is None:
            return False
        # Drop a pre-staged frame-zero return: this controller owns the phase.
        staged = self._completion_idle_switch
        self._completion_idle_switch = None
        if staged:
            staged["idle_track"].stop()
        self._stop_pending_idle_switches()
        if self._motion_building or not self._motion_settled.is_set():
            return True
        anchor = self._last_live_frame
        metadata = self._emitted_motion
        if anchor is None or metadata is None:
            return True  # Cancelled before speech was actually displayed.
        source_pose = bank.canonical(metadata["pose_id"])
        source_index = int(metadata["source_frame"])
        edge = bank.edge(source_pose, source_index, IDLE)
        if not edge["admissible"]:
            raise RuntimeError("Attempted playback from an uncovered motion phase")
        old = anchor.to_ndarray(format="bgr24")
        self._last_idle_frame = anchor
        self._idle_transition_frames = []
        self._motion_settled.clear()
        self._motion_building = True
        self._motion_token += 1
        token = self._motion_token
        started = time.monotonic()
        record = {"from_pose": source_pose, "from_frame": source_index,
                  "target_frame": edge["target_frame"], "started_at": started,
                  "source_output_frame": metadata.get("output_frame"),
                  "status": "building", "bridge_seconds": bank.bridge_seconds}
        self._motion_returns.append(record)
        self._motion_return_record = record

        def build():
            from scripts.webrtc_tracks import IdleVideoStreamTrack
            path = self._motion_idle_path
            decoder = IdleVideoStreamTrack(path, fps=bank.fps, decode_threads=16)
            try:
                target = int(edge["target_frame"])
                for _ in range(target):
                    decoder.read_frame()
                count = max(2, math.ceil(bank.bridge_seconds * self._output_fps))
                frames, ids = [], []
                previous = -1
                incoming = None
                for n in range(count):
                    offset = int(n * bank.fps / self._output_fps)
                    for _ in range(offset - previous):
                        incoming = decoder.read_frame().to_ndarray(format="bgr24")
                    previous = offset
                    t = (n + 1) / count
                    alpha = .5 - .5 * math.cos(math.pi * t)
                    blend = flow_blend(old, incoming, alpha)
                    frames.append(av.VideoFrame.from_ndarray(blend, format="bgr24").reformat(format="yuv420p"))
                    ids.append({"pose_id": IDLE,
                                "source_frame": (target+offset) % bank.count(IDLE),
                                "mode": "mouth_release_return", "progress": round(t,3)})
                return decoder, frames, ids
            except BaseException:
                decoder.stop()
                raise

        async def install():
            try:
                decoder, frames, ids = await asyncio.to_thread(build)
                if self._closed or token != self._motion_token:
                    decoder.stop()
                    return
                record["build_seconds"] = time.monotonic() - started
                record["status"] = "playing"
                self._apply_idle_switch(decoder, idle_video_path=self._motion_idle_path,
                                        pose_id=IDLE, reason="matched_motion_return",
                                        transition_frames=frames)
                self._motion_transition_ids = list(ids)
                self._motion_building = False
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                self._motion_building = False
                self._motion_settled.set()
                print(f"Motion return failed: {exc}", flush=True)
        self._motion_task = asyncio.create_task(install())
        return True

    def _note_motion_idle_frame(self):
        if not getattr(self, "motion_bank", None):
            return
        ids = getattr(self, "_motion_transition_ids", None)
        if ids:
            self._note_motion_output(ids.pop(0))
            if not ids:
                record = self._motion_return_record
                if record:
                    record["status"] = "completed"
                    record["total_seconds"] = time.monotonic() - record["started_at"]
                self._motion_settled.set()
        elif not self._motion_building:
            timing = self._idle.get_timing()
            self._note_motion_output({"pose_id": self._current_idle_pose_id,
                                      "source_frame": timing["source_frame_index"], "mode": "idle"})
