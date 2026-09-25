"""Optional motion recovery for the persistent WebRTC video track."""
import asyncio
import math
import time
from collections import deque

import av

from scripts.motion_transitions import IDLE, flow_blend


MOTION_ENTRY_PREPARE_TIMEOUT_SECONDS = 5.0
MOTION_ENTRY_FLOW_TIMEOUT_SECONDS = 1.0
MOTION_ENTRY_CLOCK_TIMEOUT_SECONDS = .5


class MotionPlaybackMixin:
    def configure_motion_bank(self, bank, source_paths=None):
        self.motion_bank = bank
        self._motion_source_paths = {pose: str(path) for pose, path in (source_paths or {}).items()
                                     if pose in bank.sources}
        self._motion_idle_path = self._motion_source_paths.get(IDLE, self._current_idle_video_path)
        self._motion_source_paths.setdefault(IDLE, self._motion_idle_path)
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
        self._motion_entries = deque(maxlen=30)
        self._motion_entry = None
        self._motion_entry_task = None
        self._motion_entry_last_frame = None
        self._motion_entry_last_metadata = None
        self._motion_entry_output_metadata = None
        self._motion_entry_failure = asyncio.Event()
        self._motion_entry_failed_record = None
        self._motion_entry_clock_catchup = False

    def motion_status(self):
        if not getattr(self, "motion_bank", None):
            return None
        return {"enabled": True,
                "bank": {"fps": self.motion_bank.fps,
                         "routing_sha256": self.motion_bank.routing_sha256,
                         "sources": {pose: {"sha256": source["sha256"],
                                            "frame_count": source["frame_count"]}
                                     for pose, source in self.motion_bank.sources.items()}},
                "settled": self._motion_settled.is_set(),
                "building": self._motion_building,
                "last_emitted": self._emitted_motion,
                "returns": list(self._motion_returns),
                "entries": [{k: v for k, v in entry.items() if k != "target_bgr"}
                            for entry in self._motion_entries],
                "trace": list(self._motion_trace)}

    async def wait_for_motion_settled(self):
        if getattr(self, "motion_bank", None):
            await asyncio.wait_for(self._motion_settled.wait(), timeout=3)

    def _close_motion(self):
        if getattr(self, "motion_bank", None):
            self._motion_token += 1
            self._cancel_motion_entry()
            self._motion_settled.set()

    def _cancel_motion_entry(self):
        entry = getattr(self, "_motion_entry", None)
        if (entry is not None and entry["status"] != "failed"
                and "first_live_output_frame" not in entry):
            # The body bridge may be complete while transport clocks still
            # reconcile. Until a live frame is emitted, this is a cancelled entry.
            entry["status"] = "cancelled"
        if entry is not None:
            entry.pop("target_bgr", None)
        self._motion_entry = None
        self._motion_entry_last_frame = None
        self._motion_entry_last_metadata = None
        self._motion_entry_output_metadata = None
        self._motion_entry_clock_catchup = False

    async def wait_for_motion_entry_failure(self, generation_id=None):
        """Allow the owning API request to cancel audio/GPU without killing recv."""
        await self._motion_entry_failure.wait()
        record = self._motion_entry_failed_record
        if record is None or (generation_id is not None and record["generation_id"] != generation_id):
            return None
        return dict(record)

    def _fail_motion_entry(self, entry, exc):
        if (self._closed or not self._live_active or self._motion_entry is not entry
                or entry["generation_id"] != self._live_generation_id):
            return
        entry["status"] = "failed"
        entry["error"] = str(exc)
        self._motion_entry_failed_record = {k: v for k, v in entry.items() if k != "target_bgr"}
        self._motion_entry_failure.set()

    def _prepare_motion_entry(self, metadata, generation_id):
        """Predecode raw frame zero while live frames remain in the FIFO.

        Inference captured an earlier idle phase. The idle decoder keeps moving
        throughout inference and this worker. Once both media are ready, a short
        body-only bridge reaches that exact source phase before audio is released.
        No generated mouth frame is faded, skipped, or displayed before its audio.
        """
        bank = getattr(self, "motion_bank", None)
        if bank is None or self._motion_entry is not None:
            return
        entry = {"generation_id": generation_id, "pose_id": None,
                 "source_frame": None, "status": "preparing",
                 "prepared_at": time.monotonic(), "frames_emitted": 0,
                 "bridge_seconds": bank.bridge_seconds}
        self._motion_entry = entry
        self._motion_entries.append(entry)
        try:
            if metadata is None or int(metadata.get("generation_frame", -1)) != 0:
                raise ValueError("Motion entry requires metadata for generation frame zero")
            pose = bank.canonical(metadata["pose_id"])
            source_frame = int(metadata["source_frame"])
            if not 0 <= source_frame < bank.count(pose):
                raise ValueError("Motion entry source frame is outside its source video")
            if pose not in self._motion_source_paths:
                raise ValueError("Motion entry has no verified runtime source path for " + pose)
            source_path = self._motion_source_paths[pose]
            entry.update(pose_id=pose, source_frame=source_frame, source_path=source_path)
        except (KeyError, TypeError, ValueError) as exc:
            # Producer callbacks catch exceptions, so raising here would leave
            # the request prebuffering indefinitely without notifying its owner.
            self._fail_motion_entry(entry, exc)
            return

        def decode():
            from scripts.webrtc_tracks import IdleVideoStreamTrack
            decoder = IdleVideoStreamTrack(source_path, fps=bank.fps, decode_threads=16)
            try:
                for _ in range(source_frame):
                    decoder.read_frame()
                return decoder.read_frame().to_ndarray(format="bgr24")
            finally:
                decoder.stop()

        def owns_entry():
            return (not self._closed and self._live_active
                    and generation_id == self._live_generation_id
                    and self._motion_entry is entry)

        async def prepare_target():
            raw = await asyncio.to_thread(decode)
            if not owns_entry():
                return None
            # The first OpenCV/flow invocation can take >100 ms on a cold
            # process. Pay that setup cost while idle continues playing,
            # not inside the first deadline-sensitive bridge recv().
            await asyncio.to_thread(flow_blend, raw, raw, .5)
            return raw

        async def prepare():
            try:
                # Cancelling to_thread cannot terminate native work. A timed-out
                # decoder still closes in its own finally; its result has no path
                # back into this or a later generation. This deadline requires a
                # responsive event loop and does not cure a native/GIL deadlock.
                raw = await asyncio.wait_for(prepare_target(), MOTION_ENTRY_PREPARE_TIMEOUT_SECONDS)
                if not owns_entry():
                    return
                entry["target_bgr"] = raw
                entry["prepare_seconds"] = time.monotonic() - entry["prepared_at"]
                entry["status"] = "ready"
            except asyncio.TimeoutError:
                self._fail_motion_entry(entry, TimeoutError(
                    f"Motion entry preparation exceeded {MOTION_ENTRY_PREPARE_TIMEOUT_SECONDS:g}s"))
            except Exception as exc:
                self._fail_motion_entry(entry, exc)
        self._motion_entry_task = asyncio.create_task(prepare())

    def _motion_entry_pending(self):
        if getattr(self, "motion_bank", None) is None or self._live_released:
            return False
        entry = self._motion_entry
        return entry is None or entry["status"] != "completed"

    async def _next_motion_entry_frame(self, idle_steps):
        # The actual current idle frame is the outgoing image on EVERY tick,
        # including while target decoding and GPU prebuffering are pending.
        # Flow work runs off the event loop so the audio RTP cadence can continue.
        outgoing = self._advance_idle_frame(idle_steps)
        entry = self._motion_entry
        if entry is None or entry["status"] == "preparing":
            return outgoing
        if entry["status"] == "failed":
            # The API failure supervisor cancels this turn. Keep the receiver
            # alive and its idle body moving while that cleanup is scheduled.
            return outgoing
        if "bridge_started_at" not in entry:
            entry["bridge_started_at"] = time.monotonic()
            entry["first_output_frame"] = self._rtp_frame_index
        count = max(2, math.ceil(self.motion_bank.bridge_seconds * self._output_fps))
        progress = (entry["frames_emitted"] + 1) / count
        alpha = .5 - .5 * math.cos(math.pi * progress)
        target = entry["target_bgr"]
        try:
            blended = await asyncio.wait_for(
                asyncio.to_thread(flow_blend, outgoing.to_ndarray(format="bgr24"), target, alpha),
                MOTION_ENTRY_FLOW_TIMEOUT_SECONDS)
        except Exception as exc:
            if isinstance(exc, asyncio.TimeoutError):
                exc = TimeoutError(f"Motion entry flow exceeded {MOTION_ENTRY_FLOW_TIMEOUT_SECONDS:g}s")
            if (self._closed or not self._live_active or self._motion_entry is not entry
                    or entry["generation_id"] != self._live_generation_id):
                # Recovery can install while this obsolete worker is awaiting.
                # Consume its actual next frame so return metadata stays paired
                # with the image emitted by the interrupted recv().
                return self._last_idle_frame if self._closed else self._advance_idle_frame(0)
            self._fail_motion_entry(entry, exc)
            if self._motion_entry_last_frame is not None:
                self._motion_entry_output_metadata = dict(self._motion_entry_last_metadata)
                return self._motion_entry_last_frame
            return outgoing
        # A user abort/close/new turn can arrive while the CPU worker is running.
        if (self._closed or not self._live_active or self._motion_entry is not entry
                or entry["generation_id"] != self._live_generation_id):
            return self._last_idle_frame if self._closed else self._advance_idle_frame(0)
        frame = av.VideoFrame.from_ndarray(blended, format="bgr24").reformat(format="yuv420p")
        self._motion_entry_last_frame = frame
        self._motion_entry_output_metadata = {
            "pose_id": entry["pose_id"], "source_frame": entry["source_frame"],
            "mode": "speech_entry_body_bridge", "progress": round(progress, 3),
            "outgoing_idle_frame": self._idle.get_timing()["source_frame_index"],
            "outgoing_pose_id": self._current_idle_pose_id,
        }
        self._motion_entry_last_metadata = dict(self._motion_entry_output_metadata)
        entry["frames_emitted"] += 1
        entry["status"] = "completed" if entry["frames_emitted"] == count else "playing"
        if entry["status"] == "completed":
            entry["bridge_render_seconds"] = time.monotonic() - entry["bridge_started_at"]
        return frame

    async def _motion_entry_clock_frame(self):
        """Catch up silent transport slots before exposing any generated lips.

        Return a raw body frame when a contiguous video timestamp needs filling;
        return None only when frame zero can start within one video frame of TTS.
        A video lead is resolved by waiting for ordinary silent audio packets.
        Neither transport's timestamp counter is rewritten.
        """
        if getattr(self, "motion_bank", None) is None or self._live_released:
            return None
        entry = self._motion_entry
        clock = self._sync_clock
        if entry is None or entry["status"] != "completed" or clock is None:
            return None
        limit = MOTION_ENTRY_CLOCK_TIMEOUT_SECONDS
        max_frames = max(2, math.ceil(limit * self._output_fps))
        # Leave one audio packet of headroom on the video-behind side. Negative
        # skew can use the full video interval because audio is still advancing.
        positive_tolerance = max(0, 1 / self._output_fps - .02)
        negative_tolerance = 1 / self._output_fps
        while True:
            if (self._closed or not self._live_active or self._motion_entry is not entry
                    or entry["generation_id"] != self._live_generation_id):
                return self._last_idle_frame if self._closed else self._advance_idle_frame(0)
            audio_pts = clock.audio_transport_next_pts_seconds
            video_pts = self._rtp_frame_index / self._output_fps
            delta = None if audio_pts is None else audio_pts - video_pts
            now = time.monotonic()
            if delta is not None and -negative_tolerance - 1e-9 <= delta <= positive_tolerance + 1e-9:
                self._motion_entry_clock_catchup = False
                if "clock_wait_started_at" in entry:
                    entry["clock_wait_seconds"] = now - entry["clock_wait_started_at"]
                    # Catchup emitted existing timestamp slots faster than their
                    # old pacing deadline. Resume regular pacing from this anchor.
                    self._last_ts = now
                return None
            if "clock_wait_started_at" not in entry:
                entry["clock_wait_started_at"] = now
                entry["clock_wait_frames"] = 0
                entry["clock_initial_skew_seconds"] = delta
            elapsed = now - entry["clock_wait_started_at"]
            entry["clock_wait_seconds"] = elapsed
            if (elapsed >= limit or entry["clock_wait_frames"] >= max_frames
                    or (delta is not None and abs(delta) > limit + 1e-9)):
                self._motion_entry_clock_catchup = False
                self._fail_motion_entry(entry, TimeoutError(
                    f"Motion entry transport clocks did not converge within {limit:g}s; skew={delta}"))
                self._motion_entry_output_metadata = dict(self._motion_entry_last_metadata,
                                                          mode="entry_transport_alignment")
                return self._motion_entry_last_frame
            if delta is not None and delta > positive_tolerance:
                # Emit the missing slot, then skip ordinary pacing on the next
                # recv. Sleeping a full video interval would let audio advance
                # equally and leave the clock difference unchanged forever.
                entry["clock_wait_frames"] += 1
                self._motion_entry_clock_catchup = True
                self._motion_entry_output_metadata = dict(self._motion_entry_last_metadata,
                                                          mode="entry_transport_alignment")
                return self._motion_entry_last_frame
            # Video is ahead, or the first silent audio timestamp is not yet
            # available. Keep generated frame zero queued until audio catches up.
            await asyncio.sleep(.005)

    def _note_motion_output(self, metadata):
        if not getattr(self, "motion_bank", None) or metadata is None:
            return
        self._emitted_motion = dict(metadata)
        self._emitted_motion["output_frame"] = self._rtp_frame_index
        self._motion_trace.append(self._emitted_motion)
        entry = self._motion_entry
        if (entry is not None and entry["status"] == "completed"
                and metadata.get("mode") == "live" and "first_live_output_frame" not in entry):
            entry["first_live_output_frame"] = self._rtp_frame_index
            entry["first_live_generation_frame"] = metadata.get("generation_frame")
            entry["additional_start_seconds"] = time.monotonic() - entry["bridge_started_at"]
            entry.pop("target_bgr", None)

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
        if anchor is None:
            # An abort during entry must release the displayed blended body,
            # not jump backwards to the decoder that advanced under that bridge.
            anchor = self._motion_entry_last_frame
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
        if self._motion_entry_output_metadata is not None:
            self._note_motion_output(self._motion_entry_output_metadata)
            self._motion_entry_output_metadata = None
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
