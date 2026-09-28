#!/usr/bin/env python
"""CPU tests for WEBRTC_NONBLOCKING_HANDOFF (plan item 1.5), no CUDA.

Covers scripts/webrtc_live_handoff.py with a fake loop/track and then the REAL
SwitchableVideoStreamTrack (real strict-FIFO queue, real VideoSyncClock in
timestamp-locked mode, real recv()) on a real idle clip:

  T1 order: start -> batches -> marker run in submission order; the marker sees
     every frame already queued; on_pushed reports prebuffer readiness.
  T2 non-blocking: with the event loop hogged for 200 ms, submit_frames returns
     in microseconds while today's run_coroutine_threadsafe(...).result() waits.
  T3 bound/backpressure: a full track queue (push blocked) fills the FIFO to the
     bound, the next submit waits exactly until space frees; timeout path.
  T4 cancel/stale: a cancelled turn's queued frames are skipped, the next turn's
     frames follow in order.
  T5 close: close() drops queued frames, wakes a waiting scheduler thread, and
     submit_marker returns False.
  T6 converter threads: WEBRTC_HANDOFF_CONVERT_THREADS=2 + WEBRTC_HANDOFF_VERIFY=1:
     converter-thread I420 == on-loop PyAV conversion, zero order/content errors.
  T7 real track end to end: frames played by recv() through the non-blocking
     handoff (with and without converter threads) are SHA-identical, in order,
     to the blocking handoff and to the direct conversion of the inputs; and
     WEBRTC_LIFETIME_COUNTERS stays monotonic across two turns.
  T8 WEBRTC_DEADLINE_PACING=1 plays the same frames in the same order.
  T9 with no flag set the track carries none of the levers (today's payload).

Run from the worktree root:
  /workspace/.venvs/musetalk_trt_stagewise/bin/python \
    docs/fps_comparisons/4070s_300fps_impl_20260928/webrtc/test_live_handoff_fifo.py
Writes test_live_handoff_fifo.json next to this file; exit 0 only if all pass.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

import scripts.webrtc_live_handoff as handoff_mod  # noqa: E402
from scripts.webrtc_live_handoff import (  # noqa: E402
    LiveFrameHandoff,
    bgr_to_yuv420p_frame,
    video_frame_from_live_item,
)

OUT = Path(__file__).with_name("test_live_handoff_fifo.json")
IDLE_CLIP = ROOT / "results/v15/avatars/chinese_bob_pink_bedroom_idle_d4b06da317/input_video.mp4"
H, W = 896, 512
RESULTS: list[dict] = []


def record(name: str, ok: bool, **detail) -> None:
    RESULTS.append({"test": name, "result": "pass" if ok else "fail", **detail})
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)}", flush=True)


def make_frames(count: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    frames = []
    for index in range(count):
        frame = base.copy()
        frame[:16, :, :] = (index * 7 + seed) % 256  # unique band per frame
        frame[16:32, : (index % W) + 1, 0] = 255
        frames.append(frame)
    return frames


def i420_sha(frame) -> str:
    return hashlib.sha256(frame.to_ndarray().tobytes()).hexdigest()


class LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


class FakeTrack:
    """Minimal track: strict FIFO asyncio.Queue + push_bgr_frames_batch."""

    def __init__(self, max_queue: int = 400, prebuffer: int = 16):
        self._max_queue = max_queue
        self._queue = asyncio.Queue(maxsize=max_queue)
        self.gate = asyncio.Event()
        self.gate.set()
        self.prebuffer = prebuffer
        self.generation = 0
        self.log: list = []
        self.queued = 0

    def start_live(self) -> int:
        self.generation += 1
        self.log.append(("start", self.generation))
        return self.generation

    async def push_bgr_frames_batch(self, frames, generation_id=None, metadata=None):
        for index, item in enumerate(frames):
            await self.gate.wait()
            frame, _converted_here = video_frame_from_live_item(item)
            await self._queue.put((generation_id, frame))
            self.queued += 1
            tag = metadata[index] if metadata else None
            self.log.append(("frame", generation_id, tag, i420_sha(frame)))
        return self.queued >= self.prebuffer


def new_handoff(lt: LoopThread, **kwargs):
    track = lt.call(_make_fake_track(**kwargs))
    return track, LiveFrameHandoff(track, lt.loop, max_pending_frames=kwargs.get("bound", 64))


async def _make_fake_track(max_queue=400, prebuffer=16, bound=64):
    return FakeTrack(max_queue=max_queue, prebuffer=prebuffer)


async def _start(track):
    return track.start_live()


async def _gate(track, open_: bool):
    (track.gate.set if open_ else track.gate.clear)()


def gate(lt: LoopThread, track, open_: bool) -> None:
    """Open/close the fake track's push gate synchronously on the loop (a
    call_soon_threadsafe could run after a drain that never yields)."""
    lt.call(_gate(track, open_))


def wait_until(predicate, timeout=10.0, step=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


# ---------------------------------------------------------------------------
def t1_order(lt: LoopThread):
    track, h = new_handoff(lt)
    frames = make_frames(80, 1)
    pushed_flags, marker_seen = [], []
    turn = h.begin_turn(lambda: _start(track), label="t1")
    for b in range(10):
        batch = frames[b * 8:(b + 1) * 8]
        h.submit_frames(turn, batch, metadata=[{"i": b * 8 + i} for i in range(8)],
                        on_pushed=pushed_flags.append)
    h.submit_marker(turn, lambda gid: marker_seen.append((gid, track.queued)))
    wait_until(lambda: marker_seen)
    tags = [entry[2]["i"] for entry in track.log if entry[0] == "frame"]
    expected_sha = [i420_sha(bgr_to_yuv420p_frame(f)) for f in frames]
    got_sha = [entry[3] for entry in track.log if entry[0] == "frame"]
    ok = (track.log[0] == ("start", 1) and tags == list(range(80)) and got_sha == expected_sha
          and marker_seen == [(1, 80)] and pushed_flags == [False, True] + [True] * 8
          and h.pending_frames() == 0)
    record("T1_order_marker_prebuffer", ok, frames=len(tags), marker=marker_seen,
           on_pushed=pushed_flags, stats=h.get_stats())


def t2_nonblocking(lt: LoopThread):
    track, h = new_handoff(lt)
    frames = make_frames(40, 2)
    turn = h.begin_turn(lambda: _start(track), label="t2")
    lt.loop.call_soon_threadsafe(time.sleep, 0.2)  # hog the loop
    time.sleep(0.01)
    submit_s = []
    for b in range(5):
        t0 = time.perf_counter()
        h.submit_frames(turn, frames[b * 8:(b + 1) * 8])
        submit_s.append(time.perf_counter() - t0)
    wait_until(lambda: track.queued == 40)
    # Today's blocking pattern against the same hog, for contrast.
    lt.loop.call_soon_threadsafe(time.sleep, 0.2)
    time.sleep(0.01)
    t0 = time.perf_counter()
    asyncio.run_coroutine_threadsafe(
        track.push_bgr_frames_batch(frames[:8], generation_id=turn.generation_id),
        lt.loop).result(10)
    blocking_s = time.perf_counter() - t0
    ok = max(submit_s) < 0.005 and blocking_s > 0.15 and track.queued == 48
    record("T2_scheduler_never_waits_on_loop", ok,
           max_submit_ms=round(max(submit_s) * 1000, 4),
           blocking_result_wait_ms=round(blocking_s * 1000, 1))


def t3_bound(lt: LoopThread):
    track, h = new_handoff(lt, bound=64)
    gate(lt, track, False)  # track queue "full": push blocks
    frames = make_frames(8, 3)
    turn = h.begin_turn(lambda: _start(track), label="t3")
    t0 = time.perf_counter()
    for _ in range(8):
        h.submit_frames(turn, frames)  # 64 frames: fills the bound, no wait
    fill_s = time.perf_counter() - t0
    waited = {}

    def ninth():
        s = time.perf_counter()
        h.submit_frames(turn, frames, wait_timeout_s=10)
        waited["s"] = time.perf_counter() - s

    thread = threading.Thread(target=ninth)
    thread.start()
    time.sleep(0.3)
    blocked = thread.is_alive()
    gate(lt, track, True)
    thread.join(5)
    wait_until(lambda: track.queued == 72)
    ok1 = (fill_s < 0.01 and blocked and 0.25 < waited.get("s", 0) < 2.0
           and h.max_pending_frames_seen <= 64 + 8 and track.queued == 72)
    record("T3_bound_waits_only_when_full", ok1, fill_ms=round(fill_s * 1000, 3),
           ninth_wait_s=round(waited.get("s", -1), 3),
           max_pending_seen=h.max_pending_frames_seen, bound=h.max_pending_frames)
    # Timeout path: bound exceeded for longer than the push timeout.
    wait_until(lambda: h.pending_frames() == 0)
    gate(lt, track, False)
    for _ in range(8):
        h.submit_frames(turn, frames)
    s = time.perf_counter()
    h.submit_frames(turn, frames, wait_timeout_s=0.3)
    timeout_s = time.perf_counter() - s
    gate(lt, track, True)
    wait_until(lambda: track.queued == 72 + 72)
    ok2 = 0.25 < timeout_s < 1.0 and track.queued == 144
    record("T3b_bound_timeout_then_queue", ok2, timeout_wait_s=round(timeout_s, 3),
           overflow_waits=h.overflow_waits)


def t4_cancel(lt: LoopThread):
    track, h = new_handoff(lt)
    gate(lt, track, False)
    a, b = make_frames(24, 4), make_frames(16, 5)
    turn_a = h.begin_turn(lambda: _start(track), label="A")
    for k in range(3):
        h.submit_frames(turn_a, a[k * 8:(k + 1) * 8])
    time.sleep(0.05)
    h.cancel_turn(turn_a)  # batch 0 is in flight (blocked on the gate)
    turn_b = h.begin_turn(lambda: _start(track), label="B")
    for k in range(2):
        h.submit_frames(turn_b, b[k * 8:(k + 1) * 8])
    done = []
    h.submit_marker(turn_b, lambda gid: done.append(gid))
    gate(lt, track, True)
    wait_until(lambda: done)
    gens = [entry[1] for entry in track.log if entry[0] == "frame"]
    ok = (gens.count(1) == 8 and gens.count(2) == 16 and gens == [1] * 8 + [2] * 16
          and turn_a.frames_skipped == 16 and done == [2])
    record("T4_cancelled_turn_skipped_next_in_order", ok, gens_seen={1: gens.count(1), 2: gens.count(2)},
           skipped=turn_a.frames_skipped)


def t5_close(lt: LoopThread):
    track, h = new_handoff(lt, bound=16)
    gate(lt, track, False)
    frames = make_frames(8, 6)
    turn = h.begin_turn(lambda: _start(track), label="t5")
    h.submit_frames(turn, frames)
    h.submit_frames(turn, frames)
    released = {}

    def blocked_submit():
        s = time.perf_counter()
        h.submit_frames(turn, frames, wait_timeout_s=20)
        released["s"] = time.perf_counter() - s

    thread = threading.Thread(target=blocked_submit)
    thread.start()
    time.sleep(0.2)
    dropped = h.close()
    thread.join(5)
    marker_ok = h.submit_marker(turn, lambda gid: None)
    gate(lt, track, True)
    time.sleep(0.1)
    ok = (not thread.is_alive() and released.get("s", 99) < 1.0 and dropped == 8
          and marker_ok is False and h.get_stats()["closed"])
    record("T5_close_drops_and_wakes", ok, dropped=dropped, waiter_released_s=round(released.get("s", -1), 3))


def t6_converter(lt: LoopThread):
    os.environ["WEBRTC_HANDOFF_CONVERT_THREADS"] = "2"
    os.environ["WEBRTC_HANDOFF_VERIFY"] = "1"
    try:
        track, h = new_handoff(lt)
        frames = make_frames(48, 7)
        turn = h.begin_turn(lambda: _start(track), label="t6")
        for b in range(6):
            h.submit_frames(turn, frames[b * 8:(b + 1) * 8])
        done = []
        h.submit_marker(turn, lambda gid: done.append(gid))
        wait_until(lambda: done, timeout=20)
        stats = h.get_stats()
        got = [entry[3] for entry in track.log if entry[0] == "frame"]
        expected = [i420_sha(bgr_to_yuv420p_frame(f)) for f in frames]
        ok = (got == expected and stats["preconvert_batches"] == 6
              and stats["verify_frames"] == 48 and stats["verify_content_mismatch"] == 0
              and stats["verify_order_errors"] == 0 and stats["verify_i420_checked"] == 48
              and stats["verify_i420_mismatch"] == 0)
        record("T6_converter_threads_exact_i420", ok, preconvert_batches=stats["preconvert_batches"],
               verify={k: v for k, v in stats.items() if k.startswith("verify_")})
    finally:
        os.environ.pop("WEBRTC_HANDOFF_CONVERT_THREADS", None)
        os.environ.pop("WEBRTC_HANDOFF_VERIFY", None)


# ---------------------------------------------------------------------------
# T7: real SwitchableVideoStreamTrack
# ---------------------------------------------------------------------------
async def _make_real_track():
    from scripts.webrtc_tracks import SwitchableVideoStreamTrack, VideoSyncClock
    clock = VideoSyncClock(source_fps=20.0)
    track = SwitchableVideoStreamTrack(
        str(IDLE_CLIP), source_fps=20.0, output_fps=20.0, sync_clock=clock,
        prebuffer_seconds=0.5, idle_source_fps=24.0,
    )
    return track, clock


async def _release(track, clock, frame_count: int):
    # What _release_webrtc_playout does for the video side (no audio transport).
    clock.set_audio_media_duration(frame_count / 20.0)
    clock.mark_audio_ready()
    clock.mark_video_ready()
    clock.release_playout(time.monotonic() + 0.05)


async def _consume_turn(track, max_frames=400):
    """recv() until the turn returns to idle; return SHAs of live frames sent."""
    live, total = [], 0
    was_live = False
    while total < max_frames:
        frame = await track.recv()
        total += 1
        if track._live_active and track._live_released and frame is track._last_live_frame:
            live.append(i420_sha(frame))
            was_live = True
        if was_live and not track._live_active:
            break
    return live


def run_real_turn(lt: LoopThread, track, clock, frames, mode: str, converter: bool):
    """Drive one turn like api_server's frame_batch_callback does."""
    consume = asyncio.run_coroutine_threadsafe(_consume_turn(track), lt.loop)
    released = []
    if mode == "blocking":
        gen = lt.call(_start(track))
        for b in range(0, len(frames), 8):
            ready = asyncio.run_coroutine_threadsafe(
                track.push_bgr_frames_batch(frames[b:b + 8], generation_id=gen), lt.loop).result(30)
            if ready and not released:
                lt.call(_release(track, clock, len(frames)))
                released.append(True)
        lt.loop.call_soon_threadsafe(track.signal_generation_complete, gen)
    else:
        if converter:
            os.environ["WEBRTC_HANDOFF_CONVERT_THREADS"] = "2"
        h = handoff_mod.get_live_handoff(track, lt.loop)

        def on_pushed(ready):
            if ready and not released:
                released.append(True)
                asyncio.ensure_future(_release(track, clock, len(frames)))

        turn = h.begin_turn(lambda: _start(track), label=mode)
        for b in range(0, len(frames), 8):
            h.submit_frames(turn, frames[b:b + 8], on_pushed=on_pushed)
        h.submit_marker(turn, lambda gid: track.signal_generation_complete(gid))
    live = consume.result(60)
    os.environ.pop("WEBRTC_HANDOFF_CONVERT_THREADS", None)
    return live


def t7_real_track(lt: LoopThread):
    if not IDLE_CLIP.exists():
        record("T7_real_track_end_to_end", False, error=f"missing {IDLE_CLIP}")
        return
    os.environ["WEBRTC_LIFETIME_COUNTERS"] = "1"
    try:
        frames = make_frames(60, 11)
        expected = [i420_sha(bgr_to_yuv420p_frame(f)) for f in frames]
        runs = {}
        for mode, converter in (("blocking", False), ("nonblocking", False), ("nonblocking_convert", True)):
            track, clock = lt.call(_make_real_track())
            live = run_real_turn(lt, track, clock, frames,
                                 "blocking" if mode == "blocking" else "nonblocking", converter)
            stats = track.lifetime_stats()
            runs[mode] = {"live_frames": len(live), "equal_to_inputs": live == expected,
                          "frames_played": stats["frames_played"],
                          "frames_duplicated": stats["frames_duplicated"],
                          "fresh_output_frames": stats["fresh_output_frames"]}
            if mode == "nonblocking":
                # Second turn on the same track: lifetime counters are monotonic.
                frames2 = make_frames(30, 12)
                expected2 = [i420_sha(bgr_to_yuv420p_frame(f)) for f in frames2]
                live2 = run_real_turn(lt, track, clock, frames2, "nonblocking", False)
                stats2 = track.lifetime_stats()
                runs["nonblocking_turn2"] = {
                    "live_frames": len(live2), "equal_to_inputs": live2 == expected2,
                    "lifetime_frames_played": stats2["frames_played"],
                    "per_turn_frames_played": track._frames_played,
                    "turns_started": stats2["turns_started"],
                    "fresh_output_frames": stats2["fresh_output_frames"],
                }
            lt.loop.call_soon_threadsafe(track.stop)
            time.sleep(0.05)
        ok = (all(runs[m]["equal_to_inputs"] for m in runs)
              and runs["blocking"]["frames_played"] == 60
              and runs["nonblocking"]["frames_played"] == 60
              and runs["nonblocking_turn2"]["lifetime_frames_played"] == 90
              and runs["nonblocking_turn2"]["per_turn_frames_played"] == 30
              and runs["nonblocking_turn2"]["turns_started"] == 2)
        record("T7_real_track_end_to_end", ok, runs=runs)
    finally:
        os.environ.pop("WEBRTC_LIFETIME_COUNTERS", None)


def t8_deadline_pacing_same_content(lt: LoopThread):
    """WEBRTC_DEADLINE_PACING=1 changes send timing only: same live frames, in order."""
    os.environ["WEBRTC_DEADLINE_PACING"] = "1"
    try:
        frames = make_frames(60, 21)
        expected = [i420_sha(bgr_to_yuv420p_frame(f)) for f in frames]
        track, clock = lt.call(_make_real_track())
        started = time.monotonic()
        live = run_real_turn(lt, track, clock, frames, "nonblocking", False)
        elapsed = time.monotonic() - started
        pacing = track._deadline_pacing
        lt.loop.call_soon_threadsafe(track.stop)
        time.sleep(0.05)
        record("T8_deadline_pacing_same_content", pacing and live == expected,
               live_frames=len(live), deadline_pacing=pacing, turn_wall_s=round(elapsed, 3))
    finally:
        os.environ.pop("WEBRTC_DEADLINE_PACING", None)


def t9_defaults_inert(lt: LoopThread):
    """No flag set: no handoff, no lifetime ring, no cache, no tap, FFmpeg auto threads,
    and get_stats() carries no lever keys (today's payload)."""
    for name in list(os.environ):
        if name.startswith(("WEBRTC_LIFETIME", "WEBRTC_IDLE_FRAME_CACHE", "WEBRTC_PREENCODE",
                            "WEBRTC_DEADLINE", "MUSETALK_THREAD_CAPS", "WEBRTC_NONBLOCKING")):
            os.environ.pop(name)
    track, _clock = lt.call(_make_real_track())
    stats = track.get_stats()
    view = track.counters_view(8)
    ok = (not hasattr(track, "_musetalk_live_handoff") and track._lifetime_send_ring is None
          and track._idle._frame_cache is None and track._idle._decode_threads == 0
          and track._preencode_tap is None and track._deadline_pacing is False
          and not {"lifetime", "live_handoff", "idle_frame_cache_backed"} & set(stats)
          and view["lifetime_enabled"] is False and track.lifetime_stats() is None)
    lt.loop.call_soon_threadsafe(track.stop)
    time.sleep(0.05)
    record("T9_defaults_inert", ok, stats_keys=len(stats), view_keys=sorted(view))


def main() -> int:
    lt = LoopThread()
    try:
        for test in (t1_order, t2_nonblocking, t3_bound, t4_cancel, t5_close, t6_converter,
                     t7_real_track, t8_deadline_pacing_same_content, t9_defaults_inert):
            try:
                test(lt)
            except Exception as exc:  # a crash is a failure, keep going
                import traceback
                traceback.print_exc()
                record(test.__name__, False, error=repr(exc))
    finally:
        lt.stop()
    passed = all(r["result"] == "pass" for r in RESULTS)
    summary = {"suite": "live_handoff_fifo", "passed": passed,
               "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               "results": RESULTS}
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(f"{'PASS' if passed else 'FAIL'} live_handoff_fifo: "
          f"{sum(r['result'] == 'pass' for r in RESULTS)}/{len(RESULTS)} -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
