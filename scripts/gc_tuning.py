"""Python cyclic-GC instrumentation and tuning for the live server (all opt-in; defaults change nothing).

Why: the live WebRTC server paces every stream from one asyncio event loop. A full (generation-2) collection
scans every tracked object in the process while holding the GIL; with models, 15 avatars and their compose
plans resident, that pause can exceed the 50 ms frame interval and freeze all streams at once
(docs/fps_comparisons/live15_r5_20260929/README.md).

  MUSETALK_GC_LOG=1            log each collection that takes >= MUSETALK_GC_LOG_MIN_MS (default 20) ms,
                               with generation, duration and objects collected; keeps running totals
  MUSETALK_GC_GARBAGE_TYPES=N  diagnostic (with MUSETALK_GC_LOG=1): every Nth (N >= 2) gen-2 collection keeps its garbage
                               (DEBUG_SAVEALL), logs the 12 most common garbage types, then releases it; this names
                               the code that creates reference cycles. The sampled collection pauses longer.
  MUSETALK_GC_FREEZE=1         after startup and after each avatar load, gc.collect() then gc.freeze(): the
                               long-lived heap moves to the permanent generation, so later full collections
                               scan only objects created since (Python >= 3.7). Frame content is unchanged.
  MUSETALK_GC_THRESHOLDS=a,b,c gc.set_threshold(a, b, c) at startup
  MUSETALK_SWITCH_INTERVAL_MS=x sys.setswitchinterval(x / 1000) at startup (CPython default 5 ms): how long a
                               thread that wants the GIL waits before forcing a hand-off, so the event loop gets it
                               back sooner after each syscall while worker threads are busy
  MUSETALK_LOOP_LAG_LOG_MS=N   diagnostic: a task wakes every 5 ms and logs each wake-up that is >= N ms late, with
                               CLOCK_MONOTONIC time, so server loop stalls can be joined to client traces. Cheap.
  MUSETALK_LOOP_STALL_DUMP_MS=N diagnostic: a watchdog thread notices when the loop heartbeat (5 ms) is >= N ms
                               overdue and logs the Python stack of every thread that is not parked in a wait, i.e.
                               who holds or wants the GIL while the loop is stalled (one dump per stall).
  MUSETALK_ASYNCIO_SLOW_MS=N   diagnostic: asyncio debug mode with slow-callback logging (every event-loop callback or
                               task step that holds the loop >= N ms is logged with its identity). Adds overhead;
                               coroutine-origin tracking is switched off to keep it small.
"""
from __future__ import annotations

import gc
import os
import threading
import time

_state = {"installed": False, "start": {}, "slow": 0, "total_ms": {0: 0.0, 1: 0.0, 2: 0.0}, "count": {0: 0, 1: 0, 2: 0},
          "max_ms": {0: 0.0, 1: 0.0, 2: 0.0}, "freezes": 0}
_lock = threading.Lock()


def _flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _callback(phase, info):
    gen = info.get("generation", -1)
    if phase == "start":
        _state["start"][gen] = time.perf_counter()
        every = _state.get("garbage_every", 0)
        if gen == 2 and every > 0:
            _state["gen2_seen"] = _state.get("gen2_seen", 0) + 1
            if _state["gen2_seen"] % every == 0:
                _state["sampling"] = True
                gc.set_debug(gc.get_debug() | gc.DEBUG_SAVEALL)
        return
    if _state.pop("sampling", False):
        gc.set_debug(gc.get_debug() & ~gc.DEBUG_SAVEALL)
        from collections import Counter

        kinds = Counter(f"{type(o).__module__}.{type(o).__qualname__}" for o in gc.garbage)
        n = len(gc.garbage)
        gc.garbage.clear()
        print(f"🧹 GC gen2 garbage sample: {n} objects; top types {kinds.most_common(12)}", flush=True)
    t0 = _state["start"].pop(gen, None)
    if t0 is None:
        return
    ms = (time.perf_counter() - t0) * 1000.0
    with _lock:
        _state["count"][gen] = _state["count"].get(gen, 0) + 1
        _state["total_ms"][gen] = _state["total_ms"].get(gen, 0.0) + ms
        _state["max_ms"][gen] = max(_state["max_ms"].get(gen, 0.0), ms)
    if ms >= _state["min_ms"]:
        _state["slow"] += 1
        print(f"🧹 GC gen{gen} {ms:.1f} ms collected={info.get('collected')} uncollectable={info.get('uncollectable')} "
              f"t_mono={time.monotonic():.3f} counts={gc.get_count()} frozen={gc.get_freeze_count()} "
              f"thread={threading.current_thread().name}", flush=True)


def install() -> None:
    """Call once at import time of the server."""
    if _state["installed"]:
        return
    _state["installed"] = True
    _state["min_ms"] = float(os.getenv("MUSETALK_GC_LOG_MIN_MS", "20"))
    every = int(os.getenv("MUSETALK_GC_GARBAGE_TYPES", "0") or 0)
    # A sampled collection frees nothing (DEBUG_SAVEALL); clearing gc.garbage leaves the cycles for the next
    # collection, so at least every other gen-2 collection must run unsampled or the garbage is never freed.
    _state["garbage_every"] = max(2, every) if every > 0 else 0
    th = os.getenv("MUSETALK_GC_THRESHOLDS", "").strip()
    if th:
        a, b, c = (int(x) for x in th.split(","))
        gc.set_threshold(a, b, c)
        print(f"🧹 GC thresholds set to {gc.get_threshold()}", flush=True)
    sw = os.getenv("MUSETALK_SWITCH_INTERVAL_MS", "").strip()
    if sw:
        import sys

        sys.setswitchinterval(float(sw) / 1000.0)
        print(f"🧹 GIL switch interval set to {sys.getswitchinterval() * 1000:.2f} ms", flush=True)
    if _flag("MUSETALK_GC_LOG"):
        gc.callbacks.append(_callback)
        print(f"🧹 GC logging on (collections >= {_state['min_ms']:.0f} ms); thresholds {gc.get_threshold()}", flush=True)


def freeze(reason: str, collect: bool = True) -> None:
    """MUSETALK_GC_FREEZE=1: move everything alive now into the permanent generation.

    collect=False skips the full collection first (gc.freeze itself only splices lists), for call sites that may
    run while streams are live, e.g. a mid-run avatar load."""
    if not _flag("MUSETALK_GC_FREEZE"):
        return
    t0 = time.perf_counter()
    if collect:
        gc.collect()
    gc.freeze()
    _state["freezes"] += 1
    print(f"🧹 GC freeze ({reason}): {gc.get_freeze_count()} objects frozen in {(time.perf_counter() - t0) * 1000:.0f} ms",
          flush=True)


async def _loop_lag_logger(min_ms: float) -> None:
    import asyncio

    period = 0.005
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(period)
        late_ms = (time.monotonic() - t0 - period) * 1000.0
        if late_ms >= min_ms:
            print(f"⏱️ loop lag {late_ms:.1f} ms t_mono={time.monotonic():.3f}", flush=True)


_IDLE_LEAVES = ("wait", "_wait_for_tstate_lock", "select", "poll", "epoll", "sleep", "acquire", "get", "_worker",
                "accept", "recv", "recv_into", "read", "readline", "join", "_recv_bytes", "_poll", "run_forever",
                "_run_once", "do_select", "_bootstrap_inner", "wait_for", "result")


async def _loop_heartbeat() -> None:
    import asyncio

    while True:
        _state["hb"] = time.monotonic()
        await asyncio.sleep(0.005)


def _stall_watchdog(min_ms: float, loop_thread_id: int) -> None:
    import sys
    import traceback

    in_stall = False
    while True:
        time.sleep(0.005)
        hb = _state.get("hb")
        if hb is None:
            continue
        late_ms = (time.monotonic() - hb) * 1000.0
        if late_ms < min_ms:
            in_stall = False
            continue
        if in_stall:
            continue
        in_stall = True
        names = {t.ident: t.name for t in threading.enumerate()}
        lines = []
        for ident, frame in sys._current_frames().items():
            if ident == threading.get_ident():
                continue
            stack = traceback.extract_stack(frame)
            if not stack:
                continue
            leaf = stack[-1]
            if ident != loop_thread_id and leaf.name in _IDLE_LEAVES:
                continue
            tail = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(stack[-7:]))
            lines.append(f"   [{names.get(ident, ident)}{' LOOP' if ident == loop_thread_id else ''}] {tail}")
        print(f"🩺 loop stall {late_ms:.0f} ms t_mono={time.monotonic():.3f} active threads:\n" + "\n".join(lines),
              flush=True)


def install_loop_diagnostics(loop) -> None:
    """MUSETALK_LOOP_LAG_LOG_MS / MUSETALK_ASYNCIO_SLOW_MS (see the module docstring)."""
    dump_ms = float(os.getenv("MUSETALK_LOOP_STALL_DUMP_MS", "0") or 0)
    if dump_ms > 0:
        _state["hb_task"] = loop.create_task(_loop_heartbeat())
        threading.Thread(target=_stall_watchdog, args=(dump_ms, threading.get_ident()), name="loop-stall-watchdog",
                         daemon=True).start()
        print(f"🩺 loop stall dumps on (heartbeat >= {dump_ms:.0f} ms overdue)", flush=True)
    lag_ms = float(os.getenv("MUSETALK_LOOP_LAG_LOG_MS", "0") or 0)
    if lag_ms > 0:
        _state["lag_task"] = loop.create_task(_loop_lag_logger(lag_ms))
        print(f"⏱️ loop lag logging on (>= {lag_ms:.0f} ms late)", flush=True)
    ms = float(os.getenv("MUSETALK_ASYNCIO_SLOW_MS", "0") or 0)
    if ms <= 0:
        return
    import logging
    import sys

    loop.set_debug(True)
    loop.slow_callback_duration = ms / 1000.0
    sys.set_coroutine_origin_tracking_depth(0)
    lg = logging.getLogger("asyncio")
    lg.setLevel(logging.WARNING)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("🐌 asyncio %(message)s t_mono=%(relativeCreated)d"))
    lg.addHandler(h)
    lg.propagate = False
    print(f"🐌 asyncio slow-callback logging on (>= {ms:.0f} ms)", flush=True)


def stats() -> dict:
    with _lock:
        return {"counts": dict(_state["count"]), "total_ms": {k: round(v, 1) for k, v in _state["total_ms"].items()},
                "max_ms": {k: round(v, 1) for k, v in _state["max_ms"].items()}, "slow": _state["slow"],
                "freezes": _state["freezes"], "frozen": gc.get_freeze_count(), "thresholds": gc.get_threshold()}
