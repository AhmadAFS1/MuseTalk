#!/usr/bin/env python
"""CPU test for WEBRTC_IDLE_FRAME_CACHE (plan item 1.7), PyAV only, no CUDA.

On real idle clips, the shared pre-decoded cache must be bit-identical to what
IdleVideoStreamTrack.read_frame() decodes today, frame by frame, across loop
boundaries, including every conversion the WebRTC path applies downstream:

  * packed I420 planes (what the encoder receives) - SHA-256 per frame;
  * to_ndarray(format="bgr24") (pose switch / motion builders);
  * reformat(width, height, format="bgr24") (idle crossfade path);
  * frame format, size, color_range, colorspace;
  * loop bookkeeping: get_timing().source_frame_index, next_frame_starts_cycle(),
    last_read_started_cycle(), completed_cycles.

Scenarios: A decode baseline; B cache-backed from frame 0; C switch-over mid-clip
(decoder -> cache when the background build finishes); D skip_frames() ==
repeated read_frame(); E reset(); F LRU budget (eviction when idle, admission
rejection when resident clips are in use -> that session decodes as today);
G MUSETALK_THREAD_CAPS=1 decoder (1 thread) == FFmpeg auto threads.

Writes test_idle_frame_cache_i420.json next to this file; exit 0 only if all pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import scripts.webrtc_idle_frame_cache as cache_mod  # noqa: E402
from scripts.webrtc_tracks import IdleVideoStreamTrack  # noqa: E402

OUT = Path(__file__).with_name("test_idle_frame_cache_i420.json")
CLIPS = {
    "bob_idle_512x896_240f": ROOT / "results/v15/avatars/chinese_bob_pink_bedroom_idle_d4b06da317/input_video.mp4",
    "latina_idle_512x832_241f": ROOT / "results/v15/avatars/latina_guided_20260925_idle_ee675cb4fd/input_video.mp4",
}
RESULTS: list[dict] = []


def record(name, ok, **detail):
    RESULTS.append({"test": name, "result": "pass" if ok else "fail", **detail})
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)}", flush=True)


def sha(array) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def fingerprint(track, frame, full: bool) -> tuple:
    timing = track.get_timing()
    entry = [sha(frame.to_ndarray()), frame.format.name, frame.width, frame.height,
             int(frame.color_range), int(frame.colorspace), timing["source_frame_index"],
             timing["completed_cycles"], track.next_frame_starts_cycle(),
             track.last_read_started_cycle()]
    if full:
        entry.append(sha(frame.to_ndarray(format="bgr24")))
        entry.append(sha(frame.reformat(width=frame.width, height=frame.height,
                                        format="bgr24").to_ndarray()))
    return tuple(entry)


def read_sequence(track, count: int, full_every: int = 1):
    out = []
    started = time.perf_counter()
    for index in range(count):
        frame = track.read_frame()
        out.append(fingerprint(track, frame, full=(index % full_every == 0)))
    return out, (time.perf_counter() - started) / max(1, count)


def set_cache(enabled: bool, max_mb: int = 1024):
    if enabled:
        os.environ["WEBRTC_IDLE_FRAME_CACHE"] = "1"
        os.environ["WEBRTC_IDLE_FRAME_CACHE_MAX_MB"] = str(max_mb)
    else:
        os.environ.pop("WEBRTC_IDLE_FRAME_CACHE", None)
        os.environ.pop("WEBRTC_IDLE_FRAME_CACHE_MAX_MB", None)
    cache_mod.reset_idle_frame_cache_for_tests()


def first_mismatch(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": index, "baseline": x, "candidate": y}
    if len(a) != len(b):
        return {"length": [len(a), len(b)]}
    return None


def run_clip(label: str, path: Path) -> None:
    probe = IdleVideoStreamTrack(str(path), fps=24.0)
    frame_count = probe.get_timing()["source_frame_count"]
    probe.stop()
    total = int(frame_count * 2.5) + 3  # crosses two loop boundaries

    # A: today's per-session decode.
    set_cache(False)
    base_track = IdleVideoStreamTrack(str(path), fps=24.0)
    baseline, decode_s = read_sequence(base_track, total)
    base_track.stop()

    # B: cache-backed from frame 0.
    set_cache(True)
    cache = cache_mod.get_idle_frame_cache()
    cache.request(str(path))
    cache.wait_idle(120)
    track = IdleVideoStreamTrack(str(path), fps=24.0)
    backed = track.cache_backed
    cached, cached_s = read_sequence(track, total)
    track.stop()
    stats = cache.get_stats()
    mismatch = first_mismatch(baseline, cached)
    record(f"B_cache_equals_decode[{label}]", backed and mismatch is None,
           frames=total, clip_frames=frame_count, cache_backed=backed,
           decode_ms_per_frame=round(decode_s * 1000, 3),
           cache_ms_per_frame=round(cached_s * 1000, 3),
           cache_mib=stats["used_mib"], first_mismatch=mismatch)

    # Pure read_frame() cost (what recv() pays on the event loop), no hashing.
    costs = {}
    for mode in ("decoder", "cache"):
        set_cache(mode == "cache")
        if mode == "cache":
            cache = cache_mod.get_idle_frame_cache()
            cache.request(str(path))
            cache.wait_idle(120)
        track = IdleVideoStreamTrack(str(path), fps=24.0)
        started = time.perf_counter()
        cpu0 = time.process_time()
        for _ in range(frame_count):
            track.read_frame()
        costs[mode] = {"wall_ms_per_frame": round((time.perf_counter() - started) / frame_count * 1000, 3),
                       "cpu_ms_per_frame": round((time.process_time() - cpu0) / frame_count * 1000, 3)}
        track.stop()
    record(f"read_frame_cost[{label}]", True, **costs)

    # C: switch-over while the clip is still being built.
    set_cache(True)
    cache = cache_mod.get_idle_frame_cache()
    track = IdleVideoStreamTrack(str(path), fps=24.0)  # miss -> decoder + background build
    started_decoding = not track.cache_backed
    head, _ = read_sequence(track, 37)
    cache.wait_idle(120)
    tail, _ = read_sequence(track, total - 37)
    switched = track.cache_backed
    track.stop()
    mismatch = first_mismatch(baseline, head + tail)
    record(f"C_switch_over_mid_clip[{label}]", started_decoding and switched and mismatch is None,
           started_decoding=started_decoding, switched_to_cache=switched, first_mismatch=mismatch)

    # D: skip_frames == repeated read_frame (cache-backed and decoder-backed).
    for mode in ("cache", "decoder"):
        set_cache(mode == "cache")
        if mode == "cache":
            cache = cache_mod.get_idle_frame_cache()
            cache.request(str(path))
            cache.wait_idle(120)
        track = IdleVideoStreamTrack(str(path), fps=24.0)
        skip = frame_count - 5
        track.skip_frames(skip)
        after, _ = read_sequence(track, 20)
        track.stop()
        mismatch = first_mismatch(baseline[skip:skip + 20], after)
        record(f"D_skip_frames_{mode}[{label}]", mismatch is None, skipped=skip,
               first_mismatch=mismatch)

    # E: reset() restarts at frame 0 with the same bookkeeping.
    results = {}
    for mode in ("decoder", "cache"):
        set_cache(mode == "cache")
        if mode == "cache":
            cache = cache_mod.get_idle_frame_cache()
            cache.request(str(path))
            cache.wait_idle(120)
        track = IdleVideoStreamTrack(str(path), fps=24.0)
        read_sequence(track, 50, full_every=50)
        track.reset()
        results[mode], _ = read_sequence(track, 10)
        track.stop()
    mismatch = first_mismatch(results["decoder"], results["cache"])
    record(f"E_reset[{label}]", mismatch is None and results["cache"][0][0] == baseline[0][0],
           first_mismatch=mismatch)

    # G: thread caps (1 decoder thread) decode identical pixels.
    set_cache(False)
    os.environ["MUSETALK_THREAD_CAPS"] = "1"
    try:
        track = IdleVideoStreamTrack(str(path), fps=24.0)
        threads = track._decode_threads
        capped, _ = read_sequence(track, frame_count + 3, full_every=1000)
        track.stop()
    finally:
        os.environ.pop("MUSETALK_THREAD_CAPS", None)
    mismatch = first_mismatch([b[:1] for b in baseline[:frame_count + 3]], [c[:1] for c in capped])
    record(f"G_thread_caps_decoder_exact[{label}]", threads == 1 and mismatch is None,
           decode_threads=threads, first_mismatch=mismatch)


def lru_budget() -> None:
    bob, latina = CLIPS["bob_idle_512x896_240f"], CLIPS["latina_idle_512x832_241f"]
    set_cache(True, max_mb=200)  # bob 157.5 MiB, latina 147 MiB: only one fits
    cache = cache_mod.get_idle_frame_cache()
    cache.request(str(bob))
    cache.wait_idle(120)
    one = cache.get_stats()
    cache.request(str(latina))  # bob has no readers -> evicted
    cache.wait_idle(120)
    two = cache.get_stats()
    reader = IdleVideoStreamTrack(str(latina), fps=24.0)  # holds latina
    cache.request(str(bob))  # cannot fit while latina is in use -> rejected
    cache.wait_idle(120)
    three = cache.get_stats()
    fallback = IdleVideoStreamTrack(str(bob), fps=24.0)
    fallback_backed = fallback.cache_backed
    fallback.stop()
    reader.stop()
    ok = (one["clips"] == 1 and two["clips"] == 1 and two["evictions"] == 1
          and three["admission_rejected"] >= 1 and not fallback_backed
          and three["used_mib"] <= 200)
    record("F_lru_budget_evict_and_reject", ok,
           after_first=one["clip_detail"], after_second={"evictions": two["evictions"],
                                                         "clips": two["clip_detail"]},
           after_third={"admission_rejected": three["admission_rejected"],
                        "used_mib": three["used_mib"]},
           fallback_session_decodes=not fallback_backed)
    set_cache(False)


def main() -> int:
    for label, path in CLIPS.items():
        if not path.exists():
            record(f"clip_present[{label}]", False, path=str(path))
            continue
        try:
            run_clip(label, path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            record(f"run_clip[{label}]", False, error=repr(exc))
    try:
        lru_budget()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        record("F_lru_budget_evict_and_reject", False, error=repr(exc))
    passed = bool(RESULTS) and all(r["result"] == "pass" for r in RESULTS)
    summary = {"suite": "idle_frame_cache_i420", "passed": passed,
               "clips": {k: str(v) for k, v in CLIPS.items()},
               "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               "results": RESULTS}
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(f"{'PASS' if passed else 'FAIL'} idle_frame_cache_i420: "
          f"{sum(r['result'] == 'pass' for r in RESULTS)}/{len(RESULTS)} "
          f"max_rss={summary['max_rss_mb']}MB -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
