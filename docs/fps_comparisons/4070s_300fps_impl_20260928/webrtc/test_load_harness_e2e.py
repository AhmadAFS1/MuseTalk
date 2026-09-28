#!/usr/bin/env python
"""CPU end-to-end test of load_test_webrtc_v2.py against a fake MuseTalk server.

The fake server (``--serve``) speaks the real HTTP/WebRTC contract the harness
uses (/webrtc/sessions/create, /offer, /stream, /stats?view=lifetime, DELETE)
and streams the REAL SwitchableVideoStreamTrack (real strict-FIFO queue, real
VideoSyncClock in timestamp-locked mode, real lifetime counters and send ring,
real aiortc VP8 encode) from a real idle clip. Only generation is fake: a thread
produces synthetic 512x896 frames at --gen-fps and hands them to the track with
the real non-blocking handoff (or today's blocking run_coroutine_threadsafe
handoff). No model, no CUDA.

Scenarios (each: server + harness, N=3 peers, 2 sequential turns):
  E1 gen 20 fps, non-blocking handoff: fresh fraction ~1, no held runs, cadence 50 ms
  E2 gen 18 fps (a synthetic ~10% hold): the harness must report ~10% held
     through a turn boundary (lifetime counters monotonic)
  E3 gen 18 fps with WEBRTC_LIFETIME_COUNTERS=0: stitched per-turn counters give
     the same held fraction (no send ring)
  E4 --chain (ffmpeg concatenated WAV) with the blocking handoff, N=2
Writes test_load_harness_e2e.json next to this file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
HERE = Path(__file__).resolve().parent
OUT = HERE / "test_load_harness_e2e.json"
IDLE_CLIP = ROOT / "results/v15/avatars/chinese_bob_pink_bedroom_idle_d4b06da317/input_video.mp4"
PY = sys.executable


# ---------------------------------------------------------------------------
# fake server
# ---------------------------------------------------------------------------
def serve(port: int, gen_fps: float, handoff: str) -> None:
    import av
    import numpy as np
    from aiohttp import web
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from scripts.webrtc_live_handoff import get_live_handoff
    from scripts.webrtc_tracks import SwitchableVideoStreamTrack, VideoSyncClock

    sessions: dict = {}
    counters = {"frames_handed_off": 0, "batches_handed_off": 0, "callback_total_s": 0.0,
                "callback_max_ms": 0.0}
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (896, 512, 3), dtype=np.uint8)
    frames = []
    for i in range(16):
        f = base.copy()
        f[:24] = i * 15
        frames.append(f)
    loop_holder = {}

    async def create(request):
        sid = uuid.uuid4().hex[:12]
        clock = VideoSyncClock(source_fps=20.0)
        track = SwitchableVideoStreamTrack(str(IDLE_CLIP), source_fps=20.0, output_fps=20.0,
                                           sync_clock=clock, prebuffer_seconds=0.5,
                                           idle_source_fps=24.0)
        sessions[sid] = {"sid": sid, "user_id": request.query.get("user_id"), "track": track,
                         "clock": clock, "pc": RTCPeerConnection(), "active": None,
                         "avatar_id": request.query.get("avatar_id"),
                         "playback_fps": int(request.query.get("playback_fps", 20))}
        return web.json_response({"session_id": sid, "ice_servers": []})

    async def offer(request):
        s = sessions[request.match_info["sid"]]
        body = await request.json()
        await s["pc"].setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
        s["pc"].addTrack(s["track"])
        answer = await s["pc"].createAnswer()
        await s["pc"].setLocalDescription(answer)
        return web.json_response({"sdp": s["pc"].localDescription.sdp,
                                  "type": s["pc"].localDescription.type})

    async def release(s, total):
        clock = s["clock"]
        clock.set_audio_media_duration(total / 20.0)
        clock.mark_audio_ready()
        clock.mark_video_ready()
        clock.release_playout(time.monotonic() + 0.05)

    async def start(s):
        return s["track"].start_live()

    async def finish(s, request_id):
        try:
            await s["track"].wait_for_playback_complete(timeout=120)
        finally:
            if s["active"] == request_id:
                s["active"] = None

    def generate(s, request_id, total, loop):
        track = s["track"]
        released = []

        def on_pushed(ready):
            if ready and not released:
                released.append(True)
                asyncio.ensure_future(release(s, total))

        if handoff == "nonblocking":
            h = get_live_handoff(track, loop)
            turn = h.begin_turn(lambda: start(s), label=request_id)
        else:
            gen = asyncio.run_coroutine_threadsafe(start(s), loop).result(30)
        t0 = time.monotonic()
        for b in range(0, total, 8):
            batch = [frames[(b + i) % len(frames)] for i in range(min(8, total - b))]
            delay = t0 + b / gen_fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            c0 = time.monotonic()
            if handoff == "nonblocking":
                h.submit_frames(turn, batch, on_pushed=on_pushed)
            else:
                ready = asyncio.run_coroutine_threadsafe(
                    track.push_bgr_frames_batch(batch, generation_id=gen), loop).result(30)
                if ready and not released:
                    released.append(True)
                    asyncio.run_coroutine_threadsafe(release(s, total), loop).result(10)
            cb = time.monotonic() - c0
            counters["frames_handed_off"] += len(batch)
            counters["batches_handed_off"] += 1
            counters["callback_total_s"] += cb
            counters["callback_max_ms"] = max(counters["callback_max_ms"], cb * 1000)

        def complete(gid):
            track.signal_generation_complete(gid)
            if not released:
                asyncio.ensure_future(release(s, total))
            asyncio.ensure_future(finish(s, request_id))

        if handoff == "nonblocking":
            h.submit_marker(turn, complete)
        else:
            loop.call_soon_threadsafe(complete, gen)

    async def stream(request):
        s = sessions[request.match_info["sid"]]
        if s["active"]:
            return web.json_response({"detail": "Session already streaming"}, status=409)
        form = await request.post()
        data = form["audio_file"].file.read()
        tmp = Path(f"/tmp/claude-0/fake_lt2_{uuid.uuid4().hex}.wav")
        tmp.write_bytes(data)
        container = av.open(str(tmp))
        stream_ = container.streams.audio[0]
        duration = float(stream_.duration * stream_.time_base) if stream_.duration else \
            float(container.duration) / 1e6
        container.close()
        tmp.unlink(missing_ok=True)
        total = max(8, int(round(duration * 20)))
        request_id = f"fake_{uuid.uuid4().hex[:8]}"
        s["active"] = request_id
        threading.Thread(target=generate, args=(s, request_id, total, loop_holder["loop"]),
                         daemon=True).start()
        return web.json_response({"request_id": request_id, "session_id": s["sid"], "status": "streaming"})

    async def stats(request):
        ring = int(request.query.get("ring", 0))
        out = []
        for s in sessions.values():
            view = s["track"].counters_view(ring)
            if os.environ.get("FAKE_TURN_VIEW") == "1":
                # Serve what a WEBRTC_LIFETIME_COUNTERS=0 server serves (per-turn counters,
                # no ring) plus the track's real lifetime totals as ground truth.
                t = s["track"]
                view = {"lifetime_enabled": False, "lifetime_truth": t.lifetime_stats(), "turn_counters": {
                    "generation_id": t._live_generation_id, "frames_received": t._frames_received,
                    "frames_played": t._frames_played, "frames_dropped": t._frames_dropped,
                    "frames_duplicated": t._frames_duplicated, "queue_underruns": t._queue_underruns,
                    "strict_video_stalls": t._strict_video_stalls,
                    "strict_video_stall_seconds": t._strict_video_stall_seconds,
                    "output_frames_sent": t._output_frames_sent, "output_fps": t._output_fps,
                    "live_active": t._live_active, "live_released": t._live_released,
                    "queue_size": t._queue.qsize()}}
            out.append({"session_id": s["sid"], "user_id": s["user_id"], "avatar_id": s["avatar_id"],
                        "status": "streaming" if s["active"] else "connected",
                        "active_stream": s["active"], "playback_fps": s["playback_fps"],
                        "counters": view})
        status = Path("/proc/self/status").read_text()
        fields = dict(l.split(":", 1) for l in status.splitlines() if ":" in l)
        return web.json_response({"view": "lifetime", "sessions": out, "server": {
            "pid": os.getpid(), "monotonic": time.monotonic(), "process_cpu_s": time.process_time(),
            "rss_mb": int(fields["VmRSS"].split()[0]) / 1024, "threads": int(fields["Threads"]),
            "lifetime_counters": os.environ.get("WEBRTC_LIFETIME_COUNTERS") == "1",
            "live_handoff_mode": handoff, "live": dict(counters), "loop_lag": {}, "media_flags": {}}})

    async def events(request):
        s = sessions[request.match_info["sid"]]
        body = await request.json()
        if body.get("event") == "assistant_turn_aborted" and s["active"]:
            s["active"] = None
            s["track"].end_live()  # what api_server does for a motion session abort
        return web.json_response({"status": "accepted"})

    async def delete(request):
        s = sessions.pop(request.match_info["sid"], None)
        if s:
            s["track"].stop()
            await s["pc"].close()
        return web.json_response({"status": "deleted"})

    async def on_startup(app):
        loop_holder["loop"] = asyncio.get_running_loop()

    app = web.Application(client_max_size=64 << 20)
    app.on_startup.append(on_startup)
    app.router.add_post("/webrtc/sessions/create", create)
    app.router.add_post("/webrtc/sessions/{sid}/offer", offer)
    app.router.add_post("/webrtc/sessions/{sid}/stream", stream)
    app.router.add_get("/webrtc/sessions/stats", stats)
    app.router.add_delete("/webrtc/sessions/{sid}", delete)
    app.router.add_post("/webrtc/sessions/{sid}/events", events)
    web.run_app(app, host="127.0.0.1", port=port, print=None)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_corpus(directory: Path, count: int = 8, seconds: float = 6.0) -> Path:
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(count):
        path = directory / f"{i:02d}_turn_tone.wav"
        t = np.arange(int(16000 * seconds)) / 16000.0
        pcm = (0.3 * np.sin(2 * np.pi * (180 + 20 * i) * t) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        entries.append({"file": path.name, "source": str(path), "class": "turn", "duration_s": seconds,
                        "sha256": str(i), "voice": "tone"})
    (directory / "manifest.json").write_text(json.dumps({"entries": entries}))
    return directory


def run_scenario(name: str, gen_fps: float, handoff: str, lifetime: bool, harness_args: list,
                 work: Path, corpus: str = "corpus_short", extra_env: dict | None = None) -> dict:
    port = free_port()
    env = dict(os.environ)
    env["WEBRTC_LIFETIME_COUNTERS"] = "1" if lifetime else "0"
    env.update(extra_env or {})
    log = open(work / f"{name}_server.log", "w")
    server = subprocess.Popen([PY, __file__, "--serve", "--port", str(port), "--gen-fps", str(gen_fps),
                               "--handoff", handoff], env=env, stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(ROOT))
    try:
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/webrtc/sessions/stats?view=lifetime", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        out_dir = work / name
        cmd = [PY, str(ROOT / "load_test_webrtc_v2.py"), "--base-url", f"http://127.0.0.1:{port}",
               "--label", name, "--out-dir", str(out_dir), "--audio-dir", str(work / corpus),
               "--ignore-ice-servers", "--cooldown-s", "0", *harness_args]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        (work / f"{name}_harness.log").write_text(proc.stdout + "\n" + proc.stderr)
        verdict_lines = [l for l in proc.stdout.splitlines() if l.startswith(("PASS level", "FAIL level",
                                                                              "INVALID level"))]
        level_json = sorted(out_dir.glob(f"{name}_n*.json"))
        summary = json.loads(level_json[0].read_text()) if level_json else None
        return {"rc": proc.returncode, "verdict_line": verdict_lines[-1] if verdict_lines else None,
                "summary": summary, "stderr_tail": proc.stderr[-1500:] if not summary else None}
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--gen-fps", type=float, default=20.0)
    ap.add_argument("--handoff", default="nonblocking")
    ap.add_argument("--only", default="", help="comma list of scenario prefixes (E1,E2,...)")
    args = ap.parse_args()
    only = [x for x in args.only.split(",") if x]

    def wanted(name):
        return not only or any(name.startswith(prefix) for prefix in only)
    if args.serve:
        serve(args.port, args.gen_fps, args.handoff)
        return 0
    work = Path("/tmp/claude-0") / f"lt2_e2e_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    make_corpus(work / "corpus_short", count=8, seconds=6.0)
    make_corpus(work / "corpus_long", count=8, seconds=30.0)
    common = ["--turns", "2", "--turn-gap-s", "1.5", "--warmup-s", "1", "--min-steady-s", "5",
              "--client-cores", "12,13,14,15"]
    results = []

    def record(name, ok, **detail):
        results.append({"test": name, "result": "pass" if ok else "fail", **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:600]}", flush=True)

    def brief(r):
        s = r["summary"]
        if not s:
            return {"rc": r["rc"], "stderr": r["stderr_tail"], "verdict": r["verdict_line"]}
        return {"verdict": s["verdict"], "failed_checks": [k for k, v in s["checks"].items() if not v],
                "invalid": [k for k, v in s["validity"].items() if not v],
                "unmeasured": s.get("unmeasured"),
                "effective_output_fps": [st["server_ring"].get("effective_output_fps") for st in s["streams"]],
                "slot_deficit": [st["server_ring"].get("slot_deficit") for st in s["streams"]],
                "fresh_fraction_min": s["aggregate"]["fresh_fraction_min"],
                "fresh_fractions": [st["server_ring"]["fresh_fraction"] for st in s["streams"]],
                "counter_fresh_fractions": [st["server_counters"].get("fresh_fraction") for st in s["streams"]],
                "max_held_run": s["aggregate"]["max_held_run"],
                "send_interval_max_s": s["aggregate"]["send_interval_max_s"],
                "agg_fresh_fps": s["aggregate"].get("aggregate_fresh_fps_mean"),
                "generated_fps": s["aggregate"].get("generated_fps_server"),
                "speakers_max": s["aggregate"].get("speakers_max"),
                "first_frame_p95_s": s["aggregate"].get("first_frame_p95_s"),
                "turns_ok": [st["turns_ok"] for st in s["streams"]],
                "stitch_resets": [st["stitch_resets"] for st in s["streams"]],
                "client_lag_p99_ms": s["client"]["loop_lag_p99_ms_max"],
                "output_fps_ok": s["checks"]["output_fps_20"], "steady_s": s["window"]["steady_state_s"],
                "server_rss_max_mb": s["server"].get("server_rss_max_mb")}

    if wanted("E1"):
        e1(work, common, record, brief)
    if wanted("E2"):
        e2(work, common, record, brief)
    if wanted("E3"):
        e3(work, common, record, brief)
    if wanted("E4"):
        e4(work, common, record, brief)
    if wanted("E5"):
        e5(work, common, record, brief)
    if wanted("E6"):
        e6(work, common, record, brief)
    passed = bool(results) and all(r["result"] == "pass" for r in results)
    previous = json.loads(OUT.read_text()).get("results", []) if (only and OUT.exists()) else []
    merged = [r for r in previous if not any(r["test"].startswith(x) for x in only)] + results
    OUT.write_text(json.dumps({"suite": "load_harness_e2e_fake_server",
                               "passed": all(r["result"] == "pass" for r in merged),
                               "work_dir": str(work), "results": merged}, indent=2, default=str))
    print(f"{'PASS' if passed else 'FAIL'} load_harness_e2e {sum(r['result'] == 'pass' for r in results)}/{len(results)} -> {OUT}")
    return 0 if passed else 1


def e1(work, common, record, brief):
    r1 = run_scenario("E1_gen20_nonblocking", 20.0, "nonblocking", True, ["--levels", "3", *common], work)
    b1 = brief(r1)
    # Only the cadence-average check may fail here: today's non-motion recv() pacing
    # re-anchors on wake-up, so asyncio oversleep makes it ~51 ms (see issues).
    ok1 = (r1["summary"] is not None and b1["fresh_fraction_min"] is not None
           and b1["fresh_fraction_min"] >= 0.97 and b1["turns_ok"] == [2, 2, 2]
           and b1["output_fps_ok"] and b1["speakers_max"] == 3
           and set(b1["failed_checks"]) <= {"send_interval_avg"} and not b1["invalid"])
    record("E1_harness_end_to_end_clean", ok1, **b1)


def e2(work, common, record, brief):
    # 30 s turns: the 10-frame prebuffer covers the first ~5 s, then ~10% of slots hold.
    r2 = run_scenario("E2_gen18_hold10", 18.0, "nonblocking", True, ["--levels", "3", *common], work,
                      corpus="corpus_long")
    b2 = brief(r2)
    ffs = [f for f in (b2.get("fresh_fractions") or []) if f is not None]
    cf2 = [f for f in (b2.get("counter_fresh_fractions") or []) if f is not None]
    ok2 = (r2["summary"] is not None and ffs and all(0.85 <= f <= 0.95 for f in ffs)
           and b2["verdict"] == "FAIL" and "fresh_fraction" in b2["failed_checks"]
           and b2["turns_ok"] == [2, 2, 2] and len(cf2) == len(ffs)
           and all(abs(a - b) < 0.02 for a, b in zip(ffs, cf2)))
    record("E2_synthetic_10pct_hold_detected_across_turns", ok2, **b2)


def e3(work, common, record, brief):
    # The server serves only per-turn counters (as with WEBRTC_LIFETIME_COUNTERS=0) and
    # the true lifetime totals on the side; the harness must stitch them exactly.
    r3 = run_scenario("E3_gen18_no_lifetime", 18.0, "nonblocking", True, ["--levels", "3", *common], work,
                      corpus="corpus_long", extra_env={"FAKE_TURN_VIEW": "1"})
    b3 = brief(r3)
    exact = []
    for st in (r3["summary"] or {}).get("streams", []):
        truth = (st.get("counters_view_last") or {}).get("lifetime_truth") or {}
        stitched = st.get("stitched_totals_end") or {}
        # frames_played / frames_duplicated freeze between turns, so stitching is exact;
        # output_frames_sent also counts idle frames up to the reset (reported, not gated).
        exact.append({k: (stitched.get(k), truth.get(k)) for k in ("frames_played", "frames_duplicated")})
        b3.setdefault("output_frames_sent_stitched_vs_truth", []).append(
            (stitched.get("output_frames_sent"), truth.get("output_frames_sent")))
    b3["stitched_vs_truth"] = exact
    cffs = [f for f in (b3.get("counter_fresh_fractions") or []) if f is not None]
    ok3 = (r3["summary"] is not None and cffs and all(0.85 <= f <= 0.95 for f in cffs)
           and all((x or 0) >= 1 for x in b3["stitch_resets"])
           and "fresh_fraction" in b3["failed_checks"] and 0.85 <= (b3["fresh_fraction_min"] or 0) <= 0.95
           and b3["unmeasured"] == ["held_run", "send_interval_max", "send_interval_avg"]
           and exact and all(a == b for e in exact for (a, b) in e.values()))
    record("E3_stitched_per_turn_counters", ok3, **b3)


def e4(work, common, record, brief):
    r4 = run_scenario("E4_chain_blocking", 20.0, "blocking", True,
                      ["--levels", "2", "--chain", "--chain-seconds", "14", "--settle-s", "1",
                       "--min-steady-s", "5", "--client-cores", "12,13,14,15"], work)
    b4 = brief(r4)
    chain = (r4["summary"] or {}).get("audio", {}).get("chain") or []
    ok4 = (r4["summary"] is not None and b4["turns_ok"] == [1, 1] and len(chain) == 2
           and all(c["duration_s"] >= 14 for c in chain) and (b4["fresh_fraction_min"] or 0) >= 0.97)
    record("E4_chain_mode_blocking_handoff", ok4, chain=[{k: c[k] for k in ("duration_s", "clips")} for c in chain],
           **b4)


def e5(work, common, record, brief):
    # WEBRTC_DEADLINE_PACING=1: same content, average send interval back on 50 ms.
    r5 = run_scenario("E5_deadline_pacing", 20.0, "nonblocking", True, ["--levels", "3", *common], work,
                      extra_env={"WEBRTC_DEADLINE_PACING": "1"})
    b5 = brief(r5)
    ok5 = (r5["summary"] is not None and b5["verdict"] == "PASS" and not b5["failed_checks"]
           and all(19.9 <= (f or 0) <= 20.1 for f in b5["effective_output_fps"])
           and (b5["fresh_fraction_min"] or 0) >= 0.97)
    record("E5_deadline_pacing_cadence_50ms", ok5, **b5)



def e6(work, common, record, brief):
    # S2/S5 plumbing: Poisson gaps for 50% duty, every turn barged in 1.5-2.5 s after accept.
    r6 = run_scenario("E6_duty_bargein", 20.0, "nonblocking", True,
                      ["--levels", "2", "--turns", "3", "--duty", "0.5", "--barge-in-fraction", "1.0",
                       "--barge-in-min-s", "1.5", "--barge-in-max-s", "2.5", "--warmup-s", "1",
                       "--min-steady-s", "1", "--client-cores", "12,13,14,15"], work)
    b6 = brief(r6)
    s = r6["summary"] or {}
    aborts = []
    for st in s.get("streams", []):
        aborts.extend(st.get("barge_in") or [])
    returns = [a["return_to_idle_s"] for a in aborts if a.get("return_to_idle_s") is not None]
    ok6 = (bool(s) and b6["turns_ok"] == [3, 3] and len(aborts) == 6
           and all(a["abort_status"] == 200 for a in aborts) and len(returns) == 6
           and max(returns) < 0.5)
    record("E6_duty_and_barge_in", ok6, barge_in=aborts,
           barge_in_return_p95_s=s.get("aggregate", {}).get("barge_in_return_p95_s"), **b6)


if __name__ == "__main__":
    sys.exit(main())
