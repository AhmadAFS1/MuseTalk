"""Keep every session of a WebRTC group talking, for a person watching the group wall in a browser.

Each session waits until its browser peer is connected, then plays back-to-back turns from the audio corpus with a
1 s gap, staggered across sessions like real calls. It stops at the deadline or when <run>/STOP_DEMO exists.
  demo_driver.py <base url> <group id> <run dir> [--minutes 45] [--stagger-s 1.5] [--gap-s 1.0]
Writes <run dir>/demo_turns.jsonl (one line per turn). Never prints the group's ICE credentials."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

CORPUS = Path(__file__).resolve().parents[1] / "throughput300_candidate" / "audio_corpus"


async def session_loop(http, base, sid, k, wavs, run, deadline, a, log):
    while time.monotonic() < deadline and not (run / "STOP_DEMO").exists():
        async with http.get(f"{base}/webrtc/sessions/{sid}/status", params={"light": "1"}) as r:
            if r.status == 404:
                return
            status = (await r.json()).get("status")
        if status in ("connected", "streaming"):
            break
        await asyncio.sleep(1.0)
    await asyncio.sleep(k * a.stagger_s)
    turn = 0
    while time.monotonic() < deadline and not (run / "STOP_DEMO").exists():
        wav = wavs[(k + turn) % len(wavs)]
        form = aiohttp.FormData()
        form.add_field("audio_file", wav.read_bytes(), filename=wav.name, content_type="audio/wav")
        t0 = time.monotonic()
        async with http.post(f"{base}/webrtc/sessions/{sid}/stream", data=form) as r:
            code = r.status
        log({"session": k, "turn": turn, "wav": wav.name, "status": code, "t": round(t0, 3),
             "post_s": round(time.monotonic() - t0, 3)})
        if code == 404:
            return
        if code != 200:
            await asyncio.sleep(2.0)  # 409 = already streaming (e.g. the wall's own buttons); try again
            continue
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            async with http.get(f"{base}/webrtc/sessions/{sid}/status", params={"light": "1"}) as r:
                if r.status == 404:
                    return
                if (await r.json()).get("active_stream") is None:
                    break
        turn += 1
        await asyncio.sleep(a.gap_s)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("group")
    ap.add_argument("run")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--stagger-s", type=float, default=1.5)
    ap.add_argument("--gap-s", type=float, default=1.0)
    a = ap.parse_args()
    run = Path(a.run)
    wavs = sorted(CORPUS.glob("*_turn_*.wav"))
    out = open(run / "demo_turns.jsonl", "a", buffering=1)

    def log(row):
        out.write(json.dumps(row) + "\n")

    deadline = time.monotonic() + a.minutes * 60
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        async with http.get(f"{a.base}/webrtc/groups/{a.group}") as r:
            sids = [s["session_id"] for s in (await r.json())["sessions"]]
        print(f"driving {len(sids)} sessions of group {a.group} with {len(wavs)} corpus clips for {a.minutes:.0f} min",
              flush=True)
        await asyncio.gather(*(session_loop(http, a.base, sid, k, wavs, run, deadline, a, log)
                               for k, sid in enumerate(sids)), return_exceptions=True)
    print("driver done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
