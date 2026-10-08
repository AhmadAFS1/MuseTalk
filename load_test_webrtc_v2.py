#!/usr/bin/env python
"""MuseTalk WebRTC load harness v2 (300 fps plan items 0.4 and 0.5).

Why a v2: load_test_webrtc.py records client receive intervals only. Held
(repeated) frames still arrive on cadence, so it cannot see them, and
start_live() zeroes every per-turn counter at each turn. v2 measures on the
server side:

  * polls GET /webrtc/sessions/stats?view=lifetime&ring=64 once per second:
    fresh = delta frames_played, held = delta frames_duplicated,
    stall = delta strict_video_stall_seconds (monotonic lifetime counters,
    WEBRTC_LIFETIME_COUNTERS=1; without them the per-turn counters are
    stitched across start_live() resets), plus the per-track ring of server
    send stamps (kind f=fresh, h=held, p=live-prebuffer idle, i=idle) for exact
    server-side cadence and fresh fraction;
  * first-frame latency per turn = server first live send - client POST time
    (both CLOCK_MONOTONIC on the same box);
  * server RSS / threads / CPU (from /proc/<pid>) and GPU util / VRAM / power /
    SM clock (nvidia-smi, no CUDA context) at 1 Hz;
  * client shards of <= 5 peers per process, pinned to the client physical
    cores (default cores 12-15 = CPUs {12-15, 28-31}, verified from sysfs),
    each with its own event-loop-lag metric (run INVALID if p99 > 20 ms).

Audio: distinct pre-synthesized WAVs from --audio-dir (manifest.json from
--build-corpus, or *.wav). No TTS is ever called. --turns N posts N sequential
turns per stream; --chain concatenates each stream's turns into one long WAV
of >= --chain-seconds so every stream speaks for the whole window (S1).

Examples
  # build the >= 20-WAV corpus (byte-identical copies + manifest, no audio is re-encoded)
  python load_test_webrtc_v2.py --build-corpus experiments/throughput300_candidate/audio_corpus

  # S1-style burst: 15 streams, chained speech, >= 180 s steady state
  python load_test_webrtc_v2.py --base-url http://127.0.0.1:8300 \\
      --avatar-ids japanese_realtime_talking_7d94520b7f --levels 15 --chain \\
      --audio-dir experiments/throughput300_candidate/audio_corpus --out-dir /tmp/lt

  # metric self-test (synthetic 10% hold, turn-boundary resets), no server
  python load_test_webrtc_v2.py --selftest --selftest-out /tmp/selftest.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from array import array
from contextlib import suppress
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent
SCHEMA = "load_test_webrtc_v2/1"
MARK = "@@LT2 "  # shard -> coordinator protocol line prefix
DEFAULT_CLIENT_CORES = "12,13,14,15"
WALL_POSE_PLAN = {"version": 2, "clock": "audio_progress",
                  "segments": [{"at_permille": 0, "pose_id": "speaking_direct"}],
                  "switch_mode": "next_boundary", "on_complete": "neutral_resting"}
# Read-only sources of existing pre-synthesized speech (the worktree's own data/audio first;
# /workspace/MuseTalk/generated only exists in the main checkout and is only read).
CORPUS_ROOTS = ("/workspace/experiments", str(ROOT / "data/audio"), "/workspace/MuseTalk/generated")
CORPUS_EXCLUDE = ("silence", "chirp", "zeros", "_webrtc")


# ===========================================================================
# small utilities
# ===========================================================================
def now() -> float:
    return time.monotonic()


def pct(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return ordered[index]


def rnd(value, digits=4):
    return None if value is None else round(float(value), digits)


def mem_available_gb() -> float:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return float("nan")


def parse_cpu_list(text: str) -> list:
    cpus = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def client_cpus_for_cores(cores_text: str) -> dict:
    """Logical CPUs of the requested physical cores, verified from sysfs + lscpu."""
    cores = [int(c) for c in cores_text.split(",") if c.strip()]
    base = Path("/sys/devices/system/cpu")
    by_core: dict = {}
    for cpu_dir in sorted(base.glob("cpu[0-9]*")):
        try:
            core = int((cpu_dir / "topology/core_id").read_text())
            package = int((cpu_dir / "topology/physical_package_id").read_text())
        except OSError:
            continue
        by_core.setdefault((package, core), []).append(int(cpu_dir.name[3:]))
    cpus, detail = [], {}
    for core in cores:
        siblings = sorted(by_core.get((0, core), []))
        detail[str(core)] = siblings
        cpus.extend(siblings)
    lscpu = None
    with suppress(Exception):
        out = subprocess.run(["lscpu", "-p=CPU,CORE"], capture_output=True, text=True, timeout=10).stdout
        pairs = [l.split(",") for l in out.splitlines() if l and not l.startswith("#")]
        lscpu = {str(core): sorted(int(c) for c, k in pairs if int(k) == core) for core in cores}
    return {"cores": cores, "cpus": sorted(cpus), "siblings": detail, "lscpu": lscpu,
            "lscpu_agrees": lscpu == detail if lscpu is not None else None}


def pin_to(cpus) -> Optional[str]:
    if not cpus:
        return None
    try:
        os.sched_setaffinity(0, set(cpus))
        return None
    except Exception as exc:  # pragma: no cover
        return str(exc)


# ===========================================================================
# audio corpus (plan item 0.5): existing WAVs only, never TTS
# ===========================================================================
def _probe_wav(path: Path) -> Optional[dict]:
    import av
    import numpy as np
    try:
        container = av.open(str(path))
    except Exception:
        return None
    try:
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate or stream.codec_context.sample_rate)
        channels = int(stream.codec_context.channels or 1)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    except Exception:
        return None
    finally:
        container.close()
    if not chunks:
        return None
    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    duration = len(pcm) / 16000.0
    hop = 320  # 20 ms
    frames = len(pcm) // hop
    if frames < 10:
        return None
    rms = np.sqrt(np.mean(pcm[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    active = db > -45.0
    pauses, run = 0, 0
    for flag in active:
        if flag:
            if run >= 10:  # >= 200 ms of silence between speech
                pauses += 1
            run = 0
        else:
            run += 1
    return {"duration_s": round(duration, 3), "sample_rate": sample_rate, "channels": channels,
            "speech_fraction": round(float(active.mean()), 3), "pauses_200ms": int(pauses),
            "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(pcm)) + 1e-9)), 1)}


def _voice_label(path: Path) -> str:
    text = str(path).lower()
    for tag in ("af_heart", "am_michael", "aoede", "kokoro", "soulx", "ltx", "chinese_bob",
                "japanese", "latina", "indian", "yongen", "sun.wav", "eng", "outputnew"):
        if tag in text:
            return tag.replace(".wav", "")
    return path.parent.name


def build_corpus(out_dir: Path, roots=CORPUS_ROOTS, min_turn: int = 20) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_dir():
            candidates.extend(sorted(root_path.rglob("*.wav")))
    seen, entries = set(), []
    for path in candidates:
        name = path.name.lower()
        if any(tag in name for tag in CORPUS_EXCLUDE) or path.stat().st_size < 40_000:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        info = _probe_wav(path)
        if info is None or info["speech_fraction"] < 0.35:
            continue
        duration = info["duration_s"]
        klass = "turn" if 3.0 <= duration <= 24.0 else ("long" if 24.0 < duration <= 300.0 else None)
        if klass is None:
            continue
        seen.add(digest)
        entries.append({"source": str(path), "sha256": digest, "class": klass,
                        "voice": _voice_label(path), **info})
    entries.sort(key=lambda e: (e["class"] != "turn", e["voice"], e["source"]))
    for old in out_dir.glob("*.wav"):
        old.unlink()
    for index, entry in enumerate(entries):
        slug = "".join(c if c.isalnum() else "_" for c in Path(entry["source"]).stem)[:40]
        target = out_dir / f"{index:02d}_{entry['class']}_{slug}.wav"
        # Byte-identical copy (sha256 recorded), so the corpus survives cleanups elsewhere.
        shutil.copyfile(entry["source"], target)
        entry["file"] = target.name
    turn = [e for e in entries if e["class"] == "turn"]
    manifest = {
        "schema": "webrtc_audio_corpus/1", "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": "existing pre-synthesized WAVs only (no TTS); distinct by SHA-256; speech_fraction >= 0.35; "
                "turn = 3-24 s, long = 24-300 s; names containing silence/chirp/zeros/_webrtc excluded",
        "roots": list(roots), "turn_count": len(turn), "long_count": len(entries) - len(turn),
        "voices": sorted({e["voice"] for e in turn}),
        "turn_seconds_total": round(sum(e["duration_s"] for e in turn), 1),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = len(turn) >= min_turn
    print(f"{'PASS' if ok else 'FAIL'} corpus: {len(turn)} distinct turn WAVs (need >= {min_turn}), "
          f"{manifest['long_count']} long, voices={len(manifest['voices'])}, "
          f"{manifest['turn_seconds_total']} s of turn audio -> {out_dir}/manifest.json")
    return manifest


def load_corpus(audio_dir: Path, klass: str = "turn") -> list:
    manifest = audio_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        entries = [e for e in data["entries"] if klass == "any" or e["class"] == klass]
        for e in entries:
            e["path"] = str(audio_dir / e["file"])
        return entries
    return [{"path": str(p), "file": p.name, "duration_s": None, "sha256": None}
            for p in sorted(audio_dir.glob("*.wav"))]


def plan_turns(corpus: list, streams: int, turns: int, seed: int) -> tuple:
    """Stream i, turn t -> corpus[(i + t*streams + offset) % M]: all distinct while
    streams*turns <= M; otherwise the reuse count is reported."""
    rng = random.Random(seed)
    offset = rng.randrange(len(corpus)) if corpus else 0
    plan = [[corpus[(i + t * streams + offset) % len(corpus)] for t in range(turns)]
            for i in range(streams)]
    uses = {}
    for row in plan:
        for entry in row:
            uses[entry["path"]] = uses.get(entry["path"], 0) + 1
    return plan, max(uses.values()) if uses else 0


def build_chain_wav(corpus: list, start: int, target_s: float, out_path: Path) -> dict:
    """Concatenate distinct corpus clips (rotation from ``start``) to >= target_s
    as one 16 kHz mono WAV (ffmpeg, CPU only)."""
    chosen, total = [], 0.0
    index = start
    while total < target_s and len(chosen) < 500:
        entry = corpus[index % len(corpus)]
        chosen.append(entry)
        total += float(entry.get("duration_s") or 10.0)
        index += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for entry in chosen:
        cmd += ["-i", entry["path"]]
    parts = "".join(f"[{i}:a]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a{i}];"
                    for i in range(len(chosen)))
    cmd += ["-filter_complex", parts + "".join(f"[a{i}]" for i in range(len(chosen)))
            + f"concat=n={len(chosen)}:v=0:a=1[out]", "-map", "[out]", "-ac", "1", "-ar", "16000",
            str(out_path)]
    subprocess.run(cmd, check=True, timeout=300)
    return {"path": str(out_path), "duration_s": round(total, 3),
            "clips": [e.get("file") or e["path"] for e in chosen]}


# ===========================================================================
# metric math (pure; covered by --selftest)
# ===========================================================================
LIFETIME_KEYS = ("frames_played", "frames_duplicated", "strict_video_stall_seconds",
                 "queue_underruns", "frames_received", "output_frames_sent")


class CounterStitcher:
    """Monotonic totals from either lifetime counters or per-turn counters.

    Per-turn counters (WEBRTC_LIFETIME_COUNTERS=0) are zeroed by start_live()
    (all of them) and end_live() (frames_received only). A reset is detected per
    counter when it decreases, and that counter's last value before the reset is
    folded into its base. Frames played after the last poll of a turn and
    before its reset would be lost, but end_live() does not zero frames_played,
    so with >= 1 poll between turns it is exact for frames_played /
    frames_duplicated / stall seconds (frozen between turns); output_frames_sent
    keeps counting idle frames until start_live(), so it can lose up to one poll
    interval of idle frames per turn. A reset followed by a count
    that already exceeds the previous turn's total within one poll interval
    (previous turn < 1 s) is not detectable."""

    def __init__(self):
        self.base = dict.fromkeys(LIFETIME_KEYS, 0.0)
        self.last = None
        self.last_generation = None
        self.resets = 0

    def update(self, counters_view: dict) -> dict:
        if counters_view.get("lifetime_enabled"):
            life = counters_view.get("lifetime") or {}
            return {k: float(life.get(k) or 0.0) for k in LIFETIME_KEYS}
        turn = counters_view.get("turn_counters") or {}
        current = {k: float(turn.get(k) or 0.0) for k in LIFETIME_KEYS}
        generation = turn.get("generation_id")
        if self.last is not None:
            folded = False
            for k in LIFETIME_KEYS:
                if current[k] < self.last[k] - 1e-9:
                    self.base[k] += self.last[k]
                    folded = True
            if folded:
                self.resets += 1
        self.last = current
        self.last_generation = generation
        return {k: self.base[k] + current[k] for k in LIFETIME_KEYS}


class RingStitcher:
    """Per-track server send stamps (seq, t, kind) merged across polls."""

    def __init__(self):
        self.seq = array("q")
        self.t = array("d")
        self.kind = bytearray()
        self.pts = array("q")
        self.gaps = 0
        self.missing = 0

    def add(self, entries) -> None:
        for entry in entries or ():
            seq, stamp, kind, pts = int(entry[0]), float(entry[1]), str(entry[2]), int(entry[3])
            if self.seq and seq <= self.seq[-1]:
                continue
            if self.seq and seq != self.seq[-1] + 1:
                self.gaps += 1
                self.missing += seq - self.seq[-1] - 1
            self.seq.append(seq)
            self.t.append(stamp)
            self.kind.append(ord(kind[0]))
            self.pts.append(pts)


def summarize_ring(ring: RingStitcher, w0: float, w1: float, fps: float) -> dict:
    """Fresh fraction, held runs and server send cadence inside [w0, w1].

    Speaking slots are sends of kind f (fresh) or h (held). slot_deficit sums,
    per contiguous speaking segment, the 20 fps wall-clock slots that were never
    sent (strict-FIFO stalls, pacing drift)."""
    speaking = (ord("f"), ord("h"))
    fresh = held = 0
    held_run = max_held_run = 0
    intervals = []
    segments = []  # [first_t, last_t, slots]
    prev_t = prev_speaking = None
    for i in range(len(ring.seq)):
        t = ring.t[i]
        kind = ring.kind[i]
        is_speaking = kind in speaking
        inside = w0 <= t <= w1
        contiguous = i > 0 and ring.seq[i] == ring.seq[i - 1] + 1
        if inside and is_speaking:
            if kind == ord("f"):
                fresh += 1
                held_run = 0
            else:
                held += 1
                held_run += 1
                max_held_run = max(max_held_run, held_run)
            if prev_t is not None and prev_speaking and contiguous:
                intervals.append(t - prev_t)
                segments[-1][1] = t
                segments[-1][2] += 1
            else:
                segments.append([t, t, 1])
        elif not is_speaking:
            held_run = 0
        prev_t, prev_speaking = t, (inside and is_speaking)
    slots = fresh + held
    span = sum(seg[1] - seg[0] for seg in segments)
    deficit = sum(max(0, int(round((seg[1] - seg[0]) * fps)) + 1 - seg[2]) for seg in segments)
    avg = sum(intervals) / len(intervals) if intervals else None
    return {
        "speaking_slots": slots, "fresh": fresh, "held": held,
        "fresh_fraction": rnd(fresh / slots, 6) if slots else None,
        "max_held_run": max_held_run,
        "speaking_segments": len(segments),
        "speaking_span_s": rnd(span, 3),
        "slot_deficit": deficit,
        "effective_output_fps": rnd(1.0 / avg, 3) if avg else None,
        "send_interval_avg_s": rnd(avg, 5),
        "send_interval_p99_s": rnd(pct(intervals, 0.99), 5),
        "send_interval_max_s": rnd(max(intervals), 5) if intervals else None,
        "send_gaps_over_100ms": sum(1 for v in intervals if v > 0.100),
        "ring_gaps": ring.gaps, "ring_missing_entries": ring.missing,
    }


def fresh_series_from_rings(rings: dict, w0: float, w1: float) -> dict:
    """Per-second aggregate fresh frames and concurrent speakers from rings."""
    if w1 <= w0:
        return {"seconds": 0}
    bins = int(math.floor(w1 - w0))
    fresh = [0] * max(1, bins)
    speakers = [set() for _ in range(max(1, bins))]
    for sid, ring in rings.items():
        for i in range(len(ring.seq)):
            t = ring.t[i]
            if not (w0 <= t < w0 + bins):
                continue
            b = int(t - w0)
            kind = ring.kind[i]
            if kind == ord("f"):
                fresh[b] += 1
            if kind in (ord("f"), ord("h")):
                speakers[b].add(sid)
    counts = [len(s) for s in speakers]
    return {"seconds": bins, "aggregate_fresh_fps_mean": rnd(sum(fresh) / bins, 3) if bins else None,
            "aggregate_fresh_fps_min_1s": min(fresh) if bins else None,
            "aggregate_fresh_fps_p05_1s": pct(fresh, 0.05) if bins else None,
            "speakers_min": min(counts) if bins else None, "speakers_max": max(counts) if bins else None,
            "speakers_mean": rnd(sum(counts) / bins, 3) if bins else None,
            "fresh_per_second": fresh, "speakers_per_second": counts}


def counter_deltas(samples: list, w0: float, w1: float) -> dict:
    """Window deltas and per-second rates from stitched counter samples [(t, totals)]."""
    inside = [(t, c) for t, c in samples if w0 <= t <= w1]
    if len(inside) < 2:
        return {"samples": len(inside)}
    (t0, c0), (t1, c1) = inside[0], inside[-1]
    dt = max(1e-9, t1 - t0)
    per_second_fresh, per_second_held = [], []
    for (ta, ca), (tb, cb) in zip(inside, inside[1:]):
        step = max(1e-9, tb - ta)
        per_second_fresh.append((cb["frames_played"] - ca["frames_played"]) / step)
        per_second_held.append((cb["frames_duplicated"] - ca["frames_duplicated"]) / step)
    fresh = c1["frames_played"] - c0["frames_played"]
    held = c1["frames_duplicated"] - c0["frames_duplicated"]
    return {"samples": len(inside), "span_s": rnd(dt, 3),
            "fresh": fresh, "held": held,
            "fresh_fraction": rnd(fresh / (fresh + held), 6) if fresh + held > 0 else None,
            "fresh_fps": rnd(fresh / dt, 3), "held_fps": rnd(held / dt, 3),
            "stall_seconds": rnd(c1["strict_video_stall_seconds"] - c0["strict_video_stall_seconds"], 4),
            "underruns": c1["queue_underruns"] - c0["queue_underruns"],
            "fresh_fps_min_1s": rnd(min(per_second_fresh), 2),
            "held_fps_max_1s": rnd(max(per_second_held), 2)}


# ===========================================================================
# shard worker: <= 5 peers per process
# ===========================================================================
def emit(event: dict) -> None:
    sys.stdout.write(MARK + json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


class LoopLag:
    def __init__(self, interval=0.05, spike_ms=20.0):
        self.interval = interval
        self.samples = array("d")
        self.spike_ms = spike_ms
        self.spikes = array("d")  # flat (t, lag_ms) pairs above spike_ms, for attributing client-side gaps

    async def run(self, stop: asyncio.Event):
        while not stop.is_set():
            t = now()
            await asyncio.sleep(self.interval)
            lag_ms = max(0.0, (now() - t - self.interval) * 1000.0)
            self.samples.append(lag_ms)
            if lag_ms > self.spike_ms:
                self.spikes.extend((now(), lag_ms))

    def summary(self):
        values = list(self.samples)
        return {"samples": len(values), "p50_ms": rnd(pct(values, 0.5), 3),
                "p99_ms": rnd(pct(values, 0.99), 3), "max_ms": rnd(max(values), 3) if values else None}


class ArrivalRecorder:
    """Writes one stream's decoded frames to MP4 with pts = arrival time (ms), in a background thread.

    Starts record_at_s after the stream connects and runs record_seconds. A delivery stall therefore shows as
    a frozen picture of the same length, unlike a fixed-fps recorder that erases gaps."""

    def __init__(self, path: Path, start_t: float, seconds: float):
        import queue
        import threading
        self.path, self.start_t, self.end_t = path, start_t, start_t + seconds
        self.q = queue.Queue(maxsize=200)
        self.dropped = 0
        self.written = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def offer(self, t: float, frame) -> None:
        if self.start_t <= t <= self.end_t:
            try:
                self.q.put_nowait((t, frame.to_ndarray(format="rgb24")))
            except Exception:
                self.dropped += 1
        elif t > self.end_t and self.q is not None:
            with suppress(Exception):
                self.q.put_nowait(None)

    def _run(self) -> None:
        import av
        from fractions import Fraction
        container = stream = None
        t0 = None
        last_pts = -1
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                t, img = item
                if container is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.path), "w")
                    stream = container.add_stream("libx264", rate=1000)
                    stream.width, stream.height = img.shape[1], img.shape[0]
                    stream.pix_fmt = "yuv420p"
                    stream.time_base = Fraction(1, 1000)
                    stream.options = {"crf": "18", "preset": "veryfast"}
                    t0 = t
                pts = int(round((t - t0) * 1000))
                if pts <= last_pts:
                    pts = last_pts + 1
                last_pts = pts
                vf = av.VideoFrame.from_ndarray(img, format="rgb24")
                vf.pts, vf.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(vf):
                    container.mux(packet)
                self.written += 1
        finally:
            if container is not None:
                with suppress(Exception):
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()


class StreamClient:
    def __init__(self, spec: dict, stream: dict, http):
        self.spec = spec
        self.stream = stream
        self.http = http
        self.base = spec["base_url"]
        self.session_id = None
        self.pc = None
        self.tasks = []
        self.video_t = array("d")
        self.video_pts = array("q")   # RTP-derived pts per decoded video frame (joins the server send ring)
        self.audio_t = array("d")
        self.audio_frames = 0
        self.errors = []
        self.turns = []
        self.timing = {}
        self.seq = 0
        self.stop = asyncio.Event()
        self.recorder = None  # --record-streams: frames written at their real arrival times (gaps stay visible)

    async def setup(self):
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        spec = self.spec
        params = {"avatar_id": self.stream["avatar_id"], "user_id": self.stream["user_id"],
                  "fps": spec["musetalk_fps"], "playback_fps": spec["playback_fps"],
                  "batch_size": spec["batch_size"], "chunk_duration": spec["chunk_duration"]}
        if spec.get("prebuffer_seconds") is not None:
            params["prebuffer_seconds"] = spec["prebuffer_seconds"]
        if spec.get("pose_set"):
            params["pose_set"] = spec["pose_set"]
        t0 = now()
        async with self.http.post(f"{self.base}/webrtc/sessions/create?{urlencode(params)}") as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"create failed {resp.status}: {body[:300]}")
            data = json.loads(body)
        self.session_id = data["session_id"]
        self.timing["create_s"] = now() - t0
        servers = []
        if not spec.get("ignore_ice_servers"):
            for entry in data.get("ice_servers") or []:
                if entry.get("urls"):
                    servers.append(RTCIceServer(urls=entry["urls"], username=entry.get("username"),
                                                credential=entry.get("credential")))
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        connected = asyncio.Event()

        @self.pc.on("connectionstatechange")
        async def _state():
            if self.pc.connectionState == "connected":
                connected.set()
            elif self.pc.connectionState in ("failed", "closed"):
                if not self.stop.is_set():  # our own close at the end is not an error
                    self.errors.append(f"pc {self.pc.connectionState}")
                connected.set()

        @self.pc.on("track")
        def _track(track):
            self.tasks.append(asyncio.ensure_future(self._consume(track)))

        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.addTransceiver("audio", direction="recvonly")
        t1 = now()
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        gathered = asyncio.Event()
        if self.pc.iceGatheringState == "complete":
            gathered.set()

        @self.pc.on("icegatheringstatechange")
        def _gather():
            if self.pc.iceGatheringState == "complete":
                gathered.set()

        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(gathered.wait(), spec["ice_gather_timeout_s"])
        async with self.http.post(f"{self.base}/webrtc/sessions/{self.session_id}/offer",
                                  json={"sdp": self.pc.localDescription.sdp,
                                        "type": self.pc.localDescription.type}) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"offer failed {resp.status}: {body[:300]}")
            answer = json.loads(body)
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        await asyncio.wait_for(connected.wait(), spec["connection_timeout_s"])
        if self.pc.connectionState != "connected":
            raise RuntimeError(f"peer not connected: {self.pc.connectionState}")
        self.timing["offer_connect_s"] = now() - t1
        if spec.get("trace_dir") and self.stream["index"] in (spec.get("record_streams") or []):
            self.recorder = ArrivalRecorder(Path(spec["trace_dir"]) / f"s{self.stream['index']:02d}_observer.mp4",
                                            now() + spec.get("record_at_s", 60.0), spec.get("record_seconds", 90.0))

    async def _consume(self, track):
        kind = track.kind
        try:
            while not self.stop.is_set():
                frame = await track.recv()
                if kind == "video":
                    self.video_t.append(now())
                    self.video_pts.append(int(frame.pts) if frame.pts is not None else -1)
                    if self.recorder is not None:
                        self.recorder.offer(self.video_t[-1], frame)
                else:
                    self.audio_frames += 1
                    self.audio_t.append(now())
        except Exception as exc:
            if not self.stop.is_set():
                self.errors.append(f"{kind} recv: {type(exc).__name__}: {exc}")

    async def post_turn(self, wav: str, index: int) -> dict:
        import aiohttp
        spec = self.spec
        record = {"turn": index, "wav": wav}
        for attempt in range(spec["post_retries"] + 1):
            form = aiohttp.FormData()
            data = Path(wav).read_bytes()
            form.add_field("audio_file", data, filename=Path(wav).name, content_type="audio/wav")
            if spec.get("pose_set"):
                self.seq += 1
                form.add_field("reaction_intent", "none")
                form.add_field("pose_id", "speaking_direct")
                form.add_field("pose_plan", json.dumps(spec.get("pose_plan") or WALL_POSE_PLAN))
                record["turn_id"] = f"lt2_{self.stream['index']}_{index}_{int(time.time() * 1000)}"
                form.add_field("turn_id", record["turn_id"])
                form.add_field("seq", str(self.seq))
                form.add_field("effective", "next_boundary")
                form.add_field("mouth_mode", "lip_sync")
                form.add_field("audio_start", "immediate")
            record["t_post"] = now()
            record["t_post_wall"] = time.time()
            try:
                async with self.http.post(f"{self.base}/webrtc/sessions/{self.session_id}/stream",
                                          data=form) as resp:
                    body = await resp.text()
                    record["status"] = resp.status
                    record["accept_s"] = now() - record["t_post"]
                    if resp.status == 200:
                        payload = json.loads(body)
                        record["request_id"] = payload.get("request_id")
                        return record
                    record["error"] = body[:300]
            except Exception as exc:
                record["status"] = -1
                record["error"] = f"{type(exc).__name__}: {exc}"
            if record.get("status") == 409 and attempt < spec["post_retries"]:
                await asyncio.sleep(0.5)
                continue
            break
        self.errors.append(f"turn {index} post failed: {record.get('status')} {record.get('error')}")
        return record


async def send_abort(client: "StreamClient", record: dict) -> None:
    """Barge-in (S5): assistant_turn_aborted for this turn (pose protocol sessions)."""
    client.seq += 1
    record["t_abort"] = now()
    try:
        async with client.http.post(f"{client.base}/webrtc/sessions/{client.session_id}/events",
                                    json={"event": "assistant_turn_aborted", "seq": client.seq,
                                          "turn_id": record.get("turn_id")}) as resp:
            record["abort_status"] = resp.status
            record["abort_body"] = (await resp.text())[:200]
    except Exception as exc:
        record["abort_status"] = -1
        record["abort_body"] = f"{type(exc).__name__}: {exc}"


async def shard_main(spec_path: str) -> int:
    import aiohttp
    spec = json.loads(Path(spec_path).read_text())
    pin_error = pin_to(spec.get("client_cpus"))
    stop = asyncio.Event()
    lag = LoopLag()
    lag_task = asyncio.ensure_future(lag.run(stop))
    timeout = aiohttp.ClientTimeout(total=spec["http_timeout_s"])
    cpu0 = time.process_time()
    async with aiohttp.ClientSession(timeout=timeout) as http:
        clients = [StreamClient(spec, stream, http) for stream in spec["streams"]]
        stagger_join = bool(spec.get("stagger_join"))
        # --stagger-join: a stream with a start delay creates its session and connects only when its delay
        # expires, i.e. while the other streams are already live (a real mid-call join)
        early = [c for c in clients if not (stagger_join and c.stream.get("start_delay_s", 0.0) > 0)]
        results = await asyncio.gather(*(c.setup() for c in early), return_exceptions=True)
        ready = []
        trace_dir = Path(spec["trace_dir"]) if spec.get("trace_dir") else None
        flushed = {}

        def flush_traces():
            if trace_dir is None:
                return
            trace_dir.mkdir(parents=True, exist_ok=True)
            import struct
            for c in clients:
                k = c.stream["index"]
                v0, a0 = flushed.get(("v", k), 0), flushed.get(("a", k), 0)
                n_v, n_a = len(c.video_t), len(c.audio_t)
                if n_v > v0:
                    with open(trace_dir / f"s{k:02d}_video.bin", "ab") as f:
                        f.write(b"".join(struct.pack("<dq", c.video_t[i], c.video_pts[i]) for i in range(v0, n_v)))
                    flushed[("v", k)] = n_v
                if n_a > a0:
                    with open(trace_dir / f"s{k:02d}_audio.bin", "ab") as f:
                        f.write(array("d", c.audio_t[a0:n_a]).tobytes())
                    flushed[("a", k)] = n_a
                meta = trace_dir / f"s{k:02d}_meta.json"
                if c.session_id and not meta.exists():
                    meta.write_text(json.dumps({"stream": k, "session_id": c.session_id, "avatar_id": c.stream["avatar_id"],
                                                "user_id": c.stream["user_id"], "timing": c.timing}))
            n_l = len(lag.spikes)
            l0 = flushed.get(("lag",), 0)
            if n_l > l0:
                with open(trace_dir / f"shard{spec['shard']}_looplag.bin", "ab") as f:
                    f.write(array("d", lag.spikes[l0:n_l]).tobytes())
                flushed[("lag",)] = n_l

        async def flusher():
            while not stop.is_set():
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), 10.0)
                with suppress(Exception):
                    flush_traces()

        flush_task = asyncio.ensure_future(flusher())
        for client, result in zip(early, results):
            if isinstance(result, Exception):
                client.errors.append(f"setup: {type(result).__name__}: {result}")
            ready.append({"stream": client.stream["index"], "session_id": client.session_id,
                          "ok": not isinstance(result, Exception), "timing": client.timing,
                          "error": None if not isinstance(result, Exception) else str(result)})
        emit({"event": "ready", "shard": spec["shard"], "pin_error": pin_error, "streams": ready})
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        go = line.strip().split()
        if not go or go[0] != "go":
            emit({"event": "aborted", "shard": spec["shard"], "line": line.strip()})
            return 2
        t_go = float(go[1]) if len(go) > 1 else now()
        active: dict = {}
        poll_started: dict = {}

        async def poll_active():
            while not stop.is_set():
                sids = {c.session_id for c in clients if c.session_id}  # late joiners appear over time
                started = now()
                if spec.get("shard_poll") == "all":
                    with suppress(Exception):
                        async with http.get(f"{spec['base_url']}/webrtc/sessions/stats",
                                            params={"view": "lifetime"}) as resp:
                            data = await resp.json()
                        for entry in data.get("sessions", []):
                            if entry["session_id"] in sids:
                                active[entry["session_id"]] = entry.get("active_stream")
                                poll_started[entry["session_id"]] = started
                else:
                    # One small per-session request instead of every session's lifetime stats: the
                    # server builds these replies on its event loop, so the poll is part of the load.
                    for sid in sids:
                        with suppress(Exception):
                            async with http.get(f"{spec['base_url']}/webrtc/sessions/{sid}/status",
                                                params={"light": "1"}) as resp:
                                entry = await resp.json()
                            active[sid] = entry.get("active_stream")
                            poll_started[sid] = started
                await asyncio.sleep(0.5)

        poll_task = asyncio.ensure_future(poll_active())

        async def run_client(client: StreamClient):
            delay = client.stream.get("start_delay_s", 0.0)
            if stagger_join and delay > 0 and not client.session_id:
                await asyncio.sleep(max(0.0, t_go + delay - now()))
                try:
                    await client.setup()
                except Exception as exc:
                    client.errors.append(f"setup: {type(exc).__name__}: {exc}")
                emit({"event": "joined", "shard": spec["shard"], "stream": client.stream["index"],
                      "session_id": client.session_id, "t": now(), "timing": client.timing,
                      "error": client.errors[-1] if client.errors else None})
                delay = 0.0
            if not client.session_id or client.errors:
                return
            if delay > 0:
                await asyncio.sleep(max(0.0, t_go + delay - now()))
            deadline = t_go + spec["max_level_s"]
            end_at = (now() + spec["duration_s"]) if spec.get("duration_s") else None
            failures = 0
            rng = random.Random(spec["seed"] * 1000 + client.stream["index"])
            durations = client.stream.get("durations") or []
            for index, wav in enumerate(client.stream["turns"]):
                if end_at is not None and now() >= end_at:
                    break  # --duration-s: stop starting new turns (not an error)
                if now() > deadline:
                    client.errors.append("level time limit reached before all turns")
                    break
                record = await client.post_turn(wav, index)
                client.turns.append(record)
                if record.get("status") != 200:
                    failures += 1
                    if failures > spec.get("max_consecutive_post_failures", 0):
                        break
                    await asyncio.sleep(min(5.0, 0.5 * 2 ** failures))  # keep the stream speaking after a rare race
                    continue
                failures = 0
                answered = now()
                abort_at = None
                if spec.get("barge_in_fraction", 0) > 0 and rng.random() < spec["barge_in_fraction"]:
                    abort_at = answered + rng.uniform(*spec["barge_in_after_s"])
                    record["barge_in_planned"] = True
                while now() < deadline:
                    await asyncio.sleep(0.05 if abort_at else 0.25)
                    if abort_at is not None and now() >= abort_at:
                        await send_abort(client, record)
                        abort_at = None
                    seen_at = poll_started.get(client.session_id, 0.0)
                    if seen_at > answered and active.get(client.session_id) != record.get("request_id"):
                        break
                record["t_done_seen"] = now()
                if index < len(client.stream["turns"]) - 1:
                    gap = spec["turn_gap_s"]
                    duty = spec.get("duty")
                    if duty and 0 < duty < 1:
                        # Poisson turns (S2): exponential idle gap for the requested duty cycle.
                        speech = float(durations[index]) if index < len(durations) and durations[index] else 10.0
                        gap = max(gap, rng.expovariate(duty / (speech * (1.0 - duty))))
                    record["gap_after_s"] = round(gap, 3)
                    await asyncio.sleep(gap)
            await asyncio.sleep(spec["tail_s"])

        await asyncio.gather(*(run_client(c) for c in clients))
        stop.set()
        for c in clients:
            if c.recorder is not None:
                with suppress(Exception):
                    c.recorder.q.put_nowait(None)
                    c.recorder.thread.join(timeout=30)
        flush_task.cancel()
        with suppress(Exception):
            flush_traces()
        for c in clients:
            c.stop.set()
        poll_task.cancel()
        lag_task.cancel()
        streams_out = []
        for c in clients:
            streams_out.append({
                "stream": c.stream["index"], "session_id": c.session_id,
                "user_id": c.stream["user_id"], "avatar_id": c.stream["avatar_id"],
                "timing": c.timing, "turns": c.turns, "errors": c.errors,
                "audio_frames": c.audio_frames,
                "video_recv_t": list(c.video_t),
            })
            if c.session_id and not spec.get("keep_sessions"):
                with suppress(Exception):
                    async with http.delete(f"{spec['base_url']}/webrtc/sessions/{c.session_id}") as resp:
                        await resp.text()
            if c.pc is not None:
                with suppress(Exception):
                    await c.pc.close()
            for task in c.tasks:
                task.cancel()
        emit({"event": "result", "shard": spec["shard"], "loop_lag": lag.summary(),
              "cpu_s": round(time.process_time() - cpu0, 3),
              "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
              "streams": streams_out})
    return 0


# ===========================================================================
# coordinator
# ===========================================================================
class SystemSampler:
    """Server process (/proc) + GPU (nvidia-smi) + MemAvailable at 1 Hz."""

    def __init__(self, pid: Optional[int], interval: float = 1.0, abort_below_gb: float = 4.0):
        self.pid = pid
        self.interval = interval
        self.abort_below_gb = abort_below_gb
        self.samples = []
        self.gpu = []
        self.vram_by_pid = []
        self.aborted = None
        self.clock_tck = os.sysconf("SC_CLK_TCK")

    def _proc(self):
        if not self.pid:
            return None
        try:
            stat = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu = (int(stat[11]) + int(stat[12])) / self.clock_tck
            status = Path(f"/proc/{self.pid}/status").read_text()
            fields = dict(l.split(":", 1) for l in status.splitlines() if ":" in l)
            return {"cpu_s": cpu, "rss_mb": int(fields["VmRSS"].split()[0]) / 1024.0,
                    "threads": int(fields["Threads"])}
        except Exception:
            return None

    async def _smi(self, args):
        if shutil.which("nvidia-smi") is None:
            return None
        proc = await asyncio.create_subprocess_exec("nvidia-smi", *args, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        return out.decode(errors="ignore").strip() if proc.returncode == 0 else None

    async def run(self, stop: asyncio.Event):
        tick = 0
        while not stop.is_set():
            t = now()
            entry = {"t": t, "mem_available_gb": round(mem_available_gb(), 3), "proc": self._proc()}
            self.samples.append(entry)
            if entry["mem_available_gb"] < self.abort_below_gb and self.aborted is None:
                self.aborted = f"MemAvailable {entry['mem_available_gb']:.2f} GB < {self.abort_below_gb} GB"
            text = await self._smi(["--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
                                    "--format=csv,noheader,nounits"])
            if text:
                with suppress(Exception):
                    u, m, p, c, temp = [v.strip() for v in text.splitlines()[0].split(",")]
                    self.gpu.append({"t": t, "util": float(u), "mem_mb": float(m),
                                     "power_w": float(p) if p not in ("N/A", "[N/A]") else None,
                                     "sm_mhz": float(c), "temp_c": float(temp)})
            if tick % 5 == 0 and self.pid:
                text = await self._smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
                if text is not None:
                    used = 0.0
                    for line in text.splitlines():
                        with suppress(Exception):
                            pid, mem = [v.strip() for v in line.split(",")]
                            if int(pid) == self.pid:
                                used += float(mem)
                    self.vram_by_pid.append({"t": t, "server_vram_mb": used})
            tick += 1
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), max(0.0, self.interval - (now() - t)))

    def summary(self, w0: float, w1: float) -> dict:
        inside = [s for s in self.samples if w0 <= s["t"] <= w1 and s["proc"]]
        out = {"mem_available_min_gb": rnd(min((s["mem_available_gb"] for s in self.samples), default=None), 3)}
        if len(inside) >= 2:
            cpu = (inside[-1]["proc"]["cpu_s"] - inside[0]["proc"]["cpu_s"]) / max(1e-9, inside[-1]["t"] - inside[0]["t"])
            out.update({"server_cpu_cores_mean": rnd(cpu, 3),
                        "server_rss_max_mb": rnd(max(s["proc"]["rss_mb"] for s in inside), 1),
                        "server_rss_end_mb": rnd(inside[-1]["proc"]["rss_mb"], 1),
                        "server_threads_max": max(s["proc"]["threads"] for s in inside)})
        gpu = [g for g in self.gpu if w0 <= g["t"] <= w1]
        if gpu:
            out.update({"gpu_util_mean": rnd(sum(g["util"] for g in gpu) / len(gpu), 2),
                        "gpu_mem_max_mb": max(g["mem_mb"] for g in gpu),
                        "gpu_power_mean_w": rnd(sum(g["power_w"] or 0 for g in gpu) / len(gpu), 1),
                        "gpu_sm_mhz_min": min(g["sm_mhz"] for g in gpu),
                        "gpu_temp_max_c": max(g["temp_c"] for g in gpu)})
        vram = [v["server_vram_mb"] for v in self.vram_by_pid if w0 <= v["t"] <= w1]
        if vram:
            out["server_vram_max_mb"] = max(vram)
        return out


async def poll_server(args, sessions_of_interest: set, state: dict, stop: asyncio.Event):
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        while not stop.is_set():
            t_req = now()
            try:
                async with http.get(f"{args.base_url}/webrtc/sessions/stats",
                                    params={"view": "lifetime", "ring": str(args.ring)}) as resp:
                    data = await resp.json()
            except Exception as exc:
                state["poll_errors"] += 1
                state["last_poll_error"] = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(args.poll_interval_s)
                continue
            t_resp = now()
            state["poll_rtt"].append(t_resp - t_req)
            server = data.get("server") or {}
            state["server_samples"].append({"t": t_resp, "server": {
                k: server.get(k) for k in ("monotonic", "process_cpu_s", "rss_mb", "threads", "live",
                                           "loop_lag", "live_handoff_mode", "lifetime_counters")}})
            for entry in data.get("sessions", []):
                sid = entry["session_id"]
                if sid not in sessions_of_interest:
                    continue
                view = entry.get("counters") or {}
                stitcher = state["stitchers"].setdefault(sid, CounterStitcher())
                totals = stitcher.update(view)
                state["counter_samples"].setdefault(sid, []).append((t_resp, totals))
                if state.get("counters_jsonl") is not None:
                    life_ = view.get("lifetime") or {}
                    rows = state.setdefault("_jsonl_rows", [])
                    rows.append(json.dumps({"t": t_resp, "sid": sid, "totals": totals,
                                            "live_active": life_.get("live_active"),
                                            "buffer_depth_frames": life_.get("buffer_depth_frames"),
                                            "queue_underruns": life_.get("queue_underruns")}))
                life = view.get("lifetime") or {}
                turn_view = view.get("turn_counters") or {}
                fps = life.get("output_fps", turn_view.get("output_fps"))
                if fps is not None:
                    state["output_fps"].setdefault(sid, set()).add(float(fps))
                if entry.get("playback_fps") is not None:
                    state["playback_fps"].setdefault(sid, set()).add(float(entry["playback_fps"]))
                if life.get("send_ring"):
                    state["rings"].setdefault(sid, RingStitcher()).add(life["send_ring"])
                state["last_totals"][sid] = totals
                state["last_view"][sid] = {k: v for k, v in view.items() if k != "lifetime"}
                if view.get("live_handoff") is not None:
                    state["handoff"][sid] = view["live_handoff"]
                if view.get("idle_cache_backed") is not None:
                    state["idle_cache_backed"][sid] = view["idle_cache_backed"]
                marks = life.get("turn")
                if marks and marks.get("generation_id") is not None:
                    turns = state["turn_marks"].setdefault(sid, {})
                    turns[marks["generation_id"]] = marks
                live_active = life.get("live_active", turn_view.get("live_active"))
                state["live"].setdefault(sid, []).append((t_resp, bool(live_active),
                                                          bool(life.get("live_released",
                                                                        turn_view.get("live_released")))))
            if state.get("counters_jsonl") is not None and state.get("_jsonl_rows"):
                with suppress(Exception):
                    state["counters_jsonl"].parent.mkdir(parents=True, exist_ok=True)
                    with open(state["counters_jsonl"], "a") as f:
                        f.write("\n".join(state["_jsonl_rows"]) + "\n")
                    state["_jsonl_rows"] = []
            elapsed = now() - t_req
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), max(0.05, args.poll_interval_s - elapsed))


def first_frame_latencies(turns: list, marks: dict) -> list:
    """Match each POSTed turn to the first server generation started after it."""
    ordered = sorted((m for m in marks.values() if m.get("started_monotonic")),
                     key=lambda m: m["started_monotonic"])
    out, used = [], set()
    for turn in turns:
        t_post = turn.get("t_post")
        if t_post is None or turn.get("status") != 200:
            continue
        match = next((m for m in ordered if m["generation_id"] not in used
                      and m["started_monotonic"] >= t_post - 0.001), None)
        if match is None:
            out.append({"turn": turn["turn"], "matched": False})
            continue
        used.add(match["generation_id"])
        first = match.get("first_live_send_monotonic")
        out.append({"turn": turn["turn"], "matched": True, "generation_id": match["generation_id"],
                    "accept_s": rnd(turn.get("accept_s"), 4),
                    "setup_s": rnd(match["started_monotonic"] - t_post, 4),
                    "first_frame_s": rnd(first - t_post, 4) if first else None})
    return out


def client_intervals(video_t: list, w0: float, w1: float) -> dict:
    inside = [t for t in video_t if w0 <= t <= w1]
    gaps = [b - a for a, b in zip(inside, inside[1:])]
    return {"frames": len(inside), "interval_avg_s": rnd(sum(gaps) / len(gaps), 5) if gaps else None,
            "interval_p99_s": rnd(pct(gaps, 0.99), 5), "interval_max_s": rnd(max(gaps), 5) if gaps else None}


async def run_level(args, level: int, corpus: list, level_dir: Path, client_cpus: list,
                    server_view: dict) -> dict:
    level_dir.mkdir(parents=True, exist_ok=True)
    avatars = [a for a in args.avatar_ids.split(",") if a]
    turns_per_stream = 1 if args.chain else args.turns
    plan, reuse = plan_turns(corpus, level, turns_per_stream, args.seed + level)
    chain_info = []
    if args.wav_list:
        fixed = [w for w in args.wav_list.split(",") if w]
        plan = [[{"path": w, "file": Path(w).name, "duration_s": None} for w in fixed]
                for _ in range(level)]
        reuse = level
        turns_per_stream = len(fixed)
    if args.chain:
        chain_dir = Path(args.chain_dir or (level_dir / "chain_audio"))
        start = random.Random(args.seed).randrange(len(corpus))
        for i in range(level):
            info = build_chain_wav(corpus, start + i * max(1, len(corpus) // max(1, level)),
                                   args.chain_seconds, chain_dir / f"stream_{i:02d}.wav")
            chain_info.append(info)
        wavs = [[info["path"]] for info in chain_info]
    else:
        wavs = [[entry["path"] for entry in row] for row in plan]
    pose_set = Path(args.pose_set_file).read_text() if args.pose_set_file else None
    stage_tag = f"lt2-{int(time.time())}-n{level}"
    durations = ([[info["duration_s"]] for info in chain_info] if args.chain else
                 [[e.get("duration_s") for e in row] for row in plan])
    streams = [{"index": i, "user_id": f"{stage_tag}-{i:02d}", "avatar_id": avatars[i % len(avatars)],
                "turns": wavs[i], "durations": durations[i], "start_delay_s": i * args.stagger_s}
               for i in range(level)]
    shard_size = max(1, min(5, args.peers_per_shard))
    shards = [streams[i:i + shard_size] for i in range(0, level, shard_size)]
    expected_audio = max(sum(float(e.get("duration_s") or 30.0) for e in row) for row in plan) if not args.chain \
        else max(i["duration_s"] for i in chain_info)
    duty_gaps = 0.0
    if args.duty and 0 < args.duty < 1:
        duty_gaps = expected_audio * (1.0 - args.duty) / args.duty * 3.0  # generous: exponential tail
    max_level_s = (expected_audio + duty_gaps + turns_per_stream * (args.turn_gap_s + 30.0)
                   + args.stagger_s * level + 120.0)
    if args.duration_s:
        max_level_s = args.duration_s + args.stagger_s * level + 600.0
    procs = []
    for k, members in enumerate(shards):
        spec = {"shard": k, "base_url": args.base_url, "streams": members, "client_cpus": client_cpus,
                "musetalk_fps": args.musetalk_fps, "playback_fps": args.playback_fps,
                "batch_size": args.batch_size, "chunk_duration": args.chunk_duration,
                "prebuffer_seconds": args.prebuffer_seconds, "pose_set": pose_set,
                "pose_plan": json.loads(args.pose_plan) if args.pose_plan else None,
                "ignore_ice_servers": args.ignore_ice_servers, "ice_gather_timeout_s": args.ice_gather_timeout_s,
                "connection_timeout_s": args.connection_timeout_s, "http_timeout_s": 120,
                "post_retries": args.post_retries, "turn_gap_s": args.turn_gap_s, "tail_s": args.tail_s,
                "trace_dir": str(Path(args.trace_dir) / f"n{level:02d}") if args.trace_dir else None,
                "stagger_join": args.stagger_join, "duration_s": args.duration_s, "shard_poll": args.shard_poll,
                "max_consecutive_post_failures": args.max_consecutive_post_failures,
                "record_streams": [int(x) for x in args.record_streams.split(",") if x != ""] if args.record_streams else [],
                "record_at_s": args.record_at_s, "record_seconds": args.record_seconds,
                "duty": args.duty, "seed": args.seed, "barge_in_fraction": args.barge_in_fraction,
                "barge_in_after_s": [args.barge_in_min_s, args.barge_in_max_s],
                "max_level_s": max_level_s, "keep_sessions": args.keep_sessions}
        spec_path = level_dir / f"shard_{k}.json"
        spec_path.write_text(json.dumps(spec))
        log = open(level_dir / f"shard_{k}.log", "w")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), "--shard-worker", str(spec_path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, cwd=str(ROOT),
            limit=1 << 26)
        procs.append((proc, log))
    events = {"ready": {}, "result": {}}
    sessions: set = set()  # polled sessions; late joiners (--stagger-join) are added as they connect
    joins: list = []

    async def reader(k, proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            text = line.decode(errors="ignore")
            if text.startswith(MARK):
                event = json.loads(text[len(MARK):])
                if event["event"] == "joined":
                    joins.append(event)
                    if event.get("session_id"):
                        sessions.add(event["session_id"])
                    continue
                events.setdefault(event["event"], {})[k] = event

    readers = [asyncio.ensure_future(reader(k, p)) for k, (p, _l) in enumerate(procs)]
    t_setup = now()
    while len(events["ready"]) < len(procs) and now() - t_setup < args.setup_timeout_s:
        if all(p.returncode is not None for p, _ in procs):
            break
        await asyncio.sleep(0.2)
    ready_streams = [s for e in events["ready"].values() for s in e["streams"]]
    sessions.update(s["session_id"] for s in ready_streams if s.get("session_id"))
    setup_s = now() - t_setup
    state = {"counters_jsonl": (Path(args.trace_dir) / f"n{level:02d}" / "server_counters.jsonl") if args.trace_dir else None,
             "stitchers": {}, "counter_samples": {}, "rings": {}, "turn_marks": {}, "live": {},
             "handoff": {}, "idle_cache_backed": {}, "last_totals": {}, "last_view": {},
             "output_fps": {}, "playback_fps": {}, "server_samples": [], "poll_rtt": [],
             "poll_errors": 0, "last_poll_error": None}
    stop = asyncio.Event()
    sampler = SystemSampler(server_view.get("pid"), abort_below_gb=args.abort_below_gb)
    background = [asyncio.ensure_future(poll_server(args, sessions, state, stop)),
                  asyncio.ensure_future(sampler.run(stop))]
    await asyncio.sleep(1.5)  # a baseline poll before any turn
    t_go = now()
    for proc, _log in procs:
        with suppress(Exception):
            proc.stdin.write(f"go {t_go}\n".encode())
            await proc.stdin.drain()
    print(f"[lt2] level N={level}: {len(ready_streams)} peers ready in {setup_s:.1f}s "
          f"({len(procs)} shards); GO; expected <= {max_level_s:.0f}s", flush=True)
    deadline = t_go + max_level_s + 60
    while len(events["result"]) < len(procs) and now() < deadline:
        if sampler.aborted:
            print(f"[lt2] ABORT: {sampler.aborted}", flush=True)
            break
        if all(p.returncode is not None for p, _ in procs):
            break
        await asyncio.sleep(0.5)
    t_end = now()
    stop.set()
    for task in background:
        with suppress(Exception):
            await asyncio.wait_for(task, 15)
    for proc, log in procs:
        if proc.returncode is None:
            with suppress(Exception):
                proc.stdin.close()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), 30)
            if proc.returncode is None:
                proc.kill()
        log.close()
    for task in readers:
        task.cancel()
    if args.trace_dir:
        # raw material for per-stream 1 s windows / gap / freshness analysis (docs/.../live15/live_trace_report.py)
        with suppress(Exception):
            tdir = Path(args.trace_dir) / f"n{level:02d}"
            tdir.mkdir(parents=True, exist_ok=True)
            import struct
            for sid, ring in state["rings"].items():
                with open(tdir / f"ring_{sid}.bin", "wb") as f:
                    f.write(b"".join(struct.pack("<qdbq", ring.seq[i], ring.t[i], ring.kind[i], ring.pts[i])
                                     for i in range(len(ring.seq))))
            (tdir / "level_meta.json").write_text(json.dumps({
                "level": level, "t_go": t_go, "t_end": t_end, "stagger_s": args.stagger_s,
                "stagger_join": args.stagger_join, "duration_s": args.duration_s, "joins": joins,
                "ready": [s_ for e in events["ready"].values() for s_ in e["streams"]],
                "server_samples": state["server_samples"], "system_samples": getattr(sampler, "samples", [])}, default=str))
    return summarize_level(args, level, t_go, t_end, events, state, sampler, plan, reuse, chain_info,
                           setup_s, server_view)


def summarize_level(args, level, t_go, t_end, events, state, sampler, plan, reuse, chain_info,
                    setup_s, server_view) -> dict:
    results = [s for e in events.get("result", {}).values() for s in e["streams"]]
    by_sid = {s["session_id"]: s for s in results if s.get("session_id")}
    rings = {sid: state["rings"].get(sid, RingStitcher()) for sid in by_sid}
    first_f = {}
    last_f = {}
    for sid, ring in rings.items():
        f_times = [ring.t[i] for i in range(len(ring.seq)) if ring.kind[i] == ord("f")]
        if f_times:
            first_f[sid], last_f[sid] = f_times[0], f_times[-1]
    if args.window_start_s is not None:
        w0 = t_go + args.window_start_s
        w1 = w0 + (args.window_s or (t_end - w0))
    elif args.chain and first_f and len(first_f) == len(by_sid):
        w0 = max(first_f.values()) + args.settle_s
        w1 = min(last_f.values()) - 0.5
    else:
        w0 = t_go + args.warmup_s
        w1 = max(last_f.values()) if last_f else t_end
    steady_s = max(0.0, w1 - w0)
    per_stream = []
    worst = {"fresh_fraction_min": None, "max_held_run": 0, "stall_seconds_max": 0.0,
             "send_interval_max_s": 0.0, "send_interval_avg_s_worst_dev": 0.0}
    all_first = []
    all_barge = []
    for sid, result in sorted(by_sid.items(), key=lambda kv: kv[1]["stream"]):
        ring_summary = summarize_ring(rings[sid], w0, w1, args.playback_fps)
        counters = counter_deltas(state["counter_samples"].get(sid, []), w0, w1)
        latencies = first_frame_latencies(result.get("turns", []), state["turn_marks"].get(sid, {}))
        barge = []
        ring = rings[sid]
        for turn in result.get("turns", []):
            t_abort = turn.get("t_abort")
            if t_abort is None:
                continue
            back = next((ring.t[i] for i in range(len(ring.seq))
                         if ring.t[i] > t_abort and ring.kind[i] in (ord("i"), ord("p"))), None)
            barge.append({"turn": turn["turn"], "abort_status": turn.get("abort_status"),
                          "return_to_idle_s": rnd(back - t_abort, 4) if back else None})
        all_barge.extend(b["return_to_idle_s"] for b in barge if b["return_to_idle_s"] is not None)
        all_first.extend(l["first_frame_s"] for l in latencies if l.get("first_frame_s") is not None)
        client = client_intervals(result.get("video_recv_t", []), w0, w1)
        fps_seen = sorted(state["output_fps"].get(sid, set()) | state["playback_fps"].get(sid, set()))
        entry = {"stream": result["stream"], "session_id": sid, "avatar_id": result["avatar_id"],
                 "turns_posted": len(result.get("turns", [])),
                 "turns_ok": sum(1 for t in result.get("turns", []) if t.get("status") == 200),
                 "server_ring": ring_summary, "server_counters": counters,
                 "first_frame": latencies, "barge_in": barge or None,
                 "client_recv": client, "output_fps_seen": fps_seen,
                 "stitch_resets": state["stitchers"].get(sid).resets if sid in state["stitchers"] else None,
                 "live_handoff_last": state["handoff"].get(sid),
                 "stitched_totals_end": state["last_totals"].get(sid),
                 "counters_view_last": state["last_view"].get(sid),
                 "idle_cache_backed": state["idle_cache_backed"].get(sid),
                 "setup_timing": result.get("timing"), "errors": result.get("errors")}
        per_stream.append(entry)
        ff = ring_summary["fresh_fraction"]
        if ff is None:
            ff = counters.get("fresh_fraction")  # lifetime ring absent: stitched counters
        if ff is not None:
            worst["fresh_fraction_min"] = ff if worst["fresh_fraction_min"] is None else min(worst["fresh_fraction_min"], ff)
        worst["max_held_run"] = max(worst["max_held_run"], ring_summary["max_held_run"])
        worst["stall_seconds_max"] = max(worst["stall_seconds_max"], counters.get("stall_seconds") or 0.0)
        worst["send_interval_max_s"] = max(worst["send_interval_max_s"], ring_summary["send_interval_max_s"] or 0.0)
        if ring_summary["send_interval_avg_s"] is not None:
            worst["send_interval_avg_s_worst_dev"] = max(worst["send_interval_avg_s_worst_dev"],
                                                        abs(ring_summary["send_interval_avg_s"] - 1.0 / args.playback_fps))
    series = fresh_series_from_rings(rings, w0, w1)
    server_live = [s for s in state["server_samples"] if w0 <= s["t"] <= w1 and s["server"].get("live")]
    generated_fps = None
    callback_ms = None
    if len(server_live) >= 2:
        a, b = server_live[0], server_live[-1]
        dt = max(1e-9, b["t"] - a["t"])
        generated_fps = (b["server"]["live"]["frames_handed_off"] - a["server"]["live"]["frames_handed_off"]) / dt
        batches = b["server"]["live"]["batches_handed_off"] - a["server"]["live"]["batches_handed_off"]
        if batches > 0:
            callback_ms = (b["server"]["live"]["callback_total_s"] - a["server"]["live"]["callback_total_s"]) / batches * 1000
    shard_results = list(events.get("result", {}).values())
    lag_p99 = max((r["loop_lag"]["p99_ms"] or 0.0) for r in shard_results) if shard_results else None
    server_lag = [s["server"].get("loop_lag") or {} for s in state["server_samples"] if w0 <= s["t"] <= w1]
    output_fps_ok = all(e["output_fps_seen"] == [float(args.playback_fps)] for e in per_stream) and bool(per_stream)
    errors = [f"stream {e['stream']}: {err}" for e in per_stream for err in (e["errors"] or [])]
    missing = level - len(per_stream)
    th = args
    ring_present = bool(per_stream) and all(e["server_ring"]["speaking_slots"] > 0 for e in per_stream)
    checks = {
        "all_streams_reported": missing == 0,
        "no_client_errors": not errors,
        "steady_state_s": steady_s >= th.min_steady_s,
        "output_fps_20": output_fps_ok,
        "fresh_fraction": (worst["fresh_fraction_min"] or 0.0) >= th.min_fresh_fraction,
        "stall_seconds": worst["stall_seconds_max"] <= th.max_stall_s,
    }
    unmeasured = []
    if ring_present:
        checks.update({
            "held_run": worst["max_held_run"] <= th.max_held_run,
            "send_interval_max": worst["send_interval_max_s"] <= th.max_send_interval_s,
            "send_interval_avg": worst["send_interval_avg_s_worst_dev"] <= th.send_interval_avg_tol_s,
        })
    else:
        # WEBRTC_LIFETIME_COUNTERS=0 on the server: no send ring. Fresh fraction and
        # stalls come from stitched per-turn counters; cadence checks are not measurable.
        unmeasured = ["held_run", "send_interval_max", "send_interval_avg"]
    valid = {"client_loop_lag_p99": (lag_p99 or 0.0) <= th.max_client_lag_ms,
             "mem_available": sampler.aborted is None,
             "poll_ok": state["poll_errors"] <= max(3, int(0.05 * max(1, len(state["server_samples"]))))}
    verdict = "PASS" if all(checks.values()) else "FAIL"
    if not all(valid.values()):
        verdict = "INVALID"
    summary = {
        "schema": SCHEMA, "label": args.label, "level": level, "mode": "chain" if args.chain else f"turns={args.turns}",
        "verdict": verdict, "checks": checks, "unmeasured": unmeasured, "validity": valid,
        "thresholds": {"min_steady_s": th.min_steady_s, "min_fresh_fraction": th.min_fresh_fraction,
                       "max_held_run": th.max_held_run, "max_stall_s": th.max_stall_s,
                       "max_send_interval_s": th.max_send_interval_s,
                       "send_interval_avg_tol_s": th.send_interval_avg_tol_s,
                       "max_client_lag_ms": th.max_client_lag_ms},
        "window": {"t_go": t_go, "w0": w0, "w1": w1, "steady_state_s": rnd(steady_s, 2),
                   "rule": ("explicit" if args.window_start_s is not None else
                            "chain: last stream's first fresh + settle .. first stream's last fresh"
                            if args.chain else "t_go + warmup .. last fresh")},
        "aggregate": {**{k: v for k, v in series.items() if not k.endswith("_per_second")},
                      "generated_fps_server": rnd(generated_fps, 3),
                      "callback_ms_per_batch": rnd(callback_ms, 3),
                      "first_frame_p50_s": rnd(pct(all_first, 0.5), 3),
                      "first_frame_p95_s": rnd(pct(all_first, 0.95), 3),
                      "first_frame_max_s": rnd(max(all_first), 3) if all_first else None,
                      "barge_in_returns": len(all_barge),
                      "barge_in_return_p95_s": rnd(pct(all_barge, 0.95), 3),
                      "barge_in_return_max_s": rnd(max(all_barge), 3) if all_barge else None,
                      **worst},
        "server": {"pid": server_view.get("pid"), "media_flags": server_view.get("media_flags"),
                   "live_handoff_mode": server_view.get("live_handoff_mode"),
                   "lifetime_counters": server_view.get("lifetime_counters"),
                   "vp8": server_view.get("vp8"), "h264": server_view.get("h264"),
                   "loop_lag_p99_ms_max": max((l.get("p99_ms") or 0.0) for l in server_lag) if server_lag else None,
                   **sampler.summary(w0, w1)},
        "client": {"shards": len(shard_results), "peers_per_shard": args.peers_per_shard,
                   "loop_lag_p99_ms_max": lag_p99, "shard_cpu_s": [r["cpu_s"] for r in shard_results],
                   "shard_max_rss_mb": [r["max_rss_mb"] for r in shard_results],
                   "setup_s": rnd(setup_s, 2)},
        "poll": {"samples": len(state["server_samples"]), "errors": state["poll_errors"],
                 "last_error": state["last_poll_error"], "rtt_p99_s": rnd(pct(state["poll_rtt"], 0.99), 4)},
        "mode_detail": {"duty": args.duty, "barge_in_fraction": args.barge_in_fraction,
                        "turn_gap_s": args.turn_gap_s},
        "audio": {"distinct_max_reuse": reuse, "chain": chain_info or None,
                  "turn_wavs": [[Path(e["path"]).name for e in row] for row in plan] if not args.chain else None},
        "streams": per_stream,
        "series": {"fresh_per_second": series.get("fresh_per_second"),
                   "speakers_per_second": series.get("speakers_per_second")},
        "errors": errors,
    }
    agg = summary["aggregate"]
    print(f"{verdict} level N={level} label={args.label} mode={summary['mode']} steady={steady_s:.1f}s "
          f"agg_fresh_fps={agg.get('aggregate_fresh_fps_mean')} (min_1s={agg.get('aggregate_fresh_fps_min_1s')}) "
          f"generated_fps={agg.get('generated_fps_server')} fresh_frac_min={worst['fresh_fraction_min']} "
          f"held_run_max={worst['max_held_run']} stall_s_max={worst['stall_seconds_max']} "
          f"send_max={worst['send_interval_max_s']} first_frame_p95={agg.get('first_frame_p95_s')} "
          f"speakers_min={agg.get('speakers_min')} client_lag_p99={lag_p99} output_fps_ok={output_fps_ok} "
          f"rss_max_mb={summary['server'].get('server_rss_max_mb')} vram_max_mb={summary['server'].get('server_vram_max_mb')} "
          f"failed_checks={[k for k, v in checks.items() if not v]} invalid={[k for k, v in valid.items() if not v]}",
          flush=True)
    return summary


async def coordinator(args) -> int:
    import aiohttp
    if args.client_cpus:
        selected = parse_cpu_list(args.client_cpus)
        if not selected or not set(selected) <= set(os.sched_getaffinity(0)):
            raise ValueError("--client-cpus must be a nonempty subset of the allocated CPUs")
        topo = {"cpus": selected, "cores": "explicit_cpu_ids", "siblings": {}, "lscpu_agrees": None}
    else:
        topo = client_cpus_for_cores(args.client_cores)
    pin_error = pin_to(topo["cpus"]) if not args.no_pin else "disabled"
    print(f"[lt2] client cores {topo['cores']} -> CPUs {topo['cpus']} (siblings {topo['siblings']}, "
          f"lscpu agrees={topo['lscpu_agrees']}) pin_error={pin_error}", flush=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
        async with http.get(f"{args.base_url}/webrtc/sessions/stats", params={"view": "lifetime"}) as resp:
            if resp.status != 200:
                print(f"FAIL server lifetime view unavailable: HTTP {resp.status}", flush=True)
                return 3
            view = await resp.json()
    server_view = view.get("server") or {}
    if not server_view.get("lifetime_counters"):
        print("[lt2] WARNING: server runs without WEBRTC_LIFETIME_COUNTERS=1: stitching per-turn "
              "counters, no send ring (fresh fraction / cadence from counters only)", flush=True)
    corpus = load_corpus(Path(args.audio_dir), args.audio_class)
    if len(corpus) < 1:
        print(f"FAIL no WAVs in {args.audio_dir}", flush=True)
        return 3
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    summaries = []
    for index, level in enumerate(levels):
        if mem_available_gb() < args.abort_below_gb + 0.5:
            print(f"[lt2] skip level {level}: MemAvailable {mem_available_gb():.2f} GB", flush=True)
            break
        summary = await run_level(args, level, corpus, out_dir / f"{args.label}_n{level:02d}", topo["cpus"],
                                  server_view)
        summary["client_topology"] = topo
        path = out_dir / f"{args.label}_n{level:02d}.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        summaries.append({k: summary[k] for k in ("level", "verdict", "checks", "validity", "aggregate", "window")}
                         | {"server": {k: v for k, v in summary["server"].items() if k not in ("vp8", "h264")},
                            "json": str(path)})
        if not args.keep_chain_audio and args.chain:
            shutil.rmtree(out_dir / f"{args.label}_n{level:02d}" / "chain_audio", ignore_errors=True)
        if index < len(levels) - 1:
            await asyncio.sleep(args.cooldown_s)
    combined = out_dir / f"{args.label}_summary.json"
    combined.write_text(json.dumps({"schema": SCHEMA, "label": args.label, "base_url": args.base_url,
                                    "args": vars(args), "levels": summaries}, indent=2, default=str))
    verdicts = [s["verdict"] for s in summaries]
    print(f"[lt2] {args.label}: verdicts {dict(zip(levels, verdicts))} -> {combined}", flush=True)
    return 0 if verdicts and all(v == "PASS" for v in verdicts) else 1


# ===========================================================================
# self-test of the metric math (no server)
# ===========================================================================
def selftest(out: Optional[str]) -> int:
    results = []

    def check(name, ok, **detail):
        results.append({"test": name, "result": "pass" if ok else "fail", **detail})
        print(f"{'PASS' if ok else 'FAIL'} selftest {name} {json.dumps(detail, default=str)}")

    # 1. synthetic 10% hold in the send ring
    ring = RingStitcher()
    t, seq, entries = 1000.0, 0, []
    for i in range(20 * 60):
        kind = "h" if i % 10 == 9 else "f"
        seq += 1
        entries.append((seq, t, kind, i * 4500))
        t += 0.05
    for k in range(0, len(entries), 64):  # delivered in overlapping polls
        ring.add(entries[max(0, k - 10):k + 64])
    s = summarize_ring(ring, 1000.0, 1000.0 + 60.0, 20)
    check("ring_10pct_hold_detected", abs(s["fresh_fraction"] - 0.9) < 1e-6 and s["max_held_run"] == 1
          and abs(s["send_interval_avg_s"] - 0.05) < 1e-6 and s["ring_gaps"] == 0, summary=s)

    # 2. counters with turn-boundary resets (lifetime off) vs lifetime on
    stitch_off, stitch_on = CounterStitcher(), CounterStitcher()
    life = dict.fromkeys(LIFETIME_KEYS, 0.0)
    samples_off, samples_on = [], []
    t = 0.0
    generation = 0
    for turn in range(3):
        per_turn = dict.fromkeys(LIFETIME_KEYS, 0.0)  # start_live(): everything zeroed
        generation += 1
        for second in range(20):
            fresh, held = 18, 2  # 10% hold every second
            per_turn["frames_played"] += fresh
            per_turn["frames_duplicated"] += held
            per_turn["frames_received"] += 20
            per_turn["output_frames_sent"] += 20
            for key in ("frames_played", "frames_duplicated", "frames_received", "output_frames_sent"):
                life[key] += {"frames_played": fresh, "frames_duplicated": held}.get(key, 20)
            t += 1.0
            samples_off.append((t, stitch_off.update({"lifetime_enabled": False, "turn_counters": {
                **per_turn, "generation_id": generation}})))
            samples_on.append((t, stitch_on.update({"lifetime_enabled": True, "lifetime": dict(life)})))
        # end_live(): generation +1, frames_received zeroed, frames_played kept until next start_live()
        generation += 1
        per_turn["frames_received"] = 0.0
        per_turn["output_frames_sent"] += 20
        life["output_frames_sent"] += 20
        t += 1.0
        samples_off.append((t, stitch_off.update({"lifetime_enabled": False, "turn_counters": {
            **per_turn, "generation_id": generation}})))
        samples_on.append((t, stitch_on.update({"lifetime_enabled": True, "lifetime": dict(life)})))
    d_off = counter_deltas(samples_off, 0.0, t)
    d_on = counter_deltas(samples_on, 0.0, t)
    last_off, last_on = samples_off[-1][1], samples_on[-1][1]
    check("turn_boundary_resets_do_not_corrupt_deltas",
          d_off["fresh"] == d_on["fresh"] == 18 * 60 - 18 and abs(d_off["fresh_fraction"] - 0.9) < 1e-6
          and d_on["fresh_fraction"] == d_off["fresh_fraction"]
          and all(abs(last_off[k] - last_on[k]) < 1e-9 for k in LIFETIME_KEYS),
          stitched=d_off, lifetime=d_on, resets=stitch_off.resets,
          totals_equal={k: (last_off[k], last_on[k]) for k in LIFETIME_KEYS})

    # 3. ring gap accounting (poll slower than the ring)
    ring2 = RingStitcher()
    ring2.add([(i, 2000 + i * 0.05, "f", 0) for i in range(1, 65)])
    ring2.add([(i, 2000 + i * 0.05, "f", 0) for i in range(100, 164)])
    check("ring_gap_counted", ring2.gaps == 1 and ring2.missing == 35, gaps=ring2.gaps, missing=ring2.missing)

    # 4. aggregate fresh fps / speakers from rings
    rings = {}
    for sid in range(3):
        r = RingStitcher()
        r.add([(i + 1, 3000 + i * 0.05, "f" if (i % 20) else "h", 0) for i in range(20 * 10)])
        rings[sid] = r
    series = fresh_series_from_rings(rings, 3000.0, 3010.0)
    check("aggregate_fresh_and_speakers", series["aggregate_fresh_fps_mean"] == 57.0
          and series["speakers_min"] == 3, series={k: v for k, v in series.items() if not k.endswith("second")})

    # 5. first-frame latency matching
    lat = first_frame_latencies(
        [{"turn": 0, "t_post": 10.0, "status": 200, "accept_s": 0.1},
         {"turn": 1, "t_post": 30.0, "status": 200, "accept_s": 0.1}],
        {1: {"generation_id": 1, "started_monotonic": 10.2, "first_live_send_monotonic": 10.9},
         2: {"generation_id": 2, "started_monotonic": 30.3, "first_live_send_monotonic": 30.8}})
    check("first_frame_latency_matching", [l["first_frame_s"] for l in lat] == [0.9, 0.8], latencies=lat)

    # 6. a 200 ms strict-FIFO stall inside a speaking segment shows up as a slot deficit
    ring3 = RingStitcher()
    stamps, t = [], 4000.0
    for i in range(100):
        stamps.append((i + 1, t, "f", 0))
        t += 0.25 if i == 49 else 0.05
    ring3.add(stamps)
    s3 = summarize_ring(ring3, 3999.0, 4100.0, 20)
    check("stall_gap_slot_deficit", s3["slot_deficit"] == 4 and s3["send_gaps_over_100ms"] == 1
          and s3["send_interval_max_s"] == 0.25, summary=s3)

    passed = all(r["result"] == "pass" for r in results)
    if out:
        Path(out).write_text(json.dumps({"suite": "load_test_webrtc_v2_selftest", "passed": passed,
                                         "results": results}, indent=2))
    print(f"{'PASS' if passed else 'FAIL'} load_test_webrtc_v2 selftest {sum(r['result'] == 'pass' for r in results)}/{len(results)}")
    return 0 if passed else 1


# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MuseTalk WebRTC load harness v2 (server-side lifetime counters)")
    p.add_argument("--base-url", default="http://127.0.0.1:8300")
    p.add_argument("--avatar-ids", default="japanese_realtime_talking_7d94520b7f",
                   help="comma list; streams are assigned round-robin (5 per identity at N=15 with 3 ids)")
    p.add_argument("--levels", default="1", help="comma list of N (streams) run sequentially")
    p.add_argument("--label", default="run")
    p.add_argument("--out-dir", default="load_test_v2_out")
    p.add_argument("--audio-dir", default=str(ROOT / "experiments/throughput300_candidate/audio_corpus"))
    p.add_argument("--audio-class", default="turn", choices=["turn", "long", "any"])
    p.add_argument("--turns", type=int, default=3, help="sequential turns per stream (ignored with --chain)")
    p.add_argument("--wav-list", default=None,
                   help="comma list of WAVs used as the turn list of EVERY stream (exactness A/B runs)")
    p.add_argument("--chain", action="store_true", help="one concatenated turn per stream (S1)")
    p.add_argument("--chain-seconds", type=float, default=220.0)
    p.add_argument("--chain-dir", default=None)
    p.add_argument("--keep-chain-audio", action="store_true")
    p.add_argument("--turn-gap-s", type=float, default=1.0)
    p.add_argument("--duty", type=float, default=None,
                   help="0..1: Poisson idle gaps for this duty cycle (S2 conversational); default fixed gaps")
    p.add_argument("--barge-in-fraction", type=float, default=0.0,
                   help="S5 (pose sessions): abort this fraction of turns with assistant_turn_aborted")
    p.add_argument("--barge-in-min-s", type=float, default=1.5)
    p.add_argument("--barge-in-max-s", type=float, default=4.0)
    p.add_argument("--tail-s", type=float, default=2.0)
    p.add_argument("--stagger-s", type=float, default=0.0)
    p.add_argument("--stagger-join", action="store_true",
                   help="with --stagger-s: create/connect each delayed stream at its start time (real mid-run join)")
    p.add_argument("--duration-s", type=float, default=None,
                   help="stop starting new turns this long after a stream's first turn (not an error)")
    p.add_argument("--post-retries", type=int, default=3, help="409 retries per turn POST (0.5 s apart)")
    p.add_argument("--max-consecutive-post-failures", type=int, default=0,
                   help="keep a stream speaking after this many failed turns in a row (default 0: stop it)")
    p.add_argument("--record-streams", default="", help="with --trace-dir: comma list of stream indexes to record")
    p.add_argument("--record-at-s", type=float, default=60.0, help="recording starts this long after the stream connects")
    p.add_argument("--record-seconds", type=float, default=90.0)
    p.add_argument("--trace-dir", default=None,
                   help="write per-frame client arrival traces, loop-lag spikes, 1 Hz server counters and send rings")
    p.add_argument("--musetalk-fps", type=int, default=20)
    p.add_argument("--playback-fps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-duration", type=int, default=1)
    p.add_argument("--prebuffer-seconds", type=float, default=None)
    p.add_argument("--pose-set-file", default=None, help="pose protocol v1 session (wall-style turns)")
    p.add_argument("--pose-plan", default=None, help="JSON pose plan (default: the wall plan)")
    p.add_argument("--peers-per-shard", type=int, default=5)
    p.add_argument("--shard-poll", choices=("session", "all"), default="session",
                   help="how each shard learns its turns ended: GET /webrtc/sessions/{id}/status per own session "
                        "(default), or the older GET /webrtc/sessions/stats?view=lifetime of every session")
    p.add_argument("--client-cores", default=DEFAULT_CLIENT_CORES, help="physical core ids for all client processes")
    p.add_argument("--client-cpus", default="", help="explicit allocated logical CPU IDs/ranges; overrides --client-cores")
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--ring", type=int, default=64)
    p.add_argument("--poll-interval-s", type=float, default=1.0)
    p.add_argument("--ignore-ice-servers", action="store_true", help="loopback: host candidates only")
    p.add_argument("--ice-gather-timeout-s", type=float, default=5.0)
    p.add_argument("--connection-timeout-s", type=float, default=30.0)
    p.add_argument("--setup-timeout-s", type=float, default=120.0)
    p.add_argument("--keep-sessions", action="store_true")
    p.add_argument("--cooldown-s", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--warmup-s", type=float, default=20.0)
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument("--window-start-s", type=float, default=None)
    p.add_argument("--window-s", type=float, default=None)
    p.add_argument("--min-steady-s", type=float, default=180.0)
    p.add_argument("--min-fresh-fraction", type=float, default=0.995)
    p.add_argument("--max-held-run", type=int, default=2)
    p.add_argument("--max-stall-s", type=float, default=0.0)
    p.add_argument("--max-send-interval-s", type=float, default=0.100)
    p.add_argument("--send-interval-avg-tol-s", type=float, default=0.001)
    p.add_argument("--max-client-lag-ms", type=float, default=20.0)
    p.add_argument("--abort-below-gb", type=float, default=4.0)
    p.add_argument("--build-corpus", default=None, metavar="DIR")
    p.add_argument("--min-corpus", type=int, default=20)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--selftest-out", default=None)
    p.add_argument("--shard-worker", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.shard_worker:
        return asyncio.run(shard_main(args.shard_worker))
    if args.selftest:
        return selftest(args.selftest_out)
    if args.build_corpus:
        manifest = build_corpus(Path(args.build_corpus), min_turn=args.min_corpus)
        return 0 if manifest["turn_count"] >= args.min_corpus else 1
    return asyncio.run(coordinator(args))


if __name__ == "__main__":
    sys.exit(main())
