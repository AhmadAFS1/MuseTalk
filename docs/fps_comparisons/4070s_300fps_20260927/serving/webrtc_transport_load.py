"""Loopback aiortc load test: can ONE Python process / ONE asyncio loop carry
K concurrent 20 fps WebRTC video+audio streams (ex-GPU)?

Sender process mirrors the MuseTalk live path cost structure:
  * mode=idle : track.recv() decodes the idle mp4 on the event loop (like
                IdleVideoStreamTrack.read_frame inside SwitchableVideoStreamTrack.recv)
  * mode=live : a single "scheduler" thread produces 8-frame BGR batches per stream
                and hands each batch to the loop via run_coroutine_threadsafe(...).result(),
                where BGR->yuv420p (from_ndarray+reformat) runs ON THE LOOP (like
                push_bgr_frames_batch); recv() pops from an asyncio.Queue.
  * enc=x264default | x264fast (veryfast, threads=1) | nvenc
Receivers run in separate processes and just count frames.
Nothing in /workspace is modified; no network egress (iceServers=[]).
"""
import argparse
import asyncio
import fractions
import json
import multiprocessing as mp
import os
import statistics
import threading
import time

IDLE = ("/workspace/experiments/chinese_bob_webrtc_20260927/h3_batch/"
        "chinese_bob_pink_bedroom_fd26f914/idle/source.mp4")
FPS = 20
TB = fractions.Fraction(1, 90000)


def patch_encoder(enc_mode):
    import av
    import aiortc.codecs.h264 as h264
    h264.MAX_BITRATE = 6_000_000
    if enc_mode == "x264default":
        return
    orig = h264.H264Encoder._encode_frame

    def _encode_frame(self, frame, force_keyframe):
        if self.codec and (frame.width != self.codec.width or frame.height != self.codec.height):
            self.codec = None
        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        else:
            frame.pict_type = av.video.frame.PictureType.NONE
        if self.codec is None:
            if enc_mode == "nvenc":
                cc = av.CodecContext.create("h264_nvenc", "w")
                opts = {"preset": "p1", "tune": "ull", "bf": "0", "rc": "cbr",
                        "delay": "0", "zerolatency": "1", "g": "3000", "profile": "baseline"}
            else:
                cc = av.CodecContext.create("libx264", "w")
                opts = {"preset": "veryfast", "tune": "zerolatency", "threads": "1",
                        "level": "31", "profile": "baseline"}
            cc.width, cc.height, cc.pix_fmt = frame.width, frame.height, "yuv420p"
            cc.bit_rate = 2_000_000
            cc.framerate = fractions.Fraction(30, 1)
            cc.time_base = fractions.Fraction(1, 30)
            cc.options = opts
            self.codec = cc
        data = b""
        for p in self.codec.encode(frame):
            data += bytes(p)
        if data:
            yield from self._split_bitstream(data)

    h264.H264Encoder._encode_frame = _encode_frame


def sender_main(args, conns, result_q):
    import av
    import numpy as np
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
    from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack, MediaStreamError
    patch_encoder(args.enc)

    # pre-decode a few BGR frames for the live producer
    c = av.open(IDLE)
    it = c.decode(c.streams.video[0])
    bgr_pool = [next(it).to_ndarray(format="bgr24") for _ in range(24)]
    c.close()

    class PacedTrack(VideoStreamTrack):
        def __init__(self):
            super().__init__()
            self._t0 = None
            self._n = 0

        async def _pace(self):
            if self._t0 is None:
                self._t0 = time.monotonic()
            target = self._t0 + self._n / FPS
            wait = target - time.monotonic()
            if wait > 0.001:
                await asyncio.sleep(wait)
            pts = int(self._n * 90000 / FPS)
            self._n += 1
            return pts

    class IdleDecodeTrack(PacedTrack):
        def __init__(self):
            super().__init__()
            self._open()
            self._acc = 0.0

        def _open(self):
            self._c = av.open(IDLE)
            self._it = self._c.decode(self._c.streams.video[0])

        def _read(self):
            try:
                f = next(self._it)
            except StopIteration:
                self._c.close()
                self._open()
                f = next(self._it)
            return f.reformat(format="yuv420p")

        async def recv(self):
            pts = await self._pace()
            self._acc += 24.0 / FPS  # 24 fps source on a 20 fps output, like the pose clips
            steps = int(self._acc)
            self._acc -= steps
            f = None
            for _ in range(max(1, steps)):
                f = self._read()
            f.pts, f.time_base = pts, TB
            return f

    _cache = []
    if args.predecoded_idle:
        cc = av.open(IDLE)
        for fr in cc.decode(cc.streams.video[0]):
            _cache.append(fr.reformat(format="yuv420p").to_ndarray())
        cc.close()

    class PreDecodedIdleTrack(PacedTrack):
        def __init__(self):
            super().__init__()
            self._i = 0
            self._acc = 0.0

        async def recv(self):
            pts = await self._pace()
            self._acc += 24.0 / FPS
            steps = int(self._acc)
            self._acc -= steps
            self._i = (self._i + max(1, steps)) % len(_cache)
            f = av.VideoFrame.from_ndarray(_cache[self._i], format="yuv420p")
            f.pts, f.time_base = pts, TB
            return f

    class LivePushTrack(PacedTrack):
        def __init__(self):
            super().__init__()
            self.q = asyncio.Queue(maxsize=400)
            self.last = None
            self.underruns = 0

        async def push_batch(self, frames):
            for b in frames:
                vf = av.VideoFrame.from_ndarray(b, format="bgr24").reformat(format="yuv420p")
                await self.q.put(vf)
            return self.q.qsize()

        async def recv(self):
            pts = await self._pace()
            try:
                f = self.q.get_nowait()
                self.last = f
            except asyncio.QueueEmpty:
                self.underruns += 1
                f = self.last
                if f is None:
                    f = av.VideoFrame.from_ndarray(bgr_pool[0], format="bgr24").reformat(format="yuv420p")
                    self.last = f
            out = av.VideoFrame.from_ndarray(f.to_ndarray(), format="yuv420p") if f is self.last and self.q.qsize() == 0 and False else f
            out.pts, out.time_base = pts, TB
            return out

    async def amain():
        loop = asyncio.get_running_loop()
        pcs, tracks = [], []
        for k in range(args.streams):
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            is_idle = args.mode == "idle" or (args.mode == "mixed" and k % 2 == 0)
            vt = (PreDecodedIdleTrack() if args.predecoded_idle else IdleDecodeTrack()) if is_idle else LivePushTrack()
            pc.addTrack(vt)
            pc.addTrack(AudioStreamTrack())
            for t in pc.getTransceivers():
                if t.kind == "video":
                    from aiortc import RTCRtpSender
                    caps = [c for c in RTCRtpSender.getCapabilities("video").codecs if c.mimeType.lower() == "video/h264"]
                    t.setCodecPreferences(caps)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            conn = conns[k % len(conns)]
            conn.send(("offer", k, pc.localDescription.sdp))
            pcs.append(pc)
            tracks.append(vt)
        for k in range(args.streams):
            conn = conns[k % len(conns)]
            kind, kk, sdp = await loop.run_in_executor(None, conn.recv)
            await pcs[kk].setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

        # loop-lag monitor
        lags = []

        async def lag_monitor():
            while True:
                t = time.perf_counter()
                await asyncio.sleep(0.005)
                lags.append((time.perf_counter() - t - 0.005) * 1000)

        mon = asyncio.create_task(lag_monitor())

        # live producer thread == single scheduler thread
        push_lat = []
        stop = threading.Event()

        def producer():
            period = 8.0 / FPS
            next_due = [time.monotonic() + 0.5] * args.streams
            while not stop.is_set():
                now = time.monotonic()
                did = False
                for k, vt in enumerate(tracks):
                    if not isinstance(vt, LivePushTrack):
                        continue
                    if now >= next_due[k]:
                        batch = [bgr_pool[(i + k) % 24].copy() for i in range(8)]
                        t = time.perf_counter()
                        asyncio.run_coroutine_threadsafe(vt.push_batch(batch), loop).result()
                        push_lat.append((time.perf_counter() - t) * 1000)
                        next_due[k] += period
                        did = True
                if not did:
                    time.sleep(0.002)

        th = None
        if args.mode in ("live", "mixed"):
            th = threading.Thread(target=producer, daemon=True)
            th.start()

        probe = []

        def gil_probe():
            def work():
                x = 0
                for i in range(20000):
                    x += i * i
                return x
            t = time.perf_counter(); work(); base = time.perf_counter() - t
            probe.append(("base", base * 1000))
            while not stop.is_set():
                t = time.perf_counter(); work(); probe.append(("run", (time.perf_counter() - t) * 1000))
                time.sleep(0.01)

        if args.gil_probe:
            threading.Thread(target=gil_probe, daemon=True).start()

        main_tid = threading.get_native_id()

        def ticks(tid=None):
            p = f"/proc/self/task/{tid}/stat" if tid else "/proc/self/stat"
            v = open(p).read().rsplit(")", 1)[1].split()
            return int(v[11]) + int(v[12])

        await asyncio.sleep(args.warmup)
        lags.clear()
        push_lat.clear()
        probe.clear()
        p0, m0, w0 = ticks(), ticks(main_tid), time.monotonic()
        await asyncio.sleep(args.duration)
        p1, m1, w1 = ticks(), ticks(main_tid), time.monotonic()
        hz = os.sysconf("SC_CLK_TCK")
        stop.set()
        mon.cancel()
        lags_sorted = sorted(lags) or [0]
        res = {
            "process_cores": (p1 - p0) / hz / (w1 - w0),
            "loop_thread_core_frac": (m1 - m0) / hz / (w1 - w0),
            "loop_lag_ms_p50": lags_sorted[len(lags_sorted) // 2],
            "loop_lag_ms_p99": lags_sorted[int(len(lags_sorted) * 0.99)],
            "loop_lag_ms_max": lags_sorted[-1],
            "threads": len(os.listdir("/proc/self/task")),
        }
        runs = sorted(v for k2, v in probe if k2 == "run")
        if runs:
            res.update({"gil_probe_ms_p50": runs[len(runs)//2], "gil_probe_ms_p99": runs[int(len(runs)*0.99)]})
        if push_lat:
            ps = sorted(push_lat)
            res.update({"push_batch_ms_p50": ps[len(ps) // 2], "push_batch_ms_p99": ps[int(len(ps) * 0.99)],
                        "push_batches": len(ps)})
        if args.mode in ("live", "mixed"):
            res["underruns_total"] = sum(t.underruns for t in tracks)
        for conn in conns:
            conn.send(("done", 0, ""))
        result_q.put(("sender", res))
        await asyncio.sleep(0.5)
        for pc in pcs:
            await pc.close()

    asyncio.run(amain())


def receiver_main(conn, duration, warmup, result_q):
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration

    async def amain():
        loop = asyncio.get_running_loop()
        pcs = {}
        arrivals = {}
        t_start = [None]

        async def consume(k, track):
            while True:
                try:
                    await track.recv()
                except Exception:
                    return
                arrivals[k].append(time.monotonic())

        while True:
            kind, k, sdp = await loop.run_in_executor(None, conn.recv)
            if kind == "offer":
                pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                arrivals[k] = []

                @pc.on("track")
                def on_track(track, k=k):
                    if track.kind == "video":
                        asyncio.ensure_future(consume(k, track))

                await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
                ans = await pc.createAnswer()
                await pc.setLocalDescription(ans)
                conn.send(("answer", k, pc.localDescription.sdp))
                pcs[k] = pc
            elif kind == "done":
                break
        now = time.monotonic()
        stats = {}
        for k, a in arrivals.items():
            win = [t for t in a if t >= now - duration]
            gaps = [(b - x) * 1000 for x, b in zip(win, win[1:])]
            stats[k] = {"fps": len(win) / duration,
                        "gap_ms_p99": sorted(gaps)[int(len(gaps) * 0.99)] if gaps else None,
                        "gap_ms_max": max(gaps) if gaps else None}
        result_q.put(("recv", stats))
        for pc in pcs.values():
            await pc.close()

    asyncio.run(amain())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--mode", choices=["idle", "live", "mixed"], default="idle")
    ap.add_argument("--gil-probe", action="store_true")
    ap.add_argument("--predecoded-idle", action="store_true")
    ap.add_argument("--enc", choices=["x264default", "x264fast", "nvenc"], default="x264default")
    ap.add_argument("--receivers", type=int, default=3)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--warmup", type=float, default=6.0)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    pairs = [ctx.Pipe() for _ in range(args.receivers)]
    recvs = [ctx.Process(target=receiver_main, args=(b, args.duration, args.warmup, result_q)) for _, b in pairs]
    for p in recvs:
        p.start()
    snd = ctx.Process(target=sender_main, args=(args, [a for a, _ in pairs], result_q))
    snd.start()
    sender_stats, recv_stats = None, {}
    for _ in range(1 + args.receivers):
        try:
            kind, payload = result_q.get(timeout=args.duration + args.warmup + 120)
        except Exception:
            break
        if kind == "sender":
            sender_stats = payload
        else:
            recv_stats.update(payload)
    snd.join(timeout=30)
    for p in recvs:
        p.join(timeout=30)
    for p in [snd] + recvs:
        if p.is_alive():
            p.terminate()
    fps = [v["fps"] for v in recv_stats.values()]
    gmax = [v["gap_ms_max"] for v in recv_stats.values() if v["gap_ms_max"] is not None]
    gp99 = [v["gap_ms_p99"] for v in recv_stats.values() if v["gap_ms_p99"] is not None]
    summary = {"config": vars(args), "sender": sender_stats,
               "recv_streams": len(fps),
               "recv_fps_min": min(fps) if fps else None,
               "recv_fps_mean": statistics.mean(fps) if fps else None,
               "recv_aggregate_fps": sum(fps) if fps else None,
               "recv_gap_ms_p99_worst": max(gp99) if gp99 else None,
               "recv_gap_ms_max_worst": max(gmax) if gmax else None}
    print(json.dumps(summary))
    with open("webrtc_transport_load_results.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
