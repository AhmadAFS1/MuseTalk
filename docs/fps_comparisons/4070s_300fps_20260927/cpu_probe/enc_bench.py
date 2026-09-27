#!/usr/bin/env python3
"""Encoder CPU probe: per-frame encode cost and N-stream concurrency.

Read-only w.r.t. /workspace. Source frames = consecutive avatar full_imgs PNGs
(ping-pong so motion is continuous), converted once to yuv420p VideoFrames so the
timed region is encode (+ aiortc RTP payload packetization for *_aiortc/native).
"""
import argparse, fractions, glob, json, multiprocessing as mp, os, resource, statistics, sys, threading, time

AVATAR = "/workspace/MuseTalk/results/v15/avatars/japanese_realtime_talking_7d94520b7f/full_imgs"
VT = fractions.Fraction(1, 90000)


def clone_frames(frames):
    import av
    return [av.VideoFrame.from_ndarray(f.to_ndarray(), format="yuv420p") for f in frames]


def load_frames(w, h, n):
    import cv2, av
    files = sorted(glob.glob(os.path.join(AVATAR, "*.png")))[:n]
    out = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out.append(av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p"))
    return out + out[-2:0:-1]  # ping-pong


def make_encoder(codec, w, h, bitrate, threads):
    """Returns encode(frame, idx)->nbytes callable and a description."""
    import av
    if codec in ("vp8_aiortc", "vp8_native"):
        if codec == "vp8_native":
            os.environ.setdefault("WEBRTC_NATIVE_VP8_MAX_BITRATE_BPS", "3500000")
            os.environ.setdefault("WEBRTC_NATIVE_VP8_BITRATE_FLOOR_BPS", str(bitrate))
            sys.path.insert(0, "/workspace/MuseTalk")
            from scripts.webrtc_native_vp8 import load_native_encoder
            cls = load_native_encoder("/workspace/MuseTalk/.runtime/native_vp8")
            if threads:
                sys.modules["aiortc.codecs._musetalk_native_vpx_111"].number_of_threads = lambda p, c, _t=threads: _t
            enc = cls()
        else:
            import aiortc.codecs.vpx as vpx
            vpx.MAX_BITRATE = max(vpx.MAX_BITRATE, bitrate)  # in-process only
            if threads:
                vpx.number_of_threads = lambda p, c, _t=threads: _t
            enc = vpx.Vp8Encoder()
        enc.target_bitrate = bitrate

        def encode(frame, idx):
            frame.pts = idx * 4500
            frame.time_base = VT
            frame.duration = 4500
            payloads, _ts = enc.encode(frame, force_keyframe=False)
            return sum(len(p) for p in payloads)
        return encode, getattr(enc, "cfg", None)

    if codec == "vp8_pyav":  # same option set as aiortc stock encoder, raw PyAV
        ctx = av.CodecContext.create("libvpx", "w")
        ctx.width, ctx.height = w, h
        ctx.bit_rate = bitrate
        ctx.pix_fmt = "yuv420p"
        ctx.framerate = fractions.Fraction(20, 1)
        ctx.time_base = VT
        ctx.gop_size = 3000
        ctx.qmin, ctx.qmax = 2, 56
        ctx.options = {"bufsize": str(bitrate), "cpu-used": "-6", "deadline": "realtime",
                       "lag-in-frames": "0", "minrate": str(bitrate), "maxrate": str(bitrate),
                       "noise-sensitivity": "4", "overshoot-pct": "15", "partitions": "0",
                       "static-thresh": "1", "undershoot-pct": "100"}
        ctx.thread_count = threads or (2 if w * h > 640 * 480 else 1)
    elif codec.startswith("x264"):
        ctx = av.CodecContext.create("libx264", "w")
        ctx.width, ctx.height = w, h
        ctx.bit_rate = bitrate
        ctx.pix_fmt = "yuv420p"
        ctx.framerate = fractions.Fraction(20, 1)
        ctx.time_base = fractions.Fraction(1, 20)
        opts = {"tune": "zerolatency", "level": "31"}
        if codec == "x264_ultrafast":
            opts["preset"] = "ultrafast"
        elif codec == "x264_veryfast":
            opts["preset"] = "veryfast"
        # x264_aiortc: aiortc default (preset medium) + zerolatency + Baseline
        opts.update({"maxrate": str(bitrate), "bufsize": str(bitrate)})
        ctx.options = opts
        ctx.profile = "Baseline"
        if threads:
            ctx.thread_count = threads
    elif codec == "nvenc":  # api_server.enable_h264_nvenc options
        ctx = av.CodecContext.create("h264_nvenc", "w")
        ctx.width, ctx.height = w, h
        ctx.bit_rate = bitrate
        ctx.pix_fmt = "yuv420p"
        ctx.framerate = fractions.Fraction(20, 1)
        ctx.time_base = fractions.Fraction(1, 20)
        ctx.options = {"preset": "p2", "tune": "ll", "bf": "0", "rc": os.environ.get("NVENC_RC", "cbr_ld_hq"),
                       "maxrate": str(bitrate), "bufsize": str(bitrate * 2), "g": "40",
                       "delay": "0", "zerolatency": "1"}
    else:
        raise ValueError(codec)
    ctx.open()

    def encode(frame, idx):
        frame.pts = idx
        frame.time_base = ctx.time_base if codec != "vp8_pyav" else VT
        if codec == "vp8_pyav":
            frame.pts = idx * 4500
        frame.pict_type = 0
        return sum(p.size for p in ctx.encode(frame))
    return encode, ctx


def run_stream(args, frames, sid, barrier, results):
    enc, _ = make_encoder(args.codec, args.w, args.h, args.bitrate, args.threads)
    # warmup (keyframe + rate control settle), untimed
    for i in range(args.warmup):
        enc(frames[i % len(frames)], i)
    if barrier is not None:
        barrier.wait()
    lat, nbytes, late = [], 0, 0
    period = 1.0 / args.fps if args.paced else 0.0
    t0 = time.perf_counter()
    c0 = time.thread_time() if args.mode == "thread" else time.process_time()
    for k in range(args.frames):
        i = args.warmup + k
        if period:
            due = t0 + k * period
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
            elif now - due > period:
                late += 1
        s = time.perf_counter()
        nbytes += enc(frames[i % len(frames)], i)
        lat.append((time.perf_counter() - s) * 1000)
    wall = time.perf_counter() - t0
    cpu = (time.thread_time() if args.mode == "thread" else time.process_time()) - c0
    results.append({"sid": sid, "wall": wall, "frames": args.frames, "lat": lat, "bytes": nbytes,
                    "late": late, "cpu_self": cpu})


def proc_entry(args, sid, barrier, q):
    frames = load_frames(args.w, args.h, args.src_frames)
    res = []
    run_stream(args, frames, sid, barrier, res)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    res[0]["cpu_total_proc"] = ru.ru_utime + ru.ru_stime
    q.put(res[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=832)
    ap.add_argument("--bitrate", type=int, default=2_500_000)
    ap.add_argument("--threads", type=int, default=0, help="codec threads (0=repo default)")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--mode", choices=["thread", "proc"], default="proc")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--src-frames", type=int, default=40)
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    ruc0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t_start = time.perf_counter()
    if args.mode == "thread":
        frames = load_frames(args.w, args.h, args.src_frames)
        barrier = threading.Barrier(args.streams + 1)
        results = []
        per = [frames] + [clone_frames(frames) for _ in range(args.streams - 1)]
        ths = [threading.Thread(target=run_stream, args=(args, per[s], s, barrier, results))
               for s in range(args.streams)]
        for t in ths:
            t.start()
        barrier.wait()
        ru_b = resource.getrusage(resource.RUSAGE_SELF)
        tb = time.perf_counter()
        for t in ths:
            t.join()
        te = time.perf_counter()
        ru_e = resource.getrusage(resource.RUSAGE_SELF)
        cpu_window = (ru_e.ru_utime + ru_e.ru_stime) - (ru_b.ru_utime + ru_b.ru_stime)
        wall_window = te - tb
    else:
        ctx = mp.get_context("fork")
        barrier = ctx.Barrier(args.streams)
        q = ctx.Queue()
        ps = [ctx.Process(target=proc_entry, args=(args, s, barrier, q)) for s in range(args.streams)]
        for p in ps:
            p.start()
        results = [q.get() for _ in ps]
        for p in ps:
            p.join()
        # windowed CPU: use per-stream timed-loop process_time (includes codec worker threads)
        cpu_window = sum(r["cpu_self"] for r in results)
        wall_window = max(r["wall"] for r in results)

    all_lat = [x for r in results for x in r["lat"]]
    total_frames = sum(r["frames"] for r in results)
    agg_fps = total_frames / wall_window
    out = {
        "tag": args.tag, "codec": args.codec, "size": f"{args.w}x{args.h}", "bitrate": args.bitrate,
        "codec_threads": args.threads or "repo_default", "streams": args.streams, "mode": args.mode,
        "paced": args.paced, "frames_per_stream": args.frames,
        "lat_ms_mean": round(statistics.fmean(all_lat), 3),
        "lat_ms_p50": round(statistics.median(all_lat), 3),
        "lat_ms_p95": round(sorted(all_lat)[int(0.95 * len(all_lat)) - 1], 3),
        "lat_ms_max": round(max(all_lat), 3),
        "agg_fps": round(agg_fps, 1),
        "wall_s": round(wall_window, 3),
        "cpu_s": round(cpu_window, 3),
        "cores_used": round(cpu_window / wall_window, 2),
        "cpu_ms_per_frame": round(cpu_window * 1000 / total_frames, 3),
        "kbps_actual": round(sum(r["bytes"] for r in results) * 8 * args.fps / total_frames / 1000, 1),
        "late_frames": sum(r["late"] for r in results),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
