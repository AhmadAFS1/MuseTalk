#!/usr/bin/env python3
"""Thread/process scaling of APIAvatar.compose_frame (+ optional colorconv) on a real avatar."""
import argparse, functools, json, multiprocessing as mp, os, sys, threading, time
import torch
torch.load = functools.partial(torch.load, map_location="cpu")
sys.path.insert(0, "/workspace/MuseTalk")
sys.path.insert(0, "/workspace/MuseTalk/scripts")
import numpy as np, cv2
cv2.setNumThreads(1)
import av
from benchmark_compose_frame import load_avatar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--avatar", default="japanese_realtime_talking_7d94520b7f")
ap.add_argument("--workers", default="1,8,16,24")
ap.add_argument("--secs", type=float, default=4.0)
ap.add_argument("--chain", choices=["compose", "compose+pyavcc", "compose+cv2cc", "pyavcc"], default="compose")
args = ap.parse_args()

avatar = load_avatar(args.avatar, 8, "v15")
n_cycle = len(avatar.coord_list_cycle)
bbox_wh = [(c[2] - c[0], c[3] - c[1]) for c in avatar.coord_list_cycle]
face = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)


def one(i):
    out = avatar.compose_frame(face, i % n_cycle) if args.chain != "pyavcc" else avatar.frame_list_cycle[i % n_cycle]
    if args.chain in ("compose+pyavcc", "pyavcc"):
        av.VideoFrame.from_ndarray(out, format="bgr24").reformat(format="yuv420p")
    elif args.chain == "compose+cv2cc":
        av.VideoFrame.from_ndarray(cv2.cvtColor(out, cv2.COLOR_BGR2YUV_I420), format="yuv420p")


def worker(stop_at, counter, idx):
    n = 0; i = idx * 37
    while time.perf_counter() < stop_at:
        one(i); i += 1; n += 1
    counter.append(n)


def proc_worker(stop_at, q, idx):
    c = []
    worker(stop_at, c, idx)
    q.put(c[0])


res = {"avatar": args.avatar, "chain": args.chain, "frame_shape": list(avatar.frame_list_cycle[0].shape),
       "bbox_w_mean": round(float(np.mean([b[0] for b in bbox_wh])), 1),
       "bbox_h_mean": round(float(np.mean([b[1] for b in bbox_wh])), 1), "runs": []}
for _ in range(50):
    one(_)
for mode in ("thread", "proc"):
    for n in [int(x) for x in args.workers.split(",")]:
        r0 = os.times()
        start = time.perf_counter() + 0.2
        stop_at = start + args.secs
        if mode == "thread":
            counts = []
            ths = [threading.Thread(target=worker, args=(stop_at, counts, k)) for k in range(n)]
            for t in ths: t.start()
            for t in ths: t.join()
        else:
            ctx = mp.get_context("fork"); q = ctx.Queue()
            ps = [ctx.Process(target=proc_worker, args=(stop_at, q, k)) for k in range(n)]
            for p in ps: p.start()
            counts = [q.get() for _ in ps]
            for p in ps: p.join()
        r1 = os.times()
        wall = time.perf_counter() - (start - 0.2)
        cpu = (r1.user - r0.user) + (r1.system - r0.system) + (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system)
        total = sum(counts)
        res["runs"].append({"mode": mode, "workers": n, "agg_fps": round(total / args.secs, 1),
                            "per_worker_ms": round(args.secs * 1000 * n / total, 3),
                            "cores_used": round(cpu / wall, 2), "cpu_ms_per_frame": round(cpu * 1000 / total, 3)})
        print(json.dumps(res["runs"][-1]), flush=True)
print(json.dumps(res))
