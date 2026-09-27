#!/usr/bin/env python3
"""MediaPipe FaceMesh (refine_landmarks=True) per-frame cost + process scaling.

Config mirrors experiments/chin_fps_validation_20260927/tracker_worker.py:
static_image_mode=False (tracking), max_num_faces=1, refine_landmarks=True, conf .5/.5,
cv2.setNumThreads(2), BGR->RGB conversion inside the timed region.
"""
import argparse, glob, json, multiprocessing as mp, os, pickle, resource, statistics, sys, time
import cv2, numpy as np

AVD = "/workspace/MuseTalk/results/v15/avatars/japanese_realtime_talking_7d94520b7f"


def load(kind, n):
    files = sorted(glob.glob(AVD + "/full_imgs/*.png"))[:n]
    coords = pickle.load(open(AVD + "/coords.pkl", "rb"))
    out = []
    for i, f in enumerate(files):
        im = cv2.imread(f)
        x1, y1, x2, y2 = [int(v) for v in coords[i]]
        if kind == "full":
            out.append(im)
        elif kind == "bbox":  # bbox + 20% margin (face crop as tracked region)
            mw, mh = int(0.2 * (x2 - x1)), int(0.2 * (y2 - y1))
            out.append(np.ascontiguousarray(im[max(0, y1 - mh):y2 + mh, max(0, x1 - mw):x2 + mw]))
        elif kind == "256":
            mw, mh = int(0.2 * (x2 - x1)), int(0.2 * (y2 - y1))
            out.append(cv2.resize(im[max(0, y1 - mh):y2 + mh, max(0, x1 - mw):x2 + mw], (256, 256)))
    return out + out[-2:0:-1]


def run(kind, frames_n, static, q=None, barrier=None, cv2_threads=2, period=0.0):
    import mediapipe as mpipe
    cv2.setNumThreads(cv2_threads)
    frames = load(kind, 60)
    mesh = mpipe.solutions.face_mesh.FaceMesh(static_image_mode=static, max_num_faces=1, refine_landmarks=True,
                                              min_detection_confidence=.5, min_tracking_confidence=.5)
    for i in range(20):
        mesh.process(cv2.cvtColor(frames[i % len(frames)], cv2.COLOR_BGR2RGB))
    if barrier is not None:
        barrier.wait()
    lat, miss = [], 0
    t0 = time.perf_counter(); c0 = time.process_time()
    for i in range(frames_n):
        if period:
            due = t0 + i * period
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
        s = time.perf_counter()
        r = mesh.process(cv2.cvtColor(frames[i % len(frames)], cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks:
            miss += 1
        else:
            _ = [(p.x, p.y) for p in r.multi_face_landmarks[0].landmark]
        lat.append((time.perf_counter() - s) * 1000)
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    mesh.close()
    res = {"wall": wall, "cpu": cpu, "lat": lat, "miss": miss, "shape": list(frames[0].shape)}
    if q is not None:
        q.put(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="full")
    ap.add_argument("--procs", default="1")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--paced", type=float, default=0.0, help="fps per process")
    a = ap.parse_args()
    for n in [int(x) for x in a.procs.split(",")]:
        if n == 1:
            rs = [run(a.kind, a.frames, a.static, period=(1/a.paced if a.paced else 0.0))]
        else:
            ctx = mp.get_context("spawn"); q = ctx.Queue(); b = ctx.Barrier(n)
            ps = [ctx.Process(target=run, args=(a.kind, a.frames, a.static, q, b, 2, (1/a.paced if a.paced else 0.0))) for _ in range(n)]
            for p in ps: p.start()
            rs = [q.get() for _ in ps]
            for p in ps: p.join()
        lat = [x for r in rs for x in r["lat"]]
        wall = max(r["wall"] for r in rs); cpu = sum(r["cpu"] for r in rs)
        print(json.dumps({"kind": a.kind, "shape": rs[0]["shape"], "static": a.static, "procs": n,
                          "lat_ms_mean": round(statistics.fmean(lat), 3), "lat_ms_p50": round(statistics.median(lat), 3),
                          "lat_ms_p95": round(sorted(lat)[int(.95 * len(lat)) - 1], 3),
                          "paced_fps": a.paced, "agg_fps": round(n * a.frames / wall, 1), "cores_used": round(cpu / wall, 2),
                          "cpu_ms_per_frame": round(cpu * 1000 / (n * a.frames), 3),
                          "misses": sum(r["miss"] for r in rs)}), flush=True)


if __name__ == "__main__":
    main()
