"""One ordered worker process per stream.

Per 8-frame batch it runs render_stage.py's per-frame code unchanged:
    g,_ = tracker.track(frames[i], face, boxes[i])            # backend.Tracker (FaceMesh subprocess)
    source,target = chin.curves(p[i], g); delta = source-target; queue.append((i,face,g,delta))
    if len(queue)==2: emit(queue.pop(0), delta)
and emit() = render_stage.emit(): d['g'][i]=g; 3-tap .25/.5/.25 filter on the jaw delta, clipped
to [0,.18]; chin.corrected_refined(frames[i], d, i, face). At the end of each 240-frame clip the last
queued frame is emitted with its own delta (render_stage's `if queue: emit(queue[0],queue[0][3])`),
the filter state is cleared and the tracker is reset (a fresh FaceMesh, as render_stage starts with),
so every loop of a stream is an independent replay of the accepted clip and must hash identically.

Faces arrive through a shared-memory ring (slots of 8 faces); each slot is copied out and released
(a credit back to the GPU process) before the batch is processed. The worker hashes, in order, the
generated faces and the raw refined frames of every clip (sha256 over tobytes, like render_stage's
hash_frames) and optionally encodes the first clip (crf<=12) for video review.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import queue as queue_mod
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chin_multistream import paths  # noqa: E402
from chin_multistream.telemetry import proc_cpu_s, rss_mib  # noqa: E402


def import_tracker():
    """backend.Tracker, imported from the unmodified accepted backend.py without loading torch.

    backend.py imports torch at module level only for setup(); Tracker uses shared_memory, numpy,
    cv2 and subprocess. A placeholder module stands in for torch during the import (saves ~180 MB
    of private RSS per worker) and is removed again right after.
    """
    paths.add_import_paths()
    if "backend" in sys.modules:
        return sys.modules["backend"].Tracker
    had_torch = "torch" in sys.modules
    if not had_torch:
        stub = types.ModuleType("torch")
        stub.__file__ = "<chin_multistream placeholder: backend.Tracker does not use torch>"
        sys.modules["torch"] = stub
    try:
        import backend
    finally:
        if not had_torch:
            sys.modules.pop("torch", None)
    if Path(backend.__file__).resolve() != (paths.WORKFLOW / "backend.py").resolve():
        raise RuntimeError(f"backend imported from {backend.__file__}")
    return backend.Tracker


class Encoder:
    """ffmpeg pipe fed from a bounded queue by a writer thread (keeps compose off the encoder's pace)."""

    def __init__(self, path, width, height, crf, audio=None, preset="medium", threads=2):
        cmd = ["ffmpeg", "-v", "error", "-xerror", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", "24", "-i", "pipe:0"]
        if audio:
            cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-t", "10"]
        cmd += ["-c:v", "libx264", "-threads", str(threads), "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path)]
        self.path = path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.q = queue_mod.Queue(maxsize=48)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.proc.stdin.write(item)
        self.proc.stdin.close()

    def write(self, frame):
        self.q.put(frame.tobytes())

    def close(self):
        self.q.put(None)
        self.t.join()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}) for {self.path}")
        return str(self.path)


class Hasher:
    """In-order sha256 of faces and raw refined frames on a helper thread.

    hashlib releases the GIL for large buffers, so hashing overlaps the FaceMesh IPC wait instead of
    adding ~0.85 ms per frame to the worker's critical path. Same bytes, same order as
    render_stage.hash_frames (sha256 over each C-contiguous frame's buffer == its tobytes()).
    """

    def __init__(self):
        self.q = queue_mod.Queue(maxsize=24)
        self.done = []
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        fh, rh = hashlib.sha256(), hashlib.sha256()
        while True:
            item = self.q.get()
            if item is None:
                break
            kind, obj = item
            if kind == "f":
                fh.update(obj)
            elif kind == "r":
                rh.update(obj)
            else:  # end of clip
                self.done.append((obj, rh.hexdigest(), fh.hexdigest()))
                fh, rh = hashlib.sha256(), hashlib.sha256()

    def face(self, arr):
        if not arr.flags.c_contiguous:
            arr = arr.tobytes()
        self.q.put(("f", arr))

    def refined(self, arr):
        if not arr.flags.c_contiguous:
            arr = arr.tobytes()
        self.q.put(("r", arr))

    def end_clip(self, loop):
        self.q.put(("e", loop))

    def finish(self):
        self.q.put(None)
        self.t.join()
        return self.done


def main(cfg: dict, conn) -> None:
    try:
        _run(cfg, conn)
    except BaseException:
        try:
            conn.send(("error", cfg["stream"], traceback.format_exc()))
        except Exception:
            pass
        raise


def _run(cfg: dict, conn) -> None:
    import numpy as np
    import cv2

    cv2.setNumThreads(int(cfg["cv2_threads"]))
    paths.add_import_paths()
    import chin
    from chin_multistream import arena

    Tracker = import_tracker()
    from multiprocessing import shared_memory

    stream = cfg["stream"]
    ring = shared_memory.SharedMemory(name=cfg["ring_name"])
    ring_arr = np.ndarray((cfg["ring_slots"], paths.BATCH) + paths.FACE_SHAPE, np.uint8, buffer=ring.buf)
    conn.send(("hello", stream, dict(pid=os.getpid(), torch_loaded="torch" in sys.modules, numpy=np.__version__,
                                     opencv=cv2.__version__, chin=chin.__file__, blending=sys.modules["musetalk.utils.blending"].__file__,
                                     blas_env={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})))
    att = None
    tracker = None
    acc = None
    try:
        while True:
            msg = conn.recv()
            kind = msg[0]
            if kind == "prep":
                _, arena_dir, boxes, cropboxes = msg
                t = time.perf_counter()
                stats = arena.build(cfg["identity"], Path(arena_dir), boxes, cropboxes)
                stats["wall_s"] = time.perf_counter() - t
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                conn.send(("prepped", stream, stats))
            elif kind == "attach":
                _, arena_dir = msg
                t = time.perf_counter()
                att = arena.Attached(Path(arena_dir))
                if att.meta["identity"] != cfg["identity"]:
                    raise RuntimeError("arena identity mismatch")
                if cfg.get("compare_accepted"):
                    src = paths.ACCEPTED_ROOT / cfg["identity"]
                    acc = dict(faces=np.load(src / "faces.npz")["faces"], g=np.load(src / "generated_landmarks.npy"),
                               chin_delta=np.load(src / "chin_delta.npy"))
                tracker = Tracker(paths.WORKSPACE, cfg["tracker_log"])
                conn.send(("attached", stream, dict(attach_s=time.perf_counter() - t, facemesh_pid=tracker.proc.pid,
                                                    rss_mib=rss_mib())))
            elif kind == "run":
                stats = _repeat(cfg, conn, msg[1], msg[2], att, tracker, ring_arr, acc, chin, np)
                conn.send(("repdone", stream, msg[1], stats))
            elif kind == "stop":
                break
            else:
                raise ValueError(kind)
    finally:
        if tracker is not None:
            tracker.close()
        del ring_arr
        ring.close()
        if att is not None:
            att.close()
    conn.send(("bye", stream))


def _repeat(cfg, conn, rep, rcfg, att, tracker, ring_arr, acc, chin, np):
    frames, d = att.frames, att.d
    boxes = d["cache"]["boxes"]
    loops = int(rcfg["loops"])
    n = paths.N_FRAMES
    first = rep == 0
    encode = bool(rcfg.get("encode")) and first
    save_arrays = bool(rcfg.get("save_arrays")) and first
    compare = acc is not None and first
    out_dir = Path(rcfg["out_dir"]) if rcfg.get("out_dir") else None
    tag = f"stream{cfg['stream']:02d}_{cfg['identity']}"
    enc_refined = enc_faces = None
    if encode:
        crf = int(rcfg.get("crf", 12))
        audio = paths.ACCEPTED_ROOT / cfg["identity"] / "speech.wav"
        enc_refined = Encoder(out_dir / f"{tag}_refined.mp4", 512, 896, crf, audio=audio)
        enc_faces = Encoder(out_dir / f"{tag}_faces.mp4", 256, 256, crf)
    T = dict(tracking_ipc_ms=0., facemesh_ms=0., compose_ms=0., filter_ms=0., hash_ms=0., copy_ms=0.,
             reset_ms=0., idle_ms=0., encode_queue_ms=0., compare_ms=0., batches=0, frames=0)
    cmp = dict(face_sse=0., face_max=0, face_identical=0, refined_sse=0., refined_max=0, refined_identical=0,
               g_max_px=0., g_sum_px=0., delta_max=0.) if compare else None
    st = dict(previous=None)
    q = []
    clip_hashes = []
    raw_faces = []
    hasher = Hasher()
    perf = time.perf_counter

    def emit(record, next_delta):
        i, face, g, delta = record
        t = perf()
        d["g"][i] = g
        old = delta if st["previous"] is None else st["previous"]
        d["chin_delta"][i] = np.clip(.25 * old + .5 * delta + .25 * next_delta, 0, .18)
        t1 = perf()
        out = chin.corrected_refined(frames[i], d, i, face)
        t2 = perf()
        st["previous"] = delta
        hasher.refined(out)
        t3 = perf()
        T["filter_ms"] += (t1 - t) * 1000
        T["compose_ms"] += (t2 - t1) * 1000
        T["hash_ms"] += (t3 - t2) * 1000
        T["frames"] += 1
        st["t_last"] = t3
        if st["loop"] == 0:
            if save_arrays:
                raw_faces.append(face.copy())  # raw uint8 generated face (the ring slot is reused)
            if enc_refined is not None:
                enc_refined.write(out)
                enc_faces.write(face)
                T["encode_queue_ms"] += (perf() - t3) * 1000
            if cmp is not None:
                t4 = perf()
                da = dict(d)
                da["g"], da["chin_delta"] = acc["g"], acc["chin_delta"]
                ref = chin.corrected_refined(frames[i], da, i, acc["faces"][i])
                diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
                cmp["refined_sse"] += float((diff.astype(np.float64) ** 2).sum())
                cmp["refined_max"] = max(cmp["refined_max"], int(diff.max()))
                cmp["refined_identical"] += int(not diff.any())
                fd = np.abs(face.astype(np.int16) - acc["faces"][i].astype(np.int16))
                cmp["face_sse"] += float((fd.astype(np.float64) ** 2).sum())
                cmp["face_max"] = max(cmp["face_max"], int(fd.max()))
                cmp["face_identical"] += int(not fd.any())
                gd = np.linalg.norm(g.astype(np.float64) - acc["g"][i], axis=1)
                cmp["g_max_px"] = max(cmp["g_max_px"], float(gd.max()))
                cmp["g_sum_px"] += float(gd.mean())
                cmp["delta_max"] = max(cmp["delta_max"], float(np.abs(d["chin_delta"][i] - acc["chin_delta"][i]).max()))
                T["compare_ms"] += (perf() - t4) * 1000

    t = perf()
    tracker.reset()  # render_stage: tracker=Tracker(...); tracker.reset() before the timed loop
    arm_reset_ms = (perf() - t) * 1000
    cpu0, fm0 = proc_cpu_s(), proc_cpu_s(tracker.proc.pid)
    conn.send(("armed", cfg["stream"], rep))
    st["loop"] = 0
    expect = (0, 0)
    t_first = None
    while True:
        t = perf()
        msg = conn.recv()
        now = perf()
        if t_first is not None:
            T["idle_ms"] += (now - t) * 1000
        if msg[0] != "b":
            raise RuntimeError(f"unexpected message during repeat: {msg[0]}")
        _, slot, mrep, loop, base = msg
        if (mrep, loop, base) != (rep,) + expect:
            raise RuntimeError(f"out-of-order batch {(mrep, loop, base)} expected {(rep,) + expect}")
        if t_first is None:
            t_first = now
        faces = ring_arr[slot].copy()
        conn.send(("r", slot))
        T["copy_ms"] += (perf() - now) * 1000
        T["batches"] += 1
        st["loop"] = loop
        for j, face in enumerate(faces):
            i = base + j
            t = perf()
            hasher.face(face)
            t0 = perf()
            g, fm_s = tracker.track(frames[i], face, boxes[i])
            t1 = perf()
            T["hash_ms"] += (t0 - t) * 1000
            T["tracking_ipc_ms"] += (t1 - t0) * 1000
            T["facemesh_ms"] += fm_s * 1000
            source, target = chin.curves(d["p"][i], g)
            delta = source - target
            q.append((i, face, g, delta))
            T["filter_ms"] += (perf() - t1) * 1000
            if len(q) == 2:
                emit(q.pop(0), delta)
        if base + paths.BATCH == n:
            if q:
                emit(q[0], q[0][3])
            q.clear()
            hasher.end_clip(loop)
            clip_hashes.append(dict(loop=loop, t_done=st["t_last"]))
            if loop == 0:
                if enc_refined is not None:
                    enc_refined_path, enc_faces_path = enc_refined.close(), enc_faces.close()
                    enc_refined = enc_faces = None
                    clip_hashes[-1]["videos"] = [enc_refined_path, enc_faces_path]
                if save_arrays:
                    np.savez_compressed(out_dir / f"{tag}_arrays.npz", generated_landmarks=d["g"], chin_delta=d["chin_delta"])
                    # Raw generated faces (render_stage's faces.npz layout) for exact E1 quality comparisons;
                    # encoded faces carry codec noise larger than the FaceMesh gates.
                    np.savez_compressed(out_dir / f"{tag}_faces.npz", faces=np.asarray(raw_faces))
                    raw_faces.clear()
            st["previous"] = None
            if loop + 1 == loops:
                break
            t = perf()
            tracker.reset()
            T["reset_ms"] += (perf() - t) * 1000
            expect = (loop + 1, 0)
        else:
            expect = (loop, base + paths.BATCH)
    t_hash_wait = perf()
    for (loop, raw, faces_sha), c in zip(hasher.finish(), clip_hashes):
        if loop != c["loop"]:
            raise RuntimeError("hasher clip order")
        c["raw_refined_sha256"], c["generated_faces_sha256"] = raw, faces_sha
    T["hash_drain_after_last_ms"] = (perf() - t_hash_wait) * 1000
    cpu1, fm1 = proc_cpu_s(), proc_cpu_s(tracker.proc.pid)
    if cmp is not None:
        m = n
        cmp_out = dict(
            frames=m,
            face_psnr_db=float(10 * np.log10(255 ** 2 / max(cmp["face_sse"] / (m * 256 * 256 * 3), 1e-12))),
            face_max_abs=cmp["face_max"], face_identical_frames=cmp["face_identical"],
            refined_psnr_db=float(10 * np.log10(255 ** 2 / max(cmp["refined_sse"] / (m * 896 * 512 * 3), 1e-12))),
            refined_max_abs=cmp["refined_max"], refined_identical_frames=cmp["refined_identical"],
            generated_landmarks_max_px=cmp["g_max_px"], generated_landmarks_mean_px=cmp["g_sum_px"] / m,
            chin_delta_max_abs=cmp["delta_max"])
    else:
        cmp_out = None
    return dict(timing_ms=T, arm_reset_ms=arm_reset_ms, t_first_recv=t_first, t_last_compose=st.get("t_last"),
                worker_cpu_s=cpu1 - cpu0, facemesh_cpu_s=fm1 - fm0, clips=clip_hashes, compare_accepted=cmp_out,
                rss_mib=rss_mib(), facemesh_rss_mib=rss_mib(tracker.proc.pid))
