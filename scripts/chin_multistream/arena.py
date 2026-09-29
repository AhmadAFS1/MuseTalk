"""Per-identity precompute shared read-only by every stream of that identity.

render_stage.py runs, before timing:
    frames   = every frame of source.mp4 via cv2.VideoCapture.read()
    d        = dict(cache=torch.load(cache.pt), masks=np.load(masks.npz), p=np.load(source_landmarks.npy),
                    g=np.zeros((240,478,2),np.float32))
    chin.prepare_refined(d, frames)
which builds d['source_masks'], d['plans'], d['chin_delta'] (zeros) and d['refined_masks'].
That costs ~2.45 GB of RAM per identity, too much for 6-12 streams on this box.

The timed loop only ever calls chin.curves, chin.corrected_refined (-> RefinedMask.current,
get_image_blending_with_plan, warp_roi). Those read:
    d['refined_masks'][i]  attributes u, relative, lateral, cb, source_hull, radius, feather,
                           mask, lip, kernel, region   (RefinedMask.current)
    d['plans'][i]          clip_slice, overlay_dst_slice, face_src_slice (+ alpha/alpha_u8, which
                           corrected_refined overwrites on its copy before use)
    d['cache']['boxes'][i], d['cache']['cropboxes'][i], d['p'][i], d['g'][i], d['chin_delta'][i]
d['source_masks'] is only read by chin.corrected / chin.standard (offline diagnostics).

"Lean" prep therefore calls the SAME constructors with the SAME arguments as
chin.prepare_source/prepare_refined, frame by frame:
    plan = chin.prepare_image_blending_plan(frame.shape, boxes[i], masks[str(i)], cropboxes[i])
    rm   = chin.RefinedMask(masks[str(i)], cropboxes[i], p[i])
and keeps every attribute except the ones listed in DROP_* (never read on the timed path).
Arrays are written to append-only files in /dev/shm and mapped read-only by the workers
(any accidental write raises). check_prep.py proves the result composes bit-exactly
(accepted faces + landmarks -> render.json raw_refined_sha256) and equals a verbatim
chin.prepare_refined attribute by attribute.
"""
from __future__ import annotations

import mmap
import os
import pickle
import time
from pathlib import Path

import numpy as np

from . import paths

# RefinedMask inherits SourceMask.__init__, which also builds `weight` (float64, full mask) and
# `base`; RefinedMask.current never reads either (it recomputes its own weight from u/relative).
DROP_REFINED_ATTRS = ("weight", "base")
# corrected_refined does plan=d['plans'][i].copy(); plan['alpha_u8']=...; plan['alpha']=...
DROP_PLAN_KEYS = ("alpha", "alpha_u8")
INLINE_BYTES = 4096  # smaller arrays stay in the pickle
ALIGN = 64


class ArrRef:
    """Placeholder for an array stored in prep.bin."""

    __slots__ = ("offset", "shape", "dtype")

    def __init__(self, offset, shape, dtype):
        self.offset, self.shape, self.dtype = int(offset), tuple(shape), str(dtype)

    def __getstate__(self):
        return (self.offset, self.shape, self.dtype)

    def __setstate__(self, state):
        self.offset, self.shape, self.dtype = state


def _load_cache_meta(identity_dir: Path):
    """boxes / cropboxes exactly as torch.load(cache.pt) returns them (np.int32 array, list of lists)."""
    import torch  # only in the owner's short prep phase; see worker.py for why workers avoid torch

    cache = torch.load(identity_dir / "cache.pt", map_location="cpu", weights_only=False)
    boxes, cropboxes = cache["boxes"], cache["cropboxes"]
    del cache
    return boxes, cropboxes


def build(identity: str, arena_dir: Path, boxes=None, cropboxes=None) -> dict:
    """Decode + lean-prepare one identity into arena_dir (frames.u8, prep.bin, meta.pkl)."""
    import cv2

    paths.add_import_paths()
    import chin

    src = paths.ACCEPTED_ROOT / identity
    arena_dir = Path(arena_dir)
    arena_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Frames: identical to render_stage (VideoCapture.read until it fails), streamed to the arena.
    cap = cv2.VideoCapture(str(src / "source.mp4"))
    n = 0
    with open(arena_dir / "frames.u8.tmp", "wb") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != paths.FRAME_SHAPE or frame.dtype != np.uint8:
                raise RuntimeError(f"{identity}: unexpected frame {frame.shape} {frame.dtype}")
            f.write(np.ascontiguousarray(frame).data)
            n += 1
    cap.release()
    if n != paths.N_FRAMES:
        raise RuntimeError(f"{identity}: {n} frames, expected {paths.N_FRAMES}")
    decode_s = time.perf_counter() - t0

    if boxes is None or cropboxes is None:
        boxes, cropboxes = _load_cache_meta(src)
    masks = np.load(src / "masks.npz")
    p = np.load(src / "source_landmarks.npy")
    t1 = time.perf_counter()
    plans, refined = [], []
    offset = 0
    with open(arena_dir / "prep.bin.tmp", "wb") as f:
        def store(arr):
            nonlocal offset
            arr_c = np.ascontiguousarray(arr)
            pad = (-offset) % ALIGN
            if pad:
                f.write(b"\0" * pad)
                offset += pad
            ref = ArrRef(offset, arr_c.shape, arr_c.dtype.str)
            f.write(arr_c.data)
            offset += arr_c.nbytes
            return ref

        for i in range(paths.N_FRAMES):
            # Same calls, same arguments, same order as chin.prepare_source / prepare_refined.
            plan = chin.prepare_image_blending_plan(paths.FRAME_SHAPE, boxes[i], masks[str(i)], cropboxes[i])
            if plan is None:
                raise RuntimeError(f"{identity}: frame {i} has no blending plan")
            plans.append({k: v for k, v in plan.items() if k not in DROP_PLAN_KEYS})
            rm = chin.RefinedMask(masks[str(i)], cropboxes[i], p[i])
            attrs = {}
            for k, v in vars(rm).items():
                if k in DROP_REFINED_ATTRS:
                    continue
                if isinstance(v, np.ndarray) and v.nbytes >= INLINE_BYTES:
                    attrs[k] = store(v)
                else:
                    attrs[k] = v
            refined.append(attrs)
            del rm, plan
    prep_s = time.perf_counter() - t1
    meta = dict(
        identity=identity, n=paths.N_FRAMES, frame_shape=paths.FRAME_SHAPE, boxes=boxes, cropboxes=cropboxes,
        p=p, plans=plans, refined=refined, grid_len=len(chin.GRID), prep_bin_bytes=offset,
        drop_refined_attrs=DROP_REFINED_ATTRS, drop_plan_keys=DROP_PLAN_KEYS,
        decode_s=decode_s, prep_s=prep_s,
        sources={name: paths.sha256_file(src / name) for name in
                 ("source.mp4", "masks.npz", "source_landmarks.npy", "cache.pt")},
    )
    with open(arena_dir / "meta.pkl.tmp", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    for name in ("frames.u8", "prep.bin", "meta.pkl"):
        os.replace(arena_dir / f"{name}.tmp", arena_dir / name)
    return dict(identity=identity, decode_s=decode_s, prep_s=prep_s,
                frames_bytes=paths.N_FRAMES * int(np.prod(paths.FRAME_SHAPE)), prep_bytes=offset)


class Attached:
    """Read-only views of an arena, laid out as render_stage's `frames` and `d`."""

    def __init__(self, arena_dir: Path):
        paths.add_import_paths()
        import chin

        arena_dir = Path(arena_dir)
        with open(arena_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.meta = meta
        self._files = []
        self._maps = []

        def mapped(name):
            fh = open(arena_dir / name, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._files.append(fh)
            self._maps.append(mm)
            return mm

        fmm = mapped("frames.u8")
        self.frames = np.frombuffer(fmm, dtype=np.uint8).reshape((meta["n"],) + tuple(meta["frame_shape"]))
        pmm = mapped("prep.bin") if meta["prep_bin_bytes"] else None

        def view(ref):
            dt = np.dtype(ref.dtype)
            count = int(np.prod(ref.shape)) if ref.shape else 1
            return np.frombuffer(pmm, dtype=dt, count=count, offset=ref.offset).reshape(ref.shape)

        refined = []
        for attrs in meta["refined"]:
            obj = object.__new__(chin.RefinedMask)
            for k, v in attrs.items():
                setattr(obj, k, view(v) if isinstance(v, ArrRef) else v)
            refined.append(obj)
        self.d = dict(
            cache=dict(boxes=meta["boxes"], cropboxes=meta["cropboxes"]),
            p=meta["p"],
            g=np.zeros((meta["n"], 478, 2), np.float32),
            plans=meta["plans"],
            refined_masks=refined,
            # chin.prepare_source: d['chin_delta']=np.zeros((len(frames),len(GRID)))
            chin_delta=np.zeros((meta["n"], meta["grid_len"])),
        )

    def close(self):
        self.d = None
        self.frames = None
        for mm in self._maps:
            try:
                mm.close()
            except BufferError:
                pass  # views still alive; the process is exiting anyway
        for fh in self._files:
            fh.close()
