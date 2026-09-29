"""CPU-only proof that the lean, shared arena reproduces the accepted composition bit for bit.

For each accepted identity:
  1. build the arena (arena.build) and attach it read-only (arena.Attached), as the workers do;
  2. compose all 240 frames with chin.corrected_refined using the accepted run's own saved faces
     (faces.npz), generated landmarks (generated_landmarks.npy) and filtered chin deltas
     (chin_delta.npy) - the exact inputs render_stage.py composed with;
  3. require sha256(raw refined frames) == render.json raw_refined_sha256 and
     sha256(faces) == render.json generated_faces_sha256.
With --verbatim IDENT it also runs chin.prepare_refined verbatim (the 2.45 GB path) for that
identity and checks every attribute the timed path reads, plan by plan and mask by mask.
No CUDA, no tracker. Usage:
  python -m chin_multistream.check_prep [--identities a,b] [--verbatim black_woman] [--out JSON]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from chin_multistream import arena, paths  # noqa: E402

READ_ATTRS = ("u", "relative", "lateral", "cb", "source_hull", "radius", "feather", "mask", "lip", "kernel", "region")
READ_PLAN_KEYS = ("face_size", "clip_slice", "overlay_dst_slice", "face_src_slice")


def same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def compose_hash(att, faces, g, delta):
    import chin

    d = att.d
    d["g"][:] = g
    d["chin_delta"][:] = delta
    h = hashlib.sha256()
    t = time.perf_counter()
    for i in range(paths.N_FRAMES):
        h.update(chin.corrected_refined(att.frames[i], d, i, faces[i]).tobytes())
    return h.hexdigest(), (time.perf_counter() - t) * 1000 / paths.N_FRAMES


def verbatim_compare(identity, att, faces, g, delta):
    """chin.prepare_refined exactly as render_stage calls it, compared with the arena views."""
    import cv2
    import torch
    import chin

    src = paths.ACCEPTED_ROOT / identity
    cap = cv2.VideoCapture(str(src / "source.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    d = dict(cache=torch.load(src / "cache.pt", map_location="cpu", weights_only=False), masks=np.load(src / "masks.npz"),
             p=np.load(src / "source_landmarks.npy"), g=np.zeros((240, 478, 2), np.float32))
    chin.prepare_refined(d, frames)
    frames_equal = all(np.array_equal(frames[i], att.frames[i]) for i in range(len(frames)))
    mism = []
    for i in range(paths.N_FRAMES):
        a, b = d["refined_masks"][i], att.d["refined_masks"][i]
        for k in READ_ATTRS:
            if not same(getattr(a, k), getattr(b, k)):
                mism.append((i, "refined." + k))
        for k in READ_PLAN_KEYS:
            if not same(d["plans"][i][k], att.d["plans"][i][k]):
                mism.append((i, "plan." + k))
    boxes_equal = same(d["cache"]["boxes"], att.d["cache"]["boxes"]) and same(d["cache"]["cropboxes"], att.d["cache"]["cropboxes"])
    p_equal = same(d["p"], att.d["p"])
    zeros_equal = same(d["chin_delta"], np.zeros_like(att.d["chin_delta"]))
    d["g"][:] = g
    d["chin_delta"][:] = delta
    h = hashlib.sha256()
    for i in range(paths.N_FRAMES):
        h.update(chin.corrected_refined(frames[i], d, i, faces[i]).tobytes())
    return dict(frames_equal=frames_equal, attribute_mismatches=mism[:20], n_mismatches=len(mism),
                boxes_equal=boxes_equal, p_equal=p_equal, chin_delta_init_equal=zeros_equal,
                verbatim_compose_sha256=h.hexdigest())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", default=",".join(paths.IDENTITIES))
    ap.add_argument("--verbatim", default="", help="identity to also check against verbatim chin.prepare_refined")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    import cv2

    cv2.setNumThreads(2)
    paths.add_import_paths()
    root = Path(f"/dev/shm/chinms_check_{os.getpid()}")
    report = dict(code=paths.code_integrity(), numpy=np.__version__, opencv=cv2.__version__, subjects={})
    ok_all = report["code"]["matches_accepted_render_json"]
    try:
        for ident in [s for s in args.identities.split(",") if s]:
            ref = paths.accepted_render(ident)
            src = paths.ACCEPTED_ROOT / ident
            stats = arena.build(ident, root / ident)
            att = arena.Attached(root / ident)
            faces = np.load(src / "faces.npz")["faces"]
            g = np.load(src / "generated_landmarks.npy")
            delta = np.load(src / "chin_delta.npy")
            fh = hashlib.sha256()
            for f in faces:
                fh.update(f.tobytes())
            raw, ms = compose_hash(att, faces, g, delta)
            row = dict(build=stats, compose_ms_per_frame=ms, raw_refined_sha256=raw,
                       raw_match=raw == ref["raw_refined_sha256"], faces_match=fh.hexdigest() == ref["generated_faces_sha256"])
            if ident == args.verbatim:
                v = verbatim_compare(ident, att, faces, g, delta)
                v["verbatim_match"] = v["verbatim_compose_sha256"] == ref["raw_refined_sha256"]
                row["verbatim"] = v
                ok_all = ok_all and v["verbatim_match"] and v["n_mismatches"] == 0 and v["frames_equal"] and \
                    v["boxes_equal"] and v["p_equal"] and v["chin_delta_init_equal"]
            ok_all = ok_all and row["raw_match"] and row["faces_match"]
            report["subjects"][ident] = row
            print("CHECK", ident, "raw_match", row["raw_match"], "faces_match", row["faces_match"],
                  "verbatim", row.get("verbatim", {}).get("verbatim_match"), flush=True)
            att.close()
            del att, faces
            shutil.rmtree(root / ident, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    report["passes"] = bool(ok_all)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("CHECK PREP", "PASS" if ok_all else "FAIL", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
