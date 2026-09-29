"""Labelled lineage video: the accepted pre-change render next to every optimization round, per identity.

Fairness: every column is rebuilt from its RAW pre-encode frames, with no intermediate mp4, using the quality
tool's reconstruction (scripts/quality_ab_metrics.py load_arm, i.e. the unchanged chin.py
corrected_refined compose):
  - BEFORE (accepted render): faces.npz + generated_landmarks.npy + chin_delta.npy, verified bit-exact
    against the render's recorded raw_refined_sha256;
  - each round: the faces, generated landmarks and chin delta its chin_multistream_render.py workers actually
    used (<capture>/<tag>_faces.npz + <tag>_arrays.npz), verified bit-exact against the raw_refined_sha256
    the harness worker recorded while rendering.
All columns are composed into one canvas and encoded once, so none has a compression advantage. (The stored
accepted refined_raw.mp4 is ~1.9 Mbit/s; the round captures are crf 12, ~5.3 Mbit/s. Comparing those files
directly would favour the rounds.)

Layout per column:
  - labels: name, measured fps, what changed;
  - full frame (scaled to the column width);
  - mouth zoom (nearest neighbour; the actual factor is printed). The crop follows BEFORE's lips, and every column
    uses the same crop;
  - face + neck region of the composited output: BEFORE's pixels, then per round 0 where identical and
    40 + 8 x |output - BEFORE output| elsewhere, so a 1 LSB change survives the encode. Computed from the raw
    pre-encode frames, so it shows exactly what reaches the viewer;
  - quality-tool metrics vs BEFORE with gate verdicts, colour-coded (green pass, orange fail).
Differences below ~1 LSB in the full-frame and zoom rows are H.264 noise. The diff row and the metrics are the
pixel-level authority.

Also writes <identity>_stills.png, lossless from the raw canvas: the widest-mouth frame and the frame where the last
column's output differs most. And lineage_report.json: bit-exactness, 1-based still frame numbers, the exact command
line and the script sha256.

usage: video_lineage.py [--ids a,b] [--out-name lineage_all_rounds] [--col-width 400] [--crf 12]
                        [--arm 'name|fps|what|capture_dir|qlabel' ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import quality_ab_metrics as qab  # noqa: E402  (also puts chin.py on sys.path)

import cv2  # noqa: E402

chin = qab.chin
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
IMPL = ROOT / "docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream"
R400 = ROOT / "docs/fps_comparisons/4070s_400fps_20260928/chin_multistream"
QRUNS = ROOT / "docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/runs"
# extra quality-run directories searched first (os.pathsep-separated), e.g. a reproduction's own quality_runs
QRUNS_EXTRA = [Path(x) for x in __import__("os").environ.get("MUSETALK_QUALITY_RUNS", "").split(__import__("os").pathsep) if x]
# BEFORE's throughput: the accepted single-stream render loop (render.json), and the same pre-change backends in the
# six-stream harness (T_baseline_pair.json: 252.0 fps, every clip bit-identical to the accepted renders)
BEFORE_HARNESS_FPS = 252.0   # T_baseline_pair (252.6 / 251.4), re-measured back to back with r5 (400.7); earlier Ta_baseline_n6 251.8
BEFORE_LABEL = ("BEFORE (previous working pipeline)", f"1 stream 148-171 fps | same 6-stream harness {BEFORE_HARNESS_FPS} fps",
                "shipping TensorRT FP16 bs8 UNet + compiled TAESD, same chin recipe (pre-change backends)")
DEFAULT_ARMS = [
    ("r2", "350.2 fps aggregate (6 streams)", "stagewise FP16 UNet + source-prefix cache + INT8 down3/mid + TRT TAESD",
     IMPL / "V_srcmix_taesdtrt", "srcmix_taesdtrt"),
    ("r3", "415.6 fps aggregate (6 streams)", "r2 + broad INT8 PTQ on 6 UNet blocks (no recovery)",
     IMPL / "V_srcv1_taesdtrt", "srcv1_taesdtrt"),
    ("r4", "414.9 fps aggregate (6 streams)", "r2 + layer-selective INT8 (146 layers; down0/up3/audio K/V FP16)",
     R400 / "V_srcblkA8", "srcblkA8"),
    ("r5 NEW", "400 fps sustained aggregate (6 streams; 404.0 to 399.96 over 5 x 64 s)", "r2 + INT8 on 117 layers chosen by error per MAC (gmac_0.50)",
     R400 / "V_srcg50", "srcg50"),
]
# repo UNet latent gate (scripts/validate_unet_backend.py; mae_max <= 0.01 and max_abs <= 0.5), main / holdout corpus
UNET_GATE = {"srcmix_taesdtrt": ((0.0025, 0.389), (0.0021, 0.242)),   # unet_fp16/gunet_srcmix_*.json
             "srcv1_taesdtrt": ((0.0382, 2.389), (0.0370, 2.369)),    # unet_fp16/gunet_v1_*.json (srcv1 = v1 + exact prefix cache)
             "srcblkA8": ((0.0071, 1.284), (0.0061, 1.803)),          # 4070s_400fps_20260928/gate/gunet_srcblkA8_*.json
             "srcg50": ((0.0044, 0.777), (0.0039, 1.260))}            # 4070s_400fps_20260928/gate/gunet_srcg50_*.json
CW, CH = 400, 700          # column width, full-frame height (512x896 scaled by 0.78125); --col-width overrides
TOP, GAP, LINE = 104, 22, 16
GREEN, AMBER, ORANGE, CYAN, GREY = "0x66ff66", "0xffe040", "0xff8030", "cyan", "0xdddddd"
LEGEND = ["gates (quality tool, vs BEFORE):", "lip corr >= 0.97, lag 0, aperture delta <= 0.5 px",
          "flicker ratios <= 1.05, chin err change <= +0.05 px", "landmarks: repo gate <= 0.05 / 0.15 px (FaceMesh floor);",
          "  proposed bar <= 0.10 / 0.35 px (not yet approved)", "UNet latent gate: mae <= 0.01, max_abs <= 0.5",
          "colours: green pass | amber only the proposed bar passes", "  | orange fail",
          "+-1 LSB random noise on faces: mouth PSNR ~51.7 dB,", "  landmarks 0.045-0.088 / 0.13-0.33 px"]


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def wrap_words(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


def gate(d, name):
    return next((g for g in d["gates"] if g["gate"] == name), None)


def metrics_lines(ident: str, qlabel: str, out_a, out_b, fa, fb) -> list[tuple[str, str]]:
    """(text, colour) lines for one round column."""
    od = np.abs(out_a.astype(np.int16) - out_b.astype(np.int16))
    gd = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
    lines = [(f"output face-region diff mean {od.mean():.3f} / max {int(od.max())} LSB", CYAN),
             (f"(generated face diff mean {gd.mean():.3f} / max {int(gd.max())})", CYAN)]
    f = next((d / f"{ident}__{qlabel}.json" for d in QRUNS_EXTRA + [QRUNS] if (d / f"{ident}__{qlabel}.json").exists()), QRUNS / "-")
    if not f.exists():
        return lines + [("quality tool: not run", ORANGE)]
    d = json.loads(f.read_text())
    ps = d["overall"]["psnr"]
    v = {k: gate(d, k)["value"] for k in ("lip.aperture_corr", "lip.mean_abs_delta_px", "flicker.mouth_ratio",
                                          "flicker.jaw_ratio", "chin.landmark_dev_mean_px", "chin.landmark_dev_p99_px")}
    chin_d = d["chin"]["B"]["target_chin_abs_error_px"]["mean"] - d["chin"]["A"]["target_chin_abs_error_px"]["mean"]
    lines += [(f"lip-sync corr {v['lip.aperture_corr']:.4f}, aperture delta {v['lip.mean_abs_delta_px']:.2f} px", CYAN),
              (f"flicker mouth {v['flicker.mouth_ratio']:.3f} / jaw {v['flicker.jaw_ratio']:.3f} (1.000 = same)", CYAN),
              (f"jaw+lip landmarks {v['chin.landmark_dev_mean_px']:.4f} / p99 {v['chin.landmark_dev_p99_px']:.3f} px", CYAN),
              (f"mouth PSNR mean {ps['mouth_roi']['mean_frame_db_capped']:.1f} / worst frame {ps['mouth_roi']['worst_frame_db']:.1f} dB", CYAN),
              (f"chin-target err change {chin_d:+.3f} px", CYAN)]
    strict = v["chin.landmark_dev_mean_px"] <= 0.05 and v["chin.landmark_dev_p99_px"] <= 0.15
    calib = v["chin.landmark_dev_mean_px"] <= 0.10 and v["chin.landmark_dev_p99_px"] <= 0.35
    others = [g["gate"] for g in d["gates"] if g["result"] == "fail" and not g["gate"].startswith("chin.landmark_dev")]
    lines += [(f"landmark repo gate: {'PASS' if strict else 'FAIL'} | proposed bar: {'PASS' if calib else 'FAIL'}",
               GREEN if strict else (AMBER if calib else ORANGE)),
              ("all other quality-tool gates: PASS" if not others else "FAIL: " + ", ".join(others), GREEN if not others else ORANGE)]
    ug = UNET_GATE.get(qlabel)
    if ug:
        ok = all(mae <= 0.01 and mx <= 0.5 for mae, mx in ug)
        lines.append((f"UNet latent gate {'PASS' if ok else 'FAIL'}: {ug[0][0]:.4f}/{ug[0][1]:.2f} main, {ug[1][0]:.4f}/{ug[1][1]:.2f} holdout",
                      GREEN if ok else ORANGE))
    return lines


def diff_vis(a, b):
    """|a - b| per channel as 0 if identical, else 40 + 8 x LSB (capped): even a 1 LSB change stays visible after encoding."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.where(d > 0, np.minimum(255, 40 + 8 * d), 0).astype(np.uint8)


def fit_diff(dv):
    """Resize the diff map to the cell. When downscaling, max-pool first so every changed source pixel stays lit."""
    s_ = dv.shape[0]
    if s_ > CW:
        k = int(np.ceil(s_ / CW)) + 1
        dv = cv2.dilate(dv, np.ones((k, k), np.uint8))
    return cv2.resize(dv, (CW, CW), interpolation=cv2.INTER_NEAREST)


def mouth_boxes(g_accepted):
    """Per-frame mouth crops of one fixed size that follow BEFORE's lips (centre smoothed over +-12 frames).

    The size covers the widest/tallest single-frame lip extent with a margin, so no frame clips the lips. The same
    boxes are used for every column, so columns stay pixel-aligned.
    """
    g = g_accepted[:, list(chin.LIPS)]
    x0, y0 = np.nanmin(g[..., 0], axis=1), np.nanmin(g[..., 1], axis=1)
    x1, y1 = np.nanmax(g[..., 0], axis=1), np.nanmax(g[..., 1], axis=1)
    k = np.hanning(27)[1:-1]
    k /= k.sum()                                      # +-12 frames: slow, smooth viewport motion
    cx = np.convolve(np.pad((x0 + x1) / 2, 12, mode="edge"), k, mode="valid")
    cy = np.convolve(np.pad((y0 + y1) / 2, 12, mode="edge"), k, mode="valid")
    # size: every frame's lips inside the (smoothed) crop with >= 6 px margin
    half_w = float(np.max(np.maximum(cx - x0, x1 - cx))) + 6
    half_h = float(np.max(np.maximum(cy - y0, y1 - cy))) + 6
    w = max(2 * half_w, float(np.max(x1 - x0)) * 1.5)
    h = max(2 * half_h, w * 0.62)
    w = max(w, h / 0.62)
    w, h = int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2)
    boxes = [(int(max(0, min(512 - w, round(a - w / 2)))), int(max(0, min(896 - h, round(b - h / 2)))), w, h)
             for a, b in zip(cx, cy)]
    return boxes


def face_box(g_accepted):
    """Fixed square crop around every frame's face landmarks plus the neck below the chin, where the chin warp acts."""
    x0, y0 = np.nanmin(g_accepted[..., 0]), np.nanmin(g_accepted[..., 1])
    x1, y1 = np.nanmax(g_accepted[..., 0]), np.nanmax(g_accepted[..., 1])
    y1 = y1 + 0.22 * (y1 - y0)
    s = int(np.ceil(max(x1 - x0, y1 - y0) * 1.08 / 2) * 2)
    s = min(s, 512)
    X = int(max(0, min(512 - s, round((x0 + x1) / 2 - s / 2))))
    Y = int(max(0, min(896 - s, round((y0 + y1) / 2 - s / 2))))
    return X, Y, s


def load_column(ident, arm, mboxes, zh, fbox):
    qab.load_arm(ident, arm, "raw")
    fx, fy, s = fbox
    full = np.stack([cv2.resize(f, (CW, CH), interpolation=cv2.INTER_AREA) for f in arm.frames])
    zoom = np.stack([cv2.resize(f[Y:Y + h, X:X + w], (CW, zh), interpolation=cv2.INTER_NEAREST)
                     for f, (X, Y, w, h) in zip(arm.frames, mboxes)])
    face = np.stack([np.ascontiguousarray(f[fy:fy + s, fx:fx + s]) for f in arm.frames])
    info = dict(arm.info)
    arm.frames = None
    return full, zoom, face, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", default=",".join(qab.DIV_IDS))
    ap.add_argument("--out-name", default="lineage_all_rounds")
    ap.add_argument("--arm", action="append", default=[], help="'name|fps|what|capture_dir|qlabel' (replaces the defaults)")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--col-width", type=int, default=400, help="column width; 512 = native full-frame resolution")
    a = ap.parse_args()
    global CW, CH
    CW, CH = a.col_width, int(round(896 * a.col_width / 512)) // 2 * 2
    arms = [tuple(s.split("|")) for s in a.arm] or DEFAULT_ARMS
    out = ROOT / "experiments/video_validation" / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    import hashlib
    report = {"command": sys.argv, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "crf": a.crf, "col_width": CW, "before": list(BEFORE_LABEL),
              "arms": [dict(name=n, fps=f, what=w, capture=str(c), quality_label=q) for n, f, w, c, q in arms], "identities": {}}
    for who in a.ids.split(","):
        ident = qab.Identity.from_dir(qab.DIV / who)
        acc = qab.Arm.parse(f"dir={qab.DIV / who}", "BEFORE")
        g_acc = np.load(qab.DIV / who / "generated_landmarks.npy")
        mboxes, fbox = mouth_boxes(g_acc), face_box(g_acc)
        mbox = mboxes[0]
        zh = int(round(CW * mbox[3] / mbox[2])) // 2 * 2
        cols = [load_column(ident, acc, mboxes, zh, fbox)]
        faces = [np.load(qab.DIV / who / "faces.npz")["faces"]]
        rep = {"BEFORE": {k: cols[0][3].get(k) for k in ("raw_matches_render_json", "frames_sha256", "protected_lip")},
               "mouth_box_size_wh": list(mbox[2:]), "mouth_box_follows": "BEFORE lip landmarks, centre smoothed +-12 frames (Hann)",
               "face_box_xys": list(fbox)}
        with tempfile.TemporaryDirectory() as td:
            for name, fps, what, cap, qlabel in arms:
                cap = Path(cap)
                tag = next(cap.glob(f"stream??_{who}_faces.npz")).name[: -len("_faces.npz")]
                arr = np.load(cap / f"{tag}_arrays.npz")
                gp, dp = Path(td) / f"{tag}_g.npy", Path(td) / f"{tag}_delta.npy"
                np.save(gp, arr["generated_landmarks"])
                np.save(dp, arr["chin_delta"])
                arm = qab.Arm(label=name, faces=cap / f"{tag}_faces.npz", g=gp, chin_delta=dp)
                col = load_column(ident, arm, mboxes, zh, fbox)
                info = col[3]
                cols.append(col)
                faces.append(np.load(cap / f"{tag}_faces.npz")["faces"])
                # bit-exactness: the rebuilt raw frames vs the raw_refined_sha256 the harness worker recorded
                harness_sha = None
                cap_json = cap.parent / f"{cap.name}.json"
                if cap_json.exists():
                    widx = str(int(tag[len("stream"):len("stream") + 2]))
                    for r_ in json.loads(cap_json.read_text()).get("repeats", []):
                        clips = r_.get("per_worker", {}).get(widx, {}).get("clips", [])
                        if clips:
                            harness_sha = clips[0].get("raw_refined_sha256")
                            break
                rep[name] = {"bit_exact_vs_harness_render": harness_sha is not None and harness_sha == info.get("frames_sha256"),
                             "harness_raw_refined_sha256": harness_sha, "frames_sha256": info.get("frames_sha256"),
                             "chin_delta_recomputed_equals_saved": info.get("chin_delta_recomputed_equals_saved"),
                             "protected_lip": info.get("protected_lip"), "capture": str(cap), "tag": tag}
        T, n = len(faces[0]), len(cols)
        mlines = [metrics_lines(who, arms[k - 1][4], cols[0][2], cols[k][2], faces[0], faces[k]) for k in range(1, n)]
        BOT = 10 + LINE * max(len(LEGEND), max(len(m) for m in mlines))
        W = CW * n
        y_zoom = TOP + CH + GAP
        y_face = y_zoom + zh + GAP
        H = y_face + CW + BOT
        ap_px = qab.aperture_px(g_acc)
        widest = int(np.nanargmax(ap_px))
        worst = int(np.argmax([np.abs(cols[-1][2][i].astype(np.int16) - cols[0][2][i]).mean() for i in range(T)]))
        if worst == widest:
            worst = (worst + T // 2) % T
        title = f"{who}: same audio and source frames, full 100% chin recipe. Every column rebuilt bit-exactly from raw pre-encode frames, encoded once."
        dt = [f"drawtext=fontfile={FONT}:expansion=none:text='{esc(title)}':x=8:y=6:fontsize=14:fontcolor=white"]
        def with_speedup(fps):
            try:
                v = float(fps.split()[0])
            except ValueError:
                return fps
            return f"{fps}, {v / BEFORE_HARNESS_FPS:.2f}x BEFORE"
        labels = [BEFORE_LABEL] + [(nm, with_speedup(fps), what) for nm, fps, what, _, _ in arms]
        for k, (nm, fps, what) in enumerate(labels):
            x = k * CW + 8
            color = "yellow" if "NEW" in nm else "white"
            dt.append(f"drawtext=fontfile={FONT}:expansion=none:text='{esc(nm)}':x={x}:y=30:fontsize=17:fontcolor={color}")
            dt.append(f"drawtext=fontfile={FONT}:expansion=none:text='{esc(fps)}':x={x}:y=52:fontsize=12:fontcolor={color}")
            for j, line in enumerate(wrap_words(what, int(CW / 6.6))[:2]):
                dt.append(f"drawtext=fontfile={FONT_R}:expansion=none:text='{esc(line)}':x={x}:y={70 + 15 * j}:fontsize=11:fontcolor={GREY}")
            ml = [(t, GREY) for t in LEGEND] if k == 0 else mlines[k - 1]
            for j, (t, c) in enumerate(ml):
                dt.append(f"drawtext=fontfile={FONT_R}:expansion=none:text='{esc(t)}':x={x}:y={y_face + CW + 8 + LINE * j}:fontsize=12:fontcolor={c}")
        dt.append(f"drawtext=fontfile={FONT_R}:expansion=none:text='{esc(f'{CW / mbox[2]:.1f}x mouth zoom (nearest, follows the lips, same crop in every column). Codec noise in this row ~2 LSB mean, up to ~15: compare pixels in the diff row')}':x=8:y={TOP + CH + 4}:fontsize=12:fontcolor=white")
        dt.append(f"drawtext=fontfile={FONT_R}:expansion=none:text='{esc('face + neck of the composited output, raw pre-encode: black = identical; a change is drawn 40 + 8 x LSB (1 LSB = 48/255, >= 27 LSB saturates), same in every column')}':x=8:y={y_zoom + zh + 4}:fontsize=12:fontcolor=white")
        dt.append(f"drawtext=fontfile={FONT}:text='frame %{{frame_num}} / {T}':x=w-150:y=6:fontsize=14:fontcolor=white:start_number=1")
        dst = out / f"{who}_lineage.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "24", "-i", "pipe:0",
               "-i", str(ident.audio), "-vf", ",".join(dt) + ",format=yuv420p", "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-crf", str(a.crf), "-preset", "medium", "-profile:v", "high", "-movflags", "+faststart",
               "-c:a", "aac", "-shortest", str(dst)]
        def canvas_for(i, canvas):
            canvas[:] = 0
            for k, (full, zoom, face, _) in enumerate(cols):
                x = k * CW
                canvas[TOP:TOP + CH, x:x + CW] = full[i]
                canvas[y_zoom:y_zoom + zh, x:x + CW] = zoom[i]
                if k == 0:
                    cell = cv2.resize(face[i], (CW, CW), interpolation=cv2.INTER_AREA if face.shape[1] > CW else cv2.INTER_LINEAR)
                else:
                    cell = fit_diff(diff_vis(face[i], cols[0][2][i]))
                canvas[y_face:y_face + CW, x:x + CW] = cell
                if k:   # column dividers: below the title, not through the two row captions
                    for a0, a1 in ((26, TOP + CH), (y_zoom, y_zoom + zh), (y_face, H)):
                        canvas[a0:a1, x - 1:x + 1] = 90
            return canvas

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        canvas = np.zeros((H, W, 3), np.uint8)
        for i in range(T):
            p.stdin.write(canvas_for(i, canvas).tobytes())
        p.stdin.close()
        assert p.wait() == 0, "ffmpeg failed"
        # lossless stills straight from the raw canvas (not from the mp4): widest mouth and the most-different output frame
        pngs = []
        for i in (widest, worst):
            vf = ",".join(dt[:-1] + [f"drawtext=fontfile={FONT}:expansion=none:text='frame {i + 1} / {T}':x=w-150:y=6:fontsize=14:fontcolor=white"])
            png = out / f".{who}_still_{i + 1}.png"
            q = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-i", "pipe:0",
                                  "-vf", vf, "-frames:v", "1", str(png)], stdin=subprocess.PIPE)
            q.stdin.write(canvas_for(i, canvas).tobytes())
            q.stdin.close()
            assert q.wait() == 0
            pngs.append(png)
        stills = out / f"{who}_stills.png"
        cv2.imwrite(str(stills), np.vstack([cv2.imread(str(x)) for x in pngs]), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        for x in pngs:
            x.unlink()
        rep["stills_frames_1based"] = {"widest_mouth": widest + 1, "last_column_most_different_output": worst + 1}
        rep["video"] = str(dst.relative_to(ROOT))
        rep["video_size"] = [W, H]
        report["identities"][who] = rep
        print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, {W}x{H}) stills frames {widest + 1},{worst + 1}; "
              + "; ".join(f"{k}: bit-exact {v['bit_exact_vs_harness_render']}" for k, v in rep.items()
                          if isinstance(v, dict) and "bit_exact_vs_harness_render" in v)
              + f"; BEFORE bit-exact: {rep['BEFORE']['raw_matches_render_json']}", flush=True)
        del cols, faces
    (out / "lineage_report.json").write_text(json.dumps(qab.clean(report), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
