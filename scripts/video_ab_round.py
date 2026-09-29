"""Labelled pre-change vs candidate A/B videos for one optimization round (full chin recipe renders).

A = the accepted pre-change render of each identity (experiments/avatar_diversity_20260927/<id>/refined_raw.mp4,
    faces.npz). B = a chin_multistream_render.py capture run with --encode --save-arrays (refined mp4 + raw faces).
Per identity: full frame A | B on top, 3x nearest-neighbour mouth zoom A | B and the exact raw 256 px face
|A-B| x8 panel below (black = identical), burned-in labels with backends and measured fps, and a metrics line
from the quality tool's run JSON when present. Also a 2x3 mosaic of all identities (full frames, B only labelled).

Usage: video_ab_round.py --round r2_srcmix --run-dir <capture dir> --label-b "..." --fps-a 165 --fps-b 350.2
"""
from __future__ import annotations

import argparse, glob, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "character_factory/h3_avatar_workflow")]
import chin  # noqa: E402

ACC = Path("/workspace/experiments/avatar_diversity_20260927")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DT = f"drawtext=fontfile={FONT}:expansion=none"


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def wrap(text: str, width: int = 128) -> list:
    """Split a label at ' | ' boundaries into lines of at most ~width characters."""
    lines, cur = [], ""
    for part in text.split(" | "):
        cand = part if not cur else cur + " | " + part
        if len(cand) > width and cur:
            lines.append(cur); cur = part
        else:
            cur = cand
    return lines + ([cur] if cur else [])


def drawlines(lines, y0, color, size=12, step=17):
    return ",".join(f"{DT}:text='{esc(t)}':x=8:y={y0 + i * step}:fontsize={size}:fontcolor={color}" for i, t in enumerate(lines))


def metrics_line(ident: str, qlabel: str) -> str:
    runs = ROOT / "docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/runs"
    f = runs / f"{ident}__{qlabel}.json"
    if not f.exists():
        return "quality metrics: (not run)"
    d = json.loads(f.read_text())
    md = (runs / f"{ident}__{qlabel}.md").read_text() if (runs / f"{ident}__{qlabel}.md").exists() else ""
    def row(name):
        for line in md.splitlines():
            if line.startswith(f"| {name}"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                return cells[3] if len(cells) > 3 else "?"
        return "?"
    return (f"verdict {d.get('verdict')} | lip corr {row('Lip aperture mean')} | flicker mouth {row('Flicker mouth')} "
            f"jaw {row('Flicker jaw')} | chin err d {row('Chin-target abs error')} px | landmarks {row('Jaw+lip landmark dev A vs B mean')}"
            f"/{row('Jaw+lip landmark dev A vs B p99')} px | sharp {row('Mouth sharpness')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label-a", default="A (left) PRE-CHANGE accepted render: TRT .ts FP16 bs8 UNet + compiled TAESD + 100% chin + seam")
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--fps-a", default="single-stream 148-171 fps (render.json)")
    ap.add_argument("--fps-b", required=True)
    ap.add_argument("--quality-label", default="")
    a = ap.parse_args()
    out_dir = ROOT / "experiments/video_validation" / a.round
    out_dir.mkdir(parents=True, exist_ok=True)
    run = Path(a.run_dir)
    made = []
    for refined_b in sorted(run.glob("stream*_refined.mp4")):
        tag = refined_b.name[: -len("_refined.mp4")]
        ident = tag.split("_", 1)[1]
        A = ACC / ident
        faces_b = run / f"{tag}_faces.npz"
        fa = np.load(A / "faces.npz")["faces"].astype(np.int16)
        fb = np.load(faces_b)["faces"].astype(np.int16) if faces_b.exists() else None
        g = np.load(A / "generated_landmarks.npy")[:, list(chin.LIPS)]
        x0, y0 = np.floor(g[..., 0].min()), np.floor(g[..., 1].min())
        x1, y1 = np.ceil(g[..., 0].max()), np.ceil(g[..., 1].max())
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = int(max(x1 - x0, 1) * 1.6) // 2 * 2
        h = int(w * 0.62) // 2 * 2
        X = int(max(0, min(512 - w, cx - w / 2)))
        Y = int(max(0, min(896 - h, cy - h / 2 + 8)))
        zw, zh = w * 3, h * 3
        diff_path = out_dir / f"{ident}_facediff.mp4"
        if fb is not None:
            diff = np.clip(np.abs(fa - fb) * 8, 0, 255).astype(np.uint8)
            stats = f"raw face diff mean {float(np.abs(fa - fb).mean()):.3f} max {int(np.abs(fa - fb).max())} LSB"
        else:
            diff = np.zeros_like(fa, dtype=np.uint8)
            stats = "raw faces not saved"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "256x256", "-r", "24",
                              "-i", "pipe:0", "-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p", str(diff_path)],
                             stdin=subprocess.PIPE)
        p.stdin.write(diff.tobytes()); p.stdin.close(); assert p.wait() == 0
        la = wrap(f"{a.label_a} | {a.fps_a}")
        lb = wrap(f"{a.label_b} | {a.fps_b}")
        lm = wrap(f"{ident}: {stats} | " + metrics_line(ident, a.quality_label or a.round))
        top_h = 8 + 17 * (len(la) + len(lb))
        bot_h = 8 + 16 * len(lm)
        fc = (f"[0:v]split=2[a0][a1];[1:v]split=2[b0][b1];"
              f"[a1]crop={w}:{h}:{X}:{Y},scale={zw}:{zh}:flags=neighbor[az];[b1]crop={w}:{h}:{X}:{Y},scale={zw}:{zh}:flags=neighbor[bz];"
              f"[2:v]scale={zh}:{zh}:flags=neighbor[dz];"
              f"[a0][b0]hstack=2[top];[az][bz][dz]hstack=3,scale=1024:-2:flags=neighbor[bot];"
              f"[top]pad=1024:ih+{top_h}:0:{top_h}:color=black,{drawlines(la, 4, 'white')},{drawlines(lb, 4 + 17 * len(la), 'yellow')}[topl];"
              f"[bot]pad=1024:ih+26:0:26:color=black,{DT}:text='3x mouth zoom (nearest) A | B | raw 256px generated face abs(A-B) x8 (black = identical)':x=8:y=5:fontsize=12:fontcolor=white[botl];"
              f"[topl][botl]vstack=2,pad=iw:ih+{bot_h}:0:0:color=black,{drawlines(lm, 0, 'cyan', 11, 16).replace(':y=', ':y=h-' + str(bot_h) + '+')},format=yuv420p[v]")
        dst = out_dir / f"{ident}_ab.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(A / "refined_raw.mp4"), "-i", str(refined_b), "-i", str(diff_path),
               "-i", str(A / "speech.wav"), "-filter_complex", fc, "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-crf", "12",
               "-preset", "medium", "-r", "24", "-c:a", "aac", "-shortest", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(ident, "FAILED", r.stderr[-500:]); continue
        diff_path.unlink(missing_ok=True)
        made.append((ident, dst, refined_b))
        print("wrote", dst, flush=True)
    if len(made) >= 2:
        # mosaic: B renders of all identities, 3 per row, labelled
        ins, fcs = [], []
        for i, (ident, _, rb) in enumerate(made[:6]):
            ins += ["-i", str(rb)]
            fcs.append(f"[{i}:v]scale=256:448,{DT}:text='{esc(ident)}':x=6:y=6:fontsize=12:fontcolor=yellow[m{i}]")
        n = len(made[:6]); cols = 3; rows = (n + cols - 1) // cols
        layout = "|".join(f"{(i % cols) * 256}_{(i // cols) * 448}" for i in range(n))
        fc = ";".join(fcs) + ";" + "".join(f"[m{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}:fill=black," \
             f"pad=iw:ih+30:0:30:color=black,{DT}:text='{esc(a.label_b + ' | ' + a.fps_b)}':x=6:y=6:fontsize=12:fontcolor=white,format=yuv420p[v]"
        mos = out_dir / "mosaic_candidate.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "14",
                            "-r", "24", str(mos)], capture_output=True, text=True)
        print("mosaic", "ok" if r.returncode == 0 else r.stderr[-300:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
