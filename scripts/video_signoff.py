"""Sign-off package: BEFORE | r2 | r5, plain side by side, plus each version's clean output clip, per identity.

The frames are the same as in scripts/video_lineage.py: every clip is rebuilt from its raw pre-encode frames with
the unchanged chin.py compose (quality_ab_metrics.load_arm), checked bit-exact against the recorded render SHAs,
and encoded with identical settings, so no version gets a compression advantage. Writes to
experiments/video_validation/signoff_r2_r5/:
  <id>_before.mp4, <id>_r2.mp4, <id>_r5.mp4   clean 512x896 output clips with the speech track
  <id>_side_by_side.mp4                        BEFORE | r2 | r5 at native resolution with a label bar
  reel_before_r2_r5.mp4                        all six side-by-sides back to back (60 s)
  web/<id>_side_by_side.mp4                    smaller encode of the side-by-side (<= ~12 MB) for a web page
  signoff_report.json                          bit-exactness per clip
usage: video_signoff.py [--ids a,b] [--crf 14] [--web-crf 21]
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
from scripts import quality_ab_metrics as qab  # noqa: E402
from scripts.video_lineage import FONT, FONT_R, esc  # noqa: E402

IMPL = ROOT / "docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream"
R400 = ROOT / "docs/fps_comparisons/4070s_400fps_20260928/chin_multistream"
ARMS = [
    ("BEFORE", "previous working pipeline", "252 fps (6 streams)", None),
    ("r2", "stagewise FP16 UNet + prefix cache + TRT TAESD", "350 fps (6 streams), 1.39x", IMPL / "V_srcmix_taesdtrt"),
    ("r5", "r2 + selective INT8 UNet (gmac_0.50)", "~400 fps sustained (6 streams), 1.59x", R400 / "V_srcg50"),
]
BAR = 58


def harness_sha(cap: Path, tag: str):
    j = cap.parent / f"{cap.name}.json"
    widx = str(int(tag[6:8]))
    for r in json.loads(j.read_text()).get("repeats", []):
        clips = r.get("per_worker", {}).get(widx, {}).get("clips", [])
        if clips:
            return clips[0].get("raw_refined_sha256")
    return None


def encode(path: Path, frames, audio: Path, crf: int, vf: str | None = None):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "24", "-i", "pipe:0",
           "-i", str(audio), "-map", "0:v", "-map", "1:a"]
    if vf:
        cmd += ["-vf", vf + ",format=yuv420p"]
    else:
        cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-profile:v", "high", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "96k", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed for {path}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", default=",".join(qab.DIV_IDS))
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--web-crf", type=int, default=21)
    a = ap.parse_args()
    out = ROOT / "experiments/video_validation/signoff_r2_r5"
    (out / "web").mkdir(parents=True, exist_ok=True)
    report = {"command": sys.argv, "crf": a.crf, "web_crf": a.web_crf, "identities": {}}
    sides = []
    for who in a.ids.split(","):
        ident = qab.Identity.from_dir(qab.DIV / who)
        rep, frames = {}, {}
        with tempfile.TemporaryDirectory() as td:
            for name, _, _, cap in ARMS:
                if cap is None:
                    arm = qab.Arm.parse(f"dir={qab.DIV / who}", name)
                    qab.load_arm(ident, arm, "raw")
                    ok = bool(arm.info.get("raw_matches_render_json"))
                else:
                    tag = next(cap.glob(f"stream??_{who}_faces.npz")).name[: -len("_faces.npz")]
                    arr = np.load(cap / f"{tag}_arrays.npz")
                    gp, dp = Path(td) / f"{tag}_g.npy", Path(td) / f"{tag}_d.npy"
                    np.save(gp, arr["generated_landmarks"])
                    np.save(dp, arr["chin_delta"])
                    arm = qab.Arm(label=name, faces=cap / f"{tag}_faces.npz", g=gp, chin_delta=dp)
                    qab.load_arm(ident, arm, "raw")
                    ok = harness_sha(cap, tag) == arm.info.get("frames_sha256")
                rep[name] = {"bit_exact": ok, "frames_sha256": arm.info.get("frames_sha256")}
                frames[name] = arm.frames
                arm.frames = None
        # clean single clips, identical encode for every version
        for name in frames:
            encode(out / f"{who}_{name.lower()}.mp4", frames[name], ident.audio, a.crf)
        # side by side with a label bar
        T = len(frames["BEFORE"])
        h, w = frames["BEFORE"][0].shape[:2]
        canvas = np.zeros((T, h + BAR, 3 * w, 3), np.uint8)
        for k, (name, _, _, _) in enumerate(ARMS):
            canvas[:, BAR:, k * w:(k + 1) * w] = np.stack(frames[name])
            if k:
                canvas[:, BAR:, k * w - 1:k * w + 1] = 90
        del frames
        dt = []
        for k, (name, what, fps, _) in enumerate(ARMS):
            x = k * w + 10
            col = "yellow" if name == "r5" else "white"
            dt.append(f"drawtext=fontfile={FONT}:expansion=none:text='{esc(name + '  ' + fps)}':x={x}:y=8:fontsize=17:fontcolor={col}")
            dt.append(f"drawtext=fontfile={FONT_R}:expansion=none:text='{esc(what)}':x={x}:y=33:fontsize=13:fontcolor=0xdddddd")
        dt.append(f"drawtext=fontfile={FONT}:expansion=none:text='{esc(who)}':x=w-tw-10:y=33:fontsize=13:fontcolor=cyan")
        vf = ",".join(dt)
        side = out / f"{who}_side_by_side.mp4"
        encode(side, canvas, ident.audio, a.crf, vf)
        encode(out / "web" / f"{who}_side_by_side.mp4", canvas, ident.audio, a.web_crf, vf)
        del canvas
        sides.append(side)
        rep["files"] = {k: f"{who}_{k.lower()}.mp4" for k in ("BEFORE", "r2", "r5")} | {"side_by_side": side.name}
        report["identities"][who] = rep
        print(f"{who}: " + ", ".join(f"{k} bit-exact {v['bit_exact']}" for k, v in rep.items() if isinstance(v, dict) and "bit_exact" in v)
              + f"; side-by-side {side.stat().st_size / 1e6:.1f} MB, web {(out / 'web' / side.name).stat().st_size / 1e6:.1f} MB",
              flush=True)
    if len(sides) > 1:
        lst = out / ".reel_list.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in sides))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(out / "reel_before_r2_r5.mp4")], check=True)
        lst.unlink()
        print(f"reel {(out / 'reel_before_r2_r5.mp4').stat().st_size / 1e6:.1f} MB", flush=True)
    (out / "signoff_report.json").write_text(json.dumps(qab.clean(report), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
