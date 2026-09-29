#!/usr/bin/env python
"""G-ENC CPU test (plan item 1.8): decoded-vs-source quality at 2.5 Mbps.

Encodes real avatar frames (chinese_bob idle + talking source clips, 512x896,
played as a 20 fps sequence) through each WebRTC encoder configuration exactly
as aiortc's RTP sender calls it (encode() -> RTP payloads), depayloads and
decodes with aiortc's own decoders, and compares the decoded I420 against the
source I420:

  PSNR-Y / PSNR-U / PSNR-V / PSNR-YUV(6:1:1), SSIM-Y (Gaussian 11x11, sigma 1.5),
  achieved bitrate, encoder CPU ms/frame (process CPU incl. encoder threads).

Configurations (one subprocess each, so module-level encoder patches never mix):
  h264_aiortc_default   today's default for H.264 clients (aiortc 1.14 libx264:
                        preset medium, zerolatency, Baseline, auto threads)
  h264_x264tuned_veryfast / h264_x264tuned_ultrafast   WEBRTC_H264_IMPL=x264tuned, 1 thread
  vp8_pyav_default      WEBRTC_VP8_ENCODER=pyav (aiortc 1.14 stock VP8, PyAV libvpx)
  vp8_native            WEBRTC_VP8_ENCODER=native (pinned libvpx v1.13.1), upstream threads
  vp8_native_threads1   + WEBRTC_NATIVE_VP8_THREADS=1
NVENC needs the GPU and is covered by gpu_sequence.sh.

Gate (G-ENC, proposed in the plan): candidate PSNR-YUV mean within 0.2 dB of the
baseline of the same codec family (h264 -> h264_aiortc_default; vp8 native ->
vp8_pyav_default), plus a cross-family report vs h264_aiortc_default, which is
what H.264-capable browsers receive today (api_server prefers H.264 unless
WEBRTC_VP8_ENCODER=native). Exit 0 when every configuration was measured; the
per-candidate verdicts are in "gates" / "g_enc_fail".

Writes test_encoder_quality.json next to this file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = Path(__file__).with_name("test_encoder_quality.json")
PY = sys.executable
BITRATE = 2_500_000
FPS = 20
CLIPS = {
    "bob_idle": ROOT / "results/v15/avatars/chinese_bob_pink_bedroom_idle_d4b06da317/input_video.mp4",
    "bob_talking": ROOT / "results/v15/avatars/chinese_bob_pink_bedroom_talking_3373c10448/input_video.mp4",
}
FRAMES_PER_CLIP = 200
WARMUP_FRAMES = 20  # rate-control settle; "steady" metrics exclude them

CONFIGS = {
    "h264_aiortc_default": {"family": "h264", "env": {"WEBRTC_H264_IMPL": "aiortc"}},
    "h264_x264tuned_veryfast": {"family": "h264", "env": {
        "WEBRTC_H264_IMPL": "x264tuned", "WEBRTC_H264_X264_PRESET": "veryfast",
        "WEBRTC_H264_X264_THREADS": "1"}},
    "h264_x264tuned_ultrafast": {"family": "h264", "env": {
        "WEBRTC_H264_IMPL": "x264tuned", "WEBRTC_H264_X264_PRESET": "ultrafast",
        "WEBRTC_H264_X264_THREADS": "1"}},
    "h264_x264tuned_faster": {"family": "h264", "env": {
        "WEBRTC_H264_IMPL": "x264tuned", "WEBRTC_H264_X264_PRESET": "faster",
        "WEBRTC_H264_X264_THREADS": "1"}},
    "h264_x264tuned_fast": {"family": "h264", "env": {
        "WEBRTC_H264_IMPL": "x264tuned", "WEBRTC_H264_X264_PRESET": "fast",
        "WEBRTC_H264_X264_THREADS": "1"}},
    "h264_x264tuned_medium": {"family": "h264", "env": {
        "WEBRTC_H264_IMPL": "x264tuned", "WEBRTC_H264_X264_PRESET": "medium",
        "WEBRTC_H264_X264_THREADS": "1"}},
    # GPU only (NVENC opens a CUDA context): run from gpu_sequence.sh with --configs.
    "h264_nvenc": {"family": "h264", "gpu": True, "env": {
        "WEBRTC_H264_IMPL": "nvenc", "WEBRTC_NVENC_MAX_SESSIONS": "12"}},
    "vp8_pyav_default": {"family": "vp8", "env": {"WEBRTC_VP8_ENCODER": "pyav"}},
    "vp8_native": {"family": "vp8", "env": {
        "WEBRTC_VP8_ENCODER": "native", "WEBRTC_NATIVE_VP8_MAX_BITRATE_BPS": str(BITRATE)}},
    "vp8_native_threads1": {"family": "vp8", "env": {
        "WEBRTC_VP8_ENCODER": "native", "WEBRTC_NATIVE_VP8_MAX_BITRATE_BPS": str(BITRATE),
        "WEBRTC_NATIVE_VP8_THREADS": "1"}},
}
BASELINE = {"h264": "h264_aiortc_default", "vp8": "vp8_pyav_default"}
GATE_DB = 0.2


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def load_source(path: Path, count: int):
    import av
    frames = []
    container = av.open(str(path))
    try:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.reformat(format="yuv420p").to_ndarray())
            if len(frames) >= count:
                break
    finally:
        container.close()
    return frames


def ssim_y(a, b):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda v: gaussian_filter(v, 1.5, truncate=3.5)  # noqa: E731 (11x11 window)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(ssim[5:-5, 5:-5].mean())


def plane_psnr(a, b):
    import numpy as np
    mse = float(np.mean((a.astype(np.int32) - b.astype(np.int32)) ** 2))
    return (99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)), mse


def build_encoder(name: str):
    """Instantiate the encoder class the server would register for this config."""
    if name.startswith("h264"):
        import aiortc.codecs as codecs
        import aiortc.codecs.h264 as h264
        from scripts.webrtc_h264_override import install_h264_encoder
        install_h264_encoder(6_000_000, label="enc_quality")  # api_server's MAX_BITRATE
        return codecs.H264Encoder(), h264.H264Decoder(), h264.h264_depayload
    import aiortc.codecs as codecs
    import aiortc.codecs.vpx as vpx
    from scripts.webrtc_native_vp8 import configure_vp8_encoder
    if CONFIGS[name]["env"]["WEBRTC_VP8_ENCODER"] == "pyav":
        vpx.MAX_BITRATE = BITRATE  # stock ceiling is 1.5 Mbps; equal-bitrate comparison
    status = configure_vp8_encoder("enc_quality")
    encoder = codecs.Vp8Encoder()
    return encoder, vpx.Vp8Decoder(), vpx.vp8_depayload, status


def worker(name: str) -> dict:
    import av
    import numpy as np
    from aiortc.jitterbuffer import JitterFrame
    built = build_encoder(name)
    encoder, decoder, depayload = built[0], built[1], built[2]
    encoder.target_bitrate = BITRATE
    result = {"config": name, "family": CONFIGS[name]["family"], "env": CONFIGS[name]["env"],
              "encoder_class": type(encoder).__name__, "clips": {}}
    if len(built) > 3:
        result["vp8_status"] = {k: v for k, v in built[3].items() if k != "packages"}
    timestamp_step = 90000 // FPS
    frame_index = 0
    for clip, path in CLIPS.items():
        source = load_source(path, FRAMES_PER_CLIP)
        rows = []
        cpu_total = wall_total = 0.0
        bytes_total = 0
        for i, planes in enumerate(source):
            frame = av.VideoFrame.from_ndarray(planes, format="yuv420p")
            frame.pts = frame_index * timestamp_step
            frame.time_base = Fraction(1, 90000)
            frame.duration = timestamp_step
            frame_index += 1
            c0, w0 = time.process_time(), time.perf_counter()
            payloads, ts = encoder.encode(frame, force_keyframe=(i == 0))
            cpu_total += time.process_time() - c0
            wall_total += time.perf_counter() - w0
            bytes_total += sum(len(p) for p in payloads)
            data = b"".join(depayload(p) for p in payloads)
            decoded = decoder.decode(JitterFrame(data, ts)) if data else []
            if not decoded:
                rows.append(None)
                continue
            out = decoded[-1].reformat(format="yuv420p").to_ndarray()
            h = planes.shape[0] * 2 // 3
            y_ref, y_out = planes[:h], out[:h]
            u_ref, u_out = planes[h:h + h // 4], out[h:h + h // 4]
            v_ref, v_out = planes[h + h // 4:], out[h + h // 4:]
            py, my = plane_psnr(y_ref, y_out)
            pu, mu = plane_psnr(u_ref, u_out)
            pv, mv = plane_psnr(v_ref, v_out)
            mse = (6 * my + mu + mv) / 8
            pyuv = 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
            rows.append({"psnr_y": py, "psnr_u": pu, "psnr_v": pv, "psnr_yuv": pyuv,
                         "ssim_y": ssim_y(y_ref, y_out) if i % 2 == 0 else None})
        good = [r for r in rows if r is not None]
        steady = [r for i, r in enumerate(rows) if r is not None and i >= WARMUP_FRAMES]

        def agg(values, key):
            vals = [v[key] for v in values if v.get(key) is not None]
            if not vals:
                return None
            vals_sorted = sorted(vals)
            return {"mean": round(float(np.mean(vals)), 4), "min": round(vals_sorted[0], 4),
                    "p05": round(vals_sorted[max(0, int(0.05 * (len(vals) - 1)))], 4)}

        result["clips"][clip] = {
            "frames": len(source), "decoded": len(good), "undecoded": len(rows) - len(good),
            "bitrate_kbps": round(bytes_total * 8 * FPS / max(1, len(source)) / 1000, 1),
            "encode_cpu_ms_per_frame": round(cpu_total / len(source) * 1000, 3),
            "encode_wall_ms_per_frame": round(wall_total / len(source) * 1000, 3),
            "all": {k: agg(good, k) for k in ("psnr_y", "psnr_u", "psnr_v", "psnr_yuv", "ssim_y")},
            "steady": {k: agg(steady, k) for k in ("psnr_y", "psnr_yuv", "ssim_y")},
        }
    codec = getattr(getattr(encoder, "codec", None), "name", None)
    result["codec_in_use"] = codec if isinstance(codec, str) else type(encoder).__name__
    if name == "h264_nvenc" and result["codec_in_use"] != "h264_nvenc":
        raise RuntimeError(f"NVENC config fell back to {result['codec_in_use']}; not an NVENC measurement")
    means = [c["steady"]["psnr_yuv"]["mean"] for c in result["clips"].values()]
    result["steady_psnr_yuv_mean"] = round(float(np.mean(means)), 4)
    result["steady_ssim_y_mean"] = round(float(np.mean(
        [c["steady"]["ssim_y"]["mean"] for c in result["clips"].values()])), 5)
    result["encode_cpu_ms_per_frame"] = round(float(np.mean(
        [c["encode_cpu_ms_per_frame"] for c in result["clips"].values()])), 3)
    result["bitrate_kbps"] = round(float(np.mean(
        [c["bitrate_kbps"] for c in result["clips"].values()])), 1)
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print("RESULT " + json.dumps(worker(sys.argv[2])), flush=True)
        return 0
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="", help="comma list (default: every CPU config)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    selected = ([c for c in args.configs.split(",") if c] if args.configs else
                [name for name, cfg in CONFIGS.items() if not cfg.get("gpu")])
    for name in selected:
        if name not in CONFIGS:
            print(f"unknown config {name}; known: {sorted(CONFIGS)}")
            return 2
    # A family baseline is always measured alongside its candidates.
    for name in list(selected):
        base = BASELINE[CONFIGS[name]["family"]]
        if base not in selected:
            selected.insert(0, base)
    if "h264_aiortc_default" not in selected:
        selected.insert(0, "h264_aiortc_default")
    out_path = Path(args.out)
    results = {}
    for name in selected:
        config = CONFIGS[name]
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("WEBRTC_H264", "WEBRTC_VP8", "WEBRTC_NATIVE_VP8"))}
        env.update(config["env"])
        started = time.perf_counter()
        proc = subprocess.run([PY, __file__, "--worker", name], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, timeout=900)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if proc.returncode != 0 or line is None:
            results[name] = {"config": name, "error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"FAIL worker {name}: rc={proc.returncode}\n{proc.stderr[-1500:]}", flush=True)
            continue
        results[name] = json.loads(line[len("RESULT "):])
        r = results[name]
        print(f"{name:26s} psnr_yuv={r['steady_psnr_yuv_mean']:.3f} dB ssim_y={r['steady_ssim_y_mean']:.5f} "
              f"bitrate={r['bitrate_kbps']:.0f} kbps encode_cpu={r['encode_cpu_ms_per_frame']:.2f} ms/frame "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    gates = []
    cross = results.get("h264_aiortc_default", {}).get("steady_psnr_yuv_mean")
    for name, r in results.items():
        if "error" in r:
            gates.append({"config": name, "result": "fail", "reason": "worker error"})
            continue
        base_name = BASELINE[r["family"]]
        base = results.get(base_name, {}).get("steady_psnr_yuv_mean")
        delta = None if base is None else round(r["steady_psnr_yuv_mean"] - base, 4)
        undecoded = sum(c["undecoded"] for c in r["clips"].values())
        ok = delta is not None and delta >= -GATE_DB and undecoded == 0
        gates.append({
            "config": name, "baseline": base_name, "result": "pass" if ok else "fail",
            "delta_psnr_yuv_db_vs_family_baseline": delta,
            "delta_psnr_yuv_db_vs_h264_today": (None if cross is None else
                                                round(r["steady_psnr_yuv_mean"] - cross, 4)),
            "delta_ssim_y_vs_family_baseline": (
                None if base_name not in results or "error" in results[base_name] else
                round(r["steady_ssim_y_mean"] - results[base_name]["steady_ssim_y_mean"], 5)),
            "bitrate_kbps": r["bitrate_kbps"], "encode_cpu_ms_per_frame": r["encode_cpu_ms_per_frame"],
            "undecoded_frames": undecoded,
        })
        print(f"{'PASS' if ok else 'FAIL'} G-ENC {name}: dPSNR-YUV={delta} dB vs {base_name} "
              f"(gate >= -{GATE_DB}), vs h264 today "
              f"{gates[-1]['delta_psnr_yuv_db_vs_h264_today']} dB", flush=True)
    # The suite passes when every configuration was measured; each candidate's
    # G-ENC verdict is reported separately (a failing candidate is a finding).
    passed = all("error" not in r for r in results.values())
    cheapest = sorted((g for g in gates if g["result"] == "pass" and g["config"].startswith("h264")
                       and g["config"] != "h264_aiortc_default"),
                      key=lambda g: g["encode_cpu_ms_per_frame"])
    out_path.write_text(json.dumps({"suite": "encoder_quality_g_enc", "bitrate_bps": BITRATE,
                               "cheapest_h264_within_gate": cheapest[0]["config"] if cheapest else None,
                               "fps": FPS, "frames_per_clip": FRAMES_PER_CLIP,
                               "warmup_frames_excluded": WARMUP_FRAMES,
                               "clips": {k: str(v) for k, v in CLIPS.items()},
                               "gate_db": GATE_DB, "passed": passed,
                               "g_enc_pass": [g["config"] for g in gates if g["result"] == "pass"],
                               "g_enc_fail": [g["config"] for g in gates if g["result"] != "pass"],
                               "gates": gates,
                               "results": results}, indent=2))
    print(f"{'PASS' if passed else 'FAIL'} encoder_quality measured {len(results)} configs; "
          f"G-ENC fail: {[g['config'] for g in gates if g['result'] != 'pass']} -> {out_path} "
          f"(cheapest H.264 within {GATE_DB} dB: {cheapest[0]['config'] if cheapest else None})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
