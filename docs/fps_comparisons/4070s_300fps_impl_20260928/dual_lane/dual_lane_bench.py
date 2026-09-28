"""Dual-lane GPU path micro-benchmark (toward 350 fps with 100% chin).

One lane = stagewise FP16 bs16 UNet (CUDA-graph replay) -> TensorRT TAESD (bs8 x2, fused uint8 BGR
post) -> non_blocking D2H into pinned memory, all on the lane's own CUDA stream. Each lane owns its
own UNet backend instance (own static buffers, arena, graph) and its own TAESD TRT instance (own
execution context), so lanes can genuinely overlap. Compares sustained aggregate frames/s for
--lanes 1 vs --lanes 2 (vs 3) on real multi-avatar corpus inputs at the power cap.

Per-lane math is identical to the single-lane path (same engines, same inputs, same order), so a
2-lane run must produce bit-identical outputs; this is checked on the first batch of every lane.

Run: scripts/box_guard.sh run --min-avail-gb 8 -- <venv python> <this file> --lanes 2 --seconds 60
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk-perf300")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.chdir(ROOT)
os.environ.update(MUSETALK_UNET_BACKEND="trt_stagewise", MUSETALK_UNET_STAGEWISE_BATCH="16",
                  MUSETALK_TAESD_BACKEND="trt", MUSETALK_TAESD_TRT_STRICT="1", MUSETALK_TAESD_TRT_BUILD=os.environ.get("MUSETALK_TAESD_TRT_BUILD", "0"),
                  MUSETALK_TRT_FALLBACK="0")

import torch  # noqa: E402

torch.backends.cudnn.benchmark = True
DEV = torch.device("cuda:0")


def smi_sampler(stop, samples):
    q = "clocks.sm,power.draw,temperature.gpu,utilization.gpu"
    while not stop.is_set():
        try:
            out = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True)
            samples.append([float(x) for x in out.strip().split(",")])
        except Exception:
            pass
        stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--inputs", type=int, default=8, help="bs16 input batches cycled")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import unet_stagewise_trt as sw
    import vae_fast_decoder as vfd

    files = sorted(glob.glob(str(ROOT / "calibration/unet_multi_avatar_20260928/unet_io_*.pt")))[: 2 * a.inputs]
    lat, aud = [], []
    for i in range(0, len(files) - 1, 2):
        d0, d1 = torch.load(files[i], map_location="cpu"), torch.load(files[i + 1], map_location="cpu")
        lat.append(torch.cat([d0["latent_batch"], d1["latent_batch"]]).half().to(DEV))
        aud.append(torch.cat([d0["audio_feature_batch"], d1["audio_feature_batch"]]).half().to(DEV))

    lanes = []
    for i in range(a.lanes):
        unet = sw.StagewiseTrtUnetBackend.load(sw.engine_dir_for_batch(16), device=DEV)
        taesd = vfd.load_taesd_trt_backend(DEV, torch.float16)
        stream = torch.cuda.Stream(device=DEV)
        pinned = [torch.empty((16, 256, 256, 3), dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        lanes.append(dict(unet=unet, taesd=taesd, stream=stream, pinned=pinned, k=0, frames=0, ev=None))
    torch.cuda.synchronize()

    def step(lane, j):
        with torch.cuda.stream(lane["stream"]):
            z = lane["unet"](lat[j], None, encoder_hidden_states=aud[j]).sample
            u8 = lane["taesd"].decode_bgr_u8(z)
            buf = lane["pinned"][lane["k"] % 2]
            buf.copy_(u8, non_blocking=True)
            lane["k"] += 1
            ev = torch.cuda.Event()
            ev.record(lane["stream"])
            return ev, u8

    # exactness: every lane's first batch equals lane 0's
    ref = None
    for li, lane in enumerate(lanes):
        _, u8 = step(lane, 0)
        torch.cuda.synchronize()
        if ref is None:
            ref = u8.clone()
        exact = bool(torch.equal(ref, u8))
        lane["exact_vs_lane0"] = exact

    stop = threading.Event(); samples = []
    th = threading.Thread(target=smi_sampler, args=(stop, samples), daemon=True)

    def run_for(seconds, record):
        pending = [None] * len(lanes)
        n = 0; t0 = time.perf_counter(); j = 0
        while time.perf_counter() - t0 < seconds:
            for li, lane in enumerate(lanes):
                if pending[li] is not None:
                    pending[li].synchronize()  # keep at most one batch in flight per lane
                    if record:
                        lane["frames"] += 16
                    n += 16
                pending[li], _ = step(lane, j % len(lat))
                j += 1
        for li, lane in enumerate(lanes):
            if pending[li] is not None:
                pending[li].synchronize()
                if record:
                    lane["frames"] += 16
                n += 16
        return n, time.perf_counter() - t0

    run_for(a.warmup, False)
    th.start()
    frames, wall = run_for(a.seconds, True)
    stop.set(); th.join(timeout=3)
    import statistics as st
    smi = {k: st.median([s[i] for s in samples]) if samples else None
           for i, k in enumerate(("sm_mhz", "power_w", "temp_c", "util_pct"))}
    res = dict(label=a.label or f"lanes{a.lanes}", lanes=a.lanes, seconds=wall, frames=frames,
               aggregate_fps=frames / wall, ms_per_frame=1000 * wall / frames,
               per_lane_frames=[l["frames"] for l in lanes], exact_vs_lane0=[l["exact_vs_lane0"] for l in lanes],
               vram_mib=torch.cuda.max_memory_allocated() / 2**20, smi_median=smi, tags="[M]")
    out = OUT / f"dual_lane_{res['label']}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
