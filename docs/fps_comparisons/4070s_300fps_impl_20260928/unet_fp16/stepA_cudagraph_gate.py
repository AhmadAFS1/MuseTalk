"""Step A gate (plan item 1.2): MUSETALK_TRT_UNET_CUDAGRAPHS=manual|runtime vs 0 on the shipping bs8 .ts UNet.

One process, one engine load (the .ts load has a ~9.5 GB host-RSS transient), modes switched on
the same TrtUnetBackend via set_cudagraphs_mode():
  1. mode 0 (today's call) over every capture of the multi-avatar corpus (main + holdout) -> reference outputs;
  2. manual and runtime modes over the same captures -> max_abs vs mode 0 (gate: exactly 0);
  3. overwrite test: out_A = f(A); out_B = f(B) with no sync in between; out_A must still equal ref(A)
     and must not share storage with out_B or any static buffer;
     plus the MultiTrtUnetBackend bs16 -> 2 x bs8 split path;
  4. speed: sustained back-to-back UNet calls per mode (CUDA events per call + wall throughput),
     host enqueue time per call, with 1 Hz nvidia-smi clocks/power.
Run under the lease:
  scripts/box_guard.sh run --min-avail-gb 12 -- /workspace/.venvs/musetalk_trt_stagewise/bin/python <this> [--seconds 30]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/workspace/MuseTalk-perf300")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("MUSETALK_UNET_BACKEND", "trt")
os.environ.setdefault("MUSETALK_TRT_FALLBACK", "0")
os.environ.setdefault("MUSETALK_TRT_UNET_PATHS", "8:models/tensorrt_unet_sm89_bs8_local/unet_trt.ts")
os.environ.pop("MUSETALK_TRT_UNET_CUDAGRAPHS", None)  # the loader must come up in mode 0

import torch  # noqa: E402

from scripts.trt_runtime import load_unet_trt_backend, MultiTrtUnetBackend, TrtUnetBackend  # noqa: E402


def smi_sampler(rows, stop):
    fields = "clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used"
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-lms=1000"],
                         stdout=subprocess.PIPE, text=True)
    while not stop.is_set():
        line = p.stdout.readline()
        if not line:
            break
        rows.append((time.time(), [x.strip() for x in line.split(",")]))
    p.terminate()


def smi_stats(rows, t0, t1):
    sel = [r for t, r in rows if t0 <= t <= t1]
    out = {"samples": len(sel)}
    for i, k in enumerate(["sm_mhz", "power_w", "temp_c", "util", "mem_mib"]):
        vals = sorted(float(r[i]) for r in sel if len(r) > i)
        if vals:
            out[k] = {"median": statistics.median(vals), "min": vals[0], "max": vals[-1]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0, help="sustained seconds per mode")
    ap.add_argument("--out", default=str(OUT / "stepA_cudagraph_gate.json"))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda:0")
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tags": "[M]"}

    t = time.time()
    multi = load_unet_trt_backend(device=dev)
    assert isinstance(multi, MultiTrtUnetBackend), type(multi)
    inner = multi.backends_by_batch["8"]
    assert isinstance(inner, TrtUnetBackend) and inner.cudagraphs_mode == "0"
    res["load_s"] = time.time() - t

    files = sorted((ROOT / "calibration/unet_multi_avatar_20260928").glob("unet_io_*.pt"))
    files += sorted((ROOT / "calibration/unet_multi_avatar_20260928/holdout").glob("unet_io_*.pt"))
    lat, aud, ref_eager = [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lat.append(d["latent_batch"].to(dev, torch.float16))
        aud.append(d["audio_feature_batch"].to(dev, torch.float16))
        ref_eager.append(d["pred_latents"].to(dev, torch.float16))
    res["captures"] = len(files)
    ts = torch.tensor([0], device=dev)

    def run_all(fn):
        outs = []
        with torch.inference_mode():
            for a, b in zip(lat, aud):
                outs.append(fn(a, ts, encoder_hidden_states=b).sample)
        torch.cuda.synchronize()
        return outs

    ref = run_all(inner)  # mode 0
    res["mode0_vs_eager_mae_max"] = max((o.float() - r.float()).abs().mean().item() for o, r in zip(ref, ref_eager))
    res["mode0_vs_eager_max_abs_max"] = max((o.float() - r.float()).abs().max().item() for o, r in zip(ref, ref_eager))
    ref_repeat = run_all(inner)
    res["mode0_repeat_max_abs"] = max((o.float() - r.float()).abs().max().item() for o, r in zip(ref_repeat, ref))
    del ref_repeat

    stop = threading.Event()
    smi_rows = []
    th = threading.Thread(target=smi_sampler, args=(smi_rows, stop), daemon=True)
    th.start()

    modes = {}
    for mode in ["manual", "runtime", "0"]:
        m = {}
        inner.set_cudagraphs_mode(mode)
        with torch.no_grad():
            inner.warmup()  # captures the graph in manual mode (as TrtUnetBackend.load does)
        outs = run_all(inner)
        diffs = [(o.float() - r.float()).abs().max().item() for o, r in zip(outs, ref)]
        m["max_abs_vs_mode0_max"] = max(diffs)
        m["files_nonzero"] = sum(1 for x in diffs if x != 0.0)
        m["bitexact"] = all(torch.equal(o, r) for o, r in zip(outs, ref))
        del outs
        # overwrite test: two different batches, no sync in between, first result read afterwards
        with torch.inference_mode():
            oa = inner(lat[0], ts, encoder_hidden_states=aud[0]).sample
            ob = inner(lat[200], ts, encoder_hidden_states=aud[200]).sample
            oc = inner(lat[1], ts, encoder_hidden_states=aud[1]).sample
        torch.cuda.synchronize()
        static_ptrs = set()
        for e in inner._graph_entries.values():
            static_ptrs |= {e.static_output.data_ptr(), e.static_latent.data_ptr(), e.static_audio.data_ptr()}
        m["overwrite_test"] = {
            "A_still_equals_ref": bool(torch.equal(oa, ref[0])),
            "B_equals_ref": bool(torch.equal(ob, ref[200])),
            "C_equals_ref": bool(torch.equal(oc, ref[1])),
            "distinct_storage": len({oa.data_ptr(), ob.data_ptr(), oc.data_ptr()}) == 3,
            "no_alias_with_static_buffers": not ({oa.data_ptr(), ob.data_ptr(), oc.data_ptr()} & static_ptrs),
        }
        # bs16 split path through MultiTrtUnetBackend (2 x bs8)
        with torch.inference_mode():
            o16 = multi(torch.cat([lat[2], lat[3]]), ts, encoder_hidden_states=torch.cat([aud[2], aud[3]])).sample
        torch.cuda.synchronize()
        m["multi_bs16_split_equals_ref"] = bool(torch.equal(o16, torch.cat([ref[2], ref[3]])))
        # runtime-mode aliasing of the raw module output (does torch_tensorrt return a fresh tensor?)
        if mode == "runtime":
            import torch_tensorrt
            with torch.inference_mode(), torch_tensorrt.runtime.enable_cudagraphs():
                ra = inner.module(lat[0], aud[0])
                ra = ra[0] if isinstance(ra, (list, tuple)) else ra
                ra_ptr = ra.data_ptr()
                rb = inner.module(lat[200], aud[200])
                rb = rb[0] if isinstance(rb, (list, tuple)) else rb
            torch.cuda.synchronize()
            m["raw_runtime_module_output_fresh"] = {"A_intact": bool(torch.equal(ra, ref[0])),
                                                    "same_ptr": ra_ptr == rb.data_ptr()}
        # speed: sustained back-to-back calls, CUDA events around each call, no per-call sync
        n_in = len(lat)
        with torch.inference_mode():
            for i in range(30):
                inner(lat[i % n_in], ts, encoder_hidden_states=aud[i % n_in])
            torch.cuda.synchronize()
            evs = []
            host = []
            t0 = time.time()
            p0 = time.perf_counter()
            i = 0
            while time.perf_counter() - p0 < args.seconds:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                h0 = time.perf_counter()
                s.record()
                inner(lat[i % n_in], ts, encoder_hidden_states=aud[i % n_in])
                e.record()
                host.append((time.perf_counter() - h0) * 1e3)
                evs.append((s, e))
                if len(evs) > 4:  # bounded run-ahead (depth ~4), as a pipelined feeder would keep
                    evs[-5][1].synchronize()
                i += 1
            torch.cuda.synchronize()
            wall = time.perf_counter() - p0
            t1 = time.time()
        ms = sorted(s.elapsed_time(e) for s, e in evs)
        # isolated per-call timing (sync per call), like the validator/probes
        iso = []
        with torch.inference_mode():
            for j in range(60):
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                inner(lat[j % n_in], ts, encoder_hidden_states=aud[j % n_in])
                e.record()
                e.synchronize()
                iso.append(s.elapsed_time(e))
        iso.sort()
        m["speed"] = {
            "calls": i, "wall_s": wall, "wall_ms_per_call": wall * 1e3 / i, "wall_fps_bs8": 8 * i / wall,
            "event_ms_per_call_median": ms[len(ms) // 2], "event_ms_p90": ms[int(len(ms) * 0.9)],
            "event_ms_p99": ms[int(len(ms) * 0.99)],
            "host_enqueue_ms_median": statistics.median(host),
            "isolated_sync_per_call_ms_median": iso[len(iso) // 2],
            "gpu": smi_stats(smi_rows, t0, t1),
        }
        m["graphs_captured"] = [list(k) for k in inner._graph_entries]
        modes[mode] = m
        print(mode, json.dumps(m, default=str), flush=True)
    stop.set()
    res["modes"] = modes
    res["torch_max_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    g = modes
    res["gate"] = {
        mode: {
            "max_abs_0": g[mode]["max_abs_vs_mode0_max"] == 0.0 and g[mode]["bitexact"],
            "overwrite_ok": all(g[mode]["overwrite_test"].values()) and g[mode]["multi_bs16_split_equals_ref"],
        } for mode in ("manual", "runtime")
    }
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["gate"]), flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
