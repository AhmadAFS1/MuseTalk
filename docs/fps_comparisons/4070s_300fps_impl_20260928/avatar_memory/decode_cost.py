"""CPU cost of MUSETALK_AVATAR_FRAME_STORE=png (PNG decode on demand) and a 300 fps core estimate.

CPU only (CUDA hidden), one process, RSS < 2 GB, cv2 on 1 thread. Sections:

  1. single_thread  cv2.imdecode of real prepared frame PNGs (every distinct pose, K unique
                    frames each), per-decode thread CPU and wall ms; plus plane-0 mask decode
                    (load time only: masks are decoded once, when compose plans are built).
  2. contended      T threads decoding at once for D seconds (SMT / memory-bandwidth effect).
  3. compose_path   compose_frame over the cycle on one avatar, sequential, three stores:
                    decoded (today), png inline (LRU 1, no readahead: every decode on the
                    critical path) and png default (LRU 24, readahead 8). CPU and wall ms.
  4. paced          S sessions on S distinct avatars, each composing at 20 fps for D seconds
                    under the png layout, the way scheduler compose workers consume frames:
                    compose latency p50/p99, share of decodes left on the critical path,
                    process CPU per frame.
  5. estimate_300fps  cores = 300 x decodes-per-fresh-frame x decode CPU ms / 1000, at the
                    single-thread and the contended per-decode cost.

The box is shared (a live server and another session's load test), so wall numbers carry
ambient contention; thread CPU time is the figure to plan with.

    cd /workspace/MuseTalk-perf300 && CUDA_VISIBLE_DEVICES= \\
      /workspace/.venvs/musetalk_trt_stagewise/bin/python \\
      docs/fps_comparisons/4070s_300fps_impl_20260928/avatar_memory/decode_cost.py
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/workspace/MuseTalk-perf300")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import memguard  # noqa: E402
import ram_attribution  # noqa: E402

memguard.set_oom_score_adj(1000)


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def summary_ms(values):
    return {"n": len(values), "mean": round(statistics.mean(values), 3), "p50": round(pct(values, .5), 3),
            "p95": round(pct(values, .95), 3), "p99": round(pct(values, .99), 3)}


def unique_sample(paths, count):
    seen, picked = set(), []
    step = max(1, len(paths) // (count * 2))
    for path in paths[::step]:
        blob = path.read_bytes()
        digest = hashlib.blake2b(blob, digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        picked.append(blob)
        if len(picked) >= count:
            break
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-pose", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--contended-s", type=float, default=3.0)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--paced-s", type=float, default=10.0)
    parser.add_argument("--compose-avatar", default="chinese_bob_pink_bedroom_talking_3373c10448")
    parser.add_argument("--out", default=str(HERE / "decode_cost.json"))
    args = parser.parse_args()
    # Wait for headroom first, then arm the watchdog (it exits 86 if the box dips below
    # 4 GB while we run; rerun then - the dip is someone else's load, not ours).
    if not memguard.wait_for_headroom(need_gb=1.5, floor_gb=4.0, timeout_s=3600):
        raise SystemExit("RAM headroom timeout")
    watchdog = ram_attribution.RssCapWatchdog(kill_below_gb=4.0, rss_cap_gb=1.95).start()

    import cv2
    import numpy as np
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    memguard.force_cpu_torch_load()
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    import api_avatar

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "loadavg_start": os.getloadavg(),
              "cpu_count": os.cpu_count(), "cv2_threads": cv2.getNumThreads()}

    # 1. single-thread decode --------------------------------------------------------
    poses = ram_attribution.interleaved_order(ram_attribution.pose_inventory())
    per_pose, all_cpu, all_wall, blobs_all = {}, [], [], []
    mask_cpu = []
    for entry in poses:
        directory = ram_attribution.AVATAR_ROOT / entry["avatar"]
        blobs = unique_sample(sorted((directory / "full_imgs").glob("*.png")), args.per_pose)
        blobs_all += blobs
        for blob in blobs[:2]:
            api_avatar._decode_png_color(blob)  # warm
        cpu_ms, wall_ms, shape = [], [], None
        for _ in range(2):
            for blob in blobs:
                c0, w0 = time.thread_time_ns(), time.perf_counter_ns()
                image = api_avatar._decode_png_color(blob)
                cpu_ms.append((time.thread_time_ns() - c0) / 1e6)
                wall_ms.append((time.perf_counter_ns() - w0) / 1e6)
                shape = image.shape
        masks = unique_sample(sorted((directory / "mask").glob("*.png")), 6)
        for blob in masks:
            c0 = time.thread_time_ns()
            api_avatar._decode_png_plane0(blob, True)
            mask_cpu.append((time.thread_time_ns() - c0) / 1e6)
        per_pose[entry["avatar"]] = {"frame_shape": list(shape), "png_kb_mean": round(
            statistics.mean(len(b) for b in blobs) / 1024, 1), "decode_cpu_ms": summary_ms(cpu_ms),
            "decode_wall_ms_p50": round(pct(wall_ms, .5), 3)}
        all_cpu += cpu_ms
        all_wall += wall_ms
    megapixels = statistics.mean(v["frame_shape"][0] * v["frame_shape"][1] for v in per_pose.values()) / 1e6
    report["single_thread"] = {"per_pose": per_pose, "decode_cpu_ms": summary_ms(all_cpu),
                               "decode_wall_ms": summary_ms(all_wall), "mean_megapixels": round(megapixels, 3),
                               "mask_plane0_decode_cpu_ms": summary_ms(mask_cpu)}
    print("single-thread decode CPU ms", report["single_thread"]["decode_cpu_ms"], flush=True)

    # 2. contended ----------------------------------------------------------------------
    stop_at = time.perf_counter() + args.contended_s
    lock = threading.Lock()
    contended_cpu = []

    def worker(offset):
        local = []
        index = offset
        while time.perf_counter() < stop_at:
            blob = blobs_all[index % len(blobs_all)]
            c0 = time.thread_time_ns()
            api_avatar._decode_png_color(blob)
            local.append((time.thread_time_ns() - c0) / 1e6)
            index += 7
        with lock:
            contended_cpu.extend(local)

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(worker, range(args.threads)))
    wall = time.perf_counter() - wall0
    report["contended"] = {"threads": args.threads, "decodes": len(contended_cpu),
                           "decodes_per_s": round(len(contended_cpu) / wall, 1),
                           "decode_cpu_ms": summary_ms(contended_cpu),
                           "cpu_ms_ratio_vs_single": round(statistics.mean(contended_cpu) / statistics.mean(all_cpu), 3)}
    print("contended", report["contended"], flush=True)
    del blobs_all
    gc.collect()

    # 3. compose critical path ------------------------------------------------------------
    stub = SimpleNamespace(model_dtype=torch.float16, runtime_dtype=torch.float16)

    def load(avatar_id, env):
        for name in ram_attribution.LAYOUT_FLAGS:
            os.environ.pop(name, None)
        os.environ.update(env)
        return api_avatar.APIAvatar(avatar_id, "", 0, 8, stub, stub, None, None,
                                    SimpleNamespace(version="v15"), preparation=False)

    faces = [np.random.RandomState(i).randint(0, 256, (256, 256, 3), dtype=np.uint8) for i in range(8)]
    png = ram_attribution.CONFIGS["png_all"]
    compose_configs = {
        "decoded_today": {},
        "png_inline_lru1_ra0": {**png, "MUSETALK_AVATAR_DECODED_LRU_FRAMES": "1", "MUSETALK_AVATAR_PNG_READAHEAD": "0"},
        "png_default_lru24_ra8": png,
    }
    report["compose_path"] = {"avatar": args.compose_avatar}
    for label, env in compose_configs.items():
        if not memguard.wait_for_headroom(need_gb=0.9, floor_gb=4.0):
            raise SystemExit("RAM headroom timeout")
        avatar = load(args.compose_avatar, env)
        count = len(avatar.coord_list_cycle)
        for position in range(4):
            avatar.compose_frame(faces[0], count - 4 + position)  # warm, far from the walk start
        cpu_ms, wall_ms = [], []
        for position in range(count):
            c0, w0 = time.thread_time_ns(), time.perf_counter_ns()
            avatar.compose_frame(faces[position % len(faces)], position)
            cpu_ms.append((time.thread_time_ns() - c0) / 1e6)
            wall_ms.append((time.perf_counter_ns() - w0) / 1e6)
        entry = {"positions": count, "compose_thread_cpu_ms": summary_ms(cpu_ms),
                 "compose_wall_ms": summary_ms(wall_ms)}
        frames = avatar.frame_list_cycle
        if hasattr(frames, "stats"):
            stats = frames.stats()
            entry["store_stats"] = stats
            entry["decodes_per_compose"] = round(
                (stats["inline_decodes"] + stats["readahead_decodes"]) / max(1, stats["requests"]), 3)
        report["compose_path"][label] = entry
        print("compose", label, entry["compose_thread_cpu_ms"], entry["compose_wall_ms"]["p50"], flush=True)
        del avatar, frames
        gc.collect()
    base_cpu = report["compose_path"]["decoded_today"]["compose_thread_cpu_ms"]["mean"]
    inline_cpu = report["compose_path"]["png_inline_lru1_ra0"]["compose_thread_cpu_ms"]["mean"]
    report["compose_path"]["inline_decode_added_cpu_ms"] = round(inline_cpu - base_cpu, 3)

    # 4. paced multi-session ------------------------------------------------------------
    talking = [p["avatar"] for p in poses if p["pose"] == "talking"][: args.sessions]
    if not memguard.wait_for_headroom(need_gb=0.35 * len(talking), floor_gb=4.0):
        raise SystemExit("RAM headroom timeout")
    avatars = [load(avatar_id, png) for avatar_id in talking]
    latencies = []
    period = 0.05
    frames_done = [0]
    lock = threading.Lock()

    def session(index):
        avatar = avatars[index]
        local = []
        position = 37 * index
        next_at = time.perf_counter() + index * period / len(avatars)
        end_at = time.perf_counter() + args.paced_s
        while next_at < end_at:
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            w0 = time.perf_counter_ns()
            avatar.compose_frame(faces[position % len(faces)], position)
            local.append((time.perf_counter_ns() - w0) / 1e6)
            position += 1
            next_at += period
        with lock:
            latencies.extend(local)
            frames_done[0] += len(local)

    for avatar in avatars:  # plans are built; start every store cold at the walk start
        avatar.frame_list_cycle.release_decoded_cache()
    cpu0, wall0 = memguard.process_cpu_seconds(), time.perf_counter()
    threads = [threading.Thread(target=session, args=(i,)) for i in range(len(avatars))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cpu = memguard.process_cpu_seconds() - cpu0
    wall = time.perf_counter() - wall0
    totals = {"requests": 0, "inline_decodes": 0, "readahead_decodes": 0, "readahead_joins": 0,
              "readahead_cancels": 0, "hits": 0, "decode_seconds": 0.0}
    for avatar in avatars:
        stats = avatar.frame_list_cycle.stats()
        for key in totals:
            totals[key] += stats[key]
    decodes = totals["inline_decodes"] + totals["readahead_decodes"]
    report["paced"] = {
        "sessions": len(avatars), "avatars": talking, "fps_per_session": 1 / period, "seconds": round(wall, 2),
        "frames": frames_done[0], "compose_latency_ms": summary_ms(latencies),
        "process_cpu_ms_per_frame": round(1e3 * cpu / max(1, frames_done[0]), 3),
        "store_totals": totals, "decodes_per_frame": round(decodes / max(1, frames_done[0]), 3),
        "critical_path_decode_share": round(totals["inline_decodes"] / max(1, decodes), 3),
    }
    print("paced", report["paced"]["compose_latency_ms"], report["paced"]["critical_path_decode_share"], flush=True)
    del avatars
    gc.collect()

    # 5. 300 fps estimate ------------------------------------------------------------------
    decodes_per_frame = report["compose_path"]["png_default_lru24_ra8"].get("decodes_per_compose", 1.0)
    single = report["single_thread"]["decode_cpu_ms"]["mean"]
    contended = report["contended"]["decode_cpu_ms"]["mean"]
    report["estimate_300fps"] = {
        "fresh_fps": 300, "decodes_per_fresh_frame": decodes_per_frame,
        "decode_cpu_ms_single": single, "decode_cpu_ms_contended": contended,
        "cores_single_thread_cost": round(300 * decodes_per_frame * single / 1e3, 2),
        "cores_contended_cost": round(300 * decodes_per_frame * contended / 1e3, 2),
        "note": ("decodes per fresh frame from a sequential cycle walk (one session per avatar); "
                 "sessions sharing one avatar in lockstep hit the LRU instead. Masks cost nothing at "
                 "runtime: they are decoded once while plans are built."),
    }
    report["loadavg_end"] = os.getloadavg()
    report["peak_rss_gb"] = round(watchdog.peak_rss / memguard.GB, 3)
    report["min_mem_available_gb"] = round(watchdog.min_seen_gb, 2)
    for name in ram_attribution.LAYOUT_FLAGS:
        os.environ.pop(name, None)
    Path(args.out).write_text(json.dumps(report, indent=1))
    est = report["estimate_300fps"]
    print(f"RESULT: PNG decode {single:.2f} ms CPU/frame single-thread ({contended:.2f} contended x{args.threads}); "
          f"300 fps needs {est['cores_single_thread_cost']}-{est['cores_contended_cost']} cores; "
          f"paced p99 compose {report['paced']['compose_latency_ms']['p99']} ms -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
