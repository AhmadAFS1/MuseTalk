"""RAM attribution of prepared avatars under each memory layout (CPU only, no CUDA).

Plan item 0.9 (avatar part) for the 300 fps work. For every layout config the parent
drives child processes that load distinct prepared poses from results/v15/avatars one
after another IN ONE PROCESS (the way the server's avatar cache accumulates them) and
reads /proc/self/smaps_rollup around each load:

  * per pose, in a FRESH child each (after a tiny warm-up avatar): anonymous-memory
    growth (primary; USS/RSS beside it, see summarize()) for load + a compose walk that fills the PNG store's decoded LRU (steady state), the
    loader's own estimate and its breakdown, load wall and CPU seconds, compose ms/frame.
    Deltas measured inside a long-lived process are also kept (coresident_delta_uss_mb):
    they swing +-100 MB per pose as glibc reuses or grows heap, but sum to the same total;
  * co-resident totals for the first 1 / 5 / 10 / all poses: measured when one child
    held them all under its RSS cap, else base + the sum of fresh per-pose costs
    (flagged "extrapolated"; the sum of co-resident deltas is reported beside it);
  * per identity (idle + talking + smiling): the sum of its poses.

Safety (shared box): each child sets oom_score_adj=1000, runs a 4 Hz watchdog that
exits (86) if MemAvailable < --kill-below-gb or (87) if its own RSS > --rss-cap-gb,
waits before each load until MemAvailable >= --floor-gb + the predicted cost, and
stops loading (exit 0, "capped") when RSS + predicted cost would pass --plan-cap-gb.
The parent then starts a fresh child for the remaining poses.

Latents: prepared latents.pt hold CUDA tensors; with CUDA hidden they load to host
RAM here, while the server keeps them in VRAM. They are reported separately
(latents_host_bytes) and subtracted for the server-equivalent host numbers.

    cd /workspace/MuseTalk-perf300 && CUDA_VISIBLE_DEVICES= \\
      /workspace/.venvs/musetalk_trt_stagewise/bin/python \\
      docs/fps_comparisons/4070s_300fps_impl_20260928/avatar_memory/ram_attribution.py
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/workspace/MuseTalk-perf300")
HERE = Path(__file__).resolve().parent
AVATAR_ROOT = ROOT / "results/v15/avatars"
sys.path[:0] = [str(HERE)]

import memguard  # noqa: E402

LAYOUT_FLAGS = ("MUSETALK_AVATAR_MASK_CHANNELS", "MUSETALK_AVATAR_MASK_STORE", "MUSETALK_AVATAR_FRAME_STORE",
                "MUSETALK_AVATAR_DECODED_LRU_FRAMES", "MUSETALK_AVATAR_PNG_READAHEAD",
                "MUSETALK_AVATAR_PNG_DECODE_WORKERS", "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA")
LEAN = {"MUSETALK_AVATAR_MASK_CHANNELS": "1", "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA": "0"}
CONFIGS = {
    "baseline": {},
    "mask1_lean": dict(LEAN),
    "mask1_lean_maskpng": {**LEAN, "MUSETALK_AVATAR_MASK_STORE": "png"},
    "png_all": {**LEAN, "MUSETALK_AVATAR_MASK_STORE": "png", "MUSETALK_AVATAR_FRAME_STORE": "png"},
}
# First-load cost guess per config before anything is measured (GB); replaced by the
# largest delta seen so far in the run.
PRIOR_GB = {"baseline": 1.0, "mask1_lean": 0.7, "mask1_lean_maskpng": 0.6, "png_all": 0.35}
POSE_RE = re.compile(r"^(?P<ident>.+?)_(?P<pose>idle|talking|smiling)(?:_(?P<tag>.+))?$")
WARMUP_AVATAR = "shared_indian_20260915"  # 50 frames: exercises every code path once


# ----------------------------------------------------------------------------- discovery
def pose_inventory():
    """Distinct prepared poses. Dirs whose sampled frame files are byte-identical to an
    earlier dir (fixed-face-height / latent-mix research variants) collapse onto it."""
    groups = {}
    for directory in sorted(AVATAR_ROOT.iterdir()):
        frames = sorted((directory / "full_imgs").glob("*.png"))
        if not (directory / "latents.pt").exists() or not frames:
            continue
        count = len(frames)
        digest = hashlib.blake2b(str(count).encode(), digest_size=16)
        for index in sorted({0, count // 4, count // 2, 3 * count // 4, count - 1}):
            digest.update(frames[index].read_bytes())
        groups.setdefault(digest.hexdigest(), []).append((directory.name, count))
    poses = []
    for members in groups.values():
        members.sort(key=lambda item: (len(item[0]), item[0]))
        name, count = members[0]
        match = POSE_RE.match(name)
        ident, pose = (match["ident"], match["pose"]) if match else (name, "single")
        poses.append({"avatar": name, "identity": ident, "pose": pose, "frames": count,
                      "collapsed_variants": [m[0] for m in members[1:]]})
    return poses


def interleaved_order(poses):
    """Round-robin over identities so the first N poses are N distinct identities."""
    by_pose = {}
    for entry in poses:
        by_pose.setdefault(entry["pose"], []).append(entry)
    ordered = []
    for pose in ("talking", "idle", "smiling", "single"):
        ordered += sorted(by_pose.get(pose, []), key=lambda e: e["identity"])
    return ordered


# ----------------------------------------------------------------------------- child
class RssCapWatchdog(memguard.Watchdog):
    def __init__(self, kill_below_gb, rss_cap_gb):
        super().__init__(kill_below_gb)
        self.rss_cap = rss_cap_gb * memguard.GB
        self.peak_rss = 0

    def _run(self):
        while not self._stop.wait(0.25):
            avail = memguard.mem_available_gb()
            self.min_seen_gb = min(self.min_seen_gb, avail)
            rss = memguard.smaps_rollup()["rss"]
            self.peak_rss = max(self.peak_rss, rss)
            if avail < self.kill_below_gb:
                print(f"[child] WATCHDOG: MemAvailable {avail:.2f} GB < {self.kill_below_gb}", flush=True)
                os._exit(86)
            if rss > self.rss_cap:
                print(f"[child] WATCHDOG: RSS {rss / memguard.GB:.2f} GB > cap", flush=True)
                os._exit(87)


def _trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _smaps_gb(snapshot):
    return {k: round(v / memguard.GB, 4) for k, v in snapshot.items()}


def child_main(args):
    memguard.set_oom_score_adj(1000)
    watchdog = RssCapWatchdog(args.kill_below_gb, args.rss_cap_gb).start()
    import cv2
    import numpy as np
    import torch

    torch.set_num_threads(1)
    memguard.force_cpu_torch_load()
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    import api_avatar

    stub = SimpleNamespace(model_dtype=torch.float16, runtime_dtype=torch.float16)
    face = np.random.RandomState(5).randint(0, 256, (256, 256, 3), dtype=np.uint8)

    def load(avatar_id):
        return api_avatar.APIAvatar(avatar_id, "", 0, 8, stub, stub, None, None,
                                    SimpleNamespace(version="v15"), preparation=False)

    def walk(avatar):
        times = []
        for position in range(args.walk_frames):
            started = time.perf_counter()
            avatar.compose_frame(face, position)
            times.append((time.perf_counter() - started) * 1e3)
        frames = avatar.frame_list_cycle
        if hasattr(frames, "stats"):
            deadline = time.time() + 2
            while frames.stats()["inflight"] and time.time() < deadline:
                time.sleep(0.01)
        return times

    # Warm-up: one tiny avatar exercises thread pools, torch.load, cv2 and tqdm once, so
    # the first measured pose does not carry one-time process costs.
    warm = load(WARMUP_AVATAR)
    walk(warm)
    del warm
    _trim()
    base = memguard.smaps_rollup()
    out = {"config": args.config, "flags": api_avatar.avatar_memory_layout_flags(),
           "base_after_import_and_warmup": _smaps_gb(base), "poses": [], "status": "done"}
    previous = base
    predicted_gb = args.predicted_gb
    held = []
    for avatar_id in args.avatars:
        rss_now = memguard.smaps_rollup()["rss"] / memguard.GB
        if rss_now + predicted_gb > args.plan_cap_gb:
            out["status"] = "capped"
            break
        if not memguard.wait_for_headroom(need_gb=predicted_gb, floor_gb=args.floor_gb, timeout_s=900):
            out["status"] = "ram_headroom_timeout"
            break
        cpu0, wall0 = memguard.process_cpu_seconds(), time.perf_counter()
        watchdog.peak_rss = 0
        avatar = load(avatar_id)
        load_wall, load_cpu = time.perf_counter() - wall0, memguard.process_cpu_seconds() - cpu0
        gc.collect()
        after_load = memguard.smaps_rollup()
        load_peak_rss = watchdog.peak_rss
        compose_ms = walk(avatar)
        gc.collect()
        after_walk = memguard.smaps_rollup()
        held.append(avatar)
        breakdown = avatar.estimate_memory_breakdown()
        frames = avatar.frame_list_cycle
        record = {
            "avatar": avatar_id,
            "cycle_positions": len(avatar.coord_list_cycle),
            "unique_frames": (frames.unique_count if hasattr(frames, "unique_count")
                              else len({id(f) for f in frames})),
            "load_wall_s": round(load_wall, 3), "load_cpu_s": round(load_cpu, 3),
            "delta_after_load": {k: after_load[k] - previous[k] for k in after_load},
            "delta_steady": {k: after_walk[k] - previous[k] for k in after_walk},
            "cumulative_steady": after_walk,
            "load_transient_rss_over_steady": max(0, load_peak_rss - after_walk["rss"]),
            "estimate_bytes": avatar.estimate_memory_usage_bytes(),
            "breakdown": breakdown,
            "compose_ms_p50": round(statistics.median(compose_ms), 3),
            "compose_ms_p95": round(sorted(compose_ms)[int(0.95 * (len(compose_ms) - 1))], 3),
            "layout": avatar.memory_layout_stats(),
        }
        out["poses"].append(record)
        previous = after_walk
        predicted_gb = max(predicted_gb * 0.5, 1.25 * record["delta_steady"]["uss"] / memguard.GB
                           + record["load_transient_rss_over_steady"] / memguard.GB)
        print(f"[child {args.config}] {avatar_id}: +{record['delta_steady']['uss'] / memguard.MB:.0f} MB USS "
              f"(est {record['estimate_bytes'] / memguard.MB:.0f} MB) load {load_wall:.2f}s", flush=True)
    gc.collect()
    out["end_untrimmed"] = memguard.smaps_rollup()
    _trim()
    out["end_trimmed"] = memguard.smaps_rollup()
    out["peak_rss_gb"] = round(max(watchdog.peak_rss, out["end_untrimmed"]["rss"]) / memguard.GB, 3)
    out["min_mem_available_gb"] = round(watchdog.min_seen_gb, 2)
    out["cv2_threads"] = cv2.getNumThreads()
    watchdog.stop()
    Path(args.child_out).write_text(json.dumps(out))
    return 0


# ----------------------------------------------------------------------------- parent
def run_chunked(config, avatars, args, scratch, tag=""):
    """Load ``avatars`` in order in as few capped children as possible (co-resident)."""
    remaining = list(avatars)
    children, predicted = [], PRIOR_GB[config]
    attempt = 0
    while remaining:
        attempt += 1
        child_out = scratch / f"ram_child_{config}_{tag}{attempt}.json"
        result = None
        if args.reuse_children and child_out.exists():
            cached = json.loads(child_out.read_text())
            loaded = [p["avatar"] for p in cached["poses"]]
            if loaded and loaded == remaining[: len(loaded)]:
                result = cached
                print(f"[reuse {config}] {child_out.name}: {len(loaded)} poses", flush=True)
        if result is None:
            env = {k: v for k, v in os.environ.items() if k not in LAYOUT_FLAGS}
            env.update(CONFIGS[config])
            env["CUDA_VISIBLE_DEVICES"] = ""
            cmd = [sys.executable, __file__, "--child", "--config", config, "--child-out", str(child_out),
                   "--predicted-gb", f"{predicted:.3f}", "--walk-frames", str(args.walk_frames),
                   "--plan-cap-gb", str(args.plan_cap_gb), "--rss-cap-gb", str(args.rss_cap_gb),
                   "--floor-gb", str(args.floor_gb), "--kill-below-gb", str(args.kill_below_gb),
                   "--avatars", *remaining]
            for retry in range(3):
                proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                if proc.returncode != 86:  # 86: MemAvailable watchdog (someone else's load)
                    break
                print(f"[parent] {config}: watchdog exit 86, waiting for headroom (retry {retry + 1})", flush=True)
                memguard.wait_for_headroom(need_gb=predicted + 0.6, floor_gb=args.floor_gb, timeout_s=900)
            log = [line for line in proc.stdout.splitlines() if line.startswith("[")]
            print("\n".join(log), flush=True)
            if proc.returncode != 0 or not child_out.exists():
                raise SystemExit(f"child for {config} failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n"
                                 f"{proc.stderr[-3000:]}")
            result = json.loads(child_out.read_text())
        loaded = [p["avatar"] for p in result["poses"]]
        if not loaded:
            raise SystemExit(f"child for {config} loaded nothing: {result['status']}")
        children.append(result)
        remaining = remaining[len(loaded):]
        predicted = max(1.25 * max(p["delta_steady"]["uss"] for p in result["poses"]) / memguard.GB,
                        0.05)
    return children


def run_fresh(config, avatars, args, scratch):
    """One fresh child per pose: its material cost without other avatars' heap to reuse."""
    fresh = {}
    for index, avatar in enumerate(avatars):
        (child,) = run_chunked(config, [avatar], args, scratch, tag=f"fresh{index:02d}_")
        fresh[avatar] = child
    return fresh


def summarize(config, children, fresh, order):
    """Primary metric: ANONYMOUS memory growth (heap + anonymous mmap), which is where
    avatar materials live. USS/RSS are kept beside it but also move with clean
    file-backed library pages (libtorch, cv2) that the kernel evicts or that flip between
    private and shared when another process (the live server, same venv) maps them: they
    swung +-150 MB per pose on this shared box while anon stayed within a few MB."""
    coresident_delta = {}
    for child in children:
        for record in child["poses"]:
            coresident_delta[record["avatar"]] = record["delta_steady"]
    info = {p["avatar"]: p for p in order}
    mb, gb = memguard.MB, memguard.GB
    per_pose = {}
    for avatar, child in fresh.items():
        (record,) = child["poses"]
        steady = record["delta_steady"]
        latents_host = record["breakdown"]["latents_host_bytes"]
        per_pose[avatar] = {
            "identity": info[avatar]["identity"], "pose": info[avatar]["pose"],
            "cycle_positions": record["cycle_positions"], "unique_frames": record.get("unique_frames"),
            "anon_mb": round(steady["anon"] / mb, 1),
            "uss_mb": round(steady["uss"] / mb, 1), "rss_mb": round(steady["rss"] / mb, 1),
            "anon_after_load_mb": round(record["delta_after_load"]["anon"] / mb, 1),
            "coresident_delta_anon_mb": round(coresident_delta[avatar]["anon"] / mb, 1),
            "server_equiv_host_mb": round((steady["anon"] - latents_host) / mb, 1),
            "estimate_mb": round(record["estimate_bytes"] / mb, 1),
            "estimate_host_server_mb": round((record["estimate_bytes"] - latents_host) / mb, 1),
            "estimate_over_measured": round(record["estimate_bytes"] / max(1, steady["anon"]), 3),
            "breakdown_mb": {k.replace("_bytes", ""): round(v / mb, 1) for k, v in record["breakdown"].items()},
            "load_wall_s": record["load_wall_s"], "load_cpu_s": record["load_cpu_s"],
            "load_transient_rss_mb": round(record["load_transient_rss_over_steady"] / mb, 1),
            "compose_ms_p50": record["compose_ms_p50"], "compose_ms_p95": record["compose_ms_p95"],
        }
        frames = record["layout"].get("frame_list_cycle", {})
        if "encoded_bytes" in frames:
            per_pose[avatar]["png_store"] = {k: frames[k] for k in (
                "unique", "lru_capacity", "readahead", "inline_decodes", "readahead_decodes", "hits",
                "requests", "decode_seconds")}
    per_identity = {}
    summed = ("anon_mb", "uss_mb", "server_equiv_host_mb", "estimate_mb", "load_wall_s", "unique_frames")
    for avatar, entry in per_pose.items():
        ident = per_identity.setdefault(entry["identity"], {"poses": {}, **{k: 0.0 for k in summed}})
        ident["poses"][entry["pose"]] = avatar
        for key in summed:
            ident[key] = round(ident[key] + (entry[key] or 0), 2)
    for ident in per_identity.values():
        ident["complete_pose_set"] = set(ident["poses"]) >= {"idle", "talking", "smiling"}
    # co-resident totals: measured in the first chunked child while it held them all, else
    # base + the sum of fresh per-pose costs; the sum of co-resident deltas is kept beside it
    first_child = children[0]
    base = {k: v * gb for k, v in first_child["base_after_import_and_warmup"].items()}
    measured_n = len(first_child["poses"])
    coresident = {}
    ordered = [p["avatar"] for p in order]
    for label, n in (("1", 1), ("5", 5), ("10", 10), ("all", len(ordered))):
        n = min(n, len(ordered))
        fresh_sum = sum(per_pose[a]["anon_mb"] for a in ordered[:n]) * mb
        delta_sum = sum(coresident_delta[a]["anon"] for a in ordered[:n])
        estimate_sum = sum(per_pose[a]["estimate_mb"] for a in ordered[:n]) * mb
        if n <= measured_n:
            cumulative = first_child["poses"][n - 1]["cumulative_steady"]
            entry = {"poses": n, "measured": True,
                     "process_rss_gb": round(cumulative["rss"] / gb, 3),
                     "process_anon_gb": round(cumulative["anon"] / gb, 3),
                     "avatars_anon_gb": round((cumulative["anon"] - base["anon"]) / gb, 3)}
        else:
            entry = {"poses": n, "measured": False,
                     "extrapolated": "base + sum of fresh per-pose anon (the RSS cap stops one process first)",
                     "process_anon_gb": round((base["anon"] + fresh_sum) / gb, 3),
                     "avatars_anon_gb": round(fresh_sum / gb, 3)}
        entry["sum_fresh_per_pose_anon_gb"] = round(fresh_sum / gb, 3)
        entry["sum_coresident_deltas_anon_gb"] = round(delta_sum / gb, 3)
        entry["estimate_sum_gb"] = round(estimate_sum / gb, 3)
        entry["estimate_over_coresident"] = round(estimate_sum / max(1, delta_sum), 3)
        latents = sum(fresh[a]["poses"][0]["breakdown"]["latents_host_bytes"] for a in ordered[:n])
        entry["avatars_server_equiv_host_gb"] = round(entry["avatars_anon_gb"] - latents / gb, 3)
        entry["avatars_coresident_server_equiv_host_gb"] = round((delta_sum - latents) / gb, 3)
        coresident[label] = entry
    return {
        "config": config, "env": CONFIGS[config], "flags": first_child["flags"],
        "base_after_import_and_warmup_gb": first_child["base_after_import_and_warmup"],
        "coresident_children": len(children), "poses_measured_coresident_in_first_child": measured_n,
        "child_peak_rss_gb": max([c["peak_rss_gb"] for c in children] + [c["peak_rss_gb"] for c in fresh.values()]),
        "child_min_mem_available_gb": min([c["min_mem_available_gb"] for c in children]
                                          + [c["min_mem_available_gb"] for c in fresh.values()]),
        "retained_free_heap_mb_first_child": round(
            (first_child["end_untrimmed"]["anon"] - first_child["end_trimmed"]["anon"]) / mb, 1),
        "per_pose": per_pose, "per_identity": per_identity, "coresident": coresident,
    }


def parent_main(args):
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    order = interleaved_order(pose_inventory())
    if args.limit:
        order = order[: args.limit]
    print(f"{len(order)} distinct poses:", [p["avatar"] for p in order], flush=True)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "host_mem_available_gb_at_start":
              round(memguard.mem_available_gb(), 2), "order": order, "walk_frames": args.walk_frames,
              "caps": {"plan_cap_gb": args.plan_cap_gb, "rss_cap_gb": args.rss_cap_gb,
                       "floor_gb": args.floor_gb, "kill_below_gb": args.kill_below_gb},
              "reused_child_results": bool(args.reuse_children), "configs": {}}
    for config in args.configs.split(","):
        avatars = [p["avatar"] for p in order]
        children = run_chunked(config, avatars, args, scratch)
        fresh = run_fresh(config, avatars, args, scratch)
        report["configs"][config] = summarize(config, children, fresh, order)
        Path(args.out).write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.time() - started, 1)
    # headline: per-identity (complete idle/talking/smiling sets) and per-pose means
    headline = {}
    for config, summary in report["configs"].items():
        complete = [v for v in summary["per_identity"].values() if v["complete_pose_set"]]
        poses = list(summary["per_pose"].values())
        sized = [p for p in poses if p["estimate_mb"] >= 20]  # the 50-frame still avatar is noise-sized
        headline[config] = {
            "per_pose_anon_mb_mean": round(statistics.mean(p["anon_mb"] for p in poses), 1),
            "per_pose_anon_mb_max": round(max(p["anon_mb"] for p in poses), 1),
            "per_pose_uss_mb_mean": round(statistics.mean(p["uss_mb"] for p in poses), 1),
            "per_pose_server_equiv_mb_mean": round(statistics.mean(p["server_equiv_host_mb"] for p in poses), 1),
            "per_identity_anon_mb_mean": round(statistics.mean(v["anon_mb"] for v in complete), 1),
            "per_identity_anon_mb_max": round(max(v["anon_mb"] for v in complete), 1),
            "per_identity_server_equiv_mb_mean": round(statistics.mean(v["server_equiv_host_mb"] for v in complete), 1),
            "per_identity_server_equiv_mb_max": round(max(v["server_equiv_host_mb"] for v in complete), 1),
            "estimate_over_measured_median": round(statistics.median(p["estimate_over_measured"] for p in sized), 3),
            "estimate_over_measured_min": round(min(p["estimate_over_measured"] for p in sized), 3),
            "estimate_over_measured_max": round(max(p["estimate_over_measured"] for p in sized), 3),
            "estimate_over_measured_sum": round(sum(p["estimate_mb"] for p in poses)
                                                / sum(p["anon_mb"] for p in poses), 3),
            "estimate_over_coresident_all": summary["coresident"]["all"]["estimate_over_coresident"],
            "load_wall_s_mean": round(statistics.mean(p["load_wall_s"] for p in poses), 2),
            "load_wall_s_max": round(max(p["load_wall_s"] for p in poses), 2),
            "compose_ms_p50_median": round(statistics.median(p["compose_ms_p50"] for p in poses), 3),
            "complete_identities": len(complete),
            "coresident": summary["coresident"],
        }
    report["headline"] = headline
    base = headline.get("baseline")
    for config, h in headline.items():
        line = (f"{config:20s} per-pose anon {h['per_pose_anon_mb_mean']:6.1f} MB (max {h['per_pose_anon_mb_max']:.0f})"
                f"  per-identity {h['per_identity_anon_mb_mean']:7.1f} MB (max {h['per_identity_anon_mb_max']:.0f})"
                f"  est/measured {h['estimate_over_measured_median']} (co-resident {h['estimate_over_coresident_all']})"
                f"  load {h['load_wall_s_mean']} s  all {len(report['order'])} poses "
                f"{h['coresident']['all']['sum_coresident_deltas_anon_gb']} GB")
        if base and config != "baseline":
            line += f"  saving {100 * (1 - h['per_pose_anon_mb_mean'] / base['per_pose_anon_mb_mean']):.0f}%"
        print(line)
    truthful = all(0.85 <= h["estimate_over_measured_min"] and h["estimate_over_measured_max"] <= 1.10
                   and 0.90 <= h["estimate_over_coresident_all"] <= 1.10 for h in headline.values())
    report["estimate_gate"] = {
        "verdict": "PASS" if truthful else "FAIL",
        "rule": ("estimate_memory_usage_bytes / measured anon growth: every pose (fresh process) in [0.85, 1.10] "
                 "and all poses co-resident in [0.90, 1.10], for every config")}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(("PASS" if truthful else "FAIL") + ": " + report["estimate_gate"]["rule"] + " ->", args.out)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--avatars", nargs="*", default=[])
    parser.add_argument("--child-out")
    parser.add_argument("--predicted-gb", type=float, default=1.0)
    parser.add_argument("--walk-frames", type=int, default=96)
    parser.add_argument("--plan-cap-gb", type=float, default=1.85)
    parser.add_argument("--rss-cap-gb", type=float, default=1.95)
    parser.add_argument("--floor-gb", type=float, default=4.0)
    parser.add_argument("--kill-below-gb", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-children", action="store_true",
                        help="reuse child JSONs already in --scratch (same pose order) instead of re-running")
    parser.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    parser.add_argument("--out", default=str(HERE / "ram_attribution.json"))
    args = parser.parse_args()
    return child_main(args) if args.child else parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
