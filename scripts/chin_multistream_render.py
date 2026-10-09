#!/usr/bin/env python
"""Multi-stream chin render harness: N accepted-identity streams, one GPU issue thread, one ordered
worker process per stream (tracker + 3-tap jaw filter + chin.corrected_refined), bit-exact per stream
with character_factory/h3_avatar_workflow/render_stage.py when run on the baseline backends.

Layout
  parent (this process)   spawns the N workers first (spawn context, same TRT venv interpreter, no
                          CUDA yet), then loads the backends (backend.setup() rooted at the worktree +
                          optional candidate flags), uploads every identity's latents/audio once, warms
                          up, has one owner worker per identity build the shared read-only arena
                          (chin_multistream/arena.py: frames + lean chin.prepare_refined, proven exact
                          by chin_multistream/check_prep.py), then issues all UNet/TAESD work from its
                          main thread: batches drawn round-robin across streams, each an 8-frame
                          stream-aligned chunk at base..base+8 (--align stream8); --pack 16 puts two
                          such chunks in one bs16 UNet call; the next batch is submitted before the
                          current one is waited on (double buffering).
  worker s                owns backend.Tracker (FaceMesh subprocess), receives faces through a
                          shared-memory ring (credits = backpressure), runs render_stage's per-frame
                          code in order and hashes faces + raw refined frames per 240-frame clip.
Timing: aggregate fps = total refined frames / (last frame composed by any worker - first GPU submit);
load, preparation, warmup and tracker start are excluded (as render_stage excludes them).

Usage (every run through scripts/box_guard.sh):
  python scripts/chin_multistream_render.py --backend baseline --streams 6 --loops 1 --label gateE_baseline_n6
  python scripts/chin_multistream_render.py --backend stagewise16 --streams 12 --loops 7 --repeats 2 --label T_sw16_n12
  python scripts/chin_multistream_render.py --mode serial --backend baseline --repeats 3 --label serial_baseline
Writes <out-root>/<label>.json (+ <label>/ with logs, optional crf<=12 videos and arrays).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chin_multistream import paths  # noqa: E402  (stdlib-only module; safe for spawn re-import)


def parse_args(argv=None):
    from chin_multistream.gpu import PRESETS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("multi", "serial"), default="multi")
    ap.add_argument("--backend", choices=sorted(PRESETS), default="baseline")
    ap.add_argument("--flags", default="", help="extra env K=V,K=V applied after the preset")
    ap.add_argument("--identities", default="all", help="comma list of accepted identities (default all six)")
    ap.add_argument("--streams", type=int, default=0, help="N streams (default = number of identities); identity = s mod len")
    ap.add_argument("--loops", type=int, default=1, help="each stream renders its 240-frame clip K times")
    ap.add_argument("--repeats", type=int, default=1, help="timed repeats in this process (median reported)")
    ap.add_argument("--align", choices=("stream8",), default="stream8")
    ap.add_argument("--pack", type=int, choices=(8, 16), default=0, help="frames per UNet call (default from preset)")
    ap.add_argument("--decode-split", type=int, choices=(0, 8), default=8, help="8: TAESD per 8-frame chunk; 0: one call")
    ap.add_argument("--depth", type=int, default=2, help="GPU batches in flight (2 = double buffering)")
    ap.add_argument("--ring-slots", type=int, default=3, help="8-face slots per stream ring")
    ap.add_argument("--encode", action="store_true", help="encode each stream's first clip (refined + faces) at --crf")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--save-arrays", action="store_true", help="save each stream's first-clip landmarks + chin deltas")
    ap.add_argument("--compare-accepted", action="store_true",
                    help="first clip: PSNR/max of faces and raw refined frames vs the accepted render (adds CPU work)")
    ap.add_argument("--cv2-threads", type=int, default=2, help="cv2.setNumThreads in workers (render_stage uses 2)")
    ap.add_argument("--tracking-overlap", action="store_true",
                    help="experimental/default-off: overlap one ordered canonical tracking call with composition")
    ap.add_argument("--blas-threads", type=int, default=1, help="OPENBLAS/OMP/MKL threads in workers")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-root", default=str(paths.OUT_ROOT))
    ap.add_argument("--min-timed-s", type=float, default=0.0, help="flag the run if a repeat is shorter than this")
    ap.add_argument("--thermal-warmup-s", type=float, default=0.0,
                    help="untimed GPU warmup after CPU preparation; records final 30s thermal range")
    a = ap.parse_args(argv)
    if a.tracking_overlap and a.mode != "multi":
        ap.error("tracking-overlap applies only to multi mode")
    a.identity_list = list(paths.IDENTITIES) if a.identities == "all" else [s for s in a.identities.split(",") if s]
    for ident in a.identity_list:
        if ident not in paths.IDENTITIES:
            ap.error(f"unknown identity {ident}")
    a.streams = a.streams or len(a.identity_list)
    a.pack = a.pack or PRESETS[a.backend]["pack"]
    a.extra_flags = dict(kv.split("=", 1) for kv in a.flags.split(",") if kv)
    return a


def median(xs):
    return statistics.median(xs) if xs else None


class Collector:
    """Receives worker messages outside the GPU loop; raises on worker errors or deaths."""

    def __init__(self, conns, procs):
        self.conns, self.procs = conns, procs
        self.inbox = {}

    def on_message(self, s, msg):
        if msg[0] == "error":
            raise RuntimeError(f"worker {s} failed:\n{msg[2]}")
        self.inbox.setdefault(msg[0], {})[s] = msg

    def wait_for(self, kind, streams, timeout=600.0):
        from multiprocessing.connection import wait as conn_wait

        deadline = time.monotonic() + timeout
        want = set(streams)
        closed = getattr(self, "closed", set())
        self.closed = closed
        while not want <= set(self.inbox.get(kind, {})):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"timed out waiting for {kind} from {sorted(want - set(self.inbox.get(kind, {})))}")
            ready = conn_wait([c for i, c in enumerate(self.conns) if i not in closed], timeout=min(left, 5.0))
            for c in ready:
                s = self.conns.index(c)
                try:
                    while c.poll():
                        self.on_message(s, c.recv())
                except EOFError:
                    # a worker closes its pipe right after its final "bye": that is a clean exit, not a failure
                    if s in self.inbox.get("bye", {}):
                        closed.add(s)
                        continue
                    raise RuntimeError(f"worker {s} exited (exitcode {self.procs[s].exitcode})")
            for s, p in enumerate(self.procs):
                if not p.is_alive() and s in want and s not in self.inbox.get(kind, {}):
                    raise RuntimeError(f"worker {s} died (exitcode {p.exitcode}) while waiting for {kind}")
        got = {s: self.inbox[kind].pop(s) for s in streams}
        return got


def run_multi(a, run_dir: Path) -> dict:
    import multiprocessing as mp
    from multiprocessing import shared_memory

    import numpy as np

    from chin_multistream import worker, telemetry

    streams = [a.identity_list[s % len(a.identity_list)] for s in range(a.streams)]
    unique = list(dict.fromkeys(streams))
    # Worker-only thread caps (the parent's own environment is restored right after spawning).
    blas_keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    saved_env = {k: os.environ.get(k) for k in blas_keys}
    for k in blas_keys:
        os.environ[k] = str(a.blas_threads)
    ctx = mp.get_context("spawn")
    slot_bytes = paths.BATCH * int(np.prod(paths.FACE_SHAPE))
    rings, ring_views, conns, procs = [], [], [], []
    arena_root = Path(f"/dev/shm/chinms_{os.getpid()}")
    record = dict(streams=[dict(stream=s, identity=i) for s, i in enumerate(streams)], unique_identities=unique)
    t_start = time.perf_counter()
    try:
        for s, ident in enumerate(streams):
            shm = shared_memory.SharedMemory(create=True, size=a.ring_slots * slot_bytes)
            rings.append(shm)
            ring_views.append(np.ndarray((a.ring_slots, paths.BATCH) + paths.FACE_SHAPE, np.uint8, buffer=shm.buf))
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            cfg = dict(stream=s, identity=ident, ring_name=shm.name, ring_slots=a.ring_slots, cv2_threads=a.cv2_threads,
                       tracking_overlap=a.tracking_overlap,
                       tracker_log=str(run_dir / "logs" / f"tracker_stream{s:02d}.log"), compare_accepted=a.compare_accepted)
            p = ctx.Process(target=worker.main, args=(cfg, child_conn), name=f"chinms-worker-{s}", daemon=True)
            p.start()
            child_conn.close()
            conns.append(parent_conn)
            procs.append(p)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        col = Collector(conns, procs)
        hello = col.wait_for("hello", range(a.streams), timeout=120)
        record["workers"] = {s: m[2] for s, m in hello.items()}
        record["workers_spawned_before_cuda_init"] = True
        record["spawn_s"] = time.perf_counter() - t_start

        # ---- GPU: backends, inputs, warmup (after the workers exist, before any arena memory)
        import torch
        from chin_multistream import gpu
        import ctypes
        import gc

        torch.set_num_threads(4)
        preset = gpu.PRESETS[a.backend]
        replay = a.backend == "replay"
        boxes, crops = {}, {}
        if replay:
            faces = {}
            for ident in unique:
                cache = torch.load(paths.ACCEPTED_ROOT / ident / "cache.pt", map_location="cpu", weights_only=False)
                boxes[ident], crops[ident] = cache["boxes"], cache["cropboxes"]
                faces[ident] = np.load(paths.ACCEPTED_ROOT / ident / "faces.npz")["faces"]
                del cache
            issuer = gpu.ReplayIssuer(faces, a.pack, a.depth)
            record["backends"] = dict(unet_name="replay(accepted faces.npz)", decoder_name="replay", cuda_initialized=False)
        else:
            warm_batches = "8" if (a.decode_split == 8 or a.pack == 8) else "8,16"
            unet, decoder, sf, env_record, load_s = gpu.setup_backends({**preset["env"], **a.extra_flags}, warm_batches)
            record["env"] = env_record
            record["load_s"] = load_s
            record["backends"] = gpu.describe_backends(unet, decoder)
            if record["backends"]["unet_name"] not in preset["unet_names"] or \
                    record["backends"]["decoder_name"] not in preset["decoder_names"]:
                raise RuntimeError(f"backend mismatch for preset {a.backend}: {record['backends']}")
            lat, aud = {}, {}
            for ident in unique:
                cache = torch.load(paths.ACCEPTED_ROOT / ident / "cache.pt", map_location="cpu", weights_only=False)
                lat[ident] = cache["latents"].cuda()
                aud[ident] = cache["audio"].cuda()
                boxes[ident], crops[ident] = cache["boxes"], cache["cropboxes"]
                del cache
            issuer = gpu.GpuIssuer(unet, decoder, sf, lat, aud, a.pack, a.decode_split, a.depth)
            t = time.perf_counter()
            with torch.inference_mode():
                issuer.warmup(unique[0])
            record["gpu_warmup_s"] = time.perf_counter() - t
            gc.collect()
            torch.cuda.empty_cache()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        record["parent_rss_after_load_mib"] = telemetry.rss_mib()
        record["mem_available_after_load_gb"] = telemetry.mem_available_gb()

        # ---- shared arenas: one owner worker per identity builds it, every stream maps it read-only
        owners = {ident: streams.index(ident) for ident in unique}
        t = time.perf_counter()
        for ident, s in owners.items():
            conns[s].send(("prep", str(arena_root / ident), boxes[ident], crops[ident]))
        prepped = col.wait_for("prepped", owners.values(), timeout=600)
        record["prep"] = {m[2]["identity"]: m[2] for m in prepped.values()}
        record["prep_wall_s"] = time.perf_counter() - t
        t = time.perf_counter()
        for s in range(a.streams):
            conns[s].send(("attach", str(arena_root / streams[s])))
        attached = col.wait_for("attached", range(a.streams), timeout=300)
        record["attach_wall_s"] = time.perf_counter() - t
        record["attach"] = {s: m[2] for s, m in attached.items()}
        record["mem_available_after_attach_gb"] = telemetry.mem_available_gb()
        facemesh_pids = {s: m[2]["facemesh_pid"] for s, m in attached.items()}

        # Optional extra preconditioning; default preserves historical reproductions.
        # It runs after CPU arenas are ready so preparation cannot cool the GPU again.
        if not replay and a.thermal_warmup_s > 0:
            thermal = telemetry.SmiSampler(500)
            t_thermal = time.perf_counter()
            with torch.inference_mode():
                while time.perf_counter() - t_thermal < a.thermal_warmup_s:
                    issuer.warmup(unique[0])
            record["thermal_warmup"] = {"seconds": time.perf_counter() - t_thermal,
                                        "last_30s": thermal.stop(tail_samples=60)}
        # ---- timed repeats
        reps = []
        for rep in range(a.repeats):
            rcfg = dict(loops=a.loops, encode=a.encode, crf=a.crf, save_arrays=a.save_arrays, out_dir=str(run_dir))
            for s in range(a.streams):
                conns[s].send(("run", rep, rcfg))
            col.wait_for("armed", range(a.streams), timeout=120)
            smi = telemetry.SmiSampler(500) if not replay else None
            mem = telemetry.MemSampler(1.0)
            mem.start()
            cpu_parent0 = telemetry.proc_cpu_s()
            sys0 = telemetry.system_cpu()
            with torch.inference_mode():
                gstats, _drain = gpu.run_repeat(issuer, streams, conns, ring_views, rep, a.loops, a.ring_slots, col.on_message)
            done = col.wait_for("repdone", range(a.streams), timeout=600)
            t_all = time.perf_counter()
            cpu_parent1 = telemetry.proc_cpu_s()
            sys1 = telemetry.system_cpu()
            smi_stats = smi.stop() if smi is not None else None
            mem_stats = mem.stop()
            wstats = {s: m[3] for s, m in done.items()}
            t0 = gstats["t0"]
            t_end = max(w["t_last_compose"] for w in wstats.values())
            wall = t_end - t0
            frames = a.streams * a.loops * paths.N_FRAMES
            worker_cpu = sum(w["worker_cpu_s"] for w in wstats.values())
            fm_cpu = sum(w["facemesh_cpu_s"] for w in wstats.values())
            parent_cpu = cpu_parent1 - cpu_parent0
            sys_busy = (sys1[0] - sys0[0]) / max(1, (sys1[1] - sys0[1])) * os.cpu_count()
            per_worker = {}
            for s, w in wstats.items():
                T = w["timing_ms"]
                busy_w = wall * 1000 - T["idle_ms"]
                per_worker[s] = dict(identity=streams[s], frames=T["frames"], fps=T["frames"] / wall,
                                     idle_frac=T["idle_ms"] / (wall * 1000), timing_ms=T,
                                     per_frame_ms={k: T[k] / max(1, T["frames"]) for k in
                                                   ("tracking_ipc_ms", "facemesh_ms", "compose_ms", "filter_ms", "hash_ms",
                                                    "copy_ms", "reset_ms", "compare_ms", "encode_queue_ms")},
                                     busy_ms=busy_w, worker_cpu_s=w["worker_cpu_s"], facemesh_cpu_s=w["facemesh_cpu_s"],
                                     arm_reset_ms=w["arm_reset_ms"], first_recv_after_t0_ms=(w["t_first_recv"] - t0) * 1000,
                                     done_after_t0_s=w["t_last_compose"] - t0, rss_mib=w["rss_mib"],
                                     facemesh_rss_mib=w["facemesh_rss_mib"], compare_accepted=w["compare_accepted"],
                                     tracking_overlap=w["tracking_overlap"], timing_semantics=w["timing_semantics"],
                                     clips=w["clips"])
                if w["tracking_overlap"]:
                    per_worker[s]["per_frame_ms"].update({k: T[k] / max(1, T["frames"]) for k in
                                                       ("tracking_overlap_wait_ms", "tracking_overlap_submit_ms")})
            agg = {k: sum(pw["timing_ms"][k] for pw in per_worker.values()) for k in
                   ("tracking_ipc_ms", "facemesh_ms", "compose_ms", "filter_ms", "hash_ms", "copy_ms", "reset_ms", "idle_ms")}
            r = dict(
                repeat=rep, frames=frames, wall_s=wall, aggregate_fps=frames / wall,
                gpu_issue_done_s=gstats["t_gpu_done"] - t0, all_reported_s=t_all - t0,
                timed_ge_min=wall >= a.min_timed_s,
                gpu=dict(jobs=gstats["jobs"], partial_jobs=gstats["partial_jobs"], chunks=gstats["chunks"],
                         busy_frac_events=gstats["gpu_job_ms"] / (wall * 1000),
                         ms_per_job=dict(total=gstats["gpu_job_ms"] / max(1, gstats["jobs"]),
                                         unet=gstats["gpu_unet_ms"] / max(1, gstats["jobs"]),
                                         decode_post_d2h=gstats["gpu_decode_post_d2h_ms"] / max(1, gstats["jobs"])),
                         ms_per_frame=gstats["gpu_job_ms"] / frames, smi=smi_stats),
                gpu_thread=dict(submit_host_ms=gstats["submit_host_ms"], sync_wait_ms=gstats["sync_wait_ms"],
                                deliver_host_ms=gstats["deliver_host_ms"], credit_wait_ms=gstats["credit_wait_ms"],
                                credit_wait_frac=gstats["credit_wait_ms"] / (wall * 1000),
                                min_free_ring_slots_seen=gstats["min_free_slots_seen"]),
                workers_aggregate_ms=agg,
                workers_mean_idle_frac=statistics.mean(pw["idle_frac"] for pw in per_worker.values()),
                cores=dict(parent=parent_cpu / wall, workers=worker_cpu / wall, facemesh=fm_cpu / wall,
                           harness_total=(parent_cpu + worker_cpu + fm_cpu) / wall, system_busy=sys_busy),
                mem_available_gb=mem_stats,
                per_worker=per_worker,
            )
            reps.append(r)
            print(f"REPEAT {rep} {a.label}: {frames} frames in {wall:.2f}s = {frames / wall:.1f} fps "
                  f"(gpu busy {r['gpu']['busy_frac_events']:.3f}, worker idle {r['workers_mean_idle_frac']:.3f}, "
                  f"cores {r['cores']['harness_total']:.1f})", flush=True)
        record["repeats"] = reps
        for s in range(a.streams):
            conns[s].send(("stop",))
        try:
            col.wait_for("bye", range(a.streams), timeout=60)
        except Exception as exc:  # pragma: no cover
            record["stop_warning"] = repr(exc)
        record["facemesh_pids"] = facemesh_pids
        return record
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for shm in rings:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        import shutil

        shutil.rmtree(arena_root, ignore_errors=True)


def evaluate(a, record) -> dict:
    """Per-stream, per-clip hashes vs the accepted render.json, plus cross-clip determinism."""
    by_ident = {}
    rows = []
    exact = True
    for r in record.get("repeats", []):
        for s, pw in r["per_worker"].items():
            ref = paths.accepted_render(pw["identity"])
            for c in pw["clips"]:
                raw_ok = c["raw_refined_sha256"] == ref["raw_refined_sha256"]
                faces_ok = c["generated_faces_sha256"] == ref["generated_faces_sha256"]
                exact = exact and raw_ok and faces_ok
                rows.append(dict(repeat=r["repeat"], stream=int(s), identity=pw["identity"], loop=c["loop"],
                                 raw_match=raw_ok, faces_match=faces_ok))
                by_ident.setdefault(pw["identity"], set()).add((c["raw_refined_sha256"], c["generated_faces_sha256"]))
    n_clips = len(rows)
    return dict(
        clips_checked=n_clips,
        clips_expected=a.repeats * a.streams * a.loops,
        all_clips_match_accepted=bool(exact and n_clips == a.repeats * a.streams * a.loops),
        clips_matching_raw=sum(x["raw_match"] for x in rows), clips_matching_faces=sum(x["faces_match"] for x in rows),
        deterministic_per_identity={k: len(v) == 1 for k, v in by_ident.items()},
        distinct_hash_pairs_per_identity={k: sorted(v) for k, v in by_ident.items()},
        per_stream_first_clip=[x for x in rows if x["repeat"] == 0 and x["loop"] == 0],
    )


def verify_videos(run_dir: Path) -> list:
    import subprocess

    out = []
    for f in sorted(run_dir.glob("*.mp4")):
        try:
            n = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                         "stream=nb_read_frames", "-of", "csv=p=0", str(f)], text=True).strip()
        except subprocess.CalledProcessError as exc:
            n = f"error {exc.returncode}"
        out.append(dict(file=f.name, frames=n, bytes=f.stat().st_size))
    return out


def main(argv=None):
    a = parse_args(argv)
    out_root = Path(a.out_root)
    run_dir = out_root / a.label
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    integrity = paths.code_integrity()
    if not integrity["matches_accepted_render_json"]:
        raise SystemExit(f"accepted recipe files changed: {integrity}")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = dict(schema="chin_multistream_v1", label=a.label, started_utc=started, tags="[M] measured by this run",
               args={k: v for k, v in vars(a).items()}, code_integrity=integrity,
               harness_code_sha256={p.name: paths.sha256_file(p) for p in
                                    [Path(__file__).resolve(), *sorted((paths.WORKTREE / "scripts" / "chin_multistream").glob("*.py"))]})
    status = "error"
    try:
        if a.mode == "serial":
            import cv2
            import torch
            from chin_multistream import gpu, serial

            torch.set_num_threads(4)
            cv2.setNumThreads(2)
            preset = gpu.PRESETS[a.backend]
            unet, decoder, sf, env_record, load_s = gpu.setup_backends({**preset["env"], **a.extra_flags})
            rec.update(env=env_record, load_s=load_s, backends=gpu.describe_backends(unet, decoder))
            res = serial.run_serial(unet, decoder, sf, a.identity_list, a.repeats, run_dir / "logs")
            rec["serial"] = res
            allfps = [x["warm_render_fps"] for v in res.values() for x in v["runs"]]
            rec["summary"] = dict(
                median_fps_over_identities=median([v["median_fps"] for v in res.values()]),
                median_fps_all_runs=median(allfps), min_fps=min(allfps), max_fps=max(allfps),
                accepted_warm_render_fps={k: v["accepted_warm_render_fps"] for k, v in res.items()},
                all_runs_match_accepted=all(x["raw_match"] and x["faces_match"] for v in res.values() for x in v["runs"]))
            import numpy as np
            rec["versions"] = dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__, python=sys.version.split()[0])
        else:
            rec.update(run_multi(a, run_dir))
            fps = [r["aggregate_fps"] for r in rec["repeats"]]
            rec["exactness"] = evaluate(a, rec)
            rec["summary"] = dict(
                streams=a.streams, loops=a.loops, repeats=a.repeats, backend=a.backend, pack=a.pack, decode_split=a.decode_split,
                aggregate_fps_per_repeat=fps, median_aggregate_fps=median(fps),
                wall_s_per_repeat=[r["wall_s"] for r in rec["repeats"]],
                all_repeats_ge_min_timed_s=all(r["timed_ge_min"] for r in rec["repeats"]),
                gpu_busy_frac_median=median([r["gpu"]["busy_frac_events"] for r in rec["repeats"]]),
                gpu_ms_per_frame_median=median([r["gpu"]["ms_per_frame"] for r in rec["repeats"]]),
                worker_idle_frac_median=median([r["workers_mean_idle_frac"] for r in rec["repeats"]]),
                harness_cores_median=median([r["cores"]["harness_total"] for r in rec["repeats"]]),
                all_clips_match_accepted=rec["exactness"]["all_clips_match_accepted"],
                deterministic_per_identity=all(rec["exactness"]["deterministic_per_identity"].values()))
            import numpy as np
            import torch
            import cv2
            rec["versions"] = dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__, python=sys.version.split()[0],
                                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None)
            if a.encode:
                rec["videos"] = verify_videos(run_dir)
        status = "complete"
    except BaseException:
        rec["error"] = traceback.format_exc()
        raise
    finally:
        rec["status"] = status
        rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out_root / f"{a.label}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        s = rec.get("summary", {})
        print("SUMMARY", a.label, status, json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)}, default=str), flush=True)


if __name__ == "__main__":
    main()
