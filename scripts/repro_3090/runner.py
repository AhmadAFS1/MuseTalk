#!/usr/bin/env python3
"""Explicit wrappers around canonical r5 GPU, aggregate, quality and local-live tools."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import report as checks

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def capture(command, env=None):
    return subprocess.check_output(command, cwd=ROOT, env=env, text=True, stderr=subprocess.PIPE).strip()


def clean_environment(ambient, values):
    """Explicit profile wins; shell leftovers must not redirect a nested harness."""
    prefixes = ("MUSETALK_", "HLS_", "WEBRTC_", "REPRO_", "LIVE15_", "OMP_", "MKL_", "OPENBLAS_",
                "TORCHINDUCTOR_", "TRITON_", "PYTORCH_")
    removed = sorted(k for k in ambient if k.startswith(prefixes))
    return {**{k: v for k, v in ambient.items() if k not in removed}, **values}, removed


def gpu_roots(candidate, comparisons, baseline_only):
    roots = [str(Path(p).resolve()) for p in (candidate, *comparisons)]
    checks.require(len(roots) == len(set(roots)), "comparison roots must be distinct from candidate and one another")
    checks.require(not (baseline_only and comparisons), "baseline-only cannot include comparison roots")
    checks.require(baseline_only or comparisons, "GPU diagnostic requires comparison roots or explicit --baseline-only")
    return roots


def verify_s3_objects(objects):
    for item in objects:
        result = json.loads(capture(["aws", "s3api", "head-object", "--bucket", item["bucket"], "--key", item["key"], "--output", "json"]))
        checks.require(result["ContentLength"] == item["bytes"], f"S3 object size mismatch: {item['key']}")


def profile(path):
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        checks.require(sep and re.fullmatch(r"(?:MUSETALK|HLS|WEBRTC|HF|TRANSFORMERS|OMP|MKL|OPENBLAS)_[A-Z0-9_]+", key),
                       f"unsafe profile key at line {number}")
        checks.require(not re.search(r"TOKEN|SECRET|PASSWORD|ACCESS_KEY|CREDENTIAL|AUTH", key), "secrets forbidden in benchmark profiles")
        checks.require(not any(c in value for c in ("$", "`", "\n")), "profiles must use literal values, no shell expansion")
        checks.require(key not in values, f"duplicate profile key: {key}")
        values[key] = value.strip('"').strip("'")
    return values


def engine(root):
    root = Path(root).resolve()
    manifest_path = root / "bs16/manifest.json"
    manifest = checks.read(manifest_path)
    checks.require(manifest.get("complete") is True and manifest.get("batch") == 16, "engine set incomplete/wrong batch")
    checks.require(manifest.get("variant") == "srccache", "not the canonical source-cache recipe")
    required = {b["name"] for b in manifest["spec"]} | {"prefix"}
    checks.require(required <= set(manifest["blocks"]), "missing engine blocks")
    hashes = {}
    for name in required:
        row = manifest["blocks"][name]
        file = root / "bs16" / row["engine_file"]
        digest = checks.sha256(file)
        checks.require(digest == row.get("engine_sha256"), f"engine hash mismatch: {name}")
        hashes[name] = digest
    for key in ("graph_equals_direct_enqueue", "deterministic_run_to_run"):
        checks.require(manifest.get("probe", {}).get(key) is True, f"missing/failed engine invariant: {key}")
    return {"root": str(root), "manifest_sha256": checks.sha256(manifest_path), "plan_sha256": hashes, "manifest": manifest}


def preflight(args, env, child_env=None):
    smi_fields = "name,uuid,compute_cap,memory.total,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu"
    rows = list(csv.reader(capture(["nvidia-smi", f"--query-gpu={smi_fields}", "--format=csv,noheader,nounits"]).splitlines()))
    checks.require(len(rows) == 1, "exactly one visible physical GPU required")
    values = [s.strip() for s in rows[0]]
    gpu = dict(zip(smi_fields.split(","), values))
    actual = checks.gpu_identity(gpu["name"], gpu["compute_cap"], args.general_gpu)
    apps = capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    checks.require(not apps, "foreign GPU workload detected; isolate/drain owned server first")
    runtime = json.loads(capture(["bash", "scripts/box_guard.sh", "run", "--wait-min", "0", "--min-avail-gb", "3", "--label", "3090_preflight", "--",
                                 args.python, "-c", "import json,torch,tensorrt,torch_tensorrt; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,tensorrt=tensorrt.__version__,torch_tensorrt=torch_tensorrt.__version__,visible_vram_bytes=torch.cuda.get_device_properties(0).total_memory,gpu=torch.cuda.get_device_name(0),compute_capability='.'.join(map(str,torch.cuda.get_device_capability(0))))))"], env=child_env))
    checks.require(runtime["gpu"] == gpu["name"], "CUDA/NVML device selection mismatch")
    checks.require(runtime["torch"] == "2.5.1+cu121" and runtime["tensorrt"].startswith("10.3.")
                   and runtime["torch_tensorrt"].startswith("2.5."), "runtime differs from pinned r5 build matrix")
    engines = [engine(args.engine_root)]
    engines += [engine(p) for p in args.comparison_root]
    expected_compat = "ampere_plus" if args.target == "portable" else "none"
    checks.require((engines[0]["manifest"].get("hardware_compatibility_level") or "none") == expected_compat,
                   "candidate native/portable label does not match manifest")
    for item in engines:
        manifest = item["manifest"]
        compat = manifest.get("hardware_compatibility_level") or "none"
        checks.require(compat in ("none", "ampere_plus"), "unknown engine compatibility")
        if compat == "none":
            checks.require(str(manifest.get("compute_capability")).replace(".", "").replace(",", "").replace(" ", "").strip("[]()") == gpu["compute_cap"].replace(".", ""), "engine GPU architecture mismatch")
            checks.require(manifest.get("gpu") == gpu["name"], "native engine GPU model mismatch")
    meta_path = Path(args.taesd_dir) / f"taesd_trt_{args.taesd_key}.json"
    meta = checks.read(meta_path)
    checks.require(meta.get("key") == args.taesd_key, "TAESD identity mismatch")
    fp = meta["fingerprint"]
    checks.require(hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:20] == args.taesd_key, "TAESD fingerprint/key mismatch")
    checks.require(fp["batch"] == 8 and fp["tensorrt"] == runtime["tensorrt"], "TAESD runtime/batch mismatch")
    checks.require(fp.get("hardware_compatibility_level", "none") == env.get("MUSETALK_TAESD_TRT_HW_COMPAT", "none"), "TAESD hardware profile mismatch")
    checks.require(fp["opt_level"] == int(env.get("MUSETALK_TAESD_TRT_OPT_LEVEL", "3")), "TAESD optimization knob was not used by this engine")
    if fp.get("hardware_compatibility_level", "none") == "none":
        checks.require(fp["gpu"] == gpu["name"] and fp["compute_capability"] == gpu["compute_cap"], "TAESD target GPU mismatch")
    for kind in ("decoder", "post"):
        checks.require(checks.sha256(Path(args.taesd_dir) / meta[f"{kind}_plan"]) == meta[f"{kind}_plan_sha256"], f"corrupt TAESD {kind}")
    inputs = checks.read(args.input_manifest)
    base = Path(args.input_manifest).resolve().parent
    checks.verify_files(inputs, base)
    paths = {str((base / row["path"]).resolve()) for row in inputs["files"]}
    for file in (ROOT / "models/musetalkV15/unet.pth", ROOT / "models/musetalkV15/musetalk.json", ROOT / "musetalk/utils/blending.py",
                 *sorted((ROOT / "character_factory/h3_avatar_workflow").glob("*.py"))):
        checks.require(str(file.resolve()) in paths, f"unhashed model/quality-critical source: {file}")
    for split, count in (("", 352), ("holdout", 96)):
        captures = list((Path(args.corpus) / split).glob("unet_io_*.pt"))
        checks.require(len(captures) == count, f"expected {count} canonical {split or 'main'} captures")
        checks.require(all(str(p.resolve()) in paths for p in captures), "unhashed corpus input")
    for ident in checks.IDENTITIES:
        files = [p for p in (Path(args.accepted_root) / ident).rglob("*") if p.is_file()]
        checks.require(bool(files) and all(str(p.resolve()) in paths for p in files), f"unhashed/missing canonical fixture: {ident}")
    # Optional live S3 HEAD checks use existing credentials, never log the environment.
    verify_s3_objects(inputs.get("s3_objects", []))
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            memory[line.split(":")[0]] = int(line.split()[1]) * 1024
    cgroup = {}
    for name in ("memory.max", "memory.current", "cpu.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        cgroup[name] = path.read_text().strip() if path.exists() else "unavailable"
    return {**actual, "gpu": gpu, "runtime": runtime, "engines": engines,
            "taesd": {"key": args.taesd_key, "meta_sha256": checks.sha256(meta_path), "decoder_plan_sha256": meta["decoder_plan_sha256"], "fingerprint": meta["fingerprint"]},
            "input_manifest_sha256": checks.sha256(args.input_manifest), "input_count": len(inputs["files"]),
            "profile_sha256": checks.sha256(args.profile), "effective_profile": env,
            "git_revision": capture(["git", "rev-parse", "HEAD"]), "git_status": capture(["git", "status", "--porcelain"]),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "memory": memory, "cgroup": cgroup,
            "disk_free_bytes": shutil.disk_usage(ROOT).free, "shm": dict(zip(("total", "used", "free"), shutil.disk_usage("/dev/shm"))),
            "foreign_gpu_apps_at_preflight": apps, "utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def child(args, out, env, label, command, gb=12):
    log = out / f"{label}.log"
    command = ["bash", "scripts/box_guard.sh", "run", "--wait-min", "0", "--min-avail-gb", str(gb),
               "--label", f"3090_{args.label}_{label}", "--", sys.executable, str(HERE / "watch.py"),
               "--out", str(out / f"{label}.gpu_watch.jsonl"), "--", *map(str, command)]
    start = time.monotonic()
    with log.open("w") as stream:
        run = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {"label": label, "returncode": run.returncode, "wall_s": time.monotonic() - start,
              "log": str(log), "command": command}
    (out / f"{label}.child.json").write_text(json.dumps(record, indent=2) + "\n")
    return run.returncode


def quality(args, out, env):
    results = []
    for split in ("main", "holdout"):
        path = out / f"unet_{split}.json"
        corpus = Path(args.corpus) / ("holdout" if split == "holdout" else "")
        rc = child(args, out, env, f"unet_{split}", [args.python, "scripts/validate_unet_backend.py", "--backend", "runtime", "--capture-dir", corpus,
                   "--padded-batch-size", "8", "--group-captures", "2", "--limit", "0", "--warmup", "1", "--iters", "2",
                   "--fail-mae", "0.01", "--fail-max-abs", "0.5", "--report-path", path], 8)
        data = checks.read(path)
        summary = data.get("summary", data)
        checks.require(summary["files"] == (176 if split == "main" else 48), f"incomplete UNet {split} coverage")
        passed = summary["mae_max"] <= .01 and summary["max_abs_max"] <= .5
        checks.require(rc == (0 if passed else 1), f"UNet {split} child exit does not match numerical verdict")
        results.append({"name": f"unet_{split}", "status": "PASS" if passed else "FAIL", "metrics": summary})
    path = out / "srccache.json"
    rc = child(args, out, env, "srccache", [args.python, "scripts/repro_400fps/srccache_exact.py", "--root", args.engine_root,
               "--baseline-root", str(out / "intentionally_no_sm89_baseline"), "--corpus", args.corpus, "--out", path], 8)
    data = checks.child_result(rc, [path])[0]
    checks.require(data["cached_equals_forward"] is True and data["permuted_rows_exact"] is True, "source-prefix invariant failed")
    results.append({"name": "source_prefix", "status": "PASS", "metrics": data})
    qenv = {**env, "REPRO_GATE_OUT": str(out / "taesd")}
    rc = child(args, out, qenv, "taesd", [args.python, "scripts/repro_400fps/gate_taesd_trt.py", "--no-record", "--corpus", args.corpus], 8)
    data = checks.read(out / "taesd/gate_taesd_trt.json")
    checks.require(data["engine"]["key"] == args.taesd_key, "quality gate loaded wrong TAESD")
    checks.require(data["gate"]["bit_exact_checks"] == "PASS", "TAESD bit-exact invariant failed")
    passed = data["gate"]["verdict"] == "PASS"
    checks.require(rc == (0 if passed else 1), "TAESD child exit does not match numerical verdict")
    results.append({"name": "taesd", "status": "PASS" if passed else "FAIL", "metrics": data["gate"]})
    # Untimed captures are intentionally separate from aggregate throughput reports.
    label = f"{args.label}_quality_capture"
    rc = child(args, out, env, "quality_capture", [args.python, "scripts/chin_multistream_render.py", "--backend", "stagewise16_taesdtrt",
               "--streams", "6", "--loops", "1", "--repeats", "1", "--save-arrays", "--encode", "--compare-accepted",
               "--out-root", out, "--label", label], 12)
    data = checks.child_result(rc, [out / f"{label}.json"])[0]
    checks.require(data["status"] == "complete" and data["summary"]["deterministic_per_identity"], "quality capture incomplete")
    checks.require(data["code_integrity"]["matches_accepted_render_json"] is True, "canonical quality composition changed")
    meta = checks.read(Path(args.taesd_dir) / f"taesd_trt_{args.taesd_key}.json")
    checks.loaded_backend(data, args.engine_root, args.taesd_key, meta["decoder_plan_sha256"], engine(args.engine_root)["manifest"])
    faces = list((out / label).glob("stream*_faces.npz"))
    checks.require(len(faces) == 6, "missing canonical quality capture")
    for ident in checks.IDENTITIES:
        matches = [f for f in faces if f.name.endswith(f"_{ident}_faces.npz")]
        checks.require(len(matches) == 1, f"missing quality capture: {ident}")
        run_name = f"{ident}__{args.label}"
        rc = child(args, out, env, run_name, [args.python, "scripts/quality_ab_metrics.py", "pair", "--identity-dir", Path(args.accepted_root) / ident,
                   "--a", f"dir={Path(args.accepted_root) / ident}", "--b", f"faces={matches[0]},label={args.label}",
                   "--profile", "e1", "--name", run_name, "--out-dir", out / "quality_metrics"], 8)
        metrics = checks.read(out / "quality_metrics" / f"{run_name}.json")
        verdict = metrics["verdict"]
        checks.require(verdict in ("PASS", "FAIL"), f"quality metrics incomplete: {ident}")
        checks.require(rc == (0 if verdict == "PASS" else 1), f"quality child failure: {ident}")
        results.append({"name": f"canonical_pixels_{ident}", "status": verdict, "report": str(out / "quality_metrics" / f"{run_name}.json")})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=("check", "gpu", "aggregate", "quality", "live"))
    for name in ("profile", "engine-root", "taesd-key", "taesd-dir", "input-manifest", "out", "label"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--comparison-root", action="append", default=[])
    p.add_argument("--baseline-only", action="store_true", help="GPU suite: one engine's blocks and >=2 180-second runs; no comparative claim")
    p.add_argument("--python", default="/workspace/.venvs/musetalk_trt_stagewise/bin/python")
    p.add_argument("--corpus", default=str(ROOT / "calibration/unet_multi_avatar_20260928"))
    p.add_argument("--accepted-root", default="/workspace/experiments/avatar_diversity_20260927")
    p.add_argument("--workspace", default="/workspace")
    p.add_argument("--general-gpu", action="store_true", help="label actual GPU; never imply RTX 3090 validation")
    p.add_argument("--target", choices=("portable", "native"), default="native")
    p.add_argument("--stages", nargs="+", choices=("T", "SUST", "N15"), default=["T", "SUST"])
    p.add_argument("--loops", type=int, default=24, help="increase if a window is <60s; never relax duration")
    p.add_argument("--gpu-repeats", type=int, default=2)
    p.add_argument("--thermal-warmup-s", type=int, default=120)
    p.add_argument("--live-avatar-file")
    p.add_argument("--live-cpus", help="available CPU IDs; local clients share these CPUs and contention is disclosed")
    p.add_argument("--live-stages", default="s0,ramp,soak")
    p.add_argument("--live-levels", default="5 10 15")
    p.add_argument("--live-soak-n", type=int, default=15)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        p.error("unsafe run label")
    out = Path(args.out).resolve() / f"{args.label}_{args.suite}"
    # Never accept stale reports from an earlier attempt.
    if out.exists():
        p.error(f"output exists; choose a fresh label: {out}")
    out.mkdir(parents=True)
    result = {"schema": "repro_3090_v1", "suite": args.suite, "label": args.label,
              "measurement_class": "measured_on_current_host", "results": [], "status": "INVALID"}
    try:
        args.engine_root = str(Path(args.engine_root).resolve())
        args.comparison_root = [str(Path(x).resolve()) for x in args.comparison_root]
        checks.require(not args.baseline_only or args.suite == "gpu", "baseline-only applies only to GPU diagnostic")
        if args.suite == "gpu":
            gpu_roots(args.engine_root, args.comparison_root, args.baseline_only)
        values = profile(args.profile)
        values.update(MUSETALK_UNET_BACKEND="trt_stagewise", MUSETALK_UNET_STAGEWISE_BATCH="16",
                      MUSETALK_UNET_STAGEWISE_CACHE_DIR=args.engine_root, MUSETALK_TRT_FALLBACK="0",
                      MUSETALK_UNET_STAGEWISE_VERIFY_SHA="1", MUSETALK_UNET_STAGEWISE_PROBE_CHECK="1",
                      MUSETALK_UNET_STAGEWISE_PROBE_TOL="0",
                      MUSETALK_VAE_BACKEND="taesd", MUSETALK_TAESD_BACKEND="trt", MUSETALK_TAESD_TRT_BUILD="0",
                      MUSETALK_TAESD_TRT_STRICT="1", MUSETALK_TAESD_TRT_BATCH="8", MUSETALK_TAESD_WARMUP_BATCHES="8",
                      MUSETALK_TAESD_TRT_DIR=str(Path(args.taesd_dir).resolve()))
        effective = out / "effective.env"
        effective.write_text("".join(f"{k}={v}\n" for k, v in sorted(values.items())))
        # Ambient tuning cannot silently beat the explicit profile. Retain credentials for optional S3 HEAD only.
        env, removed = clean_environment(os.environ, values)
        env.update(MUSETALK_REPRO_RUNTIME_ENV=str(effective), MUSETALK_REPRO_ACCEPTED=str(Path(args.accepted_root).resolve()),
                   MUSETALK_REPRO_WORKSPACE=str(Path(args.workspace).resolve()))
        result["environment"] = preflight(args, values, env)
        result["environment"]["discarded_ambient_keys"] = removed
        (out / "environment.json").write_text(json.dumps(result["environment"], indent=2) + "\n")
        def loaded(data):
            checks.loaded_backend(data, args.engine_root, args.taesd_key, result["environment"]["taesd"]["decoder_plan_sha256"],
                                  result["environment"]["engines"][0]["manifest"])
        if args.suite == "check":
            result["results"].append({"status": "PASS", "scope": "preflight_only_not_performance"})
        elif args.suite == "gpu":
            checks.require(args.gpu_repeats >= 2, "at least two GPU repetitions required")
            result["comparison_claim"] = "none: single-engine diagnostic baseline" if args.baseline_only else "interleaved synthetic block timings only"
            path = out / "blocks.json"
            roots = gpu_roots(args.engine_root, args.comparison_root, args.baseline_only)
            command = [args.python, "scripts/bench_stagewise_blocks.py", "--batch", "16", "--rounds", "9", "--out", path]
            for root in roots:
                command += ["--root", root]
            data = checks.child_result(child(args, out, env, "blocks", command, 8), [path])[0]
            result["results"].append(checks.blocks(data, {x["root"]: x["manifest"] for x in result["environment"]["engines"]}))
            for i in range(args.gpu_repeats):
                path = out / f"gpu_{i}.json"
                rc = child(args, out, env, f"gpu_{i}", [args.python, "scripts/bench_gpu_path.py", "--no-live-env", "--overlay", effective,
                           "--batch", "16", "--seconds", "180", "--warmup", "20", "--window-s", "10", "--stage-sync", "off",
                           "--capture-dir", args.corpus, "--out", path, "--label", args.label], 12)
                data = checks.child_result(rc, [path])[0]
                loaded(data)
                result["results"].append(checks.gpu_path(data))
        elif args.suite == "aggregate":
            for stage in args.stages:
                n, repeats = {"T": (6, 2), "SUST": (6, 5), "N15": (15, 10)}[stage]
                label = f"{args.label}_{stage}"
                path = out / f"{label}.json"
                rc = child(args, out, env, stage, [args.python, "scripts/chin_multistream_render.py", "--backend", "stagewise16_taesdtrt",
                           "--streams", str(n), "--loops", str(args.loops if n == 6 else max(10, args.loops // 2)), "--repeats", str(repeats),
                           "--min-timed-s", "60", "--thermal-warmup-s", str(args.thermal_warmup_s), "--out-root", out, "--label", label], 14 if n == 15 else 12)
                data = checks.child_result(rc, [path])[0]
                loaded(data)
                threshold = None if stage == "N15" else (300 if args.target == "portable" else 400)
                result["results"].append(checks.aggregate(data, stage, threshold))
        elif args.suite == "quality":
            result["results"] = quality(args, out, env)
            result["strict_original_gates"] = checks.combined_status(result["results"])
            result["quality_parity_with_reference"] = "NOT_EVALUATED"
            result["visual_inspection"] = "NOT_EVALUATED"
            result["release_quality_decision"] = "incomplete"
        elif args.suite == "live":
            checks.require(args.live_avatar_file and args.live_cpus, "live requires explicit avatar file and CPU set")
            avatar_path = Path(args.live_avatar_file).resolve()
            checks.require(avatar_path.is_file(), "missing live avatar list")
            allowed = set(os.sched_getaffinity(0))
            selected = set()
            for part in args.live_cpus.split(","):
                lo, _, hi = part.partition("-")
                selected.update(range(int(lo), int(hi or lo) + 1))
            checks.require(selected and selected <= allowed, "live CPU set outside allocation")
            levels = [int(n) for n in args.live_levels.split()]
            checks.require(levels == sorted(set(levels)) and levels[0] >= 1, "invalid live ramp order")
            checks.require(args.live_soak_n in levels, "soak capacity must be tested in ramp")
            live_env = {**env, "LIVE15_PYTHON": args.python, "LIVE15_AVATARS_FILE": str(avatar_path),
                        "LIVE15_VENV": str(Path(args.python).absolute().parent.parent),
                        "LIVE15_SERVER_CPUS": args.live_cpus, "LIVE15_CLIENT_CPUS": args.live_cpus,
                        "LIVE15_NO_PIN": "1", "LIVE15_SOAK_N": str(args.live_soak_n), "LIVE15_SOAK_RECORD": "",
                        "LIVE15_SOAK_SECONDS": "3630", "LIVE15_SOAK_MIN_STEADY": "3600"}
            overrides = ":".join(map(str, (effective, ROOT / "experiments/live15_r5/loopfix.env", ROOT / "experiments/live15_r5/serve.env", ROOT / "experiments/live15_r5/common.env")))
            rc = child(args, out, live_env, "live", ["bash", "experiments/live15_r5/run_live15.sh", args.label, out / "live", overrides, args.live_stages, args.live_levels], 14)
            checks.require(rc == 0, f"live child failed: {rc}")
            startup_log = (out / "live/api_server_8300.log").read_text()
            checks.require(f"TAESD TRT backend: key={args.taesd_key}" in startup_log, "live loaded TAESD identity not verified")
            checks.require(args.engine_root in startup_log, "live loaded UNet root not verified")
            for stage in args.live_stages.split(","):
                expected = [1, 3] if stage == "s0" else levels if stage == "ramp" else [args.live_soak_n]
                for n in expected:
                    data = checks.read(out / "live" / f"{stage}_n{n}_trace.json")
                    result["results"].append(checks.live(data, n, {"s0": 10, "ramp": 240, "soak": 3600}[stage]))
            result["live_network"] = "loopback; clients co-resident and sharing disclosed CPU set"
            result["deployment_acceptance"] = "NOT_EVALUATED: requires external real-browser EC2/TURN call"
        for prior in result["environment"]["engines"]:
            current = engine(prior["root"])
            checks.require(current["plan_sha256"] == prior["plan_sha256"]
                           and current["manifest_sha256"] == prior["manifest_sha256"], "engine changed during measurement")
        result["status"] = checks.combined_status(result["results"])
    except (checks.Invalid, KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
        result["status"] = "INVALID"
        # subprocess stderr can contain credential material; record type/command basename only.
        result["reason"] = type(exc).__name__ if isinstance(exc, subprocess.SubprocessError) else str(exc)
    finally:
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(out / "report.json"), "reason": result.get("reason")}))
    return checks.EXIT[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
