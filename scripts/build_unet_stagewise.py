"""Build-if-missing tool for the stagewise FP16 TensorRT UNet engines (plan items 2.2 / 2.3a / 2.3b).

For each engine batch (16 and/or 8) it exports the 11 top-level UNet blocks to ONNX in RAM, builds
each one with the TensorRT ONNX parser (FP16, builder optimisation level --opt-level, shared timing
cache), and writes <root>/bs<N>/<block>.plan plus manifest.json (TensorRT version, GPU, per-block ONNX
hash, build flags, engine hash, build time, host RSS). A block whose engine exists with the same ONNX
hash and build flags is skipped, so the tool resumes and can be split into short leased runs
(--max-minutes). When all 11 engines exist it finalises the set: loads it through
StagewiseTrtUnetBackend, checks CUDA-graph replay == direct enqueue (bit-exact) and run-to-run
determinism, compares the chain with the eager PyTorch UNet on a deterministic probe batch, and
records the probe output hash that the runtime checks at load.

--second-build builds each block again in RAM with the timing cache disabled (fresh tactic timing),
times both engines on the block's real inputs (interleaved CUDA-graph replays) and keeps the faster
(tactic variance ~5%); only the kept engine is written.

Run under the GPU lease, e.g.:
  scripts/box_guard.sh run --min-avail-gb 8 -- /workspace/.venvs/musetalk_trt_stagewise/bin/python \
      scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --max-minutes 20
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import unet_stagewise_trt as sw  # noqa: E402

SEED_TIMING_CACHE = ROOT / "docs/fps_comparisons/4070s_300fps_20260927/unet_probe/tt16_timing_cache.bin"


def proc_mem() -> dict:
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0] + "_MiB"] = round(int(line.split()[1]) / 1024, 1)
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                out["MemAvailable_MiB"] = round(int(line.split()[1]) / 1024, 1)
    return out


def load_eager_unet(device: torch.device):
    """Same weights as the live eager UNet (fp32 checkpoint -> fp16, round-to-nearest-even), low host RAM.

    Identical to scripts/build_unet_multi_avatar_corpus.load_eager_unet (the corpus reference)."""
    from diffusers import UNet2DConditionModel

    cfg = json.loads((ROOT / "models/musetalkV15/musetalk.json").read_text())
    model = UNet2DConditionModel(**cfg).half()
    try:
        state = torch.load(ROOT / "models/musetalkV15/unet.pth", map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        state = torch.load(ROOT / "models/musetalkV15/unet.pth", map_location="cpu")
    model.load_state_dict(state)
    del state
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(path)


def time_block_engines(engine_bytes_list, blk, tensors, device, rounds=6, reps=40):
    """Median CUDA-graph replay ms of each candidate engine for one block, interleaved."""
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    chains = []
    for data in engine_bytes_list:
        eng = runtime.deserialize_cuda_engine(data)
        chain = sw.StageChain({blk["name"]: eng}, [blk], device, require_io=False)
        for key in blk["inputs"]:
            chain.buffers[key].copy_(tensors[key])
        chain.capture_graph()
        chains.append((eng, chain))
    times = [[] for _ in chains]
    for _ in range(rounds):
        for i, (_, chain) in enumerate(chains):
            for _ in range(5):
                chain.run()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                chain.run()
            e.record()
            e.synchronize()
            times[i].append(s.elapsed_time(e) / reps)
    outs = [chain.buffers[blk["outputs"][0]].clone() for _, chain in chains]
    del chains
    torch.cuda.synchronize()
    return [sorted(t)[len(t) // 2] for t in times], outs, runtime


def build_batch(args, batch: int, model, device) -> dict:
    root = Path(args.root).resolve()
    engine_dir = root / f"bs{batch}"
    engine_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = engine_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    import tensorrt as trt

    spec = sw.chain_spec(model)
    flags = sw.build_flags_record(args.opt_level, args.workspace_gb, not args.no_timing_cache)
    base = {
        "schema": sw.MANIFEST_SCHEMA,
        "batch": batch,
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "unet_weights": {"path": "models/musetalkV15/unet.pth",
                         "bytes": (ROOT / "models/musetalkV15/unet.pth").stat().st_size},
        "unet_config": "models/musetalkV15/musetalk.json",
        "timestep": 0,
        "spec": spec,
        "build_flags": flags,
    }
    if manifest.get("spec") not in (None, spec) or manifest.get("batch") not in (None, batch):
        raise SystemExit(f"{manifest_path}: existing manifest has a different spec/batch; use a fresh --root")
    blocks = dict(manifest.get("blocks", {}))
    manifest.update(base)
    manifest["blocks"] = blocks
    manifest["complete"] = False

    emb = sw.time_embedding_t0(model, batch, device)
    wrappers = sw.make_block_wrappers(model, emb)
    lat, aud = sw.probe_inputs(batch)
    lat, aud = lat.to(device), aud.to(device)
    tensors, out_chain = sw.trace_block_inputs(model, spec, wrappers, lat, aud)
    with torch.no_grad():
        out_eager = model(lat, torch.tensor([0], device=device), encoder_hidden_states=aud).sample
    manifest["eager_wrapper_chain_vs_forward_max_abs"] = float((out_chain.float() - out_eager.float()).abs().max())

    cache_path = Path(args.timing_cache).resolve() if args.timing_cache else root / "timing_cache.bin"
    cache_bytes = None
    if not args.no_timing_cache:
        if cache_path.exists():
            cache_bytes = cache_path.read_bytes()
        elif SEED_TIMING_CACHE.exists():
            cache_bytes = SEED_TIMING_CACHE.read_bytes()
            manifest["timing_cache_seed"] = str(SEED_TIMING_CACHE.relative_to(ROOT))
    wanted = BLOCKS if not args.blocks else [b for b in sw.BLOCK_ORDER if b in args.blocks.split(",")]
    started = time.time()
    log = manifest.setdefault("build_log", [])
    for blk in spec:
        name = blk["name"]
        if name not in wanted:
            continue
        if (time.time() - started) / 60.0 > args.max_minutes:
            print(f"[bs{batch}] --max-minutes reached; stopping before {name}", flush=True)
            break
        args_t = [tensors[k] for k in blk["inputs"]]
        t0 = time.time()
        onnx_bytes = sw.export_block_onnx(wrappers[name], args_t)
        export_s = time.time() - t0
        onnx_sha = sw.sha256_bytes(onnx_bytes)
        engine_file = f"{name}.plan"
        prev = blocks.get(name, {})
        if (not args.force and prev.get("onnx_sha256") == onnx_sha and prev.get("build_flags") == flags
                and (engine_dir / engine_file).exists()
                and (not args.second_build or prev.get("second_build"))):
            print(f"[bs{batch}] {name}: engine up to date, skip", flush=True)
            del onnx_bytes
            continue
        gc.collect()
        torch.cuda.empty_cache()
        mem0 = proc_mem()
        engine_bytes, build_s, cache_out = sw.build_engine_from_onnx(
            onnx_bytes, opt_level=args.opt_level, workspace_gb=args.workspace_gb,
            timing_cache=cache_bytes, use_timing_cache=not args.no_timing_cache)
        mem1 = proc_mem()
        onnx_mb = len(onnx_bytes) / 2**20
        del onnx_bytes
        if cache_out is not None:
            cache_bytes = cache_out
            write_bytes_atomic(cache_path, cache_bytes)
        entry = {"engine_file": engine_file, "onnx_sha256": onnx_sha, "onnx_mib": onnx_mb,
                 "export_s": export_s, "build_s": build_s, "build_flags": flags,
                 "inputs": blk["inputs"], "outputs": blk["outputs"],
                 "host_mem_before_build": mem0, "host_mem_after_build": mem1}
        if args.second_build:
            onnx_b = sw.export_block_onnx(wrappers[name], args_t)
            eng_b, build_b, _ = sw.build_engine_from_onnx(
                onnx_b, opt_level=args.opt_level, workspace_gb=args.workspace_gb, use_timing_cache=False)
            del onnx_b
            ms, outs, _rt = time_block_engines([engine_bytes, eng_b], blk, tensors, device)
            keep_b = ms[1] < ms[0] * 0.995
            entry["second_build"] = {"first_ms": ms[0], "second_ms": ms[1], "second_build_s": build_b,
                                     "kept": "second" if keep_b else "first",
                                     "out0_max_abs_first_vs_second": float((outs[0].float() - outs[1].float()).abs().max())}
            if keep_b:
                engine_bytes = eng_b
            del eng_b, outs
        write_bytes_atomic(engine_dir / engine_file, engine_bytes)
        entry["engine_sha256"] = sw.sha256_bytes(engine_bytes)
        entry["engine_mib"] = len(engine_bytes) / 2**20
        del engine_bytes
        gc.collect()
        blocks[name] = entry
        log.append({"block": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "build_s": build_s, "VmHWM_MiB": mem1.get("VmHWM_MiB"), "MemAvailable_MiB": mem1.get("MemAvailable_MiB")})
        write_json_atomic(manifest_path, manifest)
        print(f"[bs{batch}] {name}: built in {build_s:.1f}s (onnx {onnx_mb:.1f} MiB, engine "
              f"{entry['engine_mib']:.1f} MiB) {entry.get('second_build', '')} mem={mem1}", flush=True)

    missing = [b for b in sw.BLOCK_ORDER if b not in blocks or not (engine_dir / blocks[b]["engine_file"]).exists()]
    manifest["missing_blocks"] = missing
    if missing:
        write_json_atomic(manifest_path, manifest)
        print(f"[bs{batch}] incomplete; missing {missing}", flush=True)
        return manifest
    del wrappers
    gc.collect()
    torch.cuda.empty_cache()

    # ---- finalise: load through the runtime backend, probe hash, graph/determinism checks
    write_json_atomic(manifest_path, manifest)
    backend = sw.StagewiseTrtUnetBackend.load(engine_dir, device=device, use_graph=True, probe_check=False,
                                              require_complete=False)
    out1 = backend.run_probe().clone()
    out2 = backend.run_probe().clone()
    backend.use_graph = False
    backend._chain.graph, graph = None, backend._chain.graph
    out_direct = backend.run_probe().clone()
    backend._chain.graph = graph
    backend.use_graph = True
    torch.cuda.synchronize()
    d = (out1.float() - out_eager.float()).abs()
    rel = float((out1.float() - out_eager.float()).norm() / out_eager.float().norm())
    probe_file = "probe_output.pt"
    torch.save(out1.cpu(), engine_dir / probe_file)
    lat_cpu, aud_cpu = sw.probe_inputs(batch)
    manifest["probe"] = {
        "seed": sw.PROBE_SEED,
        "input_sha256": sw.sha256_bytes(lat_cpu.numpy().tobytes() + aud_cpu.numpy().tobytes()),
        "output_sha256": sw.tensor_sha256(out1),
        "output_file": probe_file,
        "deterministic_run_to_run": bool(torch.equal(out1, out2)),
        "graph_equals_direct_enqueue": bool(torch.equal(out1, out_direct)),
        "vs_eager_forward": {"mae": float(d.mean()), "max_abs": float(d.max()), "rel_l2": rel},
    }
    manifest["runtime"] = backend.describe()
    manifest["complete"] = True
    manifest["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["total_engine_mib"] = sum(blocks[b]["engine_mib"] for b in sw.BLOCK_ORDER)
    write_json_atomic(manifest_path, manifest)
    print(f"[bs{batch}] complete: probe {manifest['probe']}", flush=True)
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


BLOCKS = list(sw.BLOCK_ORDER)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, action="append", help="engine batch (repeatable); default 16")
    ap.add_argument("--root", default=str(sw.default_cache_root()))
    ap.add_argument("--opt-level", type=int, default=5)
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--timing-cache", default="", help="timing cache file (default <root>/timing_cache.bin, "
                    "seeded from the probe's tt16_timing_cache.bin)")
    ap.add_argument("--no-timing-cache", action="store_true")
    ap.add_argument("--blocks", default="", help="comma list of blocks to (re)build; default all")
    ap.add_argument("--force", action="store_true", help="rebuild even when the engine is up to date")
    ap.add_argument("--second-build", action="store_true", help="build twice, keep the faster engine per block")
    ap.add_argument("--max-minutes", type=float, default=1e9, help="do not start a new block after this long")
    ap.add_argument("--report", default="", help="optional JSON summary path")
    args = ap.parse_args()
    os.chdir(ROOT)
    logging_level = os.getenv("BUILD_LOG_LEVEL", "INFO")
    import logging

    logging.basicConfig(level=getattr(logging, logging_level), format="%(asctime)s %(name)s: %(message)s")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    mem = {"start": proc_mem()}
    model = load_eager_unet(device)
    mem["after_model"] = proc_mem()
    summary = {"args": vars(args), "batches": {}}
    for batch in (args.batch or [16]):
        m = build_batch(args, batch, model, device)
        summary["batches"][batch] = {"complete": m.get("complete"), "missing": m.get("missing_blocks"),
                                     "probe": m.get("probe"), "runtime": m.get("runtime")}
    mem["end"] = proc_mem()
    summary["host_mem"] = mem
    summary["torch_max_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(Path(args.report), summary)
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
