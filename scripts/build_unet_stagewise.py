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

--hardware-compat ampere_plus builds every block with TensorRT HardwareCompatibilityLevel.AMPERE_PLUS:
the set then loads on any GPU of compute capability 8.0 or newer (RTX 3090 and 4070 SUPER alike) with
the same TensorRT version. A root holds one level only; such a build never seeds from the probe's
default-level timing cache, and finalising records the cross-GPU probe bound the loader applies on
other GPU models.

Run under the GPU lease, e.g.:
  scripts/box_guard.sh run --min-avail-gb 8 -- /workspace/.venvs/musetalk_trt_stagewise/bin/python \
      scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --max-minutes 20
"""
from __future__ import annotations

import argparse
import gc
import glob
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


def recipe_block_and_local(full_name: str, variant: str) -> tuple[str, str]:
    """UNet module name -> (stagewise block, name inside that block's wrapper)."""
    import re

    m = re.match(r"(down|up)_blocks\.(\d+)\.(.+)$", full_name)
    if m:
        blk = f"{m.group(1)}{m.group(2)}"
        if blk == "down0" and variant == "srccache":
            blk = "down0rest"
        return blk, "block." + m.group(3)
    if full_name.startswith("mid_block."):
        return "mid", "block." + full_name[len("mid_block."):]
    raise SystemExit(f"--int8-recipe: layer {full_name} is not inside a down/mid/up block")


def load_int8_recipe(path: str, variant: str) -> dict:
    """{"layers": {unet_module_name: {"input_amax": float}}} -> {block: {local_name: input_amax}}."""
    raw = json.loads(Path(path).read_text())
    per_block: dict = {}
    for full, spec in raw["layers"].items():
        blk, local = recipe_block_and_local(full, variant)
        per_block.setdefault(blk, {})[local] = float(spec["input_amax"])
    return per_block


def load_recipe_tensors(path: str, variant: str) -> tuple[dict, dict, str | None]:
    """A recovered recipe's learned tensors (int8_layer_study.py --stage recover): FP16 model edits
    {"bias_delta": {unet_module_name: delta}, "lora": {unet_module_name: {"A", "B"}}} and pinned per-output-channel
    weight amax {block: {local_name: amax}}."""
    raw = json.loads(Path(path).read_text())
    if not raw.get("tensors"):
        return {}, {}, None
    t = torch.load(raw["tensors"], map_location="cpu", weights_only=True)
    if t.get("smooth"):
        raise SystemExit("--int8-recipe: per-channel smoothing tensors are study-only (not exported)")
    w_amax: dict = {}
    for full, v in t.get("weight_amax", {}).items():
        blk, local = recipe_block_and_local(full, variant)
        w_amax.setdefault(blk, {})[local] = v
    edits = {"bias_delta": t.get("bias_delta", {}), "lora": t.get("lora", {})}
    return edits, w_amax, sw.sha256_bytes(Path(raw["tensors"]).read_bytes())[:16]


def apply_int8_recipe(module, layer_amax: dict, weight_amax: dict | None = None) -> list:
    """Keep INT8 Q/DQ only on the recipe's layers; pin their input amax (and learned weight amax) to the recipe."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    kept = []
    for name, m in module.named_modules():
        iq, wq = getattr(m, "input_quantizer", None), getattr(m, "weight_quantizer", None)
        if not (isinstance(iq, TensorQuantizer) and isinstance(wq, TensorQuantizer)):
            continue
        local = name[len("inner."):] if name.startswith("inner.") else name
        if local in layer_amax:
            iq.amax = torch.tensor(layer_amax[local], device=iq.amax.device, dtype=iq.amax.dtype)
            if weight_amax and local in weight_amax:
                wq.amax = weight_amax[local].to(wq.amax.device, wq.amax.dtype).reshape(wq.amax.shape)
            kept.append(local)
        else:
            iq.disable()
            wq.disable()
    missing = sorted(set(layer_amax) - set(kept))
    if missing:
        raise SystemExit(f"--int8-recipe: layers not found in block wrapper: {missing[:5]}")
    return kept


def build_batch(args, batch: int, model, device) -> dict:
    root = Path(args.root).resolve()
    engine_dir = root / f"bs{batch}"
    engine_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = engine_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    import tensorrt as trt

    spec = sw.chain_spec(model, args.variant)
    order = sw.block_order(args.variant)
    hw_compat = sw.hardware_compat_level(args.hardware_compat)
    flags = sw.build_flags_record(args.opt_level, args.workspace_gb, not args.no_timing_cache,
                                  hardware_compat=hw_compat)
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
        "variant": args.variant,
        "build_flags": flags,
    }
    if hw_compat != "none":
        base["hardware_compatibility_level"] = hw_compat
    if manifest.get("spec") not in (None, spec) or manifest.get("batch") not in (None, batch):
        raise SystemExit(f"{manifest_path}: existing manifest has a different spec/batch; use a fresh --root")
    if manifest.get("blocks") and sw.hardware_compat_level(manifest.get("hardware_compatibility_level")) != hw_compat:
        raise SystemExit(f"{manifest_path}: existing set has hardware compatibility "
                         f"{manifest.get('hardware_compatibility_level') or 'none'}, not {hw_compat}; use a fresh --root")
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
        elif SEED_TIMING_CACHE.exists() and hw_compat == "none":  # the seed was timed without hardware compat
            cache_bytes = SEED_TIMING_CACHE.read_bytes()
            manifest["timing_cache_seed"] = str(SEED_TIMING_CACHE.relative_to(ROOT))
    wanted = order if not args.blocks else [b for b in order if b in args.blocks.split(",")]
    int8_blocks = {b for b in (args.int8_blocks or "").split(",") if b}
    recipe = load_int8_recipe(args.int8_recipe, args.variant) if args.int8_recipe else {}
    recipe_sha = sw.sha256_bytes(Path(args.int8_recipe).read_bytes())[:16] if args.int8_recipe else None
    _, recipe_w_amax, tensors_sha = load_recipe_tensors(args.int8_recipe, args.variant) if args.int8_recipe else ({}, {}, None)
    if recipe:
        int8_blocks |= set(recipe)
    unknown = int8_blocks - set(order)
    if unknown:
        raise SystemExit(f"--int8-blocks: unknown blocks {sorted(unknown)}")
    calib = []
    if int8_blocks:
        # Calibration inputs per block, traced through the FP16 eager wrappers from real
        # scheduler-equivalent corpus batches (main split only; holdout stays unseen).
        files = sorted(glob.glob(str(Path(args.calib_dir) / "unet_io_*.pt")))
        need = args.calib_batches * (batch // 8)
        if len(files) < need:
            raise SystemExit(f"--calib-dir has {len(files)} captures; need {need}")
        step = max(1, len(files) // need)
        picked = files[::step][:need]
        for k in range(0, len(picked), batch // 8):
            ds = [torch.load(f, map_location="cpu") for f in picked[k:k + batch // 8]]
            lat_c = torch.cat([d["latent_batch"] for d in ds]).to(device=device, dtype=torch.float16)
            aud_c = torch.cat([d["audio_feature_batch"] for d in ds]).to(device=device, dtype=torch.float16)
            t_c, _ = sw.trace_block_inputs(model, spec, wrappers, lat_c, aud_c)
            calib.append({key: val for key, val in t_c.items()})
        manifest["int8_calibration"] = {"dir": str(Path(args.calib_dir)), "files": [Path(f).name for f in picked],
                                        "batches": len(calib), "algorithm": "modelopt INT8_DEFAULT_CFG (max)"}
        if recipe:
            manifest["int8_calibration"].update(
                recipe=str(args.int8_recipe), recipe_sha256_16=recipe_sha,
                algorithm="layer-selective recipe: weight per-channel max, input amax pinned from the recipe "
                          "(scripts/int8_layer_study.py); non-recipe layers stay FP16")
            if tensors_sha:
                manifest["int8_calibration"].update(
                    recipe_tensors_sha256_16=tensors_sha,
                    algorithm="layer-selective recovered recipe: learned input amax, per-channel bias corrections and "
                              "quantization-aware LoRA merged into the FP16 weights, weight amax per-channel max or learned "
                              "(scripts/int8_layer_study.py --stage recover); non-recipe layers stay FP16")
    started = time.time()
    log = manifest.setdefault("build_log", [])
    build_specs = ([sw.PREFIX_SPEC] + spec) if args.variant == "srccache" else spec
    for blk in build_specs:
        name = blk["name"]
        if name not in wanted:
            continue
        if (time.time() - started) / 60.0 > args.max_minutes:
            print(f"[bs{batch}] --max-minutes reached; stopping before {name}", flush=True)
            break
        args_t = [tensors[k] for k in blk["inputs"]]
        is_int8 = name in int8_blocks
        block_flags = dict(flags, precision="int8_qdq") if is_int8 else flags
        if is_int8 and name in recipe:
            block_flags = dict(block_flags, precision=f"int8_qdq_recipe:{recipe_sha}")
        t0 = time.time()
        export_module = wrappers[name]
        if is_int8:
            import copy
            import modelopt.torch.quantization as mtq

            export_module = copy.deepcopy(wrappers[name]).eval()

            def _calib_loop(q, _blk=blk):
                with torch.no_grad():
                    for t_c in calib:
                        q(*[t_c[k] for k in _blk["inputs"]])

            export_module = mtq.quantize(export_module, mtq.INT8_DEFAULT_CFG, _calib_loop)
            if name in recipe:
                apply_int8_recipe(export_module, recipe[name], recipe_w_amax.get(name))

            class _HalfOutputs(torch.nn.Module):
                """The fake-quant export can surface fp32 outputs; the chain's static buffers are fp16."""

                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, *xs):
                    out = self.inner(*xs)
                    if isinstance(out, (tuple, list)):
                        return tuple(o.half() for o in out)
                    return out.half()

            export_module = _HalfOutputs(export_module).eval()
        onnx_bytes = sw.export_block_onnx(export_module, args_t)
        export_s = time.time() - t0
        onnx_sha = sw.sha256_bytes(onnx_bytes)
        engine_file = f"{name}.int8.plan" if is_int8 else f"{name}.plan"
        prev = blocks.get(name, {})
        if (not args.force and prev.get("onnx_sha256") == onnx_sha and prev.get("build_flags") == block_flags
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
            timing_cache=cache_bytes, use_timing_cache=not args.no_timing_cache, int8=is_int8,
            hardware_compat=hw_compat)
        mem1 = proc_mem()
        onnx_mb = len(onnx_bytes) / 2**20
        del onnx_bytes
        if cache_out is not None:
            cache_bytes = cache_out
            write_bytes_atomic(cache_path, cache_bytes)
        entry = {"engine_file": engine_file, "onnx_sha256": onnx_sha, "onnx_mib": onnx_mb,
                 "export_s": export_s, "build_s": build_s, "build_flags": block_flags,
                 "inputs": blk["inputs"], "outputs": blk["outputs"],
                 "host_mem_before_build": mem0, "host_mem_after_build": mem1}
        if args.second_build:
            onnx_b = sw.export_block_onnx(export_module, args_t)
            eng_b, build_b, _ = sw.build_engine_from_onnx(
                onnx_b, opt_level=args.opt_level, workspace_gb=args.workspace_gb, use_timing_cache=False, int8=is_int8,
                hardware_compat=hw_compat)
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

    missing = [b for b in order if b not in blocks or not (engine_dir / blocks[b]["engine_file"]).exists()]
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
    if hw_compat != "none":
        # the loader's bound on GPU models other than this one (the build GPU stays bit-exact)
        manifest["probe"]["cross_gpu_rel_l2_max"] = sw.CROSS_GPU_PROBE_REL_L2_MAX
    manifest["runtime"] = backend.describe()
    manifest["complete"] = True
    manifest["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["total_engine_mib"] = sum(blocks[b]["engine_mib"] for b in order)
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
    ap.add_argument("--variant", choices=sw.VARIANTS, default="default",
                    help="srccache: separate per-source-frame prefix engine (conv_in + down0.resnets[0]) and a chain "
                    "starting at down0rest; default: today's 11-block chain")
    ap.add_argument("--int8-blocks", default="", help="comma list of blocks built as INT8 Q/DQ (modelopt max "
                    "calibration on real corpus batches); all other blocks stay FP16. Default: none (all FP16)")
    ap.add_argument("--int8-recipe", default="", help="layer-selective INT8 recipe JSON ({\"layers\": {unet_module: "
                    "{\"input_amax\": x}}}, from scripts/int8_layer_study.py --stage export); blocks holding a recipe "
                    "layer are built INT8 with Q/DQ on those layers only, everything else stays FP16")
    ap.add_argument("--calib-dir", default=str(ROOT / "calibration/unet_multi_avatar_20260928"),
                    help="UNet capture corpus used for INT8 calibration (main split; holdout stays unseen)")
    ap.add_argument("--calib-batches", type=int, default=8, help="engine-batch-sized calibration batches")
    ap.add_argument("--report", default="", help="optional JSON summary path")
    ap.add_argument("--hardware-compat", default="none", choices=sw.HW_COMPAT_LEVELS,
                    help="none (default: engines for this GPU's compute capability) or ampere_plus (one set for "
                         "every Ampere-or-newer GPU; TensorRT HardwareCompatibilityLevel.AMPERE_PLUS)")
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
    if args.int8_recipe:
        edits, _, _ = load_recipe_tensors(args.int8_recipe, "default")
        with torch.no_grad():
            for full, delta in edits.get("bias_delta", {}).items():
                m = model.get_submodule(full)
                m.bias.add_(delta.to(m.bias.device, m.bias.dtype))
            for full, ab in edits.get("lora", {}).items():
                # quantization-aware LoRA merged in FP32 then rounded once, as in the study's fake quant
                m = model.get_submodule(full)
                dw = (ab["B"].float() @ ab["A"].float()).view(m.weight.shape).to(m.weight.device)
                m.weight.copy_((m.weight.float() + dw).to(m.weight.dtype))
        if any(edits.values()):
            print(f"applied learned edits: bias corrections on {len(edits.get('bias_delta', {}))} layers, "
                  f"LoRA merged into {len(edits.get('lora', {}))} layers", flush=True)
        smooth = json.loads(Path(args.int8_recipe).read_text()).get("smooth_ff")
        if smooth:
            # power-of-two SmoothQuant folded into GEGLU (FP16-noise change, not bit-exact); INT8 ff.net.2 range shrinks
            from scripts import unet_int8_smooth as sq

            sq.apply_smoothing(model, smooth["exponents"])
            print(f"applied smooth_ff (alpha {smooth['alpha']}) to {len(smooth['exponents'])} ff.net.2 layers", flush=True)
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
