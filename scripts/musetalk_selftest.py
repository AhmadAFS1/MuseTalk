#!/usr/bin/env python3
"""MuseTalk GPU self-test (imports torch; run by scripts/install_musetalk.sh and by humans).

Measures what the resolver needs to pick and sanity-check the fast recipe on THIS GPU + venv and
writes <repo>/.runtime/gpu_selftest.json (schema musetalk_gpu_selftest_v1):

  gpu{name, compute_capability, ...}, torch_version, cuda_ok,
  taesd{compile_ok, compile_mode, warmup_s, ms_bs8, eager_ms_bs8, buckets, per_bucket_ms, error},
  taesd_trt{engine_present, layout, engine_dir, ok, key, ms_bs8, ms_bs8_bgr_u8, load_s, error}
            (only timed when an engine already exists: the usable engine-store entry for this key, else
            the flat models/taesd/trt/ dir; the self-test never builds one),
  unet_eager{ok, ms_bs8, error}                                           (--unet),
  trt_import_ok, trt_versions, engine_keys{source, facts, keys{unet_ts, unet_stagewise, taesd_trt}},
  engines_found{store{kind: [key, batch, dir, usable, match, reasons, env]}, legacy{kind: [...]}},
  estimate{...}, created_utc.

Every section catches its own failure (e.g. a torch.compile / Triton failure on an unsupported
arch) and records it instead of crashing, so the resolver can fall back (MUSETALK_TAESD_COMPILE=0).
Exit 0 when CUDA works and at least one TAESD path decoded; 1 otherwise; the JSON is always
written. Run it on an otherwise idle GPU (box_guard on shared hosts):

  <venv>/bin/python scripts/musetalk_selftest.py [--unet] [--buckets 8] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path

SCHEMA = "musetalk_gpu_selftest_v1"
REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 600 else text[:600] + "..."


def parse_buckets(text: str) -> list:
    out = []
    for token in (text or "").split(","):
        token = token.strip()
        if token.isdigit() and int(token) > 0 and int(token) not in out:
            out.append(int(token))
    return sorted(out) or [8]


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def gpu_timer(torch, fn, x, iters: int, warmup: int) -> float:
    """Mean milliseconds per call measured with CUDA events after `warmup` calls."""
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            fn(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(x)
        end.record()
        end.synchronize()
    return float(start.elapsed_time(end)) / max(1, iters)


# ------------------------------------------------------------------------- engine keys / store scan
def contract_unet_ts_key(facts: dict):
    """Contract fallback: sm<major><minor>-<gpu slug>-trt<tensorrt>-tt<torch_tensorrt>."""
    cc = facts.get("compute_capability") or ""
    if not re.match(r"^\d+\.\d+$", cc) or not facts.get("tensorrt_version") or not facts.get("torch_tensorrt_version"):
        return None
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", facts["gpu_name"].lower())).strip("-")
    tt = str(facts["torch_tensorrt_version"]).split("+", 1)[0]
    return f"sm{cc.replace('.', '')}-{slug}-trt{facts['tensorrt_version']}-tt{tt}"


def load_engine_keys_module():
    """scripts/musetalk_engine_keys.py (engine store component, stdlib only) or None."""
    try:
        return importlib.import_module("scripts.musetalk_engine_keys"), None
    except Exception as exc:
        return None, short_error(exc)


def engine_keys(facts: dict) -> dict:
    """Keys for this GPU/venv via musetalk_engine_keys.engine_key(kind, facts) (flat facts shape);
    falls back to the contract formula for the torch_tensorrt UNet when the module is unavailable."""
    module, error = load_engine_keys_module()
    out = {"source": "scripts.musetalk_engine_keys" if module else "selftest_fallback",
           "facts": facts, "keys": {}, "errors": {}}
    if error:
        out["errors"]["import"] = error
    for kind in ("unet_ts", "unet_stagewise", "taesd_trt"):
        key = None
        if module is not None:
            try:
                key = module.engine_key(kind, facts)
            except Exception as exc:
                out["errors"][kind] = short_error(exc)
        elif kind == "unet_ts":
            key = contract_unet_ts_key(facts)
        out["keys"][kind] = key
    return out


def read_json(path: str):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def engines_found(repo: Path, facts: dict) -> dict:
    """Engines on disk for the three kinds: store entries (with usability for THIS host) and legacy
    engines outside the store that `unet_engine_store.py adopt` could register."""
    module, error = load_engine_keys_module()
    if module is not None:
        out = {"source": "scripts.musetalk_engine_keys", "store": {}, "legacy": {}, "errors": {}}
        for kind in ("unet_ts", "unet_stagewise", "taesd_trt"):
            try:
                entries = [module.describe_entry(entry, facts) for entry in module.list_entries(kind, repo)]
                out["store"][kind] = [{k: entry.get(k) for k in ("key", "batch", "dir", "source", "usable",
                                                                  "match", "reasons", "env")}
                                      for entry in entries]
            except Exception as exc:
                out["errors"][f"store.{kind}"] = short_error(exc)
            try:
                out["legacy"][kind] = _jsonable(module.legacy_candidates(kind, repo))
            except Exception as exc:
                out["errors"][f"legacy.{kind}"] = short_error(exc)
        return out
    return dict(fallback_scan(repo, facts), source="selftest_fallback", errors={"import": error})


def fallback_scan(repo: Path, facts: dict) -> dict:
    """Minimal scan (no usability verdicts beyond GPU/TRT match) when the keys module is missing."""
    cc = facts.get("compute_capability")
    trt = facts.get("tensorrt_version")
    store = {"unet_ts": [], "unet_stagewise": [], "taesd_trt": []}
    legacy = {"unet_ts": [], "unet_stagewise": [], "taesd_trt": []}
    ts_store = Path(os.environ.get("MUSETALK_UNET_ENGINE_STORE", "") or repo / "models" / "tensorrt_unet")
    for fp_path in sorted(glob.glob(str(ts_store / "*" / "bs*" / "fingerprint.json"))):
        fp = read_json(fp_path) or {}
        store["unet_ts"].append({"key": fp.get("engine_key"), "dir": str(Path(fp_path).parent),
                                 "validated": bool((fp.get("validation") or {}).get("passed"))})
    for ts in sorted(glob.glob(str(repo / "models" / "tensorrt_unet_*" / "unet_trt.ts"))):
        legacy["unet_ts"].append({"path": ts})
    for manifest_path in sorted(glob.glob(str(repo / "models" / "tensorrt_unet_stagewise*" / "**" / "manifest.json"),
                                          recursive=True)):
        manifest = read_json(manifest_path) or {}
        mcc = manifest.get("compute_capability")
        mcc_text = f"{mcc[0]}.{mcc[1]}" if isinstance(mcc, list) and len(mcc) == 2 else str(mcc)
        legacy["unet_stagewise"].append({"path": str(Path(manifest_path).parent), "batch": manifest.get("batch"),
                                         "gpu_name": manifest.get("gpu"), "compute_capability": mcc_text,
                                         "tensorrt_version": manifest.get("tensorrt_version"),
                                         "complete": bool(manifest.get("complete")),
                                         "matches_this_gpu": mcc_text == cc and manifest.get("tensorrt_version") == trt})
    taesd_dir = Path(os.environ.get("MUSETALK_TAESD_TRT_DIR", "") or repo / "models" / "taesd" / "trt")
    for meta_path in sorted(glob.glob(str(taesd_dir / "taesd_trt_*.json"))):
        fp = (read_json(meta_path) or {}).get("fingerprint") or {}
        legacy["taesd_trt"].append({"path": meta_path, "gpu_name": fp.get("gpu"),
                                    "compute_capability": fp.get("compute_capability"),
                                    "tensorrt_version": fp.get("tensorrt"),
                                    "matches_this_gpu": fp.get("compute_capability") == cc and fp.get("tensorrt") == trt})
    return {"store": store, "legacy": legacy}


# ------------------------------------------------------------------------- sections
def section_taesd(torch, device, buckets, compile_mode, iters, warmup, skip_compile, result):
    taesd = {"compile_ok": None, "compile_mode": compile_mode, "warmup_s": None, "ms_bs8": None,
             "eager_ms_bs8": None, "buckets": buckets, "per_bucket_ms": {}, "source": None, "error": None,
             "eager_error": None}
    result["taesd"] = taesd
    model = None
    try:
        from scripts import vae_fast_decoder as vfd

        os.environ["MUSETALK_TAESD_COMPILE_MODE"] = compile_mode
        os.environ["MUSETALK_TAESD_COMPILE"] = "0" if skip_compile else "1"
        local = REPO_ROOT / "models" / "taesd" / "config.json"
        taesd["source"] = str(local.parent) if local.exists() else os.getenv("MUSETALK_TAESD_MODEL", vfd.DEFAULT_MODEL)
        backend = vfd.TaesdVaeDecodeBackend.load(device=device, runtime_dtype=torch.float16)
        model = backend.model
    except Exception as exc:
        taesd["error"] = short_error(exc)
        taesd["compile_ok"] = False
        return None
    probe8 = torch.randn((8, 4, 32, 32), device=device, dtype=torch.float16)
    # Eager first: it is the fallback the resolver picks when compile fails.
    try:
        eager = vfd.TaesdVaeDecodeBackend(model=model, device=device, runtime_dtype=torch.float16,
                                          compile_enabled=False, compile_mode=compile_mode)
        taesd["eager_ms_bs8"] = round(gpu_timer(torch, lambda x: eager.decode(x, 1.0), probe8, iters, warmup), 3)
    except Exception as exc:
        taesd["eager_error"] = short_error(exc)
    if skip_compile:
        taesd["compile_ok"] = None
        taesd["error"] = "compile skipped (--skip-compile)"
        return model
    try:
        started = time.time()
        backend.warmup(buckets)  # torch.compile(dynamic=False) once per bucket, exactly like the server
        taesd["warmup_s"] = round(time.time() - started, 2)
        if not backend.compile_enabled:
            raise RuntimeError("torch.compile unavailable (backend fell back to eager during warmup)")
        for bucket in buckets:
            probe = torch.randn((bucket, 4, 32, 32), device=device, dtype=torch.float16)
            taesd["per_bucket_ms"][str(bucket)] = round(
                gpu_timer(torch, lambda x: backend.decode(x, 1.0), probe, iters, warmup), 3)
        if 8 in buckets:
            taesd["ms_bs8"] = taesd["per_bucket_ms"]["8"]
        else:
            taesd["ms_bs8"] = round(gpu_timer(torch, lambda x: backend.decode(x, 1.0), probe8, iters, warmup), 3)
        taesd["compile_ok"] = True
    except Exception as exc:
        taesd["compile_ok"] = False
        taesd["error"] = short_error(exc)
        taesd["traceback_tail"] = traceback.format_exc()[-1500:]
    return model


def section_taesd_trt(torch, device, model, facts, iters, warmup, result):
    """Time an EXISTING TAESD TRT engine: the usable engine-store entry for this key if there is one
    (its env from musetalk_engine_keys.entry_env), else the flat legacy dir models/taesd/trt/
    (vae_fast_decoder's own fingerprint check then decides). Never builds (MUSETALK_TAESD_TRT_BUILD=0)."""
    info = {"engine_present": False, "layout": None, "engine_dir": None, "store_key": None, "ok": None,
            "key": None, "ms_bs8": None, "ms_bs8_bgr_u8": None, "load_s": None, "error": None}
    result["taesd_trt"] = info
    env = {}
    module, _ = load_engine_keys_module()
    if module is not None and facts:
        try:
            entry = module.find_engine("taesd_trt", facts, REPO_ROOT)
            if entry:
                env = dict(module.entry_env(entry))
                info.update(layout="store", engine_dir=entry.get("dir"), store_key=entry.get("key"))
        except Exception as exc:
            info["store_error"] = short_error(exc)
    if not env:
        flat = Path(os.environ.get("MUSETALK_TAESD_TRT_DIR", "") or REPO_ROOT / "models" / "taesd" / "trt")
        if glob.glob(str(flat / "taesd_trt_*.json")):
            env = {"MUSETALK_TAESD_TRT_DIR": str(flat)}
            info.update(layout="legacy_flat", engine_dir=str(flat))
    if not env:
        info["error"] = ("no TAESD TRT engine on disk (build: scripts/unet_engine_store.py build --kind taesd_trt, "
                         "or python scripts/vae_fast_decoder.py build)")
        return
    info["engine_present"] = True
    if model is None:
        info["ok"] = False
        info["error"] = "TAESD model did not load"
        return
    env["MUSETALK_TAESD_TRT_BUILD"] = "0"
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        from scripts import vae_fast_decoder as vfd

        started = time.time()
        backend = vfd.load_taesd_trt_backend(device=device, runtime_dtype=torch.float16, model=model)
        info["load_s"] = round(time.time() - started, 2)
        meta = getattr(backend, "meta", None) or {}
        info["key"] = meta.get("key")
        info["gate"] = (meta.get("gate") or {}).get("verdict")
        probe = torch.randn((8, 4, 32, 32), device=device, dtype=torch.float16)
        info["ms_bs8"] = round(gpu_timer(torch, lambda x: backend.decode(x, 1.0), probe, iters, warmup), 3)
        if hasattr(backend, "decode_bgr_u8"):
            info["ms_bs8_bgr_u8"] = round(gpu_timer(torch, backend.decode_bgr_u8, probe, iters, warmup), 3)
        info["ok"] = True
        del backend
    except FileNotFoundError as exc:
        info["ok"] = False
        info["error"] = "no engine for this GPU/TensorRT/recipe fingerprint: " + short_error(exc)
    except Exception as exc:
        info["ok"] = False
        info["error"] = short_error(exc)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def section_unet_eager(torch, device, iters, warmup, result):
    info = {"ok": False, "ms_bs8": None, "load_s": None, "error": None}
    result["unet_eager"] = info
    model = None
    try:
        from diffusers import UNet2DConditionModel

        cfg_path = REPO_ROOT / "models" / "musetalkV15" / "musetalk.json"
        weights = REPO_ROOT / "models" / "musetalkV15" / "unet.pth"
        started = time.time()
        cfg = json.loads(cfg_path.read_text())
        # Same low-peak load as scripts/build_unet_multi_avatar_corpus.load_eager_unet: mmap the fp32
        # state dict and copy into fp16 parameters (bit-identical to .half()).
        model = UNet2DConditionModel(**cfg).half()
        try:
            state = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
        except Exception:
            state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
        del state
        model = model.to(device).eval()
        model.requires_grad_(False)
        info["load_s"] = round(time.time() - started, 2)
        latents = torch.randn((8, int(cfg.get("in_channels", 8)), 32, 32), device=device, dtype=torch.float16)
        audio = torch.randn((8, 50, int(cfg.get("cross_attention_dim", 384))), device=device, dtype=torch.float16)
        timesteps = torch.tensor([0], device=device)
        info["ms_bs8"] = round(gpu_timer(
            torch, lambda x: model(x, timesteps, encoder_hidden_states=audio).sample, latents, iters, warmup), 3)
        info["ok"] = True
    except Exception as exc:
        info["error"] = short_error(exc)
    finally:
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def section_trt_import(torch, result) -> dict:
    versions = {"tensorrt": None, "torch_tensorrt": None}
    try:
        import tensorrt

        versions["tensorrt"] = tensorrt.__version__
        import torch_tensorrt  # noqa: F401  (registers torch.classes.tensorrt.Engine)

        versions["torch_tensorrt"] = torch_tensorrt.__version__
        getattr(torch.classes.tensorrt, "Engine")
        result["trt_import_ok"] = True
    except Exception as exc:
        result["trt_import_ok"] = False
        result["trt_import_error"] = short_error(exc)
    result["trt_versions"] = versions
    return versions


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    global REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--out", default=os.getenv("MUSETALK_GPU_SELFTEST_FILE", ""),
                        help="JSON output (default $MUSETALK_GPU_SELFTEST_FILE, else <repo>/.runtime/gpu_selftest.json)")
    parser.add_argument("--buckets", default=os.getenv("HLS_SCHEDULER_FIXED_BATCH_SIZES")
                        or os.getenv("MUSETALK_TAESD_WARMUP_BATCHES") or "8",
                        help="comma list of TAESD compile buckets (default: HLS_SCHEDULER_FIXED_BATCH_SIZES or 8)")
    parser.add_argument("--compile-mode", default=os.getenv("MUSETALK_TAESD_COMPILE_MODE", "max-autotune"))
    parser.add_argument("--skip-compile", action="store_true", help="only time eager TAESD")
    parser.add_argument("--unet", action="store_true", help="also time the eager fp16 UNet at bs8 (~4 GB host RAM)")
    parser.add_argument("--no-taesd-trt", action="store_true", help="do not time an existing TAESD TRT engine")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    REPO_ROOT = Path(args.repo_root).resolve()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    out_path = Path(args.out) if args.out else REPO_ROOT / ".runtime" / "gpu_selftest.json"
    buckets = parse_buckets(args.buckets)
    started = time.time()
    result = {"schema": SCHEMA, "created_utc": utc_now(), "host": platform.node(), "repo_root": str(REPO_ROOT),
              "python": sys.version.split()[0], "gpu": None, "torch_version": None, "cuda_ok": False,
              "taesd": None, "taesd_trt": None, "unet_eager": {"ok": None, "ms_bs8": None, "error": "not requested (--unet)"},
              "trt_import_ok": None, "engine_keys": None, "engines_found": None, "estimate": None}
    exit_code = 1
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        try:
            result["torch_arch_flags"] = torch._C._cuda_getArchFlags()
        except Exception:
            result["torch_arch_flags"] = None
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")
            device = torch.device(args.device)
            torch.cuda.init()
            major, minor = torch.cuda.get_device_capability(device)
            props = torch.cuda.get_device_properties(device)
            result["gpu"] = {"index": device.index or 0, "name": torch.cuda.get_device_name(device),
                             "compute_capability": f"{major}.{minor}",
                             "memory_total_mib": int(props.total_memory // 2**20),
                             "multiprocessors": int(props.multi_processor_count)}
            torch.zeros(1, device=device).add_(1)
            torch.cuda.synchronize(device)
            result["cuda_ok"] = True
        except Exception as exc:
            result["cuda_error"] = short_error(exc)
        versions = section_trt_import(torch, result)
        facts = None
        if result["gpu"]:
            facts = {"gpu_name": result["gpu"]["name"], "compute_capability": result["gpu"]["compute_capability"],
                     "tensorrt_version": versions.get("tensorrt"), "torch_tensorrt_version": versions.get("torch_tensorrt"),
                     "torch_version": torch.__version__}
            result["engine_keys"] = engine_keys(facts)
            result["engines_found"] = engines_found(REPO_ROOT, facts)
        if result["cuda_ok"]:
            torch.backends.cudnn.benchmark = True
            model = section_taesd(torch, device, buckets, args.compile_mode, args.iters, args.warmup_iters,
                                  args.skip_compile, result)
            if not args.no_taesd_trt:
                section_taesd_trt(torch, device, model, facts, args.iters, args.warmup_iters, result)
            if args.unet:
                section_unet_eager(torch, device, args.iters, args.warmup_iters, result)
            result["peak_vram_mib"] = int(torch.cuda.max_memory_allocated(device) // 2**20)
            taesd = result.get("taesd") or {}
            vae_ms = taesd.get("ms_bs8") if taesd.get("compile_ok") else taesd.get("eager_ms_bs8")
            unet_ms = (result.get("unet_eager") or {}).get("ms_bs8")
            estimate = {"vae_ms_bs8": vae_ms, "vae_path": "compiled" if taesd.get("compile_ok") else "eager",
                        "unet_eager_ms_bs8": unet_ms,
                        "note": "GPU-path only (UNet + TAESD per bs8 batch); serving adds H2D/compose/encode"}
            if vae_ms and unet_ms:
                estimate["gpu_path_fps_eager_unet"] = round(8000.0 / (vae_ms + unet_ms), 1)
            result["estimate"] = estimate
            if taesd.get("ms_bs8") or taesd.get("eager_ms_bs8"):
                exit_code = 0
    except Exception as exc:
        result["fatal_error"] = short_error(exc)
        result["traceback_tail"] = traceback.format_exc()[-2000:]
    result["duration_s"] = round(time.time() - started, 2)
    write_json_atomic(out_path, result)
    taesd = result.get("taesd") or {}
    print(f"[selftest] gpu={(result.get('gpu') or {}).get('name')} cc={(result.get('gpu') or {}).get('compute_capability')} "
          f"cuda_ok={result['cuda_ok']} trt_import_ok={result.get('trt_import_ok')} "
          f"taesd.compile_ok={taesd.get('compile_ok')} ms_bs8={taesd.get('ms_bs8')} eager_ms_bs8={taesd.get('eager_ms_bs8')} "
          f"taesd_trt.ok={(result.get('taesd_trt') or {}).get('ok')} unet_eager.ms_bs8={(result.get('unet_eager') or {}).get('ms_bs8')} "
          f"-> {out_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
