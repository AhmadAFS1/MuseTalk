"""CPU-only provenance policy for native TensorRT timing caches.

Legacy 4070 builds may opt into their historical seed path. Strict builds never
import unlabelled cache bytes, including an external --timing-cache file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint(base):
    return {key: base[key] for key in ("gpu", "compute_capability", "tensorrt_version", "torch_version", "cuda_version")} | {
        "hardware_compatibility_level": base.get("hardware_compatibility_level", "none"),
        "builder_optimization_level": base["build_flags"]["builder_optimization_level"],
        "workspace_gb": base["build_flags"]["workspace_gb"],
    }


def metadata_path(cache_path):
    return Path(str(cache_path) + ".json")


def select(cache_path, seed_path, target, *, strict, allow_seed):
    cache_path, seed_path = Path(cache_path), Path(seed_path)
    if cache_path.exists():
        data = cache_path.read_bytes()
        if strict:
            meta_path = metadata_path(cache_path)
            if not meta_path.is_file():
                raise ValueError(f"unverified timing cache {cache_path}; use a fresh root/cache")
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != target or meta.get("sha256") != hashlib.sha256(data).hexdigest():
                raise ValueError(f"timing cache target/hash mismatch: {cache_path}; use a fresh root/cache")
        return data, {"source": "existing", "path": str(cache_path), "sha256": hashlib.sha256(data).hexdigest(), "strict": strict}
    if not strict and allow_seed and seed_path.is_file():
        data = seed_path.read_bytes()
        return data, {"source": "legacy_4070_seed", "path": str(seed_path), "sha256": hashlib.sha256(data).hexdigest(), "strict": False}
    return None, {"source": "empty", "strict": strict}


def record(cache_path, data, target):
    path = metadata_path(cache_path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"schema": "trt_timing_cache_v1", "fingerprint": target,
                               "sha256": hashlib.sha256(data).hexdigest()}, sort_keys=True) + "\n")
    tmp.replace(path)


def validate_resume(manifest, base, strict):
    if strict and manifest.get("blocks"):
        # Do not relabel existing engine plans as a new target before skipping them.
        for key in ("gpu", "compute_capability", "tensorrt_version", "torch_version", "cuda_version"):
            if manifest.get(key) != base[key]:
                raise ValueError(f"existing engine target differs at {key}; use a fresh --root")
