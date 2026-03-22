# MuseTalk Separate TensorRT Environment Plan

## Purpose

This document defines the next branch for TensorRT work after the current
`/content/py310` environment proved insufficient for a trustworthy full VAE
TensorRT path.

The goal is:

- keep the current stable MuseTalk server environment untouched
- create a separate TensorRT-focused experiment environment
- use that environment only for export and benchmark work first
- only consider moving the full server there if the benchmark materially improves

## Why A Separate Environment

The current environment already proved three things:

- the repo-side TensorRT integration work is mostly in place
- a VAE engine artifact could be exported once
- but the current stack still failed at trustworthy runtime activation and then
  failed a stricter full compile because of unsupported operators

So the next problem is no longer “missing code wiring.”
It is “the current backend stack is the wrong place to keep forcing this.”

## Best Practical Choice On This Machine

This machine does **not** currently have Docker available.

That means the best practical next move is:

- a **separate Python 3.10 venv**
- on the same machine
- with a **newer coherent Torch / Torch-TensorRT / TensorRT stack**
- while leaving `/content/py310` untouched

If Docker or another container runtime becomes available later, a containerized
TensorRT environment would be even cleaner. But for now, a second venv is the
best practical option.

## Environment Layout

Recommended paths:

- current stable env:
  - `/content/py310`
- new TRT experiment env:
  - `/content/py310_trt_exp`
- shared repo:
  - `/content/MuseTalk`
- separate TRT artifact directory:
  - `/content/MuseTalk/models/tensorrt_altenv`

Using a separate artifact directory matters because we do **not** want to mix:

- current-environment TensorRT artifacts
- alternate-environment TensorRT artifacts

## Current Attempt Status

The first alternate-environment setup attempt has already started, and the
current state is now known:

- `/content/py310_trt_exp` was created successfully
- the first validation import failed with:
  - `ModuleNotFoundError: No module named 'torch'`
- that import failure was expected, because the new venv had been created but
  no backend stack had been installed yet
- the first real install attempt then hit a storage failure before `torch`
  finished installing
- a later unpinned `torch-tensorrt` install attempt also tried to pull the
  newest `torch-tensorrt 2.10.0` family and `tensorrt-cu13`, which is **not**
  the intended stack for this branch

Current machine-level storage snapshot from that attempt:

- filesystem size: about `56G`
- used: about `54G`
- available: about `2.7G`
- `/root/.cache/pip`: about `6.1G`
- partially created `/content/py310_trt_exp`: about `4.0G`

Practical meaning:

- the first alternate-environment attempt failed primarily on disk pressure
- the unpinned backend install also confirmed that we must pin the new
  Torch-TensorRT family explicitly instead of allowing pip to select the latest
  release

## Core Principle

Do **not** piece together the new environment one package at a time the way the
current environment ended up being assembled.

Instead:

- choose one coherent PyTorch + CUDA + Torch-TensorRT family
- choose the matching TensorRT Python/runtime packages for that family
- install them together in the new venv

The key mistake to avoid is mixing:

- older Torch
- newer TensorRT bits
- or export/runtime stacks that are only accidentally compatible

One more mistake to avoid:

- do **not** run an unpinned `pip install torch-tensorrt tensorrt`

That now proved to be dangerous because pip selected:

- `torch-tensorrt 2.10.0`
- a newer `torch` family
- `tensorrt-cu13`

which is outside the intended compatibility family for this branch.

## Phase 0: Storage And Cleanup Guardrail

Before recreating the alternate environment, free disk space and remove the
half-installed venv.

Recommended commands:

```bash
deactivate 2>/dev/null || true
rm -rf /root/.cache/pip
rm -rf /content/py310_trt_exp
df -h /content
```

Why:

- the first install attempt failed with `OSError: [Errno 28] No space left on device`
- Torch + CUDA wheels are large enough that low free space will produce noisy,
  partially installed states
- clearing the pip cache first is the fastest way to recover several gigabytes

## Phase 1: Create The New Venv

```bash
python3.10 -m venv /content/py310_trt_exp
source /content/py310_trt_exp/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Phase 2: Install A Coherent New Torch/TRT Stack

Use the official compatibility guidance for the chosen stack and install the
core packages together:

- `torch`
- `torchvision`
- `torchaudio`
- `torch-tensorrt`
- `tensorrt` or the matching NVIDIA runtime packages

Guidance for this phase:

- stay on Python `3.10`
- prefer one CUDA family consistently across the stack
- do **not** mix the old `/content/py310` Torch stack into this new venv
- do **not** use unpinned `torch-tensorrt`
- verify importability immediately after install

### Current Recommended First Install Attempt

Based on the machine state now known:

- Python `3.10`
- NVIDIA driver `535.113.01`
- reported CUDA version `12.2`

the next clean attempt should target a `cu121` PyTorch family with a pinned
`torch-tensorrt 2.5.x` family.

Recommended commands:

```bash
source /content/py310_trt_exp/bin/activate

python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install --no-cache-dir \
  --extra-index-url https://pypi.nvidia.com \
  torch-tensorrt==2.5.0
```

Important notes:

- `--no-cache-dir` is intentional because disk space is currently tight
- the earlier unpinned install attempted to move to `torch-tensorrt 2.10.0`
  and `tensorrt-cu13`, which is not the planned stack
- if the `torch` install fails again because of free space, stop and reclaim
  more disk before continuing

Minimum validation:

```bash
python - <<'PY'
import torch
import torch_tensorrt
import tensorrt
print("torch", torch.__version__)
print("torch_tensorrt", torch_tensorrt.__version__)
print("tensorrt", tensorrt.__version__)
print("Engine class", getattr(torch.classes.tensorrt, "Engine"))
PY
```

If that last line fails, stop there. The environment is not ready.

If the import fails with `No module named 'torch'`, that means the new torch
install never completed successfully. That is not a TensorRT compatibility
signal by itself; it is just an incomplete environment.

## Phase 3: Reinstall MuseTalk Runtime Dependencies

In the new venv, install the MuseTalk Python dependencies again, but keep the
new Torch stack intact.

That means:

- install `requirements.txt`
- reinstall the MM family packages
- keep the new Torch family from Phase 2

The goal is to reproduce the current server dependency set while preserving the
new backend stack.

## Phase 4: Keep TRT Artifacts Separate

When exporting in the new venv, do **not** overwrite the current TensorRT
artifact path first.

Use:

```bash
export MUSETALK_TRT_DIR=./models/tensorrt_altenv
```

That lets us compare:

- current-environment artifacts
- alternate-environment artifacts

without ambiguity.

## Phase 5: Export Only, Not Full Server Yet

The first job of the new environment is:

- export the VAE TensorRT engine

Not:

- run the full API server
- run the full HLS system

The export step should be isolated first.

Recommended command shape:

```bash
cd /content/MuseTalk
export MUSETALK_TRT_DIR=./models/tensorrt_altenv
python scripts/tensorrt_export.py \
  --components vae \
  --batch-sizes 4,8,16,24,32,40,48
```

## Phase 6: Runtime Loader Validation

Before benchmarking, validate that the engine can really load in the new env
with fallback disabled.

```bash
cd /content/MuseTalk
MUSETALK_TRT_ENABLED=1 \
MUSETALK_VAE_BACKEND=trt \
MUSETALK_TRT_FALLBACK=0 \
MUSETALK_TRT_DIR=./models/tensorrt_altenv \
python - <<'PY'
from scripts.trt_runtime import load_vae_trt_decoder
backend = load_vae_trt_decoder()
print("backend", type(backend).__name__, getattr(backend, "name", None))
PY
```

If this fails, do **not** run the benchmark yet.

## Phase 7: Benchmark Before Any Server Migration

Only after load succeeds, run the isolated benchmark:

```bash
cd /content/MuseTalk
MUSETALK_TRT_ENABLED=1 \
MUSETALK_VAE_BACKEND=trt \
MUSETALK_TRT_FALLBACK=0 \
MUSETALK_TRT_DIR=./models/tensorrt_altenv \
MUSETALK_COMPILE_VAE=0 \
python scripts/benchmark_pipeline.py \
  --batch-sizes 4,8,16,24,32,40,48 \
  --warmup 20 \
  --iters 50 \
  --output-json benchmark_pipeline_trt_vae_altenv.json
```

This is the real gate for whether the alternate environment is worth pursuing.

## Success Criteria

### Technical Success

- the engine exports cleanly
- the runtime loader activates the backend with fallback disabled
- benchmark logs show:
  - `VAE decode backend: tensorrt`

### Performance Success

At least one of these should happen:

- best total throughput rises materially above the current `~51 fps` baseline
- VAE full-path latency drops materially versus baseline

If neither happens, then the alternate environment is not worth migrating.

## Decision Gate

If the alternate environment succeeds technically **and** materially improves
the benchmark:

1. run the full HLS `load_test.py` there
2. compare against the stable baseline
3. only then decide whether to migrate the full API server environment

If the alternate environment still fails or does not improve the benchmark:

- stop pushing the TensorRT VAE branch
- pivot to the next backend option, such as ONNX Runtime

## What Not To Do

- do not modify the stable `/content/py310` environment further for TensorRT
- do not overwrite the stable artifact path with alternate-environment outputs
- do not migrate the full API server before the new environment wins the benchmark

## Bottom Line

The next clean branch is:

1. clear disk pressure and remove the half-installed `/content/py310_trt_exp`
2. recreate `/content/py310_trt_exp`
3. install a pinned `torch 2.5.1 / cu121` + `torch-tensorrt 2.5.0` family
4. validate imports before touching MuseTalk runtime dependencies
5. export to `models/tensorrt_altenv`
6. validate runtime load with fallback disabled
5. benchmark there before touching the full server
