# MuseTalk CPU Optimization Analysis

## Problem Statement

After migrating MuseTalk to a server with more CPU cores, **throughput decreased** instead of increasing. This document analyzes the root cause and provides a phased optimization plan.

---

## Table of Contents

- [Root Cause Analysis](#root-cause-analysis)
- [Current Pipeline Architecture](#current-pipeline-architecture)
- [Why Blind Multithreading Will Not Work](#why-blind-multithreading-will-not-work)
- [Why More Cores Hurt Performance](#why-more-cores-hurt-performance)
- [March 23 Reality Check](#march-23-reality-check)
- [Optimization Plan](#optimization-plan)
  - [Phase 0: Restore The Stable Baseline](#phase-0-restore-the-stable-baseline)
  - [Phase 1: Parallelize HLS Prep](#phase-1-parallelize-hls-prep)
  - [Phase 2: Fix Avatar Cache Miss Cost](#phase-2-fix-avatar-cache-miss-cost)
  - [Phase 3: Persistent Encode Pipeline](#phase-3-persistent-encode-pipeline)
  - [Phase 4: Compose Pipeline Refactor](#phase-4-compose-pipeline-refactor)
  - [Phase 5: Consolidate Live Serving Paths](#phase-5-consolidate-live-serving-paths)
- [Core Utilization: Before vs After](#core-utilization-before-vs-after)
- [Priority Summary](#priority-summary)

---

## Root Cause Analysis

The throughput regression is caused by **thread oversubscription**. Every library in the MuseTalk stack auto-detects the total core count and spawns that many internal threads:

| Library | Thread Pool | Detection Method |
|---------|------------|-----------------|
| PyTorch | OpenMP | `os.cpu_count()` |
| NumPy/SciPy | MKL / OpenBLAS | `os.cpu_count()` |
| OpenCV | TBB / OpenMP | `os.cpu_count()` |
| Whisper | MKL + OpenMP | `os.cpu_count()` |

### The Math

```
On the OLD server (e.g., 8 cores):
  PyTorch spawns 8 OpenMP threads
  OpenCV spawns 8 threads
  Whisper/MKL spawns 8 threads
  Total: ~24 threads on 8 cores = 3x oversubscription (manageable)

On the NEW server (e.g., 64 cores):
  PyTorch spawns 64 OpenMP threads
  OpenCV spawns 64 threads
  Whisper/MKL spawns 64 threads
  Total: ~192 threads on 64 cores = 3x oversubscription
  BUT: cache thrashing across NUMA nodes,
       64-way lock contention in thread pools,
       massive context switching overhead
```

### Contributing Factors

1. **GIL Contention (Python Global Interpreter Lock):** More threads competing for the GIL on more cores increases context-switching overhead without enabling true parallelism.
2. **NUMA-Unaware Memory Access:** Servers with more cores typically have multiple NUMA nodes. Threads scheduled across NUMA nodes experience 2-3x memory access latency due to remote memory access.
3. **False Sharing & Cache Thrashing:** More cores means more L1/L2 caches competing. Shared data structures (numpy arrays, tensors being moved CPU↔GPU) trigger expensive cache coherence protocols.
4. **CPU-GPU Transfer Bottleneck:** More CPU cores do not help when the pipeline is serialized across CPU and GPU stages. Added scheduling overhead on a larger CPU actually hurts latency-sensitive transfers.

---

## Current Pipeline Architecture

The inference pipeline is **strictly sequential** — each frame passes through every stage before the next frame begins:

```
Frame 1: [Decode → FaceDetect → Whisper → UNet(GPU) → Composite → Encode]
Frame 2:          [Decode → FaceDetect → Whisper → UNet(GPU) → Composite → Encode]
Frame 3:                   [Decode → FaceDetect → Whisper → UNet(GPU) → Composite → Encode]
                                                    ↑
                                            THIS is the bottleneck
```

### Stage Breakdown

| Stage | Bound | Description |
|-------|-------|-------------|
| Video Decode | CPU | Read and decode video frames |
| Face Detection | CPU | DWPose / face parsing preprocessing |
| Audio Features | CPU | Whisper-based feature extraction (`--use_float16`) |
| VAE + UNet | **GPU** | VAE encode → UNet denoise → VAE decode |
| Composite | CPU | Paste result back onto original frame |
| Video Encode | CPU | Encode final output frame |

With the current architecture, **62 out of 64 cores sit idle** while one frame at a time crawls through the pipeline.

---

## Why Blind Multithreading Will Not Work

Python's **Global Interpreter Lock (GIL)** prevents true parallel execution of Python code across threads. Adding multithreading to the MuseTalk pipeline will compound the oversubscription problem:

| Factor | Evidence from Codebase | Impact of Multithreading |
|--------|----------------------|--------------------------|
| Python GIL | All inference scripts are pure Python orchestrating PyTorch | Threads cannot run Python in parallel — only adds overhead |
| PyTorch internal threads | PyTorch already uses OpenMP threads for CPU ops | More threads = thread pool × thread pool = exponential contention |
| Whisper inference | CPU-heavy feature extraction already uses MKL/OpenBLAS internally | Adding threading on top causes oversubscription |
| OpenCV face parsing | `cv2` already parallelizes internally via TBB/OpenMP | Same oversubscription issue |
| Real-time target (30fps) | Latency-sensitive pipeline | Thread synchronization adds unpredictable jitter |

---

## Why More Cores Hurt Performance

More cores are **not** inherently bad. The problem is that the current codebase **accidentally turns extra cores into overhead**. The solution is not fewer cores — it is using them correctly.

| Approach | Verdict | Reason |
|----------|---------|--------|
| Blind multithreading | ❌ No | GIL + library thread pools = compounded oversubscription |
| Fewer-core CPU | ❌ Wasteful | The hardware is fine; the software configuration is wrong |
| Thread pool limiting | ✅ Immediate fix | Directly addresses root cause with zero code changes |
| Multiprocessing (pipeline parallelism) | ✅ Best long-term | Bypasses GIL, uses cores for true parallel work |

---

## March 23 Reality Check

The latest Threadripper HLS load tests changed the interpretation of this file.

Hardware facts from `lscpu`:

- AMD Ryzen Threadripper PRO 3995WX
- 64 physical cores / 128 logical CPUs
- 1 visible NUMA node in this environment

Latest measured `8`-stream HLS regression:

- `avg_time_to_live_ready_s = 3.469`
- `avg_segment_interval_s = 3.622`
- `max_segment_interval_s = 6.137`
- `avg GPU util = 37.2%`

That result matters because it proves the current problem is no longer just
"too many library threads." The GPU is now underfed. The current CPU-tuning
helper in `scripts/runtime_cpu_tuning.py` is still valid as an
oversubscription-control mechanism, but the measured `api_server.py` HLS path
shows that aggressive caps can make the pipeline worse when:

- request prep is still front-loaded and sequential
- avatar cache misses still load all frames and masks serially
- chunk encode still launches a fresh `ffmpeg` process every time
- compose is still CPU/OpenCV and ordered behind per-batch queues

So the current conclusion is:

- thread caps are **not** the main next throughput lever
- pipeline refactoring is now more important than further thread-cap tuning
- the safest default today is still the stable no-CPU-tuning baseline until
  the host-side pipeline is reworked

### First Refactor Slice Now Implemented

The first host-side refactor slice has now landed in code:

- `musetalk/utils/audio_processor.py`
  - batched feature-extractor path for multi-segment audio
  - batched Whisper encoder path for multi-segment audio
  - vectorized `build_audio_prompts()` with exact integer indexing
- `scripts/hls_gpu_scheduler.py`
  - shared prep-subtask pool
  - concurrent avatar load + audio feature extraction during `_prepare_job()`
  - idle-frame preload now overlaps with prompt/conditioning setup
- `scripts/api_avatar.py`
  - parallel avatar frame / mask loading on cache miss
  - parallel metadata/material load during `_load_existing_materials()`

Still intentionally deferred:

- persistent per-stream encoder / segmenter
- larger compose-process refactor
- convergence of older direct live-serving paths onto the shared scheduler

So the current repo is no longer just "analysis only" on the host-pipeline
branch. The first real throughput-oriented code slice is already in place.

First measured result after that refactor slice:

- `concurrency=8`
- `batch_size=4`
- `playback_fps=24`
- `musetalk_fps=12`
- `hold_seconds=30`
- `completed=8`
- `failed=0`
- `avg_time_to_live_ready_s = 1.760`
- `avg_segment_interval_s = 1.733`
- `max_segment_interval_s = 2.535`
- `avg GPU util = 82.06%`

Later March 24 ramp results tightened the interpretation further:

- `concurrency=6`
  - `avg_time_to_live_ready_s = 1.342`
  - `avg_segment_interval_s = 1.294`
  - `max_segment_interval_s = 2.032`
  - `avg GPU util = 83.84%`
- `concurrency=7`
  - `avg_time_to_live_ready_s = 1.508`
  - `avg_segment_interval_s = 1.516`
  - `max_segment_interval_s = 2.530`
  - `avg GPU util = 84.83%`
- repeated `concurrency=8`
  - `avg_time_to_live_ready_s = 1.569-1.760`
  - `avg_segment_interval_s = 1.733-1.736`
  - `max_segment_interval_s = 2.524-2.535`
  - `avg GPU util = 82.06-83.64%`

Compared with the earlier severe regression:

- `avg_time_to_live_ready_s`: `3.469 -> 1.760`
- `avg_segment_interval_s`: `3.622 -> 1.733`
- `max_segment_interval_s`: `6.137 -> 2.535`
- `avg GPU util`: `37.2% -> 82.06%`

Interpretation:

- the first multithreaded / adjacent-task host-pipeline slice materially
  recovered throughput
- the GPU is no longer obviously starving the way it was during the bad run
- the system is back in the healthy "near-threshold" band
- `concurrency=6` is now the first practical realtime milestone on this branch
  even though the strict `load_test.py` warning still trips by about `32ms`
  on the worst interval
- `concurrency=7` and `concurrency=8` now sit in the same saturation band
  because the shared HLS scheduler is still operating on the same `batch_size=4`
  regime, so the extra stream mostly affects fairness/tail behavior rather than
  changing the GPU batch shape
- but the tail is still slightly above the current `2.0s` throttling line, so
  persistent encode and later compose refactors still matter

---

## Optimization Plan

### Phase 0: Restore The Stable Baseline

Before further CPU experimentation:

- keep `MUSETALK_CPU_TUNING` disabled for the live HLS path
- keep the repaired `trt_stagewise` backend
- use the known-working stagewise HLS start block without aggressive CPU caps

Why first:

- the latest measured helper-driven CPU tuning profiles starved the GPU
- we need a trustworthy baseline before refactoring adjacent CPU stages

### Phase 1: Parallelize HLS Prep

Current hot path in `scripts/hls_gpu_scheduler.py` is too sequential:

- avatar load
- audio decode / feature extraction
- Whisper encode
- prompt construction
- positional-encoding staging
- initial conditioning allocation

Best refactor direction:

- overlap avatar load with CPU audio feature extraction
- move prompt construction into block-based work instead of one long
  front-loaded serial section
- keep the final GPU scheduler batched, but shorten the amount of CPU work
  required before a job becomes schedulable

Why first:

- current load tests show the GPU is waiting for work
- this is the cleanest place to create adjacent-task parallelism

### Phase 2: Fix Avatar Cache Miss Cost

Current behavior in `scripts/api_avatar.py` is expensive:

- load all avatar frames serially with `cv2.imread`
- load all masks serially with `cv2.imread`
- keep a very large avatar footprint in RAM after load

Best refactor direction:

- parallel frame/mask reads on cache miss
- or move to lazy / demand-driven frame material loading
- keep the existing cache, but reduce the "cache miss tax"

Why second:

- cache misses are large enough to visibly delay live readiness
- this work can use many Threadripper cores effectively

### Phase 3: Persistent Encode Pipeline

Current behavior in `scripts/api_avatar.py`:

- every chunk spawns a new `ffmpeg` process
- raw frames are written over stdin
- encode and mux start from scratch for every segment

Best refactor direction:

- persistent per-stream encoder / segmenter process
- or a bounded process pool dedicated to long-lived encode workers
- keep `libx264` and `h264_nvenc` pluggable behind the same interface

Why third:

- this is one of the most obvious adjacent CPU/process bottlenecks
- it is also the clearest place where a high-core CPU can help

### Phase 4: Compose Pipeline Refactor

Current behavior:

- compose is already dispatched via thread pool
- but the actual work is still a Python loop over frames using CPU/OpenCV
- ordered compose completion can create head-of-line blocking

Best refactor direction:

- keep batching, but evaluate a bounded process pool for compose
- or move compose to coarser batched work units so Python overhead drops
- do not start by increasing thread counts blindly

Why fourth:

- this is still hot, but the scheduler already parallelizes it somewhat
- encode and prep look more urgent first

### Phase 5: Consolidate Live Serving Paths

Current repo state:

- HLS has a shared GPU scheduler
- older direct streaming/session endpoints still run `avatar.inference_streaming`
  inside `manager.executor`

Best refactor direction:

- converge live generation paths on the shared scheduler model
- reduce duplicated orchestration logic
- keep one place where CPU prep, GPU batching, compose, and encode are tuned

Why fifth:

- architecture duplication makes throughput tuning much harder
- it also increases the risk of solving the problem in one path but not another

### Phase 1: Thread Pool Limiting (Implemented, Not Current Default)

**Time:** 5 minutes
**Impact:** 30-60% throughput recovery

This is the single most important fix. Cap all internal thread pools to a sane number regardless of total core count.

#### Current repo status

This phase is now implemented as an opt-in helper:

- module:
  - `scripts/runtime_cpu_tuning.py`
- live entrypoints:
  - `api_server.py`
  - `scripts/benchmark_pipeline.py`
  - `scripts/inference.py`
  - `scripts/realtime_inference.py`
  - `scripts/preprocess.py`

The helper is implemented, but the latest Threadripper HLS results show the
current aggressive profiles should **not** be treated as the live-serving
default.

Earlier first-pass env block:

```bash
export MUSETALK_CPU_TUNING=1
export MUSETALK_CPU_THREADS=4
export MUSETALK_CPU_INTEROP_THREADS=2
export MUSETALK_CPU_CV2_THREADS=1
# Optional:
# export MUSETALK_CPU_NUMA_NODE=0
# export MUSETALK_CPU_AFFINITY=0-15
```

Behavior summary:

- applies thread-pool env vars before heavy imports
- applies `torch.set_num_threads()` and `torch.set_num_interop_threads()`
  after `torch` is imported
- applies `cv2.setNumThreads()` after `cv2` is imported
- optionally pins the process to a NUMA node or explicit CPU set

Why `MUSETALK_CPU_CV2_THREADS=1` was recommended first:

- the live HLS path already has multiple Python worker pools for prep, compose,
  and encode
- letting each OpenCV call fan out internally as well tends to recreate nested
  oversubscription
- `1` is still the safest diagnostic starting point, but the latest live HLS
  measurements show that this setting can underfeed the GPU on the current
  stagewise + `libx264` server path

#### Option A: Shell wrapper

```bash
#!/bin/bash
# run_optimized.sh

# Limit every library's internal thread pool
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_THREAD_COUNT=4

# Disable GPU-CPU sync stalls
export CUDA_LAUNCH_BLOCKING=0

python -m scripts.realtime_inference \
    --inference_config configs/inference/realtime.yaml \
    --fps 30 \
    "$@"
```

#### Option B: Python (add at the very top of your entry point, before any imports)

```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENCV_THREAD_COUNT"] = "4"

# ...existing imports below...
```

#### Option C: Reusable module

```python
# scripts/runtime_cpu_tuning.py
import os

def set_optimal_threads(num_threads=None):
    """Restrict all internal thread pools to avoid oversubscription."""
    if num_threads is None:
        num_threads = 4

    t = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t
    os.environ["OPENCV_THREAD_COUNT"] = "1"

    try:
        import torch
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except ImportError:
        pass

    print(f"All thread pools limited to {num_threads} threads")

set_optimal_threads()
```

---

### Phase 2: NUMA Pinning

**Time:** 10 minutes
**Impact:** low on the current host, potentially useful on a different multi-NUMA deployment

Pin the process to a single NUMA node to avoid cross-socket memory latency.

Important current note:

- `lscpu` on the active Threadripper host reports **1 visible NUMA node**
- that means NUMA pinning is **not** currently the lead lever for this machine
- keep this documented for other deployments, but do not prioritize it ahead
  of prep / compose / encode refactors on this box

#### Diagnose NUMA topology

```bash
lscpu | grep -i numa
numactl --hardware
```

#### Run with NUMA binding

```bash
numactl --cpunodebind=0 --membind=0 python -m scripts.realtime_inference \
    --inference_config configs/inference/realtime.yaml
```

#### Programmatic NUMA pinning

```python
# scripts/set_affinity.py
import os
import psutil

def pin_to_numa_node(node=0):
    """Pin current process and all threads to a single NUMA node."""
    p = psutil.Process(os.getpid())

    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            cpu_str = f.read().strip()
        cpus = []
        for part in cpu_str.split(","):
            if "-" in part:
                start, end = part.split("-")
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
    except FileNotFoundError:
        all_cpus = sorted(os.sched_getaffinity(0))
        cpus = all_cpus[:len(all_cpus) // 2]

    p.cpu_affinity(cpus)
    print(f"Pinned PID {os.getpid()} to CPUs: {cpus}")
    return cpus
```

---

### Phase 3: Pipeline Parallelism via Multiprocessing

**Time:** 2-3 days
**Impact:** True utilization of 14-16 cores with overlapping stage execution

This is the correct way to use many cores. Each pipeline stage runs in its own **process** (not thread), bypassing the GIL entirely. Stages communicate via bounded queues with backpressure.

#### Target Architecture

```
Time →
Core 0-3:   [Decode F1] [Decode F2] [Decode F3] [Decode F4] ...
Core 4-7:              [Whisper F1] [Whisper F2] [Whisper F3] ...
Core 8-9:                          [GPU Inf F1] [GPU Inf F2] ...
Core 10-13:                                    [Encode F1] [Encode F2] ...
```

#### Implementation

```python
# musetalk/pipeline/parallel_pipeline.py
import os
import multiprocessing as mp
from multiprocessing import Process, Queue

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

SENTINEL = "PIPELINE_DONE"


def stage_decode(in_q, out_q, cpus):
    """Stage 1: Video decode + face detection (CPU-bound)"""
    os.sched_setaffinity(0, cpus)
    import cv2
    from musetalk.utils.face_parsing import FaceParsing

    while True:
        item = in_q.get()
        if item == SENTINEL:
            out_q.put(SENTINEL)
            break
        frame_idx, frame = item
        # Face detection + crop + parse
        # ...processing logic...
        out_q.put((frame_idx, frame, face_data))


def stage_audio(in_q, out_q, cpus):
    """Stage 2: Whisper feature extraction (CPU-bound)"""
    os.sched_setaffinity(0, cpus)
    import torch
    torch.set_num_threads(len(cpus))

    while True:
        item = in_q.get()
        if item == SENTINEL:
            out_q.put(SENTINEL)
            break
        frame_idx, frame, face_data = item
        # Extract audio features
        # ...processing logic...
        out_q.put((frame_idx, frame, face_data, audio_features))


def stage_gpu_inference(in_q, out_q, cpus):
    """Stage 3: VAE + UNet inference (GPU-bound, minimal CPU)"""
    os.sched_setaffinity(0, cpus)
    import torch
    torch.set_num_threads(2)

    while True:
        item = in_q.get()
        if item == SENTINEL:
            out_q.put(SENTINEL)
            break
        frame_idx, frame, face_data, audio_features = item
        # Run VAE encode → UNet → VAE decode
        # ...processing logic...
        out_q.put((frame_idx, frame, result))


def stage_encode(in_q, out_q, cpus):
    """Stage 4: Composite + video encode (CPU-bound)"""
    os.sched_setaffinity(0, cpus)
    import cv2

    while True:
        item = in_q.get()
        if item == SENTINEL:
            if out_q:
                out_q.put(SENTINEL)
            break
        frame_idx, frame, result = item
        # Paste result back, encode frame
        # ...processing logic...
        if out_q:
            out_q.put((frame_idx, final_frame))


class ParallelMuseTalkPipeline:
    """
    Pipeline-parallel inference using multiprocessing.

    Uses ~16 cores effectively:
      - Decode/FaceDetect:  4 cores (CPU-bound)
      - Whisper:            4 cores (CPU-bound)
      - GPU Inference:      2 cores (GPU does real work)
      - Encode/Composite:   4 cores (CPU-bound)

    Remaining cores are free for OS, GPU DMA, and headroom.
    """

    def __init__(self, numa_node=0):
        self.all_cpus = self._get_numa_cpus(numa_node)
        n = len(self.all_cpus)
        self.cpu_map = {
            "decode":    self.all_cpus[0 : n // 4],
            "audio":     self.all_cpus[n // 4 : n // 2],
            "inference": self.all_cpus[n // 2 : n // 2 + 2],
            "encode":    self.all_cpus[n // 2 + 2 : 3 * n // 4],
        }

    def _get_numa_cpus(self, node):
        try:
            with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
                cpu_str = f.read().strip()
            cpus = []
            for part in cpu_str.split(","):
                if "-" in part:
                    start, end = part.split("-")
                    cpus.extend(range(int(start), int(end) + 1))
                else:
                    cpus.append(int(part))
            return cpus
        except FileNotFoundError:
            return list(range(os.cpu_count()))

    def run(self, frames_iterator, output_callback=None):
        q1 = Queue(maxsize=16)
        q2 = Queue(maxsize=16)
        q3 = Queue(maxsize=16)
        q4 = Queue(maxsize=16)

        processes = [
            Process(target=stage_decode,
                    args=(q1, q2, self.cpu_map["decode"])),
            Process(target=stage_audio,
                    args=(q2, q3, self.cpu_map["audio"])),
            Process(target=stage_gpu_inference,
                    args=(q3, q4, self.cpu_map["inference"])),
            Process(target=stage_encode,
                    args=(q4, None, self.cpu_map["encode"])),
        ]

        for p in processes:
            p.start()

        for idx, frame in enumerate(frames_iterator):
            q1.put((idx, frame))
        q1.put(SENTINEL)

        for p in processes:
            p.join()
```

---

### Phase 4: Multi-Stream Serving

**Time:** 1 week
**Impact:** Full utilization of all 64 cores

With pipeline parallelism in place, the remaining cores can serve **multiple avatar streams simultaneously**:

```
Stream A: Cores 0-15  → Avatar 1
Stream B: Cores 16-31 → Avatar 2
Stream C: Cores 32-47 → Avatar 3
Stream D: Cores 48-63 → Avatar 4
All sharing 1 GPU with batched inference
```

This is where a high-core-count CPU truly pays off — not by making one stream faster, but by running many streams concurrently.

---

## Core Utilization: Before vs After

### Before (Current Regression)

```
Core  0-?: mixed prep / compose / encode contention
GPU: underfed
Observed avg GPU util: ~37%
Observed avg segment interval: ~3.62s
```

### After Baseline Recovery

```
Current practical goal:

- restore the known-good stagewise TRT baseline without aggressive CPU caps
- keep GPU utilization back in the ~80% band
- then refactor the host pipeline
```

### After Host-Side Pipeline Refactor

```
Core groups do different adjacent work in parallel:

- avatar load / cache miss recovery
- audio feature extraction and prompt prep
- GPU scheduler batch execution
- compose
- encode / segment mux

Desired effect:

- GPU stays fed
- live_ready compresses
- segment intervals stop drifting into the 3-6s range
```

### After Multi-Stream Scaling

```
Only after the host pipeline is efficient does it make sense to partition the
CPU budget harder across multiple simultaneous live streams.
```

---

## Priority Summary

| Priority | Phase | Fix | Expected Impact | Effort |
|----------|-------|-----|----------------|--------|
| 🔴 1 | Phase 0 | Restore no-CPU-tuning stable baseline | Prevents current regression | minutes |
| 🔴 2 | Phase 1 | Parallelize HLS prep | Faster live_ready, better GPU feed | 1-2 days |
| 🔴 3 | Phase 2 | Parallel / lazy avatar material loading | Lower cache-miss tax | 1-2 days |
| 🔴 4 | Phase 3 | Persistent encode pipeline | Lower chunk latency and CPU waste | 2-3 days |
| 🟡 5 | Phase 4 | Compose pipeline refactor | Lower CPU queueing / jitter | 2-3 days |
| 🟢 6 | Phase 5 | Consolidate live serving paths | Cleaner long-term scaling | several days |

---

## Key Takeaway

> **Your high-core CPU is still an asset, not a liability.** But the latest
> Threadripper measurements show that simple thread caps are not enough. The
> next real gains will come from restructuring the host pipeline so avatar
> loading, audio prep, compose, and encode can use adjacent cores effectively
> without starving the GPU.
