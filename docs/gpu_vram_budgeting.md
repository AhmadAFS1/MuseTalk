# GPU VRAM Budgeting

MuseTalk now derives runtime GPU budget defaults from the detected VRAM size
instead of assuming every server is a 24GB RTX 3090-class card.

## Detection and overrides

At startup the server detects GPU memory through Torch, then `nvidia-smi`, then
falls back to 24GB. Operators can override the detected values:

```bash
GPU_TOTAL_MEMORY_GB=32
GPU_RESERVED_MEMORY_GB=8
```

The reserved budget defaults to roughly 25% of total VRAM, clamped between 4GB
and 10GB. The logical direct-generation batch leases can also be overridden:

```bash
GPU_MEMORY_BATCH_GB="4:6,8:10,16:14,32:22"
```

## Throughput profile defaults

`PROFILE=throughput_record` and `PROFILE=vram_max` now choose scheduler buckets
from the VRAM class when the relevant env vars are not explicitly set.

| GPU VRAM | Max scheduler batch | Fixed/warmed buckets |
| --- | ---: | --- |
| `<20GB` | 8 | `4,8` |
| `20-29GB` | 16 | `8,16` |
| `30-44GB` | 32 | `4,8,16,32` |
| `45GB+` | 48 | `4,8,16,32,48` |

`PROFILE=baseline` remains conservative: max batch `4`, fixed/warmed bucket `4`.

## V100 32GB target

On a 32GB V100, the GPU-aware throughput defaults are:

```bash
GPU_TOTAL_MEMORY_GB=32
GPU_RESERVED_MEMORY_GB=8
HLS_SCHEDULER_MAX_BATCH=32
HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4,8,16,32
HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
```

Manual env vars still win. For example, to use the 32GB card but cap WebRTC at
the lower-latency 16-frame path:

```bash
PROFILE=throughput_record \
HLS_SCHEDULER_MAX_BATCH=16 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4,8,16 \
bash scripts/vast_server_ctl.sh restart
```

## 2026-05-23 observed 32GB behavior

The automatic 32GB default of `4,8,16,32` was too aggressive for the current
TensorRT stagewise warmup path on the tested 32GB V100-class worker. Batch `32`
OOM'd during warmup after `4,8,16` had already compiled. Batch `20` also OOM'd
when warmed alongside `4,8,16` or `8,16`.

The successful high-throughput experiment was a sparse profile:

```bash
PROFILE=throughput_record \
WEBRTC_H264_ENCODER=libx264 \
HLS_SCHEDULER_MAX_BATCH=24 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=24 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=24 \
STARTUP_TIMEOUT_SECONDS=3600 \
bash scripts/vast_server_ctl.sh restart
```

Observed behavior:

- batch `24` warmed successfully in `540.45s`
- health passed after `9m23s`
- idle VRAM after health was about `21GB`
- WebRTC load tests with avatar cache active stayed around `25-26GB`
- max scheduler batch remained `24`; the server did not try to fill the full
  `32GB`

This is expected: VRAM is allocated for the warmed engine, runtime state, avatar
cache, buffers, and active jobs. It is not automatically filled. If GPU
utilization is high while VRAM has headroom, the bottleneck is more likely
compute, scheduling, composition, or encoding than raw VRAM capacity.

With only bucket `24` warmed, smaller WebRTC/HLS requested batch sizes resolve
upward to `24`. This means wall tests with `batch_size=2` on this profile do not
measure a true batch-2 TensorRT path.
