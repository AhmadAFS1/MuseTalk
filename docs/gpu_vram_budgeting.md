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
